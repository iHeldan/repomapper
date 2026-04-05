"""
RepoMap class for generating repository maps.
"""

import json
import os
import re
import shlex
from pathlib import Path
from collections import namedtuple, defaultdict, deque
from typing import List, Dict, Set, Optional, Tuple, Callable
import shutil
import sqlite3
from dataclasses import dataclass, field
import diskcache
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python
from grep_ast import TreeContext
from utils import count_tokens, read_text, find_src_files
from scm import get_scm_fname
from importance import is_important
from parser_support import resolve_parser_config, format_parser_runtime_error
from repomap_config import load_repo_map_config
from repomap_semantics import collect_semantic_links, SemanticLink

try:
    import tomllib
except ImportError:
    tomllib = None

# Pre-import tree-sitter modules to avoid per-file import overhead (R1 Finding #7)
try:
    from grep_ast import filename_to_lang
    from grep_ast.tsl import get_language, get_parser
    from tree_sitter import Query, QueryCursor
    _HAS_GREP_AST = True
except ImportError:
    _HAS_GREP_AST = False

# R2 Finding #1: Global caches that persist across RepoMap instances (MCP requests)
_GLOBAL_QUERY_CACHE = {}
_GLOBAL_DISK_CACHES = {}


@dataclass
class RankingReason:
    code: str
    message: str


@dataclass
class RankedFile:
    path: str
    rank: float
    base_rank: float
    included_in_map: bool = False
    is_changed_file: bool = False
    changed_neighbor_distance: Optional[int] = None
    is_test_file: bool = False
    is_entrypoint_file: bool = False
    is_public_api_file: bool = False
    is_chat_file: bool = False
    is_mentioned_file: bool = False
    is_important_file: bool = False
    definitions: int = 0
    references: int = 0
    referenced_by_files: int = 0
    references_to_files: int = 0
    matched_query_terms: List[str] = field(default_factory=list)
    matched_query_path_terms: List[str] = field(default_factory=list)
    matched_query_symbol_terms: List[str] = field(default_factory=list)
    related_changed_files: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)
    related_sources: List[str] = field(default_factory=list)
    entrypoint_signals: List[str] = field(default_factory=list)
    public_api_signals: List[str] = field(default_factory=list)
    summary_kind: Optional[str] = None
    summary_items: List[str] = field(default_factory=list)
    mentioned_identifiers: List[str] = field(default_factory=list)
    sample_symbols: List[str] = field(default_factory=list)
    inbound_neighbors: List[str] = field(default_factory=list)
    outbound_neighbors: List[str] = field(default_factory=list)
    lines_of_interest: List[int] = field(default_factory=list)
    reasons: List[RankingReason] = field(default_factory=list)


@dataclass
class FileReport:
    excluded: Dict[str, str]        # File -> exclusion reason with status
    definition_matches: int         # Total definition tags
    reference_matches: int          # Total reference tags
    total_files_considered: int     # Total files provided as input
    diagnostics: List[str] = field(default_factory=list)
    query: Optional[str] = None
    query_terms: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    changed_neighbor_depth: int = 0
    ranked_files: List[RankedFile] = field(default_factory=list)
    selected_files: List[str] = field(default_factory=list)
    map_tokens: int = 0


@dataclass
class ConnectionStep:
    source: str
    target: str
    relation: str
    symbols: List[str] = field(default_factory=list)
    symbol_hops: List["SymbolTraceHop"] = field(default_factory=list)


@dataclass
class SymbolTraceHop:
    source_file: str
    target_file: str
    relation: str
    source_symbol: Optional[str] = None
    target_symbol: Optional[str] = None
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    evidence_kind: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class ConnectionReport:
    start_file: str
    end_file: str
    path: List[str] = field(default_factory=list)
    steps: List[ConnectionStep] = field(default_factory=list)
    symbol_path: List[SymbolTraceHop] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ImpactTarget:
    path: str
    seed_file: str
    distance: int
    path_from_seed: List[str] = field(default_factory=list)
    steps: List[ConnectionStep] = field(default_factory=list)
    symbol_path: List[SymbolTraceHop] = field(default_factory=list)
    seed_focus_lines: List[int] = field(default_factory=list)
    seed_hunks: List["ImpactHunk"] = field(default_factory=list)
    changed_boundary_symbols: List[str] = field(default_factory=list)
    changed_boundary_distances: Dict[str, int] = field(default_factory=dict)
    closest_changed_hunk_distance: Optional[int] = None
    boundary_symbols: List[str] = field(default_factory=list)
    boundary_relations: List[str] = field(default_factory=list)
    boundary_locations: List["ImpactLocation"] = field(default_factory=list)
    boundary_snippets: List["ImpactSnippet"] = field(default_factory=list)
    focus_lines: List[int] = field(default_factory=list)
    is_test_file: bool = False
    is_entrypoint_file: bool = False
    is_public_api_file: bool = False
    is_important_file: bool = False
    summary_kind: Optional[str] = None
    summary_items: List[str] = field(default_factory=list)
    reasons: List[RankingReason] = field(default_factory=list)


@dataclass
class ImpactSuggestion:
    priority: int
    kind: str
    target: str
    message: str
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    anchor_file: Optional[str] = None
    anchor_line: Optional[int] = None
    anchor_symbol: Optional[str] = None
    anchor_kind: Optional[str] = None
    anchor_excerpt: Optional[str] = None


@dataclass
class ImpactQuickAction:
    priority: int
    kind: str
    target: str
    message: str
    effort: str = "small"
    target_role: str = "boundary"
    risk_level: str = "low"
    confidence: float = 0.5
    focus_symbols: List[str] = field(default_factory=list)
    focus_reason: Optional[str] = None
    why_now: Optional[str] = None
    expected_outcome: Optional[str] = None
    follow_if_true: Optional[str] = None
    follow_if_false: Optional[str] = None
    location_hint: Optional[str] = None
    command_hint: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    anchor_file: Optional[str] = None
    anchor_line: Optional[int] = None
    anchor_symbol: Optional[str] = None
    anchor_kind: Optional[str] = None
    anchor_excerpt: Optional[str] = None


@dataclass
class ImpactEditCandidate:
    path: str
    target_role: str
    reason: str
    priority: int = 0
    confidence: float = 0.5
    line: Optional[int] = None
    symbol: Optional[str] = None
    symbol_kind: Optional[str] = None
    location_hint: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    source_action_kind: Optional[str] = None
    source_action_target: Optional[str] = None


@dataclass
class ImpactEditPlanStep:
    step: int
    priority: int
    title: str
    instruction: str
    target: str
    target_role: str
    confidence: float = 0.5
    action_kind: Optional[str] = None
    location_hint: Optional[str] = None
    command_hint: Optional[str] = None
    focus_symbols: List[str] = field(default_factory=list)
    why_now: Optional[str] = None
    expected_outcome: Optional[str] = None
    follow_if_true: Optional[str] = None
    follow_if_false: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    edit_candidates: List[ImpactEditCandidate] = field(default_factory=list)


@dataclass
class ImpactTestCluster:
    kind: str
    seed_file: str
    paths: List[str] = field(default_factory=list)
    covers: List[str] = field(default_factory=list)
    closest_distance: Optional[int] = None
    focus_symbols: List[str] = field(default_factory=list)
    command_hint: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ImpactLocation:
    file: str
    line: int
    kind: str
    symbol: str


@dataclass
class ImpactSnippet:
    file: str
    start_line: int
    end_line: int
    highlight_line: int
    kind: str
    symbol: str
    excerpt: str


@dataclass
class ImpactHunk:
    start_line: int
    end_line: int


@dataclass
class ImpactSymbol:
    name: str
    target_files: List[str] = field(default_factory=list)
    seed_files: List[str] = field(default_factory=list)
    target_count: int = 0
    closest_distance: Optional[int] = None
    is_changed_seed_symbol: bool = False
    closest_changed_hunk_distance: Optional[int] = None
    locations: List[ImpactLocation] = field(default_factory=list)


@dataclass
class ImpactReport:
    seed_files: List[str]
    max_depth: int
    max_results: int
    impacted_files: List[ImpactTarget] = field(default_factory=list)
    changed_lines_by_file: Dict[str, List[int]] = field(default_factory=dict)
    changed_hunks_by_file: Dict[str, List[ImpactHunk]] = field(default_factory=dict)
    changed_seed_symbols: Dict[str, List[str]] = field(default_factory=dict)
    shared_symbols: List[ImpactSymbol] = field(default_factory=list)
    quick_actions: List[ImpactQuickAction] = field(default_factory=list)
    edit_candidates: List[ImpactEditCandidate] = field(default_factory=list)
    edit_plan: List[ImpactEditPlanStep] = field(default_factory=list)
    test_clusters: List[ImpactTestCluster] = field(default_factory=list)
    suggested_checks: List[ImpactSuggestion] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ReviewChangedFile:
    path: str
    target_role: str
    changed_lines: List[int] = field(default_factory=list)
    changed_hunks: List["ImpactHunk"] = field(default_factory=list)
    changed_symbols: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)
    entrypoint_signals: List[str] = field(default_factory=list)
    public_api_signals: List[str] = field(default_factory=list)
    summary_kind: Optional[str] = None
    summary_items: List[str] = field(default_factory=list)
    is_test_file: bool = False
    is_entrypoint_file: bool = False
    is_public_api_file: bool = False
    is_important_file: bool = False


@dataclass
class ReviewFocusItem:
    priority: int
    kind: str
    title: str
    target: str
    target_role: str
    message: str
    risk_level: str = "low"
    confidence: float = 0.5
    location_hint: Optional[str] = None
    command_hint: Optional[str] = None
    focus_symbols: List[str] = field(default_factory=list)
    why_now: Optional[str] = None
    expected_outcome: Optional[str] = None
    follow_if_true: Optional[str] = None
    follow_if_false: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    anchor_file: Optional[str] = None
    anchor_line: Optional[int] = None
    anchor_symbol: Optional[str] = None
    anchor_kind: Optional[str] = None


@dataclass
class ReviewReport:
    current_branch: Optional[str]
    base_ref: Optional[str]
    max_depth: int
    max_results: int
    changed_files: List[ReviewChangedFile] = field(default_factory=list)
    changed_lines_by_file: Dict[str, List[int]] = field(default_factory=dict)
    changed_hunks_by_file: Dict[str, List["ImpactHunk"]] = field(default_factory=dict)
    changed_seed_symbols: Dict[str, List[str]] = field(default_factory=dict)
    impacted_files: List[ImpactTarget] = field(default_factory=list)
    shared_symbols: List[ImpactSymbol] = field(default_factory=list)
    quick_actions: List[ImpactQuickAction] = field(default_factory=list)
    edit_candidates: List[ImpactEditCandidate] = field(default_factory=list)
    edit_plan: List[ImpactEditPlanStep] = field(default_factory=list)
    test_clusters: List[ImpactTestCluster] = field(default_factory=list)
    suggested_checks: List[ImpactSuggestion] = field(default_factory=list)
    review_focus: List[ReviewFocusItem] = field(default_factory=list)
    changed_public_api_files: List[str] = field(default_factory=list)
    changed_entrypoint_files: List[str] = field(default_factory=list)
    changed_test_files: List[str] = field(default_factory=list)
    changed_config_files: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    error: Optional[str] = None



# Constants
CACHE_VERSION = 1

TAGS_CACHE_DIR = f".repomap.tags.cache.v{CACHE_VERSION}"
SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError)
QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "how", "into", "its", "that",
    "the", "their", "this", "those", "was", "what", "when", "where", "which",
    "with",
}
TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs"}
INTEGRATION_TEST_MARKERS = {
    "acceptance", "e2e", "end_to_end", "feature", "features",
    "integration", "integrations", "scenario", "scenarios", "system",
}
ENTRYPOINT_FILENAMES = {
    "__main__.py", "main.py", "app.py", "server.py", "cli.py", "manage.py", "wsgi.py", "asgi.py",
    "main.ts", "main.tsx", "main.js", "main.jsx",
    "app.ts", "app.tsx", "app.js", "app.jsx",
    "server.ts", "server.tsx", "server.js", "server.jsx",
    "index.ts", "index.tsx", "index.js", "index.jsx",
}
ENTRYPOINT_DIR_NAMES = {"bin", "cmd", "scripts"}
PUBLIC_API_FILENAMES = {
    "__init__.py", "index.ts", "index.tsx", "index.js", "index.jsx",
    "api.py", "api.ts", "api.tsx", "api.js", "api.jsx",
    "client.py", "client.ts", "client.js",
    "routes.py", "routes.ts", "routes.js",
    "router.py", "router.ts", "router.js",
    "urls.py", "urls.ts", "urls.js",
}
PUBLIC_API_DIR_NAMES = {"api", "apis", "route", "routes", "router", "routers", "public"}

# Tag namedtuple for storing parsed code definitions and references
Tag = namedtuple("Tag", "rel_fname fname line name kind".split())


class RepoMap:
    """Main class for generating repository maps."""
    
    def __init__(
        self,
        map_tokens: int = 1024,
        root: str = None,
        token_counter_func: Callable[[str], int] = count_tokens,
        file_reader_func: Callable[[str], Optional[str]] = read_text,
        output_handler_funcs: Dict[str, Callable] = None,
        repo_content_prefix: Optional[str] = None,
        verbose: bool = False,
        max_context_window: Optional[int] = None,
        map_mul_no_files: int = 8,
        exclude_unranked: bool = False
    ):
        """Initialize RepoMap instance."""
        self.max_map_tokens = map_tokens
        self.root = Path(root or os.getcwd()).resolve()
        self.token_count_func_internal = token_counter_func
        self.read_text_func_internal = file_reader_func
        self.repo_content_prefix = repo_content_prefix
        self.verbose = verbose
        self.max_context_window = max_context_window
        self.map_mul_no_files = map_mul_no_files
        self.exclude_unranked = exclude_unranked
        
        # Set up output handlers
        if output_handler_funcs is None:
            output_handler_funcs = {
                'info': print,
                'warning': print,
                'error': print
            }
        self.output_handlers = output_handler_funcs
        
        # Initialize caches
        self.tree_context_cache = {}
        self.map_cache = {}
        self._query_cache = _GLOBAL_QUERY_CACHE   # R2: shared across instances
        self._local_tags_cache = {}               # Per-request (mtime could change between requests)
        self._tag_failures = {}
        self._uncacheable_tag_failures = set()
        self.diagnostics = []
        self.file_summary_by_file = {}
        self.file_summary_kind_by_file = {}
        self.repo_config = load_repo_map_config(self.root, self.read_text_func_internal)
        self._test_dir_names = TEST_DIR_NAMES | set(self.repo_config.tests.dirs)
        self._candidate_test_dir_names = self._dedupe_preserve_order(
            ["tests", "test", "__tests__", "spec", "specs"] + list(self.repo_config.tests.dirs)
        )
        self._integration_test_markers = INTEGRATION_TEST_MARKERS | set(self.repo_config.tests.integration_markers)
        self._entrypoint_filenames = ENTRYPOINT_FILENAMES | set(self.repo_config.frameworks.entrypoint_files)
        self._entrypoint_dir_names = ENTRYPOINT_DIR_NAMES | set(self.repo_config.frameworks.entrypoint_dirs)
        self._public_api_filenames = PUBLIC_API_FILENAMES | set(self.repo_config.frameworks.public_api_files)
        self._public_api_dir_names = PUBLIC_API_DIR_NAMES | set(self.repo_config.frameworks.public_api_dirs)

        # Load persistent tags cache
        self.load_tags_cache()
    
    def load_tags_cache(self):
        """Load the persistent tags cache. Reuses global SQLite connection per root dir (R2 Finding #1)."""
        cache_dir = str(self.root / TAGS_CACHE_DIR)
        try:
            if cache_dir not in _GLOBAL_DISK_CACHES:
                _GLOBAL_DISK_CACHES[cache_dir] = diskcache.Cache(cache_dir)
            self.TAGS_CACHE = _GLOBAL_DISK_CACHES[cache_dir]
        except Exception as e:
            self.output_handlers['warning'](f"Failed to load tags cache: {e}")
            self.TAGS_CACHE = {}
    
    def tags_cache_error(self):
        """Handle tags cache errors."""
        try:
            cache_dir = self.root / TAGS_CACHE_DIR
            cache_dir_str = str(cache_dir)
            # Remove stale diskcache reference before rmtree so load_tags_cache creates a fresh one
            _GLOBAL_DISK_CACHES.pop(cache_dir_str, None)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            self.load_tags_cache()
        except Exception:
            self.output_handlers['warning']("Failed to recreate tags cache, using in-memory cache")
            self.TAGS_CACHE = {}
    
    def token_count(self, text: str) -> int:
        """Count tokens in text with sampling optimization for long texts."""
        if not text:
            return 0
        
        len_text = len(text)
        if len_text < 200:
            return self.token_count_func_internal(text)
        
        # Sample for longer texts
        lines = text.splitlines(keepends=True)
        num_lines = len(lines)
        
        step = max(1, num_lines // 100)
        sampled_lines = lines[::step]
        sample_text = "".join(sampled_lines)
        
        if not sample_text:
            return self.token_count_func_internal(text)

        sample_tokens = self.token_count_func_internal(sample_text)
        est_tokens = (sample_tokens / len(sample_text)) * len_text
        return int(est_tokens)
    
    def get_rel_fname(self, fname: str) -> str:
        """Get relative filename from absolute path."""
        try:
            return str(Path(fname).relative_to(self.root))
        except ValueError:
            return fname
    
    def get_mtime(self, fname: str) -> Optional[float]:
        """Get file modification time."""
        try:
            return os.stat(fname).st_mtime_ns
        except OSError:
            return None
    
    def get_tags(self, fname: str, rel_fname: str) -> List[Tag]:
        """Get tags for a file, using cache when possible."""
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []

        # Check in-memory cache first to avoid diskcache SQLite round-trips (Finding #6)
        local_entry = self._local_tags_cache.get(fname)
        if local_entry and local_entry.get("mtime") == file_mtime:
            return local_entry["data"]

        try:
            cached_entry = self.TAGS_CACHE.get(fname)

            if cached_entry and cached_entry.get("mtime") == file_mtime:
                self._local_tags_cache[fname] = cached_entry
                return cached_entry["data"]
        except SQLITE_ERRORS:
            self.tags_cache_error()

        # Cache miss or file changed
        tags = self.get_tags_raw(fname, rel_fname)

        if fname in self._uncacheable_tag_failures:
            self._uncacheable_tag_failures.discard(fname)
            return tags

        cache_entry = {"mtime": file_mtime, "data": tags}

        try:
            self.TAGS_CACHE[fname] = cache_entry
        except SQLITE_ERRORS:
            self.tags_cache_error()

        self._local_tags_cache[fname] = cache_entry
        return tags

    def _add_diagnostic(self, message: str):
        """Store a de-duplicated runtime diagnostic for the current request."""
        if message not in self.diagnostics:
            self.diagnostics.append(message)

    def _reset_request_state(self):
        """Reset per-request caches while retaining repo-config diagnostics."""
        self.diagnostics = list(self.repo_config.diagnostics)
        self._tag_failures = {}
        self._uncacheable_tag_failures = set()

    def _get_scope_exclusion_reason(self, rel_fname: str) -> Optional[str]:
        """Return a repo-config scope exclusion reason, if any."""
        return self.repo_config.scope_reason(self._normalize_rel_path(rel_fname))

    def _prepare_candidate_files(
        self,
        fnames: List[str],
        excluded: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Normalize, de-duplicate, and scope-filter candidate files."""
        scoped_fnames = []
        seen = set()
        for fname in fnames:
            abs_fname = os.path.abspath(fname)
            if abs_fname in seen:
                continue
            seen.add(abs_fname)

            if not os.path.exists(abs_fname):
                if excluded is not None:
                    excluded[abs_fname] = "File not found"
                continue

            rel_fname = self.get_rel_fname(abs_fname)
            scope_reason = self._get_scope_exclusion_reason(rel_fname)
            if scope_reason:
                if excluded is not None:
                    excluded[abs_fname] = scope_reason
                continue

            scoped_fnames.append(abs_fname)
        return scoped_fnames

    def _find_repo_files(self) -> List[str]:
        """Find repository files and apply configured include/exclude patterns."""
        return self._prepare_candidate_files(find_src_files(str(self.root)))

    def _is_important_file(self, rel_fname: str) -> bool:
        """Check built-in and repo-config important-file signals."""
        normalized = self._normalize_rel_path(rel_fname)
        return is_important(normalized) or self.repo_config.is_configured_important_file(normalized)

    def _record_tag_failure(self, fname: str, message: str, *, cacheable: bool):
        """Remember a per-file parse failure and whether it may be cached."""
        self._tag_failures[fname] = message
        self._add_diagnostic(message)
        if not cacheable:
            self._uncacheable_tag_failures.add(fname)

    def _clear_tag_failure(self, fname: str):
        """Clear a previously recorded parse failure for a file."""
        self._tag_failures.pop(fname, None)
        self._uncacheable_tag_failures.discard(fname)

    def _calculate_pagerank(self, graph: nx.MultiDiGraph, personalization: Dict[str, float]) -> Dict[str, float]:
        """Run PageRank with a pure-Python fallback when scipy/numpy is unavailable."""
        pagerank_kwargs = {"alpha": 0.85}
        if personalization:
            pagerank_kwargs["personalization"] = personalization

        try:
            return nx.pagerank(graph, **pagerank_kwargs)
        except (ImportError, ModuleNotFoundError):
            return _pagerank_python(graph, **pagerank_kwargs)

    def _score_file_tag(
        self,
        rel_fname: str,
        tag_name: str,
        file_rank: float,
        mentioned_fnames: Set[str],
        mentioned_idents: Set[str],
        chat_rel_fnames: Set[str],
        query_boost: float = 1.0,
    ) -> float:
        """Apply the standard rank boosts for a file/tag pair."""
        boost = 1.0
        weights = self.repo_config.ranking_weights
        if tag_name in mentioned_idents:
            boost *= self._scale_multiplier(10.0, weights.mentioned_identifier)
        if rel_fname in mentioned_fnames:
            boost *= self._scale_multiplier(5.0, weights.mentioned_file)
        if rel_fname in chat_rel_fnames:
            boost *= self._scale_multiplier(20.0, weights.chat_file)
        boost *= query_boost
        return file_rank * boost

    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        """Return unique values while preserving first-seen order."""
        seen = set()
        unique_values = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    @staticmethod
    def _normalize_rel_path(rel_fname: str) -> str:
        """Normalize relative paths to a forward-slash form for matching heuristics."""
        return Path(rel_fname).as_posix()

    @staticmethod
    def _scale_multiplier(base_multiplier: float, weight: float) -> float:
        """Scale an existing >1.0 multiplier with a user-configurable weight."""
        return 1.0 + max(base_multiplier - 1.0, 0.0) * max(weight, 0.0)

    @staticmethod
    def _scale_floor(base_floor: float, weight: float) -> float:
        """Scale a non-negative ranking floor with a user-configurable weight."""
        return max(base_floor, 0.0) * max(weight, 0.0)

    def _is_test_file(self, rel_fname: str) -> bool:
        """Heuristically detect whether a file is a test/spec file."""
        normalized = self._normalize_rel_path(rel_fname).lower()
        parts = normalized.split("/")
        basename = parts[-1]
        stem = Path(basename).stem.lower()

        if any(part in self._test_dir_names for part in parts[:-1]):
            return True

        return (
            basename.startswith("test_")
            or basename.endswith("_test.py")
            or ".test." in basename
            or ".spec." in basename
            or stem.startswith("test_")
            or stem.endswith("_test")
            or stem.endswith("_spec")
        )

    def _candidate_related_test_paths(self, rel_fname: str) -> List[str]:
        """Generate likely test file paths for a source file."""
        normalized = self._normalize_rel_path(rel_fname)
        path = Path(normalized)
        suffix = path.suffix
        stem = path.stem
        basename = path.name
        parent_parts = list(path.parent.parts) if str(path.parent) != "." else []
        source_relative_parts = parent_parts[:]
        if source_relative_parts and source_relative_parts[0] in {"src", "lib", "app", "source"}:
            source_relative_parts = source_relative_parts[1:]

        candidate_names = [
            basename,
            f"test_{stem}{suffix}",
            f"{stem}_test{suffix}",
            f"{stem}.test{suffix}",
            f"{stem}.spec{suffix}",
        ]

        candidate_dirs = [
            parent_parts,
        ]
        for test_dir_name in self._candidate_test_dir_names:
            candidate_dirs.append(parent_parts + [test_dir_name])
        for test_dir_name in self._candidate_test_dir_names:
            candidate_dirs.append([test_dir_name] + source_relative_parts)

        candidates = []
        for dir_parts in candidate_dirs:
            dir_path = Path(*dir_parts) if dir_parts else Path()
            for candidate_name in candidate_names:
                candidates.append((dir_path / candidate_name).as_posix())
        return self._dedupe_preserve_order(candidates)

    def _find_related_test_files(self, rel_fname: str, known_rel_fnames: Set[str]) -> List[str]:
        """Resolve likely test files for a source file from known repository files."""
        if self._is_test_file(rel_fname):
            return []
        return [
            candidate
            for candidate in self._candidate_related_test_paths(rel_fname)
            if candidate in known_rel_fnames and candidate != rel_fname and self._is_test_file(candidate)
        ]

    def _get_path_role_signals(self, rel_fname: str) -> Tuple[List[str], List[str]]:
        """Infer entrypoint/public API signals from the relative path alone."""
        normalized = self._normalize_rel_path(rel_fname).lower()
        path = Path(normalized)
        basename = path.name
        parent_parts = [part.lower() for part in path.parts[:-1]]

        entrypoint_signals = []
        public_api_signals = []

        if basename in self._entrypoint_filenames:
            entrypoint_signals.append("entrypoint_filename")
        if any(part in self._entrypoint_dir_names for part in parent_parts):
            entrypoint_signals.append("entrypoint_directory")

        if basename in self._public_api_filenames:
            public_api_signals.append("public_api_filename")
        if any(part in self._public_api_dir_names for part in parent_parts):
            public_api_signals.append("public_api_directory")

        return entrypoint_signals, public_api_signals

    def _get_runtime_role_metadata(
        self,
        fname: str,
        rel_fname: str,
    ) -> Tuple[List[str], List[str], List[Tag]]:
        """Infer entrypoint/public API signals and synthetic highlight tags from file contents."""
        text = self.read_text_func_internal(fname)
        if not text:
            return [], [], []

        entrypoint_patterns = [
            ("python_main_guard", re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']')),
            ("main_function", re.compile(r'\b(def|func)\s+main\s*\(|\bstatic\s+void\s+main\s*\(')),
            ("server_bootstrap", re.compile(r'\b(app|server)\.listen\s*\(|\buvicorn\.run\s*\(|\bcreateServer\s*\(')),
            ("cli_bootstrap", re.compile(r'\bargparse\.ArgumentParser\b|\bclick\.command\b|\btyper\.Typer\b')),
            ("web_app_factory", re.compile(r'\bFastAPI\s*\(|\bFlask\s*\(|\bexpress\s*\(')),
        ]
        public_api_patterns = [
            ("route_definition", re.compile(r'@\w*app\.route|@\w*router\.(get|post|put|delete|patch)|\b(router|app)\.(get|post|put|delete|patch)\s*\(')),
            ("api_router", re.compile(r'\bAPIRouter\s*\(|\bBlueprint\s*\(')),
            ("explicit_exports", re.compile(r'\bexport\s+(class|function|const|interface|type)\b|\bmodule\.exports\b|\bexports\.\w+\b|\b__all__\b')),
            ("package_reexports", re.compile(r'^\s*from\s+\.[\w\.]+\s+import\s+')),
            ("url_patterns", re.compile(r'\burlpatterns\b')),
        ]

        entrypoint_signals = []
        public_api_signals = []
        tag_specs = []

        for line_num, line in enumerate(text.splitlines(), start=1):
            for signal_name, pattern in entrypoint_patterns:
                if pattern.search(line):
                    entrypoint_signals.append(signal_name)
                    tag_specs.append(("entry", line_num))
            for signal_name, pattern in public_api_patterns:
                if pattern.search(line):
                    public_api_signals.append(signal_name)
                    tag_specs.append(("api", line_num))

        basename = Path(rel_fname).name
        synthetic_tags = [
            Tag(rel_fname=rel_fname, fname=fname, line=line_num, name=basename, kind=kind)
            for kind, line_num in self._dedupe_preserve_order(tag_specs)[:6]
        ]
        return (
            self._dedupe_preserve_order(entrypoint_signals),
            self._dedupe_preserve_order(public_api_signals),
            synthetic_tags,
        )

    def _get_role_rank_context(
        self,
        entrypoint_signals: List[str],
        public_api_signals: List[str],
    ) -> Tuple[float, float]:
        """Return rank floor and boost for entrypoint/public API files."""
        boost = 1.0
        rank_floor = 0.0
        weights = self.repo_config.ranking_weights

        if entrypoint_signals:
            entrypoint_boost = min(1.5 + (0.15 * max(len(entrypoint_signals) - 1, 0)), 2.0)
            entrypoint_floor = 0.05 + (0.01 * min(len(entrypoint_signals), 3))
            boost *= self._scale_multiplier(entrypoint_boost, weights.entrypoint)
            rank_floor = max(rank_floor, self._scale_floor(entrypoint_floor, weights.entrypoint))
        if public_api_signals:
            public_api_boost = min(1.3 + (0.1 * max(len(public_api_signals) - 1, 0)), 1.8)
            public_api_floor = 0.035 + (0.005 * min(len(public_api_signals), 3))
            boost *= self._scale_multiplier(public_api_boost, weights.public_api)
            rank_floor = max(rank_floor, self._scale_floor(public_api_floor, weights.public_api))

        return rank_floor, boost

    def _get_changed_rank_context(self, changed_distance: Optional[int]) -> Tuple[float, float]:
        """Return rank floor and boost for changed files and their nearby impact neighbors."""
        if changed_distance is None:
            return 0.0, 1.0
        weights = self.repo_config.ranking_weights
        if changed_distance == 0:
            return (
                self._scale_floor(0.08, weights.changed_file),
                self._scale_multiplier(8.0, weights.changed_file),
            )
        if changed_distance == 1:
            return (
                self._scale_floor(0.03, weights.changed_neighbor),
                self._scale_multiplier(2.5, weights.changed_neighbor),
            )
        return (
            self._scale_floor(0.015, weights.changed_neighbor),
            self._scale_multiplier(max(1.25, 2.25 - (0.25 * changed_distance)), weights.changed_neighbor),
        )

    @staticmethod
    def _shorten_summary_value(value: str, limit: int = 80) -> str:
        """Normalize and trim summary text for compact map output."""
        compact = " ".join(value.strip().split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _extract_dependency_names(self, values: List[str]) -> List[str]:
        """Extract package names from dependency specifications."""
        names = []
        for value in values:
            item = value.strip()
            if not item:
                continue
            if ";" in item:
                item = item.split(";", 1)[0].strip()
            if "[" in item:
                item = item.split("[", 1)[0].strip()
            match = re.match(r"([A-Za-z0-9_.-]+)", item)
            if match:
                names.append(match.group(1))
        return self._dedupe_preserve_order(names)

    def _extract_markdown_summary(self, text: str) -> List[str]:
        """Summarize Markdown/docs files using headings and the opening paragraph."""
        headings = []
        first_paragraph_lines = []
        in_paragraph = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if in_paragraph and first_paragraph_lines:
                    break
                continue

            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                if heading:
                    headings.append(self._shorten_summary_value(heading, 60))
                continue

            if not in_paragraph:
                in_paragraph = True
            if len(first_paragraph_lines) < 2:
                first_paragraph_lines.append(line)

        items = [f"heading: {heading}" for heading in headings[:4]]
        if first_paragraph_lines:
            items.append(f"overview: {self._shorten_summary_value(' '.join(first_paragraph_lines), 90)}")
        return items[:5]

    def _extract_package_json_summary(self, text: str) -> List[str]:
        """Summarize package.json metadata that helps agents orient quickly."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        items = []
        name = data.get("name")
        version = data.get("version")
        if name:
            items.append(f"package: {name}{'@' + version if version else ''}")

        scripts = list((data.get("scripts") or {}).keys())
        if scripts:
            items.append(f"scripts: {', '.join(scripts[:4])}")

        dependencies = list((data.get("dependencies") or {}).keys())
        if dependencies:
            items.append(f"dependencies: {', '.join(dependencies[:4])}")

        dev_dependencies = list((data.get("devDependencies") or {}).keys())
        if dev_dependencies:
            items.append(f"devDependencies: {', '.join(dev_dependencies[:4])}")

        return items[:5]

    def _extract_pyproject_summary(self, text: str) -> List[str]:
        """Summarize pyproject metadata for Python repos."""
        if tomllib is None:
            return []

        try:
            data = tomllib.loads(text)
        except (tomllib.TOMLDecodeError, TypeError):
            return []

        items = []
        project = data.get("project") or {}
        project_name = project.get("name")
        project_version = project.get("version")
        if project_name:
            items.append(f"project: {project_name}{'@' + project_version if project_version else ''}")

        dependencies = self._extract_dependency_names(project.get("dependencies") or [])
        if dependencies:
            items.append(f"dependencies: {', '.join(dependencies[:4])}")

        scripts = list((project.get("scripts") or {}).keys())
        if scripts:
            items.append(f"scripts: {', '.join(scripts[:4])}")

        build_backend = ((data.get("build-system") or {}).get("build-backend"))
        if build_backend:
            items.append(f"build backend: {build_backend}")

        tool_sections = list((data.get("tool") or {}).keys())
        if tool_sections:
            items.append(f"tooling: {', '.join(tool_sections[:4])}")

        return items[:5]

    def _extract_workflow_summary(self, text: str) -> List[str]:
        """Summarize GitHub Actions workflow metadata with lightweight parsing."""
        items = []
        lines = text.splitlines()

        workflow_name = None
        triggers = []
        jobs = []
        in_on_block = False
        in_jobs_block = False

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if raw_line.startswith("name:") and workflow_name is None:
                workflow_name = raw_line.split(":", 1)[1].strip().strip("'\"")
                continue

            if stripped.startswith("on:"):
                in_on_block = True
                in_jobs_block = False
                inline = stripped.split(":", 1)[1].strip()
                if inline:
                    triggers.extend(re.findall(r"[A-Za-z_]+", inline))
                continue

            if stripped.startswith("jobs:"):
                in_jobs_block = True
                in_on_block = False
                continue

            if in_on_block:
                if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
                    in_on_block = False
                elif re.match(r"\s{2,}[A-Za-z_][A-Za-z0-9_-]*\s*:", raw_line):
                    trigger = raw_line.strip().split(":", 1)[0]
                    triggers.append(trigger)
                    continue

            if in_jobs_block:
                if not raw_line.startswith(" ") and not raw_line.startswith("\t"):
                    in_jobs_block = False
                elif re.match(r"\s{2,}[A-Za-z_][A-Za-z0-9_-]*\s*:", raw_line):
                    job_name = raw_line.strip().split(":", 1)[0]
                    jobs.append(job_name)

        if workflow_name:
            items.append(f"workflow: {workflow_name}")
        if triggers:
            items.append(f"triggers: {', '.join(self._dedupe_preserve_order(triggers)[:4])}")
        if jobs:
            items.append(f"jobs: {', '.join(self._dedupe_preserve_order(jobs)[:4])}")
        return items[:5]

    def _extract_dockerfile_summary(self, text: str) -> List[str]:
        """Summarize Dockerfile base image and runtime commands."""
        items = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("FROM ") and not any(item.startswith("base image:") for item in items):
                items.append(f"base image: {self._shorten_summary_value(line[5:], 60)}")
            elif upper.startswith("EXPOSE ") and not any(item.startswith("exposes:") for item in items):
                items.append(f"exposes: {self._shorten_summary_value(line[7:], 40)}")
            elif upper.startswith("ENTRYPOINT ") and not any(item.startswith("entrypoint:") for item in items):
                items.append(f"entrypoint: {self._shorten_summary_value(line[11:], 60)}")
            elif upper.startswith("CMD ") and not any(item.startswith("cmd:") for item in items):
                items.append(f"cmd: {self._shorten_summary_value(line[4:], 60)}")
            if len(items) >= 4:
                break
        return items[:5]

    def _extract_requirements_summary(self, text: str) -> List[str]:
        """Summarize plain dependency lists like requirements.txt."""
        deps = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-r"):
                continue
            deps.append(line)
            if len(deps) >= 8:
                break
        dep_names = self._extract_dependency_names(deps)
        if not dep_names:
            return []
        return [f"dependencies: {', '.join(dep_names[:6])}"]

    def _extract_file_summary(self, fname: str, rel_fname: str) -> Tuple[Optional[str], List[str]]:
        """Extract deterministic summary bullets for important docs/config files."""
        text = self.read_text_func_internal(fname)
        if not text:
            return None, []

        normalized = self._normalize_rel_path(rel_fname).lower()
        basename = Path(rel_fname).name.lower()

        if basename.endswith((".md", ".rst", ".txt")) or basename.startswith("readme") or normalized.startswith("docs/"):
            return "doc", self._extract_markdown_summary(text)
        if basename == "package.json":
            return "config", self._extract_package_json_summary(text)
        if basename == "pyproject.toml":
            return "config", self._extract_pyproject_summary(text)
        if normalized.startswith(".github/workflows/") and basename.endswith((".yml", ".yaml")):
            return "config", self._extract_workflow_summary(text)
        if basename == "dockerfile" or basename.startswith("dockerfile."):
            return "config", self._extract_dockerfile_summary(text)
        if basename in {"requirements.txt", "pdm.lock"}:
            return "config", self._extract_requirements_summary(text)

        return None, []

    @staticmethod
    def _index_tag_lines(
        tags_by_rel_fname: Dict[str, List[Tag]],
    ) -> Tuple[Dict[Tuple[str, str], List[int]], Dict[Tuple[str, str], List[int]]]:
        """Index definition and reference lines by file/symbol for trace annotations."""
        def_lines: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        ref_lines: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        for rel_fname, tags in tags_by_rel_fname.items():
            for tag in tags:
                if tag.kind == "def":
                    def_lines[(rel_fname, tag.name)].append(tag.line)
                elif tag.kind == "ref":
                    ref_lines[(rel_fname, tag.name)].append(tag.line)

        return def_lines, ref_lines

    @staticmethod
    def _make_symbol_trace_hop_from_semantic_link(
        link: SemanticLink,
        def_lines: Dict[Tuple[str, str], List[int]],
    ) -> SymbolTraceHop:
        """Convert a semantic import/export edge into trace metadata."""
        target_line = None
        if link.target_symbol and link.target_symbol != "*":
            target_candidates = def_lines.get((link.target, link.target_symbol), [])
            if target_candidates:
                target_line = min(target_candidates)

        evidence_kind = "import_export"
        if link.relation == "imports":
            evidence_kind = "import"
        elif link.relation == "re_exports":
            evidence_kind = "re_export"
        elif link.relation == "package_reexports":
            evidence_kind = "package_boundary"

        return SymbolTraceHop(
            source_file=link.source,
            target_file=link.target,
            relation=link.relation,
            source_symbol=link.source_symbol,
            target_symbol=link.target_symbol,
            source_line=link.line,
            target_line=target_line,
            evidence_kind=evidence_kind,
            detail=link.detail,
        )

    def _build_file_reference_graph(
        self,
        fnames: List[str],
    ) -> Tuple[
        Dict[str, str],
        Dict[str, List[Tag]],
        nx.DiGraph,
        Dict[Tuple[str, str], List[str]],
        Dict[Tuple[str, str], List[SymbolTraceHop]],
    ]:
        """Build a file-level reference graph and edge labels from parser tags."""
        all_fnames = list(dict.fromkeys(os.path.abspath(fname) for fname in fnames))
        abs_to_rel = {fname: self.get_rel_fname(fname) for fname in all_fnames}
        known_rel_fnames = set(abs_to_rel.values())
        defines = defaultdict(set)
        references = defaultdict(set)
        edge_symbols = defaultdict(set)
        edge_details = defaultdict(list)
        tags_by_rel_fname = {}

        for fname in all_fnames:
            rel_fname = abs_to_rel[fname]
            if not os.path.exists(fname):
                continue

            tags = self.get_tags(fname, rel_fname)
            if self._tag_failures.get(fname):
                continue
            tags_by_rel_fname[rel_fname] = tags

            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                elif tag.kind == "ref":
                    references[tag.name].add(rel_fname)

        def_lines, ref_lines = self._index_tag_lines(tags_by_rel_fname)

        graph = nx.DiGraph()
        graph.add_nodes_from(abs_to_rel.values())

        for name, ref_fnames in references.items():
            def_fnames = defines.get(name, set())
            if len(ref_fnames) * len(def_fnames) > 1000:
                continue
            for ref_fname in ref_fnames:
                for def_fname in def_fnames:
                    if ref_fname == def_fname:
                        continue
                    graph.add_edge(ref_fname, def_fname)
                    edge_symbols[(ref_fname, def_fname)].add(name)
                    edge_details[(ref_fname, def_fname)].append(
                        SymbolTraceHop(
                            source_file=ref_fname,
                            target_file=def_fname,
                            relation="references",
                            source_symbol=name,
                            target_symbol=name,
                            source_line=min(ref_lines.get((ref_fname, name), [None])),
                            target_line=min(def_lines.get((def_fname, name), [None])),
                            evidence_kind="callsite",
                            detail=f"Reference to {name}",
                        )
                    )

        for rel_fname in known_rel_fnames:
            abs_fname = next((fname for fname, known_rel in abs_to_rel.items() if known_rel == rel_fname), None)
            if not abs_fname:
                continue
            text = self.read_text_func_internal(abs_fname)
            if not text:
                continue

            for semantic_link in collect_semantic_links(rel_fname, text, known_rel_fnames):
                graph.add_edge(semantic_link.source, semantic_link.target)
                if semantic_link.target_symbol and semantic_link.target_symbol != "*":
                    edge_symbols[(semantic_link.source, semantic_link.target)].add(semantic_link.target_symbol)
                elif semantic_link.source_symbol and semantic_link.source_symbol != "*":
                    edge_symbols[(semantic_link.source, semantic_link.target)].add(semantic_link.source_symbol)
                edge_details[(semantic_link.source, semantic_link.target)].append(
                    self._make_symbol_trace_hop_from_semantic_link(semantic_link, def_lines)
                )

        for rel_fname in known_rel_fnames:
            if self._is_test_file(rel_fname):
                continue
            for test_rel_fname in self._find_related_test_files(rel_fname, known_rel_fnames):
                graph.add_edge(rel_fname, test_rel_fname)
                graph.add_edge(test_rel_fname, rel_fname)
                edge_symbols[(rel_fname, test_rel_fname)].add("__related_test__")
                edge_symbols[(test_rel_fname, rel_fname)].add("__related_source__")
                edge_details[(rel_fname, test_rel_fname)].append(
                    SymbolTraceHop(
                        source_file=rel_fname,
                        target_file=test_rel_fname,
                        relation="related_test",
                        evidence_kind="test_link",
                        detail="Heuristic source-to-test link",
                    )
                )
                edge_details[(test_rel_fname, rel_fname)].append(
                    SymbolTraceHop(
                        source_file=test_rel_fname,
                        target_file=rel_fname,
                        relation="related_source",
                        evidence_kind="test_link",
                        detail="Heuristic test-to-source link",
                    )
                )

        return abs_to_rel, tags_by_rel_fname, graph, {
            pair: sorted(symbols)
            for pair, symbols in edge_symbols.items()
        }, {
            pair: details[:8]
            for pair, details in edge_details.items()
        }

    def trace_file_path(
        self,
        start_file: str,
        end_file: str,
        files: Optional[List[str]] = None,
        max_hops: int = 6,
    ) -> ConnectionReport:
        """Find a shortest file-level path between two files using the repo graph."""
        self._reset_request_state()

        candidate_files = files or self._find_repo_files()
        all_fnames = self._prepare_candidate_files(candidate_files)
        abs_to_rel = {fname: self.get_rel_fname(fname) for fname in all_fnames}

        start_abs = os.path.abspath(start_file)
        end_abs = os.path.abspath(end_file)
        start_rel = abs_to_rel.get(start_abs, self.get_rel_fname(start_abs))
        end_rel = abs_to_rel.get(end_abs, self.get_rel_fname(end_abs))

        if start_abs not in abs_to_rel:
            return ConnectionReport(
                start_file=start_rel,
                end_file=end_rel,
                diagnostics=list(self.diagnostics),
                error=f"Start file is not in the selected repository scope: {start_rel}",
            )
        if end_abs not in abs_to_rel:
            return ConnectionReport(
                start_file=start_rel,
                end_file=end_rel,
                diagnostics=list(self.diagnostics),
                error=f"End file is not in the selected repository scope: {end_rel}",
            )

        _, _, graph, edge_symbols, edge_details = self._build_file_reference_graph(all_fnames)
        undirected_graph = graph.to_undirected()

        if start_rel not in undirected_graph or end_rel not in undirected_graph:
            return ConnectionReport(
                start_file=start_rel,
                end_file=end_rel,
                diagnostics=list(self.diagnostics),
                error="One or both files could not be added to the repository graph.",
            )

        try:
            path = nx.shortest_path(undirected_graph, start_rel, end_rel)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return ConnectionReport(
                start_file=start_rel,
                end_file=end_rel,
                diagnostics=list(self.diagnostics),
                error=f"No file-level path found between {start_rel} and {end_rel}.",
            )

        hop_count = max(0, len(path) - 1)
        if hop_count > max_hops:
            return ConnectionReport(
                start_file=start_rel,
                end_file=end_rel,
                path=path,
                diagnostics=list(self.diagnostics),
                error=f"Shortest path is {hop_count} hops, which exceeds the max_hops limit of {max_hops}.",
            )

        steps = self._build_connection_steps(path, graph, edge_symbols, edge_details)
        symbol_path = self._flatten_symbol_path(steps)

        diagnostics = list(self.diagnostics)
        if hop_count:
            diagnostics.append(f"Found a {hop_count}-hop path between {start_rel} and {end_rel}.")

        return ConnectionReport(
            start_file=start_rel,
            end_file=end_rel,
            path=path,
            steps=steps,
            symbol_path=symbol_path,
            diagnostics=diagnostics,
        )

    def _extract_boundary_symbols(self, steps: List[ConnectionStep]) -> List[str]:
        """Collect the concrete non-synthetic symbols that connect an impact path."""
        symbols = []
        for step in steps:
            for symbol in step.symbols:
                if symbol.startswith("__related_"):
                    continue
                if symbol not in symbols:
                    symbols.append(symbol)
        return symbols[:8]

    def _build_boundary_locations(
        self,
        path: List[str],
        tags_by_rel_fname: Dict[str, List[Tag]],
        boundary_symbols: List[str],
    ) -> List[ImpactLocation]:
        """Collect concrete tag locations for the symbols that define the impact path."""
        symbol_set = set(boundary_symbols)
        if not symbol_set:
            return []

        locations = []
        seen = set()
        for rel_fname in path:
            for tag in tags_by_rel_fname.get(rel_fname, []):
                if tag.kind not in {"def", "ref"} or tag.name not in symbol_set:
                    continue
                key = (tag.rel_fname, tag.line, tag.kind, tag.name)
                if key in seen:
                    continue
                seen.add(key)
                locations.append(
                    ImpactLocation(
                        file=tag.rel_fname,
                        line=tag.line,
                        kind=tag.kind,
                        symbol=tag.name,
                    )
                )
        return locations[:12]

    def _build_boundary_snippets(
        self,
        rel_to_abs: Dict[str, str],
        boundary_locations: List[ImpactLocation],
        context_lines: int = 1,
    ) -> List[ImpactSnippet]:
        """Extract short code excerpts around important boundary locations."""
        snippets = []
        seen = set()

        for location in boundary_locations:
            key = (location.file, location.line, location.kind, location.symbol)
            if key in seen:
                continue
            seen.add(key)

            abs_fname = rel_to_abs.get(location.file)
            if not abs_fname:
                continue
            text = self.read_text_func_internal(abs_fname)
            if not text:
                continue

            lines = text.splitlines()
            if not lines:
                continue

            highlight_index = min(max(location.line - 1, 0), len(lines) - 1)
            start_index = max(0, highlight_index - context_lines)
            end_index = min(len(lines), highlight_index + context_lines + 1)
            excerpt_lines = [
                f"{line_no}: {lines[line_no - 1]}"
                for line_no in range(start_index + 1, end_index + 1)
            ]
            snippets.append(
                ImpactSnippet(
                    file=location.file,
                    start_line=start_index + 1,
                    end_line=end_index,
                    highlight_line=location.line,
                    kind=location.kind,
                    symbol=location.symbol,
                    excerpt="\n".join(excerpt_lines),
                )
            )

        return snippets[:6]

    @staticmethod
    def _group_changed_lines_into_hunks(changed_lines: List[int]) -> List[ImpactHunk]:
        """Collapse raw changed lines into contiguous diff hunk ranges."""
        if not changed_lines:
            return []

        sorted_lines = sorted(set(changed_lines))
        hunks = []
        start_line = sorted_lines[0]
        end_line = sorted_lines[0]

        for line in sorted_lines[1:]:
            if line == end_line + 1:
                end_line = line
                continue
            hunks.append(ImpactHunk(start_line=start_line, end_line=end_line))
            start_line = line
            end_line = line

        hunks.append(ImpactHunk(start_line=start_line, end_line=end_line))
        return hunks

    @staticmethod
    def _line_distance_to_hunks(line: int, hunks: List[ImpactHunk]) -> Optional[int]:
        """Return the closest line distance from a symbol to any changed hunk."""
        if not hunks:
            return None

        distances = []
        for hunk in hunks:
            if hunk.start_line <= line <= hunk.end_line:
                distances.append(0)
            elif line < hunk.start_line:
                distances.append(hunk.start_line - line)
            else:
                distances.append(line - hunk.end_line)
        return min(distances) if distances else None

    @staticmethod
    def _format_hunk(hunk: ImpactHunk) -> str:
        """Render a compact human-readable hunk label."""
        if hunk.start_line == hunk.end_line:
            return str(hunk.start_line)
        return f"{hunk.start_line}-{hunk.end_line}"

    def _extract_changed_symbol_metadata(
        self,
        tags: List[Tag],
        changed_lines: List[int],
    ) -> Tuple[List[str], Dict[str, int], List[ImpactHunk]]:
        """Infer which symbols are closest to the actually changed lines in a seed file."""
        hunks = self._group_changed_lines_into_hunks(changed_lines)
        if not tags or not changed_lines:
            return [], {}, hunks

        candidate_tags = [tag for tag in tags if tag.kind in {"def", "ref"}]
        if not candidate_tags:
            return [], {}, hunks

        def collect_within(distance_limit: int) -> Tuple[List[str], Dict[str, int]]:
            symbols = []
            distances = {}
            for tag in candidate_tags:
                distance = self._line_distance_to_hunks(tag.line, hunks)
                if distance is None or distance > distance_limit:
                    continue
                if tag.name not in symbols:
                    symbols.append(tag.name)
                current = distances.get(tag.name)
                if current is None or distance < current:
                    distances[tag.name] = distance
            return symbols, distances

        symbols, distances = collect_within(0)
        if not symbols:
            symbols, distances = collect_within(2)
        if not symbols:
            symbols, distances = collect_within(8)
        return symbols[:8], distances, hunks

    @staticmethod
    def _choose_step_relation(details: List[SymbolTraceHop], default_relation: str) -> str:
        """Prefer more semantic edge relations when rendering a trace step."""
        priority = {
            "package_reexports": 0,
            "re_exports": 1,
            "imports": 2,
            "references": 3,
            "related_test": 4,
            "related_source": 5,
        }
        if not details:
            return default_relation
        best = min(details, key=lambda item: priority.get(item.relation, 9))
        return best.relation

    @staticmethod
    def _flatten_symbol_path(steps: List[ConnectionStep]) -> List[SymbolTraceHop]:
        """Expose a flattened symbol-level trace alongside step-level annotations."""
        symbol_path = []
        seen = set()
        for step in steps:
            for hop in step.symbol_hops:
                key = (
                    hop.source_file,
                    hop.target_file,
                    hop.relation,
                    hop.source_symbol,
                    hop.target_symbol,
                    hop.source_line,
                    hop.target_line,
                    hop.detail,
                )
                if key in seen:
                    continue
                seen.add(key)
                symbol_path.append(hop)
        return symbol_path[:16]

    def _build_connection_steps(
        self,
        path: List[str],
        graph: nx.DiGraph,
        edge_symbols: Dict[Tuple[str, str], List[str]],
        edge_details: Dict[Tuple[str, str], List[SymbolTraceHop]],
    ) -> List[ConnectionStep]:
        """Convert a path into directional step metadata."""
        steps = []
        for source, target in zip(path, path[1:]):
            if graph.has_edge(source, target):
                symbols = edge_symbols.get((source, target), [])
                details = edge_details.get((source, target), [])
                relation = self._choose_step_relation(details, "references")
            elif graph.has_edge(target, source):
                symbols = edge_symbols.get((target, source), [])
                details = edge_details.get((target, source), [])
                relation = "referenced_by"
            else:
                symbols = []
                details = []
                relation = "related"

            if "__related_test__" in symbols:
                relation = "related_test"
                symbols = []
            elif "__related_source__" in symbols:
                relation = "related_source"
                symbols = []

            step_hops = []
            for hop in details:
                step_hops.append(
                    SymbolTraceHop(
                        source_file=source,
                        target_file=target,
                        relation=hop.relation,
                        source_symbol=hop.source_symbol,
                        target_symbol=hop.target_symbol,
                        source_line=hop.source_line,
                        target_line=hop.target_line,
                        evidence_kind=hop.evidence_kind,
                        detail=hop.detail,
                    )
                )

            steps.append(ConnectionStep(source=source, target=target, relation=relation, symbols=symbols[:5], symbol_hops=step_hops[:6]))
        return steps

    def analyze_file_impact(
        self,
        seed_files: List[str],
        files: Optional[List[str]] = None,
        max_depth: int = 2,
        max_results: int = 10,
        changed_lines_by_file: Optional[Dict[str, List[int]]] = None,
    ) -> ImpactReport:
        """Find nearby repository files most likely to be affected by the given seed files."""
        self._reset_request_state()

        seed_files = list(dict.fromkeys(seed_files or []))
        resolved_seed_files = [self.get_rel_fname(os.path.abspath(seed_file)) for seed_file in seed_files]
        report = ImpactReport(
            seed_files=resolved_seed_files,
            max_depth=max_depth,
            max_results=max_results,
        )

        if not seed_files:
            report.error = "At least one seed file is required for impact analysis."
            return report
        if max_depth < 1:
            report.error = "max_depth must be at least 1."
            return report
        if max_results < 1:
            report.error = "max_results must be at least 1."
            return report

        candidate_files = files or self._find_repo_files()
        all_fnames = self._prepare_candidate_files(candidate_files)
        abs_to_rel = {fname: self.get_rel_fname(fname) for fname in all_fnames}
        rel_to_abs = {rel_fname: fname for fname, rel_fname in abs_to_rel.items()}
        changed_lines_by_rel = {}
        for fname, line_numbers in (changed_lines_by_file or {}).items():
            abs_fname = os.path.abspath(fname)
            rel_fname = abs_to_rel.get(abs_fname)
            if not rel_fname or not line_numbers:
                continue
            changed_lines_by_rel[rel_fname] = sorted(set(int(line) for line in line_numbers if int(line) > 0))
        report.changed_lines_by_file = {
            rel_fname: line_numbers[:20]
            for rel_fname, line_numbers in changed_lines_by_rel.items()
        }
        changed_hunks_by_rel = {
            rel_fname: self._group_changed_lines_into_hunks(line_numbers)
            for rel_fname, line_numbers in changed_lines_by_rel.items()
        }
        report.changed_hunks_by_file = {
            rel_fname: hunks[:8]
            for rel_fname, hunks in changed_hunks_by_rel.items()
            if hunks
        }

        seed_abs_files = [os.path.abspath(seed_file) for seed_file in seed_files]
        missing_seed_files = [
            abs_to_rel.get(seed_abs, self.get_rel_fname(seed_abs))
            for seed_abs in seed_abs_files
            if seed_abs not in abs_to_rel
        ]
        if missing_seed_files:
            report.error = (
                "Seed file(s) are not in the selected repository scope: "
                + ", ".join(missing_seed_files)
            )
            report.diagnostics = list(self.diagnostics)
            return report

        _, tags_by_rel_fname, graph, edge_symbols, edge_details = self._build_file_reference_graph(all_fnames)
        undirected_graph = graph.to_undirected()

        seed_rel_files = [abs_to_rel[seed_abs] for seed_abs in seed_abs_files]
        missing_graph_files = [
            rel_fname for rel_fname in seed_rel_files
            if rel_fname not in undirected_graph
        ]
        if missing_graph_files:
            report.error = (
                "Seed file(s) could not be added to the repository graph: "
                + ", ".join(missing_graph_files)
            )
            report.diagnostics = list(self.diagnostics)
            return report

        candidate_targets = {}
        seed_rel_set = set(seed_rel_files)
        changed_seed_metadata_by_rel = {
            rel_fname: self._extract_changed_symbol_metadata(
                tags_by_rel_fname.get(rel_fname, []),
                changed_lines_by_rel.get(rel_fname, []),
            )
            for rel_fname in seed_rel_files
            if changed_lines_by_rel.get(rel_fname)
        }
        report.changed_seed_symbols = {
            rel_fname: symbols
            for rel_fname, (symbols, _, _) in changed_seed_metadata_by_rel.items()
            if symbols
        }
        for seed_rel in seed_rel_files:
            try:
                seed_paths = nx.single_source_shortest_path(
                    undirected_graph,
                    seed_rel,
                    cutoff=max_depth,
                )
            except nx.NodeNotFound:
                continue

            for target_rel, path in seed_paths.items():
                if target_rel in seed_rel_set or len(path) < 2:
                    continue
                distance = len(path) - 1
                current = candidate_targets.get(target_rel)
                candidate = (distance, seed_rel, path)
                if current is None or candidate < current:
                    candidate_targets[target_rel] = candidate

        impacted_files = []
        for target_rel, (distance, seed_rel, path) in candidate_targets.items():
            target_abs = rel_to_abs.get(target_rel)
            if not target_abs:
                continue

            steps = self._build_connection_steps(path, graph, edge_symbols, edge_details)
            symbol_path = self._flatten_symbol_path(steps)
            seed_focus_lines = changed_lines_by_rel.get(seed_rel, [])
            seed_hunks = changed_hunks_by_rel.get(seed_rel, [])
            changed_seed_symbols, changed_seed_symbol_distances, _ = changed_seed_metadata_by_rel.get(
                seed_rel,
                ([], {}, []),
            )
            boundary_symbols = self._extract_boundary_symbols(steps)
            changed_boundary_symbols = [
                symbol for symbol in boundary_symbols
                if symbol in changed_seed_symbols
            ]
            changed_boundary_distances = {
                symbol: changed_seed_symbol_distances[symbol]
                for symbol in changed_boundary_symbols
                if symbol in changed_seed_symbol_distances
            }
            closest_changed_hunk_distance = min(changed_boundary_distances.values()) if changed_boundary_distances else None
            boundary_locations = self._build_boundary_locations(path, tags_by_rel_fname, boundary_symbols)
            boundary_snippets = self._build_boundary_snippets(rel_to_abs, boundary_locations)
            boundary_relations = []
            for step in steps:
                if step.relation not in boundary_relations:
                    boundary_relations.append(step.relation)
            focus_lines = sorted({
                location.line
                for location in boundary_locations
                if location.file == target_rel
            })
            is_test_file = self._is_test_file(target_rel)
            entrypoint_signals, public_api_signals = self._get_path_role_signals(target_rel)
            is_entrypoint_file = bool(entrypoint_signals)
            is_public_api_file = bool(public_api_signals)
            is_important_file = self._is_important_file(target_rel)
            summary_kind, summary_items = self._extract_file_summary(target_abs, target_rel)

            relations_preview = " -> ".join(step.relation for step in steps[:4])
            reasons = [
                RankingReason(
                    "impact_path",
                    f"Reachable from {seed_rel} in {distance} hop(s).",
                ),
            ]
            if relations_preview:
                reasons.append(
                    RankingReason(
                        "impact_relations",
                        f"Path relations: {relations_preview}.",
                    )
                )
            symbol_preview = [
                symbol
                for step in steps
                for symbol in step.symbols[:2]
            ]
            if symbol_preview:
                reasons.append(
                    RankingReason(
                        "impact_symbols",
                        f"Connected via symbols such as: {', '.join(symbol_preview[:4])}.",
                    )
                )
            if changed_boundary_symbols:
                hunk_preview = ", ".join(self._format_hunk(hunk) for hunk in seed_hunks[:2]) if seed_hunks else ""
                distance_preview = min(changed_boundary_distances.values()) if changed_boundary_distances else 0
                reasons.append(
                    RankingReason(
                        "changed_symbol_boundary",
                        f"Touches symbol(s) changed in {seed_rel}: {', '.join(changed_boundary_symbols[:4])}.",
                    )
                )
                reasons.append(
                    RankingReason(
                        "changed_hunk_proximity",
                        (
                            f"Closest changed hunk"
                            + (f" ({hunk_preview})" if hunk_preview else "")
                            + f" is {distance_preview} line(s) away from the matched boundary."
                        ),
                    )
                )
            if is_test_file:
                reasons.append(
                    RankingReason(
                        "impact_test",
                        f"Looks like nearby validation coverage for {seed_rel}.",
                    )
                )
            if is_entrypoint_file:
                reasons.append(
                    RankingReason(
                        "impact_entrypoint",
                        f"Looks like an application entrypoint via: {', '.join(entrypoint_signals[:3])}.",
                    )
                )
            if is_public_api_file:
                reasons.append(
                    RankingReason(
                        "impact_public_api",
                        f"Looks like a public API surface via: {', '.join(public_api_signals[:3])}.",
                    )
                )
            if is_important_file:
                reasons.append(
                    RankingReason(
                        "impact_important_file",
                        "Looks like an important project file.",
                    )
                )
            if summary_items:
                summary_code = "impact_doc_summary" if summary_kind == "doc" else "impact_config_summary"
                reasons.append(
                    RankingReason(
                        summary_code,
                        f"Structured summary extracted: {', '.join(summary_items[:2])}.",
                    )
                )

            impacted_files.append(
                ImpactTarget(
                    path=target_rel,
                    seed_file=seed_rel,
                    distance=distance,
                    path_from_seed=path,
                    steps=steps,
                    symbol_path=symbol_path,
                    seed_focus_lines=seed_focus_lines[:12],
                    seed_hunks=seed_hunks[:6],
                    changed_boundary_symbols=changed_boundary_symbols,
                    changed_boundary_distances=changed_boundary_distances,
                    closest_changed_hunk_distance=closest_changed_hunk_distance,
                    boundary_symbols=boundary_symbols,
                    boundary_relations=boundary_relations[:5],
                    boundary_locations=boundary_locations,
                    boundary_snippets=boundary_snippets,
                    focus_lines=focus_lines[:8],
                    is_test_file=is_test_file,
                    is_entrypoint_file=is_entrypoint_file,
                    is_public_api_file=is_public_api_file,
                    is_important_file=is_important_file,
                    summary_kind=summary_kind,
                    summary_items=summary_items[:5],
                    reasons=reasons,
                )
            )

        impacted_files.sort(
            key=lambda item: (
                -(1 if item.changed_boundary_symbols else 0),
                item.closest_changed_hunk_distance if item.closest_changed_hunk_distance is not None else 999,
                -len(item.changed_boundary_symbols),
                item.distance,
                -(
                    (3 if item.is_test_file else 0)
                    + (2 if item.is_public_api_file else 0)
                    + (2 if item.is_entrypoint_file else 0)
                    + (1 if item.is_important_file else 0)
                    + (1 if item.summary_items else 0)
                ),
                item.path,
            )
        )
        report.impacted_files = impacted_files[:max_results]
        report.shared_symbols = self._build_impact_symbols(
            report.impacted_files,
            report.changed_seed_symbols,
        )
        report.suggested_checks = self._build_impact_suggestions(seed_rel_files, report.impacted_files)
        report.quick_actions = self._build_impact_quick_actions(report.impacted_files, report.suggested_checks)
        report.edit_candidates = self._build_impact_edit_candidates(report.impacted_files, report.quick_actions)
        report.edit_plan = self._build_impact_edit_plan(report.quick_actions, report.edit_candidates)
        report.test_clusters = self._build_impact_test_clusters(report.impacted_files)

        diagnostics = list(self.diagnostics)
        if report.impacted_files:
            diagnostics.append(
                f"Found {len(report.impacted_files)} impacted file(s) within {max_depth} hop(s) of {', '.join(seed_rel_files)}."
            )
        else:
            diagnostics.append(
                f"No impacted files were found within {max_depth} hop(s) of {', '.join(seed_rel_files)}."
            )
        report.diagnostics = diagnostics
        return report

    def _classify_review_target_role(
        self,
        *,
        is_test_file: bool,
        is_entrypoint_file: bool,
        is_public_api_file: bool,
        summary_kind: Optional[str],
    ) -> str:
        """Classify a changed file into the review lane it most belongs to."""
        if is_test_file:
            return "test"
        if summary_kind == "config":
            return "config"
        if is_public_api_file:
            return "public_api"
        if is_entrypoint_file:
            return "entrypoint"
        return "boundary"

    def _build_review_changed_files(
        self,
        changed_rel_files: List[str],
        rel_to_abs: Dict[str, str],
        known_rel_fnames: Set[str],
        changed_lines_by_file: Dict[str, List[int]],
        changed_hunks_by_file: Dict[str, List["ImpactHunk"]],
        changed_seed_symbols: Dict[str, List[str]],
    ) -> List[ReviewChangedFile]:
        """Summarize the changed files that anchor a review session."""
        changed_files = []
        for rel_fname in changed_rel_files:
            abs_fname = rel_to_abs.get(rel_fname)
            if not abs_fname:
                continue

            path_entrypoint_signals, path_public_api_signals = self._get_path_role_signals(rel_fname)
            runtime_entrypoint_signals, runtime_public_api_signals, _ = self._get_runtime_role_metadata(
                abs_fname,
                rel_fname,
            )
            entrypoint_signals = self._dedupe_preserve_order(path_entrypoint_signals + runtime_entrypoint_signals)
            public_api_signals = self._dedupe_preserve_order(path_public_api_signals + runtime_public_api_signals)
            is_test_file = self._is_test_file(rel_fname)
            summary_kind, summary_items = self._extract_file_summary(abs_fname, rel_fname)
            related_tests = self._find_related_test_files(rel_fname, known_rel_fnames)
            changed_files.append(
                ReviewChangedFile(
                    path=rel_fname,
                    target_role=self._classify_review_target_role(
                        is_test_file=is_test_file,
                        is_entrypoint_file=bool(entrypoint_signals),
                        is_public_api_file=bool(public_api_signals),
                        summary_kind=summary_kind,
                    ),
                    changed_lines=changed_lines_by_file.get(rel_fname, [])[:20],
                    changed_hunks=changed_hunks_by_file.get(rel_fname, [])[:8],
                    changed_symbols=changed_seed_symbols.get(rel_fname, [])[:8],
                    related_tests=related_tests[:5],
                    entrypoint_signals=entrypoint_signals[:5],
                    public_api_signals=public_api_signals[:5],
                    summary_kind=summary_kind,
                    summary_items=summary_items[:5],
                    is_test_file=is_test_file,
                    is_entrypoint_file=bool(entrypoint_signals),
                    is_public_api_file=bool(public_api_signals),
                    is_important_file=self._is_important_file(rel_fname),
                )
            )

        role_priority = {"public_api": 0, "entrypoint": 1, "config": 2, "test": 3, "boundary": 4}
        changed_files.sort(key=lambda item: (role_priority.get(item.target_role, 9), item.path))
        return changed_files

    def _build_review_focus(
        self,
        changed_files: List[ReviewChangedFile],
        impact_report: ImpactReport,
    ) -> List[ReviewFocusItem]:
        """Build a review-first queue that combines changed surfaces and impact heuristics."""
        focus_items = []
        seen = set()

        def add_focus(
            *,
            priority: int,
            kind: str,
            title: str,
            target: str,
            target_role: str,
            message: str,
            risk_level: str = "low",
            confidence: float = 0.5,
            location_hint: Optional[str] = None,
            command_hint: Optional[str] = None,
            focus_symbols: Optional[List[str]] = None,
            why_now: Optional[str] = None,
            expected_outcome: Optional[str] = None,
            follow_if_true: Optional[str] = None,
            follow_if_false: Optional[str] = None,
            seed_file: Optional[str] = None,
            path_from_seed: Optional[List[str]] = None,
            anchor_file: Optional[str] = None,
            anchor_line: Optional[int] = None,
            anchor_symbol: Optional[str] = None,
            anchor_kind: Optional[str] = None,
        ) -> None:
            key = (kind, target)
            if key in seen:
                return
            seen.add(key)
            focus_items.append(
                ReviewFocusItem(
                    priority=priority,
                    kind=kind,
                    title=title,
                    target=target,
                    target_role=target_role,
                    message=message,
                    risk_level=risk_level,
                    confidence=round(confidence, 2),
                    location_hint=location_hint,
                    command_hint=command_hint,
                    focus_symbols=(focus_symbols or [])[:3],
                    why_now=why_now,
                    expected_outcome=expected_outcome,
                    follow_if_true=follow_if_true,
                    follow_if_false=follow_if_false,
                    seed_file=seed_file,
                    path_from_seed=(path_from_seed or [])[:],
                    anchor_file=anchor_file,
                    anchor_line=anchor_line,
                    anchor_symbol=anchor_symbol,
                    anchor_kind=anchor_kind,
                )
            )

        for changed_file in changed_files:
            location_hint = None
            anchor_line = changed_file.changed_lines[0] if changed_file.changed_lines else None
            if anchor_line:
                location_hint = f"{changed_file.path}:{anchor_line}"
            focus_symbols = changed_file.changed_symbols[:3]

            if changed_file.is_public_api_file:
                add_focus(
                    priority=0,
                    kind="review_changed_public_api",
                    title="Check changed public API",
                    target=changed_file.path,
                    target_role="public_api",
                    message="This change lands on a public surface that can fan out quickly across callers.",
                    risk_level="medium",
                    confidence=0.94,
                    location_hint=location_hint or changed_file.path,
                    focus_symbols=focus_symbols,
                    why_now="Public API changes are the fastest way to spot downstream contract risk.",
                    expected_outcome="Confirm whether the exported signature, route shape, or boundary behavior changed.",
                    follow_if_true="If the contract changed, review impacted callers, tests, and entrypoints next.",
                    follow_if_false="If the contract stayed stable, move to the closest impacted boundary or test.",
                    anchor_file=changed_file.path,
                    anchor_line=anchor_line,
                    anchor_symbol=focus_symbols[0] if focus_symbols else None,
                    anchor_kind="file",
                )
            if changed_file.is_entrypoint_file:
                add_focus(
                    priority=0,
                    kind="review_changed_entrypoint",
                    title="Check changed entrypoint",
                    target=changed_file.path,
                    target_role="entrypoint",
                    message="Entrypoint changes can redirect execution before downstream impact becomes obvious.",
                    risk_level="medium",
                    confidence=0.9,
                    location_hint=location_hint or changed_file.path,
                    focus_symbols=focus_symbols,
                    why_now="Execution-flow changes are easiest to validate at the boundary where they begin.",
                    expected_outcome="Confirm whether the startup, CLI, server, or routing flow now reaches different code paths.",
                    follow_if_true="If the flow changed, inspect the nearest impacted boundary and related tests next.",
                    follow_if_false="If the flow is unchanged, continue with the highest-confidence impacted boundary.",
                    anchor_file=changed_file.path,
                    anchor_line=anchor_line,
                    anchor_symbol=focus_symbols[0] if focus_symbols else None,
                    anchor_kind="file",
                )
            if changed_file.summary_kind == "config":
                add_focus(
                    priority=1,
                    kind="review_changed_config",
                    title="Check changed config",
                    target=changed_file.path,
                    target_role="config",
                    message="Config drift can invalidate the rest of the review path quickly.",
                    risk_level="medium",
                    confidence=0.82,
                    location_hint=location_hint or changed_file.path,
                    focus_symbols=focus_symbols,
                    why_now="A config change can explain multiple downstream effects at once.",
                    expected_outcome="Confirm whether configuration changes alter tests, entrypoints, or boundary assumptions.",
                    follow_if_true="If config changed behavior, review dependent entrypoints and tests immediately.",
                    follow_if_false="If config is neutral, continue with the closest impacted code boundary.",
                    anchor_file=changed_file.path,
                    anchor_line=anchor_line,
                    anchor_symbol=focus_symbols[0] if focus_symbols else None,
                    anchor_kind="file",
                )
            if changed_file.is_test_file:
                add_focus(
                    priority=1,
                    kind="review_changed_test",
                    title="Check changed test",
                    target=changed_file.path,
                    target_role="test",
                    message="A changed test often encodes the intended behavior shift or the regression being fixed.",
                    risk_level="low",
                    confidence=0.84,
                    location_hint=location_hint or changed_file.path,
                    command_hint=self._suggest_test_command(changed_file.path),
                    focus_symbols=focus_symbols,
                    why_now="Changed tests usually clarify expected behavior faster than reading implementation code first.",
                    expected_outcome="Confirm what behavior is newly expected or what regression is being guarded.",
                    follow_if_true="If the expectation changed, inspect the nearest impacted implementation boundary next.",
                    follow_if_false="If the test only refactors setup, continue with the strongest code-side impact target.",
                    anchor_file=changed_file.path,
                    anchor_line=anchor_line,
                    anchor_symbol=focus_symbols[0] if focus_symbols else None,
                    anchor_kind="file",
                )

        for action in impact_report.quick_actions:
            title = {
                "open_changed_boundary": "Inspect changed boundary",
                "run_nearby_test": "Run nearby test",
                "check_config_assumption": "Check config assumption",
                "open_direct_neighbor": "Inspect direct neighbor",
                "start_here": "Start with anchored boundary",
            }.get(action.kind, "Inspect impacted boundary")
            add_focus(
                priority=action.priority + 2,
                kind=action.kind,
                title=title,
                target=action.target,
                target_role=action.target_role,
                message=action.message,
                risk_level=action.risk_level,
                confidence=action.confidence,
                location_hint=action.location_hint,
                command_hint=action.command_hint,
                focus_symbols=action.focus_symbols,
                why_now=action.why_now,
                expected_outcome=action.expected_outcome,
                follow_if_true=action.follow_if_true,
                follow_if_false=action.follow_if_false,
                seed_file=action.seed_file,
                path_from_seed=action.path_from_seed,
                anchor_file=action.anchor_file,
                anchor_line=action.anchor_line,
                anchor_symbol=action.anchor_symbol,
                anchor_kind=action.anchor_kind,
            )

        for cluster in impact_report.test_clusters:
            if not cluster.paths:
                continue
            cluster_title = {
                "sibling": "Run sibling test cluster",
                "nearby": "Run nearby test cluster",
                "integration": "Run integration test cluster",
            }.get(cluster.kind, "Run impacted test cluster")
            add_focus(
                priority=3 if cluster.kind == "sibling" else 4,
                kind=f"review_{cluster.kind}_test_cluster",
                title=cluster_title,
                target=cluster.paths[0],
                target_role="test",
                message=cluster.reason or "Validate the affected change set with the closest related tests.",
                risk_level="low",
                confidence=0.8 if cluster.kind == "sibling" else 0.74,
                location_hint=cluster.paths[0],
                command_hint=cluster.command_hint,
                focus_symbols=cluster.focus_symbols,
                why_now="Grouped test validation can confirm the impact trail quickly before deeper review.",
                expected_outcome="Confirm whether the closest validating tests still pass or reveal the broken boundary.",
                follow_if_true="If the cluster reveals failures, inspect the matching impacted boundary immediately.",
                follow_if_false="If the cluster passes, continue with the next highest-risk changed surface.",
                seed_file=cluster.seed_file,
            )

        role_priority = {"public_api": 0, "entrypoint": 1, "config": 2, "test": 3, "boundary": 4, "neighbor": 5}
        focus_items.sort(
            key=lambda item: (
                item.priority,
                role_priority.get(item.target_role, 9),
                -item.confidence,
                item.target,
            )
        )
        return focus_items[:10]

    def build_review_report(
        self,
        changed_files: List[str],
        files: Optional[List[str]] = None,
        current_branch: Optional[str] = None,
        base_ref: Optional[str] = None,
        max_depth: int = 2,
        max_results: int = 10,
        changed_lines_by_file: Optional[Dict[str, List[int]]] = None,
    ) -> ReviewReport:
        """Combine git changes, impact analysis, test clusters, and review priorities."""
        changed_files = list(dict.fromkeys(changed_files or []))
        report = ReviewReport(
            current_branch=current_branch,
            base_ref=base_ref,
            max_depth=max_depth,
            max_results=max_results,
        )
        if not changed_files:
            report.error = "At least one changed file is required for review mode."
            report.diagnostics = list(self.repo_config.diagnostics)
            return report

        candidate_files = files or self._find_repo_files()
        all_fnames = self._prepare_candidate_files(candidate_files)
        abs_to_rel = {fname: self.get_rel_fname(fname) for fname in all_fnames}
        rel_to_abs = {rel_fname: fname for fname, rel_fname in abs_to_rel.items()}
        changed_abs_files = [os.path.abspath(path) for path in changed_files]
        changed_rel_files = [abs_to_rel[path] for path in changed_abs_files if path in abs_to_rel]
        missing_changed_files = [
            self.get_rel_fname(path)
            for path in changed_abs_files
            if path not in abs_to_rel
        ]
        if missing_changed_files:
            report.error = (
                "Changed file(s) are not in the selected repository scope: "
                + ", ".join(missing_changed_files)
            )
            report.diagnostics = list(self.repo_config.diagnostics)
            return report

        impact_report = self.analyze_file_impact(
            changed_abs_files,
            files=all_fnames,
            max_depth=max_depth,
            max_results=max_results,
            changed_lines_by_file=changed_lines_by_file,
        )
        if impact_report.error:
            report.error = impact_report.error
            report.diagnostics = impact_report.diagnostics[:]
            return report

        known_rel_fnames = set(abs_to_rel.values())
        changed_file_details = self._build_review_changed_files(
            changed_rel_files,
            rel_to_abs,
            known_rel_fnames,
            impact_report.changed_lines_by_file,
            impact_report.changed_hunks_by_file,
            impact_report.changed_seed_symbols,
        )
        report.changed_files = changed_file_details
        report.changed_lines_by_file = impact_report.changed_lines_by_file.copy()
        report.changed_hunks_by_file = {
            rel_fname: hunks[:]
            for rel_fname, hunks in impact_report.changed_hunks_by_file.items()
        }
        report.changed_seed_symbols = {
            rel_fname: symbols[:]
            for rel_fname, symbols in impact_report.changed_seed_symbols.items()
        }
        report.impacted_files = impact_report.impacted_files[:]
        report.shared_symbols = impact_report.shared_symbols[:]
        report.quick_actions = impact_report.quick_actions[:]
        report.edit_candidates = impact_report.edit_candidates[:]
        report.edit_plan = impact_report.edit_plan[:]
        report.test_clusters = impact_report.test_clusters[:]
        report.suggested_checks = impact_report.suggested_checks[:]
        report.changed_public_api_files = [item.path for item in changed_file_details if item.is_public_api_file]
        report.changed_entrypoint_files = [item.path for item in changed_file_details if item.is_entrypoint_file]
        report.changed_test_files = [item.path for item in changed_file_details if item.is_test_file]
        report.changed_config_files = [item.path for item in changed_file_details if item.summary_kind == "config"]
        report.review_focus = self._build_review_focus(changed_file_details, impact_report)
        report.diagnostics = impact_report.diagnostics[:]
        if current_branch:
            report.diagnostics.append(f"Reviewing branch {current_branch}.")
        if base_ref:
            report.diagnostics.append(f"Compared against base ref {base_ref}.")
        return report

    def _build_impact_symbols(
        self,
        impacted_files: List[ImpactTarget],
        changed_seed_symbols: Dict[str, List[str]],
    ) -> List[ImpactSymbol]:
        """Aggregate the shared symbols that most often define impact boundaries."""
        by_symbol = {}
        changed_symbol_names = {
            symbol
            for symbols in changed_seed_symbols.values()
            for symbol in symbols
        }
        for target in impacted_files:
            for symbol in target.boundary_symbols:
                entry = by_symbol.setdefault(
                    symbol,
                    {
                        "target_files": [],
                        "seed_files": [],
                        "closest_distance": None,
                        "closest_changed_hunk_distance": None,
                        "locations": [],
                    },
                )
                if target.path not in entry["target_files"]:
                    entry["target_files"].append(target.path)
                if target.seed_file not in entry["seed_files"]:
                    entry["seed_files"].append(target.seed_file)
                if entry["closest_distance"] is None or target.distance < entry["closest_distance"]:
                    entry["closest_distance"] = target.distance
                symbol_hunk_distance = target.changed_boundary_distances.get(symbol)
                if symbol_hunk_distance is not None and (
                    entry["closest_changed_hunk_distance"] is None
                    or symbol_hunk_distance < entry["closest_changed_hunk_distance"]
                ):
                    entry["closest_changed_hunk_distance"] = symbol_hunk_distance
                for location in target.boundary_locations:
                    if location.symbol != symbol:
                        continue
                    if any(
                        existing.file == location.file
                        and existing.line == location.line
                        and existing.kind == location.kind
                        and existing.symbol == location.symbol
                        for existing in entry["locations"]
                    ):
                        continue
                    entry["locations"].append(location)

        symbols = []
        for name, data in by_symbol.items():
            symbols.append(
                ImpactSymbol(
                    name=name,
                    target_files=sorted(data["target_files"]),
                    seed_files=sorted(data["seed_files"]),
                    target_count=len(data["target_files"]),
                    closest_distance=data["closest_distance"],
                    is_changed_seed_symbol=name in changed_symbol_names,
                    closest_changed_hunk_distance=data["closest_changed_hunk_distance"],
                    locations=data["locations"][:8],
                )
            )

        symbols.sort(
            key=lambda item: (
                -(1 if item.is_changed_seed_symbol else 0),
                item.closest_changed_hunk_distance if item.closest_changed_hunk_distance is not None else 999,
                -item.target_count,
                item.closest_distance or 99,
                item.name.lower(),
            )
        )
        return symbols[:12]

    def _choose_target_anchor(
        self,
        target: ImpactTarget,
        *,
        prefer_changed: bool = False,
        prefer_target_file: bool = True,
    ) -> Optional[ImpactSnippet]:
        """Pick the most actionable boundary snippet for a target."""
        snippets = target.boundary_snippets or []
        if not snippets:
            return None
        if prefer_target_file and not any(snippet.file == target.path for snippet in snippets):
            return None

        scored = []
        for index, snippet in enumerate(snippets):
            score = 0
            if prefer_changed and snippet.symbol in target.changed_boundary_symbols:
                score += 4
            if prefer_target_file and snippet.file == target.path:
                score += 2
            if snippet.highlight_line in target.focus_lines:
                score += 1
            scored.append((-score, index, snippet))

        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][2] if scored else snippets[0]

    def _build_target_file_anchor(self, target: ImpactTarget) -> Optional[ImpactSnippet]:
        """Create a file-level anchor when no stronger symbol-level anchor exists."""
        abs_fname = self.root / target.path
        text = self.read_text_func_internal(str(abs_fname))
        if not text:
            return None

        lines = text.splitlines()
        if not lines:
            return None

        highlight_line = target.focus_lines[0] if target.focus_lines else 1
        highlight_index = min(max(highlight_line - 1, 0), len(lines) - 1)
        start_index = max(0, highlight_index - 1)
        end_index = min(len(lines), highlight_index + 2)
        excerpt = "\n".join(
            f"{line_no}: {lines[line_no - 1]}"
            for line_no in range(start_index + 1, end_index + 1)
        )
        return ImpactSnippet(
            file=target.path,
            start_line=start_index + 1,
            end_line=end_index,
            highlight_line=highlight_index + 1,
            kind="file",
            symbol=Path(target.path).name,
            excerpt=excerpt,
        )

    @staticmethod
    def _build_action_anchor(action: ImpactQuickAction) -> Optional[ImpactSnippet]:
        """Convert quick action anchor fields into a snippet-like structure."""
        if not action.anchor_file or not action.anchor_line:
            return None
        symbol = action.anchor_symbol or Path(action.anchor_file).name
        kind = action.anchor_kind or "file"
        excerpt = action.anchor_excerpt or ""
        return ImpactSnippet(
            file=action.anchor_file,
            start_line=action.anchor_line,
            end_line=action.anchor_line,
            highlight_line=action.anchor_line,
            kind=kind,
            symbol=symbol,
            excerpt=excerpt,
        )

    @staticmethod
    def _get_quick_action_target_role(
        kind: str,
        target: Optional[ImpactTarget],
    ) -> str:
        """Classify the role of the file a quick action points at."""
        kind_roles = {
            "run_nearby_test": "test",
            "check_config_assumption": "config",
            "open_direct_neighbor": "neighbor",
        }

        if target:
            if target.is_test_file:
                return "test"
            if target.summary_kind == "config":
                return "config"
            if target.is_public_api_file:
                return "public_api"
            if target.is_entrypoint_file:
                return "entrypoint"

        return kind_roles.get(kind, "boundary")

    @staticmethod
    def _get_target_role_sort_key(target_role: str) -> int:
        """Keep more actionable code-facing edit roles ahead of generic neighbors."""
        order = {
            "boundary": 0,
            "public_api": 1,
            "entrypoint": 2,
            "config": 3,
            "test": 4,
            "neighbor": 5,
        }
        return order.get(target_role, 9)

    def _describe_edit_candidate(
        self,
        action: ImpactQuickAction,
        target: Optional[ImpactTarget],
        anchor: Optional[ImpactSnippet],
    ) -> str:
        """Explain why a concrete file/symbol is a plausible edit point."""
        role = action.target_role
        seed_file = action.seed_file or "the seed file"
        symbol = None
        if anchor and anchor.kind != "file":
            symbol = anchor.symbol
        elif action.focus_symbols:
            symbol = action.focus_symbols[0]

        if role == "test":
            return f"Closest validation file covering the impact path from {seed_file}."
        if role == "config":
            return f"Config surface on the impact path from {seed_file} that can invalidate deeper edits quickly."
        if role == "public_api":
            return (
                f"Public API boundary {symbol} is the likeliest contract edit point."
                if symbol
                else "Public API surface likely needs a matching contract review."
            )
        if role == "entrypoint":
            return (
                f"Entrypoint boundary {symbol} is where the changed flow most likely surfaces first."
                if symbol
                else "Entrypoint surface likely needs a matching flow review."
            )
        if role == "neighbor":
            return f"One-hop neighbor from {seed_file} that can absorb or rule out the change quickly."
        if symbol:
            return f"Boundary symbol {symbol} is the strongest concrete edit point on this path."
        if target and target.changed_boundary_symbols:
            return f"Changed boundary from {seed_file} stays active in this file."
        return f"Most direct impacted boundary reachable from {seed_file}."

    def _build_impact_edit_candidates(
        self,
        impacted_files: List[ImpactTarget],
        quick_actions: List[ImpactQuickAction],
    ) -> List[ImpactEditCandidate]:
        """Turn quick actions into concrete file/symbol edit suggestions."""
        candidates = []
        seen = set()
        impacted_by_path = {target.path: target for target in impacted_files}

        def add_candidate(
            action: ImpactQuickAction,
            target: Optional[ImpactTarget],
            anchor: Optional[ImpactSnippet],
        ) -> None:
            path = action.target
            line = None
            symbol = None
            symbol_kind = None
            location_hint = action.location_hint or action.target

            if anchor:
                path = anchor.file or path
                line = anchor.highlight_line
                location_hint = f"{path}:{line}" if line else path
                if anchor.kind != "file":
                    symbol = anchor.symbol
                    symbol_kind = anchor.kind

            key = (path, line, symbol, action.target_role)
            if key in seen:
                return
            seen.add(key)

            candidates.append(
                ImpactEditCandidate(
                    path=path,
                    target_role=action.target_role,
                    reason=self._describe_edit_candidate(action, target, anchor),
                    priority=action.priority,
                    confidence=action.confidence,
                    line=line,
                    symbol=symbol,
                    symbol_kind=symbol_kind,
                    location_hint=location_hint,
                    seed_file=action.seed_file,
                    path_from_seed=action.path_from_seed[:],
                    source_action_kind=action.kind,
                    source_action_target=action.target,
                )
            )

        for action in quick_actions:
            target = impacted_by_path.get(action.target)
            anchor = self._build_action_anchor(action)
            if target and (not anchor or anchor.file != action.target):
                anchor = self._choose_target_anchor(
                    target,
                    prefer_changed=action.kind == "open_changed_boundary",
                    prefer_target_file=True,
                ) or self._build_target_file_anchor(target)
            add_candidate(action, target, anchor)

        candidates.sort(
            key=lambda item: (
                item.priority,
                -item.confidence,
                self._get_target_role_sort_key(item.target_role),
                len(item.path_from_seed),
                item.path,
                item.line or 0,
            )
        )
        return candidates[:8]

    def _get_edit_plan_title(self, action: ImpactQuickAction) -> str:
        """Render a short title for a compact edit-plan step."""
        if action.target_role == "test":
            return "Run nearby test"
        if action.target_role == "config":
            return "Check config assumption"
        if action.target_role == "public_api":
            return "Inspect public API boundary"
        if action.target_role == "entrypoint":
            return "Inspect entrypoint boundary"
        if action.target_role == "neighbor":
            return "Inspect direct neighbor"

        titles = {
            "open_changed_boundary": "Inspect changed boundary",
            "start_here": "Start with anchored boundary",
        }
        return titles.get(action.kind, "Inspect impacted boundary")

    def _build_edit_plan_instruction(
        self,
        action: ImpactQuickAction,
        candidate: Optional[ImpactEditCandidate],
    ) -> str:
        """Describe the next practical move in edit-oriented language."""
        candidate_ref = action.location_hint or action.target
        if candidate:
            candidate_ref = candidate.location_hint or candidate.path
            if candidate.symbol:
                candidate_ref = f"{candidate_ref} ({candidate.symbol})"

        if action.target_role == "test":
            return f"Validate the impact trail in {candidate_ref} before editing deeper code."
        if action.target_role == "config":
            return f"Check {candidate_ref} first so config drift does not send the edit down the wrong branch."
        if action.target_role == "public_api":
            return f"Open {candidate_ref} and confirm the exported contract still matches the changed behavior."
        if action.target_role == "entrypoint":
            return f"Open {candidate_ref} and confirm the main execution flow still reaches the changed boundary."
        if action.target_role == "neighbor":
            return f"Inspect {candidate_ref} as the smallest one-hop file that may need a matching update."
        return f"Inspect {candidate_ref} as the strongest concrete boundary on this impact path."

    def _build_impact_edit_plan(
        self,
        quick_actions: List[ImpactQuickAction],
        edit_candidates: List[ImpactEditCandidate],
    ) -> List[ImpactEditPlanStep]:
        """Project quick actions into a compact what-to-edit-next plan."""
        candidates_by_action = defaultdict(list)
        for candidate in edit_candidates:
            key = (
                candidate.source_action_kind,
                candidate.source_action_target,
                candidate.seed_file,
            )
            candidates_by_action[key].append(candidate)

        steps = []
        for index, action in enumerate(quick_actions[:4], start=1):
            key = (action.kind, action.target, action.seed_file)
            step_candidates = candidates_by_action.get(key, [])[:3]
            primary_candidate = step_candidates[0] if step_candidates else None
            steps.append(
                ImpactEditPlanStep(
                    step=index,
                    priority=action.priority,
                    title=self._get_edit_plan_title(action),
                    instruction=self._build_edit_plan_instruction(action, primary_candidate),
                    target=action.target,
                    target_role=action.target_role,
                    confidence=action.confidence,
                    action_kind=action.kind,
                    location_hint=(primary_candidate.location_hint if primary_candidate else action.location_hint),
                    command_hint=action.command_hint,
                    focus_symbols=action.focus_symbols[:3],
                    why_now=action.why_now,
                    expected_outcome=action.expected_outcome,
                    follow_if_true=action.follow_if_true,
                    follow_if_false=action.follow_if_false,
                    seed_file=action.seed_file,
                    path_from_seed=action.path_from_seed[:],
                    edit_candidates=step_candidates,
                )
            )

        return steps

    def _is_integration_test_path(self, rel_path: str) -> bool:
        """Detect tests that look broader than a direct sibling/unit test."""
        normalized = self._normalize_rel_path(rel_path).lower()
        path = Path(normalized)
        if any(part in self._integration_test_markers for part in path.parts):
            return True

        stem = path.stem
        return any(marker in stem for marker in self._integration_test_markers)

    def _suggest_test_cluster_command(self, rel_paths: List[str]) -> Optional[str]:
        """Suggest a runner command for a small group of related test files."""
        rel_paths = [path for path in rel_paths if path]
        if not rel_paths:
            return None
        if len(rel_paths) == 1:
            return self._suggest_test_command(rel_paths[0])

        suffixes = {Path(path).suffix.lower() for path in rel_paths}
        quoted_paths = " ".join(shlex.quote(path) for path in rel_paths[:6])

        if suffixes == {".py"} and self._uses_pytest():
            return f"pytest {quoted_paths}"

        if suffixes.issubset({".js", ".jsx", ".ts", ".tsx"}):
            js_runner = self._detect_js_test_runner()
            if js_runner == "vitest":
                return f"npx vitest run {quoted_paths}"
            if js_runner == "jest":
                return f"npx jest {quoted_paths}"
            if js_runner == "mocha":
                return f"npx mocha {quoted_paths}"

        return None

    def _build_impact_test_clusters(
        self,
        impacted_files: List[ImpactTarget],
    ) -> List[ImpactTestCluster]:
        """Group impacted tests into nearby, sibling, and integration clusters."""
        impacted_tests = [target for target in impacted_files if target.is_test_file]
        if not impacted_tests:
            return []

        impacted_non_tests = [target for target in impacted_files if not target.is_test_file]
        sibling_sources_by_test = {}
        for test_target in impacted_tests:
            sibling_sources = []
            for source_target in impacted_non_tests:
                if test_target.path in self._candidate_related_test_paths(source_target.path):
                    sibling_sources.append(source_target.path)
            sibling_sources_by_test[test_target.path] = sibling_sources

        grouped = {}
        for test_target in impacted_tests:
            sibling_sources = sibling_sources_by_test.get(test_target.path, [])
            if self._is_integration_test_path(test_target.path):
                cluster_kind = "integration"
            elif sibling_sources:
                cluster_kind = "sibling"
            else:
                cluster_kind = "nearby"

            key = (test_target.seed_file, cluster_kind)
            entry = grouped.setdefault(
                key,
                {
                    "paths": [],
                    "covers": [],
                    "closest_distance": None,
                    "focus_symbols": [],
                },
            )

            if test_target.path not in entry["paths"]:
                entry["paths"].append(test_target.path)
            for source_path in sibling_sources:
                if source_path not in entry["covers"]:
                    entry["covers"].append(source_path)
            if entry["closest_distance"] is None or test_target.distance < entry["closest_distance"]:
                entry["closest_distance"] = test_target.distance
            for symbol in test_target.boundary_symbols:
                if symbol and symbol not in entry["focus_symbols"]:
                    entry["focus_symbols"].append(symbol)

        clusters = []
        for (seed_file, cluster_kind), data in grouped.items():
            paths = sorted(data["paths"])
            covers = sorted(data["covers"])
            if cluster_kind == "integration":
                reason = f"Integration-style tests reachable from {seed_file}."
            elif cluster_kind == "sibling":
                reason = (
                    f"Sibling tests closely matched to impacted file(s): {', '.join(covers[:3])}."
                    if covers
                    else f"Sibling tests closely matched to the impact path from {seed_file}."
                )
            else:
                reason = f"Nearby validation tests in the same impact neighborhood as {seed_file}."

            clusters.append(
                ImpactTestCluster(
                    kind=cluster_kind,
                    seed_file=seed_file,
                    paths=paths[:6],
                    covers=covers[:5],
                    closest_distance=data["closest_distance"],
                    focus_symbols=data["focus_symbols"][:5],
                    command_hint=self._suggest_test_cluster_command(paths[:6]),
                    reason=reason,
                )
            )

        kind_priority = {"sibling": 0, "nearby": 1, "integration": 2}
        clusters.sort(
            key=lambda item: (
                item.closest_distance if item.closest_distance is not None else 99,
                kind_priority.get(item.kind, 9),
                item.seed_file,
                item.paths[0] if item.paths else "",
            )
        )
        return clusters[:6]

    def _build_impact_quick_actions(
        self,
        impacted_files: List[ImpactTarget],
        suggested_checks: List[ImpactSuggestion],
    ) -> List[ImpactQuickAction]:
        """Build low-risk next actions from the broader impact checklist."""
        quick_actions = []
        seen = set()
        impacted_by_path = {target.path: target for target in impacted_files}

        def add_action(
            suggestion: ImpactSuggestion,
            *,
            target: Optional[ImpactTarget],
            kind: str,
            message: str,
            priority: Optional[int] = None,
            effort: str = "small",
        ) -> None:
            key = (kind, suggestion.target)
            if key in seen:
                return
            seen.add(key)
            location_hint = None
            if suggestion.anchor_file and suggestion.anchor_line:
                location_hint = f"{suggestion.anchor_file}:{suggestion.anchor_line}"
            elif suggestion.target:
                location_hint = suggestion.target

            command_hint = None
            if kind == "run_nearby_test":
                command_hint = self._suggest_test_command(suggestion.target)

            risk_level, why_now, expected_outcome, follow_if_true, follow_if_false = self._describe_quick_action(
                kind,
                target,
                suggestion,
            )
            confidence = self._score_quick_action_confidence(kind, target)
            focus_symbols, focus_reason = self._select_quick_action_focus(kind, target, suggestion)
            target_role = self._get_quick_action_target_role(kind, target)

            quick_actions.append(
                ImpactQuickAction(
                    priority=suggestion.priority if priority is None else priority,
                    kind=kind,
                    target=suggestion.target,
                    message=message,
                    effort=effort,
                    target_role=target_role,
                    risk_level=risk_level,
                    confidence=confidence,
                    focus_symbols=focus_symbols,
                    focus_reason=focus_reason,
                    why_now=why_now,
                    expected_outcome=expected_outcome,
                    follow_if_true=follow_if_true,
                    follow_if_false=follow_if_false,
                    location_hint=location_hint,
                    command_hint=command_hint,
                    seed_file=suggestion.seed_file,
                    path_from_seed=suggestion.path_from_seed[:],
                    anchor_file=suggestion.anchor_file,
                    anchor_line=suggestion.anchor_line,
                    anchor_symbol=suggestion.anchor_symbol,
                    anchor_kind=suggestion.anchor_kind,
                    anchor_excerpt=suggestion.anchor_excerpt,
                )
            )

        for suggestion in suggested_checks:
            target = impacted_by_path.get(suggestion.target)
            if suggestion.kind == "review_changed_symbol_boundary":
                add_action(
                    suggestion,
                    target=target,
                    kind="open_changed_boundary",
                    message="Open this changed boundary first and verify the nearby symbol contract.",
                    priority=0,
                )
            elif suggestion.kind == "review_test":
                add_action(
                    suggestion,
                    target=target,
                    kind="run_nearby_test",
                    message="Run or inspect this nearby test before making broader edits.",
                    priority=0 if target and target.distance == 1 else 1,
                )
            elif suggestion.kind == "review_config":
                add_action(
                    suggestion,
                    target=target,
                    kind="check_config_assumption",
                    message="Verify this config assumption before changing deeper logic.",
                    priority=1,
                )
            elif suggestion.kind == "inspect_neighbor" and target and target.distance == 1:
                add_action(
                    suggestion,
                    target=target,
                    kind="open_direct_neighbor",
                    message="Open this one-hop neighbor at the anchored line first.",
                    priority=2,
                )

        if not quick_actions and suggested_checks:
            first = suggested_checks[0]
            add_action(
                first,
                target=impacted_by_path.get(first.target),
                kind="start_here",
                message="Start with the first anchored follow-up item.",
                priority=first.priority,
            )

        quick_actions.sort(key=lambda item: (item.priority, -item.confidence, len(item.path_from_seed), item.target))
        return quick_actions[:6]

    def _score_quick_action_confidence(
        self,
        kind: str,
        target: Optional[ImpactTarget],
    ) -> float:
        """Estimate how strong the repository evidence is for a quick action."""
        base_scores = {
            "open_changed_boundary": 0.9,
            "run_nearby_test": 0.82,
            "open_direct_neighbor": 0.72,
            "check_config_assumption": 0.64,
            "start_here": 0.52,
        }
        score = base_scores.get(kind, 0.6)

        if not target:
            return round(score, 2)

        score -= max(target.distance - 1, 0) * 0.05

        if target.closest_changed_hunk_distance is not None:
            if target.closest_changed_hunk_distance == 0:
                score += 0.05
            elif target.closest_changed_hunk_distance == 1:
                score += 0.03
            elif target.closest_changed_hunk_distance >= 3:
                score -= 0.04

        if target.is_test_file and kind == "run_nearby_test":
            score += 0.04
        if target.summary_kind == "config" and kind == "check_config_assumption":
            score += 0.04
        if target.is_public_api_file and kind == "open_changed_boundary":
            score += 0.03
        if target.is_entrypoint_file and kind == "open_direct_neighbor":
            score += 0.02

        score = min(max(score, 0.15), 0.98)
        return round(score, 2)

    def _select_quick_action_focus(
        self,
        kind: str,
        target: Optional[ImpactTarget],
        suggestion: ImpactSuggestion,
    ) -> Tuple[List[str], Optional[str]]:
        """Pick the most useful symbols to focus on for a quick action and explain why."""
        candidates = []
        focus_reason = None

        if suggestion.anchor_symbol and suggestion.anchor_kind != "file":
            candidates.append(suggestion.anchor_symbol)
            focus_reason = "Focused on the anchor symbol highlighted for this action."

        if target:
            if kind in {"open_changed_boundary", "run_nearby_test"}:
                candidates.extend(target.changed_boundary_symbols)
                if target.changed_boundary_symbols:
                    focus_reason = "Focused on changed boundary symbols nearest to the seed diff."
            if not candidates:
                if kind == "open_changed_boundary":
                    candidates.extend(target.boundary_symbols[:1])
                elif kind == "run_nearby_test":
                    candidates.extend(target.boundary_symbols[:1])
                else:
                    candidates.extend(target.boundary_symbols)
                if candidates:
                    focus_reason = "Focused on the strongest shared boundary symbols on this impact path."

        seen = set()
        focus_symbols = []
        for symbol in candidates:
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            focus_symbols.append(symbol)
            if len(focus_symbols) >= 3:
                break

        return focus_symbols, focus_reason

    def _describe_quick_action(
        self,
        kind: str,
        target: Optional[ImpactTarget],
        suggestion: ImpactSuggestion,
    ) -> Tuple[str, str, str, str, str]:
        """Provide compact prioritization and success criteria for a quick action."""
        seed_file = suggestion.seed_file or "the seed file"

        if kind == "run_nearby_test":
            return (
                "low",
                f"This is the fastest validation signal close to {seed_file}.",
                "Confirm whether the nearby test already passes or pinpoints the broken behavior.",
                "If it fails, follow the failing assertion or stack trace to the impacted boundary immediately.",
                "If it passes, continue with the nearest non-test impact boundary or direct neighbor.",
            )

        if kind == "open_changed_boundary":
            if target and target.closest_changed_hunk_distance is not None:
                return (
                    "low",
                    f"This boundary is only {target.closest_changed_hunk_distance} line(s) from the changed hunk.",
                    "Confirm whether this boundary symbol or call site needs a matching update.",
                    "If it does need a change, trace outward to callers, tests, and public API edges touching this symbol.",
                    "If it does not, move to the next closest impacted file or config assumption.",
                )
            return (
                "low",
                "This is the narrowest downstream boundary touched by the changed symbol.",
                "Confirm whether the downstream boundary still matches the changed symbol contract.",
                "If it does not match, update connected callers and nearby tests next.",
                "If it still matches, continue to the next impacted boundary on the path.",
            )

        if kind == "open_direct_neighbor":
            return (
                "low",
                "A one-hop neighbor is usually the smallest non-test file worth checking next.",
                "Confirm whether the closest neighboring file is affected or can be ruled out quickly.",
                "If it is affected, keep expanding along its boundary symbols and related tests.",
                "If it is not, deprioritize this branch and inspect the next quick action.",
            )

        if kind == "check_config_assumption":
            return (
                "medium",
                "Config mismatches can invalidate the rest of the impact trail quickly.",
                "Confirm whether configuration still matches the changed execution path or test setup.",
                "If config changed, review entrypoints, setup docs, and the tests that depend on it.",
                "If config is unchanged, return to code-level boundaries and neighbors.",
            )

        if kind == "start_here":
            return (
                "medium",
                "No lower-risk shortcut was found, so this is the best anchored starting point.",
                "Collect the first concrete signal about whether the impact trail is real.",
                "If you find a concrete mismatch, pivot toward the closest affected boundary and tests.",
                "If you do not, continue down the remaining suggested checks in priority order.",
            )

        return (
            "low",
            "This is an anchored next step near the current impact boundary.",
            "Confirm whether this anchored boundary changes the likely follow-up path.",
            "If it does, continue along the impacted branch from this point outward.",
            "If it does not, step back and inspect the next prioritized action.",
        )

    def _read_root_file(self, rel_path: str) -> Optional[str]:
        """Read a root-level project file if it exists."""
        abs_path = self.root / rel_path
        if not abs_path.is_file():
            return None
        return self.read_text_func_internal(str(abs_path))

    def _get_python_test_runner(self) -> Optional[str]:
        """Return the configured or inferred Python test runner."""
        if self.repo_config.tests.python_runner:
            return self.repo_config.tests.python_runner
        if self._uses_pytest():
            return "pytest"
        return None

    def _suggest_test_command(self, rel_path: str) -> Optional[str]:
        """Suggest a concrete test command when the repository gives a clear enough signal."""
        suffix = Path(rel_path).suffix.lower()
        quoted_path = shlex.quote(rel_path)

        if suffix == ".py":
            if self._get_python_test_runner() == "pytest":
                return f"pytest {quoted_path}"
            return None

        if suffix in {".js", ".jsx", ".ts", ".tsx"}:
            js_runner = self._detect_js_test_runner()
            if js_runner == "vitest":
                return f"npx vitest run {quoted_path}"
            if js_runner == "jest":
                return f"npx jest {quoted_path}"
            if js_runner == "mocha":
                return f"npx mocha {quoted_path}"

        return None

    def _uses_pytest(self) -> bool:
        """Infer whether pytest is the project's Python test runner."""
        if self.repo_config.tests.python_runner:
            return self.repo_config.tests.python_runner == "pytest"

        pyproject_text = self._read_root_file("pyproject.toml")
        if pyproject_text and tomllib is not None:
            try:
                data = tomllib.loads(pyproject_text)
            except (tomllib.TOMLDecodeError, TypeError):
                data = {}
            tool = data.get("tool") or {}
            if "pytest" in tool or "pytest_env" in tool:
                return True

            project = data.get("project") or {}
            dependency_names = set(self._extract_dependency_names(project.get("dependencies") or []))
            optional_dependencies = project.get("optional-dependencies") or {}
            for dep_list in optional_dependencies.values():
                dependency_names.update(self._extract_dependency_names(dep_list or []))
            if "pytest" in dependency_names:
                return True

        for config_name in ("pytest.ini", "tox.ini", "setup.cfg"):
            config_text = self._read_root_file(config_name)
            if config_text and "pytest" in config_text.lower():
                return True

        requirements_text = self._read_root_file("requirements.txt")
        if requirements_text and re.search(r"(?mi)^pytest(?:[\[<=>].*)?$", requirements_text):
            return True

        return False

    def _detect_js_test_runner(self) -> Optional[str]:
        """Infer the dominant JS/TS test runner from package.json metadata."""
        if self.repo_config.tests.js_runner:
            return self.repo_config.tests.js_runner

        package_text = self._read_root_file("package.json")
        if not package_text:
            return None

        try:
            package_data = json.loads(package_text)
        except json.JSONDecodeError:
            return None

        scripts = package_data.get("scripts") or {}
        test_script = str(scripts.get("test") or "").lower()
        dependencies = {
            str(name).lower()
            for name in (package_data.get("dependencies") or {}).keys()
        }
        dependencies.update(
            str(name).lower()
            for name in (package_data.get("devDependencies") or {}).keys()
        )

        if "vitest" in test_script or "vitest" in dependencies:
            return "vitest"
        if "jest" in test_script or "jest" in dependencies:
            return "jest"
        if "mocha" in test_script or "mocha" in dependencies:
            return "mocha"
        return None

    def _build_impact_suggestions(
        self,
        seed_rel_files: List[str],
        impacted_files: List[ImpactTarget],
    ) -> List[ImpactSuggestion]:
        """Build a short, prioritized follow-up checklist for impact analysis."""
        suggestions = []
        seen = set()

        def add_suggestion(
            priority: int,
            kind: str,
            target: str,
            message: str,
            seed_file: Optional[str] = None,
            path_from_seed: Optional[List[str]] = None,
            anchor: Optional[ImpactSnippet] = None,
        ) -> None:
            key = (kind, target)
            if key in seen:
                return
            seen.add(key)
            suggestions.append(
                ImpactSuggestion(
                    priority=priority,
                    kind=kind,
                    target=target,
                    message=message,
                    seed_file=seed_file,
                    path_from_seed=(path_from_seed or [])[:],
                    anchor_file=anchor.file if anchor else None,
                    anchor_line=anchor.highlight_line if anchor else None,
                    anchor_symbol=anchor.symbol if anchor else None,
                    anchor_kind=anchor.kind if anchor else None,
                    anchor_excerpt=anchor.excerpt if anchor else None,
                )
            )

        for target in impacted_files:
            if target.is_test_file:
                add_suggestion(
                    0,
                    "review_test",
                    target.path,
                    f"Review or run this nearby test for changes around {target.seed_file}.",
                    seed_file=target.seed_file,
                    path_from_seed=target.path_from_seed,
                    anchor=self._choose_target_anchor(target) or self._build_target_file_anchor(target),
                )
            if target.is_public_api_file:
                add_suggestion(
                    1,
                    "review_public_api",
                    target.path,
                    f"Check whether the public API contract exposed by {target.path} changed.",
                    seed_file=target.seed_file,
                    path_from_seed=target.path_from_seed,
                    anchor=self._choose_target_anchor(target) or self._build_target_file_anchor(target),
                )
            if target.is_entrypoint_file:
                add_suggestion(
                    1,
                    "verify_entrypoint",
                    target.path,
                    f"Verify the main execution flow still reaches {target.path} correctly.",
                    seed_file=target.seed_file,
                    path_from_seed=target.path_from_seed,
                    anchor=self._choose_target_anchor(target) or self._build_target_file_anchor(target),
                )
            if target.summary_kind == "config":
                add_suggestion(
                    2,
                    "review_config",
                    target.path,
                    f"Inspect this config file because it sits on the impact path from {target.seed_file}.",
                    seed_file=target.seed_file,
                    path_from_seed=target.path_from_seed,
                    anchor=self._choose_target_anchor(target) or self._build_target_file_anchor(target),
                )
            elif target.summary_kind == "doc":
                add_suggestion(
                    3,
                    "read_doc",
                    target.path,
                    f"Check this documentation file for setup or workflow assumptions around {target.seed_file}.",
                    seed_file=target.seed_file,
                    path_from_seed=target.path_from_seed,
                    anchor=self._choose_target_anchor(target, prefer_target_file=False) or self._build_target_file_anchor(target),
                )
            if target.distance == 1 and not target.is_test_file:
                add_suggestion(
                    2,
                    "inspect_neighbor",
                    target.path,
                    f"Inspect this direct neighbor because it is one hop away from {target.seed_file}.",
                    seed_file=target.seed_file,
                    path_from_seed=target.path_from_seed,
                    anchor=self._choose_target_anchor(target) or self._build_target_file_anchor(target),
                )
            if target.steps and any(step.symbols for step in target.steps):
                if target.boundary_symbols:
                    add_suggestion(
                        2,
                        "follow_symbols",
                        target.path,
                        f"Trace shared symbols such as {', '.join(target.boundary_symbols[:3])}.",
                        seed_file=target.seed_file,
                        path_from_seed=target.path_from_seed,
                        anchor=self._choose_target_anchor(target) or self._build_target_file_anchor(target),
                    )
            if target.changed_boundary_symbols:
                add_suggestion(
                    0 if (target.closest_changed_hunk_distance == 0) else 1,
                    "review_changed_symbol_boundary",
                    target.path,
                    (
                        f"Start with changed boundary symbols: {', '.join(target.changed_boundary_symbols[:3])}"
                        + (
                            f" (closest hunk distance {target.closest_changed_hunk_distance})."
                            if target.closest_changed_hunk_distance is not None
                            else "."
                        )
                    ),
                    seed_file=target.seed_file,
                    path_from_seed=target.path_from_seed,
                    anchor=self._choose_target_anchor(target, prefer_changed=True) or self._build_target_file_anchor(target),
                )

        if not suggestions and impacted_files:
            first_target = impacted_files[0]
            add_suggestion(
                2,
                "inspect_neighbor",
                first_target.path,
                f"Start with the closest impacted file on the path from {first_target.seed_file}.",
                seed_file=first_target.seed_file,
                path_from_seed=first_target.path_from_seed,
                anchor=self._choose_target_anchor(first_target) or self._build_target_file_anchor(first_target),
            )

        suggestions.sort(key=lambda item: (item.priority, len(item.path_from_seed), item.target))
        return suggestions[:8]

    def _get_test_file_tags(self, fname: str, rel_fname: str) -> List[Tag]:
        """Generate synthetic lines of interest for test files lacking parser definitions."""
        text = self.read_text_func_internal(fname)
        if not text:
            return []

        candidate_lines = []
        fallback_lines = []
        test_patterns = ("def test_", "class test", "it(", "test(", "describe(", "scenario(")

        for line_num, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip().lower()
            if not stripped:
                continue
            if len(fallback_lines) < 4:
                fallback_lines.append(line_num)
            if any(pattern in stripped for pattern in test_patterns):
                candidate_lines.append(line_num)
            if len(candidate_lines) >= 6:
                break

        lois = candidate_lines or fallback_lines
        basename = Path(rel_fname).name
        return [
            Tag(rel_fname=rel_fname, fname=fname, line=line_num, name=basename, kind="test")
            for line_num in lois
        ]

    def _extract_query_terms(self, query: Optional[str]) -> List[str]:
        """Extract code-relevant terms from a free-form query string."""
        if not query:
            return []

        terms = []
        for token in re.split(r"[^A-Za-z0-9]+", query.lower()):
            if len(token) < 2 or token in QUERY_STOP_WORDS:
                continue
            terms.append(token)
        return self._dedupe_preserve_order(terms)

    def _match_query_terms(
        self,
        rel_fname: str,
        tags: List[Tag],
        query_terms: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Find query terms that match the file path or extracted symbols."""
        if not query_terms:
            return [], []

        rel_lower = rel_fname.lower()
        path_matches = [term for term in query_terms if term in rel_lower]

        symbol_names = {tag.name.lower() for tag in tags if tag.name}
        symbol_matches = []
        for term in query_terms:
            if any(term in symbol_name for symbol_name in symbol_names):
                symbol_matches.append(term)

        return self._dedupe_preserve_order(path_matches), self._dedupe_preserve_order(symbol_matches)

    def _get_query_rank_context(
        self,
        file_rank: float,
        path_matches: List[str],
        symbol_matches: List[str],
    ) -> Tuple[float, float]:
        """Return an effective base rank and boost for query-matched files."""
        if not path_matches and not symbol_matches:
            return file_rank, 1.0

        query_weight = self.repo_config.ranking_weights.query
        query_floor = self._scale_floor(
            (0.02 * len(path_matches)) + (0.05 * len(symbol_matches)),
            query_weight,
        )
        path_boost = 1.0 + (0.75 * len(path_matches) * max(query_weight, 0.0))
        symbol_boost = 1.0 + (1.25 * len(symbol_matches) * max(query_weight, 0.0))
        query_boost = min(path_boost * symbol_boost, self._scale_multiplier(8.0, query_weight))
        return max(file_rank, query_floor), query_boost

    def _build_file_reasons(
        self,
        *,
        rel_fname: str,
        is_changed_file: bool,
        changed_neighbor_distance: Optional[int],
        related_changed_files: List[str],
        is_test_file: bool,
        is_entrypoint_file: bool,
        is_public_api_file: bool,
        is_chat_file: bool,
        is_mentioned_file: bool,
        is_important_file: bool,
        matched_query_path_terms: List[str],
        matched_query_symbol_terms: List[str],
        related_tests: List[str],
        related_sources: List[str],
        entrypoint_signals: List[str],
        public_api_signals: List[str],
        summary_kind: Optional[str],
        summary_items: List[str],
        mentioned_identifiers: List[str],
        inbound_neighbors: List[str],
        outbound_neighbors: List[str],
        definition_count: int,
        reference_count: int,
    ) -> List[RankingReason]:
        """Build a short machine-readable explanation for why a file ranked highly."""
        reasons = []

        if is_changed_file:
            reasons.append(RankingReason("changed_file", "File has direct local changes in the current git selection."))
        elif changed_neighbor_distance is not None:
            preview = ", ".join(related_changed_files[:3]) if related_changed_files else "changed files"
            reasons.append(
                RankingReason(
                    "changed_neighbor",
                    f"Within changed-file impact neighborhood (distance {changed_neighbor_distance}) via {preview}.",
                )
            )
        if is_chat_file:
            reasons.append(RankingReason("chat_file", "File is part of the active chat/edit context."))
        if is_mentioned_file:
            reasons.append(RankingReason("mentioned_file", "File was explicitly mentioned by the caller."))
        if is_entrypoint_file:
            reasons.append(
                RankingReason(
                    "entrypoint_file",
                    f"Looks like an application entrypoint via: {', '.join(entrypoint_signals[:3])}.",
                )
            )
        if is_public_api_file:
            reasons.append(
                RankingReason(
                    "public_api_file",
                    f"Looks like a public API surface via: {', '.join(public_api_signals[:3])}.",
                )
            )
        if summary_items:
            reason_code = "doc_summary" if summary_kind == "doc" else "config_summary"
            reasons.append(
                RankingReason(
                    reason_code,
                    f"Structured summary extracted: {', '.join(summary_items[:2])}.",
                )
            )
        if matched_query_path_terms:
            reasons.append(
                RankingReason(
                    "query_path_match",
                    f"Path matches query terms: {', '.join(matched_query_path_terms[:4])}.",
                )
            )
        if matched_query_symbol_terms:
            reasons.append(
                RankingReason(
                    "query_symbol_match",
                    f"Symbol names match query terms: {', '.join(matched_query_symbol_terms[:4])}.",
                )
            )
        if related_tests:
            reasons.append(
                RankingReason(
                    "related_tests",
                    f"Has nearby related test file(s): {', '.join(related_tests[:3])}.",
                )
            )
        if is_test_file and related_sources:
            reasons.append(
                RankingReason(
                    "related_source",
                    f"Likely tests related source file(s): {', '.join(related_sources[:3])}.",
                )
            )
        if mentioned_identifiers:
            preview = ", ".join(mentioned_identifiers[:3])
            if len(mentioned_identifiers) > 3:
                preview += f", +{len(mentioned_identifiers) - 3} more"
            reasons.append(
                RankingReason(
                    "mentioned_identifier",
                    f"Defines or references explicitly mentioned identifiers: {preview}.",
                )
            )
        if is_important_file:
            reasons.append(
                RankingReason(
                    "important_file",
                    "Important docs/config file retained even without parser-extracted symbols.",
                )
            )
        if inbound_neighbors:
            reasons.append(
                RankingReason(
                    "referenced_by",
                    f"Referenced by {len(inbound_neighbors)} file(s): {', '.join(inbound_neighbors[:3])}.",
                )
            )
        if outbound_neighbors:
            reasons.append(
                RankingReason(
                    "references",
                    f"Links to {len(outbound_neighbors)} related file(s): {', '.join(outbound_neighbors[:3])}.",
                )
            )
        if definition_count:
            reasons.append(
                RankingReason("definitions", f"Defines {definition_count} symbol(s) captured by the parser.")
            )
        if reference_count:
            reasons.append(
                RankingReason("identifier_references", f"Contains {reference_count} symbol reference(s).")
            )

        return reasons

    def _get_important_file_tags(self, fname: str, rel_fname: str) -> List[Tag]:
        """Generate synthetic tags for important docs/configs that lack parser tags."""
        text = self.read_text_func_internal(fname)
        if not text:
            return []

        candidate_lines = []
        fallback_lines = []
        for line_num, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            if len(fallback_lines) < 4:
                fallback_lines.append(line_num)

            if (
                line_num == 1
                or stripped.startswith(("#", "[", "{"))
                or "=" in stripped
                or ":" in stripped
            ):
                candidate_lines.append(line_num)

            if len(candidate_lines) >= 6:
                break

        lois = candidate_lines or fallback_lines
        basename = Path(rel_fname).name
        return [
            Tag(rel_fname=rel_fname, fname=fname, line=line_num, name=basename, kind="doc")
            for line_num in lois
        ]
    
    def get_tags_raw(self, fname: str, rel_fname: str) -> List[Tag]:
        """Parse file to extract tags using Tree-sitter."""
        if not _HAS_GREP_AST:
            message = "grep-ast is required. Install with: pip install grep-ast"
            self._record_tag_failure(fname, message, cacheable=False)
            self.output_handlers['error'](message)
            return []

        lang = filename_to_lang(fname)
        if not lang:
            self._clear_tag_failure(fname)
            return []

        code = self.read_text_func_internal(fname)
        if not code:
            self._clear_tag_failure(fname)
            return []

        # Vue SFC: tree-sitter-vue treats <script> as raw_text, so we
        # extract script blocks and parse them with JS/TS parsers instead.
        if lang == "vue":
            return self._get_vue_tags(fname, rel_fname, code)

        parser_lang, query_lang = resolve_parser_config(fname, lang)
        scm_fname = get_scm_fname(query_lang)
        if not scm_fname:
            self._clear_tag_failure(fname)
            return []

        try:
            language = get_language(parser_lang)
            parser = get_parser(parser_lang)
        except Exception as err:
            message = format_parser_runtime_error(parser_lang, err)
            self._record_tag_failure(fname, message, cacheable=False)
            self.output_handlers['error'](f"Skipping file {fname}: {message}")
            return []

        try:
            tree = parser.parse(bytes(code, "utf-8"))

            # Use cached compiled query instead of re-reading SCM file per parse (Finding #2)
            if parser_lang not in self._query_cache:
                query_text = read_text(scm_fname, silent=True)
                if not query_text:
                    self._clear_tag_failure(fname)
                    return []
                self._query_cache[parser_lang] = Query(language, query_text)

            query = self._query_cache[parser_lang]
            cursor = QueryCursor(query)
            captures = cursor.captures(tree.root_node)
            
            tags = []
            # Process captures as a dictionary
            for capture_name, nodes in captures.items():
                for node in nodes:
                    if "name.definition" in capture_name:
                        kind = "def"
                    elif "name.reference" in capture_name:
                        kind = "ref"
                    else:
                        # Skip other capture types like 'reference.call' if not needed for tagging
                        continue 
                    
                    line_num = node.start_point[0] + 1
                    # Handle potential None value
                    name = node.text.decode('utf-8') if node.text else ""
                    
                    tags.append(Tag(
                        rel_fname=rel_fname,
                        fname=fname,
                        line=line_num,
                        name=name,
                        kind=kind
                    ))
            
            self._clear_tag_failure(fname)
            return tags
            
        except Exception as e:
            message = f"Failed to parse {rel_fname}: {e}"
            self._record_tag_failure(fname, message, cacheable=True)
            self.output_handlers['error'](f"Error parsing {fname}: {e}")
            return []

    # Match <script> tags allowing > inside quoted attributes (Vue 3.3+ generics)
    _VUE_SCRIPT_RE = re.compile(
        r'<script((?:[^>"\']|"[^"]*"|\'[^\']*\')*?)>(.*?)</script>',
        re.DOTALL
    )

    def _get_vue_tags(self, fname: str, rel_fname: str, vue_code: str) -> List[Tag]:
        """Extract tags from Vue SFC by parsing <script> blocks with JS/TS parsers."""
        all_tags = []

        for match in self._VUE_SCRIPT_RE.finditer(vue_code):
            attrs, script_content = match.group(1), match.group(2)
            if not script_content.strip():
                continue

            # Detect language from <script lang="ts"> or <script lang="tsx">
            lang_match = re.search(r'lang=["\'](\w+)["\']', attrs)
            script_lang = lang_match.group(1) if lang_match else "js"
            if script_lang == "tsx":
                parser_lang = "tsx"
                query_lang = "typescript"  # tsx uses typescript queries
            elif script_lang in ("ts", "typescript"):
                parser_lang = "typescript"
                query_lang = "typescript"
            else:
                parser_lang = "javascript"
                query_lang = "javascript"

            try:
                language = get_language(parser_lang)
                parser = get_parser(parser_lang)
            except Exception as err:
                message = format_parser_runtime_error(parser_lang, err)
                self._record_tag_failure(fname, message, cacheable=False)
                continue

            # Line offset: count newlines before script content starts
            line_offset = vue_code[:match.start(2)].count('\n')

            try:
                tree = parser.parse(bytes(script_content, "utf-8"))

                # Cache key: parser_lang (tsx Query differs from typescript Query
                # even though they share the same SCM text, because Query is
                # compiled against a specific Language object)
                if parser_lang not in self._query_cache:
                    scm_fname = get_scm_fname(query_lang)
                    if not scm_fname:
                        continue
                    query_text = read_text(scm_fname, silent=True)
                    if not query_text:
                        continue
                    self._query_cache[parser_lang] = Query(language, query_text)

                query = self._query_cache[parser_lang]
                cursor = QueryCursor(query)
                captures = cursor.captures(tree.root_node)

                for capture_name, nodes in captures.items():
                    for node in nodes:
                        if "name.definition" in capture_name:
                            kind = "def"
                        elif "name.reference" in capture_name:
                            kind = "ref"
                        else:
                            continue

                        line_num = node.start_point[0] + 1 + line_offset
                        name = node.text.decode('utf-8') if node.text else ""

                        all_tags.append(Tag(
                            rel_fname=rel_fname,
                            fname=fname,
                            line=line_num,
                            name=name,
                            kind=kind
                        ))
            except Exception as e:
                message = f"Failed to parse Vue script in {rel_fname}: {e}"
                self._record_tag_failure(fname, message, cacheable=True)
                self.output_handlers['error'](f"Error parsing Vue script in {fname}: {e}")

        if all_tags:
            self._clear_tag_failure(fname)
        return all_tags

    def get_ranked_tags(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        changed_fnames: Optional[Set[str]] = None,
        changed_neighbor_depth: int = 0,
        query: Optional[str] = None,
    ) -> Tuple[List[Tuple[float, Tag]], FileReport]:
        """Get ranked tags using PageRank algorithm with file report."""
        self._reset_request_state()

        # Return empty list and empty report if no files
        if not chat_fnames and not other_fnames:
            return [], FileReport(
                {},
                0,
                0,
                0,
                list(self.diagnostics),
                query=query,
                query_terms=self._extract_query_terms(query),
                changed_files=sorted(self.get_rel_fname(f) for f in (changed_fnames or set())),
                changed_neighbor_depth=changed_neighbor_depth,
            )
            
        if mentioned_fnames is None:
            mentioned_fnames = set()
        if mentioned_idents is None:
            mentioned_idents = set()
        if changed_fnames is None:
            changed_fnames = set()
        changed_neighbor_depth = max(0, changed_neighbor_depth)
        query_terms = self._extract_query_terms(query)

        raw_chat_fnames = [os.path.abspath(f) for f in chat_fnames]
        raw_other_fnames = [os.path.abspath(f) for f in other_fnames]
        raw_changed_fnames = {os.path.abspath(f) for f in changed_fnames}

        included: List[str] = []
        excluded: Dict[str, str] = {}
        total_definitions = 0
        total_references = 0
        tags_by_file: Dict[str, List[Tag]] = {}
        supplemental_tags_by_file: Dict[str, List[Tag]] = {}
        summary_kind_by_file: Dict[str, str] = {}
        summary_items_by_file: Dict[str, List[str]] = {}
        definitions_by_file: Dict[str, List[str]] = defaultdict(list)
        references_by_file: Dict[str, List[str]] = defaultdict(list)
        entrypoint_signals_by_file: Dict[str, List[str]] = {}
        public_api_signals_by_file: Dict[str, List[str]] = {}
        query_path_matches_by_file: Dict[str, List[str]] = {}
        query_symbol_matches_by_file: Dict[str, List[str]] = {}
        mentioned_identifiers_by_file: Dict[str, Set[str]] = defaultdict(set)
        semantic_links: List[SemanticLink] = []

        defines = defaultdict(set)
        references = defaultdict(set)

        # R4 Finding B1-F2: Pre-compute rel_fname once per file to avoid redundant Path operations
        raw_all_fnames = list(dict.fromkeys(raw_chat_fnames + raw_other_fnames))
        all_fnames = self._prepare_candidate_files(raw_all_fnames, excluded)
        abs_to_rel: Dict[str, str] = {f: self.get_rel_fname(f) for f in all_fnames}
        known_rel_fnames = set(abs_to_rel.values())
        chat_fnames = [fname for fname in raw_chat_fnames if fname in abs_to_rel]
        other_fnames = [fname for fname in raw_other_fnames if fname in abs_to_rel]
        changed_fnames = {fname for fname in raw_changed_fnames if fname in abs_to_rel}
        changed_rel_fnames = {abs_to_rel[f] for f in changed_fnames if f in abs_to_rel}
        is_test_file_by_rel = {rel_fname: self._is_test_file(rel_fname) for rel_fname in known_rel_fnames}
        related_tests_by_source = {
            rel_fname: self._find_related_test_files(rel_fname, known_rel_fnames)
            for rel_fname in known_rel_fnames
            if not is_test_file_by_rel.get(rel_fname, False)
        }
        related_sources_by_test: Dict[str, List[str]] = defaultdict(list)
        for source_rel_fname, related_tests in related_tests_by_source.items():
            for test_rel_fname in related_tests:
                related_sources_by_test[test_rel_fname].append(source_rel_fname)

        personalization = {}
        chat_fnames_set = set(chat_fnames)
        chat_rel_fnames = {abs_to_rel[f] for f in chat_fnames if f in abs_to_rel}

        for fname in all_fnames:
            rel_fname = abs_to_rel[fname]

            included.append(fname)
            
            tags = self.get_tags(fname, rel_fname)
            tags_by_file[fname] = tags
            failure_reason = self._tag_failures.get(fname)
            if failure_reason:
                excluded[fname] = failure_reason
                continue
            
            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    definitions_by_file[rel_fname].append(tag.name)
                    total_definitions += 1
                elif tag.kind == "ref":
                    references[tag.name].add(rel_fname)
                    references_by_file[rel_fname].append(tag.name)
                    total_references += 1

                if tag.name in mentioned_idents:
                    mentioned_identifiers_by_file[rel_fname].add(tag.name)

            if not tags and self._is_important_file(rel_fname):
                synthetic_tags = self._get_important_file_tags(fname, rel_fname)
                if synthetic_tags:
                    supplemental_tags_by_file[fname] = synthetic_tags

            summary_kind, summary_items = self._extract_file_summary(fname, rel_fname)
            if summary_items:
                summary_kind_by_file[rel_fname] = summary_kind or "config"
                summary_items_by_file[rel_fname] = summary_items[:5]

            has_definitions = any(tag.kind == "def" for tag in tags)
            if is_test_file_by_rel.get(rel_fname, False) and not has_definitions:
                synthetic_test_tags = self._get_test_file_tags(fname, rel_fname)
                if synthetic_test_tags:
                    supplemental_tags_by_file.setdefault(fname, []).extend(synthetic_test_tags)

            path_entrypoint_signals, path_public_api_signals = self._get_path_role_signals(rel_fname)
            runtime_entrypoint_signals, runtime_public_api_signals, runtime_role_tags = self._get_runtime_role_metadata(
                fname,
                rel_fname,
            )
            entrypoint_signals_by_file[rel_fname] = self._dedupe_preserve_order(
                path_entrypoint_signals + runtime_entrypoint_signals
            )
            public_api_signals_by_file[rel_fname] = self._dedupe_preserve_order(
                path_public_api_signals + runtime_public_api_signals
            )
            if runtime_role_tags:
                supplemental_tags_by_file.setdefault(fname, []).extend(runtime_role_tags)

            semantic_text = self.read_text_func_internal(fname)
            if semantic_text:
                semantic_links.extend(collect_semantic_links(rel_fname, semantic_text, known_rel_fnames))

            path_matches, symbol_matches = self._match_query_terms(
                rel_fname,
                tags + supplemental_tags_by_file.get(fname, []),
                query_terms,
            )
            query_path_matches_by_file[rel_fname] = path_matches
            query_symbol_matches_by_file[rel_fname] = symbol_matches
            
            # Set personalization for chat files
            if fname in chat_fnames_set:
                personalization[rel_fname] = 100.0
        
        # Build graph
        G = nx.MultiDiGraph()
        
        # Add nodes — use pre-computed rel_fnames
        G.add_nodes_from(abs_to_rel[f] for f in all_fnames)
        
        # R3 Finding B1-2: Batch edge addition instead of one-at-a-time
        edges_to_add = []
        for name, ref_fnames in references.items():
            def_fnames = defines.get(name, set())
            if len(ref_fnames) * len(def_fnames) > 1000:
                continue
            for ref_fname in ref_fnames:
                for def_fname in def_fnames:
                    if ref_fname != def_fname:
                        edges_to_add.append((ref_fname, def_fname, {'name': name}))
        for link in semantic_links:
            edge_name = link.target_symbol or link.source_symbol or f"__semantic_{link.relation}__"
            edges_to_add.append((link.source, link.target, {'name': edge_name}))
        G.add_edges_from(edges_to_add)
        
        if not G.nodes():
            return [], FileReport(
                excluded,
                total_definitions,
                total_references,
                len(raw_all_fnames),
                list(self.diagnostics),
                query=query,
                query_terms=query_terms,
                changed_files=sorted(changed_rel_fnames),
                changed_neighbor_depth=changed_neighbor_depth,
            )
        
        # Run PageRank
        try:
            ranks = self._calculate_pagerank(G, personalization)
        except Exception:
            # Fallback to uniform ranking
            ranks = {node: 1.0 for node in G.nodes()}
        
        # Add status prefix to exclusion reasons
        for fname in excluded:
            excluded[fname] = f"[EXCLUDED] {excluded[fname]}"

        # Create file report
        file_report = FileReport(
            excluded=excluded,
            definition_matches=total_definitions,
            reference_matches=total_references,
            total_files_considered=len(raw_all_fnames),
            diagnostics=list(self.diagnostics),
            query=query,
            query_terms=query_terms,
            changed_files=sorted(changed_rel_fnames),
            changed_neighbor_depth=changed_neighbor_depth,
        )
        
        # Collect and rank tags
        ranked_tags = []
        max_rank_by_file: Dict[str, float] = defaultdict(float)
        effective_rank_by_file: Dict[str, float] = {}
        context_boost_by_file: Dict[str, float] = {}
        changed_distance_by_file: Dict[str, int] = {}
        changed_anchors_by_file: Dict[str, List[str]] = defaultdict(list)

        if changed_rel_fnames:
            adjacency = defaultdict(set)
            for src_rel_fname, dst_rel_fname in G.edges():
                adjacency[src_rel_fname].add(dst_rel_fname)
                adjacency[dst_rel_fname].add(src_rel_fname)

            for changed_rel_fname in changed_rel_fnames:
                changed_distance_by_file.setdefault(changed_rel_fname, 0)
                if changed_rel_fname not in changed_anchors_by_file[changed_rel_fname]:
                    changed_anchors_by_file[changed_rel_fname].append(changed_rel_fname)

                if changed_neighbor_depth <= 0:
                    continue

                queue = [(changed_rel_fname, 0)]
                seen = {changed_rel_fname}
                while queue:
                    current_rel_fname, distance = queue.pop(0)
                    if distance >= changed_neighbor_depth:
                        continue
                    next_distance = distance + 1
                    heuristic_neighbors = set(adjacency.get(current_rel_fname, set()))
                    heuristic_neighbors.update(related_tests_by_source.get(current_rel_fname, []))
                    heuristic_neighbors.update(related_sources_by_test.get(current_rel_fname, []))
                    for neighbor_rel_fname in heuristic_neighbors:
                        if neighbor_rel_fname not in known_rel_fnames or neighbor_rel_fname in seen:
                            continue
                        seen.add(neighbor_rel_fname)
                        queue.append((neighbor_rel_fname, next_distance))
                        best_distance = changed_distance_by_file.get(neighbor_rel_fname)
                        if best_distance is None or next_distance < best_distance:
                            changed_distance_by_file[neighbor_rel_fname] = next_distance
                            changed_anchors_by_file[neighbor_rel_fname] = [changed_rel_fname]
                        elif next_distance == best_distance and changed_rel_fname not in changed_anchors_by_file[neighbor_rel_fname]:
                            changed_anchors_by_file[neighbor_rel_fname].append(changed_rel_fname)

        allowed_changed_scope = None
        if changed_rel_fnames:
            if changed_neighbor_depth > 0:
                allowed_changed_scope = {
                    rel_fname
                    for rel_fname, distance in changed_distance_by_file.items()
                    if distance <= changed_neighbor_depth
                }
            else:
                allowed_changed_scope = set(changed_rel_fnames)

        for rel_fname in known_rel_fnames:
            effective_rank, query_boost = self._get_query_rank_context(
                ranks.get(rel_fname, 0.0),
                query_path_matches_by_file.get(rel_fname, []),
                query_symbol_matches_by_file.get(rel_fname, []),
            )
            role_floor, role_boost = self._get_role_rank_context(
                entrypoint_signals_by_file.get(rel_fname, []),
                public_api_signals_by_file.get(rel_fname, []),
            )
            changed_floor, changed_boost = self._get_changed_rank_context(changed_distance_by_file.get(rel_fname))
            effective_rank_by_file[rel_fname] = max(effective_rank, role_floor, changed_floor)
            context_boost_by_file[rel_fname] = query_boost * role_boost * changed_boost
        
        for fname in included:
            rel_fname = abs_to_rel[fname]
            effective_file_rank = effective_rank_by_file.get(rel_fname, ranks.get(rel_fname, 0.0))
            context_boost = context_boost_by_file.get(rel_fname, 1.0)

            if allowed_changed_scope is not None and rel_fname not in allowed_changed_scope:
                continue

            if is_test_file_by_rel.get(rel_fname, False):
                related_source_support = 0.0
                for source_rel_fname in related_sources_by_test.get(rel_fname, []):
                    source_support = effective_rank_by_file.get(source_rel_fname, ranks.get(source_rel_fname, 0.0))
                    if source_rel_fname in chat_rel_fnames:
                        source_support *= 4.0
                    if source_rel_fname in mentioned_fnames:
                        source_support *= 1.5
                    source_support *= min(context_boost_by_file.get(source_rel_fname, 1.0), 2.0)
                    related_source_support = max(
                        related_source_support,
                        source_support * (0.5 * max(self.repo_config.ranking_weights.related_test, 0.0)),
                    )
                effective_file_rank = max(effective_file_rank, related_source_support)

            # Exclude files with low Page Rank if exclude_unranked is True
            if self.exclude_unranked and effective_file_rank <= 0.0001 and fname not in supplemental_tags_by_file:
                continue
            
            tags = tags_by_file.get(fname, [])
            for tag in tags:
                if tag.kind == "def":
                    final_rank = self._score_file_tag(
                        rel_fname,
                        tag.name,
                        effective_file_rank,
                        mentioned_fnames,
                        mentioned_idents,
                        chat_rel_fnames,
                        context_boost,
                    )
                    ranked_tags.append((final_rank, tag))
                    max_rank_by_file[rel_fname] = max(max_rank_by_file[rel_fname], final_rank)

            for tag in supplemental_tags_by_file.get(fname, []):
                final_rank = self._score_file_tag(
                    rel_fname,
                    tag.name,
                    max(effective_file_rank * 4.0, 0.02),
                    mentioned_fnames,
                    mentioned_idents,
                    chat_rel_fnames,
                    context_boost,
                )
                ranked_tags.append((final_rank, tag))
                max_rank_by_file[rel_fname] = max(max_rank_by_file[rel_fname], final_rank)

        ranked_files = []
        for fname in included:
            rel_fname = abs_to_rel[fname]
            tags = tags_by_file.get(fname, [])
            supplemental_tags = supplemental_tags_by_file.get(fname, [])

            if allowed_changed_scope is not None and rel_fname not in allowed_changed_scope:
                continue
            if not tags and not supplemental_tags:
                continue
            if rel_fname not in max_rank_by_file:
                continue

            inbound_neighbors = sorted(G.predecessors(rel_fname))
            outbound_neighbors = sorted(G.successors(rel_fname))
            definition_names = self._dedupe_preserve_order(definitions_by_file.get(rel_fname, []))
            reference_names = self._dedupe_preserve_order(references_by_file.get(rel_fname, []))
            entrypoint_signals = entrypoint_signals_by_file.get(rel_fname, [])
            public_api_signals = public_api_signals_by_file.get(rel_fname, [])
            summary_kind = summary_kind_by_file.get(rel_fname)
            summary_items = summary_items_by_file.get(rel_fname, [])
            matched_query_path_terms = query_path_matches_by_file.get(rel_fname, [])
            matched_query_symbol_terms = query_symbol_matches_by_file.get(rel_fname, [])
            related_tests = related_tests_by_source.get(rel_fname, [])
            related_sources = related_sources_by_test.get(rel_fname, [])
            mentioned_identifiers = sorted(mentioned_identifiers_by_file.get(rel_fname, set()))
            lines_of_interest = sorted({
                tag.line
                for tag in tags
                if tag.kind == "def"
            } | {
                tag.line
                for tag in supplemental_tags
            })

            ranked_files.append(
                RankedFile(
                    path=rel_fname,
                    rank=max_rank_by_file[rel_fname],
                    base_rank=ranks.get(rel_fname, 0.0),
                    is_changed_file=rel_fname in changed_rel_fnames,
                    changed_neighbor_distance=changed_distance_by_file.get(rel_fname),
                    is_test_file=is_test_file_by_rel.get(rel_fname, False),
                    is_entrypoint_file=bool(entrypoint_signals),
                    is_public_api_file=bool(public_api_signals),
                    is_chat_file=fname in chat_fnames_set,
                    is_mentioned_file=rel_fname in mentioned_fnames,
                    is_important_file=any(tag.kind == "doc" for tag in supplemental_tags),
                    definitions=len(definitions_by_file.get(rel_fname, [])),
                    references=len(references_by_file.get(rel_fname, [])),
                    referenced_by_files=len(inbound_neighbors),
                    references_to_files=len(outbound_neighbors),
                    matched_query_terms=self._dedupe_preserve_order(
                        matched_query_path_terms + matched_query_symbol_terms
                    ),
                    matched_query_path_terms=matched_query_path_terms,
                    matched_query_symbol_terms=matched_query_symbol_terms,
                    related_changed_files=changed_anchors_by_file.get(rel_fname, [])[:5],
                    related_tests=related_tests[:5],
                    related_sources=related_sources[:5],
                    entrypoint_signals=entrypoint_signals[:5],
                    public_api_signals=public_api_signals[:5],
                    summary_kind=summary_kind,
                    summary_items=summary_items[:5],
                    mentioned_identifiers=mentioned_identifiers,
                    sample_symbols=(definition_names or reference_names or [Path(rel_fname).name])[:5],
                    inbound_neighbors=inbound_neighbors[:5],
                    outbound_neighbors=outbound_neighbors[:5],
                    lines_of_interest=lines_of_interest[:10],
                    reasons=self._build_file_reasons(
                        rel_fname=rel_fname,
                        is_changed_file=rel_fname in changed_rel_fnames,
                        changed_neighbor_distance=changed_distance_by_file.get(rel_fname),
                        related_changed_files=changed_anchors_by_file.get(rel_fname, []),
                        is_test_file=is_test_file_by_rel.get(rel_fname, False),
                        is_entrypoint_file=bool(entrypoint_signals),
                        is_public_api_file=bool(public_api_signals),
                        is_chat_file=fname in chat_fnames_set,
                        is_mentioned_file=rel_fname in mentioned_fnames,
                        is_important_file=any(tag.kind == "doc" for tag in supplemental_tags),
                        matched_query_path_terms=matched_query_path_terms,
                        matched_query_symbol_terms=matched_query_symbol_terms,
                        related_tests=related_tests,
                        related_sources=related_sources,
                        entrypoint_signals=entrypoint_signals,
                        public_api_signals=public_api_signals,
                        summary_kind=summary_kind,
                        summary_items=summary_items,
                        mentioned_identifiers=mentioned_identifiers,
                        inbound_neighbors=inbound_neighbors,
                        outbound_neighbors=outbound_neighbors,
                        definition_count=len(definitions_by_file.get(rel_fname, [])),
                        reference_count=len(references_by_file.get(rel_fname, [])),
                    ),
                )
            )

        ranked_files.sort(key=lambda entry: entry.rank, reverse=True)
        file_report.ranked_files = ranked_files
        self.file_summary_by_file = {path: items[:] for path, items in summary_items_by_file.items()}
        self.file_summary_kind_by_file = summary_kind_by_file.copy()
        
        # Sort by rank (descending)
        ranked_tags.sort(key=lambda x: x[0], reverse=True)
        
        return ranked_tags, file_report
    
    def render_tree(self, abs_fname: str, rel_fname: str, lois: List[int]) -> str:
        """Render a code snippet with specific lines of interest."""
        # Cache formatted result per (file, lois) to avoid re-rendering in binary search
        file_mtime = self.get_mtime(abs_fname)
        cache_key = (rel_fname, file_mtime, tuple(sorted(set(lois))))
        if cache_key in self.tree_context_cache:
            return self.tree_context_cache[cache_key]

        code = self.read_text_func_internal(abs_fname)
        if not code:
            return ""

        try:
            tc = TreeContext(rel_fname, code, color=False)
            tc.add_lines_of_interest(lois)
            tc.add_context()
            result = tc.format()
        except Exception:
            # Fallback to simple line extraction
            lines = code.splitlines()
            result_lines = [f"{rel_fname}:"]
            for loi in sorted(set(lois)):
                if 1 <= loi <= len(lines):
                    result_lines.append(f"{loi:4d}: {lines[loi-1]}")
            result = "\n".join(result_lines)

        self.tree_context_cache[cache_key] = result
        return result
    
    def _group_and_sort_tags_by_file(self, tags: List[Tuple[float, Tag]]) -> List[Tuple[str, List[Tuple[float, Tag]]]]:
        """R3 Finding B1-1: Groups tags by file and sorts files by max rank. Called once before binary search."""
        if not tags:
            return []

        file_tags = defaultdict(list)
        for rank, tag in tags:
            file_tags[tag.rel_fname].append((rank, tag))

        return sorted(
            file_tags.items(),
            key=lambda x: max(rank for rank, tag in x[1]),
            reverse=True
        )

    def _format_summary_block(self, rel_fname: str) -> str:
        """Format structured summary bullets for a file when available."""
        summary_items = self.file_summary_by_file.get(rel_fname, [])
        if not summary_items:
            return ""

        summary_kind = self.file_summary_kind_by_file.get(rel_fname)
        label = "Doc Highlights" if summary_kind == "doc" else "Config Highlights"
        summary_lines = [f"({label})"]
        summary_lines.extend(f"- {item}" for item in summary_items[:5])
        return "\n".join(summary_lines)

    def _render_file_section(self, rel_fname: str, file_tag_list: List[Tuple[float, Tag]]) -> str:
        """Render one file's final map section including summaries."""
        lois = [tag.line for rank, tag in file_tag_list]
        abs_fname = str(self.root / rel_fname)
        max_rank = max(rank for rank, tag in file_tag_list)

        rendered = self.render_tree(abs_fname, rel_fname, lois)
        if not rendered:
            return ""

        rendered_lines = rendered.splitlines()
        first_line = rendered_lines[0]
        code_lines = rendered_lines[1:]
        summary_block = self._format_summary_block(rel_fname)
        sections = [f"{first_line}\n(Rank value: {max_rank:.4f})"]
        if summary_block:
            sections.append(summary_block)
        if code_lines:
            sections.append("\n".join(code_lines))
        return "\n\n".join(section for section in sections if section)

    def to_tree(self, sorted_files: List[Tuple[str, List[Tuple[float, Tag]]]]) -> str:
        """Convert pre-sorted files and tags to formatted tree output."""
        if not sorted_files:
            return ""

        tree_parts = []

        for rel_fname, file_tag_list in sorted_files:
            rendered = self._render_file_section(rel_fname, file_tag_list)
            if rendered:
                tree_parts.append(rendered)

        return "\n\n".join(tree_parts)
    
    def get_ranked_tags_map(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        changed_fnames: Optional[Set[str]] = None,
        changed_neighbor_depth: int = 0,
        query: Optional[str] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Get the ranked tags map with caching."""
        cache_key = (
            tuple(sorted(chat_fnames)),
            tuple(sorted(other_fnames)),
            max_map_tokens,
            tuple(sorted(mentioned_fnames or [])),
            tuple(sorted(mentioned_idents or [])),
            tuple(sorted(changed_fnames or [])),
            changed_neighbor_depth,
            query or "",
        )
        
        if not force_refresh and cache_key in self.map_cache:
            return self.map_cache[cache_key]
        
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens,
            mentioned_fnames, mentioned_idents, changed_fnames, changed_neighbor_depth, query
        )
        
        self.map_cache[cache_key] = result
        return result
    
    def get_ranked_tags_map_uncached(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        changed_fnames: Optional[Set[str]] = None,
        changed_neighbor_depth: int = 0,
        query: Optional[str] = None,
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the ranked tags map without caching."""
        ranked_tags, file_report = self.get_ranked_tags(
            chat_fnames,
            other_fnames,
            mentioned_fnames,
            mentioned_idents,
            changed_fnames,
            changed_neighbor_depth,
            query,
        )
        
        if not ranked_tags:
            return None, file_report

        # R3 Finding B1-1: Hoist grouping/sorting out of binary search loop
        sorted_files_with_tags = self._group_and_sort_tags_by_file(ranked_tags)

        # Filter out files whose individual tree exceeds the token limit
        fitting_files = []
        for entry in sorted_files_with_tags:
            rel_fname, file_tag_list = entry
            rendered = self._render_file_section(rel_fname, file_tag_list)
            if rendered and self.token_count(rendered) <= max_map_tokens:
                fitting_files.append(entry)

        if not fitting_files:
            return None, file_report

        def try_files(num_files: int) -> Tuple[Optional[str], int]:
            if num_files <= 0:
                return None, 0

            selected_file_entries = fitting_files[:num_files]
            tree_output = self.to_tree(selected_file_entries)

            if not tree_output:
                return None, 0

            tokens = self.token_count(tree_output)
            return tree_output, tokens

        # Binary search for optimal number of files (start at 1; try_files(0) is always None)
        left, right = 1, len(fitting_files)
        best_tree = None
        best_selected_files: List[str] = []
        best_tokens = 0

        while left <= right:
            mid = (left + right) // 2
            tree_output, tokens = try_files(mid)

            if tree_output and tokens <= max_map_tokens:
                best_tree = tree_output
                best_selected_files = [rel_fname for rel_fname, _ in fitting_files[:mid]]
                best_tokens = tokens
                left = mid + 1
            else:
                right = mid - 1

        selected_paths = set(best_selected_files)
        for ranked_file in file_report.ranked_files:
            ranked_file.included_in_map = ranked_file.path in selected_paths
        file_report.selected_files = best_selected_files
        file_report.map_tokens = best_tokens

        return best_tree, file_report
    
    def get_repo_map(
        self,
        chat_files: List[str] = None,
        other_files: List[str] = None,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        changed_fnames: Optional[Set[str]] = None,
        changed_neighbor_depth: int = 0,
        query: Optional[str] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the repository map with file report."""
        if chat_files is None:
            chat_files = []
        if other_files is None:
            other_files = []

        self._reset_request_state()
        self.file_summary_by_file = {}
        self.file_summary_kind_by_file = {}
            
        # Create empty report for error cases
        empty_report = FileReport(
            {},
            0,
            0,
            0,
            list(self.diagnostics),
            query=query,
            query_terms=self._extract_query_terms(query),
            changed_files=sorted(self.get_rel_fname(f) for f in (changed_fnames or set())),
            changed_neighbor_depth=changed_neighbor_depth,
        )
        
        if self.max_map_tokens <= 0 or not other_files:
            return None, empty_report
        
        # Adjust max_map_tokens if no chat files
        max_map_tokens = self.max_map_tokens
        if not chat_files and self.max_context_window:
            padding = 1024
            available = self.max_context_window - padding
            max_map_tokens = min(
                max_map_tokens * self.map_mul_no_files,
                available
            )
        
        try:
            # get_ranked_tags_map returns (map_string, file_report)
            map_string, file_report = self.get_ranked_tags_map(
                chat_files, other_files, max_map_tokens,
                mentioned_fnames, mentioned_idents, changed_fnames, changed_neighbor_depth, query, force_refresh
            )
        except RecursionError:
            message = "Disabling repo map, git repo too large?"
            self._add_diagnostic(message)
            self.output_handlers['error'](message)
            self.max_map_tokens = 0
            return None, FileReport(
                {},
                0,
                0,
                0,
                list(self.diagnostics),
                query=query,
                query_terms=self._extract_query_terms(query),
                changed_files=sorted(self.get_rel_fname(f) for f in (changed_fnames or set())),
                changed_neighbor_depth=changed_neighbor_depth,
            )  # Ensure consistent return type
        
        if map_string is None:
            return None, file_report
        
        if self.verbose:
            tokens = self.token_count(map_string)
            self.output_handlers['info'](f"Repo-map: {tokens / 1024:.1f} k-tokens")
        
        # Format final output
        other = "other " if chat_files else ""
        
        if self.repo_content_prefix:
            repo_content = self.repo_content_prefix.format(other=other)
        else:
            repo_content = ""
        
        repo_content += map_string
        
        return repo_content, file_report
