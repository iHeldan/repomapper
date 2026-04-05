"""
RepoMap class for generating repository maps.
"""

import os
import re
from pathlib import Path
from collections import namedtuple, defaultdict
from typing import List, Dict, Set, Optional, Tuple, Callable
import shutil
import sqlite3
from dataclasses import dataclass, field
import diskcache
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python
from grep_ast import TreeContext
from utils import count_tokens, read_text
from scm import get_scm_fname
from importance import is_important
from parser_support import resolve_parser_config, format_parser_runtime_error

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
        if tag_name in mentioned_idents:
            boost *= 10.0
        if rel_fname in mentioned_fnames:
            boost *= 5.0
        if rel_fname in chat_rel_fnames:
            boost *= 20.0
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

    def _is_test_file(self, rel_fname: str) -> bool:
        """Heuristically detect whether a file is a test/spec file."""
        normalized = self._normalize_rel_path(rel_fname).lower()
        parts = normalized.split("/")
        basename = parts[-1]
        stem = Path(basename).stem.lower()

        if any(part in TEST_DIR_NAMES for part in parts[:-1]):
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
            parent_parts + ["tests"],
            parent_parts + ["test"],
            parent_parts + ["__tests__"],
            parent_parts + ["spec"],
            ["tests"] + source_relative_parts,
            ["test"] + source_relative_parts,
            ["__tests__"] + source_relative_parts,
            ["spec"] + source_relative_parts,
        ]

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

        if basename in ENTRYPOINT_FILENAMES:
            entrypoint_signals.append("entrypoint_filename")
        if any(part in ENTRYPOINT_DIR_NAMES for part in parent_parts):
            entrypoint_signals.append("entrypoint_directory")

        if basename in PUBLIC_API_FILENAMES:
            public_api_signals.append("public_api_filename")
        if any(part in PUBLIC_API_DIR_NAMES for part in parent_parts):
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

        if entrypoint_signals:
            boost *= min(1.5 + (0.15 * max(len(entrypoint_signals) - 1, 0)), 2.0)
            rank_floor = max(rank_floor, 0.05 + (0.01 * min(len(entrypoint_signals), 3)))
        if public_api_signals:
            boost *= min(1.3 + (0.1 * max(len(public_api_signals) - 1, 0)), 1.8)
            rank_floor = max(rank_floor, 0.035 + (0.005 * min(len(public_api_signals), 3)))

        return rank_floor, boost

    def _get_changed_rank_context(self, changed_distance: Optional[int]) -> Tuple[float, float]:
        """Return rank floor and boost for changed files and their nearby impact neighbors."""
        if changed_distance is None:
            return 0.0, 1.0
        if changed_distance == 0:
            return 0.08, 8.0
        if changed_distance == 1:
            return 0.03, 2.5
        return 0.015, max(1.25, 2.25 - (0.25 * changed_distance))

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

        query_floor = (0.02 * len(path_matches)) + (0.05 * len(symbol_matches))
        path_boost = 1.0 + (0.75 * len(path_matches))
        symbol_boost = 1.0 + (1.25 * len(symbol_matches))
        query_boost = min(path_boost * symbol_boost, 8.0)
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
        # Return empty list and empty report if no files
        if not chat_fnames and not other_fnames:
            return [], FileReport(
                {},
                0,
                0,
                0,
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

        chat_fnames = [os.path.abspath(f) for f in chat_fnames]
        other_fnames = [os.path.abspath(f) for f in other_fnames]
        changed_fnames = {os.path.abspath(f) for f in changed_fnames}

        included: List[str] = []
        excluded: Dict[str, str] = {}
        total_definitions = 0
        total_references = 0
        tags_by_file: Dict[str, List[Tag]] = {}
        supplemental_tags_by_file: Dict[str, List[Tag]] = {}
        definitions_by_file: Dict[str, List[str]] = defaultdict(list)
        references_by_file: Dict[str, List[str]] = defaultdict(list)
        entrypoint_signals_by_file: Dict[str, List[str]] = {}
        public_api_signals_by_file: Dict[str, List[str]] = {}
        query_path_matches_by_file: Dict[str, List[str]] = {}
        query_symbol_matches_by_file: Dict[str, List[str]] = {}
        mentioned_identifiers_by_file: Dict[str, Set[str]] = defaultdict(set)

        defines = defaultdict(set)
        references = defaultdict(set)

        # R4 Finding B1-F2: Pre-compute rel_fname once per file to avoid redundant Path operations
        all_fnames = list(dict.fromkeys(chat_fnames + other_fnames))
        abs_to_rel: Dict[str, str] = {f: self.get_rel_fname(f) for f in all_fnames}
        known_rel_fnames = set(abs_to_rel.values())
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
            
            if not os.path.exists(fname):
                reason = "File not found"
                excluded[fname] = reason
                self.output_handlers['warning'](f"Repo-map can't include {fname}: {reason}")
                continue
                
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

            if not tags and is_important(rel_fname):
                synthetic_tags = self._get_important_file_tags(fname, rel_fname)
                if synthetic_tags:
                    supplemental_tags_by_file[fname] = synthetic_tags

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
        G.add_edges_from(edges_to_add)
        
        if not G.nodes():
            return [], FileReport(
                excluded,
                total_definitions,
                total_references,
                len(all_fnames),
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
            total_files_considered=len(all_fnames),
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
                    related_source_support = max(related_source_support, source_support * 0.5)
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

    def to_tree(self, sorted_files: List[Tuple[str, List[Tuple[float, Tag]]]]) -> str:
        """Convert pre-sorted files and tags to formatted tree output."""
        if not sorted_files:
            return ""

        tree_parts = []

        for rel_fname, file_tag_list in sorted_files:
            lois = [tag.line for rank, tag in file_tag_list]
            abs_fname = str(self.root / rel_fname)
            max_rank = max(rank for rank, tag in file_tag_list)

            rendered = self.render_tree(abs_fname, rel_fname, lois)
            if rendered:
                rendered_lines = rendered.splitlines()
                first_line = rendered_lines[0]
                code_lines = rendered_lines[1:]

                tree_parts.append(
                    f"{first_line}\n"
                    f"(Rank value: {max_rank:.4f})\n\n"
                    + "\n".join(code_lines)
                )

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
            lois = [tag.line for _, tag in file_tag_list]
            abs_fname = str(self.root / rel_fname)
            rendered = self.render_tree(abs_fname, rel_fname, lois)
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

        self.diagnostics = []
        self._tag_failures = {}
        self._uncacheable_tag_failures = set()
            
        # Create empty report for error cases
        empty_report = FileReport(
            {},
            0,
            0,
            0,
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
