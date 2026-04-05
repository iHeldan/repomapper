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
from dataclasses import dataclass
import diskcache
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_python
from grep_ast import TreeContext
from utils import count_tokens, read_text
from scm import get_scm_fname

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
class FileReport:
    excluded: Dict[str, str]        # File -> exclusion reason with status
    definition_matches: int         # Total definition tags
    reference_matches: int          # Total reference tags
    total_files_considered: int     # Total files provided as input



# Constants
CACHE_VERSION = 1

TAGS_CACHE_DIR = f".repomap.tags.cache.v{CACHE_VERSION}"
SQLITE_ERRORS = (sqlite3.OperationalError, sqlite3.DatabaseError)

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
        cache_entry = {"mtime": file_mtime, "data": tags}

        try:
            self.TAGS_CACHE[fname] = cache_entry
        except SQLITE_ERRORS:
            self.tags_cache_error()

        self._local_tags_cache[fname] = cache_entry
        return tags

    def _calculate_pagerank(self, graph: nx.MultiDiGraph, personalization: Dict[str, float]) -> Dict[str, float]:
        """Run PageRank with a pure-Python fallback when scipy/numpy is unavailable."""
        pagerank_kwargs = {"alpha": 0.85}
        if personalization:
            pagerank_kwargs["personalization"] = personalization

        try:
            return nx.pagerank(graph, **pagerank_kwargs)
        except (ImportError, ModuleNotFoundError):
            return _pagerank_python(graph, **pagerank_kwargs)
    
    def get_tags_raw(self, fname: str, rel_fname: str) -> List[Tag]:
        """Parse file to extract tags using Tree-sitter."""
        if not _HAS_GREP_AST:
            self.output_handlers['error']("grep-ast is required. Install with: pip install grep-ast")
            return []

        lang = filename_to_lang(fname)
        if not lang:
            return []

        code = self.read_text_func_internal(fname)
        if not code:
            return []

        # Vue SFC: tree-sitter-vue treats <script> as raw_text, so we
        # extract script blocks and parse them with JS/TS parsers instead.
        if lang == "vue":
            return self._get_vue_tags(fname, rel_fname, code)

        try:
            language = get_language(lang)
            parser = get_parser(lang)
        except Exception as err:
            self.output_handlers['error'](f"Skipping file {fname}: {err}")
            return []

        try:
            tree = parser.parse(bytes(code, "utf-8"))

            # Use cached compiled query instead of re-reading SCM file per parse (Finding #2)
            if lang not in self._query_cache:
                scm_fname = get_scm_fname(lang)
                if not scm_fname:
                    return []
                query_text = read_text(scm_fname, silent=True)
                if not query_text:
                    return []
                self._query_cache[lang] = Query(language, query_text)

            query = self._query_cache[lang]
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
            
            return tags
            
        except Exception as e:
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
            except Exception:
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
                self.output_handlers['error'](f"Error parsing Vue script in {fname}: {e}")

        return all_tags

    def get_ranked_tags(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None
    ) -> Tuple[List[Tuple[float, Tag]], FileReport]:
        """Get ranked tags using PageRank algorithm with file report."""
        # Return empty list and empty report if no files
        if not chat_fnames and not other_fnames:
            return [], FileReport({}, 0, 0, 0)
            
        if mentioned_fnames is None:
            mentioned_fnames = set()
        if mentioned_idents is None:
            mentioned_idents = set()

        chat_fnames = [os.path.abspath(f) for f in chat_fnames]
        other_fnames = [os.path.abspath(f) for f in other_fnames]

        included: List[str] = []
        excluded: Dict[str, str] = {}
        total_definitions = 0
        total_references = 0

        defines = defaultdict(set)
        references = defaultdict(set)

        # R4 Finding B1-F2: Pre-compute rel_fname once per file to avoid redundant Path operations
        all_fnames = list(dict.fromkeys(chat_fnames + other_fnames))
        abs_to_rel: Dict[str, str] = {f: self.get_rel_fname(f) for f in all_fnames}

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
            
            for tag in tags:
                if tag.kind == "def":
                    defines[tag.name].add(rel_fname)
                    total_definitions += 1
                elif tag.kind == "ref":
                    references[tag.name].add(rel_fname)
                    total_references += 1
            
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
            return [], FileReport(excluded, total_definitions, total_references, len(all_fnames))
        
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
            total_files_considered=len(all_fnames)
        )
        
        # Collect and rank tags
        ranked_tags = []
        
        for fname in included:
            rel_fname = abs_to_rel[fname]
            file_rank = ranks.get(rel_fname, 0.0)

            # Exclude files with low Page Rank if exclude_unranked is True
            if self.exclude_unranked and file_rank <= 0.0001:  # Use a small threshold to exclude near-zero ranks
                continue
            
            tags = self.get_tags(fname, rel_fname)
            for tag in tags:
                if tag.kind == "def":
                    # Boost for mentioned identifiers
                    boost = 1.0
                    if tag.name in mentioned_idents:
                        boost *= 10.0
                    if rel_fname in mentioned_fnames:
                        boost *= 5.0
                    if rel_fname in chat_rel_fnames:
                        boost *= 20.0
                    
                    final_rank = file_rank * boost
                    ranked_tags.append((final_rank, tag))
        
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
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Get the ranked tags map with caching."""
        cache_key = (
            tuple(sorted(chat_fnames)),
            tuple(sorted(other_fnames)),
            max_map_tokens,
            tuple(sorted(mentioned_fnames or [])),
            tuple(sorted(mentioned_idents or [])),
        )
        
        if not force_refresh and cache_key in self.map_cache:
            return self.map_cache[cache_key]
        
        result = self.get_ranked_tags_map_uncached(
            chat_fnames, other_fnames, max_map_tokens,
            mentioned_fnames, mentioned_idents
        )
        
        self.map_cache[cache_key] = result
        return result
    
    def get_ranked_tags_map_uncached(
        self,
        chat_fnames: List[str],
        other_fnames: List[str],
        max_map_tokens: int,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the ranked tags map without caching."""
        ranked_tags, file_report = self.get_ranked_tags(
            chat_fnames, other_fnames, mentioned_fnames, mentioned_idents
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

            selected_files = fitting_files[:num_files]
            tree_output = self.to_tree(selected_files)

            if not tree_output:
                return None, 0

            tokens = self.token_count(tree_output)
            return tree_output, tokens

        # Binary search for optimal number of files (start at 1; try_files(0) is always None)
        left, right = 1, len(fitting_files)
        best_tree = None

        while left <= right:
            mid = (left + right) // 2
            tree_output, tokens = try_files(mid)

            if tree_output and tokens <= max_map_tokens:
                best_tree = tree_output
                left = mid + 1
            else:
                right = mid - 1

        return best_tree, file_report
    
    def get_repo_map(
        self,
        chat_files: List[str] = None,
        other_files: List[str] = None,
        mentioned_fnames: Optional[Set[str]] = None,
        mentioned_idents: Optional[Set[str]] = None,
        force_refresh: bool = False
    ) -> Tuple[Optional[str], FileReport]:
        """Generate the repository map with file report."""
        if chat_files is None:
            chat_files = []
        if other_files is None:
            other_files = []
            
        # Create empty report for error cases
        empty_report = FileReport({}, 0, 0, 0)
        
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
                mentioned_fnames, mentioned_idents, force_refresh
            )
        except RecursionError:
            self.output_handlers['error']("Disabling repo map, git repo too large?")
            self.max_map_tokens = 0
            return None, FileReport({}, 0, 0, 0)  # Ensure consistent return type
        
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
