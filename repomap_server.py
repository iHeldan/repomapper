import asyncio
import dataclasses
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastmcp import FastMCP, settings
from repomap_class import RepoMap
from utils import count_tokens, read_text, find_src_files, is_within_directory

# Security: only allow project roots under these directories (pre-resolved at module load)
_ALLOWED_ROOTS = [
    (Path.home() / "AI").resolve(),
    (Path.home() / "Projects").resolve(),
    (Path.home() / "Coding").resolve(),
]

def _check_project_root(project_root: str) -> Optional[Dict[str, str]]:
    """Validate project_root exists and is within allowed directories. Returns error dict or None."""
    if not os.path.isdir(project_root):
        return {"error": f"Project root directory not found: {project_root}"}
    resolved = Path(project_root).resolve()
    for allowed in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(allowed)
            return None
        except ValueError:
            continue
    return {"error": f"Project root '{project_root}' is outside allowed directories: {[str(a) for a in _ALLOWED_ROOTS]}"}

def _validate_path_containment(file_path: str, root_str: str) -> bool:
    """Check that file_path resolves to within root_str."""
    candidate = Path(file_path)
    resolved_path = candidate if candidate.is_absolute() else Path(root_str) / candidate
    return is_within_directory(str(resolved_path), root_str)

# R3 Finding B2-1: Cache RepoMap instances for search_identifiers
_REPO_MAP_CACHE: Dict[str, Any] = {}


def _get_project_state(file_paths: List[str]) -> tuple:
    """Build a cheap fingerprint so cached search indexes can be invalidated on edits."""
    state = []
    for file_path in file_paths:
        try:
            stat_result = os.stat(file_path)
        except OSError:
            continue
        state.append((file_path, stat_result.st_mtime_ns, stat_result.st_size))
    return tuple(sorted(state))

# Configure logging - only show errors
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)

# Create console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter('%(levelname)-5s %(asctime)-15s %(name)s:%(funcName)s:%(lineno)d - %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Suppress FastMCP logs
fastmcp_logger = logging.getLogger('fastmcp')
fastmcp_logger.setLevel(logging.ERROR)
# Suppress server startup message
server_logger = logging.getLogger('fastmcp.server')
server_logger.setLevel(logging.ERROR)

log = logging.getLogger(__name__)

# Set global stateless_http setting
settings.stateless_http = True

# Create MCP server
mcp = FastMCP("RepoMapServer")

@mcp.tool()
async def repo_map(
    project_root: str,
    chat_files: Optional[List[str]] = None,
    other_files: Optional[List[str]] = None,
    token_limit: Any = 8192,  # Accept any type to handle empty strings
    exclude_unranked: bool = False,
    force_refresh: bool = False,
    mentioned_files: Optional[List[str]] = None,
    mentioned_idents: Optional[List[str]] = None,
    verbose: bool = False,
    max_context_window: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a repository map for the specified files, providing a list of function prototypes and variables for files as well as relevant related
    files. Provide filenames relative to the project_root. In addition to the files provided, relevant related files will also be included with a
    very small ranking boost.

    :param project_root: Root directory of the project to search.  (must be an absolute path!)
    :param chat_files: A list of file paths that are currently in the chat context. These files will receive the highest ranking.
    :param other_files: A list of other relevant file paths in the repository to consider for the map. They receive a lower ranking boost than mentioned_files and chat_files.
    :param token_limit: The maximum number of tokens the generated repository map should occupy. Defaults to 8192.
    :param exclude_unranked: If True, files with a PageRank of 0.0 will be excluded from the map. Defaults to False.
    :param force_refresh: If True, forces a refresh of the repository map cache. Defaults to False.
    :param mentioned_files: Optional list of file paths explicitly mentioned in the conversation and receive a mid-level ranking boost.
    :param mentioned_idents: Optional list of identifiers explicitly mentioned in the conversation, to boost their ranking.
    :param verbose: If True, enables verbose logging for the RepoMap generation process. Defaults to False.
    :param max_context_window: Optional maximum context window size for token calculation, used to adjust map token limit when no chat files are provided.
    :returns: A dictionary containing:
        - 'map': the generated repository map string
        - 'report': a dictionary with file processing details including:
            - 'excluded': dictionary of excluded files with reasons
            - 'definition_matches': count of matched definitions
            - 'reference_matches': count of matched references
            - 'total_files_considered': total files processed
        Or an 'error' key if an error occurred.
    """
    if error := _check_project_root(project_root):
        return error

    # 1. Handle and validate parameters
    # Convert token_limit to integer with fallback
    try:
        token_limit = int(token_limit) if token_limit else 8192
    except (TypeError, ValueError):
        token_limit = 8192
    
    # Ensure token_limit is positive
    if token_limit <= 0:
        token_limit = 8192
    
    chat_files_list = chat_files or []
    mentioned_fnames_set = set(mentioned_files) if mentioned_files else None
    mentioned_idents_set = set(mentioned_idents) if mentioned_idents else None

    root_path = Path(project_root).resolve()

    # R2 Finding #2: move all blocking I/O (file discovery + RepoMap) into thread
    root_str = str(root_path)

    def _prepare_and_run():
        effective_other_files = other_files if other_files is not None else find_src_files(project_root)

        if not chat_files_list and not effective_other_files:
            return {"map": "No files found to generate a map."}

        def _to_abs(f: str) -> str:
            """Convert to absolute path, handling both relative and already-absolute paths."""
            p = Path(f)
            return str(p if p.is_absolute() else root_path / f)

        abs_chat = [_to_abs(f) for f in chat_files_list if _validate_path_containment(f, root_str)]
        abs_other = [_to_abs(f) for f in effective_other_files if _validate_path_containment(f, root_str)]
        abs_chat_set = set(abs_chat)
        abs_other = [f for f in abs_other if f not in abs_chat_set]

        repo_mapper = RepoMap(
            map_tokens=token_limit,
            root=str(root_path),
            token_counter_func=lambda text: count_tokens(text, "gpt-4"),
            file_reader_func=read_text,
            output_handler_funcs={'info': log.info, 'warning': log.warning, 'error': log.error},
            verbose=verbose,
            exclude_unranked=exclude_unranked,
            max_context_window=max_context_window
        )

        map_content, file_report = repo_mapper.get_repo_map(
            chat_files=abs_chat,
            other_files=abs_other,
            mentioned_fnames=mentioned_fnames_set,
            mentioned_idents=mentioned_idents_set,
            force_refresh=force_refresh
        )
        return map_content, file_report

    try:
        result = await asyncio.to_thread(_prepare_and_run)
        if isinstance(result, dict):
            return result
        map_content, file_report = result
        
        return {
            "map": map_content or "No repository map could be generated.",
            "report": dataclasses.asdict(file_report)
        }
    except Exception as e:
        log.exception(f"Error generating repository map for project '{project_root}': {e}")
        return {"error": f"Error generating repository map: {str(e)}"}
    
@mcp.tool()
async def search_identifiers(
    project_root: str,
    query: str,
    max_results: int = 50,
    context_lines: int = 2,
    include_definitions: bool = True,
    include_references: bool = True
) -> Dict[str, Any]:
    """Search for identifiers in code files. Get back a list of matching identifiers with their file, line number, and context.
       When searching, just use the identifier name without any special characters, prefixes or suffixes. The search is 
       case-insensitive.

    Args:
        project_root: Root directory of the project to search.  (must be an absolute path!)
        query: Search query (identifier name)
        max_results: Maximum number of results to return
        context_lines: Number of lines of context to show
        include_definitions: Whether to include definition occurrences
        include_references: Whether to include reference occurrences
    
    Returns:
        Dictionary containing search results or error message
    """
    if error := _check_project_root(project_root):
        return error

    # Finding #1: run blocking work in thread to avoid freezing async event loop
    def _run_search():
        # R3 Finding B2-1: Reuse cached RepoMap to preserve _local_tags_cache
        if project_root not in _REPO_MAP_CACHE:
            _REPO_MAP_CACHE[project_root] = RepoMap(
                root=project_root,
                token_counter_func=lambda text: count_tokens(text, "gpt-4"),
                file_reader_func=read_text,
                output_handler_funcs={'info': log.info, 'warning': log.warning, 'error': log.error},
                verbose=False,
                exclude_unranked=True
            )
        repo_map = _REPO_MAP_CACHE[project_root]

        all_files = find_src_files(project_root)
        project_state = _get_project_state(all_files)

        # Rebuild cached search index whenever any tracked file changes.
        if getattr(repo_map, '_project_tags_index_state', None) != project_state:
            all_tags = []
            for file_path in all_files:
                rel_path = repo_map.get_rel_fname(file_path)
                tags = repo_map.get_tags(file_path, rel_path)
                all_tags.extend(tags)
            repo_map._project_tags_index = all_tags
            repo_map._project_tags_index_state = project_state
        all_tags = repo_map._project_tags_index

        matching_tags = []
        query_lower = query.lower()

        for tag in all_tags:
            if query_lower in tag.name.lower():
                if (tag.kind == "def" and include_definitions) or \
                   (tag.kind == "ref" and include_references):
                    matching_tags.append(tag)

        matching_tags.sort(key=lambda x: (x.kind != "def", x.name.lower().find(query_lower)))
        matching_tags = matching_tags[:max_results]

        results = []
        for tag in matching_tags:
            file_path = str(Path(project_root) / tag.rel_fname)

            start_line = max(1, tag.line - context_lines)
            end_line = tag.line + context_lines
            context_range = list(range(start_line, end_line + 1))

            context = repo_map.render_tree(
                file_path,
                tag.rel_fname,
                context_range
            )

            if context:
                results.append({
                    "file": tag.rel_fname,
                    "line": tag.line,
                    "name": tag.name,
                    "kind": tag.kind,
                    "context": context
                })

        return {"results": results}

    try:
        return await asyncio.to_thread(_run_search)
    except Exception as e:
        log.exception(f"Error searching identifiers in project '{project_root}': {e}")
        return {"error": f"Error searching identifiers: {str(e)}"}    

# --- Main Entry Point ---
def main():
    # Run the MCP server
    log.debug("Starting FastMCP server...")
    mcp.run()

if __name__ == "__main__":
    main()
