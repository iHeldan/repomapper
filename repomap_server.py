import asyncio
import dataclasses
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastmcp import FastMCP, settings
from git_support import get_changed_files, get_current_branch
from parser_support import infer_parser_languages, warm_languages, get_downloaded_parser_languages
from repomap_class import RepoMap, ImpactReport, ReviewReport
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
_DEFAULT_MCP_RANKED_FILES_LIMIT = 50
_DEFAULT_MCP_RANKED_FILES_PREVIEW_LIMIT = 10


def _get_project_state(file_paths: List[str]) -> tuple:
    """Build a cheap fingerprint so cached search indexes can be invalidated on edits."""
    state = []
    for file_path in file_paths:
        try:
            stat_result = os.stat(file_path)
        except OSError:
            continue
        state.append((file_path, stat_result.st_mtime_ns, stat_result.st_size))
    return tuple(sorted(state)), tuple(get_downloaded_parser_languages())


def _serialize_repo_map_report(
    file_report,
    *,
    include_ranked_files: bool = True,
    ranked_files_limit: Optional[int] = _DEFAULT_MCP_RANKED_FILES_LIMIT,
) -> Dict[str, Any]:
    """Serialize FileReport with MCP-friendly ranked_files trimming."""
    report = dataclasses.asdict(file_report)
    ranked_files = report.get("ranked_files") or []
    total_ranked_files = len(ranked_files)

    if not include_ranked_files:
        returned_ranked_files = []
    elif ranked_files_limit is None or ranked_files_limit <= 0:
        returned_ranked_files = ranked_files
    else:
        returned_ranked_files = ranked_files[:ranked_files_limit]

    report["ranked_files"] = returned_ranked_files
    report["ranked_files_total"] = total_ranked_files
    report["ranked_files_returned"] = len(returned_ranked_files)
    report["ranked_files_omitted"] = max(0, total_ranked_files - len(returned_ranked_files))
    report["ranked_files_truncated"] = len(returned_ranked_files) < total_ranked_files
    report["ranked_files_limit_applied"] = (
        None if not include_ranked_files or ranked_files_limit is None or ranked_files_limit <= 0 else ranked_files_limit
    )
    report["ranked_files_preview_limit"] = _DEFAULT_MCP_RANKED_FILES_PREVIEW_LIMIT
    report["ranked_files_preview"] = [
        {
            "path": entry.get("path"),
            "rank": entry.get("rank"),
            "base_rank": entry.get("base_rank"),
            "is_changed_file": entry.get("is_changed_file", False),
            "is_test_file": entry.get("is_test_file", False),
            "is_entrypoint_file": entry.get("is_entrypoint_file", False),
            "is_public_api_file": entry.get("is_public_api_file", False),
            "is_important_file": entry.get("is_important_file", False),
            "sample_symbols": (entry.get("sample_symbols") or [])[:3],
            "matched_query_terms": (entry.get("matched_query_terms") or [])[:3],
            "related_tests": (entry.get("related_tests") or [])[:3],
            "related_changed_files": (entry.get("related_changed_files") or [])[:3],
            "reason_codes": [
                reason.get("code")
                for reason in (entry.get("reasons") or [])[:5]
                if reason.get("code")
            ],
        }
        for entry in ranked_files[:_DEFAULT_MCP_RANKED_FILES_PREVIEW_LIMIT]
    ]
    report["ranked_files_counts"] = {
        "changed_files": sum(1 for entry in ranked_files if entry.get("is_changed_file")),
        "test_files": sum(1 for entry in ranked_files if entry.get("is_test_file")),
        "entrypoint_files": sum(1 for entry in ranked_files if entry.get("is_entrypoint_file")),
        "public_api_files": sum(1 for entry in ranked_files if entry.get("is_public_api_file")),
        "important_files": sum(1 for entry in ranked_files if entry.get("is_important_file")),
    }
    return report

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
    query: Optional[str] = None,
    token_limit: Any = 8192,  # Accept ints, auto presets, or structured AI-guided hints
    exclude_unranked: bool = False,
    force_refresh: bool = False,
    changed_only: bool = False,
    base_ref: Optional[str] = None,
    changed_neighbors: int = 0,
    download_missing_parsers: bool = False,
    mentioned_files: Optional[List[str]] = None,
    mentioned_idents: Optional[List[str]] = None,
    verbose: bool = False,
    max_context_window: Optional[int] = None,
    include_ranked_files: bool = True,
    ranked_files_limit: int = _DEFAULT_MCP_RANKED_FILES_LIMIT,
) -> Dict[str, Any]:
    """Generate a repository map for the specified files, providing a list of function prototypes and variables for files as well as relevant related
    files. Provide filenames relative to the project_root. In addition to the files provided, relevant related files will also be included with a
    very small ranking boost.

    :param project_root: Root directory of the project to search.  (must be an absolute path!)
    :param chat_files: A list of file paths that are currently in the chat context. These files will receive the highest ranking.
    :param other_files: A list of other relevant file paths in the repository to consider for the map. They receive a lower ranking boost than mentioned_files and chat_files.
    :param query: Optional free-form task/query string used to bias ranking toward matching paths and symbols.
    :param token_limit: The map token budget. Accepts an integer, `auto`, `small`, `medium`, `large`,
        or a structured object like {"mode": "ai_guided", "hint": "large"}. Defaults to 8192.
    :param exclude_unranked: If True, files with a PageRank of 0.0 will be excluded from the map. Defaults to False.
    :param force_refresh: If True, forces a refresh of the repository map cache. Defaults to False.
    :param changed_only: If True, limit the map to git-changed files under project_root.
    :param base_ref: Optional git ref to compare against when changed_only is enabled.
    :param changed_neighbors: When non-zero, include repository neighbors up to this graph distance around changed files.
    :param download_missing_parsers: If True, attempts to download required parser runtimes before parsing files.
    :param mentioned_files: Optional list of file paths explicitly mentioned in the conversation and receive a mid-level ranking boost.
    :param mentioned_idents: Optional list of identifiers explicitly mentioned in the conversation, to boost their ranking.
    :param verbose: If True, enables verbose logging for the RepoMap generation process. Defaults to False.
    :param max_context_window: Optional maximum context window size for token calculation, used to adjust map token limit when no chat files are provided.
    :param include_ranked_files: If False, omit detailed ranked_files entries from the MCP report while keeping summary counters.
    :param ranked_files_limit: Maximum number of ranked_files entries to include in the MCP report. Use 0 to return the full list.
    :returns: A dictionary containing:
        - 'map': the generated repository map string
        - 'report': a dictionary with file processing details including:
            - 'excluded': dictionary of excluded files with reasons
            - 'definition_matches': count of matched definitions
            - 'reference_matches': count of matched references
            - 'total_files_considered': total files processed
            - 'map_token_budget' / 'map_token_budget_mode': the effective cap and whether it was fixed vs auto/AI-guided
            - 'query' / 'query_terms': task query context used for ranking, when provided
            - 'changed_files' / 'changed_neighbor_depth': changed-file focus metadata for impact views
            - 'ranked_files': per-file rank metadata and reason codes (top 50 by default in MCP)
            - 'ranked_files_total' / 'ranked_files_returned' / 'ranked_files_truncated': summary metadata for omitted ranked file rows
            - 'ranked_files_preview': a compact top-ranked preview that stays small even for large repositories
        Or an 'error' key if an error occurred.
    """
    if error := _check_project_root(project_root):
        return error

    # 1. Handle and validate parameters
    changed_only = changed_only or bool(base_ref) or changed_neighbors > 0
    
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

        git_result = None
        changed_files = []
        if changed_only:
            git_result = get_changed_files(root_str, base_ref)
            if git_result.error:
                return {"error": git_result.error}

            changed_set = set(git_result.files)
            changed_files = [path for path in abs_other if path in changed_set] if other_files is not None else git_result.files
            if changed_neighbors > 0:
                if not changed_files:
                    abs_other = []
            else:
                abs_other = changed_files

        warmup_result = None
        if download_missing_parsers:
            warmup_result = warm_languages(infer_parser_languages(abs_chat + abs_other))

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
            changed_fnames=set(changed_files) if changed_files else None,
            changed_neighbor_depth=changed_neighbors,
            query=query,
            force_refresh=force_refresh
        )

        if warmup_result:
            if warmup_result.downloaded:
                file_report.diagnostics.append(
                    f"Downloaded parser runtimes: {', '.join(warmup_result.downloaded)}"
                )
            if warmup_result.error:
                file_report.diagnostics.append(warmup_result.error)
        if git_result:
            file_report.diagnostics.extend(git_result.diagnostics)
        return map_content, file_report

    try:
        result = await asyncio.to_thread(_prepare_and_run)
        if isinstance(result, dict):
            return result
        map_content, file_report = result
        
        return {
            "map": map_content or "No repository map could be generated.",
            "report": _serialize_repo_map_report(
                file_report,
                include_ranked_files=include_ranked_files,
                ranked_files_limit=ranked_files_limit,
            ),
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
    include_references: bool = True,
    download_missing_parsers: bool = False,
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
        download_missing_parsers: Whether to download required parser runtimes before indexing
    
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
        if download_missing_parsers:
            warm_languages(infer_parser_languages(all_files))
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


@mcp.tool()
async def trace_file_path(
    project_root: str,
    start_file: str,
    end_file: str,
    other_files: Optional[List[str]] = None,
    max_hops: int = 6,
    download_missing_parsers: bool = False,
) -> Dict[str, Any]:
    """Trace a shortest connection path between two files in the repository graph.

    :param project_root: Root directory of the project to search. (must be an absolute path!)
    :param start_file: Start file path relative to project_root, or an absolute path under it.
    :param end_file: End file path relative to project_root, or an absolute path under it.
    :param other_files: Optional file scope to limit the search. Defaults to all source files under project_root.
    :param max_hops: Maximum allowed hop count for the traced path. Defaults to 6.
    :param download_missing_parsers: If True, attempts to download required parser runtimes before building the graph.
    :returns: A dictionary containing the path, step metadata, symbol-level trace evidence, diagnostics, or an error.
    """
    if error := _check_project_root(project_root):
        return error

    root_path = Path(project_root).resolve()
    root_str = str(root_path)

    def _to_abs(f: str) -> str:
        p = Path(f)
        return str(p if p.is_absolute() else root_path / f)

    def _run_trace():
        effective_other_files = other_files if other_files is not None else find_src_files(project_root)
        abs_other = [_to_abs(f) for f in effective_other_files if _validate_path_containment(f, root_str)]
        abs_start = _to_abs(start_file)
        abs_end = _to_abs(end_file)

        if not _validate_path_containment(abs_start, root_str):
            return {"error": f"Start file resolves outside the project root: {start_file}"}
        if not _validate_path_containment(abs_end, root_str):
            return {"error": f"End file resolves outside the project root: {end_file}"}

        if download_missing_parsers:
            warm_languages(infer_parser_languages(abs_other + [abs_start, abs_end]))

        repo_mapper = RepoMap(
            root=str(root_path),
            token_counter_func=lambda text: count_tokens(text, "gpt-4"),
            file_reader_func=read_text,
            output_handler_funcs={'info': log.info, 'warning': log.warning, 'error': log.error},
            verbose=False,
            exclude_unranked=True
        )

        report = repo_mapper.trace_file_path(
            abs_start,
            abs_end,
            files=abs_other,
            max_hops=max_hops,
        )
        return dataclasses.asdict(report)

    try:
        return await asyncio.to_thread(_run_trace)
    except Exception as e:
        log.exception(f"Error tracing file path in project '{project_root}': {e}")
        return {"error": f"Error tracing file path: {str(e)}"}


@mcp.tool()
async def analyze_file_impact(
    project_root: str,
    seed_files: Optional[List[str]] = None,
    other_files: Optional[List[str]] = None,
    changed_only: bool = False,
    base_ref: Optional[str] = None,
    max_depth: int = 2,
    max_results: int = 10,
    download_missing_parsers: bool = False,
) -> Dict[str, Any]:
    """Analyze likely impact neighbors around the given seed files.

    :param project_root: Root directory of the project to search. (must be an absolute path!)
    :param seed_files: Seed file paths relative to project_root, or absolute paths under it.
    :param other_files: Optional file scope to limit the search. Defaults to all source files under project_root.
    :param changed_only: If True, derive the impact seed set from git-changed files under project_root.
    :param base_ref: Optional git ref to compare against when changed_only is enabled.
    :param max_depth: Maximum graph distance to consider. Defaults to 2.
    :param max_results: Maximum impacted files to return. Defaults to 10.
    :param download_missing_parsers: If True, attempts to download required parser runtimes before building the graph.
    :returns: A dictionary containing impacted files, quick actions, edit candidates, edit plan steps,
        grouped test clusters, symbol-level path evidence, reason metadata, diagnostics, or an error.
    """
    if error := _check_project_root(project_root):
        return error
    if changed_only and seed_files:
        return {"error": "Provide either seed_files or changed_only=true, not both."}
    if not changed_only and not seed_files:
        return {"error": "Provide seed_files, or set changed_only=true to derive them from git changes."}

    root_path = Path(project_root).resolve()
    root_str = str(root_path)

    def _to_abs(f: str) -> str:
        p = Path(f)
        return str(p if p.is_absolute() else root_path / f)

    def _run_impact():
        effective_other_files = other_files if other_files is not None else find_src_files(project_root)
        abs_other = [_to_abs(f) for f in effective_other_files if _validate_path_containment(f, root_str)]
        git_result = None
        provided_seed_files = seed_files or []

        if changed_only:
            git_result = get_changed_files(root_str, base_ref)
            if git_result.error:
                return {"error": git_result.error}
            changed_set = set(git_result.files)
            abs_seed_files = [path for path in abs_other if path in changed_set] if other_files is not None else git_result.files
            abs_seed_files = list(dict.fromkeys(abs_seed_files))
        else:
            abs_seed_files = [_to_abs(f) for f in provided_seed_files]

        invalid_seed_files = [
            seed_file for seed_file, abs_seed in zip(provided_seed_files, abs_seed_files)
            if not _validate_path_containment(abs_seed, root_str)
        ]
        if invalid_seed_files and not changed_only:
            return {
                "error": (
                    "Seed file(s) resolve outside the project root: "
                    + ", ".join(invalid_seed_files)
                )
            }

        if changed_only and not abs_seed_files:
            return dataclasses.asdict(
                ImpactReport(
                    seed_files=[],
                    max_depth=max_depth,
                    max_results=max_results,
                    diagnostics=(git_result.diagnostics[:] if git_result else []) + ["No changed files found for impact analysis."],
                )
            )

        if download_missing_parsers:
            warm_languages(infer_parser_languages(abs_other + abs_seed_files))

        repo_mapper = RepoMap(
            root=str(root_path),
            token_counter_func=lambda text: count_tokens(text, "gpt-4"),
            file_reader_func=read_text,
            output_handler_funcs={'info': log.info, 'warning': log.warning, 'error': log.error},
            verbose=False,
            exclude_unranked=True
        )

        report = repo_mapper.analyze_file_impact(
            abs_seed_files,
            files=abs_other,
            max_depth=max_depth,
            max_results=max_results,
            changed_lines_by_file=(git_result.changed_lines if git_result else None),
        )
        if git_result:
            report.diagnostics = list(dict.fromkeys(git_result.diagnostics + report.diagnostics))
        return dataclasses.asdict(report)

    try:
        return await asyncio.to_thread(_run_impact)
    except Exception as e:
        log.exception(f"Error analyzing file impact in project '{project_root}': {e}")
        return {"error": f"Error analyzing file impact: {str(e)}"}


@mcp.tool()
async def review_changes(
    project_root: str,
    other_files: Optional[List[str]] = None,
    base_ref: Optional[str] = None,
    max_depth: int = 2,
    max_results: int = 10,
    download_missing_parsers: bool = False,
) -> Dict[str, Any]:
    """Review git-changed files with combined branch, diff, test, public-API, and impact context.

    :param project_root: Root directory of the project to search. (must be an absolute path!)
    :param other_files: Optional file scope to limit the review. Defaults to all source files under project_root.
    :param base_ref: Optional git ref to compare against.
    :param max_depth: Maximum graph distance to consider for impact expansion. Defaults to 2.
    :param max_results: Maximum impacted files to return. Defaults to 10.
    :param download_missing_parsers: If True, attempts to download required parser runtimes before building the graph.
    :returns: A review-first payload combining changed files, review_focus, impact results, tests, and diagnostics.
    """
    if error := _check_project_root(project_root):
        return error

    root_path = Path(project_root).resolve()
    root_str = str(root_path)

    def _to_abs(f: str) -> str:
        p = Path(f)
        return str(p if p.is_absolute() else root_path / f)

    def _run_review():
        effective_other_files = other_files if other_files is not None else find_src_files(project_root)
        abs_other = [_to_abs(f) for f in effective_other_files if _validate_path_containment(f, root_str)]

        git_result = get_changed_files(root_str, base_ref)
        if git_result.error:
            return {"error": git_result.error}

        changed_set = set(git_result.files)
        abs_changed = [path for path in abs_other if path in changed_set] if other_files is not None else git_result.files
        abs_changed = list(dict.fromkeys(abs_changed))
        current_branch = get_current_branch(root_str)

        if not abs_changed:
            return dataclasses.asdict(
                ReviewReport(
                    current_branch=current_branch,
                    base_ref=base_ref,
                    max_depth=max_depth,
                    max_results=max_results,
                    diagnostics=git_result.diagnostics[:] + ["No changed files found for review mode."],
                )
            )

        if download_missing_parsers:
            warm_languages(infer_parser_languages(abs_other + abs_changed))

        repo_mapper = RepoMap(
            root=str(root_path),
            token_counter_func=lambda text: count_tokens(text, "gpt-4"),
            file_reader_func=read_text,
            output_handler_funcs={'info': log.info, 'warning': log.warning, 'error': log.error},
            verbose=False,
            exclude_unranked=True
        )

        report = repo_mapper.build_review_report(
            abs_changed,
            files=abs_other,
            current_branch=current_branch,
            base_ref=base_ref,
            max_depth=max_depth,
            max_results=max_results,
            changed_lines_by_file=git_result.changed_lines,
        )
        report.diagnostics = list(dict.fromkeys(git_result.diagnostics + report.diagnostics))
        return dataclasses.asdict(report)

    try:
        return await asyncio.to_thread(_run_review)
    except Exception as e:
        log.exception(f"Error reviewing changes in project '{project_root}': {e}")
        return {"error": f"Error reviewing changes: {str(e)}"}

# --- Main Entry Point ---
def main():
    # Run the MCP server
    log.debug("Starting FastMCP server...")
    mcp.run()

if __name__ == "__main__":
    main()
