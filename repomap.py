#!/usr/bin/env python3
"""
Standalone RepoMap Tool

A command-line tool that generates a "map" of a software repository,
highlighting important files and definitions based on their relevance.
Uses Tree-sitter for parsing and PageRank for ranking importance.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from git_support import get_changed_files
from utils import count_tokens, read_text, find_src_files
from parser_support import infer_parser_languages, warm_languages
from repomap_class import RepoMap


def resolve_repo_path(root_path: Path, path_str: str) -> Path:
    """Resolve CLI file arguments relative to the repository root."""
    path = Path(path_str)
    return path if path.is_absolute() else root_path / path


def expand_path_specs(root_path: Path, path_specs: list[str]) -> list[str]:
    """Expand file/directory specs into absolute file paths."""
    expanded_paths = []
    for path_spec in path_specs:
        expanded_paths.extend(find_src_files(str(resolve_repo_path(root_path, path_spec))))
    return [str(Path(path).resolve()) for path in expanded_paths]


def tool_output(*messages):
    """Print informational messages."""
    print(*messages, file=sys.stdout)


def tool_info_stderr(message):
    """Print informational messages to stderr."""
    print(message, file=sys.stderr)


def tool_warning(message):
    """Print warning messages."""
    print(f"Warning: {message}", file=sys.stderr)


def tool_error(message):
    """Print error messages."""
    print(f"Error: {message}", file=sys.stderr)


def report_parser_warmup(result, info_handler, warning_handler) -> bool:
    """Report parser warmup status. Returns True when all requested parsers are available."""
    if result.downloaded:
        info_handler(f"Downloaded parser runtimes: {', '.join(result.downloaded)}")
    elif result.available:
        info_handler(f"Parser runtimes already available: {', '.join(result.available)}")

    if result.error:
        warning_handler(result.error)

    return not result.missing


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a repository map showing important code structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .                    # Map current directory
  %(prog)s src/ --map-tokens 2048  # Map src/ with 2048 token limit
  %(prog)s file1.py file2.py    # Map specific files
  %(prog)s --chat-files main.py --other-files src/  # Specify chat vs other files
        """
    )
    
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to include in the map"
    )
    
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root directory (default: current directory)"
    )
    
    parser.add_argument(
        "--map-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for the generated map (default: 8192)"
    )
    
    parser.add_argument(
        "--chat-files",
        nargs="*",
        help="Files currently being edited (given higher priority)"
    )
    
    parser.add_argument(
        "--other-files",
        nargs="*",
        help="Other files to consider for the map"
    )
    
    parser.add_argument(
        "--mentioned-files",
        nargs="*",
        help="Files explicitly mentioned (given higher priority)"
    )
    
    parser.add_argument(
        "--mentioned-idents",
        nargs="*",
        help="Identifiers explicitly mentioned (given higher priority)"
    )

    parser.add_argument(
        "--query",
        help="Free-form task or search query used to bias ranking toward relevant paths and symbols"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model name for token counting (default: gpt-4)"
    )
    
    parser.add_argument(
        "--max-context-window",
        type=int,
        help="Maximum context window size"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of caches"
    )

    parser.add_argument(
        "--exclude-unranked",
        action="store_true",
        help="Exclude files with Page Rank 0 from the map"
    )

    parser.add_argument(
        "--download-missing-parsers",
        action="store_true",
        help="Download required parser runtimes before generating the map"
    )

    parser.add_argument(
        "--warm-languages",
        nargs="+",
        help="Download parser runtimes and exit. Use 'auto' to infer languages from the selected files."
    )

    parser.add_argument(
        "--changed",
        action="store_true",
        help="Limit the map to files changed in git (includes staged, unstaged, and untracked files)"
    )

    parser.add_argument(
        "--base-ref",
        help="When used with --changed, also include committed changes since the merge-base with this git ref"
    )

    parser.add_argument(
        "--changed-neighbors",
        type=int,
        default=0,
        help="When used with --changed, include repository neighbors up to this graph distance around changed files"
    )

    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="Output as human-readable text or machine-readable JSON (default: text)"
    )
    
    args = parser.parse_args()
    
    # Set up token counter with specified model
    def token_counter(text: str) -> int:
        return count_tokens(text, args.model)
    
    # Set up output handlers
    info_handler = tool_output if args.output_format == "text" else tool_info_stderr
    output_handlers = {
        'info': info_handler,
        'warning': tool_warning,
        'error': tool_error
    }
    
    args.changed = args.changed or bool(args.base_ref) or args.changed_neighbors > 0

    # Process file arguments
    root_path = Path(args.root).resolve()
    chat_files_from_args = args.chat_files or [] # These are the paths as strings from the CLI
    explicit_other_specs = bool(args.other_files or args.paths)
    
    # Determine the list of unresolved path specifications that will form the 'other_files'
    # These can be files or directories. find_src_files will expand them.
    unresolved_paths_for_other_files_specs = []
    if args.other_files:  # If --other-files is explicitly provided, it's the source
        unresolved_paths_for_other_files_specs.extend(args.other_files)
    elif args.paths:  # Else, if positional paths are given, they are the source
        unresolved_paths_for_other_files_specs.extend(args.paths)
    else:
        unresolved_paths_for_other_files_specs.append(str(root_path))
    # If neither, unresolved_paths_for_other_files_specs remains empty.

    # Expand relative path specs against the repository root before collecting files.
    chat_files = [str(resolve_repo_path(root_path, f).resolve()) for f in chat_files_from_args]
    other_files = expand_path_specs(root_path, unresolved_paths_for_other_files_specs)

    changed_files = []

    if args.changed:
        git_result = get_changed_files(str(root_path), args.base_ref)
        if git_result.error:
            tool_error(git_result.error)
            sys.exit(1)

        changed_set = set(git_result.files)
        changed_files = [path for path in other_files if path in changed_set] if explicit_other_specs else git_result.files
        if args.changed_neighbors > 0:
            if not changed_files:
                other_files = []
        else:
            other_files = changed_files

        if args.verbose:
            for diagnostic in git_result.diagnostics:
                info_handler(diagnostic)
            info_handler(f"Changed files selected: {len(changed_files)}")
            if args.changed_neighbors > 0 and changed_files:
                info_handler(
                    f"Including repository neighbors up to distance {args.changed_neighbors} around changed files."
                )

    inferred_parser_languages = infer_parser_languages(chat_files + other_files)

    if args.warm_languages:
        requested_languages = inferred_parser_languages if args.warm_languages == ["auto"] else args.warm_languages
        warmup_result = warm_languages(requested_languages)
        success = report_parser_warmup(warmup_result, info_handler, tool_warning)
        if not requested_languages:
            tool_warning("No supported parser runtimes were inferred from the selected files.")
        sys.exit(0 if success else 1)

    if args.download_missing_parsers:
        warmup_result = warm_languages(inferred_parser_languages)
        report_parser_warmup(warmup_result, info_handler, tool_warning)

    if args.verbose:
        info_handler(f"Chat files: {chat_files}")

    # Convert mentioned files to sets
    mentioned_fnames = set(args.mentioned_files) if args.mentioned_files else None
    mentioned_idents = set(args.mentioned_idents) if args.mentioned_idents else None
    
    # Create RepoMap instance
    repo_map = RepoMap(
        map_tokens=args.map_tokens,
        root=str(root_path),
        token_counter_func=token_counter,
        file_reader_func=read_text,
        output_handler_funcs=output_handlers,
        verbose=args.verbose,
        max_context_window=args.max_context_window,
        exclude_unranked=args.exclude_unranked
    )
    
    # Generate the map
    try:
        map_content, file_report = repo_map.get_repo_map(
            chat_files=chat_files,
            other_files=other_files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            changed_fnames=set(changed_files) if changed_files else None,
            changed_neighbor_depth=args.changed_neighbors,
            query=args.query,
            force_refresh=args.force_refresh
        )

        if map_content:
            if args.output_format == "json":
                print(json.dumps({
                    "map": map_content,
                    "report": dataclasses.asdict(file_report),
                }, indent=2))
            else:
                if args.verbose:
                    tokens = repo_map.token_count(map_content)
                    tool_output(f"Generated map: {len(map_content)} chars, ~{tokens} tokens")
                    tool_output(f"Files considered: {file_report.total_files_considered}, "
                               f"Definitions: {file_report.definition_matches}, "
                               f"References: {file_report.reference_matches}")

                print(map_content)
        else:
            if args.output_format == "json":
                print(json.dumps({
                    "map": None,
                    "report": dataclasses.asdict(file_report),
                }, indent=2))
            else:
                tool_output("No repository map generated.")
                for diagnostic in file_report.diagnostics:
                    tool_warning(diagnostic)
            
    except KeyboardInterrupt:
        tool_error("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        tool_error(f"Error generating repository map: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
