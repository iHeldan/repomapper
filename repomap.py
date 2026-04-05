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
from repomap_class import RepoMap, ImpactReport


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


def format_connection_report(report) -> str:
    """Render a connection path report as readable text."""
    if report.error:
        lines = [f"No path found: {report.error}"]
        if report.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            lines.extend(f"- {message}" for message in report.diagnostics)
        return "\n".join(lines)

    lines = [
        f"Connection path: {report.start_file} -> {report.end_file}",
        f"Hops: {max(0, len(report.path) - 1)}",
        "",
        "Path:",
    ]
    lines.extend(f"- {path}" for path in report.path)
    if report.steps:
        lines.append("")
        lines.append("Steps:")
        for step in report.steps:
            symbol_suffix = f" via {', '.join(step.symbols[:4])}" if step.symbols else ""
            lines.append(f"- {step.source} --{step.relation}{symbol_suffix}--> {step.target}")
    if report.diagnostics:
        lines.append("")
        lines.append("Diagnostics:")
        lines.extend(f"- {message}" for message in report.diagnostics)
    return "\n".join(lines)


def format_impact_report(report) -> str:
    """Render an impact analysis report as readable text."""
    if report.error:
        lines = [f"Impact analysis failed: {report.error}"]
        if report.diagnostics:
            lines.append("")
            lines.append("Diagnostics:")
            lines.extend(f"- {message}" for message in report.diagnostics)
        return "\n".join(lines)

    lines = [
        f"Impact analysis from: {', '.join(report.seed_files) if report.seed_files else '(none)'}",
        f"Depth: <= {report.max_depth} hops",
        f"Results: {len(report.impacted_files)}",
    ]

    for target in report.impacted_files:
        lines.append("")
        lines.append(f"{target.path} (distance {target.distance}, from {target.seed_file})")
        lines.append(f"Path: {' -> '.join(target.path_from_seed)}")
        if target.seed_focus_lines:
            lines.append(f"Seed lines: {', '.join(str(line) for line in target.seed_focus_lines[:6])}")
        if target.seed_hunks:
            lines.append(
                "Seed hunks: "
                + ", ".join(
                    str(hunk.start_line) if hunk.start_line == hunk.end_line else f"{hunk.start_line}-{hunk.end_line}"
                    for hunk in target.seed_hunks[:4]
                )
            )
        if target.changed_boundary_symbols:
            lines.append(f"Changed boundary symbols: {', '.join(target.changed_boundary_symbols[:5])}")
        if target.changed_boundary_distances:
            lines.append(
                "Changed boundary distances: "
                + ", ".join(
                    f"{symbol}:{distance}"
                    for symbol, distance in list(target.changed_boundary_distances.items())[:5]
                )
            )
        if target.closest_changed_hunk_distance is not None:
            lines.append(f"Closest changed hunk distance: {target.closest_changed_hunk_distance}")
        if target.boundary_symbols:
            lines.append(f"Boundary symbols: {', '.join(target.boundary_symbols[:5])}")
        if target.boundary_relations:
            lines.append(f"Boundary relations: {', '.join(target.boundary_relations[:4])}")
        if target.focus_lines:
            lines.append(f"Target lines: {', '.join(str(line) for line in target.focus_lines[:6])}")
        if target.boundary_locations:
            preview = "; ".join(
                f"{location.file}:{location.line} {location.kind} {location.symbol}"
                for location in target.boundary_locations[:5]
            )
            lines.append(f"Boundary locations: {preview}")
        if target.boundary_snippets:
            lines.append("Boundary snippets:")
            for snippet in target.boundary_snippets[:3]:
                lines.append(f"  {snippet.file}:{snippet.highlight_line} {snippet.kind} {snippet.symbol}")
                lines.extend(f"    {line}" for line in snippet.excerpt.splitlines())
        if target.steps:
            relation_chunks = []
            for step in target.steps:
                symbol_suffix = f"[{', '.join(step.symbols[:2])}]" if step.symbols else ""
                relation_chunks.append(f"{step.relation}{symbol_suffix}")
            lines.append(
                "Relations: "
                + " -> ".join(relation_chunks)
            )
        if target.reasons:
            lines.append(f"Why: {target.reasons[0].message}")

    if report.quick_actions:
        lines.append("")
        lines.append("Quick actions:")
        for action in report.quick_actions:
            lines.append(f"- [P{action.priority}] ({action.effort}, risk {action.risk_level}, confidence {action.confidence:.2f}) {action.kind} {action.target}: {action.message}")
            if action.focus_symbols:
                lines.append(f"  focus: {', '.join(action.focus_symbols[:3])}")
            if action.why_now:
                lines.append(f"  why now: {action.why_now}")
            if action.expected_outcome:
                lines.append(f"  expect: {action.expected_outcome}")
            if action.follow_if_true:
                lines.append(f"  if yes: {action.follow_if_true}")
            if action.follow_if_false:
                lines.append(f"  if no: {action.follow_if_false}")
            if action.location_hint:
                lines.append(f"  open {action.location_hint}")
            if action.command_hint:
                lines.append(f"  run {action.command_hint}")
            if action.anchor_file and action.anchor_line:
                anchor_suffix = f" ({action.anchor_kind} {action.anchor_symbol})" if action.anchor_symbol else ""
                lines.append(f"  at {action.anchor_file}:{action.anchor_line}{anchor_suffix}")
            if action.anchor_excerpt:
                lines.extend(f"    {line}" for line in action.anchor_excerpt.splitlines()[:3])

    if report.suggested_checks:
        lines.append("")
        lines.append("Suggested checks:")
        for suggestion in report.suggested_checks:
            lines.append(f"- [P{suggestion.priority}] {suggestion.kind} {suggestion.target}: {suggestion.message}")
            if suggestion.anchor_file and suggestion.anchor_line:
                anchor_suffix = f" ({suggestion.anchor_kind} {suggestion.anchor_symbol})" if suggestion.anchor_symbol else ""
                lines.append(f"  at {suggestion.anchor_file}:{suggestion.anchor_line}{anchor_suffix}")
            if suggestion.anchor_excerpt:
                lines.extend(f"    {line}" for line in suggestion.anchor_excerpt.splitlines()[:3])

    if report.shared_symbols:
        lines.append("")
        lines.append("Shared symbols:")
        for symbol in report.shared_symbols[:8]:
            distance_suffix = f", closest hop {symbol.closest_distance}" if symbol.closest_distance is not None else ""
            changed_suffix = " [changed]" if symbol.is_changed_seed_symbol else ""
            changed_hunk_suffix = (
                f", hunk distance {symbol.closest_changed_hunk_distance}"
                if symbol.closest_changed_hunk_distance is not None
                else ""
            )
            location_suffix = ""
            if symbol.locations:
                location_preview = ", ".join(
                    f"{location.file}:{location.line} {location.kind}"
                    for location in symbol.locations[:3]
                )
                location_suffix = f" [{location_preview}]"
            lines.append(
                f"- {symbol.name}{changed_suffix}: {symbol.target_count} target(s){distance_suffix}{changed_hunk_suffix} -> {', '.join(symbol.target_files[:4])}{location_suffix}"
            )

    if report.changed_seed_symbols:
        lines.append("")
        lines.append("Changed seed symbols:")
        for seed_file, symbols in sorted(report.changed_seed_symbols.items()):
            seed_lines = report.changed_lines_by_file.get(seed_file, [])
            seed_hunks = report.changed_hunks_by_file.get(seed_file, [])
            line_suffix = f" (lines {', '.join(str(line) for line in seed_lines[:6])})" if seed_lines else ""
            hunk_suffix = ""
            if seed_hunks:
                hunk_suffix = " hunks " + ", ".join(
                    str(hunk.start_line) if hunk.start_line == hunk.end_line else f"{hunk.start_line}-{hunk.end_line}"
                    for hunk in seed_hunks[:4]
                )
            lines.append(f"- {seed_file}{line_suffix}{hunk_suffix}: {', '.join(symbols[:6])}")

    if report.diagnostics:
        lines.append("")
        lines.append("Diagnostics:")
        lines.extend(f"- {message}" for message in report.diagnostics)
    return "\n".join(lines)


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
        "--trace-from",
        help="Find a file-level connection path starting from this file"
    )

    parser.add_argument(
        "--trace-to",
        help="Find a file-level connection path ending at this file"
    )

    parser.add_argument(
        "--trace-max-hops",
        type=int,
        default=6,
        help="Maximum allowed hop count when tracing file connections (default: 6)"
    )

    parser.add_argument(
        "--impact-from",
        nargs="+",
        help="Analyze which nearby files are most likely affected by the given file(s)"
    )

    parser.add_argument(
        "--impact-changed",
        action="store_true",
        help="Analyze likely impact radius around git-changed files"
    )

    parser.add_argument(
        "--impact-max-depth",
        type=int,
        default=2,
        help="Maximum graph distance to consider for impact analysis (default: 2)"
    )

    parser.add_argument(
        "--impact-max-results",
        type=int,
        default=10,
        help="Maximum impacted files to return in impact mode (default: 10)"
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
    
    map_changed_mode = args.changed or args.changed_neighbors > 0 or (bool(args.base_ref) and not args.impact_changed)
    impact_mode = bool(args.impact_from) or args.impact_changed

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
    trace_from = str(resolve_repo_path(root_path, args.trace_from).resolve()) if args.trace_from else None
    trace_to = str(resolve_repo_path(root_path, args.trace_to).resolve()) if args.trace_to else None
    impact_from = [str(resolve_repo_path(root_path, f).resolve()) for f in (args.impact_from or [])]

    if bool(trace_from) != bool(trace_to):
        tool_error("Both --trace-from and --trace-to must be provided together.")
        sys.exit(1)
    if args.impact_changed and impact_from:
        tool_error("Use either --impact-from or --impact-changed, not both.")
        sys.exit(1)
    if trace_from and impact_mode:
        tool_error("Trace mode and impact mode cannot be used together.")
        sys.exit(1)
    if impact_mode and (args.changed or args.changed_neighbors > 0):
        tool_error("Map-focused changed mode cannot be combined with impact mode. Use --impact-changed and optional --base-ref instead.")
        sys.exit(1)

    git_result = None
    changed_files = []
    impact_seed_files = impact_from[:]

    if map_changed_mode or args.impact_changed:
        git_result = get_changed_files(str(root_path), args.base_ref)
        if git_result.error:
            tool_error(git_result.error)
            sys.exit(1)

        changed_set = set(git_result.files)
        if args.verbose:
            for diagnostic in git_result.diagnostics:
                info_handler(diagnostic)
        if map_changed_mode:
            changed_files = [path for path in other_files if path in changed_set] if explicit_other_specs else git_result.files
            if args.changed_neighbors > 0:
                if not changed_files:
                    other_files = []
            else:
                other_files = changed_files

            if args.verbose:
                info_handler(f"Changed files selected: {len(changed_files)}")
                if args.changed_neighbors > 0 and changed_files:
                    info_handler(
                        f"Including repository neighbors up to distance {args.changed_neighbors} around changed files."
                    )

        if args.impact_changed:
            impact_seed_files = [path for path in other_files if path in changed_set] if explicit_other_specs else git_result.files
            impact_seed_files = list(dict.fromkeys(impact_seed_files))
            if args.verbose:
                info_handler(f"Impact seed files selected from git: {len(impact_seed_files)}")

    inferred_parser_languages = infer_parser_languages(
        chat_files + other_files + impact_seed_files + [path for path in (trace_from, trace_to) if path]
    )

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
        if trace_from and trace_to:
            connection_report = repo_map.trace_file_path(
                trace_from,
                trace_to,
                files=other_files,
                max_hops=args.trace_max_hops,
            )

            if args.output_format == "json":
                print(json.dumps(dataclasses.asdict(connection_report), indent=2))
            else:
                print(format_connection_report(connection_report))

            if connection_report.error:
                sys.exit(1)
            return

        if impact_mode:
            if not impact_seed_files:
                impact_report = ImpactReport(
                    seed_files=[],
                    max_depth=args.impact_max_depth,
                    max_results=args.impact_max_results,
                    diagnostics=(git_result.diagnostics[:] if git_result else []) + ["No changed files found for impact analysis."],
                )
            else:
                impact_report = repo_map.analyze_file_impact(
                    impact_seed_files,
                    files=other_files,
                    max_depth=args.impact_max_depth,
                    max_results=args.impact_max_results,
                    changed_lines_by_file=(git_result.changed_lines if git_result and args.impact_changed else None),
                )
                if git_result and args.impact_changed:
                    impact_report.diagnostics = list(dict.fromkeys(git_result.diagnostics + impact_report.diagnostics))

            if args.output_format == "json":
                print(json.dumps(dataclasses.asdict(impact_report), indent=2))
            else:
                print(format_impact_report(impact_report))

            if impact_report.error:
                sys.exit(1)
            return

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
