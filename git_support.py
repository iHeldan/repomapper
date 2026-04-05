"""Git-aware file selection helpers for RepoMap."""

from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
from typing import Dict, Iterable, List, Optional

from utils import is_within_directory

INTERNAL_GIT_IGNORE_PREFIXES = (".repomap.",)


@dataclass
class GitFileSelectionResult:
    files: List[str] = field(default_factory=list)
    changed_lines: Dict[str, List[int]] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _run_git(root_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Run a git command rooted at the selected project path."""
    return subprocess.run(
        ["git", "-C", str(root_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _normalize_git_paths(root_path: Path, rel_paths: Iterable[str]) -> List[str]:
    """Normalize git-relative paths to absolute paths inside root_path."""
    normalized = []
    for rel_path in rel_paths:
        rel_path = rel_path.strip()
        if not rel_path:
            continue

        abs_path = (root_path / rel_path).resolve()
        if not is_within_directory(str(abs_path), str(root_path)):
            continue
        if abs_path.exists() and not abs_path.is_dir():
            normalized.append(str(abs_path))

    return normalized


def _is_internal_tool_artifact(root_path: Path, abs_path: str) -> bool:
    """Ignore RepoMap's own cache artifacts in git-aware workflows."""
    try:
        rel_path = Path(abs_path).resolve().relative_to(root_path.resolve())
    except (ValueError, OSError):
        return False
    return any(part.startswith(INTERNAL_GIT_IGNORE_PREFIXES) for part in rel_path.parts)


def _parse_diff_changed_lines(diff_text: str) -> Dict[str, set[int]]:
    """Parse unified diff hunks into changed line numbers for the new file."""
    changed_lines: Dict[str, set[int]] = {}
    current_path: Optional[str] = None

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            new_path = line[4:].strip()
            if new_path == "/dev/null":
                current_path = None
                continue
            if new_path.startswith("b/"):
                new_path = new_path[2:]
            current_path = new_path
            changed_lines.setdefault(current_path, set())
            continue

        if not current_path or not line.startswith("@@"):
            continue

        match = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if not match:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count <= 0:
            continue
        changed_lines[current_path].update(range(start, start + count))

    return changed_lines


def _merge_changed_lines(
    root_path: Path,
    target: Dict[str, set[int]],
    rel_line_map: Dict[str, set[int]],
) -> None:
    """Merge git-relative changed lines into an absolute-path keyed mapping."""
    for rel_path, line_numbers in rel_line_map.items():
        abs_path = (root_path / rel_path).resolve()
        if not is_within_directory(str(abs_path), str(root_path)):
            continue
        if not abs_path.exists() or abs_path.is_dir():
            continue
        target.setdefault(str(abs_path), set()).update(line_numbers)


def _all_file_lines(path: Path) -> List[int]:
    """Return line numbers for an existing text file, used for untracked files."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return list(range(1, len(text.splitlines()) + 1))


def get_changed_files(root: str, base_ref: str | None = None) -> GitFileSelectionResult:
    """Collect changed files from the git repo under root.

    Includes staged, unstaged, and untracked files. When base_ref is provided,
    also includes committed changes since the merge-base with that ref.
    """
    root_path = Path(root).resolve()

    repo_check = _run_git(root_path, ["rev-parse", "--is-inside-work-tree"])
    if repo_check.returncode != 0:
        return GitFileSelectionResult(error=f"Git-aware mode requires a git repository under {root_path}")

    changed_paths = set()
    changed_lines: Dict[str, set[int]] = {}
    diagnostics = []

    if base_ref:
        diff_patch_proc = _run_git(
            root_path,
            ["diff", "--unified=0", "--diff-filter=ACMR", "--relative", f"{base_ref}...HEAD", "--", "."],
        )
        if diff_patch_proc.returncode != 0:
            error_text = diff_patch_proc.stderr.strip() or f"Failed to diff against base ref '{base_ref}'"
            return GitFileSelectionResult(error=error_text)
        _merge_changed_lines(root_path, changed_lines, _parse_diff_changed_lines(diff_patch_proc.stdout))

        diff_proc = _run_git(
            root_path,
            ["diff", "--name-only", "--diff-filter=ACMR", "--relative", f"{base_ref}...HEAD", "--", "."],
        )
        if diff_proc.returncode != 0:
            error_text = diff_proc.stderr.strip() or f"Failed to diff against base ref '{base_ref}'"
            return GitFileSelectionResult(error=error_text)
        changed_paths.update(diff_proc.stdout.splitlines())
        diagnostics.append(f"Included committed changes since {base_ref}.")
    else:
        diagnostics.append("Included staged, unstaged, and untracked changes from the working tree.")

    for patch_args, name_args in (
        (
            ["diff", "--unified=0", "--diff-filter=ACMR", "--relative", "--", "."],
            ["diff", "--name-only", "--diff-filter=ACMR", "--relative", "--", "."],
        ),
        (
            ["diff", "--cached", "--unified=0", "--diff-filter=ACMR", "--relative", "--", "."],
            ["diff", "--name-only", "--cached", "--diff-filter=ACMR", "--relative", "--", "."],
        ),
    ):
        patch_proc = _run_git(root_path, patch_args)
        if patch_proc.returncode != 0:
            error_text = patch_proc.stderr.strip() or "Failed to diff changed files from git"
            return GitFileSelectionResult(error=error_text)
        _merge_changed_lines(root_path, changed_lines, _parse_diff_changed_lines(patch_proc.stdout))

        proc = _run_git(root_path, name_args)
        if proc.returncode != 0:
            error_text = proc.stderr.strip() or "Failed to list changed files from git"
            return GitFileSelectionResult(error=error_text)
        changed_paths.update(proc.stdout.splitlines())

    untracked_proc = _run_git(root_path, ["ls-files", "--others", "--exclude-standard", "--", "."])
    if untracked_proc.returncode != 0:
        error_text = untracked_proc.stderr.strip() or "Failed to list changed files from git"
        return GitFileSelectionResult(error=error_text)
    changed_paths.update(untracked_proc.stdout.splitlines())

    files = sorted(
        path
        for path in dict.fromkeys(_normalize_git_paths(root_path, changed_paths))
        if not _is_internal_tool_artifact(root_path, path)
    )
    for abs_path in files:
        if abs_path in changed_lines:
            continue
        if abs_path in _normalize_git_paths(root_path, untracked_proc.stdout.splitlines()):
            line_numbers = _all_file_lines(Path(abs_path))
            if line_numbers:
                changed_lines[abs_path] = set(line_numbers)

    return GitFileSelectionResult(
        files=files,
        changed_lines={
            path: sorted(line_numbers)
            for path, line_numbers in sorted(changed_lines.items())
            if not _is_internal_tool_artifact(root_path, path)
        },
        diagnostics=diagnostics,
    )


def get_current_branch(root: str) -> Optional[str]:
    """Return the current branch name, or a short HEAD sha when detached."""
    root_path = Path(root).resolve()

    repo_check = _run_git(root_path, ["rev-parse", "--is-inside-work-tree"])
    if repo_check.returncode != 0:
        return None

    branch_proc = _run_git(root_path, ["branch", "--show-current"])
    if branch_proc.returncode == 0:
        branch_name = branch_proc.stdout.strip()
        if branch_name:
            return branch_name

    head_proc = _run_git(root_path, ["rev-parse", "--short", "HEAD"])
    if head_proc.returncode == 0:
        short_head = head_proc.stdout.strip()
        if short_head:
            return f"detached@{short_head}"

    return None
