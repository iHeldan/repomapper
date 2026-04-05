"""Git-aware file selection helpers for RepoMap."""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Iterable, List, Optional

from utils import is_within_directory


@dataclass
class GitFileSelectionResult:
    files: List[str] = field(default_factory=list)
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
    diagnostics = []

    if base_ref:
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

    for args in (
        ["diff", "--name-only", "--diff-filter=ACMR", "--relative", "--", "."],
        ["diff", "--name-only", "--cached", "--diff-filter=ACMR", "--relative", "--", "."],
        ["ls-files", "--others", "--exclude-standard", "--", "."],
    ):
        proc = _run_git(root_path, args)
        if proc.returncode != 0:
            error_text = proc.stderr.strip() or "Failed to list changed files from git"
            return GitFileSelectionResult(error=error_text)
        changed_paths.update(proc.stdout.splitlines())

    files = sorted(dict.fromkeys(_normalize_git_paths(root_path, changed_paths)))
    return GitFileSelectionResult(files=files, diagnostics=diagnostics)
