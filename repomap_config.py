"""
Repository-local configuration loader for RepoMap.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, List, Optional, Tuple

try:
    import tomllib
except ImportError:
    tomllib = None


def _normalize_pattern(value: str) -> str:
    """Normalize config path patterns to forward-slash repository-relative form."""
    pattern = str(value or "").strip().replace("\\", "/")
    if pattern.startswith("./"):
        pattern = pattern[2:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    return pattern


def _normalize_token(value: str) -> str:
    """Normalize case-insensitive file or directory names from config."""
    return str(value or "").strip().lower()


def _string_list(data: object, field_name: str, diagnostics: List[str]) -> Tuple[str, ...]:
    """Parse a list of strings while keeping config loading forgiving."""
    if data is None:
        return ()
    if not isinstance(data, list):
        diagnostics.append(f"Ignoring `{field_name}` in .repomapper.toml: expected a list of strings.")
        return ()

    values = []
    for item in data:
        if not isinstance(item, str):
            diagnostics.append(f"Ignoring non-string item in `{field_name}` in .repomapper.toml.")
            continue
        normalized = _normalize_pattern(item)
        if normalized:
            values.append(normalized)
    return tuple(dict.fromkeys(values))


def _token_list(data: object, field_name: str, diagnostics: List[str]) -> Tuple[str, ...]:
    """Parse a list of lowercase filename/directory tokens."""
    if data is None:
        return ()
    if not isinstance(data, list):
        diagnostics.append(f"Ignoring `{field_name}` in .repomapper.toml: expected a list of strings.")
        return ()

    values = []
    for item in data:
        if not isinstance(item, str):
            diagnostics.append(f"Ignoring non-string item in `{field_name}` in .repomapper.toml.")
            continue
        normalized = _normalize_token(item)
        if normalized:
            values.append(normalized)
    return tuple(dict.fromkeys(values))


def _float_value(data: object, field_name: str, diagnostics: List[str], default: float = 1.0) -> float:
    """Parse a non-negative ranking weight."""
    if data is None:
        return default
    if isinstance(data, (int, float)):
        return max(0.0, float(data))
    diagnostics.append(f"Ignoring `{field_name}` in .repomapper.toml: expected a non-negative number.")
    return default


def _matches_path_pattern(rel_path: str, pattern: str) -> bool:
    """Match repository-relative paths against a forgiving glob syntax."""
    rel_value = str(PurePosixPath(str(rel_path).replace("\\", "/")))
    normalized_pattern = _normalize_pattern(pattern)
    if not normalized_pattern:
        return False

    if normalized_pattern.endswith("/"):
        normalized_pattern = normalized_pattern.rstrip("/")
        return rel_value == normalized_pattern or rel_value.startswith(normalized_pattern + "/")

    if not any(char in normalized_pattern for char in "*?[]"):
        basename = PurePosixPath(rel_value).name
        return (
            rel_value == normalized_pattern
            or rel_value.startswith(normalized_pattern + "/")
            or basename == normalized_pattern
        )

    rel_path_obj = PurePosixPath(rel_value)
    if rel_path_obj.match(normalized_pattern):
        return True

    if "/" not in normalized_pattern:
        return fnmatch.fnmatch(rel_path_obj.name, normalized_pattern)

    return False


@dataclass(frozen=True)
class RepoMapRankingWeights:
    mentioned_identifier: float = 1.0
    mentioned_file: float = 1.0
    chat_file: float = 1.0
    query: float = 1.0
    entrypoint: float = 1.0
    public_api: float = 1.0
    changed_file: float = 1.0
    changed_neighbor: float = 1.0
    related_test: float = 1.0


@dataclass(frozen=True)
class RepoMapFrameworkSignals:
    entrypoint_files: Tuple[str, ...] = ()
    entrypoint_dirs: Tuple[str, ...] = ()
    public_api_files: Tuple[str, ...] = ()
    public_api_dirs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RepoMapTestConfig:
    dirs: Tuple[str, ...] = ()
    integration_markers: Tuple[str, ...] = ()
    python_runner: Optional[str] = None
    js_runner: Optional[str] = None


@dataclass(frozen=True)
class RepoMapConfig:
    path: Optional[str] = None
    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ()
    important_files: Tuple[str, ...] = ()
    frameworks: RepoMapFrameworkSignals = field(default_factory=RepoMapFrameworkSignals)
    tests: RepoMapTestConfig = field(default_factory=RepoMapTestConfig)
    ranking_weights: RepoMapRankingWeights = field(default_factory=RepoMapRankingWeights)
    diagnostics: Tuple[str, ...] = ()

    def _matches_any(self, rel_path: str, patterns: Tuple[str, ...]) -> bool:
        return any(_matches_path_pattern(rel_path, pattern) for pattern in patterns)

    def scope_reason(self, rel_path: str) -> Optional[str]:
        """Return a human-readable exclusion reason when config filters out a file."""
        if self.include and not self._matches_any(rel_path, self.include):
            return "Outside configured include patterns"
        if self.exclude and self._matches_any(rel_path, self.exclude):
            return "Matched configured exclude pattern"
        return None

    def is_configured_important_file(self, rel_path: str) -> bool:
        """Return True when the config explicitly marks a file as important."""
        return self._matches_any(rel_path, self.important_files)


def load_repo_map_config(
    root: str | Path,
    file_reader_func: Callable[[str], Optional[str]],
) -> RepoMapConfig:
    """Load .repomapper.toml from the repository root when present."""
    config_path = Path(root).resolve() / ".repomapper.toml"
    if not config_path.is_file():
        return RepoMapConfig()

    diagnostics: List[str] = []
    if tomllib is None:
        diagnostics.append("Ignoring .repomapper.toml because tomllib is unavailable in this Python runtime.")
        return RepoMapConfig(path=str(config_path), diagnostics=tuple(diagnostics))

    raw_text = file_reader_func(str(config_path))
    if not raw_text:
        diagnostics.append("Ignoring .repomapper.toml because it could not be read.")
        return RepoMapConfig(path=str(config_path), diagnostics=tuple(diagnostics))

    try:
        data = tomllib.loads(raw_text)
    except (tomllib.TOMLDecodeError, TypeError) as exc:
        diagnostics.append(f"Ignoring .repomapper.toml because it could not be parsed: {exc}.")
        return RepoMapConfig(path=str(config_path), diagnostics=tuple(diagnostics))

    if not isinstance(data, dict):
        diagnostics.append("Ignoring .repomapper.toml because the top-level document is not a TOML table.")
        return RepoMapConfig(path=str(config_path), diagnostics=tuple(diagnostics))

    include = _string_list(data.get("include"), "include", diagnostics)
    exclude = _string_list(data.get("exclude"), "exclude", diagnostics)
    important_files = _string_list(data.get("important_files"), "important_files", diagnostics)

    frameworks_table = data.get("frameworks") or {}
    if not isinstance(frameworks_table, dict):
        diagnostics.append("Ignoring `[frameworks]` in .repomapper.toml: expected a table.")
        frameworks_table = {}

    tests_table = data.get("tests") or {}
    if not isinstance(tests_table, dict):
        diagnostics.append("Ignoring `[tests]` in .repomapper.toml: expected a table.")
        tests_table = {}

    weights_table = data.get("ranking_weights") or {}
    if not isinstance(weights_table, dict):
        diagnostics.append("Ignoring `[ranking_weights]` in .repomapper.toml: expected a table.")
        weights_table = {}

    python_runner = tests_table.get("python_runner")
    if python_runner is not None and not isinstance(python_runner, str):
        diagnostics.append("Ignoring `tests.python_runner` in .repomapper.toml: expected a string.")
        python_runner = None
    python_runner = _normalize_token(python_runner) if python_runner else None

    js_runner = tests_table.get("js_runner")
    if js_runner is not None and not isinstance(js_runner, str):
        diagnostics.append("Ignoring `tests.js_runner` in .repomapper.toml: expected a string.")
        js_runner = None
    js_runner = _normalize_token(js_runner) if js_runner else None

    return RepoMapConfig(
        path=str(config_path),
        include=include,
        exclude=exclude,
        important_files=important_files,
        frameworks=RepoMapFrameworkSignals(
            entrypoint_files=_token_list(frameworks_table.get("entrypoint_files"), "frameworks.entrypoint_files", diagnostics),
            entrypoint_dirs=_token_list(frameworks_table.get("entrypoint_dirs"), "frameworks.entrypoint_dirs", diagnostics),
            public_api_files=_token_list(frameworks_table.get("public_api_files"), "frameworks.public_api_files", diagnostics),
            public_api_dirs=_token_list(frameworks_table.get("public_api_dirs"), "frameworks.public_api_dirs", diagnostics),
        ),
        tests=RepoMapTestConfig(
            dirs=_token_list(tests_table.get("dirs"), "tests.dirs", diagnostics),
            integration_markers=_token_list(
                tests_table.get("integration_markers"),
                "tests.integration_markers",
                diagnostics,
            ),
            python_runner=python_runner,
            js_runner=js_runner,
        ),
        ranking_weights=RepoMapRankingWeights(
            mentioned_identifier=_float_value(
                weights_table.get("mentioned_identifier"),
                "ranking_weights.mentioned_identifier",
                diagnostics,
            ),
            mentioned_file=_float_value(
                weights_table.get("mentioned_file"),
                "ranking_weights.mentioned_file",
                diagnostics,
            ),
            chat_file=_float_value(
                weights_table.get("chat_file"),
                "ranking_weights.chat_file",
                diagnostics,
            ),
            query=_float_value(
                weights_table.get("query"),
                "ranking_weights.query",
                diagnostics,
            ),
            entrypoint=_float_value(
                weights_table.get("entrypoint"),
                "ranking_weights.entrypoint",
                diagnostics,
            ),
            public_api=_float_value(
                weights_table.get("public_api"),
                "ranking_weights.public_api",
                diagnostics,
            ),
            changed_file=_float_value(
                weights_table.get("changed_file"),
                "ranking_weights.changed_file",
                diagnostics,
            ),
            changed_neighbor=_float_value(
                weights_table.get("changed_neighbor"),
                "ranking_weights.changed_neighbor",
                diagnostics,
            ),
            related_test=_float_value(
                weights_table.get("related_test"),
                "ranking_weights.related_test",
                diagnostics,
            ),
        ),
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )
