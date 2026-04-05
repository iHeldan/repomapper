# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Changed

- MCP `repo_map` now trims `ranked_files` to the top 20 entries by default and reports `ranked_files_total`, `ranked_files_returned`, `ranked_files_omitted`, and `ranked_files_truncated` summary fields for large repositories
- MCP `repo_map` now also exposes a compact `ranked_files_preview` list and `ranked_files_counts` summary so agents can inspect the top-ranked paths without pulling the full metadata payload
- MCP callers can opt back into the full ranked file list with `ranked_files_limit=0`, or suppress detailed ranked file rows entirely with `include_ranked_files=false`
- MCP `repo_map` now also trims large `excluded` dictionaries by default, adds `excluded_total` / `excluded_preview` / `excluded_reason_counts` summary metadata, and lets callers opt back into full excluded rows with `excluded_limit=0`
- Repo config glob matching now handles recursive subtree patterns like `opensrc/**` correctly instead of only matching the first nested segment
- Repo map ranking now softens entrypoint/public-API boosts inside test directories so factory/mock barrel files remain visible without outranking primary production surfaces

## 0.2.1 - 2026-04-06

### Changed

- Repo map output is now deterministic for equal-rank ties, which keeps repeated runs and eval comparisons stable
- Map selection now uses a lazy oversize-file fallback: it first searches the full ranked prefix, then filters individually oversized files only when needed
- Regression coverage now explicitly guards deterministic ordering and lazy-filtering behavior

## 0.2.0 - 2026-04-06

### Added

- Query-aware ranking that boosts matching paths and symbols for task-focused map generation
- Git-aware changed-file and changed-neighbor workflows for narrowing maps to active work
- Impact analysis with `quick_actions`, `edit_candidates`, `edit_plan`, `test_clusters`, and structured boundary evidence
- Symbol-level trace support for import, re-export, and Python package-boundary chains
- Review mode that combines changed files, impact surfaces, nearby tests, and prioritized `review_focus` guidance
- Repo-local `.repomapper.toml` support for include/exclude rules, framework signals, test conventions, and ranking weights
- Fixture-backed evals with golden outputs for regression checking
- Dynamic token budgeting with fixed, `auto`, and AI-guided budget requests
- MCP/JSON payload expansions for richer agent-oriented follow-up flows

### Changed

- RepoMap report and review payload models now live in a dedicated `repomap_models.py` module to keep the core engine easier to evolve
