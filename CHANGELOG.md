# Changelog

All notable changes to this project will be documented in this file.

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

