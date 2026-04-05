# RepoMapper

Code intelligence for AI agents. Generates a ranked map of your codebase — functions, classes, interfaces and their relationships — compressed to fit an LLM's context window.

Based on [pdavis68/RepoMapper](https://github.com/pdavis68/RepoMapper) (which is based on [Aider's](https://aider.chat/) repo-map). This fork adds Vue/Nuxt SFC support and performance optimizations.

## What it does

1. **Parses** every source file with Tree-sitter to extract definitions and references
2. **Builds a graph** where files are nodes and symbol references are edges
3. **Ranks** files using PageRank — heavily referenced modules surface to the top
4. **Keeps** important docs and config files visible even when they don't contain parser-extracted symbols
5. **Biases** ranking toward task-relevant files when you provide a free-form query
6. **Surfaces** nearby related tests for highly relevant source files
7. **Elevates** entrypoints and public API surfaces so agents can find where execution starts
8. **Expands** git-changed views with nearby impact neighbors when needed
9. **Extracts** lightweight summaries from key docs and config files
10. **Traces** shortest file-level connection paths across the repository graph
11. **Analyzes** likely impact radius around one or more seed files, including nearby tests and boundary files
12. **Projects** that impact analysis into concrete edit candidates and a lightweight "what to edit next" plan
13. **Compresses** the output to fit within a token budget (default 8192 tokens)

The result: an AI agent gets structural understanding of a 1000+ file codebase in ~4k tokens, instead of reading dozens of files (~50k+ tokens).

## Example

```
$ python repomap.py /path/to/project --map-tokens 2048

server/services/ProductService.ts:
(Rank value: 1.0000)

  │export class ProductService {
  █  private readonly productRepo: ProductRepository
  │  async getProductTree(): Promise<ProductsResponse> {
  █    const products = await this.productRepo.getActiveProducts()
  ⋮
  │  private buildTree(products: ProductRow[]): ProductTreeNode[] {

app/components/FilterPanel.vue:
(Rank value: 0.8542)

  │<script setup lang="ts">
  │interface Props {
  █  mode: 'grid' | 'list'
  ⋮
  │function handleFilterChange(value: string | number): void {
  █  const parsed = parseFilter(value)
```

## Fork changes

### Vue/Nuxt SFC support

Standard RepoMapper can't parse `.vue` files because Vue's tree-sitter grammar treats `<script>` content as opaque `raw_text`. This fork:

- Extracts `<script>` blocks from Vue SFCs using regex
- Detects `lang="ts"` / `lang="tsx"` and routes to the correct parser
- Handles Vue 3.3+ `generic` attributes (angle brackets in tag attributes)
- Supports multiple script blocks (`<script>` + `<script setup>`)
- Offsets line numbers correctly so repo map references point to the right lines in the `.vue` file

### Performance optimizations

- Global query cache shared across MCP requests (R2)
- Per-request in-memory tags cache to avoid SQLite round-trips (R2)
- Batch edge addition for graph building (R3)
- Pre-computed relative paths to avoid redundant Path operations (R4)
- Pre-sorted file grouping hoisted out of binary search loop (R3)

### Better repository context

- Important files like `README.md`, `pyproject.toml`, workflow YAMLs and docs can still appear in the map even when they have no tree-sitter symbol tags
- Standalone `.tsx` files use the TSX parser/runtime instead of falling back to plain TypeScript parsing
- Parser bootstrap failures are surfaced as user-facing diagnostics instead of silently collapsing into an empty map
- CLI and MCP responses now include structured per-file ranking metadata, including reason codes such as `chat_file`, `mentioned_identifier`, `important_file`, and `referenced_by`
- Free-form queries can bias ranking toward matching file paths and symbol names, with explicit `query_path_match` and `query_symbol_match` reasons in the report
- Nearby test files can be lifted into the map even when they lack parser definitions, with `related_tests` / `related_source` reasons in the report
- Entrypoints and public API files get automatic heuristic boosts and synthetic highlights, with `entrypoint_file` / `public_api_file` reasons in the report
- Changed-file mode can optionally include graph-near neighbors, with `changed_file` / `changed_neighbor` reasons and explicit changed-file metadata in the report
- Important docs and config files can expose structured highlights such as README headings, package scripts, workflow jobs and Docker entrypoints
- File-to-file tracing can explain how two parts of the repo connect through references and source/test relationships
- File tracing and impact analysis now understand TS/JS import + re-export chains and Python package boundaries, not just raw same-name references
- Repo-local `.repomapper.toml` files can now scope the repository view, mark extra important files, extend framework/test signals, and tune ranking weights without changing the CLI or MCP payload shape
- Impact analysis can explain which nearby files are most likely affected by a change, with shortest paths, reasons, and related tests
- Impact analysis now also emits a prioritized `suggested_checks` checklist so agents can decide what to inspect first
- Impact analysis also emits a smaller `quick_actions` lane for low-risk next steps such as opening the closest changed boundary, running a nearby test, or checking a config assumption
- `quick_actions` now include explicit `location_hint` anchors and, when the repo config clearly signals a runner, concrete `command_hint` test commands
- `quick_actions` now also carry `risk_level` and `why_now`, so agents can prioritize the safest high-signal step first
- `quick_actions` now also include `expected_outcome`, describing what signal the agent should gain from taking that step
- `quick_actions` now also include `follow_if_true` / `follow_if_false` hints so an agent can branch immediately after the first finding
- `quick_actions` now also carry a lightweight `confidence` score based on how direct the impact evidence is
- `quick_actions` now also include `focus_symbols`, highlighting the 1-3 symbols most worth checking before opening the file
- `quick_actions` now also include `focus_reason`, explaining why those symbols were chosen
- `quick_actions` now also include `target_role`, so agents can distinguish test/config/public API/entrypoint/neighbor boundaries at a glance
- Impact analysis now also emits concrete `edit_candidates` with file/symbol anchors for the most likely next edits
- Impact analysis now also emits a lightweight `edit_plan` lane and a matching `--edit-plan` CLI mode for compact "what to edit next" workflows
- Impact analysis now also groups impacted tests into `test_clusters` such as `sibling`, `nearby`, and `integration`, with optional grouped test commands
- Impact analysis now also surfaces shared boundary symbols, changed seed symbols, diff hunks, concrete file/line locations, and short boundary snippets so agents can jump straight to the likely change boundary
- `suggested_checks` can now point directly at a boundary line/snippet instead of only naming a file
- Review mode now combines branch context, changed-file diff anchors, public API / entrypoint / config surfaces, nearby tests, and impact-based `review_focus` priorities into a single "what to check first" view
- Next-style `app/api/*/route.ts` handlers are now treated as public API surfaces instead of being misclassified as tests when an endpoint segment happens to be named `test`

## Supported languages

All Tree-sitter languages from the upstream project, plus:

- **Vue/Nuxt SFCs** (`.vue`) — JavaScript, TypeScript, and TSX script blocks

Full list: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, C#, Ruby, PHP, Kotlin, Scala, Swift, Dart, Elixir, Elm, Gleam, Lua, OCaml, R, Racket, HCL, Solidity, and more.

## Installation

```bash
# Clone and set up
git clone git@github.com:iHeldan/repomapper.git
cd repomapper
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs two console scripts:

- `repomap` for the CLI
- `repomap-mcp` for the MCP server
- `repomap-eval` for fixture-backed golden evals

Note: `tree-sitter-language-pack` may download parser binaries on first use. If you need offline execution, pre-warm the languages you care about while network access is available.

## Usage

### CLI

```bash
# Map a project directory
repomap /path/to/project

# Or point --root at a repo and map everything under it
repomap --root /path/to/project

# With custom token budget
repomap /path/to/project --map-tokens 4096

# Prioritize specific files (e.g., files you're editing)
repomap /path/to/project --chat-files src/main.ts

# Exclude low-rank files
repomap /path/to/project --exclude-unranked

# Focus only on local git changes
repomap --root /path/to/project --changed

# Compare the current branch against a git base ref
repomap --root /path/to/project --changed --base-ref origin/main

# Include immediate graph neighbors around changed files
repomap --root /path/to/project --changed-neighbors 1

# Bias ranking toward a task like auth debugging or payment flow tracing
repomap --root /path/to/project --query "auth login flow"

# Trace a file-level path between two parts of the repo
repomap --root /path/to/project --trace-from app.py --trace-to api/routes.py

# Analyze likely impact neighbors around one or more seed files
repomap --root /path/to/project --impact-from app.py service.py

# Analyze likely impact radius around your current git changes
repomap --root /path/to/project --impact-changed --base-ref origin/main

# Review the current branch like a PR, with branch + diff + impact context
repomap --root /path/to/project --review --base-ref origin/main

# Render a compact "what to edit next" plan for an impact analysis
repomap --root /path/to/project --impact-from app.py --edit-plan

# Download missing parser runtimes before mapping
repomap --root /path/to/project --download-missing-parsers

# Pre-warm parser runtimes for the selected files and exit
repomap --root /path/to/project --warm-languages auto

# Emit structured JSON for agent workflows
repomap --root /path/to/project --output-format json
```

When using `--output-format json`, the CLI returns both the rendered text map and a structured `report` with:

- `ranked_files`: ranked file entries with scores, sample symbols, neighbor files, lines of interest, and machine-readable reasons
- `query` / `query_terms`: the task prompt used for query-aware ranking plus the extracted ranking terms
- `changed_files` / `changed_neighbor_depth`: the changed-file focus set and the neighbor expansion depth used for impact views
- `related_tests` / `related_sources`: heuristic source-to-test relationships to help agents find validating coverage quickly
- `entrypoint_signals` / `public_api_signals`: heuristics explaining why a file looks like an app entrypoint or public API surface
- `summary_kind` / `summary_items`: lightweight structured highlights extracted from important docs and config files
- `selected_files`: which ranked files actually fit into the token budget and were rendered into the text map
- `map_tokens`: estimated token cost of the rendered map

When using `--trace-from` and `--trace-to`, the CLI switches to path-tracing mode and returns either:

- a readable hop-by-hop explanation in text mode
- or a structured `path` + `steps` + `symbol_path` payload in JSON mode, where each step can also include symbol-level `symbol_hops` evidence for callsites, imports, re-exports, and package boundaries

When using `--impact-from` or `--impact-changed`, the CLI switches to impact-analysis mode and returns either:

- a readable list of nearby impacted files with shortest paths and relations in text mode
- or a structured `seed_files` + `impacted_files` + `shared_symbols` + `quick_actions` + `edit_candidates` + `edit_plan` + `test_clusters` + `suggested_checks` payload in JSON mode, including `changed_seed_symbols`, `changed_hunks_by_file`, `seed_hunks`, `seed_focus_lines`, `changed_boundary_symbols`, `changed_boundary_distances`, `boundary_locations`, `boundary_snippets`, per-target `symbol_path`, target `focus_lines`, and per-action/per-suggestion anchor fields plus `location_hint`/`command_hint`, `risk_level`/`why_now`, `expected_outcome`, `follow_if_true` / `follow_if_false`, `confidence`, `focus_symbols`, `focus_reason`, and `target_role` where available

If you pass `--edit-plan`, text output switches to a compact edit-oriented view that prioritizes the first few high-signal next steps, along with their best concrete edit candidates.

When using `--review`, the CLI switches to review mode and returns either:

- a readable PR-style summary with changed surfaces, diff anchors, grouped test clusters, closest impact targets, and a prioritized `review_focus` lane
- or a structured payload containing `current_branch`, `base_ref`, `changed_files`, `changed_public_api_files`, `changed_entrypoint_files`, `changed_config_files`, `changed_test_files`, `review_focus`, plus the underlying impact-driven `quick_actions`, `edit_candidates`, `edit_plan`, `test_clusters`, `impacted_files`, and `shared_symbols`

### Repository config

If the repo root contains a `.repomapper.toml`, RepoMapper loads it automatically and applies it to map, trace, and impact workflows.

Example:

```toml
include = ["src/**", "tests/**", "package.json"]
exclude = ["src/generated/**", "dist/**"]
important_files = ["ops/*.conf", "contracts/**"]

[frameworks]
entrypoint_files = ["boot.py", "worker.ts"]
entrypoint_dirs = ["services"]
public_api_files = ["exports.ts"]
public_api_dirs = ["exports", "contracts"]

[tests]
dirs = ["checks", "integration"]
integration_markers = ["contract", "scenario"]
python_runner = "pytest"
js_runner = "vitest"

[ranking_weights]
query = 1.0
entrypoint = 1.25
public_api = 1.1
changed_file = 1.0
changed_neighbor = 0.8
related_test = 1.2
```

Supported config knobs:

- `include` / `exclude`: repository-relative glob patterns that scope the files considered by ranking, tracing, and impact analysis
- `important_files`: extra docs/config/contracts that should stay visible even without parser-extracted symbols
- `[frameworks]`: additional entrypoint/public API filenames and directory signals
- `[tests]`: extra test directories, extra integration markers, and optional runner overrides for `pytest`, `vitest`, `jest`, or `mocha`
- `[ranking_weights]`: non-negative multipliers that tune how strongly query matches, changed files, related tests, entrypoints, public APIs, chat files, mentioned files, and mentioned identifiers affect ranking

### MCP Server

Add to your Claude Code config (`.claude.json` or similar):

```json
{
  "mcpServers": {
    "repomapper": {
      "type": "stdio",
      "command": "/path/to/repomapper/.venv/bin/python",
      "args": ["/path/to/repomapper/repomap_server.py"]
    }
  }
}
```

The server exposes a `repo_map` tool that any MCP-compatible AI agent can call.

Security note: the bundled MCP server only accepts `project_root` values under `~/AI`, `~/Projects`, or `~/Coding`, and it rejects file paths that resolve outside the selected root, including symlink escapes.

You can also ask the MCP tool to download missing parser runtimes by passing `download_missing_parsers=true`.
For change-focused workflows, pass `changed_only=true` and optionally `base_ref="origin/main"` to restrict the map to git-changed files.
For impact-focused workflows, pass `changed_neighbors=1` (or higher) to include nearby graph neighbors around those changed files.
For task-focused workflows, pass `query="auth login flow"` to bias ranking toward matching paths and symbols.
The `report` payload also includes structured `ranked_files`, `selected_files`, and `map_tokens` fields for agent-friendly follow-up logic.
The server also exposes `trace_file_path` for shortest-path explanations between two files. Its response now includes symbol-level evidence such as callsites, imports, TS/JS re-exports, and Python package-boundary hops.
The server also exposes `analyze_file_impact` for "what else is likely affected?" workflows around one or more seed files, or around git-changed files via `changed_only=true` and optional `base_ref`. Its response now includes changed seed symbols from the diff, grouped changed hunks, shared boundary symbols, concrete file/line boundary locations, symbol-level path evidence, a lightweight `quick_actions` lane for low-risk next moves, concrete `edit_candidates`, a compact `edit_plan`, grouped `test_clusters`, and prioritized `suggested_checks` items such as nearby tests, boundary APIs, entrypoints, and config files worth verifying next. When the repository clearly signals a test runner, quick actions can also include a ready-to-run `command_hint`, plus `risk_level`, `why_now`, `expected_outcome`, `follow_if_true` / `follow_if_false`, `confidence`, `focus_symbols`, `focus_reason`, and `target_role` fields for fast prioritization.
The server also exposes `review_changes` for PR/review-style workflows. It combines git-changed files, branch metadata, changed diff anchors, public API and entrypoint surfaces, grouped nearby tests, and a prioritized `review_focus` queue on top of the existing impact analysis payload.

### Evals

Fixture-backed eval cases live under `evals/fixtures/`, and their expected normalized outputs live under `evals/goldens/`.

Run all evals:

```bash
repomap-eval
```

Update goldens after an intentional behavior change:

```bash
repomap-eval --update
```

## Dependencies

- `tree-sitter` + `grep-ast` — code parsing
- `networkx` — PageRank graph algorithm
- `tiktoken` — token counting
- `diskcache` — persistent tag cache
- `fastmcp` — MCP server framework

## License

MIT (see [LICENSE](LICENSE)). Original work by Pete Davis.
