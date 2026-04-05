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
12. **Compresses** the output to fit within a token budget (default 8192 tokens)

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
- Impact analysis can explain which nearby files are most likely affected by a change, with shortest paths, reasons, and related tests
- Impact analysis now also emits a prioritized `suggested_checks` checklist so agents can decide what to inspect first
- Impact analysis now also surfaces shared boundary symbols so agents can see which APIs, classes, or functions likely carry the change across files

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
- or a structured `path` + `steps` payload in JSON mode

When using `--impact-from` or `--impact-changed`, the CLI switches to impact-analysis mode and returns either:

- a readable list of nearby impacted files with shortest paths and relations in text mode
- or a structured `seed_files` + `impacted_files` + `shared_symbols` + `suggested_checks` payload in JSON mode

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
The server also exposes `trace_file_path` for shortest-path explanations between two files.
The server also exposes `analyze_file_impact` for "what else is likely affected?" workflows around one or more seed files, or around git-changed files via `changed_only=true` and optional `base_ref`. Its response now includes shared boundary symbols plus prioritized `suggested_checks` items such as nearby tests, boundary APIs, entrypoints, and config files worth verifying next.

## Dependencies

- `tree-sitter` + `grep-ast` — code parsing
- `networkx` — PageRank graph algorithm
- `tiktoken` — token counting
- `diskcache` — persistent tag cache
- `fastmcp` — MCP server framework

## License

MIT (see [LICENSE](LICENSE)). Original work by Pete Davis.
