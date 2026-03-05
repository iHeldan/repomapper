# RepoMapper

Code intelligence for AI agents. Generates a ranked map of your codebase — functions, classes, interfaces and their relationships — compressed to fit an LLM's context window.

Based on [pdavis68/RepoMapper](https://github.com/pdavis68/RepoMapper) (which is based on [Aider's](https://aider.chat/) repo-map). This fork adds Vue/Nuxt SFC support and performance optimizations.

## What it does

1. **Parses** every source file with Tree-sitter to extract definitions and references
2. **Builds a graph** where files are nodes and symbol references are edges
3. **Ranks** files using PageRank — heavily referenced modules surface to the top
4. **Compresses** the output to fit within a token budget (default 1024 tokens)

The result: an AI agent gets structural understanding of a 1000+ file codebase in ~4k tokens, instead of reading dozens of files (~50k+ tokens).

## Example

```
$ python repomap.py /path/to/project --map-tokens 2048

server/services/CategoryService.ts:
(Rank value: 1.0000)

  │export class CategoryService {
  █  private readonly categoryRepo: CategoryRepository
  │  async getCategoryTree(): Promise<CategoriesResponse> {
  █    const categories = await this.categoryRepo.getActiveCategories()
  ⋮
  │  private buildTree(categories: CategoryRow[]): CategoryTreeNode[] {

app/components/PricingFields.vue:
(Rank value: 0.8542)

  │<script setup lang="ts">
  │interface Props {
  █  saleMethod: 'auction' | 'buy-now'
  ⋮
  │function handleStartingPriceChange(value: string | number): void {
  █  const price = parsePrice(value)
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

## Usage

### CLI

```bash
# Map a project directory
python repomap.py /path/to/project

# With custom token budget
python repomap.py /path/to/project --map-tokens 4096

# Prioritize specific files (e.g., files you're editing)
python repomap.py /path/to/project --chat-files src/main.ts

# Exclude low-rank files
python repomap.py /path/to/project --exclude-unranked
```

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

## Dependencies

- `tree-sitter` + `grep-ast` — code parsing
- `networkx` — PageRank graph algorithm
- `tiktoken` — token counting
- `diskcache` — persistent tag cache
- `fastmcp` — MCP server framework

## License

MIT (see [LICENSE](LICENSE)). Original work by Pete Davis.
