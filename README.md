# pi-local-rag

> **Fork** — This is a fork of the original [pi-local-rag](https://github.com/vahidkowsari/pi-local-rag) by @vahidkowsari.

Local hybrid RAG pipeline for the [Pi coding agent](https://github.com/badlogic/pi-mono). Index your codebase and get fast semantic + keyword search — **zero cloud dependency, works fully offline**.

Backed by **SQLite FTS5** (BM25) and **sqlite-vec** (vector search) for fast retrieval over large codebases.

## Features

- **Hybrid BM25 + vector search** — SQLite FTS5 keyword scoring blended with ONNX embeddings via sqlite-vec
- **Fast** — O(log n) index lookups, not O(n) in-memory scans
- **Smart chunking** — splits files into ~50-line blocks at natural blank-line boundaries
- **PDF + DOCX + HTML support** — extracts text from binary documents via `pdf-parse` and `mammoth`; HTML files are converted to Markdown via `turndown` (strips scripts, styles, nav/footer noise)
- **Incremental indexing** — skips unchanged files (SHA-256 hash check)
- **Auto-refresh** — stale index (24h+) is silently refreshed before auto-injection
- **Exclude patterns** — gitignore-style patterns to skip files or directories
- **Zero cloud dependency** — local Transformers.js model + SQLite, fully offline
- **3 AI tools** — `rag_index`, `rag_query`, `rag_status` for the agent to use directly

## Install

```bash
pi install git:github.com/theli-ua/pi-local-rag
```

## Commands

| Command | Description |
|---|---|
| `/rag index <path>` | Index a file or directory |
| `/rag search <query>` | Search indexed content |
| `/rag find <glob>` | Find indexed files by glob pattern |
| `/rag status` | Show index stats (files, chunks, tokens) |
| `/rag rebuild` | Re-index changed files, prune deleted |
| `/rag refresh` | Refresh tracked paths (re-embed changed files) |
| `/rag clear` | Wipe the entire index |
| `/rag exclude <pattern>` | Add exclude pattern (prefix `-` to remove) |
| `/rag on` | Enable auto-injection |
| `/rag off` | Disable auto-injection |
| `/rag auto-refresh` | Toggle periodic reindexing |

## AI Tools

The extension registers three tools the agent can call directly:

- **`rag_index`** — Index a path into the pipeline
- **`rag_query`** — Hybrid BM25+vector search, returns file paths + line numbers + previews
- **`rag_status`** — Show index stats and RAG config

## Supported File Types

**Text files** (< 500 KB): `.md`, `.txt`, `.ts`, `.js`, `.py`, `.rs`, `.go`, `.java`, `.c`, `.cpp`, `.h`, `.cs`, `.css`, `.html`, `.json`, `.yaml`, `.yml`, `.toml`, `.xml`, `.csv`, `.sh`, `.sql`, `.graphql`, `.proto`, `.env`, `.gitignore`, `.dockerfile`

**Binary documents** (< 10 MB): `.pdf`, `.docx`

> **Note:** `.html` files are converted to Markdown in memory via [`turndown`](https://github.com/mixmark-io/turndown) before indexing — scripts, styles, nav, and footer elements are stripped, producing clean semantic content for both BM25 and vector search.

Directories named `node_modules`, `.git`, `.next`, `dist`, `build`, `__pycache__`, `.venv`, `venv`, `.cache` and any hidden directory (starting with `.`) are always skipped.

## How It Works

1. **Index** — files are chunked (~50 lines each), embedded with `Xenova/all-MiniLM-L6-v2` (384-dim), and stored in a local SQLite database with FTS5 (BM25) and sqlite-vec (vector) indexes
2. **Search** — hybrid scoring: `alpha × BM25 + (1-alpha) × cosine_similarity` (default `alpha=0.4`). BM25 via FTS5's native `bm25()` function, vectors via sqlite-vec nearest-neighbor
3. **Auto-inject** — before every agent turn, the prompt is searched and relevant chunks are prepended to the system prompt. Periodic reindexing (when `ragAutoRefresh` is enabled) refreshes changed files if the index is stale (>24h).

## Storage

The plugin keeps each project's index and config in its own `.pi/rag/` directory. The active store is resolved per-cwd:

1. `$PI_RAG_DIR` — explicit override, if set.
2. **Walk-up** — climb upward from the working directory looking for an existing `.pi/rag/`. Walk-up stops *before* `$HOME`, so the global store never wins via walk-up.
3. **Auto-create** — the first `/rag index` (or `rag_index` tool call) in a directory with no walk-up hit creates `./.pi/rag/` at the cwd. Other commands never auto-create.
4. **Global fallback** — `~/.pi/rag/` is used when no project store is found.

The index is stored as a **SQLite database** (`rag.db`) with:
- `chunks` table — text content, line numbers, metadata
- `chunks_fts` — FTS5 virtual table for BM25 keyword search
- `chunks_vec` — sqlite-vec virtual table for 384-dim vector similarity
- `files` table — file-level hash tracking for incremental re-indexing
- `metadata` table — key/value pairs (`last_build`, `embedding_model`)

Config is stored as JSON in `config.json`. `/rag status` shows the active store's path and labels it `(project)` or `(global)`. Legacy `index.json` files are auto-migrated to SQLite on first run. If you previously used `~/.pi/lens/`, that directory is migrated to `~/.pi/rag/` on first run.

## Exclude Patterns

Use `/rag exclude` to add gitignore-style patterns. Matched files are skipped during indexing and pruned on rebuild:

```
/rag exclude node_modules      # add pattern
/rag exclude -node_modules     # remove pattern
/rag exclude                   # list current patterns
```

Patterns support globs: `**/test/**/*.ts`, `*.log`, `fixtures/`, etc.

## Configuration

Auto-injection is on by default. Tune via `/rag status`:

| Setting | Default | Description |
|---|---|---|
| `ragEnabled` | `false` | Auto-inject context before each turn |
| `ragTopK` | `5` | Max chunks to inject |
| `ragScoreThreshold` | `0.1` | Min hybrid score to include |
| `ragAlpha` | `0.4` | BM25/vector blend (0=pure vector, 1=pure BM25) |
| `ragAutoRefresh` | `false` | Periodic reindexing when index is stale (>24h) |
