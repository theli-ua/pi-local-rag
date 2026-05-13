# pi-local-rag

Local BM25 RAG pipeline for the [Pi coding agent](https://github.com/badlogic/pi-mono). Index your local files and search them with keyword matching — **zero cloud dependency, works fully offline**.

## Features

- **Hybrid BM25 + vector search** — TF-IDF scoring with exact phrase and filename boosts, combined with local ONNX embeddings
- **Smart chunking** — splits files into ~50-line blocks at natural blank-line boundaries
- **Incremental indexing** — skips unchanged files (SHA-256 hash check)
- **Zero cloud dependency** — uses only Node.js built-ins + local Transformers.js model
- **3 AI tools** — `rag_index`, `rag_query`, `rag_status` for the agent to use directly

## Install

```bash
pi install npm:pi-local-rag
```

Or via git:

```bash
pi install git:github.com/vahidkowsari/pi-local-rag
```

## Commands

| Command | Description |
|---|---|
| `/rag index <path>` | Index a file or directory |
| `/rag search <query>` | Search indexed content |
| `/rag status` | Show index stats (files, chunks, tokens) |
| `/rag rebuild` | Re-index changed files, prune deleted |
| `/rag clear` | Wipe the entire index |
| `/rag on` | Enable auto-injection |
| `/rag off` | Disable auto-injection |

## AI Tools

The extension registers three tools the agent can call directly:

- **`rag_index`** — Index a path into the pipeline
- **`rag_query`** — Hybrid BM25+vector search, returns file paths + line numbers + previews
- **`rag_status`** — Show index stats and RAG config

## How It Works

1. **Index** — files are chunked (~50 lines each), embedded with `Xenova/all-MiniLM-L6-v2` (384-dim), and stored in the active RAG store (see [Storage](#storage))
2. **Search** — hybrid scoring: `alpha × BM25 + (1-alpha) × cosine_similarity` (default `alpha=0.4`)
3. **Auto-inject** — before every agent turn, the prompt is searched and relevant chunks are prepended to the system prompt

## Storage

The plugin keeps each project's index and config in its own `.pi/rag/` directory. The active store is resolved per-cwd:

1. `$PI_RAG_DIR` — explicit override, if set.
2. **Walk-up** — climb upward from the working directory looking for an existing `.pi/rag/`. Walk-up stops *before* `$HOME`, so the global store never wins via walk-up.
3. **Auto-create** — the first `/rag index` (or `rag_index` tool call) in a directory with no walk-up hit creates `./.pi/rag/` at the cwd. Other commands never auto-create.
4. **Global fallback** — `~/.pi/rag/` is used when no project store is found.

`/rag status` shows the active store's path and labels it `(project)` or `(global)`. If you previously used `~/.pi/lens/`, the directory is migrated to `~/.pi/rag/` on first run.

## Configuration

Auto-injection is on by default. Tune via `/rag status`:

| Setting | Default | Description |
|---|---|---|
| `ragEnabled` | `true` | Auto-inject context before each turn |
| `ragTopK` | `5` | Max chunks to inject |
| `ragScoreThreshold` | `0.1` | Min hybrid score to include |
| `ragAlpha` | `0.4` | BM25/vector blend (0=pure vector, 1=pure BM25) |
