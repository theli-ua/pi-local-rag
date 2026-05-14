# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run build          # Type-check only (noEmit: true, --noCheck — essentially a lint pass)
npm test               # Run tests once (vitest run)
npm run test:watch     # Run tests in watch mode
```

Node >=20 required. There is no lint script configured.

## Architecture

**pi-local-rag** is a multi-file TypeScript extension for the [Pi coding agent](https://github.com/badlogic/pi-mono). It ships as TypeScript source — Pi compiles it at install time. No build output is committed.

### Module structure

| File | Responsibility |
|---|---|
| `index.ts` | Extension entry point: registers `/rag` command, 3 tools, `before_agent_start` hook. Re-exports for tests. |
| `constants.ts` | Shared constants: file extension sets, size limits, skip dirs, embedding model name, vector dim. |
| `store.ts` | Store resolution: `getRagDir()` walk-up logic, `ensureDir()`, path helpers, legacy `lens→rag` migration. |
| `config.ts` | Config load/save: `RagConfig` type, `loadConfig()`, `saveConfig()`, `defaultConfig()`. |
| `db.ts` | Database layer: `openDb()`, `initSchema()`, `getIndexStats()`, `loadIndex()`, `float32ToBuffer()`, legacy JSON→SQLite migration. Defines `Chunk`, `IndexMeta`, `IndexStats` types. |
| `chunking.ts` | File ingestion: `chunkText()`, `collectFiles()`, `collectFromTracked()`, `isExcludedByConfig()`, `extractText()` (PDF/DOCX/plain), `sha256()`. |
| `embed.ts` | Embedding: `getEmbedder()` (lazy singleton), `embed()`, `embedBatch()` with progress callback. |
| `search.ts` | Retrieval: `hybridSearch()` (FTS5 BM25 + sqlite-vec), `cosineSimilarity()`, `normalize()`. Defines `ScoredChunk` type. |
| `indexing.ts` | Indexing pipeline: `indexFiles()` (parallel read → batch embed → DB insert), `isIndexStale()`, progress helpers. |

### Extension integration

The default export in `index.ts` calls `api.registerCommand`, `api.registerTool` (×3), and `api.on("before_agent_start", ...)`. The Pi agent supplies the `ExtensionAPI` at runtime; `@earendil-works/pi-coding-agent` is a peer dependency used only for types.

### Data model

A single SQLite database (`rag.db`) lives at the active RAG store directory, resolved per-cwd by `getRagDir()`:

1. `$PI_RAG_DIR` env var, if set (used for tests / explicit overrides).
2. Walk up from `process.cwd()`, stopping before `homedir()`, returning the first ancestor that contains a `.pi/rag/` directory.
3. With `createIfMissing` (only set by `/rag index` and `rag_index`), create `${cwd}/.pi/rag/`.
4. Otherwise fall back to the global `~/.pi/rag/`.

Stopping walk-up before `$HOME` is the key invariant — it makes `~/.pi/rag/` reachable only as the explicit fallback, not via climbing through ancestors of any cwd inside the home tree.

SQLite tables:
- **`chunks`** — `id`, `file_path`, `chunk_content`, `line_start`, `line_end`, `chunk_hash`, `indexed_at`, `tokens`
- **`chunks_fts`** — FTS5 virtual table over `chunk_content` + `file_path` (kept in sync via triggers)
- **`chunks_vec`** — sqlite-vec virtual table, `embedding float[384]`
- **`files`** — `path`, `hash`, `chunks`, `indexed`, `size`, `embedded`
- **`metadata`** — key/value pairs (`last_build`, `embedding_model`)

Config is stored as JSON in `config.json`: `{ ragEnabled, ragTopK, ragScoreThreshold, ragAlpha, trackedPaths, excludePatterns }`.

### Indexing pipeline

1. Walk directory tree, filtering by `TEXT_EXTS` and skipping `SKIP_DIRS` plus hidden dirs. Files >500 KB skipped (binary docs <10 MB).
2. SHA-256 hash the file content; skip if hash matches existing DB entry with `embedded=1`.
3. Chunk each file: split on blank lines, cap at 50 lines, backtrack up to 15 lines to find a blank-line boundary. Discard chunks <20 chars.
4. Parallel read + chunk (bounded concurrency=32), then batch-embed via `@xenova/transformers` (`Xenova/all-MiniLM-L6-v2`, 384-dim ONNX) in cross-file groups of 256.
5. Insert chunks + vectors into SQLite in a single transaction.

### Search

`hybridSearch(query, index, limit, alpha, db)` blends two signals:

- **BM25** via SQLite FTS5: native `bm25()` scoring on `chunks_fts`, with filename boost for first query term.
- **Vector** via sqlite-vec: L2 nearest-neighbor search on `chunks_vec`, converted to cosine similarity.

Both score arrays are min-max normalized to [0,1], then combined: `alpha × BM25 + (1-alpha) × cosine`. Default `alpha=0.4` (slightly vector-leaning). Results below `ragScoreThreshold` are dropped; top `ragTopK` are returned.

### Auto-injection hook

`before_agent_start` runs a silent `hybridSearch` against the current user prompt and prepends matching chunks as a fenced code block to the system prompt. If the index is stale (>24h), tracked paths are re-walked and changed files are refreshed before searching. Controlled by `ragEnabled` config flag.

### Legacy migration

On first run, if `~/.pi/lens/` exists and `~/.pi/rag/` does not, `ensureDir()` renames the directory automatically. If a legacy `index.json` exists in the store, `openDb()` migrates it to SQLite and deletes the JSON file. Both migrations only trigger when the resolved store *is* the home-dir global.
