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

**pi-local-rag** is a single-file TypeScript extension (`index.ts`) for the [Pi coding agent](https://github.com/badlogic/pi-mono). It ships as TypeScript source — Pi compiles it at install time. No build output is committed.

### Extension integration

The default export calls `api.registerCommand`, `api.registerTool` (×3), and `api.on("before_agent_start", ...)`. The Pi agent supplies the `ExtensionAPI` at runtime; `@mariozechner/pi-coding-agent` is a peer dependency used only for types.

### Data model

Two JSON files live at the active RAG store directory. The store is resolved per-cwd by `getRagDir()`:

1. `$PI_RAG_DIR` env var, if set (used for tests / explicit overrides).
2. Walk up from `process.cwd()`, stopping before `homedir()`, returning the first ancestor that contains a `.pi/rag/` directory.
3. With `createIfMissing` (only set by `/rag index` and `rag_index`), create `${cwd}/.pi/rag/`.
4. Otherwise fall back to the global `~/.pi/rag/`.

Stopping walk-up before `$HOME` is the key invariant — it makes `~/.pi/rag/` reachable only as the explicit fallback, not via climbing through ancestors of any cwd inside the home tree.

- **`index.json`** — `IndexMeta`: flat `chunks[]` array + per-file metadata map (`files`). Each `Chunk` carries `{ id, file, content, lineStart, lineEnd, hash, indexed, tokens, vector? }`. `vector` is a 384-dim float array added after the embed step.
- **`config.json`** — `RagConfig`: `{ ragEnabled, ragTopK, ragScoreThreshold, ragAlpha, trackedPaths, excludePatterns }`.

### Indexing pipeline

1. Walk directory tree, filtering by `TEXT_EXTS` and skipping `SKIP_DIRS` plus hidden dirs. Files >500 KB are skipped.
2. SHA-256 hash the file content; skip if hash matches existing index entry (`files[fp].embedded === true`).
3. Chunk each file: split on blank lines, cap at 50 lines, backtrack up to 15 lines to find a blank-line boundary. Discard chunks <20 chars.
4. Batch-embed chunks via `@xenova/transformers` (`Xenova/all-MiniLM-L6-v2`, 384-dim ONNX). The pipeline singleton (`_pipeline`) is lazy-initialized on first embed call.
5. Write updated `index.json`.

### Search

`hybridSearch(query, index, cfg)` blends two signals:

- **BM25**: IDF-weighted TF scoring with boosts for exact phrase matches and filename hits.
- **Vector**: cosine similarity between query embedding and each chunk's stored vector.

Both score arrays are min-max normalized, then combined: `alpha × BM25 + (1-alpha) × cosine`. Default `alpha=0.4` (slightly vector-leaning). Results below `ragScoreThreshold` are dropped; top `ragTopK` are returned.

### Auto-injection hook

`before_agent_start` runs a silent `hybridSearch` against the current user prompt and prepends matching chunks as a fenced code block to the system prompt. Controlled by `ragEnabled` config flag — toggling it does not require re-indexing.

### Legacy migration

On first run, if `~/.pi/lens/` exists and `~/.pi/rag/` does not, `ensureDir()` renames the directory automatically (falls back to copy+delete for cross-filesystem moves). The migration only triggers when the resolved store *is* the home-dir global — project stores have nothing to migrate.
