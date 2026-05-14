/**
 * pi-local-rag — Hybrid RAG Pipeline (BM25 + Vector + Auto-injection)
 *
 * Index local files → chunk → embed → store → retrieve → inject into LLM context.
 * Uses Transformers.js (ONNX) for local embeddings — zero cloud dependency.
 *
 * Storage: SQLite (better-sqlite3) with FTS5 for BM25 and sqlite-vec for vector search.
 *
 * Storage is per-cwd: walk up from the working directory looking for a `.pi/rag/`
 * project store; fall back to `~/.pi/rag/` as the global default. The first
 * `/rag index` in a directory with no parent store creates one at cwd.
 *
 * /rag index <path>     → index + embed a file or directory
 * /rag search <query>   → hybrid search (BM25 + vector)
 * /rag status           → show index stats
 * /rag rebuild          → rebuild entire index
 * /rag clear            → clear index
 * /rag on|off           → toggle auto-injection
 *
 * Tools: rag_index, rag_query, rag_status
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { AutocompleteItem } from "@earendil-works/pi-tui";
import { Type } from "@sinclair/typebox";
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync, renameSync, unlinkSync } from "node:fs";
import { join, extname, basename, resolve, relative, dirname } from "node:path";
import { homedir } from "node:os";
import { createHash } from "node:crypto";
import ignore from "ignore";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";

// ─── Constants ───────────────────────────────────────────────────────────────

const LEGACY_DIR = join(homedir(), ".pi", "lens"); // renamed from lens → rag
const GLOBAL_RAG_DIR = () => join(homedir(), ".pi", "rag");

const RST = "\x1b[0m", B = "\x1b[1m", D = "\x1b[2m";
const GREEN = "\x1b[32m", YELLOW = "\x1b[33m", CYAN = "\x1b[36m", RED = "\x1b[31m", MAGENTA = "\x1b[35m";

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const VECTOR_DIM = 384;

const TEXT_EXTS = new Set([
  ".md", ".txt", ".ts", ".js", ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".cs",
  ".css", ".html", ".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".sh",
  ".sql", ".graphql", ".proto", ".env", ".gitignore", ".dockerfile",
]);

const BINARY_DOC_EXTS = new Set([".pdf", ".docx"]);

const TEXT_MAX_BYTES = 500_000;
const BINARY_DOC_MAX_BYTES = 10_000_000;

const SKIP_DIRS = new Set([
  "node_modules", ".git", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".cache",
]);

// ─── Types ───────────────────────────────────────────────────────────────────

interface Chunk {
  id: string;
  file: string;
  content: string;
  lineStart: number;
  lineEnd: number;
  hash: string;
  indexed: string;
  tokens: number;
  vector?: number[]; // 384-dim embedding, present after embed step
}

interface IndexMeta {
  chunks: Chunk[];
  files: Record<string, { hash: string; chunks: number; indexed: string; size: number; embedded?: boolean }>;
  lastBuild: string;
  embeddingModel?: string;
}

interface RagConfig {
  ragEnabled: boolean;
  ragTopK: number;
  ragScoreThreshold: number;
  ragAlpha: number; // 0 = pure vector, 1 = pure BM25
  trackedPaths: string[];      // absolute paths previously passed to /rag index
  excludePatterns: string[];   // gitignore-style patterns
}

interface IndexStats {
  totalChunks: number;
  totalFiles: number;
  totalTokens: number;
  embeddedCount: number;
  lastBuild: string;
  embeddingModel: string;
}

interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
}

type RagCommandCtx = Parameters<NonNullable<Parameters<ExtensionAPI["registerCommand"]>[0]["handler"]>>[1];

// ─── Store resolution ────────────────────────────────────────────────────────

/**
 * Resolve the active RAG store directory for the current cwd.
 *
 * 1. `$PI_RAG_DIR` — explicit override, wins over everything.
 * 2. Walk upward from `process.cwd()` looking for an existing `.pi/rag/`,
 *    stopping before `homedir()` so the global store at `~/.pi/rag/` is only
 *    reached as an explicit fallback (not via walk-up).
 * 3. With `createIfMissing`, create `${cwd}/.pi/rag/`.
 * 4. Otherwise, fall back to `${homedir()}/.pi/rag/`.
 */
export function getRagDir(opts: { createIfMissing?: boolean } = {}): string {
  const override = process.env.PI_RAG_DIR;
  if (override) {
    if (!existsSync(override)) mkdirSync(override, { recursive: true });
    return override;
  }
  const home = homedir();
  let dir = process.cwd();
  // Walk-up search, stopping before $HOME.
  while (true) {
    if (dir === home) break;
    const candidate = join(dir, ".pi", "rag");
    if (existsSync(candidate)) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break; // reached filesystem root
    dir = parent;
  }
  if (opts.createIfMissing) {
    const local = join(process.cwd(), ".pi", "rag");
    mkdirSync(local, { recursive: true });
    return local;
  }
  // Fallback: home-dir global. ensureDir handles creation + lens→rag migration.
  const global = GLOBAL_RAG_DIR();
  ensureDir(global);
  return global;
}

function dbFile(ragDir: string): string { return join(ragDir, "rag.db"); }
function legacyIndexFile(ragDir: string): string { return join(ragDir, "index.json"); }
function configFile(ragDir: string): string { return join(ragDir, "config.json"); }

function ensureDir(ragDir: string) {
  if (existsSync(ragDir)) return;
  // Lens→rag migration only applies at the home-dir global store.
  if (ragDir === GLOBAL_RAG_DIR() && existsSync(LEGACY_DIR)) {
    try {
      renameSync(LEGACY_DIR, ragDir);
      return;
    } catch { /* fall through to mkdir */ }
  }
  mkdirSync(ragDir, { recursive: true });
}

// ─── Config ──────────────────────────────────────────────────────────────────

export function loadConfig(): RagConfig {
  const ragDir = getRagDir();
  const cfgFile = configFile(ragDir);
  if (!existsSync(cfgFile)) return defaultConfig();
  try {
    return { ...defaultConfig(), ...JSON.parse(readFileSync(cfgFile, "utf-8")) };
  } catch { return defaultConfig(); }
}

export function defaultConfig(): RagConfig {
  return {
    ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4,
    trackedPaths: [], excludePatterns: [],
  };
}

export function saveConfig(config: RagConfig) {
  const ragDir = getRagDir();
  writeFileSync(configFile(ragDir), JSON.stringify(config, null, 2));
}

// ─── Database ────────────────────────────────────────────────────────────────

/**
 * Open (or create) the RAG SQLite database.
 * Migrates from legacy index.json if present.
 */
export function openDb(ragDir?: string): Database.Database {
  const dir = ragDir ?? getRagDir();
  ensureDir(dir);
  const path = dbFile(dir);
  const db = new Database(path);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");
  loadVec(db);
  initSchema(db);

  // Migrate from legacy index.json if it exists and rag.db is empty
  const legacyPath = legacyIndexFile(dir);
  if (existsSync(legacyPath)) {
    const chunkCount = db.prepare("SELECT COUNT(*) as c FROM chunks").get() as { c: number };
    if (chunkCount.c === 0) {
      migrateFromJson(db, legacyPath);
    }
  }

  return db;
}

/** Get the default database (resolved from cwd). */
export function getDb(): Database.Database {
  return openDb();
}

export function initSchema(db: Database.Database) {
  // Drop old triggers first (IF NOT EXISTS doesn't overwrite)
  db.exec(`DROP TRIGGER IF EXISTS chunks_ai; DROP TRIGGER IF EXISTS chunks_ad;`);
  db.exec(`
    CREATE TABLE IF NOT EXISTS metadata (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id          TEXT PRIMARY KEY,
      file_path   TEXT NOT NULL,
      chunk_content TEXT NOT NULL,
      line_start  INTEGER NOT NULL,
      line_end    INTEGER NOT NULL,
      chunk_hash  TEXT NOT NULL,
      indexed_at  TEXT NOT NULL,
      tokens      INTEGER NOT NULL
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      chunk_content,
      file_path,
      content_rowid=rowid
    );

    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, chunk_content, file_path)
      VALUES (new.rowid, new.chunk_content, new.file_path);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      DELETE FROM chunks_fts WHERE rowid = old.rowid;
    END;

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
      embedding float[${VECTOR_DIM}]
    );

    CREATE TABLE IF NOT EXISTS files (
      path      TEXT PRIMARY KEY,
      hash      TEXT NOT NULL,
      chunks    INTEGER NOT NULL,
      indexed   TEXT NOT NULL,
      size      INTEGER NOT NULL,
      embedded  INTEGER NOT NULL DEFAULT 0
    );
  `);
}

/**
 * Migrate data from legacy index.json into SQLite.
 * Deletes the legacy file on success.
 */
function migrateFromJson(db: Database.Database, jsonPath: string): void {
  let data: IndexMeta;
  try {
    data = JSON.parse(readFileSync(jsonPath, "utf-8"));
  } catch {
    return;
  }

  if (!data.chunks || data.chunks.length === 0) {
    try { unlinkSync(jsonPath); } catch {}
    return;
  }

  const tx = db.transaction(() => {
    // Insert chunks
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insVec = db.prepare("INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)");
    const insFile = db.prepare(`
      INSERT OR REPLACE INTO files(path, hash, chunks, indexed, size, embedded)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    for (const c of data.chunks) {
      const chunkResult = insChunk.run(c.id, c.file, c.content, c.lineStart, c.lineEnd, c.hash, c.indexed, c.tokens);
      if (c.vector && c.vector.length === VECTOR_DIM) {
        insVec.run(Number(chunkResult.lastInsertRowid), float32ToBuffer(c.vector));
      }
    }

    for (const [fp, info] of Object.entries(data.files || {})) {
      insFile.run(fp, info.hash, info.chunks, info.indexed, info.size, info.embedded ? 1 : 0);
    }

    if (data.lastBuild) {
      db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_build', ?)").run(data.lastBuild);
    }
    if (data.embeddingModel) {
      db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('embedding_model', ?)").run(data.embeddingModel);
    }
  });

  tx();

  // Delete legacy file on success
  try { unlinkSync(jsonPath); } catch {}
}

function float32ToBuffer(arr: number[]): Buffer {
  const f = new Float32Array(arr);
  return Buffer.from(f.buffer, f.byteOffset, f.byteLength);
}

/** Get index statistics from the database. */
export function getIndexStats(db: Database.Database): IndexStats {
  const chunkRow = db.prepare(`
    SELECT COUNT(*) as totalChunks,
           COALESCE(SUM(tokens), 0) as totalTokens
    FROM chunks
  `).get() as { totalChunks: number; totalTokens: number };

  const fileRow = db.prepare("SELECT COUNT(*) as totalFiles FROM files").get() as { totalFiles: number };

  const vecRow = db.prepare("SELECT COUNT(*) as embeddedCount FROM chunks_vec").get() as { embeddedCount: number };

  const lastBuild = db.prepare("SELECT value FROM metadata WHERE key = 'last_build'").get() as { value?: string } | undefined;
  const embeddingModel = db.prepare("SELECT value FROM metadata WHERE key = 'embedding_model'").get() as { value?: string } | undefined;

  return {
    totalChunks: chunkRow.totalChunks,
    totalFiles: fileRow.totalFiles,
    totalTokens: chunkRow.totalTokens,
    embeddedCount: vecRow.embeddedCount,
    lastBuild: lastBuild?.value ?? "",
    embeddingModel: embeddingModel?.value ?? "",
  };
}

/**
 * For backward compatibility: returns an IndexMeta-like object from the DB.
 * Only used by code that expects the old shape.
 */
export function loadIndex(): IndexMeta {
  const db = getDb();
  try {
    const chunks = db.prepare(`
      SELECT c.id, c.file_path as file, c.chunk_content as content,
             c.line_start as lineStart, c.line_end as lineEnd,
             c.chunk_hash as hash, c.indexed_at as indexed, c.tokens
      FROM chunks c
    `).all() as Chunk[];

    const filesRaw = db.prepare("SELECT * FROM files").all() as Array<{
      path: string; hash: string; chunks: number; indexed: string; size: number; embedded: number;
    }>;
    const files: IndexMeta["files"] = {};
    for (const f of filesRaw) {
      files[f.path] = { hash: f.hash, chunks: f.chunks, indexed: f.indexed, size: f.size, embedded: !!f.embedded };
    }

    const lastBuild = db.prepare("SELECT value FROM metadata WHERE key = 'last_build'").get() as { value?: string } | undefined;
    const embeddingModel = db.prepare("SELECT value FROM metadata WHERE key = 'embedding_model'").get() as { value?: string } | undefined;

    return {
      chunks,
      files,
      lastBuild: lastBuild?.value ?? "",
      embeddingModel: embeddingModel?.value,
    };
  } finally {
    db.close();
  }
}

function saveIndex(_index: IndexMeta) {
  // No-op: data is stored in SQLite, not JSON
}

export function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex").slice(0, 12);
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

let _pipeline: any = null;

async function getEmbedder() {
  if (_pipeline) return _pipeline;
  const { pipeline } = await import("@xenova/transformers");
  _pipeline = await pipeline("feature-extraction", EMBEDDING_MODEL);
  return _pipeline;
}

async function embed(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

const EMBED_BATCH_SIZE = 64;

export async function embedBatch(texts: string[], onProgress?: (i: number, total: number) => void): Promise<number[][]> {
  const embedder = await getEmbedder();
  const results: number[][] = [];
  for (let start = 0; start < texts.length; start += EMBED_BATCH_SIZE) {
    const batch = texts.slice(start, start + EMBED_BATCH_SIZE);
    try {
      const output = await embedder(batch, { pooling: "mean", normalize: true, truncate: true });
      for (let b = 0; b < batch.length; b++) {
        results.push(Array.from(output[b].data as Float32Array));
      }
    } catch (err) {
      process.stderr.write(`\n[embedBatch ERROR] batch ${start}/${texts.length}: ${err instanceof Error ? err.message : String(err)}\n`);
      throw err;
    }
    const done = start + batch.length;
    for (let b = 0; b < batch.length; b++) onProgress?.(done - batch.length + b + 1, texts.length);
  }
  return results;
}

// ─── Math ────────────────────────────────────────────────────────────────────

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function normalize(scores: number[]): number[] {
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  if (range === 0) return scores.map(() => 0);
  return scores.map(s => (s - min) / range);
}

// L2 distance → cosine similarity (valid when vectors are unit-length)
function l2ToCosine(l2Dist: number): number {
  return 1 - (l2Dist * l2Dist) / 2;
}

// ─── Chunking & File Collection ──────────────────────────────────────────────

export function chunkText(text: string, maxLines = 50): { content: string; lineStart: number; lineEnd: number }[] {
  const lines = text.split("\n");
  const chunks: { content: string; lineStart: number; lineEnd: number }[] = [];
  let i = 0;
  while (i < lines.length) {
    let end = Math.min(i + maxLines, lines.length);
    for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
      if (lines[j]?.trim() === "") { end = j + 1; break; }
    }
    const chunk = lines.slice(i, end).join("\n");
    if (chunk.trim().length > 20) {
      chunks.push({ content: chunk, lineStart: i + 1, lineEnd: end });
    }
    i = end;
  }
  return chunks;
}

export function collectFiles(dirPath: string, excludePatterns: string[] = []): string[] {
  const ig = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const files: string[] = [];

  function isExcluded(absPath: string, root: string): boolean {
    if (!ig) return false;
    const rel = relative(root, absPath);
    if (!rel || rel.startsWith("..")) return false;
    return ig.ignores(rel);
  }

  function acceptable(fp: string, size: number): boolean {
    const ext = extname(fp).toLowerCase();
    if (TEXT_EXTS.has(ext)) return size < TEXT_MAX_BYTES;
    if (BINARY_DOC_EXTS.has(ext)) return size < BINARY_DOC_MAX_BYTES;
    return false;
  }

  try {
    const stat = statSync(dirPath);
    if (stat.isFile()) {
      if (!acceptable(dirPath, stat.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch { return []; }

  const root = dirPath;
  function walk(dir: string) {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const fp = join(dir, entry.name);
        if (entry.isDirectory()) {
          if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
          if (isExcluded(fp, root)) continue;
          walk(fp);
        } else {
          const ext = extname(entry.name).toLowerCase();
          if (!TEXT_EXTS.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
          if (isExcluded(fp, root)) continue;
          try {
            if (acceptable(fp, statSync(fp).size)) files.push(fp);
          } catch {}
        }
      }
    } catch {}
  }
  walk(root);
  return files;
}

export function collectFromTracked(cfg: RagConfig): string[] {
  const out = new Set<string>();
  for (const p of cfg.trackedPaths) {
    if (!existsSync(p)) continue;
    for (const f of collectFiles(p, cfg.excludePatterns)) out.add(f);
  }
  return [...out];
}

/** Returns true if `file` is matched by `excludePatterns` relative to any of `roots`. */
export function isExcludedByConfig(file: string, roots: string[], excludePatterns: string[]): boolean {
  if (!excludePatterns.length) return false;
  const ig = ignore().add(excludePatterns);
  for (const root of roots) {
    const rel = relative(root, file);
    if (!rel || rel.startsWith("..")) continue;
    if (ig.ignores(rel)) return true;
  }
  return false;
}

// ─── Indexing ─────────────────────────────────────────────────────────────────

/**
 * Read and decode a file into UTF-8 text. PDF and DOCX are routed through
 * extraction libraries; everything else is read as plain UTF-8. Hash is
 * computed over the raw bytes for binaries (so the source file's identity
 * drives skip-on-rebuild) and over the decoded text for plain text files.
 */
// pdfjs (bundled inside pdf-parse) routes warnings through console.log with a
// "Warning: " prefix. On real-world PDFs this fires thousands of times per
// document ("Ran out of space in font private use area", missing glyphs, …).
// The font warnings come from pdf.worker.js, which is a separate webpack
// bundle whose verbosity is not externally configurable (its setVerbosityLevel
// export exists only as a placeholder at the outer module level). Filtering
// console.log for the known pdfjs prefixes is the only reliable approach.
const PDFJS_LOG_PREFIX = /^(Warning|Info|Deprecated API usage):/;
async function withPdfjsSilenced<T>(fn: () => Promise<T>): Promise<T> {
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    const first = args[0];
    if (typeof first === "string" && PDFJS_LOG_PREFIX.test(first)) return;
    origLog(...args);
  };
  try {
    return await fn();
  } finally {
    console.log = origLog;
  }
}

export async function extractText(fp: string): Promise<{ text: string; hash: string; size: number }> {
  const ext = extname(fp).toLowerCase();
  if (ext === ".pdf") {
    const buf = readFileSync(fp);
    const { default: pdf } = await import("pdf-parse/lib/pdf-parse.js");
    const data = await withPdfjsSilenced(() => pdf(buf));
    return { text: data.text, hash: sha256(buf.toString("binary")), size: buf.length };
  }
  if (ext === ".docx") {
    const buf = readFileSync(fp);
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return { text: value, hash: sha256(buf.toString("binary")), size: buf.length };
  }
  const text = readFileSync(fp, "utf-8");
  return { text, hash: sha256(text), size: text.length };
}

interface ProgressCallbacks {
  onFile?: (current: number, total: number, filename: string, skipped: number) => void;
  onChunk?: (fileChunk: number, totalChunks: number, filename: string) => void;
  onSave?: () => void;
}

/** Yield to the event loop so the TUI can re-render between heavy operations */
const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

/** Write overwriting progress line to stderr (visible in terminal even during tool calls) */
function stderrProgress(msg: string) {
  if (_suppressStderr) return;
  process.stderr.write(`\r\x1b[2K${msg}`);
}

/** Whether stderr progress should be suppressed (e.g. when TUI callbacks handle display) */
let _suppressStderr = false;

interface _FileWork {
  fp: string;
  hash: string;
  size: number;
  rawChunks: { content: string; lineStart: number; lineEnd: number }[];
  _vectors?: number[][]; // populated during embed phase
}

export async function indexFiles(
  paths: string[],
  progress?: ProgressCallbacks,
  _db?: Database.Database
): Promise<{ indexed: number; chunks: number; skipped: number; durationMs: number }> {
  // Suppress stderr progress when TUI callbacks handle display to avoid flashing
  const hadCallbacks = !!progress;
  if (hadCallbacks) _suppressStderr = true;
  const database = _db ?? openDb();
  let indexed = 0, chunked = 0, skipped = 0;
  const startMs = Date.now();
  const total = paths.length;

  try {

  const delChunks = database.prepare("DELETE FROM chunks WHERE file_path = ?");
  const delVec = database.prepare("DELETE FROM chunks_vec WHERE rowid IN (SELECT rowid FROM chunks WHERE file_path = ?)");

  // ── Phase 1: read + chunk all files; skip unchanged ──
  const toIndex: _FileWork[] = [];
  for (let i = 0; i < paths.length; i++) {
    const fp = paths[i];
    const pct = Math.round(((i + 1) / total) * 100);
    const name = basename(fp);

    try {
      const { text: content, hash, size } = await extractText(fp);

      const existing = database.prepare("SELECT hash, embedded FROM files WHERE path = ?").get(fp) as { hash?: string; embedded?: number } | undefined;
      if (existing?.hash === hash && existing?.embedded) {
        skipped++;
        stderrProgress(`[${i + 1}/${total}] ${pct}% skipped ${name}`);
        progress?.onFile?.(i + 1, total, name, skipped);
        await yield_();
        continue;
      }

      delVec.run(fp);
      delChunks.run(fp);

      const rawChunks = chunkText(content);
      stderrProgress(`[${i + 1}/${total}] ${pct}% chunked ${name} (${rawChunks.length} chunks)`);
      progress?.onFile?.(i + 1, total, name, skipped);
      await yield_();

      toIndex.push({ fp, hash, size, rawChunks });
    } catch (err) {
      skipped++;
      stderrProgress(`[${i + 1}/${total}] ERROR ${name}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  // ── Phase 2: embed in cross-file groups (bounded memory) ──
  // Collect chunks across files until we hit a group target, then embed.
  // Keeps memory bounded for 100k+ files while still batching across files.
  const EMBED_GROUP_TARGET = 256;
  const groupChunks: { fw: _FileWork; ci: number }[] = [];
  let globalChunkIdx = 0;

  const flushGroup = async () => {
    if (groupChunks.length === 0) return;
    const texts = groupChunks.map(g => g.fw.rawChunks[g.ci].content);
    const totalChunks = toIndex.reduce((s, f) => s + f.rawChunks.length, 0);
    stderrProgress(`Embedding ${globalChunkIdx - groupChunks.length + 1}…${globalChunkIdx}/${totalChunks} chunks`);
    await yield_();
    const vectors = await embedBatch(texts);
    // Store vectors on the _FileWork for DB insert phase
    for (let vi = 0; vi < groupChunks.length; vi++) {
      const g = groupChunks[vi];
      g.fw._vectors ??= new Array(g.fw.rawChunks.length);
      g.fw._vectors[g.ci] = vectors[vi];
    }
    groupChunks.length = 0;
  };

  for (const fw of toIndex) {
    for (let j = 0; j < fw.rawChunks.length; j++) {
      groupChunks.push({ fw, ci: j });
      globalChunkIdx++;
      if (groupChunks.length >= EMBED_GROUP_TARGET) await flushGroup();
    }
  }
  await flushGroup(); // remaining

  // ── Phase 3: insert chunks + vectors into DB ──
  const insChunk = database.prepare(`
    INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insVecRowid = database.prepare("INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)");
  const upsertFile = database.prepare(`
    INSERT INTO files(path, hash, chunks, indexed, size, embedded)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(path) DO UPDATE SET
      hash=excluded.hash, chunks=excluded.chunks, indexed=excluded.indexed,
      size=excluded.size, embedded=excluded.embedded
  `);

  const tx = database.transaction(() => {
    for (const fw of toIndex) {
      const vectors = fw._vectors;
      for (let j = 0; j < fw.rawChunks.length; j++) {
        const c = fw.rawChunks[j];
        const chunkResult = insChunk.run(
          `${sha256(fw.fp)}-${c.lineStart}`,
          fw.fp,
          c.content,
          c.lineStart,
          c.lineEnd,
          sha256(c.content),
          new Date().toISOString(),
          Math.ceil(c.content.length / 4),
        );
        if (vectors?.[j]) {
          insVecRowid.run(Number(chunkResult.lastInsertRowid), float32ToBuffer(vectors[j]));
        }
        chunked++;
      }
      upsertFile.run(fw.fp, fw.hash, fw.rawChunks.length, new Date().toISOString(), fw.size, 1);
    }
    indexed = toIndex.length;
  });

  tx();

  // Clear stderr progress line
  if (!hadCallbacks) process.stderr.write(`\r\x1b[2K`);

  progress?.onSave?.();
  database.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_build', ?)").run(new Date().toISOString());
  database.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('embedding_model', ?)").run(EMBEDDING_MODEL);

  return { indexed, chunks: chunked, skipped, durationMs: Date.now() - startMs };
  } finally {
    if (hadCallbacks) _suppressStderr = false;
  }
}

// ─── Staleness ───────────────────────────────────────────────────────────────

export function isIndexStale(index: IndexMeta, maxAgeMs = 24 * 60 * 60 * 1000): boolean {
  if (!index.lastBuild) return false;
  return Date.now() - new Date(index.lastBuild).getTime() > maxAgeMs;
}

// ─── Search ───────────────────────────────────────────────────────────────────

/**
 * Hybrid search using SQLite FTS5 (BM25) + sqlite-vec (vector).
 *
 * BM25 scores from FTS5 are negative floats (closer to 0 = better).
 * Vector distances from sqlite-vec are L2 distances (smaller = better).
 * Both are min-max normalized to [0, 1] then blended.
 */
export async function hybridSearch(
  query: string,
  _index: IndexMeta, // kept for API compat; actual search uses DB
  limit = 10,
  alpha = 0.4,
  _db?: Database.Database
): Promise<ScoredChunk[]> {
  const database = _db ?? openDb();

  try {
    const chunkCount = database.prepare("SELECT COUNT(*) as c FROM chunks").get() as { c: number };
    if (chunkCount.c === 0) return [];

    // ── BM25 via FTS5 ──
    // Escape the query as a literal FTS5 string: wrap in double-quotes,
    // doubling any embedded double-quotes.  This safely handles single
    // quotes, parens, boolean operators, etc.
    const ftsQuery = `"${query.replace(/"/g, '""')}"`;
    const ftsResults = database.prepare(`
      SELECT chunks_fts.rowid, bm25(chunks_fts) as bm25_score
      FROM chunks_fts
      WHERE chunks_fts MATCH ?
      ORDER BY bm25(chunks_fts)
    `).all(ftsQuery);

    // ── Vector via sqlite-vec ──
    const queryVec = await embed(query);
    const queryBuf = float32ToBuffer(queryVec);
    const vecResults = database.prepare(`
      SELECT rowid, distance
      FROM chunks_vec
      WHERE embedding MATCH ?
      LIMIT ?
    `).bind(queryBuf, Math.max(limit, 50)).all();

    // ── Fetch chunk details for all candidate rowids ──
    const ftsRowIds = new Set(ftsResults.map((r: any) => r.rowid));
    const vecRowIds = new Set(vecResults.map((r: any) => r.rowid));
    const allRowIds = new Set([...ftsRowIds, ...vecRowIds]);

    if (allRowIds.size === 0) return [];

    // Build rowid → chunk map
    const rowidPlaceholders = Array.from(allRowIds).map(() => "?").join(",");
    const rowidValues = Array.from(allRowIds);
    const chunks = database.prepare(`
      SELECT rowid, id, file_path, chunk_content, line_start, line_end,
             chunk_hash, indexed_at, tokens
      FROM chunks
      WHERE rowid IN (${rowidPlaceholders})
    `).all(...rowidValues) as Array<{
      rowid: number; id: string; file_path: string; chunk_content: string;
      line_start: number; line_end: number; chunk_hash: string;
      indexed_at: string; tokens: number;
    }>;

    const chunkMap = new Map<number, typeof chunks[0]>();
    for (const c of chunks) chunkMap.set(c.rowid, c);

    // ── Score computation ──
    const bm25Map = new Map<number, number>();
    for (const r of ftsResults) bm25Map.set(r.rowid, r.bm25_score);

    const distMap = new Map<number, number>();
    for (const r of vecResults) distMap.set(r.rowid, r.distance);

    // Collect all BM25 scores for normalization
    const bm25Scores = ftsResults.map((r: any) => r.bm25_score);
    const hasBm25 = bm25Scores.length > 0;

    // Collect all distances for normalization
    const distances = vecResults.map((r: any) => r.distance);
    const hasVectors = distances.length > 0;

    // Normalize BM25 (negatives: closer to 0 = better → higher normalized score)
    const bm25NormMap = new Map<number, number>();
    if (hasBm25) {
      const bm25Max = Math.max(...bm25Scores); // closest to 0
      const bm25Min = Math.min(...bm25Scores);
      const bm25Range = bm25Max - bm25Min;
      if (bm25Range === 0) {
        // Single result or all equal → give max score
        for (const r of ftsResults) bm25NormMap.set(r.rowid, 1);
      } else {
        for (const r of ftsResults) {
          bm25NormMap.set(r.rowid, (r.bm25_score - bm25Min) / bm25Range);
        }
      }
    }

    // Normalize distances (smaller = better → higher normalized score)
    const vecNormMap = new Map<number, number>();
    if (hasVectors) {
      const distMax = Math.max(...distances);
      const distMin = Math.min(...distances);
      const distRange = distMax - distMin;
      for (const r of vecResults) {
        // Convert L2 to cosine similarity, then normalize
        const cosine = l2ToCosine(r.distance);
        vecNormMap.set(r.rowid, cosine);
      }
      // Min-max normalize cosine scores
      const cosines = Array.from(vecNormMap.values());
      const cosMax = Math.max(...cosines);
      const cosMin = Math.min(...cosines);
      const cosRange = cosMax - cosMin;
      if (cosRange > 0) {
        const normalized = new Map<number, number>();
        for (const [rowid, cos] of vecNormMap) {
          normalized.set(rowid, (cos - cosMin) / cosRange);
        }
        vecNormMap.clear();
        for (const [k, v] of normalized) vecNormMap.set(k, v);
      } else {
        // Single result or all equal → give max score
        for (const k of vecNormMap.keys()) vecNormMap.set(k, 1);
      }
    }

    // ── Build scored results ──
    const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);

    const scored: ScoredChunk[] = [];
    for (const rowid of allRowIds) {
      const c = chunkMap.get(rowid);
      if (!c) continue;

      const bm25Raw = bm25Map.get(rowid) ?? 0;
      const bm25Norm = bm25NormMap.get(rowid) ?? 0;
      const vecNorm = vecNormMap.get(rowid) ?? 0;

      // Filename boost (matching original behavior)
      let bm25Final = bm25Norm;
      if (c.file_path.toLowerCase().includes(terms[0] ?? "")) {
        bm25Final = Math.min(1, bm25Final * 1.5);
      }

      const hybrid = hasVectors
        ? alpha * bm25Final + (1 - alpha) * vecNorm
        : bm25Final;

      scored.push({
        chunk: {
          id: c.id,
          file: c.file_path,
          content: c.chunk_content,
          lineStart: c.line_start,
          lineEnd: c.line_end,
          hash: c.chunk_hash,
          indexed: c.indexed_at,
          tokens: c.tokens,
        },
        bm25: bm25Final,
        vector: vecNorm,
        hybrid,
      });
    }

    return scored
      .filter(s => s.hybrid > 0)
      .sort((a, b) => b.hybrid - a.hybrid)
      .slice(0, limit);
  } finally {
    database.close();
  }
}

// ─── Extension ────────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // ── Auto-inject RAG context before every agent turn ──
  pi.on("before_agent_start", async (event, _ctx) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;

    const database = openDb();
    try {
      const stats = getIndexStats(database);
      if (stats.totalChunks === 0) return;

      // Check staleness
      const indexMeta: IndexMeta = { chunks: [], files: {}, lastBuild: stats.lastBuild, embeddingModel: stats.embeddingModel };
      if (isIndexStale(indexMeta)) {
        const files = config.trackedPaths.length
          ? collectFromTracked(config)
          : Object.keys(loadIndex().files).filter(f => existsSync(f));
        if (files.length) {
          stderrProgress(`[rag] Index stale, refreshing ${files.length} files…`);
          await indexFiles(files, {}, database);
          process.stderr.write(`\r\x1b[2K`);
        }
      }

      const results = await hybridSearch(event.prompt, indexMeta, config.ragTopK, config.ragAlpha, database);
      const relevant = results.filter(r => r.hybrid >= config.ragScoreThreshold);
      if (!relevant.length) return;

      const context = relevant.map(r =>
        `### ${basename(r.chunk.file)} (lines ${r.chunk.lineStart}-${r.chunk.lineEnd})\n` +
        `\`\`\`\n${r.chunk.content.slice(0, 600)}\n\`\`\``
      ).join("\n\n");

      return {
        message: {
          customType: "rag",
          content:
            `[pi-local-rag] Automatic RAG lookup triggered by the user's message above.\n` +
            `Retrieved ${relevant.length} chunk${relevant.length === 1 ? "" : "s"} via hybrid search (BM25 + vector). ` +
            `These are search hits, not statements from the user.\n\n` +
            context,
          display: false,
        },
      };
    } finally {
      database.close();
    }
  });

  // ── /rag command helpers ──

  function progressBar(n: number, total: number, width = 24): string {
    const filled = Math.round((n / total) * width);
    return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
  }

  async function cmdIndex(path: string, ctx: RagCommandCtx) {
    if (!existsSync(path)) { ctx.ui.notify(`Path not found: ${path}`, "error"); return; }
    getRagDir({ createIfMissing: true });
    const config = loadConfig();
    const absPath = resolve(path);
    if (!config.trackedPaths.includes(absPath)) {
      config.trackedPaths.push(absPath);
      saveConfig(config);
    }
    const files = collectFiles(absPath, config.excludePatterns);
    if (!files.length) { ctx.ui.notify(`No indexable files found in: ${path}`, "warning"); return; }

    ctx.ui.notify(`Found ${files.length} files to index`, "info");

    const database = openDb();
    try {
      const result = await indexFiles(files, {
        onFile(current, total, filename, skipped) {
          const pct = Math.round((current / total) * 100);
          const bar = progressBar(current, total);
          ctx.ui.setStatus("rag", `■ Indexing ${pct}% │ ${current}/${total} files │ ${skipped} unchanged`);
          ctx.ui.setWidget("rag", [
            `${B}${CYAN}Indexing${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
            `${D}file:    ${RST}${filename}`,
            `${D}done:    ${RST}${GREEN}${current - skipped} embedded${RST}  ${D}${skipped} unchanged${RST}`,
          ]);
        },
        onChunk(ci, total, filename) {
          ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
        },
        onSave() {
          ctx.ui.setStatus("rag", `■ Saving index...`);
        },
      }, database);

      ctx.ui.setStatus("rag", undefined);
      ctx.ui.setWidget("rag", undefined);

      const secs = (result.durationMs / 1000).toFixed(1);
      const ragDir = getRagDir();
      const scope = ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
      ctx.ui.notify(`Indexed: ${result.indexed} files (${result.chunks} chunks) │ ${result.skipped} unchanged │ ${secs}s │ tracking ${config.trackedPaths.length} path(s) │ ${scope}`, "success");
    } finally {
      database.close();
    }
  }

  async function cmdSearch(query: string, ctx: RagCommandCtx) {
    if (!query) { pi.sendMessage({ customType: "rag-search", content: "**Usage:** `/rag search <query>`", display: true }); return; }
    const config = loadConfig();
    const database = openDb();
    try {
      const stats = getIndexStats(database);
      if (stats.totalChunks === 0) {
        pi.sendMessage({ customType: "rag-search", content: `No results for: \`${query}\``, display: true });
        return;
      }
      const results = await hybridSearch(query, { chunks: [], files: {}, lastBuild: stats.lastBuild }, 10, config.ragAlpha, database);
      if (!results.length) {
        pi.sendMessage({ customType: "rag-search", content: `No results for: \`${query}\``, display: true });
        return;
      }

      const hasVectors = stats.embeddedCount > 0;
      let md = `## 🔍 ${results.length} results for "${query}"\n\n`;
      md += `*${hasVectors ? "hybrid BM25+vector" : "BM25 only — run /rag index to add vectors"}*\n\n`;

      for (const r of results) {
        const bar = "█".repeat(Math.round(r.hybrid * 10)) + "░".repeat(10 - Math.round(r.hybrid * 10));
        md += `- **${basename(r.chunk.file)}**:${r.chunk.lineStart}-${r.chunk.lineEnd} \`bm25=${r.bm25.toFixed(2)} vec=${r.vector.toFixed(2)} hybrid=${r.hybrid.toFixed(2)}\` ${bar}\n`;
        const preview = r.chunk.content.split("\n").slice(0, 3).join("\n");
        md += "  ```\n" + preview.slice(0, 200) + "\n  ```\n\n";
      }
      pi.sendMessage({ customType: "rag-search", content: md, display: true });
    } finally {
      database.close();
    }
  }

  function cmdToggle(mode: "on" | "off", ctx: RagCommandCtx) {
    const config = loadConfig();
    config.ragEnabled = mode === "on";
    saveConfig(config);
    ctx.ui.notify(mode === "on" ? "RAG auto-injection enabled" : "RAG auto-injection disabled", mode === "on" ? "success" : "warning");
  }

  async function cmdRebuild(ctx: RagCommandCtx) {
    const database = openDb();
    const config = loadConfig();

    try {
      const indexedFiles = database.prepare("SELECT path FROM files").all() as Array<{ path: string }>;
      const indexedFileSet = new Set(indexedFiles.map(f => f.path));
      const trackedFiles = collectFromTracked(config);

      // Union of currently-indexed files and files discovered by walking tracked paths.
      const targetSet = new Set<string>([...trackedFiles]);
      for (const f of indexedFileSet) {
        if (existsSync(f) && !isExcludedByConfig(f, config.trackedPaths, config.excludePatterns)) {
          targetSet.add(f);
        }
      }
      const targetFiles = [...targetSet];

      if (!targetFiles.length && !indexedFileSet.size) {
        ctx.ui.notify("No files to rebuild. Run /rag index <path> first.", "warning");
        return;
      }

      // Files in the index but no longer present (deleted, excluded, or untracked).
      const droppedFiles = [...indexedFileSet].filter(f => !targetSet.has(f));
      for (const f of droppedFiles) {
        database.prepare("DELETE FROM chunks_vec WHERE rowid IN (SELECT rowid FROM chunks WHERE file_path = ?)").run(f);
        database.prepare("DELETE FROM chunks WHERE file_path = ?").run(f);
        database.prepare("DELETE FROM files WHERE path = ?").run(f);
      }

      // Force re-embed all target files
      for (const f of targetFiles) {
        database.prepare("UPDATE files SET embedded = 0 WHERE path = ?").run(f);
      }

      const newFiles = targetFiles.filter(f => !indexedFileSet.has(f));
      if (droppedFiles.length) ctx.ui.notify(`Pruned ${droppedFiles.length} files (deleted/excluded)`, "info");
      if (newFiles.length) ctx.ui.notify(`Discovered ${newFiles.length} new files`, "info");
      ctx.ui.notify(`Rebuilding ${targetFiles.length} files...`, "info");

      const result = await indexFiles(targetFiles, {
        onFile(current, total, filename, skipped) {
          const pct = Math.round((current / total) * 100);
          const bar = progressBar(current, total);
          ctx.ui.setStatus("rag", `■ Rebuilding ${pct}% │ ${current}/${total} │ ${skipped} unchanged`);
          ctx.ui.setWidget("rag", [
            `${B}${CYAN}Rebuilding${RST}  ${bar}  ${GREEN}${pct}%${RST}`,
            `${D}file:    ${RST}${filename}`,
            `${D}done:    ${RST}${GREEN}${current - skipped} re-embedded${RST}  ${D}${skipped} unchanged${RST}`,
          ]);
        },
        onChunk(ci, total, filename) {
          ctx.ui.setStatus("rag", `■ Embedding ${filename} — chunk ${ci}/${total}`);
        },
        onSave() {
          ctx.ui.setStatus("rag", `■ Saving index...`);
        },
      }, database);

      ctx.ui.setStatus("rag", undefined);
      ctx.ui.setWidget("rag", undefined);

      const secs = (result.durationMs / 1000).toFixed(1);
      ctx.ui.notify(`Rebuilt: ${result.indexed} re-indexed │ ${result.skipped} unchanged │ ${droppedFiles.length} deleted │ ${result.chunks} chunks │ ${secs}s`, "success");
    } finally {
      database.close();
    }
  }

  function cmdClear(ctx: RagCommandCtx) {
    const database = openDb();
    try {
      database.exec(`
        DELETE FROM chunks_vec;
        DELETE FROM chunks;
        DELETE FROM files;
        DELETE FROM metadata;
      `);
      // Rebuild FTS5 after clearing
      database.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
    } finally {
      database.close();
    }
    ctx.ui.notify("Index cleared.", "success");
  }

  function cmdStatus(ctx: RagCommandCtx) {
    const database = openDb();
    const config = loadConfig();
    try {
      const stats = getIndexStats(database);
      const fileCount = stats.totalFiles;
      const totalTokens = stats.totalTokens;
      const embeddedCount = stats.embeddedCount;
      const totalChunks = stats.totalChunks;
      const vectorCoverage = totalChunks ? Math.round(embeddedCount / totalChunks * 100) : 0;
      const ragDir = getRagDir();
      const scope = ragDir === GLOBAL_RAG_DIR() ? "global" : "project";

      let md = `## 🔍 pi-local-rag Status\n\n`;
      md += `| Metric | Value |\n|---|---|\n`;
      md += `| Files indexed | ${fileCount} |\n`;
      md += `| Chunks | ${totalChunks} |\n`;
      md += `| Vectors | ${embeddedCount} (${vectorCoverage}% coverage) |\n`;
      md += `| Total tokens | ${totalTokens.toLocaleString()} |\n`;
      md += `| Embedding model | ${stats.embeddingModel || "none"} |\n`;
      md += `| Last build | ${stats.lastBuild || "never"} |\n`;
      md += `| Storage | \`${ragDir}\` (${scope}) |\n\n`;
      md += `**RAG injection:** ${config.ragEnabled ? "enabled ✅" : "disabled ⚠️"}  \n`;
      md += `\`topK=${config.ragTopK}\`  \`threshold=${config.ragScoreThreshold}\`  \`alpha=${config.ragAlpha}\`\n`;

      if (fileCount) {
        md += `\n### File types\n\n`;
        const byExt: Record<string, number> = {};
        const files = database.prepare("SELECT path FROM files").all() as Array<{ path: string }>;
        for (const f of files) byExt[extname(f.path)] = (byExt[extname(f.path)] || 0) + 1;
        for (const [ext, count] of Object.entries(byExt).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
          md += `- \`${ext}\`: ${count}\n`;
        }
      }

      md += `\n### Tracked paths\n\n`;
      if (config.trackedPaths.length) {
        for (const p of config.trackedPaths) md += `- \`${p}\`\n`;
      } else {
        md += `*(none — run /rag index <path> to track)*\n`;
      }

      md += `\n### Exclude patterns\n\n`;
      if (config.excludePatterns.length) {
        for (const p of config.excludePatterns) md += `- \`${p}\`\n`;
      } else {
        md += `*(none — add with /rag exclude <pattern>)*\n`;
      }

      pi.sendMessage({ customType: "rag-status", content: md, display: true });
    } finally {
      database.close();
    }
  }

  function cmdExclude(expr: string, ctx: RagCommandCtx) {
    const config = loadConfig();

    if (!expr) {
      if (!config.excludePatterns.length) {
        ctx.ui.notify("No exclude patterns set. Add one with: /rag exclude <pattern>", "warning");
        return;
      }
      let md = `## Exclude patterns (${config.excludePatterns.length})\n\n`;
      for (const p of config.excludePatterns) md += `- \`${p}\`\n`;
      pi.sendMessage({ customType: "rag", content: md, display: true });
      return;
    }

    if (expr.startsWith("-")) {
      const target = expr.slice(1);
      const before = config.excludePatterns.length;
      config.excludePatterns = config.excludePatterns.filter(p => p !== target);
      if (config.excludePatterns.length === before) {
        ctx.ui.notify(`Pattern not found: ${target}`, "warning");
        return;
      }
      saveConfig(config);
      ctx.ui.notify(`Removed exclude: ${target} (${config.excludePatterns.length} remain). Run /rag rebuild to re-apply.`, "success");
      return;
    }

    if (config.excludePatterns.includes(expr)) {
      ctx.ui.notify(`Already excluded: ${expr}`, "warning");
      return;
    }
    config.excludePatterns.push(expr);
    saveConfig(config);
    ctx.ui.notify(`Added exclude: ${expr} (${config.excludePatterns.length} total). Run /rag rebuild to re-apply.`, "success");
  }

  function cmdFind(glob: string, ctx: RagCommandCtx) {
    if (!glob) {
      ctx.ui.notify("Usage: /rag find <glob>   e.g. *.html, page*, foo.js, src/*.ts", "warning");
      return;
    }

    const database = openDb();
    try {
      const cwd = process.cwd();
      const ig = ignore().add([glob]);
      const files = database.prepare("SELECT path FROM files").all() as Array<{ path: string }>;

      const matches: string[] = [];
      for (const f of files) {
        const rel = relative(cwd, f.path);
        const candidate = rel && !rel.startsWith("..") ? rel : basename(f.path);
        if (ig.ignores(candidate)) matches.push(f.path);
      }
      matches.sort();

      if (!matches.length) {
        ctx.ui.notify(`No indexed files match: ${glob}`, "warning");
        return;
      }

      let md = `## 🔍 ${matches.length} indexed file${matches.length === 1 ? "" : "s"} matching "${glob}"\n\n`;
      for (const fp of matches) md += `- \`${fp}\`\n`;
      pi.sendMessage({ customType: "rag", content: md, display: true });
    } finally {
      database.close();
    }
  }

  // ── /rag command ──
  const RAG_SUBCOMMANDS: { value: string; label: string; description: string }[] = [
    { value: "index", label: "index", description: "Index a file or directory" },
    { value: "search", label: "search", description: "Search the index" },
    { value: "find", label: "find", description: "Find indexed files by glob" },
    { value: "status", label: "status", description: "Show index statistics" },
    { value: "rebuild", label: "rebuild", description: "Rebuild entire index" },
    { value: "clear", label: "clear", description: "Clear the index" },
    { value: "exclude", label: "exclude", description: "Manage exclude patterns" },
    { value: "on", label: "on", description: "Enable auto-injection" },
    { value: "off", label: "off", description: "Disable auto-injection" },
  ];

  pi.registerCommand("rag", {
    description: "pi-local-rag: /rag index|search|find|status|rebuild|clear|exclude|on|off",
    getArgumentCompletions: (prefix: string): AutocompleteItem[] | null => {
      const filtered = RAG_SUBCOMMANDS
        .filter((s) => s.value.startsWith(prefix))
        .map((s) => ({ value: s.value, label: s.label, description: s.description }));
      return filtered.length > 0 ? filtered : null;
    },
    handler: async (args, ctx) => {
      const parts = (args || "").trim().split(/\s+/);
      const cmd = parts[0] || "status";

      switch (cmd) {
        case "index":
          await cmdIndex(parts[1] || ".", ctx);
          break;
        case "search":
          await cmdSearch(parts.slice(1).join(" "), ctx);
          break;
        case "exclude":
          cmdExclude(parts.slice(1).join(" ").trim(), ctx);
          break;
        case "find":
          cmdFind(parts.slice(1).join(" ").trim(), ctx);
          break;
        case "on":
        case "off":
          cmdToggle(cmd, ctx);
          break;
        case "rebuild":
          await cmdRebuild(ctx);
          break;
        case "clear":
          cmdClear(ctx);
          break;
        default:
          cmdStatus(ctx);
      }
    },
  });

  // ── Tools ──

  pi.registerTool({
    name: "rag_index",
    label: "RAG index",
    description: "Index a file or directory into the local pi-local-rag pipeline. Chunks text files (including PDF and DOCX), generates embeddings, stores for hybrid BM25+vector search.",
    parameters: Type.Object({
      path: Type.String({ description: "File or directory path to index" }),
    }),
    execute: async (_toolCallId, params) => {
      if (!existsSync(params.path)) return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }], details: undefined };
      getRagDir({ createIfMissing: true });
      const config = loadConfig();
      const absPath = resolve(params.path);
      if (!config.trackedPaths.includes(absPath)) {
        config.trackedPaths.push(absPath);
        saveConfig(config);
      }
      const files = collectFiles(absPath, config.excludePatterns);
      if (!files.length) return { content: [{ type: "text" as const, text: `No indexable files found in: ${params.path}` }], details: undefined };
      const database = openDb();
      try {
        const result = await indexFiles(files, {}, database);
        process.stderr.write(`\n`);
        return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` }], details: undefined };
      } finally {
        database.close();
      }
    },
  });

  pi.registerTool({
    name: "rag_query",
    label: "RAG query",
    description: "Search the local pi-local-rag index using hybrid BM25+vector search. Returns relevant chunks with file paths, line numbers, and relevance scores.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    execute: async (_toolCallId, params) => {
      const database = openDb();
      try {
        const stats = getIndexStats(database);
        if (stats.totalChunks === 0) return { content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }], details: undefined };
        const config = loadConfig();
        const results = await hybridSearch(params.query, { chunks: [], files: {}, lastBuild: stats.lastBuild }, params.limit ?? 10, config.ragAlpha, database);
        if (!results.length) return { content: [{ type: "text" as const, text: `No results for: ${params.query}` }], details: undefined };
        const text = JSON.stringify(results.map(r => ({
          file: r.chunk.file,
          lines: `${r.chunk.lineStart}-${r.chunk.lineEnd}`,
          tokens: r.chunk.tokens,
          scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
          preview: r.chunk.content.slice(0, 300),
        })), null, 2);
        return { content: [{ type: "text" as const, text }], details: undefined };
      } finally {
        database.close();
      }
    },
  });

  pi.registerTool({
    name: "rag_status",
    label: "RAG status",
    description: "Show pi-local-rag index statistics: file count, chunk count, vector coverage, embedding model, RAG config.",
    parameters: Type.Object({}),
    execute: async (_toolCallId) => {
      const database = openDb();
      try {
        const stats = getIndexStats(database);
        const config = loadConfig();
        const text = JSON.stringify({
          files: stats.totalFiles,
          chunks: stats.totalChunks,
          vectorsEmbedded: stats.embeddedCount,
          vectorCoverage: stats.totalChunks ? `${Math.round(stats.embeddedCount / stats.totalChunks * 100)}%` : "0%",
          embeddingModel: stats.embeddingModel ?? "none",
          totalTokens: stats.totalTokens,
          lastBuild: stats.lastBuild || "never",
          ragConfig: config,
          storagePath: getRagDir(),
          storageScope: getRagDir() === GLOBAL_RAG_DIR() ? "global" : "project",
        }, null, 2);
        return { content: [{ type: "text" as const, text }], details: undefined };
      } finally {
        database.close();
      }
    },
  });
}
