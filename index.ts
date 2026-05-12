/**
 * pi-local-rag — Hybrid RAG Pipeline (BM25 + Vector + Auto-injection)
 *
 * Index local files → chunk → embed → store → retrieve → inject into LLM context.
 * Uses Transformers.js (ONNX) for local embeddings — zero cloud dependency.
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
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync, renameSync } from "node:fs";
import { join, extname, basename, resolve, relative, dirname } from "node:path";
import { homedir } from "node:os";
import { createHash } from "node:crypto";
import ignore from "ignore";

// ─── Constants ───────────────────────────────────────────────────────────────

const LEGACY_DIR = join(homedir(), ".pi", "lens"); // renamed from lens → rag
const GLOBAL_RAG_DIR = () => join(homedir(), ".pi", "rag");

const RST = "\x1b[0m", B = "\x1b[1m", D = "\x1b[2m";
const GREEN = "\x1b[32m", YELLOW = "\x1b[33m", CYAN = "\x1b[36m", RED = "\x1b[31m", MAGENTA = "\x1b[35m";

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const VECTOR_DIM = 384;

const TEXT_EXTS = new Set([
  ".md", ".txt", ".ts", ".js", ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h",
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

function indexFile(ragDir: string): string { return join(ragDir, "index.json"); }
function configFile(ragDir: string): string { return join(ragDir, "config.json"); }

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

// ─── Index I/O ───────────────────────────────────────────────────────────────

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

function loadIndex(): IndexMeta {
  const ragDir = getRagDir();
  const idxFile = indexFile(ragDir);
  if (!existsSync(idxFile)) return { chunks: [], files: {}, lastBuild: "" };
  try {
    const data = JSON.parse(readFileSync(idxFile, "utf-8"));
    return {
      chunks: Array.isArray(data.chunks) ? data.chunks : [],
      files: data.files && typeof data.files === "object" ? data.files : {},
      lastBuild: data.lastBuild ?? "",
      embeddingModel: data.embeddingModel,
    };
  } catch { return { chunks: [], files: {}, lastBuild: "" }; }
}

function saveIndex(index: IndexMeta) {
  const ragDir = getRagDir();
  writeFileSync(indexFile(ragDir), JSON.stringify(index, null, 2));
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

export async function embedBatch(texts: string[], onProgress?: (i: number, total: number) => void): Promise<number[][]> {
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    results.push(await embed(texts[i]));
    onProgress?.(i + 1, texts.length);
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
export async function extractText(fp: string): Promise<{ text: string; hash: string; size: number }> {
  const ext = extname(fp).toLowerCase();
  if (ext === ".pdf") {
    const buf = readFileSync(fp);
    const { default: pdf } = await import("pdf-parse/lib/pdf-parse.js");
    const data = await pdf(buf);
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
  process.stderr.write(`\r\x1b[2K${msg}`);
}

export async function indexFiles(
  paths: string[],
  progress?: ProgressCallbacks
): Promise<{ indexed: number; chunks: number; skipped: number; durationMs: number }> {
  const index = loadIndex();
  let indexed = 0, chunked = 0, skipped = 0;
  const startMs = Date.now();
  const total = paths.length;

  for (let i = 0; i < paths.length; i++) {
    const fp = paths[i];
    const pct = Math.round(((i + 1) / total) * 100);
    const name = basename(fp);

    try {
      const { text: content, hash, size } = await extractText(fp);

      if (index.files[fp]?.hash === hash && index.files[fp]?.embedded) {
        skipped++;
        stderrProgress(`[${i + 1}/${total}] ${pct}% skipped ${name}`);
        progress?.onFile?.(i + 1, total, name, skipped);
        await yield_(); // let TUI paint
        continue;
      }

      index.chunks = index.chunks.filter(c => c.file !== fp);
      const rawChunks = chunkText(content);

      stderrProgress(`[${i + 1}/${total}] ${pct}% embedding ${name} (${rawChunks.length} chunks)`);
      progress?.onFile?.(i + 1, total, name, skipped);
      await yield_();

      const vectors = await embedBatch(
        rawChunks.map(c => c.content),
        (ci) => {
          stderrProgress(`[${i + 1}/${total}] ${pct}% ${name} — chunk ${ci}/${rawChunks.length}`);
          progress?.onChunk?.(ci, rawChunks.length, name);
        }
      );

      for (let j = 0; j < rawChunks.length; j++) {
        const chunk = rawChunks[j];
        index.chunks.push({
          id: `${sha256(fp)}-${chunk.lineStart}`,
          file: fp,
          content: chunk.content,
          lineStart: chunk.lineStart,
          lineEnd: chunk.lineEnd,
          hash: sha256(chunk.content),
          indexed: new Date().toISOString(),
          tokens: Math.ceil(chunk.content.length / 4),
          vector: vectors[j],
        });
        chunked++;
      }

      index.files[fp] = { hash, chunks: rawChunks.length, indexed: new Date().toISOString(), size, embedded: true };
      indexed++;
    } catch { skipped++; }
  }

  // Clear stderr progress line
  process.stderr.write(`\r\x1b[2K`);

  progress?.onSave?.();
  index.lastBuild = new Date().toISOString();
  index.embeddingModel = EMBEDDING_MODEL;
  saveIndex(index);
  return { indexed, chunks: chunked, skipped, durationMs: Date.now() - startMs };
}

// ─── Staleness ───────────────────────────────────────────────────────────────

export function isIndexStale(index: IndexMeta, maxAgeMs = 24 * 60 * 60 * 1000): boolean {
  if (!index.lastBuild) return false;
  return Date.now() - new Date(index.lastBuild).getTime() > maxAgeMs;
}

// ─── Search ───────────────────────────────────────────────────────────────────

interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
}

export async function hybridSearch(
  query: string,
  index: IndexMeta,
  limit = 10,
  alpha = 0.4
): Promise<ScoredChunk[]> {
  if (!index.chunks.length) return [];

  // ── BM25 ──
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 1);
  const queryLower = query.toLowerCase();
  const idfMap = new Map<string, number>();
  for (const term of terms) {
    const docsWithTerm = index.chunks.filter(c => c.content.toLowerCase().includes(term)).length;
    idfMap.set(term, Math.log(1 + index.chunks.length / (1 + docsWithTerm)));
  }

  const bm25Raw = index.chunks.map(chunk => {
    const lower = chunk.content.toLowerCase();
    let score = 0;
    for (const term of terms) {
      const count = (lower.match(new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g")) || []).length;
      if (count > 0) score += Math.log(1 + count) * (idfMap.get(term) ?? 0);
    }
    if (lower.includes(queryLower)) score *= 2;
    if (chunk.file.toLowerCase().includes(terms[0] ?? "")) score *= 1.5;
    return score;
  });

  const bm25Norm = normalize(bm25Raw);

  // ── Vector ──
  const chunksWithVectors = index.chunks.filter(c => c.vector && c.vector.length === VECTOR_DIM);
  const hasVectors = chunksWithVectors.length > 0;

  let vectorNorm: number[] = new Array(index.chunks.length).fill(0);

  if (hasVectors) {
    const queryVec = await embed(query);
    const vectorRaw = index.chunks.map(chunk =>
      chunk.vector && chunk.vector.length === VECTOR_DIM
        ? cosineSimilarity(queryVec, chunk.vector)
        : 0
    );
    vectorNorm = normalize(vectorRaw);
  }

  // ── Hybrid ──
  const scored: ScoredChunk[] = index.chunks.map((chunk, i) => ({
    chunk,
    bm25: bm25Norm[i],
    vector: vectorNorm[i],
    hybrid: hasVectors
      ? alpha * bm25Norm[i] + (1 - alpha) * vectorNorm[i]
      : bm25Norm[i],
  }));

  return scored
    .filter(s => s.hybrid > 0)
    .sort((a, b) => b.hybrid - a.hybrid)
    .slice(0, limit);
}

// ─── Extension ────────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // ── Auto-inject RAG context before every agent turn ──
  pi.on("before_agent_start", async (event, _ctx) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;

    let index = loadIndex();
    if (!index.chunks.length) return;

    if (isIndexStale(index)) {
      // Re-walk tracked paths so new files (and files of newly-supported
      // extensions, e.g. PDF/DOCX added in a later version) are picked up.
      // For pre-trackedPaths indexes, fall back to refreshing only known files.
      const files = config.trackedPaths.length
        ? collectFromTracked(config)
        : Object.keys(index.files).filter(f => existsSync(f));
      if (files.length) {
        stderrProgress(`[rag] Index stale, refreshing ${files.length} files…`);
        await indexFiles(files);
        process.stderr.write(`\r\x1b[2K`);
        index = loadIndex();
      }
    }

    const results = await hybridSearch(event.prompt, index, config.ragTopK, config.ragAlpha);
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
  });

  // ── /rag command ──
  pi.registerCommand("rag", {
    description: "pi-local-rag: /rag index|search|status|rebuild|clear|exclude|on|off",
    handler: async (args, ctx) => {
      const reply = (text: string) => pi.sendMessage({ customType: "rag", content: text, display: true });
      const parts = (args || "").trim().split(/\s+/);
      const cmd = parts[0] || "status";

      // ── index ──
      if (cmd === "index") {
        const path = parts[1] || ".";
        if (!existsSync(path)) return reply(`${RED}Path not found:${RST} ${path}`);
        getRagDir({ createIfMissing: true });
        const config = loadConfig();
        const absPath = resolve(path);
        if (!config.trackedPaths.includes(absPath)) {
          config.trackedPaths.push(absPath);
          saveConfig(config);
        }
        const files = collectFiles(absPath, config.excludePatterns);
        if (!files.length) return reply(`${YELLOW}No indexable files found in:${RST} ${path}`);

        const total = files.length;
        ctx.ui.notify(`Found ${total} files to index`, "info");

        function progressBar(n: number, total: number, width = 24): string {
          const filled = Math.round((n / total) * width);
          return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
        }

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
        });

        ctx.ui.setStatus("rag", undefined);
        ctx.ui.setWidget("rag", undefined);

        const secs = (result.durationMs / 1000).toFixed(1);
        const ragDir = getRagDir();
        const scope = ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
        return reply(`${GREEN}✅ Indexed:${RST} ${result.indexed} files (${result.chunks} chunks) │ ${result.skipped} unchanged │ ${secs}s\n` +
          `${D}Now tracking ${config.trackedPaths.length} path(s) │ Model: ${EMBEDDING_MODEL} │ Storage: ${ragDir} (${scope})${RST}`);
      }

      // ── search ──
      if (cmd === "search") {
        const query = parts.slice(1).join(" ");
        if (!query) return reply(`${YELLOW}Usage:${RST} /rag search <query>`);
        const index = loadIndex();
        const config = loadConfig();
        const results = await hybridSearch(query, index, 10, config.ragAlpha);
        if (!results.length) return reply(`${YELLOW}No results for:${RST} ${query}`);

        const hasVectors = index.chunks.some(c => c.vector);
        let out = `${B}${CYAN}🔍 ${results.length} results for "${query}"${RST}`;
        out += ` ${D}(${hasVectors ? "hybrid BM25+vector" : "BM25 only — run /rag index to add vectors"})${RST}\n\n`;

        for (const r of results) {
          const bar = "█".repeat(Math.round(r.hybrid * 10)) + "░".repeat(10 - Math.round(r.hybrid * 10));
          out += `${GREEN}${basename(r.chunk.file)}${RST}:${r.chunk.lineStart}-${r.chunk.lineEnd} `;
          out += `${D}bm25=${r.bm25.toFixed(2)} vec=${r.vector.toFixed(2)} hybrid=${r.hybrid.toFixed(2)}${RST} ${CYAN}${bar}${RST}\n`;
          const preview = r.chunk.content.split("\n").slice(0, 3).join("\n");
          out += `${D}${preview.slice(0, 200)}${RST}\n\n`;
        }
        return reply(out);
      }

      // ── on/off toggle ──
      if (cmd === "on" || cmd === "off") {
        const config = loadConfig();
        config.ragEnabled = cmd === "on";
        saveConfig(config);
        return reply(cmd === "on"
          ? `${GREEN}✅ RAG auto-injection enabled${RST}`
          : `${YELLOW}RAG auto-injection disabled${RST}`);
      }

      // ── rebuild ──
      if (cmd === "rebuild") {
        const index = loadIndex();
        const config = loadConfig();
        const indexedFiles = Object.keys(index.files);
        const trackedFiles = collectFromTracked(config);

        // Union of currently-indexed files and files discovered by walking tracked paths.
        const targetSet = new Set<string>([...trackedFiles]);
        for (const f of indexedFiles) {
          if (existsSync(f) && !isExcludedByConfig(f, config.trackedPaths, config.excludePatterns)) {
            targetSet.add(f);
          }
        }
        const targetFiles = [...targetSet];

        if (!targetFiles.length && !indexedFiles.length) {
          return reply(`${YELLOW}No files to rebuild. Run /rag index <path> first.${RST}`);
        }

        // Files in the index but no longer present (deleted, excluded, or untracked).
        const droppedFiles = indexedFiles.filter(f => !targetSet.has(f));
        for (const f of droppedFiles) {
          index.chunks = index.chunks.filter(c => c.file !== f);
          delete index.files[f];
        }

        // Force re-embed all target files (treat new ones as not yet embedded).
        for (const f of targetFiles) {
          if (index.files[f]) index.files[f].embedded = false;
        }
        saveIndex(index);

        const newFiles = targetFiles.filter(f => !indexedFiles.includes(f));
        if (droppedFiles.length) ctx.ui.notify(`Pruned ${droppedFiles.length} files (deleted/excluded)`, "info");
        if (newFiles.length) ctx.ui.notify(`Discovered ${newFiles.length} new files`, "info");
        ctx.ui.notify(`Rebuilding ${targetFiles.length} files...`, "info");

        const existingFiles = targetFiles;
        const deletedFiles = droppedFiles;

        function progressBar(n: number, total: number, width = 24): string {
          const filled = Math.round((n / total) * width);
          return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
        }

        const result = await indexFiles(existingFiles, {
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
        });

        ctx.ui.setStatus("rag", undefined);
        ctx.ui.setWidget("rag", undefined);

        const secs = (result.durationMs / 1000).toFixed(1);
        return reply(`${GREEN}✅ Rebuilt:${RST} ${result.indexed} re-indexed │ ${result.skipped} unchanged │ ${deletedFiles.length} deleted │ ${result.chunks} chunks │ ${secs}s`);
      }

      // ── exclude ──
      if (cmd === "exclude") {
        const config = loadConfig();
        const expr = parts.slice(1).join(" ").trim();

        if (!expr) {
          if (!config.excludePatterns.length) return reply(`${YELLOW}No exclude patterns set.${RST}\n${D}Add one with: /rag exclude <pattern>${RST}`);
          let out = `${B}${CYAN}Exclude patterns (${config.excludePatterns.length}):${RST}\n`;
          for (const p of config.excludePatterns) out += `  ${p}\n`;
          return reply(out);
        }

        if (expr.startsWith("-")) {
          const target = expr.slice(1);
          const before = config.excludePatterns.length;
          config.excludePatterns = config.excludePatterns.filter(p => p !== target);
          if (config.excludePatterns.length === before) {
            return reply(`${YELLOW}Pattern not found:${RST} ${target}`);
          }
          saveConfig(config);
          return reply(`${GREEN}✅ Removed exclude:${RST} ${target}\n${D}${config.excludePatterns.length} pattern(s) remain. Run /rag rebuild to re-apply.${RST}`);
        }

        if (config.excludePatterns.includes(expr)) {
          return reply(`${YELLOW}Already excluded:${RST} ${expr}`);
        }
        config.excludePatterns.push(expr);
        saveConfig(config);
        return reply(`${GREEN}✅ Added exclude:${RST} ${expr}\n${D}${config.excludePatterns.length} pattern(s) total. Run /rag rebuild to re-apply.${RST}`);
      }

      // ── clear ──
      if (cmd === "clear") {
        saveIndex({ chunks: [], files: {}, lastBuild: "" });
        return reply(`${GREEN}✅ Index cleared.${RST}`);
      }

      // ── status (default) ──
      const index = loadIndex();
      const config = loadConfig();
      const fileCount = Object.keys(index.files).length;
      const totalTokens = index.chunks.reduce((sum, c) => sum + c.tokens, 0);
      const embeddedCount = index.chunks.filter(c => c.vector).length;
      const vectorCoverage = index.chunks.length ? Math.round(embeddedCount / index.chunks.length * 100) : 0;

      let out = `${B}${CYAN}🔍 pi-local-rag Status${RST}\n\n`;
      out += `  Files indexed:   ${GREEN}${fileCount}${RST}\n`;
      out += `  Chunks:          ${GREEN}${index.chunks.length}${RST}\n`;
      out += `  Vectors:         ${GREEN}${embeddedCount}${RST} ${D}(${vectorCoverage}% coverage)${RST}\n`;
      out += `  Total tokens:    ${GREEN}${totalTokens.toLocaleString()}${RST}\n`;
      out += `  Embedding model: ${D}${index.embeddingModel || "none"}${RST}\n`;
      out += `  Last build:      ${index.lastBuild || "never"}\n`;
      const ragDir = getRagDir();
      const scope = ragDir === GLOBAL_RAG_DIR() ? "global" : "project";
      out += `  Storage:         ${D}${ragDir}${RST} ${D}(${scope})${RST}\n\n`;
      out += `  RAG injection:   ${config.ragEnabled ? `${GREEN}enabled${RST}` : `${YELLOW}disabled${RST}`}`;
      out += `  topK=${config.ragTopK}  threshold=${config.ragScoreThreshold}  alpha=${config.ragAlpha}\n`;

      if (fileCount) {
        out += `\n  ${B}File types:${RST}\n`;
        const byExt: Record<string, number> = {};
        for (const f of Object.keys(index.files)) byExt[extname(f)] = (byExt[extname(f)] || 0) + 1;
        for (const [ext, count] of Object.entries(byExt).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
          out += `    ${ext}: ${count}\n`;
        }
      }

      out += `\n  ${B}Tracked paths:${RST}\n`;
      if (config.trackedPaths.length) {
        for (const p of config.trackedPaths) out += `    ${p}\n`;
      } else {
        out += `    ${D}(none — run /rag index <path> to track)${RST}\n`;
      }

      out += `\n  ${B}Exclude patterns:${RST}\n`;
      if (config.excludePatterns.length) {
        for (const p of config.excludePatterns) out += `    ${p}\n`;
      } else {
        out += `    ${D}(none — add with /rag exclude <pattern>)${RST}\n`;
      }

      reply(out);
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
      const result = await indexFiles(files, {});
      process.stderr.write(`\n`);
      return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` }], details: undefined };
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
      const index = loadIndex();
      if (!index.chunks.length) return { content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }], details: undefined };
      const config = loadConfig();
      const results = await hybridSearch(params.query, index, params.limit ?? 10, config.ragAlpha);
      if (!results.length) return { content: [{ type: "text" as const, text: `No results for: ${params.query}` }], details: undefined };
      const text = JSON.stringify(results.map(r => ({
        file: r.chunk.file,
        lines: `${r.chunk.lineStart}-${r.chunk.lineEnd}`,
        tokens: r.chunk.tokens,
        scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
        preview: r.chunk.content.slice(0, 300),
      })), null, 2);
      return { content: [{ type: "text" as const, text }], details: undefined };
    },
  });

  pi.registerTool({
    name: "rag_status",
    label: "RAG status",
    description: "Show pi-local-rag index statistics: file count, chunk count, vector coverage, embedding model, RAG config.",
    parameters: Type.Object({}),
    execute: async (_toolCallId) => {
      const index = loadIndex();
      const config = loadConfig();
      const embeddedCount = index.chunks.filter(c => c.vector).length;
      const text = JSON.stringify({
        files: Object.keys(index.files).length,
        chunks: index.chunks.length,
        vectorsEmbedded: embeddedCount,
        vectorCoverage: index.chunks.length ? `${Math.round(embeddedCount / index.chunks.length * 100)}%` : "0%",
        embeddingModel: index.embeddingModel ?? "none",
        totalTokens: index.chunks.reduce((s, c) => s + c.tokens, 0),
        lastBuild: index.lastBuild || "never",
        ragConfig: config,
        storagePath: getRagDir(),
        storageScope: getRagDir() === GLOBAL_RAG_DIR() ? "global" : "project",
      }, null, 2);
      return { content: [{ type: "text" as const, text }], details: undefined };
    },
  });
}
