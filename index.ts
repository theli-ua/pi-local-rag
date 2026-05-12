/**
 * pi-local-rag — Hybrid RAG Pipeline (BM25 + Vector + Auto-injection)
 *
 * Index local files → chunk → embed → store → retrieve → inject into LLM context.
 * Uses Transformers.js (ONNX) for local embeddings — zero cloud dependency.
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
import type { AutocompleteItem } from "@earendil-works/pi-tui";
import { Type } from "@sinclair/typebox";
import { existsSync, readFileSync, writeFileSync, mkdirSync, readdirSync, statSync, renameSync } from "node:fs";
import { join, extname, basename } from "node:path";
import { homedir } from "node:os";
import { createHash } from "node:crypto";

// ─── Constants ───────────────────────────────────────────────────────────────

const RAG_DIR = join(homedir(), ".pi", "rag");
const LEGACY_DIR = join(homedir(), ".pi", "lens"); // renamed from lens → rag
const INDEX_FILE = join(RAG_DIR, "index.json");
const CONFIG_FILE = join(RAG_DIR, "config.json");

const RST = "\x1b[0m", B = "\x1b[1m", D = "\x1b[2m";
const GREEN = "\x1b[32m", YELLOW = "\x1b[33m", CYAN = "\x1b[36m", RED = "\x1b[31m", MAGENTA = "\x1b[35m";

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const VECTOR_DIM = 384;

const TEXT_EXTS = new Set([
  ".md", ".txt", ".ts", ".js", ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".cs",
  ".css", ".html", ".json", ".yaml", ".yml", ".toml", ".xml", ".csv", ".sh",
  ".sql", ".graphql", ".proto", ".env", ".gitignore", ".dockerfile",
]);

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
}

type RagCommandCtx = Parameters<NonNullable<Parameters<ExtensionAPI["registerCommand"]>[0]["handler"]>>[1];

// ─── Config ──────────────────────────────────────────────────────────────────

function loadConfig(): RagConfig {
  ensureDir();
  if (!existsSync(CONFIG_FILE)) return defaultConfig();
  try {
    return { ...defaultConfig(), ...JSON.parse(readFileSync(CONFIG_FILE, "utf-8")) };
  } catch { return defaultConfig(); }
}

function defaultConfig(): RagConfig {
  return { ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4 };
}

function saveConfig(config: RagConfig) {
  ensureDir();
  writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
}

// ─── Index I/O ───────────────────────────────────────────────────────────────

function ensureDir() {
  if (!existsSync(RAG_DIR)) {
    // Migrate legacy .pi/lens directory to .pi/rag on first run
    if (existsSync(LEGACY_DIR)) {
      try {
        renameSync(LEGACY_DIR, RAG_DIR);
      } catch {
        mkdirSync(RAG_DIR, { recursive: true });
      }
    } else {
      mkdirSync(RAG_DIR, { recursive: true });
    }
  }
}

function loadIndex(): IndexMeta {
  ensureDir();
  if (!existsSync(INDEX_FILE)) return { chunks: [], files: {}, lastBuild: "" };
  try {
    const data = JSON.parse(readFileSync(INDEX_FILE, "utf-8"));
    return {
      chunks: Array.isArray(data.chunks) ? data.chunks : [],
      files: data.files && typeof data.files === "object" ? data.files : {},
      lastBuild: data.lastBuild ?? "",
      embeddingModel: data.embeddingModel,
    };
  } catch { return { chunks: [], files: {}, lastBuild: "" }; }
}

function saveIndex(index: IndexMeta) {
  ensureDir();
  writeFileSync(INDEX_FILE, JSON.stringify(index, null, 2));
}

function sha256(data: string): string {
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

async function embedBatch(texts: string[], onProgress?: (i: number, total: number) => void): Promise<number[][]> {
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    results.push(await embed(texts[i]));
    onProgress?.(i + 1, texts.length);
  }
  return results;
}

// ─── Math ────────────────────────────────────────────────────────────────────

function cosineSimilarity(a: number[], b: number[]): number {
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

function normalize(scores: number[]): number[] {
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;
  if (range === 0) return scores.map(() => 0);
  return scores.map(s => (s - min) / range);
}

// ─── Chunking & File Collection ──────────────────────────────────────────────

function chunkText(text: string, maxLines = 50): { content: string; lineStart: number; lineEnd: number }[] {
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

function collectFiles(dirPath: string): string[] {
  const files: string[] = [];
  function walk(dir: string) {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
          if (!SKIP_DIRS.has(entry.name) && !entry.name.startsWith(".")) walk(join(dir, entry.name));
        } else if (TEXT_EXTS.has(extname(entry.name).toLowerCase())) {
          const fp = join(dir, entry.name);
          try {
            if (statSync(fp).size < 500_000) files.push(fp);
          } catch {}
        }
      }
    } catch {}
  }
  try {
    const stat = statSync(dirPath);
    if (stat.isFile()) {
      if (!TEXT_EXTS.has(extname(dirPath).toLowerCase()) || stat.size >= 500_000) return [];
      return [dirPath];
    }
  } catch { return []; }
  walk(dirPath);
  return files;
}

// ─── Indexing ─────────────────────────────────────────────────────────────────

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

async function indexFiles(
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
      const content = readFileSync(fp, "utf-8");
      const hash = sha256(content);

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

      index.files[fp] = { hash, chunks: rawChunks.length, indexed: new Date().toISOString(), size: content.length, embedded: true };
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

// ─── Search ───────────────────────────────────────────────────────────────────

interface ScoredChunk {
  chunk: Chunk;
  bm25: number;
  vector: number;
  hybrid: number;
}

async function hybridSearch(
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
  ensureDir();

  // ── Auto-inject RAG context before every agent turn ──
  pi.on("before_agent_start", async (event, _ctx) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;

    const index = loadIndex();
    if (!index.chunks.length) return;

    const results = await hybridSearch(event.prompt, index, config.ragTopK, config.ragAlpha);
    const relevant = results.filter(r => r.hybrid >= config.ragScoreThreshold);
    if (!relevant.length) return;

    const context = relevant.map(r =>
      `### ${basename(r.chunk.file)} (lines ${r.chunk.lineStart}-${r.chunk.lineEnd})\n` +
      `\`\`\`\n${r.chunk.content.slice(0, 600)}\n\`\`\``
    ).join("\n\n");

    return {
      systemPrompt: event.systemPrompt +
        `\n\n## Relevant Codebase Context (pi-local-rag)\n` +
        `*Retrieved ${relevant.length} chunks via hybrid search (BM25 + vector)*\n\n` +
        context,
    };
  });

  // ── /rag command helpers ──

  function progressBar(n: number, total: number, width = 24): string {
    const filled = Math.round((n / total) * width);
    return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
  }

  async function cmdIndex(path: string, ctx: RagCommandCtx) {
    if (!existsSync(path)) { ctx.ui.notify(`Path not found: ${path}`, "error"); return; }
    const files = collectFiles(path);
    if (!files.length) { ctx.ui.notify(`No indexable files found in: ${path}`, "warning"); return; }

    ctx.ui.notify(`Found ${files.length} files to index`, "info");

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
    ctx.ui.notify(`Indexed: ${result.indexed} files (${result.chunks} chunks) │ ${result.skipped} unchanged │ ${secs}s`, "success");
  }

  async function cmdSearch(query: string, ctx: RagCommandCtx) {
    if (!query) { pi.sendMessage({ customType: "rag-search", content: "**Usage:** `/rag search <query>`", display: true }); return; }
    const index = loadIndex();
    const config = loadConfig();
    const results = await hybridSearch(query, index, 10, config.ragAlpha);
    if (!results.length) {
      pi.sendMessage({ customType: "rag-search", content: `No results for: \`${query}\``, display: true });
      return;
    }

    const hasVectors = index.chunks.some(c => c.vector);
    let md = `## 🔍 ${results.length} results for "${query}"\n\n`;
    md += `*${hasVectors ? "hybrid BM25+vector" : "BM25 only — run /rag index to add vectors"}*\n\n`;

    for (const r of results) {
      const bar = "█".repeat(Math.round(r.hybrid * 10)) + "░".repeat(10 - Math.round(r.hybrid * 10));
      md += `- **${basename(r.chunk.file)}**:${r.chunk.lineStart}-${r.chunk.lineEnd} \`bm25=${r.bm25.toFixed(2)} vec=${r.vector.toFixed(2)} hybrid=${r.hybrid.toFixed(2)}\` ${bar}\n`;
      const preview = r.chunk.content.split("\n").slice(0, 3).join("\n");
      md += "  ```\n" + preview.slice(0, 200) + "\n  ```\n\n";
    }
    pi.sendMessage({ customType: "rag-search", content: md, display: true });
  }

  function cmdToggle(mode: "on" | "off", ctx: RagCommandCtx) {
    const config = loadConfig();
    config.ragEnabled = mode === "on";
    saveConfig(config);
    ctx.ui.notify(mode === "on" ? "RAG auto-injection enabled" : "RAG auto-injection disabled", mode === "on" ? "success" : "warning");
  }

  async function cmdRebuild(ctx: RagCommandCtx) {
    const index = loadIndex();
    const allFiles = Object.keys(index.files);
    if (!allFiles.length) { ctx.ui.notify("No files in index. Run /rag index <path> first.", "warning"); return; }

    const existingFiles = allFiles.filter(f => existsSync(f));
    const deletedFiles = allFiles.filter(f => !existsSync(f));

    for (const f of deletedFiles) {
      index.chunks = index.chunks.filter(c => c.file !== f);
      delete index.files[f];
    }
    for (const f of existingFiles) { if (index.files[f]) index.files[f].embedded = false; }
    saveIndex(index);

    if (deletedFiles.length) ctx.ui.notify(`Pruned ${deletedFiles.length} deleted files`, "info");
    ctx.ui.notify(`Rebuilding ${existingFiles.length} files...`, "info");

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
    ctx.ui.notify(`Rebuilt: ${result.indexed} re-indexed │ ${result.skipped} unchanged │ ${deletedFiles.length} deleted │ ${result.chunks} chunks │ ${secs}s`, "success");
  }

  function cmdClear(ctx: RagCommandCtx) {
    saveIndex({ chunks: [], files: {}, lastBuild: "" });
    ctx.ui.notify("Index cleared.", "success");
  }

  function cmdStatus(ctx: RagCommandCtx) {
    const index = loadIndex();
    const config = loadConfig();
    const fileCount = Object.keys(index.files).length;
    const totalTokens = index.chunks.reduce((sum, c) => sum + c.tokens, 0);
    const embeddedCount = index.chunks.filter(c => c.vector).length;
    const vectorCoverage = index.chunks.length ? Math.round(embeddedCount / index.chunks.length * 100) : 0;

    let md = `## 🔍 pi-local-rag Status\n\n`;
    md += `| Metric | Value |\n|---|---|\n`;
    md += `| Files indexed | ${fileCount} |\n`;
    md += `| Chunks | ${index.chunks.length} |\n`;
    md += `| Vectors | ${embeddedCount} (${vectorCoverage}% coverage) |\n`;
    md += `| Total tokens | ${totalTokens.toLocaleString()} |\n`;
    md += `| Embedding model | ${index.embeddingModel || "none"} |\n`;
    md += `| Last build | ${index.lastBuild || "never"} |\n`;
    md += `| Storage | \`${RAG_DIR}\` |\n\n`;
    md += `**RAG injection:** ${config.ragEnabled ? "enabled ✅" : "disabled ⚠️"}  \n`;
    md += `\`topK=${config.ragTopK}\`  \`threshold=${config.ragScoreThreshold}\`  \`alpha=${config.ragAlpha}\`\n`;

    if (fileCount) {
      md += `\n### File types\n\n`;
      const byExt: Record<string, number> = {};
      for (const f of Object.keys(index.files)) byExt[extname(f)] = (byExt[extname(f)] || 0) + 1;
      for (const [ext, count] of Object.entries(byExt).sort((a, b) => b[1] - a[1]).slice(0, 8)) {
        md += `- \`${ext}\`: ${count}\n`;
      }
    }
    pi.sendMessage({ customType: "rag-status", content: md, display: true });
  }

  // ── /rag command ──
  const RAG_SUBCOMMANDS: { value: string; label: string; description: string }[] = [
    { value: "index", label: "index", description: "Index a file or directory" },
    { value: "search", label: "search", description: "Search the index" },
    { value: "status", label: "status", description: "Show index statistics" },
    { value: "rebuild", label: "rebuild", description: "Rebuild entire index" },
    { value: "clear", label: "clear", description: "Clear the index" },
    { value: "on", label: "on", description: "Enable auto-injection" },
    { value: "off", label: "off", description: "Disable auto-injection" },
  ];

  pi.registerCommand("rag", {
    description: "pi-local-rag: /rag index|search|status|rebuild|clear|on|off",
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
    description: "Index a file or directory into the local pi-local-rag pipeline. Chunks text files, generates embeddings, stores for hybrid BM25+vector search.",
    parameters: Type.Object({
      path: Type.String({ description: "File or directory path to index" }),
    }),
    execute: async (_toolCallId, params) => {
      if (!existsSync(params.path)) return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }] };
      const files = collectFiles(params.path);
      if (!files.length) return { content: [{ type: "text" as const, text: `No indexable text files found in: ${params.path}` }] };
      const result = await indexFiles(files, {});
      process.stderr.write(`\n`);
      return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` }] };
    },
  });

  pi.registerTool({
    name: "rag_query",
    description: "Search the local pi-local-rag index using hybrid BM25+vector search. Returns relevant chunks with file paths, line numbers, and relevance scores.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query" }),
      limit: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
    }),
    execute: async (_toolCallId, params) => {
      const index = loadIndex();
      if (!index.chunks.length) return { content: [{ type: "text" as const, text: "pi-local-rag index is empty. Run rag_index first." }] };
      const config = loadConfig();
      const results = await hybridSearch(params.query, index, params.limit ?? 10, config.ragAlpha);
      if (!results.length) return { content: [{ type: "text" as const, text: `No results for: ${params.query}` }] };
      const text = JSON.stringify(results.map(r => ({
        file: r.chunk.file,
        lines: `${r.chunk.lineStart}-${r.chunk.lineEnd}`,
        tokens: r.chunk.tokens,
        scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
        preview: r.chunk.content.slice(0, 300),
      })), null, 2);
      return { content: [{ type: "text" as const, text }] };
    },
  });

  pi.registerTool({
    name: "rag_status",
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
        storagePath: RAG_DIR,
      }, null, 2);
      return { content: [{ type: "text" as const, text }] };
    },
  });
}
