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

// ─── Re-exports (for tests) ──────────────────────────────────────────────────
export { isIndexStale } from "./indexing.js";
export { getRagDir } from "./store.js";
export { loadConfig, saveConfig, defaultConfig } from "./config.js";
export { openDb, getIndexStats, initSchema } from "./db.js";
export { embedBatch } from "./embed.js";
export { hybridSearch } from "./search.js";
export { indexFiles } from "./indexing.js";
export { chunkText, collectFiles, collectFromTracked, isExcludedByConfig, extractText, sha256 } from "./chunking.js";
export { cosineSimilarity, normalize } from "./search.js";

// ─── Imports ─────────────────────────────────────────────────────────────────
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { AutocompleteItem } from "@earendil-works/pi-tui";
import { Type } from "@sinclair/typebox";
import { existsSync } from "node:fs";
import { extname, basename, resolve, relative } from "node:path";
import ignore from "ignore";

import { GLOBAL_RAG_DIR, getRagDir } from "./store.js";
import { loadConfig, saveConfig } from "./config.js";
import { openDb, getIndexStats, loadIndex } from "./db.js";
import { indexFiles, isIndexStale } from "./indexing.js";
import { hybridSearch } from "./search.js";
import { collectFiles, collectFromTracked, isExcludedByConfig } from "./chunking.js";

const RST = "\x1b[0m", B = "\x1b[1m", D = "\x1b[2m";
const GREEN = "\x1b[32m", CYAN = "\x1b[36m";

type RagCommandCtx = Parameters<NonNullable<Parameters<ExtensionAPI["registerCommand"]>[0]["handler"]>>[1];

let _suppressStderr = false;

function progressBar(n: number, total: number, width = 24): string {
  const filled = Math.round((n / total) * width);
  return CYAN + "█".repeat(filled) + D + "░".repeat(width - filled) + RST;
}

function stderrProgress(msg: string) {
  if (_suppressStderr) return;
  process.stderr.write(`\r\x1b[2K${msg}`);
}

export default function (pi: ExtensionAPI) {
  // ── Auto-inject RAG context ──────────────────────────────────────────────
  let lastStaleCheckMs = 0;
  const STALE_CHECK_INTERVAL_MS = 60 * 60 * 1000; // 1 hour

  pi.on("before_agent_start", async (event, _ctx) => {
    const config = loadConfig();
    if (!config.ragEnabled) return;

    const database = openDb();
    try {
      const stats = getIndexStats(database);
      if (stats.totalChunks === 0) return;

      const indexMeta = { chunks: [], files: {}, lastBuild: stats.lastBuild, embeddingModel: stats.embeddingModel };
      const now = Date.now();
      if (isIndexStale(indexMeta) && now - lastStaleCheckMs > STALE_CHECK_INTERVAL_MS) {
        lastStaleCheckMs = now;
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

  // ── /rag subcommands ─────────────────────────────────────────────────────

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
        onSave() { ctx.ui.setStatus("rag", `■ Saving index...`); },
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

      const droppedFiles = [...indexedFileSet].filter(f => !targetSet.has(f));
      for (const f of droppedFiles) {
        database.prepare("DELETE FROM chunks_vec WHERE rowid IN (SELECT rowid FROM chunks WHERE file_path = ?)").run(f);
        database.prepare("DELETE FROM chunks WHERE file_path = ?").run(f);
        database.prepare("DELETE FROM files WHERE path = ?").run(f);
      }
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
        onSave() { ctx.ui.setStatus("rag", `■ Saving index...`); },
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
      database.exec(`DELETE FROM chunks_vec; DELETE FROM chunks; DELETE FROM files; DELETE FROM metadata;`);
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
    if (!glob) { ctx.ui.notify("Usage: /rag find <glob>   e.g. *.html, page*, foo.js, src/*.ts", "warning"); return; }
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
      if (!matches.length) { ctx.ui.notify(`No indexed files match: ${glob}`, "warning"); return; }
      let md = `## 🔍 ${matches.length} indexed file${matches.length === 1 ? "" : "s"} matching "${glob}"\n\n`;
      for (const fp of matches) md += `- \`${fp}\`\n`;
      pi.sendMessage({ customType: "rag", content: md, display: true });
    } finally {
      database.close();
    }
  }

  // ── /rag command ─────────────────────────────────────────────────────────
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
        case "index": await cmdIndex(parts[1] || ".", ctx); break;
        case "search": await cmdSearch(parts.slice(1).join(" "), ctx); break;
        case "exclude": cmdExclude(parts.slice(1).join(" ").trim(), ctx); break;
        case "find": cmdFind(parts.slice(1).join(" ").trim(), ctx); break;
        case "on": case "off": cmdToggle(cmd, ctx); break;
        case "rebuild": await cmdRebuild(ctx); break;
        case "clear": cmdClear(ctx); break;
        default: cmdStatus(ctx);
      }
    },
  });

  // ── Tools ────────────────────────────────────────────────────────────────

  pi.registerTool({
    name: "rag_index",
    label: "RAG index",
    description: "Index a file or directory into the local pi-local-rag pipeline. Chunks text files (including PDF and DOCX), generates embeddings, stores for hybrid BM25+vector search.",
    parameters: Type.Object({ path: Type.String({ description: "File or directory path to index" }) }),
    execute: async (_toolCallId, params) => {
      if (!existsSync(params.path)) return { content: [{ type: "text" as const, text: `Path not found: ${params.path}` }], details: undefined };
      getRagDir({ createIfMissing: true });
      const config = loadConfig();
      const absPath = resolve(params.path);
      if (!config.trackedPaths.includes(absPath)) { config.trackedPaths.push(absPath); saveConfig(config); }
      const files = collectFiles(absPath, config.excludePatterns);
      if (!files.length) return { content: [{ type: "text" as const, text: `No indexable files found in: ${params.path}` }], details: undefined };
      const database = openDb();
      try {
        const result = await indexFiles(files, {}, database);
        process.stderr.write(`\n`);
        return { content: [{ type: "text" as const, text: `Indexed ${result.indexed} files (${result.chunks} chunks, embeddings generated). ${result.skipped} unchanged. ${(result.durationMs / 1000).toFixed(1)}s` }], details: undefined };
      } finally { database.close(); }
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
          file: r.chunk.file, lines: `${r.chunk.lineStart}-${r.chunk.lineEnd}`, tokens: r.chunk.tokens,
          scores: { bm25: r.bm25.toFixed(3), vector: r.vector.toFixed(3), hybrid: r.hybrid.toFixed(3) },
          preview: r.chunk.content.slice(0, 300),
        })), null, 2);
        return { content: [{ type: "text" as const, text }], details: undefined };
      } finally { database.close(); }
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
          files: stats.totalFiles, chunks: stats.totalChunks,
          vectorsEmbedded: stats.embeddedCount,
          vectorCoverage: stats.totalChunks ? `${Math.round(stats.embeddedCount / stats.totalChunks * 100)}%` : "0%",
          embeddingModel: stats.embeddingModel ?? "none",
          totalTokens: stats.totalTokens, lastBuild: stats.lastBuild || "never",
          ragConfig: config, storagePath: getRagDir(),
          storageScope: getRagDir() === GLOBAL_RAG_DIR() ? "global" : "project",
        }, null, 2);
        return { content: [{ type: "text" as const, text }], details: undefined };
      } finally { database.close(); }
    },
  });
}
