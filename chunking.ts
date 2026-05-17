import { readFileSync, existsSync, statSync, readdirSync, promises as fsPromises } from "node:fs";
import { extname, basename, join, relative } from "node:path";
import ignore from "ignore";
import { createHash } from "node:crypto";
import { RagConfig } from "./config.js";
import { TEXT_EXTS, BINARY_DOC_EXTS, TEXT_MAX_BYTES, BINARY_DOC_MAX_BYTES, SKIP_DIRS } from "./constants.js";

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

export function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex").slice(0, 12);
}

// ─── Chunking ─────────────────────────────────────────────────────────────────

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

// ─── File collection ──────────────────────────────────────────────────────────

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

export async function collectFilesAsync(dirPath: string, excludePatterns: string[] = []): Promise<string[]> {
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

  async function walk(dir: string, root: string): Promise<void> {
    let entries: import("node:fs").Dirent[];
    try {
      entries = await fsPromises.readdir(dir, { withFileTypes: true });
    } catch { return; }
    for (const entry of entries) {
      const fp = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        if (isExcluded(fp, root)) continue;
        await walk(fp, root);
      } else {
        const ext = extname(entry.name).toLowerCase();
        if (!TEXT_EXTS.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
        if (isExcluded(fp, root)) continue;
        try {
          const st = await fsPromises.stat(fp);
          if (acceptable(fp, st.size)) files.push(fp);
        } catch {}
      }
    }
    // Yield periodically so the event loop can process UI updates.
    await yield_();
  }

  try {
    const st = await fsPromises.stat(dirPath);
    if (st.isFile()) {
      if (!acceptable(dirPath, st.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch { return []; }

  await walk(dirPath, dirPath);
  return files;
}

export async function collectFromTrackedAsync(cfg: RagConfig): Promise<string[]> {
  const out = new Set<string>();
  for (const p of cfg.trackedPaths) {
    if (!existsSync(p)) continue;
    for (const f of await collectFilesAsync(p, cfg.excludePatterns)) out.add(f);
  }
  return [...out];
}

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

// ─── Text extraction ─────────────────────────────────────────────────────────

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
  if (ext === ".html") {
    const { default: TurndownService } = await import("turndown");
    const raw = readFileSync(fp, "utf-8");
    const td = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced",
      blankReplacement: (_content: unknown, node: Node) => (node as HTMLElement).tagName === "BR" ? "\n" : "",
    });
    // Remove script/style elements before conversion
    td.remove(["script", "style"]);
    // Strip navigation/footer elements that add noise
    td.remove(["nav", "footer"]);
    const text = td.turndown(raw);
    return { text, hash: sha256(raw), size: raw.length };
  }
  const text = readFileSync(fp, "utf-8");
  return { text, hash: sha256(text), size: text.length };
}
