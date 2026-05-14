import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, writeFileSync, rmSync, mkdtempSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";

// Mock transformers to avoid real ONNX model downloads
vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async () => ({
      data: new Float32Array(384).fill(0.1),
    }))
  ),
}));

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, "fixtures");
const SAMPLE_PDF = readFileSync(join(FIXTURES_DIR, "sample.pdf"));

import {
  chunkText,
  cosineSimilarity,
  normalize,
  sha256,
  collectFiles,
  collectFromTracked,
  isExcludedByConfig,
  hybridSearch,
  defaultConfig,
  extractText,
  initSchema,
} from "../index.ts";

async function buildMinimalDocx(text: string): Promise<Buffer> {
  const { default: JSZip } = await import("jszip");
  const zip = new JSZip();
  zip.file(
    "[Content_Types].xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>`,
  );
  zip.folder("_rels")!.file(
    ".rels",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>`,
  );
  zip.folder("word")!.file(
    "document.xml",
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>${text}</w:t></w:r></w:p>
  </w:body>
</w:document>`,
  );
  return await zip.generateAsync({ type: "nodebuffer" });
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Create an in-memory SQLite DB with the RAG schema, pre-populated with chunks. */
function createTestDb(chunks: Array<{
  id?: string; file?: string; content: string; lineStart?: number; lineEnd?: number;
  vector?: number[];
}>): Database.Database {
  const db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  loadVec(db);
  initSchema(db);

  const insChunk = db.prepare(`
    INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insVec = db.prepare(
    "INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)",
  );
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    const result = insChunk.run(
      c.id ?? `chunk-${i}`,
      c.file ?? "/src/file.ts",
      c.content,
      c.lineStart ?? 1,
      c.lineEnd ?? 10,
      sha256(c.content),
      new Date().toISOString(),
      Math.ceil(c.content.length / 4),
    );
    if (c.vector) {
      const f = new Float32Array(c.vector);
      insVec.run(Number(result.lastInsertRowid), Buffer.from(f.buffer, f.byteOffset, f.byteLength));
    }
  }

  return db;
}

// ─── sha256 ──────────────────────────────────────────────────────────────────

describe("sha256", () => {
  it("returns a 12-char hex string", () => {
    const h = sha256("hello");
    expect(h).toMatch(/^[0-9a-f]{12}$/);
  });

  it("is deterministic", () => {
    expect(sha256("test")).toBe(sha256("test"));
  });

  it("produces different hashes for different inputs", () => {
    expect(sha256("foo")).not.toBe(sha256("bar"));
  });

  it("handles empty string", () => {
    const h = sha256("");
    expect(h).toMatch(/^[0-9a-f]{12}$/);
  });
});

// ─── cosineSimilarity ────────────────────────────────────────────────────────

describe("cosineSimilarity", () => {
  it("identical unit vectors → 1", () => {
    const v = [1, 0, 0];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1);
  });

  it("orthogonal vectors → 0", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it("opposite unit vectors → -1", () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  it("zero vector → 0", () => {
    expect(cosineSimilarity([0, 0], [1, 0])).toBe(0);
  });

  it("mismatched lengths → 0", () => {
    expect(cosineSimilarity([1, 2], [1, 2, 3])).toBe(0);
  });

  it("known diagonal vectors are equal similarity", () => {
    const a = [1, 1, 0];
    const b = [1, 1, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(1);
  });
});

// ─── normalize ───────────────────────────────────────────────────────────────

describe("normalize", () => {
  it("single element → 0", () => {
    expect(normalize([5])).toEqual([0]);
  });

  it("all equal → all zeros", () => {
    expect(normalize([3, 3, 3])).toEqual([0, 0, 0]);
  });

  it("[0, 1] stays [0, 1]", () => {
    const result = normalize([0, 1]);
    expect(result[0]).toBeCloseTo(0);
    expect(result[1]).toBeCloseTo(1);
  });

  it("[1, 3] maps to [0, 1]", () => {
    const result = normalize([1, 3]);
    expect(result[0]).toBeCloseTo(0);
    expect(result[1]).toBeCloseTo(1);
  });

  it("middle value maps to 0.5", () => {
    const result = normalize([0, 1, 2]);
    expect(result[1]).toBeCloseTo(0.5);
  });

  it("preserves relative order", () => {
    const [a, b, c] = normalize([10, 30, 20]);
    expect(a).toBeLessThan(c);
    expect(c).toBeLessThan(b);
  });
});

// ─── chunkText ───────────────────────────────────────────────────────────────

describe("chunkText", () => {
  it("empty string → no chunks", () => {
    expect(chunkText("")).toHaveLength(0);
  });

  it("short content (< 20 chars per line) → no chunks", () => {
    expect(chunkText("hi\nok")).toHaveLength(0);
  });

  it("a single block of text → one chunk", () => {
    const text = Array(10).fill("const x = 1; // comment here").join("\n");
    const chunks = chunkText(text);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].lineStart).toBe(1);
    expect(chunks[0].lineEnd).toBe(10);
  });

  it("lineStart and lineEnd are 1-indexed", () => {
    const text = Array(5).fill("function foo() { return 42; }").join("\n");
    const chunks = chunkText(text);
    expect(chunks[0].lineStart).toBe(1);
  });

  it("splits into multiple chunks when text exceeds maxLines", () => {
    const line = "const value = someFunction(param) + anotherCall();";
    const text = Array(60).fill(line).join("\n");
    const chunks = chunkText(text, 50);
    expect(chunks.length).toBeGreaterThanOrEqual(2);
  });

  it("prefers splitting at blank lines", () => {
    const block1 = Array(45).fill("function doSomethingUseful() { return true; }").join("\n");
    const blank = "\n\n";
    const block2 = Array(10).fill("const answer = computeResult(input);").join("\n");
    const text = block1 + blank + block2;
    const chunks = chunkText(text, 50);
    expect(chunks.length).toBeGreaterThanOrEqual(2);
    // second chunk should start after the blank lines
    const lastChunk = chunks[chunks.length - 1];
    expect(lastChunk.lineStart).toBeGreaterThan(45);
  });

  it("chunks contain the correct content", () => {
    const lines = Array(10).fill("export const foo = () => 'bar';");
    const text = lines.join("\n");
    const chunks = chunkText(text);
    expect(chunks[0].content).toBe(text);
  });

  it("respects custom maxLines", () => {
    const line = "const x = computeExpensiveOperation(input, config);";
    const text = Array(10).fill(line).join("\n");
    const chunks = chunkText(text, 5);
    // 10 lines with maxLines=5 → 2 chunks
    expect(chunks.length).toBe(2);
  });
});

// ─── collectFiles ────────────────────────────────────────────────────────────

describe("collectFiles", () => {
  let tmp: string;

  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), "rag-test-"));
  });

  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it("non-existent path → []", () => {
    expect(collectFiles(join(tmp, "does-not-exist"))).toEqual([]);
  });

  it("single .ts file → returns it", () => {
    const fp = join(tmp, "foo.ts");
    writeFileSync(fp, "export const x = 1;");
    expect(collectFiles(fp)).toEqual([fp]);
  });

  it("unsupported extension is excluded", () => {
    writeFileSync(join(tmp, "image.png"), Buffer.alloc(10));
    expect(collectFiles(tmp)).toEqual([]);
  });

  it("collects supported files from a directory", () => {
    writeFileSync(join(tmp, "a.ts"), "x");
    writeFileSync(join(tmp, "b.py"), "x");
    writeFileSync(join(tmp, "c.md"), "x");
    const files = collectFiles(tmp);
    expect(files.length).toBe(3);
  });

  it("skips node_modules directory", () => {
    const nm = join(tmp, "node_modules");
    mkdirSync(nm);
    writeFileSync(join(nm, "lib.js"), "module.exports = {};");
    expect(collectFiles(tmp)).toEqual([]);
  });

  it("skips hidden directories (starting with .)", () => {
    const hidden = join(tmp, ".cache");
    mkdirSync(hidden);
    writeFileSync(join(hidden, "data.ts"), "export {};");
    expect(collectFiles(tmp)).toEqual([]);
  });

  it("skips files >= 500 KB", () => {
    const large = join(tmp, "big.ts");
    writeFileSync(large, Buffer.alloc(500_000)); // exactly at limit → excluded
    expect(collectFiles(tmp)).toEqual([]);
  });

  it("includes files just under 500 KB", () => {
    const fp = join(tmp, "ok.ts");
    writeFileSync(fp, Buffer.alloc(499_999));
    expect(collectFiles(tmp)).toEqual([fp]);
  });

  it("single file with wrong extension → []", () => {
    const fp = join(tmp, "data.bin");
    writeFileSync(fp, "x");
    expect(collectFiles(fp)).toEqual([]);
  });

  it("recurses into subdirectories", () => {
    const sub = join(tmp, "src");
    mkdirSync(sub);
    const fp = join(sub, "util.ts");
    writeFileSync(fp, "export {};");
    const files = collectFiles(tmp);
    expect(files).toContain(fp);
  });

  it("excludePatterns: skips files matching a pattern", () => {
    writeFileSync(join(tmp, "keep.ts"), "x");
    writeFileSync(join(tmp, "drop.ts"), "x");
    const files = collectFiles(tmp, ["drop.ts"]);
    expect(files).toEqual([join(tmp, "keep.ts")]);
  });

  it("excludePatterns: directory glob skips whole subtree", () => {
    const sub = join(tmp, "fixtures");
    mkdirSync(sub);
    writeFileSync(join(sub, "a.ts"), "x");
    writeFileSync(join(sub, "b.ts"), "x");
    writeFileSync(join(tmp, "main.ts"), "x");
    const files = collectFiles(tmp, ["fixtures/"]);
    expect(files).toEqual([join(tmp, "main.ts")]);
  });

  it("excludePatterns: ** glob matches nested files", () => {
    const sub = join(tmp, "deep", "nested");
    mkdirSync(sub, { recursive: true });
    writeFileSync(join(sub, "x.ts"), "x");
    writeFileSync(join(tmp, "main.ts"), "x");
    const files = collectFiles(tmp, ["**/x.ts"]);
    expect(files).toEqual([join(tmp, "main.ts")]);
  });

  it("excludePatterns: empty list is no-op", () => {
    writeFileSync(join(tmp, "a.ts"), "x");
    expect(collectFiles(tmp, [])).toEqual([join(tmp, "a.ts")]);
  });

  it("excludePatterns: applied to single-file input", () => {
    const fp = join(tmp, "secret.ts");
    writeFileSync(fp, "x");
    expect(collectFiles(fp, ["secret.ts"])).toEqual([]);
  });

  it("includes .pdf files", () => {
    const fp = join(tmp, "doc.pdf");
    writeFileSync(fp, SAMPLE_PDF);
    expect(collectFiles(tmp)).toEqual([fp]);
  });

  it("includes .docx files", async () => {
    const fp = join(tmp, "doc.docx");
    writeFileSync(fp, await buildMinimalDocx("Hello"));
    expect(collectFiles(tmp)).toEqual([fp]);
  });

  it("binary docs use the 10 MB size cap, not the 500 KB text cap", () => {
    const fp = join(tmp, "big.pdf");
    writeFileSync(fp, Buffer.alloc(600_000)); // would be rejected as text, accepted as binary doc
    expect(collectFiles(tmp)).toEqual([fp]);
  });

  it("skips binary docs >= 10 MB", () => {
    const fp = join(tmp, "huge.pdf");
    writeFileSync(fp, Buffer.alloc(10_000_000));
    expect(collectFiles(tmp)).toEqual([]);
  });
});

// ─── extractText ─────────────────────────────────────────────────────────────

describe("extractText", () => {
  let tmp: string;

  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), "rag-extract-"));
  });

  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it("reads plain text files as utf-8", async () => {
    const fp = join(tmp, "a.txt");
    writeFileSync(fp, "hello world");
    const { text, hash, size } = await extractText(fp);
    expect(text).toBe("hello world");
    expect(hash).toBe(sha256("hello world"));
    expect(size).toBe(11);
  });

  it("extracts text from a .pdf", async () => {
    const fp = join(tmp, "a.pdf");
    writeFileSync(fp, SAMPLE_PDF);
    const { text, hash, size } = await extractText(fp);
    expect(text).toContain("RagPdfMarker");
    expect(hash).toMatch(/^[0-9a-f]{12}$/);
    expect(size).toBe(SAMPLE_PDF.length);
  });

  it("extracts text from a .docx", async () => {
    const fp = join(tmp, "a.docx");
    writeFileSync(fp, await buildMinimalDocx("RagDocxMarker"));
    const { text } = await extractText(fp);
    expect(text).toContain("RagDocxMarker");
  });

  it("hash is stable across reads of the same binary file (skip-on-rebuild)", async () => {
    const fp = join(tmp, "stable.pdf");
    writeFileSync(fp, SAMPLE_PDF);
    const a = await extractText(fp);
    const b = await extractText(fp);
    expect(a.hash).toBe(b.hash);
  });
});

// ─── collectFromTracked ──────────────────────────────────────────────────────

describe("collectFromTracked", () => {
  let tmpA: string;
  let tmpB: string;

  beforeEach(() => {
    tmpA = mkdtempSync(join(tmpdir(), "rag-tracked-a-"));
    tmpB = mkdtempSync(join(tmpdir(), "rag-tracked-b-"));
  });

  afterEach(() => {
    rmSync(tmpA, { recursive: true, force: true });
    rmSync(tmpB, { recursive: true, force: true });
  });

  it("walks every tracked path and unions results", () => {
    writeFileSync(join(tmpA, "a.ts"), "x");
    writeFileSync(join(tmpB, "b.ts"), "x");
    const cfg = { ...defaultConfig(), trackedPaths: [tmpA, tmpB] };
    const files = collectFromTracked(cfg);
    expect(files).toContain(join(tmpA, "a.ts"));
    expect(files).toContain(join(tmpB, "b.ts"));
    expect(files.length).toBe(2);
  });

  it("dedupes when tracked paths overlap", () => {
    const sub = join(tmpA, "shared");
    mkdirSync(sub);
    writeFileSync(join(sub, "x.ts"), "x");
    const cfg = { ...defaultConfig(), trackedPaths: [tmpA, sub] };
    const files = collectFromTracked(cfg);
    expect(files).toEqual([join(sub, "x.ts")]);
  });

  it("skips non-existent tracked paths", () => {
    writeFileSync(join(tmpA, "a.ts"), "x");
    const cfg = { ...defaultConfig(), trackedPaths: [tmpA, join(tmpA, "missing")] };
    expect(collectFromTracked(cfg)).toEqual([join(tmpA, "a.ts")]);
  });

  it("applies excludePatterns across all tracked paths", () => {
    writeFileSync(join(tmpA, "keep.ts"), "x");
    writeFileSync(join(tmpA, "skip.ts"), "x");
    writeFileSync(join(tmpB, "skip.ts"), "x");
    const cfg = { ...defaultConfig(), trackedPaths: [tmpA, tmpB], excludePatterns: ["skip.ts"] };
    const files = collectFromTracked(cfg);
    expect(files).toEqual([join(tmpA, "keep.ts")]);
  });

  it("empty trackedPaths → []", () => {
    expect(collectFromTracked(defaultConfig())).toEqual([]);
  });
});

// ─── isExcludedByConfig ──────────────────────────────────────────────────────

describe("isExcludedByConfig", () => {
  it("returns false when patterns list is empty", () => {
    expect(isExcludedByConfig("/proj/src/foo.ts", ["/proj"], [])).toBe(false);
  });

  it("matches a file inside a tracked root", () => {
    expect(isExcludedByConfig("/proj/src/foo.ts", ["/proj"], ["src/foo.ts"])).toBe(true);
  });

  it("does not match files outside any tracked root", () => {
    expect(isExcludedByConfig("/other/src/foo.ts", ["/proj"], ["src/foo.ts"])).toBe(false);
  });

  it("evaluates patterns against the nearest matching root", () => {
    expect(
      isExcludedByConfig("/a/b/c/x.ts", ["/a", "/a/b"], ["c/x.ts"])
    ).toBe(true);
  });
});

// ─── defaultConfig ───────────────────────────────────────────────────────────

describe("defaultConfig", () => {
  it("has expected shape and defaults", () => {
    const cfg = defaultConfig();
    expect(cfg.ragEnabled).toBe(true);
    expect(cfg.ragTopK).toBe(5);
    expect(cfg.ragScoreThreshold).toBe(0.1);
    expect(cfg.ragAlpha).toBe(0.4);
    expect(cfg.trackedPaths).toEqual([]);
    expect(cfg.excludePatterns).toEqual([]);
  });

  it("returns a new object each call (no shared reference)", () => {
    const a = defaultConfig();
    const b = defaultConfig();
    a.ragTopK = 99;
    a.trackedPaths.push("/foo");
    a.excludePatterns.push("bar");
    expect(b.ragTopK).toBe(5);
    expect(b.trackedPaths).toEqual([]);
    expect(b.excludePatterns).toEqual([]);
  });
});

// ─── hybridSearch (using SQLite) ─────────────────────────────────────────────

describe("hybridSearch (BM25 via FTS5, no vectors)", () => {
  it("empty index → []", async () => {
    const db = createTestDb([]);
    const results = await hybridSearch("query", { chunks: [], files: {}, lastBuild: "" }, 10, 0.4, db);
    db.close();
    expect(results).toEqual([]);
  });

  it("returns scored result for matching content", async () => {
    const db = createTestDb([
      { content: "function authenticate(user, password) { return checkCredentials(user, password); }" },
      { content: "function renderTemplate(html) { return sanitize(html); }" },
    ]);
    const results = await hybridSearch("authenticate", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].chunk.content).toContain("authenticate");
  });

  it("non-matching query → no results", async () => {
    const db = createTestDb([{ content: "function computeSquareRoot(n) { return Math.sqrt(n); }" }]);
    const results = await hybridSearch("unrelated query term xyz", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    const nonZero = results.filter(r => r.hybrid > 0);
    expect(nonZero.length).toBe(0);
  });

  it("exact phrase match scores higher than partial match", async () => {
    const db = createTestDb([
      { content: "function handle user authentication: validate token from request" },
      { content: "function handle request: process data from input" },
    ]);
    const results = await hybridSearch("user authentication", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    const first = results[0]?.chunk.content ?? "";
    expect(first).toContain("authentication");
  });

  it("respects limit parameter", async () => {
    const chunks = Array.from({ length: 10 }, (_, i) => ({
      content: `function processItem${i}(value) { return transform(value); }`,
    }));
    const db = createTestDb(chunks);
    const results = await hybridSearch("function process", { chunks: [], files: {}, lastBuild: "" }, 3, 1.0, db);
    db.close();
    expect(results.length).toBeLessThanOrEqual(3);
  });

  it("result shape has bm25, vector, hybrid, chunk fields", async () => {
    const db = createTestDb([{ content: "export function calculateTotal(items) { return items.reduce((a, b) => a + b, 0); }" }]);
    const results = await hybridSearch("calculate total", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    if (results.length > 0) {
      expect(results[0]).toHaveProperty("bm25");
      expect(results[0]).toHaveProperty("vector");
      expect(results[0]).toHaveProperty("hybrid");
      expect(results[0]).toHaveProperty("chunk");
    }
  });

  it("filename boost: first query term matching filename scores higher", async () => {
    const db = createTestDb([
      { file: "/src/auth module", content: "export function login for user verification" },
      { file: "/src/render module", content: "export function display for user rendering" },
    ]);
    const results = await hybridSearch("auth user", { chunks: [], files: {}, lastBuild: "" }, 10, 1.0, db);
    db.close();
    // auth module should rank first due to filename boost on first term "auth"
    expect(results[0]?.chunk.file).toContain("auth");
  });
});

// ─── hybridSearch with vectors ──────────────────────────────────────────────

describe("hybridSearch with vectors", () => {
  const vec = (seed: number) => Array.from({ length: 384 }, (_, i) => (i === seed ? 1 : 0));

  it("uses vector scores when chunks have embeddings", async () => {
    const db = createTestDb([
      { content: "handle user login with password verification and auth", vector: vec(0) },
      { content: "render the homepage template with context data", vector: vec(1) },
    ]);
    const results = await hybridSearch("login", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    expect(results.length).toBeGreaterThan(0);
    // Both bm25 and vector scores are present
    expect(results[0]).toHaveProperty("bm25");
    expect(results[0]).toHaveProperty("vector");
    expect(results[0]).toHaveProperty("hybrid");
  });

  it("hybrid score is blend of bm25 and vector when alpha=0.5", async () => {
    const db = createTestDb([
      { content: "authenticate user credentials and verify identity", vector: vec(0) },
      { content: "logout session token and destroy active session", vector: vec(1) },
    ]);
    const results = await hybridSearch("authenticate", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    expect(results.length).toBeGreaterThan(0);
    const r = results[0];
    const expectedHybrid = 0.5 * r.bm25 + 0.5 * r.vector;
    expect(r.hybrid).toBeCloseTo(expectedHybrid, 5);
  });

  it("falls back to pure bm25 when no chunks have valid vectors", async () => {
    const db = createTestDb([
      { content: "process payment amount through payment gateway charge" },
      { content: "refund order through payment gateway refund" },
    ]);
    const results = await hybridSearch("payment", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    if (results.length > 0) {
      expect(results[0].hybrid).toBe(results[0].bm25);
    }
  });
});
