import { vi, describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";

// Must be declared before the import so vitest hoists it above the module load.
vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async (texts: string | string[]) => {
      if (Array.isArray(texts)) {
        return texts.map(() => ({ data: new Float32Array(384).fill(0.1) }));
      }
      return { data: new Float32Array(384).fill(0.1) };
    })
  ),
}));

import {
  hybridSearch,
  indexFiles,
  initSchema,
  getIndexStats,
  sha256,
} from "../index.ts";

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Create an in-memory SQLite DB with the RAG schema. */
function makeDb(): Database.Database {
  const db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  loadVec(db);
  initSchema(db);
  return db;
}

/**
 * Insert chunks with vectors whose rowids are explicitly aligned
 * (matches what indexFiles does after the fix).
 */
function insertWithAlignedVectors(
  db: Database.Database,
  rows: Array<{ id: string; file: string; content: string; vector: number[] }>,
) {
  const insChunk = db.prepare(`
    INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insVec = db.prepare(
    "INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)",
  );

  for (const r of rows) {
    const result = insChunk.run(
      r.id,
      r.file,
      r.content,
      1,
      r.content.split("\n").length,
      sha256(r.content),
      new Date().toISOString(),
      Math.ceil(r.content.length / 4),
    );
    const f = new Float32Array(r.vector);
    insVec.run(Number(result.lastInsertRowid), Buffer.from(f.buffer, f.byteOffset, f.byteLength));
  }
}

// ─── initSchema ──────────────────────────────────────────────────────────────

describe("initSchema", () => {
  it("is idempotent — calling twice does not error", () => {
    const db = makeDb();
    expect(() => initSchema(db)).not.toThrow();
    db.close();
  });

  it("creates all required tables", () => {
    const db = makeDb();
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
      .all() as Array<{ name: string }>;
    const names = tables.map((t) => t.name);
    expect(names).toContain("chunks");
    expect(names).toContain("chunks_fts");
    expect(names).toContain("chunks_vec");
    expect(names).toContain("files");
    expect(names).toContain("metadata");
    db.close();
  });
});

// ─── getIndexStats ───────────────────────────────────────────────────────────

describe("getIndexStats", () => {
  it("returns zeros for empty DB", () => {
    const db = makeDb();
    const stats = getIndexStats(db);
    expect(stats.totalChunks).toBe(0);
    expect(stats.totalFiles).toBe(0);
    expect(stats.totalTokens).toBe(0);
    expect(stats.embeddedCount).toBe(0);
    expect(stats.lastBuild).toBe("");
    db.close();
  });

  it("returns accurate counts after inserts", () => {
    const db = makeDb();
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insVec = db.prepare(
      "INSERT INTO chunks_vec(rowid, embedding) VALUES (CAST(? AS INTEGER), ?)",
    );
    const insFile = db.prepare(`
      INSERT OR REPLACE INTO files(path, hash, chunks, indexed, size, embedded)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    const buf = Buffer.from(new Float32Array(384).buffer);
    insChunk.run("c1", "/a.ts", "hello world content here", 1, 1, sha256("a"), new Date().toISOString(), 5);
    insVec.run(Number(db.prepare("SELECT last_insert_rowid()").get().last_insert_rowid), buf);
    insChunk.run("c2", "/a.ts", "another chunk of text", 2, 2, sha256("b"), new Date().toISOString(), 4);
    insVec.run(Number(db.prepare("SELECT last_insert_rowid()").get().last_insert_rowid), buf);
    insFile.run("/a.ts", sha256("file"), 2, new Date().toISOString(), 100, 1);

    const stats = getIndexStats(db);
    expect(stats.totalChunks).toBe(2);
    expect(stats.totalFiles).toBe(1);
    expect(stats.totalTokens).toBe(9);
    expect(stats.embeddedCount).toBe(2);
    db.close();
  });
});

// ─── indexFiles — rowid alignment ────────────────────────────────────────────

describe("indexFiles (rowid alignment)", () => {
  let tmp: string;

  beforeAll(() => {
    tmp = mkdtempSync(join(tmpdir(), "rag-rowid-"));
  });

  afterAll(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it("aligns chunks_vec rowids with chunks rowids for a single-chunk file", async () => {
    const fp = join(tmp, "single.ts");
    writeFileSync(fp, "export function singleChunk() { return true; }\n");
    const db = makeDb();
    await indexFiles([fp], {}, db);

    const chunkRowids = db.prepare("SELECT rowid FROM chunks ORDER BY rowid").all() as Array<{ rowid: number }>;
    const vecRowids = db.prepare("SELECT rowid FROM chunks_vec ORDER BY rowid").all() as Array<{ rowid: number }>;

    expect(chunkRowids.map((r) => r.rowid)).toEqual(vecRowids.map((r) => r.rowid));
    db.close();
  });

  it("aligns rowids for a multi-chunk file", async () => {
    const fp = join(tmp, "multi.ts");
    // 150 lines → multiple chunks (maxLines=50)
    writeFileSync(fp, Array(150).fill("export function doSomething(x: number): number { return x * 2; }\n").join(""));
    const db = makeDb();
    const result = await indexFiles([fp], {}, db);
    expect(result.chunks).toBeGreaterThan(1);

    const chunkRowids = db.prepare("SELECT rowid FROM chunks ORDER BY rowid").all() as Array<{ rowid: number }>;
    const vecRowids = db.prepare("SELECT rowid FROM chunks_vec ORDER BY rowid").all() as Array<{ rowid: number }>;

    expect(chunkRowids.map((r) => r.rowid)).toEqual(vecRowids.map((r) => r.rowid));
    db.close();
  });

  it("aligns rowids after re-indexing the same file", async () => {
    const fp = join(tmp, "reindex.ts");
    writeFileSync(fp, Array(10).fill("export function firstVersion() { return 1; }\n").join(""));
    const db = makeDb();

    // First index
    await indexFiles([fp], {}, db);

    // Modify and re-index
    writeFileSync(fp, Array(10).fill("export function secondVersion() { return 2; }\n").join(""));
    await indexFiles([fp], {}, db);

    const chunkRowids = db.prepare("SELECT rowid FROM chunks ORDER BY rowid").all() as Array<{ rowid: number }>;
    const vecRowids = db.prepare("SELECT rowid FROM chunks_vec ORDER BY rowid").all() as Array<{ rowid: number }>;

    expect(chunkRowids.map((r) => r.rowid)).toEqual(vecRowids.map((r) => r.rowid));
    db.close();
  });

  it("aligns rowids when indexing multiple files", async () => {
    const fp1 = join(tmp, "a.ts");
    const fp2 = join(tmp, "b.ts");
    writeFileSync(fp1, Array(10).fill("export function funcA() { return 'a'; }\n").join(""));
    writeFileSync(fp2, Array(10).fill("export function funcB() { return 'b'; }\n").join(""));
    const db = makeDb();
    await indexFiles([fp1, fp2], {}, db);

    const chunkRowids = db.prepare("SELECT rowid FROM chunks ORDER BY rowid").all() as Array<{ rowid: number }>;
    const vecRowids = db.prepare("SELECT rowid FROM chunks_vec ORDER BY rowid").all() as Array<{ rowid: number }>;

    expect(chunkRowids.map((r) => r.rowid)).toEqual(vecRowids.map((r) => r.rowid));
    db.close();
  });
});

// ─── hybridSearch — vector normalization ─────────────────────────────────────

describe("hybridSearch (vector normalization)", () => {
  // Mocked embedder returns [0.1, 0.1, ...] for every query.
  // Stored vectors use distinct patterns so distances differ.

  const vec = (seed: number) => Array.from({ length: 384 }, (_, i) => (i === seed ? 1 : 0));

  it("single result: vector score is 1.0 (cosRange=0 branch)", async () => {
    const db = makeDb();
    insertWithAlignedVectors(db, [
      {
        id: "c1",
        file: "/src/only.ts",
        content: "export function theOnlyFunction() { return 42; }",
        vector: vec(0),
      },
    ]);

    const results = await hybridSearch(
      "the only function",
      { chunks: [], files: {}, lastBuild: "" },
      10,
      0.4,
      db,
    );
    db.close();

    expect(results).toHaveLength(1);
    // Single result → cosRange=0 → vec normalized to 1.0
    expect(results[0].vector).toBeCloseTo(1, 4);
  });

  it("multiple results: best vector score is 1.0, ranking reflects similarity", async () => {
    const db = makeDb();
    insertWithAlignedVectors(db, [
      { id: "c1", file: "/a.ts", content: "handle user authentication and login flow with token", vector: vec(0) },
      { id: "c2", file: "/b.ts", content: "render the homepage template with token and data", vector: vec(100) },
      { id: "c3", file: "/c.ts", content: "process database query results with token and cache", vector: vec(200) },
    ]);

    // Query matches all 3 via BM25 (all contain "token").
    // With alpha=0.4, BM25 lifts all results above the hybrid>0 filter.
    const results = await hybridSearch(
      "token",
      { chunks: [], files: {}, lastBuild: "" },
      10,
      0.4, // blended — BM25 ensures all pass filter, vector differentiates
      db,
    );
    db.close();

    // All 3 matched by BM25, all have vectors → all should appear
    expect(results.length).toBeGreaterThanOrEqual(2);
    // Best vector score is always 1.0 after min-max normalization
    const bestVec = Math.max(...results.map((r) => r.vector));
    expect(bestVec).toBeCloseTo(1, 4);
    // Results are sorted by hybrid score descending
    for (let i = 1; i < results.length; i++) {
      expect(results[i].hybrid).toBeLessThanOrEqual(results[i - 1].hybrid);
    }
  });

  it("vector-only match: query has no BM25 overlap but vector finds it", async () => {
    const db = makeDb();
    insertWithAlignedVectors(db, [
      {
        id: "c1",
        file: "/src/math.ts",
        content: "compute the numerical derivative using finite difference approximation",
        vector: vec(0),
      },
      {
        id: "c2",
        file: "/src/auth.ts",
        content: "verify user credentials against the authentication database server",
        vector: vec(0),
      },
    ]);

    // "calculus slope" shares no keywords with either chunk,
    // but is semantically closer to the math chunk.
    // With mocked embeddings (all identical), vector distances are equal,
    // so we verify that when bm25=0 the result still appears via vector.
    const results = await hybridSearch(
      "calculus slope tangent line",
      { chunks: [], files: {}, lastBuild: "" },
      10,
      0.0, // pure vector
      db,
    );
    db.close();

    // Even with zero BM25 overlap, vector search returns results
    expect(results.length).toBeGreaterThan(0);
    // BM25 contribution is 0 (no keyword match)
    expect(results[0].bm25).toBe(0);
    // Vector score drives the result
    expect(results[0].vector).toBeGreaterThan(0);
  });

  it("alpha=1.0 → pure BM25 (vector ignored)", async () => {
    const db = makeDb();
    insertWithAlignedVectors(db, [
      { id: "c1", file: "/a.ts", content: "export function calculate the total sum of all values in the list", vector: vec(0) },
    ]);

    const results = await hybridSearch(
      "calculate sum",
      { chunks: [], files: {}, lastBuild: "" },
      10,
      1.0, // pure BM25
      db,
    );
    db.close();

    expect(results.length).toBeGreaterThan(0);
    // hybrid == bm25 when alpha=1
    expect(results[0].hybrid).toBeCloseTo(results[0].bm25, 5);
  });

  it("multi-word query: BM25 matches chunks containing any term (OR semantics)", async () => {
    // Regression test: pre-fix, multi-word queries were joined with implicit
    // AND (space-separated quoted phrases in FTS5). Since no single chunk
    // contained every query term, FTS5 returned zero rows and every result
    // had bm25 = 0. Post-fix, terms are OR-joined, so chunks matching any
    // single term receive a BM25 score.
    const db = makeDb();
    insertWithAlignedVectors(db, [
      { id: "c1", file: "/a.ts", content: "keycloak provider configuration",  vector: vec(0) },
      { id: "c2", file: "/b.ts", content: "argocd application sync settings", vector: vec(1) },
    ]);

    // No chunk contains BOTH "keycloak" AND "argocd" — only one each.
    // Pre-fix: implicit AND → 0 FTS rows → bm25 = 0 for every result.
    // Post-fix: explicit OR  → 2 FTS rows → at least one result has bm25 > 0.
    const results = await hybridSearch(
      "keycloak argocd",
      { chunks: [], files: {}, lastBuild: "" },
      10,
      0.4,
      db,
    );
    db.close();

    expect(results.some(r => r.bm25 > 0)).toBe(true);
  });

  it("alpha=0.0 → pure vector (BM25 ignored)", async () => {
    const db = makeDb();
    insertWithAlignedVectors(db, [
      { id: "c1", file: "/a.ts", content: "export function compute the aggregate total for items", vector: vec(0) },
    ]);

    const results = await hybridSearch(
      "calculate sum", // no BM25 keyword match, vector-only
      { chunks: [], files: {}, lastBuild: "" },
      10,
      0.0, // pure vector
      db,
    );
    db.close();

    expect(results.length).toBeGreaterThan(0);
    // hybrid == vector when alpha=0 (single result → vec=1.0)
    expect(results[0].hybrid).toBeCloseTo(results[0].vector, 5);
  });
});

// ─── hybridSearch — end-to-end via indexFiles ────────────────────────────────

describe("hybridSearch (end-to-end via indexFiles)", () => {
  let tmp: string;

  beforeAll(() => {
    tmp = mkdtempSync(join(tmpdir(), "rag-e2e-"));
  });

  afterAll(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it("finds content indexed via indexFiles", async () => {
    const fp = join(tmp, "sample.ts");
    writeFileSync(
      fp,
      Array(10)
        .fill("export function processData(input: string): string { return input.trim().toLowerCase(); }")
        .join("\n"),
    );
    const db = makeDb();
    await indexFiles([fp], {}, db);

    const results = await hybridSearch(
      "process data",
      { chunks: [], files: {}, lastBuild: "" },
      10,
      0.4,
      db,
    );
    db.close();

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].chunk.content).toContain("processData");
  });

  it("search returns empty when index has no matching content", async () => {
    const fp = join(tmp, "unrelated.ts");
    writeFileSync(fp, "export const PI = 3.14159;\n");
    const db = makeDb();
    await indexFiles([fp], {}, db);

    const results = await hybridSearch(
      "quantum entanglement photon polarization",
      { chunks: [], files: {}, lastBuild: "" },
      10,
      1.0, // pure BM25 — no keyword match at all
      db,
    );
    db.close();

    // BM25-only search returns nothing for completely unrelated query
    const nonZero = results.filter((r) => r.hybrid > 0);
    expect(nonZero.length).toBe(0);
  });
});

// ─── indexFiles — files table tracking ────────────────────────────────────────

describe("indexFiles (files table)", () => {
  let tmp: string;

  beforeAll(() => {
    tmp = mkdtempSync(join(tmpdir(), "rag-files-"));
  });

  afterAll(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it("updates files table with correct chunk count and embedded flag", async () => {
    const fp = join(tmp, "tracked.ts");
    writeFileSync(fp, Array(10).fill("export function trackedFunc() { return true; }\n").join(""));
    const db = makeDb();
    await indexFiles([fp], {}, db);

    const fileRow = db.prepare("SELECT * FROM files WHERE path = ?").get(fp) as {
      chunks: number; embedded: number; hash: string;
    };
    expect(fileRow.chunks).toBeGreaterThan(0);
    expect(fileRow.embedded).toBe(1);
    expect(fileRow.hash).toMatch(/^[0-9a-f]{12}$/);
    db.close();
  });

  it("updates files table on re-index with new chunk count", async () => {
    const fp = join(tmp, "resize.ts");
    // First: small file (1 chunk)
    writeFileSync(fp, "export function small() { return 1; }\n");
    const db = makeDb();
    await indexFiles([fp], {}, db);

    let fileRow = db.prepare("SELECT chunks, embedded FROM files WHERE path = ?").get(fp) as {
      chunks: number; embedded: number;
    };
    const firstChunks = fileRow.chunks;

    // Re-index with larger file (more chunks)
    writeFileSync(fp, Array(120).fill("export function larger() { return computeComplexValue(x, y, z); }\n").join(""));
    await indexFiles([fp], {}, db);

    fileRow = db.prepare("SELECT chunks, embedded FROM files WHERE path = ?").get(fp) as {
      chunks: number; embedded: number;
    };
    expect(fileRow.chunks).toBeGreaterThan(firstChunks);
    expect(fileRow.embedded).toBe(1);
    db.close();
  });

  it("stores last_build and embedding_model in metadata", async () => {
    const fp = join(tmp, "meta.ts");
    writeFileSync(fp, "export function metaFunc() { return 'x'; }\n");
    const db = makeDb();
    await indexFiles([fp], {}, db);

    const lastBuild = db.prepare("SELECT value FROM metadata WHERE key = 'last_build'").get() as { value?: string };
    const model = db.prepare("SELECT value FROM metadata WHERE key = 'embedding_model'").get() as { value?: string };
    expect(lastBuild.value).toBeDefined();
    expect(model.value).toContain("all-MiniLM-L6-v2");
    db.close();
  });
});

// ─── FTS5 trigger sync ────────────────────────────────────────────────────────

describe("FTS5 trigger sync", () => {
  it("chunks_fts is populated on chunk insert", () => {
    const db = makeDb();
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    insChunk.run("c1", "/a.ts", "hello world content", 1, 1, sha256("a"), new Date().toISOString(), 5);

    const ftsCount = db.prepare("SELECT COUNT(*) as c FROM chunks_fts").get() as { c: number };
    expect(ftsCount.c).toBe(1);

    const ftsMatch = db.prepare("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'hello'").all();
    expect(ftsMatch).toHaveLength(1);
    db.close();
  });

  it("chunks_fts is cleaned up on chunk delete", () => {
    const db = makeDb();
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    insChunk.run("c1", "/a.ts", "hello world content", 1, 1, sha256("a"), new Date().toISOString(), 5);
    insChunk.run("c2", "/b.ts", "goodbye world content", 1, 1, sha256("b"), new Date().toISOString(), 5);

    expect(db.prepare("SELECT COUNT(*) as c FROM chunks_fts").get().c).toBe(2);

    db.prepare("DELETE FROM chunks WHERE id = 'c1'").run();

    const remaining = db.prepare("SELECT COUNT(*) as c FROM chunks_fts").get() as { c: number };
    expect(remaining.c).toBe(1);

    const helloMatch = db.prepare("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'hello'").all();
    expect(helloMatch).toHaveLength(0);
    db.close();
  });
});
