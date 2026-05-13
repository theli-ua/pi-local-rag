import { vi, describe, it, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// Must be declared before the import so vitest hoists it above the module load.
vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async () => ({
      data: new Float32Array(384).fill(0.1),
    }))
  ),
}));

import { embedBatch, hybridSearch, indexFiles, sha256, openDb, getIndexStats } from "../index.ts";

// ─── embedBatch ───────────────────────────────────────────────────────────────

describe("embedBatch", () => {
  it("returns one 384-dim vector per text", async () => {
    const vecs = await embedBatch(["hello world", "foo bar"]);
    expect(vecs).toHaveLength(2);
    expect(vecs[0]).toHaveLength(384);
    expect(vecs[1]).toHaveLength(384);
  });

  it("returns empty array for empty input", async () => {
    expect(await embedBatch([])).toEqual([]);
  });

  it("calls onProgress once per item with correct index and total", async () => {
    const calls: [number, number][] = [];
    await embedBatch(["a", "b", "c"], (i, total) => calls.push([i, total]));
    expect(calls).toEqual([[1, 3], [2, 3], [3, 3]]);
  });

  it("works without an onProgress callback", async () => {
    await expect(embedBatch(["single text"])).resolves.toHaveLength(1);
  });
});

// ─── hybridSearch with vectors ────────────────────────────────────────────────

describe("hybridSearch with vectors", () => {
  beforeEach(() => {
    // Clear shared DB state between tests
    const db = openDb();
    db.exec("DELETE FROM chunks_vec; DELETE FROM chunks; DELETE FROM files; DELETE FROM metadata;");
    db.close();
  });
  const vec = (seed: number) => Array.from({ length: 384 }, (_, i) => (i === seed ? 1 : 0));

  it("uses vector scores when chunks have embeddings", async () => {
    // All chunks have identical mocked embed result (0.1 everywhere), so vector
    // scores will be equal — BM25 drives ranking at alpha=1
    const db = openDb();
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insVec = db.prepare("INSERT INTO chunks_vec(embedding) VALUES (?)");

    insChunk.run("c1", "/src/test.ts", "handle user login with password verification and auth", 1, 10, sha256("login"), new Date().toISOString(), 15);
    insChunk.run("c2", "/src/test.ts", "render the homepage template with context data", 1, 10, sha256("render"), new Date().toISOString(), 12);

    const f1 = new Float32Array(vec(0));
    const f2 = new Float32Array(vec(1));
    insVec.run(Buffer.from(f1.buffer, f1.byteOffset, f1.byteLength));
    insVec.run(Buffer.from(f2.buffer, f2.byteOffset, f2.byteLength));

    const results = await hybridSearch("login", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    expect(results.length).toBeGreaterThan(0);
    // Both bm25 and vector scores are present
    expect(results[0]).toHaveProperty("bm25");
    expect(results[0]).toHaveProperty("vector");
    expect(results[0]).toHaveProperty("hybrid");
  });

  it("hybrid score is blend of bm25 and vector when alpha=0.5", async () => {
    const db = openDb();
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const insVec = db.prepare("INSERT INTO chunks_vec(embedding) VALUES (?)");

    insChunk.run("c1", "/src/test.ts", "authenticate user credentials and verify identity", 1, 10, sha256("auth"), new Date().toISOString(), 12);
    insChunk.run("c2", "/src/test.ts", "logout session token and destroy active session", 1, 10, sha256("logout"), new Date().toISOString(), 10);

    const f1 = new Float32Array(vec(0));
    const f2 = new Float32Array(vec(1));
    insVec.run(Buffer.from(f1.buffer, f1.byteOffset, f1.byteLength));
    insVec.run(Buffer.from(f2.buffer, f2.byteOffset, f2.byteLength));

    const results = await hybridSearch("authenticate", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    expect(results.length).toBeGreaterThan(0);
    const r = results[0];
    const expectedHybrid = 0.5 * r.bm25 + 0.5 * r.vector;
    expect(r.hybrid).toBeCloseTo(expectedHybrid, 5);
  });

  it("falls back to pure bm25 when no chunks have valid vectors", async () => {
    const db = openDb();
    const insChunk = db.prepare(`
      INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    insChunk.run("c1", "/src/test.ts", "process payment amount through payment gateway charge", 1, 10, sha256("pay"), new Date().toISOString(), 10);
    insChunk.run("c2", "/src/test.ts", "refund order through payment gateway refund", 1, 10, sha256("ref"), new Date().toISOString(), 10);

    const results = await hybridSearch("payment", { chunks: [], files: {}, lastBuild: "" }, 10, 0.5, db);
    db.close();
    if (results.length > 0) {
      expect(results[0].hybrid).toBe(results[0].bm25);
    }
  });
});

// ─── indexFiles ───────────────────────────────────────────────────────────────

describe("indexFiles", () => {
  let tmp: string;

  beforeAll(() => {
    tmp = mkdtempSync(join(tmpdir(), "rag-idx-"));
    // Clear DB once before all indexFiles tests (they share state intentionally)
    const db = openDb();
    db.exec("DELETE FROM chunks_vec; DELETE FROM chunks; DELETE FROM files; DELETE FROM metadata;");
    db.close();
  });

  afterAll(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it("indexes a file and returns stats", async () => {
    const fp = join(tmp, "util.ts");
    writeFileSync(fp, Array(10).fill("export function computeValue(x: number) { return x * 2; }").join("\n"));
    const db = openDb();
    const result = await indexFiles([fp], {}, db);
    db.close();
    expect(result.indexed).toBe(1);
    expect(result.chunks).toBeGreaterThan(0);
    expect(result.skipped).toBe(0);
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("skips a file whose hash is unchanged", async () => {
    const fp = join(tmp, "util.ts"); // same file written above
    const db = openDb();
    const result = await indexFiles([fp], {}, db);
    db.close();
    expect(result.skipped).toBe(1);
    expect(result.indexed).toBe(0);
  });

  it("skips a non-existent file (caught by try/catch)", async () => {
    const db = openDb();
    const result = await indexFiles([join(tmp, "ghost.ts")], {}, db);
    db.close();
    expect(result.skipped).toBe(1);
    expect(result.indexed).toBe(0);
  });

  it("calls onFile and onSave progress callbacks", async () => {
    const fp = join(tmp, "callback.ts");
    writeFileSync(fp, Array(10).fill("export const answer = computeResult(42);").join("\n"));
    const fileCalls: number[] = [];
    let saveCalled = false;
    const db = openDb();
    await indexFiles([fp], {
      onFile: (current) => fileCalls.push(current),
      onSave: () => { saveCalled = true; },
    }, db);
    db.close();
    expect(fileCalls).toContain(1);
    expect(saveCalled).toBe(true);
  });

  it("handles empty path list", async () => {
    const db = openDb();
    const result = await indexFiles([], {}, db);
    db.close();
    expect(result.indexed).toBe(0);
    expect(result.chunks).toBe(0);
    expect(result.skipped).toBe(0);
  });
});
