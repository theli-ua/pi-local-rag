import { vi, describe, it, expect, beforeAll, afterAll } from "vitest";
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

import { embedBatch, hybridSearch, indexFiles, sha256 } from "../index.ts";

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

type Chunk = {
  id: string; file: string; content: string;
  lineStart: number; lineEnd: number;
  hash: string; indexed: string; tokens: number;
  vector?: number[];
};
type IndexMeta = {
  chunks: Chunk[];
  files: Record<string, { hash: string; chunks: number; indexed: string; size: number; embedded?: boolean }>;
  lastBuild: string;
  embeddingModel?: string;
};

function makeChunk(content: string, vector?: number[]): Chunk {
  return {
    id: sha256(content),
    file: "/src/test.ts",
    content,
    lineStart: 1,
    lineEnd: 10,
    hash: sha256(content),
    indexed: new Date().toISOString(),
    tokens: Math.ceil(content.length / 4),
    vector,
  };
}

describe("hybridSearch with vectors", () => {
  const vec = (seed: number) => Array.from({ length: 384 }, (_, i) => (i === seed ? 1 : 0));

  it("uses vector scores when chunks have embeddings", async () => {
    // All chunks have identical mocked embed result (0.1 everywhere), so vector
    // scores will be equal — BM25 drives ranking at alpha=1
    const idx: IndexMeta = {
      chunks: [
        makeChunk("function handleLogin(user, password) { return auth.verify(user, password); }", vec(0)),
        makeChunk("function renderHomepage(ctx) { return template.render(ctx); }", vec(1)),
      ],
      files: {},
      lastBuild: "",
    };
    const results = await hybridSearch("login", idx, 10, 0.5);
    expect(results.length).toBeGreaterThan(0);
    // Both bm25 and vector scores are present
    expect(results[0]).toHaveProperty("bm25");
    expect(results[0]).toHaveProperty("vector");
    expect(results[0]).toHaveProperty("hybrid");
  });

  it("hybrid score is blend of bm25 and vector when alpha=0.5", async () => {
    const idx: IndexMeta = {
      chunks: [
        makeChunk("function authenticateUser(credentials) { return verify(credentials); }", vec(0)),
        makeChunk("function logoutSession(token) { return session.destroy(token); }", vec(1)),
      ],
      files: {},
      lastBuild: "",
    };
    const results = await hybridSearch("authenticate", idx, 10, 0.5);
    expect(results.length).toBeGreaterThan(0);
    const r = results[0];
    const expectedHybrid = 0.5 * r.bm25 + 0.5 * r.vector;
    expect(r.hybrid).toBeCloseTo(expectedHybrid, 5);
  });

  it("falls back to pure bm25 when no chunks have valid vectors", async () => {
    const idx: IndexMeta = {
      chunks: [
        makeChunk("function processPayment(amount) { return gateway.charge(amount); }"),
        makeChunk("function refundOrder(orderId) { return gateway.refund(orderId); }"),
      ],
      files: {},
      lastBuild: "",
    };
    const results = await hybridSearch("payment", idx, 10, 0.5);
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
  });

  afterAll(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it("indexes a file and returns stats", async () => {
    const fp = join(tmp, "util.ts");
    writeFileSync(fp, Array(10).fill("export function computeValue(x: number) { return x * 2; }").join("\n"));
    const result = await indexFiles([fp], {});
    expect(result.indexed).toBe(1);
    expect(result.chunks).toBeGreaterThan(0);
    expect(result.skipped).toBe(0);
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("skips a file whose hash is unchanged", async () => {
    const fp = join(tmp, "util.ts"); // same file written above
    const result = await indexFiles([fp], {});
    expect(result.skipped).toBe(1);
    expect(result.indexed).toBe(0);
  });

  it("skips a non-existent file (caught by try/catch)", async () => {
    const result = await indexFiles([join(tmp, "ghost.ts")], {});
    expect(result.skipped).toBe(1);
    expect(result.indexed).toBe(0);
  });

  it("calls onFile and onSave progress callbacks", async () => {
    const fp = join(tmp, "callback.ts");
    writeFileSync(fp, Array(10).fill("export const answer = computeResult(42);").join("\n"));
    const fileCalls: number[] = [];
    let saveCalled = false;
    await indexFiles([fp], {
      onFile: (current) => fileCalls.push(current),
      onSave: () => { saveCalled = true; },
    });
    expect(fileCalls).toContain(1);
    expect(saveCalled).toBe(true);
  });

  it("handles empty path list", async () => {
    const result = await indexFiles([], {});
    expect(result.indexed).toBe(0);
    expect(result.chunks).toBe(0);
    expect(result.skipped).toBe(0);
  });
});
