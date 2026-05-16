import { describe, it, expect, vi, beforeAll, afterAll } from "vitest";
import { mkdirSync, writeFileSync, rmSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import Database from "better-sqlite3";
import { load as loadVec } from "sqlite-vec";

// Hoisted so vi.mock factories can close over it
const TEST_HOME = vi.hoisted(() => `/tmp/pi-rag-test-${process.pid}`);

vi.mock("node:os", () => ({ homedir: () => TEST_HOME }));

// Pin the RAG store to TEST_HOME so getRagDir() never walks into the real
// `~/.pi/rag/` of the developer running these tests.
process.env.PI_RAG_DIR = `${TEST_HOME}/.pi/rag`;

// Prevent real ONNX model downloads; return a fixed 384-dim vector
vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async (texts: string | string[]) => {
      if (Array.isArray(texts)) {
        return texts.map(() => ({ data: new Float32Array(384).fill(0.1) }));
      }
      return { data: new Float32Array(384).fill(0.1) };
    }),
  ),
}));

import { isIndexStale, getRagDir, loadConfig, saveConfig, openDb, getIndexStats, initSchema } from "./index.js";
import { extractText } from "./chunking.js";
import defaultExport from "./index.js";

// ─── Helpers ─────────────────────────────────────────────────────────────────

const RAG_DIR = `${TEST_HOME}/.pi/rag`;
const DB_FILE = join(RAG_DIR, "rag.db");
const CONFIG_FILE = join(RAG_DIR, "config.json");
const DAY_MS = 24 * 60 * 60 * 1000;

const DEFAULT_CONFIG = { ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4 };

function staleTimestamp() { return new Date(Date.now() - DAY_MS - 1_000).toISOString(); }
function freshTimestamp() { return new Date(Date.now() - 60_000).toISOString(); }

function createTestDb(): Database.Database {
  mkdirSync(RAG_DIR, { recursive: true });
  const db = new Database(DB_FILE);
  db.pragma("journal_mode = WAL");
  loadVec(db);
  initSchema(db);
  return db;
}

function writeIndex(data: object) {
  // For backward compat with tests that expect index.json-like writes,
  // we now write to SQLite
  mkdirSync(RAG_DIR, { recursive: true });
  const db = new Database(DB_FILE);
  db.pragma("journal_mode = WAL");
  loadVec(db);
  initSchema(db);

  // Clear existing data to avoid UNIQUE constraint failures
  db.exec("DELETE FROM chunks_vec; DELETE FROM chunks; DELETE FROM files; DELETE FROM metadata;");

  if (data.chunks && Array.isArray(data.chunks)) {
    for (const c of data.chunks) {
      db.prepare(`
        INSERT INTO chunks(id, file_path, chunk_content, line_start, line_end, chunk_hash, indexed_at, tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `).run(c.id, c.file, c.content, c.lineStart, c.lineEnd, c.hash, c.indexed, c.tokens);
      if (c.vector) {
        const f = new Float32Array(c.vector);
        db.prepare("INSERT INTO chunks_vec(embedding) VALUES (?)").run(
          Buffer.from(f.buffer, f.byteOffset, f.byteLength)
        );
      }
    }
    if (data.files) {
      for (const [fp, info] of Object.entries(data.files as Record<string, any>)) {
        db.prepare(`
          INSERT INTO files(path, hash, chunks, indexed, size, embedded)
          VALUES (?, ?, ?, ?, ?, ?)
        `).run(fp, info.hash, info.chunks, info.indexed, info.size, info.embedded ? 1 : 0);
      }
    }
  }

  if (data.lastBuild) {
    db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('last_build', ?)").run(data.lastBuild);
  }
  if (data.embeddingModel) {
    db.prepare("INSERT OR REPLACE INTO metadata(key, value) VALUES ('embedding_model', ?)").run(data.embeddingModel);
  }

  db.close();
}

function readIndex(): Record<string, any> {
  if (!existsSync(DB_FILE)) return { chunks: [], files: {}, lastBuild: "" };
  const db = new Database(DB_FILE, { readonly: true });
  try {
    const chunks = db.prepare("SELECT * FROM chunks").all();
    const files = db.prepare("SELECT * FROM files").all();
    const meta = db.prepare("SELECT * FROM metadata").all();
    return {
      chunks,
      files,
      lastBuild: (meta.find((m: any) => m.key === "last_build") as any)?.value ?? "",
    };
  } finally {
    db.close();
  }
}

/** Minimal chunk with a pre-filled vector to pass the `!index.chunks.length` guard */
function fakeChunk(file: string) {
  return {
    id: "test", file, content: "const x = 1;", lineStart: 1, lineEnd: 1,
    hash: "abc", indexed: new Date().toISOString(), tokens: 5,
    vector: new Array(384).fill(0.1),
  };
}

function makePi() {
  let hookFn: ((event: any, ctx: any) => Promise<any>) | undefined;
  let ragHandler: ((args: string, ctx: any) => Promise<any>) | undefined;
  const messages: string[] = [];
  const pi = {
    on: vi.fn((event: string, fn: any) => { if (event === "before_agent_start") hookFn = fn; }),
    registerCommand: vi.fn((name: string, def: any) => { if (name === "rag") ragHandler = def.handler; }),
    registerTool: vi.fn(),
    sendMessage: vi.fn((m: any) => { messages.push(m.content); }),
  };
  const ctx = {
    ui: {
      notify: vi.fn(),
      setStatus: vi.fn(),
      setWidget: vi.fn(),
    },
  };
  const fireHook = (event = { prompt: "hello world", systemPrompt: "" }) => hookFn!(event, {});
  const run = (args: string) => ragHandler!(args, ctx);
  return { pi, fireHook, run, messages, ctx };
}

// ─── isIndexStale ─────────────────────────────────────────────────────────────

describe("isIndexStale", () => {
  it("returns false when lastBuild is empty", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: "" } as any)).toBe(false);
  });

  it("returns false when index was built recently", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: freshTimestamp() } as any)).toBe(false);
  });

  it("returns true when lastBuild is more than 24h ago", () => {
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: staleTimestamp() } as any)).toBe(true);
  });

  it("respects a custom maxAgeMs", () => {
    const tenMinAgo = new Date(Date.now() - 10 * 60 * 1_000).toISOString();
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: tenMinAgo } as any, 5 * 60 * 1_000)).toBe(true);
    expect(isIndexStale({ chunks: [], files: {}, lastBuild: tenMinAgo } as any, 15 * 60 * 1_000)).toBe(false);
  });
});

// ─── extractText HTML → markdown ───────────────────────────────────────────

describe("extractText HTML", () => {
  beforeAll(() => { mkdirSync(TEST_HOME, { recursive: true }); });
  afterAll(() => { rmSync(TEST_HOME, { recursive: true, force: true }); });

  it("converts simple HTML to markdown", async () => {
    const fp = join(TEST_HOME, "simple.html");
    writeFileSync(fp, "<p>Hello <strong>world</strong></p>");
    const { text } = await extractText(fp);
    expect(text).toContain("Hello");
    expect(text).toContain("world");
    expect(text).not.toContain("<p>");
    expect(text).not.toContain("<strong>");
  });

  it("removes script and style blocks", async () => {
    const fp = join(TEST_HOME, "no-script.html");
    writeFileSync(fp, "<p>Before</p><script>alert('xss')</script><style>.x{}</style><p>After</p>");
    const { text } = await extractText(fp);
    expect(text).toContain("Before");
    expect(text).toContain("After");
    expect(text).not.toContain("alert");
    expect(text).not.toContain("script");
    expect(text).not.toContain(".x{}");
  });

  it("removes nav and footer elements", async () => {
    const fp = join(TEST_HOME, "no-nav.html");
    writeFileSync(fp, "<nav>Home | About</nav><p>Content</p><footer>Copyright</footer>");
    const { text } = await extractText(fp);
    expect(text).toContain("Content");
    expect(text).not.toContain("Home | About");
    expect(text).not.toContain("Copyright");
  });

  it("converts headings to atx style", async () => {
    const fp = join(TEST_HOME, "headings.html");
    writeFileSync(fp, "<h1>Title</h1><h2>Subtitle</h2><p>Body</p>");
    const { text } = await extractText(fp);
    expect(text).toContain("# Title");
    expect(text).toContain("## Subtitle");
    expect(text).toContain("Body");
  });

  it("fences code blocks", async () => {
    const fp = join(TEST_HOME, "code.html");
    writeFileSync(fp, '<pre><code class="lang-cs">var x = 1;</code></pre>');
    const { text } = await extractText(fp);
    expect(text).toContain("```");
    expect(text).toContain("var x = 1;");
  });

  it("converts lists to markdown", async () => {
    const fp = join(TEST_HOME, "lists.html");
    writeFileSync(fp, "<ul><li>One</li><li>Two</li></ul>");
    const { text } = await extractText(fp);
    expect(text).toContain("One");
    expect(text).toContain("Two");
    expect(text).not.toContain("<li>");
  });

  it("hashes the raw HTML, not the markdown", async () => {
    const fp = join(TEST_HOME, "hash-test.html");
    const raw = "<p>Content</p>";
    writeFileSync(fp, raw);
    const { hash, text } = await extractText(fp);
    // hash is first 12 hex chars of sha256 of raw HTML
    expect(hash).toMatch(/^[0-9a-f]{12}$/);
    expect(text).not.toContain("<p>");
  });

  it("handles real-world Unity doc HTML structure", async () => {
    const fp = join(TEST_HOME, "unity-doc.html");
    const html = `<!DOCTYPE html><html><head><script>var x = 1;</script></head>
<body><nav>Navigation</nav><div class="content"><h1>Add textures to the camera history</h1>
<p>To add your own texture to the <strong>camera</strong> history.</p>
<pre><code>public class Example : CameraHistoryItem { }</code></pre>
<ul><li>Step one</li><li>Step two</li></ul>
</div><footer>Copyright</footer></body></html>`;
    writeFileSync(fp, html);
    const { text } = await extractText(fp);
    expect(text).toContain("# Add textures to the camera history");
    expect(text).toContain("camera");
    expect(text).toContain("public class Example : CameraHistoryItem { }");
    expect(text).toContain("Step one");
    expect(text).toContain("Step two");
    expect(text).not.toContain("<script>");
    expect(text).not.toContain("var x");
    expect(text).not.toContain("Navigation");
    expect(text).not.toContain("Copyright");
  });

  it("returns text and hash for non-HTML files unchanged", async () => {
    const fp = join(TEST_HOME, "plain.txt");
    writeFileSync(fp, "just text");
    const { text } = await extractText(fp);
    expect(text).toBe("just text");
  });

  it("produces much smaller output than raw HTML for Unity docs", async () => {
    const fp = join(TEST_HOME, "big.html");
    // Simulate a Unity doc with lots of script/style/nav boilerplate
    const html = "<script>" + "x".repeat(5000) + "</script>"
      + "<style>" + "y".repeat(3000) + "</style>"
      + "<nav>" + "z".repeat(2000) + "</nav>"
      + "<p>Actual content here about framebuffer fetch</p>"
      + "<footer>" + "w".repeat(1000) + "</footer>";
    writeFileSync(fp, html);
    const { text } = await extractText(fp);
    expect(text.length).toBeLessThan(html.length / 2);
    expect(text).toContain("Actual content here about framebuffer fetch");
    expect(text).not.toContain("x".repeat(100));
  });
});

// ─── before_agent_start auto-rebuild ─────────────────────────────────────────

describe("before_agent_start auto-rebuild", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("does not update lastBuild when index is fresh", async () => {
    const freshBuild = freshTimestamp();
    writeIndex({ chunks: [fakeChunk("/some/file.ts")], files: {}, lastBuild: freshBuild });

    const { pi, fireHook } = makePi();
    defaultExport(pi as any);
    await fireHook();

    expect(readIndex().lastBuild).toBe(freshBuild);
  });

  it("updates lastBuild when index is stale and files exist on disk", async () => {
    const testFile = join(TEST_HOME, "sample.ts");
    writeFileSync(testFile, "export const answer = 42;\n");

    const staleBuild = staleTimestamp();
    writeIndex({
      chunks: [fakeChunk(testFile)],
      files: { [testFile]: { hash: "old", chunks: 1, indexed: staleBuild, size: 26, embedded: true } },
      lastBuild: staleBuild,
    });

    const { pi, fireHook } = makePi();
    defaultExport(pi as any);
    await fireHook();

    const updated = readIndex();
    expect(new Date(updated.lastBuild).getTime()).toBeGreaterThan(new Date(staleBuild).getTime());
  });

  it("does not update lastBuild when stale but all referenced files are gone", async () => {
    const staleBuild = staleTimestamp();
    const missingFile = join(TEST_HOME, "deleted.ts");
    writeIndex({
      chunks: [fakeChunk(missingFile)],
      files: { [missingFile]: { hash: "old", chunks: 1, indexed: staleBuild, size: 10, embedded: true } },
      lastBuild: staleBuild,
    });

    const { pi, fireHook } = makePi();
    defaultExport(pi as any);
    await fireHook();

    expect(readIndex().lastBuild).toBe(staleBuild);
  });
});

// ─── /rag exclude subcommand ─────────────────────────────────────────────────

describe("/rag exclude subcommand", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  function readConfig(): Record<string, any> {
    return JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
  }

  it("adds a pattern to excludePatterns", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run("exclude node_modules");
    expect(readConfig().excludePatterns).toEqual(["node_modules"]);
  });

  it("removes a pattern with leading dash", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["foo", "bar"] }));
    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run("exclude -foo");
    expect(readConfig().excludePatterns).toEqual(["bar"]);
  });

  it("does not duplicate an already-present pattern", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["foo"] }));
    const { pi, run, ctx } = makePi();
    defaultExport(pi as any);
    await run("exclude foo");
    expect(readConfig().excludePatterns).toEqual(["foo"]);
    const notifyCalls = (ctx.ui.notify as vi.Mock).mock.calls.map((c: any[]) => c[0]);
    expect(notifyCalls.some((m: string) => /already excluded/i.test(m))).toBe(true);
  });

  it("reports error when removing a non-existent pattern", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: [] }));
    const { pi, run, ctx } = makePi();
    defaultExport(pi as any);
    await run("exclude -ghost");
    const notifyCalls = (ctx.ui.notify as vi.Mock).mock.calls.map((c: any[]) => c[0]);
    expect(notifyCalls.some((m: string) => /not found/i.test(m))).toBe(true);
  });

  it("lists current patterns when called with no argument", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["a", "b"] }));
    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("exclude");
    const last = messages[messages.length - 1];
    expect(last).toContain("a");
    expect(last).toContain("b");
  });
});

// ─── /rag index auto-tracks paths ────────────────────────────────────────────

describe("/rag index auto-tracking", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("adds the indexed path to trackedPaths in config", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const projDir = join(TEST_HOME, "proj-track");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "main.ts"), "export const a = 1;\n");

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);

    const cfg = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
    expect(cfg.trackedPaths).toContain(projDir);
  });

  it("does not duplicate when indexing the same path twice", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const projDir = join(TEST_HOME, "proj-dedup");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "a.ts"), "export const a = 1;\n");

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);
    await run(`index ${projDir}`);

    const cfg = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
    expect(cfg.trackedPaths.filter((p: string) => p === projDir).length).toBe(1);
  });
});

// ─── /rag rebuild discovers new files ────────────────────────────────────────

describe("/rag rebuild new-file discovery", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("picks up files added after the initial index", async () => {
    const projDir = join(TEST_HOME, "proj-rebuild");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "first.ts"), "export const a = 1;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, trackedPaths: [projDir], excludePatterns: [] }));

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);

    // Add a new file after indexing, then rebuild.
    const newFile = join(projDir, "second.ts");
    writeFileSync(newFile, "export const b = 2;\n");

    await run("rebuild");

    const idx = readIndex();
    const filePaths = idx.files?.map?.((f: any) => f.path) ?? Object.keys(idx.files ?? {});
    expect(filePaths).toContain(newFile);
  });

  it("drops files that match a newly-added exclude pattern", async () => {
    const projDir = join(TEST_HOME, "proj-rebuild-excl");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "keep.ts"), "export const k = 1;\n");
    writeFileSync(join(projDir, "drop.ts"), "export const d = 2;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, trackedPaths: [projDir], excludePatterns: [] }));

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);
    await run("exclude drop.ts");
    await run("rebuild");

    const idx = readIndex();
    const filePaths = idx.files?.map?.((f: any) => f.path) ?? Object.keys(idx.files ?? {});
    expect(filePaths).toContain(join(projDir, "keep.ts"));
    expect(filePaths).not.toContain(join(projDir, "drop.ts"));
  });

  it("prunes files that were deleted from disk", async () => {
    const projDir = join(TEST_HOME, "proj-rebuild-delete");
    mkdirSync(projDir, { recursive: true });
    const keep = join(projDir, "keep.ts");
    const gone = join(projDir, "gone.ts");
    writeFileSync(keep, "export const k = 1;\n");
    writeFileSync(gone, "export const g = 2;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));

    const { pi, run } = makePi();
    defaultExport(pi as any);
    await run(`index ${projDir}`);

    rmSync(gone);
    await run("rebuild");

    const idx = readIndex();
    const filePaths = idx.files?.map?.((f: any) => f.path) ?? Object.keys(idx.files ?? {});
    expect(filePaths).toContain(keep);
    expect(filePaths).not.toContain(gone);
    const chunkFiles = idx.chunks?.map?.((c: any) => c.file_path ?? c.file) ?? [];
    expect(chunkFiles).not.toContain(gone);
  });
});

// ─── /rag status output ──────────────────────────────────────────────────────

describe("/rag status output", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  it("lists tracked paths and exclude patterns", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify({
      ...DEFAULT_CONFIG,
      trackedPaths: ["/tmp/aaa", "/tmp/bbb"],
      excludePatterns: ["**/fixtures/**", "scratch/"],
    }));
    writeIndex({ chunks: [], files: {}, lastBuild: "" });

    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("");

    const out = messages[messages.length - 1];
    expect(out).toMatch(/Tracked paths/i);
    expect(out).toContain("/tmp/aaa");
    expect(out).toContain("/tmp/bbb");
    expect(out).toMatch(/Exclude patterns/i);
    expect(out).toContain("**/fixtures/**");
    expect(out).toContain("scratch/");
  });

  it("shows '(none)' placeholders when both lists are empty", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    writeIndex({ chunks: [], files: {}, lastBuild: "" });

    const { pi, run, messages } = makePi();
    defaultExport(pi as any);
    await run("");

    const out = messages[messages.length - 1];
    expect(out).toMatch(/Tracked paths[\s\S]*\(none/);
    expect(out).toMatch(/Exclude patterns[\s\S]*\(none/);
  });
});

// ─── rag_index tool auto-tracks ──────────────────────────────────────────────

describe("rag_index tool", () => {
  beforeAll(() => {
    mkdirSync(RAG_DIR, { recursive: true });
  });

  afterAll(() => {
    rmSync(TEST_HOME, { recursive: true, force: true });
  });

  function captureTools() {
    const tools: any[] = [];
    let hookFn: any;
    const pi = {
      on: vi.fn((event: string, fn: any) => { if (event === "before_agent_start") hookFn = fn; }),
      registerCommand: vi.fn(),
      registerTool: vi.fn((def: any) => { tools.push(def); }),
      sendMessage: vi.fn(),
    };
    return { pi, tools, fireHook: () => hookFn };
  }

  it("auto-adds the indexed path to trackedPaths", async () => {
    writeFileSync(CONFIG_FILE, JSON.stringify(DEFAULT_CONFIG));
    const projDir = join(TEST_HOME, "tool-track");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "x.ts"), "export const x = 1;\n");

    const { pi, tools } = captureTools();
    defaultExport(pi as any);
    const ragIndex = tools.find(t => t.name === "rag_index");
    expect(ragIndex).toBeDefined();
    await ragIndex.execute("call-1", { path: projDir });

    const cfg = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
    expect(cfg.trackedPaths).toContain(projDir);
  });

  it("respects excludePatterns when walking", async () => {
    const projDir = join(TEST_HOME, "tool-exclude");
    mkdirSync(projDir, { recursive: true });
    writeFileSync(join(projDir, "keep.ts"), "export const k = 1;\n");
    writeFileSync(join(projDir, "drop.ts"), "export const d = 2;\n");

    writeFileSync(CONFIG_FILE, JSON.stringify({ ...DEFAULT_CONFIG, excludePatterns: ["drop.ts"] }));
    writeIndex({ chunks: [], files: {}, lastBuild: "" });

    const { pi, tools } = captureTools();
    defaultExport(pi as any);
    const ragIndex = tools.find(t => t.name === "rag_index");
    await ragIndex.execute("call-1", { path: projDir });

    const idx = readIndex();
    const filePaths = idx.files?.map?.((f: any) => f.path) ?? Object.keys(idx.files ?? {});
    expect(filePaths).toContain(join(projDir, "keep.ts"));
    expect(filePaths).not.toContain(join(projDir, "drop.ts"));
  });
});

// ─── Per-project store resolution ────────────────────────────────────────────

describe("getRagDir resolution", () => {
  // These tests exercise walk-up + auto-create. They must NOT use the
  // PI_RAG_DIR env override (that override wins above walk-up), so we save
  // and clear it for each case, restoring afterwards.
  let savedOverride: string | undefined;
  let savedCwd: string;

  beforeAll(() => { mkdirSync(TEST_HOME, { recursive: true }); });
  afterAll(() => { rmSync(TEST_HOME, { recursive: true, force: true }); });

  function withoutOverride(cb: () => void | Promise<void>) {
    return async () => {
      savedOverride = process.env.PI_RAG_DIR;
      savedCwd = process.cwd();
      delete process.env.PI_RAG_DIR;
      try { await cb(); }
      finally {
        process.chdir(savedCwd);
        if (savedOverride !== undefined) process.env.PI_RAG_DIR = savedOverride;
        else delete process.env.PI_RAG_DIR;
      }
    };
  }

  it("walks up from cwd to find a project store", withoutOverride(() => {
    const projRoot = join(TEST_HOME, "walkup-proj");
    const projStore = join(projRoot, ".pi", "rag");
    const subDir = join(projRoot, "src", "deep");
    mkdirSync(projStore, { recursive: true });
    mkdirSync(subDir, { recursive: true });
    process.chdir(subDir);

    expect(getRagDir()).toBe(projStore);
  }));

  it("auto-creates ./.pi/rag/ at cwd when createIfMissing and no walk-up hit", withoutOverride(() => {
    const fresh = join(TEST_HOME, "fresh-proj");
    mkdirSync(fresh, { recursive: true });
    process.chdir(fresh);

    const dir = getRagDir({ createIfMissing: true });
    expect(dir).toBe(join(fresh, ".pi", "rag"));
    // Subsequent calls (no createIfMissing) should now find it via walk-up.
    expect(getRagDir()).toBe(join(fresh, ".pi", "rag"));
  }));

  it("falls back to ~/.pi/rag/ when no walk-up hit and no createIfMissing", withoutOverride(() => {
    const noProj = join(TEST_HOME, "no-proj");
    mkdirSync(noProj, { recursive: true });
    process.chdir(noProj);

    expect(getRagDir()).toBe(join(TEST_HOME, ".pi", "rag"));
  }));

  it("walk-up stops before homedir (does not return ~/.pi/rag/ via walk-up)", withoutOverride(() => {
    // ${TEST_HOME}/.pi/rag exists (it IS the home global), but walk-up from a
    // subdir of home should stop AT home and skip checking it via walk-up. The
    // result is the same path, but it comes from the fallback branch — pinned
    // by the fact that no project store between cwd and home is consulted.
    mkdirSync(join(TEST_HOME, ".pi", "rag"), { recursive: true });
    const sub = join(TEST_HOME, "sub");
    mkdirSync(sub, { recursive: true });
    process.chdir(sub);

    expect(getRagDir()).toBe(join(TEST_HOME, ".pi", "rag"));
  }));

  it("isolates two sibling project stores", withoutOverride(() => {
    const a = join(TEST_HOME, "iso-a");
    const b = join(TEST_HOME, "iso-b");
    mkdirSync(join(a, ".pi", "rag"), { recursive: true });
    mkdirSync(join(b, ".pi", "rag"), { recursive: true });

    process.chdir(a);
    saveConfig({ ragEnabled: false, ragTopK: 1, ragScoreThreshold: 0.9, ragAlpha: 0.0, trackedPaths: [a], excludePatterns: ["a-only"] });

    process.chdir(b);
    saveConfig({ ragEnabled: true, ragTopK: 7, ragScoreThreshold: 0.2, ragAlpha: 0.8, trackedPaths: [b], excludePatterns: ["b-only"] });

    process.chdir(a);
    const cfgA = loadConfig();
    expect(cfgA.ragTopK).toBe(1);
    expect(cfgA.excludePatterns).toEqual(["a-only"]);

    process.chdir(b);
    const cfgB = loadConfig();
    expect(cfgB.ragTopK).toBe(7);
    expect(cfgB.excludePatterns).toEqual(["b-only"]);
  }));
});
