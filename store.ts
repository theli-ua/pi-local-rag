import { existsSync, mkdirSync, renameSync } from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";

export const LEGACY_DIR = join(homedir(), ".pi", "lens");
export const GLOBAL_RAG_DIR = () => join(homedir(), ".pi", "rag");

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
  while (true) {
    if (dir === home) break;
    const candidate = join(dir, ".pi", "rag");
    if (existsSync(candidate)) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  if (opts.createIfMissing) {
    const local = join(process.cwd(), ".pi", "rag");
    mkdirSync(local, { recursive: true });
    return local;
  }
  const global = GLOBAL_RAG_DIR();
  ensureDir(global);
  return global;
}

export function dbFile(ragDir: string): string { return join(ragDir, "rag.db"); }
export function configFile(ragDir: string): string { return join(ragDir, "config.json"); }
export function legacyIndexFile(ragDir: string): string { return join(ragDir, "index.json"); }

export function ensureDir(ragDir: string) {
  if (existsSync(ragDir)) return;
  if (ragDir === GLOBAL_RAG_DIR() && existsSync(LEGACY_DIR)) {
    try {
      renameSync(LEGACY_DIR, ragDir);
      return;
    } catch { /* fall through */ }
  }
  mkdirSync(ragDir, { recursive: true });
}
