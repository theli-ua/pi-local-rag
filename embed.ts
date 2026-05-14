import { EMBEDDING_MODEL } from "./constants.js";

let _pipeline: any = null;

export async function getEmbedder() {
  if (_pipeline) return _pipeline;
  const { pipeline } = await import("@xenova/transformers");
  _pipeline = await pipeline("feature-extraction", EMBEDDING_MODEL);
  return _pipeline;
}

export async function embed(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

const EMBED_BATCH_SIZE = 64;

export async function embedBatch(texts: string[], onProgress?: (i: number, total: number) => void): Promise<number[][]> {
  const embedder = await getEmbedder();
  const results: number[][] = [];
  for (let start = 0; start < texts.length; start += EMBED_BATCH_SIZE) {
    const batch = texts.slice(start, start + EMBED_BATCH_SIZE);
    try {
      const output = await embedder(batch, { pooling: "mean", normalize: true, truncate: true });
      for (let b = 0; b < batch.length; b++) {
        results.push(Array.from(output[b].data as Float32Array));
      }
    } catch (err) {
      process.stderr.write(`\n[embedBatch ERROR] batch ${start}/${texts.length}: ${err instanceof Error ? err.message : String(err)}\n`);
      throw err;
    }
    const done = start + batch.length;
    for (let b = 0; b < batch.length; b++) onProgress?.(done - batch.length + b + 1, texts.length);
  }
  return results;
}
