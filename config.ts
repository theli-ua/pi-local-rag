import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { getRagDir, configFile } from "./store.js";

export interface RagConfig {
  ragEnabled: boolean;
  ragTopK: number;
  ragScoreThreshold: number;
  ragAlpha: number;
  trackedPaths: string[];
  excludePatterns: string[];
  ragAutoRefresh: boolean;
}

export function defaultConfig(): RagConfig {
  return {
    ragEnabled: false, ragTopK: 5, ragScoreThreshold: 0.1, ragAlpha: 0.4,
    trackedPaths: [], excludePatterns: [], ragAutoRefresh: false,
  };
}

export function loadConfig(): RagConfig {
  const ragDir = getRagDir();
  const cfgFile = configFile(ragDir);
  if (!existsSync(cfgFile)) return defaultConfig();
  try {
    return { ...defaultConfig(), ...JSON.parse(readFileSync(cfgFile, "utf-8")) } as RagConfig;
  } catch { return defaultConfig(); }
}

export function saveConfig(config: RagConfig) {
  const ragDir = getRagDir();
  writeFileSync(configFile(ragDir), JSON.stringify(config, null, 2));
}
