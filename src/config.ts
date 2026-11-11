import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

export interface OutClassifyConfig {
  thresholdChars: number;
  // Fewer images than this are not classified.
  thresholdImages: number;
  maxSectionChars: number;
  // A Python with Docling to use instead of the venv out-classify installs itself.
  pythonPath?: string;
}

const DEFAULT_THRESHOLD_CHARS = 8000;
const DEFAULT_THRESHOLD_IMAGES = 5;
const DEFAULT_MAX_SECTION_CHARS = 2000;

// Loads an integer key from the config, using the default when missing and requiring at least `min`.
function loadKeyInteger(
  config: Record<string, unknown>,
  key: string,
  fallback: number,
  min: number,
): number {
  const value = config[key] === undefined ? fallback : config[key];

  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min) {
    throw new Error(`Out-classify ${key} must be an integer of at least ${min}.`);
  }

  return value;
}

// Loads a non-empty path key; relative paths resolve from the agent directory, like the config itself.
function loadKeyPath(config: Record<string, unknown>, key: string): string {
  const value = config[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Out-classify ${key} must be a non-empty path.`);
  }
  return resolve(getAgentDir(), value.trim());
}

export async function loadConfig(): Promise<OutClassifyConfig> {
  const configPath = resolve(getAgentDir(), "out-classify.json");
  let config: Record<string, unknown>;

  try {
    config = JSON.parse(await readFile(configPath, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    // No config file: treat it as empty, so every key gets its default below.
    config = {};
  }

  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new Error("Out-classify config must be a JSON object.");
  }

  return {
    thresholdChars: loadKeyInteger(config, "thresholdChars", DEFAULT_THRESHOLD_CHARS, 1),
    thresholdImages: loadKeyInteger(config, "thresholdImages", DEFAULT_THRESHOLD_IMAGES, 1),
    // The splitters need at least 100 characters per section.
    maxSectionChars: loadKeyInteger(config, "maxSectionChars", DEFAULT_MAX_SECTION_CHARS, 100),
    ...(config.pythonPath === undefined ? {} : { pythonPath: loadKeyPath(config, "pythonPath") }),
  };
}
