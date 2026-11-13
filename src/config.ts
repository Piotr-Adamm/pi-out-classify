import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

export interface OutClassifyConfig {
  thresholdChars: number;
  // Fewer images than this are not classified.
  thresholdImages: number;
  maxSectionChars: number;
  // A Python with Docling to use instead of the venv out-classify installs itself.
  pythonPath?: string;
  // "provider/model" of a vision model that transcribes kept PDF formulas to LaTeX. Without it,
  // formulas are returned as images.
  formulaModel?: { provider: string; id: string };
  // Cached formula transcriptions.
  cacheDir: string;
  // Cache entries unused for this many days are deleted; 0 keeps them forever.
  cacheMaxAgeDays: number;
}

const DEFAULT_THRESHOLD_CHARS = 8000;
const DEFAULT_THRESHOLD_IMAGES = 5;
const DEFAULT_MAX_SECTION_CHARS = 2000;
const CACHE_DIR_NAME = "pi-out-classify";
const DEFAULT_CACHE_MAX_AGE_DAYS = 30;

// The user's cache directory: $XDG_CACHE_HOME if set to an absolute path (XDG Base Directory spec),
// %LOCALAPPDATA% on Windows, otherwise ~/.cache.
function defaultCacheDir(): string {
  const xdg = process.env.XDG_CACHE_HOME;
  if (xdg && isAbsolute(xdg)) return join(xdg, CACHE_DIR_NAME);
  const localAppData = process.env.LOCALAPPDATA;
  if (process.platform === "win32" && localAppData) return join(localAppData, CACHE_DIR_NAME);
  return join(homedir(), ".cache", CACHE_DIR_NAME);
}

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

// Loads a non-empty path key; ~ is the home directory, and relative paths resolve from the agent
// directory, like the config itself.
function loadKeyPath(config: Record<string, unknown>, key: string): string {
  const value = config[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Out-classify ${key} must be a non-empty path.`);
  }
  const path = value.trim().replace(/^~(?=$|[\\/])/, homedir());
  return resolve(getAgentDir(), path);
}

// Loads a "provider/model" key; the model ID may contain further slashes.
function loadKeyModel(config: Record<string, unknown>, key: string): { provider: string; id: string } {
  const value = config[key];
  const slash = typeof value === "string" ? value.indexOf("/") : -1;
  if (typeof value !== "string" || slash < 1 || slash === value.length - 1) {
    throw new Error(`Out-classify ${key} must be "provider/model", e.g. "openai-codex/gpt-6-luna".`);
  }
  return { provider: value.slice(0, slash), id: value.slice(slash + 1) };
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
    ...(config.formulaModel === undefined ? {} : { formulaModel: loadKeyModel(config, "formulaModel") }),
    cacheDir: config.cacheDir === undefined ? defaultCacheDir() : loadKeyPath(config, "cacheDir"),
    cacheMaxAgeDays: loadKeyInteger(config, "cacheMaxAgeDays", DEFAULT_CACHE_MAX_AGE_DAYS, 0),
  };
}
