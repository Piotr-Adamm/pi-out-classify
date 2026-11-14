import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { isAbsolute, join, resolve } from 'node:path'
import { getAgentDir } from '@earendil-works/pi-coding-agent'

export interface OutClassifyConfig {
  thresholdChars: number
  // Fewer images than this are not classified.
  thresholdImages: number
  maxSectionChars: number
  // A Python with Docling to use instead of the venv out-classify installs itself.
  pythonPath?: string
  // "provider/model" of a vision model that transcribes kept PDF formulas to Typst math. Without it,
  // formulas are returned as images.
  formulaModel?: { provider: string; id: string }
  // Reasoning effort of the formula model; "off" sends none.
  formulaThinkingLevel: FormulaThinkingLevel
  // Cached formula transcriptions.
  cacheDir: string
  // Cache entries unused for this many days are deleted; 0 keeps them forever.
  cacheMaxAgeDays: number
}

const DEFAULT_THRESHOLD_CHARS = 8000
const DEFAULT_THRESHOLD_IMAGES = 5
const DEFAULT_MAX_SECTION_CHARS = 2000
const CACHE_DIR_NAME = 'pi-out-classify'
const DEFAULT_CACHE_MAX_AGE_DAYS = 30
// Lowest to highest; the formula cache compares levels by this order.
export const FORMULA_THINKING_LEVELS = [
  'off',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
] as const
export type FormulaThinkingLevel = (typeof FORMULA_THINKING_LEVELS)[number]
// Transcription barely gains from reasoning, but it adds latency.
const DEFAULT_FORMULA_THINKING_LEVEL: FormulaThinkingLevel = 'minimal'

// The user's cache directory: $XDG_CACHE_HOME if set to an absolute path (XDG Base Directory spec),
// %LOCALAPPDATA% on Windows, otherwise ~/.cache.
function defaultCacheDir(): string {
  const xdg = process.env.XDG_CACHE_HOME
  if (xdg && isAbsolute(xdg)) {
    return join(xdg, CACHE_DIR_NAME)
  }
  const localAppData = process.env.LOCALAPPDATA
  if (process.platform === 'win32' && localAppData) {
    return join(localAppData, CACHE_DIR_NAME)
  }
  return join(homedir(), '.cache', CACHE_DIR_NAME)
}

// Loads an integer key from the config, using the default when missing and requiring at least `min`.
function loadKeyInteger(
  config: Record<string, unknown>,
  key: string,
  fallback: number,
  min: number
): number {
  const value = config[key] === undefined ? fallback : config[key]

  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min) {
    throw new Error(`Out-classify ${key} must be an integer of at least ${min}.`)
  }

  return value
}

// Loads a non-empty path key; ~ is the home directory, and relative paths resolve from the agent
// directory, like the config itself.
function loadKeyPath(config: Record<string, unknown>, key: string): string {
  const value = config[key]
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`Out-classify ${key} must be a non-empty path.`)
  }
  const path = value.trim().replace(/^~(?=$|[\\/])/, homedir())
  return resolve(getAgentDir(), path)
}

// Loads a "provider/model" key; the model ID may contain further slashes.
function loadKeyModel(
  config: Record<string, unknown>,
  key: string
): { provider: string; id: string } {
  const value = config[key]
  const slash = typeof value === 'string' ? value.indexOf('/') : -1
  if (typeof value !== 'string' || slash < 1 || slash === value.length - 1) {
    throw new Error(`Out-classify ${key} must be "provider/model", e.g. "openai-codex/gpt-6-luna".`)
  }
  return { provider: value.slice(0, slash), id: value.slice(slash + 1) }
}

// Loads a key that must be one of `choices`, using the default when missing.
function loadKeyChoice<T extends string>(
  config: Record<string, unknown>,
  key: string,
  choices: readonly T[],
  fallback: T
): T {
  const value = config[key] === undefined ? fallback : config[key]
  if (!choices.includes(value as T)) {
    const list = choices.map((choice) => `"${choice}"`).join(', ')
    throw new Error(`Out-classify ${key} must be one of ${list}.`)
  }
  return value as T
}

export async function loadConfig(): Promise<OutClassifyConfig> {
  const configPath = resolve(getAgentDir(), 'out-classify.json')
  let config: Record<string, unknown>

  try {
    config = JSON.parse(await readFile(configPath, 'utf8'))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error
    }
    // No config file: treat it as empty, so every key gets its default below.
    config = {}
  }

  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('Out-classify config must be a JSON object.')
  }

  return {
    thresholdChars: loadKeyInteger(config, 'thresholdChars', DEFAULT_THRESHOLD_CHARS, 1),
    thresholdImages: loadKeyInteger(config, 'thresholdImages', DEFAULT_THRESHOLD_IMAGES, 1),
    // The splitters need at least 100 characters per section.
    maxSectionChars: loadKeyInteger(config, 'maxSectionChars', DEFAULT_MAX_SECTION_CHARS, 100),
    ...(config.pythonPath === undefined ? {} : { pythonPath: loadKeyPath(config, 'pythonPath') }),
    ...(config.formulaModel === undefined
      ? {}
      : { formulaModel: loadKeyModel(config, 'formulaModel') }),
    formulaThinkingLevel: loadKeyChoice(
      config,
      'formulaThinkingLevel',
      FORMULA_THINKING_LEVELS,
      DEFAULT_FORMULA_THINKING_LEVEL
    ),
    cacheDir: config.cacheDir === undefined ? defaultCacheDir() : loadKeyPath(config, 'cacheDir'),
    cacheMaxAgeDays: loadKeyInteger(config, 'cacheMaxAgeDays', DEFAULT_CACHE_MAX_AGE_DAYS, 0),
  }
}
