/**
 * @module
 * Transcribes cropped PDF formulas to Typst math with a vision model, cached per formula image.
 */

import { createHash } from 'node:crypto'
import { mkdir, readdir, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { Api, ImageContent, Model, TextContent } from '@earendil-works/pi-ai'
import type { ExtensionContext } from '@earendil-works/pi-coding-agent'
import type { Figure } from './extractor.ts'

/**
 * Formulas of one page go into one request, at most this many; more are split into further requests.
 */
const maxFormulasPerRequest = 12
const maxParallelRequests = 8
const dayMs = 24 * 60 * 60 * 1000

/**
 * Part of every cache key: change it when the prompt or output format changes, so old entries miss.
 */
const formatVersion = 'typst-1'

const systemPrompt =
  'You transcribe images of mathematical formulas into Typst math syntax (the content between $ $). ' +
  'Each image is preceded by its label, "Formula <n>:". Transcribe exactly what is shown, including ' +
  'equation labels, answer labels such as (A) and words (as quoted text, e.g. "The solution is"), without ' +
  'solving, simplifying or correcting anything. Use Typst notation: frac(a, b) or (a)/(b) for fractions, ' +
  'x^2 and e^(pi t) for exponents, u_x for subscripts, sqrt(x), integral_a^b, sum_(n=1)^oo, hat(u), ' +
  'cases(..., ...) for braces with several cases, \\ for line breaks and & for alignment. Answer with one ' +
  'block per formula, in order:\nFormula <n>:\n<Typst math without $ delimiters or code fences>'

type ModelRegistry = ExtensionContext['modelRegistry']

export interface TranscribeOptions {
  registry: ModelRegistry
  model: Model<Api>
  cacheDir: string
  /**
   * Cache entries unused for this many days are deleted; 0 keeps them forever.
   */
  cacheMaxAgeDays: number
  signal: AbortSignal | undefined
  /**
   * Gets the error of a failed request; its formulas stay untranscribed (they are returned as images).
   */
  onError: (error: unknown) => void
}

/**
 * Formula ID → Typst math for the formulas that could be transcribed, from the cache or the model.
 * Only cancellation (Esc) throws.
 */
export const transcribeFormulas = async (
  formulas: Figure[],
  options: TranscribeOptions
): Promise<Map<string, string>> => {
  const typst = new Map<string, string>()
  const cache = createCache(options.cacheDir, `${options.model.provider}/${options.model.id}`)

  const missing: Figure[] = []
  for (const formula of formulas) {
    const cached = await cache.get(formula.image)
    if (cached === undefined) {
      missing.push(formula)
    } else {
      typst.set(formula.id, cached)
    }
  }

  await mapWithLimit(requestGroups(missing), maxParallelRequests, async (group) => {
    let transcribed: Map<string, string>
    try {
      transcribed = await transcribeGroup(group, options)
    } catch (error) {
      if (options.signal?.aborted) {
        throw error
      }
      options.onError(error)
      return
    }
    for (const formula of group) {
      const text = transcribed.get(formula.id)
      if (text === undefined) {
        continue
      }
      typst.set(formula.id, text)
      // A failing cache only costs speed.
      await cache.set(formula.image, text).catch(() => {})
    }
  })

  if (options.cacheMaxAgeDays > 0) {
    await cache.cleanUp(options.cacheMaxAgeDays * dayMs)
  }
  return typst
}

/**
 * Formulas grouped by page, in document order, each group at most maxFormulasPerRequest.
 */
const requestGroups = (formulas: Figure[]): Figure[][] => {
  const byPage = new Map<number | null, Figure[]>()
  for (const formula of formulas) {
    byPage.set(formula.page, [...(byPage.get(formula.page) ?? []), formula])
  }
  const groups: Figure[][] = []
  for (const page of byPage.values()) {
    for (let i = 0; i < page.length; i += maxFormulasPerRequest) {
      groups.push(page.slice(i, i + maxFormulasPerRequest))
    }
  }
  return groups
}

/**
 * One request for one group. Formulas missing from the answer are left out of the result.
 */
const transcribeGroup = async (
  group: Figure[],
  options: TranscribeOptions
): Promise<Map<string, string>> => {
  const content: (TextContent | ImageContent)[] = [
    { type: 'text', text: `Transcribe these ${group.length} formulas.` },
  ]
  group.forEach((formula, index) => {
    content.push({ type: 'text', text: `Formula ${index + 1}:` })
    content.push(formula.image)
  })

  const message = await options.registry
    .streamSimple(
      options.model,
      { systemPrompt, messages: [{ role: 'user', content, timestamp: Date.now() }] },
      { reasoning: 'minimal', signal: options.signal }
    )
    .result()
  if (message.stopReason === 'error' || message.stopReason === 'aborted') {
    throw new Error(message.errorMessage ?? `formula transcription ${message.stopReason}`)
  }

  const answer = message.content
    .filter((block) => block.type === 'text')
    .map((block) => block.text)
    .join('')
  const result = new Map<string, string>()
  // "Formula <n>:" starts each block; the Typst follows on the same or the next lines.
  const parts = answer.split(/^[ \t]*\**Formula (\d+):\**[ \t]*/m)
  for (let i = 1; i < parts.length; i += 2) {
    const formula = group[Number(parts[i]) - 1]
    const typst = cleanTypst(parts[i + 1] ?? '')
    if (formula && typst && !result.has(formula.id)) {
      result.set(formula.id, typst)
    }
  }
  return result
}

/**
 * Removes what models add despite the instructions: code fences and $ delimiters.
 */
const cleanTypst = (text: string): string => {
  const typst = text
    .trim()
    .replace(/^```[a-z]*\n?|\n?```$/g, '')
    .trim()
  return typst.length > 2 && typst.startsWith('$') && typst.endsWith('$')
    ? typst.slice(1, -1).trim()
    : typst
}

interface FormulaCache {
  get: (image: ImageContent) => Promise<string | undefined>
  set: (image: ImageContent, typst: string) => Promise<void>
  /**
   * Deletes entries unused for maxAgeMs, once per process and directory.
   */
  cleanUp: (maxAgeMs: number) => Promise<void>
}

const cleanedUpDirs = new Set<string>()

/**
 * One file per formula, named by a hash of the format, the model and the formula image: moved,
 * renamed or edited PDFs still hit for unchanged formulas, and another model computes its own. The
 * files hold document content, so only their owner may read them.
 */
const createCache = (dir: string, model: string): FormulaCache => {
  const pathOf = (image: ImageContent): string => {
    const hash = createHash('sha256')
      .update(`${formatVersion}\0${model}\0`)
      .update(image.data)
      .digest('hex')
    return join(dir, `${hash}.typ`)
  }

  return {
    get: async (image) => {
      const path = pathOf(image)
      try {
        const typst = await readFile(path, 'utf8')
        const now = new Date()
        // Used: its age for cleanUp starts again.
        await utimes(path, now, now).catch(() => {})
        return typst
      } catch {
        return undefined
      }
    },
    set: async (image, typst) => {
      await mkdir(dir, { recursive: true, mode: 0o700 })
      await writeFile(pathOf(image), typst, { mode: 0o600 })
    },
    cleanUp: async (maxAgeMs) => {
      if (cleanedUpDirs.has(dir)) {
        return
      }
      cleanedUpDirs.add(dir)
      const names = await readdir(dir).catch((): string[] => [])
      const cutoff = Date.now() - maxAgeMs
      await Promise.all(
        names
          .filter((name) => name.endsWith('.typ'))
          .map(async (name) => {
            const path = join(dir, name)
            const info = await stat(path).catch(() => undefined)
            if (info && info.mtimeMs < cutoff) {
              await rm(path, { force: true })
            }
          })
      )
    },
  }
}

const mapWithLimit = async <T>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<void>
): Promise<void> => {
  let next = 0
  const worker = async (): Promise<void> => {
    while (next < items.length) {
      await fn(items[next++])
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
}
