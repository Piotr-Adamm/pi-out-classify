/**
 * @module
 * Transcribes cropped PDF formulas to Typst math with a vision model, cached per formula image.
 */

import { createHash } from 'node:crypto'
import { mkdir, readdir, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { Api, ImageContent, Model, TextContent } from '@earendil-works/pi-ai'
import type { ExtensionContext } from '@earendil-works/pi-coding-agent'
import { FORMULA_THINKING_LEVELS } from './config.ts'
import type { FormulaThinkingLevel } from './config.ts'
import type { Figure } from './extractor.ts'

/**
 * Formulas of one page go into one request, at most this many; more are split into further requests.
 */
const maxFormulasPerRequest = 12
const maxParallelRequests = 8
const dayMs = 24 * 60 * 60 * 1000

/**
 * Part of every cache key: raise it when the prompt or output format changes, so old entries miss.
 */
const cacheVersion = 2

const systemPrompt =
  'You transcribe images of mathematical formulas into Typst math syntax (the content between $ $). ' +
  'Each image is preceded by its label, "Formula <n>:". Transcribe exactly what is shown, including ' +
  'equation labels, answer labels such as (A) and words (as quoted text, e.g. "The solution is"), without ' +
  'solving, simplifying or correcting anything. Use Typst notation: frac(a, b) or (a)/(b) for fractions, ' +
  'x^2 and e^(pi t) for exponents, u_x for subscripts, sqrt(x), integral_a^b, sum_(n=1)^oo, hat(u), ' +
  '\\ for line breaks and & for alignment. For braces with several cases, use cases() with each row on ' +
  'its own line, ending with a comma, and the condition after &:\ncases(\n  f(x) = 1, & x > 0,\n  ' +
  'f(x) = 0, & x <= 0,\n)\nKeep letter styles: cal(L) for calligraphic, bb(R) for blackboard bold, ' +
  'frak(g) for fraktur, bold(v) for bold. Do not use LaTeX commands such as \\quad or \\{; write quad. ' +
  'Answer with one block per formula, in order:\nFormula <n>:\n' +
  '<Typst math without $ delimiters or code fences>'

type ModelRegistry = ExtensionContext['modelRegistry']

export interface TranscribeOptions {
  registry: ModelRegistry
  model: Model<Api>
  /**
   * Reasoning effort of the formula model; "off" sends none.
   */
  thinkingLevel: FormulaThinkingLevel
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
    const cached = await cache.get(formula.image, options.thinkingLevel)
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
      await cache.set(formula.image, text, options.thinkingLevel).catch(() => {})
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
      {
        ...(options.thinkingLevel === 'off' ? {} : { reasoning: options.thinkingLevel }),
        signal: options.signal,
      }
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
 * Removes what models add despite the instructions (code fences, $ delimiters) and escapes the commas
 * inside cases() rows.
 */
const cleanTypst = (text: string): string => {
  const typst = text
    .trim()
    .replace(/^```[a-z]*\n?|\n?```$/g, '')
    .trim()
  return escapeCasesRows(
    typst.length > 2 && typst.startsWith('$') && typst.endsWith('$')
      ? typst.slice(1, -1).trim()
      : typst
  )
}

/**
 * In cases(), commas separate the rows, but models rarely escape commas inside a row (f(0) = a\, ...).
 * So the prompt asks for one row per line, and in a multi-line cases() every comma directly inside it
 * becomes \, except the one ending a line. Commas in nested calls (frac(a, b)) and quoted text stay;
 * a single-line cases() is left as it is.
 */
const escapeCasesRows = (typst: string): string => {
  let result = ''
  let index = 0
  while (index < typst.length) {
    const start = typst.indexOf('cases(', index)
    if (start === -1) {
      return result + typst.slice(index)
    }
    const bodyStart = start + 'cases('.length
    const bodyEnd = closingParentheses(typst, bodyStart)
    const body = typst.slice(bodyStart, bodyEnd)
    result +=
      typst.slice(index, bodyStart) +
      (body.includes('\n') ? body.split('\n').map(escapeRowCommas).join('\n') : body)
    index = bodyEnd
  }
  return result
}

/**
 * Index of the ) that closes the call whose arguments start at `from`; the text length if unclosed.
 */
const closingParentheses = (typst: string, from: number): number => {
  let depth = 0
  let quoted = false
  for (let i = from; i < typst.length; i++) {
    const char = typst[i]
    if (char === '"' && typst[i - 1] !== '\\') {
      quoted = !quoted
    } else if (!quoted && '([{'.includes(char)) {
      depth++
    } else if (!quoted && ')]}'.includes(char)) {
      if (depth === 0) {
        return i
      }
      depth--
    }
  }
  return typst.length
}

/**
 * Escapes the unescaped commas of one row line outside nested calls and quotes, except a final one.
 */
const escapeRowCommas = (line: string): string => {
  const end = line.trimEnd().length
  let depth = 0
  let quoted = false
  let result = ''
  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    if (char === '"' && line[i - 1] !== '\\') {
      quoted = !quoted
    } else if (!quoted && '([{'.includes(char)) {
      depth++
    } else if (!quoted && ')]}'.includes(char)) {
      depth--
    }
    const inRow = char === ',' && !quoted && depth === 0 && line[i - 1] !== '\\'
    result += inRow && i < end - 1 ? '\\,' : char
  }
  return result
}

interface FormulaCache {
  /**
   * The cached Typst, if it was made with this thinking level or a higher one.
   */
  get: (image: ImageContent, thinkingLevel: FormulaThinkingLevel) => Promise<string | undefined>
  set: (image: ImageContent, typst: string, thinkingLevel: FormulaThinkingLevel) => Promise<void>
  /**
   * Deletes entries unused for maxAgeMs, once per process and directory.
   */
  cleanUp: (maxAgeMs: number) => Promise<void>
}

const cleanedUpDirs = new Set<string>()

/**
 * One file per formula, named by a hash of the cache version, the model and the formula image: moved,
 * renamed or edited PDFs still hit for unchanged formulas, and another model computes its own. Each
 * file holds { thinkingLevel, typst }; a transcription at a lower level than configured is redone and
 * replaced. The files hold document content, so only their owner may read them.
 */
const createCache = (dir: string, model: string): FormulaCache => {
  const pathOf = (image: ImageContent): string => {
    const hash = createHash('sha256')
      .update(`${cacheVersion}\0${model}\0`)
      .update(image.data)
      .digest('hex')
    return join(dir, `${hash}.json`)
  }
  const rank = (level: unknown): number => FORMULA_THINKING_LEVELS.indexOf(level as never)

  return {
    get: async (image, thinkingLevel) => {
      const path = pathOf(image)
      let entry: { thinkingLevel?: unknown; typst?: unknown }
      try {
        entry = JSON.parse(await readFile(path, 'utf8'))
      } catch {
        return undefined
      }
      if (typeof entry?.typst !== 'string' || rank(entry.thinkingLevel) < rank(thinkingLevel)) {
        return undefined
      }
      const now = new Date()
      // Used: its age for cleanUp starts again.
      await utimes(path, now, now).catch(() => {})
      return entry.typst
    },
    set: async (image, typst, thinkingLevel) => {
      await mkdir(dir, { recursive: true, mode: 0o700 })
      await writeFile(pathOf(image), JSON.stringify({ thinkingLevel, typst }), { mode: 0o600 })
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
          .filter((name) => name.endsWith('.json'))
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
