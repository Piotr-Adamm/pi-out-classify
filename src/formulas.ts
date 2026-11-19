/**
 * @module
 * Transcribes cropped PDF formulas to Typst math with a vision model, cached in the document's cache
 * item, one file per model.
 */

import type { Api, ImageContent, Model, TextContent } from '@earendil-works/pi-ai'
import type { ExtensionContext } from '@earendil-works/pi-coding-agent'
import type { CacheItem } from './cache.ts'
import { FORMULA_THINKING_LEVELS } from './config.ts'
import type { FormulaThinkingLevel } from './config.ts'
import type { Figure } from './extractor.ts'

/**
 * Formulas of one page go into one request, at most this many; more are split into further requests.
 */
const maxFormulasPerRequest = 12
const maxParallelRequests = 8

/**
 * Stored in every transcription file: raise it when the prompt or output format changes, so files
 * with another version are ignored and overwritten.
 */
const formulaCacheVersion = 3

/**
 * A model's transcription file in a document's cache item. Formula IDs are stable for a cached
 * document, so they key the transcriptions; an edited PDF is another document, so its formulas are
 * transcribed again.
 */
interface TranscriptionFile {
  formulaCacheVersion: number
  model: string
  formulas: Record<string, { thinkingLevel: FormulaThinkingLevel; typst: string }>
}

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
  'An image may be followed by the text layer of the PDF for that formula. It loses structure and letter ' +
  'styles (s2 can be s^2, fractions are flattened, a calligraphic L is a plain L), but its symbols are ' +
  'exact: take structure and letter styles from the image, and use the text layer to check which ' +
  'symbols, digits and letters are meant (pi, not 9). ' +
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
  /**
   * The document's cache item; without it, nothing is cached.
   */
  cache: CacheItem | undefined
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
  const model = `${options.model.provider}/${options.model.id}`
  const typst = await cachedTranscriptions(
    formulas,
    options.model,
    options.thinkingLevel,
    options.cache
  )
  const missing = formulas.filter((formula) => !typst.has(formula.id))

  const transcribed = new Map<string, string>()
  try {
    await mapWithLimit(requestGroups(missing), maxParallelRequests, async (group) => {
      let answer: Map<string, string>
      try {
        answer = await transcribeGroup(group, options)
      } catch (error) {
        if (options.signal?.aborted) {
          throw error
        }
        options.onError(error)
        return
      }
      for (const [id, text] of answer) {
        typst.set(id, text)
        transcribed.set(id, text)
      }
    })
  } finally {
    // Also when cancelled: groups that finished need no request next time. A failing cache only
    // costs speed.
    if (options.cache && transcribed.size > 0) {
      await saveTranscriptions(options.cache, model, transcribed, options.thinkingLevel).catch(
        () => {}
      )
    }
  }
  return typst
}

/**
 * Formula ID → Typst math for the formulas whose transcription is stored, at this thinking level or a
 * higher one. Calls no model; nothing without a cache item.
 */
export const cachedTranscriptions = async (
  formulas: Figure[],
  model: Model<Api>,
  thinkingLevel: FormulaThinkingLevel,
  cache: CacheItem | undefined
): Promise<Map<string, string>> => {
  const typst = new Map<string, string>()
  if (!cache) {
    return typst
  }
  const stored = await readTranscriptionFile(cache, `${model.provider}/${model.id}`)
  for (const formula of formulas) {
    const entry = stored[formula.id]
    if (typeof entry?.typst === 'string' && rank(entry.thinkingLevel) >= rank(thinkingLevel)) {
      typst.set(formula.id, entry.typst)
    }
  }
  return typst
}

const rank = (level: unknown): number => FORMULA_THINKING_LEVELS.indexOf(level as never)

/**
 * The model's stored transcriptions; empty if there are none, they are unreadable, or they were made
 * with another formulaCacheVersion or another model (two models whose file names clash).
 */
const readTranscriptionFile = async (
  cache: CacheItem,
  model: string
): Promise<Partial<TranscriptionFile['formulas']>> => {
  const file = (await cache.readTranscriptions(model)) as Partial<TranscriptionFile> | undefined
  if (
    file?.formulaCacheVersion !== formulaCacheVersion ||
    file.model !== model ||
    typeof file.formulas !== 'object' ||
    file.formulas === null
  ) {
    return {}
  }
  return file.formulas
}

/**
 * Merges new transcriptions into the model's file. It is read again right before writing, so what
 * another session stored meanwhile is kept, unless it is the same formula at a lower thinking level.
 */
const saveTranscriptions = async (
  cache: CacheItem,
  model: string,
  transcribed: Map<string, string>,
  thinkingLevel: FormulaThinkingLevel
): Promise<void> => {
  const formulas = { ...(await readTranscriptionFile(cache, model)) }
  for (const [id, typst] of transcribed) {
    const current = formulas[id]
    if (!current || rank(current.thinkingLevel) <= rank(thinkingLevel)) {
      formulas[id] = { thinkingLevel, typst }
    }
  }
  const file: TranscriptionFile = {
    formulaCacheVersion,
    model,
    formulas: formulas as TranscriptionFile['formulas'],
  }
  await cache.writeTranscriptions(model, file)
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
    // The image shows the structure, the text layer the exact symbols (π was misread as 9 without it).
    if (formula.textLayer) {
      content.push({ type: 'text', text: `Text layer: ${formula.textLayer}` })
    }
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
