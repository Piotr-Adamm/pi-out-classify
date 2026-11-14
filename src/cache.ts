/**
 * @module
 * The document cache: one folder per extracted document, keyed by a hash of the PDF's content, holding
 * the extraction and the formula transcriptions made for it:
 *
 *   <cacheDir>/<key>/document.json               Markdown and figure metadata
 *   <cacheDir>/<key>/figures/<id>.png            figure and formula images
 *   <cacheDir>/<key>/transcriptions/<model>.json formula transcriptions of one model
 *
 * Items are written to <key>.tmp-<random hex> and renamed, so an item under its key is always
 * complete. The files hold document content, so only their owner may read them.
 */

import { createHash, randomBytes } from 'node:crypto'
import { mkdir, readdir, readFile, rename, rm, stat, utimes, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type { ExtractedDocument, Figure } from './extractor.ts'

const DOCUMENT_FILE = 'document.json'
const FIGURES_DIR = 'figures'
const TRANSCRIPTIONS_DIR = 'transcriptions'
const DAY_MS = 24 * 60 * 60 * 1000
// A temporary folder older than this was left by a crashed or killed session.
const STALE_TEMP_MS = 60 * 60 * 1000
// Only these names are ever deleted, so a cacheDir shared with other files is safe.
const ITEM_NAME = /^[0-9a-f]{64}$/
const TEMP_NAME = /^[0-9a-f]{64}\.tmp-[0-9a-f]+$/
const FIGURE_ID = /^(fig|formula)-\d+$/

// A figure in document.json: everything but the image, which is figures/<id>.png.
type StoredFigure = Omit<Figure, 'image'>

export interface CacheItem {
  // The model's transcription file as parsed JSON; undefined if there is none or it is unreadable.
  readTranscriptions: (model: string) => Promise<unknown>
  // Replaces the model's transcription file (atomically) and updates the item's modification date.
  writeTranscriptions: (model: string, value: unknown) => Promise<void>
}

// SHA-256 of the parts (versions the content was made with) and the content.
export function cacheKey(content: Uint8Array, parts: string[]): string {
  const hash = createHash('sha256')
  for (const part of parts) {
    hash.update(`${part}\0`)
  }
  return hash.update(content).digest('hex')
}

// The cached document under key, or undefined if there is none. A damaged item (unreadable
// document.json, missing image) is deleted, so it is extracted and stored again.
export async function loadDocument(
  cacheDir: string,
  key: string
): Promise<{ document: ExtractedDocument; item: CacheItem } | undefined> {
  const dir = join(cacheDir, key)
  let stored: { markdown?: unknown; figures?: unknown }
  try {
    stored = JSON.parse(await readFile(join(dir, DOCUMENT_FILE), 'utf8'))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      await rm(dir, { recursive: true, force: true }).catch(() => {})
    }
    return undefined
  }

  try {
    if (typeof stored.markdown !== 'string' || !Array.isArray(stored.figures)) {
      throw new Error('invalid document.json')
    }
    const figures = await Promise.all(
      (stored.figures as StoredFigure[]).map(async (figure): Promise<Figure> => {
        if (!FIGURE_ID.test(String(figure?.id))) {
          throw new Error('invalid figure')
        }
        const data = await readFile(join(dir, FIGURES_DIR, `${figure.id}.png`))
        return {
          ...figure,
          image: { type: 'image', data: data.toString('base64'), mimeType: 'image/png' },
        }
      })
    )
    await updateModificationDate(dir)
    return { document: { markdown: stored.markdown, figures }, item: createCacheItem(dir) }
  } catch {
    await rm(dir, { recursive: true, force: true }).catch(() => {})
    return undefined
  }
}

// Stores the document under key and returns its item; undefined if it could not be written (a
// failing cache only costs speed). If another session stored it first, that item is used.
export async function storeDocument(
  cacheDir: string,
  key: string,
  document: ExtractedDocument
): Promise<CacheItem | undefined> {
  const dir = join(cacheDir, key)
  // Random, so two sessions storing the same PDF at once never write into the same folder.
  const temp = `${dir}.tmp-${randomBytes(4).toString('hex')}`
  try {
    await mkdir(join(temp, FIGURES_DIR), { recursive: true, mode: 0o700 })
    await Promise.all(
      document.figures.map((figure) =>
        writeFile(
          join(temp, FIGURES_DIR, `${figure.id}.png`),
          Buffer.from(figure.image.data, 'base64'),
          { mode: 0o600 }
        )
      )
    )
    const figures: StoredFigure[] = document.figures.map(({ image: _image, ...figure }) => figure)
    await writeFile(
      join(temp, DOCUMENT_FILE),
      JSON.stringify({ markdown: document.markdown, figures }),
      { mode: 0o600 }
    )
    try {
      await rename(temp, dir)
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code
      if (code !== 'EEXIST' && code !== 'ENOTEMPTY') {
        throw error
      }
      await rm(temp, { recursive: true, force: true })
    }
    return createCacheItem(dir)
  } catch {
    await rm(temp, { recursive: true, force: true }).catch(() => {})
    return undefined
  }
}

const cleanedUpDirs = new Set<string>()

// Once per process and cacheDir: deletes items not modified for maxAgeDays (0 keeps them) and
// temporary folders left by crashed sessions.
export async function cleanUpCache(cacheDir: string, maxAgeDays: number): Promise<void> {
  if (cleanedUpDirs.has(cacheDir)) {
    return
  }
  cleanedUpDirs.add(cacheDir)
  const names = await readdir(cacheDir).catch((): string[] => [])
  const now = Date.now()
  await Promise.all(
    names.map(async (name) => {
      const maxAgeMs = TEMP_NAME.test(name)
        ? STALE_TEMP_MS
        : ITEM_NAME.test(name) && maxAgeDays > 0
          ? maxAgeDays * DAY_MS
          : undefined
      if (maxAgeMs === undefined) {
        return
      }
      const path = join(cacheDir, name)
      const info = await stat(path).catch(() => undefined)
      if (info?.isDirectory() && now - info.mtimeMs > maxAgeMs) {
        await rm(path, { recursive: true, force: true }).catch(() => {})
      }
    })
  )
}

function createCacheItem(dir: string): CacheItem {
  // Model IDs may contain / or :; the file also holds the model, so a clash is detected on reading.
  const transcriptionPath = (model: string) =>
    join(dir, TRANSCRIPTIONS_DIR, `${model.replace(/[^\w.-]/g, '_')}.json`)

  return {
    readTranscriptions: async (model) => {
      try {
        return JSON.parse(await readFile(transcriptionPath(model), 'utf8'))
      } catch {
        return undefined
      }
    },
    writeTranscriptions: async (model, value) => {
      const path = transcriptionPath(model)
      const temp = `${path}.tmp-${randomBytes(4).toString('hex')}`
      await mkdir(join(dir, TRANSCRIPTIONS_DIR), { recursive: true, mode: 0o700 })
      try {
        await writeFile(temp, JSON.stringify(value), { mode: 0o600 })
        await rename(temp, path)
      } catch (error) {
        await rm(temp, { force: true }).catch(() => {})
        throw error
      }
      // Writing inside transcriptions/ does not change the item folder's own modification date.
      await updateModificationDate(dir)
    },
  }
}

// Sets the item folder's modification date to now: cleanUpCache counts maxAgeDays from the last use,
// not from creation.
async function updateModificationDate(dir: string): Promise<void> {
  const now = new Date()
  await utimes(dir, now, now).catch(() => {})
}
