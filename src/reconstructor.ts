import type { ImageContent, TextContent } from '@earendil-works/pi-ai'
import type { OutputSection } from './batcher.ts'
import type { ClassifierResults } from './classifier.ts'
import type { Figure } from './extractor.ts'

type Content = TextContent | ImageContent
type ImageItem = { id: string; contentIndex: number; block: ImageContent }
// Text of a kept section, a kept image, or a removed section (becomes a marker in merge).
type Piece = string | ImageContent | OutputSection

export interface ReconstructionInput {
  sections: OutputSection[] // As classified: batched order, split pieces included.
  results: ClassifierResults // Missing ID = not classified → kept.
  document?: { markdown: string; figures: Figure[] }
  images: ImageItem[] // Non-documents only.
  readOffset?: number // First file line of the output (read's offset); default 1.
  // Formula ID → Typst math for transcribed formulas; the others are returned as images, like figures.
  formulaText?: Map<string, string>
}

const PAGE_BREAK = '<!-- page break -->'
const PAGE_BREAK_AT_END = /<!-- page break -->$/
const NOTHING_RELEVANT = '[out-classify: nothing in this output was deemed relevant to the task.]'

export function reconstruct(input: ReconstructionInput): Content[] {
  const { sections, results, document } = input
  const kept = (id: string) => results.get(id) !== 'discard'
  // Formulas are not classified themselves; they follow their section.
  const imageIds = document
    ? document.figures.filter((figure) => figure.kind === 'figure').map((figure) => figure.id)
    : input.images.map((image) => image.id)

  const nothingKept = !sections.some((section) => kept(section.id)) && !imageIds.some(kept)
  if (nothingKept) {
    return [{ type: 'text', text: NOTHING_RELEVANT }]
  }

  const content = document
    ? reconstructDocument(
        sections,
        document.figures,
        document.markdown,
        kept,
        input.formulaText ?? new Map()
      )
    : reconstructOutput(sections, input.images, input.readOffset ?? 1, kept)
  return [...content, ...agentNotes(sections, imageIds, results)]
}

// Replaces the links of transcribed formulas with their Typst math, as a display equation.
export function inlineFormulas(text: string, formulaText: Map<string, string>): string {
  return text.replace(/!\[Formula\]\((formula-\d+)\)/g, (link, id: string) => {
    const typst = formulaText.get(id)
    return typst === undefined ? link : `$ ${typst} $`
  })
}

// Formulas linked from a kept section, in document order; formulas in removed sections are dropped.
export function keptFormulas(
  figures: Figure[],
  sections: OutputSection[],
  results: ClassifierResults
): Figure[] {
  return figures.filter(
    (figure) =>
      figure.kind === 'formula' &&
      sections.some(
        (section) =>
          results.get(section.id) !== 'discard' && section.text.includes(`](${figure.id})`)
      )
  )
}

// Sections in order; each followed by the kept figures it links to, even if the section is removed.
// Formulas follow their section: in a kept one, transcribed formulas stand in place of their link and
// the others follow it as images; in a removed one, they are dropped with it.
function reconstructDocument(
  sections: OutputSection[],
  figures: Figure[],
  markdown: string,
  kept: (id: string) => boolean,
  formulaText: Map<string, string>
): Content[] {
  const pieces: Piece[] = []
  for (const section of sections) {
    const sectionKept = kept(section.id)
    pieces.push(sectionKept ? inlineFormulas(section.text, formulaText) : section)
    for (const figure of figures) {
      if (!section.text.includes(`](${figure.id})`)) {
        continue
      }
      const keptFigure =
        figure.kind === 'formula' ? sectionKept && !formulaText.has(figure.id) : kept(figure.id)
      if (keptFigure) {
        pieces.push(figure.image)
      }
    }
  }
  return merge(pieces, (run) => pageMarker(run, markdown))
}

// Sections and images back in their original order (contentIndex).
function reconstructOutput(
  sections: OutputSection[],
  images: ImageItem[],
  readOffset: number,
  kept: (id: string) => boolean
): Content[] {
  const items = [...sections, ...images].sort((a, b) => a.contentIndex! - b.contentIndex!)
  const pieces: Piece[] = []
  for (const item of items) {
    if ('block' in item) {
      if (kept(item.id)) {
        pieces.push(item.block)
      }
    } else {
      pieces.push(kept(item.id) ? item.text : item)
    }
  }
  return merge(pieces, (run) => lineMarker(run, sections, readOffset))
}

// Adjacent text joins into one block; an image ends the block. A run (consecutive removed
// sections) becomes one marker line.
function merge(pieces: Piece[], marker: (run: OutputSection[]) => string): Content[] {
  const content: Content[] = []
  let text = ''
  let run: OutputSection[] = []

  const endRun = () => {
    if (run.length === 0) {
      return
    }
    if (text && !text.endsWith('\n')) {
      text += '\n'
    }
    text += `${marker(run)}\n`
    run = []
  }

  for (const piece of pieces) {
    if (typeof piece === 'string') {
      endRun()
      text += piece
    } else if ('type' in piece) {
      endRun()
      if (text) {
        content.push({ type: 'text', text })
      }
      text = ''
      content.push(piece)
    } else {
      run.push(piece)
    }
  }
  endRun()
  if (text) {
    content.push({ type: 'text', text })
  }
  return content
}

// "lines 120–184": counted within the section's own text block, shifted by read's offset.
function lineMarker(run: OutputSection[], sections: OutputSection[], readOffset: number): string {
  const firstLine = (section: OutputSection) =>
    readOffset +
    sections
      .filter((other) => other.contentIndex === section.contentIndex && other.start < section.start)
      .reduce((lines, other) => lines + newlines(other.text), 0)
  const last = run[run.length - 1]
  const from = firstLine(run[0])
  const to = firstLine(last) + newlines(last.text.replace(/\n$/, ''))
  const lines = from === to ? `line ${from}` : `lines ${from}–${to}`
  return `[out-classify: ${lines} omitted, deemed not relevant to the task]`
}

// "pages 3–5 (## A, ## B)": page = 1 + page breaks before; headings = first heading of each section.
function pageMarker(run: OutputSection[], markdown: string): string {
  const pageAt = (offset: number) => markdown.slice(0, offset).split(PAGE_BREAK).length
  const last = run[run.length - 1]
  const from = pageAt(run[0].start)
  const to = pageAt(last.start + last.text.trimEnd().replace(PAGE_BREAK_AT_END, '').length)
  const pages = from === to ? `page ${from}` : `pages ${from}–${to}`
  const headings = run.map((section) => section.text.match(/^#{1,6} .*/m)?.[0]).filter(Boolean)
  const where = headings.length > 0 ? `${pages} (${headings.join(', ')})` : pages
  return `[out-classify: text omitted on ${where}, deemed not relevant to the task]`
}

// At most one note per kind: unclassified parts are included unfiltered.
function agentNotes(
  sections: OutputSection[],
  imageIds: string[],
  results: ClassifierResults
): TextContent[] {
  const classified = sections.filter((section) => results.has(section.id))
  const unclassifiedSections = classified.filter(
    (section) => results.get(section.id) === 'unclassified'
  ).length
  const unclassifiedImages = imageIds.filter((id) => results.get(id) === 'unclassified').length

  const notes: string[] = []
  if (unclassifiedSections > 0) {
    notes.push(
      `[out-classify: ${unclassifiedSections} of ${classified.length} sections could not be classified ` +
        '(TypeSafe error) and are included unfiltered.]'
    )
  }
  if (unclassifiedImages > 0) {
    notes.push(
      `[out-classify: ${unclassifiedImages} images could not be classified and are included unfiltered.]`
    )
  }
  return notes.length > 0 ? [{ type: 'text', text: notes.join('\n') }] : []
}

function newlines(text: string): number {
  return text.split('\n').length - 1
}
