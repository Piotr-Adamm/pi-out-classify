import type { Figure } from "./extractor.ts";

export interface OutputSection {
  id: string;
  // Offsets into all text items joined in order (see splitTextBuffer), so IDs are unique.
  start: number;
  end: number;
  text: string;
  // Position of the source block in event.content, for the rebuild. Unset for documents,
  // whose extracted text replaces the whole output.
  contentIndex?: number;
}

// One piece of text in the buffer: a text block of the tool output, or an extracted document.
export interface TextItem {
  text: string;
  // Documents and .md reads use the Markdown splitter, everything else the line splitter.
  markdown: boolean;
  contentIndex?: number;
}

// Everything the hook collects for text classification, in output order. Filled in the hook's
// tool_result: one item per text block, or a single item for an extracted document.
export type TextBuffer = TextItem[];

export interface OutputBatch {
  sections: OutputSection[];
  estimatedTokens: number;
}

// Smallest section size, for the splitters and for re-splitting oversized sections.
const MIN_SECTION_CHARS = 100;

function checkSplitArguments(maxSectionChars: number, startOffset: number): void {
  if (!Number.isSafeInteger(maxSectionChars) || maxSectionChars < MIN_SECTION_CHARS) {
    throw new Error(`maxSectionChars must be an integer of at least ${MIN_SECTION_CHARS}.`);
  }
  if (!Number.isSafeInteger(startOffset) || startOffset < 0) {
    throw new Error("startOffset must be a non-negative integer.");
  }
}

// Offsets and limits use JavaScript string units, not model tokens.
export function splitIntoSections(
  toolOutput: string,
  maxSectionChars: number,
  // Position of this text in the original output, including during refinement.
  startOffset = 0,
): OutputSection[] {
  checkSplitArguments(maxSectionChars, startOffset);

  const sections: OutputSection[] = [];
  let start = 0;

  while (start < toolOutput.length) {
    let end = Math.min(start + maxSectionChars, toolOutput.length);

    if (end < toolOutput.length) {
      // Prefer a complete line; hard-split lines longer than the limit.
      const newline = toolOutput.lastIndexOf("\n", end - 1);
      if (newline >= start) {
        end = newline + 1;
      }

      // Keep CRLF endings and UTF-16 surrogate pairs together.
      if (toolOutput[end - 1] === "\r" && toolOutput[end] === "\n") {
        end--;
      }
      const previous = toolOutput.charCodeAt(end - 1);
      const next = toolOutput.charCodeAt(end);
      if (
        previous >= 0xd800 && previous <= 0xdbff &&
        next >= 0xdc00 && next <= 0xdfff
      ) {
        end--;
      }
    }

    const sectionStart = startOffset + start;
    const sectionEnd = startOffset + end;
    sections.push({
      id: `section_${sectionStart}_${sectionEnd}`,
      start: sectionStart,
      end: sectionEnd,
      text: toolOutput.slice(start, end),
    });

    start = end;
  }

  return sections;
}

const HEADING = /^#{1,6}\s/;
const CODE_FENCE = /^(```|~~~)/;

// Markdown only: sections start at headings and otherwise follow paragraphs.
export function splitMarkdownIntoSections(
  markdown: string,
  maxSectionChars: number,
  startOffset = 0,
): OutputSection[] {
  checkSplitArguments(maxSectionChars, startOffset);

  // Find where blocks start: headings and paragraphs after blank lines.
  // Lines inside code fences never start a block, so `# comment` lines stay code.
  const blocks: { start: number; heading: boolean; image: boolean }[] = [];
  let inFence = false;
  let afterBlank = true;
  let offset = 0;
  for (const line of markdown.split(/(?<=\n)/)) {
    const content = line.trim();
    if (inFence) {
      if (CODE_FENCE.test(content)) inFence = false;
    } else if (!content) {
      afterBlank = true;
    } else {
      const heading = HEADING.test(content);
      if (afterBlank || heading) {
        blocks.push({ start: offset, heading, image: content.startsWith("![") });
      }
      if (CODE_FENCE.test(content)) inFence = true;
      // The line after a heading always starts its own block.
      afterBlank = heading;
    }
    offset += line.length;
  }

  // Group blocks into section ranges. A section ends at a new topic or the size limit.
  const ranges: [number, number][] = [];
  let sectionStart = 0;
  let sectionIsEmpty = true; // Only headings so far.

  for (const [i, block] of blocks.entries()) {
    const blockEnd = blocks[i + 1]?.start ?? markdown.length;

    const isFirstInSection = block.start === sectionStart;
    const startsNewTopic = block.heading && !sectionIsEmpty;
    const wouldExceedLimit = blockEnd - sectionStart > maxSectionChars;

    // Figure links never start a section, so they stay with the caption above them.
    if (!isFirstInSection && !block.image && (startsNewTopic || wouldExceedLimit)) {
      ranges.push([sectionStart, block.start]);
      sectionStart = block.start;
      sectionIsEmpty = true;
    }

    if (!block.heading) sectionIsEmpty = false;
  }
  if (sectionStart < markdown.length) ranges.push([sectionStart, markdown.length]);

  // Ranges still over the limit fall back to line-based splitting.
  return ranges.flatMap(([start, end]) => {
    const text = markdown.slice(start, end);
    return text.length > maxSectionChars
      ? splitIntoSections(text, maxSectionChars, startOffset + start)
      : [{ id: `section_${startOffset + start}_${startOffset + end}`, start: startOffset + start, end: startOffset + end, text }];
  });
}

// Splits every item with its splitter and keeps a running offset across items,
// so sections of different blocks never share offsets or IDs.
export function splitTextBuffer(buffer: TextBuffer, maxSectionChars: number): OutputSection[] {
  const sections: OutputSection[] = [];
  let offset = 0;

  for (const item of buffer) {
    const split = item.markdown
      ? splitMarkdownIntoSections(item.text, maxSectionChars, offset)
      : splitIntoSections(item.text, maxSectionChars, offset);

    for (const section of split) {
      sections.push(item.contentIndex === undefined ? section : { ...section, contentIndex: item.contentIndex });
    }
    offset += item.text.length;
  }

  return sections;
}

export interface FigureWithContext extends Figure {
  // Text of the section that links to the figure; only set when there is no caption.
  context?: string;
}

export function addFigureContext(figures: Figure[], sections: OutputSection[]): FigureWithContext[] {
  return figures.map((figure) => ({
    ...figure,
    // Without a caption, the section around the figure link is its only description.
    context: figure.caption
      ? undefined
      : sections.find((section) => section.text.includes(`](${figure.id})`))?.text,
  }));
}

interface EstimatedSection {
  section: OutputSection;
  tokens: number;
}

// Only for a section too big for even an empty batch: halve it until every piece fits.
// All pieces are kept, in order, in place of the original. They keep absolute offsets
// (unique IDs) and any extra fields of the original section.
function fitSection(
  section: OutputSection,
  maxSectionTokens: number,
  estimate: (section: OutputSection) => number,
): EstimatedSection[] {
  const tokens = estimate(section);
  if (tokens <= maxSectionTokens) return [{ section, tokens }];

  const half = Math.floor(section.text.length / 2);
  if (half < MIN_SECTION_CHARS) {
    // Only reachable when the base (purpose, tool arguments) leaves almost no room.
    throw new Error(
      `Section ${section.id} still exceeds the batch budget at ${MIN_SECTION_CHARS} characters; ` +
      "the request base (purpose, tool arguments) is too large.",
    );
  }

  return splitIntoSections(section.text, half, section.start)
    .flatMap((piece) => fitSection({ ...section, ...piece }, maxSectionTokens, estimate));
}

// A batch costs baseTokens (sent with every request) plus the tokens of each of its sections.
// Each section is estimated once; the batch total is a running sum. Sections are never cut to
// fill a batch: if the next one does not fit, the batch is closed and the next one starts.
export function batchSections(
  sections: OutputSection[],
  maxBatchTokens: number,
  baseTokens: number,
  estimateSectionTokens: (section: OutputSection) => number,
): OutputBatch[] {
  if (!Number.isSafeInteger(maxBatchTokens) || maxBatchTokens <= 0) {
    throw new Error("maxBatchTokens must be a positive integer.");
  }
  if (!Number.isSafeInteger(baseTokens) || baseTokens < 0) {
    throw new Error("baseTokens must be a non-negative integer.");
  }
  if (baseTokens >= maxBatchTokens) {
    throw new Error("The request base (purpose, tool arguments) alone exceeds the batch budget.");
  }

  const estimate = (section: OutputSection): number => {
    const tokens = estimateSectionTokens(section);
    if (!Number.isSafeInteger(tokens) || tokens < 0) {
      throw new Error("The section token estimate must be a non-negative integer.");
    }
    return tokens;
  };

  const batches: OutputBatch[] = [];
  let pending: OutputSection[] = [];
  let pendingTokens = baseTokens;

  for (const original of sections) {
    for (const { section, tokens } of fitSection(original, maxBatchTokens - baseTokens, estimate)) {
      // Close the batch if this section would push it over the budget.
      if (pending.length > 0 && pendingTokens + tokens > maxBatchTokens) {
        batches.push({ sections: pending, estimatedTokens: pendingTokens });
        pending = [];
        pendingTokens = baseTokens;
      }

      pending.push(section);
      pendingTokens += tokens;
    }
  }

  if (pending.length > 0) {
    batches.push({ sections: pending, estimatedTokens: pendingTokens });
  }

  return batches;
}
