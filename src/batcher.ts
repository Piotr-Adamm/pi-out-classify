import type { Figure } from "./extractor.ts";

export interface OutputSection {
  id: string;
  start: number;
  end: number;
  text: string;
}

export interface OutputBatch {
  sections: OutputSection[];
  estimatedTokens: number;
}

function checkSplitArguments(maxSectionChars: number, startOffset: number): void {
  if (!Number.isSafeInteger(maxSectionChars) || maxSectionChars < 100) {
    throw new Error("maxSectionChars must be an integer of at least 100.");
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

// The estimator must include task context, tool arguments, questions, and overhead.
export function batchSections(
  sections: OutputSection[],
  maxBatchTokens: number,
  estimateRequestTokens: (sections: OutputSection[]) => number,
): OutputBatch[] {
  if (!Number.isSafeInteger(maxBatchTokens) || maxBatchTokens <= 0) {
    throw new Error("maxBatchTokens must be a positive integer.");
  }

  const estimate = (items: OutputSection[]): number => {
    const tokens = estimateRequestTokens(items);
    if (!Number.isSafeInteger(tokens) || tokens < 0) {
      throw new Error("The request token estimate must be a non-negative integer.");
    }
    return tokens;
  };

  const batches: OutputBatch[] = [];
  let pending: OutputSection[] = [];
  let pendingTokens = 0;

  for (const section of sections) {
    const singleTokens = estimate([section]);
    if (singleTokens > maxBatchTokens) {
      throw new Error(`Section ${section.id} exceeds the batch budget; split it further.`);
    }

    const candidate = [...pending, section];
    const candidateTokens = pending.length === 0
      ? singleTokens
      : estimate(candidate);

    if (candidateTokens > maxBatchTokens) {
      batches.push({ sections: pending, estimatedTokens: pendingTokens });
      pending = [section];
      pendingTokens = singleTokens;
    } else {
      pending = candidate;
      pendingTokens = candidateTokens;
    }
  }

  if (pending.length > 0) {
    batches.push({ sections: pending, estimatedTokens: pendingTokens });
  }

  return batches;
}
