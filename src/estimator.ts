import { countTokens } from 'gpt-tokenizer/encoding/o200k_base'
import type { OutputSection } from './batcher.ts'
import { requestBase, buildQuestion, requestSectionId, sectionState } from './classifier.ts'
import type { ClassificationTaskContext } from './classifier.ts'

// o200k only approximates Jev's tokenizer; maxBatchTokens keeps a margin for that.
// Special tokens such as <|endoftext|> can appear in tool output and are counted as plain text.
function tokens(value: unknown): number {
  return countTokens(JSON.stringify(value), { disallowedSpecial: new Set() })
}

// Counted once per tool result: goal, tool arguments, and the JSON around the sections.
export function estimateBaseTokens(context: ClassificationTaskContext): number {
  return tokens(requestBase(context))
}

// Counted once per section: its state entry plus its question. Added to the base for each batch.
// The request ID depends on the section's place in its batch, which is not known yet; a three-digit
// one (s100) is never shorter.
export function estimateSectionTokens(
  section: OutputSection,
  context: ClassificationTaskContext
): number {
  const id = requestSectionId(99)
  return (
    tokens(sectionState(id, section, context)) +
    tokens({ [id]: buildQuestion(`section ${id}`, 'section text', context) })
  )
}
