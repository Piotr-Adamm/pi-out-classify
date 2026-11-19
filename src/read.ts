import { createReadToolDefinition } from '@earendil-works/pi-coding-agent'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { Type } from 'typebox'
import type { Static } from 'typebox'

// What scripts (codemode) get for a PDF, in addition to native read's text and image shapes: the
// Markdown with its figures and formulas, each with its image block. The figures are extractor.ts's
// Figure without the text layer, so hook.ts builds them without listing the fields again.
export const DOCUMENT_OUTPUT_SCHEMA = Type.Object(
  {
    type: Type.Literal('document'),
    text: Type.String({
      description:
        'Extracted Markdown. Figures and formulas are linked as ![Image](fig-<n>) and ![Formula](formula-<n>); transcribed formulas are inlined as $ Typst $.',
    }),
    figures: Type.Array(
      Type.Object({
        id: Type.String({ description: 'The ID in the Markdown link.' }),
        kind: Type.Union([Type.Literal('figure'), Type.Literal('formula')]),
        caption: Type.String(),
        page: Type.Union([Type.Number(), Type.Null()]),
        image: Type.Object(
          { type: Type.Literal('image'), data: Type.String(), mimeType: Type.String() },
          { description: 'Image block; show it with image().' }
        ),
        typst: Type.Optional(
          Type.String({ description: 'Formulas: the Typst math, if a transcription is cached.' })
        ),
      })
    ),
  },
  { description: 'A PDF.' }
)

export type DocumentOutput = Static<typeof DOCUMENT_OUTPUT_SCHEMA>

export function registerReadOverride(pi: ExtensionAPI): void {
  const nativeRead = createReadToolDefinition(process.cwd())

  pi.registerTool({
    ...nativeRead,
    description:
      nativeRead.description +
      ' PDFs are returned as extracted Markdown with their figures.' +
      ' Optionally supply goal to explain what this read should establish. Parts of large outputs that are' +
      ' not relevant to the goal (or, without one, to the latest user messages) may be omitted and marked' +
      ' with a note such as "[out-classify: lines 120–184 omitted, …]".',
    promptGuidelines: [
      ...(nativeRead.promptGuidelines ?? []),
      'Provide goal describing the immediate information needed from a read.',
      'If an omitted part is needed after all, read it again with offset/limit for the omitted lines.',
    ],
    outputSchema: nativeRead.outputSchema
      ? Type.Union([nativeRead.outputSchema, DOCUMENT_OUTPUT_SCHEMA])
      : undefined,
    parameters: Type.Object({
      ...nativeRead.parameters.properties,
      goal: Type.Optional(
        Type.String({
          description: 'The immediate question or information this read should answer.',
        })
      ),
    }),
    async execute(toolCallId, input, signal, onUpdate, ctx) {
      const { goal: _goal, ...readInput } = input
      return nativeRead.execute(toolCallId, readInput, signal, onUpdate, ctx)
    },
  })
}
