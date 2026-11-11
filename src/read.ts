import { createReadToolDefinition, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export function registerReadOverride(pi: ExtensionAPI): void {
  const nativeRead = createReadToolDefinition(process.cwd());

  pi.registerTool({
    ...nativeRead,
    description:
      nativeRead.description +
      " PDFs are returned as extracted Markdown with their figures." +
      " Optionally supply purpose to explain what this read should establish. Parts of large outputs that are" +
      " not relevant to the purpose (or, without one, to the latest user messages) may be omitted and marked" +
      " with a note such as \"[out-classify: lines 120–184 omitted, …]\".",
    promptGuidelines: [
      ...(nativeRead.promptGuidelines ?? []),
      "Provide purpose describing the immediate information needed from a read.",
      "If an omitted part is needed after all, read it again with offset/limit for the omitted lines.",
    ],
    parameters: Type.Object({
      ...nativeRead.parameters.properties,
      purpose: Type.Optional(Type.String({
        description: "The immediate question or information this read should answer.",
      })),
    }),
    async execute(toolCallId, input, signal, onUpdate, ctx) {
      const { purpose: _purpose, ...readInput } = input;
      return nativeRead.execute(toolCallId, readInput, signal, onUpdate, ctx);
    },
  });
}
