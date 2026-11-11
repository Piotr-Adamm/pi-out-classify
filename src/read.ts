import { createReadToolDefinition, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export function registerReadOverride(pi: ExtensionAPI): void {
  const nativeRead = createReadToolDefinition(process.cwd());

  pi.registerTool({
    ...nativeRead,
    description:
      nativeRead.description +
      " Optionally supply purpose to explain what this read should establish. Reads without a purpose bypass output classification.",
    promptGuidelines: [
      ...(nativeRead.promptGuidelines ?? []),
      "Provide purpose describing the immediate information needed from a read. Omit it when raw, unfiltered output is required.",
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
