import type { ImageContent, TextContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import { addFigureContext, splitIntoSections, splitMarkdownIntoSections, type FigureWithContext, type OutputSection } from "./batcher.ts";
import { loadConfig, type OutClassifyConfig } from "./config.ts";
import { extractDocument, supportsDocumentExtraction, type ExtractedDocument } from "./extractor.ts";

export type OutputType = "text" | "images" | "document" | "mixed" | "empty";

interface BufferedBlock<T> {
  contentIndex: number;
  block: T;
}

export function identifyOutputType(event: ToolResultEvent): OutputType {
  const path = event.toolName === "read" ? event.input.path : undefined;
  if (typeof path === "string" && supportsDocumentExtraction(path)) {
    return "document";
  }

  const hasText = event.content.some((part) => part.type === "text");
  const hasImages = event.content.some((part) => part.type === "image");

  if (hasText && hasImages) return "mixed";
  if (hasImages) return "images";
  if (hasText) return "text";
  return "empty";
}

function lastUserMessage(ctx: ExtensionContext): string | undefined {
  const branch = ctx.sessionManager.getBranch();
  for (let i = branch.length - 1; i >= 0; i--) {
    const entry = branch[i];
    if (entry.type !== "message" || entry.message.role !== "user") continue;

    const { content } = entry.message;
    const text = typeof content === "string"
      ? content
      : content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
    if (text.trim()) return text.trim();
  }
  return undefined;
}

export function registerOutputHook(pi: ExtensionAPI): void {
  // Loaded once per session; /reload fires session_start again, so config edits apply then.
  let config: OutClassifyConfig | undefined;

  pi.on("session_start", async (_event, ctx) => {
    try {
      config = await loadConfig();
    } catch (error) {
      config = undefined;
      ctx.ui.notify(`out-classify disabled: ${(error as Error).message}`, "error");
    }
  });

  pi.on("tool_result", async (event, ctx) => {
    if (!config) return;

    const outputType = identifyOutputType(event);
    if (outputType === "empty") return;
    if (event.isError) return;

    // Without an explicit purpose, judge the output against the user's latest request.
    const inputPurpose = typeof event.input.purpose === "string" ? event.input.purpose.trim() : "";
    const purpose = inputPurpose || lastUserMessage(ctx);
    if (!purpose) return;
    const purposeSource = inputPurpose ? "tool_call" : "user_message";

    const text: BufferedBlock<TextContent>[] = [];
    const images: BufferedBlock<ImageContent>[] = []; // Non-document images only.
    let sections: OutputSection[] = [];
    let figures: FigureWithContext[] = [];

    if (outputType === "document") {
      // Native read returns raw PDF syntax, so never fall back to it.
      let document: ExtractedDocument;
      try {
        document = await extractDocument(event.input.path as string);
      } catch (error) {
        return {
          content: [{ type: "text", text: `Could not extract this PDF: ${(error as Error).message}` }],
          isError: true,
        };
      }

      // The extracted document replaces the whole raw output, so it is split directly.
      sections = splitMarkdownIntoSections(document.markdown, config.maxSectionChars);
      figures = addFigureContext(document.figures, sections);
    } else {
      // Pure and mixed outputs feed the same buffers without copying or reordering blocks.
      for (const [contentIndex, block] of event.content.entries()) {
        if (block.type === "text") {
          text.push({ contentIndex, block });
        } else if (block.type === "image") {
          images.push({ contentIndex, block });
        }
      }
    }

    // TODO: Use splitMarkdownIntoSections for .md reads.
    for (const { contentIndex, block } of text) {
      sections.push(...splitIntoSections(block.text, config.maxSectionChars, contentIndex));
    }

    if (sections.length === 0 && figures.length === 0 && images.length === 0) return;

    // TODO: Batch sections with batchSections once the token estimate exists.

    // TODO: Classify text batches: one request per batch, one question per section (purpose, purposeSource).
    // TODO: Classify figures: one request per figure; caption-only when the model lacks image input.

    // TODO: Return keep + uncertain in original order (content AND structuredContent).
    // On classifier failure, return the original output (for documents, the extracted content).
  });
}
