import type { ImageContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ToolResultEvent } from "@earendil-works/pi-coding-agent";
import type { TypeSafeClient } from "@typesafe-ai/sdk";
import { createClient } from "../../jevisual/src/api.ts";
import {
  addFigureContext,
  batchSections,
  splitTextBuffer,
  type FigureWithContext,
  type OutputBatch,
  type TextBuffer,
} from "./batcher.ts";
import type { ClassificationTaskContext } from "./classifier.ts";
import { loadConfig, type OutClassifyConfig } from "./config.ts";
import { estimateBaseTokens, estimateSectionTokens } from "./estimator.ts";
import { extractDocument, supportsDocumentExtraction, type ExtractedDocument } from "./extractor.ts";

export type OutputType = "text" | "images" | "document" | "mixed" | "empty";

// Jev's context window if the catalog has no entry; a share of it is the token budget per request.
const FALLBACK_CONTEXT_WINDOW = 64000;
const BATCH_TOKEN_MARGIN = 0.8;
// Without a purpose argument, the output is judged against this many of the latest user messages.
const PURPOSE_MESSAGE_COUNT = 2;
const MARKDOWN_PATH = /\.(md|markdown)$/i;

// Everything resolved once per session_start; undefined means out-classify is off.
interface SessionState {
  config: OutClassifyConfig;
  client: TypeSafeClient;
  maxBatchTokens: number;
}

// One image block of the tool output, kept unchanged for the rebuild. Counterpart of TextItem.
interface ImageItem {
  contentIndex: number;
  block: ImageContent;
}

// Non-document images, in output order. Counterpart of TextBuffer.
type ImageBuffer = ImageItem[];

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

// The last `count` non-empty user messages, oldest first; fewer if the session has fewer.
function lastUserMessages(ctx: ExtensionContext, count: number): string[] {
  const branch = ctx.sessionManager.getBranch();
  const messages: string[] = [];

  for (let i = branch.length - 1; i >= 0 && messages.length < count; i--) {
    const entry = branch[i];
    if (entry.type !== "message" || entry.message.role !== "user") continue;

    const { content } = entry.message;
    const text = typeof content === "string"
      ? content
      : content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
    if (text.trim()) messages.unshift(text.trim());
  }

  return messages;
}

export function registerOutputHook(pi: ExtensionAPI): void {
  // Resolved once per session; /reload fires session_start again, so config and key changes apply then.
  let session: SessionState | undefined;

  pi.on("session_start", async (_event, ctx) => {
    session = undefined;

    let config: OutClassifyConfig;
    try {
      config = await loadConfig();
    } catch (error) {
      ctx.ui.notify(`out-classify disabled: ${(error as Error).message}`, "error");
      return;
    }

    const apiKey = await ctx.modelRegistry.getApiKeyForProvider("typesafe");
    if (!apiKey) {
      ctx.ui.notify("out-classify disabled: no TypeSafe key found.", "error");
      return;
    }

    const contextWindow =
      ctx.modelRegistry.findOfType("classifier", "typesafe", "jev-latest")?.contextWindow
      ?? FALLBACK_CONTEXT_WINDOW;

    session = {
      config,
      client: createClient(apiKey),
      maxBatchTokens: Math.floor(contextWindow * BATCH_TOKEN_MARGIN),
    };
  });

  pi.on("tool_result", async (event, ctx) => {
    if (!session) return;
    const { config } = session;

    const outputType = identifyOutputType(event);
    if (outputType === "empty") return;
    if (event.isError) return;

    // Without an explicit purpose, judge the output against the user's latest messages.
    const inputPurpose = typeof event.input.purpose === "string" ? event.input.purpose.trim() : "";
    const userMessages = inputPurpose ? [] : lastUserMessages(ctx, PURPOSE_MESSAGE_COUNT);
    const purpose = inputPurpose || userMessages.join("\n\n");
    if (!purpose) return;
    const purposeSource = inputPurpose ? "tool_call" : "user_message";
    // Passed on as purposeMessageCount, so the classifier instructions name the actual number.
    const purposeMessageCount = userMessages.length;

    const textBuffer: TextBuffer = [];
    const imageBuffer: ImageBuffer = [];
    let document: ExtractedDocument | undefined;

    if (outputType === "document") {
      // Native read returns raw PDF syntax, so never fall back to it.
      try {
        document = await extractDocument(event.input.path as string);
      } catch (error) {
        return {
          content: [{ type: "text", text: `Could not extract this PDF: ${(error as Error).message}` }],
          isError: true,
        };
      }

      // The extracted document replaces the whole raw output: one item, no contentIndex.
      textBuffer.push({ text: document.markdown, markdown: true });
    } else {
      // Pure and mixed outputs feed the same buffers without copying or reordering blocks.
      const path = event.toolName === "read" ? event.input.path : undefined;
      const markdown = typeof path === "string" && MARKDOWN_PATH.test(path);
      for (const [contentIndex, block] of event.content.entries()) {
        if (block.type === "text") {
          textBuffer.push({ text: block.text, markdown, contentIndex });
        } else if (block.type === "image") {
          imageBuffer.push({ contentIndex, block });
        }
      }
    }

    const sections = splitTextBuffer(textBuffer, config.maxSectionChars);
    const figures: FigureWithContext[] = document ? addFigureContext(document.figures, sections) : [];

    if (sections.length === 0 && figures.length === 0 && imageBuffer.length === 0) return;

    const context: ClassificationTaskContext = {
      purpose,
      purposeSource,
      ...(purposeSource === "user_message" ? { purposeMessageCount } : {}),
      toolName: event.toolName,
      // Tool arguments arrive as parsed JSON.
      toolArguments: event.input as ClassificationTaskContext["toolArguments"],
    };

    let batches: OutputBatch[] = [];
    if (sections.length > 0) {
      try {
        batches = batchSections(
          sections,
          session.maxBatchTokens,
          estimateBaseTokens(context),
          (section) => estimateSectionTokens(section, context),
        );
      } catch (error) {
        // Only when the base (purpose, tool arguments) leaves no room: pass the output through.
        ctx.ui.notify(`out-classify skipped this output: ${(error as Error).message}`, "warning");
        // TODO: For documents, return the extracted content with figures (same as the rebuild).
        return document ? { content: [{ type: "text", text: document.markdown }] } : undefined;
      }
    }

    // TODO: Classify text batches: one request per batch, one question per section (purpose, purposeSource).
    // TODO: Classify figures and images with jevisual (session.client), with captions.

    // TODO: Return keep + uncertain in original order (content AND structuredContent).
    // On classifier failure, return the original output (for documents, the extracted content).
  });
}
