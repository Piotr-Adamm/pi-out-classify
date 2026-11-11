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
import {
  classifyImages,
  classifyTextBatches,
  type ClassificationTaskContext,
  type ClassifierResults,
  type ImageToClassify,
} from "./classifier.ts";
import { CODEMODE_TOOL_NAME, parseCodemodeGoal } from "./codemode.ts";
import { loadConfig, type OutClassifyConfig } from "./config.ts";
import { reconstruct } from "./reconstructor.ts";
import { estimateBaseTokens, estimateSectionTokens } from "./estimator.ts";
import { extractDocument, supportsDocumentExtraction, type ExtractedDocument } from "./extractor.ts";

export type OutputType = "text" | "images" | "document" | "mixed" | "empty";

// Jev's context window if the catalog has no entry; a share of it is the token budget per request.
const FALLBACK_CONTEXT_WINDOW = 64000;
const BATCH_TOKEN_MARGIN = 0.8;
// Without a goal argument, the output is judged against this many of the latest user messages.
const GOAL_MESSAGE_COUNT = 2;
const MARKDOWN_PATH = /\.(md|markdown)$/i;
// pi's read appends this when it truncates a file, e.g. "[Showing lines 1-2000 of 5000. Use offset=2001 to
// continue.]". It is never classified, so the agent always learns where to continue. Checked for every
// tool, not only read: a codemode script that returns a read's text ends with the same footer.
const READ_FOOTER = /\n\n\[[^\]\n]*Use offset=\d+ to continue\.\]\s*$/;

// Everything resolved once per session_start; undefined means out-classify is off.
interface SessionState {
  config: OutClassifyConfig;
  client: TypeSafeClient;
  maxBatchTokens: number;
}

// One image block of the tool output, kept unchanged for the rebuild. Counterpart of TextItem.
interface ImageItem {
  id: string; // image_<contentIndex>; key in ClassifierResults.
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

// The extracted document as tool output, used whenever it is not (or cannot be) classified:
// everything kept, figures after their linking section. structuredContent (read's output for scripts)
// becomes the extracted Markdown instead of the raw PDF text.
function documentOutput(document: ExtractedDocument, maxSectionChars: number) {
  const sections = splitTextBuffer([{ text: document.markdown, markdown: true }], maxSectionChars);
  return {
    content: reconstruct({ sections, results: new Map(), images: [], document }),
    structuredContent: document.markdown,
  };
}

// The goal the tool call states itself: read's goal parameter (added by out-classify's read override)
// or the `// goal:` line of a codemode script. Empty if there is none.
function toolCallGoal(event: ToolResultEvent): string {
  if (event.toolName === "read" && typeof event.input.goal === "string") return event.input.goal.trim();
  if (event.toolName.toLowerCase() === CODEMODE_TOOL_NAME) return parseCodemodeGoal(event.input.code);
  return "";
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

    // Only what the agent gets is filtered. Calls made by another tool (e.g. tools.read inside a
    // codemode script) go to that tool, which needs the raw data; its own result is filtered.
    if (event.parentToolCallId !== undefined) return;

    // Error results are filtered too (e.g. a long failing build log); isError stays as it is.
    const outputType = identifyOutputType(event);
    if (outputType === "empty") return;

    // Thresholds: a part below its threshold passes through unclassified; both below → nothing to do.
    // Documents are checked after extraction, because only the extracted text counts. Batching and
    // image classification check their threshold again on the buffers.
    if (outputType !== "document") {
      const textChars = event.content.reduce((sum, block) => block.type === "text" ? sum + block.text.length : sum, 0);
      const imageCount = event.content.filter((block) => block.type === "image").length;
      if (textChars < config.thresholdChars && imageCount < config.thresholdImages) return;
    }

    // Without an explicit goal, judge the output against the user's latest messages.
    const inputGoal = toolCallGoal(event);
    const userMessages = inputGoal ? [] : lastUserMessages(ctx, GOAL_MESSAGE_COUNT);
    const goal = inputGoal || userMessages.join("\n\n");
    // Documents still need extraction without a goal; they return unclassified below.
    if (!goal && outputType !== "document") return;
    const goalSource = inputGoal ? "tool_call" : "user_message";
    // Passed on as goalMessageCount, so the classifier instructions name the actual number.
    const goalMessageCount = userMessages.length;

    const textBuffer: TextBuffer = [];
    const imageBuffer: ImageBuffer = [];
    let readFooter = "";
    let document: ExtractedDocument | undefined;

    if (outputType === "document") {
      // Native read returns raw PDF syntax, so never fall back to it.
      try {
        document = await extractDocument(event.input.path as string);
      } catch (error) {
        const text = `Could not extract this PDF: ${(error as Error).message}`;
        return { content: [{ type: "text", text }], structuredContent: text, isError: true };
      }

      if (
        !goal
        || (document.markdown.length < config.thresholdChars && document.figures.length < config.thresholdImages)
      ) return documentOutput(document, config.maxSectionChars);

      // The extracted document replaces the whole raw output: one item, no contentIndex.
      textBuffer.push({ text: document.markdown, markdown: true });
    } else {
      // Pure and mixed outputs feed the same buffers without copying or reordering blocks.
      const path = event.toolName === "read" ? event.input.path : undefined;
      const markdown = typeof path === "string" && MARKDOWN_PATH.test(path);
      for (const [contentIndex, block] of event.content.entries()) {
        if (block.type === "text") {
          const footer = block.text.match(READ_FOOTER)?.[0] ?? "";
          readFooter ||= footer.trim();
          textBuffer.push({ text: block.text.slice(0, block.text.length - footer.length), markdown, contentIndex });
        } else if (block.type === "image") {
          imageBuffer.push({ id: `image_${contentIndex}`, contentIndex, block });
        }
      }
    }

    const sections = splitTextBuffer(textBuffer, config.maxSectionChars);
    const figures: FigureWithContext[] = document ? addFigureContext(document.figures, sections) : [];

    if (sections.length === 0 && figures.length === 0 && imageBuffer.length === 0) return;

    const context: ClassificationTaskContext = {
      goal,
      goalSource,
      ...(goalSource === "user_message" ? { goalMessageCount } : {}),
      toolName: event.toolName,
      // Tool arguments arrive as parsed JSON.
      toolArguments: event.input as ClassificationTaskContext["toolArguments"],
    };

    // Sections are always split (figure context needs them), but only batched if text is classified.
    let batches: OutputBatch[] = [];
    const textChars = textBuffer.reduce((sum, item) => sum + item.text.length, 0);
    if (textChars >= config.thresholdChars && sections.length > 0) {
      try {
        batches = batchSections(
          sections,
          session.maxBatchTokens,
          estimateBaseTokens(context),
          (section) => estimateSectionTokens(section, context),
        );
      } catch (error) {
        // Only when the base (goal, tool arguments) leaves no room: pass the output through.
        ctx.ui.notify(`out-classify skipped this output: ${(error as Error).message}`, "warning");
        return document ? documentOutput(document, config.maxSectionChars) : undefined;
      }
    }

    // Figures (documents) or non-document images, never both. Figures carry their caption, or without
    // one the text of their linking section; other images are judged by the goal only.
    const images: ImageToClassify[] = document
      ? figures.map((figure) => ({ id: figure.id, image: figure.image, caption: figure.caption || figure.context }))
      : imageBuffer.map((image) => ({ id: image.id, image: image.block }));

    // Text and images in parallel; a part below its threshold sends nothing. Failures come back as
    // "unclassified"; only cancellation (Esc) throws.
    let textError: unknown;
    let imageError: unknown;
    let textResults: ClassifierResults;
    let imageResults: ClassifierResults;
    try {
      [textResults, imageResults] = await Promise.all([
        classifyTextBatches(batches, context, session.client, ctx.signal, (error) => { textError ??= error; }),
        classifyImages(
          images.length >= config.thresholdImages ? images : [],
          context,
          session.client,
          ctx.signal,
          (error) => { imageError ??= error; },
        ),
      ]);
    } catch {
      return document ? documentOutput(document, config.maxSectionChars) : undefined;
    }

    const unclassifiedSections = [...textResults.values()].filter((result) => result === "unclassified").length;
    if (unclassifiedSections > 0) {
      ctx.ui.notify(
        `out-classify: ${unclassifiedSections} of ${textResults.size} sections could not be classified ` +
        `(${(textError as Error)?.message ?? "unknown error"}).`,
        "warning",
      );
    }
    const unclassifiedImages = [...imageResults.values()].filter((result) => result === "unclassified").length;
    if (unclassifiedImages > 0) {
      ctx.ui.notify(
        `out-classify: ${unclassifiedImages} of ${imageResults.size} images could not be classified ` +
        `(${(imageError as Error)?.message ?? "animated or multi-page"}).`,
        "warning",
      );
    }

    // One map for the reconstruction: results.get(section.id), results.get(image.id), results.get(figure.id).
    const results: ClassifierResults = new Map([...textResults, ...imageResults]);

    // Nothing removed or unclassified: the original output passes through unchanged (documents still
    // need their extracted content).
    const changed = [...results.values()].some((result) => result === "discard" || result === "unclassified");
    if (!changed) return document ? documentOutput(document, config.maxSectionChars) : undefined;

    // Batched sections carry the IDs in results (split pieces have their own); unbatched text is kept.
    const classifiedSections = batches.length > 0 ? batches.flatMap((batch) => batch.sections) : sections;
    const readOffset = event.toolName === "read" && typeof event.input.offset === "number" ? event.input.offset : 1;
    // Only content (what the agent sees) is filtered. structuredContent goes to scripts and is passed
    // on unchanged (documents: the extracted Markdown, not the raw file); returning content without
    // it would drop it.
    return {
      structuredContent: document ? document.markdown : event.structuredContent,
      content: [
        ...reconstruct({ sections: classifiedSections, results, document, images: imageBuffer, readOffset }),
        ...(readFooter ? [{ type: "text" as const, text: readFooter }] : []),
      ],
    };
  });
}
