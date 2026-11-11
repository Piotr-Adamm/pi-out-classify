import type { ImageContent } from "@earendil-works/pi-ai";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pythonPath = fileURLToPath(new URL("../.venv/bin/python", import.meta.url));
const scriptPath = fileURLToPath(new URL("./extractor.py", import.meta.url));

// Enable additional Docling formats when their extraction paths are implemented.
const documentExtensions = new Set([".pdf"]);

export interface Figure {
  id: string; // Matches the ![Image](id) reference in the Markdown.
  caption: string;
  page: number | null;
  image: ImageContent;
}

export interface ExtractedDocument {
  markdown: string;
  figures: Figure[];
}

export function supportsDocumentExtraction(path: string): boolean {
  return documentExtensions.has(extname(path).toLowerCase());
}

export async function extractDocument(path: string): Promise<ExtractedDocument> {
  const imageDir = await mkdtemp(join(tmpdir(), "out-classify-"));
  try {
    const child = spawn(pythonPath, [scriptPath, resolve(path), imageDir], {
      stdio: ["ignore", "pipe", "inherit"],
    });

    let output = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => { output += chunk; });

    const [code] = await once(child, "close");
    if (code !== 0) throw new Error(`Docling exited with code ${code}`);

    const { markdown, figures } = JSON.parse(output) as {
      markdown: string;
      figures: { id: string; path: string; caption: string; page: number | null }[];
    };

    // Load the images into memory so the directory can be removed right away.
    return {
      markdown,
      figures: await Promise.all(figures.map(async ({ path, ...figure }) => ({
        ...figure,
        image: {
          type: "image" as const,
          data: (await readFile(path)).toString("base64"),
          mimeType: "image/png",
        },
      }))),
    };
  } finally {
    await rm(imageDir, { recursive: true, force: true });
  }
}
