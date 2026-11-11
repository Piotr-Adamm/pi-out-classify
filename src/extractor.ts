import type { ImageContent } from "@earendil-works/pi-ai";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(new URL("./extractor.py", import.meta.url));

// The Docling version out-classify installs and is tested with. Changing it reinstalls the venv and
// downloads the models again.
const DOCLING_VERSION = "2.144.0";
// Written last by each setup step and holding what it was set up for, so an interrupted or outdated
// step is repeated.
const INSTALLED_MARKER = ".installed";
// A lock older than this is left over from a crashed setup and is removed.
const STALE_LOCK_MS = 60 * 60 * 1000;
// Lines of a failed command's stderr (e.g. a Python traceback) included in the error.
const STDERR_TAIL_LINES = 5;

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

export interface ExtractorOptions {
  // A Python with Docling to use instead of the venv out-classify installs.
  pythonPath?: string;
  // Called once per setup, e.g. when pythonPath has another Docling version than DOCLING_VERSION.
  onWarning?: (message: string) => void;
}

interface Setup {
  python: string;
  modelsDir: string;
}

// Everything out-classify installs lives here: venv (Docling) and models (its weights). Outside the
// package, so package updates keep them.
function dataDir(): string {
  return join(getAgentDir(), "out-classify");
}

function venvPython(venvDir: string): string {
  return process.platform === "win32" ? join(venvDir, "Scripts", "python.exe") : join(venvDir, "bin", "python");
}

export function supportsDocumentExtraction(path: string): boolean {
  return documentExtensions.has(extname(path).toLowerCase());
}

// Runs a command and returns its stdout. On a non-zero exit, the error ends with stderr's last lines.
async function run(command: string, args: string[]): Promise<string> {
  const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  let errors = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => { output += chunk; });
  child.stderr.on("data", (chunk: string) => { errors += chunk; });

  let code: number | null;
  try {
    [code] = await once(child, "close");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      throw new Error(command === "uv"
        ? "uv is not installed; it is needed to install Docling (https://docs.astral.sh/uv/)."
        : `${command} was not found.`);
    }
    throw error;
  }
  if (code !== 0) {
    const tail = errors.trim().split("\n").slice(-STDERR_TAIL_LINES).join("\n");
    throw new Error(`${command} exited with code ${code}${tail ? `:\n${tail}` : ""}`);
  }
  return output;
}

async function readMarker(directory: string): Promise<string | undefined> {
  return readFile(join(directory, INSTALLED_MARKER), "utf8").then((text) => text.trim(), () => undefined);
}

// Holds a directory lock while `task` runs, so two pi sessions never install at the same time.
async function withLock<T>(lockDir: string, task: () => Promise<T>): Promise<T> {
  for (;;) {
    try {
      await mkdir(lockDir);
      break;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      const age = await stat(lockDir).then((info) => Date.now() - info.mtimeMs, () => 0);
      if (age > STALE_LOCK_MS) await rm(lockDir, { recursive: true, force: true });
      else await sleep(2000);
    }
  }
  try {
    return await task();
  } finally {
    await rm(lockDir, { recursive: true, force: true });
  }
}

// Installs what is missing or outdated and returns the Python to run extractor.py with. With
// pythonPath, that Python is used as it is: nothing is installed, only its models are downloaded.
async function setUp({ pythonPath, onWarning }: ExtractorOptions): Promise<Setup> {
  const venvDir = join(dataDir(), "venv");
  const modelsDir = join(dataDir(), "models");
  const python = pythonPath ?? venvPython(venvDir);
  const requirement = `docling==${DOCLING_VERSION}`;

  const doclingVersion = pythonPath === undefined
    ? DOCLING_VERSION
    : (await run(python, [scriptPath, "version"])).trim();
  if (doclingVersion !== DOCLING_VERSION) {
    onWarning?.(`pythonPath has Docling ${doclingVersion}; out-classify is tested with ${DOCLING_VERSION}.`);
  }

  const venvReady = async () => pythonPath !== undefined || await readMarker(venvDir) === requirement;
  const modelsReady = async () => await readMarker(modelsDir) === doclingVersion;
  if (await venvReady() && await modelsReady()) return { python, modelsDir };

  await mkdir(dataDir(), { recursive: true });
  await withLock(join(dataDir(), "setup.lock"), async () => {
    // Another session may have finished the setup while this one waited for the lock.
    if (!await venvReady()) {
      await rm(venvDir, { recursive: true, force: true });
      await run("uv", ["venv", "--quiet", "--python", ">=3.10", venvDir]);
      await run("uv", ["pip", "install", "--quiet", "--python", python, requirement]);
      await writeFile(join(venvDir, INSTALLED_MARKER), requirement);
    }
    // Downloads into a temporary directory, swaps it in and writes the marker with its Docling version.
    if (!await modelsReady()) await run(python, [scriptPath, "models", modelsDir]);
  });
  return { python, modelsDir };
}

// One setup per process and pythonPath; parallel reads wait for the same one. A failed setup is tried
// again next time.
let setup: { pythonPath: string | undefined; promise: Promise<Setup> } | undefined;

export async function extractDocument(path: string, options: ExtractorOptions = {}): Promise<ExtractedDocument> {
  if (!setup || setup.pythonPath !== options.pythonPath) {
    const promise = setUp(options).catch((error) => {
      if (setup?.promise === promise) setup = undefined;
      throw error;
    });
    setup = { pythonPath: options.pythonPath, promise };
  }
  const { python, modelsDir } = await setup.promise;

  const imageDir = await mkdtemp(join(tmpdir(), "out-classify-"));
  try {
    const output = await run(python, [scriptPath, "extract", resolve(path), imageDir, modelsDir]);
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
