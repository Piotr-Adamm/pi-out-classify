/**
 * @module
 * Shows what the agent gets from a read or a bash command, through the real hook outside pi: your
 * out-classify config, your logins (TypeSafe, formula model) and the formula cache, as in a session.
 *
 * Usage: npm run preview.test -- <file> [goal] [--offset N] [--limit N]
 *        npm run preview.test -- --bash '<command>' [goal]
 *
 * bash has no goal parameter: the goal is passed as the user's latest message, like a request in chat.
 *
 * Text goes to stdout as the agent sees it; images are saved as PNG files and shown as
 * [image n: path]. Notifications (warnings, errors) and a summary go to stderr. Without a goal, the
 * output is not classified (a read without goal and without user messages). Other files go through
 * pi's read first, with its line limit and truncation footer.
 */

import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import type { ImageContent, TextContent } from '@earendil-works/pi-ai'
import {
  createBashToolDefinition,
  createReadToolDefinition,
  ModelRegistry,
  ModelRuntime,
} from '@earendil-works/pi-coding-agent'
import type {
  ExtensionAPI,
  ExtensionContext,
  ExtensionToolContext,
  ToolResultEvent,
} from '@earendil-works/pi-coding-agent'
import { resolveReadPath, supportsDocumentExtraction } from '../src/extractor.ts'
import { registerOutputHook } from '../src/hook.ts'

type Handler = (event: unknown, ctx: ExtensionContext) => unknown

const usage =
  'Usage: npm run preview.test -- <file> [goal] [--offset N] [--limit N]\n' +
  "       npm run preview.test -- --bash '<command>' [goal]"

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: { offset: { type: 'string' }, limit: { type: 'string' }, bash: { type: 'string' } },
})
const command = values.bash
// Words after the file are the goal, so an unquoted goal works too.
const [file, ...goalWords] = command === undefined ? positionals : ['', ...positionals]
const goal = goalWords.join(' ')
if (command === undefined ? !file : !command.trim()) {
  console.error(usage)
  process.exit(1)
}
const path = command === undefined ? resolveReadPath(file, process.cwd()) : ''

// Only the parts of pi the hook uses: event registration, the model registry, notifications.
const handlers = new Map<string, Handler>()
const pi = {
  on: (event: string, handler: Handler) => {
    handlers.set(event, handler)
  },
} as unknown as ExtensionAPI
registerOutputHook(pi)

const notify = (message: string, type = 'info') => console.error(`[${type}] ${message}`)
const ctx = {
  cwd: process.cwd(),
  modelRegistry: new ModelRegistry(await ModelRuntime.create()),
  // For bash, the goal is the user's latest message.
  // bash puts the session ID and file into the command's environment.
  sessionManager: {
    getSessionId: () => 'preview',
    getSessionFile: () => undefined,
    getBranch: () =>
      command !== undefined && goal
        ? [{ type: 'message', message: { role: 'user', content: goal } }]
        : [],
  },
  ui: { notify },
  signal: undefined,
} as unknown as ExtensionContext

await handlers.get('session_start')?.({ type: 'session_start' }, ctx)

// The tools only use the context's cwd, model and session; the rest of a tool context is unused.
const toolCtx = ctx as ExtensionToolContext

let event: ToolResultEvent
if (command !== undefined) {
  // pi's bash returns a failing command as an error, which the agent gets (and out-classify filters)
  // like any other result.
  const input = { command }
  let content: (TextContent | ImageContent)[]
  let isError = false
  try {
    content = (
      await createBashToolDefinition(process.cwd()).execute(
        'preview',
        input,
        undefined,
        undefined,
        toolCtx
      )
    ).content
  } catch (error) {
    content = [{ type: 'text', text: (error as Error).message }]
    isError = true
  }
  event = {
    type: 'tool_result',
    toolName: 'bash',
    toolCallId: 'preview',
    input,
    content,
    details: undefined,
    isError,
  } as unknown as ToolResultEvent
} else {
  // As the agent passes it (~/…, relative); the hook resolves it like pi's read.
  const input = {
    path: file,
    ...(goal ? { goal } : {}),
    ...(values.offset ? { offset: Number(values.offset) } : {}),
    ...(values.limit ? { limit: Number(values.limit) } : {}),
  }
  // Documents are extracted by the hook; other files come in as pi's read returns them.
  const { goal: _goal, ...readInput } = input
  const content: (TextContent | ImageContent)[] = supportsDocumentExtraction(path)
    ? []
    : (
        await createReadToolDefinition(process.cwd()).execute(
          'preview',
          readInput,
          undefined,
          undefined,
          toolCtx
        )
      ).content
  event = {
    type: 'tool_result',
    toolName: 'read',
    toolCallId: 'preview',
    input,
    content,
    details: undefined,
    isError: false,
  } as unknown as ToolResultEvent
}
const original = event.content

const started = Date.now()
const result = (await handlers.get('tool_result')?.(event, ctx)) as
  | { content?: (TextContent | ImageContent)[]; structuredContent?: unknown }
  | undefined
const seconds = ((Date.now() - started) / 1000).toFixed(1)
const content = result?.content ?? original

await mkdir(join(tmpdir(), 'out-classify-preview'), { recursive: true })
const imageDir = await mkdtemp(join(tmpdir(), 'out-classify-preview', 'run-'))
let images = 0
let chars = 0
for (const block of content) {
  if (block.type === 'text') {
    chars += block.text.length
    process.stdout.write(block.text.endsWith('\n') ? block.text : `${block.text}\n`)
  } else {
    images++
    const imagePath = join(imageDir, `${images}.png`)
    await writeFile(imagePath, Buffer.from(block.data, 'base64'))
    console.log(`[image ${images}: ${imagePath}]`)
  }
}

const full =
  typeof result?.structuredContent === 'string'
    ? result.structuredContent.length
    : original.reduce((sum, block) => sum + (block.type === 'text' ? block.text.length : 0), 0)
console.error(
  `\n--- ${result ? 'filtered by out-classify' : 'passed through unchanged'} in ${seconds}s: ` +
    `${chars} of ${full} characters, ${images} images`
)
