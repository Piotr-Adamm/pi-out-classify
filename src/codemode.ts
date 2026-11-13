import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'

// The goal of a codemode script, as introduced by pi-vcc-codemode: a `// goal:` comment at the top.

export const CODEMODE_TOOL_NAME = 'codemode'

// pi renders every prompt guideline as a list item, so the text carries no bullet of its own.
export const CODEMODE_GOAL_GUIDELINE =
  "Start each codemode script with a one-line `// goal: <intended outcome>` in the same language as the user's request, describing intent rather than success; if you use `// @options:`, keep it first and put the intent on line 2."

const OPTIONS_LINE = /^\s*\/\/\s*@options\s*:/i
const GOAL_LINE = /^\s*\/\/\s*goal\s*:\s*(.*?)\s*$/i
const MAX_GOAL_CHARS = 200
const HEADER_COMMENT_LINES = 3

// The goal from the script's opening comment region: at most three lines after an optional first-line
// `// @options:`, stopping at the first non-empty line that is not a comment. The first non-empty goal
// of at most 200 characters wins; comments elsewhere in the script are not goals.
export function parseCodemodeGoal(code: unknown): string {
  if (typeof code !== 'string' || !code) {
    return ''
  }

  const lines = code.split(/\r?\n/)
  const start = OPTIONS_LINE.test(lines[0] ?? '') ? 1 : 0
  for (let i = start; i < lines.length && i - start < HEADER_COMMENT_LINES; i++) {
    const line = lines[i].trim()
    if (!line) {
      continue
    }
    if (!line.startsWith('//')) {
      break
    }

    const goal = GOAL_LINE.exec(line)?.[1].trim()
    if (goal && goal.length <= MAX_GOAL_CHARS) {
      return goal
    }
  }
  return ''
}

// Adds the goal rule to the run's prompt guidelines while codemode is an active tool.
export function registerCodemodeGoal(pi: ExtensionAPI): void {
  pi.on('before_agent_start', (event) => {
    const codemodeActive = pi
      .getActiveTools()
      .some((tool) => tool.toLowerCase() === CODEMODE_TOOL_NAME)
    if (!codemodeActive) {
      return
    }

    const { promptGuidelines } = event.systemPromptOptions
    if (!promptGuidelines.includes(CODEMODE_GOAL_GUIDELINE)) {
      promptGuidelines.push(CODEMODE_GOAL_GUIDELINE)
    }
  })
}
