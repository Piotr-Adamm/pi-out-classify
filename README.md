# pi-out-classify

A [pi](https://pi.dev) extension that keeps large tool outputs out of the agent's context. Before a large tool result reaches the model, TypeSafe's Jev (`typesafe/jev-latest`) judges each part of it against what the agent is trying to do. Irrelevant parts are replaced by a short marker:

```
[out-classify: lines 120–184 omitted, deemed not relevant to the task]
```

It works on text, images and PDFs.

## Install

```bash
pi install npm:pi-out-classify
```

Requirements:

- A TypeSafe API key, either in the `TYPESAFE_API_KEY` environment variable or stored in pi for the `typesafe` provider. Without a key, out-classify is off for the session, with a notice.
- For PDFs: [uv](https://docs.astral.sh/uv/). Not needed for text and images.

## How it works

1. **Which outputs:** every result the agent itself receives, including error results. Calls made inside a codemode script are not filtered, because the script needs the raw data; the script's own result is. Outputs below the thresholds pass through unchanged.
2. **The goal** the output is judged against:
   - `read` gets an optional `goal` parameter ("what should this read establish?").
   - Codemode scripts state theirs in a first line `// goal: …`. While the `codemode` tool is active, out-classify adds a rule asking for it to the system prompt.
   - Otherwise: the user's two latest messages.
3. **Text** is split into sections of at most `maxSectionChars`, Markdown (`.md` files, PDFs) at headings and paragraphs, everything else at line boundaries. Each section is classified as keep, uncertain or discard; only discarded ones are removed. Warnings, errors and needed context are kept, and in doubt sections are kept. Markers give the omitted line range. `read`'s "Use offset=N to continue" line is always kept.
4. **Images** in the output are classified by [pi-jevisual](https://www.npmjs.com/package/pi-jevisual), against the same goal.
5. **PDFs** (`read` on a `.pdf`) are converted with [Docling](https://github.com/docling-project/docling) to Markdown plus figures, instead of pi's raw PDF text. Large ones are then classified like text, with each figure judged together with the text that refers to it. Markers name the pages and headings.
6. **Formulas** in PDFs are cropped from the page as images, because the PDF's text layer flattens them (`s2` for `s^2`, fractions on one line). They belong to the section they appear in: the classifier judges that section with the formulas' text layer, and they are kept or removed with it. They are not classified as images and don't count toward `thresholdImages`.
   - With a `formulaModel` configured, the kept formulas are transcribed to Typst math with that vision model and stand in the text as `$ … $`. The model gets the image and the text layer: the image for structure and letter styles, the text layer to check the exact symbols.
   - Without one, or for formulas that could not be transcribed, they are returned as images after their section.

   Transcriptions are cached per formula image and model, so a formula is transcribed only once, even if the PDF is moved, renamed or edited elsewhere. A transcription made with a lower `formulaThinkingLevel` than configured is redone.

If classification fails, the affected sections and images are kept, with a notice. A PDF that cannot be extracted gives an error instead of raw PDF syntax.

## Examples

What the agent gets, shortened. Images are real image blocks in the result, shown here as `<image: …>`.

**Text:** `read` on pi's `docs/extensions.md` from line 1124, with the goal "how do I register a keyboard shortcut". Of 51,234 characters, 1,196 are kept. read's footer is always kept, so the agent can continue:

````
[out-classify: lines 1124–1835 omitted, deemed not relevant to the task]
[...]
### pi.registerShortcut(shortcut, options)

Register a keyboard shortcut. See [keybindings.md](keybindings.md) for the shortcut format and built-in keybindings.

```typescript
pi.registerShortcut("ctrl+shift+p", {
  description: "Toggle plan mode",
  handler: async (ctx) => {
    ctx.ui.notify("Toggled!");
  },
});
```
[...]
[out-classify: lines 1873–2275 omitted, deemed not relevant to the task]
[Showing lines 1124-2275 of 3348 (50.0KB limit). Use offset=2276 to continue.]
````

**Command output:** `bash` with `log show --last 2m --style compact | tail -n 400` (macOS's system log, like `journalctl`), after the user asked "are there any Wi-Fi problems in the system log". bash has no `goal` parameter, so the user's message is the goal. Of 51,196 characters, 21,252 are kept: the Wi-Fi and network entries stay, and the blocks in between are removed. Sections are judged as a whole (at most `maxSectionChars`), so unrelated lines next to relevant ones stay too.

```
[out-classify: lines 1–98 omitted, deemed not relevant to the task]
[...]
2026-11-14 12:35:00.533 Df wifip2pd[62200:bc6b75] [com.apple.awdl:interface] findConflictingService: allowOnenessWombatCoex is true
[out-classify: lines 112–194 omitted, deemed not relevant to the task]
[...]
2026-11-14 12:35:00.962 Df kernel[0:bc6286] () wlan0:com.apple.p2p.awdl0: setScheduleState[11180]: reason:UserTriggered sc:Infra Priority and force:NO, AWDL-restore:No
2026-11-14 12:35:00.971 Df WindowServer[517:bc59c2] [com.apple.Multitouch:Plugin-Apple Internal Keyboard / Trackpad] [TP] Dispatching digitizer event …
2026-11-14 12:35:00.980 Df kernel[0:bc6286] () wlan0:com.apple.p2p.en0:   set Datapath Open needsTrigger 0
[...]
2026-11-14 12:35:01.065 Df kernel[0:500e15] () wlan0:com.apple.p2p.awdl0: inputPacket: Dropping Multicast input packet because no peer was found for xx:xx:xx:xx:xx:xx
[...]
[out-classify: lines 302–330 omitted, deemed not relevant to the task]
```

**PDF:** an exam, with the goal "the first two questions". The cover pages and Question 3 are omitted. The formula tables on the cover pages are kept, because the questions need them. Formulas are transcribed to Typst math:

```
[out-classify: text omitted on pages 1–2 (## Exam Analysis III, ## Please do not turn the page yet!), deemed not relevant to the task]
<image: table of Laplace transforms>
[...]
## Question 1

- 1.MC1 [3 Points] Consider the following initial value problem:

$ cases(
  f'''(t) = cos(2t) + e^(-3t)\, & t > 0,
  f(0) = 0 quad f'(0) = 0 quad f''(0) = pi,
) $

Find the Laplace transform L(f) = F of the function f.

$ (A) F(s) = 1/(s^2(s^2 + 2)) + 1/(s^3(s + 3)) - pi/s^3 $
[...]
[out-classify: text omitted on page 7 (## Question 3), deemed not relevant to the task]
```

If nothing is relevant, the whole output becomes one line:

```
[out-classify: nothing in this output was deemed relevant to the task.]
```

If TypeSafe fails, the unclassified parts are included and the agent is told so:

```
[...]
[out-classify: 2 of 9 sections could not be classified (TypeSafe error) and are included unfiltered.]
```

## PDF setup

On the first PDF read, out-classify sets up Docling in pi's data directory (`~/.pi/agent/out-classify/` by default). That takes a few minutes and about 1.9 GB:

| Folder | Content |
|---|---|
| `venv/` | Python environment with Docling 2.144.0, created with uv (~1.2 GB) |
| `models/` | Docling's layout and table models (~670 MB) |

Both are kept across out-classify updates. When a new version pins another Docling version, they are rebuilt automatically. Delete the folder to remove them.

To use a Python that already has Docling instead, set `pythonPath` (see below). Then nothing is installed except the models. Another Docling version than 2.144.0 works, but gives a warning.

## Configuration

Optional `out-classify.json` in pi's data directory (`~/.pi/agent/out-classify.json` by default):

```json
{
  "thresholdChars": 8000,
  "thresholdImages": 3,
  "maxSectionChars": 2000,
  "pythonPath": "/path/to/venv/bin/python",
  "formulaModel": "openai/gpt-5-mini",
  "formulaThinkingLevel": "minimal",
  "cacheDir": "~/.cache/pi-out-classify",
  "cacheMaxAgeDays": 30
}
```

| Key | Default | Description |
|---|---|---|
| `thresholdChars` | 8000 | Text shorter than this (in total per output) is not classified. |
| `thresholdImages` | 3 | Fewer images than this are not classified. |
| `maxSectionChars` | 2000 | Maximum section length; at least 100. |
| `pythonPath` | – | Python with Docling for PDFs, instead of the managed venv. |
| `formulaModel` | – | `"provider/model"` of a vision model in pi's model registry that transcribes PDF formulas to Typst math. It needs image input and a login or key in pi; otherwise there is a warning and formulas stay images. Without it, formulas are returned as images. |
| `formulaThinkingLevel` | `minimal` | Reasoning effort of the formula model: `off`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. `off` sends none. Higher levels transcribe more carefully but take longer. |
| `cacheDir` | `~/.cache/pi-out-classify` | Folder for cached formula transcriptions. The default follows `$XDG_CACHE_HOME`, or `%LOCALAPPDATA%` on Windows. |
| `cacheMaxAgeDays` | 30 | Cached transcriptions unused for this many days are deleted; `0` keeps them forever. |

Paths may start with `~`; relative paths are relative to pi's data directory.

Changes apply after `/reload`.

## Notes

- **Privacy:** the parts of tool outputs that are classified, the goal and the tool arguments are sent to TypeSafe's API. With a `formulaModel`, the images and text layers of kept formulas are sent to that model's provider. The cache holds document content and is readable only by your user.
- **Costs:** classification calls are billed by TypeSafe. They are not counted as LLM usage in pi. Formula transcriptions are billed by the `formulaModel`'s provider (or count toward its subscription); cached formulas cost nothing.
- out-classify replaces pi's built-in `read` tool with one that has the extra `goal` parameter and otherwise behaves the same.
