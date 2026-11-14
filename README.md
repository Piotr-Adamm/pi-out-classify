# pi-out-classify

A [pi](https://pi.dev) extension that keeps large tool outputs out of the agent's context. Before a large
tool result reaches the model, TypeSafe's Jev (`typesafe/jev-latest`) judges each part of it against
what the agent is trying to do. Irrelevant parts are replaced by a short marker:

```
[out-classify: lines 120–184 omitted, deemed not relevant to the task]
```

It works on text, images and PDFs.

## Install

```bash
pi install npm:pi-out-classify
```

Requirements:

- A TypeSafe API key, either in the `TYPESAFE_API_KEY` environment variable or stored in pi for the
  `typesafe` provider. Without a key, out-classify is off for the session, with a notice.
- For PDFs: [uv](https://docs.astral.sh/uv/). Not needed for text and images.

## How it works

1. **Which outputs:** every result the agent itself receives, including error results. Calls made
   inside a codemode script are not filtered, because the script needs the raw data; the script's own
   result is. Outputs below the thresholds pass through unchanged.
2. **The goal** the output is judged against:
   - `read` gets an optional `goal` parameter ("what should this read establish?").
   - Codemode scripts state theirs in a first line `// goal: …`. While the `codemode` tool is active,
     out-classify adds a rule asking for it to the system prompt.
   - Otherwise: the user's two latest messages.
3. **Text** is split into sections of at most `maxSectionChars`: Markdown (`.md` files, PDFs) at headings
   and paragraphs, everything else at line boundaries. Each
   section is classified as keep, uncertain or discard; only discarded ones are removed. Warnings,
   errors and needed context are kept, and in doubt sections are kept. Markers give the omitted line
   range. `read`'s "Use offset=N to continue" line is always kept.
4. **Images** in the output are classified by
   [pi-jevisual](https://www.npmjs.com/package/pi-jevisual), against the same goal.
5. **PDFs** (`read` on a `.pdf`) are converted with [Docling](https://github.com/docling-project/docling)
   to Markdown plus figures, instead of pi's raw PDF text. Large ones are then classified like text, with
   each figure judged together with the text that refers to it. Markers name the pages and headings.

If classification fails, the affected sections and images are kept, with a notice. A PDF that cannot be
extracted gives an error instead of raw PDF syntax.

## PDF setup

On the first PDF read, out-classify sets up Docling in pi's data directory (`~/.pi/agent/out-classify/`
by default). That takes a few minutes and about 1.9 GB:

| Folder | Content |
|---|---|
| `venv/` | Python environment with Docling 2.144.0, created with uv (~1.2 GB) |
| `models/` | Docling's layout and table models (~670 MB) |

Both are kept across out-classify updates. When a new version pins another Docling version, they are
rebuilt automatically. Delete the folder to remove them.

To use a Python that already has Docling instead, set `pythonPath` (see below). Then nothing is
installed except the models. Another Docling version than 2.144.0 works, but gives a warning.

## Configuration

Optional `out-classify.json` in pi's data directory (`~/.pi/agent/out-classify.json` by default):

```json
{
  "thresholdChars": 8000,
  "thresholdImages": 3,
  "maxSectionChars": 2000,
  "pythonPath": "/path/to/venv/bin/python"
}
```

| Key | Default | Description |
|---|---|---|
| `thresholdChars` | 8000 | Text shorter than this (in total per output) is not classified. |
| `thresholdImages` | 3 | Fewer images than this are not classified. |
| `maxSectionChars` | 2000 | Maximum section length; at least 100. |
| `pythonPath` | – | Python with Docling for PDFs, instead of the managed venv. |

Changes apply after `/reload`.

## Notes

- **Privacy:** the parts of tool outputs that are classified, the goal and the tool arguments are sent
  to TypeSafe's API.
- **Costs:** classification calls are billed by TypeSafe. They are not counted as LLM usage in pi.
- out-classify replaces pi's built-in `read` tool with one that has the extra `goal` parameter and
  otherwise behaves the same.
