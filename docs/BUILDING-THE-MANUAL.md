# Rebuilding the manual

Current documentation edition: **2026-09-29**, incubation workflow checked against
NanoScope build **40.15.59**. Other chapters and retained illustrations carry their
own earlier review dates. See `MANUAL_REVISION_2026-09-29.md` for scope.

    /opt/anaconda3/bin/python3 docs/gen_page.py --pdf --zip

That is the whole build. It runs from a clean checkout: every path is resolved relative to
`docs/gen_page.py`, nothing is read from `/tmp`, and nothing is written outside `docs/`.
Needs Python-Markdown (`python3 -m pip install markdown`); `--pdf` also needs Google Chrome
in `/Applications`.

When headless Chrome is unavailable, use the offline ReportLab renderer:

    python3 docs/gen_page.py --pdf --zip --pdf-engine reportlab --pdf-python /path/to/python-with-reportlab

For this edition's offline build, a verified environment combination is:

```sh
/opt/anaconda3/bin/python3 docs/gen_page.py --pdf --zip --pdf-engine reportlab \
  --pdf-python /Users/abrahamcampos/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

These are environment-specific interpreter locations, not paths required by the
manual content. Use equivalent local interpreters elsewhere.

The authoring Python needs `markdown`; the PDF Python needs `reportlab`, `Pillow` and `pypdf`.
They may be different environments. The ReportLab route reads the same verified HTML, includes
its prose, tables and embedded illustrations, and adds page numbers and a generated PDF contents
page. It uses the macOS Arial/Courier font files; adapt those paths before building on another OS.
Inspect rendered pages after either route. Browser and ReportLab pagination can differ.

The previous PDF is preserved if rendering fails. `--zip` requires `--pdf` in the same invocation
so an updated HTML document cannot silently ship beside an older PDF.

## What comes from where

| Source | Holds |
|---|---|
| `docs/manual-markdown/manual.md` | **all of the prose.** Edit this file — it is the source of truth. |
| `docs/manual-markdown/images/*.webp` | the screenshots. Dropping a `.webp` in and referencing it from the markdown is all that is needed; it gets inlined as a `data:` URI. |
| `docs/inventory.json` | the control inventory. `gen_manual.py` renders the `Control reference` section from it. |

The markdown's own `Control reference` subsections are **dropped** at build time and replaced
by the inventory-rendered tables: the inventory carries each control's shortcut, notes and
corrections as separate fields, which a markdown table flattens into one run-on cell. The
section's preamble paragraphs in the markdown are kept. Screenshots the prose already shows
are not repeated in the reference — the build tracks that through the `data-img` attribute
it puts on every figure.

Which screenshot heads each reference area, and its caption, is `AREA_IMG` in
`docs/gen_manual.py`. That is the only place manual captions live outside the markdown.

## Outputs

| File | Made by |
|---|---|
| `docs/manual.html` | `gen_page.py` — one self-contained file, images inlined, loads nothing from the network |
| `docs/manual-markdown/manual.html` | a byte-identical copy of it (`shasum -a 256` the two to check) |
| `docs/nanoscope-manual.pdf` | `--pdf`: Chrome with the page's print stylesheet, or the selected offline ReportLab renderer |
| `docs/nanoscope-manual.zip` | `--zip`: markdown + images + html + pdf, under `nanoscope-manual-md/` |
| `docs/manual-build-manifest.json` | edition, rendering engine, validation checks and source/output hashes |
| `docs/manual-current-captures.json` | current-source screenshot provenance, unchanged input-project hashes |
| `docs/manual-incubation-captures.json` | provenance for the 2026-09-29 incubation illustrations; earlier capture manifests remain dated |
| `docs/_build/reference.html` | only if you run `gen_manual.py` by itself; a scratch view of the reference section, not shipped |

## The build checks itself

`gen_page.py` fails rather than shipping a broken page. It refuses to write if an `<img>` is
not inlined, if the markdown references an image that is not in `images/`, if an internal
link points at an id that is not on the page, if a heading is missing from the contents rail,
or if the page grows a `<script>`, `<iframe>` or external stylesheet. The contents rail and
the section anchors are both generated from the headings, so they cannot drift apart.

The ZIP also includes the written User Guide, incubation companion, Zn/Cu/Acid workflow
guides, recording-specific reference guide, edition notes, inventory and verification
manifests. Finalize companion Markdown and captures before the distribution build.
Application executables/DMGs are not rebuilt by this command;
installed Help retains the bundled edition until a separate application release is built/installed.

## History

Before 2026-08-20 both scripts wrote into a git worktree that no longer exists and read
`/tmp/inventory.json`, `/tmp/docimgs.json` and `/tmp/manual_parts.json`, so neither would
run; and the page's prose was a second copy of the manual held as HTML string literals
inside `gen_page.py`, which is how it went stale against `manual.md`. There is now one copy
of the prose, and it is the markdown.
