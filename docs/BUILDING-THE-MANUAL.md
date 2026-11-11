# Rebuilding the manual

Documentation edition **2026-11-11** describes the controls in installed **NanoScope build
46.31.52**, build ID `20261111T163237642239Z-36aa140fa2b2-dirty-8c6f5e37d549`. Its source SHA-256 is
`8c6f5e37d54985a5869391c104f783eb55b29b1ec3b29bf91492254c2894209c`.
The prose covers the changes since the 2026-11-05 edition; the illustrations were not
re-captured and keep the build named in their captions. Earlier scientific chapters,
apo-ferritin outputs and capture manifests retain their original dates and limitations.
See `MANUAL_REVISION_2026-11-11.md` for the current scope.

## Authoritative sources

| Source | Purpose |
|---|---|
| `manual-markdown/manual.md` | Illustrated manual prose; edit this source rather than generated HTML/PDF |
| `USER_GUIDE.md` | Companion written guide; keep shared control names and procedures aligned |
| `inventory.json` | Control reference, including individual notes and shortcuts |
| `manual-markdown/images/` | Actual PNG/WebP screenshots, embedded into generated HTML |
| `gen_manual.py` | Generates the reference tables; `AREA_IMG` contains reference-area captions |
| `gen_page.py` | Builds and validates HTML, PDF, ZIP and build manifest |
| `render_manual_pdf.py` | Offline ReportLab PDF rendering |
| `manual-usability-captures.json` | Current-build screenshot hashes, original evidence paths and fixture descriptions |

The markdown's own Control reference subsections are replaced by the inventory-rendered
tables; the section preamble remains. The build omits a repeated reference screenshot
when the prose has already shown it. Historical images and manifests must not be relabelled
as new captures. The current images are unaltered offscreen captures from synthetic
validation fixtures, not fresh apo-ferritin measurements.

## Build without using the screen

Run from the application checkout or the staged workspace containing `docs/`:

```sh
/Users/abrahamcampos/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  docs/gen_page.py --pdf --zip --pdf-engine reportlab \
  --pdf-python /Users/abrahamcampos/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

Paths are environment-specific; use equivalent local runtimes elsewhere. The preferred
Markdown engine is Python-Markdown. When it is absent, `markdown_backend.py` uses the
bundled Node `marked` engine through the local bridge while retaining legacy heading IDs.
The selected engine is recorded in `manual-build-manifest.json`. The PDF interpreter
needs ReportLab, Pillow and pypdf. ReportLab uses local macOS Arial/Courier fonts;
adjust the font paths for another operating system. No screen or desktop browser is
needed for this route.

All content paths resolve relative to `gen_page.py`; outputs stay under `docs/`.
The previous PDF is preserved if rendering fails. Use `--pdf --zip` together so an
updated HTML file cannot ship with an older PDF. The optional Chrome renderer remains
available in the generator, but is not needed for the background documentation build.

## Verify and distribute

1. Confirm the prose, inventory and screenshot captions name the intended build.
   Match every current image to `manual-usability-captures.json`; preserve older manifests.
2. Run the build. It checks image embedding, missing image files, internal links,
   contents headings and absence of scripts, iframes and external stylesheets.
3. Render the PDF to page images and inspect the new control chapters, reference tables,
   captions, page breaks and representative retained chapters. PDF pagination can differ
   from browser print layout. Text/link checks alone do not establish readable layout.
4. Confirm both HTML copies have identical SHA-256; inspect the ZIP for the matching
   PDF, source prose, images, inventory, companion guides and capture/build manifests.
5. Distribute the generated files and record their hashes. Documentation changes do not
   require modifying or replacing the application bundle.

| Output | Contents |
|---|---|
| `manual.html` | Single offline HTML manual with embedded images |
| `manual-markdown/manual.html` | Byte-identical HTML copy beside the source |
| `nanoscope-manual.pdf` | Paginated offline manual with contents and page numbers |
| `nanoscope-manual.zip` | Matching manual, source, images, PDF and companion documents under `nanoscope-manual-md/` |
| `manual-build-manifest.json` | Edition/build identity, render engines, validation checks and hashes |
| `_build/reference.html` | Optional scratch output from running `gen_manual.py` directly; not a second manual source |

The installed 45.37.98 app's reviewed packaging specification does not collect these
manual files, and the reviewed main-window menu does not provide a User Guide/manual
action. Open the distributed HTML or PDF directly. Do not claim that an in-app Help
manual has been updated or that documentation must wait for another app release.

## Retained provenance

`manual-current-captures.json`, `manual-plot-navigation-captures.json`,
`manual-phase1-captures.json`, `manual-phase4-captures.json`,
`manual-design-connections-captures.json` and `manual-incubation-captures.json`
document earlier captures and keep their original identities. Their corresponding
capture-plan files describe those earlier handoffs. The 2026-09-29 incubation chapter
and 2026-09-11 main-analysis review remain historical in this edition. The build includes
available companion guides and dated revision notes without claiming every scientific
workflow was rerun for this documentation update.

Before 2026-08-20, the scripts depended on a removed worktree and temporary inventory
files, and the prose existed twice. The maintained build now reads local documentation
sources, with one authoritative illustrated prose file.
