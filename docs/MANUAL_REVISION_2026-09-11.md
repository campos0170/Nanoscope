# Manual revision - 11 September 2026

Updated against NanoScope0.1.0 source `703a96a6a50a7150f2158e5aa19d84055dc7c0d7`.
This is a documentation-only revision, not an application release or scientific algorithm change.

## What changed

- Current navigation, Essential/Advanced controls, Results tabs, Analysis home and Explorer.
- Automatic, known-current and selected-interval reference workflows, preview/redetection and refusal rules.
- Display-only subtraction versus analytical drift removal and Savitzky-Golay smoothing/subtraction.
- Optional amplitude definitions, editable event-level line and separation from original median depth.
- Background sub-state fitting, saved-result integrity, project snapshots and recovery.
- Distribution summaries, comparison figures, Figure Library, multi-panel composition and publication export.
- Current apo-ferritin and synthetic replication checkpoints, with calibration and oracle-assistance caveats.
- Updated control inventory and written User Guide; historical walkthroughs are explicitly scoped.
- Five current-source offscreen UI captures, loaded from unchanged saved projects without rerunning detection.
- Retired unsupported claims that normalization certifies comparability or changed counts prove false positives.
- Recorded the confirmed event-count heatmap display-orientation limitation; no app fix is included.

The current screenshots show source widgets, not an interactive installed-app smoke test. The
capture harness used isolated preferences/recovery settings and verified input-project hashes.
An exception was observed when switching away from the curation project after capturing its
Inspector (closed-source amplitude synchronization). Captures completed; this is not represented
as a clean end-to-end GUI validation. No application source was modified to suppress it.

## Analysis and plotting navigation expansion

The September 11 follow-up gives Analysis/plotting special emphasis:

- A navigation guide separating the Inspector, Analysis menu, Explorer tool tabs and Section dropdown.
- A complete first histogram exercise with source checkboxes, pending changes, styling, Library/project saving and export.
- Nine numbered current-gallery recipes plus scatter/lasso, KDE, fitting and PCA/clustering exercises.
- Exact measurement-name translation across Distributions, Distribution summaries and Comparison figures.
- Expanded Figure Library, composition and physical-export instructions, plus a symptom-based troubleshooting table.
- Four additional current-source screenshots of restored plotting pages; their capture manifest records an unchanged input project.
- PDF subsection bookmarks so individual recipes can be found without scanning the full control reference.

The companion User Guide now distinguishes live Distributions redraw/refresh from opt-in
Generate workflows, and correctly places scientific warnings in summaries/Details rather than artwork.
No application algorithm or GUI implementation was changed.

## Updated distribution

Canonical sources: `manual-markdown/manual.md`, `inventory.json`, and `USER_GUIDE.md`.
Generated outputs: `manual.html`, its identical Markdown-folder copy, `nanoscope-manual.pdf`,
and `nanoscope-manual.zip`. Build checks and hashes are in `manual-build-manifest.json`.

Rebuilding these documents does not replace the manual already embedded in an installed app or
DMG. Application rebuilding, installation and GitHub publication are separate explicit actions.
