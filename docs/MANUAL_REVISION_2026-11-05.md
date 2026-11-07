# Manual revision — 2026-11-05

This edition updates the plotting and analysis instructions for installed NanoScope
**45.37.98**, build ID `20261105T043813486839Z-nogit-5447ddda145a`.
Source SHA-256: `5447ddda145a83c1e898dab5d309da3a703739fd339745d340cb473f5c498813`.
Executable SHA-256: `d4c89fed8124f7726c1145bd0940a7e3191e25cdf394dc572dbb44de3b89c1c9`.

The current procedures were checked against the installed build's source and completed
offscreen acceptance evidence. Twelve scientific/lifecycle suites and ten compiled UI
workflows passed for that build. These results are application-validation evidence;
they are not a claim that this documentation revision reran every experiment or audited
every historical scientific result.

## Changed instructions

- **Figures and navigation:** the remembered Figures toggle and Ctrl/⌘Shift+L shortcut;
  direct New entries for Pair-plot matrix, Faceted comparisons and Individual-event overlays;
  separate Data & analysis and Format operations; viewport reset versus figure appearance reset.
- **Pending and completed results:** analytical drafts require Update plot; failed updates
  preserve the last result and pending/stale identity; exporting does not run a draft.
- **Curve fitting:** the current Curve fit… entry and Quick fit distinction; Source/Model,
  storage-unit fit range, Advanced starts/constraints/loss, Estimate starts, Reset constraints,
  Draft data and Last completed fit, diagnostics, Apply to figure, Remove applied fit and
  Export last completed fit… . The export instructions identify the original completed
  source/model/settings even when a later draft exists.
- **Subpopulations:** pinned recording/channel/detector identity, explicit curation update,
  refused missing/mismatched sources, original-source requirements for gallery/trace actions,
  and the distinction between an inspectable stored report and a valid live detection.
- **UMAP:** Generate UMAP from stored PCA scores, preserved existing report results,
  automatic display selection and saved-coordinate reopening. The older programmatic
  example retains its historical origin while the current button route is now explicit.
- **Noise and layout:** visible α/R²/descriptive-fit qualification, Full details and scrolling
  in small windows. No molecular-identity or calibrated-noise-model inference is implied.

Both `manual-markdown/manual.md` and `USER_GUIDE.md` carry the current procedure. The
control inventory gains an explicit Curve fitting area and the missing Figures/UMAP
controls. The generated edition header and manifest identify the installed build.

## Current illustrations and retained evidence

Nine selected screenshots from the compiled offscreen acceptance workflows are copied
unchanged to `manual-markdown/images/current_453798_*.png`. Their original paths,
fixture descriptions, screenshot SHA-256 and evidence-report SHA-256 are in
`manual-usability-captures.json`. They demonstrate current controls using synthetic
data. They do not replace or relabel earlier apo-ferritin figures.

The incubation chapter retains its dated 2026-09-29 review; main-analysis chapters and
historical recordings retain their earlier stated review dates, recipes, counts and caveats.
Earlier capture manifests and plan files remain unchanged. The retained Phase 4 captures
continue to identify development source `403311a6cb44`.

## Build and access

The manual is distributed as self-contained offline HTML, PDF and ZIP. The reviewed
45.37.98 app packaging and main-window menu do not provide a bundled manual/User Guide
route. Open these distributed files directly; updating them does not rebuild the app.
`BUILDING-THE-MANUAL.md` documents the background ReportLab build and the bundled
Markdown fallback. Final rendering checks and output hashes belong to
`manual-build-manifest.json` and the delivery verification report, not to a pre-build claim.
