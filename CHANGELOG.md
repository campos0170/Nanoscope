# Changelog

All notable NanoScope changes are recorded here. The project is currently in private beta; beta
tags identify tested development baselines and do not promise a stable file or API surface.

## Unreleased

### Added

- Publication-workflow hardening: revise saved compositions from their embedded original panels,
  restore layout controls, preview and save as a new figure with portable revision lineage. Nested
  panels and absent source entries are supported; project-generation guards prevent modal saves
  into another project, including an identical reopened project.
- Extended detached shared ranges to exact built-in readable-axis formatters, including Population
  millisecond/log axes and PSD notation, while preserving uniform point styling. Versioned models
  prevent older readers from silently dropping unit-aware formatting. Unknown overrides and varying
  point styles remain unsupported; actual PNG dimensions are bounded before decoding.
- Next-phase plotting: frozen multi-panel composition with ordering, panel labels, grid layouts,
  preview-before-save and compatible shared ranges on detached Cartesian graphics. Each original
  panel's exact tables and resources remain in composition bundles; unsupported plots retain
  independent axes instead of silently losing annotations or image content.
- Selection-only lasso for Population Scatter/PCA with exact event IDs, modifier toggles,
  explicit multi-source choice, gesture cancellation and unchanged curation/analysis results.
- Opt-in Distribution summaries page with validated ECDF, uncensored empirical survival,
  linear-quartile box plots and Scott-KDE violins. Explicit generation, exclusion/comparability
  disclosures, presentation-only styling, frozen workspace restore and Figure Library capture
  are separate from existing detection, baseline, calibration and fitting behavior.
- Completed the core plotting-workflow additions: shared Data / Analysis / Appearance / Export
  sections, reusable appearance-only presets, persistent warning summaries with full details,
  and explicit event-count heatmap versus KDE terminology.
- Project-owned Figure Library, separate from Event Gallery, with named thumbnails, exact frozen
  data/configuration/provenance, source status, duplication, renaming, export and reveal-file actions.
- Graphical publication-export previews, explicit residual-panel inclusion and self-contained,
  checksummed figure ZIP bundles containing PNG/SVG, CSV/Parquet and analytical resources.
- Extended in-place Population styling to colours, markers, lines, legends and fit decorations,
  including appearance-only undo/redo and preservation of pending analytical settings.
- Opt-in event amplitudes: whole-event median, mean, peak and histogram mode; explicit
  selected-interval/state median or mean; and an editable equivalent level in Curate.
  Original median `depth_pA`, fractional depth, SNR, detection masks and state fits are unchanged.
- Measurement-domain Inspector traces, explicit units/reference labels, amplitude-only undo,
  project/CSV audit fields, guarded batch application and stale-choice invalidation after
  mask, merge or baseline changes. Plot pickers distinguish selected amplitudes from median
  depth and disclose missing values and non-comparable measurement choices.
- Shared opt-in plot typography, I-V/voltage-series appearance presets, and project-saved
  presentation overrides. Title/label/color-map edits in event/noise plots and text/canvas
  edits in Distributions update existing items without a new analysis or zoom reset.
- A shared **Export current figure** dialog with PNG/PDF/SVG, native or mm/in sizing,
  aspect lock, 85/180 mm width presets, and PNG DPI metadata. Standard report generation
  is labelled separately. Existing explicit-path exports and numerical CSVs are unchanged.
- Implemented project-v11 complete analytical projects: resumable plot and source state; frozen
  PCA, clustering and subpopulation results with preprocessing and seeds; PSD, density, I-V and
  voltage-dependence snapshots; fingerprinted ML-model references; and live analysis-panel state.
- Restored analytical snapshots lazily without reopening windows or silently rerunning detectors,
  solvers, fits or spectrum kernels.
- Added stable recording/channel/result lineage, missing-source and missing-model carry-through,
  and opaque future/plugin artifact preservation.
- Added explicit backward compatibility for v1-v10 projects and refusal by older readers rather
  than allowing them to strip v11 analytical data.

### Validation

- 2026-09-06 packaging: rebuilt the macOS application and replaced the local and SanDisk
  DMGs. All 15 applicable packaged smoke checks passed; three optional real-recording/model
  checks were skipped because no inputs were supplied. Strict ad-hoc signature verification
  and both DMG integrity checks passed, and the DMGs have matching SHA-256 hashes. All 122
  embedded NanoScope modules match the current source code apart from rewritten filenames.
  The installed `/Applications` copy was left unchanged.
- Publication-hardening pass: 88 composition/revision checks passed, including real Population
  linear/log millisecond axes, real PSD replay, original snapshot/resource preservation, model
  version guards and project-switch cancellation. Additional targeted runs passed 108 export/
  capture checks, 74 existing axis/selection checks and 79 Library/Explorer/workspace checks
  (these groups overlap). Native and physical-size shared-panel exports and revision layout
  were visually verified. No scientific kernels or analysis defaults changed in this pass.
- Next-phase verification covered numerical references and unit-change invariants for all four
  new summaries, exact frozen restore/export, invalid-input refusal, composition resources and
  shared-axis safety, real lasso gestures and event-table routing, and unchanged automatic zoom.
  The existing baseline, detection, characterization, amplitude and workspace regression group
  passed all 394 checks; Explorer/workspace integration passed 101. Offscreen visual QA covered
  all four summary types and the multi-panel composer. These are targeted runs, not a full suite.
- Core plotting verification: 182 targeted checks passed across frozen previews/exports,
  appearance invariance, Figure Library persistence/provenance, all-page capture and staged-write
  safety. Broader plotting, numerical, workspace and responsive-layout suites also passed in
  targeted runs. Visual QA covered publication previews and smaller Explorer windows.
- Long combined GUI runs can intermittently hit the existing PySide6 6.11.1 signal-manager
  crash seen before this work; this is not claimed as a clean full-suite run.
- 870 distinct focused amplitude, detector, characterization, persistence, curation and
  plotting regression checks passed; offscreen visual checks verified editable-level and
  selected-interval layouts. These source checks preceded the packaged build described above.
- 8,114 tests passed and 8 corpus/platform tests skipped in the complete Stage 3 regression.

## 0.1.0 Beta 1 — 2026-09-03

### Added

- Interactive ABF, HDF5, TDMS, CSV, and text-trace viewing and calibration.
- Schmitt, CUSUM, ML, and manual event detection with provenance-preserving reconciliation.
- Event inspection, batch curation, structured labels, and supervised-learning exports.
- Histograms, density and scatter plots, fitting, PCA, clustering, PSD, I–V, and voltage-dependence
  analysis.
- Project persistence with guarded atomic saves and external-change conflict detection.
- Crash-recovery autosaves after 15 seconds idle, bounded by a 60-second maximum, with two atomic
  generations and Recover, Discard, and Show in Finder startup actions.
- Adaptive local-context trace navigation for efficient manual annotation.

### Validation

- 8,060 tests passed and 6 corpus/platform tests skipped in the complete Stage 2 regression.
- The packaged macOS application passed all 18 bundle self-tests, strict ad-hoc signature
  verification, and DMG checksum verification.

### Distribution limits

- The beta build targets Apple Silicon and macOS 15 or later.
- The application is ad-hoc signed but not Apple-notarized; Gatekeeper may require the documented
  first-launch workaround.
- Raw recordings are referenced by path and are not embedded in projects or recovery snapshots.
