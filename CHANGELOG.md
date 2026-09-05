# Changelog

All notable NanoScope changes are recorded here. The project is currently in private beta; beta
tags identify tested development baselines and do not promise a stable file or API surface.

## Unreleased

### Added

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

- 8,114 tests passed and 8 corpus/platform tests skipped in the complete Stage 3 regression.
- App and DMG build verification is intentionally deferred until the next implementation stage.

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
