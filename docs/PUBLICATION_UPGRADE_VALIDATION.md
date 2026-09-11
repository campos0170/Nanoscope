# Plotting and publication upgrade — 2026-09-11

Implemented in the working tree based on `779d4ec`. No application rebuild,
installation, DMG replacement, commit or GitHub push was performed in this task.

## Delivered

- Distribution summaries use explicit recorded-unit axes. Disabling an automatic
  SI prefix also clears its old tick multiplier; ECDF probabilities are no longer
  displayed as 0–1000 while their label omits the multiplier.
- Shared light/dark styling includes legend-text contrast. Reset restores the
  captured appearance, without changing plotted coordinates.
- Comparison and distribution pages have figure-local source labels/colors.
  Optional project defaults use existing source keys and the existing color
  registry; aliases do not change source identity, measurements or freshness.
  Population Plots can promote an edited name or chosen source color to a project
  default. Source-color changes there require a subsequent explicit Plot action;
  single-series, fit and cluster palettes keep their own meaning.
- Export provides conservative printed-text/scalable-stroke estimates, advisory
  targets, canvas enlargement and an uncalibrated print-size preview. Preflight
  messages stay in the dialog, not in exported artwork.
- Composition supports row/column spans, spacing and panel-label typography,
  plus a shared legend for faithfully captured, identically styled keys. Saved
  originals remain embedded without numerical changes; revisions save as new.

## Boundaries

This does not independently resize every axis font at a fixed journal width.
The enlargement action scales the complete physical canvas. At a fixed width,
adjust Appearance and capture again. Outlined glyphs, embedded/nested artwork,
cosmetic strokes and unsupported styling require manual inspection. Neither the
preflight nor its screen preview certifies journal compliance or absence of clipping.

Older figures and exported example files are not silently regenerated or
rewritten. Old snapshots remain usable; capturing new source figures is necessary
to obtain the extended Cartesian legend model. Unsupported shared legends/axes
are refused rather than dropping scientific keys or annotations.

## Verification

Interpreter: `/opt/anaconda3/bin/python`, Python 3.11.7, PySide6 6.11.1.
Commands used `QT_QPA_PLATFORM=offscreen` and the existing preference-isolated test
harness. The final five fresh-process groups passed **772 tests**:

| Group | Tests passed |
| --- | ---: |
| Distribution panels/workspace, axes, shared appearance, plotted-data and vector exports | 148 |
| Print preflight, export options, frozen bundles, Figure Library and analytical exports | 140 |
| Composition, axes, revisions, publication layouts and composer controls | 105 |
| Comparison panels/workspace/renderer, source-style registry, project persistence and editor | 131 |
| Distribution properties, baseline/Savitzky–Golay, amplitude/lifecycle/persistence, fit censoring/honesty and calibration | 248 |

The first group reported one joblib warning: physical-core discovery failed and
fell back to logical cores. Tests still passed.

An earlier long combined GUI process terminated with a native segmentation fault
while constructing `pyqtgraph.ViewBoxMenu` during the Figure Library fixture's
I–V window setup. It was **not a completed passing run**. The same selected modules
subsequently passed in the fresh-process groups above. The cause of that combined
process failure has not been established; this is a verification limitation, not
a claimed fix or proof of full application lifetime stability.

Regression assertions cover exact numerical-table preservation, no analysis
calls on appearance changes/restoration/export, retained zoom and source keys,
unchanged Figure Library freshness, residual opt-in behavior, legacy snapshots,
invalid metadata rejection and untouched original composition resources.
No detection, fitting, baseline or calibration algorithm files were changed.
`git diff --check` passed.

Visual inspection covered the corrected synthetic ECDF, source editor, collapsed
and expanded export dialog, composer controls and an unequal-panel shared-legend
result. Screenshots are in the task workspace under `work/plot-publication-qa/`,
`work/publication-export-upgrade/` and `work/export-preflight-*.png`; synthetic
examples are display checks, not additional apo-ferritin results.
