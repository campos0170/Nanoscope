# Changelog

All notable NanoScope changes are recorded here. The project is currently in private beta; beta
tags identify tested development baselines and do not promise a stable file or API surface.

## Unreleased

### Changed

- The event table, its Summary / Fits / Quality tabs and the Merge sub-events strip moved out of
  the Inspector into a new **Results** dock under the trace (View ▸ Panels ▸ Results, ⇧⌘R; close
  it to give the trace its full height — nothing is lost). The Inspector keeps only the parameter
  forms, so they get its full height back (Detection page 256 → 592 px at 1512x900). The Results
  dock opens tall enough for six table rows, capped at a third of the window (at least four whole
  rows at 1512x900), and its two header rows have a fixed height: the toolbar (Table selector,
  Columns…, Merge…, the table's constant columns, advisories) and one status line — the event
  count, or the stale notice, which is now one amber line (verdict, action, Details for the full
  sentence) instead of a wrapped banner up to 254 px tall. When the action is unavailable the
  line starts its reason, giving up the verdict first on a narrow panel. Table rows are 24 px on
  a Mac (one text line + 8 px). A short panel (or Merge… open) scrolls the Events page instead of
  squeezing the table. ⇧⌘R hides the panel with the focus parked on the trace and, shown again,
  hands it back to the table, so a/r/f keep working.
  Saved layouts keep their dock widths and gain the Results dock; the layout is now saved as
  version 1, so a downgraded build resets the layout once and names `windowState` on its info bar.
- Keep scientific warnings and fit verdicts in Details/readouts and saved figure
  provenance, not on plot titles, fit legends or Event Gallery waveform captions.
  Optional fit statistics, units, curves and numerical exports are unchanged.
- Standard report PDF/PNG files retain diagnostic context in document/image
  metadata. New Figure Library captures include full Details text; existing frozen
  figures and previously generated guides are not rewritten.

### Added

- The **Event inspector can dock beside the trace** (opt-in; it still opens as its own window by
  default): **Dock beside trace** / **Own window** on its toolbar, or **Window ▸ Dock Event
  inspector beside the trace**, moves the same window — event, filter and context kept — into a tab
  beside the Inspector panel (left or right column only), and the choice is remembered. Its
  minimum fell from 1107×827 to about 452×550 px (452×440 docked): the toolbar is two rows in a
  wide window and five short ones in a column, the amplitude row is two, the waveform's floor is
  180 px, and the readout scrolls and collapses under **Details ▾ / ▸** (hidden when docked, shown
  in a window, each remembered), with the Label row in the fixed bar under the waveform. The
  staleness sentence is a one-line chip (all of it in its tooltip, ⚠ on the Details button).
  Docked, ⌘S / ⇧⌘S / ⌘Z / ⇧⌘Z
  are the main window's own items (so neither copy goes dead as an ambiguous pair), a click anywhere
  in it or on its tab takes the keyboard, a tab sent behind hands the keyboard to the trace, and Tab
  never lands in the panel behind. In a short column the part under its toolbar scrolls, so no
  amplitude method, note or Details state pushes the main window past a 1280×800 screen. The
  waveform's title no longer widens the plot past its widget: a narrow plot draws a shorter form
  that keeps depth and dI/I0 whole (all of it in the tooltip), and a short axis reads **I (pA)**.
  The amplitude note reserves its height, so a staleness flip no longer moves the waveform.
- A **Window** menu: Bring All to Front; the main window, Event inspector, Event gallery,
  Analysis Explorer (and its open pages), Figure Library and the other windows NanoScope has
  opened, checked while on screen; Dock Event inspector beside the trace; and **Reset Window
  Layout…** (asks first). Child windows now reopen at their remembered size and place, validated
  like the main window's and brought back onto the current screen.
- Curation throughput in the Event inspector and the Event gallery: **Advance after decision**
  (off by default, one remembered choice for both windows) makes the A / R / Space / label keys
  move on to the next event of the filter — in the gallery only for a one-tile selection, turning
  pages; buttons never move on and decision keys no longer auto-repeat. New **Unreviewed**,
  **Unlabelled** and **Labelled: key=value** filters; **First unreviewed** (Ctrl/⌘+U); a
  fixed-height progress line (*Reviewed 312 / 870 · accepted 801 · labels: …*); a Label row
  (Apply / Clear) for the inspector's current event; ⌘Z / ⇧⌘Z in the inspector on the main
  undo history. Edit ▸ Undo / Redo no longer reset the inspector to All / the first event or move
  the gallery to tile 0 when the restored table holds the same events. An inspector readout line
  too long for the window now ends in "…" instead of widening the window, and in the gallery a
  decision that takes the focused tile out of the Show subset leaves the cursor beside it (the
  next ← / → selects its neighbour, not the page's first tile).
- Signal Conditioning window (Analysis ▸ Signal conditioning…, Ctrl+Shift+F; also the
  Preprocess card's Open button, the Filter step and the baseline workspace's Advanced
  signal-chain settings): the classical
  detectors' stage editor, moved out of the Inspector into one built-once window with an
  execution-order stage list, notices below the form, the full-height response view, protocol
  Import/Export (versioned JSON, never overwriting) and Copy as Python. The Preprocess tab keeps
  a fixed-height card. When the form's chain differs from the Schmitt/CUSUM detection on screen,
  the event table shows "⚠ Chain changed" with Revert / Run / Open, and the card and window name
  the differences. Ctrl+S / Ctrl+Shift+S save the project from the window, and its Run
  detection… follows the Detect button's gate. Recalling a protocol now sets every stage
  (stages it does not name, and the
  low-pass, are turned off), and its name stays on the combo only while the form equals it —
  a value the recording cannot hold (a cutoff above its Nyquist limit) drops it to "— manual —"
  and the status bar names what changed.
- Signal Conditioning previews of the main trace's visible window, computed on one background
  thread 250 ms after the last edit (a newer edit cancels the older preview): Time (raw against
  processed — Overlay, Split, Raw − processed — split onto the processed trace's real unit when
  Normalize or a high-pass changes it), Spectrum (Welch before/after with a deterministic line
  finder and Send to notch through the form's own checks) and Response (Magnitude plus Phase,
  Group delay and Step, applied zero-phase against the single-pass design). The status line says
  "Window estimate" while baseline drift or Normalize (or a stage settling longer than the 40 s
  padding, which it names) make a window inexact; Compute exact runs the whole recording on
  demand only, and the status line leads with what is happening (Computing exact…, Exact pass
  failed…, Updating…) before the current chain's verdict. A mains line is snapped onto its
  harmonic only when Δf ≤ 1 Hz. Send to notch on an off notch that keeps a list adds to that list
  and leaves the notch off. The window keeps its one preview thread from first open to quit.
- Opt-in Comparison figures page: shared-axis per-recording facets, feature-pair matrices with
  common-bin count histograms, and strip/swarm individual-event plots with optional descriptive
  boxes. Seaborn renders prepared observations; generation is explicit, source identities and
  exclusions are retained, and large/crowded inputs are refused instead of silently sampled.
  Appearance and workspace restore use frozen geometry. Native vector export, publication preview,
  Figure Library capture and bundles preserve exact tables and provenance; warnings stay in Details.
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

### Fixed

- Reopening a project with ML detections no longer shows a false "open-pore estimator
  changed" banner when the stored windows differ from the defaults. The ML branch of the
  stored-detection seeding restored only the ML tab and left the baseline form on the app
  default, so the banner appeared and its "Re-measure depths" button would have re-measured with
  the defaults instead of the windows the table was measured with. ML results saved without
  stored parameters (pre-v3 projects) are unchanged. The baseline form is shared by every
  detector on a channel, so a table of another detector measured with different windows still
  reads as stale until it is selected.
- The ML preprocessing median (`_decimated_median`) is a rolling order statistic instead of
  ndimage's O(n*w) rank filter: about 40-50x faster at the 2501-tap Host-guest / Zn windows
  (6.2 s to 0.16 s at 3M samples; 31 s to 0.6 s at 1.5M samples after decimation). Values are
  identical on 240 synthetic cases and the 35 real-recording corpus tests (the sign of a zero
  can differ when -0.0 and 0.0 are mixed). Non-finite input and windows longer than the signal
  keep the previous path. The pinned md5 of `ml_preprocess.py` was re-pinned with this
  verification; the median's arithmetic now also depends on
  `acid_incubation_numerics.median_filter_1d`, which has its own tests.
- Voltage dependence and its menu gate read tables from any primary detector (Schmitt, CUSUM,
  ML): the one stored on the most recordings that can become a point (Schmitt on a tie) is
  pooled for the whole series, recorded in saved workspaces, and named in the Pooled provenance
  block when it is not Schmitt. An ML-only project previously had the action greyed out.
- Event inspector: opens on the event selected in the table (and, re-opened, where you were)
  instead of event 0; keyboard focus starts on the waveform instead of the "Event N" spin box,
  so A/R/T and label digits reach their shortcuts; the first event no longer draws against a
  0..1 pA placeholder range when the stale-measurement note arrives after the first draw.
- Analysis > Event labels... fits a laptop screen: its explanation line wraps instead of
  forcing a minimum width wider than the screen (1777 px at the app's 13 pt font).
- The trace view gives the current channel more height than the command-voltage pane (5:2
  instead of equal rows).
- "Known open-pore current" accepted 0 pA (and anything that underflows to 0 in the
  float32 pipeline that reads it, e.g. 1e-300) with no error, then silently produced a
  NaN fractional depth on every event and a NaN Summary median with no explanation.
  Refused at the editor, at the precision the analysis pipeline actually reads it under.
- Three numeric boxes (open-pore current, notch lines, acquisition bandwidth cutoff)
  each guarded only against PEP 515 underscore grouping (`1_20` -> 120.0); none guarded
  against the non-ASCII digit scripts bare `float()` also accepts (`١٢٠` -> 120.0). A
  shared strict-ASCII parser now backs all three.
- The per-event local baseline-intrusion advisory could report "assessed" with a
  near-zero fraction on exactly the two canonical Automatic-mode baseline failures
  (isolated long events; high-duty recordings), because it compared a fragment's two
  flanks to each other rather than to a wider, already-computed reference. Gated on the
  recording-level baseline-health verdict baseline.py already produces; advisory only,
  no detector output changes.
- The HDF5 reader read t0 from a `/time` column it had already declared unusable for a
  sample rate (all-NaN or reversed), producing a NaN or backwards origin with no issue
  naming the problem. Returns the documented "unavailable" sentinel with an explicit
  issue instead.
- `recharacterize_span` accepted a negative sample rate (a wrong-signed row, no error)
  and raised the wrong exception type for zero, infinite, or `None`; now raises
  `ValueError` naming `fs`, matching `detect_events`'s existing convention.
- Opening Analysis > I-V on a large, never-before-medianed recording blocked the GUI
  thread for the whole whole-channel median (~300 ms observed) with no busy state. The
  first gather now runs on a background thread once a channel exceeds ~500,000 samples;
  the raw arrays it needs are resolved on the GUI thread first and handed to the worker
  as plain data, never a live `Session` reference, after an earlier version of this fix
  was found and rejected in review for exactly that hazard.
- A project reset/switch deleted the I-V window directly, bypassing the retirement step
  every other background-worker-holding window goes through at teardown; a gather still
  in flight was left running with nothing left to deliver its result to.
- Practice-recording duplicate detection compared paths with `==` instead of the one
  function that answers "is this file already open" -- the literal pattern an
  architectural guard exists to forbid -- and carried a latent case-sensitivity defect
  on a case-insensitive volume as a result.
- Two dialogs (Help > About; the open-pore reference preview) set `WA_DeleteOnClose`,
  the one lifetime pattern `nanoscope/gui` otherwise avoids everywhere: the app's
  teardown sweep calls `close()` on every owned window as a stop mechanism, which is a
  hide for every other window and would have been a delete under a live Python wrapper
  for these two.
- Several architectural AST guards had gone stale as legitimate shared infrastructure
  (plot-appearance/workflow-bar/label modules, a lasso-selection primitive, a
  publication-export dialog) was adopted by multiple analysis windows without the
  guards' exact-import-set assertions being updated to match. Each newly-allowed module
  was individually verified -- no widget of its own, no further reach beyond the same
  shared, non-panel set -- before being added, rather than the assertions being loosened
  wholesale.

### Validation

- 2026-09-13: all six items confirmed still open from the 2026-09-12 beta sweep (a
  zero open-pore current; non-ASCII digit acceptance in three boxes; the baseline-
  intrusion advisory's false negative; the HDF5 t0 defect; `recharacterize_span`'s
  fs validation; the I-V first-open stall) are fixed and independently reviewed;
  root-cause fixes throughout, not input-specific patches. Nine other items from that
  sweep, including the acquisition-metadata save failure, were found already resolved
  by intervening work and are not claimed here. 358+ targeted tests re-verified across
  measurement quality, the reader contract, characterize/span/flag-invariant guards
  and I-V; 455 of 457 tests across the sibling-import guard files (the two remaining
  failures are a pre-existing, unrelated geometry pin).
- 2026-09-13: root cause found for the intermittent PySide6 signal-manager crash noted
  below -- once a Qt wrapper's C++ side is destroyed, its Python object becomes
  permanently invisible to Python's own cyclic garbage collector. Confirmed on PySide6
  6.11.1 and 6.11.2 both, and confirmed independent of this suite's `gc.disable()`
  policy (a separate, unrelated deadlock avoidance; re-enabling it does not free these).
  Measured: ~734 leaked `QObject`s and ~3.6 MB RSS per `MainWindow` construct/destroy
  cycle, linear with no plateau across 40 cycles. This is a PySide6 characteristic, not
  fixed here; the manual full-regression CI job now runs the `gui` shard as 16 smaller
  processes instead of one, bounding rather than eliminating the accumulation a single
  long process would otherwise reach.
- Comparison-figure source pass: 571 focused tests passed across rendering, page interaction,
  workspace persistence, Library capture, native vector/export composition and scientific
  regressions. All 37 recorded analysis-module hashes are unchanged. Read-only apo-ferritin
  QA used all 2,104 saved accepted events for the three new views, checked PNG/PDF/SVG exports,
  and left original event tables and project bytes unchanged. This source pass preceded the
  packaged comparison release below.
- 2026-09-06 comparison packaging: rebuilt the Apple Silicon app and replaced both the local
  and SanDisk DMGs. All 16 applicable in-bundle checks passed, including Seaborn facets, pair
  matrices, strip/swarm views, source-free restore and twelve PNG/PDF/SVG exports; three optional
  real-recording/model checks were skipped because no inputs were supplied. All 27 packaging
  regressions passed. Strict ad-hoc signature verification and both DMG integrity checks passed;
  the images have matching SHA-256 `3ffe9e901c7032169496b3303b4dc27ebec56e3eb8c0c9c70fad52b79c9a48d6`.
  All 125 embedded NanoScope modules plus the packaged self-test match current source bytecode,
  apart from normalized filenames. The installed `/Applications` copy was not replaced in this pass.
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
