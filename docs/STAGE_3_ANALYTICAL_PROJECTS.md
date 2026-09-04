# Stage 3 complete analytical-project contract

Status: **implemented and verified in project format v11**. The implementation preserves the
project-owned analytical workspace independently of whether its standalone analysis windows are
open. The complete headless regression finished with 8,114 passed, 8 skipped, and no failures.

## Outcome

A `.nanoscope` project must reopen as an analytical project, not merely as a collection of input
references and event tables. If a user configures a population plot, runs PCA or clustering,
finishes a subpopulation report, measures a PSD, or fits an I-V or voltage-dependence series, a
normal project save must keep enough structured state to recover that work without silently running
the analysis again.

Stage 3 has four governing rules:

1. **A saved scientific result is a snapshot.** Reopening restores its stored numbers, warnings,
   source identity, parameters, preprocessing decisions, and random seeds. It does not substitute a
   new run from a newer NumPy, SciPy, scikit-learn, TensorFlow, or plugin version.
2. **A recipe is not a result.** Plot controls, selected sources, styles, and analysis-panel forms
   are saved even when no computation has been run. Expensive or stochastic results are stored in
   addition to their recipes.
3. **Graphics are views.** Qt widgets, pyqtgraph items, Matplotlib figures, caches, and pixels are
   rebuilt from structured state. They are never serialized as Python objects or screenshots.
4. **Missing or changed inputs are disclosed.** A stored result remains readable and exportable
   when possible, but it is marked unavailable or stale for refresh. NanoScope never calls a
   detector, loads a model, or reruns an analysis merely because a project was opened.

Raw recordings remain path references. Stage 3 is analytical completeness, not a portable raw-data
bundle.

## User-visible behaviour

1. **File ▸ Save Project** and **Save Project As** save the analytical workspace through the same
   guarded, atomic `MainWindow` writer used in Stages 1 and 2. Gallery and Inspector remain intent
   emitters; no child window becomes a second project writer.
2. Changing a durable analysis control, accepting a completed analysis result, changing a model
   reference, or replacing a saved report marks the project visibly unsaved in the main window,
   Gallery, and Inspector. Hovering, resizing, moving a window, changing focus, selecting a table
   row, or mouse-panning an unpinned plot does not.
3. A project open does not produce a wall of top-level windows. Analytical surface state is restored
   into the project and installed when that surface is next opened. A surface that was closed before
   the save still keeps its last committed recipe and result; closing a widget is not a data-loss
   operation. Window visibility, position, and geometry are not project data.
4. Reopening a v11 project restores stored analytical results without dispatching background jobs.
   **Refresh**, **Run PCA**, **Run clustering**, **Run subpopulation analysis**, or a detector button
   is the explicit boundary at which a new result may replace the saved snapshot.
5. If a source recording or event table is absent, the saved analysis block is carried through the
   next save. The corresponding surface names the unavailable source and keeps any self-contained
   stored report/result viewable. It must not quietly reset to defaults or be dropped from the file.
6. If a source is present but its fingerprint or result lineage differs, the stored result is shown
   as **stale**. The old numbers and caveats remain the old numbers and caveats; the user may refresh
   deliberately.
7. A saved ML-model reference has three visible states: **matched**, **missing**, or **changed since
   save**. A missing/changed model disables a new ML run until the user locates or explicitly accepts
   a model, but never hides a previously stored ML detection.
8. A Stage 3-capable build continues to open projects written by versions 1–10. Their absent Stage
   3 blocks read as empty analytical state, without invented windows, results, model verification,
   or a false warning. Saving one with the v11 writer upgrades the file to v11.
9. An older application whose maximum format is v10 refuses a v11 project as needing a newer
   NanoScope. It must refuse before opening source recordings or replacing the current session.

## The v11 state boundary

The table describes the logical fields. Concrete dataclass names may differ, but the ownership and
round-trip guarantees may not. Every source reference uses the project's persistent recording key,
channel identity, and producing module id where relevant—never a live `doc_id`, list position, or
display label.

| Block | Durable inputs and view state | Durable scientific snapshot | Reopen rule |
| --- | --- | --- | --- |
| Analysis panel | Selected persistent target/channel/result; active detector; the complete live detection and preprocessing form (both Schmitt and CUSUM parameters); baseline configuration; ML preprocessing/decode configuration; manual-measurement method and other analysis controls that affect a future run | No unrun detector output is invented; existing detector/fits continue through the v1–v10 result records | Install after the selected source/result has been resolved. A draft form is restored even if it differs from the last stored run, and remains visibly unsaved until saved. |
| Population plots | `PlotConfig` in full: source keys/order, accepted-only filter, plot type, histogram/scatter/KDE choices, feature set, fit proposal and applied fit, axis pins and units, style, legend edits, and export size | Existing stored distribution fits plus the exact PCA/clustering snapshots described below | Feed restored sources first, apply the config second, attach computed snapshots last. This ordering prevents `set_data` from clearing the restored PCA/clustering result. |
| Clustering | Source-result identity and digest; ordered features; finite-row policy; scaling/preprocessing mode and learned centre/scale where applicable; method; requested/selected `k`; module/library version; integer random seed | Row-aligned labels including dropped rows, cluster sizes, means in original units, BIC search table, chosen `k`, warnings, and diagnostic metadata | Draw from the stored labels/means. A rerun uses the stored recipe and seed but is a new snapshot; bit identity across library versions is not promised by rerunning. |
| PCA | Source-result identity and digest; ordered features; finite-row policy; scaling/preprocessing mode and learned centre/scale; component count; module/library version; explicit random seed | Kept-row identities/order, scores, explained-variance values, components/loadings, warnings, and diagnostic metadata | Draw the stored scores/loadings. Do not call the PCA solver during project open. |
| Subpopulation report | Target recording/channel/detector and source-result digest; the complete `SubpopConfig`, including both seeds and all preprocessing/gating choices; accepted event-id set used by the run | Feature/gate tables, exploratory labels, clustering, mixture quantification and intervals, embeddings, median waveforms, event-to-subpopulation assignment, metadata, warnings, and saved headline | Reconstruct the report window/export from stored structures. Trace/Gallery colouring is reattached only when the target lineage matches; otherwise the report remains viewable with a stale/unavailable explanation. |
| PSD / FFT / spectrogram / all-points | Primary and comparison source keys; selected view; requested sample span; Welch length/window/overlap; PSD/ASD and axis choices; RMS band; spectrogram resolution/colour map; histogram bins/clipping/level markers; title, colours, background, and export size | The current computed per-source PSD curves and summaries and the active auxiliary view's numerical result, with source/calibration fingerprints and warnings | Rebuild plot items from the stored arrays. **Refresh** recomputes from the recordings and replaces the snapshot. A missing recording does not erase the stored spectrum. Non-active transient view caches need not be stored. |
| Density | Source identities/order, amplitude measure, accepted-only rule, bin count, log axes, colour map, titles and labels | Frozen two-dimensional counts, physical bin edges, marginal counts, accepted-event count, units and ensemble snapshot; the authoritative event table is not duplicated | Draw the stored grid and ensemble without calling density or ensemble kernels. A user edit/refresh creates a replacement snapshot. |
| I-V / conductance | Ordered recording identities, calibration/source fingerprints, and the point/exclusion provenance used by the view | Frozen point/exclusion table, fitted slope/intercept/conductance/R², units, warnings, and source membership | Rebuild the view from the stored point/fit snapshot. **Refresh** explicitly gathers the live recordings and creates a replacement snapshot. |
| Voltage dependence | Metric; accepted-only setting; signed or `|V|` axis; log-y flag; proposed and applied fit; source-result identities/digests and comparability state | Frozen per-recording points, exclusions, pooled-provenance verdict, fitted parameters/R²/dropped inputs, warnings, and details metadata | Rebuild from the stored series and fit. **Refresh** explicitly re-summarizes current event tables. Pending-vs-applied fit state must not collapse on reopen. |
| ML model | Path spelling, file size, full-file SHA-256 fingerprint, and model-format/input-shape metadata obtainable without importing TensorFlow | The model bytes are not embedded and the model is never deserialized on project open | A byte match restores the reference as verified. Missing or mismatched bytes produce the states above. Locating identical bytes at a new path updates the path and marks the project dirty; accepting different bytes establishes a new identity and never rewrites an old result's provenance. |
| Opaque extensions | Namespaced producer id/version and JSON-safe metadata; safe references to owned archive members | Plugin-defined tables or binary members, uninterpreted when their producer is unavailable | Preserve unknown namespaced blocks and their members byte-for-byte across open/save. Never import code or unpickle data merely to carry it. |

### Stored versus regenerated

This distinction is user-visible and testable:

| On reopen | Treatment |
| --- | --- |
| Curated event/measurement/fit tables | Loaded exactly as in v10. |
| PCA, clustering, subpopulation, PSD, I-V, and voltage-dependence numerical snapshots | Loaded, not rerun. Arrays, indexes, dtypes, warnings, source stamps, preprocessing choices, and seeds survive. |
| A fitted distribution's callable density | Regenerated from its already-stored parameters and metadata through the existing `density_from` path; the optimizer is not rerun. |
| Histogram/scatter geometry, fitted-line sample grids, captions, and Details formatting | Regenerated from saved structured data and configuration. |
| Qt widgets, windows, focus, selection highlights, worker objects, plot items, undo stacks, caches, and rendered pixels | Never stored. |
| Raw recordings and `.keras` model bytes | Never stored; paths plus identity fingerprints are stored. |

“Regenerated” above means a pure view calculation over stored numbers. It does not mean rerunning a
detector, preprocessing a raw trace, refitting a model, choosing clusters again, or replacing saved
caveats with the current build's opinion.

One explicit limitation remains: Population Plots preserves a KDE recipe but not a frozen sampled
KDE grid. Reopening that plot displays **Press Plot to recompute**; it never runs the density kernel
silently. Event Density is separate and does preserve its frozen count grid and bin edges.

## Identity, freshness, and replacement

Each analytical result records the exact inputs it describes:

- persistent recording/segment key, channel name/kind, and analysis module id/version;
- a digest of the source result or raw-source/calibration fingerprint, as appropriate;
- ordered source keys and the accepted-row/event-id set when selection affects the answer;
- preprocessing specification and learned scaling values;
- every random seed; and
- warnings/caveats from the original run.

Stage 3 does not add immutable event UUIDs. Therefore a PCA, clustering, or report result may be
reattached only to the exact stored source-result digest. Reusing equal-looking `event_id` values
from a re-detection is forbidden. Curation-only edits may be described as stale rather than
destroying the saved snapshot; a new detection lineage cannot inherit the old assignment.

An explicit rerun/refresh replaces only the analytical snapshot it owns. It must not overwrite a
different surface's config, reset its source choices, or clear an unavailable opaque record. The
replacement is a durable edit and participates in the same dirty-state/recovery revision as every
other project mutation.

## Format, integrity, and opaque carry-through

- The ordinary project manifest version becomes `11`; `MIN_READ_VERSION` stays `1`.
- New state lives under one versioned analytical-workspace section whose own optional sub-blocks
  have safe empty defaults. Large tables use Parquet as existing project tables do; numeric arrays
  use a non-executable, dtype/shape-preserving representation. Pickle and arbitrary Python object
  deserialization are forbidden.
- Archive member names are validated before reading. Analytical/extension members may not escape
  their namespace, shadow `manifest.json`, or collide with existing result members.
- A malformed known v11 block raises `ProjectError` naming the block/member. It is not silently
  defaulted away. An absent block in v1–v10 is different and reads as empty.
- Unknown plugin/future data must be namespaced. The loader retains its raw JSON and referenced
  members without interpreting them; a later save writes them back even if the plugin is absent or
  the associated recording could not load. Unknown core-format versions are still refused—opaque
  carry-through is not permission for a v11 reader to rewrite a v12 project.
- The entire file continues through the existing staged atomic writer and conflict guard. A failed
  v11 save leaves the previous project byte-for-byte intact and no partial analytical member behind.
- Recovery snapshots wrap the same `ProjectState`. Their GUI-thread clone must deeply detach every
  new DataFrame, index, array, mapping, and report object before the serializer worker receives it.

## Restore order

The order is part of the contract because several existing panel methods clear derived state:

1. Parse and validate the complete project before replacing the current session.
2. Restore file-backed documents, segments, calibrations, layers, results, fits, measurements, and
   colour identity through the established v10 path.
3. Resolve every analytical source reference from persistent identity to live identity. Retain
   unresolved blocks instead of deleting them.
4. Restore project-owned analysis-panel drafts and surface recipes. Do not emit dirty signals while
   installing saved state.
5. Attach stored numerical snapshots only after each surface has received its source data and
   configuration.
6. Publish stale, changed-input, missing-input, and missing-model states. No automatic job may have
   been dispatched.
7. Finish clean only when the in-memory state faithfully represents the project. A recovery restore
   remains visibly dirty under Stage 2's rule.

## Acceptance matrix and implemented tests

The Stage 3 tests cross the real project boundary and exercise both the Qt-free archive format and
`MainWindow` capture, lazy restore, source identity, dirty-state, recovery and no-recomputation
contracts. The principal additions are `tests/core/test_project_workspace.py`,
`tests/core/test_analysis_workspace.py`, `tests/gui/test_stage3_main_window_workspace.py`, and
`tests/gui/test_stage3_workspace_change_signals.py`; the existing persistence, recovery and panel
regressions provide the surrounding compatibility coverage.

| ID | Acceptance test |
| --- | --- |
| AP-01 | A fully populated v11 `ProjectState` round-trips every old and new field, with exact DataFrame indexes/dtypes and exact array dtype/shape/values. |
| AP-02 | The v11 manifest reports version 11 and references every required analytical member; no raw recording, model, widget, pickle, or absolute temporary member path enters the archive. |
| AP-03 | A v1–v10 fixture loads with empty analytical defaults, then saves as v11 without losing any legacy field. |
| AP-04 | A v12/future project is refused as needing a newer app. A frozen v10 reader refuses a v11 fixture before opening any recording. |
| AP-05 | `PlotConfig`, source order/selection, applied-vs-proposed fit state, axes, style, and legend edits survive a real MainWindow save/reopen. Sources are installed before config and results. |
| AP-06 | GMM, k-means, and PCA round-trip full snapshots, preprocessing/scaler state, integer seeds, dropped-row mapping, warnings, and source-result digest without invoking their solver functions during open. |
| AP-07 | A subpopulation report round-trips each optional branch (`None` and populated), tables, embeddings, waveforms, mixture/bootstrap output, seeds, warnings, headline, and event assignment. Its export after reopen is numerically identical. |
| AP-08 | PSD state round-trips primary/comparison sources, span, all estimator/view/style controls, per-source fingerprints, curves/summaries, and the active auxiliary result. Reopen invokes no PSD/FFT/spectrogram kernel. |
| AP-09 | Density recipe and publication style round-trip; reopening regenerates the same structured plot data from restored events without storing a raster or second event table. |
| AP-10 | I-V and voltage-dependence point/exclusion/fit snapshots and provenance round-trip. Voltage metric, accepted filter, axis flags, comparability warning, and pending/applied fit remain distinct. No fit function runs on open. |
| AP-11 | A matched model fingerprint restores verified; missing and same-path/different-bytes models are visibly unavailable/changed and never loaded. Relocating identical bytes can be accepted without changing the fingerprint. |
| AP-12 | A model and recording fingerprint detect a same-size, same-mtime byte rewrite. Fingerprints are content based, not metadata-only. |
| AP-13 | Changing each durable analytical control or completing/replacing each result advances the session revision and shows unsaved state on all save surfaces. Navigation-only actions do not. Restore itself creates no phantom edit. |
| AP-14 | Ctrl/⌘+S from Main, Gallery, and Inspector reaches the one writer and captures analytical state exactly once; cancelled/failed saves preserve the prior file, project path, disk stamp, dirty state, and recovery candidate. |
| AP-15 | A source/result changed after the saved run leaves the frozen result visible with a stale reason. Explicit Refresh/Run replaces it and marks dirty; opening alone does not. |
| AP-16 | A missing/unreadable source or model leaves its analytical block in carried state. Saving the partial restore and later restoring the input recovers the whole analysis. Deliberately deleting a result through an explicit UI action does not resurrect it. |
| AP-17 | Closing and reopening an analytical widget within one session does not discard project-owned state; opening a different project does not leak the first project's state into it. No top-level analysis windows are opened automatically. |
| AP-18 | Unknown namespaced JSON and binary plugin members survive an open/save cycle byte-for-byte with the plugin absent. They are never imported or executed. Malformed known blocks still fail clearly. |
| AP-19 | A Stage 2 recovery snapshot deeply detaches every new analytical frame/array, recovers it as visibly dirty, and an explicit revision-fenced save retires only the covered analytical revision. |
| AP-20 | Large analytical payloads remain bounded and atomic: injected ENOSPC/member-write failure keeps the old project intact and leaves no staging/replacing orphan. |
| AP-21 | The live analysis-panel draft round-trips independently of stored result parameters: persistent target/result selection, both Schmitt and CUSUM controls, preprocessing and baseline controls, ML preprocess/decode/range controls, and manual-measurement method. Installing it neither starts a run nor marks a clean restore dirty. |

## Verification commands

Run the focused contract first:

```bash
cd /Users/abrahamcampos/Desktop/nanoscope
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  /opt/anaconda3/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/core/test_project_workspace.py \
  tests/core/test_analysis_workspace.py \
  tests/gui/test_stage3_main_window_workspace.py \
  tests/gui/test_stage3_workspace_change_signals.py
```

Then run the persistence/recovery regression around it:

```bash
cd /Users/abrahamcampos/Desktop/nanoscope
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  /opt/anaconda3/bin/python3 -m pytest -q -p no:cacheprovider \
  tests/core/test_atomic_save.py \
  tests/core/test_project_guard.py \
  tests/core/test_project_segments.py \
  tests/core/test_recovery.py \
  tests/gui/test_project_io.py \
  tests/gui/test_adversarial_project_io.py \
  tests/gui/test_restore_carry_and_identity.py \
  tests/gui/test_dataloss_regressions.py \
  tests/gui/test_project_save_surfaces.py \
  tests/gui/test_project_recovery.py \
  tests/gui/test_recovery_controller.py
```

Finally run the complete headless suite before building a beta:

```bash
cd /Users/abrahamcampos/Desktop/nanoscope
PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg \
  /opt/anaconda3/bin/python3 -m pytest -q -p no:cacheprovider
```

## Explicitly outside Stage 3

Stage 3 does not embed raw recordings or model files, add cloud sync, create multi-user locks,
persist undo/redo history, add immutable event UUIDs, record reviewer identities/audit history, or
import label packages. It also does not promise that rerunning a stochastic method under a different
library version reproduces its last bits—the stored v11 snapshot is what preserves that result.
