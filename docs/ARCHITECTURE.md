# NanoScope — Architecture

Design notes for developers and maintainers.

## Layering

The package is layered so the analysis core never depends on Qt, and the GUI never re-implements
analysis:

```
nanoscope/
  io/        format readers (ABF/HDF5/TDMS/text) behind one dispatch registry, over one shared
             answer-contract (io/contract.py) — no GUI, no analysis
  core/      data model + state hub (Session, Document, calibration, identity, types), the analysis
             *seam*, the undo door (history), the project (de)serializer, the write seam
  analysis/  pure DSP/ML: filters, baseline (+ baseline_probe), detect, characterize, events, stats,
             populations, measure, remeasure, psd, iv, merge, scoring, truth, and the ML pipeline
  plot/      matplotlib publication figures + style (static export path, separate from the live viewer)
  gui/       PySide6 + pyqtgraph widgets; the only layer that imports Qt
```

Rule of thumb: `analysis/` and `io/` are importable and testable with no display; `gui/` orchestrates
them. Everything in `analysis/`/`core/` runs headless in CI.

## State model

`core/session.Session` is the **single mutable state hub**. Each GUI panel holds exactly one
reference — to `Session` — and communicates through its mutators and Qt signals. **No panel imports
or references another panel**; `main_window.py` is the *only* GUI module allowed to import other
panels, and it wires them together. This keeps the panel graph a star, not a web, and is enforced by
a test that walks the `gui/` ASTs.

- **Documents.** `core/document.TraceDocument` is an immutable value: metadata + per-channel
  contiguous **float32** signal arrays. The time axis is **float64** (a float32 time axis is
  non-monotonic at 15 M samples and breaks rendering).
- **Calibration** is separate from the document (`core/calibration`): the same raw data can be viewed
  under different gains/offsets. Calibrated arrays are produced on demand and cached; identity
  calibration returns the raw array by reference (no copy).
- **Signal loading** is eager (whole channel into RAM) and happens **off the Qt main thread** via a
  worker; ~15 M samples/channel is the design ceiling, above which the loader offers a sample range.
- **Identity** is three questions, not one, and `core/identity.py` is where they are kept apart.
  Conflating them into "the resolved path" caused three separate data-safety defects.
  - *Which load?* — `doc_id`. Opaque, unique to **this load in this session**; only `core/identity`
    may build it (`make_doc_id`) or take it apart (`source_of`), and an AST guard fails the build if
    anything else derives an identity from `meta.path.resolve()`. It is stamped once in
    `TraceDocument.__post_init__` — the one funnel every recording passes through — and preserved
    across `dataclasses.replace`, so it is stable for the life of the load. This is what makes the
    ~50 `session.has_document(res.doc_id)` guards mean what they always assumed: a detection that
    lands after its recording was closed **and reopened** no longer installs itself onto the new
    document.
  - *Which recording, across sessions?* — `core/segment.persistent_key(doc)`: the resolved source
    path for a file (`Document.source_key`), the segment uid for a derived segment. Never the
    display name (a segment may be named after its parent) and never `doc_id`. Results,
    calibrations, layers, manual measurements **and the analysis windows' overlay sources**
    (`core/sources.py`) are keyed by it. "Is this *file* already open?" is therefore
    `Session.doc_id_for_source`, not `has_document`.
  - *Which bytes?* — `identity.SourceFingerprint`, stamped into each project's `DocumentRef` (v4).
    A path is a name, not a recording: re-exporting over the same filename used to reattach saved
    events to completely different data in silence. On mismatch the restore **warns and explains**
    (naming what differs and how many restored events now lie past the end of the file); it never
    refuses to open, and an absent or unreadable stamp reads as *unknown*, never as *verified*.
- **Undo** is `core/history.EventHistory`, one snapshot stack over every `EventSet` mutation in the
  session, keyed `(doc_id, channel, module_id)`. It is deliberately **not** a `QUndoStack`, and this
  is the load-bearing decision in the file: `QUndoStack.push()` *performs* the command, so every
  mutation site would have to build a `QUndoCommand` that knows how to do and un-do itself **before**
  it mutates. This app mutates an `EventSet` from **eleven** call sites — six install a new result
  (a finished detection, a batch detection, a merge from any of its three entry points, a hand-added
  missed event, a re-measure, a project restore) and the other **five write the shared `DataFrame`
  in place** from inside a Qt slot on a widget (the table's `accepted` checkbox, the Inspector's
  single accept/reject, Accept-all/Reject-all, a mask drag, the table's flag cell). Retro-fitting a
  command object onto each is exactly the "fix applied at N call sites" shape this project has
  watched come back three times: the twelfth mutation, written next month, forgets to push and is
  silently un-undoable. Instead the history keeps a pristine deep copy of every live result and
  **compares**.
  A mutation does not have to announce itself to be captured; announcing it only sharpens the menu
  label, and even the label is derived from the diff (`describe_change`). A forgotten announcement
  costs a word, never a row. `main_window.py` is the only module that touches it (AST-guarded) — it
  is the one object that already receives every announcement of an `EventSet` mutation
  (`resultsChanged`, `eventAcceptanceChanged`, `eventFlagChanged`, the Inspector's `eventEdited` and
  `curationBatchChanged`), all routed to the one recorder. The redundancy there is deliberate and
  free: a checkpoint that finds nothing changed pushes nothing, so wiring a signal twice cannot
  invent an undo step. Undo re-installs through
  `Session.add_result`, so the existing `resultsChanged` refresh is the only repaint path.
  `checkpoint_all()` is reached from four places, and the last two are what make the promise above
  true rather than aspirational: an undo/redo, the Edit menu opening, **a project save and a quit**.
  Two things the diff alone cannot see are read off the RESULT instead: whether the `EventSet`
  OBJECT was replaced (an install) or edited in place (a curation click), and whether the new table
  carries any of the hand curation the old one held. Pressing Detect twice with the same settings
  produces an identical frame with the accept/reject column reset — indistinguishable from curation
  in the frame, unmistakable in the result — so it is named "Re-run detection" and announced.
  Optional non-schema columns (`added`) are reconciled inside the comparison: it is materialised
  only when a detection is DISPLAYED, so comparing column lists made every batch-detected table
  that was never on screen report a mutation on every checkpoint.
- **The hand-measurement table gets the same rule from a sibling class**,
  `core/history.MeasurementHistory`, not from a sixth key on `EventHistory`. Everything that made
  the event stack right carries over — a pristine deep copy that is compared, a derived label, a
  bounded stack — but the PAYLOAD does not: every step of `EventHistory` goes through
  `AnalysisResult`/`EventSet` (the frozen 20-column schema, `ensure_added_column`,
  `describe_change`'s column groups, `Session.add_result` as the install door), and a measurement row
  shares none of it. A reserved key would have meant branching on that key in five places inside the
  newest and most-repaired module in the app. The frame is read and written through two callables
  the wirer supplies, so the class stays free of Qt widgets and of the measurement schema. There is
  still exactly ONE Edit ▸ Undo: `main_window` keeps a small journal of which stack each step landed
  on and a press takes back the most recent step that can actually be applied.
- **A channel the user unticked is `Layer.excluded`, not a deleted layer.** `visible` answers "is
  this the recording on screen"; `excluded` answers "did the user say no". Conflating them forced the
  untick to be destructive — `show_only` re-shows every layer of the recording it switches to, so a
  merely-hidden channel came back — and that took the layer's colour and its `time_offset_s`
  alignment with it, both of which are persisted. `excluded` is an additive, optional manifest key:
  absent means False, which is exactly what an older project meant.
- **What the analysis windows may plot** is `core/sources.discover_plot_sources` — a PURE (Qt-free,
  Session-free) rule that turns the open documents plus a result lookup into a list of `PlotSource`
  value objects: one per `(document, channel)` with a stored detection, segments included, plus the
  manual-measurement table. Each carries its OWN detection provenance (so a per-source fit resolves
  its own `t_min`), its own Okabe-Ito colour and its own calibration fingerprint. `MainWindow` calls
  it; the windows never walk the Session themselves.

## The reader door — `io/contract.py`

Four readers decode four formats (`abf`, `hdf5`, `tdms`, `text`), and the recurring defect was never
in the decoding: it was **four different answers to one question**. `io/contract.py` is where the
questions and their single answers live. Nothing in it decodes anything — it is deliberately the
boring half of `io/`: the sentences, the clamp, and one logging bridge. All four readers import it.

The four questions, each with the defect that put it there:

1. **How many samples are really here?** A truncated TDMS reported whatever nptdms recovered with
   `issues=()`; ABF correctly refused. That number is the denominator of every event rate, capture
   rate and duration in the app, so a short read silently re-attributes the whole result to a
   recording the user does not have. `short_read_issue` is the one sentence for it.
2. **What is the sample rate, and how sure are we?** The text and HDF5 readers measure `dt` from the
   first ~1000 rows and then describe the whole file with it: on a CSV whose interval changes at row
   1500, 0.0600 s was reported for a 0.3300 s recording — 5.5× wrong, `issues=()`. Changing how the
   rate is derived would move every dwell time on every well-formed file, so the fix is not the rate:
   it is *saying that the rest of the file disagrees* (`rate_disagrees_issue`).
3. **Which channels exist, and why was one rejected?** HDF5 silently dropped a `/current` stored as
   int32 raw ADC counts, or shaped `(1, N)` — absent from the Files list with nothing anywhere saying
   why. A refusal the user cannot see is indistinguishable from a file that never had the channel
   (`rejected_channel_issue`).
4. **What encoding is this text in?** The probe opened with `errors="replace"` and `pd.read_csv` did
   not, so a Latin-1 CSV off Windows instrument software probed clean with a channel called `curr?nt`
   and then failed at load (`fallback_encoding_note`).

Plus the structural rule all four re-implemented, three of them differently — **the load window has
two halves**:

- `meta.loaded_range` is derived from what was **read**, never from what was **asked**. The text
  reader assigned the request verbatim, so `(500, 8000)` on a 2000-row CSV produced `n_samples=1500`
  beside `loaded_range=(500, 8000)` — a value that then went into the project manifest and was
  replayed on restore. **`window_loaded` is the only writer of that field**, and one level below the
  readers `core.types.RecordingMeta.__post_init__` refuses any meta whose window and sample count
  disagree, so a fifth reader cannot get it wrong quietly. `short_window_issue` is the separate
  sentence for "the request outran the file" — the file is not damaged, and saying so matters on a
  project restore whose recording has been re-exported shorter.
- `meta.t0_s` is the half nobody unified: it is the experimental time of the **first loaded sample**,
  because that is what `Session.x_axis` builds (`arange(n)/fs + meta.t0_s`), what the Event Inspector
  inverts, and what `core.segment` rebases against. HDF5 read it at the window's own index and the
  other three left it at the file's origin, so the same window of the same experiment carried two
  different clocks depending on the format (measured on the owner's ABF: a window starting 20 s in
  reported the file's own 61.303 s). `window_t0` is the one answer, and `start == 0` returns the
  file's `t0` unchanged so a whole-file load is bit-identical to what it always was.

`ReaderReport` is the accumulator both `probe()` and `load()` write into. It keeps **issues**
(non-fatal *problems* — the recording is not what it looks like) apart from **notes** (honest
*advisories*), deduplicates (probe and load share one scan, and a rebuilt `RecordingMeta` must not
say the same sentence twice), and both halves are rendered verbatim by `gui/file_info_panel.py` —
the **Inspector ▸ File Info** tab, in its own **Notes** and **Issues** boxes, NOT the Files dock,
which lists recordings and channels only. That rendering is the entire point, since a problem that
reaches only Python logging has not been reported to anybody.
That is also what `decoder_messages` exists for: nptdms recovers what it can from a truncated file
and says so *to the `nptdms` logger*, i.e. to a terminal nobody has open. It attaches a
thread-scoped handler rather than reconfiguring logging, lowers a logger's level only when the
effective level would have swallowed the warning outright and only downward, and puts it back.

## The analysis seam

`core/analysis.py` declares the extension points that let analysis modules plug into the GUI without
either side importing the other:

- `AnalysisModule` / `AnalysisContext` — a channel-in, result-out protocol. One `register_all()` at
  app start wires up both detectors: `analysis/modules.DetectSchmitt` and the runtime-loaded Keras
  `analysis/ml_modules.MLDetect`. Registration is idempotent (a second test session shares the
  process-global registry), and importing `ml_modules` only *probes* for TensorFlow rather than
  importing it, so start-up stays cheap.
- `AnalysisResult` carries a payload (e.g. an `EventSet`) plus a `calibration_fingerprint` for
  staleness detection.
- `OverlayHint` (int64 sample-index arrays) + an overlay-renderer registry (`gui/overlays.py`) turn
  analysis output into trace graphics. Detected events render as **one batched span item**, not N
  region items.
- `ResultStore` on the `Session` holds results; `resultsChanged` drives the UI.

Detection runs off-thread through `gui/analysis_worker.AnalysisController`; results are installed on
the main thread.

### Background work: one seam, four rules

Every background job in the app — a load, a probe, a detection, an ML run, a surrogate study, an
open-pore probe, a re-measure, a batch pass — is started and retired through
`gui/worker_thread.py`. That module is where the thread-lifetime rules live, once, instead of three
times over in the three controllers:

1. **A worker thread is never terminated**, and no wait is ever unbounded. A thread that will not
   stop is *detached*: severed from its controller, parked, and left to wind down with nothing
   listening. `retire_worker_thread` returns whether it actually stopped, and the caller must not
   claim the work was cancelled when it was not.
2. **Cancellation is an ambient token** (`analysis/cancel.py`), installed by the worker and polled
   by every kernel underneath — no call site forwards it, so no call site can forget it. Anything
   long-running added later must poll.
3. **One analysis at a time, app-wide** — THE RUN GATE. `AnalysisController` and
   `BatchDetectController` write into the same `ResultStore`, so two of them in flight is two
   answers for one key with the last writer winning. Every job registers a `domain` and a `label`
   as a side effect of `start_worker_thread` (both are *required* keyword arguments, so a
   controller written later cannot fall outside the gate by omission); `blocking_job` decides, and
   `refusal_text` writes the sentence. A refused start emits `refused(str)`, returns `False`, and
   **stops nothing**. Two jobs of the same domain with the same owner are allowed — a controller's
   own `start()` aborts its predecessor first, which drops its answer.
   Loads are a separate domain and are deliberately not exclusive with analyses.
4. **Quitting never aborts the process.** `MainWindow._teardown` aborts every controller, syncs
   `QSettings`, and *uses* `drain_detached`'s return value; `gui/app.main` gives anything still
   running one more bounded chance and then leaves via
   `worker_thread.exit_without_destroying_live_threads`, because a C++ `QThread` destroyed while
   its OS thread runs calls `qFatal` (measured: SIGABRT, 4/4 runs, plus a macOS crash report).

The gate is deliberately NOT expressed as a disabled Detect button: Qt never delivers a click to a
disabled widget, so a control greyed out by a batch could not tell the user it was the batch. The
click lands and the refusal names the run that is in the way.

## Rendering

- pyqtgraph with peak-preserving **min/max decimation** (`setDownsampling(method="peak")`) so spikes
  survive zoom-out. There is no custom decimation pyramid — measured unnecessary at the sample
  ceiling.
- Annotation items added to a pane (event spans, the measure region, markers) use
  `ignoreBounds=True` so they never drive the y-autorange (a real bug fixed once: view-tracking items
  otherwise ratchet the axis wide).
- The trace's x-range is owned explicitly by `TraceView` (load/reset set the full data extent);
  autoRange is not relied on because it interacts badly with `clipToView`.

## Pipeline

```
open_document (io) ──▶ Session.add_document
                          │
   Calibration ◀── calibrated(doc, ch)  (raw × gain + offset, cached, float32)
                          │
   Filters (analysis.modules.preprocess_for_detection: notch → high-pass → baseline-detrend → savgol → running-mean/median)
                          │
   Detection (analysis.detect.detect_events: Schmitt two-threshold; low-pass applied here)
                          │
   Characterization (analysis.characterize → 20-column EventSet, median depth, ECD, rise/fall, n_levels)
                          │
   ├─ Event table + span overlays + summary (gui.analysis_panel)
   ├─ Event inspector (gui.event_inspector — per-event waveform + curation)
   ├─ Manual measurement (analysis.measure — cursor-selected stats, own schema)
   └─ Plots + ML (analysis.populations + analysis.stats; gui.plots_panel — hist/scatter/PCA/clustering/fits,
        MULTI-SOURCE: core.sources.discover_plot_sources offers every open recording/segment with a
        detection, keyed by segment.persistent_key, so two pores overlay with shared bins, a legend
        and ONE FIT PER SOURCE resolved from that source's own provenance)
```

## Analysis core modules

| Module | Responsibility |
|---|---|
| `analysis/filters.py` | Zero-phase low-pass (Bessel), high-pass, notch, Savitzky-Golay, running-mean, median; MAD noise; rise time; PSD. |
| `analysis/baseline.py` | `BaselineConfig`; the two-pass masked-median open-pore estimator (kernel imported from `ml_preprocess`), the block-median/percentile/ASLS + PCHIP detrenders, `measured_polarity`, `baseline_health`, `remove_baseline_wander`, and `characterisation_baseline` — THE door. |
| `analysis/baseline_probe.py` | Measures a recording's own timescales and **proposes** `rough_s`/`base_s` from them. Reports; never applies. |
| `analysis/detect.py` | Schmitt two-threshold detector; measured polarity; merged-exclusion analyzed-duration. |
| `analysis/merge.py` | Post-detection sub-event merge (window-based and selection-based), re-characterizing each merged span through `characterize`; the occupancy floor that refuses a merge whose span would be mostly open pore. Reuses `detect.group_starts_by_gap`, so the gap rule is not re-derived. |
| `analysis/characterize.py` | Per-event dwell, **median** depth, ECD, SNR, rise/fall, sub-state count → the events DataFrame. |
| `analysis/events.py` | The frozen 20-column event schema + `EventSet` (accept filtering, pooling that refuses cross-pore). |
| `analysis/stats.py` | Censored-MLE exponential dwell fit; hand-rolled Gaussian-mixture amplitude fit; log-bin histograms with the Jacobian. |
| `analysis/populations.py` | GMM/k-means clustering (BIC k-selection), PCA, and linear/power-law fits — all scikit-learn/NumPy. |
| `analysis/measure.py` | Manual-selection statistics (dwell, amplitude by method, skewness, kurtosis, …) + a 28-column measurement schema (`MEASUREMENT_SCHEMA`), separate from the frozen 20-column event schema. |
| `analysis/truth.py` | Synthetic trace + AAFT surrogate generators used to validate detection (recall/precision) and for null controls. |

## The baseline seam — one config, one door, one fingerprint

Every blockade depth in this app is a distance from a baseline, so "which baseline?" is not a detail.
The app used to answer it four times: a 10 ms `block_median` inside `detect.py`, a 200 ms window in
`ml_modules.py` (widened to fix a real zoom-dependence defect, but only there), a ±3 ms flank median
in `characterize.recharacterize_span`, and the 10 ms default again in `truth.py`. One event on the
owner's −600 mV apo recording therefore read **32.04 pA** (Schmitt), **37.54 pA** (rebuild) and
**67.69 pA** (ML), and the population median depth was 23.23 pA *or* 111.31 pA depending on which
detector found it.

- **`BaselineConfig` is the single owner of the estimator's shape** — method, `rough_s`, `base_s`,
  `decim`, `mask_sigma`, `dilate_ms`, `window_s`, `despike_size`. One object flows from the GUI
  through detection, characterisation, ML characterisation and re-characterisation. Deliberately NOT
  on it: `percentile` and the four `asls_*` knobs, which tune a *detrender*, not the I0-tracking
  shape that has to be identical across the four call sites.
- **`baseline.characterisation_baseline` is THE door**, and `baseline.DEFAULT_BASELINE` is the only
  baseline-parameter literal in the package. Parameters alone would have fixed four sites, not the
  *class*: `estimate_baseline` is public and takes `window_s=` with a silent default, so the fifth
  site to hit a baseline problem would fix it locally exactly as `ml_modules` did — for a good
  reason, with a measurement behind it — and the app would report two depths again. The AST guards in
  `tests/analysis/test_baseline_seam.py` fail the build if another module calls `estimate_baseline`,
  defines a baseline-window constant, or hands the door a literal window.
- **The kernel is imported, never re-implemented.** `baseline.py` pulls `baseline_masked_median`,
  `baseline_robust_median`, `_decimated_median` and `despike_median` out of the frozen
  `analysis/ml_preprocess.py`, which is bit-exact against the owner's persisted training corpus and
  already hardened against the non-deterministic 1-D `scipy.ndimage.median_filter`. A second copy of
  a twenty-line function with five silent bit-exactness traps was judged the single largest risk in
  that change; `tests/analysis/test_masked_baseline.py` asserts there is exactly one definition of
  `_decimated_median` in the tree.
- **`BaselineConfig.fingerprint()` follows the `calibration_fingerprint` precedent exactly.** Rows
  carry the stamp (in `provenance`, via `to_provenance` — never a new column, the 20-column schema is
  frozen); a stored result whose stamp is absent or differs from the live config is marked STALE and
  the user is offered a re-run. Nothing is ever silently re-measured, because `depth_pA` moving 15×
  between two openings of the same project is indistinguishable from data corruption. `fs` is folded
  in by resolving `decim` first, so two recordings at different rates cannot share a fingerprint
  while running different strides. `from_provenance` always answers (callers need a config);
  `reproducible_from` **refuses** — returning `None` for a pre-change row or one whose detector was
  handed a baseline array — because a rebuild that guesses is worse than one that falls back and says
  so.

- **A recording too short for the configured window is FITTED, never substituted.**
  `adapt_to_length(config, n, fs)` is a three-rung ladder, and each rung is named in `provenance` as
  `baseline_fallback`: (1) the configured estimator fits and nothing is touched — **the only rung a
  normal-length recording can reach, which is why no shipped number moved**; (2) `FALLBACK_CLAMPED`,
  the *same* estimator with `rough_s` clamped to the whole recording and `base_s` scaled to preserve
  the user's own `base_s/rough_s` ratio, `decim` re-derived; (3) `FALLBACK_TOO_SHORT`, a global
  median under its true name, `method='constant'`. What this replaced was a substitution: the old
  `_length_fallback` stood the pre-change 10 ms block median up in place of the configured
  estimator, i.e. the fallback that existed to avoid a degenerate answer *was* the degenerate
  answer. Measured on a synthetic trace with known truth (baseline −35 pA, σ 4.5 pA, one 20 ms
  −88 pA blockade), depth reported as a fraction of the true 88 pA at **n = 3072** — the owner's own
  ML window: **0.0094 before, 1.02 after**; at n ≥ 5000 both agree to four decimals. Below ~2000
  samples the clamp collapses too, for the documented reason (`rough_s` must be ≥ ~3–4× the longest
  state it is meant to expose, and a 20 ms event fills a 30 ms window) — that regime is **not
  claimed to be sound**, it is warned about, and it is not retuned, because tuning
  `MASK_FRACTION_CEILING` against one recording is the auto-tune this module refuses.
  **Clamping is not the auto-tune R5 forbids:** there is no candidate set, no metric is consulted,
  and the ratio that decides the two-pass structure is the user's own. `BaselineResult.fallback_token`
  carries which rung was taken; `provenance()` used to hardcode `FALLBACK_TOO_SHORT` whatever
  happened, so no consumer could tell a clamped measurement from an untracked one.
  `estimate_baseline_result` still **refuses** a too-short window — it is the estimator's strict
  contract and the corpus runs through it — but names the config that would work and points at the
  door. `'constant'` is exempt from the length check in both: a global median reads no window, and
  `estimate_baseline` answered a 200-sample `'constant'` call while `estimate_baseline_result`
  refused it.

### `analysis/baseline_probe.py` — propose, never decide

There is no defensible global default. The apo `_0006_` file has a **521 ms partial-clog state at
t = 37.13 s**, and a 500 ms rough window sitting inside a 521 ms state *is* that state: nothing gets
masked and the baseline lands 22 pA off (−56.6 against a true −34.5). It needs `rough_s = 2.0`.
`ferritin_data3` is a gating pore stepping 296 pA with a ~35 ms level dwell, where a 2 s window
cannot follow the steps at all and the reference notebook uses `rough_s = 0.03` — 67× apart across
four real recordings. So the app **measures each recording and proposes**: `probe_timescales` returns
a `BaselineProbe` (a robust one-sided sigma, a dwell distribution, a gating verdict), and
`propose_baseline` turns it into a `BaselineProposal` — a config, a `param_source` of `'proposed'` or
`'default'`, and a `reason` written to be shown verbatim beside the numbers. It reproduces the known
good parameters on three of the four recordings from the data alone.

Nothing here is ever auto-applied, for two measured reasons rather than stylistic ones. First, the
substitute sigma estimator is defensible but over-reads 3–4× on a gating pore (the obvious
delta-based sigma reads 7× *low* on these already-low-passed exports, which is why it is not used).
Second, **the reference's own QC triple is a degenerate objective**: `qc_open_pore_level` / `drift` /
`mad_noise` are monotone in window length, so optimising them selects the shortest possible
window — a degenerate `0.01/0.002/1` set beats the shipped parameters on all three for every corpus
file. Auto-tuning against them would reproduce the exact failure the seam removes. The defence is
therefore architectural: a human confirms.

The probe also may not construct a `BaselineConfig`. Every config it returns is
`DEFAULT_BASELINE.replace(...)` touching only `rough_s` / `base_s` / `decim`, which makes it
structurally impossible for a probe to invent a `mask_sigma`, a `dilate_ms` or a `despike_size`, and
keeps `DEFAULT_BASELINE` the one literal the seam guard protects. An undecidable probe returns
`DEFAULT_BASELINE` *by identity*. `decim` is always **derived**, never proposed. The probe runs off
the GUI thread through the ordinary run gate (`label="baseline_probe"`) — measured 106 ms at 1.2 M
samples and 579 ms at 8.6 M — and `proposal_summary` builds the dialog's rows, so the sentence the
user reads and the numbers the estimator runs come from one function over one object.

## Load-bearing invariants

Preserved deliberately; changing them silently corrupts the science:

- **Detected (not assumed) polarity**; work in rectified space, never rectify stored data.
- **The mask direction is a property of the recording, not of the parameters.**
  `baseline.measured_polarity` votes across a fixed ladder of detrend scales, drops any scale that
  measured nothing (`derive_polarity` returns `+1` when it has no evidence, and that default was
  being reported as a measurement), and ignores the `BaselineConfig` it is handed. It used to
  detrend at the configured `rough_s` with a stride derived from `base_s`: on `ferritin_data3`,
  stride 264 answered −1 and stride 333 answered +1, and the one-sided mask pointed backwards moves
  the baseline by 312 pA.
- **No schema column ever holds ±Inf.** `EventSet` demotes an overflowed magnitude to `NaN`, this
  app's "not measured" sentinel; the row keeps its `non_finite` flag and its rejection. `±Inf` is
  the only non-value that survives `dropna()`.
- **A control checks its own premise.** `truth.null_control` verifies that its AAFT surrogate is a
  null of the recording it came from, and reports no rate rather than a wrong one. Phase
  randomisation is a null for a stationary spectrum plus Gaussian noise; a drifting real pore is
  not, and the control was returning `estimated_fdr = 1.00` on every real recording measured.
- **MAD** noise, never `std` (std absorbs the events being detected).
- **Median** blockade depth, not max (max is biased by dwell → spurious dwell–depth correlation).
- **Two-threshold** Schmitt with `k_out < k_in` (single threshold shatters events).
- **PCHIP, never a cubic spline**, whenever block knots are expanded back to full rate
  (`baseline._expand`) — Pchip is shape-preserving and cannot overshoot; a natural cubic spline
  rings at a drift step and the overshoot lobe is a phantom deviation detection reports as an event.
  This is the *detrending* family (`block_median` / `percentile` / `asls`) and the stored-result
  compatibility path. The estimator every depth is now measured against is the two-pass masked
  median — see *The baseline seam* — which interpolates nothing.
- **Censored-MLE** dwell fit (a naive `expon.fit` is biased long by `t_min`).
- **Zero-phase** filtering everywhere (event edges must not shift).
- Signals stored **float32 contiguous**; time axis **float64**.
- **A recording has at least one sample.** `n_samples == 0` is refused at `TraceDocument`
  construction — the one door the four readers and every segment pass through — and again in
  `io.base.probe_path`, which builds no document. HDF5 and TDMS both used to accept one with
  `issues=()`, and selecting it raised out of the calibration panel halfway through
  `_on_selection`, leaving the analysis target on the previous recording while the title bar named
  the empty one.
- **A selection retargets the whole window, or the user is told which panel did not follow.**
  `MainWindow._on_selection` runs its seventeen consumers through one runner; an exception in any of
  them is recorded, printed, and does not abandon the rest. PySide6 already swallows an exception
  that escapes a slot, so the real choice was never "raise or swallow" — it was "half a retarget
  and silence" or "a whole retarget and a sentence".
- **A calibration gain is never zero.** Refused at `ChannelCalibration.__post_init__`, the one
  door a sidecar, a project, the Gain box and any future loader must all pass through — after the
  float32 cast `apply()` will make, so a gain that underflows to zero is caught too. It is a
  different rule from the non-finite one: `gain=0` makes every sample equal to the offset, a
  perfectly *finite* constant that no read-out downstream can tell from a recording (it printed
  "Total RMS noise: 0 pA" with `meta.issues = ()` on the owner's real ABF). A NEGATIVE gain stays
  legal — headstages invert, and a sign flip loses nothing.
- **What the calibration did to a channel travels with the channel.** `meta.issues` describes the
  recording *as loaded*; every analysis reads the *calibrated* array, and nothing re-checked it, so
  a gain that overflowed float32 left 2,420 non-finite samples with `meta.issues = ()` while the
  PSD panel told the user to go read the Issues box. `Session.calibrated` now fills
  `CalibratedChannel.issues` (memoised per `(doc, ch, gain, offset)`, the same key as the cached
  array), and the File Info Issues box shows both halves. Empty means "the calibration added
  nothing", never "not checked".

## The project file — `core/project.py`

`core/project.py` is the **pure (de)serializer** for a `.nanoscope` file: no Qt, no `Session`. It
moves a plain `ProjectState` — documents-as-references, segments, calibrations, layers, view mode,
per-result event tables + provenance + params, manual measurements — to and from one self-contained
file, so the GUI restore code in `main_window` stays a thin translator between `Session` and a
dataclass. That split is what makes the format testable headless, and `tests/core/test_project.py`,
`test_project_segments.py` and `tests/gui/test_adversarial_project_io.py` all lean on it.

**The file is a ZIP** holding `manifest.json` (everything except the bulk tables),
`results/<i>.parquet` (one per `AnalysisResult`, the FULL curated frame — `accepted`, edited
`start_idx`/`end_idx`, recomputed columns, non-schema columns like `added` — stored losslessly with
pandas dtypes via pyarrow) and `measurements.parquet` when the hand-measurement table is non-empty.

**The 180 MB of raw signal is never stored.** Documents are persisted as REFERENCES — absolute path,
the loaded `SampleRange`, and the `identity.SourceFingerprint` — and reopened from disk on load. What
a user cannot cheaply recreate (calibration, layer view state, curated detections, manual
measurements) is what we keep. A **segment** is likewise a derived recording, stored as
`(segment_uid, parent_key, i0, i1, name, t0_abs_s)` and re-derived from its parent on load rather
than saved as signal.

Three decisions in it are load-bearing:

- **Records are keyed by the PERSISTENT key, never by `doc_id`.** The `doc_id` field inside
  `calibrations` / `layers` / `results` holds `core.segment.persistent_key(doc)` — a file's resolved
  source path, or a segment's `segment_uid`. A live `doc_id` now carries a per-load token (see
  *Identity*, above), so persisting it would make **every** project round-trip fail to re-attach: the
  reopened load's token is by construction a different one. The values `persistent_key` returns are
  exactly what they were before that token existed, so projects written by either build read
  identically.
- **The version is a RANGE, and every extension is additive.** This build writes `VERSION = 11` and
  reads back to `MIN_READ_VERSION = 1`; older formats are strict subsets. v2 added `segments`, v3
  each result's `params` and `calibration_fingerprint`, v4 each document's `source` fingerprint, v5
  each calibration's `note` — the one field a project could not carry, which is why committing a
  calibration after a restore used to blank the user's sidecar. v6 added the `fits` list: the
  DERIVED results (`fit.dwell` / `fit.depth` / `fit.levels`), which until then were dropped on save
  without a word. v7 added `channel_name` / `channel_kind` to all four per-channel records
  (`CalibrationRec`, `LayerRec`, `ResultRec`, `FitRec`), because an index is a *position* and not an
  *identity*: re-export a recording with its signal columns in the other order — same sample count,
  same channel count, byte-identical size — and a `pA` calibration lands on the VOLTAGE channel,
  a detection is reattached to the voltage trace, and the result's stored fingerprint MATCHES the
  misplaced calibration, so the staleness system positively certifies those events as current-valid.
  v8 added `CalibrationRec.kind`, the user's *"treat this channel as"* override — NOT `channel_kind`,
  which records what the file header declared so a channel can be found again; this one is the
  correction made when the header is wrong or silent, and it drives both the pane order and the
  overlay grouping. v9 added `colours`, the project's identity-keyed colour map
  (`core.layer.ColourRegistry`): which slot each recording+channel holds in the categorical cycle,
  and the ink the user assigned it by hand. Before it, a recording's colour was persisted for
  exactly ONE of the four surfaces that draw it — `layers[].color` carried the trace view's ink,
  while the Plots panel recoloured from *position* on every refresh — so a hand-picked colour did
  not survive a reopen and "the Zn run is the green one" was true in one window and false in the
  next. Keyed by the persistent key like `calibrations` / `layers` / `results`, so it re-attaches to
  a segment re-derived on load. v10 added each result's `module_version` and top-level
  `baseline_fingerprint`. This preserves the exact producer version for derived/plugin results and
  the baseline lineage needed by Reconciled snapshots instead of silently substituting the current
  registry version or dropping provenance on round-trip. v11 added the versioned `workspace`
  envelope: ordered, stable-id analytical artifacts with small JSON payloads, Parquet-backed tables,
  and opaque byte members. It carries population-plot recipes and computed PCA/clustering/report/QC
  snapshots without serializing Qt or executable Python objects; unknown producer kinds and their
  extension fields/resources are deliberately retained for a later build or plugin. Each missing key
  has a defined reading:
  absent `params` is `{}`, an absent
  calibration fingerprint falls back to the restored calibration, an absent `source` reads as
  *unknown* (which raises no warning rather than a false one), an absent `note` means "this file has
  no opinion", leaving the sidecar's note alone, an absent `fits` is `[]`, an absent
  `channel_name` / `channel_kind` is `None`, which `calibration.resolve_channel` treats as "resolve
  by index, exactly as before, silently", and an absent `kind` means "no override, the header
  stands", and an absent `colours` is an EMPTY registry, meaning "nobody has assigned anything
  yet" — the first discovery then hands out slots in the order it always did, which is visually what
  that project last showed, because a project without a registry was drawing positional colours
  anyway. An absent `module_version` falls back to the currently registered module version for
  legacy projects, and an absent top-level baseline fingerprint falls back to provenance when
  available. An absent `workspace` in a v1–v10 file is an empty analytical workspace; a v11 file
  must carry a valid workspace envelope. A version above
  `VERSION` is refused with *"needs a newer NanoScope"* rather than half-read.
- **Additive does not mean an older build may write the file.** Additive-ness governs whether THIS
  build can read OLDER projects; it says nothing about whether older builds can safely read this one,
  and they cannot: a build that predates `fits` accepts the file, ignores the key and the
  `fits/<i>.parquet` members, and DROPS THEM ON ITS NEXT SAVE — silently, because it never knew they
  were there. That is why every one of v6 / v7 / v8 / v9 / v10 / v11 bumps `VERSION` even though each is
  purely additive: the bump converts a silent strip into `ProjectVersionError`'s clean refusal, and costs
  nothing in the other direction, since `MIN_READ_VERSION` is still 1.
- **`load_project` promises exactly ONE exception type.** It kept that promise for the failures
  somebody had thought of — missing manifest, bad JSON, absent parquet member — and broke it for
  every other malformation of the same file: a `layers` entry without `color` raised a bare
  `KeyError('color')`, a `loaded_range` written as a string raised `TypeError`, a non-numeric
  `channel` raised `ValueError`. Six shapes, each escaping as whatever the decoding expression
  happened to throw. `_manifest_section` wraps each section — not each field, because the failure is
  not a field: it is *"this manifest is not shaped like a manifest"*, and the useful thing to say is
  which part of it and what Python found there. The GUI's blanket `except Exception` is precisely why
  it stayed wrong for so long; the contract is for the API's other callers and for the next reader.

Two smaller rules keep the round-trip honest. Provenance dicts are coerced JSON-safe (`_json_safe`:
numpy scalars and arrays down to builtins, an unrecognised leaf to `str`, so a stray object can never
abort a save). And the per-result **overlay hints are not stored** — their sample-index arrays are
rebuilt from the event frame on restore, so a numpy array never reaches the JSON in the first place.

**The save is atomic**, via `atomic_write.staged_write` (see the next section; `_staged_write` here is
a one-line pass-through kept for this module's callers). This is not defensive decoration:
`zipfile.ZipFile(path, "w")` truncates the target the instant it opens, so the pre-atomic version
destroyed a good project *before* it knew whether it could write a new one — an injected ENOSPC on
the second member turned a 6949-byte project into a 3251-byte corpse that then failed to load with
"missing its manifest.json". A save that fails must cost the user nothing. `save_project` also takes
an optional `notes` list, which collects sentences about what the write did to the **file** — that a
symlink was followed to its destination, or that other hard-linked names now hold the previous
contents.

## The write seam — `core/atomic_write`

PORT_PLAN rule 21: **no write may destroy what it is replacing until the replacement exists.** The
rule was stated for "every file the app writes over user data" and implemented in
`core/project._staged_write` alone; `calibration.save_sidecar`, `scoring`'s two exports,
`measure.write_measurements_csv` and `plot.figures.save_figure` all wrote in place. It now lives in
one module and an AST guard (`tests/core/test_persist_invariant.py`) fails the build if any module
under `nanoscope/` writes a path it did not stage.

Three guarantees, three different failures:

1. **A write never half-lands.** Staged in a temporary sibling, moved with one `os.replace`.
2. **A write never changes WHICH FILE it wrote.** `os.replace` installs a new inode, so a naive
   stage-and-replace turned a **symlink** into a regular file (the canonical project silently
   keeping the previous save), destroyed **extended attributes** on every save, could not stage a
   name of 239 characters at all, and overwrote a `chmod 400` project without a word.
   `resolve_write_target`, `dress_like`, `temp_sibling` and `refuse_if_write_protected` close
   those. A **hard link** is the one thing an atomic replace cannot preserve — writing in place is
   the only alternative and it is the destructive path guarantee 1 forbids — so the trade stands
   and is *reported* through `notes`.
3. **A SET of files never half-lands.** One figure export writes twelve files under one `figures_*`
   stem; nine failing left three from today beside nine survivors of a previous export with no
   marker. `StagedSet` stages every member first, commits together, and restores what it replaced
   if a commit fails part way.

The same rule at the value level is why `CalibrationRec` carries every field
`ChannelCalibration` has (checked by a test): a field the record lacks is not "not persisted", it
is *blanked* by the next ordinary Apply. `note` is three-state — a string, `""`, or absent, where
absent means "this project file has no opinion" and leaves the sidecar's note alone.

And the same rule about *what to ask before*: `sidecar_effect` answers a question about the FILE,
so the confirmation built on it only fires when the whole file goes. Reset one channel of a
recording whose other channel is still calibrated and the file survives (`SIDECAR_WRITTEN`) while
the note goes out of it — measured: no prompt, status line *"Applied · saved to …"*.
`calibration.notes_lost_by` (→ `Session.sidecar_notes_lost_by`) is the companion question — *what
does this write destroy* — derived from the same `_sidecar_payload` the write is derived from, so
the prediction cannot drift from the act. The panel asks one of the two, never both.

## The save-dialog seam — `gui/export_paths`

`core/atomic_write` guarantees a write does not half-land. This module answers the question that
comes *before* it: **which files is the user actually about to lose?**

`QFileDialog.getSaveFileName` confirms an overwrite of **the one name the user typed**, and that
confirmation is only truthful for an export that writes exactly that file. Several of ours do not.
*Analysis ▸ Export figure…* proposed `figures.pdf` — a file it **never writes** — and then wrote
`{stem}_{dwell,depth,scatter,trace}.{pdf,png,csv}`: up to **twelve** files, none of them mentioned,
silently overwriting whatever was there, with the `.csv` source-data files the ones a user is most
likely to have hand-edited. *Export events for training…* proposed `events_training.csv` and also
wrote `events_training.meta.json` beside it, unmentioned.

Patching those two commands would have left the next export free to reintroduce the defect, so the
guarantee moved to where every export must pass: **the path a GUI export writes to is obtained from
`ask_export_path`**, which is handed an `outputs` callable that enumerates the REAL output set
(`figures.figure_output_paths`, `scoring.training_export_paths`, …) and confirms whatever part of
that set already exists. `outputs=None` is the honest declaration "this export writes exactly the
chosen path and nothing else". Seven GUI modules and ten call sites go through it. An AST guard in
`tests/gui/test_export_overwrite.py` fails the build if any module under `nanoscope/gui` calls
`getSaveFileName` outside this file — that guard, not the docstring, is what keeps the seam.

Three details are deliberate:

- **The native overwrite prompt is suppressed** (`DontConfirmOverwrite`). Leaving it on would ask
  about one file and then this module would ask about the real set: two prompts, the first of them
  misleading. One prompt, naming what is actually at stake.
- **`confirm_overwrite` names every file**, because "3 files will be replaced" is not something a
  user can check. It is a module-level function rather than a method so a test can substitute it
  without a live modal.
- **Buttons are named, never indexed.** `QMessageBox.buttons()` returns *layout* order, not insertion
  order, and on this platform layout order put the destructive button last — so `buttons()[-1]` made
  **Replace** the Enter-key default on a prompt whose entire job is to stop an accidental overwrite,
  and left no escape button at all. Return and Esc must both keep the user's files.

`ask_export_path` returns the chosen path **verbatim**; normalising it is the caller's job and must
match what its own `outputs` enumerator assumed, or the set that was confirmed is not the set that
gets written.

## Environment

The Anaconda distribution ships a `qt.conf` that points Qt6 at conda's Qt5 plugins, breaking Qt
plugin discovery. `nanoscope/bootstrap.py` sets the plugin path to PySide6's own plugins before any
Qt import, so the app launches without manual environment variables. It is imported first by
`nanoscope/__init__.py` and by the test bootstrap.

## Testing

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Runs headless. GUI tests use `pytest-qt`. A `@pytest.mark.corpus` subset drives real recordings and
skips when they are absent. Detection is gated in tests to recover synthetic ground truth at
recall/precision > 0.95.

Two hazards are the environment's, not the code's, and both are handled once in `tests/conftest.py`
rather than per test. Both were live defects in this suite.

**1. A modal reached under `QT_QPA_PLATFORM=offscreen` hangs the whole run, silently.** There is no
event source that can dismiss a `QDialog.exec()`, so the process sits in `poll()` at 0 % CPU with no
output — indistinguishable, from the outside, from a slow test. Five modals are defaulted for every
test by the autouse `_no_close_guard_modal` fixture. Three are **confirmations the user's own click
asks for**: `MainWindow._confirm_close_choice` → *Discard* (pytest-qt closes every registered widget
at teardown, which fires `closeEvent`), `_confirm_close_discarding_events` → True, and
`MeasurePanel._confirm_remove` → True. Four are those confirmations plus one of a different and more
dangerous shape — a **modal a RESULT opens**: `AnalysisPanel._show_sweep`, where a finished
`sweep_threshold` raises `SweepDialog` on a queued signal. A test never has to click anything to meet
that one; it only has to let the job finish, so whether it hangs is a race against machine speed —
won on a loaded machine, lost on an idle one.
`test_run_gate.py::test_the_surrogate_chain_runs_off_the_gui_thread` starts a real sweep to prove
conditioning runs off the GUI thread, waits only until conditioning *begins*, then pumps 20 more
times before cancelling; it hung deterministically in isolation until this fixture covered it.
`_show_comparison` (`EstimatorCompareDialog`) is deliberately **not** in the fixture: it is reached
only after a test has itself called `compare_baseline_estimators()`, and the migration test patches
`EstimatorCompareDialog.exec` to return *Accepted* so it can assert the re-measure that branch
launches — neutralising the method would delete that branch and the test with it. A per-test patch of
the *dialog* is right when the dialog's answer is what the test is about; the fixture is for modals
nobody meant to open. **Any new result-opened modal must get the fixture's treatment**, and note its
one limitation: it patches only what is already in `sys.modules` when it runs, so a test that is the
first to import `nanoscope.gui.main_window` is not covered.

**2. The suite must not read the developer's own saved window layout.** `MainWindow._save_settings`
writes `geometry` and `windowState` into `QSettings` on every close, so any real launch of the app, a
screenshot harness, or a `--self-test` run leaves a layout behind that the next `MainWindow()` in a
test restores instead of its defaults. The session `qapp` fixture therefore points the whole run at a
private namespace (`NanoScopeTest` / `NanoScopeSuite`) and clears it. Measured: three geometry- and
focus-sensitive tests failed on this machine and passed the instant `geometry` and `windowState` were
deleted from the real store — `test_analysis_launcher.py::test_a_real_window_drives_the_ladder_and_never_clips_below_the_minimum`,
`test_measure_tab.py::test_dragging_the_inspector_narrow_still_sticks_with_the_measure_tab_raised`
and `test_p3_regressions.py::test_the_documented_curation_keys_are_reachable_on_the_table`.
`test_window_layout.py` and `test_shown_recording.py` already carried their own namespaces for this
reason; the rule is now the suite's, not two modules'.

To diagnose a hang: `sample <pid>` and look for `Sbk_QDialogFunc_exec` / `QDialog::exec()` in the
main-thread call graph. Re-running with `-v` and reading the last line of the log names the test.
