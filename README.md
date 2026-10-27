# NanoScope

A desktop application for **solid-state nanopore / resistive-pulse signal analysis** — load a
recording, view and calibrate the trace, detect and characterize translocation events, reconcile
several detectors, label events by hand, and explore the population with histograms, scatter plots,
clustering, and fits.

Built on **PySide6 + pyqtgraph** for a fast, interactive trace viewer, over a pure
NumPy/SciPy/scikit-learn analysis core.

---

## Install

Requires **Python ≥ 3.11** (developed against the Anaconda distribution).

```bash
cd nanoscope
python -m pip install -e .
```

This installs the dependencies (PySide6, pyqtgraph, numpy, scipy, pandas, scikit-learn, h5py,
npTDMS, neo, matplotlib) and a console entry point.

## Run

```bash
nanoscope          # the console script
# or
python -m nanoscope
```

Then **File ▸ Open…** a recording. On a fresh machine the app fixes a Qt plugin-path conflict
automatically (see `nanoscope/bootstrap.py`); no manual `QT_*` environment variables are needed.

---

## What it does

| Stage | Capability |
|---|---|
| **Load** | ABF (Axon), HDF5, TDMS (LabVIEW), CSV/TXT — a unified reader that reports units and sample rate honestly |
| **View** | Multi-channel stacked/overlaid panes, an adaptive overview/local-context Navigator, overlapping event-scale Scan controls, box-zoom, and a value crosshair |
| **Calibrate** | Per-channel gain/offset with a live raw-vs-calibrated readout and a command-voltage check |
| **Filter** | Low-pass, high-pass, notch, Savitzky-Golay, running-mean, median, and baseline-drift removal; an orange preview shows the candidate-localization chain |
| **Detect** | Schmitt and CUSUM classical detection with automatic polarity, plus model-based detection |
| **Reconcile** | Converge overlapping Schmitt, CUSUM, ML, and hand-added candidates into a separate, provenance-carrying table without changing the source detections |
| **Characterize** | Per-event dwell, blockade depth, ECD, SNR, rise/fall, sub-state count |
| **Fit levels** | CUSUM sub-state fitting — per-level amplitude and dwell inside multi-level events |
| **Measure** | Cursor-based manual event measurement and insertion of detector misses, with a selectable amplitude method + rich statistics |
| **Label** | Review positives and negatives, apply structured `key=value` labels, curate in bulk, and export CSV/Parquet training regions with provenance |
| **Inspect** | An event browser: step through events, see each waveform, accept/reject to curate |
| **Explore** | One Analysis Explorer for histograms, scatter, event waveforms, noise, I-V, voltage series, subpopulations, fits, clustering, and PCA; plot points link back to events |
| **Export** | Events & measurements to CSV; live plot data to CSV; publication and live figures to PNG/PDF/SVG |

See **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** for a full walkthrough and
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the design and module map.

The current private-beta project format v11 preserves the core session and the analytical
workspace: recording references, calibration/view state, curated detections, measurements, fits,
labels, colours, plot recipes, PCA/clustering, subpopulation reports, PSD, density, I-V,
voltage-dependence, and fingerprinted ML-model references. Raw recordings and model bytes remain
external. The complete contract is documented in
**[docs/STAGE_3_ANALYTICAL_PROJECTS.md](docs/STAGE_3_ANALYTICAL_PROJECTS.md)**.

---

## Supported formats

| Format | Reader | Notes |
|--------|--------|-------|
| Axon ABF (`.abf`) | `neo.rawio.AxonRawIO` | float-format ABF v1; the `CPYA` variant is refused with a specific message |
| HDF5 (`.h5`, `.hdf5`) | `h5py` | flat `/current` `/voltage` `/conductance`; units are undeclared and reported as such |
| TDMS (`.tdms`) | `npTDMS` | modern (rate known) and legacy (rate unknown) layouts; raw-Volts channels flagged |
| Text (`.csv`, `.txt`) | `pandas` | trace tables; channel kind inferred from a `current`/`voltage` column name; the time column's header may name its unit (`time_ms`, `Time (ms)`) |

---

## Quick start

1. **File ▸ Open…** a recording (e.g. an `.abf`). The current channel is selected automatically.
2. Set units in the **Calibration** tab if needed (see the note on current gain below).
3. Press **Ctrl+D** (Analysis ▸ Run detection) to detect events. They appear in the event table and
   as shaded spans on the trace.
4. **Analysis ▸ Event inspector…** to browse and curate events; **Analysis ▸ Plots…** to plot
   dwell/amplitude distributions, scatter, clusters, and fits.

### High-pass and quantitative measurements

For new Schmitt and CUSUM runs, high-pass is a **candidate-localization aid only**. The orange trace
and combined filter-response plot show that localization branch. After finding candidates,
NanoScope measures baseline, depth, fractional depth, mean/max deviation and SNR on the otherwise
identical chain with high-pass omitted. Auto polarity is anchored to that no-high-pass branch; if
ringing makes the high-pass trace vote the other way, localization is repeated with the physical
direction.

High-pass can still move candidate boundaries. Start/end, dwell and rise/fall therefore remain
high-pass-dependent, and ECD uses those inherited integration limits. Results made by an older
build with high-pass enabled retain their original candidate-only measurements and should be
re-run. The open-pore **Measure this recording** probe also uses the no-high-pass characterization
branch. See [Filtering](docs/USER_GUIDE.md#5-filtering) for the full contract. Normalize is a
separate dimensionless transform and retains the caveat documented there.

## Detector reconciliation and training labels

To build a labelled set from several detectors, normally run at least two detector types on the same
current channel. Agreement (`support > 1`) requires identical analysed ranges; a union
(`support = 1`) can retain candidates from compatible source tables with different complete
coverage, but NanoScope suppresses event-rate claims when that coverage cannot be compared safely.
Alternatively, one detector table becomes eligible after it
contains both automatic candidates and hand-added missed events, so manual and detector candidates
can be deduplicated. Add misses to that **source detector table** first, then choose **Analysis ▸
Reconcile detector results…**. Preview an overlap rule and the required detector support, create the
separate **Reconciled** table, and use the **Event table** selector to move between it and the
unchanged sources. Accept/reject and apply structured labels in the Event Inspector or Event Gallery.
Resolve any reported label or contradictory accept/reject conflicts before choosing **Analysis ▸
Export events for training…**; an unresolved review decision is excluded from Reviewed export until
you explicitly accept or reject it.

**Reviewed decisions** is the recommended export because it includes human-reviewed positives and
negatives while excluding untouched candidates and automatic rejects.

NanoScope is an **annotation and provenance-preserving export front end**, not an in-app model
trainer. Training happens in an external notebook or pipeline. It currently has no dataset-split
manager, annotation import/round-trip, reviewer identity/timestamps/audit log, or multi-reviewer /
inter-annotator workflow. Save a `.nanoscope` project to preserve source detections, the reconciled
table, curation, labels, and the label palette.

## A note on calibration

Absolute current values (pA/nA) are only physically meaningful once the **current-channel gain** is
set in the Calibration tab. Detection, dwell times, and the *shapes* of all distributions are correct
regardless; the amplitude *axis* becomes publishable once the gain is dialed in. See the User Guide.

## Testing

```bash
python -m pip install -e ".[dev]"
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

The suite runs headless. A `@pytest.mark.corpus` subset exercises real recordings from the author's
data directory and is skipped when those files are absent. The in-tree harness disables cyclic
garbage collection before Qt is imported to avoid a PySide6 wrapper-finalization deadlock; the
small `tests/lifetime` suite explicitly re-enables it for object-lifetime regressions.

## Private beta development

Source, tests, documentation, and packaging recipes belong in Git. Recordings, user projects,
calibration sidecars, trained models, crash-recovery snapshots, and built applications do not.
Report bugs through the repository's issue form and describe the recording format and reproduction
steps; do not attach confidential recordings. Tested DMGs are published as release assets rather
than committed to the repository.
