I have all the evidence I need. I've verified every target seam against the actual source, confirmed the environment, and spot-checked the SOURCE load-bearing details (including the confirmed `np.trapezoid` blocker and the exact 20-column SCHEMA). Here is the authoritative port plan.

---

# NANOSCOPE DSP PORT PLAN

Authoritative build spec. Port ALL analysis/DSP from `~/nanoscope` (SOURCE) into `~/Desktop/nanoscope` (TARGET). A build team follows only this document.

## 0. Verified environment & ground truth (read first)

Confirmed by direct inspection of both trees and the target Anaconda interpreter (`/opt/anaconda3/bin/python3`):

| Fact | Value | Consequence for the port |
|---|---|---|
| numpy | **1.26.4** (`np.trapezoid` ABSENT — verified `hasattr` False) | **HARD BLOCKER.** SOURCE `stats.py:300` calls `np.trapezoid`. Must shim (D2). |
| scipy | 1.15.1 | OK — all filter/baseline/interp APIs present. |
| pandas | 2.2.3 | OK — `string` extension dtype in SCHEMA works. |
| matplotlib | 3.10.0 (installed, NOT in `pyproject` deps) | figures port works; add to deps. |
| sklearn / hmmlearn | 1.7.0 / 0.3.3 (installed) | **Not needed** — stats EM and level-counting are hand-rolled numpy. Do not introduce a dependency on them. |
| pyqtgraph / PySide6 | 0.14.0 / 6.11.1 | OK. |
| pyarrow | 14.0.2 (installed, NOT in deps) | Available for optional parquet EventTable export. |
| Target layout | flat: `nanoscope/{core,gui,io}/`; **no `analysis/`, no `plot/`, no `conftest.py`** | Create them (D1). |
| Target tests | pytest, `QT_QPA_PLATFORM=offscreen`, `import nanoscope` first (bootstrap), `create_app([])` qapp fixture, `@pytest.mark.corpus` marker. **No pytest-qt** in deps. | SOURCE panel tests need pytest-qt — add to dev deps. |
| SOURCE `Recording` surface (to adapt away) | `.current`(float32), `.fs`, `.polarity`(∈{-1,+1}, validated), `.n`, `.duration_s`, `.dt`, `.time()`, `.slice_s()`, `.with_current(cur, **meta)`, `.name`, `.source`, `.voltage_mV`, `.meta`(dict) | Every DSP entry point takes a `Recording`; replace per §3. |
| SOURCE `events.SCHEMA` | **exactly 20 columns** (the detect study's "21" is wrong) | See §3.3 for the canonical list. |

All target integration seams (`core/analysis.py`, `document.py`, `session.py`, `types.py`, `calibration.py`, `layer.py`, `gui/trace_view.py`, `gui/workers.py`, `gui/main_window.py`, `gui/calibration_panel.py`, `gui/app.py`) were read in full and match the supplied TARGET contract verbatim. The 6 gaps the contract names (no concrete `AnalysisContext`; no analysis worker; no `resultsChanged` emitter/`add_result`; empty `_OVERLAY_RENDERERS`; no results panel/params form; no `register_module` calls) are all real and are filled below.

## 1. Key decisions (resolve before coding)

- **D1 — DSP lives in a new `nanoscope/analysis/` package, not in `core/`.** The individual studies each say "becomes `core/filters.py`" etc., but `core/analysis.py` already exists as the *seam*. Putting DSP modules next to it in `core/` invites confusion (`core.analysis` the seam vs `core.filters` the DSP). Decision: `core/` stays data-model + seam; all ported DSP + AnalysisModule adapters go in `nanoscope/analysis/`. No import clash (`nanoscope.analysis` package vs `nanoscope.core.analysis` module are distinct dotted paths). This also matches the GOAL's requested tree.
- **D2 — `np.trapezoid` shim (mandatory).** Top of `analysis/stats.py`: `_trapz = getattr(np, "trapezoid", np.trapz)` and call `_trapz(...)`. Do **not** upgrade numpy (would risk the neo/h5py/nptdms reader stack the v1 app depends on). One-line shim only.
- **D3 — Fits and truth-validation are NOT channel-driven `AnalysisModule`s.** The `AnalysisModule.run(ctx)` contract is channel-in. Only **detection** fits it. Dwell/depth **fits** consume the detection result's event table (not a raw channel); **null/sweep/recovery** re-run the detector on surrogates. Decision: detection = an `AnalysisModule`; fits + truth-validation = plain functions invoked by the analysis controller, their outputs stored as derived `AnalysisResult`s (module_ids `fit.dwell`, `fit.depth`) or fed to dialogs. This keeps the seam clean and channel-first.
- **D4 — Filtered trace = a managed preview curve, not an `OverlayHint`.** A low-pass output is a full 1e6–1e7-sample float32 *signal*, not a span/marker array; `OverlayHint` is the wrong vehicle (it carries int64 index arrays). Decision: the filter live-preview is a managed `pg.PlotDataItem` added to the pane (peak-downsampled by pyqtgraph), owned by the analysis panel, toggled on/off. A full derived-document path (reuses panes/calibration) is a v3 refinement, flagged in §10.
- **D5 — Export = matplotlib (SOURCE `plot/figures.py`), kept separate from the live pyqtgraph view.** The two studies confirm this is the intended split: pyqtgraph = interactive; matplotlib = publication export with source-data CSV. Port both.

## 2. Target file tree to add

```
nanoscope/analysis/
  __init__.py            # register_all(): imports modules.py (registers AnalysisModules) — call at app start
  _compat.py             # _trapz shim (D2); shared tiny helpers
  filters.py             # signal-first kernels + measure_rise_time + noise_rms + psd + _design_sos
  events.py              # SCHEMA (20 cols) + empty_df + EventSet  (VERBATIM, io import dropped)
  characterize.py        # characterize(current, baseline, starts, ends, fs, polarity, ...)
  baseline.py            # derive_polarity + estimate_baseline + open_pore_current + baseline_intrusion + BaselineWarning
  detect.py              # detect_polarity (verbatim) + detect_events engine + _runs/_run_maxima
  stats.py               # FitResult + fit_dwell_exponential + fit_depth_gaussians  (+ _trapz shim)
  truth.py               # SynthRecording shim + synth_trace + aaft_surrogate + null_control + sweep_threshold + recovery_map
  context.py             # SimpleAnalysisContext (concrete AnalysisContext — fills contract gap #1)
  modules.py             # DetectSchmitt(AnalysisModule) + register_module(...) ; fit_dwell/fit_depth wrappers

nanoscope/plot/
  __init__.py
  style.py               # VERBATIM from SOURCE (pure matplotlib; zero adaptation)
  figures.py             # event-stats figures (data-model-agnostic) + trace_preview (→ CalibratedChannel) + save_figure

nanoscope/gui/
  panels.py              # DetectionPanel, StatsPanel, EventTable, SweepDialog  (ported from SOURCE panels.py)
  overlays.py            # SpanOverlayItem + render_spans/render_hline/render_marker + register_all_renderers()  (fills gap #4)
  analysis_worker.py     # AnalysisController + AnalysisWorker  (mirror workers.py; fills gap #2)
  analysis_panel.py      # AnalysisPanel: params form + Run + results table + event nav + staleness  (fills gap #5)

nanoscope/core/
  session.py             # EDIT: add Session.add_result(res)  (fills gap #3)

nanoscope/gui/
  main_window.py         # EDIT: build AnalysisController + AnalysisPanel tab, wire selection, Run action, overlay lifecycle
  app.py                 # EDIT: main() calls analysis.register_all() + overlays.register_all_renderers() before MainWindow

tests/
  conftest.py            # qapp fixture, register_all(), synth CalibratedChannel/context builders
  analysis/test_filters.py test_baseline.py test_characterize.py test_events.py
  analysis/test_detect.py  (carries make_trace backbone + recall/precision>0.95)
  analysis/test_stats.py test_truth.py
  plot/test_figures.py
  gui/test_panels.py       (pytest-qt) test_overlays.py test_analysis_integration.py

pyproject.toml           # EDIT: runtime deps += scipy, matplotlib, pyarrow ; dev deps += pytest-qt
```

## 3. Per-module port spec

General adaptation rules applied to every DSP module:
1. **Drop** `from nanoscope.io.recording import Recording` and every `rec.*` access.
2. Refactor each `Recording`-consuming function into a **signal-first kernel** taking `(data: np.ndarray, fs: float, ...)`; keep every internal algorithm byte-for-byte.
3. `rec.current`→`data`, `rec.fs`→`fs`, `rec.n`→`data.size`, `rec.polarity`→a passed-in/derived `int`, `rec.with_current`→drop (provenance is returned separately), `rec.meta['pore_id']`/`rec.name`→explicit `pore_id`/`recording` string args.
4. **float32→float64**: CalibratedChannel.data is float32; upcast to float64 internally exactly as SOURCE already does (`np.asarray(x, dtype=np.float64)`). Return float32 contiguous for *signals* (matches `CalibratedChannel.data` rule); return float64 for *baselines* and *fit inputs* (SOURCE rule).
5. **Units**: SOURCE hard-codes `pA` in docstrings only; code is unit-agnostic. Carry `CalibratedChannel.units` through into provenance/labels. Keep SCHEMA column names (`depth_pA`, `ecd_pA_s`, …) as the fixed contract even though magnitudes are in the channel's own units.

### 3.1 `analysis/filters.py`
Port verbatim (Recording-free already): `noise_rms(x)`, `measure_rise_time(fs, cutoff_hz, *, order=8, kind="bessel")`, `_design_sos`, `_gaussian_sigma_samples`, `_filter_fn`, `_validate_cutoff`.
Re-seam the three Recording-consumers to kernels:
- `lowpass(data, fs, cutoff_hz, *, order=8, kind="bessel") -> np.ndarray` (float32 contiguous, same length).
- `notch(data, fs, freq_hz=60.0, *, q=30.0, harmonics=3) -> np.ndarray` (float32).
- `psd(data, fs, *, nperseg=65536) -> tuple[np.ndarray, np.ndarray]`.
The meta stamps (`filter_kind`, `filter_cutoff_hz`, `filter_order`, `filter_zero_phase=True`, `filter_rise_s`) move to the *caller* (detector provenance / preview provenance).
**Invariants (must survive):** zero-phase always (`sosfiltfilt`/symmetric Gaussian — edges must not shift); Bessel **`norm='mag'`** (not scipy default `'phase'`); IIR **`output='sos'`** (not `'ba'`); filtfilt delivers **−6 dB** at cutoff (report resolution via *measured* rise time, never 0.34/fc); rise time measured by pushing a step through the *exact same `_filter_fn`* `lowpass` uses (they must never diverge); `noise_rms = 1.4826·MAD`, never `np.std`; Gaussian σ gives −3 dB at cutoff; cutoff validation raises at/above Nyquist, warns above fs/2.5; PSD `scaling='density'`.

### 3.2 `analysis/events.py`
Port **verbatim** (pure numpy/pandas, no io dependency): `SCHEMA`, `REQUIRED`, `empty_df()`, `EventSet` (with `.accepted`, `.n_pores`, `.rate_hz()`, `.pool()`, `.describe()`, `__repr__`/`_repr_html_`), `concat()`.
**Canonical 20 columns** (verified — this is the cross-module contract): `event_id`(int64), `recording`(string), `pore_id`(string), `start_idx`(int64), `end_idx`(int64), `t_start_s`(float64), `dwell_s`(float64), `baseline_pA`(float64), `depth_pA`(float64), `frac_depth`(float64), `ecd_pA_s`(float64), `max_dev_pA`(float64), `mean_dev_pA`(float64), `rise_s`(float64), `fall_s`(float64), `snr`(float64), `n_levels`(int64), `filter_limited`(bool), `accepted`(bool), `flag`(string).
**Invariants:** `pool()`/`concat()` **refuse to span pores** unless `allow_multi_pore=True` (pore-to-pore variance dominates counting error); `rate_hz()` requires `provenance['analyzed_duration_s']`; `string` is the pandas extension dtype, not object.
**`flag` is a SET, not a slot** (C1/C21/C10). The plan described it as "why rejected, or ''", a single value, and every writer therefore OVERWROTE it — which is how a merge erased `'dwell>max_dwell_s'` and un-rejected the row (240 accepted rows over a declared 2 ms limit on the owner's -600 mV recording, worst 24.88 ms) and how a mask resize left a row rejected with no reason at all. The column holds `FLAG_SEP`-joined tokens; the only two operations on it are `combine_flags` (union) and `inherited_curation` (what a REBUILT row carries over: curator tokens and hand rejections yes, automatic rejections re-judged). Read it with `has_flag`, **never** `df['flag'] == token` — the equality misses a row carrying the token alongside another.

### 3.3 `analysis/characterize.py`
Signature adaptation (drop Recording; take primitives):
```python
def characterize(current: np.ndarray, baseline: np.ndarray, starts: np.ndarray,
                 ends: np.ndarray, fs: float, polarity: int, *, noise_rms: float,
                 filter_rise_s: float | None = None, max_dwell_s: float | None = None,
                 clip_bounds: tuple[int, int] | None = None,
                 inherit_flag=None, inherit_accepted=None, pore_id: str = "",
                 recording: str = "", units: str = "pA") -> pd.DataFrame
```
**This function owns `flag` and `accepted`.** The plan originally left the two automatic rejection rules to the CALLER — `max_dwell_s` and the boundary clip were both applied to the frame `detect_events` got back — and that is exactly what produced C1/C21/C10: the ML detector calls `characterize` directly and so never got the boundary rule at all, and the three row-rebuild paths (merge, Event Inspector resize, hand-added event) each re-decided `flag`/`accepted` their own way. All three rules (`zero_baseline`, `dwell>max_dwell_s`, `clip_bounds` → `clipped_by_boundary`) live here now, and `recharacterize_span` is the ONE padded-window rebuild every non-detector path calls. Two AST guards in `tests/analysis/test_flag_invariant.py` keep it that way.
Everything internal ports **verbatim**: `_grouped_median` (lexsort), `_rise_fall`, `_cross`, `_count_levels`/`_segment_count`/`_best_split`/`_seg_rss`/`_bic`, the flat-gather + `reduceat` reductions. Cast `current`/`baseline` to float64 at entry (mandatory — float32 in).
**Invariants:** **depth = MEDIAN of in-event dev, not max** (max is biased high by ~σ√(2 ln N), grows with dwell → manufactures a spurious dwell–depth correlation; `max_dev_pA` still reported but never used as depth); `dev = polarity·(current−baseline)` positive-in-event for both polarities; baseline is full-length local I0 subtracted per-sample; `frac_depth` against `|baseline|==0` is **NaN + flag `'zero_baseline'`**, never inf; `rise_s`/`fall_s` **NaN** when the transition isn't in-window, never 0; hard floors `_MIN_SAMPLES_FOR_LEVELS=10`, `_MAX_LEVELS=4`, `_MIN_LEVEL_SAMPLES=3`, `_EDGE_PAD_SAMPLES=64`; output is exactly `list(SCHEMA)` order/dtypes; loud ValueErrors with the tested substrings (`'full-length'`, `'differ in length'`, `'noise_rms'`, `'zero-length'`, `'outside the trace'`, `'ascending'`, `'overlap'`, `'filter_rise_s'`); vectorized path must match a naive per-event loop to rtol 1e-9.

### 3.4 `analysis/baseline.py`
Add polarity derivation (target stores no polarity — analysis.py Rule 5):
```python
def derive_polarity(data: np.ndarray) -> int      # sign from a robust baseline median → +1/−1
def estimate_baseline(data: np.ndarray, fs: float, *, polarity: int | None = None,
                      window_s=0.01, method="block_median", two_pass=False,
                      min_dwell_s=2e-5, percentile=80.0, asls_lam=6e4, asls_p=1e-3,
                      asls_niter=10, asls_decim=0.1, two_pass_k=4.0,
                      two_pass_pad_s=1e-4) -> np.ndarray   # float64, full length
def open_pore_current(data: np.ndarray, fs: float, baseline: np.ndarray | None = None) -> float
def baseline_intrusion(current, baseline, events, fs, polarity, *,
                       flank_s=5e-3, pad_s=2e-4) -> dict
class BaselineWarning(UserWarning): ...
```
`polarity=None` → `derive_polarity(data)`. All internals verbatim: `_single_pass`, `_block_stat` (reshape reduction), `_expand` (**PchipInterpolator**, extrapolate), `_asls_knots` (Eilers–Boelens on decimated knots), `_mask_events` (MAD dilate), `_spans` (keep the EventSet/.df / DataFrame / (starts,ends) duck typing — v2 SPAN overlay `starts`/`stops` feed it directly).
**Guard sample_rate_hz is None**: `estimate_baseline` requires a real rate; raise a clear error when `fs is None`. The owning module sets `requires_time_base=True`.
**Invariants:** Pchip not cubic spline (no drift-step overshoot → no phantom events); MAD not std for the two-pass σ (survives ~50% occupancy); percentile q flips with polarity; asls runs on decimated knots with tiny `asls_p`; returned baseline always float64 full-length; `open_pore_current` is **signed**; `baseline_intrusion` score is **NaN not 0** when no measurable events, `severe` at >0.1, verdict escalates at >0.3; `window_s ≥ 20·min_dwell_s` enforced as `BaselineWarning`; over-long window / non-finite / unknown method **raise**.

### 3.5 `analysis/detect.py`
`detect_polarity(current, baseline, *, k=5.0) -> tuple[int, dict]` — **verbatim** (already signal-first). `_runs`, `_run_maxima`, `_as_baseline`, `_exclusion_samples`, `_merge_intervals` — verbatim.
Refactor `detect(rec, ...)` into the engine:
```python
def detect_events(signal: np.ndarray, fs: float, *, polarity: int | None = None,
                  k_in=5.0, k_out=2.0, min_dwell_s=2e-5, max_dwell_s=None,
                  baseline: np.ndarray | float | None = None,
                  filter_cutoff_hz: float | None = None, merge_gap_s: float | None = None,
                  exclude_regions=None, pore_id="", recording="",
                  units="pA") -> tuple[pd.DataFrame, dict]
```
Adaptations: `polarity=None` → **measure** via `detect_polarity` (verdict `'measured'`), not a stored field; **drop** the `rec.polarity` cross-check (nothing to contradict) — optionally keep a soft warning in provenance when `evidence['verdict']=='undetermined'`; internal filter path calls `filters.lowpass(signal, fs, cutoff)` + `filters.measure_rise_time(fs, cutoff)`; baseline path calls `estimate_baseline(signal, fs, polarity=...)`; features via `characterize(signal, base, starts, ends, fs, polarity, noise_rms=..., filter_rise_s=..., pore_id=..., recording=..., units=units)`. Returns `(df, provenance)` with **`analyzed_duration_s`** (exclusions merged, not double-counted) and the full ~40-key provenance.
**Invariants:** Schmitt two-threshold **`k_out < k_in`** (enforced by raise; single-threshold shatters events >1.5× and drops precision <0.6); MAD noise; `_MIN_RUN_SAMPLES=2` coherence; the **+1 spare `-inf` trailing slot** on `dev_padded` for `reduceat`; excluded regions detected-through then dropped (not zeroed → no spurious boundary events); `max_dwell_s` **keeps** long events `accepted=False` flag `'dwell>max_dwell_s'` (a clog is real signal) — but the RULE is `characterize`'s (§3.3), forwarded from here, not applied to the returned frame; the same goes for `clipped_by_boundary`, which was applied here and therefore never reached the ML detector (C10); `min_dwell_s` drops; `filter_limited := dwell_s < 2·filter_rise_s`; non-finite input raises; 5M samples < 1 s (all per-sample work vectorized).

### 3.6 `analysis/stats.py`
Port **verbatim** with the **D2 shim only**: `FitResult`, `fit_dwell_exponential(dwell_s, *, t_min, n_components=1, efficiency=None, n_boot=1000, seed=0)`, `fit_depth_gaussians(frac_depth, *, n_components=1, n_boot=400, seed=0)`, `_Normalizer`, `_fit_mix`, `_em_gauss`, `_clean_1d`, `_resolve_efficiency`. Replace `np.trapezoid` → `_trapz`.
**Invariants:** left-truncation is the whole point — for k=1 the MLE is **exactly `mean(t)−t_min`** (a naive `expon.fit` is biased long by exactly `t_min`); the k=1 closed-form **self-check** (RuntimeError if numerical MLE disagrees >1e-3 rel) stays; dwell in **seconds**, `t_min` from the detector's real resolution (min-width samples / fs), never hard-coded; `.params` = untruncated physical τ/weights, `.predict()` = **censored observed density** (overlays a density histogram); components sorted (τ for dwell, μ for depth); efficiency Z split at grid top (quadrature below, analytic survival tail above); every fail-loud validation and message substring preserved.

### 3.7 `analysis/truth.py`
Add a minimal shim to replace `io.Recording` (synth traces are invented, not read from Session):
```python
@dataclass(frozen=True)
class SynthRecording:
    data: np.ndarray; sample_rate_hz: float; polarity: int
    name: str = "synthetic"; units: str = "pA"; meta: dict = field(default_factory=dict)
    @property
    def n(self): return self.data.size
    @property
    def duration_s(self): return self.n / self.sample_rate_hz
    def with_current(self, data, **m): return replace(self, data=np.asarray(data, np.float32),
                                                      meta={**self.meta, **m})
```
Port `aaft_surrogate` **verbatim** (pure numpy). `synth_trace` returns `(SynthRecording, DataFrame)`. `null_control`/`sweep_threshold`/`recovery_map` adapted to call `detect_events(rec.data, rec.sample_rate_hz, polarity=rec.polarity, ...)` (via a lazy import of `analysis.detect`) instead of `detect(rec)`; keep `_place_events`, `_ranges`, `_pink_noise`, `_draw_noise`, `_match`, `_longest_run`, `_make_efficiency_fn` (RegularGridInterpolator), `_quiet_noise_pool`, `_event_free_trace` verbatim.
**Invariants:** AAFT null built from **event-removed** noise (surrogating the raw trace pins fdr≈1); surrogate/probe detect runs pass `polarity=` explicitly + drop any provided baseline; a mirror/flip control is **rejected** (enhancement is real physics); noise RMS rescaled exactly, pink noise is a variance fraction; blockades multiplicative on local baseline; efficiency_fn interpolates in **log(dwell)**, clipped to [0,1]; `_place_events` **raises** rather than truncating; the sign convention already matches analysis.py Rule 5.

### 3.8 `plot/style.py` + `plot/figures.py`
`style.py`: **verbatim** (pure matplotlib; `pdf.fonttype=42`, Okabe-Ito palette, scoped `rc_context` — must not set rcParams at import).
`figures.py`: the four **event-statistics figures** (`dwell_histogram`, `depth_histogram`, `scatter_dwell_depth`, `event_rate_plot`) + `save_figure` are data-model-agnostic — port unchanged; keep `_unpack` duck-typed (`getattr(events,'df',events)` + `getattr(events,'provenance',{})`) so it eats the EventSet, a raw DataFrame, or `AnalysisResult.payload`. Keep the lazy `_stats_module()` tri-state (`fit=True|False|FitResult`) → repoint to `analysis.stats`. Adapt only `trace_preview` to read `Session.calibrated(doc_id,ch).data` + `Session.x_axis` and label from `CalibratedChannel.units`; keep `_minmax_envelope` verbatim.
**Invariants:** **log-bin Jacobian** `expected_count = N·f(t)·t·ln(10)·du` (the `t·ln 10` factor is the classic silent bug — port the exact-expectation test *and its negative control*); log-spaced dwell bins by default; geometric bin centers; `np.nextafter` edge nudging; accepted-only events; mandatory below-detection shaded band; density-vs-density for depth (no Jacobian); `event_rate_plot` never averages pores (requires `pore_id`); filter-limited events drawn distinctly; **min/max envelope** decimation (never subsample — a 20 µs spike must survive); every figure attaches source data and `save_figure` writes PDF+PNG+CSV.

## 4. AnalysisModule seam wiring

**`analysis/context.py` (gap #1):**
```python
@dataclass
class SimpleAnalysisContext:                      # satisfies core.analysis.AnalysisContext (runtime_checkable)
    channel: CalibratedChannel
    meta: RecordingMeta
    params: Mapping[str, Any]
    progress_cb: Callable[[float, str], None] = lambda f, m="": None
    cancel_flag: Callable[[], bool] = lambda: False
    def report_progress(self, frac, msg=""): self.progress_cb(frac, msg)
    def cancelled(self): return self.cancel_flag()
```

**`analysis/modules.py` — `DetectSchmitt` (gap #6 registration):**
```python
class DetectSchmitt:
    id = "detect.schmitt"; label = "Detect events (Schmitt)"; version = "1.0"
    accepts_kinds = (ChannelKind.CURRENT,)     # gate: corpus has voltage channels
    requires_time_base = True                  # dwell is undefined without fs → GUI disables on rate-unknown docs
    def run(self, ctx) -> AnalysisResult:
        ch = ctx.channel
        if ch.sample_rate_hz is None: raise RuntimeError("detector requires a sample rate")
        doc_id = str(ctx.meta.path.resolve()); channel = ch.info.index
        fp = tuple(ctx.params.get("_calibration_fingerprint", (1.0, 0.0)))   # injected by controller
        df, prov = detect_events(ch.data, ch.sample_rate_hz, units=ch.units,
                                 recording=ctx.meta.path.name,
                                 pore_id=ctx.params.get("pore_id",""),
                                 **{k: v for k,v in ctx.params.items() if not k.startswith("_")})
        events = EventSet(df, prov)
        spans = OverlayHint(OverlayKind.SPAN, starts=df["start_idx"].to_numpy(np.int64),
                            stops=df["end_idx"].to_numpy(np.int64), label="events")
        # baseline HLINE overlay: emit DECIMATED (centres→int64 idx, knots→values), not 5M points
        scalars = {"n_events": float(len(df)), "rate_hz": events.rate_hz(),
                   "noise_rms": prov["noise_rms_pA"], "median_baseline": prov["baseline_median_pA"],
                   "filter_rise_s": prov.get("filter_rise_s") or float("nan")}
        return AnalysisResult(module_id=self.id, module_version=self.version,
                              doc_id=doc_id, channel=channel, params=dict(ctx.params),
                              calibration_fingerprint=fp, payload=events,
                              overlays=(spans,), scalars=scalars, warnings=_build_warnings(prov, df))

def register_all():
    register_module(DetectSchmitt())
```
The controller **injects** `_calibration_fingerprint = session.calibration(doc_id,ch).fingerprint()` and `pore_id` into `params` before the run (the module has no Session). doc_id/channel are derived from `ctx.meta.path`/`ctx.channel.info.index`.

**`core/session.py` edit (gap #3):**
```python
def add_result(self, res: AnalysisResult) -> None:
    self._results.put(res)
    self.resultsChanged.emit(res.doc_id, res.module_id)
```

**`gui/analysis_worker.py` (gap #2) — mirror `workers.py` exactly:** `AnalysisWorker(QObject)` with `progress(float,str)/finished(object)/failed(str)/cancelled()`, a `threading.Event` cancel via a **plain method** (NOT a `@Slot` — the documented reason: a `@Slot` cancel queues behind `run()` in the blocked event loop). `run()` builds a `SimpleAnalysisContext(progress_cb=self.progress.emit, cancel_flag=self._cancel.is_set)` and calls `module.run(ctx)`; never lets an exception escape (`failed(str)` / `cancelled()`). `AnalysisController` mirrors `LoadController`: `_spin_up`, `abort` (detach-then-cancel-then-quit/bounded-wait), `_is_active()` sender guard. **NOT terminate** — the plan originally said `quit/wait/terminate`, and that is precisely what deadlocked the app (C19): `QThread.terminate()` kills a worker holding the GIL and the unbounded `wait()` after it can never be satisfied. All three controllers now spin up and retire through `gui/worker_thread.py`, which never terminates and never waits without a bound; a worker that will not stop is detached and left to wind down. Cancellation itself is cooperative and ambient — see `analysis/cancel.py`. Add a generic job path (`run_callable(fn) -> emits finished(obj)`) for null/sweep/recovery (they return dicts/DataFrames, not `AnalysisResult`). On `finished(AnalysisResult)`, the **main thread** calls `session.add_result(res)`.

## 5. Overlay rendering (gap #4)

**`gui/overlays.py`:**
- `class SpanOverlayItem(pg.GraphicsObject)` — holds x-start/x-stop float arrays; `boundingRect()` spans full y of the viewbox; `paint()` clips to the current x-range and draws one `QRectF` per **visible** span with a semi-transparent brush. This is the anti-`10k-LinearRegionItem` design the seam exists for — N rects in one `paint()`. Connect its repaint to `TraceView.sigVisibleRangeChanged` (lazy repaint on zoom).
- `render_spans(plot, session, doc_id, hint) -> [SpanOverlayItem]`: convert `hint.starts/stops` (int64 sample indices) → x via `session.x_axis(doc_id).values[idx]`, add the item to `plot`, return `[item]`.
- `render_hline(plot, session, doc_id, hint)`: baseline track — plot `values` at `starts` indices as a light step/curve (baseline was emitted decimated).
- `render_marker(plot, session, doc_id, hint)`: one highlighted rect / `InfiniteLine` pair for the **selected** event (this is the ≤1 real interactive item allowed).
- `register_all_renderers()`: `register_overlay_renderer(OverlayKind.SPAN, render_spans)` + HLINE + MARKER.
Index→x uses `x_axis(doc_id)` at offset 0 (the doc's native t0); layer-time-offset-aware overlays are a v3 refinement (the renderer signature carries no channel/offset) — flag in §10.
**Callers** (also a gap — nothing calls `add_overlay` in v1): the `AnalysisPanel` calls `trace_view.add_overlay(doc_id, channel, hint)` for each `result.overlays` after a run, stores the returned `overlay_id`s, and `remove_overlay`s them on re-run / doc close / result-cleared.

## 6. GUI integration

**Ported `gui/panels.py`** (from SOURCE, adapted): `DetectionPanel` (baseline/filter/threshold/polarity controls; `params()` → the `ctx.params` mapping with the **None off-sentinels** for `filter_cutoff_hz` and `merge_gap_s`; live rise-time readout via `analysis.filters.measure_rise_time`; **`k_out<k_in` enforced live**; `set_recording(fs)` — but **tolerate `fs is None`**: disable Run + show "rate unknown" instead of calling `set_recording` with a fake rate). `StatsPanel` (summary rows, FDR readout, gated fit controls, monospace fit text). `EventTable` (`QAbstractTableModel` over the DataFrame — never one widget per row; uses ported `events.SCHEMA`; `a`/`r`/`f` curation writes back in place; `eventSelected(event_id)`). `SweepDialog` (pyqtgraph plateau plot; reads `sweep_df` columns `k`/`n_events`/optional `in_plateau`/`n_surrogate` + `attrs['k_recommended']`).

**`gui/analysis_panel.py`** — mirror `CalibrationPanel`: constructed with `(session, controller, trace_view)`; composes `DetectionPanel` + `StatsPanel` + `EventTable`; `set_selection(doc_id, channel)` pushed from `MainWindow._on_selection`; subscribes to `session.resultsChanged`, `session.calibrationChanged`, `session.documentClosed`; **never talks to another panel**. Behaviours:
- **Run**: on `DetectionPanel.detectRequested(params)` → gate on `accepts_kinds`/`requires_time_base` (kind==CURRENT and `meta.sample_rate_hz` not None) → `controller.run_module("detect.schmitt", doc_id, channel, params)`.
- On `resultsChanged(doc_id, module_id)` for the current selection → refresh `EventTable`/summary; remove old overlays; `trace_view.add_overlay(...)` for each `result.overlays`; keep `overlay_id`s.
- **Event navigation**: `EventTable.eventSelected` → `trace_view.zoom_x_to(t_start, t_start+dwell)` and draw the MARKER overlay (dive into the event).
- **Staleness**: compare `result.calibration_fingerprint` to `session.calibration(doc_id,ch).fingerprint()`; on mismatch (or on `calibrationChanged` for the selection) show a "stale — re-run" banner (the user's deferred current-gain workflow guarantees this fires).
- **Filter live-preview (D4)**: `DetectionPanel` filter toggle → compute `filters.lowpass(ch.data, fs, cutoff)` and hand to a new `TraceView.set_filter_preview(doc_id, channel, data|None)` that adds/removes a managed peak-downsampled `pg.PlotDataItem` on the pane (reuse the four-step `make_curve` order). Recompute on cutoff change; the pane's own clip-to-view handles re-render on zoom.
- **Sweep/Null/Recovery**: `DetectionPanel` buttons → `controller.run_callable(...)` wrapping `truth.sweep_threshold`/`null_control`/`recovery_map`; results → `SweepDialog` / `StatsPanel.set_fdr` / stored `efficiency_fn` (enables `StatsPanel.use_efficiency`).
- **Fits**: `StatsPanel.fitRequested` → read the current detection `AnalysisResult.payload` (dwell_s/frac_depth, `t_min` from provenance) → `stats.fit_dwell_exponential`/`fit_depth_gaussians` → store derived `AnalysisResult`s (`fit.dwell`/`fit.depth`) via `session.add_result` → `StatsPanel.set_fit_text` + drive figure overlays.

**`gui/main_window.py` edits:** build `AnalysisController(self)`; wire its `probed`/`finished`/`failed`/`progressed`/`cancelled` like `LoadController`; build `AnalysisPanel(self._session, self._analysis_controller, self._trace_view)` and add it as a **third Inspector tab** ("Analysis") in `_build_docks` (same `QScrollArea` pattern) — or a bottom dock if vertical room is tight; extend `_on_selection` to also call `self._analysis_panel.set_selection(doc_id, channel)`; add an **"Analysis" menu** with a "Run detection" action (enabled only when selection is a CURRENT channel on a rate-known doc) and "Export figure…"; overlay lifecycle handled inside `AnalysisPanel`.

**`gui/app.py` edit:** in `main()`, before `MainWindow()`: `from ..analysis import register_all; register_all()` and `from .overlays import register_all_renderers; register_all_renderers()` (registration must precede any module lookup / overlay dispatch). Tests call the same two functions in `conftest.py`.

**Figure export:** an "Export figure…" action runs `plot.figures.*` on the current detection result's EventSet and `plot.figures.save_figure(fig, path)` (PDF+PNG+CSV). Optional: parquet EventTable export via pyarrow.

> **CORRECTED (C11/C18).** The sentence above says `save_figure(fig, path)` writes "PDF+PNG+CSV" and then leaves the GUI to ask the user for *one* `path`. That is the defect: the file dialog's overwrite prompt covered the single name the user typed (`figures.pdf`, which is **never written**) while the command silently replaced up to **twelve** real files — `{stem}_{dwell,depth,scatter,trace}.{pdf,png,csv}` — the `.csv` source-data ones most damagingly. Same shape in "Export events for training…", which also writes an unmentioned `<stem>.meta.json`. **Rule: a writer that produces siblings must EXPORT AN ENUMERATION of them (`figures.figure_output_paths`, `scoring.training_export_paths`), and every GUI save dialog goes through `gui/export_paths.ask_export_path`, which confirms that enumeration.** No module under `nanoscope/gui` may call `QFileDialog.getSaveFileName` itself; `tests/gui/test_export_overwrite.py` holds the AST guard.

## 7. Load-bearing invariants — master checklist (any one broken = silently wrong science)

1. **Detected (measured) polarity** — never assume "blockade = down"; derive from the sign of a robust baseline / count coherent excursion runs; enhancement (+1) is real physics; corpus has both signs (ABF ≈ −49 pA, nisin CSV ≈ +0.295 nA). Work in rectified space, never rectify stored data in place.
2. **`dev = polarity·(current−baseline)` positive-in-event** for both polarities; depth/ECD/frac_depth are positive magnitudes; direction lives only in `polarity`.
3. **MAD noise** (`1.4826·MAD`), never `np.std` — std absorbs the events being detected and drops the smallest real ones.
4. **Schmitt two-threshold** `k_out < k_in` (enforced by raise); single-threshold shatters events.
5. **Block-median + Pchip baseline** — Pchip (shape-preserving), never cubic spline (rings → phantom events); window ≥ 20× longest dwell; two-pass σ uses MAD.
6. **Depth = MEDIAN of in-event dev**, not max (max bias grows with dwell → spurious dwell–depth correlation).
7. **Censored-MLE dwell** — k=1 MLE is exactly `mean(t)−t_min`; a naive `expon.fit` is biased long by `t_min`; keep the closed-form self-check; `.predict()` returns the censored observed density.
8. **Log-bin histograms + Jacobian** `N·f(t)·t·ln 10·du` for the count-histogram overlay; density-vs-density (no Jacobian) for depth; log-spaced dwell bins by default; geometric centers; `nextafter` edge nudging.
9. **AAFT surrogate** built from event-removed noise; a mirror/flip control is rejected.
10. **Measured rise time** through the exact `_filter_fn`; filter-limited := `dwell_s < 2·rise`; report resolution via rise time, never 0.34/fc; filtfilt is −6 dB at cutoff.
11. **`pool()`/`concat()` refuse cross-pore** unless `allow_multi_pore=True`.
12. **Bessel `norm='mag'` + IIR `output='sos'`**; zero-phase everywhere (edges must not shift).
13. **Min/max envelope decimation** (`trace_preview` and the live pane) — never subsample; a one-sample spike must survive.
14. **`analyzed_duration_s`** subtracts merged (not double-counted) exclusions — the denominator of every rate.
15. **`max_dwell_s` keeps** long events `accepted=False`; `min_dwell_s` drops.
16. **float32 discipline**: signals out are float32 contiguous; baselines/fit inputs float64; upcast internally.
17. **`requires_time_base=True`** on the detector → GUI disables it (honest reason) when `sample_rate_hz is None`, never fabricates dwell.
18. **`calibration_fingerprint`** stamped on every result; UI marks stale + offers re-run when it diverges from `session.calibration(...).fingerprint()`.
19. **`OverlayHint.starts/stops` are int64 sample indices** (lossless; survive a rate fix); overlays rendered as batched rects, not per-event items.
20. **`np.trapezoid` shim** (D2) — target numpy is 1.26.4.
21. **No write may destroy what it is replacing until the replacement exists.** Every file the
    app writes over user data — the `.nanoscope` project above all — is staged in a temporary
    sibling and moved into place with `os.replace`. `zipfile.ZipFile(path, "w")` truncates on
    OPEN, so the pre-fix save destroyed a good project before it knew whether it could write a
    new one (C13). **`core/atomic_write` is the seam, and it is now the seam for all of them:**
    this sentence said "every file" while `calibration.save_sidecar`, `scoring`'s two exports,
    `measure.write_measurements_csv` and `plot.figures.save_figure` all wrote in place. An AST
    guard (`tests/core/test_persist_invariant.py`) fails the build if any module under
    `nanoscope/` writes a path it did not stage. The seam also owns what an `os.replace`
    otherwise silently changes about the FILE — a symlink resolved rather than replaced,
    extended attributes carried across, a temp name that fits `PC_NAME_MAX`, a read-only file
    refused — and, for a SET of files (one figure export writes twelve), that they land
    together or not at all.
22. **A save dialog must confirm the write's REAL output set, not the name the user typed.**
    A writer with siblings exports an enumeration of them; every GUI save dialog goes through
    `gui/export_paths.ask_export_path` (C11/C18 — see §6).
23. **Deleting a file the app does not own is never a side effect.** `<datafile>.npcal.json` is
    the user's, and `save_sidecar` removes it whenever every channel is back at its default —
    reachable from *Reset to default* AND from *Apply* with default values typed in. Ask via
    `calibration.sidecar_effect` BEFORE committing, report the outcome after, and keep the
    calibration panel's commits on ONE path (C12).
24. **Non-finite input never comes back as a confident number.** `np.isfinite` is gated at three
    invariants, not at each caller: `TraceDocument.__post_init__` names every channel carrying
    NaN/inf in `meta.issues` (so all four readers report); `ChannelCalibration.__post_init__`
    refuses a non-finite gain/offset (so the sidecar, the project and the Gain box are all
    covered by one check); and `psd.band_rms` returns **NaN with a warning**, never `0.0`, for a
    band it cannot integrate — one NaN sample poisons every Welch bin, and `0.0` reads as a
    flawless pore (C4/C8/C41). Read-outs must render these through an is-finite check.

## 8. Test port plan

Bring over every SOURCE test, swapping the `Recording`/`make_rec` builder for the target constructor and repointing imports to `nanoscope.analysis.*` / `nanoscope.plot.*`. All numeric assertions transfer unchanged.

- **Backbone (port first):** the SOURCE `make_trace(...)` ground-truth generator (`test_detect.py`) and `match_one_to_one()` (strict one-to-one overlap). Put a shared version in `tests/conftest.py` producing a `SynthRecording` (or a `CalibratedChannel` + fs + polarity). **The detector must recover synthetic ground truth at recall > 0.95 AND precision > 0.95** (and dwell within 3/fs, frac_depth within 0.02) on clean, enhancement, and drift cases — this is the anchor.
- `tests/analysis/test_filters.py` (44 tests): `_design_sos` import path → `nanoscope.analysis.filters._design_sos`; Bessel `norm='mag'` gives `|H(fc)|=1/√2` and scipy's `'phase'` is proven wrong; rise-time / zero-phase / MAD / notch / PSD pins. Swap `make_rec`→kernel calls.
- `tests/analysis/test_baseline.py` (17): construct data arrays + explicit polarity; drift tracking, short-window intrusion hazard, two-pass at 30%, 60%-occupancy percentile/asls, Pchip no-overshoot, 5M < 2 s.
- `tests/analysis/test_characterize.py` (25) + `test_events.py`: median-vs-max science tests, level counting, rise/fall NaN, SCHEMA exactness, vectorized==naive-loop, pooling guardrails.
- `tests/analysis/test_detect.py` (536 lines): recall/precision 1.0, enhancement, hysteresis-beats-single-threshold, polarity verdicts, exclusions, merge/min/max dwell, 5M < 1 s.
- `tests/analysis/test_stats.py` (28): naive-uncensored-is-biased-long (bias == exactly `t_min`), closed-form self-check, mixtures + BIC, efficiency correction, all fail-loud. **Runs on numpy 1.26 via the D2 shim — add a test that `_trapz` is bound.**
- `tests/analysis/test_truth.py` (555 lines): install the reference detector only if absent (target now has one — point it at `detect_events`); AAFT amplitude/PSD preservation + event destruction; null-control event-removal regression (fdr < 0.1); sweep plateau; recovery efficiency in [0,1].
- `tests/plot/test_figures.py` (40+, Agg backend): **port the log-bin-Jacobian exact-expectation test AND its negative control unchanged** (the scientific guardrail); accepted-only; envelope spike survival; save_figure PDF+PNG+CSV.
- `tests/gui/test_panels.py` (**needs pytest-qt**, offscreen): `k_out<k_in` after adversarial edits; None off-sentinels; rise-time readout monotonic in cutoff; EventTable curation writes back; SweepDialog enable/disable.
- `tests/gui/test_overlays.py`: `render_spans` produces **one** `SpanOverlayItem` for N events (assert not N `LinearRegionItem`s); index→x mapping correct.
- `tests/gui/test_analysis_integration.py`: end-to-end — synth `TraceDocument` → `Session` → `DetectSchmitt` via `AnalysisWorker` → `session.add_result` emits `resultsChanged` → `trace_view.add_overlay` paints spans; assert `requires_time_base` disables the module on a rate-`None` doc.
`tests/conftest.py`: `qapp` fixture (`create_app([])`), calls `analysis.register_all()` + `overlays.register_all_renderers()`, and builds a `CalibratedChannel`/`SimpleAnalysisContext` from a synth array.

## 9. Build phases & dependencies

- **Phase 0 — Scaffolding.** Create `analysis/` + `plot/` packages; `_compat.py` (D2 shim); `pyproject` deps (+scipy, +matplotlib, +pyarrow) and dev (+pytest-qt); `tests/conftest.py`; port `plot/style.py` verbatim. *No dependencies.*
- **Phase 1 — Pure DSP leaves.** `events.py` (verbatim), `filters.py` (kernels), `stats.py` (+shim). Port `test_filters.py`, `test_stats.py`, `test_events.py`. *Depends on Phase 0.*
- **Phase 2 — Baseline + characterize.** `baseline.py` (+`derive_polarity`), `characterize.py`. Port `test_baseline.py`, `test_characterize.py`. *Depends on Phase 1 (events, filters, noise_rms).*
- **Phase 3 — Detector engine.** `detect.py` (`detect_events` + `detect_polarity`). Port the `make_trace` backbone + `test_detect.py`; **gate the phase on recall/precision > 0.95.** *Depends on Phases 1–2.*
- **Phase 4 — Truth/validation.** `truth.py` (+`SynthRecording`). Port `test_truth.py`. *Depends on Phase 3 (lazy detect import).*
- **Phase 5 — Seam adapters.** `context.py`, `modules.py` (`DetectSchmitt` + `register_module`), `Session.add_result`. Unit-test module registration + `AnalysisResult` shape. *Depends on Phase 3.*
- **Phase 6 — Overlay + worker.** `gui/overlays.py` (renderers + `register_all_renderers`), `gui/analysis_worker.py`. Port `test_overlays.py`. *Depends on Phase 5.*
- **Phase 7 — Panels + AnalysisPanel + main_window wiring.** Port `gui/panels.py`; build `analysis_panel.py`; edit `main_window.py`/`app.py`. Port `test_panels.py`; write `test_analysis_integration.py`. *Depends on Phases 5–6.*
- **Phase 8 — Figures + export.** `plot/figures.py`; "Export figure…" action. Port `test_figures.py` (Jacobian + negative control). *Depends on Phases 1–3 (stats/events) + Phase 7 for the action.*

Critical path: 0 → 1 → 2 → 3 → 5 → 6 → 7. Phases 4 and 8 are parallelizable after their deps.

## 10. Risks & decisions requiring a call

- **R1 (blocker, resolved by D2):** numpy 1.26.4 has no `np.trapezoid`. Shim required or `test_stats.py`/`stats.py` crash on import. Do not upgrade numpy (risks the neo reader stack).
- **R2 — calibration_fingerprint injection.** The module has no Session; the controller must inject `_calibration_fingerprint` (and `pore_id`) into `ctx.params`. Reserve the `_`-prefix convention and strip it before calling `detect_events`. (Spec'd in §4.)
- **R3 — layer time-offset-aware overlays. CLOSED.** The overlay-renderer signature `(plot, session, doc_id, hint)` carried no channel/offset, so overlays mapped index→x at the doc's native t0 and every span/marker on an aligned layer sat `time_offset_s` from the event it annotated. Closed the way this note anticipated: `add_overlay` knows the channel, so it looks the layer's offset up (`TraceView.layer_time_offset`) and passes it to the renderer as a keyword-only `time_offset_s`, defaulted to 0.0 — the native axis, i.e. the unchanged answer for an unaligned layer and for any renderer written against the original four-argument signature. Fixed alongside C8, which was the same offset-0 mistake in `AnalysisPanel`'s visible-range and event-zoom mappings.
- **R4 — filtered-trace representation (D4).** v2 uses a managed preview curve (`TraceView.set_filter_preview`), not a derived document. If the team wants filtered traces to be first-class comparison layers (their own pane, calibration, overlays), that needs a `DerivedTraceDocument` (synthetic unique `doc_id`, parent-close cascade) — larger, defer to v3.
- **R5 — fits/truth are not AnalysisModules (D3).** Confirm the team is comfortable that only detection goes through the `AnalysisModule`/worker path; fits + null/sweep/recovery are controller-invoked functions storing derived results. (Recommended; keeps the seam channel-first.)
- **R6 — matplotlib vs pyqtgraph (D5).** Two rendering stacks coexist by design (live = pyqtgraph, export = matplotlib). No unification; do not try to render publication figures in pyqtgraph.
- **R7 — sklearn/hmmlearn are installed but unused.** The stats EM and characterize level-counting are hand-rolled numpy on purpose. Do not "simplify" them onto sklearn/hmmlearn — it would change the pinned numbers the tests encode.
- **R8 — pytest-qt / matplotlib / scipy / pyarrow not in `pyproject`.** Installed in the Anaconda env but undeclared. Add them (runtime: scipy, matplotlib, pyarrow; dev: pytest-qt) so a fresh checkout is reproducible.
- **R9 — units are per-channel and may be blank or nA.** Do not hard-code "pA" in any label; thread `CalibratedChannel.units` into provenance/figures/depth reporting. SCHEMA column *names* keep `_pA` as the frozen contract; values are in the channel's units.
- **R10 — `requires_time_base` on legacy TDMS.** `example.tdms` has `sample_rate_hz = None`. The DetectionPanel and the "Run detection" action must disable honestly (no fabricated dwell). Covered by the integration test.

**Files referenced (all absolute):** SOURCE DSP `/Users/abrahamcampos/nanoscope/src/nanoscope/core/{filters,baseline,detect,events,characterize,stats,truth}.py`, `/Users/abrahamcampos/nanoscope/src/nanoscope/plot/{figures,style}.py`, `/Users/abrahamcampos/nanoscope/src/nanoscope/gui/panels.py`, tests `/Users/abrahamcampos/nanoscope/tests/test_*.py`. TARGET seams `/Users/abrahamcampos/Desktop/nanoscope/nanoscope/core/{analysis,document,session,types,calibration,layer}.py`, `/Users/abrahamcampos/Desktop/nanoscope/nanoscope/gui/{trace_view,workers,main_window,calibration_panel,app}.py`, config `/Users/abrahamcampos/Desktop/nanoscope/pyproject.toml`.