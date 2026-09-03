"""The runtime-loaded Keras detector as an :class:`AnalysisModule` (Phase 2).

:class:`MLDetect` is the adapter that plugs the pure DSP core in
:mod:`nanoscope.analysis.ml_detect` into the v2 analysis seam, so an ML run
produces the **exact same** :class:`~nanoscope.analysis.events.EventSet` the
built-in Schmitt detector does and flows into the rest of the pipeline (event
table, span overlays, event inspector, plots, exports) unchanged.

It is channel-in / :class:`AnalysisResult`-out, which is the ``run(ctx)``
contract, so it runs on the shared
:class:`~nanoscope.gui.analysis_worker.AnalysisController` off the Qt main thread
with progress + cancel -- inference is slow (~0.24x realtime on CPU), so this is
non-negotiable.

What the module does, given ``ctx``:

  1. Pull the **calibrated current channel** and slice it to the requested sample
     range (``range_start``/``range_stop``; default whole recording). The range
     keeps the slow inference to what the user actually wants.
  2. Load the ``.keras`` model by PATH (``ctx.params['model_path']``) via
     :func:`ml_detect.load_keras_model` -- cached by ``(path, mtime)``, and it is
     loaded HERE (off the main thread) so the app never imports TensorFlow on the
     GUI thread. The panel only ever holds a path string.
  3. Run :func:`ml_detect.ml_detect_regions` (preprocess -> [RAW: global normalize
     | FEATURE: per-window multiscale features] -> windowed predict -> decode) to
     get INCLUSIVE ``[start, end]`` regions in slice-local coordinates. The RAW
     ``(None, W, 1)`` vs FEATURE ``(None, W, 12)`` path is auto-selected from
     ``model.channels`` inside ``ml_detect`` -- this adapter is channel-count
     agnostic.

     The preprocessing is the ML detector's OWN chain
     (:mod:`nanoscope.analysis.ml_preprocess`), chosen by preset from
     ``ctx.params['ml_preprocess']`` / ``['ml_decode']`` -- **never** the app's
     general :func:`~nanoscope.analysis.modules.preprocess_for_detection` Preprocess-tab
     chain. A learned model only behaves as measured when it is fed the pipeline it
     was trained on. Absent params keep the pre-existing ``"apo_raw_global"``
     (formerly ``"apo_legacy"``) behaviour bit-for-bit.

     When the normalization scope is ``"recording"`` (the default) and only a
     sub-range is being inferred, the WHOLE channel is preprocessed once here so
     ``max|x|``, the normalization MEAN (apo), the masked-median MAD and the SavGol
     envelope all see the same span training did; the analysed slice is then cut out
     of that.
  4. Characterise those regions on the **raw calibrated** slice (real pA, real
     open-pore baseline) with the SAME :func:`characterize` the Schmitt path uses,
     then OFFSET the sample-index columns back to full-file coordinates so the
     spans/markers/zoom land on the right samples.
  5. Wrap it as an :class:`AnalysisResult` (``module_id='detect.keras'``) with a
     SPAN overlay and the provenance (model file, thresholds, window, range).

The model is loaded lazily and CPU is forced inside ``ml_detect`` -- merely
importing this module never touches TensorFlow (it only probes for it), so the
app starts fast and degrades gracefully when TF is absent.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np

from ..core.analysis import AnalysisResult, OverlayHint, OverlayKind
from ..core.identity import doc_id_for_meta
from ..core.types import ChannelKind
from . import ml_detect
from .baseline import characterisation_baseline
from .characterize import characterize
from .detect import detect_polarity
from .events import (
    EventSet,
    clipped_boundary_warning,
    empty_df,
    n_clipped_boundary,
    span_provenance,
    stamp_original_bounds,
    stamp_source_status,
)
from .filters import noise_rms
from .ml_preprocess import (
    DECODE_PRESETS,
    PREPROCESS_PRESETS,
    MLDecodeConfig,
    MLPreprocessConfig,
    apply_ml_preprocess,
    model_input_normalization,
    normalization_center,
    normalization_scale,
    resolve_preset_name,
)

__all__ = ["MLDetect", "register_ml"]

# Identity calibration -> no gain/offset. The controller overrides this with the
# real ``session.calibration(doc_id, ch).fingerprint()`` so an ML result carries the
# same staleness stamp a Schmitt result does; the default keeps the module runnable
# off a Session (tests build a bare SimpleAnalysisContext).
_DEFAULT_FINGERPRINT: tuple[float, float] = (1.0, 0.0)

# The k (in noise RMS) used only to MEASURE polarity for characterisation. The ML
# model itself found the regions; this just tells characterize which way dev points
# so depths come out as positive magnitudes (the events.py sign convention).
_POLARITY_K = 5.0

# THERE IS NO BASELINE WINDOW CONSTANT IN THIS MODULE, AND THERE MUST NOT BE ONE.
#
# One lived here: ``_CHAR_BASELINE_WINDOW_S = 0.20``, widened from
# ``estimate_baseline``'s silent 10 ms default to fix a real defect (a 6 ms
# translocation fills 63% of a 10 ms block, so the block median sat ON the blockade
# and reported 7.7-29.8 pA for an event whose true depth is 67.1 pA). The fix was
# right about the recording and wrong about the scope: the Schmitt detector kept
# the 10 ms default, so the app reported TWO DIFFERENT DEPTHS FOR ONE EVENT
# depending on which detector found it -- population median 23.23 pA (Schmitt)
# against 111.31 pA (ML) on the owner's -600 mV file.
#
# The window now arrives as a ``BaselineConfig`` from
# ``baseline.baseline_config_from_params``, the same object the Schmitt path,
# ``truth`` and ``characterize.recharacterize_span`` receive, so the two detectors
# stamp the same ``baseline_fingerprint`` and their depths are comparable. A
# constant re-added here would be caught by the AST guard in
# ``tests/analysis/test_baseline_seam.py``.
#
# What is NOT delegated is the SCOPE: the baseline is measured over the WHOLE
# CHANNEL, never the analysed slice. That was the other half of the same defect --
# the 10 ms block grid anchors at ``range_start``, so sliding the user's range by
# 2 ms moved the reported depth 7.7 -> 29.8 pA -- and it matters more, not less,
# for the masked estimator, whose mask threshold is a GLOBAL MAD.


def _fingerprint(params: Mapping[str, Any]) -> tuple[float, float]:
    raw = params.get("_calibration_fingerprint", _DEFAULT_FINGERPRINT)
    vals = tuple(float(v) for v in raw)
    if len(vals) != 2:
        raise ValueError(
            f"_calibration_fingerprint must be a (gain, offset) pair, got {raw!r}"
        )
    return vals  # type: ignore[return-value]


def _resolve_range(params: Mapping[str, Any], n: int) -> tuple[int, int]:
    """Clamp the requested ``[range_start, range_stop)`` to ``[0, n]``.

    The panel resolves the user's "Visible view / Whole recording" choice into
    concrete sample indices before the run; this only validates them. Anything
    missing or degenerate falls back to the whole recording (a safe superset).
    """
    i0 = params.get("range_start")
    i1 = params.get("range_stop")
    if i0 is None or i1 is None:
        return 0, n
    i0 = max(0, min(int(i0), n))
    i1 = max(0, min(int(i1), n))
    if i1 <= i0:
        return 0, n
    return i0, i1


def _resolve_configs(
    params: Mapping[str, Any], enter: float, exit_thr: float
) -> tuple[MLPreprocessConfig, MLDecodeConfig | None]:
    """Pull the ML detector's OWN preprocessing/decode configs out of ``params``.

    The panel sends them as plain dicts (JSON-safe, so a stored result stays
    readable). Absent -> the pre-existing behaviour: the ``"apo_raw_global"``
    preprocessing chain (bit-for-bit :func:`ml_detect.detrend_for_model`) and
    ``decode=None``, which routes to the legacy :func:`ml_detect.decode_regions`
    driven by ``enter``/``exit``. That keeps every existing caller -- and the
    stored params of every result made before this existed -- numerically
    unchanged.

    When a decode config IS given, the panel's live ``enter``/``exit`` spins win
    over whatever the preset carried, because those two are the user's headline
    knobs and have always been top-level params.

    They only win when they were actually SENT, though. ``enter``/``exit`` are
    resolved by the caller with :data:`ml_detect.DEFAULT_ENTER` / ``DEFAULT_EXIT``
    (0.5 / 0.3, the legacy apo values) as fallbacks, so overriding
    unconditionally silently replaced the exit threshold of every notebook preset
    (0.15 / 0.15 / 0.20) with 0.3 for any caller that carried ``ml_decode`` but no
    top-level thresholds -- a stored result re-run from its own params, or any
    programmatic caller. Measured on ferritin_data1 that moved the median dwell by
    ~2.7 %. Only keys PRESENT in ``params`` are applied.
    """
    pre = MLPreprocessConfig.from_dict(params.get("ml_preprocess"))
    if pre is None:
        pre = PREPROCESS_PRESETS["apo_raw_global"]
    dec = MLDecodeConfig.from_dict(params.get("ml_decode"))
    if dec is not None:
        overrides: dict[str, float] = {}
        if params.get("enter") is not None:
            overrides["enter_threshold"] = float(enter)
        if params.get("exit") is not None:
            overrides["exit_threshold"] = float(exit_thr)
        if overrides:
            dec = dec.replace(**overrides)
    return pre, dec


class MLDetect:
    """Runtime-loaded Keras event detection as an :class:`AnalysisModule`.

    Mirrors :class:`~nanoscope.analysis.modules.DetectSchmitt`'s shape (same
    ``accepts_kinds`` / ``requires_time_base`` gate, same EventSet payload + SPAN
    overlay) so the GUI treats an ML detection exactly like a Schmitt one.
    """

    id = "detect.keras"
    label = "Detect events (Keras ML)"
    version = "1.0"
    accepts_kinds = (ChannelKind.CURRENT,)   # gate: current channels only
    requires_time_base = True                # dwell is undefined without fs

    def run(self, ctx: Any) -> AnalysisResult:
        ch = ctx.channel
        if ch.sample_rate_hz is None:
            raise RuntimeError(
                "ML detector requires a sample rate, but this channel has none "
                "(requires_time_base is True); the GUI should have disabled it."
            )
        fs = float(ch.sample_rate_hz)
        # WHICH LOAD, not which path — see the same stamp in `modules.DetectSchmitt.run`
        # and `core.identity`, which owns the only writer.
        doc_id = doc_id_for_meta(ctx.meta)
        channel = int(ch.info.index)
        fp = _fingerprint(ctx.params)
        # ONE resolution of the characterisation-baseline knobs, through the SAME resolver
        # the two threshold detectors use (`modules._resolve_run_baseline`, S7-DET1): the
        # form's source is judged for THIS recording, so a batch pass over recordings the
        # windows were not proposed for stamps ``'batch'`` (and the copied-windows
        # advisory fires) rather than claiming "proposed" for a recording nobody measured;
        # and the params RECORDED on the result carry that resolved source and drop the
        # per-load scope key, so a stored result re-runs to the same stamp and never
        # writes a ``doc_id`` into a project file. This module used to record
        # ``dict(ctx.params)`` untouched and resolve the config alone, which is exactly
        # the gap agent E's report named for the day the ML detector batch-runs.
        # Imported at call time: `modules` imports THIS module lazily (register_all), so
        # a module-level import here would be the circular half of that arrangement.
        from .modules import _resolve_run_baseline

        base_cfg, base_source, params = _resolve_run_baseline(ctx.params, (doc_id, channel))

        model_path = params.get("model_path")
        if not model_path:
            raise RuntimeError(
                "no ML model was provided (params['model_path'] is empty); load a "
                ".keras model before running ML detection."
            )
        enter = float(params.get("enter", ml_detect.DEFAULT_ENTER))
        exit_thr = float(params.get("exit", ml_detect.DEFAULT_EXIT))

        pre_cfg, dec_cfg = _resolve_configs(params, enter, exit_thr)
        if dec_cfg is not None:
            # The decode config is now the single source of truth for the two
            # thresholds. Adopt them here so the provenance this adapter stamps
            # (``_provenance`` writes enter/exit last, over the DSP core's) reports
            # what actually ran rather than the params fallbacks.
            enter = float(dec_cfg.enter_threshold)
            exit_thr = float(dec_cfg.exit_threshold)

        n_total = int(ch.data.size)
        i0, i1 = _resolve_range(params, n_total)
        sig = np.ascontiguousarray(ch.data[i0:i1], dtype=np.float32)

        # THE MODEL IS LOADED FIRST, BEFORE ANY SIGNAL IS TOUCHED.
        # Every way this run can be refused for a reason the user can act on -- the file
        # is gone, it is not a detector, its window or channel count is wrong, its output
        # is per-window rather than per-sample -- is decided here, by `load_keras_model`.
        # It used to run AFTER the whole-recording notch + SavGol pass, so a mistyped or
        # moved model path charged the user the entire preprocessing of a 3 M-sample
        # recording before saying "model file not found". Loading is cached by
        # (path, mtime), so on the runs that DO proceed this costs nothing the next line
        # would not have cost anyway.
        # Loaded OFF the main thread; cached by (path, mtime). Forces CPU inside.
        ctx.report_progress(0.0, "loading model")
        model = ml_detect.load_keras_model(model_path)

        # -- NORMALIZATION SCOPE ------------------------------------------------
        # Training normalized by max|x| over the WHOLE recording (student_predict;
        # the apo notebook's process_trace likewise passes the full raw_trace).
        # Measuring it on the analysed slice instead makes the model's input scale
        # depend on the user's zoom -- silently, since the range combo used to
        # default to "Visible view". With scope="recording" (the default) the whole
        # channel is preprocessed once, the scale is taken from THAT, and the slice
        # is cut out of it. That also gives masked-median its global MAD and SavGol
        # its full-length envelope, exactly as dataLoader.ipynb intends.
        #
        # The apo chain's mean is a SECOND whole-recording statistic and must come
        # from the same span, in the same order the notebook computes it: the mean
        # first, then max|x - mean| about it. Measuring either on the slice would
        # reintroduce the zoom dependence the scope exists to remove.
        detrended = None
        gate_signal = None
        norm_scale = None
        norm_center = None
        sub_range = (i0, i1) != (0, n_total)
        norm_scope = "range"

        # -- WHAT THE AMPLITUDE GATES MEASURE ON -------------------------------
        # ``min_depth_sigma`` / ``min_depth_pa`` / ``min_fracblock`` are depth
        # thresholds in PICOAMPS (or in dI/I0, which is picoamps over picoamps).
        # On the robust-sigma path below, the trace handed to the DSP core as
        # ``detrended`` is not picoamps at all -- it is the finished model input:
        # re-centred on a twice-trimmed median, divided by a robust sigma, and
        # CLIPPED at +-clip_sigma. Reading "at least 10 pA" against that trace
        # would compare a picoamp threshold to a unitless, saturated number and
        # keep or drop events essentially at random. So the PRE-NORMALIZATION
        # detrended trace is kept and passed separately.
        #
        # It is only materialised when a gate is actually on -- for every preset
        # that ships gates off, this costs one boolean and no memory.
        gates_on = dec_cfg is not None and (
            float(dec_cfg.min_depth_sigma) > 0
            or float(dec_cfg.min_depth_pa) > 0
            or float(dec_cfg.min_fracblock) > 0
        )

        # -- ROBUST-SIGMA NORMALIZATION IS APPLIED HERE, NOT DOWNSTREAM ---------
        # `ml_detect.ml_detect_regions` normalizes with `normalize_global`, i.e. one
        # affine map ``(x - center)/scale``. `filters.robust_normalize` is NOT affine --
        # it re-estimates a twice-trimmed median/MAD and then CLIPS -- so there is no
        # (center, scale) pair that expresses it and no way to hand the DSP core the
        # recording's statistics for a slice. The only faithful arrangement is to
        # normalize the whole span HERE and hand the core a finished model input,
        # telling it not to normalize again.
        #
        # ``run_cfg`` is what the core is given; ``pre_cfg`` stays the truth and is what
        # provenance, the deviation report and the preset comparison all use. The
        # ``normalize=False`` on run_cfg is not a fib to the core -- from where it
        # stands, it really does apply no normalization. The keys the core would
        # otherwise stamp about a normalization it did not perform are corrected below.
        #
        # It only applies to the RAW contract: a FEATURE model (channels == 12)
        # standardizes every column inside `create_multiscale_features_A`, so the core
        # skips global normalization entirely and this must too.
        run_cfg = pre_cfg
        # The CONFIG is tested first and the model second, deliberately: for every
        # pre-existing preset this short-circuits on the first clause and nothing about
        # the model is read at all, so the pre-robust path is byte-for-byte the code it
        # always was. ``channels`` is read through getattr for the same reason -- a
        # stubbed model wrapper that never needed the attribute must not start needing
        # it, and 1 (the RAW contract) is the only value a model without one can mean.
        robust_input = (
            pre_cfg.normalize_method == "robust_sigma"
            and pre_cfg.normalize
            and int(getattr(model, "channels", 1)) == 1
        )
        if robust_input:
            ctx.report_progress(0.0, "preprocessing (whole recording)")
            if sub_range and pre_cfg.normalize_scope == "recording":
                # Whole-recording scope: preprocess and normalize the WHOLE channel,
                # then cut the slice out of the finished model input. Slicing after the
                # normalization is what "recording scope" means for a kernel that
                # cannot be handed a precomputed scale.
                full = np.ascontiguousarray(ch.data, dtype=np.float32)
                full_detr = apply_ml_preprocess(full, fs, pre_cfg)
                span = model_input_normalization(full_detr, pre_cfg)
                detrended = np.ascontiguousarray(span[i0:i1])
                if gates_on:
                    gate_signal = np.ascontiguousarray(full_detr[i0:i1])
                norm_scope = "recording"
            else:
                slice_detr = apply_ml_preprocess(sig, fs, pre_cfg)
                detrended = model_input_normalization(slice_detr, pre_cfg)
                if gates_on:
                    gate_signal = np.ascontiguousarray(slice_detr)
                norm_scope = "recording" if not sub_range else "range"
            run_cfg = pre_cfg.replace(normalize=False)
        elif not sub_range:
            norm_scope = "recording"        # the slice IS the recording
        elif pre_cfg.normalize_scope == "recording":
            ctx.report_progress(0.0, "preprocessing (whole recording)")
            full = np.ascontiguousarray(ch.data, dtype=np.float32)
            full_detr = apply_ml_preprocess(full, fs, pre_cfg)
            norm_center = (
                normalization_center(full_detr) if pre_cfg.normalize_center else 0.0
            )
            norm_scale = normalization_scale(full_detr, norm_center)
            detrended = np.ascontiguousarray(full_detr[i0:i1])
            norm_scope = "recording"

        # preprocess -> (RAW: normalize | FEATURE: per-window features) -> windowed
        # predict -> decode. Slow; reports 0..1 and aborts cleanly on cancel.
        # ml_detect_regions runs the ML detector's OWN preprocessing (never the
        # general Preprocess-tab chain) and auto-selects the RAW/FEATURE path from
        # model.channels; regions come back in slice-local samples.
        try:
            regions, _prob, ml_prov = ml_detect.ml_detect_regions(
                sig,
                fs,
                model,
                enter=enter,
                exit=exit_thr,
                preprocess=run_cfg,
                decode=dec_cfg,
                detrended=detrended,
                gate_signal=gate_signal,
                norm_scale=norm_scale,
                norm_center=norm_center,
                norm_scope=norm_scope,
                progress=ctx.report_progress,
                cancel=ctx.cancelled,
            )
        except ml_detect.MLCancelled:
            # Translate the DSP core's cancel into the worker's cancel protocol
            # WITHOUT importing gui: the worker sees ctx.cancelled() is True and
            # drops this (empty) result, emitting cancelled() rather than failed().
            return self._cancelled_result(doc_id, channel, fp, params, fs, i0, i1)

        # A deviation from the preset the model was trained under is a CORRECTNESS
        # risk, not a preference, so it is recorded against the named preset rather
        # than being flattened into "custom". Derived from the configs themselves,
        # never from what the UI claimed. ``resolve_preset_name`` is what keeps a
        # result stored under a superseded key (``apo_legacy``) reporting deviations
        # against the preset it actually means instead of silently reporting none.
        ml_prov = dict(ml_prov)
        if robust_input:
            # PUT THE TRUTH BACK. The core was handed ``run_cfg`` (normalize=False) and
            # honestly reported that it normalized nothing; what actually ran is
            # ``pre_cfg``, and provenance is the only record of it. Overwriting these
            # five keys here -- rather than letting the core's answer stand -- is what
            # keeps a stored result re-runnable and keeps `deviations_from` comparing
            # the chain the user chose against the preset, not a doctored copy of it.
            ml_prov["ml_preprocess"] = pre_cfg.to_dict()
            ml_prov["ml_preprocess_desc"] = pre_cfg.describe()
            ml_prov["preprocess"] = pre_cfg.describe()
            ml_prov["normalized"] = True
            ml_prov["norm_scope"] = norm_scope
            # There is no single scale: the kernel estimates one internally and clips,
            # so a number here would be a fabrication. The method and its bound are the
            # reproducible facts, and they are stamped as such.
            ml_prov["norm_scale"] = None
            ml_prov["norm_center"] = None
            ml_prov["ml_normalize_method"] = str(pre_cfg.normalize_method)
            ml_prov["ml_normalize_clip_sigma"] = float(pre_cfg.normalize_clip_sigma)
            ml_prov["ml_preprocess_scope"] = norm_scope
        base_pre = PREPROCESS_PRESETS.get(resolve_preset_name(pre_cfg.preset) or "")
        pre_dev = pre_cfg.deviations_from(base_pre) if base_pre is not None else ()
        base_dec = (
            DECODE_PRESETS.get(resolve_preset_name(dec_cfg.preset) or "")
            if dec_cfg is not None else None
        )
        dec_dev = dec_cfg.deviations_from(base_dec) if base_dec is not None else ()
        ml_prov["ml_preprocess_deviations"] = list(pre_dev)
        ml_prov["ml_decode_deviations"] = list(dec_dev)
        ml_prov["ml_preset_customized"] = bool(pre_dev or dec_dev)

        # Characterise on the RAW calibrated slice (real pA against the real
        # open-pore baseline) -- NOT the model's internal detrended signal, whose
        # only purpose was to feed the network. Same characterize the Schmitt path
        # uses, so ML events are indistinguishable downstream.
        #
        # THE BASELINE IS MEASURED ON THE WHOLE RECORDING, NOT THE SLICE.
        # It used to be ``estimate_baseline(sig_slice, ...)`` at the default
        # ``window_s=0.01``, which made the reported DEPTH depend on where the user
        # had zoomed -- and "Visible view" was the default range. Two independent
        # failures, both measured on the owner's -600 mV recording against one real
        # 6.26 ms event whose true depth is 67.1 pA:
        #
        #   * the 10 ms block grid anchors at ``range_start``, so sliding the range
        #     by 100 samples (2 ms) moved the depth 7.7 -> 29.8 pA (3.9x) with a
        #     period of exactly 500 samples = one block; changing only the range
        #     WIDTH gave 6.5 -> 41.6 pA. Every run reported verdict="measured",
        #     warnings=().
        #   * a 313-sample event fills 63% of a 500-sample block, so the block median
        #     IS the blockade level and the baseline dives into the event: 2-9x low
        #     in absolute terms even before the zoom dependence.
        #
        # Estimating over the full channel fixes the first (the grid is anchored to
        # the recording, wherever the user zoomed); the second is fixed by the
        # event-masked estimator itself, which is what the shared `BaselineConfig`
        # now selects for BOTH detectors instead of a 200 ms window here and a 10 ms
        # one in `detect`. Both are stamped into provenance. NOTE this is the
        # CHARACTERISATION baseline only -- the ML PREPROCESSING chain above is
        # verified bit-exact against the owner's notebooks and is not touched.
        # ``base_cfg`` / ``base_source`` were resolved at the top of ``run`` (one resolver,
        # shared with the threshold detectors); nothing is re-read from the form here.
        full = np.asarray(ch.data, dtype=np.float64)
        # The DOOR is handed the channel's own array, not this call's float64 copy: the door
        # upcasts identically, and its whole-recording memo is keyed on the identity of what it
        # was given. Handing it `full` memoised an object that dies with this call. Same rule as
        # `detect.detect_events` — no caller converts before the door.
        base_result = characterisation_baseline(
            np.asarray(ch.data), fs, config=base_cfg, polarity=None, param_source=base_source,
        )
        base_full = base_result.baseline
        cur = full[i0:i1]
        base = base_full[i0:i1]
        # The SNR denominator is the same kind of statistic and had the same
        # dependence; the Schmitt path measures it over its whole trace, so this one
        # does too. It is also what a later merge / mask-resize of an ML row reads
        # back out of provenance as its own noise scale.
        noise = float(noise_rms(full - base_full))
        measured_polarity, evidence = detect_polarity(cur, base, k=_POLARITY_K)
        polarity = measured_polarity
        polarity_warning = None
        if evidence.get("verdict") == "undetermined":
            polarity_warning = (
                "polarity is undetermined (no coherent excursions above threshold); "
                f"proceeding with polarity={polarity:+d}."
            )

        recording = ctx.meta.path.name
        pore_id = str(params.get("pore_id", ""))

        if regions:
            starts_local = np.asarray([s for s, _ in regions], dtype=np.int64)
            # ml_detect regions are INCLUSIVE [start, end]; characterize wants
            # one-past-the-last exclusive ends.
            ends_local = np.asarray([e + 1 for _, e in regions], dtype=np.int64)
            df = characterize(
                cur,
                base,
                starts_local,
                ends_local,
                fs,
                polarity,
                noise_rms=noise,
                filter_rise_s=None,   # the ML path applies no low-pass; nothing is filter-limited
                # The analysed span's edges, in the slice's own coordinates. An ML run
                # on a sub-range admits events cut off by the ZOOM RECTANGLE, whose
                # dwell is a lower bound and whose depth is the median of whatever
                # fraction of the event happened to be inside it. The Schmitt path has
                # rejected those since the boundary-clip rule was written; this path
                # never saw it, because the rule lived in ``detect_events`` instead of
                # in ``characterize``. It lives in ``characterize`` now, so this is the
                # identical determination rather than a second copy of it.
                clip_bounds=(0, int(cur.size)),
                pore_id=pore_id,
                recording=recording,
                units=ch.units,
            )
            # Offset the sample-index / time columns back to FULL-FILE coordinates
            # when a sub-range was analysed, so overlays + zoom land on the right samples.
            if i0:
                df["start_idx"] = df["start_idx"].to_numpy(np.int64) + i0
                df["end_idx"] = df["end_idx"].to_numpy(np.int64) + i0
                df["t_start_s"] = df["t_start_s"].to_numpy(np.float64) + (i0 / fs)
        else:
            df = empty_df()

        n_clipped = n_clipped_boundary(df)
        prov = self._provenance(
            ml_prov, model, model_path, params, fs, n_total, i0, i1,
            enter, exit_thr, noise, base_result, polarity, measured_polarity,
            evidence, polarity_warning, recording, pore_id, ch.units, len(df),
            n_clipped,
        )
        # THE DETECTOR'S OWN BOUNDS, recorded before any hand edit can move them --
        # the same stamp, at the same moment and for the same reason as the Schmitt
        # path (`modules.DetectSchmitt.run`). Deliberately NOT done in
        # `_cancelled_result`: an empty frame has nothing to record, and stamping after
        # an edit would file the edit under the detector's name.
        #
        # ...and WHO FOUND IT, on the same line and for the same reason. ``self.id``
        # is ``detect.keras``, which `events.SOURCE_KIND_BY_MODULE` maps to
        # ``KIND_ML`` -- the owner's "conventional detector / ML detector" split,
        # made per ROW instead of per table. That split is the one thing this
        # detector could not previously say about a table it shares with a hand-added
        # event, and it is what "final analyses should normally use all accepted
        # events, regardless of how they were detected" needs in order to be a
        # CHOICE rather than an accident. ``review`` is ``unreviewed``: the model
        # emits candidates exactly as the threshold detectors do. ``accepted`` is
        # untouched, so no exported number moves.
        events = EventSet(stamp_source_status(stamp_original_bounds(df), self.id), prov)

        spans = OverlayHint(
            OverlayKind.SPAN,
            starts=df["start_idx"].to_numpy(dtype=np.int64),
            stops=df["end_idx"].to_numpy(dtype=np.int64),
            label="events",
        )

        scalars: dict[str, float] = {
            "n_events": float(len(df)),
            "noise_rms": float(noise),
            "median_baseline": float(np.median(base)),
        }
        try:
            scalars["rate_hz"] = float(events.rate_hz())
        except ValueError:
            scalars["rate_hz"] = float("nan")

        warnings: list[str] = []
        warnings.extend(base_result.warnings)
        if polarity_warning:
            warnings.append(polarity_warning)
        if n_clipped:
            # The SAME sentence the Schmitt path prints, from the one place it is
            # written (``events``); ``modules`` cannot be imported from here, and a
            # second copy of the text is how the two detectors start disagreeing
            # about what a clipped event is.
            warnings.append(clipped_boundary_warning(n_clipped, len(df)))

        ctx.report_progress(1.0, f"{len(df)} event(s)")
        return AnalysisResult(
            module_id=self.id,
            module_version=self.version,
            doc_id=doc_id,
            channel=channel,
            params=params,
            calibration_fingerprint=fp,
            payload=events,
            overlays=(spans,),
            scalars=scalars,
            warnings=tuple(warnings),
            baseline_fingerprint=prov.get("baseline_fingerprint"),
        )

    # -- helpers ----------------------------------------------------------------
    def _provenance(
        self, ml_prov, model, model_path, params, fs, n_total, i0, i1,
        enter, exit_thr, noise, base_result, polarity, measured_polarity,
        evidence, polarity_warning, recording, pore_id, units, n_events,
        n_clipped,
    ) -> dict[str, Any]:
        """Build the self-describing provenance for the result.

        Starts from the DSP core's provenance (preprocess/inference strings,
        window, decode params) and augments it with what the ADAPTER adds:
        characterisation baseline/noise/polarity, the analysed range, and
        ``analyzed_duration_s`` -- REQUIRED, since ``EventSet.rate_hz()`` divides
        the event count by it.
        """
        prov: dict[str, Any] = dict(ml_prov)
        prov.update(
            detector="ml_keras",
            model_path=str(model_path),
            model_file=os.path.basename(str(model_path)),
            model_window=int(model.window),
            model_input_shape=tuple(model.input_shape),
            model_output_shape=tuple(model.output_shape),
            enter=float(enter),
            exit=float(exit_thr),
            range_source=str(params.get("range_source", "whole")),
            range_start=int(i0),
            range_stop=int(i1),
            # THE SAME NUMBERS, under the keys the rest of the app reads. These two
            # existed as ``range_start``/``range_stop`` and nothing read them, so
            # every rebuild of an ML row judged ``clipped_by_boundary`` against the
            # whole recording and a row cut by the zoom rectangle quietly
            # un-rejected itself on the first merge. ``range_*`` stay for the
            # panel's staleness comparison and for results already on disk; the
            # boundary rule reads the canonical pair, written by the one writer
            # both detectors share.
            **span_provenance(i0, i1),
            fs=float(fs),
            n_samples=int(i1 - i0),
            n_samples_total=int(n_total),
            duration_s=n_total / fs,
            analyzed_duration_s=(i1 - i0) / fs,     # REQUIRED for EventSet.rate_hz()
            noise_rms_pA=float(noise),
            noise_method="MAD (filters.noise_rms)",
            # THE SAME ``baseline_*`` BLOCK THE SCHMITT PATH STAMPS, from the one
            # writer of these keys. Identical config in -> identical
            # ``baseline_fingerprint`` out, which is what makes an ML row and a
            # Schmitt row on this recording directly comparable. They were not: this
            # module measured at 200 ms and `detect` at 10 ms, and one event read
            # 67.69 pA or 32.04 pA depending on which detector found it.
            **base_result.provenance(),
            baseline_provided=False,
            baseline_warnings=list(base_result.warnings),
            # The characterisation baseline is a WHOLE-RECORDING estimate, so the
            # depths do not move when the user zooms. Stamped so a result says which
            # basis its depths were measured on. The WINDOW is no longer a constant
            # of this module -- it is whichever tracking window the shared config
            # names (``base_s`` for the two-pass estimator).
            characterize_baseline_scope="recording",
            characterize_baseline_window_s=float(
                base_result.config.tracking_window_s
            ),
            #: Regions still above threshold at the first or last analysed sample --
            #: cut off by the range the user chose, not by the pore. Flagged
            #: 'clipped_by_boundary' and NOT accepted, exactly as on the Schmitt path.
            n_clipped_boundary=int(n_clipped),
            polarity=int(polarity),
            polarity_source="measured",
            polarity_measured=int(measured_polarity),
            polarity_evidence=evidence,
            polarity_warning=polarity_warning,
            units=units,
            recording=recording,
            pore_id=pore_id,
            n_events=int(n_events),
            n_filter_limited=0,
        )
        return prov

    def _cancelled_result(
        self, doc_id, channel, fp, params, fs, i0, i1,
    ) -> AnalysisResult:
        """A minimal empty result returned when the run was cancelled mid-inference.

        The worker checks ``ctx.cancelled()`` after ``run()`` returns; because a
        cancel is the ONLY thing that raises MLCancelled, that flag is set, so the
        worker emits ``cancelled()`` and DROPS this object -- it is never stored."""
        events = EventSet(
            empty_df(),
            {
                "detector": "ml_keras",
                "cancelled": True,
                "analyzed_duration_s": max((i1 - i0) / fs, 1e-9),
            },
        )
        return AnalysisResult(
            module_id=self.id,
            module_version=self.version,
            doc_id=doc_id,
            channel=channel,
            params=dict(params),
            calibration_fingerprint=fp,
            payload=events,
            overlays=(),
            scalars={},
            warnings=(),
        )


def register_ml() -> None:
    """Register :class:`MLDetect` into ``ANALYSIS_REGISTRY`` (idempotent).

    Called from :func:`nanoscope.analysis.modules.register_all` so the one
    ``register_all`` at app/test start wires up both detectors.
    """
    from ..core.analysis import ANALYSIS_REGISTRY, register_module

    if MLDetect.id not in ANALYSIS_REGISTRY:
        register_module(MLDetect())
