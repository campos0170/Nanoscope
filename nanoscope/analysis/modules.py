"""AnalysisModule adapters that plug the ported DSP into the v2 seam.

Only **detection** is an :class:`~nanoscope.core.analysis.AnalysisModule`: it is
channel-in / channel-out, which is exactly the ``run(ctx)`` contract (PORT_PLAN
D3). The **fits** are not modules -- they consume a detection result's event
table, not a raw channel -- so ``fit_dwell`` / ``fit_depth`` are plain functions
the analysis controller calls, returning derived :class:`AnalysisResult`s
(``module_id`` ``fit.dwell`` / ``fit.depth``) that land in the same
:class:`~nanoscope.core.analysis.ResultStore` via ``Session.add_result``.

The module has no :class:`~nanoscope.core.session.Session`. The controller injects
``_calibration_fingerprint`` (and ``pore_id``) into ``ctx.params`` before the run;
``_``-prefixed keys are stripped before reaching :func:`detect_events` (PORT_PLAN
R2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd

from ..core.analysis import (
    ANALYSIS_REGISTRY,
    AnalysisResult,
    OverlayHint,
    OverlayKind,
    register_module,
)
from ..core.identity import doc_id_for_meta
from ..core.acquisition import stamp_acquisition_bandwidth, stamp_reference_availability
from ..core.types import ChannelKind
from .baseline import (
    BASELINE_PARAM_KEYS,
    BATCH_SCOPE_KEY,
    DEFAULT_DETREND_WINDOW_S,
    SAVGOL_DECIM,
    SAVGOL_POLY,
    AmplifiedWanderWarning,
    ConstantBaselineWarning,
    baseline_config_from_params,
    baseline_progress_scope,
    characterisation_baseline,
    collect_baseline_warnings,
    remove_baseline_wander,
    resolve_param_source,
)
from .cancel import check_cancelled
from .detect import detect_events, detect_polarity
from .detect_cusum import (
    DEFAULT_CUSUM_H,
    DEFAULT_CUSUM_K,
    DEFAULT_CUSUM_MERGE_GAP_S,
    DEFAULT_CUSUM_MIN_DWELL_S,
    detect_events_cusum,
)
from .characterize import FLAG_SATURATED, characterize, saturated_warning
from .events import (
    EventSet,
    clipped_boundary_warning,
    ensure_source_status,
    has_flag,
    reconcile_row_counts,
    stamp_original_bounds,
    stamp_source_status,
)
from .filters import (
    MODEL_INPUT_CLIP_SIGMA,
    MODEL_INPUT_MODE,
    highpass,
    lowpass,
    median_filter,
    noise_rms,
    normalize,
    notch,
    notch_frequencies,
    notch_settle_s,
    notch_unsettled_s,
    robust_normalize,
    running_mean,
    running_mean_samples,
    savgol,
    savgol_window_samples,
)
from .cusum import (
    CUSUM_VERSION,
    DEFAULT_CUSUM,
    SIGMA_ATTR as CUSUM_SIGMA_ATTR,
    TAU_ATTR as CUSUM_TAU_ATTR,
    CusumConfig,
    fit_event_levels,
    levels_summary,
)
from .stats import FitResult, fit_depth_gaussians, fit_dwell_exponential

__all__ = [
    "DetectSchmitt",
    "register_all",
    "reconcile_result",
    "fit_dwell",
    "fit_depth",
    "fit_levels",
    "preprocess_for_detection",
    "characterization_params",
    "preprocess_params_from_provenance",
    "detection_params_from_provenance",
    "DETECT_PARAM_KEYS",
    "detection_signal",
    "detection_signal_key",
    "any_preprocess_enabled",
    "smoothing_stage_labels",
    "smoothing_depth_retention",
    "smoothing_resolution_dwell_s",
    "notch_settling_from_provenance",
    "detrend_method_kwargs",
    "DETREND_PARAM_KEYS",
    "DETREND_METHOD_PARAMS",
    "FILTER_PARAM_KEYS",
    "DwellFitInputs",
    "DepthFitInputs",
    "dwell_fit_inputs",
    "depth_fit_inputs",
    "resolve_t_min",
    "depth_floor_from_provenance",
    "stamp_fit_provenance",
    "FIT_DETECTION_KEY",
    "FIT_CURRENT",
    "FIT_OTHER_TABLE",
    "FIT_CALIBRATION_MOVED",
    "FIT_EVENTS_CHANGED",
    "FIT_UNRECORDED",
    "FitVerdict",
    "judge_stored_fit",
    "stored_fit_problem",
    "NO_TMIN_WARNING",
]

# Identity calibration -> no gain/offset applied. The controller overrides this
# with session.calibration(doc_id, ch).fingerprint() so results carry the real
# staleness stamp; the default keeps the module runnable off a Session.
_DEFAULT_FINGERPRINT: tuple[float, float] = (1.0, 0.0)

# ctx.params keys that steer baseline-wander removal (detrending). The module
# consumes them itself via preprocess_for_detection BEFORE detect_events and must
# never forward them to the detector, which has no such parameters. Public so the
# GUI panels can build and strip the same set without re-spelling the keys.
# The trailing five are the selected METHOD's own shape parameters, forwarded to
# estimate_baseline as **method_kwargs. Only the ones belonging to the active method are
# forwarded (see _detrend_method_kwargs), so 'block_median' -- which has no shape
# parameters -- calls estimate_baseline exactly as it did before they existed.
DETREND_PARAM_KEYS = (
    "detrend_enabled", "detrend_method", "detrend_window_s",
    "detrend_percentile",
    "detrend_asls_lam", "detrend_asls_p", "detrend_asls_niter", "detrend_asls_decim",
    # Savitzky-Golay BASELINE (subtract) -- the Cu recipe's estimator, not the SavGol
    # SMOOTHER two stages later. Two opposite operations share the name: the smoother
    # RETURNS the polynomial fit, this SUBTRACTS it. The keys are prefixed
    # ``detrend_`` / ``savgol_`` respectively so a params dict can carry both at once.
    "detrend_sg_poly", "detrend_sg_decim",
    # The two masked/robust median estimators' own shape knobs. Without these the
    # methods ran at estimate_baseline's bare defaults (rough 0.5 / base 0.1 /
    # sigma 3.5 / dilate 1.0 ms) and none of the owner's three per-project settings
    # -- host-guest base 0.02 s dilate 0.2 ms, Zn/Cu base 0.2 s sigma 3.0 dilate
    # 2.0 ms -- could be expressed at all.
    "detrend_rough_s", "detrend_base_s", "detrend_mask_sigma", "detrend_dilate_ms",
    "detrend_decim",
)

# ctx.params keys that steer the OTHER pre-detection filters (notch / median / high-pass
# / running-mean / Savitzky-Golay). Like the detrend keys, preprocess_for_detection
# consumes them before detect_events, so they must never be forwarded to the detector.
# Public so the GUI panels can build and strip the same set without re-spelling the keys.
#
# NOTE the low-pass keys (``filter_cutoff_hz`` / ``filter_order`` / ``filter_kind``) are
# deliberately NOT here: the low-pass is applied INSIDE detect_events, so those three are
# real detect_events kwargs and must be forwarded, not consumed.
FILTER_PARAM_KEYS = (
    # ``notch_freqs`` is the EXPLICIT line list and, when present and non-empty, it
    # REPLACES (freq_hz, harmonics) outright -- those two are then not read at all.
    # It exists because the owner's own Welch censuses found line sets with GAPS that
    # a fundamental+harmonics series cannot express: host-guest is
    # [60,120,180,300,420,540] (240/360/480 quiet) and Zn/Cu is
    # [60,120,180,240,300,360,420,540,660] (480 absent). Reaching 540 with
    # ``harmonics=9`` would notch three quiet bands.
    "notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q", "notch_freqs",
    "median_enabled", "median_window_s",
    "hp_enabled", "hp_cutoff_hz", "hp_order", "hp_kind",
    "rmean_enabled", "rmean_window_s",
    "savgol_enabled", "savgol_window_s", "savgol_polyorder",
    # ``normalize_clip_sigma`` is read ONLY by the 'model_input' mode, where it is
    # both the clip bound and the divisor.
    "normalize_enabled", "normalize_mode", "normalize_clip_sigma",
)

# Defaults for the pre-detection filters, mirrored by the DetectionPanel widgets.
# These are only the fallbacks preprocess_for_detection uses when a key is absent;
# the panel always supplies its widget values in a real run.
DEFAULT_NOTCH_FREQ_HZ = 60.0
DEFAULT_NOTCH_HARMONICS = 3
DEFAULT_NOTCH_Q = 30.0
DEFAULT_HP_CUTOFF_HZ = 100.0
DEFAULT_HP_ORDER = 4
DEFAULT_HP_KIND = "butter"
DEFAULT_MEDIAN_WINDOW_S = 1e-4
DEFAULT_RMEAN_WINDOW_S = 5e-4
DEFAULT_SAVGOL_WINDOW_S = 1e-3
DEFAULT_SAVGOL_POLYORDER = 3

# Per-method shape parameters for the baseline-drift stage. Each entry maps a
# ``detrend_*`` param key to the ``estimate_baseline`` keyword it feeds, and the values
# below are EXACTLY estimate_baseline's own defaults -- so exposing them changes no
# result unless the user moves one. 'block_median' and 'constant' have no shape
# parameters and so have no entry: they call estimate_baseline with nothing extra,
# byte-for-byte as before.
#: The ``detrend_decim`` value that means "let the kernel derive the stride".
#:
#: ``estimate_baseline``'s own default for ``decim`` is ``None`` -- derive it from the
#: tracking window (`derive_decim`) -- and the DETREND_METHOD_PARAMS tuple mechanism
#: cannot express a None default: `detrend_method_kwargs` coerces every value with
#: ``int(value) if isinstance(default, int) else float(value)``, and ``float(None)``
#: raises. So 0 is the sentinel, translated back to None on the way out. It is the
#: DEFAULT, which is what keeps the two median estimators byte-identical to the
#: kernel until the user pins a stride -- the same "0 == the off/auto state" spelling
#: `max_dwell_s`, `merge_gap_s` and `filter_cutoff_hz` already use in this app.
_DECIM_AUTO = 0

DETREND_METHOD_PARAMS: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "percentile": (("detrend_percentile", "percentile", 80.0),),
    "asls": (
        ("detrend_asls_lam", "asls_lam", 6e4),
        ("detrend_asls_p", "asls_p", 1e-3),
        ("detrend_asls_niter", "asls_niter", 10),
        ("detrend_asls_decim", "asls_decim", 0.1),
    ),
    # SAVITZKY-GOLAY BASELINE. The defaults are IMPORTED, never re-spelled: their
    # int-ness is load-bearing, because `detrend_method_kwargs` coerces with
    # ``int(value) if isinstance(default, int) else float(value)`` and ``sg_decim``
    # must reach the kernel as an int. Verified: window_s 0.2 / poly 3 / decim 50
    # reproduces the ``baseline_pA`` column the owner wrote for all four cu_*
    # recordings BIT-IDENTICALLY (max |diff| exactly 0.0 pA over 13.9-20.2 M samples
    # each), once the nine-line notch has run first.
    "savgol": (
        ("detrend_sg_poly", "sg_poly", SAVGOL_POLY),
        ("detrend_sg_decim", "sg_decim", SAVGOL_DECIM),
    ),
    # EVENT-MASKED MEDIAN. Note ``window_s`` is NOT in this list and is genuinely a
    # NO-OP for this method -- it reads rough_s/base_s only, and two runs at
    # window_s=0.05 and 5.0 are byte-identical (tests/analysis/
    # test_detrend_methods_reachable.py pins it). The panel disables that spinbox here.
    # 'decim' is deliberately absent: estimate_baseline derives it from the tracking
    # window when it is None, and this tuple mechanism cannot express a None default.
    # ``detrend_decim`` below pins it for the two methods that want a fixed stride.
    "masked_median": (
        ("detrend_rough_s", "rough_s", 0.5),
        ("detrend_base_s", "base_s", 0.1),
        ("detrend_mask_sigma", "mask_sigma", 3.5),
        ("detrend_dilate_ms", "dilate_ms", 1.0),
        ("detrend_decim", "decim", _DECIM_AUTO),
    ),
    "robust_median": (
        ("detrend_decim", "decim", _DECIM_AUTO),
    ),
}

# ctx.params keys the module owns / consumes itself and must NOT forward to
# detect_events (it passes recording/units/pore_id explicitly, consumes the detrend
# and filter keys before detection, and _-keys are the controller's private channel).
_CONSUMED_PARAM_KEYS = frozenset(
    {"pore_id", "recording", "units",
     # WHICH detector to run. The panel emits it so the controller can pick a module
     # id; neither detector function has such a parameter, and forwarding it would
     # raise exactly as `baseline_fingerprint` once did.
     "detector",
     *DETREND_PARAM_KEYS, *FILTER_PARAM_KEYS,
     # The characterisation-baseline knobs are resolved into ONE `BaselineConfig`
     # by `baseline.baseline_config_from_params` and passed as `baseline_config=`.
     # They must not also arrive as loose kwargs -- `detect_events` has no
     # `baseline_rough_s` parameter and never will; the config object is the whole
     # point, and a per-key kwarg is how the four call sites drifted apart.
     *BASELINE_PARAM_KEYS, "baseline_config", "baseline_param_source"}
)


def detrend_method_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    """The ``estimate_baseline`` shape kwargs for the method selected in ``params``.

    Only the ACTIVE method's parameters are forwarded. That is not tidiness: it is what
    makes exposing them safe. 'block_median' has no shape parameters, so it keeps calling
    ``estimate_baseline`` with method/window_s/polarity and nothing else -- exactly the
    call it made before these knobs existed -- and a stale ``detrend_asls_lam`` left in a
    params dict from an earlier method cannot leak into it.

    Each value falls back to ``estimate_baseline``'s own default when the key is absent,
    so an old params dict (a restored project, a test) reproduces the frozen behaviour.
    """
    method = str(params.get("detrend_method", "block_median"))
    spec = DETREND_METHOD_PARAMS.get(method, ())
    out: dict[str, Any] = {}
    for param_key, kernel_key, default in spec:
        value = params.get(param_key, default)
        if kernel_key == "decim" and (value is None or int(value) == _DECIM_AUTO):
            # `_DECIM_AUTO` -> the kernel's own None, i.e. derive the stride from the
            # tracking window. Written here rather than dropped from the dict so the
            # keyword is always present and the intent is explicit in one place.
            out[kernel_key] = None
            continue
        out[kernel_key] = int(value) if isinstance(default, int) else float(value)
    return out


def notch_freqs_param(params: Mapping[str, Any]) -> list[float] | None:
    """The EXPLICIT notch line list in ``params``, or ``None`` for the series spelling.

    THE ONE READER of ``notch_freqs``, so the preview, the run and the provenance
    stamp can never disagree about which of the two spellings is in force. Present and
    non-empty means explicit mode and ``notch_freq_hz`` / ``notch_harmonics`` are NOT
    read at all (`filters.notch` ignores them outright rather than merging).

    An EMPTY list is passed through as an empty list, NOT folded to ``None``: the
    kernel raises "an explicit notch line list must name at least one frequency" for
    it, and a silent fallback to a different filter than the one the params describe
    is exactly the class of defect this key exists to fix. The panel never emits ``[]``
    (it omits the key instead) -- this guard is for a hand-built or hand-edited dict.
    """
    v = params.get("notch_freqs")
    if v is None:
        return None
    return [float(f) for f in v]


def any_preprocess_enabled(params: Mapping[str, Any]) -> bool:
    """True if ANY pre-detection filter stage is enabled in ``params``.

    The stages :func:`preprocess_for_detection` may apply are notch, median, high-pass,
    baseline-wander removal (detrend), running-mean, Savitzky-Golay and normalize. This is
    the single source of truth for "is the preprocessed trace different from the raw one?",
    used by the GUI to decide when the on-trace preview is meaningful. It deliberately does
    NOT include the low-pass, which is applied later by :func:`detect_events`, not here.
    """
    return any(
        bool(params.get(k))
        for k in (
            "notch_enabled", "median_enabled", "hp_enabled",
            "detrend_enabled", "rmean_enabled", "savgol_enabled",
            "normalize_enabled",
        )
    )


def preprocess_for_detection(
    signal: np.ndarray, fs: float | None, params: Mapping[str, Any]
) -> np.ndarray:
    """Apply the pre-detection signal conditioning DetectSchmitt runs BEFORE detect_events.

    The enabled stages are applied in a FIXED, documented order, each gated by its own
    ``*_enabled`` flag in ``params``:

        1. NOTCH           -- pull mains hum (and harmonics) before anything widens it.
        2. MEDIAN          -- despike early: delete single-sample impulses/glitches while
                              they are still narrow, before any smooth spreads them out.
        3. HIGH-PASS       -- strip DC / slow drift. WARNS in its docstring that it
                              distorts square events; the panel tooltip repeats it.
        4. BASELINE DETREND-- flatten slow open-pore wander while keeping event shape
                              (:func:`remove_baseline_wander`).
        5. RUNNING-MEAN    -- box-average the noise once the baseline is flat (a fast,
                              unbiased noise floor; rounds peaks -- keep it short).
        6. SAVITZKY-GOLAY  -- smooth noise last, once the baseline is flat, so the
                              polynomial fit is not chasing drift.
        7. NORMALIZE       -- rescale the whole conditioned trace onto a dimensionless
                              scale (:func:`~nanoscope.analysis.filters.normalize`).

    The order is not arbitrary: despike then line/DC/drift removal precede the smoothing
    stages so each stage sees a cleaner input than the last. The result is returned as
    contiguous ``float32``.

    NORMALIZE IS LAST, AND IT TAKES THE PICOAMPS WITH IT. It is a single affine map over
    the whole trace, so putting it last is what keeps the six conditioning stages -- and
    :func:`smoothing_depth_retention`, whose answer is a RATIO and therefore survives any
    rescale -- measuring exactly what they measured before the stage existed. Everything
    DOWNSTREAM is a different matter: ``depth_pA``, the baseline, the thresholds and every
    exported column are still spelled "pA" while carrying sigmas or span-fractions. The
    detector cannot detect that, so the disclosure is the panel's job -- see
    ``DetectionPanel._refresh_normalize_units_note``, which keeps a banner up on every
    page of the Signal-filters group for as long as the stage is on.

    The LOW-PASS is deliberately NOT applied here. :func:`detect_events` owns it (it
    must, to measure the real rise time and flag filter-limited events), and applying
    it twice would over-smooth the trace and corrupt those numbers. This helper is the
    single source of truth for the *extra* transforms so the live preview and the
    surrogate studies (sweep/null/recovery) condition the signal identically to the
    detector -- they hand the preprocessed signal to detect_events, which then low-passes.

    Returns the signal unchanged when the rate is unknown (every stage needs a time base;
    detection is gated off in that case anyway) or when no stage is enabled.

    CANCELLABLE BETWEEN STAGES. Each stage is one monolithic scipy call that nothing can
    interrupt from the outside, so a poll BETWEEN them is the finest grain available;
    the worst-case latency is therefore one stage, not the whole chain. It matters
    because this function is the front of the two slowest paths in the app: the
    surrogate studies condition the whole recording through it before ``sweep_threshold``
    even begins, and ``MLDetect.run`` runs the whole-recording notch + SavGol before it
    loads a model. Nothing about the numerics moved — :func:`check_cancelled` returns
    immediately when no cancel token is installed, which is every non-worker caller.
    """
    # Validate BEFORE the first filter. Several conditioning kernels are local
    # smoothers (especially the median) and can replace an isolated NaN/Inf with a
    # finite neighbour. Waiting for ``detect_events`` to validate its input would
    # then turn corrupt acquisition data into an apparently healthy trace. The
    # preprocessing boundary is therefore the fail-closed door for every caller.
    source = np.asarray(signal)
    if source.ndim != 1:
        raise ValueError(f"signal must be 1-D before preprocessing, got shape {source.shape}")
    try:
        finite = bool(np.isfinite(source).all())
    except TypeError as exc:
        raise ValueError("signal must contain numeric samples before preprocessing") from exc
    if not finite:
        raise ValueError(
            "signal contains NaN/Inf before preprocessing; filtering is refused because "
            "a median or smoother could hide the corrupt samples"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        arr = np.ascontiguousarray(source, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(
            "signal contains values outside the float32 processing range; converting "
            "them would create NaN/Inf"
        )
    if fs is None:
        # Every OTHER stage needs a time base. Normalize does not -- no window, no kernel,
        # no time constant -- so an unknown rate is no reason to skip the one stage whose
        # effect is a factor on every sample, and skipping it here would let the preview
        # disagree with the run.
        return _apply_normalize(arr, params)
    fs_f = float(fs)

    check_cancelled()
    # 1. Notch out mains hum and its harmonics.
    if params.get("notch_enabled"):
        arr = notch(
            arr,
            fs_f,
            freq_hz=float(params.get("notch_freq_hz", DEFAULT_NOTCH_FREQ_HZ)),
            q=float(params.get("notch_q", DEFAULT_NOTCH_Q)),
            harmonics=int(params.get("notch_harmonics", DEFAULT_NOTCH_HARMONICS)),
            freqs=notch_freqs_param(params),
        )

    check_cancelled()
    # 2. Median despike: remove single-sample impulses/glitches while they are still narrow.
    if params.get("median_enabled"):
        arr = median_filter(
            arr, fs_f, window_s=float(params.get("median_window_s", DEFAULT_MEDIAN_WINDOW_S))
        )

    check_cancelled()
    # 3. High-pass to remove DC / slow drift (distorts square events -- for drift only).
    if params.get("hp_enabled"):
        arr = highpass(
            arr,
            fs_f,
            float(params.get("hp_cutoff_hz", DEFAULT_HP_CUTOFF_HZ)),
            order=int(params.get("hp_order", DEFAULT_HP_ORDER)),
            kind=str(params.get("hp_kind", DEFAULT_HP_KIND)),
        )

    check_cancelled()
    # 4. Baseline-wander removal (detrend), the event-shape-preserving drift flattener.
    if params.get("detrend_enabled"):
        arr = remove_baseline_wander(
            arr,
            fs_f,
            method=str(params.get("detrend_method", "block_median")),
            window_s=float(params.get("detrend_window_s", DEFAULT_DETREND_WINDOW_S)),
            polarity=params.get("polarity"),
            **detrend_method_kwargs(params),
        )

    # 5 and 6: the two smoothing stages, through the one helper a probe can reuse.
    arr = _apply_smoothers(arr, fs_f, params)

    check_cancelled()
    # 7. Normalize LAST, so the six stages above -- and the depth-retention probe, whose
    #    answer is a ratio and so survives any rescale -- keep measuring what they always
    #    measured. See this function's docstring for what it costs downstream.
    arr = _apply_normalize(arr, params)

    check_cancelled()
    return np.ascontiguousarray(arr, dtype=np.float32)


def _apply_normalize(arr: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Stage 7 of the conditioning chain: the dimensionless rescale, or a no-op.

    Its own helper for the same reason `_apply_smoothers` is one: two call sites reach it
    (the ordinary chain and the unknown-rate early return), and a second copy of the call
    is how a stage starts being applied on one path and not the other.
    """
    if not params.get("normalize_enabled"):
        return arr
    mode = str(params.get("normalize_mode", "zscore"))
    if mode == MODEL_INPUT_MODE:
        # THE ONLY PLACE THE TWO KERNELS ARE CHOSEN BETWEEN. `robust_normalize` is
        # deliberately not a fourth mode of `filters.normalize` -- it is a trimmed
        # three-pass re-estimation followed by a CLIP (not affine, bounded, and with
        # the OPPOSITE degenerate-input contract: it falls back on a flat trace where
        # `normalize` raises). Dispatching here keeps that separation while letting
        # one params key select either.
        return robust_normalize(
            arr,
            clip_sigma=float(
                params.get("normalize_clip_sigma", MODEL_INPUT_CLIP_SIGMA)),
        )
    return normalize(arr, mode=mode)


def _apply_smoothers(
    arr: np.ndarray, fs: float, params: Mapping[str, Any]
) -> np.ndarray:
    """Stages 5 (running-mean) and 6 (Savitzky-Golay) of the conditioning chain.

    Factored out of :func:`preprocess_for_detection` because these are the two stages
    that ATTENUATE EVENT DEPTH, and :func:`smoothing_depth_retention` has to push a
    synthetic square event through the EXACT code the data went through in order to
    say by how much. A second copy of the two calls is how a disclosure starts
    describing a smoother nobody applied.

    Cancellation is polled between the stages exactly as it was when they were inline.
    """
    check_cancelled()
    if params.get("rmean_enabled"):
        arr = running_mean(
            arr, fs, window_s=float(params.get("rmean_window_s", DEFAULT_RMEAN_WINDOW_S))
        )

    check_cancelled()
    if params.get("savgol_enabled"):
        arr = savgol(
            arr,
            fs,
            window_s=float(params.get("savgol_window_s", DEFAULT_SAVGOL_WINDOW_S)),
            polyorder=int(params.get("savgol_polyorder", DEFAULT_SAVGOL_POLYORDER)),
        )
    return arr


def smoothing_stage_labels(params: Mapping[str, Any]) -> tuple[str, ...]:
    """Human names (with their windows) of the enabled depth-attenuating stages.

    Written here, beside the stages themselves, so the sentence a user reads names
    the same windows :func:`_apply_smoothers` applied.
    """
    labels: list[str] = []
    if params.get("rmean_enabled"):
        win = float(params.get("rmean_window_s", DEFAULT_RMEAN_WINDOW_S))
        labels.append(f"running-mean ({win * 1e3:g} ms window)")
    if params.get("savgol_enabled"):
        win = float(params.get("savgol_window_s", DEFAULT_SAVGOL_WINDOW_S))
        order = int(params.get("savgol_polyorder", DEFAULT_SAVGOL_POLYORDER))
        labels.append(f"Savitzky-Golay ({win * 1e3:g} ms window, order {order})")
    return tuple(labels)


def smoothing_depth_retention(
    fs: float, params: Mapping[str, Any], dwell_s: float
) -> float | None:
    """Fraction of a square event's depth that survives the enabled smoothing stages.

    ``None`` when no smoothing stage is enabled -- there is then nothing to disclose.

    WHY THIS IS MEASURED RATHER THAN ASSERTED. The low-pass inside
    :func:`~nanoscope.analysis.detect.detect_events` has a whole disclosure
    machinery for exactly this damage: it measures its own rise time, marks every
    affected row ``filter_limited``, and :func:`_build_warnings` prints the fraction.
    The SavGol and running-mean stages do the same physical damage one rung earlier
    and the detector never learns they ran, so no row can carry the flag. Measured,
    20 square 1.000 ms / 100 pA events at 100 kHz:

        SavGol at the APP DEFAULT window (1 ms, order 3)  depth 76.74 pA (-23.3%)
        SavGol 5.0 ms                                     depth  6.56 pA (-93.4%)
        running mean 3.0 ms                               depth 33.09 pA (-66.9%)

    -- all with ``filter_limited`` false on every row and no warning of any kind,
    while the 1 kHz low-pass in the same table, with a SMALLER error, said so.

    The number returned is the MEDIAN of the smoothed pulse over the pulse's own span
    -- measured the way :func:`~nanoscope.analysis.characterize.characterize` measures
    ``depth_pA``, so it predicts what the table will report rather than some other
    statistic. (The peak would not: a Savitzky-Golay window overshoots at an edge, so
    the peak of a well-resolved smoothed pulse reads 1.08 of its true depth while the
    depth the app reports is 1.00.) The real detected window is wider than the pulse,
    which pulls the reported median down further, so this is an upper bound on what an
    event of this duration keeps -- the exact per-event attenuation is not recoverable
    from the smoothed samples.
    """
    if not any(params.get(k) for k in ("rmean_enabled", "savgol_enabled")):
        return None
    fs_f = float(fs)
    width = max(1, int(round(float(dwell_s) * fs_f)))
    # Pad generously: `savgol` clamps a window longer than the trace, which would
    # measure a different (milder) filter than the one that ran on the recording.
    kernel = max(
        running_mean_samples(fs_f, float(params.get("rmean_window_s", DEFAULT_RMEAN_WINDOW_S)))
        if params.get("rmean_enabled") else 1,
        savgol_window_samples(
            fs_f,
            float(params.get("savgol_window_s", DEFAULT_SAVGOL_WINDOW_S)),
            int(params.get("savgol_polyorder", DEFAULT_SAVGOL_POLYORDER)),
        ) if params.get("savgol_enabled") else 1,
    )
    pad = max(4 * kernel, 4 * width, 64)
    probe = np.zeros(2 * pad + width, dtype=np.float32)
    probe[pad : pad + width] = -1.0  # one unit-depth square blockade
    out = np.asarray(_apply_smoothers(probe, fs_f, params), dtype=np.float64)
    return float(-np.median(out[pad : pad + width]))


def smoothing_resolution_dwell_s(
    fs: float, params: Mapping[str, Any], *, keep: float = 0.9
) -> float | None:
    """Shortest square event that keeps ``keep`` of its depth through the smoothers.

    The smoothers' equivalent of the low-pass's ``2 * filter_rise_s`` resolution floor,
    and the honest way to state their cost: it is a property of the FILTER, so unlike a
    count of short rows it cannot be fooled by the dwells the same filter inflated.

    ``None`` when no smoothing stage is enabled. Bisected on
    :func:`smoothing_depth_retention`, which rises monotonically with the event's
    duration; the search is capped at 32 kernel lengths, far past where every stage
    here reaches full retention.
    """
    if smoothing_depth_retention(fs, params, 1.0 / float(fs)) is None:
        return None
    fs_f = float(fs)
    lo, hi = 1, 1
    cap = 32 * max(
        running_mean_samples(fs_f, float(params.get("rmean_window_s", DEFAULT_RMEAN_WINDOW_S)))
        if params.get("rmean_enabled") else 1,
        savgol_window_samples(
            fs_f,
            float(params.get("savgol_window_s", DEFAULT_SAVGOL_WINDOW_S)),
            int(params.get("savgol_polyorder", DEFAULT_SAVGOL_POLYORDER)),
        ) if params.get("savgol_enabled") else 1,
    )
    while hi <= cap and (smoothing_depth_retention(fs_f, params, hi / fs_f) or 0.0) < keep:
        lo, hi = hi, hi * 2
    if hi > cap:
        return float(cap / fs_f)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if (smoothing_depth_retention(fs_f, params, mid / fs_f) or 0.0) < keep:
            lo = mid
        else:
            hi = mid
    return float(hi / fs_f)


def preprocess_params_from_provenance(prov: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a detection's provenance back into ``preprocess_for_detection`` params.

    :class:`DetectSchmitt` stamps every enabled stage's parameters onto provenance so a
    result is self-describing, and it uses the same key names -- with ONE exception: the
    detrend method's shape parameters are stamped under their ``estimate_baseline``
    KERNEL names (``percentile``, ``asls_lam``, ...) while
    :func:`preprocess_for_detection` reads them under their PARAM names
    (``detrend_percentile``, ``detrend_asls_lam``, ...). Feeding provenance straight back
    in would therefore silently fall back to the kernel defaults for exactly the knobs
    the user moved. This translates them, using :data:`DETREND_METHOD_PARAMS` -- the same
    table :func:`detrend_method_kwargs` maps the other way.
    """
    params = dict(prov)
    method = str(prov.get("detrend_method", "block_median"))
    for param_key, kernel_key, _default in DETREND_METHOD_PARAMS.get(method, ()):
        if kernel_key in prov:
            params[param_key] = prov[kernel_key]

    # ``notch_freqs`` is the one LIST-valued preprocessing key, and a project file is
    # JSON, so it comes back as a list of numbers rather than the tuple the stamp
    # wrote. Re-spelling it as a list of floats here is what keeps a restored chain
    # comparing equal to the live one (`detection_signal_key` hashes these values, and
    # a tuple and a list of the same numbers are different keys). An explicit None --
    # the series spelling -- is dropped so it cannot shadow ``notch_freq_hz``.
    if "notch_freqs" in params:
        lines = params["notch_freqs"]
        if lines is None:
            params.pop("notch_freqs")
        else:
            params["notch_freqs"] = [float(f) for f in lines]

    # ``polarity`` is the second trap. ``preprocess_for_detection`` reads the value the
    # USER set -- often None, which makes ``remove_baseline_wander`` derive it from the
    # unfiltered signal -- while provenance's ``polarity`` is what ``detect_events``
    # resolved afterwards, from the LOW-PASSED trace. Those can disagree, and feeding the
    # resolved one back in would reconstruct a detrend that never ran.
    # ``polarity_requested`` is the original, so prefer it when provenance has it.
    if "polarity_requested" in prov:
        params["polarity"] = prov["polarity_requested"]
    return params


def characterization_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the conditioning recipe used for quantitative event measurement.

    High-pass is a candidate-localisation aid: it may remove enough wander for a
    threshold detector to find stable boundaries, but it also removes the open-pore DC
    reference and makes a square event droop/ring. Quantitative rows therefore use the
    otherwise identical chain with only ``hp_enabled`` forced off. The input mapping is
    never mutated, and all of the high-pass design values remain available in the run's
    provenance to reproduce the candidate search.
    """
    out = dict(params)
    out["hp_enabled"] = False
    return out


# The detect_events kwargs the DetectionPanel form owns, in the spelling ``params()`` uses.
# The preprocessing keys are DETREND_PARAM_KEYS + FILTER_PARAM_KEYS; together the three
# tuples are the whole Schmitt form.
DETECT_PARAM_KEYS = (
    "polarity", "k_in", "k_out", "min_dwell_s", "max_dwell_s", "merge_gap_s",
    "filter_cutoff_hz", "filter_order", "filter_kind",
)

# The same list for the CUSUM detector. It shares every key with the Schmitt form
# EXCEPT the two thresholds: ``k_in``/``k_out`` are a hysteresis pair in noise sigmas,
# ``cusum_k``/``cusum_h`` are a sequential test's drift and decision threshold -- also
# dimensionless, also in MAD-sigma units, but they are NOT the same quantity and a
# value carried across would be meaningless (k_in=5.0 as a CUSUM drift detects
# nothing; cusum_h=28 as an enter threshold detects nothing).
DETECT_CUSUM_PARAM_KEYS = (
    "polarity", "cusum_k", "cusum_h", "min_dwell_s", "max_dwell_s", "merge_gap_s",
    "filter_cutoff_hz", "filter_order", "filter_kind",
)

#: The keys that belong to ONE detector and must never reach the other one. PUBLIC:
#: the GUI needs it too -- the three Schmitt-only surrogate studies (sweep / null /
#: recovery) hand the form's params straight to `detect_events`, so they have to strip
#: the same set, and a second spelling of it in a GUI file is how the two would drift.
#:
#: The Detection tab carries both forms at once (so switching detectors does not
#: discard what the user dialled into the other, and so a saved protocol round-trips
#: whichever was selected), which means ``ctx.params`` always contains both sets. This
#: table is what keeps a CUSUM run from receiving ``k_in`` -- which `detect_events_cusum`
#: has no parameter for and would raise a TypeError on -- and vice versa. It is keyed by
#: MODULE ID so a third detector cannot be added without declaring its own private keys.
DETECTOR_PRIVATE_KEYS: dict[str, frozenset[str]] = {
    "detect.schmitt": frozenset({"k_in", "k_out"}),
    "detect.cusum": frozenset({"cusum_k", "cusum_h"}),
}


def detection_params_from_provenance(prov: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the SETTINGS a Schmitt detection ran under, from its provenance alone.

    The last-resort source for "what produced these events": a project written before the
    result's ``params`` were persisted still carries the detector's own stamp of every
    threshold and every enabled preprocessing stage, and that stamp is the truth about the
    run. Used to re-seed the detection form (so the tab agrees with the table on screen) and
    to re-run the same detection under a new calibration.

    Only keys provenance actually carries are returned — a caller merges them over the
    form's current values, so an absent key means "unchanged", never "default". Two traps are
    handled by :func:`preprocess_params_from_provenance`: the detrend shape parameters are
    stamped under their kernel names, and ``polarity`` must come from ``polarity_requested``
    (what the user asked for) rather than the value the detector resolved. A third is handled
    here: ``filter_kind`` / ``filter_order`` are stamped ``None`` when the low-pass was off
    (and the order is ``None`` for the order-less 'gaussian' kind), and a None there means
    "not applicable", not a setting — feeding it back would blank a real widget.
    """
    if (prov.get("baseline_reference_restore_error")
            or prov.get("baseline_open_pore_reference") is not None
            or prov.get("baseline_reference_applied") or prov.get("baseline_estimator_bypassed")):
        # Validate the full saved instruction before projecting away derived flags.
        # Otherwise a missing recipe could become an apparently valid Automatic run.
        from .baseline import BaselineConfig
        BaselineConfig.reproducible_from(prov)
    base = preprocess_params_from_provenance(prov)
    out: dict[str, Any] = {}
    for key in (*DETECT_PARAM_KEYS, *DETREND_PARAM_KEYS, *FILTER_PARAM_KEYS,
                # A FOURTH trap, and the one this list existed without: the
                # characterisation-baseline knobs. Re-running a stored detection
                # under a new calibration must re-measure it with the estimator that
                # produced it, not with whatever the live default happens to be --
                # otherwise the "same detection, new calibration" path silently
                # changes depth_pA by up to 15x and the two results are not
                # comparable, which is the whole reason the fingerprint exists.
                *BASELINE_PARAM_KEYS, "baseline_open_pore_reference"):
        if key in base:
            out[key] = base[key]
    for key in ("filter_kind", "filter_order"):
        if out.get(key) is None:
            out.pop(key, None)
    if any(k in out for k in BASELINE_PARAM_KEYS):
        # Reconstructed from a stored result, not chosen now. `PARAM_SOURCES`.
        out["baseline_param_source"] = "restored"
    return out


def detection_signal(
    signal: np.ndarray, fs: float | None, prov: Mapping[str, Any]
) -> np.ndarray:
    """The trace quantitative event rows were measured on, rebuilt from provenance.

    ``preprocess_for_detection`` conditions the signal and then ``detect_events`` applies
    its own low-pass. Ordinarily, the current every reported ``depth_pA`` /
    ``baseline_pA`` / ``rise_s`` was read off is
    ``lowpass(preprocess_for_detection(raw))`` -- NOT the raw channel. For a new
    classical result stamped ``highpass_localization_only``, the same reconstruction
    forces only high-pass off: its bounds came from the candidate trace, while its row
    measurements came from the companion characterization trace. Anything that
    re-measures an event after the fact (a merge, an Event Inspector mask resize, a
    hand-added missed event) has to start from this row-measurement array or it is
    reporting a different measurement under the same column names.

    Measured on the user's -600 mV recording, re-characterizing merged spans on the raw
    channel instead of this shifted ``frac_depth`` by 2.6% at the median (10 kHz Bessel,
    the app default) and much further with a slower filter.

    Only the stages provenance actually describes are applied. A legacy high-pass
    result without the split marker retains its historical all-on-one-trace replay;
    an old project, hand-built EventSet or ML run with no classical chain is otherwise
    left alone rather than guessed at. Returns contiguous float32.
    """
    arr = np.ascontiguousarray(signal, dtype=np.float32)
    cutoff = prov.get("filter_cutoff_hz")
    if fs is None:
        if cutoff is not None:
            raise ValueError("The saved low-pass filter requires a known sample rate.")
        return arr
    fs_f = float(fs)
    kind = str(prov.get("filter_kind") or "bessel")
    order = 8
    if cutoff is not None:
        try:
            cutoff = float(cutoff)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("The saved low-pass cutoff is not a valid frequency.") from exc
        if not np.isfinite(fs_f) or fs_f <= 0:
            raise ValueError("The saved low-pass filter requires a finite, positive sample rate.")
        if not np.isfinite(cutoff) or not (0.0 < cutoff < fs_f / 2.0):
            raise ValueError("The saved low-pass cutoff must be finite, positive and below Nyquist.")
        saved_order = prov.get("filter_order")
        # An omitted/None order is the historical IIR default. Gaussian does
        # not use order at all. An EXPLICIT invalid IIR order is not permission
        # to substitute 8 (or truncate a fractional value) during remeasurement.
        if kind != "gaussian" and saved_order is not None:
            try:
                order = int(saved_order)
                valid_order = (
                    not isinstance(saved_order, (bool, np.bool_))
                    and order >= 1
                    and float(saved_order) == order
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("The saved low-pass IIR order must be a positive integer.") from exc
            if not valid_order:
                raise ValueError("The saved low-pass IIR order must be a positive integer.")
    preprocess = preprocess_params_from_provenance(prov)
    # New classical runs use high-pass only to localise candidate boundaries. All
    # post-hoc row rebuilds (merge, mask resize, hand-add) must therefore reconstruct
    # the non-high-passed CHARACTERISATION trace. Absence of the stamp means a legacy
    # result that really was measured on high-pass, so its historical semantics remain
    # reproducible rather than being silently rewritten on open.
    if (
        prov.get("highpass_localization_only")
        or prov.get("characterization_hp_omitted")
    ):
        preprocess = characterization_params(preprocess)
    arr = preprocess_for_detection(arr, fs_f, preprocess)

    if cutoff is None:
        return np.ascontiguousarray(arr, dtype=np.float32)
    return np.ascontiguousarray(
        lowpass(arr, fs_f, cutoff, order=order, kind=kind), dtype=np.float32
    )


def detection_signal_key(prov: Mapping[str, Any]) -> tuple[Any, ...]:
    """A hashable digest of everything :func:`detection_signal` reads from provenance.

    Callers cache the conditioned trace (it is ~0.12 s for a 3 M-sample recording, and
    an Event Inspector drag would otherwise pay that on every release); this is the key
    that says when the cached copy is still the right one.
    """
    effective = dict(prov)
    if (
        prov.get("highpass_localization_only")
        or prov.get("characterization_hp_omitted")
    ):
        effective["hp_enabled"] = False
        for key in ("hp_cutoff_hz", "hp_order", "hp_kind"):
            effective.pop(key, None)
    keys = (
        "filter_cutoff_hz", "filter_kind", "filter_order",
        "polarity", "polarity_requested",
        *DETREND_PARAM_KEYS, *FILTER_PARAM_KEYS,
    )
    method = str(prov.get("detrend_method", "block_median"))
    kernel_keys = tuple(k for _p, k, _d in DETREND_METHOD_PARAMS.get(method, ()))
    return tuple(effective.get(k) for k in (*keys, *kernel_keys))


def _fingerprint(params: Mapping[str, Any]) -> tuple[float, float]:
    raw = params.get("_calibration_fingerprint", _DEFAULT_FINGERPRINT)
    vals = tuple(float(v) for v in raw)
    if len(vals) != 2:
        raise ValueError(
            f"_calibration_fingerprint must be a (gain, offset) pair, got {raw!r}"
        )
    return vals  # type: ignore[return-value]


#: The whole ``baseline_`` namespace belongs to `analysis.baseline`, and NONE of it is
#: a ``detect_events`` keyword. Naming the eight parameter keys individually (which is
#: what `_CONSUMED_PARAM_KEYS` does) leaves a gap the moment a caller merges a
#: `BaselineConfig.to_provenance` dict into its params, because that dict also carries
#: ``baseline_fingerprint`` -- a DERIVED value, not a parameter, and so not in
#: `BASELINE_PARAM_KEYS`. Measured: the GUI baseline form did exactly that, and
#: ``detect_events()`` raised ``unexpected keyword argument 'baseline_fingerprint'``
#: on every run -- which the window turned into a modal error box.
#:
#: The prefix is the boundary, so the gap cannot reopen for the next derived key
#: (``baseline_health``, ``baseline_median_pA``, ``baseline_fallback``...). Safe
#: because ``detect_events``' own baseline parameter is spelled ``baseline``, with no
#: underscore after it.
_BASELINE_NAMESPACE = "baseline_"


def _detect_kwargs(
    params: Mapping[str, Any], module_id: str = "detect.schmitt"
) -> dict[str, Any]:
    """The subset of ctx.params forwarded to ``module_id``'s detector function: drop
    the module's own private (``_``-prefixed) keys, the ones passed
    positionally/explicitly, the whole ``baseline_`` namespace (see
    `_BASELINE_NAMESPACE`) and EVERY OTHER DETECTOR'S private thresholds
    (`DETECTOR_PRIVATE_KEYS`).

    That last clause is what lets one form carry both detectors' knobs. Without it a
    CUSUM run would hand ``k_in``/``k_out`` to `detect_events_cusum` and die on
    ``unexpected keyword argument 'k_in'`` -- the same failure the ``baseline_``
    namespace rule was written for, one detector later.
    """
    foreign = frozenset().union(
        *(v for k, v in DETECTOR_PRIVATE_KEYS.items() if k != module_id)
    )
    return {
        k: v
        for k, v in params.items()
        if not k.startswith("_") and k not in _CONSUMED_PARAM_KEYS
        and k not in foreign
        and not k.startswith(_BASELINE_NAMESPACE)
    }


def notch_settling_from_provenance(
    prov: Mapping[str, Any]
) -> tuple[float, float] | None:
    """``(settle_s, unsettled_s)`` for the notch a provenance block describes.

    ``None`` when the notch did not run, or when provenance does not carry enough to
    say (no rate, no length, an unusable design). THE ONE COMPUTATION behind both the
    stamped keys and the sentence in :func:`_build_warnings`, so the provenance and
    the warning can never disagree about how long the filter needed.

    ``unsettled_s`` is 0.0 for any ordinary recording: :func:`~nanoscope.analysis.
    filters.notch` pads with its own ring-down and only falls short on a trace shorter
    than ~1.8 s at the app's defaults (a hand-cut segment), where the head really does
    still carry the filter's startup transient.
    """
    if not prov.get("notch_enabled"):
        return None
    try:
        fs = float(prov["fs"])
        n_samples = int(prov["n_samples"])
        freq = float(prov.get("notch_freq_hz", DEFAULT_NOTCH_FREQ_HZ))
        q = float(prov.get("notch_q", DEFAULT_NOTCH_Q))
        harmonics = int(prov.get("notch_harmonics", DEFAULT_NOTCH_HARMONICS))
        # THE LINES THAT ACTUALLY RAN, not the series they might have been asked for.
        # A settling time sized from a filter that did not run is worse than none:
        # the nine-line Zn/Cu comb settles for longer than the three-line default, and
        # reporting the default's ring-down would under-state the unsettled head.
        freqs = notch_freqs_param(prov)
        settle = notch_settle_s(fs, freq, q=q, harmonics=harmonics, freqs=freqs)
        unsettled = notch_unsettled_s(
            fs, n_samples, freq, q=q, harmonics=harmonics, freqs=freqs)
    except (KeyError, TypeError, ValueError, RuntimeError):
        return None
    return float(settle), float(unsettled)


def _smoothing_warning(prov: Mapping[str, Any], df: pd.DataFrame) -> str | None:
    """The one sentence that discloses what the conditioning smoothers did to depth.

    The low-pass inside ``detect_events`` gets ``filter_limited`` plus a printed
    fraction; the SavGol / running-mean stages that ran one step earlier get nothing,
    because the detector is never told they happened. This is that disclosure, and it
    is quantitative on purpose: a bare "a smoother was enabled" would be a caveat the
    user cannot act on, while "an event of your median dwell keeps at most 77% of its
    depth" is the number the depths in the table are wrong by.

    IT IS JUDGED AT HALF THE MEDIAN MEASURED DWELL, not at the measured dwell itself.
    A smoother widens every event it attenuates, so asking "what would an event of the
    dwell in this table have lost?" asks the question of a number the same stage
    inflated -- measured, 1.000 ms events came back as 1.510 ms through the app's
    default SavGol window, and evaluated there the answer is "nothing" while the
    reported depth was 23% low. Half the measured dwell is a deliberately conservative
    stand-in for the true one (the observed inflation runs from 1.5x to 2.7x).

    Silent when no smoother ran, when no events were found (no depth is being reported,
    so there is nothing to caveat) and when the enabled smoother is short enough that
    even an event of that halved dwell keeps >= 95% of its depth -- an advisory that
    fires on a harmless window teaches the user to ignore it.
    """
    labels = smoothing_stage_labels(prov)
    if not labels or df.empty:
        return None
    dwells = pd.to_numeric(df["dwell_s"], errors="coerce").to_numpy(dtype=float)
    dwells = dwells[np.isfinite(dwells) & (dwells > 0)]
    if dwells.size == 0:
        return None
    fs = prov.get("fs")
    if not fs:
        return None
    probe_dwell = 0.5 * float(np.median(dwells))
    try:
        keep_probe = smoothing_depth_retention(float(fs), prov, probe_dwell)
        floor_s = smoothing_resolution_dwell_s(float(fs), prov, keep=0.9)
    except (TypeError, ValueError):
        return None
    if keep_probe is None or floor_s is None or keep_probe >= 0.95:
        return None
    return (
        f"{' + '.join(labels)} smoothed the trace before detection: a square event has "
        f"to last {floor_s * 1e3:.3g} ms to keep 90% of its depth through that stage, "
        f"and one lasting half the median dwell in this table ({probe_dwell * 1e3:.3g} "
        f"ms) keeps only {100 * keep_probe:.0f}%. Depths here are attenuated by it and "
        "dwells biased long -- and because the same stage widens what it attenuates, "
        "the dwell column cannot tell you which rows are affected. No row carries a "
        "flag for this: filter_limited covers only the detector's own low-pass."
    )


def normalized_units_warning(prov: Mapping[str, Any]) -> str:
    """The one sentence that says this table's "pA" columns are not picoamps.

    THE ONE WRITER of it, for the same reason ``stamp_preprocess_provenance`` is the
    one writer of the block it reads: the fact has to reach the Plots window, the
    Summary panel and every export, and the way it failed to was by living in ONE
    surface -- ``DetectionPanel._refresh_normalize_units_note``, a live banner that
    disappears the moment the user leaves the panel and never existed on the table.

    :func:`~nanoscope.core.sources.signal_domain_of` supplies the descriptor so the
    per-run sentence and the cross-run pooling warning name the stage identically;
    a user who sees both must not have to work out that they are the same stage.
    """
    # Imported HERE rather than at module scope: this is the only thing in the
    # detector modules that needs the plot-source layer, and a top-level import would
    # pull core.document + core.segment into every `import nanoscope.analysis.modules`
    # -- including the headless notebook paths that never draw anything. There is no
    # cycle to dodge (analysis.voltage_dep imports core.sources at module scope); this
    # is about not widening the import graph for one sentence.
    from ..core.sources import signal_domain_of

    domain = signal_domain_of(prov)
    mode = str(prov.get("normalize_mode", "zscore"))
    # 'model_input' is the one mode whose damage is not merely a change of unit: it
    # is BOUNDED, so two events of different real depth come out identical above the
    # clip and no row carries a flag for it. The panel banner splits on exactly this;
    # so does this sentence, and for exactly that reason.
    clipped = ""
    if mode == MODEL_INPUT_MODE:
        clip = float(prov.get("normalize_clip_sigma", MODEL_INPUT_CLIP_SIGMA))
        clipped = (
            f" It is also BOUNDED: anything deeper than {clip:g} sigma is CLIPPED to "
            "the bound, so depth is no longer monotone in the real picoamps above it "
            "and a histogram grows a spike at the edge that looks like a population."
        )
    return (
        f"Preprocess ▸ Normalize was ON, so the detector measured a DIMENSIONLESS "
        f"trace [{domain}]. depth_pA, baseline_pA, ecd_pA_s, max_dev_pA, mean_dev_pA "
        f"and other current-derived pA/pA·s export fields retain physical-unit labels "
        f"while carrying transformed values — the label is written from the channel, "
        f"not measured, so nothing "
        f"downstream can see it.{clipped} Do not pool these depths with an "
        f"un-normalized run, and do not reach for frac_depth as the way out: the "
        f"rescale subtracts a location as well as dividing, so |baseline_pA| lands "
        f"at ~0 and dI/I0 blows up (measured on a 100 pA blockade in a 1000 pA pore: "
        f"0.100 raw against 105 through 'robust'). Index, time, source, review and "
        f"status fields keep their ordinary meanings. Model-input clipping can also "
        f"move threshold crossings or change the event count."
    )


def highpass_measurement_warning(prov: Mapping[str, Any]) -> str:
    """Explain the classical detector's high-pass localization/measurement split."""
    cutoff = prov.get("hp_cutoff_hz", DEFAULT_HP_CUTOFF_HZ)
    try:
        cutoff_text = f"{float(cutoff):g} Hz"
    except (TypeError, ValueError):
        cutoff_text = "an unknown cutoff"
    if not (
        prov.get("highpass_localization_only")
        or prov.get("characterization_hp_omitted")
    ):
        # Backward-compatible truth for a legacy result loaded from a project written
        # before the split existed. Its values really were measured on high-pass and
        # must not be retroactively described as repaired.
        return (
            f"Preprocess ▸ High-pass was ON ({cutoff_text}) in a legacy run that found "
            "AND characterised events on the zero-DC trace. baseline_pA, frac_depth, "
            "depth_pA, mean_dev_pA, max_dev_pA, ecd_pA_s and SNR are not reliable physical "
            "measurements; treat these rows as candidate locations and re-run them."
        )
    # ``detection_polarity`` is the sign APPLIED to candidate localization and is
    # intentionally the characterization-domain sign. ``detection_polarity_measured``
    # is the unconstrained vote on the high-pass trace -- the diagnostic that reveals
    # ringing tried to reverse it.
    detection_pol = prov.get("detection_polarity_measured")
    measurement_pol = prov.get("polarity")
    polarity_clause = ""
    if detection_pol in (-1, 1) and measurement_pol in (-1, 1) \
            and int(detection_pol) != int(measurement_pol):
        if prov.get("polarity_retry_after_highpass_mismatch"):
            polarity_clause = (
                f" The high-pass trace itself voted {int(detection_pol):+d}, opposite "
                f"the {int(measurement_pol):+d} physical-branch vote; NanoScope "
                "repeated localization with the physical sign instead of accepting "
                "the ringing direction. Inspect these candidates closely."
            )
        else:
            polarity_clause = (
                f" The high-pass trace itself voted {int(detection_pol):+d}, opposite "
                f"the user's {int(measurement_pol):+d} polarity override; localization "
                "honored the user's direction. Inspect these candidates closely."
            )
    return (
        f"Preprocess ▸ High-pass was ON ({cutoff_text}) for CANDIDATE LOCALIZATION ONLY. "
        "baseline_pA, depth_pA, frac_depth, mean_dev_pA, max_dev_pA, ecd_pA_s and SNR "
        "were measured on the otherwise identical conditioned trace with High-pass "
        "omitted, so High-pass itself does not erase their DC/open-pore reference. "
        "Every other enabled stage still applies; Normalize, in particular, remains "
        "dimensionless and is disclosed separately. Event "
        "start/end, dwell, rise/fall and ECD's integration limits still inherit the "
        "high-pass detector boundaries; ringing or plateau droop can therefore move "
        "those boundaries. Review them when the cutoff approaches the event bandwidth."
        + polarity_clause
    )


def _build_warnings(prov: dict[str, Any], df: pd.DataFrame) -> tuple[str, ...]:
    """Surface the detector's soft advisories on the result (undetermined
    polarity, filter-limited fraction, what the conditioning chain cost, and whether
    the chain left the trace in picoamps at all)."""
    warns: list[str] = []
    # The estimator's own advisories first: a baseline that fell back, or whose
    # health checks fired, is upstream of every other number in the table.
    warns.extend(str(w) for w in (prov.get("baseline_warnings") or ()))
    # A split run estimates another baseline on the high-pass candidate trace to set
    # its thresholds. Its failure does not invalidate the measurement baseline above,
    # but it can invalidate the localized bounds, so a distinct advisory must travel.
    measurement_baseline = set(
        str(w) for w in (prov.get("baseline_warnings") or ())
    )
    for warning in (prov.get("detection_baseline_warnings") or ()):
        if str(warning) not in measurement_baseline:
            warns.append(f"Candidate-localization baseline: {warning}")
    # Then the preprocess detrend's own advisories (captured in run() from the
    # kernel): "window yields 1 block ... cannot track drift" means the enabled
    # detrend was a no-op, which nothing downstream can see from the data — the
    # trace simply still drifts — so it must be SAID here or nowhere.
    warns.extend(str(w) for w in (prov.get("detrend_warnings") or ()))
    # In split high-pass runs the primary warning block above belongs to the
    # quantitative trace. Candidate-only detrending can fail differently and that
    # matters to the boundaries, so surface distinct detector-branch advisories too.
    measurement_detrend = set(str(w) for w in (prov.get("detrend_warnings") or ()))
    for warning in (prov.get("detection_detrend_warnings") or ()):
        if str(warning) not in measurement_detrend:
            warns.append(f"Candidate-localization chain: {warning}")
    pw = prov.get("polarity_warning")
    if pw:
        warns.append(str(pw))
    n_fl = int(prov.get("n_filter_limited", 0) or 0)
    if n_fl:
        n = max(len(df), 1)
        warns.append(
            f"{n_fl} of {len(df)} event(s) ({100 * n_fl / n:.0f}%) are "
            "filter-limited: depths attenuated, dwell biased long."
        )
    n_clip = int(prov.get("n_clipped_boundary", 0) or 0)
    if n_clip:
        # One sentence, written in `events`, printed identically by both detectors.
        warns.append(clipped_boundary_warning(n_clip, len(df)))
    # The amplitude-domain sibling of that rule: rows whose depth was read off a rail.
    # Counted from the ROWS that carry the token, never from a parallel computation,
    # so this sentence cannot claim events the table does not flag.
    if not df.empty:
        n_sat = int(np.asarray(has_flag(df["flag"], FLAG_SATURATED), dtype=bool).sum())
        if n_sat:
            warns.append(saturated_warning(n_sat, len(df)))
    # What the CONDITIONING chain cost, which no row can carry: the notch's own
    # startup transient on a trace too short for it to settle, and the depth the
    # smoothers ate. Both are damage done before detect_events saw the trace.
    settling = notch_settling_from_provenance(prov)
    omitted_lines = prov.get("notch_omitted_freqs") or ()
    if prov.get("notch_enabled") and omitted_lines:
        omitted = ", ".join(str(float(f)) for f in omitted_lines)
        applied = ", ".join(str(float(f)) for f in prov["notch_freqs"])
        warns.append(
            f"Requested notch lines {omitted} Hz were omitted because they are at "
            f"or above Nyquist ({float(prov['fs']) / 2:g} Hz). "
            f"Applied notch lines: {applied} Hz."
        )
    if settling is not None and settling[1] > 0:
        settle_s, unsettled_s = settling
        warns.append(
            f"The notch needs {settle_s:.3g} s to settle and this trace is shorter, so "
            f"its first {unsettled_s:.3g} s still carry the filter's own startup "
            "transient (its amplitude scales with the hum removed). Events found in "
            "that head may be the filter ringing rather than translocations."
        )
    smoothing = _smoothing_warning(prov, df)
    if smoothing:
        warns.append(smoothing)
    # High-pass remains scientifically important after the dual-signal fix: row
    # magnitudes are now measured without it, but the event bounds still came from its
    # frequency-shaped trace. Legacy runs without the split stamp retain the stronger
    # candidate-only warning above.
    if prov.get("hp_enabled"):
        warns.append(highpass_measurement_warning(prov))
    # ...and the one conditioning stage that changes the UNIT rather than the value.
    # LAST, because it is the widest claim in the list: the two above say some rows
    # are wrong, this one says the axis is. It fires on an EMPTY table too (unlike the
    # smoothing advisory, which needs dwells to probe): a run that found no events
    # still reports a baseline and a threshold in the wrong units.
    if prov.get("normalize_enabled"):
        warns.append(normalized_units_warning(prov))
    return tuple(warns)


def _resolve_run_baseline(
    params: Mapping[str, Any], target: tuple[str, int], *,
    units: str | None = None, module_id: str | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """``(config, param_source, params_to_record)`` for ONE detector run over ``target``.

    The config and the form's source come from the one resolver
    (`baseline_config_from_params`); the source is then judged for THIS recording by
    `baseline.resolve_param_source`, which is what stamps ``'batch'`` on a recording
    the batch form's windows were not proposed for (S7-DET1). The params recorded on
    the result carry the RESOLVED source and not `BATCH_SCOPE_KEY`: a "Re-run" from a
    stored result then reproduces the same stamp, and a per-load ``doc_id`` never
    reaches a project file. Shared by both threshold detectors so they cannot
    disagree about whose windows these were.
    """
    cfg, source = baseline_config_from_params(params)
    from .reference_context import validate_open_pore_context
    validate_open_pore_context(params, units=units, module_id=module_id)
    source = resolve_param_source(params, target, source)
    record = {k: v for k, v in params.items() if k != BATCH_SCOPE_KEY}
    if "baseline_param_source" in record or BATCH_SCOPE_KEY in params:
        record["baseline_param_source"] = source
    return cfg, source, record


def reconcile_result(
    res: AnalysisResult, events: EventSet | None = None, **changes: Any
) -> AnalysisResult:
    """``res`` re-installed over ``events`` with everything ROW-DERIVED re-derived.

    THE ONE DOOR for a detection whose frame has changed shape or measurement under
    it -- a merge (all / in selection / with next), a hand-added missed event, a
    re-measure. Each of those used to be ``replace(res, payload=new_events)`` at its
    own call site, carrying ``provenance`` and ``warnings`` forward untouched or
    (the re-measure) replacing the warnings with the NEW baseline's alone, and sweep 7
    measured what that cost on the owner's -600 mV recording:

      * Merge all, 317 -> 247 rows: ``provenance['n_events']`` stayed 317, the Stats
        panel read "1 of 317 event(s) were still above threshold..." over a 247-row
        table, and the subpopulation report -- written to be read without the app
        beside it -- printed "CURATION: 71 of the detector's 317 region(s) were
        curated OUT" for a user who had curated out exactly one (S7-DET4).
      * Re-measure depths, low-pass at 2 kHz: FOUR result warnings before, ONE after.
        "3 of 251 event(s) (1%) are filter-limited: depths attenuated" -- a caveat
        about ACCEPTED rows that are still in the table and still exported -- was
        deleted, along with the boundary-clipped sentence, because the install took
        ``warnings=tuple(comparison.warnings)`` = the new `BaselineResult`'s own list
        and nothing else; ``voltage_dep`` then copied the truncated tuple into every
        pooled point's ``detection_warnings`` (S7 MISSED-1). Provenance also kept
        ``n_filter_limited = 3`` over rows flagging 2, into the exported sidecar
        (S7 MISSED-2).

    So this re-derives all of it from the rows, through the SAME two functions the
    detectors use at detection time -- `events.reconcile_row_counts` for the counts
    and :func:`_build_warnings` for the sentences -- and nothing else. What survives
    untouched is everything that is NOT a statement about the rows: the baseline's
    own advisories (``provenance['baseline_warnings']`` -- the re-measure engine
    stamps the NEW estimator's there, exactly as ``detect.py`` stamps the detector's),
    the ``baseline_*`` block, the ``merged`` / ``remeasured`` history records, the
    conditioning stamps and the polarity evidence. The scalars a summary reads
    (``n_events``, ``rate_hz``) follow the rows too.

    ``changes`` are extra `dataclasses.replace` fields the caller owns (the
    re-measure passes its ``params`` and ``baseline_fingerprint``); ``payload`` and
    ``warnings`` are this function's and cannot be overridden through it.

    IT ALSO MATERIALISES ``source`` / ``review``, and that is structural rather than
    remembered. `events.SCHEMA` is the frozen 20 and `characterize` ends with
    ``df[list(SCHEMA)]``, so every path that rebuilds a frame drops anything outside
    it -- which is exactly what makes an old project still loadable, and exactly what
    makes "re-apply the pair" a thing four separate call sites could each forget. They
    all already come through here, so the re-application lives here once.
    `events.ensure_source_status` only FILLS what is missing: a stamped frame is
    returned unchanged, a hand-added row is forced to ``manual`` + ``reviewed`` (the
    ``added`` marker is the pre-existing recorded fact and ``source`` must agree with
    it rather than contradict it), and ``review`` is deliberately left ``pd.NA`` for
    everything else -- a frame that reaches this door without the pair is a detection
    from before the column existed, and nobody recorded whether a person ruled on
    those rows. Saying "unreviewed" for them would claim a fact about a curator's
    attention that no one ever wrote down.

    ``source`` is filled from ``res.module_id``, which is the per-TABLE answer this
    column exists to replace -- but for a legacy frame it is also the only answer
    there is, and combined with ``added`` it is exactly as accurate as what the app
    could already say. Nothing here touches ``accepted``, so no number moves.
    """
    from dataclasses import replace

    if events is None:
        events = res.payload
    if not isinstance(events, EventSet):
        raise TypeError(
            f"reconcile_result needs an EventSet payload, got {type(events).__name__}"
        )
    for key in ("payload", "warnings"):
        if key in changes:
            raise TypeError(f"{key!r} is derived here and cannot be passed as a change")
    df = events.df
    # IN PLACE, like `scoring.ensure_added_column`: the results table, the Event
    # Inspector and the undo recorder share ONE DataFrame object, so a copy would
    # give the columns to nobody. An empty ``module_id`` leaves ``source`` unknown
    # rather than raising -- a result with no id cannot name an author.
    ensure_source_status(df, source=str(res.module_id) if res.module_id else pd.NA)
    prov = reconcile_row_counts(events.provenance, df)
    scalars = dict(res.scalars or {})
    if "n_events" in scalars:
        scalars["n_events"] = float(len(df))
    if "rate_hz" in scalars:
        try:
            scalars["rate_hz"] = float(EventSet(df, prov).rate_hz())
        except ValueError:
            pass   # no analysed duration on record: the old value is the only one there is
    return replace(
        res,
        payload=EventSet(df, prov),
        warnings=_build_warnings(prov, df),
        scalars=scalars,
        **changes,
    )


def stamp_preprocess_provenance(
    prov: dict[str, Any],
    params: Mapping[str, Any],
    fs: float,
    detrend_warns: Sequence[Warning] = (),
) -> dict[str, Any]:
    """Stamp what `preprocess_for_detection` did onto a detector's provenance.

    THE ONE WRITER of the preprocessing block, shared by every detector that runs on
    the conditioned trace (`DetectSchmitt` and `DetectCusum`). It existed as inline
    code in one module while there was one such detector; a second copy is how the two
    would start disagreeing about what "the same chain" means -- and provenance is the
    only thing `detection_signal` has to rebuild the exact trace a row was measured on,
    so a stage missing from one detector's stamp is an event whose depth cannot be
    reproduced.

    Only the ENABLED stages stamp their parameters; a disabled stage records just its
    ``False`` flag, so provenance is never ambiguous about what ran.

    ``detrend_warns`` is the detrend kernel's own advisory list, collected by the
    caller through `baseline.collect_baseline_warnings` around its
    `preprocess_for_detection` call. Empty is a valid value (no advisories); the caller
    must still open that scope, because a warning that dies in the warnings machinery
    while ``detrend_enabled`` reads True is a silent no-op no surface would disclose.

    Mutates and returns ``prov``.
    """
    # Record what preprocessing produced the conditioned signal, so a result is
    # self-describing (the events were detected on a filtered trace, not the raw).
    # Only the enabled stages stamp their parameters; a disabled stage records just
    # its False flag so provenance is never ambiguous about what ran.
    prov["detrend_enabled"] = bool(params.get("detrend_enabled", False))
    if prov["detrend_enabled"]:
        prov["detrend_method"] = str(params.get("detrend_method", "block_median"))
        prov["detrend_window_s"] = float(
            params.get("detrend_window_s", DEFAULT_DETREND_WINDOW_S)
        )
        # The selected method's shape parameters, under their kernel names, so the
        # baseline can be re-derived from provenance alone. Empty for block_median.
        prov.update(detrend_method_kwargs(params))
        # HONEST PROVENANCE for a stage that ran but achieved nothing. The kernel
        # is the authority on its own degeneracy: ConstantBaselineWarning means
        # the window yielded ONE block, the estimate was a flat level, and the
        # detrend changed nothing. Its own words are stamped here so
        # _build_warnings can put them in res.warnings (the Stats panel's
        # surface), and detrend_effective lets a provenance reader see the no-op
        # without parsing prose — detrend_enabled=True alone would claim work
        # that never happened.
        prov["detrend_warnings"] = tuple(str(w) for w in detrend_warns)
        prov["detrend_effective"] = not any(
            isinstance(w, ConstantBaselineWarning) for w in detrend_warns
        )
        # The stage's OTHER honest-failure mode, worse than the no-op: the kernel
        # measured its own output carrying MORE wander than its input (wander
        # faster than the window is phase-inverted by the block median and ADDED
        # back — see baseline.AmplifiedWanderWarning). The sentence itself rides
        # into res.warnings with the rest of detrend_warnings; this key is the
        # structural form, so a provenance reader can see "the detrend made the
        # trace worse" without parsing prose, exactly as detrend_effective works
        # for "the detrend did nothing".
        prov["detrend_amplified"] = any(
            isinstance(w, AmplifiedWanderWarning) for w in detrend_warns
        )

    prov["notch_enabled"] = bool(params.get("notch_enabled", False))
    if prov["notch_enabled"]:
        prov["notch_freq_hz"] = float(
            params.get("notch_freq_hz", DEFAULT_NOTCH_FREQ_HZ)
        )
        prov["notch_harmonics"] = int(
            params.get("notch_harmonics", DEFAULT_NOTCH_HARMONICS)
        )
        prov["notch_q"] = float(params.get("notch_q", DEFAULT_NOTCH_Q))
        # THE RESOLVED LINES, not the request. `notch_frequencies` answers from
        # the same resolver `notch` designs its cascade from, so what provenance
        # says was notched is what touched the data -- including the Nyquist drop,
        # which is invisible in (freq_hz, harmonics) alone. Stamped for BOTH
        # spellings: with an explicit list it echoes the user's own order, and
        # with the series it makes the derived comb explicit for a reader who
        # would otherwise have to re-derive it.
        prov["notch_freqs"] = list(
            notch_frequencies(
                float(fs),
                prov["notch_freq_hz"],
                harmonics=prov["notch_harmonics"],
                freqs=notch_freqs_param(params),
            )
        )
        requested = notch_freqs_param(params)
        if requested is None:
            requested = [
                prov["notch_freq_hz"] * h
                for h in range(1, prov["notch_harmonics"] + 1)
            ]
        prov["notch_requested_freqs"] = list(requested)
        applied = set(prov["notch_freqs"])
        prov["notch_omitted_freqs"] = [f for f in requested if f not in applied]
        # How long this notch needs to forget its own initial state, and how much
        # of that it could not get inside this trace. A high-Q notch rings for
        # ~1/(its width) at each end, and filtered at scipy's default pad that
        # ringing arrived as ACCEPTED EVENTS at the head of the trace. `notch`
        # now pads with its own ring-down, so `notch_unsettled_s` is 0.0 for any
        # ordinary recording -- and when it is not (a hand-cut segment shorter
        # than the ring-down) provenance says so and _build_warnings prints it,
        # instead of the events table carrying filter artefacts as rows.
        settling = notch_settling_from_provenance(prov)
        if settling is not None:
            prov["notch_settle_s"], prov["notch_unsettled_s"] = settling

    prov["median_enabled"] = bool(params.get("median_enabled", False))
    if prov["median_enabled"]:
        prov["median_window_s"] = float(
            params.get("median_window_s", DEFAULT_MEDIAN_WINDOW_S)
        )

    prov["hp_enabled"] = bool(params.get("hp_enabled", False))
    if prov["hp_enabled"]:
        prov["hp_cutoff_hz"] = float(
            params.get("hp_cutoff_hz", DEFAULT_HP_CUTOFF_HZ)
        )
        prov["hp_order"] = int(params.get("hp_order", DEFAULT_HP_ORDER))
        prov["hp_kind"] = str(params.get("hp_kind", DEFAULT_HP_KIND))

    prov["rmean_enabled"] = bool(params.get("rmean_enabled", False))
    if prov["rmean_enabled"]:
        prov["rmean_window_s"] = float(
            params.get("rmean_window_s", DEFAULT_RMEAN_WINDOW_S)
        )

    prov["savgol_enabled"] = bool(params.get("savgol_enabled", False))
    if prov["savgol_enabled"]:
        prov["savgol_window_s"] = float(
            params.get("savgol_window_s", DEFAULT_SAVGOL_WINDOW_S)
        )
        prov["savgol_polyorder"] = int(
            params.get("savgol_polyorder", DEFAULT_SAVGOL_POLYORDER)
        )

    prov["normalize_enabled"] = bool(params.get("normalize_enabled", False))
    if prov["normalize_enabled"]:
        prov["normalize_mode"] = str(params.get("normalize_mode", "zscore"))
        if prov["normalize_mode"] == MODEL_INPUT_MODE:
            # Only 'model_input' reads it, so only 'model_input' stamps it -- a
            # clip bound recorded against a z-score run would be a parameter that
            # did nothing, which is how a reader ends up believing the wrong
            # kernel ran.
            prov["normalize_clip_sigma"] = float(
                params.get("normalize_clip_sigma", MODEL_INPUT_CLIP_SIGMA)
            )
    return prov


def _recharacterize_highpass_candidates(
    raw_signal: np.ndarray,
    fs: float,
    params: Mapping[str, Any],
    df: pd.DataFrame,
    prov: dict[str, Any],
    *,
    baseline_config: Any,
    baseline_param_source: str,
    pore_id: str,
    recording: str,
    units: str,
) -> tuple[pd.DataFrame, dict[str, Any], tuple[Warning, ...]]:
    """Measure high-pass-localised spans on the equivalent non-high-passed trace.

    The detector call has already finished, so ``df[start_idx, end_idx]`` is the
    candidate-localisation result and every detector counter/threshold in ``prov`` is
    final. This function replaces only the row measurements and the top-level
    measurement-domain baseline/noise/polarity. Candidate-domain values are retained
    under explicit ``detection_*`` keys, making both halves independently auditable.

    Sequential execution is deliberate. The candidate detector's full work arrays have
    returned before the characterization trace is built, so a large recording does not
    require both filter pipelines simultaneously. The baseline caches remove entries as
    soon as those temporary trace objects die (see ``baseline._BaselineCache``).
    """
    if not prov.get("hp_enabled"):
        return df, prov, ()

    # Preserve every baseline fact that produced the threshold/localisation decision
    # before replacing the top-level baseline namespace with the physical measurement
    # baseline. A nested block avoids inventing a second hand-maintained list of keys.
    detection_baseline = {
        key: value for key, value in prov.items() if key.startswith("baseline_")
    }
    detection_noise = float(prov["noise_rms_pA"])
    detection_noise_method = str(prov.get("noise_method", "MAD (filters.noise_rms)"))
    detection_polarity = int(prov["polarity"])
    candidate_polarity_source = str(prov.get("polarity_source", "unknown"))
    candidate_polarity_measured = prov.get("polarity_measured")
    candidate_polarity_evidence = prov.get("polarity_evidence")
    candidate_polarity_warning = prov.get("polarity_warning")

    prov["highpass_localization_only"] = True
    prov["characterization_hp_omitted"] = True
    prov["detection_signal_domain"] = "conditioned trace including high-pass"
    prov["characterization_signal_domain"] = (
        "same conditioned trace with high-pass omitted"
    )

    # ``detection_signal`` is the single provenance-driven builder used by every
    # post-hoc remeasurement. The stamp above switches it to the characterization
    # domain while preserving legacy high-pass results that lack the stamp.
    with collect_baseline_warnings() as measurement_detrend_warns:
        measurement = detection_signal(raw_signal, fs, prov)

    requested_polarity = params.get("polarity")
    base_result = characterisation_baseline(
        measurement,
        fs,
        config=baseline_config,
        # Auto polarity must be measured in the physical, non-high-passed domain.
        # The one-sided baseline mask therefore gets the user's override when one
        # exists and otherwise uses the baseline door's robust multi-scale vote.
        polarity=(None if requested_polarity is None else int(requested_polarity)),
        param_source=baseline_param_source,
    )
    base = np.asarray(base_result.baseline, dtype=np.float64)
    current = np.asarray(measurement, dtype=np.float64)
    measurement_noise = float(noise_rms(current - base))
    if not np.isfinite(measurement_noise) or measurement_noise <= 0:
        raise ValueError(
            f"measurement noise RMS is {measurement_noise!r}; cannot characterize "
            "high-pass-localised events on a trace with no measurable noise."
        )

    # Resolve the detector direction again on the trace whose amplitudes will be
    # reported. A high-pass step response is biphasic; letting its ringing cast the
    # only polarity vote can point localization at the edge lobes rather than at the
    # events. Schmitt historically uses k_in for this vote; CUSUM uses the app-wide
    # 5-sigma polarity threshold.
    polarity_k = (
        float(prov.get("k_in", 5.0))
        if prov.get("detector") == "schmitt_hysteresis"
        else 5.0
    )
    measured_polarity, measurement_polarity_evidence = detect_polarity(
        current, base, k=polarity_k
    )
    measurement_polarity = (
        int(measured_polarity)
        if requested_polarity is None
        else int(requested_polarity)
    )

    # The first candidate pass is intentionally allowed to vote on its own trace: in
    # the common case that saves a whole extra conditioning/baseline pass. If the two
    # domains disagree, however, those are candidates in the wrong direction. Signal
    # the adapter to discard them and repeat candidate localization with the physical
    # branch's sign. The retry remains sequential, so the two full-size arrays and
    # baselines are never retained together.
    if requested_polarity is None and detection_polarity != measurement_polarity:
        raise _HighpassPolarityMismatch(
            measurement_polarity,
            measurement_polarity_evidence,
            candidate_polarity=detection_polarity,
        )

    starts = df["start_idx"].to_numpy(dtype=np.int64)
    ends = df["end_idx"].to_numpy(dtype=np.int64)
    measured = characterize(
        current,
        base,
        starts,
        ends,
        fs,
        measurement_polarity,
        noise_rms=measurement_noise,
        filter_rise_s=prov.get("filter_rise_s"),
        max_dwell_s=prov.get("max_dwell_s"),
        clip_bounds=(0, int(current.size)),
        pore_id=pore_id,
        recording=recording,
        units=units,
    )
    # Preserve the detector functions' public contract: their final conservative
    # dwell-vs-rise check is OR-ed with characterize's half-height-width test.
    rise_s = prov.get("filter_rise_s")
    if rise_s is not None and not measured.empty:
        limited = measured["dwell_s"].to_numpy(dtype=float) < 2.0 * float(rise_s)
        measured["filter_limited"] = (
            measured["filter_limited"].to_numpy(dtype=bool) | limited
        )

    # Top-level baseline/noise/polarity always describe the values in the event rows.
    # Candidate-domain values remain explicit and reproduce the threshold decision.
    for key in tuple(prov):
        if key.startswith("baseline_"):
            prov.pop(key, None)
    prov.update(base_result.provenance())
    prov["baseline_method"] = str(base_result.config.method)
    prov["baseline_provided"] = False
    prov["baseline_warnings"] = list(base_result.warnings)
    prov["baseline_median_pA"] = float(base_result.median_pA)
    prov["detection_baseline"] = detection_baseline
    prov["detection_baseline_median_pA"] = float(
        detection_baseline.get("baseline_median_pA", np.nan)
    )
    prov["detection_baseline_warnings"] = list(
        detection_baseline.get("baseline_warnings") or ()
    )

    prov["detection_noise_rms_pA"] = detection_noise
    prov["detection_noise_method"] = detection_noise_method
    prov["measurement_noise_rms_pA"] = measurement_noise
    prov["noise_rms_pA"] = measurement_noise
    prov["noise_method"] = (
        "MAD (filters.noise_rms) on characterization trace without high-pass"
    )

    # Preserve what an unconstrained vote on the high-pass candidate trace said; it is
    # useful evidence of ringing, but it is not allowed to choose the event direction.
    prov["candidate_signal_polarity_source"] = candidate_polarity_source
    prov["candidate_signal_polarity_measured"] = candidate_polarity_measured
    prov["candidate_signal_polarity_evidence"] = candidate_polarity_evidence
    prov["candidate_signal_polarity_warning"] = candidate_polarity_warning
    prov["detection_polarity"] = measurement_polarity
    prov["detection_polarity_source"] = (
        "characterization_trace_measured"
        if requested_polarity is None else "user_provided"
    )
    prov["detection_polarity_measured"] = candidate_polarity_measured
    prov["detection_polarity_evidence"] = candidate_polarity_evidence
    prov["detection_polarity_warning"] = candidate_polarity_warning
    prov["characterization_polarity_measured"] = int(measured_polarity)
    prov["characterization_polarity_evidence"] = measurement_polarity_evidence
    prov["polarity_domain_disagreement"] = (
        candidate_polarity_measured in (-1, 1)
        and int(candidate_polarity_measured) != int(measured_polarity)
    )
    measurement_polarity_warning = _polarity_warning_from_evidence(
        measurement_polarity_evidence,
        measurement_polarity,
        measured=requested_polarity is None,
    )
    # Top-level polarity describes both the sign used to localize and the sign used
    # for quantitative rows. ``polarity_requested`` remains the user's request (None
    # for Auto), even when the adapter had to force the resolved sign on a retry.
    prov["polarity"] = measurement_polarity
    prov["polarity_requested"] = requested_polarity
    prov["polarity_source"] = (
        "measured" if requested_polarity is None else "provided"
    )
    prov["polarity_measured"] = int(measured_polarity)
    prov["polarity_evidence"] = measurement_polarity_evidence
    prov["polarity_warning"] = measurement_polarity_warning

    # Detection thresholds stay in the candidate domain. Top-level noise describes
    # the row SNR denominator, so spell the threshold source explicitly rather than
    # leaving a reader to (incorrectly) multiply by the replacement value.
    prov["threshold_noise_source"] = "detection_noise_rms_pA"
    if prov.get("detector") == "cusum_sequential":
        prov["cusum_sigma_source"] = (
            "detection_noise_rms_pA: MAD residual on high-pass candidate trace"
        )

    prov.update(reconcile_row_counts(prov, measured))
    # The candidate-domain report cannot describe quantitative measurements rebuilt
    # on the non-high-pass trace. Reassess against that branch's actual baseline.
    from .measurement_quality import build_measurement_quality
    prov["measurement_quality"] = build_measurement_quality(current, base, measured, fs, prov)
    return measured, prov, tuple(measurement_detrend_warns)


class _HighpassPolarityMismatch(RuntimeError):
    """Internal retry signal when high-pass ringing reversed Auto polarity."""

    def __init__(
        self,
        polarity: int,
        evidence: Mapping[str, Any],
        *,
        candidate_polarity: int,
    ) -> None:
        self.polarity = int(polarity)
        self.evidence = dict(evidence)
        self.candidate_polarity = int(candidate_polarity)
        super().__init__(
            "candidate and characterization traces resolved opposite polarities: "
            f"{self.candidate_polarity:+d} vs {self.polarity:+d}"
        )


def _polarity_warning_from_evidence(
    evidence: Mapping[str, Any], polarity: int, *, measured: bool
) -> str | None:
    """The adapter-level polarity advisory for the characterization-domain vote."""
    if not measured:
        return None
    verdict = str(evidence.get("verdict", "undetermined"))
    if verdict == "undetermined":
        return (
            "polarity is undetermined on the characterization trace (no coherent "
            "excursions above threshold); proceeding with "
            f"polarity={int(polarity):+d}."
        )
    if evidence.get("decided_by") == "coherent_samples":
        return (
            "polarity run vote was ambiguous on the characterization trace and the "
            "direction was taken from coherent signal mass instead "
            f"(confidence {float(evidence.get('confidence', 0.0)):.2f})."
        )
    if (
        "decided_by" in evidence
        and not evidence.get("corroborated_by_samples", False)
    ):
        return (
            "polarity was measured at low confidence on the characterization trace "
            f"({float(evidence.get('confidence', 0.0)):.2f}); proceeding with "
            f"polarity={int(polarity):+d}."
        )
    return None


def _run_highpass_dual_path(
    ctx: Any,
    *,
    detector: Any,
    module_id: str,
    baseline_config: Any,
    baseline_param_source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Localize with high-pass, then measure the same spans without it.

    Auto polarity gets one optimistic candidate pass. The characterization branch
    independently measures the physical direction; if high-pass ringing reversed the
    candidate vote, both temporary traces are released and localization is repeated
    once with that resolved sign. This makes the common case two sequential pipelines
    and the pathological case four, rather than retaining two recording-sized signals
    and baselines at the same time.
    """
    ch = ctx.channel
    fs = float(ch.sample_rate_hz)
    raw = np.asarray(ch.data)
    forced_polarity: int | None = None
    initial_candidate_polarity: int | None = None
    characterization_resolved_polarity: int | None = None

    for attempt in range(2):
        if attempt == 0:
            candidate_lo, candidate_hi = 0.02, 0.44
            measure_lo, measure_hi = 0.46, 0.90
        else:
            candidate_lo, candidate_hi = 0.90, 0.94
            measure_lo, measure_hi = 0.94, 0.99

        # The conditioning recipe remains the user's recipe on a retry. Only the
        # detector's direction is forced; in particular an Auto detrend still uses
        # its own recorded Auto semantics rather than becoming a different filter.
        with collect_baseline_warnings() as detrend_warns:
            candidate_signal = preprocess_for_detection(raw, fs, ctx.params)
        detector_params = dict(ctx.params)
        if forced_polarity is not None:
            detector_params["polarity"] = forced_polarity

        with baseline_progress_scope(
            lambda f, m, lo=candidate_lo, hi=candidate_hi: ctx.report_progress(
                lo + (hi - lo) * f, f"detecting candidates ({m})")
        ):
            df, prov = detector(
                candidate_signal,
                fs,
                units=ch.units,
                recording=ctx.meta.path.name,
                pore_id=str(ctx.params.get("pore_id", "")),
                baseline_config=baseline_config,
                baseline_param_source=baseline_param_source,
                **_detect_kwargs(detector_params, module_id),
            )
        stamp_preprocess_provenance(prov, ctx.params, fs, detrend_warns)
        # On a retry the low-level engine quite correctly records the forced value as
        # its request. The result contract records what the USER requested; reset it
        # before detection_signal reconstructs the no-high-pass branch so an Auto
        # detrend is replayed as Auto.
        prov["polarity_requested"] = ctx.params.get("polarity")
        candidate_detrend_warns = tuple(detrend_warns)

        del candidate_signal
        ctx.report_progress(measure_lo, "measuring candidates without high-pass")
        try:
            with baseline_progress_scope(
                lambda f, m, lo=measure_lo, hi=measure_hi: ctx.report_progress(
                    lo + (hi - lo) * f, f"measuring candidates ({m})")
            ):
                df, prov, measurement_detrend_warns = (
                    _recharacterize_highpass_candidates(
                        raw,
                        fs,
                        ctx.params,
                        df,
                        prov,
                        baseline_config=baseline_config,
                        baseline_param_source=baseline_param_source,
                        pore_id=str(ctx.params.get("pore_id", "")),
                        recording=ctx.meta.path.name,
                        units=ch.units,
                    )
                )
        except _HighpassPolarityMismatch as exc:
            if forced_polarity is not None:
                raise RuntimeError(
                    "high-pass polarity retry did not converge on the "
                    "characterization-domain direction"
                ) from exc
            # Keep only scalars. Holding the exception itself would retain its
            # traceback, which retains the helper frame and therefore the full
            # measurement signal + float64 baseline while the candidate retry runs.
            initial_candidate_polarity = exc.candidate_polarity
            characterization_resolved_polarity = exc.polarity
            forced_polarity = exc.polarity
            exc.__traceback__ = None
            ctx.report_progress(
                measure_hi,
                "high-pass ringing reversed Auto polarity; repeating localization",
            )
            continue

        if prov.get("detrend_enabled"):
            prov["detection_detrend_warnings"] = tuple(
                str(w) for w in candidate_detrend_warns
            )
            prov["detection_detrend_effective"] = not any(
                isinstance(w, ConstantBaselineWarning)
                for w in candidate_detrend_warns
            )
            prov["detection_detrend_amplified"] = any(
                isinstance(w, AmplifiedWanderWarning)
                for w in candidate_detrend_warns
            )
        # Primary detrend provenance describes the trace the quantitative rows use.
        stamp_preprocess_provenance(
            prov,
            ctx.params,
            fs,
            measurement_detrend_warns,
        )
        if initial_candidate_polarity is not None:
            prov["polarity_retry_after_highpass_mismatch"] = True
            prov["candidate_initial_polarity"] = initial_candidate_polarity
            prov["characterization_resolved_polarity"] = (
                characterization_resolved_polarity
            )
            prov["candidate_localization_attempts"] = 2
        else:
            prov["polarity_retry_after_highpass_mismatch"] = False
            prov["candidate_localization_attempts"] = 1
        return df, prov

    raise RuntimeError("high-pass dual-signal localization exhausted its retry")


class DetectSchmitt:
    """Schmitt-trigger event detection as an :class:`AnalysisModule`."""

    id = "detect.schmitt"
    label = "Detect events (Schmitt)"
    version = "1.0"
    accepts_kinds = (ChannelKind.CURRENT,)  # gate: the corpus also carries voltage channels
    requires_time_base = True               # dwell is undefined without fs; GUI disables when None

    def run(self, ctx: Any) -> AnalysisResult:
        ch = ctx.channel
        if ch.sample_rate_hz is None:
            raise RuntimeError(
                "detector requires a sample rate, but this channel has none "
                "(requires_time_base is True); the GUI should have disabled it."
            )

        # WHICH LOAD, not which path. The one writer of this stamp lives in
        # core.identity; a detector that re-derived `str(ctx.meta.path.resolve())` here
        # would stamp an identity shared with every other load of the same file, and a
        # result landing after the recording was closed and reopened would install itself
        # onto a document it was never computed from. AST-guarded in the suite.
        doc_id = doc_id_for_meta(ctx.meta)
        channel = int(ch.info.index)
        fp = _fingerprint(ctx.params)

        # Baseline-wander removal (if enabled) runs on the calibrated signal BEFORE
        # detection; detect_events still owns the low-pass. preprocess_for_detection
        # is the shared transform the preview and surrogate studies reuse verbatim.
        ctx.report_progress(0.0, "detecting events")
        # ONE resolution of the characterisation-baseline knobs, from the one
        # resolver, into the one config object that also reaches the ML
        # characteriser, `truth` and every row rebuild. This module holds no
        # baseline window constant and must never acquire one.
        base_cfg, base_source, params_record = _resolve_run_baseline(
            ctx.params, (doc_id, channel), units=ch.units, module_id=self.id)
        # LIVENESS FOR THE ONE JOB THAT IS SLOW AT SCALE (GA5-01). On a whole-channel
        # detect of the owner's 606 MB ATP recording this call runs ~30 s, and this
        # method used to report progress only at 0.0 and 1.0 -- the status-bar
        # QProgressBar sat determinate at 0% for the entire run, indistinguishable
        # from a stall. `detect_events` is not this module's to change, but ~86% of
        # its time at that scale is `characterisation_baseline` [MEASURED on a
        # 50 M-sample trace: door 20.5 s of run 23.9 s], and the baseline module has
        # an ambient progress channel for exactly this shape of problem (the same
        # thread-local pattern the cancel token uses, so no unowned signature has to
        # grow a callback). The door's own 0..1 is mapped onto [0.02, 0.88] of this
        # job -- its measured share -- and the fractions it delivers are monotone by
        # construction; the remaining Schmitt + characterise stages ride from 0.88
        # to the 1.0 reported below. On a small file the whole ladder flies by in
        # milliseconds; at the ceiling the bar now moves every few seconds.
        highpass_split = bool(ctx.params.get("hp_enabled"))
        if highpass_split:
            df, prov = _run_highpass_dual_path(
                ctx,
                detector=detect_events,
                module_id=self.id,
                baseline_config=base_cfg,
                baseline_param_source=base_source,
            )
        else:
            # Collect the preprocessing kernels' own advisories (thread-local sink,
            # see collect_baseline_warnings). This is deliberately the historical
            # one-array path when high-pass is off; its numerics stay untouched.
            with collect_baseline_warnings() as detrend_warns:
                signal = preprocess_for_detection(
                    ch.data, ch.sample_rate_hz, ctx.params)
            with baseline_progress_scope(
                lambda f, m: ctx.report_progress(
                    0.02 + 0.86 * f, f"detecting candidates ({m})")
            ):
                df, prov = detect_events(
                    signal,
                    ch.sample_rate_hz,
                    units=ch.units,
                    recording=ctx.meta.path.name,
                    pore_id=str(ctx.params.get("pore_id", "")),
                    baseline_config=base_cfg,
                    baseline_param_source=base_source,
                    **_detect_kwargs(ctx.params, self.id),
                )
            stamp_preprocess_provenance(
                prov, ctx.params, float(ch.sample_rate_hz), detrend_warns
            )
        ctx.report_progress(1.0, f"{len(df)} event(s)")

        # THE DETECTOR'S OWN BOUNDS, recorded before any hand edit can move them.
        # Stamped HERE and only here (detection is the one moment they are the
        # detector's), so the Event Inspector's Revert can put an edited event back
        # where the detector actually put it. `stamp_original_bounds` refuses to
        # overwrite an existing pair, and it no-ops on a frame with no rows.
        #
        # WHO FOUND IT, AND THAT NOBODY HAS RULED ON IT YET -- the same moment and
        # the same reason. This line is the one instant at which every row in the
        # frame is provably THIS module's, so ``source`` is stamped ``self.id``
        # verbatim. Before it, "which detector produced this row" was answerable
        # only from the result's ``module_id``, which is a statement about the
        # TABLE and stops being true of every row the first time somebody hand-adds
        # a missed event into the same result. ``review`` is stamped ``unreviewed``
        # because a detector emits CANDIDATES (notes §7: "automatic detection should
        # generate candidate events; user curation should determine the final
        # dataset") -- and ``accepted`` is NOT touched, so every export, figure and
        # fit reads exactly the rows it read before. MEASURED on the owner's 15 s
        # -1000 mV recording: 115 events, sha256 of the frozen 20 columns
        # a880ca47…28dc, identical before and after this line existed.
        stamp_acquisition_bandwidth(prov, ctx)
        stamp_reference_availability(prov, ctx)
        events = EventSet(stamp_source_status(stamp_original_bounds(df), self.id), prov)

        spans = OverlayHint(
            OverlayKind.SPAN,
            starts=df["start_idx"].to_numpy(dtype=np.int64),
            stops=df["end_idx"].to_numpy(dtype=np.int64),
            label="events",
        )

        rise = prov.get("filter_rise_s")
        scalars = {
            "n_events": float(len(df)),
            "rate_hz": float(events.rate_hz()),
            "noise_rms": float(prov["noise_rms_pA"]),
            "median_baseline": float(prov["baseline_median_pA"]),
            "filter_rise_s": float(rise) if rise else float("nan"),
        }

        return AnalysisResult(
            module_id=self.id,
            module_version=self.version,
            doc_id=doc_id,
            channel=channel,
            params=params_record,
            calibration_fingerprint=fp,
            payload=events,
            overlays=(spans,),
            scalars=scalars,
            warnings=_build_warnings(prov, df),
            baseline_fingerprint=prov.get("baseline_fingerprint"),
        )



class DetectCusum:
    """Sequential-CUSUM event detection as an :class:`AnalysisModule`.

    NOT `analysis.cusum`, which is the sub-state LEVEL FITTER that runs INSIDE an
    event a detector already found. This is a DETECTOR: it decides where the events
    are. The two share only the name of the statistic.

    Shaped after :class:`DetectSchmitt`, not
    :class:`~nanoscope.analysis.ml_modules.MLDetect`: it runs on the same
    `preprocess_for_detection` output, resolves the same shared `BaselineConfig`,
    stamps the same preprocessing block through the same
    `stamp_preprocess_provenance`, and returns the same 20-column frame. A user can
    therefore compare a Schmitt row and a CUSUM row on one recording and know the
    depths were measured against the same open pore -- which is the whole reason the
    baseline config is shared rather than re-derived per detector.

    THE THRESHOLDS ARE DIMENSIONLESS. ``cusum_k`` and ``cusum_h`` are in units of the
    MAD noise sigma the detector measures on the baseline-subtracted residual (the
    same sigma the table's ``snr`` column is divided by), NOT picoamps. Do not feed
    this detector a `robust_normalize`d trace: that kernel divides by 30, which would
    make the effective thresholds 30x too large and it would find nothing.
    """

    id = "detect.cusum"
    label = "Detect events (CUSUM)"
    version = "1.0"
    accepts_kinds = (ChannelKind.CURRENT,)  # gate: the corpus also carries voltage channels
    requires_time_base = True               # dwell is undefined without fs

    def run(self, ctx: Any) -> AnalysisResult:
        ch = ctx.channel
        if ch.sample_rate_hz is None:
            raise RuntimeError(
                "CUSUM detector requires a sample rate, but this channel has none "
                "(requires_time_base is True); the GUI should have disabled it."
            )

        # WHICH LOAD, not which path -- the one writer of this stamp is core.identity.
        doc_id = doc_id_for_meta(ctx.meta)
        channel = int(ch.info.index)
        fp = _fingerprint(ctx.params)

        ctx.report_progress(0.0, "detecting events (CUSUM)")
        base_cfg, base_source, params_record = _resolve_run_baseline(
            ctx.params, (doc_id, channel), units=ch.units, module_id=self.id)

        # Same progress ladder and same reasoning as DetectSchmitt.run: the
        # characterisation baseline is ~86% of the wall time at scale and has an
        # ambient progress channel, so its own 0..1 is mapped onto [0.02, 0.88] of
        # this job. The CUSUM core itself is 0.23 s on an 11.3 M-sample trace, so the
        # remaining span is almost entirely `characterize`.
        highpass_split = bool(ctx.params.get("hp_enabled"))
        if highpass_split:
            df, prov = _run_highpass_dual_path(
                ctx,
                detector=detect_events_cusum,
                module_id=self.id,
                baseline_config=base_cfg,
                baseline_param_source=base_source,
            )
        else:
            with collect_baseline_warnings() as detrend_warns:
                signal = preprocess_for_detection(
                    ch.data, ch.sample_rate_hz, ctx.params)
            with baseline_progress_scope(
                lambda f, m: ctx.report_progress(
                    0.02 + 0.86 * f, f"detecting candidates ({m})")
            ):
                df, prov = detect_events_cusum(
                    signal,
                    ch.sample_rate_hz,
                    units=ch.units,
                    recording=ctx.meta.path.name,
                    pore_id=str(ctx.params.get("pore_id", "")),
                    baseline_config=base_cfg,
                    baseline_param_source=base_source,
                    **_detect_kwargs(ctx.params, self.id),
                )
            stamp_preprocess_provenance(
                prov, ctx.params, float(ch.sample_rate_hz), detrend_warns
            )
        ctx.report_progress(1.0, f"{len(df)} event(s)")

        # Same two stamps, same moment, same reasons as the Schmitt path -- see
        # there. ``self.id`` is ``detect.cusum``, so a table that pooled a Schmitt
        # run and a CUSUM run can say per ROW which of them found each event; that
        # is the whole difference between this column and the result's module_id.
        stamp_acquisition_bandwidth(prov, ctx)
        stamp_reference_availability(prov, ctx)
        events = EventSet(stamp_source_status(stamp_original_bounds(df), self.id), prov)

        spans = OverlayHint(
            OverlayKind.SPAN,
            starts=df["start_idx"].to_numpy(dtype=np.int64),
            stops=df["end_idx"].to_numpy(dtype=np.int64),
            label="events",
        )

        rise = prov.get("filter_rise_s")
        scalars = {
            "n_events": float(len(df)),
            "rate_hz": float(events.rate_hz()),
            "noise_rms": float(prov["noise_rms_pA"]),
            "median_baseline": float(prov["baseline_median_pA"]),
            "filter_rise_s": float(rise) if rise else float("nan"),
        }

        return AnalysisResult(
            module_id=self.id,
            module_version=self.version,
            doc_id=doc_id,
            channel=channel,
            params=params_record,
            calibration_fingerprint=fp,
            payload=events,
            overlays=(spans,),
            scalars=scalars,
            warnings=_build_warnings(prov, df),
            baseline_fingerprint=prov.get("baseline_fingerprint"),
        )


def register_all() -> None:
    """Register the channel-driven modules. Idempotent: a second call (e.g. a
    second test session sharing the process-global registry) is a no-op rather
    than the duplicate-id error ``register_module`` raises on a fresh id.

    Registers all three built-in detectors -- Schmitt, sequential CUSUM and the
    runtime-loaded Keras ML detector (:class:`~nanoscope.analysis.ml_modules.MLDetect`)
    -- so the one ``register_all`` at app start wires up every detector. Importing ``ml_modules``
    does NOT import TensorFlow (it only probes for it), so this stays cheap."""
    if DetectSchmitt.id not in ANALYSIS_REGISTRY:
        register_module(DetectSchmitt())
    if DetectCusum.id not in ANALYSIS_REGISTRY:
        register_module(DetectCusum())

    from .ml_modules import register_ml
    register_ml()


# ---------------------------------------------------------------------------
# Controller-invoked fit wrappers (D3): events-table in, AnalysisResult out.
# These are NOT AnalysisModules -- they never touch a raw channel.
# ---------------------------------------------------------------------------
def _unpack_events(events: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Duck-type an EventSet, an AnalysisResult (payload=EventSet), a bare
    DataFrame, or an explicit ``(DataFrame, provenance)`` pair into
    ``(df, provenance)``.

    The pair form exists for the GUI: the Plots window is pushed a snapshot
    DataFrame and its detection provenance side by side, and must resolve t_min
    from the same dict the controller does."""
    if isinstance(events, tuple) and len(events) == 2 and isinstance(events[0], pd.DataFrame):
        return events[0], dict(events[1] or {})
    payload = getattr(events, "payload", None)
    if payload is not None:  # an AnalysisResult carrying an EventSet
        events = payload
    df = getattr(events, "df", events)
    prov = dict(getattr(events, "provenance", {}) or {})
    return df, prov


def _accepted(df: pd.DataFrame) -> pd.DataFrame:
    if "accepted" in df.columns:
        return df[df["accepted"].to_numpy(dtype=bool)]
    return df


# ---------------------------------------------------------------------------
# ONE resolution of "what did the detector actually let through?".
#
# Every consumer of a distribution fit -- fit_dwell/fit_depth below, the static
# figures in plot.figures, and the interactive Plots window -- must answer the
# same three questions the same way, or the same events yield two different
# taus depending on which window you read them in (that was bug B1):
#
#   * where is the detection floor t_min?                 -> resolve_t_min
#   * is the sample truncated on the right as well?       -> dwell_fit_inputs
#   * are filter-limited events in or out?                -> dwell_fit_inputs
#
# These helpers are that single answer. Do not re-derive any of it at a call
# site; call these and show what they report.
# ---------------------------------------------------------------------------

#: Said (and shown) whenever the detection floor had to be guessed from the data
#: instead of read from provenance. ``plot.figures`` raises this as a
#: RuntimeWarning; the GUI prints it next to the number it changed.
NO_TMIN_WARNING = (
    "provenance has no 'min_dwell_s'; using the shortest observed dwell "
    "({t_min:g} s) as the detection limit. The censored fit's tau depends on this -- "
    "pass an EventSet from detect() to get it right."
)


def resolve_t_min(
    prov: Mapping[str, Any], *, observed_min: float | None = None
) -> tuple[float, str, str | None]:
    """The detection floor for a dwell fit: ``(t_min, source, note)``.

    Resolution order, best evidence first:

    1. ``provenance['min_dwell_s']`` -- the gate the detector actually applied.
       ``source='min_dwell_s'``, no note.
    2. ``1/fs`` -- a dwell cannot be shorter than one sample. ``source='1/fs'``,
       with a note: this is a floor, not *the* floor, and under-corrects if the
       detector really did gate at 50 us.
    3. ``observed_min`` (the caller's shortest observed dwell), if given.
       ``source='observed_min'``, with :data:`NO_TMIN_WARNING` as the note.

    Raises:
        ValueError: when none of the three is available -- the fit would have to
            invent its own censoring model, which is how a 25% error in tau gets
            published.
    """
    md = prov.get("min_dwell_s")
    if md and float(md) > 0:
        return float(md), "min_dwell_s", None
    fs = prov.get("fs")
    if fs:
        t_min = 1.0 / float(fs)
        return t_min, "1/fs", (
            f"provenance has no 'min_dwell_s'; falling back to one sample period "
            f"({t_min:g} s) as the detection limit. If the detector gated on a longer "
            f"min_dwell_s than this, tau is under-corrected."
        )
    if observed_min is not None and float(observed_min) > 0:
        t_min = float(observed_min)
        return t_min, "observed_min", NO_TMIN_WARNING.format(t_min=t_min)
    raise ValueError(
        "cannot infer t_min from provenance (no positive min_dwell_s and no fs); "
        "pass t_min explicitly to fit_dwell."
    )


def _t_min_from_prov(prov: Mapping[str, Any]) -> float:
    """The detector's real resolution: the shortest dwell it could report.

    Back-compatible thin wrapper over :func:`resolve_t_min` (same order, same
    raise); prefer ``resolve_t_min`` where the *source* and the note matter.
    """
    return resolve_t_min(prov)[0]


def depth_floor_from_provenance(
    prov: Mapping[str, Any],
    *,
    column: str = "frac_depth",
    baseline_pA: Any = None,
):
    """The detector's effective depth floor in the units of ``column``, or None.

    A split high-pass run has no defensible measurement-domain floor: its exit
    threshold is in candidate-trace pA while the fitted depths were measured on the
    no-high-pass trace. They are related by the filter response, not by one scalar, so
    this returns ``None`` whenever ``highpass_localization_only`` (or its durable
    ``characterization_hp_omitted`` alias) is stamped.

    The Schmitt trigger keeps a run only while ``dev > k_out * noise``, and
    ``depth_pA`` is the MEDIAN in-event deviation, so the reported depths are
    truncated at roughly ``threshold_out_pA`` -- not at ``threshold_in_pA``, and
    not at zero. In pA that IS the floor, and it is exact.

    ``frac_depth`` is the awkward one, and it used to be WRONG here. The column is
    ``depth_pA / |baseline_pA|`` where ``baseline_pA`` is each event's OWN local
    open-pore level (the mean of the baseline track across that event's samples --
    see ``characterize``). This function divided by ``provenance['baseline_median_pA']``
    instead, the median of the baseline track over the WHOLE recording. Those are
    different denominators, and on a real recording they are not close: on the user's
    -600 mV file the global median baseline is 34.6 pA while the per-event levels run
    39-199 pA (median 122), so the floor came out 3.5x too high and **965 of 1579
    detected events (61.1%) sat below their own supposed floor** -- an impossibility,
    since every one of them cleared the trigger. It then fired ``DEPTH_FLOOR_WARNING``
    on that fiction (``mass_below_floor = 0.42``), telling the user a clean fit was
    floor-biased.

    So the floor is now computed with the SAME denominator the column uses. Pass the
    ``baseline_pA`` of exactly the rows being fit and the answer is
    ``threshold_out_pA / max|baseline_pA|`` -- the smallest of the per-event floors,
    which is the only single number that is a floor for *every* row in the sample
    (a row with a bigger open pore has a *lower* fractional floor, so the minimum is
    the one nothing can be below). With it, 0 of those 1579 events are below the
    floor, matching ``depth_pA``, which was right all along.

    Without ``baseline_pA`` there is no honest answer for ``frac_depth`` and this
    returns ``None`` (the caller then simply does not run the floor check) rather
    than resurrecting the global-median number.

    Args:
        prov: The EventSet provenance mapping.
        column: The depth-like column the floor is wanted in.
        baseline_pA: The ``baseline_pA`` values of the rows being fit. Required for
            ``column='frac_depth'``; ignored for the pA columns, whose floor is
            ``threshold_out_pA`` outright.
    """
    if (
        prov.get("highpass_localization_only")
        or prov.get("characterization_hp_omitted")
    ):
        return None
    thr = prov.get("threshold_out_pA")
    if thr is None or not np.isfinite(float(thr)) or float(thr) <= 0:
        return None
    if column in ("depth_pA", "max_dev_pA", "mean_dev_pA"):
        return float(thr)
    if column != "frac_depth":
        return None
    if baseline_pA is None:
        return None
    base = np.abs(np.asarray(baseline_pA, dtype=np.float64).ravel())
    base = base[np.isfinite(base) & (base > 0)]
    if base.size == 0:
        return None
    return float(thr) / float(base.max())


@dataclass
class DwellFitInputs:
    """Exactly what a dwell fit should be handed, and why (B1 / B4a / B4c).

    Attributes:
        dwell: The dwell times to fit, in seconds, after the accepted-only and
            filter-limited filters.
        t_min: The detection floor to condition the likelihood on.
        t_min_source: Where ``t_min`` came from -- ``'min_dwell_s'``, ``'1/fs'``,
            ``'observed_min'``, ``'2*filter_rise_s'`` or ``'explicit'``.
        t_max: Upper truncation point to MODEL, or None.
        n_excluded_filter_limited: Events dropped as filter-limited -- their width
            at half their own peak is within ~2x the filter rise time, so their
            depth is attenuated and their (longer) measured dwell is biased. The
            test is on that width, not on ``dwell_s``, because the same filter
            inflates ``dwell_s``; see ``characterize``.
        n_input: Rows considered before any of these filters.
        notes: Human-readable, display-ready warnings. Every one of these must
            end up somewhere the user can see next to the number it qualifies.
    """

    dwell: np.ndarray
    t_min: float
    t_min_source: str
    t_max: float | None = None
    use_filter_limited: bool = False
    n_excluded_filter_limited: int = 0
    n_input: int = 0
    notes: list[str] = field(default_factory=list)

    def meta(self) -> dict[str, Any]:
        """The provenance block to stamp onto a FitResult's ``meta``."""
        return {
            "t_min": float(self.t_min),
            "t_min_source": self.t_min_source,
            "t_max": self.t_max,
            "n_excluded_filter_limited": int(self.n_excluded_filter_limited),
            "use_filter_limited": bool(self.use_filter_limited),
            "n_input_events": int(self.n_input),
        }


@dataclass
class DepthFitInputs:
    """What a depth fit should be handed (B4b / B4c). See :class:`DwellFitInputs`."""

    values: np.ndarray
    depth_floor: float | None = None
    use_filter_limited: bool = False
    n_excluded_filter_limited: int = 0
    n_input: int = 0
    notes: list[str] = field(default_factory=list)

    def meta(self) -> dict[str, Any]:
        return {
            "depth_floor": self.depth_floor,
            "n_excluded_filter_limited": int(self.n_excluded_filter_limited),
            "use_filter_limited": bool(self.use_filter_limited),
            "n_input_events": int(self.n_input),
        }


def _filter_limited_mask(df: pd.DataFrame) -> np.ndarray:
    """The ``filter_limited`` flag as a clean bool mask.

    A MISSING flag is not a set flag: ``np.asarray([nan], dtype=bool)`` is
    ``True``, so a hand-built frame with an unpopulated column would silently
    exclude every event from the fit. NaN reads as False here.
    """
    if "filter_limited" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    col = df["filter_limited"]
    if pd.api.types.is_bool_dtype(col):
        return col.to_numpy(dtype=bool)
    return np.asarray(col.where(col.notna(), False), dtype=bool)


def _right_truncation(
    full: pd.DataFrame, prov: Mapping[str, Any], kept: np.ndarray
) -> tuple[float | None, list[str]]:
    """Detect a right-truncated dwell sample: ``(t_max_to_model, notes)``.

    Two ways the long tail goes missing, and they are not the same thing:

    * The detector was given ``max_dwell_s`` -- a hard, *known* cut, so it can
      be modelled: the returned ``t_max`` goes straight into the likelihood.
    * The user curated long events out by hand -- the cut point is not knowable
      (the boundary lies somewhere between the longest kept and the shortest
      rejected event), so this only warns. Modelling it at a guessed t_max would
      trade a known bias for an unknown one.
    """
    notes: list[str] = []
    md = prov.get("max_dwell_s")
    if md is not None and float(md) > 0 and np.isfinite(float(md)):
        return float(md), notes

    if "accepted" not in full.columns or "dwell_s" not in full.columns or kept.size == 0:
        return None, notes
    rejected = full[~full["accepted"].to_numpy(dtype=bool)]
    if rejected.empty:
        return None, notes
    hi = float(kept.max())
    long_rejects = rejected["dwell_s"].to_numpy(dtype=np.float64)
    n_long = int(np.count_nonzero(long_rejects > hi))
    if n_long:
        notes.append(
            f"RIGHT CENSORING (curation): {n_long} rejected event(s) are longer than the "
            f"longest event in this fit ({hi:g} s), so the fitted sample is cut off on the "
            f"right as well as the left. The likelihood models the LEFT truncation only, so "
            f"tau is biased SHORT (measured -3.7% when the cut sits at 5 tau, -24% at 2.5 "
            f"tau). Re-accept those events, or set max_dwell_s so the cut is a known number "
            f"the fit can correct for."
        )
    return None, notes


def dwell_fit_inputs(
    events: Any,
    *,
    accepted_only: bool = True,
    use_filter_limited: bool = False,
    t_min: float | None = None,
) -> DwellFitInputs:
    """Resolve the dwell sample, its detection floor and its truncation (B1/B4a/B4c).

    ``events`` may be an :class:`~nanoscope.analysis.events.EventSet`, an
    ``AnalysisResult`` carrying one, or a ``(DataFrame, provenance)`` pair -- the
    GUI holds bare frames, the controller holds EventSets, and both must get the
    same answer.

    Args:
        events: The event table (+ provenance) to fit.
        accepted_only: Use curated-in events only, as every figure does.
        use_filter_limited: Keep events the detector marked ``filter_limited``
            (their width at half their own peak is within ~2x the filter rise
            time). Default **False**: their depth is attenuated and their dwell
            biased long, so they are excluded and counted. The width can never
            exceed the dwell, so every row that SURVIVES the exclusion has
            ``dwell_s >= 2*filter_rise_s``: the detection floor moves up to
            ``2*filter_rise_s`` and this function moves ``t_min`` with it,
            otherwise the exclusion would re-introduce exactly the
            left-truncation bias the censored MLE exists to remove.
        t_min: Override the resolved detection floor (rarely wanted).
    """
    df, prov = _unpack_events(events)
    full = df
    sel = _accepted(full) if accepted_only else full
    n_input = int(len(sel))

    notes: list[str] = []
    limited = _filter_limited_mask(sel)
    n_excluded = 0
    if not use_filter_limited and limited.any():
        n_excluded = int(limited.sum())
        sel = sel[~limited]
    dwell = sel["dwell_s"].to_numpy(dtype=np.float64)

    observed_min = float(dwell.min()) if dwell.size else None
    if t_min is not None:
        t_min_used, source = float(t_min), "explicit"
    else:
        t_min_used, source, note = resolve_t_min(prov, observed_min=observed_min)
        if note:
            notes.append(note)

    if n_excluded:
        rise = prov.get("filter_rise_s")
        raised = None
        if rise and float(rise) > 0:
            # The exclusion is a hard cut at 2*rise; that IS the floor now. Never
            # claim a floor above the smallest surviving event, or the fit's own
            # guardrail would (rightly) refuse the sample.
            raised = 2.0 * float(rise)
            if observed_min is not None:
                raised = min(raised, observed_min)
        if raised is not None and raised > t_min_used:
            notes.append(
                f"{n_excluded} filter-limited event(s) excluded from the fit (dwell within 2x "
                f"the {float(rise):g} s filter rise time: depth attenuated, dwell biased long). "
                f"The detection floor for this fit is therefore t_min = {raised:g} s, not "
                f"min_dwell_s = {t_min_used:g} s."
            )
            t_min_used, source = float(raised), "2*filter_rise_s"
        else:
            notes.append(
                f"{n_excluded} filter-limited event(s) excluded from the fit (depth attenuated, "
                f"dwell biased long); the detection floor is unchanged at t_min = {t_min_used:g} s."
            )

    t_max, right_notes = _right_truncation(full, prov, dwell)
    if t_max is not None and dwell.size and float(dwell.max()) > t_max * (1 + 1e-9):
        # A max_dwell_s is on the books but this sample still holds longer events
        # (the caller is fitting the un-curated frame). Nothing was truncated, so
        # modelling a cut that did not happen would be its own lie.
        right_notes.append(
            f"provenance records max_dwell_s = {t_max:g} s, but this sample contains events "
            f"up to {float(dwell.max()):g} s, so the right limit was NOT applied to it and is "
            f"not modelled here."
        )
        t_max = None
    notes.extend(right_notes)

    return DwellFitInputs(
        dwell=dwell, t_min=float(t_min_used), t_min_source=source, t_max=t_max,
        use_filter_limited=bool(use_filter_limited),
        n_excluded_filter_limited=n_excluded, n_input=n_input, notes=notes,
    )


def depth_fit_inputs(
    events: Any,
    *,
    column: str = "frac_depth",
    accepted_only: bool = True,
    use_filter_limited: bool = False,
) -> DepthFitInputs:
    """Resolve the depth sample and the detector's floor under it (B4b/B4c).

    The floor is derived from the per-event ``baseline_pA`` of exactly the rows that
    survive the accepted / filter-limited filters -- the same denominator
    ``frac_depth`` itself used -- so it cannot contradict the sample it qualifies.
    See :func:`depth_floor_from_provenance`.
    """
    df, prov = _unpack_events(events)
    sel = _accepted(df) if accepted_only else df
    n_input = int(len(sel))

    notes: list[str] = []
    limited = _filter_limited_mask(sel)
    n_excluded = 0
    if not use_filter_limited and limited.any():
        n_excluded = int(limited.sum())
        sel = sel[~limited]
        notes.append(
            f"{n_excluded} filter-limited event(s) excluded from the fit: the low-pass never "
            f"let the current reach the true blockade level, so their depths are attenuated "
            f"(biased shallow) and would drag the shallowest component down."
        )
    values = sel[column].to_numpy(dtype=np.float64)
    base = (
        sel["baseline_pA"].to_numpy(dtype=np.float64)
        if "baseline_pA" in sel.columns
        else None
    )
    return DepthFitInputs(
        values=values,
        depth_floor=depth_floor_from_provenance(prov, column=column, baseline_pA=base),
        use_filter_limited=bool(use_filter_limited),
        n_excluded_filter_limited=n_excluded, n_input=n_input, notes=notes,
    )


def fit_dwell(
    events: Any,
    *,
    doc_id: str,
    channel: int,
    t_min: float | None = None,
    n_components: int = 1,
    efficiency: Any = None,
    use_filter_limited: bool = False,
    calibration_fingerprint: tuple[float, float] = _DEFAULT_FINGERPRINT,
    n_boot: int = 1000,
    seed: int | None = 0,
    detection_module_id: str | None = None,
) -> AnalysisResult:
    """Fit dwell times of the accepted events with a censored exponential MLE.

    The sample, its detection floor and any right truncation all come from
    :func:`dwell_fit_inputs`, which is the SAME resolution the interactive Plots
    window uses -- so the tau printed in one window is the tau printed in the
    other. Filter-limited events are excluded by default (``use_filter_limited``
    puts them back), and every choice that moved a number lands in
    ``FitResult.meta`` and in ``AnalysisResult.warnings``.

    ``detection_module_id`` is the ``module_id`` of the stored detection ``events`` came
    from; the fit records it (:data:`FIT_DETECTION_KEY`) so it can later be told apart from
    the fit of another table. Left out, the fit records none and
    :func:`judge_stored_fit` will not vouch for it.

    Returns a derived ``AnalysisResult`` (``module_id='fit.dwell'``) whose payload
    is the :class:`~nanoscope.analysis.stats.FitResult`; the controller stores it
    via ``Session.add_result``.
    """
    inp = dwell_fit_inputs(events, use_filter_limited=use_filter_limited, t_min=t_min)
    fit = fit_dwell_exponential(
        inp.dwell, t_min=inp.t_min, t_max=inp.t_max, n_components=n_components,
        efficiency=efficiency, n_boot=n_boot, seed=seed,
    )
    stamp_fit_provenance(fit, inp, detection_module_id=detection_module_id)
    return _fit_result("fit.dwell", fit, doc_id, channel,
                       {"t_min": float(inp.t_min), "t_max": inp.t_max,
                        "n_components": int(n_components),
                        "use_filter_limited": bool(use_filter_limited)},
                       calibration_fingerprint)


def fit_depth(
    events: Any,
    *,
    doc_id: str,
    channel: int,
    n_components: int = 1,
    use_filter_limited: bool = False,
    calibration_fingerprint: tuple[float, float] = _DEFAULT_FINGERPRINT,
    n_boot: int = 400,
    seed: int | None = 0,
    detection_module_id: str | None = None,
) -> AnalysisResult:
    """Fit a Gaussian mixture to the accepted events' ``frac_depth``.

    The detector's depth floor (``threshold_out_pA`` divided by the largest
    per-event ``|baseline_pA|`` in the fitted sample -- the same denominator
    ``frac_depth`` uses; see :func:`depth_floor_from_provenance`) is handed to the
    fit, which warns when the shallowest component sits on it -- an untruncated
    mixture fit to a floor-truncated sample returns mu too high and sigma too low.

    ``detection_module_id`` is recorded as it is by :func:`fit_dwell`.

    Returns a derived ``AnalysisResult`` (``module_id='fit.depth'``) carrying the
    :class:`~nanoscope.analysis.stats.FitResult`.
    """
    inp = depth_fit_inputs(events, use_filter_limited=use_filter_limited)
    fit = fit_depth_gaussians(inp.values, n_components=n_components,
                              depth_floor=inp.depth_floor, n_boot=n_boot, seed=seed)
    stamp_fit_provenance(fit, inp, detection_module_id=detection_module_id)
    return _fit_result("fit.depth", fit, doc_id, channel,
                       {"n_components": int(n_components),
                        "use_filter_limited": bool(use_filter_limited)},
                       calibration_fingerprint)


def fit_levels(
    events: Any,
    signal: Any,
    fs: float,
    *,
    doc_id: str,
    channel: int,
    baseline: Any,
    polarity: int,
    provenance: Mapping[str, Any] | None = None,
    config: CusumConfig = DEFAULT_CUSUM,
    accepted_only: bool = True,
    sigma: float | None = None,
    tau: float | None = None,
    calibration_fingerprint: tuple[float, float] = _DEFAULT_FINGERPRINT,
    baseline_fingerprint: str | None = None,
    cancel: Any = None,
) -> AnalysisResult:
    """Resolve the sub-state levels inside each accepted event with CUSUM.

    A third derived fit alongside :func:`fit_dwell` and :func:`fit_depth`, and a
    plain function for the same reason they are (PORT_PLAN D3): it consumes an
    EVENT TABLE, not a channel, so it does not fit the channel-in
    ``AnalysisModule.run(ctx)`` contract. It differs from those two in needing the
    SAMPLES as well -- a level lives inside an event, and the event table only
    carries the event's boundaries.

    ``signal`` must be the trace the events were detected in. If detection ran on
    a low-passed copy, pass THAT copy: fitting levels on the unfiltered trace while
    the boundaries came from the filtered one puts every transition a filter rise
    time away from the edge it belongs to.

    Returns a derived ``AnalysisResult`` (``module_id='fit.levels'``) whose payload
    is the LEVEL_SCHEMA frame. Both staleness stamps are carried through: a level
    table is measured against a baseline and a calibration exactly as an event
    depth is, and is exactly as wrong when either is revised.

    ``sigma`` / ``tau`` override the two estimators
    :func:`~nanoscope.analysis.cusum.fit_event_levels` otherwise resolves for itself;
    both are RECORDED in ``params`` whichever way they were arrived at, which is the
    point. A level table describes spans, spans move, and re-fitting one edited event
    has to segment it at the same significance scale as the untouched events it is
    spliced beside -- so the resolved pair has to be readable off the result rather
    than living for the duration of one call. It survives a project round-trip with
    the rest of ``params``, so a fit restored from disk still states what measured it.

    Keys are omitted rather than written as NaN when a value was never resolved (an
    empty event set resolves no ``tau``): absence is this app's "not measured", and a
    NaN in ``params`` is not valid JSON for the project writer.
    """
    levels = fit_event_levels(
        signal, fs, events, baseline=baseline, polarity=polarity,
        sigma=sigma, tau=tau,
        provenance=provenance, config=config, accepted_only=accepted_only,
        cancel=cancel,
    )
    # What the fit ACTUALLY ran with, off the frame it returned -- not the arguments,
    # which are None on the common path. See cusum.SIGMA_ATTR / TAU_ATTR.
    estimators: dict[str, float] = {}
    for key, attr in (("sigma_pA", CUSUM_SIGMA_ATTR), ("tau_samples", CUSUM_TAU_ATTR)):
        try:
            value = float(levels.attrs.get(attr))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            estimators[key] = value
    warnings: list[str] = []
    if levels.empty:
        warnings.append("no events were fitted: the table is empty or all events are rejected")
    return AnalysisResult(
        module_id="fit.levels",
        module_version=CUSUM_VERSION,
        doc_id=doc_id,
        channel=int(channel),
        params={
            "step_sigma": float(config.step_sigma),
            "threshold_sigma": float(config.threshold_sigma),
            "min_level_samples": int(config.min_level_samples),
            "warmup_samples": int(config.warmup_samples),
            "max_levels": int(config.max_levels),
            "accepted_only": bool(accepted_only),
            "cusum_fingerprint": config.fingerprint(),
            **estimators,
        },
        calibration_fingerprint=tuple(float(v) for v in calibration_fingerprint),  # type: ignore[arg-type]
        payload=levels,
        overlays=(),
        scalars=levels_summary(levels),
        warnings=tuple(warnings),
        baseline_fingerprint=baseline_fingerprint,
    )


#: ``FitResult.meta`` key naming the STORED DETECTION a distribution fit was made on: the
#: ``module_id`` of the table (``detect.schmitt``, ``detect.cusum``, ``detect.keras``,
#: ``detect.reconciled``, ...) whose accepted events were handed to the fitter.
#:
#: The result store keeps ONE ``fit.dwell`` and ONE ``fit.depth`` per (recording, channel), so
#: every table of a recording shares one fit slot, and until this key existed nothing recorded
#: which of them a fit belonged to: the Results dock's Fits tab printed Schmitt's n = 316 under
#: CUSUM's table, and a Fit that half-failed left one table's dwell fit beside another's depth
#: fit. It is an ADDED ``meta`` key, not a new store key or module id: ``fit_to_json`` writes
#: ``meta`` wholesale, so the project file carries it, while an older build reading that file
#: carries the unknown entry through untouched and finds its fits where it always did. A fit
#: WITHOUT the key (written by an earlier build, or filed by the Plots window) is "unrecorded".
FIT_DETECTION_KEY = "detection_module_id"


def stamp_fit_provenance(
    fit: FitResult,
    inp: DwellFitInputs | DepthFitInputs,
    *,
    detection_module_id: str | None = None,
) -> None:
    """Fold the resolution's provenance + notes into the FitResult.

    The notes go FIRST: an exclusion or a right-censoring warning is the reason
    the number moved, so it must be the first thing under the estimate in
    ``summary()``. A right-censoring warning also amends ``censoring`` -- the fit
    must stop claiming left-truncation-only the moment the data say otherwise.

    ``detection_module_id`` is the stored detection the sample was taken from
    (:data:`FIT_DETECTION_KEY`); a caller that does not say leaves the fit unrecorded.
    """
    meta = fit.meta
    for key, value in inp.meta().items():
        meta.setdefault(key, value)
    meta["n_excluded_filter_limited"] = int(inp.n_excluded_filter_limited)
    meta["use_filter_limited"] = bool(inp.use_filter_limited)
    if detection_module_id:
        meta[FIT_DETECTION_KEY] = str(detection_module_id)
    if isinstance(inp, DwellFitInputs):
        meta["t_min"] = float(inp.t_min)
        meta["t_min_source"] = inp.t_min_source
    warnings_ = list(meta.get("warnings", []))
    meta["warnings"] = list(inp.notes) + warnings_
    if any(n.startswith("RIGHT CENSORING") for n in inp.notes):
        cens = str(meta.get("censoring", ""))
        if "right" not in cens.lower():
            meta["censoring"] = (
                cens + " -- but the SAMPLE is also right-censored by curation "
                "(see the warning below); the model does not correct for that"
            )


#: The answers of :func:`judge_stored_fit`.
FIT_CURRENT = "current"                        # it belongs to this table, as the table stands
FIT_OTHER_TABLE = "other_table"                # it names a different table
FIT_CALIBRATION_MOVED = "calibration_moved"    # fitted under another gain/offset than the table
FIT_EVENTS_CHANGED = "events_changed"          # the table's accepted count is not the one fitted
FIT_UNRECORDED = "unrecorded"                  # it names no table, so nothing can vouch for it

#: What Export figure says for each of the two guards it has always had -- the wording of its
#: status line, which tests pin, so it lives beside the rule instead of at the call site.
_EXPORT_WORDS = {
    FIT_CALIBRATION_MOVED: "saved fit predates a calibration change",
    FIT_EVENTS_CHANGED: "saved fit is stale (the accepted events changed since it ran)",
}


class FitVerdict(NamedTuple):
    """What :func:`judge_stored_fit` found, with the numbers a sentence about it needs.

    ``made_on`` is the table the fit names (``""`` when it names none), ``n_fit`` the accepted
    events it says it was fitted to (None when it did not record the count) and ``n_now`` the
    accepted events the table holds now.
    """

    reason: str
    made_on: str = ""
    n_fit: int | None = None
    n_now: int = 0

    @property
    def current(self) -> bool:
        return self.reason == FIT_CURRENT


def _fit_counts(stored: AnalysisResult, table: AnalysisResult) -> tuple[int | None, int]:
    """``(accepted events the fit says it used, accepted events ``table`` has now)``."""
    n_fit = (getattr(stored.payload, "meta", None) or {}).get("n_input_events")
    n_now = int(table.payload.df["accepted"].to_numpy(dtype=bool).sum())
    return (None if n_fit is None else int(n_fit)), n_now


def _fit_staleness(stored: AnalysisResult, table: AnalysisResult) -> str:
    """The two guards Export figure has always applied: ``FIT_CALIBRATION_MOVED``,
    ``FIT_EVENTS_CHANGED`` or ``""``. Calibration first, and a fit that carries no
    ``n_input_events`` cannot disagree on the count -- the order and the leniency are the
    export's, kept verbatim."""
    if tuple(stored.calibration_fingerprint) != tuple(table.calibration_fingerprint):
        return FIT_CALIBRATION_MOVED
    n_fit, n_now = _fit_counts(stored, table)
    if n_fit is not None and n_fit != n_now:
        return FIT_EVENTS_CHANGED
    return ""


def stored_fit_problem(stored: AnalysisResult, table: AnalysisResult) -> str:
    """Why ``stored`` cannot stand for the events ``table`` holds now, in the figure export's
    own words -- or ``""`` when neither of its two guards objects.

    This is the rule ``MainWindow._stored_fit`` applies before a saved fit is allowed onto an
    exported histogram: a fit taken under another gain/offset describes different depths, and a
    fit of a different number of accepted events (the table was curated, merged or re-run
    since) is "worse than a documented refit". It says nothing about WHICH table the fit was
    made on -- that is :func:`judge_stored_fit`, which applies these same two guards and the
    lineage on top.
    """
    return _EXPORT_WORDS.get(_fit_staleness(stored, table), "")


def judge_stored_fit(stored: AnalysisResult, table: AnalysisResult) -> FitVerdict:
    """Does the stored dwell/depth fit ``stored`` describe the table ``table`` on screen?

    THE RULE THE RESULTS DOCK'S FITS TAB PRINTS BY: the export's own two guards
    (:func:`stored_fit_problem`) extended with the lineage the fit records
    (:data:`FIT_DETECTION_KEY`). Checked in this order, because each answer is more specific
    than the next and says what to do:

    1. ``FIT_OTHER_TABLE`` -- it names a different table than ``table``, whatever the counts
       (two tables can hold the same number of accepted events by coincidence);
    2. ``FIT_CALIBRATION_MOVED`` -- fitted under another gain/offset than the table was measured
       under;
    3. ``FIT_EVENTS_CHANGED`` -- the accepted count is not the one the fit used (curation, a
       merge, a re-run with other settings, or an Undo of any of them since);
    4. ``FIT_UNRECORDED`` -- it names no table at all;
    5. ``FIT_CURRENT`` -- otherwise.

    THE CONSERVATIVE RULE FOR A FIT THAT NAMES NO TABLE. Saved projects written before the stamp
    existed hold fits with no lineage, and so does anything the Plots window files. Which table
    such a fit was made on is not recoverable from the file: a Schmitt fit and a CUSUM fit of the
    same recording carry the same keys, and the count that the export trusts can coincide across
    tables. So an unrecorded fit is never ASSUMED to match -- it is ``FIT_UNRECORDED`` even when
    the calibration and the count agree, and the caller is expected to show nothing and say why.
    It is not discarded: it stays in the store and in the project, and the export (whose rule
    ``stored_fit_problem`` is unchanged) keeps using it exactly as before. Pressing Fit makes a
    fit that records its table.
    """
    meta = getattr(stored.payload, "meta", None) or {}
    made_on = str(meta.get(FIT_DETECTION_KEY) or "")
    n_fit, n_now = _fit_counts(stored, table)

    def verdict(reason: str) -> FitVerdict:
        return FitVerdict(reason, made_on, n_fit, n_now)

    if made_on and made_on != table.module_id:
        return verdict(FIT_OTHER_TABLE)
    moved = _fit_staleness(stored, table)
    if moved:
        return verdict(moved)
    return verdict(FIT_CURRENT if made_on else FIT_UNRECORDED)


def _fit_result(
    module_id: str,
    fit: FitResult,
    doc_id: str,
    channel: int,
    params: dict[str, Any],
    fingerprint: tuple[float, float],
) -> AnalysisResult:
    scalars = {k: float(v) for k, v in fit.params.items()}
    scalars.update(aic=float(fit.aic), bic=float(fit.bic), n=float(fit.n))
    return AnalysisResult(
        module_id=module_id,
        module_version="1.0",
        doc_id=doc_id,
        channel=int(channel),
        params=params,
        calibration_fingerprint=tuple(float(v) for v in fingerprint),  # type: ignore[arg-type]
        payload=fit,
        overlays=(),
        scalars=scalars,
        warnings=tuple(fit.meta.get("warnings", ())),
    )
