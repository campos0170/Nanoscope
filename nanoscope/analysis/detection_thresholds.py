"""What a classical detector COMPARED the signal with, as lines a trace can carry.

"Draw the threshold" sounds like one horizontal line at ``k_in`` sigma. It is not, and the
reason is the whole design of this module. Both classical detectors work on

    dev = polarity * (loc - base_loc)

where ``loc`` is the localization signal (``lowpass(preprocess(raw))``) and ``base_loc`` is
`baseline.characterisation_baseline` of THAT signal -- a full-length moving estimate, not a
constant. Schmitt (`detect._hysteresis_runs`) takes every run where ``dev > k_out*sigma`` and
keeps the runs whose peak also passes ``k_in*sigma``: an event STARTS AND ENDS at the exit
level, and the enter level only decides whether the run counts. So in current units both are
TRACKS, ``base_loc(t) + polarity*k*sigma``, and a flat line drawn at ``median + 5 sigma`` would
sit off the level the detector really used by however far the open pore drifted. CUSUM has no
current threshold at all: evidence accumulates while ``dev > k*sigma`` and an event is called
when that evidence reaches ``h`` sigma-samples. Its one drawable level is that drift rail;
``h`` is an amount of evidence, not a current, and is never drawn as one.

THE BASELINE ARRAY IS NOT STORED, ONLY ITS RECIPE. A table carries its estimator in the
``baseline_*`` provenance namespace, so the track is REPRODUCED -- from the table's own
provenance, never from the live form, which may describe the next run -- and then CHECKED:
the reproduced global sigma must be the ``noise_rms_pA`` the detector stamped, and the reproduced
median the ``baseline_median_pA`` it stamped. A reproduction that does not land on the
detector's own numbers is a line the detector never used, and :class:`ThresholdsUnavailable`
says so instead of drawing it.

Schmitt can additionally compare against a local noise track. Its noise policy and
version travel with the table and are replayed here; an old table without that record
uses the historical global sigma. Decimation is applied to each complete comparison
rail, so a change in noise cannot disappear between the baseline's knots.

AND SOMETIMES THE HONEST ANSWER IS "NOT ON THIS TRACE". With high-pass, baseline-drift
removal or Normalize in the chain, the localization signal does not sit at the raw trace's
level (USER_GUIDE section 5: with high-pass on, the thresholds act on the high-passed
localization branch, near zero, while the trace on screen is at the open pore). A rail drawn
there would mark crossings the drawn trace never made, so these refuse with the reason.

Nothing here imports Qt. The view is told ``(label, idx, values)`` knots and decides how to
paint them; the refusal and the description are sentences the panel shows as they are.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
from typing import Any, Mapping

import numpy as np

# THE HEAVY IMPORTS ARE LOCAL, deliberately. `gui.overlays` imports this module for the three
# rail labels at app start, and `analysis.baseline` is 5.5k lines that pull SciPy -- the same
# reason `ChannelPane.set_layer_data` imports it where it is used. A trace that never shows a
# threshold must not pay for one.

__all__ = [
    "RAIL_ENTER",
    "RAIL_EXIT",
    "RAIL_DRIFT",
    "ThresholdRail",
    "ThresholdTracks",
    "ThresholdsUnavailable",
    "threshold_refusal",
    "threshold_signal_key",
    "detection_threshold_tracks",
    "decimate_extrema",
]

#: The rails' IDENTITIES (``OverlayHint.label``), shared with the renderer that picks each
#: one's dash from it. Spelled once, here, beside the code that decides what each one means.
RAIL_ENTER = "threshold.enter"
RAIL_EXIT = "threshold.exit"
RAIL_DRIFT = "threshold.drift"

_SCHMITT = "schmitt_hysteresis"
_CUSUM = "cusum_sequential"

#: How close a reproduced sigma must land on the stamped one. The rebuilt localization
#: signal is the float32 the app caches for every row rebuild while the detector held the
#: low-pass output in float64; that costs ~1e-7 relatively. A different estimator, chain or
#: calibration moves sigma by percent. 1e-3 sits between the two by three orders each way.
_REPRODUCE_RTOL = 1e-3

#: Stages that move the localization signal OFF the raw trace's level, each with the clause
#: that says why, read after "Detection thresholds are not drawn: ".
_LEVEL_CHANGING_STAGES: tuple[tuple[str, str], ...] = (
    ("hp_enabled",
     "this table's thresholds acted on the high-passed localization signal, which sits "
     "near zero rather than at the open pore drawn here, so a line on this trace would "
     "mark crossings the detector never saw."),
    ("detrend_enabled",
     "this table was detected after baseline-drift removal, so its thresholds follow the "
     "flattened signal, not the drifting trace drawn here."),
    ("normalize_enabled",
     "this table was detected on a Normalized signal, whose thresholds are in sigmas of "
     "that signal, not in the units drawn here."),
)


class ThresholdsUnavailable(ValueError):
    """The thresholds cannot be drawn honestly; ``str(exc)`` is the sentence to show."""


@dataclass(frozen=True)
class ThresholdRail:
    """One line, as ``OverlayHint`` knots: int64 sample indices and absolute current there.

    ``values`` are in the LOCALIZATION signal's coordinates (which, by the refusals above, is
    the raw trace's level). A baseline-subtracted view subtracts its own display baseline at
    ``idx``, exactly as it does for the estimated-baseline rail.
    """

    label: str
    name: str
    idx: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ThresholdTracks:
    rails: tuple[ThresholdRail, ...]
    #: What is drawn, in one sentence, for the status bar.
    description: str


def threshold_refusal(module_id: str, prov: Mapping[str, Any] | None, *,
                      n_samples: int | None = None) -> str | None:
    """Why this table's thresholds cannot be drawn on its trace, or ``None`` when they can.

    Answered from the module and the provenance alone -- the cheap half, asked before any
    signal is rebuilt. The reproduction check in :func:`detection_threshold_tracks` is the
    expensive half and has its own refusals.
    """
    from .baseline import BaselineConfig
    from .modules import preprocess_params_from_provenance

    prov = dict(prov or {})
    detector = prov.get("detector")
    if module_id == "detect.keras" or str(detector or "").startswith("ml"):
        return ("an ML table has no detection threshold of this kind: the model scores "
                "windows of the signal, it does not compare the current with a level.")
    if module_id == "detect.manual" or detector == "manual":
        return "a hand-made table has no detector, so it has no threshold."
    if module_id == "detect.reconciled":
        return ("a reconciled table combines rows from several detectors, so there is no "
                "single threshold to draw. Show one of its source tables instead.")
    if detector not in (_SCHMITT, _CUSUM):
        return "this table's detector compares no current level that can be drawn."
    chain = preprocess_params_from_provenance(prov)
    for key, why in _LEVEL_CHANGING_STAGES:
        if chain.get(key) or (key == "hp_enabled" and (
                prov.get("highpass_localization_only")
                or prov.get("characterization_hp_omitted"))):
            return why
    # `remeasure.REMEASURED_KEY`; present (even as an empty record) once a re-measure ran.
    if prov.get("remeasured") is not None:
        return ("this table's depths were re-measured against another open-pore estimator, "
                "so the baseline its thresholds were set against is no longer recorded.")
    try:
        cfg = BaselineConfig.reproducible_from(prov)
    except ValueError as exc:
        return f"its open-pore reference cannot be restored ({exc})."
    if cfg is None:
        return ("this table does not record which open-pore estimator set its thresholds "
                "(it predates that record, or its baseline was supplied by hand), so the "
                "line cannot be rebuilt.")
    if prov.get("noise_rms_pA") is None or prov.get("polarity") not in (-1, 1):
        return "this table does not record its noise sigma and polarity."
    if n_samples is not None and prov.get("n_samples") is not None \
            and int(prov["n_samples"]) != int(n_samples):
        return (f"this table was detected on {int(prov['n_samples'])} samples and the "
                f"channel now has {int(n_samples)}.")
    if detector == _CUSUM and prov.get("cusum_k") is None:
        return "this CUSUM table does not record its drift k."
    if detector == _SCHMITT:
        from .detection_noise import NOISE_ALGORITHM_VERSION
        policy = prov.get("noise_policy", "global")
        if policy not in ("auto", "global", "local"):
            return "this table records an unknown detection noise policy."
        revision = prov.get("noise_algorithm_version")
        if "noise_policy" in prov and (type(revision) is not int or revision != NOISE_ALGORITHM_VERSION):
            return "this table's detection noise algorithm revision cannot be reproduced."
        if "noise_policy" in prov and prov.get("noise_policy_effective") not in ("global", "local"):
            return "this table does not record the effective detection noise policy."
        if "noise_policy" in prov:
            try:
                value = prov["noise_window_s"]
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise ValueError("noise window must be a number")
                window = float(value)
            except (KeyError, TypeError, ValueError, OverflowError):
                return "this table does not record a valid detection noise window."
            if not np.isfinite(window) or window <= 0:
                return "this table does not record a valid detection noise window."
        if "noise_policy" not in prov and prov.get("noise_policy_effective") == "local":
            return "this table's local detection noise policy is incomplete."
        if ("noise_policy" not in prov
                and (prov.get("noise_profile_sha256") is not None or (
                    any(key in prov for key in ("noise_profile_tile_ends", "noise_profile_scales_pA"))
                    and (prov.get("threshold_in_pA") is None or prov.get("threshold_out_pA") is None)))):
            return "this table's local detection noise policy is incomplete."
        thresholds = [prov.get("k_in"), prov.get("k_out")]
        try:
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in thresholds):
                raise ValueError("threshold must be a number")
            thresholds = [float(value) for value in thresholds]
            if not np.isfinite(thresholds).all():
                raise ValueError("threshold must be finite")
        except (ValueError, TypeError, OverflowError):
            return "this table does not record finite Schmitt threshold multipliers."
        if not 0 < thresholds[1] < thresholds[0]:
            return "this table's Schmitt threshold multipliers are inconsistent."
    return None


def threshold_signal_key(prov: Mapping[str, Any]) -> tuple[Any, ...]:
    """Everything the reproduced rails depend on, as a hashable key for a caller's cache.

    Curation edits change none of it (they move rows, not the detector's comparison), so a
    panel that keys its cache on this recomputes only when a different table is shown.
    """
    from .modules import detection_signal_key

    p = dict(prov or {})
    base = tuple(sorted((k, repr(v)) for k, v in p.items()
                        if str(k).startswith("baseline_") and k != "baseline_health"))
    noise = tuple(sorted((k, repr(v)) for k, v in p.items()
                         if str(k).startswith(("noise_policy", "noise_window_", "noise_local_", "noise_profile_"))
                         or k == "noise_algorithm_version"))
    return (detection_signal_key(p), base, p.get("detector"), p.get("polarity"),
            p.get("polarity_requested"), p.get("noise_rms_pA"), p.get("threshold_in_pA"),
            p.get("threshold_out_pA"), p.get("k_in"), p.get("k_out"), p.get("cusum_k"),
            p.get("cusum_h"), p.get("n_samples"), p.get("fs"), noise)


#: Knots `decimate_extrema` keeps per bin: its min, its max and its largest step, each with
#: the sample before it.
_KNOTS_PER_BIN = 6


def decimate_extrema(track: np.ndarray, max_points: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """``(idx, value)`` knots of a full-length track: per bin, its MIN, its MAX and its LARGEST
    SAMPLE-TO-SAMPLE STEP, each with the sample just before it.

    The baseline rail's uniform knots (`baseline.decimate_track`) are right for an envelope
    with no spike to lose; a threshold rail is that same envelope plus a constant, but what
    a user checks it against is a brief excursion of the signal, and the moving baseline
    under it can step where the event mask hands the short pass a change. Uniform knots land
    either side of such a step and draw it as a ramp -- AND SO DO A BIN'S EXTREMES ALONE,
    which this kept at first: ``argmin``/``argmax`` take the first sample of each value, so
    the knot before a step was the bin's start, and a 10 pA step mid-bin drew as a ramp a bin
    wide, 9.9 pA off at worst (measured on 3 M samples). The sample BEFORE each kept one
    closes that gap, and the largest step is kept because on a sloping stretch a step is
    neither the bin's min nor its max. So a step -- on a flat or a sloping stretch -- is drawn
    at its own sample, and a feature narrower than a bin still shows as the vertical
    excursion it really is. Several sharp steps inside ONE bin are not all promised a knot
    pair: the rule keeps the largest, and the others are drawn through the bin's extremes.
    Knots stay in time order at their own indices. Between knots the line is straight: on a
    synthetic 15 s moving baseline the worst error was 0.47 pA, against 0.72 pA for the
    baseline rail's own 20 000 uniform knots.

    At most ``max_points`` knots (six per bin), plus the two endpoints. ``None`` is three times
    the baseline rail's budget (`baseline.BASELINE_TRACK_MAX_POINTS`, whose note measures it):
    the same 10 000 bins a min/max pair per bin fitted in that budget, with all six knots each.
    The renderer peak-decimates them again to the pixels on screen. ``0`` is the exact
    full-length track.
    """
    if max_points is None:
        from .baseline import BASELINE_TRACK_MAX_POINTS
        max_points = 3 * BASELINE_TRACK_MAX_POINTS
    b = np.asarray(track, dtype=np.float64)
    n = int(b.size)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    m = int(max_points)
    if m <= 0 or n <= m:
        return np.arange(n, dtype=np.int64), np.ascontiguousarray(b)
    width = int(np.ceil(n / max(1, m // _KNOTS_PER_BIN)))
    full = (n // width) * width
    offsets = np.arange(full // width, dtype=np.int64) * width
    # The step INTO sample i (0 at the first), so its knot pair is (i - 1, i).
    step = np.empty(n, dtype=np.float64)
    step[0] = 0.0
    np.subtract(b[1:], b[:-1], out=step[1:])
    np.abs(step, out=step)

    def per_bin(values: np.ndarray, pick) -> np.ndarray:
        # Whole bins as a reshaped VIEW and the short last bin on its own: no padded copy of
        # a track that can be tens of millions of samples long.
        out = offsets + pick(values[:full].reshape(-1, width), axis=1)
        return np.append(out, full + int(pick(values[full:]))) if full < n else out

    kept = np.concatenate((per_bin(b, np.argmin), per_bin(b, np.argmax),
                           per_bin(step, np.argmax)))
    idx = np.unique(np.concatenate(([0, n - 1], kept, np.maximum(kept - 1, 0)))
                    .astype(np.int64))
    return idx, np.ascontiguousarray(b[idx])


def _side(polarity: int) -> str:
    return "below" if int(polarity) < 0 else "above"


def detection_threshold_tracks(localization: np.ndarray, fs: float,
                               prov: Mapping[str, Any], *, module_id: str = "",
                               max_points: int | None = None) -> ThresholdTracks:
    """The rails this table's detector compared the signal with.

    ``localization`` must be the table's own rebuilt localization signal,
    `modules.detection_signal` of the raw channel under ``prov`` (for the tables this
    accepts, that is exactly the array the detector thresholded). Raises
    :class:`ThresholdsUnavailable` with the sentence to show when the table is refused or
    the reproduction does not land on the detector's stamped numbers.
    """
    from .baseline import BaselineConfig, characterisation_baseline
    from .filters import noise_rms

    prov = dict(prov or {})
    x = np.asarray(localization)
    reason = threshold_refusal(module_id, prov, n_samples=int(x.size))
    if reason is not None:
        raise ThresholdsUnavailable(reason)
    cfg = BaselineConfig.reproducible_from(prov)
    requested = prov.get("polarity_requested")
    try:
        result = characterisation_baseline(
            x, float(fs), config=cfg,
            polarity=None if requested is None else int(requested),
            param_source=str(prov.get("baseline_param_source", "restored")), health=False)
    except (ValueError, TypeError) as exc:
        raise ThresholdsUnavailable(
            f"the detector's open-pore baseline could not be rebuilt ({exc}).") from exc
    base = np.asarray(result.baseline, dtype=np.float64)
    if base.size != x.size:
        base = np.broadcast_to(base, x.shape).astype(np.float64)
    sigma = float(noise_rms(np.asarray(x, dtype=np.float64) - base))
    stamped = float(prov["noise_rms_pA"])
    off = not np.isfinite(sigma) or abs(sigma - stamped) > _REPRODUCE_RTOL * abs(stamped)
    # ...and the LEVEL, which sigma alone cannot pin: a baseline shifted by a constant
    # leaves the residual's MAD exactly where it was. A hundredth of a sigma is far
    # below a pixel and far above the float32 rounding of the rebuilt signal.
    median_stamped = prov.get("baseline_median_pA")
    if median_stamped is not None and np.isfinite(float(median_stamped)):
        off = off or abs(float(np.median(base)) - float(median_stamped)) > 0.01 * abs(stamped)
    if off:
        raise ThresholdsUnavailable(
            f"rebuilding the detector's baseline from this table's record gave sigma "
            f"{sigma:.4g} against the {stamped:.4g} it recorded, so the line would not be "
            f"the one the detector used.")
    pol = int(prov["polarity"])
    units = str(prov.get("units") or "pA")
    low = prov.get("filter_cutoff_hz")
    where = (f"the signal after its {float(low) / 1000.0:g} kHz low-pass" if low
             else "the conditioned signal")
    rails: list[ThresholdRail] = []
    if prov.get("detector") == _SCHMITT:
        from .detection_noise import detection_noise_profile

        try:
            noise_track, noise_prov = detection_noise_profile(
                np.asarray(x, dtype=np.float64) - base, float(fs),
                policy=prov.get("noise_policy", "global"),
                window_s=prov.get("noise_window_s", 0.25), global_noise=sigma)
        except (ValueError, TypeError) as exc:
            raise ThresholdsUnavailable(
                f"the detector's noise comparison could not be rebuilt ({exc}).") from exc
        if "noise_policy" in prov:
            if noise_prov["noise_policy_effective"] != prov["noise_policy_effective"]:
                raise ThresholdsUnavailable(
                    "rebuilding the detector's noise changed its effective policy, so the "
                    "line would not be the one the detector used.")
            if noise_track is not None:
                # The small tile profile is the recorded comparison, not a
                # sample-length array. Verify its own digest before comparing
                # every tile against replay (float32 caches need a tolerance).
                try:
                    raw_ends = np.asarray(prov["noise_profile_tile_ends"])
                    raw_scales = np.asarray(prov["noise_profile_scales_pA"])
                    width = prov["noise_window_samples"]
                    if (raw_ends.ndim != 1 or not 1 <= raw_ends.size <= 4096
                            or raw_ends.dtype.kind not in "iu"
                            or raw_scales.ndim != 1 or raw_scales.shape != raw_ends.shape
                            or raw_scales.dtype.kind not in "fiu"
                            or not np.isfinite(raw_scales).all() or np.any(raw_scales <= 0)
                            or np.any(raw_ends <= 0) or np.any(raw_ends > x.size)
                            or np.any(raw_ends[1:] <= raw_ends[:-1]) or raw_ends[-1] != x.size
                            or type(width) is not int or not 0 < width <= x.size):
                        raise ValueError("invalid local noise profile shape, boundaries or scales")
                    recorded_ends = raw_ends.astype("<i8")
                    recorded_scales = raw_scales.astype("<f8")
                    maximum_width = prov["noise_window_max_samples"]
                    tile_count = prov["noise_local_tiles"]
                    if (type(maximum_width) is not int
                            or maximum_width != int(np.max(np.diff(np.r_[0, recorded_ends])))
                            or type(tile_count) is not int or tile_count != recorded_ends.size):
                        raise ValueError("recorded local noise window geometry is inconsistent")
                    reproduced_ends = np.asarray(noise_prov["noise_profile_tile_ends"], dtype="<i8")
                    reproduced_scales = np.asarray(noise_prov["noise_profile_scales_pA"], dtype="<f8")
                    digest = hashlib.sha256()
                    digest.update(np.asarray([
                        prov["noise_algorithm_version"], x.size,
                        prov["noise_window_samples"]], dtype="<i8").tobytes())
                    digest.update(recorded_ends.tobytes())
                    digest.update(recorded_scales.tobytes())
                    profile_matches = (
                        digest.hexdigest() == prov["noise_profile_sha256"]
                        and np.array_equal(recorded_ends, reproduced_ends)
                        and recorded_scales.shape == reproduced_scales.shape
                        and np.allclose(recorded_scales, reproduced_scales,
                                        rtol=_REPRODUCE_RTOL, atol=0.0))
                    # These range fields qualify the table's detection limits.
                    # A changed multiplier must not leave a contradictory range
                    # or a fictitious scalar truncation floor beside the rails.
                    for threshold, multiplier in (("in", prov["k_in"]), ("out", prov["k_out"])):
                        recorded_range = np.asarray(prov[f"threshold_{threshold}_range_pA"])
                        expected_range = multiplier * np.asarray([
                            recorded_scales.min(), recorded_scales.max()])
                        profile_matches = profile_matches and (
                            prov.get(f"threshold_{threshold}_pA") is None
                            and recorded_range.shape == (2,)
                            and recorded_range.dtype.kind in "fiu"
                            and np.isfinite(recorded_range).all()
                            and np.allclose(recorded_range, expected_range, rtol=1e-12, atol=0))
                except (KeyError, TypeError, ValueError, OverflowError):
                    profile_matches = False
                if not profile_matches:
                    raise ThresholdsUnavailable(
                        "the detector's recorded local noise profile cannot be reproduced; "
                        "the line would not be the one the detector used.")
            for key in ("noise_local_scale_min_pA", "noise_local_scale_median_pA",
                        "noise_local_scale_max_pA"):
                try:
                    matches = key in prov and np.isclose(
                        float(noise_prov[key]), float(prov[key]),
                        rtol=_REPRODUCE_RTOL, atol=0.0)
                except (ValueError, TypeError):
                    matches = False
                if noise_track is not None and not matches:
                    raise ThresholdsUnavailable(
                        "rebuilding the detector's local noise did not reproduce its "
                        "record, so the line would not be the one the detector used.")
        k_in, k_out = float(prov.get("k_in", np.nan)), float(prov.get("k_out", np.nan))
        if noise_track is None:
            thr_in = float(prov.get("threshold_in_pA", k_in * stamped))
            thr_out = float(prov.get("threshold_out_pA", k_out * stamped))
            idx, knots = decimate_extrema(base, max_points)
            rails.append(ThresholdRail(RAIL_ENTER, f"enter, k_in = {k_in:g} sigma",
                                       idx, knots + pol * thr_in))
            rails.append(ThresholdRail(RAIL_EXIT, f"exit, k_out = {k_out:g} sigma",
                                       idx, knots + pol * thr_out))
            enter_amount, exit_amount = f"{thr_in:.3g} {units}", f"{thr_out:.3g} {units}"
            noise_label = "sigma"
        else:
            for label, name, multiplier in ((RAIL_ENTER, "enter, k_in", k_in),
                                             (RAIL_EXIT, "exit, k_out", k_out)):
                idx, values = decimate_extrema(base + pol * multiplier * noise_track, max_points)
                rails.append(ThresholdRail(label, f"{name} = {multiplier:g} local sigma", idx, values))
            enter_amount = f"{k_in * noise_track.min():.3g}–{k_in * noise_track.max():.3g} {units}"
            exit_amount = f"{k_out * noise_track.min():.3g}–{k_out * noise_track.max():.3g} {units}"
            noise_label = "local sigma"
        # THE RULE AS `_hysteresis_runs` APPLIES IT: the span is the run past the EXIT level,
        # start and end alike, and the enter level is a test of its peak. "Entered at k_in"
        # would send a user checking a span's start to the dashed rail, which the trace
        # crosses a few samples after the span began.
        description = (
            f"Detection thresholds drawn: an event is a run where {where} stayed more than "
            f"{k_out:g} {noise_label} ({exit_amount}) {_side(pol)} its moving baseline "
            f"(dotted, the exit level), kept only if it also passed {k_in:g} {noise_label} "
            f"({enter_amount}, dashed, the enter level).")
        if noise_track is not None:
            description += " Thresholds vary with the recorded local noise policy; the table's SNR uses its global noise estimate."
    else:
        idx, knots = decimate_extrema(base, max_points)
        k = float(prov["cusum_k"])
        h = prov.get("cusum_h")
        drift = k * stamped
        rails.append(ThresholdRail(RAIL_DRIFT, f"CUSUM drift, k = {k:g} sigma",
                                   idx, knots + pol * drift))
        description = (
            f"CUSUM drift level drawn (dash-dot): evidence built up while {where} was "
            f"beyond {k:g} sigma ({drift:.3g} {units}) {_side(pol)} its moving baseline. "
            f"An event was called when that evidence reached h = "
            f"{float(h) if h is not None else float('nan'):g} sigma-samples, an amount of "
            f"evidence and not a current, so it has no line.")
    if low:
        description += (" The raw trace is noisier than the signal compared, so its noise "
                        "can cross a line the detector's signal did not.")
    return ThresholdTracks(rails=tuple(rails), description=description)
