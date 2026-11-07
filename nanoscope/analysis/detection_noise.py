"""Versioned local noise scales for Schmitt detection, separate from row SNR.

The default only leaves the historical global-MAD path when robust noise scales
vary by at least a factor of three. First differences make scale estimates
insensitive to event plateau levels; a neighboring-window median of the
level/difference MAD ratio restores colored or filtered noise locally. The global
MAD is a conservative floor, so adaptation never lowers thresholds. Tile maxima are
deliberately conservative around variance changes. This cannot distinguish a
very short noise burst from an event or guarantee recovery below local noise.
"""
from __future__ import annotations

import hashlib
import math
import copy
import threading
from collections import OrderedDict
import numpy as np
from scipy.ndimage import median_filter

from .cancel import CancelFlag, check_cancelled
from .filters import noise_rms

NOISE_ALGORITHM_VERSION = 1
NOISE_POLICIES = ("auto", "global", "local")
_MIN_TILE_SAMPLES = 128
_MAX_TILES = 4096
_AUTO_VARIATION_RATIO = 3.0
# For standard Gaussian Z conditioned on |Z|<=2, median(|Z|) is
# Phi^-1(1/4 + Phi(2)/2). Its reciprocal makes the clipped MAD consistent.
_CLIPPED_MAD_NORMAL_FACTOR = 1.5646711991906783
_PROFILE_CACHE: OrderedDict = OrderedDict()
_PROFILE_CACHE_LOCK = threading.Lock()
_PROFILE_CACHE_LIMIT = 4


def _calibration_level_scale(values: np.ndarray) -> float:
    """Gaussian-consistent level scale that rejects minority event plateaus.

    Ordinary level MAD can be inflated by neighboring events even though first
    differences remain clean. Iterative symmetric clipping removes that level
    contamination before it can be misread as noise correlation. It does not
    recenter the signal or detection threshold, and never edits input samples.
    Correlated Gaussian noise retains its level distribution, so no global cap
    is imposed on the level/difference calibration factor.
    """
    # A median/MAD start can sit between two nearly equally occupied levels:
    # its huge scale then clips neither mode and calls the state separation
    # noise. Start from the shortest half-sample interval instead. For Gaussian
    # noise that interval spans the central50%, whose width is1.3489795 sigma.
    # Its center follows a dominant level without knowing event times/polarity.
    ordered = np.sort(values)
    half = max(2, (values.size + 1) // 2)
    spans = ordered[half - 1:] - ordered[:values.size - half + 1]
    begin = int(np.argmin(spans))
    location = np.median(ordered[begin:begin + half])
    scale = float(spans[begin] / 1.3489795003921634)
    if not np.isfinite(scale) or scale <= 0:
        return scale
    for _ in range(6):
        keep = np.abs(values - location) <= 2. * scale
        if np.count_nonzero(keep) < max(16, values.size // 2):
            break
        retained = values[keep]
        location = np.median(retained)
        refined = float(_CLIPPED_MAD_NORMAL_FACTOR * np.median(np.abs(retained - location)))
        if not np.isfinite(refined) or refined <= 0:
            break
        converged = abs(refined - scale) <= .001 * scale
        scale = refined
        if converged:
            break
    return scale


def _compute_detection_noise_profile(
    residual: np.ndarray,
    fs: float,
    *,
    policy: str = "auto",
    window_s: float = 0.25,
    global_noise: float | None = None,
    cancel: CancelFlag | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Return a positive local noise track, or None for exact scalar behavior.

    The returned provenance contains a bounded recipe, at most 4096 tile scales,
    and a profile digest, never a sample-length array. Rebuilding rails uses this
    same helper and revision. Tile scales permit tolerance-checked replay when a
    GUI trace cache has reduced the original float64 calculation to float32.
    ``noise_rms_pA`` elsewhere remains the global measurement/SNR reference.
    A short final tile is combined with its predecessor. At least four tiles are
    required for automatic adaptation; an explicit local policy needs two.
    """
    if policy not in NOISE_POLICIES:
        raise ValueError(f"noise_policy must be one of {NOISE_POLICIES}, got {policy!r}")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive for noise estimation")
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("noise_window_s must be finite and positive")
    x = np.asarray(residual, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("noise residual must be a nonempty finite one-dimensional array")
    if global_noise is None:
        global_noise = float(noise_rms(x))
    if not np.isfinite(global_noise) or global_noise <= 0:
        raise ValueError("global noise must be finite and positive")
    # Bound Python loops and provenance size even for unusually long, low-rate
    # recordings. The effective sample window is explicitly recorded.
    requested = float(window_s) * float(fs)
    if not np.isfinite(requested):
        raise ValueError("noise_window_s * fs must be finite")
    width = max(_MIN_TILE_SAMPLES, min(x.size, int(round(requested))),
                math.ceil(x.size / _MAX_TILES))
    info = {
        "noise_policy": policy,
        "noise_policy_effective": "global",
        "noise_window_s": float(window_s),
        "noise_window_samples": width,
        "noise_window_max_samples": min(width, int(x.size)),
        "noise_algorithm_version": NOISE_ALGORITHM_VERSION,
        "noise_local_scale_min_pA": float(global_noise),
        "noise_local_scale_median_pA": float(global_noise),
        "noise_local_scale_max_pA": float(global_noise),
        "noise_local_variation_ratio": 1.0,
        "noise_local_calibration": None,
        "noise_local_calibration_range": None,
        "noise_local_tiles": 0,
        "noise_profile_sha256": None,
        "noise_profile_tile_ends": [],
        "noise_profile_scales_pA": [],
        "noise_warnings": [],
        "noise_policy_reason": "explicit global noise policy",
    }
    check_cancelled(cancel)
    if policy == "global":
        return None, info
    edges = np.arange(0, x.size, width, dtype=np.int64)
    if edges.size > 1 and x.size - int(edges[-1]) < _MIN_TILE_SAMPLES:
        edges = edges[:-1]
    ends = np.r_[edges[1:], x.size]
    info["noise_window_max_samples"] = int(np.max(ends - edges))
    info["noise_local_tiles"] = int(edges.size)
    minimum = 4 if policy == "auto" else 2
    if edges.size < minimum:
        info["noise_policy_reason"] = "too few independent noise windows; global fallback"
        info["noise_warnings"] = [
            f"Local noise estimation needs at least {minimum} windows with at least "
            f"{_MIN_TILE_SAMPLES} samples each; only {edges.size} window(s) were available. "
            "Detection used global noise thresholds, so changes in noise within this "
            "short trace are not resolved."
        ]
        return None, info
    differences = np.empty(edges.size)
    shifted = np.empty(max(0, edges.size - 1))
    shifted_levels = np.empty(shifted.size)
    levels = np.empty(edges.size)
    for j, (start, stop) in enumerate(zip(edges, ends)):
        check_cancelled(cancel)
        tile = x[start:stop]
        levels[j] = _calibration_level_scale(tile)
        differences[j] = noise_rms(np.diff(tile)) / np.sqrt(2.0)
        if j < shifted.size:
            # Half-overlapping windows remove dependence on where a variance
            # change falls relative to tile boundaries. A burst longer than one
            # window can otherwise contaminate both neighboring MAD estimates.
            shift_start = int(start) + width // 2
            shift_stop = min(x.size, shift_start + width)
            shifted[j] = noise_rms(np.diff(x[shift_start:shift_stop])) / np.sqrt(2.0)
            shifted_levels[j] = _calibration_level_scale(x[shift_start:shift_stop])
    valid = np.isfinite(differences) & (differences > 0) & np.isfinite(levels) & (levels > 0)
    if np.count_nonzero(valid) < minimum:
        info["noise_policy_reason"] = "too few windows with measurable noise; global fallback"
        info["noise_warnings"] = [
            "Too few local windows have measurable noise. Detection used global "
            "thresholds; quantized or locally constant data cannot establish a "
            "reliable local noise scale."
        ]
        return None, info
    # A zero-noise tile has no reliable local threshold; carry the nearest valid
    # measured scale instead of inventing an epsilon that would detect roundoff.
    if not valid.all():
        differences[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), differences[valid])
        info["noise_warnings"] = [
            f"{np.count_nonzero(~valid)} local noise window(s) were constant or quantized; "
            "their noise scales were interpolated from measurable neighboring windows. "
            "Local detection sensitivity in those windows is uncertain."
        ]
    # Interleave full and half-shifted windows in time. A LOCAL calibration is
    # essential: equal-amplitude white and strongly correlated noise have very
    # different first-difference scales. A single global factor would create
    # false events in the correlated segment. Five-window medians reject the
    # minority of ratios whose level MAD is inflated by a true event boundary.
    diff_all = np.empty(edges.size + shifted.size)
    level_all = np.empty(diff_all.size)
    diff_all[::2], level_all[::2] = differences, levels
    diff_all[1::2], level_all[1::2] = shifted, shifted_levels
    measurable = np.isfinite(diff_all) & (diff_all > 0) & np.isfinite(level_all) & (level_all > 0)
    ratios = np.ones(diff_all.size)
    ratios[measurable] = level_all[measurable] / diff_all[measurable]
    if not measurable.all():
        ratios[~measurable] = np.interp(np.flatnonzero(~measurable), np.flatnonzero(measurable), ratios[measurable])
    calibrations = median_filter(ratios[None, :], size=(1, 5), mode="nearest")[0]
    all_scales = np.where(np.isfinite(diff_all) & (diff_all > 0), diff_all, 0.) * calibrations
    scales = all_scales[::2]
    # Conservative one-window guards prevent a high-noise step from borrowing
    # the low threshold on its other side. This also avoids interpolation ramps.
    guarded = np.maximum.reduce((scales, np.r_[scales[0], scales[:-1]],
                                 np.r_[scales[1:], scales[-1]]))
    if shifted.size:
        guarded[:-1] = np.maximum(guarded[:-1], all_scales[1::2])
        guarded[1:] = np.maximum(guarded[1:], all_scales[1::2])
    variation = float(np.max(guarded) / np.min(guarded))
    info["noise_local_variation_ratio"] = variation
    if policy == "auto" and variation < _AUTO_VARIATION_RATIO:
        info["noise_policy_reason"] = "local noise variation below 3x; exact global path"
        return None, info
    # Never reduce the historical threshold. Apart from conservative behavior in
    # quiet segments this protects against local calibration becoming unreliable
    # when correlation changes faster than the estimation window.
    np.maximum(guarded, global_noise, out=guarded)
    if not np.isfinite(guarded).all() or np.any(guarded <= 0):
        raise ValueError("local noise scale is not finite and positive")
    track = np.repeat(guarded, ends - edges)
    check_cancelled(cancel)
    digest = hashlib.sha256()
    digest.update(np.asarray([NOISE_ALGORITHM_VERSION, x.size, width], dtype="<i8").tobytes())
    digest.update(np.asarray(ends, dtype="<i8").tobytes())
    digest.update(np.asarray(guarded, dtype="<f8").tobytes())
    info.update(
        noise_policy_effective="local",
        noise_local_scale_min_pA=float(np.min(guarded)),
        noise_local_scale_median_pA=float(np.median(track)),
        noise_local_scale_max_pA=float(np.max(guarded)),
        noise_local_calibration=float(np.median(calibrations)),
        noise_local_calibration_range=[float(np.min(calibrations)), float(np.max(calibrations))],
        noise_profile_sha256=digest.hexdigest(),
        noise_profile_tile_ends=ends.tolist(),
        noise_profile_scales_pA=guarded.tolist(),
        noise_policy_reason="explicit local policy" if policy == "local" else "robust local noise varies by at least 3x",
    )
    longest_window_s = info["noise_window_max_samples"] / fs
    info["noise_warnings"].append(
        f"Local detection thresholds use noise windows up to {longest_window_s:.4g} s "
        f"with a conservative guard extending up to {longest_window_s:.4g} s beside a noisier "
        "window. Brief noise bursts shorter than the window may remain unresolved; "
        "weak events beside a noise change may be suppressed. Inspect these intervals "
        "before interpreting event counts."
    )
    return track, info


def detection_noise_profile(
    residual: np.ndarray,
    fs: float,
    *,
    policy: str = "auto",
    window_s: float = 0.25,
    global_noise: float | None = None,
    cancel: CancelFlag | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Compute the versioned profile, reusing identical bounded recipes.

    Automatic baseline correction and its health check can ask for the identical
    full-recording profile. An exact content digest plus every numerical input
    avoids repeating its robust statistics. Only four bounded tile recipes are
    retained, never raw signals or sample-length tracks. Each caller receives a
    fresh track and metadata, so caller mutation cannot corrupt a later result.
    Explicit cancellation callbacks bypass reuse to retain their stage polling;
    ambient cancellation is checked before hashing and before every return.
    """
    check_cancelled(cancel)
    if cancel is not None or policy == "global":
        return _compute_detection_noise_profile(residual, fs, policy=policy,
            window_s=window_s, global_noise=global_noise, cancel=cancel)
    x = np.asarray(residual, dtype=np.float64)
    # Invalid inputs belong to the numerical implementation's descriptive gate,
    # not to cache key conversion or a previously cached valid result.
    if (policy not in NOISE_POLICIES or x.ndim != 1 or not x.size
            or not np.isfinite(fs) or fs <= 0
            or not np.isfinite(window_s) or window_s <= 0
            or (global_noise is not None and (not np.isfinite(global_noise) or global_noise <= 0))):
        return _compute_detection_noise_profile(x, fs, policy=policy,
            window_s=window_s, global_noise=global_noise, cancel=cancel)
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast('B')).digest()
    key = (NOISE_ALGORITHM_VERSION, float(fs).hex(), policy, float(window_s).hex(),
           None if global_noise is None else float(global_noise).hex(), x.size, digest)
    with _PROFILE_CACHE_LOCK:
        stored = _PROFILE_CACHE.get(key)
        if stored is not None:
            _PROFILE_CACHE.move_to_end(key)
            info = copy.deepcopy(stored)
        else:
            info = None
    if info is not None:
        check_cancelled(cancel)
        if info["noise_policy_effective"] == "local":
            ends = np.asarray(info["noise_profile_tile_ends"], dtype=np.int64)
            track = np.repeat(np.asarray(info["noise_profile_scales_pA"], dtype=np.float64),
                              np.diff(np.r_[0, ends]))
        else:
            track = None
        check_cancelled(cancel)
        return track, info
    track, info = _compute_detection_noise_profile(x, fs, policy=policy,
        window_s=window_s, global_noise=global_noise, cancel=cancel)
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE[key] = copy.deepcopy(info)
        _PROFILE_CACHE.move_to_end(key)
        while len(_PROFILE_CACHE) > _PROFILE_CACHE_LIMIT:
            _PROFILE_CACHE.popitem(last=False)
    check_cancelled(cancel)
    return track, info
