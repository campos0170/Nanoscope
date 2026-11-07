"""Masking and gap interpolation for automatic baseline estimation."""
from __future__ import annotations

import numpy as np

from .detection_noise import NOISE_ALGORITHM_VERSION, detection_noise_profile
from .cancel import check_cancelled
from .sample_count import floor_sample_count


MASK_NOISE_POLICY = f"dual-source-robust-masked-baseline-v2:noise-v{NOISE_ALGORITHM_VERSION}"

# A local anchor is a robust estimate of the level immediately beside a gap,
# not a second, potentially seconds-wide baseline fit. Bound its statistical
# support as well as its time span; otherwise long tracking windows mix remote
# level changes into the anchor and repeatedly sort millions of observations.
_MAX_FLANK_SAMPLES = 512


def baseline_mask_noise_scale(rough: np.ndarray, fs: float, global_sigma: float,
                              *, signal: np.ndarray | None = None):
    """Keep global masking unless robust local noise varies by at least 3x.

    A one-sided global threshold otherwise masks most negative excursions in a
    noisy interval, interpolates from its positive samples and creates an upward
    baseline bias. Use the local scale only to RAISE that threshold's magnitude;
    never recenter samples, lower the original scale or change event polarity.
    Difference-based variation and robust calibration resist event plateaus and
    account for correlated noise. Require both the original canonical signal
    and its rough residual to support local elevation. A rough median entering
    dense events can add artificial residual variation; the original signal
    can contain physical drift. The smaller estimate rejects either artefact,
    and neither can lower the historical global residual scale. An automatic
    fallback in either profile means there is no evidence to raise that scale.
    Short/unmeasurable traces keep the old path.
    """
    if (not np.isfinite(global_sigma) or global_sigma <= 0
            or not np.isfinite(rough).all()):
        return global_sigma
    noise_signal = rough if signal is None else np.asarray(signal)
    local, _ = detection_noise_profile(noise_signal, fs, policy="auto", window_s=.25,
                                       global_noise=float(global_sigma))
    if local is None:
        return global_sigma
    if signal is not None:
        residual_local, _ = detection_noise_profile(rough, fs, policy="auto", window_s=.25,
                                                     global_noise=float(global_sigma))
        if residual_local is None:
            return global_sigma
        np.minimum(local, residual_local, out=local)
    np.maximum(local, global_sigma, out=local)
    return local


def _adjacent_flank(x: np.ndarray, indices: np.ndarray, *, left: bool) -> np.ndarray:
    """Do not average a resolved baseline step into the adjacent gap anchor."""
    if indices.size < 6:
        return indices
    values = np.asarray(x[indices], dtype=np.float64)
    spacing = np.diff(indices)
    differences = np.diff(values)
    slope = float(np.median(differences / spacing))
    changes = differences - slope * spacing
    centre = float(np.median(changes))
    sigma = float(1.4826 * np.median(np.abs(changes - centre)))
    tolerance = max(4.5 * sigma, 64 * np.finfo(float).eps * max(1., float(np.max(np.abs(values)))))
    detrended = values - slope * (indices - indices[0])
    jumps = []
    for cut in np.flatnonzero(np.abs(changes - centre) > tolerance) + 1:
        if cut < 3 or indices.size - cut < 3:
            continue
        # Require a sustained level change, not a single extreme sample. The
        # same symmetric rule applies on either side of a positive/negative event.
        before = float(np.median(detrended[max(0, cut - 5):cut]))
        after = float(np.median(detrended[cut:min(indices.size, cut + 5)]))
        if abs(after - before) > tolerance:
            jumps.append(int(cut))
    if not jumps:
        return indices
    return indices[jumps[-1]:] if left else indices[:jumps[0]]


def fill_masked_background(signal: np.ndarray, mask: np.ndarray, fs: float,
                           window_s: float, *, robust_flanks: bool = False) -> np.ndarray:
    """Interpolate masked gaps with optional robust local baseline anchors.

    A noisy endpoint is otherwise copied across an entire event. Each robust
    anchor is the median of the nearest at most 512 unmasked samples within
    half the tracking window, placed at their median sample index. This
    reproduces a linear baseline and
    leaves every unmasked sample unchanged. Nearby events never enter a flank;
    resolved level steps bound its support instead of being averaged across.
    Both flanks need at least three samples; unsupported edges retain the
    historical interpolation rather than extrapolating an invented slope.
    """
    x = np.asarray(signal)
    filled = x.copy()
    if not mask.any() or np.count_nonzero(~mask) <= 2:
        return filled
    idx = np.arange(x.size)
    filled[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    if not robust_flanks:
        return filled
    flank = max(1, floor_sample_count(.5 * float(window_s) * float(fs)))
    if flank < 3:
        return filled
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    valid = idx[~mask]
    # Search sorted surviving observations once. This avoids repeatedly scanning
    # an entire tracking window around every event in high-rate recordings.
    left_stops = np.searchsorted(valid, starts)
    right_starts = np.searchsorted(valid, ends)
    left_limits = np.searchsorted(valid, np.maximum(0, starts - flank))
    right_limits = np.searchsorted(valid, np.minimum(x.size, ends + flank))
    for j, (start, stop) in enumerate(zip(starts, ends)):
        if j % 64 == 0:
            check_cancelled()
        ls, rs = int(left_stops[j]), int(right_starts[j])
        li = valid[max(int(left_limits[j]), ls - _MAX_FLANK_SAMPLES):ls]
        ri = valid[rs:min(int(right_limits[j]), rs + _MAX_FLANK_SAMPLES)]
        li = _adjacent_flank(x, li, left=True)
        ri = _adjacent_flank(x, ri, left=False)
        if li.size < 3 or ri.size < 3:
            continue
        anchor_t = (float(np.median(li)), float(np.median(ri)))
        anchor_y = (float(np.median(x[li])), float(np.median(x[ri])))
        filled[start:stop] = np.interp(idx[start:stop], anchor_t, anchor_y)
    check_cancelled()
    return filled
