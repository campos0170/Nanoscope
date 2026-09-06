"""Opt-in, uncensored distribution summaries; independent of existing analyses.

Definitions follow NumPy's ``quantile(method='linear')``, SciPy's uncensored
ECDF/complement and ``gaussian_kde(bw_method='scott')``, and the conventional
1.5-IQR sample whiskers documented by Matplotlib's ``boxplot``. Nothing here
estimates censoring, detects events, changes a baseline, or modifies inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.stats import gaussian_kde

KINDS = ("ecdf", "survival", "box", "violin")
MAX_VIOLIN_SAMPLES = 250_000


def _frozen(values):
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DistributionSummary:
    kind: str
    samples: np.ndarray
    x: np.ndarray
    y: np.ndarray
    stats: Mapping[str, float]
    n_total: int
    n_excluded: int
    warnings: tuple[str, ...]

    @property
    def n(self):
        return len(self.samples)

    def step_coordinates(self):
        """Vertical jumps at observed values; horizontal segments are right-continuous."""
        if self.kind not in ("ecdf", "survival"):
            raise ValueError("Step coordinates apply to ECDF and empirical survival only.")
        if not self.n:
            return _frozen([]), _frozen([])
        before = np.r_[0.0 if self.kind == "ecdf" else 1.0, self.y[:-1]]
        return _frozen(np.repeat(self.x, 2)), _frozen(np.column_stack((before, self.y)).ravel())


def summarize_distribution(values, *, kind="ecdf", violin_points=256) -> DistributionSummary:
    """Summarize finite, unweighted observations, disclosing every exclusion.

    ECDF is F(x)=count(X<=x)/n and survival is S(x)=count(X>x)/n; ties share a
    jump. Box quartiles use linear interpolation and whiskers are actual sample
    extrema inside Q1-1.5*IQR and Q3+1.5*IQR. Violin density uses Scott's rule
    on 3+ nonconstant samples, evaluated only across the observed range. Its
    tails are not renormalized and its displayed width is not a sample count.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown distribution summary: {kind}")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError("Distribution summaries require a one-dimensional real numeric array.")
    # longdouble is only float64 on some platforms; compare native integers
    # directly before any floating conversion (and avoid abs(INT64_MIN)).
    if raw.dtype.kind in "iu" and raw.size and (
            np.any(raw > 2 ** 53) or (raw.dtype.kind == "i" and np.any(raw < -(2 ** 53)))):
        raise ValueError("Integer observations above 2**53 cannot be represented exactly on this float axis.")
    sample = np.array(raw, dtype=np.float64, copy=True)
    n_total = len(sample)
    sample = sample[np.isfinite(sample)]
    excluded = n_total - len(sample)
    warnings = [f"Excluded {excluded} non-finite observation(s); no values were imputed."] if excluded else []
    stats = {}
    x = y = np.empty(0, dtype=np.float64)
    if not len(sample):
        warnings.append("No finite observations are available; no distribution was estimated.")
    elif kind in ("ecdf", "survival"):
        x, counts = np.unique(sample, return_counts=True)
        cumulative = np.cumsum(counts, dtype=np.int64)
        y = (cumulative / len(sample) if kind == "ecdf" else
             (len(sample) - cumulative) / len(sample))
    elif kind == "box":
        q1, median, q3 = np.quantile(sample, [.25, .5, .75], method="linear")
        with np.errstate(over="ignore", invalid="ignore"):
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if not np.all(np.isfinite([q1, median, q3, lo, hi])):
            raise ValueError("The box summary exceeds the finite numerical range.")
        inside = sample[(sample >= lo) & (sample <= hi)]
        stats = dict(q1=float(q1), median=float(median), q3=float(q3),
                     iqr=float(iqr), lower_fence=float(lo), upper_fence=float(hi),
                     lower_whisker=float(inside.min()), upper_whisker=float(inside.max()))
        x = np.array([q1, median, q3, inside.min(), inside.max()])
        if len(sample) < 4:
            warnings.append("Fewer than four observations: quartiles interpolate sparse samples.")
    else:
        if type(violin_points) is not int or not 32 <= violin_points <= 2048:
            raise ValueError("Violin grid size must be an integer from 32 to 2048.")
        if len(sample) > MAX_VIOLIN_SAMPLES:
            raise ValueError("Violin KDE is limited to 250,000 finite observations per source; "
                             "select a smaller source. No sampling is performed automatically.")
        if len(sample) < 3 or np.all(sample == sample[0]):
            x = np.unique(sample)
            y = np.zeros(len(x))
            warnings.append("No violin density estimated: fewer than three observations or constant data. "
                            "Only the observed values are shown.")
        else:
            try:
                kernel = gaussian_kde(sample, bw_method="scott")
                x = np.linspace(sample.min(), sample.max(), violin_points)
                y = kernel(x)
                bandwidth = float(np.sqrt(kernel.covariance[0, 0]))
                if not np.all(np.isfinite(y)) or not np.isfinite(bandwidth) or bandwidth <= 0:
                    raise ValueError("non-finite density or bandwidth")
                stats = {"scott_factor": float(kernel.factor), "bandwidth": bandwidth}
            except (ValueError, np.linalg.LinAlgError) as exc:
                x = np.unique(sample)
                y = np.zeros(len(x))
                warnings.append(f"No violin density estimated ({exc}); only observations are shown.")
            if len(sample) < 10:
                warnings.append("Fewer than ten observations: the violin shape is weakly supported.")
        warnings.append("Violin: Gaussian KDE, Scott bandwidth; drawn only over the observed range. "
                        "Tails are not renormalized, and equal peak widths do not imply equal event counts.")
    if kind == "survival":
        warnings.append("Empirical survival is P(X > x) for uncensored observed values, not Kaplan–Meier. "
                        "No censoring or missed-event correction is inferred.")
    return DistributionSummary(kind, _frozen(sample), _frozen(x), _frozen(y),
                               MappingProxyType(stats), n_total, excluded, tuple(warnings))
