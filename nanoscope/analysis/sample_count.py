"""Stable integer boundaries for durations expressed on a sample grid.

Rates remain exactly as declared or measured. Only the final sample count is
snapped, and only within floating-point round-off of an integer; a genuine
fractional sample retains the caller's floor/ceil convention.
"""
from __future__ import annotations

import math


SAMPLE_COUNT_POLICY = "integer-boundary-16ulp-v1"


def _integer_boundary(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("sample count must be finite")
    nearest = round(value)
    # Eight ULPs arise in the CSV-inferred 1999.9999999999982 Hz regression;
    # sixteen also covers the multiply/divide steps used for decimated windows.
    # The absolute cap prevents a large count's wide ULP from absorbing a real
    # fraction of a sample. In particular, this is not a relative rate tolerance.
    tolerance = min(16.0 * math.ulp(value), 1e-9)
    return float(nearest) if nearest != 0 and abs(value - nearest) <= tolerance else value


def floor_sample_count(value: float) -> int:
    """Floor a sample count, tolerating round-off at an integer boundary."""
    return math.floor(_integer_boundary(value))


def ceil_sample_count(value: float) -> int:
    """Ceil a sample count, tolerating round-off at an integer boundary."""
    return math.ceil(_integer_boundary(value))
