"""Bounded-memory validation of the timestamps whose signal samples are loaded."""
from __future__ import annotations

import numpy as np


class LoadedClockAudit:
    """Check every timestamp and adjacent interval, including chunk boundaries.

    Readers retain their existing rate estimator for clean inputs. A bad loaded clock
    invalidates that estimate instead of inventing a replacement uniform clock. The
    0.1% interval tolerance allows decimal rounding without hiding a local rate change.
    """

    def __init__(self, source: str, rate: float | None, start: int = 0,
                 scope: str = "loaded samples") -> None:
        self.source = source
        self.scope = scope
        self.start = start
        self.dt = 1.0 / rate if rate is not None and rate > 0 else None
        self.count = 0
        self.first: float | None = None
        self.previous: float | None = None
        self.previous_ulp: float | None = None
        self.problem: str | None = None

    def observe(self, values: np.ndarray) -> None:
        stored = np.asarray(values).reshape(-1)
        values = np.asarray(stored, dtype=np.float64)
        if not values.size:
            return
        # An HDF5 float32 clock is already quantized before we read it. Preserve
        # that resolution through the float64 audit and across chunk boundaries.
        rounded = stored if np.issubdtype(stored.dtype, np.floating) else values
        with np.errstate(invalid="ignore", over="ignore"):
            ulps = np.maximum(np.abs(np.spacing(rounded)).astype(np.float64),
                              np.abs(np.spacing(values)))
        if self.first is None:
            self.first = float(values[0])
        if self.problem is None:
            bad = np.flatnonzero(~np.isfinite(values))
            if bad.size:
                self.problem = f"a non-finite timestamp at sample {self.start + self.count + int(bad[0]):,}"
            else:
                joined = (np.concatenate(([self.previous], values))
                          if self.previous is not None else values)
                joined_ulps = (np.concatenate(([self.previous_ulp], ulps))
                               if self.previous is not None else ulps)
                intervals = np.diff(joined)
                base = self.start + self.count - (self.previous is not None)
                bad = np.flatnonzero(intervals <= 0)
                if bad.size:
                    self.problem = f"a non-increasing timestamp at sample {base + int(bad[0]) + 1:,}"
                elif self.dt is not None:
                    # Floating-point subtraction at a large absolute t0 has a finite
                    # resolution. Do not mistake that rounding for a changed clock.
                    precision = joined_ulps[:-1] + joined_ulps[1:]
                    bad = np.flatnonzero(np.abs(intervals - self.dt) > self.dt * 1e-3 + precision)
                    if bad.size:
                        i = int(bad[0])
                        self.problem = (f"a non-uniform interval at sample {base + i + 1:,} "
                                        f"({intervals[i]:.9g} s instead of {self.dt:.9g} s)")
        self.previous = float(values[-1])
        self.previous_ulp = float(ulps[-1])
        self.count += int(values.size)

    def issue(self) -> str | None:
        if self.problem is None:
            return None
        return (f"{self.source} has {self.problem} in the {self.scope}; a single sample "
                f"rate cannot describe this clock. Sample rate is unknown, the x-axis is "
                f"in samples, and detection in seconds is unavailable. Re-export a uniformly "
                f"sampled time base in seconds; no timestamps were silently repaired.")
