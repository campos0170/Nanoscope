"""Bounded, display-only residual storage; scientific subtraction stays in analysis."""
from collections import OrderedDict
import weakref

import numpy as np


class DisplayResidualCache:
    """Small LRU across pane rebuilds, with explicit baseline/calibration invalidation.

    Session samples and installed baselines are immutable by application contract.
    Only weak references to inputs are retained; callers invalidate when replacing
    a baseline (even the same array) or changing calibration. Oversized residuals
    can still be displayed but are not kept in this cache.
    """

    def __init__(self, max_bytes=64 * 1024 * 1024, chunk_samples=262_144):
        self.max_bytes = max(0, int(max_bytes))
        self.chunk_samples = max(1, int(chunk_samples))
        self._entries = OrderedDict()
        self.nbytes = 0

    def invalidate(self, key):
        entry = self._entries.pop(key, None)
        if entry is not None:
            self.nbytes -= entry[2].nbytes

    def retain(self, keys):
        for key in tuple(self._entries):
            if key not in keys:
                self.invalidate(key)

    def residual(self, key, signal, baseline):
        x, b = np.asarray(signal), np.asarray(baseline)
        if x.ndim != 1 or x.shape != b.shape:
            raise ValueError("Display residual requires matching one-dimensional arrays.")
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is x and entry[1]() is b:
            self._entries.move_to_end(key)
            return entry[2]
        self.invalidate(key)
        # Call the existing numerical implementation in bounded pieces, retaining
        # its float64 subtraction -> float32 rounding exactly. No full-length
        # float64 copies are needed merely to draw a residual.
        from ..analysis.baseline import baseline_subtracted
        out = np.empty(x.shape, dtype=np.float32)
        for start in range(0, x.size, self.chunk_samples):
            stop = start + self.chunk_samples
            out[start:stop] = baseline_subtracted(x[start:stop], b[start:stop])
        out.setflags(write=False)
        if out.nbytes <= self.max_bytes:
            while self._entries and self.nbytes + out.nbytes > self.max_bytes:
                self.invalidate(next(iter(self._entries)))
            self._entries[key] = (weakref.ref(x), weakref.ref(b), out)
            self.nbytes += out.nbytes
        return out
