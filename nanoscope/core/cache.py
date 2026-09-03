from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np

__all__ = ["LruArrayCache"]


class LruArrayCache:
    """Bounded, evictable, keyed by CONTENT.

    WHY bounded: 4 open comparison files x (60 MB calibrated + 120 MB time axis) = 720 MB of
    derived arrays on top of the raw signal. An unbounded cache-on-the-document leaks all of it.

    Eviction drops OUR reference only. An array still referenced by a live curve stays alive —
    that is correct: the budget exists to stop orphans accumulating, not to free displayed data.
    """

    def __init__(self, budget_mb: int) -> None:
        self._budget = int(budget_mb) * 1024 * 1024
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0

    def get_or_compute(self, key: tuple, compute: Callable[[], np.ndarray]) -> np.ndarray:
        """Hit -> mark most-recently-used and return. Miss -> compute, insert, evict to budget.

        An array larger than the whole budget is returned but NOT stored: caching it would evict
        everything else and then be evicted itself on the next insert.
        """
        hit = self._entries.get(key)
        if hit is not None:
            self._entries.move_to_end(key)
            return hit

        value = compute()
        nbytes = value.nbytes
        if nbytes <= self._budget:
            self._entries[key] = value
            self._bytes += nbytes
            self._evict_to_budget()
        return value

    def _evict_to_budget(self) -> None:
        while self._bytes > self._budget and self._entries:
            _, evicted = self._entries.popitem(last=False)   # least-recently-used
            self._bytes -= evicted.nbytes

    def invalidate_doc(self, doc_id: str) -> None:
        """Drops every key whose key[1] == doc_id. Both key layouts put doc_id at index 1:
        ("cal", doc_id, channel, gain, offset) and ("time", doc_id, t_shift)."""
        for key in [k for k in self._entries if len(k) > 1 and k[1] == doc_id]:
            self.invalidate(key)

    def invalidate(self, key: tuple) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry.nbytes

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    @property
    def bytes_used(self) -> int:
        return self._bytes
