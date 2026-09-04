"""Temporary dual-signal arrays must release their cached baseline resources."""

from __future__ import annotations

import weakref

import numpy as np

from nanoscope.analysis import baseline as B


def test_dead_temporary_signal_evicts_baseline_and_polarity_memos_immediately():
    B.clear_baseline_cache()
    rng = np.random.default_rng(4)
    signal = np.ascontiguousarray(
        100.0 + rng.normal(0.0, 1.0, 20_000), dtype=np.float32)
    signal[8_000:8_300] -= 20.0
    ref = weakref.ref(signal)

    B.characterisation_baseline(signal, 10_000.0)
    assert B._CACHE._entries
    assert B._POLARITY_MEMO._entries

    del signal
    # NumPy arrays are reference-counted, so the weakref callbacks run at deletion;
    # no later lookup or four-entry eviction should be needed to release the large
    # BaselineResult arrays a real recording would have cached.
    assert ref() is None
    assert not B._CACHE._entries
    assert not B._POLARITY_MEMO._entries
