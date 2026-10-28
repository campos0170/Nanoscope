"""S8-NUM1 (analysis layer) — the key of the "Compute exact" memo must name the rate the pass was
run at, by itself.

MEASURED DEFECT (sweep 8, -600 mV apo-ferritin ABF). Signal Conditioning ▸ Compute exact with a
baseline-drift stage and a 5 kHz Bessel low-pass, then Inspector ▸ File Info ▸ Sample rate / time
unit… ▸ 25,000 Hz, then Signal Conditioning opened again: the status line read "Whole recording
(exact)", the button "Done: the preview shows this chain's whole-recording (exact) samples", and the
trace drawn was the pass computed at 50 kHz — 26 pA away from what the chain makes of the same
samples at the declared rate, on a recording whose noise sigma is 4.5 pA. The memo of exact passes
(``ConditionedTraceMemo``) was keyed ``memo_key(source key, chain, fs)`` and that key did not hold
the rate: ``fs`` went into ``chain_signature``, whose only use of it is to prune a notch's lines at
Nyquist, so a median, a baseline drift, a low-pass or a high-pass is "the same chain" at 50 kHz and
at 25 kHz while the array it makes is not.

THE FIX. S8-SC3 (already in the base) put the rate into ``ConditioningSource.key``, and the key of a
re-read recording stopped colliding. That fixes the window, and leaves the memo's key only as good as
the source key its caller hands in: ``memo_key`` still accepted ``fs`` and discarded it. The key now
carries the rate it is given, whoever built the source key (``analysis/conditioning_preview.py``,
``memo_key``). These tests are the Qt-free half; the window half is
tests/gui/test_s8_num1_exact_pass_cache_follows_the_sample_rate.py.

What they drive: the key function and the memo, with the source key spelled the way the older tests
spell it (no rate in it), and with the real ``ConditioningSource.key`` for the hazards the fix must
not touch — a hit for unchanged inputs, a calibration edit dropping the pass, the two-slot LRU, and
a closed document forgetting its passes.
"""

from __future__ import annotations

import numpy as np

from nanoscope.analysis import conditioning_preview as cp
from nanoscope.analysis.conditioning_recipes import complete_recipe

#: The finding's chain: baseline drift (block median, 0.5 s, blockade polarity) + 5 kHz Bessel.
CHAIN = dict(complete_recipe({"detrend_enabled": True, "detrend_method": "block_median",
                              "detrend_window_s": 0.5, "filter_cutoff_hz": 5_000.0}),
             polarity=-1)         # polarity is a conditioning key but not one of the recipe's
#: A source key as the pre-S8-SC3 window built it: document, channel, gain, offset, length.
RATE_LESS = ("rec#1", 0, 1.0, 0.0, 6_000_000)


def _source(fs: float | None = 50_000.0, *, doc_id: str = "rec#1", n: int = 1_000,
            gain: float = 1.0, offset: float = 0.0, channel: int = 0) -> cp.ConditioningSource:
    return cp.ConditioningSource(doc_id=doc_id, channel=channel, label="rec · current",
                                 data=np.zeros(n, dtype=np.float32), fs=fs, units="pA",
                                 x_offset_s=0.0, cal_gain=gain, cal_offset=offset)


def _key(source: cp.ConditioningSource, params=CHAIN) -> tuple:
    return cp.memo_key(source.key, params, source.fs)


def test_the_memo_key_names_the_rate_whatever_the_source_key_holds():
    """Same document, channel, calibration, length and chain — only the rate differs. The pass at
    25 kHz must not be filed where the pass at 50 kHz was (S8-NUM1: 26 pA off, shown as exact).
    The chain's signature cannot tell them apart (its only use of ``fs`` is to prune a notch's
    lines at Nyquist, and this chain has no notch), so the key has to."""
    at_50 = cp.memo_key(RATE_LESS, CHAIN, 50_000.0)
    at_25 = cp.memo_key(RATE_LESS, CHAIN, 25_000.0)
    assert at_50 != at_25
    assert cp.memo_key(RATE_LESS, CHAIN, 50_000.0) == at_50, "the same inputs still meet"
    unknown = cp.memo_key(RATE_LESS, CHAIN, None)
    assert unknown not in (at_50, at_25), "no rate is not any rate (a legacy TDMS)"
    assert cp.memo_key(RATE_LESS, CHAIN, None) == unknown


def test_a_pass_stored_at_one_rate_is_not_found_at_another_and_both_can_be_held():
    """The memo a window consults: the old pass is invisible at the declared rate, comes back when
    the declaration is taken away, and the pass at the new rate sits beside it instead of over it."""
    memo = cp.ConditionedTraceMemo(capacity=2)
    pass_at_50 = np.arange(5, dtype=np.float32)
    pass_at_25 = pass_at_50 + 26.0
    memo.put(cp.memo_key(RATE_LESS, CHAIN, 50_000.0), pass_at_50)
    assert memo.get(cp.memo_key(RATE_LESS, CHAIN, 25_000.0)) is None
    memo.put(cp.memo_key(RATE_LESS, CHAIN, 25_000.0), pass_at_25)
    assert len(memo) == 2, "two rates, two passes — a shared key would have overwritten the first"
    assert memo.get(cp.memo_key(RATE_LESS, CHAIN, 50_000.0))[0] is pass_at_50
    assert memo.get(cp.memo_key(RATE_LESS, CHAIN, 25_000.0))[0] is pass_at_25


def test_the_rate_is_a_cheap_scalar_in_a_key_that_stays_hashable_and_findable_by_document():
    """No array, no object: the key is a flat tuple of scalars and one string, hashable, and
    ``drop_doc`` still finds a document's passes by the first element."""
    key = _key(_source())
    assert hash(key) == hash(_key(_source()))
    assert all(isinstance(part, (str, int, float, type(None))) for part in key), key
    assert key[0] == "rec#1"
    memo = cp.ConditionedTraceMemo(capacity=4)
    for fs in (50_000.0, 25_000.0):
        memo.put(_key(_source(fs)), np.zeros(2, dtype=np.float32))
    other = _key(_source(50_000.0, doc_id="rec#2"))
    memo.put(other, np.zeros(2, dtype=np.float32))
    memo.drop_doc("rec#1")
    assert memo.keys() == [other], "a closed document's passes go at every rate"


def test_everything_else_the_pass_depends_on_still_moves_the_key():
    """The fix adds to the key; it must not have loosened it. Calibration (gain, offset), the
    channel, the length and the document each still make another pass, and a unit relabel or a
    moved clock origin — which change no sample — still do not."""
    base = _key(_source())
    assert _key(_source(gain=1.1)) != base and _key(_source(offset=2.0)) != base
    assert _key(_source(channel=1)) != base and _key(_source(n=999)) != base
    assert _key(_source(doc_id="rec#2")) != base
    moved = cp.ConditioningSource(doc_id="rec#1", channel=0, label="renamed", data=np.zeros(
        1_000, dtype=np.float32), fs=50_000.0, units="nA", x_offset_s=12.5)
    assert cp.memo_key(moved.key, CHAIN, moved.fs) == base
    assert _key(_source(), complete_recipe({"detrend_enabled": True})) != base, "another chain"


def test_the_memo_still_holds_two_passes_and_evicts_the_least_recently_used():
    """Capacity is unchanged: three rates of one recording leave the two most recently used."""
    memo = cp.ConditionedTraceMemo(capacity=2)
    k50, k25, k20 = (_key(_source(fs)) for fs in (50_000.0, 25_000.0, 20_000.0))
    for key in (k50, k25):
        memo.put(key, np.zeros(2, dtype=np.float32))
    assert memo.get(k50) is not None, "touch the 50 kHz pass: 25 kHz is now the oldest"
    memo.put(k20, np.zeros(2, dtype=np.float32))
    assert memo.keys() == [k50, k20] and k25 not in memo
