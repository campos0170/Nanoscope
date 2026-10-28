"""S8-SC3 (analysis layer) — what identifies the samples the Signal Conditioning previews read
must include the rate they are read at.

MEASURED DEFECT (sweep 8). ``ConditioningSource.key`` was ``(doc, channel, gain, offset, n)``.
A time-base declaration (``Session.set_sample_rate_declaration``) re-reads the SAME samples at
another rate — nothing in the array changes, so nothing in that key did — and every product of
the window depends on the rate: the seconds on the x axis, the Welch grid and the Nyquist a
refusal quotes, the settling pad that decides "exact", and the whole-recording ("Compute
exact") pass. The window compared the key of the source it was pushed with the key of the one
it held, found them equal, and neither rescheduled nor discarded an answer that was still
coming back from the worker at the old rate (``SignalConditioningWindow.set_source`` /
``_on_outcome``); and the exact-pass memo, which is keyed ``(*source.key, chain signature)``,
answered for the new rate with the pass computed at the old one (S8-NUM1: 26 pA off a
4.5 pA noise sigma, under "Whole recording (exact)").

THE FIX is at the one place the key is built: ``ConditioningSource.key`` ends with the rate, so
the window's "same samples?" test, the stale-answer test and the memo key all see a re-read
recording as another source. These tests are the Qt-free half: the key itself and the memo that
is keyed by it. The window half is tests/gui/test_s8_sc3_open_signal_conditioning_follows_the_
time_base.py.
"""

from __future__ import annotations

import numpy as np

from nanoscope.analysis import conditioning_preview as cp
from nanoscope.analysis.conditioning_recipes import complete_recipe

FS = 50_000.0


def _source(fs: float | None = FS, *, doc_id: str = "doc#1", n: int = 1_000,
            gain: float = 1.0, offset: float = 0.0, x_offset_s: float | None = 0.0,
            data: np.ndarray | None = None) -> cp.ConditioningSource:
    data = np.zeros(n, dtype=np.float32) if data is None else data
    return cp.ConditioningSource(doc_id=doc_id, channel=0, label="rec · current", data=data,
                                 fs=fs, units="pA", x_offset_s=x_offset_s, cal_gain=gain,
                                 cal_offset=offset)


def test_the_same_samples_at_another_rate_are_another_source():
    """The declaration case: one array, two clocks. The key may not call them equal."""
    samples = np.zeros(1_000, dtype=np.float32)
    as_read = _source(50.0, data=samples)
    declared = _source(50_000.0, data=samples)
    assert as_read.data is declared.data, "nothing about the samples changed"
    assert as_read.key != declared.key


def test_a_source_re_pushed_unchanged_keeps_its_key():
    """The other direction matters as much: the window skips a recompute for a re-cached
    array, a renamed unit or a moved clock origin — so the key must NOT move for those."""
    a = _source(FS, data=np.zeros(1_000, dtype=np.float32))
    b = _source(FS, data=np.zeros(1_000, dtype=np.float32), x_offset_s=12.5)    # re-cached, shifted
    assert a.key == b.key
    assert _source(FS, gain=1.1).key != a.key and _source(FS, offset=2.0).key != a.key
    assert _source(FS, n=999).key != a.key and _source(FS, doc_id="doc#2").key != a.key


def test_a_recording_with_no_rate_has_a_key_of_its_own():
    """``fs is None`` (a legacy TDMS) is a state the key must be able to express and compare;
    it is not any rate, and an unknown rate that becomes known is a change."""
    unknown = _source(None)
    assert unknown.key == _source(None).key
    assert unknown.key != _source(FS).key


def test_the_exact_pass_memo_never_answers_for_another_rate():
    """S8-NUM1's mechanism: the memo key is the source key plus the chain's signature, so a
    pass stored at one rate is invisible at another and visible again when the rate comes back
    (undoing a declaration), while the same chain at the same rate still hits."""
    params = complete_recipe({"detrend_enabled": True, "filter_cutoff_hz": 5_000.0})
    old, new = _source(50_000.0), _source(25_000.0)
    key_old = cp.memo_key(old.key, params, old.fs)
    key_new = cp.memo_key(new.key, params, new.fs)
    assert key_old != key_new
    memo = cp.ConditionedTraceMemo(capacity=2)
    exact_at_old = np.arange(4, dtype=np.float32)
    memo.put(key_old, exact_at_old)
    assert memo.get(key_new) is None, "the pass computed at 50 kHz is not the pass at 25 kHz"
    assert memo.get(cp.memo_key(old.key, params, old.fs))[0] is exact_at_old
    memo.put(key_new, exact_at_old + 1)
    assert len(memo) == 2 and memo.get(key_old)[0] is exact_at_old


def test_closing_a_document_forgets_its_passes_at_every_rate():
    """``drop_doc`` finds a document's passes by the key's first element; adding the rate to
    the key may not hide a pass from it."""
    params = complete_recipe({"detrend_enabled": True})
    memo = cp.ConditionedTraceMemo(capacity=4)
    for fs in (50.0, 50_000.0):
        memo.put(cp.memo_key(_source(fs).key, params, fs), np.zeros(2, dtype=np.float32))
    other = cp.memo_key(_source(50_000.0, doc_id="doc#2").key, params, 50_000.0)
    memo.put(other, np.zeros(2, dtype=np.float32))
    memo.drop_doc("doc#1")
    assert memo.keys() == [other]
