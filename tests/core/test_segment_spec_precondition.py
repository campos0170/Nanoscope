"""H-2 (pure half) — a saved segment window is an ASSERTION, and the builder's clamp is not.

``build_segment_document`` clamps ``[i0, i1)`` into ``[0, n]`` deliberately: a live
rubber-band cut is a REQUEST and the drag may legitimately run past the end of the trace.
``tests/core/test_segment.py`` pins that, and it stays pinned.

A :class:`SegmentRec` restored from a project is the opposite kind of statement — "this
segment IS samples 40,000-60,000 of that recording" — and a whole curated event table is
filed under its identity. Re-deriving it against a shorter parent substituted a different
recording under the same identity and said nothing.

So the rule lives next to the clamp it excepts, and this file pins BOTH halves plus the
exact boundary between them.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.core import segment as S
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta


def _api(name: str):
    obj = getattr(S, name, None)
    assert obj is not None, f"nanoscope.core.segment has no {name!r}"
    return obj


def _doc(n: int, tmp_path) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "rec.csv", format_name="synthetic", n_samples=n,
                         sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(np.zeros(n, dtype=np.float32),))


def test_a_window_that_still_fits_is_not_refused():
    refusal = _api("spec_window_refusal")
    assert refusal(100_000, 40_000, 60_000) is None
    assert refusal(60_000, 40_000, 60_000) is None, "exactly to the end still fits"
    assert refusal(100_000, 0, 2) is None


def test_a_window_needing_samples_the_parent_no_longer_has_is_refused():
    """THE SILENT BAND. Measured before: a [40 000, 60 000) segment whose parent was
    re-exported at 45 000 samples came back as [40 000, 45 000) — a QUARTER of the analysed
    window — with ``meta.issues == ()``, all 4 saved events reattached and 3 of them lying
    entirely past the end of the segment they were attached to."""
    refusal = _api("spec_window_refusal")
    why = refusal(45_000, 40_000, 60_000)
    assert why is not None
    assert "60,000" in why and "45,000" in why


@pytest.mark.parametrize("n", [40_001, 44_999, 59_999])
def test_every_length_in_the_partial_overlap_band_is_refused(n):
    """The band is exactly ``i0 <= n-2 < i1``. The app already got the ADJACENT case right —
    a window landing entirely past the new end leaves <2 samples, the builder raises, and
    the restore reports it — and lost it on the sibling."""
    assert _api("spec_window_refusal")(n, 40_000, 60_000) is not None


def test_the_degenerate_shapes_are_refused_too():
    refusal = _api("spec_window_refusal")
    assert refusal(100_000, -5, 60_000) is not None
    assert refusal(100_000, 60_000, 40_000) is not None
    assert refusal(100_000, 40_000, 40_001) is not None, "a 1-sample segment carries nothing"


def test_the_live_cut_still_clamps(tmp_path):
    """PASSES BOTH — the premise, deliberately. If this ever fails, the fix has been
    applied to the builder instead of to the restore and a rubber-band drag past the end of
    the trace has stopped working. ``tests/core/test_segment.py:214`` pins the same thing;
    it is repeated here so a reader of the refusal finds the exception beside it."""
    parent = _doc(45_000, tmp_path)
    seg = S.build_segment_document(parent, 40_000, 60_000, name="cut", segment_uid="u")
    assert (seg.origin.i0, seg.origin.i1) == (40_000, 45_000)
    assert seg.meta.n_samples == 5_000


def test_the_builder_still_raises_where_it_always_did(tmp_path):
    """PASSES BOTH — the adjacent branch the app already handled. The refusal must not
    change what happens here, only cover the sibling case beside it."""
    parent = _doc(45_000, tmp_path)
    with pytest.raises(ValueError):
        S.build_segment_document(parent, 50_000, 60_000, name="gone", segment_uid="u")
    with pytest.raises(ValueError):
        S.build_segment_document(parent, 44_999, 60_000, name="one", segment_uid="u")


def test_the_refusal_and_the_builder_agree_across_the_whole_boundary(tmp_path):
    """The property, over the plane rather than at a point: for every parent length, a
    window the refusal accepts must build to EXACTLY that window, and one it rejects must
    either raise or (the silent band) build to a different one."""
    for n in range(39_998, 60_002, 313):
        parent = _doc(n, tmp_path)
        why = _api("spec_window_refusal")(n, 40_000, 60_000)
        try:
            seg = S.build_segment_document(parent, 40_000, 60_000, name="c", segment_uid="u")
        except ValueError:
            assert why is not None, (
                f"the builder refused a window the precondition allowed at n={n}")
            continue
        got = (seg.origin.i0, seg.origin.i1)
        if why is None:
            assert got == (40_000, 60_000), f"n={n}: accepted but re-derived at {got}"
        else:
            assert got != (40_000, 60_000), f"n={n}: refused a window that rebuilt exactly"
