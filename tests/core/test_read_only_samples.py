"""A recording's samples cannot be written in memory (design §5.1, decision 7).

``TraceDocument`` was frozen in its attributes and not in its samples. At an identity
calibration ``Session.calibrated`` hands out the recording's OWN array, and at any other the
one cached calibrated copy is shared by reference, so a single in-place ``arr -= x`` anywhere
changed what every surface read for the rest of the session — with nothing said. The arrays
now carry the read-only flag (without a copy), and such a write raises at the line that
tried it, the way the worker snapshots already do.

Synthetic documents only; nothing here reads a recording from disk.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import (Session is a QObject). ---------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.segment import build_segment_document, follow_parent_clock
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 50_000.0
N = 2_000


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return (rng.normal(-100.0, 5.0, N).astype(np.float32),
            np.full(N, -6000.0, dtype=np.float32))


def _doc(arrays=None, *, name: str = "rec.dat") -> TraceDocument:
    cur = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    volt = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(path=Path("/synthetic") / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(cur, volt))
    return TraceDocument(meta=meta, channels=_arrays() if arrays is None else arrays)


def _assert_unwritable(arr: np.ndarray) -> None:
    """Every spelling of an in-place write raises, and none of them landed."""
    before = arr.copy()
    assert not arr.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        arr[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        arr[10:20] -= np.float32(1.0)
    with pytest.raises(ValueError, match="read-only"):
        np.multiply(arr, np.float32(2.0), out=arr)
    with pytest.raises(ValueError, match="read-only"):
        arr.fill(0.0)
    view = arr[100:200]                       # a view is no way round it
    with pytest.raises(ValueError, match="read-only"):
        view += np.float32(1.0)
    assert np.array_equal(arr, before)


# ================================================================================================
# the document
# ================================================================================================

def test_every_channel_is_read_only_once_the_document_is_built():
    doc = _doc()
    for ch in range(len(doc.channels)):
        _assert_unwritable(doc.raw(ch))


def test_the_flag_is_set_without_a_copy():
    """180 MB a channel: the read-only document is the array the reader handed over."""
    arrays = _arrays()
    doc = _doc(arrays)
    for given, held in zip(arrays, doc.channels):
        assert held is given


def test_a_rebuilt_document_keeps_the_same_read_only_arrays():
    """``dataclasses.replace`` is how the app re-makes a document (sidecar issues, a time-base
    declaration); it re-enters ``__post_init__`` with the same arrays."""
    doc = _doc()
    again = replace(doc, meta=replace(doc.meta, issues=("a note",)))
    for old, new in zip(doc.channels, again.channels):
        assert new is old
        _assert_unwritable(new)


def test_a_copy_is_the_way_to_write():
    """A caller that needs to write takes a copy, and the recording does not move."""
    doc = _doc()
    work = np.array(doc.raw(0))
    work -= np.float32(5.0)
    assert work.flags.writeable
    assert not np.array_equal(work, doc.raw(0))
    assert np.array_equal(doc.raw(0) - np.float32(5.0), work)


# ================================================================================================
# segments
# ================================================================================================

def test_a_segments_channels_are_read_only():
    parent = _doc()
    seg = build_segment_document(parent, 400, 900, name="seg", segment_uid="u")
    for a in seg.channels:
        assert a.base is None, "still the segment's own copy, not a view of the parent"
        _assert_unwritable(a)
    # ...and after it follows its parent onto another clock (a replace, like any rebuild)
    followed = follow_parent_clock(seg, parent)
    for a in followed.channels:
        _assert_unwritable(a)


def test_a_segment_cut_through_the_session_is_read_only(qapp):
    session = Session()
    pid = session.add_document(_doc())
    seg_id = session.create_segment(pid, 500, 1500, name="seg A")
    for ch in range(2):
        _assert_unwritable(session.document(seg_id).raw(ch))
        _assert_unwritable(session.calibrated(seg_id, ch).data)


# ================================================================================================
# the session's calibrated arrays
# ================================================================================================

def test_an_identity_calibration_hands_out_the_recordings_own_read_only_array(qapp):
    session = Session()
    doc = _doc()
    doc_id = session.add_document(doc)
    assert session.calibration(doc_id, 0).fingerprint() == (1.0, 0.0)
    data = session.calibrated_array(doc_id, 0)
    assert data is doc.raw(0)
    _assert_unwritable(data)
    _assert_unwritable(session.calibrated(doc_id, 0).data)


def test_a_cached_calibrated_array_is_read_only(qapp):
    """The voltage channel opens at gain 0.1 (its -6000 mV reads as the 10x encoding), so its
    calibrated array is the cached copy every surface shares."""
    session = Session()
    doc_id = session.add_document(_doc())
    assert session.calibration(doc_id, 1).gain == pytest.approx(0.1)
    data = session.calibrated_array(doc_id, 1)
    assert data is session.calibrated_array(doc_id, 1), "one shared cached copy"
    _assert_unwritable(data)
    _assert_unwritable(session.calibrated(doc_id, 1).data)


def test_a_calibration_change_still_works_and_the_values_are_right(qapp):
    session = Session()
    doc = _doc()
    raw = doc.raw(0).copy()
    doc_id = session.add_document(doc)

    session.set_calibration(doc_id, 0, ChannelCalibration(gain=2.0, offset=1.0,
                                                          target_units="pA"),
                            write_sidecar=False)
    scaled = session.calibrated(doc_id, 0).data
    expected = raw * np.float32(2.0) + np.float32(1.0)
    assert scaled.dtype == np.float32
    assert np.array_equal(scaled, expected)
    _assert_unwritable(scaled)

    # A -> B -> A: the identity comes back as the recording itself, untouched by any of it.
    session.set_calibration(doc_id, 0, ChannelCalibration(gain=0.5), write_sidecar=False)
    assert np.array_equal(session.calibrated(doc_id, 0).data, raw * np.float32(0.5))
    session.set_calibration(doc_id, 0, ChannelCalibration(), write_sidecar=False)
    assert session.calibrated(doc_id, 0).data is doc.raw(0)
    assert np.array_equal(doc.raw(0), raw)
