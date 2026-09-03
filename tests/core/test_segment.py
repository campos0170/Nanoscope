"""Unit tests for signal segmentation — the DERIVED-RECORDING model (``core/segment.py``).

A *segment* is a crop of a parent recording turned into a brand-new
:class:`~nanoscope.core.document.TraceDocument` (a :class:`SegmentDocument`). These
tests pin the pure cropper :func:`build_segment_document`:

  * the crop yields the right sample count + duration, with the time axis rebased to 0;
  * every channel's kind / units / ``is_dead`` (and the nominal voltage) is inherited;
  * each channel's data equals the parent slice and is an OWNED copy (independent of the
    parent's array);
  * the :class:`SegmentOrigin` records parent + window + absolute t0 + uid + name;
  * ranges are clamped, and an empty / single-sample crop is rejected with a clear error;
  * :func:`derive_segment_uid` is deterministic (never random) and disambiguates;
  * :func:`persistent_key` returns a file's path but a segment's uid.

Two Session-backed tests (they need a QObject, so the ``qapp`` fixture) pin
:meth:`~nanoscope.core.session.Session.create_segment`: it inherits the parent's LIVE
per-channel calibration, emits ``documentOpened`` (dirty-for-free), and yields a document
that survives the parent being closed.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import (the Session-backed tests build a QObject). -------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.segment import (
    MIN_SEGMENT_SAMPLES,
    SegmentDocument,
    SegmentOrigin,
    build_segment_document,
    derive_segment_uid,
    persistent_key,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 50_000.0
N = 2_000


# ================================================================================================
# builders
# ================================================================================================

def _parent(*, fs: float | None = FS, t0_s: float = 3.0, n: int = N) -> TraceDocument:
    """A 2-channel parent: a live current (pA) + a dead voltage (mV), a known nominal voltage
    (so grouping can be checked), and a non-zero t0 (so rebasing is observable)."""
    cur = np.arange(n, dtype=np.float32)               # 0,1,2,… — trivially checkable slices
    volt = (np.arange(n, dtype=np.float32) * -0.25)
    info_cur = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    info_volt = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE,
                            is_dead=True, dead_reason="no signal")
    meta = RecordingMeta(
        path=Path("/data/pore_-700 mV.abf"), format_name="Axon ABF", n_samples=n,
        sample_rate_hz=fs, rate_provenance=(Provenance.UNKNOWN if fs is None else Provenance.FILE_META),
        channels=(info_cur, info_volt), t0_s=t0_s, nominal_voltage_mV=-700.0,
        voltage_provenance=Provenance.FILENAME)
    return TraceDocument(meta=meta, channels=(cur, volt))


# ================================================================================================
# crop geometry: sample count, duration, time rebased to 0
# ================================================================================================

def test_crop_yields_right_sample_count_and_duration():
    parent = _parent()
    seg = build_segment_document(parent, 400, 900, name="s", segment_uid="u")
    assert seg.meta.n_samples == 500
    assert all(a.size == 500 for a in seg.channels)
    assert seg.meta.duration_s == pytest.approx(500 / FS)


def test_time_axis_rebased_to_zero_but_absolute_t0_recorded():
    parent = _parent(t0_s=3.0)
    seg = build_segment_document(parent, 400, 900, name="s", segment_uid="u")
    assert seg.meta.t0_s == 0.0                                  # the segment's own clock starts at 0
    # …but the absolute experimental time of sample 400 in the parent is remembered.
    assert seg.origin.t0_abs_s == pytest.approx(3.0 + 400 / FS)


def test_rate_unknown_parent_does_not_fabricate_seconds():
    parent = _parent(fs=None, t0_s=0.0)
    seg = build_segment_document(parent, 100, 250, name="s", segment_uid="u")
    assert seg.meta.sample_rate_hz is None
    assert seg.meta.duration_s is None                          # no rate -> no seconds
    assert seg.origin.t0_abs_s == pytest.approx(0.0)           # not fabricated from a fake fs


# ================================================================================================
# inheritance: kinds / units / is_dead / nominal voltage
# ================================================================================================

def test_channel_kinds_units_isdead_inherited():
    parent = _parent()
    seg = build_segment_document(parent, 0, 500, name="s", segment_uid="u")
    assert len(seg.meta.channels) == 2
    c0, c1 = seg.meta.channels
    assert (c0.kind, c0.raw_units, c0.name, c0.is_dead) == (ChannelKind.CURRENT, "pA", "IN15", False)
    assert (c1.kind, c1.raw_units, c1.name, c1.is_dead) == (ChannelKind.VOLTAGE, "mV", "IN14", True)


def test_nominal_voltage_inherited_so_segment_groups_with_parent():
    parent = _parent()
    seg = build_segment_document(parent, 0, 500, name="s", segment_uid="u")
    assert seg.meta.nominal_voltage_mV == -700.0
    assert seg.meta.voltage_group_mV == parent.meta.voltage_group_mV   # same Files-list group


# ================================================================================================
# data: equals the parent slice, and is an OWNED copy
# ================================================================================================

def test_per_channel_data_equals_parent_slice():
    parent = _parent()
    i0, i1 = 400, 900
    seg = build_segment_document(parent, i0, i1, name="s", segment_uid="u")
    assert np.array_equal(seg.channels[0], parent.channels[0][i0:i1])
    assert np.array_equal(seg.channels[1], parent.channels[1][i0:i1])


def test_segment_channels_are_owned_float32_contiguous_copies():
    parent = _parent()
    seg = build_segment_document(parent, 400, 900, name="s", segment_uid="u")
    for a in seg.channels:
        assert a.dtype == np.float32
        assert a.flags["C_CONTIGUOUS"]
        assert a.base is None                       # a real copy — NOT a view into the parent

    # Mutating the parent's array must not bleed into the segment (proves independence).
    before = seg.channels[0].copy()
    parent.channels[0][400:900] += 1000.0
    assert np.array_equal(seg.channels[0], before)


# ================================================================================================
# origin stamp + identity
# ================================================================================================

def test_origin_records_parent_window_uid_name():
    parent = _parent()
    seg = build_segment_document(parent, 400, 900, name="my seg", segment_uid="uid-9",
                                 parent_key="/data/pore_-700 mV.abf")
    o = seg.origin
    assert isinstance(o, SegmentOrigin)
    assert (o.parent_key, o.i0, o.i1, o.segment_uid, o.name) == (
        "/data/pore_-700 mV.abf", 400, 900, "uid-9", "my seg")


def test_segment_is_a_tracedocument_with_unique_pathderived_docid():
    parent = _parent()
    seg = build_segment_document(parent, 400, 900, name="my seg", segment_uid="uid-9")
    assert isinstance(seg, TraceDocument)                        # every per-recording path runs on it
    assert isinstance(seg, SegmentDocument)
    # SOURCE identity is the resolved synthetic path — the SAME contract every module uses
    # to stamp a result (AnalysisResult.doc_id = identity.doc_id_for_meta(ctx.meta)), so
    # detection lands on the segment, not the parent.
    assert seg.source_key == str(seg.meta.path.resolve())
    assert "uid-9" in seg.source_key                             # the uid is folded into the path
    assert seg.source_key != parent.source_key                   # distinct source -> no clash
    assert seg.display_name == "my seg"                          # …but the NAME is the user's
    # A segment's stem_key must not collide with the parent's (no spurious duplicate banner).
    assert seg.meta.stem_key != parent.meta.stem_key
    # Re-deriving from the same parent + uid is deterministic in every respect that persists
    # (the same synthetic source, the same uid) — see the next test for what does NOT.
    again = build_segment_document(parent, 400, 900, name="renamed", segment_uid="uid-9")
    assert again.source_key == seg.source_key
    assert again.origin.segment_uid == seg.origin.segment_uid


def test_a_re_derived_segment_is_a_new_load_and_says_so():
    """Two crops with the same uid are the same RECORDING and two different LOADS.

    ``doc_id`` carries a per-load token (``core.identity``), and a segment is not exempt:
    it is built by ``TraceDocument.__post_init__`` like everything else, so re-deriving one
    — which is exactly what a project restore does — mints a fresh id while its PERSISTENT
    key (the uid) stays put. That split is what lets the restore re-attach the saved results
    while an in-flight detection from the previous derivation still gets dropped.
    """
    parent = _parent()
    a = build_segment_document(parent, 400, 900, name="s", segment_uid="uid-9")
    b = build_segment_document(parent, 400, 900, name="s", segment_uid="uid-9")
    assert a.doc_id != b.doc_id                       # different LOADS
    assert persistent_key(a) == persistent_key(b) == "uid-9"     # same RECORDING
    assert a.meta.load_uid and b.meta.load_uid and a.meta.load_uid != b.meta.load_uid
    # A segment never inherits its parent's token: it is its own load, not a view of one.
    assert a.meta.load_uid != parent.meta.load_uid


def test_spec_round_trips_the_origin():
    parent = _parent()
    seg = build_segment_document(parent, 400, 900, name="my seg", segment_uid="uid-9")
    s = seg.spec
    # parent_key is the PERSISTENT key of the parent — its path, never its per-load doc_id,
    # which would be a dangling reference the moment the project is reopened.
    assert (s.segment_uid, s.parent_key, s.i0, s.i1, s.name) == (
        "uid-9", parent.source_key, 400, 900, "my seg")
    assert persistent_key(parent) == s.parent_key


# ================================================================================================
# clamping + guards on bad ranges
# ================================================================================================

def test_ranges_are_clamped_and_reordered():
    parent = _parent()
    # Reversed + out of range: clamps to [0, N] and orders low->high.
    seg = build_segment_document(parent, 10_000, -50, name="s", segment_uid="u")
    assert seg.origin.i0 == 0 and seg.origin.i1 == N
    assert seg.meta.n_samples == N


@pytest.mark.parametrize("i0,i1", [(500, 500), (500, 501), (0, 1), (N, N)])
def test_empty_or_single_sample_crop_is_rejected(i0, i1):
    parent = _parent()
    with pytest.raises(ValueError, match="at least"):
        build_segment_document(parent, i0, i1, name="s", segment_uid="u")
    assert (i1 - i0) < MIN_SEGMENT_SAMPLES


def test_non_trace_parent_is_rejected():
    with pytest.raises(TypeError):
        build_segment_document(object(), 0, 10, name="s", segment_uid="u")  # type: ignore[arg-type]


# ================================================================================================
# uid derivation + persistent key
# ================================================================================================

def test_derive_uid_is_deterministic_and_disambiguates():
    a = derive_segment_uid("/data/pore.abf", 100, 300)
    b = derive_segment_uid("/data/pore.abf", 100, 300)
    assert a == b                                        # deterministic — no uuid4/random
    assert a.startswith("seg-")
    # A different window -> a different id.
    assert derive_segment_uid("/data/pore.abf", 100, 400) != a
    # The base id already taken -> a counter-disambiguated id, still deterministic.
    dis = derive_segment_uid("/data/pore.abf", 100, 300, existing={a})
    assert dis == f"{a}-2"
    assert derive_segment_uid("/data/pore.abf", 100, 300, existing={a, dis}) == f"{a}-3"


def test_persistent_key_is_path_for_files_and_uid_for_segments():
    parent = _parent()
    # A file's persistent key is its SOURCE, never its doc_id: the doc_id carries this
    # session's per-load token, so persisting it would mean no project could ever re-attach
    # anything. The VALUE is unchanged from before that token existed, which is what keeps
    # projects written by either build readable by both.
    assert persistent_key(parent) == parent.source_key        # file -> its resolved path
    assert persistent_key(parent) == str(parent.meta.path.resolve())
    assert persistent_key(parent) != parent.doc_id
    seg = build_segment_document(parent, 400, 900, name="s", segment_uid="uid-9")
    assert persistent_key(seg) == "uid-9"                     # segment -> its uid (NOT "segment:uid-9")


def test_nested_segment_parent_key_is_parent_segment_uid():
    parent = _parent()
    seg = build_segment_document(parent, 0, 1000, name="s", segment_uid="uid-a")
    nested = build_segment_document(seg, 100, 400, name="n", segment_uid="uid-b")
    assert nested.origin.parent_key == "uid-a"               # traces to the parent SEGMENT, not the file


# ================================================================================================
# Session.create_segment (needs a QObject -> the qapp fixture)
# ================================================================================================

def test_create_segment_inherits_calibration_and_emits_opened(qapp):
    from nanoscope.core.calibration import ChannelCalibration
    from nanoscope.core.session import Session

    session = Session()
    pid = session.add_document(_parent())
    # A non-default USER calibration on the current channel — the segment must inherit THIS.
    session.set_calibration(pid, 0, ChannelCalibration(
        gain=0.05, offset=2.0, target_units="pA", provenance=Provenance.UNKNOWN))

    opened: list[str] = []
    session.documentOpened.connect(opened.append)
    seg_id = session.create_segment(pid, 500, 1500, name="seg A")

    assert opened == [seg_id]                                 # documentOpened -> dirty-for-free
    assert isinstance(session.document(seg_id), SegmentDocument)
    seg_cal = session.calibration(seg_id, 0)
    assert (seg_cal.gain, seg_cal.offset, seg_cal.target_units) == (0.05, 2.0, "pA")
    # The calibrated segment signal equals the calibrated parent slice (same gain/offset applied).
    parent_cal = session.calibrated(pid, 0).data
    assert np.allclose(session.calibrated(seg_id, 0).data, parent_cal[500:1500])


def test_segment_survives_parent_close(qapp):
    from nanoscope.core.session import Session

    session = Session()
    pid = session.add_document(_parent())
    seg_id = session.create_segment(pid, 500, 1500, name="seg A")
    expected = session.calibrated(seg_id, 0).data.copy()

    session.close_document(pid)                              # the parent goes away…
    assert session.has_document(seg_id)                      # …the derived segment does not
    assert np.array_equal(session.calibrated(seg_id, 0).data, expected)  # its own signal intact
    assert session.segment_documents()[0].doc_id == seg_id


# ================================================================================================
# N5 — a segment boundary that lands ON an event
#
# Segmenting is how the user is *encouraged* to slice a recording, so the boundary cutting an
# event in half is the common case, not the corner one. Detection on the segment used to report
# the two halves as complete, accepted events (see
# ``tests/analysis/test_detect.py`` for the pA-level numbers). This is the same claim through
# the real ``build_segment_document`` -> ``detect_events`` path the app takes.
# ================================================================================================

def _event_parent(*, fs: float = FS, n: int = 60_000, seed: int = 41) -> TraceDocument:
    """A parent recording with one 2.4 ms event whose midpoint is a nice round index."""
    rng = np.random.default_rng(seed)
    x = np.full(n, -1000.0) + rng.normal(0.0, 12.0, n)
    for s, e in [(10_000, 10_120), (30_000, 30_120), (50_000, 50_120)]:
        x[s:e] -= 300.0
    info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=Path("/data/pore_-600 mV.abf"), format_name="Axon ABF", n_samples=n,
        sample_rate_hz=fs, rate_provenance=Provenance.FILE_META, channels=(info,), t0_s=0.0)
    return TraceDocument(meta=meta, channels=(np.ascontiguousarray(x, np.float32),))


def test_a_segment_boundary_through_an_event_flags_both_halves():
    from nanoscope.analysis.detect import detect_events

    parent = _event_parent()
    mid = 30_060                                     # the middle of the 30_000..30_120 event

    head = build_segment_document(parent, 5_000, mid, name="A", segment_uid="a")
    tail = build_segment_document(parent, mid, 55_000, name="B", segment_uid="b")

    hdf, hprov = detect_events(np.asarray(head.channels[0]), FS,
                               baseline=-1000.0, min_dwell_s=1e-4)
    tdf, tprov = detect_events(np.asarray(tail.channels[0]), FS,
                               baseline=-1000.0, min_dwell_s=1e-4)

    # One clipped row on each side of the cut -- the two halves of one real event.
    assert hprov["n_clipped_boundary"] == 1
    assert tprov["n_clipped_boundary"] == 1
    assert hdf.iloc[-1]["flag"] == "clipped_by_boundary"
    assert tdf.iloc[0]["flag"] == "clipped_by_boundary"
    assert not bool(hdf.iloc[-1]["accepted"])
    assert not bool(tdf.iloc[0]["accepted"])

    # The events wholly inside each segment are untouched.
    assert (hdf.iloc[:-1]["flag"] == "").all()
    assert bool(hdf.iloc[:-1]["accepted"].all())
    assert (tdf.iloc[1:]["flag"] == "").all()
    assert bool(tdf.iloc[1:]["accepted"].all())


def test_a_segment_cut_between_events_flags_nothing():
    """The control: a boundary in open pore is not a clipped event."""
    from nanoscope.analysis.detect import detect_events

    parent = _event_parent()
    seg = build_segment_document(parent, 20_000, 40_000, name="clean", segment_uid="c")
    df, prov = detect_events(np.asarray(seg.channels[0]), FS,
                             baseline=-1000.0, min_dwell_s=1e-4)
    assert prov["n_clipped_boundary"] == 0
    assert len(df) == 1                               # just the 30_000 event, whole
    assert df.iloc[0]["flag"] == ""
    assert bool(df.iloc[0]["accepted"])
