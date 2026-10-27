"""A user's sample-rate / time-unit declaration: resolved, applied by rebuilding the document,
undoable, inherited by segments, saved with the project, never written beside the recording."""
from copy import deepcopy

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.project import (
    AnalyticalWorkspace, DocumentRef, ProjectState, load_project, save_project,
)
from nanoscope.core.sample_rate import TIME_UNITS, SampleRateDeclaration, TimeBaseAsRead
from nanoscope.core.sample_rate_state import (
    UNREADABLE_KEY, WORKSPACE_KEY, capture_sample_rate_declarations,
    prepare_sample_rate_workspace, restore_sample_rate_declarations,
)
from nanoscope.core.segment import persistent_key
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

MS = SampleRateDeclaration("time_unit", time_unit="ms", note="exporter writes ms")
HZ = SampleRateDeclaration("sample_rate", sample_rate_hz=12_345.0)


def _doc(path, *, rate=100.0, prov=Provenance.MEASURED, t0=0.05, scale=1.0, refused=None):
    data = np.random.default_rng(813).normal(120, 1, 12000).astype(np.float32)
    infos = (ChannelInfo(0, "current", "", ChannelKind.CURRENT),)
    meta = RecordingMeta(path, "Text trace", len(data), rate, prov, infos, t0_s=t0,
                         time_scale_s=scale, refused_rate_hz=refused)
    return TraceDocument(meta, (data,))


def _live(tmp_path, **kw):
    session = Session()
    doc = _doc(tmp_path / "recording.csv", **kw)
    return session, doc, session.add_document(doc)


# -- the value object ------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(mode="minutes"),
    dict(mode="time_unit", time_unit="min"),
    dict(mode="time_unit", time_unit="ms", sample_rate_hz=10.0),
    dict(mode="sample_rate", sample_rate_hz=0),
    dict(mode="sample_rate", sample_rate_hz=float("nan")),
    dict(mode="sample_rate", sample_rate_hz=float("inf")),
    dict(mode="sample_rate", sample_rate_hz=True),
    dict(mode="sample_rate", sample_rate_hz="1000"),
    dict(mode="sample_rate", sample_rate_hz=1000.0, time_unit="s"),
    dict(mode="time_unit", time_unit="ms", source="file"),
])
def test_a_declaration_that_cannot_be_right_is_refused_at_construction(kwargs):
    with pytest.raises(ValueError):
        SampleRateDeclaration(**kwargs)


def test_resolution_scales_rate_and_start_time_together():
    read = TimeBaseAsRead(100.0, Provenance.MEASURED, 0.05, 1.0)     # a ms column read as s
    assert MS.resolve(read) == pytest.approx((100_000.0, 5e-5, 1e-3))
    # ...and relative to how the column was ALREADY read: a header said ms, the user says µs.
    read_ms = TimeBaseAsRead(100_000.0, Provenance.MEASURED, 5e-5, 1e-3)
    us = SampleRateDeclaration("time_unit", time_unit="us")
    assert us.resolve(read_ms) == pytest.approx((100_000_000.0, 5e-8, 1e-6))
    assert HZ.resolve(read) == pytest.approx((12_345.0, 0.05, 1.0)), "a rate leaves t0 as read"
    with pytest.raises(ValueError, match="no usable time column"):
        MS.resolve(TimeBaseAsRead(None, Provenance.UNKNOWN, 0.0, 1.0))


def test_json_round_trip_is_strict_both_ways():
    for decl in (MS, HZ):
        assert SampleRateDeclaration.from_dict(decl.to_dict()) == decl
    with pytest.raises(ValueError, match="unrecognized"):
        SampleRateDeclaration.from_dict({**MS.to_dict(), "future": 1})
    with pytest.raises(ValueError):
        SampleRateDeclaration.from_dict("ms")
    assert set(TIME_UNITS) == {"s", "ms", "us", "ns"}


# -- the Session mutator ---------------------------------------------------------------------

def test_declaring_rebuilds_the_document_and_resetting_restores_it_exactly(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    axis_before = session.x_axis(key)
    hits, cal_hits, result_hits = [], [], []
    session.sampleRateChanged.connect(lambda d: hits.append(d))
    session.calibrationChanged.connect(lambda *a: cal_hits.append(a))
    session.resultsChanged.connect(lambda *a: result_hits.append(a))

    session.set_sample_rate_declaration(key, MS)
    meta = session.document(key).meta
    assert meta.sample_rate_hz == pytest.approx(100_000.0)
    assert meta.rate_provenance is Provenance.USER
    assert meta.t0_s == pytest.approx(5e-5)
    assert meta.time_scale_s == 1e-3
    assert session.document(key).doc_id == key, "a rebuilt meta keeps load_uid, so the id"
    assert session.document(key).channels[0] is doc.channels[0], "arrays shared, never copied"
    axis = session.x_axis(key)
    assert axis.values is not axis_before.values, "the cached axis was dropped"
    assert axis.values[1] - axis.values[0] == pytest.approx(1e-5)
    assert session.sample_rate_declaration(key) == MS
    assert session.time_base_as_read(key) == TimeBaseAsRead(100.0, Provenance.MEASURED, 0.05, 1.0,
                                                            None)

    session.set_sample_rate_declaration(key, MS)        # idempotent: no second emit
    session.set_sample_rate_declaration(key, HZ)
    meta = session.document(key).meta
    assert (meta.sample_rate_hz, meta.t0_s, meta.time_scale_s) == (12_345.0, 0.05, 1.0)

    session.set_sample_rate_declaration(key, None)
    meta = session.document(key).meta
    assert (meta.sample_rate_hz, meta.rate_provenance, meta.t0_s, meta.time_scale_s) == (
        100.0, Provenance.MEASURED, 0.05, 1.0)
    assert session.sample_rate_declaration(key) is None
    assert hits == [key, key, key]
    assert cal_hits == result_hits == []
    assert list(tmp_path.iterdir()) == [], "nothing is written beside the recording"


def test_only_a_user_declaration_is_accepted(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    for value in ({}, "ms", 1000.0, MS.to_dict()):
        with pytest.raises(ValueError):
            session.set_sample_rate_declaration(key, value)
    with pytest.raises(KeyError):
        session.set_sample_rate_declaration("no such document", MS)
    assert session.document(key).meta.sample_rate_hz == 100.0


def test_a_rateless_recording_takes_a_rate_but_not_a_unit(qapp, tmp_path):
    """The legacy-TDMS case: nothing to re-read in another unit, so only a rate can be
    declared — and declaring one turns the samples axis into seconds."""
    session, doc, key = _live(tmp_path, rate=None, prov=Provenance.UNKNOWN, t0=0.0)
    assert session.x_axis(key).unit == "sample"
    with pytest.raises(ValueError, match="declare its sample rate instead"):
        session.set_sample_rate_declaration(key, MS)
    assert session.document(key).meta.sample_rate_hz is None, "a refusal changes nothing"
    session.set_sample_rate_declaration(key, HZ)
    assert session.document(key).meta.sample_rate_hz == 12_345.0
    assert session.x_axis(key).unit == "s"
    session.set_sample_rate_declaration(key, None)
    assert session.x_axis(key).unit == "sample"


def test_segments_follow_the_parent_s_declaration_and_refuse_one_of_their_own(qapp, tmp_path):
    """A segment is a window of its parent: cut before OR after the declaration, it is on the
    parent's clock (rate, provenance, t0_abs), and so is a segment of a segment; it refuses a
    declaration of its own; the parent's reset returns every one of them."""
    session, doc, key = _live(tmp_path)
    before = session.create_segment(key, 1000, 3000, name="cut before")
    nested = session.create_segment(before, 100, 500, name="nested")
    hits = []
    session.sampleRateChanged.connect(hits.append)
    session.set_sample_rate_declaration(key, MS)
    after = session.create_segment(key, 5000, 6000, name="cut after")
    for seg_id, i0 in ((before, 1000), (after, 5000)):
        seg = session.document(seg_id)
        assert seg.meta.sample_rate_hz == pytest.approx(100_000.0), seg_id
        assert seg.meta.rate_provenance is Provenance.USER
        assert seg.meta.time_scale_s == 1e-3
        assert seg.origin.t0_abs_s == pytest.approx(5e-5 + i0 / 100_000.0)
        assert "t₀=" in seg.meta.notes[0]
        assert session.sample_rate_declaration(seg_id) is None
    assert session.document(nested).meta.sample_rate_hz == pytest.approx(100_000.0)
    assert session.document(nested).origin.t0_abs_s == pytest.approx(5e-5 + 1100 / 100_000.0)
    assert hits == [key, before, nested], "the parent first, then each segment, each once"
    with pytest.raises(ValueError, match="follows the time base"):
        session.set_sample_rate_declaration(before, HZ)
    session.set_sample_rate_declaration(key, None)
    for seg_id, i0 in ((before, 1000), (nested, 1100), (after, 5000)):
        seg = session.document(seg_id)
        assert seg.meta.sample_rate_hz == 100.0 and seg.meta.rate_provenance is Provenance.MEASURED
        assert seg.origin.t0_abs_s == pytest.approx(0.05 + i0 / 100.0)
    session.close_document(key)
    with pytest.raises(KeyError):
        session.sample_rate_declaration(key)
    assert key not in session._rate_declarations and key not in session._time_base_as_read


def test_an_orphan_segment_follows_its_parent_when_the_parent_lands_again(qapp, tmp_path):
    """Declare on P, cut S, close P, reopen P (the file's own clock, no declaration): S is
    on P's clock again — as it would be after a project reopen — and a segment already on
    its parent's clock is left alone (nothing fires)."""
    session, doc, key = _live(tmp_path)
    session.set_sample_rate_declaration(key, MS)
    seg = session.create_segment(key, 1000, 3000, name="orphan")
    session.close_document(key)
    assert session.document(seg).meta.sample_rate_hz == pytest.approx(100_000.0)
    hits = []
    session.sampleRateChanged.connect(hits.append)
    again = session.add_document(_doc(tmp_path / "recording.csv"))      # same persistent key
    assert session.document(seg).meta.sample_rate_hz == 100.0
    assert session.document(seg).meta.rate_provenance is Provenance.MEASURED
    assert session.document(seg).origin.t0_abs_s == pytest.approx(0.05 + 1000 / 100.0)
    assert hits == [seg]
    session.close_document(again)
    session.add_document(_doc(tmp_path / "recording.csv"))
    assert hits == [seg], "already on this clock: nothing moved, nothing fired"


def test_a_nested_orphan_follows_a_parent_segment_re_cut_at_the_same_window(qapp, tmp_path):
    """Cut S from P and N from S; close S (N is an orphan keyed by S's deterministic uid);
    declare on P (only live children follow); cut S again at the same window — same uid —
    and N must be on the new S's clock."""
    session, doc, key = _live(tmp_path)
    s = session.create_segment(key, 1000, 5000, name="S")
    n = session.create_segment(s, 100, 500, name="N")
    uid = session.document(s).origin.segment_uid
    session.close_document(s)
    session.set_sample_rate_declaration(key, MS)
    assert session.document(n).meta.sample_rate_hz == 100.0, "orphaned: not yet followed"
    s_again = session.create_segment(key, 1000, 5000, name="S")
    assert session.document(s_again).origin.segment_uid == uid
    assert session.document(n).meta.sample_rate_hz == pytest.approx(100_000.0)
    assert session.document(n).origin.t0_abs_s == pytest.approx(5e-5 + 1100 / 100_000.0)


def test_an_orphan_follows_a_parent_reopened_under_another_spelling_of_its_path(qapp, tmp_path):
    """The app answers 'same recording' with same_source (os.path.samefile), never string
    equality; the orphan follow must too."""
    import os
    upper, lower = tmp_path / "Case.csv", tmp_path / "case.csv"
    upper.write_text("fixture")
    if not (lower.exists() and os.path.samefile(upper, lower)):
        pytest.skip("needs a case-insensitive filesystem")
    session = Session()
    key = session.add_document(_doc(upper))
    session.set_sample_rate_declaration(key, MS)
    seg = session.create_segment(key, 1000, 3000, name="orphan")
    session.close_document(key)
    session.add_document(_doc(lower))
    assert session.document(seg).meta.sample_rate_hz == 100.0
    assert session.document(seg).meta.rate_provenance is Provenance.MEASURED


def test_a_live_declaration_supersedes_a_dead_record_for_the_same_recording(qapp, tmp_path):
    """A project's 'time column is in ms' that this session could not apply is kept — unless
    the user has declared something else since, which is their current statement and must
    be the only record a later restore sees first."""
    session, doc, key = _live(tmp_path, rate=None, prov=Provenance.UNKNOWN, t0=0.0)
    dead = {"doc_id": persistent_key(doc), "declaration": MS.to_dict()}
    assert restore_sample_rate_declarations(session, key, [dead])
    assert capture_sample_rate_declarations(session, [dead]) == [dead], "kept while nothing else"
    session.set_sample_rate_declaration(key, HZ)
    assert capture_sample_rate_declarations(session, [dead]) == [
        {"doc_id": persistent_key(doc), "declaration": HZ.to_dict()}]


def test_a_refused_column_is_declared_by_its_unit_and_a_bare_rate_zeroes_its_raw_t0(qapp, tmp_path):
    """The microsecond column the floor refuses: its interval is kept (refused_rate_hz), so
    the unit — the one missing fact — resolves rate AND t0 (2,500,000 µs = 2.5 s). A bare
    rate cannot place a t0 read in an unknown unit, so it starts the recording at 0 rather
    than at 2,500,000 "seconds"."""
    read = TimeBaseAsRead(None, Provenance.UNKNOWN, 2_500_000.0, 1.0, refused_rate_hz=0.1)
    assert read.can_declare_unit
    us = SampleRateDeclaration("time_unit", time_unit="us")
    assert us.resolve(read) == pytest.approx((100_000.0, 2.5, 1e-6))
    assert HZ.resolve(read) == pytest.approx((12_345.0, 0.0, 1.0))
    session, doc, key = _live(tmp_path, rate=None, prov=Provenance.UNKNOWN, t0=2_500_000.0)
    assert not session.time_base_as_read(key).can_declare_unit, "nothing was refused: unknown"
    with pytest.raises(ValueError, match="no usable time column"):
        session.set_sample_rate_declaration(key, us)


def test_a_rate_declared_in_a_header_cannot_be_re_read_in_another_unit(qapp, tmp_path):
    """An ABF's 100 kHz came from its header, not from a column: 'the column is in ms'
    would multiply a correct rate by 1,000. Only a rate can be declared for it."""
    session, doc, key = _live(tmp_path, rate=100_000.0, prov=Provenance.FILE_META, t0=0.0)
    assert not session.time_base_as_read(key).can_declare_unit
    with pytest.raises(ValueError, match="declared in its header"):
        session.set_sample_rate_declaration(key, MS)
    assert session.document(key).meta.sample_rate_hz == 100_000.0
    session.set_sample_rate_declaration(key, HZ)
    assert session.document(key).meta.sample_rate_hz == 12_345.0


# -- the project ------------------------------------------------------------------------------

def test_project_round_trip_restores_the_declaration_by_persistent_key(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    session.set_sample_rate_declaration(key, MS)
    records = capture_sample_rate_declarations(session)
    assert records == [{"doc_id": persistent_key(doc), "declaration": MS.to_dict()}]
    assert "#load-" not in records[0]["doc_id"]
    state = ProjectState(documents=[DocumentRef(persistent_key(doc))],
                         workspace=AnalyticalWorkspace(extra={WORKSPACE_KEY: records}))
    saved = load_project(save_project(tmp_path / "rate.nanoscope", state))
    other, other_doc, other_id = _live(tmp_path)
    assert restore_sample_rate_declarations(
        other, other_id, saved.workspace.extra[WORKSPACE_KEY], clear_existing=True) == ()
    assert other.document(other_id).meta.sample_rate_hz == pytest.approx(100_000.0)
    assert other.sample_rate_declaration(other_id) == MS
    assert capture_sample_rate_declarations(other, saved.workspace.extra[WORKSPACE_KEY]) == records


def test_restore_reaches_a_segment_through_its_parent_and_a_segment_record_is_never_written(
        qapp, tmp_path):
    """On reopen a segment is re-derived from its restored parent, so the parent's declaration
    is the only one that exists: capture writes none for a segment, restore on a segment is a
    no-op, and a segment cut from the restored parent is on the declared clock."""
    session, doc, key = _live(tmp_path)
    session.set_sample_rate_declaration(key, MS)
    seg_id = session.create_segment(key, 1000, 3000, name="child")
    records = capture_sample_rate_declarations(session)
    assert [r["doc_id"] for r in records] == [persistent_key(doc)]
    assert restore_sample_rate_declarations(session, seg_id, records, clear_existing=True) == ()
    assert session.document(seg_id).meta.sample_rate_hz == pytest.approx(100_000.0)

    other, other_doc, other_id = _live(tmp_path)
    assert restore_sample_rate_declarations(other, other_id, records, clear_existing=True) == ()
    child = other.create_segment(other_id, 1000, 3000, name="child",
                                 segment_uid=session.document(seg_id).origin.segment_uid)
    assert other.document(child).meta.sample_rate_hz == pytest.approx(100_000.0)
    assert other.document(child).origin.t0_abs_s == pytest.approx(5e-5 + 1000 / 100_000.0)


def test_a_record_this_session_could_not_apply_is_kept_on_the_next_save(qapp, tmp_path):
    """A project's 'time column is in ms' for a recording that now opens with no column to
    re-read (here: a header-declared rate) is refused on restore — and must survive the next
    save, as the acquisition precedent keeps a record whose channel is gone."""
    session, doc, key = _live(tmp_path, rate=100_000.0, prov=Provenance.FILE_META, t0=0.0)
    record = {"doc_id": persistent_key(doc), "declaration": MS.to_dict()}
    messages = restore_sample_rate_declarations(session, key, [record])
    assert messages and "declared in its header" in messages[0]
    assert session.sample_rate_declaration(key) is None
    assert capture_sample_rate_declarations(session, [record]) == [record]


def test_missing_malformed_future_and_duplicate_records_are_kept_not_applied(qapp, tmp_path):
    session, doc, key = _live(tmp_path)
    good = {"doc_id": persistent_key(doc), "declaration": MS.to_dict()}
    absent = {"doc_id": str(tmp_path / "missing.csv"), "declaration": HZ.to_dict()}
    future = {"doc_id": persistent_key(doc), "declaration": {**MS.to_dict(), "mode": "future"}}
    extra = {**good, "future_field": 1}
    junk = [None, "not a record", 7, {"future": True}]
    second = deepcopy(good)
    second["declaration"]["time_unit"] = "us"

    messages = restore_sample_rate_declarations(session, key, junk + [future, extra])
    assert messages.count("Malformed sample-rate declaration entry was not applied.") == 3
    assert sum("unknown declaration mode" in m for m in messages) == 1, messages
    assert sum("unsupported sample-rate declaration fields" in m for m in messages) == 1
    assert session.sample_rate_declaration(key) is None
    assert capture_sample_rate_declarations(session, junk + [future, extra, absent]) == (
        junk + [future, extra, absent]), "preserved verbatim, never applied"

    said = " ".join(restore_sample_rate_declarations(session, key, [good, second]))
    assert "duplicate" in said
    assert session.sample_rate_declaration(key) == MS, "the first entry wins"
    assert capture_sample_rate_declarations(session, [good, second, absent]) == [
        absent, good], "every readable record for this recording is replaced by the live one"


@pytest.mark.parametrize("unreadable", [None, {}, "unknown", 3])
def test_an_unreadable_section_is_quarantined_never_overwritten(qapp, tmp_path, unreadable):
    extra = {WORKSPACE_KEY: unreadable, UNREADABLE_KEY: {"older": True}}
    note = prepare_sample_rate_workspace(extra)
    assert note and extra[WORKSPACE_KEY] == []
    assert extra[UNREADABLE_KEY] == {"older": True} and extra[UNREADABLE_KEY + "_1"] == unreadable
    session, doc, key = _live(tmp_path)
    with pytest.raises(ValueError, match="unreadable"):
        capture_sample_rate_declarations(session, unreadable)
    assert prepare_sample_rate_workspace({}) == () and prepare_sample_rate_workspace(extra) == ()
