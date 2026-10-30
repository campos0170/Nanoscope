"""Save and crash recovery after "Close only this recording" leaves a segment without its parent.

MEASURED DEFECT. ``MainWindow._gather_project_state`` walks each segment up to the recording it
was cut from (``source_document``), to find the source fingerprint its results were measured
against, and it looked each parent up with ``by_key[doc.spec.parent_key]``. "Close only this
recording" keeps the segments open on purpose, so after that answer the lookup raised
``KeyError`` with the parent's path, and both callers of the gather failed on it:

* Save, Save As and the quit prompt's Save showed "Could not save the project:" and the
  parent's path in quotes, and wrote nothing, not even another recording's work. The designed
  "N segment(s) cannot be saved" question, with its "Save without them" button, was never
  reached.
* The crash-recovery capture failed the same way on every attempt. The status bar said "Could
  not update the crash-recovery copy: KeyError: ... Save the project manually", and the
  recovery copy stayed the one taken before the close.

The lookup came in with the source-evidence work (89a41fa), and the orphaned-segment tests in
test_dataloss_regressions.py, test_adv_verify_gui.py (C3) and test_p3_regressions.py (C35)
have failed since.

THE FIX. ``source_document`` answers None once a segment's parent, or an ancestor, is not open,
and the three places that use it skip a None root: the live fingerprint a fresh result is
stamped with, the recorded evidence of the live results, and that of the carried records. The
save then reaches the warning, and a result on the orphaned segment is written with its own
recorded evidence, or as explicitly unknown. A save that drops segments no longer advances
the explicit-save fence either, so the recovery copy that still holds them stays live.

Those older tests use results made in this session, which have no recorded evidence yet, so
the first skip alone turns them green. These tests pin the recovery capture, the other two
skips (a result restored from a project, a record carried by one) and a segment cut from a
segment. They drive a real MainWindow on synthetic recordings (CSVs written to tmp_path).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from nanoscope.core.project import (
    DocumentRef,
    ProjectState,
    ResultRec,
    SegmentRec,
    load_project,
    save_project,
)
from nanoscope.core.segment import SegmentDocument, derive_segment_uid
from nanoscope.gui.main_window import _CLOSE_PARENT_ONLY, MainWindow

from .test_dataloss_regressions import (
    _MODULE,
    _doc,
    _event_trace,
    _fake_events,
    _install,
    _install_fake_detection,
    _write_csv,
)

_COULD_NOT_SAVE = "Could not save the project"


# ================================================================================================
# builders
# ================================================================================================

def _window(qtbot, **kwargs) -> MainWindow:
    w = MainWindow(**kwargs)
    qtbot.addWidget(w)
    return w


def _recording_with_segment(w: MainWindow, tmp_path: Path) -> dict:
    """parent.csv open, "crop A" cut from it, and a detection on the crop."""
    csv = tmp_path / "parent.csv"
    arr = _event_trace()
    _write_csv(csv, arr)
    parent_id, _ch = _install(w, _doc(csv, arr))
    seg_id = w._session.create_segment(parent_id, 1_000, 20_000, name="crop A")
    w._create_default_layers(w._session.document(seg_id))
    _install_fake_detection(w, seg_id)
    return {"parent_id": parent_id, "seg_id": seg_id,
            "seg_uid": w._session.document(seg_id).origin.segment_uid}


def _close_only_the_parent(w: MainWindow, parent_id: str, monkeypatch) -> list[int]:
    """Close ``parent_id`` answering "Close only this recording". Returns the segment counts
    the question named."""
    asked: list[int] = []
    monkeypatch.setattr(w, "_confirm_close_with_segments",
                        lambda name, n: (asked.append(n), _CLOSE_PARENT_ONLY)[1], raising=False)
    w._close_document(parent_id)
    assert not w._session.has_document(parent_id)
    return asked


def _save(w: MainWindow, path: Path, monkeypatch, *, anyway: bool) -> tuple[bool, list]:
    """``_write_project`` with the drop warning answered "Save without them" (``anyway``) or
    "Cancel". Returns what the save returned and the (segments, results) each warning named."""
    warned: list = []
    monkeypatch.setattr(w, "_confirm_save_dropping_segments",
                        lambda n_seg, n_res: (warned.append((n_seg, n_res)), anyway)[1],
                        raising=False)
    return w._write_project(str(path)), warned


def _reopen(qtbot, proj: Path) -> MainWindow:
    """File ▸ Open Project after its dialog: the real restore, until it says it has settled."""
    w = _window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(50)
    return w


def _restored(w: MainWindow) -> tuple[str, SegmentDocument]:
    """The restored parent's doc_id and its one restored segment."""
    docs = w._session.documents()
    segments = [d for d in docs if isinstance(d, SegmentDocument)]
    parents = [d.doc_id for d in docs if not isinstance(d, SegmentDocument)]
    assert len(segments) == 1 and len(parents) == 1, docs
    return parents[0], segments[0]


# ================================================================================================
# crash recovery
# ================================================================================================

def test_crash_recovery_keeps_writing_after_close_only_this_recording(qapp, qtbot, tmp_path,
                                                                      monkeypatch):
    """REGRESSION: after "Close only this recording" every recovery capture raised ``KeyError``
    on the closed parent's path. No generation was written, and the status bar said "Could not
    update the crash-recovery copy: KeyError: ... Save the project manually", though a manual
    save failed on the same lookup. A crash then lost everything done after the close, another
    recording's work included.

    The capture the idle timer runs must write a generation without a word, and that generation
    must hold the other recording, still checkable, the orphaned segment and both detections."""
    w = _window(qtbot, recovery_root=tmp_path / "recovery", recovery_enabled=True)
    ctx = _recording_with_segment(w, tmp_path)
    other_csv = tmp_path / "other.csv"
    other_arr = _event_trace(1)
    _write_csv(other_csv, other_arr)
    other_id, _ch = _install(w, _doc(other_csv, other_arr))
    _install_fake_detection(w, other_id)
    _close_only_the_parent(w, ctx["parent_id"], monkeypatch)

    controller = w._recovery_controller
    assert controller is not None
    saved: list[int] = []
    failed: list[str] = []
    controller.saved.connect(lambda revision, candidate: saved.append(revision))
    controller.failed.connect(lambda revision, message: failed.append(message))
    w._label_vocab.add("molecule", "apo-ferritin", "1")    # an edit made after the close
    w._mark_dirty()
    controller.flush()                                     # what the idle timer does
    qtbot.waitUntil(lambda: bool(saved or failed), timeout=15_000)
    status = w.statusBar().currentMessage()
    store = w._recovery_store
    mine = [c for c in store.list_candidates() if c.session_id == store.session_id]
    w._dirty = False                                       # no quit prompt at teardown
    controller.abort()

    assert failed == [], failed
    assert "Could not update the crash-recovery copy" not in status, status
    assert saved and len(mine) == 1 and mine[0].revision == saved[-1], (saved, mine)
    state = store.load(mine[0]).state
    assert [Path(d.path).name for d in state.documents] == ["other.csv"]
    assert state.documents[0].source is not None
    assert [s.segment_uid for s in state.segments] == [ctx["seg_uid"]]
    assert sorted(r.doc_id for r in state.results) == sorted(
        [ctx["seg_uid"], state.documents[0].path])


def test_a_save_without_its_segment_leaves_the_recovery_copy_that_holds_it(qapp, qtbot, tmp_path,
                                                                           monkeypatch):
    """REGRESSION, reachable once the save above works: "Save without them" writes a file that
    does not hold the orphaned segment and keeps the session dirty, but it also advanced the
    explicit-save fence to the revision it wrote. The recovery capture of that same revision,
    which the idle timer runs after the save, was then retired the moment it landed
    (``_on_recovery_saved``), so the segment and its detection were in neither the project file
    nor the recovery store, and a crash lost them (measured: saved [2], candidates []).
    docs/STAGE_2_RECOVERY.md: a save that omits live work keeps it recoverable and visibly
    dirty.

    After "Save without them", a capture must leave a generation that holds the segment."""
    w = _window(qtbot, recovery_root=tmp_path / "recovery", recovery_enabled=True)
    ctx = _recording_with_segment(w, tmp_path)
    _close_only_the_parent(w, ctx["parent_id"], monkeypatch)
    ok, warned = _save(w, tmp_path / "kept.nanoscope", monkeypatch, anyway=True)
    assert ok is True and warned == [(1, 1)]

    controller = w._recovery_controller
    assert controller is not None
    saved: list[int] = []
    failed: list[str] = []
    controller.saved.connect(lambda revision, candidate: saved.append(revision))
    controller.failed.connect(lambda revision, message: failed.append(message))
    controller.flush()                                     # what the idle timer does
    qtbot.waitUntil(lambda: bool(saved or failed), timeout=15_000)
    qtbot.wait(50)                                         # past the landing's retirement fence
    store = w._recovery_store
    mine = [c for c in store.list_candidates() if c.session_id == store.session_id]
    w._dirty = False                                       # no quit prompt at teardown
    controller.abort()

    assert failed == [], failed
    assert len(mine) == 1, (saved, mine)
    state = store.load(mine[0]).state
    assert [s.segment_uid for s in state.segments] == [ctx["seg_uid"]]
    assert [r.doc_id for r in state.results] == [ctx["seg_uid"]]


# ================================================================================================
# the save reaches "N segment(s) cannot be saved"
# ================================================================================================

def test_a_restored_result_saves_without_its_segment_once_the_parent_closes(
        qapp, qtbot, tmp_path, monkeypatch, blocked_modals):
    """REGRESSION: a detection restored from a project carries the source fingerprint it was
    measured against, so the save checks that against the segment's root recording. With the
    parent closed there is no root: the save failed with ``KeyError`` on the parent's path, and
    a fix that only made the lookup safe failed here instead with "Could not save the project:
    'NoneType' object has no attribute 'doc_id'".

    "Save without them" must write the file, and the restored detection must keep its own
    fingerprint in it rather than lose it with the parent."""
    w = _window(qtbot)
    _recording_with_segment(w, tmp_path)
    first = tmp_path / "first.nanoscope"
    assert w._write_project(str(first)) is True
    measured_on = load_project(str(first)).documents[0].source
    assert measured_on is not None

    w2 = _reopen(qtbot, first)
    parent_id, segment = _restored(w2)
    restored = w2._session.results.get(segment.doc_id, 0, _MODULE)
    assert restored.source_fingerprint_recorded and restored.source_fingerprint == measured_on
    _close_only_the_parent(w2, parent_id, monkeypatch)

    second = tmp_path / "second.nanoscope"
    ok, warned = _save(w2, second, monkeypatch, anyway=True)
    failures = [m for m in blocked_modals if _COULD_NOT_SAVE in m.text]
    assert failures == [], failures
    assert ok is True and warned == [(1, 1)], (ok, warned)
    written = load_project(str(second))
    assert written.documents == []
    assert [r.source_fingerprint for r in written.results] == [measured_on]
    assert w2._dirty is True                           # work was left out, so still unsaved


def test_a_carried_record_saves_without_its_segment_once_the_parent_closes(
        qapp, qtbot, tmp_path, monkeypatch, blocked_modals):
    """REGRESSION: a record the restore could not re-attach (here a detection on a channel the
    re-exported recording no longer has) is written back verbatim, and the save first finds
    the open document it belongs to, to check its evidence against that document's root. On a
    segment whose parent was then closed that failed: with ``KeyError`` on the parent's path,
    and, with only the lookup made safe, with "'NoneType' object has no attribute 'doc_id'".

    "Save without them" must write the file with the carried record still in it."""
    csv = tmp_path / "parent.csv"
    _write_csv(csv, _event_trace())
    key = str(csv.resolve())
    uid = derive_segment_uid(key, 1_000, 20_000)
    proj = tmp_path / "carried.nanoscope"
    save_project(proj, ProjectState(
        documents=[DocumentRef(key)],
        segments=[SegmentRec(segment_uid=uid, parent_key=key, i0=1_000, i1=20_000,
                             name="crop A")],
        results=[ResultRec(uid, 1, _MODULE, _fake_events().df, provenance={},
                           channel_name="current_b", channel_kind="current")]))

    w = _reopen(qtbot, proj)
    parent_id, segment = _restored(w)
    assert segment.origin.segment_uid == uid
    assert [r.channel_name for r in (w._carried_over or {}).get("results", [])] == ["current_b"]
    _close_only_the_parent(w, parent_id, monkeypatch)

    out = tmp_path / "out.nanoscope"
    ok, warned = _save(w, out, monkeypatch, anyway=True)
    failures = [m for m in blocked_modals if _COULD_NOT_SAVE in m.text]
    assert failures == [], failures
    assert ok is True and warned == [(1, 1)], (ok, warned)
    written = load_project(str(out))
    assert [(r.doc_id, r.channel_name) for r in written.results] == [(uid, "current_b")]


def test_a_segment_cut_from_a_segment_is_counted_when_the_root_closes(
        qapp, qtbot, tmp_path, monkeypatch, blocked_modals):
    """REGRESSION: "Close only this recording" on the root of a segment cut from a segment
    orphans both, and the walk from the inner segment passes the outer one, still open, before
    it reaches the closed root. The save failed there with ``KeyError`` on the root's path.

    It must reach the warning and count both segments and both detections; declining it writes
    nothing and keeps the work flagged unsaved."""
    w = _window(qtbot)
    ctx = _recording_with_segment(w, tmp_path)
    inner = w._session.create_segment(ctx["seg_id"], 100, 5_000, name="crop A.1")
    _install_fake_detection(w, inner)
    assert _close_only_the_parent(w, ctx["parent_id"], monkeypatch) == [2]
    assert w._session.has_document(ctx["seg_id"]) and w._session.has_document(inner)

    proj = tmp_path / "nested.nanoscope"
    ok, warned = _save(w, proj, monkeypatch, anyway=False)
    failures = [m for m in blocked_modals if _COULD_NOT_SAVE in m.text]
    assert failures == [], failures
    assert warned == [(2, 2)], warned
    assert ok is False and not proj.exists()
    assert w._dirty is True
