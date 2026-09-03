"""Stage 2 crash-recovery integration contract.

These tests deliberately cross the boundary between ``MainWindow`` and the pure recovery
store.  The controller's timer/coalescing machinery and the store's archive parser have
their own focused tests; this file pins the promises a user can observe:

* persistent edits advance one revision clock and schedule that exact revision;
* the GUI hands an immutable ``ProjectState`` to the writer and never snapshots clean work;
* corrupt-newest recovery falls back to the preceding atomic generation;
* Recover / Discard / Show in Finder have distinct, non-ambiguous effects;
* recovered work is visibly unsaved on every editing surface;
* explicit-save retirement is fenced at the revision actually written; and
* sessions are isolated and clean shutdown stops the writer without erasing a checkpoint.

All recovery roots are pytest temporary directories.  The suite must never discover,
offer, retire, or create a user's real recovery files.
"""

from __future__ import annotations

# Qt bootstrap must precede any Qt import.
import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from uuid import uuid4

import pandas as pd
from PySide6 import QtGui

from nanoscope.analysis.labels import LabelVocabulary
from nanoscope.core.project import ProjectState
from nanoscope.core.recovery import RecoveryStore
from nanoscope.gui.main_window import (
    _RECOVERY_DISCARD,
    _RECOVERY_RECOVER,
    _RECOVERY_SHOW,
    MainWindow,
)


def _session_id() -> str:
    return str(uuid4())


def _state(value: str = "apo-ferritin") -> ProjectState:
    """A small but genuinely valuable session that restores synchronously (no raw file)."""
    return ProjectState(label_vocabulary=[{
        "key": "molecule", "value": value, "hotkey": "1", "note": "",
    }])


def _window(qtbot, root: Path, *, enabled: bool = True) -> MainWindow:
    window = MainWindow(recovery_root=root, recovery_enabled=enabled)
    qtbot.addWidget(window)
    return window


class _ControllerSpy:
    """Small replacement for revision wiring and teardown tests."""

    def __init__(self) -> None:
        self.dirty: list[int] = []
        self.aborts = 0

    def note_dirty(self, revision: int) -> None:
        self.dirty.append(int(revision))

    def abort(self) -> None:
        self.aborts += 1


def _replace_controller(window: MainWindow) -> _ControllerSpy:
    old = window._recovery_controller
    if old is not None:
        old.abort()
    spy = _ControllerSpy()
    window._recovery_controller = spy
    return spy


def _valuable_edit(window: MainWindow, value: str = "apo-ferritin") -> None:
    window._label_vocab.add("molecule", value, "1")
    window._mark_dirty()


def _quiet(window: MainWindow) -> None:
    """Keep pytest-qt cleanup away from the unrelated unsaved-work modal."""
    window._dirty = False
    controller = window._recovery_controller
    if controller is not None:
        controller.abort()


def test_persistent_edits_advance_and_schedule_the_same_revision(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "recovery")
    controller = _replace_controller(window)

    _valuable_edit(window)
    window._mark_dirty()

    assert window._session_revision == 2
    assert controller.dirty == [1, 2]
    assert window._dirty is True
    _quiet(window)


def test_snapshot_is_detached_from_live_frames_and_clean_work_is_not_snapshotted(
        qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "recovery")
    _replace_controller(window)
    measurements = pd.DataFrame([{
        "measurement_id": 7,
        "recording": "before",
        "start_idx": 100,
        "end_idx": 200,
    }])
    window._analysis_panel.measure_panel.set_measurements_df(measurements)
    window._mark_dirty()
    revision = window._session_revision

    snapshot = window._capture_recovery_snapshot(revision)
    assert snapshot is not None
    # The safety checkpoint may discover an unannounced in-place edit and advance the
    # clock while gathering.  The snapshot must name what it actually captured, never the
    # older timer request.
    assert snapshot.revision == window._session_revision
    assert snapshot.revision >= revision
    assert snapshot.state.measurements.iloc[0]["recording"] == "before"

    # ``_gather_project_state`` normally references this live DataFrame.  The recovery
    # boundary must clone it before a worker can observe a later in-place edit.
    live = window._analysis_panel.measure_panel.measurements_df
    live.loc[live.index[0], "recording"] = "after"
    assert snapshot.state.measurements.iloc[0]["recording"] == "before"

    window._mark_clean()
    assert window._capture_recovery_snapshot(revision) is None
    _quiet(window)


def test_corrupt_newest_generation_falls_back_to_previous(qapp, tmp_path):
    root = tmp_path / "recovery"
    store = RecoveryStore(root=root, session_id=_session_id())
    store.write(_state("first"), revision=1)
    newest = store.write(_state("second"), revision=2)

    newest.envelope_path.write_bytes(b"interrupted replacement")

    candidates = store.list_candidates()
    mine = [c for c in candidates if c.session_id == store.session_id]
    assert len(mine) == 1
    assert mine[0].revision == 1
    restored = store.load(mine[0])
    assert restored.state.label_vocabulary[0]["value"] == "first"


def test_two_recovery_sessions_are_isolated_and_discard_is_scoped(qapp, tmp_path):
    root = tmp_path / "recovery"
    one = RecoveryStore(root=root, session_id=_session_id())
    two = RecoveryStore(root=root, session_id=_session_id())
    one.write(_state("one"), revision=3)
    two.write(_state("two"), revision=8)

    candidates = one.list_candidates()
    by_session = {candidate.session_id: candidate for candidate in candidates}
    assert {one.session_id, two.session_id} <= set(by_session)

    one.discard(by_session[one.session_id])

    left = one.list_candidates()
    assert all(candidate.session_id != one.session_id for candidate in left)
    assert any(candidate.session_id == two.session_id and candidate.revision == 8
               for candidate in left)


def test_recover_opens_normal_payload_but_stays_visibly_unsaved_everywhere(
        qapp, qtbot, tmp_path):
    root = tmp_path / "recovery"
    source = str(tmp_path / "named-before-crash.nanoscope")
    writer = RecoveryStore(root=root, session_id=_session_id())
    writer.write(_state(), revision=4, source_project_path=source)
    candidate = next(c for c in writer.list_candidates()
                     if c.session_id == writer.session_id)
    window = _window(qtbot, root)

    assert window._recover_candidate(candidate) is True

    assert window._label_vocab.to_json() == _state().label_vocabulary
    assert window._project_path is None, "a private checkpoint must not impersonate a project"
    assert window._recovery_suggested_project_path == source
    assert window._dirty is True
    assert window._visible_project_dirty() is True
    assert window.windowTitle().startswith("• NanoScope")

    window._open_event_gallery()
    window._open_event_inspector()
    assert "Unsaved changes" in window._event_gallery._project_save_label.text()
    assert "Unsaved changes" in window._event_inspector._project_save_label.text()
    assert "no project file yet" in window._event_gallery._project_save_label.text()
    assert "no project file yet" in window._event_inspector._project_save_label.text()
    _quiet(window)


def test_show_in_finder_is_non_destructive_and_returns_to_the_same_decision(
        qapp, qtbot, tmp_path, monkeypatch):
    root = tmp_path / "recovery"
    writer = RecoveryStore(root=root, session_id=_session_id())
    writer.write(_state(), revision=2)
    window = _window(qtbot, root)
    shown = []
    choices = iter((_RECOVERY_SHOW, _RECOVERY_DISCARD))
    monkeypatch.setattr(window, "_choose_recovery_action", lambda candidate: next(choices))
    monkeypatch.setattr(window, "_show_recovery_in_finder", lambda candidate: shown.append(candidate))

    window._offer_startup_recovery()

    assert len(shown) == 1
    assert shown[0].session_id == writer.session_id
    assert all(candidate.session_id != writer.session_id
               for candidate in window._recovery_store.list_candidates())
    _quiet(window)


def test_closing_the_startup_choice_keeps_the_candidate(qapp, qtbot, tmp_path, monkeypatch):
    root = tmp_path / "recovery"
    writer = RecoveryStore(root=root, session_id=_session_id())
    writer.write(_state(), revision=5)
    window = _window(qtbot, root)
    monkeypatch.setattr(window, "_choose_recovery_action", lambda candidate: None)

    window._offer_startup_recovery()

    assert any(candidate.session_id == writer.session_id and candidate.revision == 5
               for candidate in window._recovery_store.list_candidates())
    _quiet(window)


def test_discarding_one_startup_candidate_continues_to_the_next_session(
        qapp, qtbot, tmp_path, monkeypatch):
    root = tmp_path / "recovery"
    one = RecoveryStore(root=root, session_id=_session_id())
    two = RecoveryStore(root=root, session_id=_session_id())
    one.write(_state("one"), revision=1, timestamp=100.0)
    two.write(_state("two"), revision=2, timestamp=200.0)
    window = _window(qtbot, root)
    offered: list[str] = []

    def discard(candidate):
        offered.append(candidate.session_id)
        return _RECOVERY_DISCARD

    monkeypatch.setattr(window, "_choose_recovery_action", discard)
    window._offer_startup_recovery()

    assert offered == [two.session_id, one.session_id]
    assert window._recovery_store.list_candidates() == []
    _quiet(window)


def test_startup_recovery_offer_runs_only_once_per_window(qapp, qtbot, tmp_path, monkeypatch):
    root = tmp_path / "recovery"
    writer = RecoveryStore(root=root, session_id=_session_id())
    writer.write(_state(), revision=1)
    window = _window(qtbot, root)
    offered = []
    monkeypatch.setattr(
        window, "_choose_recovery_action",
        lambda candidate: (offered.append(candidate), None)[1])

    for _ in range(2):
        event = QtGui.QShowEvent()
        window.showEvent(event)
        qapp.processEvents()

    assert len(offered) == 1
    _quiet(window)


def test_successful_explicit_save_retires_its_real_recovery_generation(
        qapp, qtbot, tmp_path):
    root = tmp_path / "recovery"
    window = _window(qtbot, root)
    controller = _replace_controller(window)
    _valuable_edit(window)
    revision = window._session_revision
    snapshot = window._capture_recovery_snapshot(revision)
    assert snapshot is not None
    candidate = window._persist_recovery_snapshot(snapshot)
    window._on_recovery_saved(snapshot.revision, candidate)
    assert any(c.session_id == window._recovery_store.session_id
               for c in window._recovery_store.list_candidates())

    assert window._write_project(str(tmp_path / "saved.nanoscope")) is True

    assert window._dirty is False
    assert window._explicit_saved_revision == snapshot.revision
    assert all(c.session_id != window._recovery_store.session_id
               for c in window._recovery_store.list_candidates())
    assert controller.dirty
    _quiet(window)


def test_successful_explicit_save_retires_only_the_revision_it_committed(
        qapp, qtbot, tmp_path, monkeypatch):
    import nanoscope.core.project as project_module

    window = _window(qtbot, tmp_path / "recovery")
    controller = _replace_controller(window)
    _valuable_edit(window)
    written_revision = window._session_revision
    retired: list[int] = []
    monkeypatch.setattr(window, "_retire_recovery_through",
                        lambda revision: retired.append(int(revision)))
    original_save = project_module.save_project

    def save_then_edit(path, state, notes=None):
        result = original_save(path, state, notes)
        # A nested event loop or a slow filesystem can let this happen while the explicit
        # write is in flight.  This edit is not represented by ``state`` above.
        window._label_vocab.add("condition", "late", "2")
        window._mark_dirty()
        return result

    monkeypatch.setattr(project_module, "save_project", save_then_edit)
    assert window._write_project(str(tmp_path / "saved.nanoscope")) is True

    assert retired == [written_revision]
    assert window._explicit_saved_revision == written_revision
    assert window._session_revision == written_revision + 1
    assert controller.dirty[-1] == written_revision + 1
    assert window._dirty is True
    assert window.windowTitle().startswith("• ")
    _quiet(window)


def test_failed_explicit_save_retires_nothing(qapp, qtbot, tmp_path, monkeypatch):
    import nanoscope.core.project as project_module

    window = _window(qtbot, tmp_path / "recovery")
    _replace_controller(window)
    _valuable_edit(window)
    snapshot = window._capture_recovery_snapshot(window._session_revision)
    assert snapshot is not None
    window._persist_recovery_snapshot(snapshot)
    retired: list[int] = []
    warnings: list[str] = []
    monkeypatch.setattr(window, "_retire_recovery_through",
                        lambda revision: retired.append(int(revision)))
    monkeypatch.setattr(project_module, "save_project",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(
        "nanoscope.gui.main_window.QtWidgets.QMessageBox.warning",
        lambda _parent, _title, text: warnings.append(str(text)))

    assert window._write_project(str(tmp_path / "failed.nanoscope")) is False
    assert retired == []
    assert window._explicit_saved_revision == 0
    assert window._dirty is True
    assert any("disk full" in text for text in warnings)
    assert any(c.session_id == window._recovery_store.session_id
               and c.revision == snapshot.revision
               for c in window._recovery_store.list_candidates())
    _quiet(window)


def test_shutdown_stops_recovery_writer_without_erasing_unsaved_checkpoint(
        qapp, qtbot, tmp_path):
    root = tmp_path / "recovery"
    window = _window(qtbot, root)
    old = window._recovery_controller
    if old is not None:
        old.abort()
    writer = window._recovery_store
    writer.write(_state(), revision=6)
    controller = _ControllerSpy()
    window._recovery_controller = controller

    window._teardown()

    assert controller.aborts >= 1
    assert any(candidate.session_id == writer.session_id and candidate.revision == 6
               for candidate in writer.list_candidates())
    window._dirty = False


def test_discarding_current_work_fences_old_revision_but_future_autosaves_still_work(
        qapp, qtbot, tmp_path):
    """Discard ends one edit generation, not the entire lifetime of this app process."""
    root = tmp_path / "recovery"
    window = _window(qtbot, root)
    _valuable_edit(window)
    first = window._capture_recovery_snapshot(window._session_revision)
    assert first is not None
    assert window._persist_recovery_snapshot(first) is not None

    window._discard_recovery()
    assert all(c.session_id != window._recovery_store.session_id
               for c in window._recovery_store.list_candidates())

    _valuable_edit(window, "after-discard")
    second = window._capture_recovery_snapshot(window._session_revision)
    assert second is not None and second.revision > first.revision
    committed = window._persist_recovery_snapshot(second)

    assert committed is not None
    assert any(c.session_id == window._recovery_store.session_id
               and c.revision == second.revision
               for c in window._recovery_store.list_candidates())
    _quiet(window)


def test_a_late_old_session_write_is_retired_without_blocking_the_newer_revision(
        qapp, qtbot, tmp_path):
    """A write already inside compression can finish after Discard/reset_session.

    ``RecoveryController.staleSaved`` is the completion handshake for that race.  Without
    a MainWindow receiver, the candidate reappears on the next launch after the user has
    explicitly discarded it.
    """
    root = tmp_path / "recovery"
    window = _window(qtbot, root)
    controller = window._recovery_controller
    assert controller is not None
    _valuable_edit(window)
    first = window._capture_recovery_snapshot(window._session_revision)
    assert first is not None
    candidate = window._persist_recovery_snapshot(first)
    assert candidate is not None
    old_generation = controller.generation
    window._discard_recovery()

    # This is the completion notification an already-running old-generation worker
    # delivers after the discard fence.  Its metadata is enough to reapply the fence even
    # though the old generation file has already been removed.
    controller.staleSaved.emit(old_generation, first.revision, candidate)
    qapp.processEvents()
    assert all(c.session_id != candidate.session_id
               for c in window._recovery_store.list_candidates())

    _valuable_edit(window, "new-generation")
    second = window._capture_recovery_snapshot(window._session_revision)
    assert second is not None and second.revision > first.revision
    committed = window._persist_recovery_snapshot(second)
    assert committed is not None
    assert any(c.session_id == candidate.session_id and c.revision == second.revision
               for c in window._recovery_store.list_candidates())
    _quiet(window)
