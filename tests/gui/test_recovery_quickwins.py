"""Crash-recovery quick wins in the window (recoverability plan Q1-Q7, Q10).

Each test pins one failure the recoverability audit measured with synthetic data and a
simulated crash (the recovery controller aborted, the session's ``live.lock`` released
as the kernel would, a fresh window on the same root):

* s1/s2: up to 15 s idle / 60 s of curation lost before a copy existed;
* s4: an exported CSV newer than the recovery copy;
* s6 moved: "skipped, along with any of their results" for work that was kept;
* s7: a second window offered, and discarded, a live window's recovery;
* s11: unreadable copies skipped in silence;
* s12: a failed copy was a 12-second status message, retried only on the next edit;
* Discard deleted at once; scripts wrote into the real Recovery folder.

All recovery roots are pytest temporary directories; recordings are synthetic.
"""

from __future__ import annotations

# Qt bootstrap must precede any Qt import.
import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui

from nanoscope.analysis.modules import register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import DocumentRef, ProjectState, ResultRec
from nanoscope.core.recovery import RecoveryStore, _lineage_is_live
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import _CLOSE_DISCARD, _RECOVERY_DISCARD, MainWindow

FS = 50_000.0
N = 60_000
EVENTS = [(5_000, 5_300), (15_000, 15_400), (25_000, 25_250), (35_000, 35_300)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _window(qtbot, root: Path) -> MainWindow:
    window = MainWindow(recovery_root=root)
    qtbot.addWidget(window)
    return window


def _quiet(window: MainWindow) -> None:
    window._dirty = False
    if window._recovery_controller is not None:
        window._recovery_controller.abort()


def _crash(window: MainWindow) -> None:
    """Process death: nothing more is written, and the kernel drops ``live.lock``."""
    _quiet(window)
    window._recovery_store.release_liveness()


def _valuable_edit(window: MainWindow, value: str = "apo-ferritin") -> None:
    window._label_vocab.add("molecule", value, "1")
    window._mark_dirty()


def _settle(window: MainWindow, qtbot, timeout: float = 20.0) -> None:
    controller = window._recovery_controller
    deadline = time.monotonic() + timeout
    while controller.is_busy() or controller.has_pending():
        qtbot.wait(20)
        assert time.monotonic() < deadline, "the recovery writer did not finish"
    qtbot.wait(20)


def _offered(window: MainWindow, answer=None) -> list:
    offered: list = []

    def choose(candidate):
        offered.append(candidate)
        return answer

    window._choose_recovery_action = choose
    window._offer_startup_recovery()
    return offered


def _open_synthetic(window: MainWindow, qtbot, path: Path) -> str:
    rng = np.random.default_rng(0)
    trace = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for start, end in EVENTS:
        trace[start:end] -= 30.0
    trace = np.ascontiguousarray(trace, dtype=np.float32)
    pd.DataFrame({"time": np.arange(N) / FS, "current": trace.astype(np.float64)}).to_csv(
        path, index=False)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = window._session.add_document(TraceDocument(meta=meta, channels=(trace,)))
    window._create_default_layers(window._session.document(doc_id))
    window._on_selection(doc_id, 0)
    qtbot.wait(20)
    return doc_id


def _detect(window: MainWindow, qtbot):
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(60)
    return panel.current_detection().payload.df


# -- Q3: the loss window --------------------------------------------------------------------------
def test_a_copy_is_due_five_seconds_after_an_edit_and_never_later_than_twenty(qapp, qtbot,
                                                                              tmp_path):
    """s1/s2: at 15 s / 60 s a crash before the first copy lost a minute of curation."""
    window = _window(qtbot, tmp_path / "R")
    controller = window._recovery_controller
    assert (controller.idle_interval_ms, controller.max_interval_ms) == (5_000, 20_000)
    _valuable_edit(window)
    assert controller.scheduled_revision == window._session_revision
    assert 0 < controller._idle_timer.remainingTime() <= 5_000
    assert 0 < controller._max_timer.remainingTime() <= 20_000
    _quiet(window)


# -- Q2: natural flush points ---------------------------------------------------------------------
def test_export_events_flushes_recovery_so_a_crash_right_after_loses_nothing(
        qapp, qtbot, tmp_path, monkeypatch):
    """s4: the CSV held a rejection the recovery copy offered after the crash did not."""
    import nanoscope.gui.main_window as main_window

    root = tmp_path / "R"
    window = _window(qtbot, root)
    _open_synthetic(window, qtbot, tmp_path / "rec.csv")
    ids = list(_detect(window, qtbot)["event_id"])
    window._analysis_panel.set_event_accepted(int(ids[0]), False)
    window._flush_recovery_now()
    _settle(window, qtbot)
    window._analysis_panel.set_event_accepted(int(ids[2]), False)   # the last edit...
    out = tmp_path / "events.csv"
    monkeypatch.setattr(main_window, "ask_export_path", lambda *a, **k: str(out))
    window._export_events()                                        # ...then export at once
    assert len(pd.read_csv(out)) == len(ids) - 2
    assert window._recovery_controller.scheduled_revision is None
    _settle(window, qtbot)
    live_revision = window._session_revision
    _crash(window)

    relaunched = _window(qtbot, root)
    offered = _offered(relaunched)
    assert [c.revision for c in offered] == [live_revision]
    restored = relaunched._recovery_store.load(offered[0]).state.results[0].events
    assert sorted(restored.loc[~restored["accepted"], "event_id"]) == sorted(ids[i] for i in (0, 2))
    _quiet(relaunched)


def test_a_run_starting_or_the_app_going_inactive_flushes_recovery(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "R")
    controller = window._recovery_controller

    _valuable_edit(window, "before-a-run")
    window._analysis_controller.started.emit("detect.schmitt")
    assert controller.scheduled_revision is None, "captured as the run started"
    _settle(window, qtbot)
    assert window._recovery_revision == window._session_revision

    _valuable_edit(window, "before-switching-apps")
    window._on_application_state_changed(QtCore.Qt.ApplicationInactive)
    window._on_application_state_changed(QtCore.Qt.ApplicationHidden)   # coalesced
    assert controller.scheduled_revision is not None, "deferred to the next turn"
    qtbot.waitUntil(lambda: controller.scheduled_revision is None, timeout=2000)
    _settle(window, qtbot)
    assert window._recovery_revision == window._session_revision

    _valuable_edit(window, "before-another-window")
    window.changeEvent(QtCore.QEvent(QtCore.QEvent.ActivationChange))
    qtbot.waitUntil(lambda: controller.scheduled_revision is None, timeout=2000)
    _settle(window, qtbot)

    # The connections themselves, not only the slots: a batch run, and the
    # application's own state signal.
    _valuable_edit(window, "before-a-batch")
    window._batch_controller.started.emit(2)
    assert controller.scheduled_revision is None, "captured as the batch started"
    _settle(window, qtbot)
    _valuable_edit(window, "before-the-app-hides")
    QtGui.QGuiApplication.instance().applicationStateChanged.emit(
        QtCore.Qt.ApplicationInactive)
    qtbot.waitUntil(lambda: controller.scheduled_revision is None, timeout=2000)
    _quiet(window)


# -- Q1: a second window never takes a live window's recovery ------------------------------------
def test_a_second_window_does_not_offer_a_live_windows_session(qapp, qtbot, tmp_path):
    """s7: B offered A's live session; Discard tombstoned it; A's later copies vanished."""
    root = tmp_path / "R"
    first = _window(qtbot, root)
    _valuable_edit(first, "first")
    first._flush_recovery_now()
    _settle(first, qtbot)

    second = _window(qtbot, root)
    assert _offered(second, _RECOVERY_DISCARD) == []
    _valuable_edit(first, "first-again")
    first._flush_recovery_now()
    _settle(first, qtbot)
    assert first._recovery_revision == first._session_revision
    _quiet(second)

    _crash(first)
    third = _window(qtbot, root)
    offered = _offered(third)
    assert [(c.session_id, c.revision) for c in offered] == [
        (first._recovery_store.session_id, first._session_revision)]
    _quiet(third)


def test_a_write_the_store_refuses_is_reported_not_dropped(qapp, qtbot, tmp_path):
    """A lineage retired from outside used to swallow every later copy in silence."""
    root = tmp_path / "R"
    window = _window(qtbot, root)
    _valuable_edit(window, "first")
    window._flush_recovery_now()
    _settle(window, qtbot)
    assert not window._recovery_failing
    # A tool (or an older build) tombstones the lineage behind the window's back.
    tool = RecoveryStore(root)
    tool._write_retirement(window._recovery_store.session_directory, 0, True)

    _valuable_edit(window, "second")
    window._flush_recovery_now()
    _settle(window, qtbot)
    assert window._recovery_failing
    assert not window._recovery_bar.isHidden()
    text = window._recovery_bar._label.text()
    assert "retired by another NanoScope window" in text
    # Every later copy is refused as well: no retry is promised, because none would land.
    assert "tries again" not in text
    assert "off for as long as this window stays open" in text
    _quiet(window)


def test_a_closed_window_no_longer_claims_its_lineage(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path / "R")
    _valuable_edit(window)
    window._flush_recovery_now()
    _settle(window, qtbot)
    lineage = window._recovery_store.session_directory
    assert _lineage_is_live(lineage)
    window._dirty = False
    window._teardown()
    assert not _lineage_is_live(lineage)


# -- Q5: health is visible --------------------------------------------------------------------------
def test_a_failed_copy_stays_on_screen_and_is_retried_without_an_edit(
        qapp, qtbot, tmp_path, monkeypatch):
    """s12: a 12-second status message, and no retry until the next edit."""
    window = _window(qtbot, tmp_path / "R")
    controller = window._recovery_controller
    _valuable_edit(window)
    real_gather = window._gather_project_state

    def broken():
        raise RuntimeError("synthetic gather failure")

    monkeypatch.setattr(window, "_gather_project_state", broken)
    window._flush_recovery_now()
    assert not window._recovery_bar.isHidden()
    assert "synthetic gather failure" in window._recovery_bar._label.text()
    assert "No copy of this session's work exists yet" in window._recovery_bar._label.text()
    assert window._recovery_label.text() == "Recovery copy not saved"
    assert controller.retry_scheduled()
    assert controller._retry_timer.interval() == 30_000

    monkeypatch.setattr(window, "_gather_project_state", real_gather)
    controller._retry_timer.setInterval(10)
    controller._retry_timer.start()
    qtbot.waitUntil(lambda: window._recovery_revision == window._session_revision,
                    timeout=10_000)
    assert window._recovery_bar.isHidden()
    assert window._recovery_label.text().startswith("Recovery copy saved ")
    assert not controller.retry_scheduled()
    _quiet(window)


def test_a_failed_write_is_retried_without_an_edit(qapp, qtbot, tmp_path, monkeypatch):
    """s12 for the disk itself (full, read-only), not only for gathering the state."""
    window = _window(qtbot, tmp_path / "R")
    controller = window._recovery_controller
    store = window._recovery_store
    real_write = store.write

    def full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store, "write", full)
    _valuable_edit(window)
    window._flush_recovery_now()
    qtbot.waitUntil(lambda: window._recovery_failing, timeout=10_000)
    assert controller.retry_scheduled()
    assert "No space left on device" in window._recovery_bar._label.text()

    monkeypatch.setattr(store, "write", real_write)
    controller._retry_timer.setInterval(10)
    controller._retry_timer.start()
    qtbot.waitUntil(lambda: window._recovery_revision == window._session_revision,
                    timeout=10_000)
    assert window._recovery_bar.isHidden()
    _quiet(window)


def test_an_older_copy_landing_does_not_clear_a_newer_failure(qapp, qtbot, tmp_path,
                                                             monkeypatch):
    """An older write finishing after a newer capture failed cleared the warning and the
    retry, leaving the newer work uncovered in silence (the s12 state again)."""
    window = _window(qtbot, tmp_path / "R")
    controller = window._recovery_controller
    store = window._recovery_store
    real_write = store.write
    gate = threading.Event()

    def held(*args, **kwargs):
        gate.wait(10)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(store, "write", held)
    _valuable_edit(window, "older")
    window._flush_recovery_now()
    assert controller.is_busy()
    monkeypatch.setattr(window, "_gather_project_state",
                        lambda: (_ for _ in ()).throw(RuntimeError("synthetic")))
    _valuable_edit(window, "newer")
    window._flush_recovery_now()
    assert window._recovery_failing and controller.retry_scheduled()

    gate.set()
    qtbot.waitUntil(lambda: not controller.is_busy(), timeout=10_000)
    qtbot.wait(20)
    assert 0 < window._recovery_revision < window._session_revision
    assert window._recovery_failing
    assert not window._recovery_bar.isHidden()
    assert window._recovery_label.text() == "Recovery copy not saved"
    assert controller.retry_scheduled(), "the newer revision is still retried"
    _quiet(window)


def test_a_discard_at_the_open_guard_clears_the_recovery_health(qapp, qtbot, tmp_path,
                                                                monkeypatch):
    """The banner (and "saved HH:MM") outlived a Discard that let the work go."""
    window = _window(qtbot, tmp_path / "R")
    _valuable_edit(window, "kept")
    window._flush_recovery_now()
    _settle(window, qtbot)
    assert window._recovery_label.text().startswith("Recovery copy saved ")
    monkeypatch.setattr(window, "_gather_project_state",
                        lambda: (_ for _ in ()).throw(RuntimeError("synthetic")))
    _valuable_edit(window, "failing")
    window._flush_recovery_now()
    assert not window._recovery_bar.isHidden()
    monkeypatch.setattr(window, "_confirm_close_choice", lambda: _CLOSE_DISCARD)

    assert window._offer_to_save_unsaved_work()
    assert window._recovery_bar.isHidden()
    assert window._recovery_label.text() == ""
    assert not window._recovery_failing
    assert len(window._recovery_store.list_discarded()) == 1
    _quiet(window)


def test_an_explicit_save_clears_the_recovery_failure(qapp, qtbot, tmp_path, monkeypatch):
    window = _window(qtbot, tmp_path / "R")
    _valuable_edit(window)
    real_gather = window._gather_project_state
    monkeypatch.setattr(window, "_gather_project_state",
                        lambda: (_ for _ in ()).throw(RuntimeError("synthetic")))
    window._flush_recovery_now()
    assert not window._recovery_bar.isHidden()
    monkeypatch.setattr(window, "_gather_project_state", real_gather)

    assert window._write_project(str(tmp_path / "saved.nanoscope"))
    assert window._recovery_bar.isHidden()
    assert window._recovery_label.text() == ""
    assert not window._recovery_failing
    _quiet(window)


# -- Q6: the prompt names the work; Discard is reversible -----------------------------------------
def _crashed_lineage(root: Path, recording: Path, value: str = "apo-ferritin"):
    events = pd.DataFrame({
        "event_id": pd.Series([0, 1, 2], dtype="int64"),
        "start_idx": pd.Series([10, 40, 70], dtype="int64"),
        "end_idx": pd.Series([20, 50, 80], dtype="int64"),
        "accepted": pd.Series([True, False, True], dtype="bool"),
    })
    state = ProjectState(
        documents=[DocumentRef(path=str(recording))],
        results=[ResultRec(doc_id=str(recording), channel=0, module_id="detect.schmitt",
                           events=events)],
        label_vocabulary=[{"key": "molecule", "value": value, "hotkey": "1", "note": ""}])
    writer = RecoveryStore(root, session_id=str(uuid4()))
    writer.write(state, revision=6)
    return writer


def test_the_startup_prompt_names_recordings_tables_and_the_build(qapp, qtbot, tmp_path):
    root = tmp_path / "R"
    _crashed_lineage(root, tmp_path / "gone" / "pore7.abf")
    window = _window(qtbot, root)
    candidate = window._recovery_store.list_candidates()[0]
    detail = window._recovery_candidate_detail(candidate)
    assert "Recordings: pore7.abf." in detail
    assert "the threshold detector on pore7.abf: 2 of 3 event(s) accepted" in detail
    assert "Written by NanoScope " in detail
    assert "run from Python" in detail, "a source run says so; the packaged app does not"
    _quiet(window)


def test_discard_on_the_startup_prompt_keeps_the_copy_for_a_week(qapp, qtbot, tmp_path):
    root = tmp_path / "R"
    writer = _crashed_lineage(root, tmp_path / "pore7.abf")
    window = _window(qtbot, root)
    assert [c.session_id for c in _offered(window, _RECOVERY_DISCARD)] == [writer.session_id]
    store = window._recovery_store
    assert store.list_candidates() == []
    discarded = store.list_discarded()
    assert [item.session_id for item in discarded] == [writer.session_id]

    store.restore_discarded(discarded[0])
    later = _window(qtbot, root)
    assert [c.session_id for c in _offered(later)] == [writer.session_id]
    _quiet(window)
    _quiet(later)


def test_close_guard_discard_after_a_recovery_keeps_both_lineages_for_a_week(
        qapp, qtbot, tmp_path, monkeypatch):
    """main_window's close guard: the recovered lineage AND this session's newer copies."""
    root = tmp_path / "R"
    writer = _crashed_lineage(root, tmp_path / "pore7.abf")
    window = _window(qtbot, root)
    candidate = window._recovery_store.list_candidates()[0]
    with qtbot.waitSignal(window.projectRestored, timeout=15_000):
        assert window._recover_candidate(candidate)
    _valuable_edit(window, "after-recovery")
    window._flush_recovery_now()
    _settle(window, qtbot)
    own = window._recovery_store.session_id
    monkeypatch.setattr(window, "_confirm_close_choice", lambda: _CLOSE_DISCARD)

    window.closeEvent(QtGui.QCloseEvent())

    reader = RecoveryStore(root)
    assert reader.list_candidates() == []
    assert {item.session_id for item in reader.list_discarded()} == {writer.session_id, own}
    window._dirty = False


def test_unreadable_copies_are_reported_at_startup(qapp, qtbot, tmp_path, monkeypatch):
    root = tmp_path / "R"
    writer = _crashed_lineage(root, tmp_path / "pore7.abf")
    raw = writer.current_path.read_bytes()
    writer.current_path.write_bytes(raw[: len(raw) // 2])
    window = _window(qtbot, root)
    reported: list = []
    shown: list = []
    answers = iter(("show", None))
    monkeypatch.setattr(window, "_report_unreadable_recoveries",
                        lambda items: (reported.append(list(items)), next(answers))[1])
    monkeypatch.setattr(window, "_show_recovery_in_finder", lambda item: shown.append(item))
    assert _offered(window) == []
    assert [item.session_id for item in reported[0]] == [writer.session_id]
    assert shown and shown[0].directory == writer.session_directory
    text = window._unreadable_recovery_text(reported[0])
    assert text.startswith("1 recovery copy could not be read by this build")
    _quiet(window)


# -- Q7: missing recordings: the work was kept ----------------------------------------------------
def test_a_recovery_with_a_missing_recording_says_its_tables_were_kept(qapp, qtbot, tmp_path):
    """s6 moved: the banner said the tables were skipped; a Save wrote them back intact."""
    root = tmp_path / "R"
    _crashed_lineage(root, tmp_path / "moved-away" / "pore7.abf")
    window = _window(qtbot, root)
    candidate = window._recovery_store.list_candidates()[0]
    with qtbot.waitSignal(window.projectRestored, timeout=15_000):
        assert window._recover_candidate(candidate)
    text = window._info_bar._label.text()
    assert "1 source file(s) could not be found: pore7.abf." in text
    assert "Their 1 event table(s) (3 row(s), 2 accepted) were kept and will be saved " \
           "with the project" in text
    assert "skipped" not in text
    _quiet(window)


def test_a_recovery_with_an_unreadable_recording_says_its_tables_were_kept(qapp, qtbot,
                                                                          tmp_path):
    """The sibling sentence: a file found but refused by its reader is carried the same way."""
    root = tmp_path / "R"
    recording = tmp_path / "pore7.abf"
    recording.write_bytes(b"not an ABF file")
    _crashed_lineage(root, recording)
    window = _window(qtbot, root)
    candidate = window._recovery_store.list_candidates()[0]
    with qtbot.waitSignal(window.projectRestored, timeout=15_000):
        assert window._recover_candidate(candidate)
    text = window._info_bar._label.text()
    assert "1 source file(s) were found but could not be read." in text
    assert "Their 1 event table(s) (3 row(s), 2 accepted) were kept and will be saved " \
           "with the project" in text
    assert "pore7.abf — " in text, "the reader's own reason is still given"
    assert "skipped" not in text
    _quiet(window)


# -- Q10: scripts stay out of the real Recovery folder -------------------------------------------
def test_a_bare_main_window_writes_no_recovery(qapp, qtbot, tmp_path, monkeypatch):
    root = tmp_path / "default-root"
    monkeypatch.setenv("NANOSCOPE_RECOVERY_ROOT", str(root))
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._recovery_store is None and window._recovery_controller is None
    _valuable_edit(window)
    window._offer_startup_recovery()
    assert not root.exists()
    window._dirty = False

    opted_in = MainWindow(recovery_enabled=True)
    qtbot.addWidget(opted_in)
    assert opted_in._recovery_store is not None
    assert opted_in._recovery_store.root == root
    _quiet(opted_in)


def test_the_app_launch_path_is_what_turns_recovery_on(qapp, monkeypatch):
    import nanoscope.gui.app as app_module
    import nanoscope.gui.main_window as main_window
    import nanoscope.gui.worker_thread as worker_thread
    from PySide6 import QtWidgets

    made: list[dict] = []
    # main() ends by leaving through os._exit when a detached worker survives its drain;
    # an earlier test's leftover must not take this process down with it.
    monkeypatch.setattr(worker_thread, "drain_detached", lambda *_a, **_k: 0)

    class _Window:
        def __init__(self, *args, **kwargs):
            made.append(dict(kwargs))

        def show(self):
            pass

    monkeypatch.setattr(main_window, "MainWindow", _Window)
    monkeypatch.setattr(app_module, "create_app", lambda argv: qapp)
    monkeypatch.setattr(app_module, "_apply_theme", lambda app: None)
    monkeypatch.setattr(app_module, "configure_pyqtgraph", lambda: None)
    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda self: 0)
    assert app_module.main([]) == 0
    assert made == [{"recovery_enabled": True}]
