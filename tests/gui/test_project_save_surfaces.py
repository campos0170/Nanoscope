"""Stage 1 project-save safety from the two standalone curation windows.

The Event Gallery and Event Inspector edit the same event table the main window owns, but
they are separate top-level windows.  A save affordance in either one is therefore intent,
never a second persistence implementation: both roads must end at MainWindow's existing
``_save_project`` / ``_save_project_as`` methods, and those methods remain the only GUI road
to :func:`nanoscope.core.project.save_project`.

These tests pin the user-visible contract that matters for a labeling session:

* Save and Save As are browsable and local to each child window;
* one key press dispatches exactly once to the main owner;
* the current path and unsaved state are seeded on first open and mirrored thereafter;
* cancelling or failing a save never clears the dirty state or changes the file on disk;
* closing a curation window does not discard the shared edits.
"""

from __future__ import annotations

# Qt bootstrap must precede every Qt import.
import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core import project as project_mod
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import load_project
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as main_window_mod
from nanoscope.gui.event_gallery import EventGalleryWindow
from nanoscope.gui.event_inspector import EventInspectorWindow
from nanoscope.gui.main_window import MainWindow


FS = 50_000.0
I0 = -1_000.0
SPANS = ((5_000, 5_450), (15_000, 15_550), (28_000, 28_700), (43_000, 43_500))
N = 50_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(731)
    data = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 7.0, N)
    for start, end in SPANS:
        data[start:end] -= 300.0
    return np.ascontiguousarray(data, dtype=np.float32)


def _normalise_control_text(text: str) -> str:
    return " ".join(text.replace("&", "").replace("…", "").split()).lower()


def _browsable_control_texts(window: QtWidgets.QWidget) -> list[str]:
    texts = [_normalise_control_text(action.text())
             for action in window.findChildren(QtGui.QAction) if action.text()]
    texts.extend(_normalise_control_text(button.text())
                 for button in window.findChildren(QtWidgets.QAbstractButton)
                 if button.text())
    return texts


def _shortcut_objects(window: QtWidgets.QWidget,
                      sequence: QtGui.QKeySequence) -> list[object]:
    found: list[object] = []
    for action in window.findChildren(QtGui.QAction):
        if any(key == sequence for key in action.shortcuts()):
            found.append(action)
    for shortcut in window.findChildren(QtGui.QShortcut):
        if shortcut.key() == sequence:
            found.append(shortcut)
    return found


def _shortcut_context(shortcut: object):
    if isinstance(shortcut, QtGui.QAction):
        return shortcut.shortcutContext()
    return shortcut.context()


def _visible_state_text(window: QtWidgets.QWidget) -> list[str]:
    texts = [window.windowTitle()]
    texts.extend(label.text() for label in window.findChildren(QtWidgets.QLabel)
                 if label.isVisible() and label.text())
    return texts


def _assert_project_state_visible(window: QtWidgets.QWidget, path: str,
                                  unsaved: bool) -> None:
    QtWidgets.QApplication.processEvents()
    texts = _visible_state_text(window)
    joined = " | ".join(texts).lower()
    if unsaved:
        assert window.windowTitle().startswith("• ") or "unsaved" in joined, (
            f"{type(window).__name__} gives no visible warning for unsaved work: {texts}")
    else:
        assert not window.windowTitle().startswith("• "), texts
    if path:
        assert Path(path).name.lower() in joined, (
            f"{type(window).__name__} does not name the project it would save: {texts}")


@pytest.fixture
def make_window(qapp, qtbot, tmp_path):
    """Build a fast, fully-wired MainWindow with a real four-event detection."""
    made: list[MainWindow] = []

    def make(name: str = "labels.dat") -> tuple[MainWindow, str, pd.DataFrame]:
        data = _signal()
        info = ChannelInfo(index=0, name="current", raw_units="pA",
                           kind=ChannelKind.CURRENT)
        meta = RecordingMeta(
            path=tmp_path / name, format_name="synthetic", n_samples=N,
            sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
            channels=(info,))
        doc = TraceDocument(meta=meta, channels=(data,))

        window = MainWindow()
        qtbot.addWidget(window)
        window.resize(1200, 800)
        window.show()
        doc_id = window._session.add_document(doc)
        window._create_default_layers(doc)
        window._on_selection(doc_id, 0)
        frame, provenance = detect_events(
            data, FS, baseline=I0, min_dwell_s=1e-4)
        assert len(frame) == len(SPANS), frame
        window._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={"min_dwell_s": 1e-4},
            calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
            payload=EventSet(frame, provenance)))
        QtWidgets.QApplication.processEvents()
        made.append(window)
        return window, doc_id, frame

    yield make

    for window in made:
        # Test cleanup is not a user quit and must never raise the unsaved-work modal.
        window._dirty = False
        for child_name in ("_event_gallery", "_event_inspector"):
            child = getattr(window, child_name, None)
            if child is not None:
                child.hide()
        for controller_name in ("_controller", "_analysis_controller", "_batch_controller"):
            controller = getattr(window, controller_name, None)
            if controller is not None:
                controller.abort()


@pytest.mark.parametrize("surface_cls", [EventGalleryWindow, EventInspectorWindow])
def test_each_curation_window_exposes_local_browsable_save_controls(
        qapp, qtbot, tmp_path, surface_cls):
    surface = surface_cls(Session())
    qtbot.addWidget(surface)
    surface.show()

    assert hasattr(surface, "saveProjectRequested")
    assert hasattr(surface, "saveProjectAsRequested")
    assert callable(getattr(surface, "set_project_save_state", None))

    texts = _browsable_control_texts(surface)
    assert any(text in {"save", "save project"} for text in texts), texts
    assert any("save" in text and " as" in text for text in texts), texts

    sequences = (
        QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save),
        QtGui.QKeySequence("Ctrl+Shift+S"),
    )
    for sequence in sequences:
        owners = _shortcut_objects(surface, sequence)
        assert len(owners) == 1, (
            f"{type(surface).__name__} must have one owner for {sequence.toString()!r}, "
            f"not {len(owners)}")
        assert _shortcut_context(owners[0]) != Qt.ApplicationShortcut, (
            "a child save shortcut must not answer for another top-level window")

    project = str(tmp_path / "curation.nanoscope")
    surface.set_project_save_state(project, True)
    _assert_project_state_visible(surface, project, True)
    surface.set_project_save_state(project, False)
    _assert_project_state_visible(surface, project, False)


def test_child_shortcuts_dispatch_once_to_the_main_windows_existing_paths(
        make_window, monkeypatch, qtbot):
    calls: list[str] = []

    def save(self):
        calls.append("save")
        return True

    def save_as(self):
        calls.append("save-as")
        return True

    # Patch before construction: both the main File actions and the lazily-created child
    # connections must resolve these same two owner methods.
    monkeypatch.setattr(MainWindow, "_save_project", save)
    monkeypatch.setattr(MainWindow, "_save_project_as", save_as)
    window, _doc_id, _frame = make_window()
    window._open_event_gallery()
    window._open_event_inspector()
    gallery = window._event_gallery
    inspector = window._event_inspector

    gallery.activateWindow()
    gallery.view.setFocus(Qt.OtherFocusReason)
    qtbot.wait(20)
    QTest.keyClick(gallery.view, Qt.Key_S, Qt.ControlModifier)
    qtbot.wait(20)
    assert calls == ["save"], calls

    calls.clear()
    inspector.activateWindow()
    inspector._next_btn.setFocus(Qt.OtherFocusReason)
    qtbot.wait(20)
    QTest.keyClick(inspector._next_btn, Qt.Key_S,
                   Qt.ControlModifier | Qt.ShiftModifier)
    qtbot.wait(20)
    assert calls == ["save-as"], calls

    # Direct intent is the wiring seam. Each emission still has exactly one receiver even
    # after both children have been opened together.
    calls.clear()
    gallery.saveProjectAsRequested.emit()
    inspector.saveProjectRequested.emit()
    assert calls == ["save-as", "save"]


def test_children_are_seeded_immediately_and_mirror_every_main_state_change(
        make_window, tmp_path, qtbot):
    window, _doc_id, _frame = make_window()
    project = str(tmp_path / "seeded.nanoscope")
    window._project_path = project
    window._dirty = True
    window._update_title()

    states: list[tuple[str, bool]] = []
    window.projectSaveStateChanged.connect(
        lambda path, unsaved: states.append((path, bool(unsaved))))
    window._open_event_gallery()
    window._open_event_inspector()
    gallery = window._event_gallery
    inspector = window._event_inspector

    # No state-change signal occurred after construction: this proves each window was seeded,
    # rather than waiting for the next edit before learning the current state.
    _assert_project_state_visible(gallery, project, True)
    _assert_project_state_visible(inspector, project, True)

    window._mark_clean()
    qtbot.wait(10)
    assert states[-1] == (project, False)
    _assert_project_state_visible(gallery, project, False)
    _assert_project_state_visible(inspector, project, False)

    window._mark_dirty()
    qtbot.wait(10)
    assert states[-1] == (project, True)
    _assert_project_state_visible(gallery, project, True)
    _assert_project_state_visible(inspector, project, True)


def test_save_dialogs_belong_to_the_child_that_started_the_flow(
        make_window, tmp_path, monkeypatch):
    """A native macOS sheet parented to MainWindow can appear behind the active child."""
    window, _doc_id, _frame = make_window()
    window._open_event_gallery()
    window._open_event_inspector()
    gallery = window._event_gallery
    inspector = window._event_inspector

    chooser_parents: list[QtWidgets.QWidget] = []

    def cancel_chooser(parent, **_kwargs):
        chooser_parents.append(parent)
        return None

    monkeypatch.setattr(main_window_mod, "ask_export_path", cancel_chooser)
    inspector.saveProjectAsRequested.emit()
    assert chooser_parents == [inspector]

    project = tmp_path / "owned-dialog.nanoscope"
    window._project_path = str(project)
    window._mark_dirty()
    monkeypatch.setattr(
        window, "_gather_project_state",
        lambda: (_ for _ in ()).throw(OSError("injected gather failure")))
    warning_parents: list[QtWidgets.QWidget] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda parent, *_a, **_k: warning_parents.append(parent)))

    gallery.saveProjectRequested.emit()

    assert warning_parents == [gallery]
    assert window._project_save_requester is None


def test_save_from_a_child_uses_the_one_core_serializer_and_persists_curation(
        make_window, tmp_path, monkeypatch, qtbot):
    window, _doc_id, frame = make_window()
    project = tmp_path / "from-inspector.nanoscope"
    window._project_path = str(project)
    window._update_title()
    window._open_event_gallery()
    window._open_event_inspector()
    gallery = window._event_gallery
    inspector = window._event_inspector

    inspector.set_accepted_current(False)
    qtbot.wait(10)
    assert window._dirty is True
    assert bool(frame.loc[frame["event_id"] == 0, "accepted"].iloc[0]) is False

    real_save = project_mod.save_project
    calls: list[Path] = []

    def counted_save(path, state, *args, **kwargs):
        calls.append(Path(path))
        return real_save(path, state, *args, **kwargs)

    monkeypatch.setattr(project_mod, "save_project", counted_save)

    # Reopening the same child must not accumulate another connection to the owner.
    inspector.close()
    window._open_event_inspector()
    window._event_inspector.saveProjectRequested.emit()
    qtbot.wait(10)

    assert calls == [project], "one child gesture must perform exactly one project write"
    assert window._dirty is False
    saved = load_project(project)
    restored = saved.results[0].events
    assert bool(restored.loc[restored["event_id"] == 0, "accepted"].iloc[0]) is False
    _assert_project_state_visible(gallery, str(project), False)
    _assert_project_state_visible(window._event_inspector, str(project), False)


def test_cancelling_the_first_child_save_as_keeps_everything_unsaved(
        make_window, tmp_path, monkeypatch, qtbot):
    window, _doc_id, frame = make_window()
    window._open_event_gallery()
    window._open_event_inspector()
    gallery = window._event_gallery
    inspector = window._event_inspector
    before = frame.copy(deep=True)
    recent_before = tuple(window._recent_projects)
    last_dir_before = window._last_project_dir

    monkeypatch.setattr(main_window_mod, "ask_export_path", lambda *a, **k: None)
    gallery.saveProjectRequested.emit()  # no current path -> the owner's Save As flow
    qtbot.wait(10)

    assert window._project_path is None
    assert window._dirty is True
    assert tuple(window._recent_projects) == recent_before
    assert window._last_project_dir == last_dir_before
    assert list(tmp_path.glob("*.nanoscope")) == []
    assert_frame_equal(frame, before)
    _assert_project_state_visible(gallery, "", True)
    _assert_project_state_visible(inspector, "", True)


def test_cancelling_child_save_as_does_not_abandon_the_current_project(
        make_window, tmp_path, monkeypatch, qtbot):
    window, _doc_id, _frame = make_window()
    current = tmp_path / "current.nanoscope"
    assert window._write_project(str(current)) is True
    before = current.read_bytes()
    window._open_event_inspector()
    inspector = window._event_inspector
    inspector.set_accepted_current(False)
    assert window._dirty is True

    monkeypatch.setattr(main_window_mod, "ask_export_path", lambda *a, **k: None)
    inspector.saveProjectAsRequested.emit()
    qtbot.wait(10)

    assert window._project_path == str(current)
    assert current.read_bytes() == before
    assert window._dirty is True
    _assert_project_state_visible(inspector, str(current), True)


@pytest.mark.parametrize("failure_at", ["gather", "serializer"])
def test_failed_child_save_preserves_the_previous_file_and_dirty_state(
        make_window, tmp_path, monkeypatch, qtbot, failure_at):
    window, _doc_id, _frame = make_window()
    project = tmp_path / "safe.nanoscope"
    assert window._write_project(str(project)) is True
    before = project.read_bytes()
    stamp_before = window._project_disk_stamp
    recent_before = tuple(window._recent_projects)
    window._open_event_gallery()
    window._open_event_inspector()
    gallery = window._event_gallery
    inspector = window._event_inspector
    inspector.set_accepted_current(False)
    assert window._dirty is True

    def fail(*_args, **_kwargs):
        raise OSError(f"injected {failure_at} failure")

    if failure_at == "gather":
        monkeypatch.setattr(window, "_gather_project_state", fail)
    else:
        monkeypatch.setattr(project_mod, "save_project", fail)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda _parent, title, text, *a, **k:
                     warnings.append((str(title), str(text)))))

    inspector.saveProjectRequested.emit()
    qtbot.wait(10)

    assert project.read_bytes() == before
    assert window._project_path == str(project)
    assert window._project_disk_stamp == stamp_before
    assert tuple(window._recent_projects) == recent_before
    assert window._dirty is True
    assert len(warnings) == 1 and "could not save" in warnings[0][1].lower(), warnings
    _assert_project_state_visible(gallery, str(project), True)
    _assert_project_state_visible(inspector, str(project), True)


def test_closing_and_reopening_children_keeps_shared_edits_and_unsaved_state(
        make_window, tmp_path, monkeypatch, qtbot):
    window, _doc_id, frame = make_window()
    project = tmp_path / "close-child.nanoscope"
    assert window._write_project(str(project)) is True
    window._open_event_gallery()
    window._open_event_inspector()
    gallery = window._event_gallery
    inspector = window._event_inspector
    inspector.set_accepted_current(False)
    qtbot.wait(10)
    assert window._dirty is True

    prompts: list[int] = []
    monkeypatch.setattr(
        window, "_confirm_close_choice", lambda: (prompts.append(1), "cancel")[1])
    gallery.close()
    inspector.close()
    qtbot.wait(10)

    assert prompts == [], "closing a view must not invoke the application's quit guard"
    assert window._dirty is True
    assert bool(frame.loc[frame["event_id"] == 0, "accepted"].iloc[0]) is False

    window._open_event_gallery()
    window._open_event_inspector()
    assert window._event_gallery is gallery
    assert window._event_inspector is inspector
    _assert_project_state_visible(gallery, str(project), True)
    _assert_project_state_visible(inspector, str(project), True)


def test_large_gallery_batch_survives_child_save_and_project_reload_exactly(
        make_window, tmp_path, qtbot):
    """75k shared rows: batch curation + labels survive the child-to-owner save road.

    The waveform stays tiny: this is a bounded stress regression for the annotation table,
    the child intent wiring and project persistence, not a detector or plotting benchmark.
    """
    from nanoscope.analysis import events as event_mod
    from nanoscope.analysis.labels import apply_label

    window, doc_id, _small = make_window("large-label-table.dat")
    n_events = 75_000
    ids = np.arange(n_events, dtype=np.int64)
    starts = (ids * 3) % (N - 3)
    stops = starts + 2
    frame = pd.DataFrame({
        "event_id": ids,
        "recording": "large-label-table",
        "pore_id": "stress-pore",
        "start_idx": starts,
        "end_idx": stops,
        "t_start_s": starts.astype(np.float64) / FS,
        "dwell_s": np.full(n_events, 2.0 / FS),
        "baseline_pA": np.full(n_events, I0),
        "depth_pA": 200.0 + (ids % 17),
        "frac_depth": 0.2 + (ids % 7) * 0.001,
        "ecd_pA_s": np.full(n_events, 0.008),
        "max_dev_pA": 210.0 + (ids % 13),
        "mean_dev_pA": 195.0 + (ids % 11),
        "rise_s": np.full(n_events, 1.0 / FS),
        "fall_s": np.full(n_events, 1.0 / FS),
        "snr": 20.0 + (ids % 9),
        "n_levels": 1 + (ids % 3),
        "filter_limited": False,
        "accepted": True,
        "flag": "",
    })[event_mod.REQUIRED].astype(
        {name: spec[0] for name, spec in event_mod.SCHEMA.items()})
    frame = event_mod.stamp_source_status(frame, "detect.schmitt")

    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="stress", doc_id=doc_id, channel=0,
        params={"fixture": "75k annotations"},
        calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(frame, {"module": "detect.schmitt", "bounded_stress": True})))
    window._open_event_gallery()
    gallery = window._event_gallery
    assert gallery.frame() is frame

    # Exercise the real whole-gallery mutation, then leave a nontrivial accepted pattern
    # and batch-label a sparse subset directly on the same shared frame.
    gallery.reject_all()
    frame.loc[(ids % 3) == 0, "accepted"] = True
    labelled_rows = np.flatnonzero((ids % 97) == 0)
    assert apply_label(frame, labelled_rows, "molecule", "apo-ferritin") \
        == len(labelled_rows)
    gallery.labelsChanged.emit()
    qtbot.wait(10)
    assert window._dirty is True

    key_columns = [
        "event_id", "accepted", "flag", event_mod.SOURCE_COL, event_mod.REVIEW_COL,
    ]
    expected = frame[key_columns].copy(deep=True).reset_index(drop=True)
    project = tmp_path / "large-child-save.nanoscope"
    window._project_path = str(project)
    window._update_title()

    # Pointer and keyboard both end at this QAction; triggering it exercises the child
    # signal and MainWindow's sole save implementation without a file chooser.
    gallery.act_save_project.trigger()
    qtbot.wait(10)
    assert project.is_file()
    assert window._dirty is False

    saved = load_project(project)
    restored_result = next(
        result for result in saved.results if result.module_id == "detect.schmitt")
    restored = restored_result.events[key_columns].reset_index(drop=True)
    assert len(restored) == n_events
    assert_frame_equal(restored, expected, check_dtype=True, check_exact=True)
