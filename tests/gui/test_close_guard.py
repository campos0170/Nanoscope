"""Unsaved-work close guard: the Save / Discard / Cancel confirmation on window close
(offscreen, pytest-qt).

MainWindow keeps a private ``_dirty`` flag off the SAME session/panel signals that drive
the workflow strip — documentOpened / documentClosed, resultsChanged, calibrationChanged,
``analysis_panel.eventAcceptanceChanged`` and (lazily) ``event_inspector.curationBatchChanged``
— plus the ``MeasurePanel.measurementsChanged`` manual-measurement edit signal. When the
window is closed with unsaved work AND at least one document open, it prompts before
tearing down; the button choice and the save both route through overridable methods
(``_confirm_close_choice`` / ``_run_save``) so the suite never faces a live modal.

``documentOpened`` and ``documentClosed`` are the conditional members of that set (B3,
C29). Opening a recording and
touching nothing creates nothing: the session holds what File ▸ Open rebuilds in a second
from a path already on the Recent list, and a project stores a document as a mere reference.
It counts as a change only once a project file exists that claims to describe this session
and no longer does — i.e. when ``_project_path`` is set. Every other signal in the set is
work the user cannot recover by reopening a file, and is tracked unconditionally.

Decision table exercised below (Save / Discard / Cancel × save outcome):

    not dirty OR nothing to lose -> accept, prompt never shown
    Save  + file written         -> accept  (save invoked once)
    Save  + cancelled/failed     -> ignore  (window stays open, work kept)
    Discard                      -> accept  (save NOT called)
    Cancel / Esc                 -> ignore  (save NOT called)

"Nothing to lose" is :meth:`MainWindow._has_work_to_lose` — a named project, open documents,
a standalone label vocabulary, OR saved manual measurements. It is deliberately broader than
"any documents open": vocabulary and measurements can exist before a recording is opened, and
removing the last recording from an existing project is itself a project change. All three are
state a project save would persist, so the guard has to fire for them.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6 import QtGui, QtWidgets

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import (
    _CLOSE_CANCEL,
    _CLOSE_DISCARD,
    _CLOSE_SAVE,
    MainWindow,
)

FS = 50_000.0
N = 20_000


# ================================================================================================
# builders
# ================================================================================================

def _fresh_window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0, dtype=np.float32) + rng.normal(0.0, 1.5, N).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    """An in-memory doc whose meta.path IS ``path`` (so its doc_id == the resolved path)."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(win: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Install a document exactly as the open pipeline would, then select its channel."""
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    return doc_id, 0


def _install_with_work(win: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Install a document AND make one real, unsaved analysis edit on it (a calibration
    change), so the window is authentically dirty.

    The close-guard cases below are about the Save/Discard/Cancel decision table, not about
    what marks dirty — and opening a recording deliberately does NOT (B3), so they need an
    actual edit to have something to lose."""
    doc_id, ch = _install(win, doc)
    win._session.set_calibration(
        doc_id, ch, ChannelCalibration(0.5, 0.0, "pA", Provenance.UNKNOWN))
    assert win._dirty is True
    return doc_id, ch


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the TextTraceReader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _close(win: MainWindow) -> QtGui.QCloseEvent:
    """Send a close event through the real ``closeEvent`` and hand it back for inspection."""
    ev = QtGui.QCloseEvent()
    win.closeEvent(ev)
    return ev


def _spy_run_save(calls: list, result: bool):
    """A stand-in ``_run_save`` that records each call and returns a fixed outcome."""
    def _run() -> bool:
        calls.append(1)
        return result
    return _run


# ================================================================================================
# dirty tracking + title marker
# ================================================================================================

def test_fresh_window_is_clean_with_plain_title(qapp, qtbot):
    w = _fresh_window(qtbot)
    assert w._dirty is False
    assert w.windowTitle() == "NanoScope"


def test_first_real_edit_marks_dirty_and_shows_title_marker(qapp, qtbot, tmp_path):
    w = _fresh_window(qtbot)
    assert not w.windowTitle().startswith("•")
    _install_with_work(w, _doc(tmp_path / "a.csv", _trace()))
    assert w._dirty is True
    assert w.windowTitle().startswith("• ")   # leading bullet marker


# ================================================================================================
# B3 regression: opening a recording is not, by itself, unsaved work
# ================================================================================================

def test_opening_a_recording_alone_is_not_unsaved_work(qapp, qtbot, tmp_path, monkeypatch):
    """REGRESSION (B3): open a recording, touch nothing, quit — and the app raised the
    Save/Discard/Cancel modal about work that does not exist. Nothing was created: the
    document is a path reference, which File ▸ Open (or Recent) rebuilds in a second."""
    w = _fresh_window(qtbot)
    _install(w, _doc(tmp_path / "a.csv", _trace()))

    assert w._dirty is False
    assert not w.windowTitle().startswith("•")     # no marker for nothing

    prompts: list = []
    monkeypatch.setattr(w, "_confirm_close_choice",
                        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])
    ev = _close(w)
    assert prompts == []          # ...and quitting asks nothing
    assert ev.isAccepted()


def test_opening_a_second_recording_into_a_saved_project_is_unsaved_work(qapp, qtbot, tmp_path):
    """The other half of B3: once a project FILE describes this session, adding a recording
    to the session means the file no longer does — that IS a change worth prompting about."""
    w = _fresh_window(qtbot)
    _install(w, _doc(tmp_path / "a.csv", _trace()))
    proj = tmp_path / "session.nanoscope"
    assert w._write_project(str(proj)) is True     # adopt a project path; clears dirty
    assert w._dirty is False

    _install(w, _doc(tmp_path / "b.csv", _trace(seed=7)))
    assert w._dirty is True
    assert w.windowTitle().startswith("• ")


def test_analysis_on_an_opened_recording_still_marks_dirty(qapp, qtbot, tmp_path):
    """B3 must not disarm the guard for real work: the very first edit on a freshly opened
    recording — with no project in sight — still arms it."""
    w = _fresh_window(qtbot)
    doc_id, ch = _install(w, _doc(tmp_path / "a.csv", _trace()))
    assert w._dirty is False
    w._session.set_calibration(
        doc_id, ch, ChannelCalibration(0.5, 0.0, "pA", Provenance.UNKNOWN))
    assert w._dirty is True
    assert w._has_work_to_lose() is True


def test_calibration_change_marks_dirty(qapp, qtbot, tmp_path):
    w = _fresh_window(qtbot)
    doc_id, ch = _install(w, _doc(tmp_path / "a.csv", _trace()))
    w._mark_clean()
    assert w._dirty is False
    w._session.set_calibration(
        doc_id, ch, ChannelCalibration(0.5, 0.0, "pA", Provenance.UNKNOWN))
    assert w._dirty is True


def test_adding_manual_measurement_marks_dirty(qapp, qtbot, tmp_path):
    w = _fresh_window(qtbot)
    doc_id, _ch = _install(w, _doc(tmp_path / "a.csv", _trace()))
    w._mark_clean()
    mp = w._analysis_panel.measure_panel
    mp.set_active(True)
    vals = w._session.x_axis(doc_id).values
    w._trace_view.set_measure_region(float(vals[2_000]), float(vals[5_000]))
    qtbot.wait(5)
    assert mp.current_measurement() is not None
    assert mp.add_current() is True
    assert w._dirty is True


def test_closing_a_recording_with_no_project_open_does_not_mark_dirty(qapp, qtbot, tmp_path):
    """C29. ``documentClosed`` is the MIRROR of ``documentOpened`` and gets the mirror rule.

    Opening two recordings and closing one is not unsaved work: what is left is a set of
    file paths the Recent list already holds. Marking it dirty grew the title's ``•`` and
    armed the quit prompt over nothing — the exact nag B3 removed, reached by the opposite
    door. (A recording carrying RESULTS is guarded separately, by a confirmation that names
    what is discarded before it goes.)"""
    w = _fresh_window(qtbot)
    _install(w, _doc(tmp_path / "a.csv", _trace()))
    id_b, _ = _install(w, _doc(tmp_path / "b.csv", _trace(seed=7)))
    w._mark_clean()
    assert w._project_path is None

    w._session.close_document(id_b)
    assert w._dirty is False
    assert w.windowTitle() == "NanoScope"


def test_closing_a_recording_with_a_project_open_does_mark_dirty(qapp, qtbot, tmp_path):
    """...and the other half of the mirror: once a file on disk claims to describe this
    session, removing a recording from it IS a change that file does not have."""
    w = _fresh_window(qtbot)
    _install(w, _doc(tmp_path / "a.csv", _trace()))
    id_b, _ = _install(w, _doc(tmp_path / "b.csv", _trace(seed=7)))
    proj = tmp_path / "session.nanoscope"
    assert w._write_project(str(proj)) is True
    assert w._dirty is False

    w._session.close_document(id_b)
    assert w._dirty is True
    assert w.windowTitle().startswith("• ")


def test_each_specified_signal_marks_dirty(qapp, qtbot, tmp_path):
    """Every UNCONDITIONAL signal in the dirty set flips the flag from clean (proves the
    wiring). ``documentOpened`` and ``documentClosed`` are the conditional members — they
    describe WHICH FILES are open, which a project stores as mere references — and have
    their own tests above and below."""
    w = _fresh_window(qtbot)
    doc_id, ch = _install(w, _doc(tmp_path / "a.csv", _trace()))
    mp = w._analysis_panel.measure_panel

    checks = [
        ("resultsChanged", lambda: w._session.resultsChanged.emit(doc_id, "detect.schmitt")),
        ("calibrationChanged", lambda: w._session.calibrationChanged.emit(doc_id, ch)),
        ("eventAcceptanceChanged",
         lambda: w._analysis_panel.eventAcceptanceChanged.emit(0, False)),
        ("measurementsChanged", lambda: mp.measurementsChanged.emit()),
    ]
    for name, emit in checks:
        w._mark_clean()
        assert w._dirty is False, name
        emit()
        assert w._dirty is True, name


def test_curation_batch_change_marks_dirty(qapp, qtbot, tmp_path):
    """The Event Inspector's Accept-all / Reject-all (curationBatchChanged) is wired lazily."""
    w = _fresh_window(qtbot)
    _install(w, _doc(tmp_path / "a.csv", _trace()))
    w._open_event_inspector()   # lazily builds the inspector AND wires curationBatchChanged
    w._mark_clean()
    assert w._dirty is False
    w._event_inspector.curationBatchChanged.emit()
    assert w._dirty is True


# ================================================================================================
# dirty clears on save + restore
# ================================================================================================

def test_successful_save_clears_dirty_and_marker(qapp, qtbot, tmp_path):
    w = _fresh_window(qtbot)
    _install_with_work(w, _doc(tmp_path / "a.csv", _trace()))
    assert w._dirty is True
    assert w.windowTitle().startswith("• ")

    proj = tmp_path / "session.nanoscope"
    assert w._write_project(str(proj)) is True
    assert proj.exists()
    assert w._dirty is False
    assert w.windowTitle() == f"NanoScope — {proj.name}"   # marker gone, name shown


def test_load_project_clears_dirty(qapp, qtbot, tmp_path):
    """Restoring a project settles to a clean session, even when the window was dirty first."""
    csv = tmp_path / "pore.csv"
    arr = _trace(seed=3)
    _write_csv(csv, arr)

    src = _fresh_window(qtbot)
    _install(src, _doc(csv, arr))
    proj = tmp_path / "session.nanoscope"
    assert src._write_project(str(proj)) is True
    src._controller.abort()

    # A fresh window, deliberately dirtied, then opens the saved project.
    w = _fresh_window(qtbot)
    _install_with_work(w, _doc(tmp_path / "scratch.csv", _trace(seed=9)))
    assert w._dirty is True
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(20)
    assert w._dirty is False
    assert not w.windowTitle().startswith("•")
    w._controller.abort()


# ================================================================================================
# closeEvent — the decision table
# ================================================================================================

def test_clean_empty_session_closes_without_prompt(qapp, qtbot, monkeypatch):
    w = _fresh_window(qtbot)
    prompts: list = []
    monkeypatch.setattr(w, "_confirm_close_choice",
                        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])
    ev = _close(w)
    assert ev.isAccepted()
    assert prompts == []   # nothing to lose -> never prompted


def test_dirty_with_nothing_to_lose_closes_without_prompt(qapp, qtbot, monkeypatch):
    """Dirty flag set but the session holds NOTHING a save would persist — no documents and
    no manual measurements — so there is genuinely nothing to lose and no prompt is raised."""
    w = _fresh_window(qtbot)
    w._dirty = True
    assert w._has_work_to_lose() is False
    prompts: list = []
    monkeypatch.setattr(w, "_confirm_close_choice",
                        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])
    ev = _close(w)
    assert ev.isAccepted()
    assert prompts == []


def test_close_save_success_accepts_and_saves_once(qapp, qtbot, tmp_path, monkeypatch):
    w = _fresh_window(qtbot)
    _install_with_work(w, _doc(tmp_path / "a.csv", _trace()))
    saves: list = []
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_SAVE)
    monkeypatch.setattr(w, "_run_save", _spy_run_save(saves, True))
    ev = _close(w)
    assert ev.isAccepted()
    assert len(saves) == 1


def test_close_save_success_real_resave_in_place(qapp, qtbot, tmp_path, monkeypatch):
    """Authentic Save path: a project with a path re-saves in place, clears dirty, closes."""
    w = _fresh_window(qtbot)
    _install(w, _doc(tmp_path / "a.csv", _trace()))
    proj = tmp_path / "session.nanoscope"
    assert w._write_project(str(proj)) is True   # adopt a project_path, clears dirty
    w._mark_dirty()
    assert w._dirty is True
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_SAVE)
    ev = _close(w)
    assert ev.isAccepted()
    assert w._dirty is False
    assert proj.exists()


def test_close_save_returns_falsy_ignores(qapp, qtbot, tmp_path, monkeypatch):
    w = _fresh_window(qtbot)
    _install_with_work(w, _doc(tmp_path / "a.csv", _trace()))
    saves: list = []
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_SAVE)
    monkeypatch.setattr(w, "_run_save", _spy_run_save(saves, False))
    ev = _close(w)
    assert not ev.isAccepted()            # ignored -> the window stays open
    assert len(saves) == 1
    assert w._dirty is True               # the work is untouched


def test_close_save_with_cancelled_save_as_keeps_open(qapp, qtbot, tmp_path, monkeypatch):
    """Authentic cancel: Save runs the real flow, the Save-As dialog is cancelled -> ignore."""
    w = _fresh_window(qtbot)
    _install_with_work(w, _doc(tmp_path / "a.csv", _trace()))
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_SAVE)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    ev = _close(w)
    assert not ev.isAccepted()
    assert w._dirty is True
    assert w._project_path is None


def test_close_discard_accepts_without_saving(qapp, qtbot, tmp_path, monkeypatch):
    w = _fresh_window(qtbot)
    _install_with_work(w, _doc(tmp_path / "a.csv", _trace()))
    saves: list = []
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_DISCARD)
    monkeypatch.setattr(w, "_run_save", _spy_run_save(saves, True))
    ev = _close(w)
    assert ev.isAccepted()
    assert saves == []                    # Discard never invokes a save


def test_close_cancel_keeps_window_open_without_saving(qapp, qtbot, tmp_path, monkeypatch):
    w = _fresh_window(qtbot)
    _install_with_work(w, _doc(tmp_path / "a.csv", _trace()))
    saves: list = []
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_CANCEL)
    monkeypatch.setattr(w, "_run_save", _spy_run_save(saves, True))
    ev = _close(w)
    assert not ev.isAccepted()
    assert saves == []
    assert w._dirty is True


def test_prompt_only_when_dirty_and_work_to_lose(qapp, qtbot, tmp_path, monkeypatch):
    """The prompt is raised only in the dirty-AND-work-to-lose quadrant; the others close."""
    w = _fresh_window(qtbot)
    prompts: list = []
    monkeypatch.setattr(w, "_confirm_close_choice",
                        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])

    # clean + nothing to lose
    assert _close(w).isAccepted()
    assert prompts == []

    # dirty + nothing to lose (no documents, no measurements)
    w._dirty = True
    assert _close(w).isAccepted()
    assert prompts == []

    # documents + clean
    _install(w, _doc(tmp_path / "a.csv", _trace()))
    w._mark_clean()
    assert _close(w).isAccepted()
    assert prompts == []

    # documents + dirty -> the one case that prompts (Cancel -> ignore)
    w._mark_dirty()
    assert not _close(w).isAccepted()
    assert len(prompts) == 1


# ================================================================================================
# C4 regression: manual measurements ARE work, even with no document open
# ================================================================================================

def _measure_only_window(qtbot, tmp_path) -> MainWindow:
    """A window whose ONLY unsaved work is a manual measurement: the document it was taken on
    is opened, measured, then closed again — exactly the "I measured a few events by hand and
    closed the recording" session that used to evaporate on quit."""
    w = _fresh_window(qtbot)
    doc_id, _ch = _install(w, _doc(tmp_path / "a.csv", _trace()))
    mp = w._analysis_panel.measure_panel
    mp.set_active(True)
    vals = w._session.x_axis(doc_id).values
    w._trace_view.set_measure_region(float(vals[2_000]), float(vals[5_000]))
    qtbot.wait(5)
    assert mp.add_current() is True
    w._session.close_document(doc_id)
    assert w._session.documents() == ()
    assert len(mp.measurements_df) == 1        # the measurement outlives its recording
    return w


def test_measurements_only_session_counts_as_work_to_lose(qapp, qtbot, tmp_path):
    """REGRESSION: the guards used "are any documents open" as a proxy for "is there work".
    Manual measurements live in the MeasurePanel, not the Session, so a measurements-only
    session answered "nothing to lose" and was thrown away without a word."""
    w = _measure_only_window(qtbot, tmp_path)
    assert w._has_work_to_lose() is True


def test_label_vocabulary_only_session_is_visible_and_guarded(qapp, qtbot, monkeypatch):
    """A palette can be authored before a recording is opened and is project data too."""
    from nanoscope.analysis.labels import LabelVocabulary

    w = _fresh_window(qtbot)
    vocab = LabelVocabulary()
    vocab.add("molecule", "apo-ferritin", "1")
    w._label_vocab = vocab
    w._mark_dirty()

    assert w._has_work_to_lose() is True
    assert w.windowTitle().startswith("• ")
    prompts: list[int] = []
    monkeypatch.setattr(
        w, "_confirm_close_choice",
        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])

    ev = _close(w)

    assert prompts == [1]
    assert not ev.isAccepted()


def test_removing_the_last_recording_from_a_named_project_is_guarded(
        qapp, qtbot, tmp_path, monkeypatch):
    """The empty session is the unsaved edit; it must not bypass the close prompt."""
    w = _fresh_window(qtbot)
    doc_id, _ = _install(w, _doc(tmp_path / "last.csv", _trace()))
    assert w._write_project(str(tmp_path / "named.nanoscope")) is True

    w._session.close_document(doc_id)

    assert w._session.documents() == ()
    assert w._dirty is True
    assert w._has_work_to_lose() is True
    assert w.windowTitle().startswith("• ")
    prompts: list[int] = []
    monkeypatch.setattr(
        w, "_confirm_close_choice",
        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])

    ev = _close(w)

    assert prompts == [1]
    assert not ev.isAccepted()


def test_measurements_only_session_shows_the_dirty_marker(qapp, qtbot, tmp_path):
    """REGRESSION: no `•` in the title meant the user had no on-screen hint the measurements
    were unsaved."""
    w = _measure_only_window(qtbot, tmp_path)
    assert w._dirty is True
    assert w.windowTitle().startswith("• ")


def test_measurements_only_session_prompts_before_closing(qapp, qtbot, tmp_path, monkeypatch):
    """REGRESSION: closing with only hand measurements went straight through the guard —
    no prompt, no chance to save, measurements gone."""
    w = _measure_only_window(qtbot, tmp_path)
    prompts: list = []
    monkeypatch.setattr(w, "_confirm_close_choice",
                        lambda: (prompts.append(1), _CLOSE_CANCEL)[1])
    ev = _close(w)
    assert len(prompts) == 1        # it asked...
    assert not ev.isAccepted()      # ...and Cancel kept the window (and the measurements) open


def test_measurements_only_session_can_still_be_saved_from_the_prompt(qapp, qtbot, tmp_path,
                                                                     monkeypatch):
    """The whole point of prompting: Save from the close guard really persists the rows."""
    w = _measure_only_window(qtbot, tmp_path)
    proj = tmp_path / "measured.nanoscope"
    monkeypatch.setattr(w, "_confirm_close_choice", lambda: _CLOSE_SAVE)
    monkeypatch.setattr(w, "_save_project", lambda: w._write_project(str(proj)))
    ev = _close(w)
    assert ev.isAccepted()
    assert proj.exists()

    from nanoscope.core.project import load_project
    assert len(load_project(str(proj)).measurements) == 1


def test_run_save_delegates_to_save_project(qapp, qtbot, monkeypatch):
    """The injectable seam forwards to the real Ctrl+S flow and reports its success bool."""
    w = _fresh_window(qtbot)
    monkeypatch.setattr(w, "_save_project", lambda: True)
    assert w._run_save() is True
    monkeypatch.setattr(w, "_save_project", lambda: False)
    assert w._run_save() is False
