"""EXP3-01 — a cancelled project restore says what it did not load, and never claims to be
a finished restore.

The chaos2-001 fix made Cancel abandon the whole open QUEUE, which is right for a
File ▸ Open of several recordings. A project restore drives the SAME queue
(``_restore_documents`` → ``open_paths`` → ``_open_queue``), and nothing there tested the
second consumer: ``tests/gui/test_s2_chaos_cancel_queue.py`` drives only ``open_paths``
with ``_restoring`` False, and the three restore suites contain no occurrence of "cancel"
at all.

Measured on the tree as found — one press of the status-bar Cancel while the second of
three recordings was loading:

    documents restored : 1 of 3
    info bar           : ''                      (the missing-source note is appended only by
                                                  ``_on_failed``, never by a cancel)
    status bar         : 'Project restored.'     (overwriting "Cancelled. 1 more queued
                                                  file(s) were not opened." inside the same
                                                  slot, so it was never painted)
    window title       : 'NanoScope — three.nanoscope'   (no •: ``_mark_clean`` had run)

An independent run cancelled on the FIRST recording and got an entirely EMPTY session
reporting "Project restored." with the project path adopted and a clean title.

``docs/USER_GUIDE.md`` §11a promises *"The project still opens; nothing is thrown away"*
and ``_finalize_restore`` pairs every other restore shortfall — a missing source, an
unrebuilt segment, a bad calibration, a moved channel — with a named banner. A cancel is
the eighth shortfall and is now named beside them. Nothing on disk was ever lost
(``_capture_unrestorable`` carries the unreached refs), which is exactly why the session
must not look complete: the user's only cue to reopen is the app saying so.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.project import load_project
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0


def _write_csv(path: Path, n: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": arr}).to_csv(path, index=False)
    return path


def _new_window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _settle(w, qtbot) -> None:
    qtbot.waitUntil(lambda: w._active_open is None and not w._open_queue, timeout=30_000)
    qtbot.wait(120)


@pytest.fixture
def project(qapp, qtbot, tmp_path):
    """A three-recording project on disk, written by the real save path."""
    recs = [_write_csv(tmp_path / f"rec{i}.csv", 300_000 if i else 40_000, seed=i)
            for i in range(3)]
    w = _new_window(qtbot)
    w.open_paths([str(p) for p in recs])
    _settle(w, qtbot)
    assert len(w._session.documents()) == 3
    proj = tmp_path / "three.nanoscope"
    assert w._write_project(str(proj)) is True
    w._controller.abort()
    w._analysis_controller.abort()
    w.close()
    yield proj, recs


def _cancel_mid_restore(w, qtbot, proj: Path) -> int:
    """Open the project and press Cancel while a recording is still loading.

    Returns how many recordings the restore never reached. Uses the real
    ``_cancel_load`` slot, i.e. exactly what the status-bar button is wired to.
    """
    w._open_project(str(proj))
    qtbot.waitUntil(lambda: w._restoring and w._active_open is not None, timeout=20_000)
    expected = len(w._open_queue) + 1
    w._cancel_load()
    _settle(w, qtbot)
    return expected


# ================================================================================================
def test_a_cancelled_restore_does_not_announce_itself_as_restored(project, qtbot):
    proj, recs = project
    w = _new_window(qtbot)
    try:
        missed = _cancel_mid_restore(w, qtbot, proj)
        assert 0 < len(w._session.documents()) + missed
        assert len(w._session.documents()) < 3, (
            "premise: the cancel really did truncate the restore")

        status = w.statusBar().currentMessage()
        assert status != "Project restored.", status
        assert "cancel" in status.lower(), status
        assert str(missed) in status, status
    finally:
        w._controller.abort()
        w._analysis_controller.abort()


def test_it_names_the_recordings_it_did_not_load(project, qtbot):
    """The banner, beside the other seven shortfalls. An empty info bar was the whole
    problem: nothing on screen contradicted "Project restored."."""
    proj, recs = project
    w = _new_window(qtbot)
    try:
        missed = _cancel_mid_restore(w, qtbot, proj)
        assert not w._info_bar.isHidden(), "the info bar stayed hidden after a cancel"
        blob = w._info_bar._label.text()
        assert "cancel" in blob.lower(), blob
        loaded = {Path(d.meta.path).name for d in w._session.documents()}
        for rec in recs:
            if rec.name not in loaded:
                assert rec.name in blob, f"{rec.name} was not named: {blob}"
        assert str(missed) in blob, blob
    finally:
        w._controller.abort()
        w._analysis_controller.abort()


def test_a_cancelled_restore_is_not_marked_clean(project, qtbot):
    """``_mark_clean`` strips the • from the title and disarms the quit guard. The session
    holds less than the file does, so the two are not in agreement and the marker says so."""
    proj, _recs = project
    w = _new_window(qtbot)
    try:
        _cancel_mid_restore(w, qtbot, proj)
        assert w._dirty is True, "a truncated session was presented as a saved project"
        assert "•" in w.windowTitle(), w.windowTitle()
    finally:
        w._controller.abort()
        w._analysis_controller.abort()


def test_the_unloaded_recordings_are_still_written_back_by_a_later_save(project, qtbot,
                                                                       tmp_path):
    """Nothing is destroyed on disk — the carry already worked — and the save's own note may
    not call them "could not be opened": they were never attempted."""
    proj, _recs = project
    w = _new_window(qtbot)
    try:
        _cancel_mid_restore(w, qtbot, proj)
        again = tmp_path / "three_again.nanoscope"
        assert w._write_project(str(again)) is True
        assert len(load_project(again).documents) == 3
        msg = w.statusBar().currentMessage()
        assert "could not be opened" not in msg, msg
        assert "kept in the project, not dropped" in msg, msg
    finally:
        w._controller.abort()
        w._analysis_controller.abort()


def test_an_uncancelled_restore_still_says_project_restored_and_is_clean(project, qtbot):
    """THE CONTROL. The ordinary restore keeps its one-line confirmation and its clean
    title — a fix that made every restore look doubtful would be its own defect."""
    proj, _recs = project
    w = _new_window(qtbot)
    try:
        w._open_project(str(proj))
        _settle(w, qtbot)
        assert len(w._session.documents()) == 3
        assert w.statusBar().currentMessage() == "Project restored."
        assert w._dirty is False
        assert "•" not in w.windowTitle(), w.windowTitle()
    finally:
        w._controller.abort()
        w._analysis_controller.abort()


def test_cancel_outside_a_restore_still_reports_the_plain_dropped_queue(qapp, qtbot,
                                                                       tmp_path):
    """The chaos2-001 contract is untouched: an ordinary multi-file open still says how
    many queued files were dropped, and says nothing about a project."""
    first = _write_csv(tmp_path / "a.csv", 400_000, seed=0)
    second = _write_csv(tmp_path / "b.csv", 20_000, seed=1)
    w = _new_window(qtbot)
    try:
        w.open_paths([str(first), str(second)])
        assert w._active_open is not None
        w._cancel_load()
        msg = w.statusBar().currentMessage()
        assert "Cancel" in msg and "1 more queued file(s) were not opened." in msg, msg
        assert "project" not in msg.lower(), msg
        assert w._restore_cancelled == []
    finally:
        w._controller.abort()
        w._analysis_controller.abort()
