"""Regression tests for chaos-001: a finishing load must not strip a running job's chrome.

MainWindow shares ONE progress bar + Cancel button between the load, analysis and
batch controllers. ``_on_loaded`` hid them unconditionally when a load landed — so
opening a recording while a threshold sweep ran (explicitly allowed: "Opening a file
is not blocked by an analysis", USER_GUIDE) hid the bar AND the only Cancel control
while the sweep was still running and ``_has_cancellable_job()`` was still True. A
sweep started via ``run_callable`` without ``wants_progress`` emits no ticks, so
nothing ever re-showed them: the UI read idle for the rest of a multi-second run.

The fix routes every load-finish slot through ``_release_load_chrome``, which hides
the chrome only when nothing cancellable is left — and otherwise hands it back to
the surviving job, repainting the label its started-slot put up (the bar was still
claiming "Opening <file>…" about a load that had just ended).

The background job here is a real ``run_callable`` worker gated on a threading.Event,
so it is genuinely running and genuinely cancellable while the assertions look; the
load is a real CSV driven through ``open_paths`` — the user's own File ▸ Open path.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 40_000


# ================================================================================================
# builders
# ================================================================================================

def _write_csv(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": arr}).to_csv(path, index=False)
    return path


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._controller.abort()


@pytest.fixture
def running_job(window, qtbot):
    """A real cancellable background job on the analysis controller, held open by an
    Event — the same no-progress-ticks shape as the threshold sweep — and always
    released at teardown so the worker thread never outlives the test."""
    release = threading.Event()

    def job():
        release.wait(30.0)
        return None

    assert window._analysis_controller.run_callable(job, label="sweep_threshold")
    assert window._analysis_controller.is_busy()
    yield release
    release.set()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=30_000)


def _chrome(window) -> tuple[bool, bool]:
    """(bar visible, Cancel visible AND enabled) — relative to the status bar, since
    an offscreen top-level window is never itself visible."""
    bar = window._progress.isVisibleTo(window.statusBar())
    cancel = (window._cancel_btn.isVisibleTo(window.statusBar())
              and window._cancel_btn.isEnabled())
    return bar, cancel


def _open_and_wait(window, qtbot, csv: Path) -> None:
    window.open_paths([str(csv)])
    # Drain the open pipeline rather than waiting a fixed interval: the load is
    # asynchronous, so a fixed wait can report success before anything happened.
    qtbot.waitUntil(lambda: window._active_open is None and not window._open_queue,
                    timeout=30_000)
    qtbot.wait(30)


# ================================================================================================
# a load landing mid-job hands the chrome back instead of taking it down
# ================================================================================================

def test_load_finishing_mid_job_keeps_bar_and_cancel(window, qtbot, tmp_path, running_job):
    assert _chrome(window) == (True, True), "the job put the chrome up"

    _open_and_wait(window, qtbot, _write_csv(tmp_path / "pore.csv"))

    assert window._analysis_controller.is_busy(), "the job must still be running"
    assert window._has_cancellable_job()
    bar, cancel = _chrome(window)
    assert bar, "the progress bar must survive a load finishing mid-job"
    assert cancel, "Cancel is offered while something is genuinely stoppable"
    # ...and the bar speaks for the SURVIVING job, not the load that just ended.
    assert window._progress.format() == "Analysing (sweep_threshold)…"

    # When the job actually ends, the chrome comes down as always.
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        running_job.set()
    qtbot.wait(30)
    assert _chrome(window) == (False, False)


def test_failed_load_mid_job_keeps_bar_and_cancel(window, qtbot, monkeypatch, running_job):
    """The failure leg of the same lifecycle: a load that FAILS mid-job must not
    strip the survivor's chrome either."""
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: QtWidgets.QMessageBox.Ok)   # the failure dialog

    window._on_failed("synthetic: could not open")

    assert window._analysis_controller.is_busy()
    assert _chrome(window) == (True, True)
    assert window._progress.format() == "Analysing (sweep_threshold)…"


# ================================================================================================
# with nothing running, a finishing load still takes the chrome down
# ================================================================================================

def test_plain_load_still_hides_the_chrome(window, qtbot, tmp_path):
    _open_and_wait(window, qtbot, _write_csv(tmp_path / "quiet.csv"))

    assert len(list(window._session.documents())) == 1
    assert not window._has_cancellable_job()
    assert _chrome(window) == (False, False)
