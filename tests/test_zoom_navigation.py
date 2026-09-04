"""Regression tests for trace-view zoom/navigation — each locks a real bug fixed on 2026-07-17.

Driven headless (QT_QPA_PLATFORM=offscreen). Uses the smallest real recording so the suite stays
fast; the behaviours under test are size-independent.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import nanoscope  # noqa: F401  -> bootstrap.py fixes the Qt plugin path before any Qt import
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.main_window import MainWindow

# The two smallest real recordings, with DIFFERENT time extents (the reopen bug needed that).
ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0000_Process_-1000 mV.abf")  # ~20 s
H5 = Path("/Users/abrahamcampos/2_3 electroporation-005_Cut_1.h5")                                # 40 s

pytestmark = pytest.mark.skipif(
    not (ABF.exists() and H5.exists()), reason="real corpus files not present")


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _open(win, app, path, pumps=80):
    win.open_paths([str(path)])
    for _ in range(pumps):
        app.processEvents()
        QtCore.QThread.msleep(12)


def _pump(app, n=8):
    for _ in range(n):
        app.processEvents()


def _xspan(win):
    lo, hi = win._trace_view.anchor_plot().vb.viewRange()[0]
    return hi - lo


def _true_extent(win, path):
    # By SOURCE: a doc_id names the LOAD, not the file (nanoscope/core/identity.py), so it
    # cannot be reconstructed from a path — and this helper is called after a reopen, which
    # is precisely when the previous load's id is stale.
    did = win._session.doc_id_for_source(str(Path(path).resolve()))
    assert did is not None, f"{Path(path).name} is not open"
    vals = win._session.x_axis(did, 0.0).values
    return float(vals[-1] - vals[0])


def test_fresh_load_shows_whole_trace(qapp):
    """BUG: autoRange()+clipToView framed only ~half the recording on load. Must show all of it."""
    win = MainWindow()
    _open(win, qapp, ABF)
    assert _xspan(win) == pytest.approx(_true_extent(win, ABF), abs=0.5)


def test_reset_returns_to_full_extent_after_deep_zoom(qapp):
    """BUG: Reset used autoRange(), which a clipToView curve defeats — it barely moved. Must
    restore the full data extent from a millisecond-deep zoom."""
    win = MainWindow()
    _open(win, qapp, ABF)
    full = _xspan(win)
    tv = win._trace_view
    lo, hi = tv.anchor_plot().vb.viewRange()[0]
    c = (lo + hi) / 2
    tv.zoom_x_to(c - 0.004, c + 0.004)   # 8 ms
    _pump(qapp)
    assert _xspan(win) < 0.05            # actually dove in
    tv.reset_zoom()
    _pump(qapp)
    assert _xspan(win) == pytest.approx(full, abs=0.5)


def test_reopen_does_not_inherit_previous_files_extent(qapp):
    """BUG: opening a file after closing a shorter-extent one truncated the new view to the old
    file's end, via the navigator region's stale bounds feeding back through the two-way binding."""
    win = MainWindow()
    _open(win, qapp, H5)                      # 40 s file first
    win._session.close_document(str(H5.resolve()))
    _pump(qapp, 12)
    _open(win, qapp, ABF)                     # then the 20 s file
    assert _xspan(win) == pytest.approx(_true_extent(win, ABF), abs=0.5)


def test_zoom_history_back_and_forward(qapp):
    """Dive into an event, step Back to the prior framing, Forward to the dive again."""
    win = MainWindow()
    _open(win, qapp, ABF)
    tv = win._trace_view
    settle_ms = tv._settle.interval() + 60

    def settle():
        for _ in range(int(settle_ms / 15) + 4):
            qapp.processEvents()
            QtCore.QThread.msleep(15)

    lo, hi = tv.anchor_plot().vb.viewRange()[0]
    c = (lo + hi) / 2
    tv.zoom_x_to(c - 1.0, c + 1.0); settle()          # browse: 2 s
    browse = _xspan(win)
    tv.zoom_x_to(c - 0.004, c + 0.004); settle()      # dive: 8 ms
    dived = _xspan(win)
    assert dived < 0.05 < browse

    tv.view_back(); _pump(qapp)
    assert _xspan(win) == pytest.approx(browse, abs=0.2)
    tv.view_forward(); _pump(qapp)
    assert _xspan(win) == pytest.approx(dived, rel=0.2)


def test_wheel_pans_signal_vertically_without_zooming(qapp):
    """Wheel up/down moves the waveform only up/down; neither axis span may change."""
    win = MainWindow()
    _open(win, qapp, ABF)
    vb = win._trace_view.anchor_plot().vb
    vb.setYRange(-60.0, -40.0, padding=0)
    x0 = tuple(vb.viewRange()[0]); y0 = tuple(vb.viewRange()[1])

    class _Wheel:
        def __init__(self, d): self._d = d
        def delta(self): return self._d
        def accept(self): pass
        def ignore(self): pass

    vb.wheelEvent(_Wheel(120))
    _pump(qapp)
    x1 = tuple(vb.viewRange()[0]); y1 = tuple(vb.viewRange()[1])
    assert x1 == pytest.approx(x0, abs=1e-9)
    assert (y1[1] - y1[0]) == pytest.approx(y0[1] - y0[0], abs=1e-9)
    assert y1[0] < y0[0] and y1[1] < y0[1]  # lower view range -> fixed signal moves up on screen

    vb.wheelEvent(_Wheel(-120))
    _pump(qapp)
    x2 = tuple(vb.viewRange()[0]); y2 = tuple(vb.viewRange()[1])
    assert x2 == pytest.approx(x0, abs=1e-9)
    assert y2 == pytest.approx(y0, abs=1e-9)


def test_navigator_drag_still_drives_view(qapp):
    """The guard must NOT break the intended direction: a user drag of the region scrubs the view."""
    win = MainWindow()
    _open(win, qapp, ABF)
    lo, hi = win._trace_view.anchor_plot().vb.viewRange()[0]
    target_lo, target_hi = lo + 2.0, lo + 6.0
    win._navigator._region.setRegion((target_lo, target_hi))   # not _syncing -> a "user" drag
    _pump(qapp)
    vlo, vhi = win._trace_view.anchor_plot().vb.viewRange()[0]
    assert vlo == pytest.approx(target_lo, abs=0.3)
    assert vhi == pytest.approx(target_hi, abs=0.3)
