"""The shared rig for the ``test_trace_*`` files: a synthetic two-channel recording on a real
``TraceView`` with the Measure panel, the Navigator and the Scan bar bound to it, plus the mouse,
wheel and cursor synthesis the gesture tests need.

SYNTHETIC ONLY. Nothing here opens a recording from disk: opening one through the app's
calibration code can write a sidecar that silently rescales it. The "recording" is two arrays in a
``TraceDocument`` whose path does not exist.

Pane order is current first, then voltage (``plan_panes`` ranks CURRENT above VOLTAGE), so
``rig.view.panes()[0]`` is the Measure target (channel 1, "I") and ``panes()[1]`` is the other
pane ("V") in both Stacked and Overlay.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.measure_panel import MeasurePanel
from nanoscope.gui.navigator import Navigator
from nanoscope.gui.pan_scrollbar import PanScrollBar
from nanoscope.gui.trace_view import TraceView

Qt = QtCore.Qt

FS = 10_000.0
N = 100_000                      # 10 s
CURRENT = 1                      # channel index of the Measure target ("I", pA)
VOLTAGE = 0                      # the other pane ("V", mV)


def make_document(n: int = N, name: str = "trace_rig.dat") -> TraceDocument:
    rng = np.random.default_rng(11)
    current = (100.0 + rng.normal(0.0, 1.0, n)).astype(np.float32)
    voltage = np.full(n, -100.0, dtype=np.float32)
    meta = RecordingMeta(
        path=Path("/nonexistent") / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
        channels=(ChannelInfo(index=0, name="V", raw_units="mV", kind=ChannelKind.VOLTAGE),
                  ChannelInfo(index=1, name="I", raw_units="pA", kind=ChannelKind.CURRENT)))
    return TraceDocument(meta=meta, channels=(voltage, current))


@dataclass
class Rig:
    session: Session
    view: TraceView
    panel: MeasurePanel
    navigator: Navigator
    bar: PanScrollBar
    doc_id: str

    @property
    def target_vb(self):
        return self.view.pane_for_layer(f"{self.doc_id}#{CURRENT}").plot.vb

    @property
    def other_vb(self):
        return self.view.pane_for_layer(f"{self.doc_id}#{VOLTAGE}").plot.vb

    def xs(self) -> np.ndarray:
        return self.session.x_axis(self.doc_id).values

    def x_range(self) -> tuple[float, float]:
        lo, hi = self.view.anchor_plot().vb.viewRange()[0]
        return float(lo), float(hi)

    def y_range(self, vb=None) -> tuple[float, float]:
        lo, hi = (vb or self.target_vb).viewRange()[1]
        return float(lo), float(hi)


def build_rig(qapp, *, parent: QtWidgets.QWidget | None = None, show: bool = True) -> Rig:
    session = Session()
    doc = make_document()
    session.add_document(doc)
    session.add_layer(doc.doc_id, VOLTAGE)
    session.add_layer(doc.doc_id, CURRENT)
    view = TraceView(session, parent)
    panel = MeasurePanel(session, view)
    navigator = Navigator()
    navigator.bind(session, view)
    navigator.set_target(doc.doc_id, CURRENT)
    bar = PanScrollBar()
    bar.bind(session, view)
    bar.set_target(doc.doc_id, CURRENT)
    panel.set_selection(doc.doc_id, CURRENT)
    if parent is None:
        # The offscreen screen is 800x800 and QApplication.widgetAt (which the pointer helpers
        # use) finds nothing beyond it, so the trace has to fit inside the screen.
        view.resize(780, 700)
    if show:
        view.show()
    qapp.processEvents()
    return Rig(session, view, panel, navigator, bar, doc.doc_id)


@pytest.fixture
def free_mouse():
    """pyqtgraph drops mouse moves closer together than 10 ms (``mouseRateLimit``); a synthetic
    gesture sends them back to back. Restored afterwards: it is a process-wide option."""
    old = pg.getConfigOption("mouseRateLimit")
    pg.setConfigOption("mouseRateLimit", -1)
    yield
    pg.setConfigOption("mouseRateLimit", old)


@pytest.fixture
def rig(qapp, free_mouse):
    r = build_rig(qapp)
    yield r
    # No view.close(): pyqtgraph's close() clears the scene and detaches the viewport, which
    # segfaults in PySide wrapper finalization here. The other trace tests simply let go too.
    r.view.hide()


# -- geometry ---------------------------------------------------------------------------------
def zoom(rig: Rig, lo: float, hi: float) -> None:
    rig.view.zoom_x_to(lo, hi)
    QtWidgets.QApplication.processEvents()


def zoom_middle(rig: Rig, half_width: float) -> tuple[float, float]:
    e0, e1 = rig.view.full_x_extent()
    c = 0.5 * (e0 + e1)
    zoom(rig, c - half_width, c + half_width)
    return rig.x_range()


def point_at(view: TraceView, vb, x: float, yfrac: float = 0.5) -> QtCore.QPointF:
    """Viewport position of data-x ``x`` at vertical fraction ``yfrac`` of ``vb``."""
    sx = vb.mapViewToScene(QtCore.QPointF(x, 0.0)).x()
    r = vb.sceneBoundingRect()
    sy = r.top() + yfrac * r.height()
    return QtCore.QPointF(view.mapFromScene(QtCore.QPointF(sx, sy)))


def global_of(view: TraceView, pos) -> QtCore.QPoint:
    p = pos.toPoint() if hasattr(pos, "toPoint") else pos
    return view.viewport().mapToGlobal(p)


# -- mouse ------------------------------------------------------------------------------------
def send_mouse(view: TraceView, kind, pos, button=Qt.NoButton, buttons=Qt.NoButton,
               modifiers=Qt.NoModifier) -> None:
    vp = view.viewport()
    p = QtCore.QPointF(pos)
    ev = QtGui.QMouseEvent(kind, p, QtCore.QPointF(global_of(view, p)), button, buttons, modifiers)
    QtWidgets.QApplication.sendEvent(vp, ev)
    QtWidgets.QApplication.processEvents()


def hover(view: TraceView, pos, modifiers=Qt.NoModifier) -> None:
    QtGui.QCursor.setPos(global_of(view, pos))
    send_mouse(view, QtCore.QEvent.MouseMove, pos, modifiers=modifiers)


def press(view, pos, button=Qt.LeftButton, modifiers=Qt.NoModifier) -> None:
    send_mouse(view, QtCore.QEvent.MouseButtonPress, pos, button, button, modifiers)


def release(view, pos, button=Qt.LeftButton, modifiers=Qt.NoModifier) -> None:
    send_mouse(view, QtCore.QEvent.MouseButtonRelease, pos, button, Qt.NoButton, modifiers)


def drag_move(view, pos, buttons=Qt.LeftButton, modifiers=Qt.NoModifier) -> None:
    send_mouse(view, QtCore.QEvent.MouseMove, pos, Qt.NoButton, buttons, modifiers)


def lerp(a: QtCore.QPointF, b: QtCore.QPointF, t: float) -> QtCore.QPointF:
    return QtCore.QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)


def drag(view, p0, p1, *, frames: int = 8, modifiers=Qt.NoModifier, release_it: bool = True) -> None:
    """Hover on ``p0`` (the scene only hands a drag to what it saw under the pointer), press,
    ``frames`` moves to ``p1``, release."""
    hover(view, p0, modifiers)
    press(view, p0, modifiers=modifiers)
    for k in range(1, frames + 1):
        drag_move(view, lerp(p0, p1, k / frames), modifiers=modifiers)
    if release_it:
        release(view, p1, modifiers=modifiers)


def double_click(view, pos) -> None:
    """press, release, then the second press as a DoubleClick event, and its release."""
    hover(view, pos)
    press(view, pos)
    release(view, pos)
    send_mouse(view, QtCore.QEvent.MouseButtonDblClick, pos, Qt.LeftButton, Qt.LeftButton)
    release(view, pos)


# -- wheel ------------------------------------------------------------------------------------
def wheel(view, pos, *, pixel=(0, 0), angle=(0, 0), phase=Qt.NoScrollPhase,
          modifiers=Qt.NoModifier) -> None:
    vp = view.viewport()
    p = QtCore.QPointF(pos)
    ev = QtGui.QWheelEvent(p, QtCore.QPointF(global_of(view, p)), QtCore.QPoint(*pixel),
                           QtCore.QPoint(*angle), Qt.NoButton, modifiers, phase, False)
    QtWidgets.QApplication.sendEvent(vp, ev)
    QtWidgets.QApplication.processEvents()


def swipe(view, pos, steps, *, begin: bool = True, end: bool = False) -> None:
    """A trackpad gesture: ``steps`` is a list of ``(pixel_x, pixel_y)`` updates. Qt gives a
    pixel-precise device an angle delta of two per pixel."""
    if begin:
        wheel(view, pos, phase=Qt.ScrollBegin)
    for px, py in steps:
        wheel(view, pos, pixel=(px, py), angle=(2 * px, 2 * py), phase=Qt.ScrollUpdate)
    if end:
        wheel(view, pos, phase=Qt.ScrollEnd)


class Recorder:
    """Collects a signal's emissions."""

    def __init__(self, *signals) -> None:
        self.calls: list[tuple] = []
        for s in signals:
            s.connect(self._on)

    def _on(self, *args) -> None:
        self.calls.append(args)

    def clear(self) -> None:
        self.calls.clear()

    def __len__(self) -> int:
        return len(self.calls)
