"""Trace-input regressions: wheel is vertical pan, while horizontal scale-out remains
clamped at MAX_X_SPAN_FACTOR x the data extent so no supported x-zoom path can drive the
span astronomically past the recording and crash pyqtgraph's auto-downsampler.

The defect, measured on the pristine code: from full extent on an 8 s / 2M-sample
recording, 123 plain 120-delta wheel notches (a held wheel or one trackpad inertia
fling) drove the visible span to 5.9e16 s, at which point every repaint raised
``ValueError: Maximum allowed dimension exceeded`` from PlotDataItem's downsampler —
a dead, non-updating curve, +1 console traceback per further notch, recoverable only
by knowing the reset action. The y-side has had a ``minYRange`` floor since
v1; the x zoom-out side was the one unguarded direction.

The x-ceiling fix (gui/trace_view.py): ``_apply_x_zoom_ceiling`` sets
``setLimits(maxXRange=MAX_X_SPAN_FACTOR * extent)`` on every pane at each ``rebuild``,
so right-drag pan-mode scaling and programmatic ``setXRange`` both hit the same ceiling
inside ``ViewBox.setRange`` itself. Only the SPAN is limited, never the
position — panning off the recording stays a legitimate, reversible gesture that other
surfaces already answer honestly.

The wheel contract is driven through a wired MainWindow with real QWheelEvents on the
viewport (the shipped gesture), offscreen.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.trace_view import MAX_X_SPAN_FACTOR

FS = 50_000.0
N = 200_000          # 4 s at 50 kHz — the ceiling is a RATIO, so a small doc tests it fully


def _synth_doc(tmp_path: Path, *, n: int = N, name: str = "zoomout.dat") -> TraceDocument:
    rng = np.random.default_rng(3)
    arr = np.ascontiguousarray(rng.normal(-50.0, 2.0, n), dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1000, 700)
    w.show()
    yield w
    w._analysis_controller.abort()


def _wheel(view, qapp, delta: int, notches: int = 1) -> None:
    """Send real wheel events at the anchor pane's centre."""
    vp = view.viewport()
    anchor = view.anchor_plot()
    sc = anchor.vb.sceneBoundingRect()
    pos = QtCore.QPointF(sc.center().x(), sc.center().y())
    gpf = QtCore.QPointF(vp.mapToGlobal(pos.toPoint()))
    for _ in range(notches):
        ev = QtGui.QWheelEvent(pos, gpf, QtCore.QPoint(0, 0), QtCore.QPoint(0, delta),
                               QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                               QtCore.Qt.NoScrollPhase, False)
        QtWidgets.QApplication.sendEvent(vp, ev)
        qapp.processEvents()


def _extent_span(view) -> float:
    extent = view.full_x_extent()
    assert extent is not None
    return extent[1] - extent[0]


def test_real_wheel_events_pan_only_vertically_in_the_requested_direction(
        window, qapp, qtbot, tmp_path):
    """Up moves a fixed signal point up, down moves it down, with both spans unchanged."""
    _install(window, _synth_doc(tmp_path))
    qtbot.wait(50)
    view = window._trace_view
    anchor = view.anchor_plot()
    view.zoom_x_to(1.0, 2.0)
    anchor.vb.setYRange(-60.0, -40.0, padding=0)
    qapp.processEvents()
    x0 = tuple(anchor.vb.viewRange()[0])
    y0 = tuple(anchor.vb.viewRange()[1])
    fixed = QtCore.QPointF(1.5, -50.0)
    scene_y0 = float(anchor.vb.mapViewToScene(fixed).y())

    _wheel(view, qapp, 120)

    x1 = tuple(anchor.vb.viewRange()[0])
    y1 = tuple(anchor.vb.viewRange()[1])
    scene_y1 = float(anchor.vb.mapViewToScene(fixed).y())
    assert x1 == pytest.approx(x0, abs=1e-9)
    assert (y1[1] - y1[0]) == pytest.approx(y0[1] - y0[0], abs=1e-9)
    assert y1[0] < y0[0] and y1[1] < y0[1]
    assert scene_y1 < scene_y0

    _wheel(view, qapp, -120)

    x2 = tuple(anchor.vb.viewRange()[0])
    y2 = tuple(anchor.vb.viewRange()[1])
    scene_y2 = float(anchor.vb.mapViewToScene(fixed).y())
    assert x2 == pytest.approx(x0, abs=1e-9)
    assert y2 == pytest.approx(y0, abs=1e-9)
    assert scene_y2 == pytest.approx(scene_y0, abs=1e-6)


def test_programmatic_and_scale_paths_hit_the_same_ceiling(window, qapp, qtbot, tmp_path):
    """The clamp lives in ViewBox.setRange, so the pan-mode right-drag scaling route
    (scaleBy — the verifier's independent path) and a programmatic setXRange are bounded
    by the same rule: one limit, every horizontal zoom-out path."""
    _install(window, _synth_doc(tmp_path))
    qtbot.wait(50)
    view = window._trace_view
    anchor = view.anchor_plot()
    ceiling = MAX_X_SPAN_FACTOR * _extent_span(view)

    for _ in range(200):    # pristine code raised after ~126 scaleBy steps
        anchor.vb.scaleBy(x=1.35)
    qapp.processEvents()
    r = anchor.vb.viewRange()[0]
    assert (r[1] - r[0]) <= ceiling * (1 + 1e-6)

    anchor.vb.setXRange(-1e17, 1e17, padding=0)
    qapp.processEvents()
    r = anchor.vb.viewRange()[0]
    assert (r[1] - r[0]) <= ceiling * (1 + 1e-6)


def test_zoom_history_stays_sane_after_a_scale_out_storm(window, qapp, qtbot, tmp_path):
    """After a scale-out storm the Back/Forward trail holds only render-safe framings
    (every recorded span under the ceiling) and Back still restores a real view."""
    _install(window, _synth_doc(tmp_path))
    qtbot.wait(50)
    view = window._trace_view
    anchor = view.anchor_plot()
    span_full = _extent_span(view)
    ceiling = MAX_X_SPAN_FACTOR * span_full

    # A dive first, so there is a framing to come back to...
    view.zoom_x_to(1.0, 1.5)
    view._settle.stop(); view._commit_view()       # the suite's deterministic-settle idiom
    # ...then a right-drag-equivalent scale-out storm, settled into history.
    for _ in range(150):
        anchor.vb.scaleBy(x=1.35)
    view._settle.stop(); view._commit_view()

    assert all((b - a) <= ceiling * (1 + 1e-6) for a, b in view._view_hist)
    assert view.can_go_back()
    view.view_back()
    r = anchor.vb.viewRange()[0]
    assert (r[1] - r[0]) <= ceiling * (1 + 1e-6)
    assert np.isfinite(r[0]) and np.isfinite(r[1])


def test_ceiling_rederived_per_rebuild_and_zoom_in_untouched(window, qapp, qtbot, tmp_path):
    """The ceiling is re-derived by every rebuild (stale state is re-derived, never
    carried), and it is a CEILING only — diving into a millisecond event is unchanged."""
    doc = _synth_doc(tmp_path)
    doc_id, ch = _install(window, doc)
    qtbot.wait(50)
    view = window._trace_view
    anchor = view.anchor_plot()
    span_full = _extent_span(view)

    # Zoom-in is untouched: frame a 2 ms window and it sticks exactly.
    view.zoom_x_to(1.0, 1.002)
    r = anchor.vb.viewRange()[0]
    assert (r[1] - r[0]) == pytest.approx(0.002, rel=1e-3)

    # A rebuild (the layer-toggle path) recreates the panes; the new ViewBoxes carry the
    # same ceiling, not pyqtgraph's default of none.
    view.rebuild()
    qapp.processEvents()
    anchor = view.anchor_plot()
    anchor.vb.setXRange(-1e17, 1e17, padding=0)
    r = anchor.vb.viewRange()[0]
    assert (r[1] - r[0]) <= MAX_X_SPAN_FACTOR * span_full * (1 + 1e-6)
