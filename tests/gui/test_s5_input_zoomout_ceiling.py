"""Sweep-5 IF-01 regression: wheel zoom-out is CLAMPED at MAX_X_SPAN_FACTOR x the data
extent, so a zoom-out storm coasts to a stop instead of driving the span astronomically
past the recording and crashing pyqtgraph's auto-downsampler.

The defect, measured on the pristine code: from full extent on an 8 s / 2M-sample
recording, 123 plain 120-delta wheel notches (a held wheel or one trackpad inertia
fling) drove the visible span to 5.9e16 s, at which point every repaint raised
``ValueError: Maximum allowed dimension exceeded`` from PlotDataItem's downsampler —
a dead, non-updating curve, +1 console traceback per further notch, recoverable only
by knowing Ctrl+0 / double-click reset. The y-side has had a ``minYRange`` floor since
v1; the x zoom-out side was the one unguarded direction.

The fix (gui/trace_view.py): ``_apply_x_zoom_ceiling`` sets
``setLimits(maxXRange=MAX_X_SPAN_FACTOR * extent)`` on every pane at each ``rebuild``,
so the wheel, right-drag pan-mode scaling and programmatic ``setXRange`` all hit the
same ceiling inside ``ViewBox.setRange`` itself. Only the SPAN is limited, never the
position — panning off the recording stays a legitimate, reversible gesture that other
surfaces already answer honestly.

Driven through a wired MainWindow with real QWheelEvents on the viewport (the shipped
gesture), offscreen.
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


def _wheel_out(view, qapp, notches: int) -> None:
    """Send real 120-delta zoom-out QWheelEvents at the anchor pane's centre — the exact
    gesture of a held wheel / trackpad inertia fling over the trace."""
    vp = view.viewport()
    anchor = view.anchor_plot()
    sc = anchor.vb.sceneBoundingRect()
    pos = QtCore.QPointF(sc.center().x(), sc.center().y())
    gpf = QtCore.QPointF(vp.mapToGlobal(pos.toPoint()))
    for _ in range(notches):
        ev = QtGui.QWheelEvent(pos, gpf, QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
                               QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                               QtCore.Qt.NoScrollPhase, False)
        QtWidgets.QApplication.sendEvent(vp, ev)
        qapp.processEvents()


def _extent_span(view) -> float:
    extent = view.full_x_extent()
    assert extent is not None
    return extent[1] - extent[0]


def test_wheel_zoomout_storm_coasts_to_the_ceiling(window, qapp, qtbot, tmp_path):
    """200 held zoom-out notches (well past the pristine code's 123-notch crash point)
    leave the span AT the ceiling, the view finite, and the curve still updatable."""
    _install(window, _synth_doc(tmp_path))
    qtbot.wait(50)
    view = window._trace_view
    anchor = view.anchor_plot()
    span_full = _extent_span(view)

    _wheel_out(view, qapp, 200)

    x0, x1 = anchor.vb.viewRange()[0]
    span = x1 - x0
    assert np.isfinite(x0) and np.isfinite(x1)
    # AT the ceiling (the storm coasted to a stop there), never past it. Small relative
    # margin: pyqtgraph applies the clamp through float scale math.
    ceiling = MAX_X_SPAN_FACTOR * span_full
    assert span <= ceiling * (1 + 1e-6)
    assert span == pytest.approx(ceiling, rel=1e-3)

    # The renderer is still alive: a repaint-driving range change goes through without
    # the downsampler's ValueError (on the pristine code this raised on every repaint).
    anchor.vb.setXRange(x0, x1 - span * 0.25, padding=0)
    qapp.processEvents()

    # ...and reset still recovers the full framing, exactly as before the fix.
    view.reset_zoom()
    qapp.processEvents()
    r = anchor.vb.viewRange()[0]
    assert r[1] - r[0] == pytest.approx(span_full, rel=1e-3)


def test_programmatic_and_scale_paths_hit_the_same_ceiling(window, qapp, qtbot, tmp_path):
    """The clamp lives in ViewBox.setRange, so the pan-mode right-drag scaling route
    (scaleBy — the verifier's independent path) and a programmatic setXRange are bounded
    by the same rule as the wheel: one limit, every zoom-out path."""
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


def test_zoom_history_stays_sane_after_a_storm(window, qapp, qtbot, tmp_path):
    """After a zoom-out storm the Back/Forward trail holds only render-safe framings
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
    # ...then the storm, settled into history.
    _wheel_out(view, qapp, 150)
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
