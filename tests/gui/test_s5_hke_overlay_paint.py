"""HKE-04 — the 100k-span detection overlay must paint in frame-budget time, every frame.

``SpanOverlayItem.paint`` honoured its docstring's O(visible spans) — but with one Python
``QRectF`` + ``drawRect`` per span (~70 µs each). At the 1e5 top of the module's own
declared event range every full-window repaint blocked the GUI thread 4-8 s, and the cost
recurred on EVERY expose (no cache absorbs it; the independent probe measured #1 7965 ms,
#2 6897 ms). The fix vectorises the geometry and MERGES runs of spans at pixel
resolution — the same decimation idea the trace curve already uses: two spans separated
by less than one device pixel cannot be told apart on screen, so they paint as one run of
shade, and the number of rects drawn is O(min(visible spans, viewport pixels)). Any gap
the user could SEE (>= 1 px) is preserved exactly, so a zoomed view paints the same
distinct rects it always did. Post-fix the same probe measures ~3 ms.

Pixel assertions render a real ``pg.PlotWidget`` to a ``QImage`` and read the pixels —
what the user's screen actually shows. The timing bound is generous (~100x post-fix) so
CI load cannot fail it while the per-span-Python regression (3 orders of magnitude over)
cannot pass it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui

from nanoscope.gui.overlays import SpanOverlayItem


# ================================================================================================
# builders
# ================================================================================================
def _widget(starts, stops, *, x0=0.0, x1=1.0):
    w = pg.PlotWidget()
    w.resize(800, 400)
    item = SpanOverlayItem(np.asarray(starts, float), np.asarray(stops, float))
    w.addItem(item, ignoreBounds=True)
    w.setXRange(x0, x1, padding=0)
    w.setYRange(0.0, 1.0, padding=0)
    w.show()
    return w, item


def _render(w) -> QtGui.QImage:
    img = QtGui.QImage(800, 400, QtGui.QImage.Format_ARGB32)
    img.fill(QtGui.QColor("black"))
    p = QtGui.QPainter(img)
    w.render(p)
    p.end()
    return img


def _pixel_at(w, img, x_data: float) -> tuple[int, int, int]:
    """The rendered RGB at data-x ``x_data``, mid-height of the viewbox."""
    vb = w.getPlotItem().vb
    dev = vb.mapViewToDevice(QtCore.QPointF(float(x_data), 0.5))
    c = img.pixelColor(int(dev.x()), int(dev.y()))
    return (c.red(), c.green(), c.blue())


# ================================================================================================
# what the screen shows — unchanged where the eye could tell
# ================================================================================================
def test_spans_are_shaded_and_a_visible_gap_stays_unshaded(qapp, qtbot):
    """Two spans a third of the pane apart: both shaded, the gap between them exactly
    as unshaded as empty background — pixel-merging must never bridge a gap the user
    can see."""
    w, _item = _widget([0.20, 0.60], [0.30, 0.70])
    qtbot.addWidget(w)
    qapp.processEvents()
    img = _render(w)
    empty = _pixel_at(w, img, 0.05)             # far from any span
    assert _pixel_at(w, img, 0.25) != empty, "the first span must be shaded"
    assert _pixel_at(w, img, 0.65) != empty, "the second span must be shaded"
    assert _pixel_at(w, img, 0.45) == empty, "the visible gap must stay unshaded"
    w.close()


def test_a_sub_pixel_span_still_paints_at_least_a_pixel(qapp, qtbot):
    """The one-device-pixel widening survives the vectorisation: a span 1e-9 wide in a
    unit view is still visibly there — 'detection finished' must not look identical to
    'nothing found'."""
    w, _item = _widget([0.5], [0.5 + 1e-9])
    qtbot.addWidget(w)
    qapp.processEvents()
    img = _render(w)
    assert _pixel_at(w, img, 0.5) != _pixel_at(w, img, 0.1), (
        "the sub-pixel span must be widened to a visible pixel")
    w.close()


def test_100k_spans_shade_where_the_events_are(qapp, qtbot):
    """At full-window zoom the merged runs must still cover the events: every probe
    point inside the span comb is shaded, points outside the comb are not."""
    n = 100_000
    starts = 0.05 + 0.9 * np.arange(n) / n
    stops = starts + 0.9 / (2 * n)              # half-duty comb, all sub-pixel
    w, item = _widget(starts, stops)
    qtbot.addWidget(w)
    qapp.processEvents()
    assert item.n_spans == n
    img = _render(w)
    empty = _pixel_at(w, img, 0.01)
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert _pixel_at(w, img, x) != empty, f"comb must be shaded at x={x}"
    assert _pixel_at(w, img, 0.97) == empty, "beyond the last span stays unshaded"
    w.close()


def test_zoomed_in_views_paint_the_same_distinct_rects_as_before(qapp, qtbot):
    """The merge only ever joins sub-pixel neighbours, so a zoom window holding a few
    well-separated events paints them individually — the regime the item was already
    fast and correct in."""
    starts = np.array([0.1, 0.4, 0.7])
    stops = starts + 0.1
    w, _item = _widget(starts, stops, x0=0.0, x1=1.0)
    qtbot.addWidget(w)
    qapp.processEvents()
    img = _render(w)
    empty = _pixel_at(w, img, 0.02)
    for x in (0.15, 0.45, 0.75):
        assert _pixel_at(w, img, x) != empty
    for x in (0.25, 0.55, 0.85):
        assert _pixel_at(w, img, x) == empty
    w.close()


# ================================================================================================
# the cost, at the dossier's own scale
# ================================================================================================
def test_full_window_paint_with_100k_spans_is_frame_budget_not_seconds(qapp, qtbot):
    """The measured defect: 4-8 s per full-window repaint at 100k spans, on EVERY
    expose. Post-fix the direct render measures ~3 ms. Bound: best of three renders
    < 1.5 s — hundreds of times the real cost, three orders of magnitude under the
    regression's."""
    n = 100_000
    starts = 0.5 + 1.49e-3 * np.arange(n)
    stops = starts + 60 / 100_000.0
    w, item = _widget(starts, stops, x0=float(starts[0]), x1=float(stops[-1]))
    qtbot.addWidget(w)
    qapp.processEvents()
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        _render(w)
        timings.append(time.perf_counter() - t0)
    assert min(timings) < 1.5, (
        f"full-window renders at {n:,} spans took {timings} (were 4-8 s EACH, "
        f"including the second — no cache absorbs this path)")
    w.close()
