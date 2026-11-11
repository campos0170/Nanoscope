"""The Measure mask's look: ``[`` and ``]`` brackets in the annotation ink over a dark casing, grip
tabs near the bottom, a ring on a fitted edge's sample, and one look per state (fitted, placed,
drawn, start only, being drawn). The hit areas are the ones they always were.

On the shared synthetic rig (``trace_rig``) with the pane set to the app's dark ``#111316``
explicitly: offscreen, `TraceView` inherits a light background. Pixels are read from a grab of
the viewport at the places `TraceView.mask_look` reports.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.gui.trace_view import ANNOTATION_INK

from .trace_rig import (  # noqa: F401  (fixtures are used by name)
    FS, Qt, drag, drag_move, free_mouse, hover, point_at, press, release, rig, zoom_middle)

INK = QtGui.QColor(ANNOTATION_INK)
PANE = "#111316"


def _dist(c: QtGui.QColor, ref: QtGui.QColor) -> float:
    return float(np.sqrt((c.red() - ref.red()) ** 2 + (c.green() - ref.green()) ** 2
                         + (c.blue() - ref.blue()) ** 2))


def _is_ink(c: QtGui.QColor) -> bool:
    return _dist(c, INK) < 70.0


def _is_dark(c: QtGui.QColor) -> bool:
    return c.lightness() < 40


class _Shot:
    def __init__(self, view) -> None:
        QtWidgets.QApplication.processEvents()
        self.img = view.viewport().grab().toImage()

    def at(self, x: float, y: float) -> QtGui.QColor:
        return QtGui.QColor(self.img.pixel(int(round(x)), int(round(y))))

    def ink_near(self, x: float, y: float, r: int = 1) -> bool:
        return any(_is_ink(self.at(x + dx, y + dy))
                   for dx in range(-r, r + 1) for dy in range(-r, r + 1))


def _masked(rig, states=("drawn", "drawn"), half=0.15):
    rig.view.setBackground(PANE)
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    rig.view.set_measure_region(c - half, c + half)
    rig.view.set_measure_edge_states(*states)
    QtWidgets.QApplication.processEvents()
    return c, rig.view.mask_look()


def _edge(look, side):
    return next(e for e in look["edges"] if e["side"] == side)


def test_mask_edges_are_drawn_in_the_annotation_ink_with_a_casing(rig):
    _c, look = _masked(rig)
    shot = _Shot(rig.view)
    y = look["top"] + 40
    for edge in look["edges"]:
        assert shot.ink_near(edge["x"], y), edge["side"]
        # The dark casing on the side away from the span (the span side is the faint fill).
        outside = edge["x"] - 2.5 * edge["inward"]
        assert _is_dark(shot.at(outside, y)), edge["side"]


def test_grip_tabs_sit_in_the_bottom_strip_not_mid_pane(rig):
    _c, look = _masked(rig)
    shot = _Shot(rig.view)
    height = look["bottom"] - look["top"]
    mid = look["top"] + 0.5 * height
    for edge in look["edges"]:
        tab = edge["tab"]
        assert tab is not None and tab.bottom() <= look["bottom"]
        assert tab.top() > look["top"] + 0.85 * height              # the bottom strip
        assert _is_ink(shot.at(tab.center().x() + 4.5, tab.center().y()))   # the filled tab
        # Nothing of the grip at mid-height beside the line (the old circle handle sat there).
        assert not shot.ink_near(edge["x"] + 6 * edge["inward"], mid, r=0)


def test_start_and_finish_are_mirror_brackets(rig):
    _c, look = _masked(rig)
    shot = _Shot(rig.view)
    start, end = _edge(look, "start"), _edge(look, "end")
    for y in (look["top"], look["bottom"]):
        assert shot.ink_near(start["x"] + 6, y) and not shot.ink_near(start["x"] - 6, y, r=0)
        assert shot.ink_near(end["x"] - 6, y) and not shot.ink_near(end["x"] + 6, y, r=0)


def test_a_fitted_edge_has_a_ring_on_its_snapped_sample(rig):
    c, look = _masked(rig, ("fit", "drawn"))
    start = _edge(look, "start")
    assert start["ring"] is not None and _edge(look, "end")["ring"] is None
    # On the target layer's own sample under the edge, at its drawn value.
    view = rig.view
    lo, _hi = view.measure_bounds()
    xs = rig.xs()
    ys = view.pane_for_layer(f"{rig.doc_id}#1").layer_y_data(f"{rig.doc_id}#1")
    scene = rig.target_vb.mapViewToScene(QtCore.QPointF(xs[lo], float(ys[lo])))
    expect = view.mapFromScene(scene)
    assert start["ring"].x() == pytest.approx(expect.x(), abs=1.5)
    assert start["ring"].y() == pytest.approx(expect.y(), abs=1.5)
    shot = _Shot(view)
    ring = start["ring"]
    assert sum(shot.ink_near(ring.x() + 6 * np.cos(a), ring.y() + 6 * np.sin(a))
               for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)) >= 9


def test_a_placed_edge_is_dashed_and_tagged_not_fitted(rig):
    _c, look = _masked(rig, ("fit", "placed"))
    end, start = _edge(look, "end"), _edge(look, "start")
    assert end["dashed"] and not start["dashed"]
    text, box = end["tag"]
    assert text == "not fitted" and box.left() > end["x"]           # outside the span
    assert start["tag"] is None
    shot = _Shot(rig.view)

    def runs(x):
        # The pixel column wholly inside the 2 px stroke (x is fractional).
        col = [_is_ink(shot.at(np.floor(x), y))
               for y in range(int(look["top"] + 15), int(look["top"] + 120))]
        assert sum(col) > 20                                      # it is the edge's column
        return sum(1 for a, b in zip(col, col[1:]) if a != b)

    assert runs(end["x"]) >= 10 and runs(start["x"]) <= 1        # dashed against solid


def test_start_alone_draws_one_bracket_and_no_fill(rig):
    _c, look = _masked(rig, ("fit", None))
    assert [e["side"] for e in look["edges"]] == ["start"]
    assert rig.view._measure_region.brush.color().alpha() == 0
    rig.view.set_measure_edge_states("fit", "fit")
    assert rig.view._measure_region.brush.color().alpha() > 0


def test_a_draw_in_progress_is_dashed_without_tabs(rig):
    rig.view.setBackground(PANE)
    c = sum(zoom_middle(rig, 0.5)) / 2
    rig.panel.set_active(True, seed=False)
    assert rig.view.set_draw_mode(True)
    p0 = point_at(rig.view, rig.target_vb, c - 0.2)
    p1 = point_at(rig.view, rig.target_vb, c + 0.1)
    drag(rig.view, p0, p1, release_it=False)
    look = rig.view.mask_look()
    assert look["drawing"] and len(look["edges"]) == 2
    assert all(e["dashed"] and e["tab"] is None and e["ring"] is None for e in look["edges"])
    release(rig.view, p1)
    look = rig.view.mask_look()
    assert not look["drawing"]
    assert all(not e["dashed"] and e["tab"] is not None for e in look["edges"])


def test_tabs_lift_above_the_pane_caption(rig):
    view = rig.view
    view.setBackground(PANE)
    lo, hi = zoom_middle(rig, 0.5)
    rig.panel.set_active(True, seed=False)
    view.set_target_label("→ Manual · 0 hand-made · 0 in view · a long caption to sit under")
    view.set_measure_region(lo + 0.02, lo + 0.12)                # its start over the caption
    QtWidgets.QApplication.processEvents()
    label = view._target_label
    caption = view.mapFromScene(label.sceneBoundingRect()).boundingRect()
    look = view.mask_look()
    tab = _edge(look, "start")["tab"]
    assert caption.left() <= _edge(look, "start")["x"] <= caption.right()
    assert tab.bottom() < caption.top()


def test_edge_hover_cursor_is_horizontal_resize(rig):
    _masked(rig)
    for line in rig.view._measure_region.lines:
        assert line.cursor().shape() == Qt.SizeHorCursor


def test_the_edges_keep_their_grip_sized_hit_strip(rig):
    """The pens are transparent now; the invisible marker still widens each line's hit strip
    over the whole pane height, wide enough to hold the grip tab drawn on it."""
    _c, look = _masked(rig)
    for line, edge in zip(rig.view._measure_region.lines, look["edges"]):
        assert line.currentPen.color().alpha() == 0
        strip = rig.view.mapFromScene(line.sceneBoundingRect()).boundingRect()
        assert strip.width() >= 16.0
        assert strip.left() <= edge["tab"].left() and edge["tab"].right() <= strip.right() + 1
        assert strip.top() <= edge["tab"].top() and edge["tab"].bottom() <= strip.bottom() + 1


def test_crosshair_line_stays_above_the_mask(rig):
    _c, look = _masked(rig)
    view = rig.view
    edge = _edge(look, "start")
    x0, _x1 = view._measure_region.getRegion()
    view.session.set_cursor(float(x0))                            # the crosshair on the edge
    shot = _Shot(view)
    grey = QtGui.QColor("#9aa2ab")
    hits = sum(1 for y in range(int(look["top"] + 60), int(look["bottom"] - 60))
               if any(_dist(shot.at(edge["x"] + dx, y), grey) < 40 for dx in (-1, 0, 1)))
    assert hits >= 20                                             # its dashes show over the edge
