"""Comparison figures: the pair-matrix feature checklist, and the canvas opening fitted.

Measured before (plotux/columns, plotux/ux L5): a click on a pair-matrix row's TEXT did
not tick it (only the 18 px box did), nothing counted against "two to five" until
Generate refused, and a figure generated with the canvas tall stayed at that scale when
the Analysis controls shrank the canvas, so it opened scrolled into the figure with its
title cut off. Synthetic sources only.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6 import QtCore
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from nanoscope.core.sources import PlotSource
from nanoscope.gui.comparison_figures_panel import METRICS, ComparisonFiguresWindow


def source(key="first", label="Recording A"):
    frame = pd.DataFrame({"event_id": [5, 6, 9, 11, 14, 17],
        "dwell_s": [.001, .002, .003, .004, .005, .006],
        "depth_pA": [10., 20., 30., 40., 50., 60.],
        "amplitude_pA": [12., 25., 35., 42., 55., 64.],
        "ecd_pA_s": [.01, .04, .09, .16, .25, .36], "snr": [5., 8., 10., 12., 15., 16.],
        "frac_depth": [.1, .2, .3, .4, .5, .6],
        "accepted": [True, True, False, True, True, True]})
    return PlotSource(key, label, frame, kind="manual", doc_key=key, current=True)


@pytest.fixture
def page(qtbot):
    page = ComparisonFiguresWindow()
    qtbot.addWidget(page)
    page.resize(1100, 800)
    page.show()
    qtbot.waitExposed(page)
    return page


def _click_text(checklist, key):
    proxy = checklist.proxy
    for r in range(proxy.rowCount()):
        index = proxy.index(r, 0)
        if index.data(Qt.UserRole) == key:
            checklist.view.scrollTo(index)
            rect = checklist.view.visualRect(index)
            QTest.mouseClick(checklist.view.viewport(), Qt.LeftButton, Qt.NoModifier,
                             QPoint(rect.left() + 80, rect.center().y()))
            return
    raise AssertionError(key)


def test_pair_features_tick_on_the_row_text_and_the_sixth_is_refused(page, qtbot):
    page.set_sources([source()])
    page.set_config({"kind": "pairs"})
    qtbot.wait(10)
    features = page._features
    assert page.config()["features"] == ["dwell_s", "depth_pA", "ecd_pA_s", "snr"]
    assert features.count.text() == "4 of 8 selected · pick 2–5"
    drafts = []
    page.workspaceChanged.connect(lambda: drafts.append(page.config()["features"]))
    _click_text(features, "frac_depth")
    assert page.config()["features"] == ["dwell_s", "depth_pA", "frac_depth", "ecd_pA_s", "snr"]
    assert drafts == [page.config()["features"]], "one draft change per click"
    assert "untick one to choose another" in features.count.text()
    _click_text(features, "rise_s")
    assert "rise_s" not in page.config()["features"], "a sixth feature was accepted"
    held = next(features.model.item(r) for r in range(features.model.rowCount())
                if features.model.item(r).data(Qt.UserRole) == "rise_s")
    assert not (held.flags() & Qt.ItemIsEnabled)
    assert "At most 5" in held.toolTip()
    # The draft keeps METRICS order whatever order the ticks came in.
    order = [key for key, _label in METRICS]
    picked = page.config()["features"]
    assert picked == sorted(picked, key=order.index)


def test_pair_features_restore_and_template_round_trip(page):
    page.set_config({"kind": "pairs", "features": ["snr", "dwell_s"]})
    assert page.config()["features"] == ["dwell_s", "snr"]
    settings, _appearance = page.template_settings()
    other = ComparisonFiguresWindow()
    try:
        other.apply_template_settings(settings, page.appearance_state())
        assert other.config()["features"] == ["dwell_s", "snr"]
        assert other._features.count.text() == "2 of 8 selected · pick 2–5"
    finally:
        other.deleteLater()


def _figure_fits(page):
    view = page._view
    seen = view.mapFromScene(page._figure_item.boundingRect()).boundingRect()
    return view.viewport().rect().adjusted(-1, -1, 1, 1).contains(seen)


def test_a_generated_figure_is_still_whole_when_the_canvas_shrinks(page, qtbot):
    page.set_sources([source()])
    page._workflow.set_section("Data")
    qtbot.wait(20)
    assert page.analyze()
    qtbot.wait(20)
    assert _figure_fits(page)
    tall = page._view.viewport().height()
    page._workflow.set_section("Analysis")     # its controls are taller than Data's
    qtbot.waitUntil(lambda: page._view.viewport().height() < tall - 40, timeout=2000)
    qtbot.wait(20)
    assert _figure_fits(page), "the canvas opened scrolled into the figure"
    assert page._view.verticalScrollBar().value() == page._view.verticalScrollBar().minimum()


def test_a_zoomed_canvas_is_left_alone_by_a_resize(page, qtbot):
    page.set_sources([source()])
    assert page.analyze()
    qtbot.wait(20)
    page._view.zoom(1.4)
    scale = page._view.transform().m11()
    page.resize(1000, 700)
    qtbot.wait(30)
    assert page._view.transform().m11() == scale
    page.fit_canvas()
    page.resize(1100, 800)
    qtbot.wait(30)
    assert _figure_fits(page)


def test_a_reopened_project_opens_at_fit_canvas_in_a_shorter_canvas(page, qtbot):
    """Saved while the figure was fitted (Section ▸ Data, the canvas tall), reopened
    where the canvas is shorter: the figure is fitted again, not scrolled into."""
    page.set_sources([source()])
    page._workflow.set_section("Data")
    qtbot.wait(20)
    assert page.analyze()
    qtbot.wait(20)
    payload, tables = page.snapshot_state()
    assert payload["canvas_view"]["fitted"] is True
    other = ComparisonFiguresWindow()
    qtbot.addWidget(other)
    other.resize(1100, 560)
    other.show()
    qtbot.waitExposed(other)
    other._workflow.set_section("Analysis")
    qtbot.wait(20)
    other.restore_state(payload, tables)
    qtbot.wait(20)
    assert _figure_fits(other), "the reopened figure opened scrolled into itself"
    other.resize(1100, 700)
    qtbot.wait(30)
    assert _figure_fits(other), "a fitted restore keeps fitting on a resize"
