"""Plot windows: Reset view, zoom history, the app right-click menu and the pointer readout.

SYNTHETIC ONLY. The helper is exercised on a bare PlotWidget, then installed in every plot
window it serves: Distributions and Distribution summaries standalone, and the rest through a
real MainWindow on two synthetic recordings with detected events. Nothing is read from disk.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401 -- Qt bootstrap before PySide
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plot_appearance import PlotAppearance, apply_plot_appearance
from nanoscope.gui.plot_interaction import (SETTLE_MS, PlotReadout, app_menu_for,
                                            axis_value_text, install_plot_interaction,
                                            navigation_for, readout_text, reset_plot_history,
                                            view_button)

#: pyqtgraph's generic menu entries. None of them may be reachable from a plot window.
GENERIC = ("View All", "X axis", "Y axis", "Mouse Mode", "Plot Options", "Export...",
           "Transforms", "Log X", "Invert")


def _menu_texts(menu: QtWidgets.QMenu) -> list[str]:
    out = []
    for action in menu.actions():
        if action.isSeparator():
            out.append("—")
        else:
            out.append(action.text() + ("" if action.isEnabled() else " (disabled)"))
    return out


def _page_texts(menu: QtWidgets.QMenu) -> list[str]:
    """The page's own entries: without figure_text's "Add title…" ("Add title (all
    views)…" where the views share one title), which leads a plain right-click's menu
    while the title draws nothing (an empty automatic title)."""
    texts = _menu_texts(menu)
    if texts[:1] and texts[0].startswith("Add title") and texts[1:2] == ["—"]:
        return texts[2:]
    return texts


@pytest.fixture
def bare(qapp, qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(500, 400)
    widget.show()
    plot = widget.getPlotItem()
    plot.plot(np.arange(10.0), np.arange(10.0) ** 2)
    qapp.processEvents()
    plot.autoRange()          # framed at the laid-out size, as Reset view will frame it
    exported = []
    nav = install_plot_interaction(plot, exports=(("Export current figure…",
                                                   lambda: exported.append("figure")),))
    nav._exported = exported
    return widget, plot, nav


class _RightClick:
    """A right-button click as pyqtgraph's ``mouseClickEvent`` reads one."""

    def button(self):
        return QtCore.Qt.MouseButton.RightButton

    def accept(self):
        pass

    def screenPos(self):
        return QtCore.QPointF(10, 10)


def _user_zoom(plot, x, y):
    """A user's zoom as pyqtgraph reports one: the range moves, then 'manually'."""
    vb = plot.getViewBox()
    vb.setRange(xRange=x, yRange=y, padding=0)
    vb.sigRangeChangedManually.emit([True, True])


def _settle(qtbot):
    qtbot.wait(SETTLE_MS + 120)


def _x(plot):
    return tuple(round(v, 6) for v in plot.viewRange()[0])


def _framed_like(plot, drawn):
    """The same auto-range framing. pyqtgraph's padding follows the ViewBox width, which
    follows the tick-label width, so two auto-ranges of one plot agree to a pixel or so."""
    lo, hi = plot.viewRange()[0]
    span = drawn[1] - drawn[0]
    return abs(lo - drawn[0]) < 0.02 * span and abs(hi - drawn[1]) < 0.02 * span


# -- the menu ------------------------------------------------------------------------------------
def test_the_app_menu_replaces_pyqtgraphs_generic_menu(bare, monkeypatch):
    _widget, plot, nav = bare
    menu = app_menu_for(plot)
    assert _menu_texts(menu) == ["Reset view", "Zoom back (disabled)",
                                 "Zoom forward (disabled)", "—", "Export current figure…"]
    # The PlotItem's "Plot Options" (log / FFT / invert transforms) and the ViewBox's own
    # entries can no longer be appended by any child item's menu either.
    assert plot.getContextMenus(None) is None
    assert plot.getViewBox().getContextMenus(None) == []
    shown = []
    monkeypatch.setattr(QtWidgets.QMenu, "popup", lambda self, *a: shown.append(_menu_texts(self)))
    plot.getViewBox().mouseClickEvent(_RightClick())
    assert shown and shown[-1][0] == "Reset view"
    assert not any(g in entry for entry in shown[-1] for g in GENERIC)
    next(a for a in menu.actions() if a.text() == "Export current figure…").trigger()
    assert nav._exported == ["figure"]


def test_menu_greys_out_zoom_back_and_forward_when_there_is_nowhere_to_go(bare, qtbot):
    _widget, plot, nav = bare
    _user_zoom(plot, (2, 4), (4, 16))
    _settle(qtbot)
    assert _menu_texts(app_menu_for(plot))[:3] == ["Reset view", "Zoom back",
                                                   "Zoom forward (disabled)"]
    nav.view_back()
    assert _menu_texts(app_menu_for(plot))[:3] == ["Reset view", "Zoom back (disabled)",
                                                   "Zoom forward"]


# -- the history ---------------------------------------------------------------------------------
def test_back_and_forward_step_through_settled_views_and_a_new_zoom_prunes(bare, qtbot):
    _widget, plot, nav = bare
    drawn = _x(plot)
    _user_zoom(plot, (2, 4), (4, 16))
    _settle(qtbot)
    _user_zoom(plot, (3, 3.5), (9, 12))
    _settle(qtbot)
    assert len(nav.history()) == 3
    nav.view_back()
    assert _x(plot) == (2.0, 4.0)
    nav.view_back()
    assert _x(plot) == drawn
    assert not nav.can_go_back()
    nav.view_forward()
    assert _x(plot) == (2.0, 4.0)
    _user_zoom(plot, (5, 6), (25, 36))       # a new place: the forward branch is gone
    _settle(qtbot)
    assert not nav.can_go_forward()
    assert [round(v[0][0], 6) for v in nav.history()] == [drawn[0], 2.0, 5.0]


def test_a_quick_back_flushes_the_unsettled_zoom_first(bare, qtbot):
    _widget, plot, nav = bare
    drawn = _x(plot)
    _user_zoom(plot, (2, 4), (4, 16))        # no settle: Back is pressed at once
    assert nav.can_go_back()
    nav.view_back()
    assert _x(plot) == drawn
    nav.view_forward()                        # the dive was kept, not dropped
    assert _x(plot) == (2.0, 4.0)


def test_the_pages_own_framing_replaces_instead_of_piling_up(bare, qtbot):
    _widget, plot, nav = bare
    vb = plot.getViewBox()
    vb.setRange(xRange=(0, 5), yRange=(0, 25), padding=0)    # e.g. a typed axis pin
    _settle(qtbot)
    vb.setRange(xRange=(0, 6), yRange=(0, 36), padding=0)    # a late auto-range
    _settle(qtbot)
    assert nav.history() == [((0.0, 6.0), (0.0, 36.0))]
    assert not nav.can_go_back()


def test_a_zoom_inside_the_settle_after_a_draw_still_returns_to_the_drawn_view(bare, qtbot):
    _widget, plot, nav = bare
    plot.clear()
    reset_plot_history(plot)
    plot.plot(np.arange(100.0), np.arange(100.0))
    plot.autoRange()                          # the draw's framing, not yet settled
    drawn = _x(plot)
    _user_zoom(plot, (10, 20), (10, 20))
    nav.view_back()
    assert _x(plot) == drawn


def test_new_geometry_resets_the_history_and_appearance_does_not(bare, qtbot):
    _widget, plot, nav = bare
    _user_zoom(plot, (2, 4), (4, 16))
    _settle(qtbot)
    apply_plot_appearance(plot, PlotAppearance(title="Restyled", line_width=3.0))
    _settle(qtbot)
    assert nav.can_go_back() and len(nav.history()) == 2
    assert _x(plot) == (2.0, 4.0)             # appearance never moves the zoom
    plot.clear()
    reset_plot_history(plot)
    plot.plot(np.arange(50.0), np.arange(50.0))
    plot.autoRange()
    _settle(qtbot)
    assert not nav.can_go_back() and len(nav.history()) == 1


def test_reset_view_is_one_step_back_from_the_zoom_it_reset(bare, qtbot):
    _widget, plot, nav = bare
    drawn = _x(plot)
    _user_zoom(plot, (2, 4), (4, 16))
    nav.reset_view()                          # flushes the unsettled zoom first
    assert _framed_like(plot, drawn)
    nav.view_back()
    assert _x(plot) == (2.0, 4.0)


@pytest.fixture
def linked(qapp, qtbot):
    """A plot and an x-linked strip sharing its navigation (Distributions' residual strip)."""
    layout = pg.GraphicsLayoutWidget()
    qtbot.addWidget(layout)
    layout.resize(600, 500)
    layout.show()
    main = layout.addPlot(row=0, col=0)
    strip = layout.addPlot(row=1, col=0)
    strip.setXLink(main)
    main.plot(np.arange(10.0), np.arange(10.0))
    strip.plot(np.arange(10.0), np.sin(np.arange(10.0)))
    qapp.processEvents()
    main.autoRange()
    nav = install_plot_interaction(main)
    install_plot_interaction(strip, navigation=nav)
    _settle(qtbot)
    return layout, main, strip, nav     # the layout too: it owns the plots' C++ objects


def test_a_zoom_on_a_linked_strip_is_a_step_of_the_shared_history(linked, qtbot):
    _layout, main, strip, nav = linked
    drawn = _x(main)
    _user_zoom(strip, (2, 4), (-0.2, 0.2))   # a wheel on the strip moves the plot by its link
    _settle(qtbot)
    zoomed = _x(main)   # lined up with the strip on screen, so within a few px of (2, 4)
    assert zoomed[1] - zoomed[0] < 3.0, zoomed
    assert nav.can_go_back(), "the drawn framing was overwritten instead of kept one step back"
    nav.view_back()
    assert _x(main) == drawn
    nav.view_forward()
    assert _x(main) == zoomed


def test_a_strip_gesture_that_moves_nothing_here_records_no_stale_view(linked, qtbot):
    _layout, main, strip, nav = linked
    main.clear()
    reset_plot_history(main)
    main.plot(np.arange(100.0), np.arange(100.0))
    main.autoRange()                          # a new draw, still settling...
    drawn = main.viewRange()
    qtbot.wait(20)                            # ...when the user's gesture arrives
    _user_zoom(strip, tuple(strip.viewRange()[0]), (-0.5, 0.5))    # y only: the plot stays
    _settle(qtbot)
    assert nav.history() == [tuple(tuple(pair) for pair in drawn)]
    assert not nav.can_go_back()


def test_the_view_button_resets_and_greys_its_drop_down(bare, qtbot):
    _widget, plot, nav = bare
    button = view_button(lambda: nav, (nav,))
    qtbot.addWidget(button)
    assert not button._nanoscope_back.isEnabled()
    drawn = _x(plot)
    _user_zoom(plot, (2, 4), (4, 16))
    _settle(qtbot)
    assert button._nanoscope_back.isEnabled()
    button.click()
    assert _framed_like(plot, drawn)
    button._nanoscope_back.trigger()
    assert _x(plot) == (2.0, 4.0)
    assert button._nanoscope_forward.isEnabled()


# -- the readout ---------------------------------------------------------------------------------
def test_readout_shows_real_values_on_log_axes_and_follows_tick_scale(qapp, qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    plot.setLogMode(x=True, y=False)
    plot.setLabel("bottom", "Frequency (Hz)")
    plot.setLabel("left", "dwell (ms)")
    plot.getAxis("left").setScale(1000.0)     # seconds plotted, milliseconds on the ticks
    text = readout_text(plot, QtCore.QPointF(2.0, 0.0125))
    assert text.startswith("Frequency (Hz) 100")
    assert "dwell (ms) 12.5" in text
    # Units declared on the axis come back with their SI prefix, as the ticks print them.
    plot.getAxis("left").setScale(1.0)
    plot.setLabel("left", "Current", units="A")
    assert axis_value_text(plot.getAxis("left"), 1.5e-9) == "1.5 nA"
    # A page that draws log10 values on a linear axis says so.
    plot.setLogMode(x=False, y=False)
    assert axis_value_text(plot.getAxis("bottom"), -3.0, log=True) == "0.001"


def test_leaving_a_plot_empties_its_readout_without_an_event_filter_on_the_view(qapp, qtbot):
    """The pointer leaving a plot empties the readout through the PlotItem's own hover exit,
    never through a Python event filter on the GraphicsView: that filter ran for every event
    the view got and made the PySide construction segfault about four times as frequent.
    Moving from one plot of a scene to another keeps the new plot's values: pyqtgraph emits
    the move before the old plot's exit, and an exit clears only the plot that owns the line."""
    from pyqtgraph.GraphicsScene.mouseEvents import HoverEvent

    from nanoscope.gui import plot_interaction

    assert plot_interaction._SceneHover.eventFilter is QtCore.QObject.eventFilter
    layout = pg.GraphicsLayoutWidget()
    qtbot.addWidget(layout)
    layout.resize(600, 400)
    top, bottom = layout.addPlot(row=0, col=0), layout.addPlot(row=1, col=0)
    for plot in (top, bottom):
        plot.setXRange(0.0, 1.0, padding=0)
        plot.setYRange(0.0, 1.0, padding=0)
    readout = PlotReadout()
    qtbot.addWidget(readout)
    install_plot_interaction(top, readout=readout)
    install_plot_interaction(bottom, readout=readout)
    layout.show()
    qtbot.waitExposed(layout)
    scene = top.scene()

    def over(plot):
        vb = plot.getViewBox()
        scene.sigMouseMoved.emit(vb.mapViewToScene(QtCore.QPointF(0.5, 0.5)))

    over(top)
    assert readout.full_text() != ""
    over(bottom)                              # the move hands the line to the bottom plot...
    shown = readout.full_text()
    top.hoverEvent(HoverEvent(None, False))   # ...before the top plot's exit arrives
    assert readout.full_text() == shown != ""
    bottom.hoverEvent(HoverEvent(None, False))  # the pointer leaves the view
    assert readout.full_text() == ""


def test_readout_names_categories_and_the_nearest_point(qapp, qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    plot = widget.getPlotItem()
    scatter = pg.ScatterPlotItem(x=[0.0, 1.0], y=[5.0, 7.0], data=[17, 23])
    plot.addItem(scatter)
    plot.getAxis("bottom").setTicks([[(0, "pore A"), (1, "pore B")]])
    plot.setRange(xRange=(-0.6, 1.6), yRange=(0, 10), padding=0)
    qapp.processEvents()
    text = readout_text(plot, QtCore.QPointF(1.02, 7.05),
                        points=lambda: [(scatter, lambda v: f"event_id {v}")])
    assert "pore B" in text
    assert "nearest event_id 23" in text
    far = readout_text(plot, QtCore.QPointF(0.5, 1.0),
                       points=lambda: [(scatter, lambda v: f"event_id {v}")])
    assert "nearest" not in far


def test_the_readout_is_one_fixed_line_that_never_widens_its_host(qapp, qtbot):
    readout = PlotReadout()
    qtbot.addWidget(readout)
    height = readout.height()
    readout.set_readout("x " * 400)
    assert readout.height() == height
    assert readout.sizePolicy().horizontalPolicy() == QtWidgets.QSizePolicy.Ignored
    assert readout.full_text().startswith("x x")


# -- Distributions -------------------------------------------------------------------------------
def _source(key: str, frame: pd.DataFrame) -> PlotSource:
    return PlotSource(key=key, label=key, df=frame, provenance={}, colour="#0072B2",
                      doc_id=f"doc-{key}", channel=0, module_id="detect.threshold",
                      calibration_fingerprint=(1.0, 0.0))


@pytest.fixture
def plots(qapp, qtbot):
    from nanoscope.gui.plots_panel import PlotsWindow

    frame = pd.DataFrame({"event_id": [17, 23, 41, 59],
                          "dwell_s": [0.001, 0.002, 0.004, 0.008],
                          "depth_pA": [100.0, 150.0, 220.0, 300.0],
                          "accepted": [True, True, True, True]})
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("A", frame)])
    w.resize(1150, 750)
    w.show()
    qapp.processEvents()
    return w


def test_distributions_reset_view_keeps_the_typed_pin_and_never_touches_config(plots, qtbot):
    w = plots
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_scatter_log(False, False)
    w._range_edits["ymin"].setText("50")
    w._range_edits["ymax"].setText("400")
    w._apply_axis_range()
    _settle(qtbot)
    config, depth = w.config(), w._history.depth()
    _user_zoom(w._plot, (0.0015, 0.003), (120, 200))
    _settle(qtbot)
    w._view_button.click()                     # the toolbar's Reset view
    assert tuple(round(v, 6) for v in w._plot.viewRange()[1]) == (50.0, 400.0)
    assert w.config() == config and w._history.depth() == depth
    assert not w._stale
    w._plot_nav.view_back()
    assert tuple(round(v, 6) for v in w._plot.viewRange()[1]) == (120.0, 200.0)
    assert w.config() == config and w._history.depth() == depth


def test_distributions_menu_and_readout_name_the_nearest_event(plots, qapp):
    w = plots
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_scatter_log(False, False)
    qapp.processEvents()
    texts = _page_texts(app_menu_for(w._plot))
    assert texts[0] == "Reset view" and "Export current figure…" in texts
    assert "Export plot data…" in texts
    assert not any(g in t for t in texts for g in GENERIC)
    assert app_menu_for(w._resid) is not None and navigation_for(w._resid) is w._plot_nav
    # Time unit ms (the default) scales the dwell ticks by 1000; the readout reads the same
    # number the ticks do, and names the point a click there would select.
    text = readout_text(w._plot, QtCore.QPointF(0.004, 220.0), points=w._readout_points)
    assert text.startswith("dwell (ms) 4   ·   "), text
    assert text.endswith("nearest A event_id 41 (4, 220)"), text
    # On a log axis the view holds log10(seconds); the readout still says 4 ms.
    w.set_scatter_log(True, False)
    qapp.processEvents()
    text = readout_text(w._plot, QtCore.QPointF(math.log10(0.004), 220.0),
                        points=w._readout_points)
    assert text.startswith("dwell (ms, log) 4   ·   "), text
    assert "nearest A event_id 41 (4, 220)" in text


def test_distributions_redraw_resets_history_but_restyling_does_not(plots, qtbot):
    w = plots
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    _settle(qtbot)
    _user_zoom(w._plot, (0.0015, 0.003), (120, 200))
    _settle(qtbot)
    assert w._plot_nav.can_go_back()
    w._appearance_from_user(PlotAppearance(title="Restyled"))
    _settle(qtbot)
    assert w._plot_nav.can_go_back()
    assert tuple(round(v, 6) for v in w._plot.viewRange()[1]) == (120.0, 200.0)
    w.set_plot_type("Histogram")              # a different plot type: new geometry
    _settle(qtbot)
    assert not w._plot_nav.can_go_back()


def test_distributions_hover_names_the_nearest_event_on_the_status_line(plots, qapp):
    """Through the scene's real mouse-move signal, not readout_text: the install's
    ``points=`` wiring is what puts the event_id on the status line."""
    w = plots
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_scatter_log(False, False)
    qapp.processEvents()
    vb = w._plot.getViewBox()
    w._plot.scene().sigMouseMoved.emit(vb.mapViewToScene(QtCore.QPointF(0.004, 220.0)))
    text = w._readout.full_text()
    assert text.startswith("dwell (ms) 4"), text
    assert text.endswith("nearest A event_id 41 (4, 220)"), text


def test_distributions_stale_mark_redraw_keeps_the_zoom_and_its_history(plots, qtbot,
                                                                       monkeypatch):
    """A moved mark redraws the same table and puts the zoom back; the places you zoomed
    to still mean what they meant, so the history survives the redraw's clear."""
    w = plots
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    _settle(qtbot)
    _user_zoom(w._plot, (0.0015, 0.003), (120, 200))
    _settle(qtbot)
    _user_zoom(w._plot, (0.0018, 0.0025), (130, 160))
    _settle(qtbot)
    history = w._plot_nav.history()
    monkeypatch.setattr(w, "_stale_signature", lambda: ("a mark moved",))
    w._redraw_stale_marks()
    _settle(qtbot)
    assert tuple(round(v, 6) for v in w._plot.viewRange()[1]) == (130.0, 160.0)
    assert w._plot_nav.history() == history and w._plot_nav.position() == 2
    w._plot_nav.view_back()
    assert tuple(round(v, 6) for v in w._plot.viewRange()[1]) == (120.0, 200.0)


def _drain_notes(w, qtbot, timeout_s: float = 240.0) -> None:
    """Wait for Distributions' deferred notes: their arrival redraws, which would restart
    the history in the middle of a test (`test_plots_axes_and_fit_stats._drain`)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qtbot.wait(10)
        if not w._pending_notes and not w._note_timer.isActive() and not w._inflight_notes:
            qtbot.wait(10)
            return
    raise AssertionError("the deferred notes never settled")


def test_distributions_residual_strip_zoom_is_a_step_and_reset_view_reframes_it(qapp, qtbot):
    from nanoscope.gui.plots_panel import PlotsWindow

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"event_id": np.arange(600),
                          "dwell_s": 2e-5 + rng.exponential(2.94e-4, 600),
                          "depth_pA": np.abs(rng.normal(40.0, 6.0, 600)),
                          "accepted": np.ones(600, dtype=bool)})
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("A", frame)])
    w.resize(1150, 750)
    w.show()
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    w._style = replace(w._style, show_residuals=True)
    w._redraw()
    _drain_notes(w, qtbot)
    _settle(qtbot)
    assert not w._resid_widget.isHidden() and w._resid.items
    span = w._resid_view_span
    drawn = tuple(w._plot.viewRange()[0])
    lo, hi = drawn
    _user_zoom(w._resid, (lo + 0.25 * (hi - lo), lo + 0.5 * (hi - lo)), (-1.0, 1.0))
    _settle(qtbot)                            # a zoom made on the strip, x-linked to the plot
    zoomed = _x(w._plot)
    assert zoomed[1] - zoomed[0] < 0.5 * (hi - lo), zoomed
    assert w._plot_nav.can_go_back()
    assert _menu_texts(app_menu_for(w._resid))[0] == "Reset view"
    app_menu_for(w._resid).actions()[0].trigger()          # the strip's own Reset view
    assert np.allclose(w._resid.viewRange()[1], (-span, span))
    assert _framed_like(w._plot, drawn)
    w._plot_nav.view_back()
    assert _x(w._plot) == zoomed


def test_distributions_readout_lives_on_the_status_line_outside_the_figure(plots, qapp):
    w = plots
    readout = w._readout
    assert readout.parent() is w.statusBar()
    scene_widgets = {item.widget() for item in w._plot.scene().items()
                     if isinstance(item, QtWidgets.QGraphicsProxyWidget)}
    assert readout not in scene_widgets
    assert w.figure_export_target() is w._plot


# -- Distribution summaries ----------------------------------------------------------------------
def _summaries(qtbot):
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow

    page = DistributionSummaryWindow()
    qtbot.addWidget(page)
    page.resize(1000, 740)
    page.show()
    page.set_sources([PlotSource("measurements", "Measured events",
                                 pd.DataFrame({"dwell_s": [1., 2., 3., 4., 9.]}),
                                 kind="manual", doc_key="measurements", current=True)])
    return page


def test_distribution_summaries_reset_view_on_an_ecdf_keeps_the_zoom_one_step_back(qapp,
                                                                                   qtbot):
    """The ECDF framing only ENABLES auto-range, which pyqtgraph applies at the next paint.
    Reset view must fit at once, or the late fit replaces the zoom's entry and Back loses it."""
    page = _summaries(qtbot)
    page.set_config({**page.config(), "kind": "ecdf"})
    page.analyze()
    _settle(qtbot)
    _user_zoom(page._plot, (2.0, 3.0), (0.3, 0.5))
    _settle(qtbot)
    page._plot_nav.reset_view()
    reset = np.asarray(page._plot.viewRange())
    assert reset[0][0] < 1.0 and reset[0][1] > 9.0, reset   # framed now, not at the next paint
    _settle(qtbot)
    page._plot_nav.view_back()
    assert _x(page._plot) == (2.0, 3.0)
    page._plot_nav.view_forward()
    assert np.allclose(page._plot.viewRange(), reset)
    page.analyze()                            # Generate: new geometry, a new history
    _settle(qtbot)
    assert not page._plot_nav.can_go_back()


def test_distribution_summaries_reset_view_keeps_the_category_slots(qapp, qtbot):
    page = _summaries(qtbot)
    page.set_config({**page.config(), "kind": "box"})
    page.analyze()
    qapp.processEvents()
    framed = tuple(round(v, 6) for v in page._plot.viewRange()[0])
    assert framed == (-0.6, 0.6)
    _user_zoom(page._plot, (-0.1, 0.1), (2, 3))
    page._plot_nav.reset_view()
    assert tuple(round(v, 6) for v in page._plot.viewRange()[0]) == framed
    text = readout_text(page._plot, QtCore.QPointF(0.05, 2.5))
    assert "Measured events" in text
    texts = _menu_texts(app_menu_for(page._plot))
    assert "Export current figure…" in texts and "Export plotted coordinates CSV…" in texts


# -- every other plot window, inside a real MainWindow -------------------------------------------
@pytest.fixture
def main_window(qapp, qtbot, monkeypatch):
    from nanoscope.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self, *a, **k: None,
                        raising=False)
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    yield w
    w._analysis_controller.abort()


def _events_doc(tmp_path, name, seed):
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

    rng = np.random.default_rng(seed)
    n = 400_000
    y = -1000.0 + rng.normal(0.0, 8.0, n)
    starts = np.sort(rng.choice(np.arange(5_000, n - 5_000, 6_000), 40, replace=False))
    for s in starts:
        y[s:s + int(rng.integers(40, 1200))] += rng.choice([250.0, 400.0])
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=n,
                         sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(y.astype(np.float32),))


def _update_managed_page(w, page):
    """Commit the chosen page through its actual shared Update plot button."""
    explorer = w._analysis_explorer
    key = next(key for key in explorer.page_ids() if explorer.page(key) is page)
    assert explorer.raise_page(key)
    assert explorer.toolbar().update_button.isEnabled()
    explorer.toolbar().update_button.click()


def _open_every_plot_window(w, qtbot, tmp_path):
    """Two synthetic recordings with detected events, and every MainWindow plot page open."""
    from nanoscope.analysis.modules import register_all
    from nanoscope.gui import overlays

    register_all()
    overlays.register_all_renderers()
    for i, name in enumerate(("poreA.dat", "poreB.dat")):
        did = w._session.add_document(_events_doc(tmp_path, name, seed=i + 5))
        w._create_default_layers(w._session.document(did))
        w._on_selection(did, 0)
        qtbot.wait(30)
        panel = w._analysis_panel
        with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
            assert panel._run_detection(panel.detection_params())
        qtbot.wait(60)
    for opener in ("_open_psd", "_open_iv", "_open_vdep", "_open_event_plots",
                   "_open_subpopulations_view"):
        getattr(w, opener)()
        # Managed openers show a staged figure. Commit its inputs before testing
        # the geometry/history; the Subpopulations opener remains a report viewer.
        if opener != "_open_subpopulations_view":
            explorer = w._analysis_explorer
            _update_managed_page(w, explorer.page(explorer.current_page_id()))
        qtbot.wait(300)
    return (w._psd_window, w._iv_window, w._vdep_window, w._event_plot_window,
            w._subpop_window)


def test_every_plot_window_has_reset_view_history_menu_and_readout(main_window, qtbot,
                                                                     tmp_path):
    w = main_window
    psd, iv, vdep, waves, subpop = _open_every_plot_window(w, qtbot, tmp_path)
    plots = [psd._plot, psd._fft_plot, psd._spec_plot, psd._hist_plot, iv._plot, vdep._plot,
             waves._density_main, waves._density_top, waves._density_right,
             waves._ensemble_plot, subpop.embedding_view()._plot.getPlotItem(),
             subpop.waveform_view()._plot.getPlotItem()]
    for plot in plots:
        texts = _page_texts(app_menu_for(plot))
        assert texts[:1] == ["Reset view"], texts
        assert not any(g in t for t in texts for g in GENERIC), texts
        assert plot.getContextMenus(None) is None
        assert plot.getViewBox().getContextMenus(None) == []
    for page in (psd, iv, vdep, waves):
        readout = page._readout
        assert isinstance(readout, PlotReadout) and readout.parent() is page.statusBar()
    assert isinstance(subpop._readout, PlotReadout)

    # Zoom, step back, reset: the spectrum, the I-V scatter and the waveform ensemble.
    for plot in (psd._plot, iv._plot, waves._ensemble_plot):
        nav = navigation_for(plot)
        _settle(qtbot)
        drawn = np.asarray(plot.viewRange())
        lo, hi = drawn[0]
        _user_zoom(plot, (lo + 0.25 * (hi - lo), lo + 0.5 * (hi - lo)), tuple(drawn[1]))
        _settle(qtbot)
        nav.reset_view()
        reset = np.asarray(plot.viewRange())
        assert reset[0][0] < lo + 0.25 * (hi - lo) and reset[0][1] > lo + 0.5 * (hi - lo)
        if all(isinstance(item, pg.TextItem) for item in plot.items):
            assert np.allclose(reset, drawn)  # a message frame: back to its own 0..1 frame
        else:
            plot.autoRange()  # Reset view on these pages IS the draw's auto-range
            assert np.allclose(plot.viewRange(), reset, rtol=1e-6, atol=1e-9)
        nav.view_back()
        assert np.isclose(plot.viewRange()[0][0], lo + 0.25 * (hi - lo))

    # Hovering fills the window's status line: the PSD reads real frequencies off its
    # log axis, and the figure the page exports never contains that line.
    psd._plot_widget.resize(700, 500)
    qtbot.wait(50)
    vb = psd._plot.getViewBox()
    (x0, x1), (y0, y1) = psd._plot.viewRange()
    view = QtCore.QPointF(0.5 * (x0 + x1), 0.5 * (y0 + y1))
    psd._plot.scene().sigMouseMoved.emit(vb.mapViewToScene(view))
    text = psd._readout.full_text()
    assert psd._plot.getAxis("bottom").logMode
    assert text.startswith(f"Frequency (Hz) {10 ** view.x():.5g}"), text
    assert psd._readout not in {item.widget() for item in psd._plot.scene().items()
                                if isinstance(item, QtWidgets.QGraphicsProxyWidget)}

    # The density map draws log10 values on a linear axis: the readout shows the real
    # value, names the metric without "(log10)", and adds the event count under the pointer.
    w._open_event_plots()
    _settle(qtbot)
    before = np.asarray(waves._density_main.viewRange())
    history = navigation_for(waves._density_main).history()
    waves._logx.setChecked(True)
    assert np.allclose(waves._density_main.viewRange(), before)
    assert navigation_for(waves._density_main).history() == history
    _update_managed_page(w, waves)
    qtbot.wait(100)
    main = waves._density_main
    (x0, x1), (y0, y1) = main.viewRange()
    view = QtCore.QPointF(0.5 * (x0 + x1), 0.5 * (y0 + y1))
    main.scene().sigMouseMoved.emit(main.getViewBox().mapViewToScene(view))
    text = waves._readout.full_text()
    name = waves._metric_label(waves._x_combo.currentData())
    assert waves._x_combo.currentData() == "dwell_s" and name == "Dwell (ms)"
    # The geometry remains log10(seconds); ticks and readout explicitly present ms.
    assert text.startswith(f"{name} (log) {1000.0 * 10 ** view.x():.5g}"), text
    assert "(log10)" not in text and "Event count " in text   # as the colour bar names it
    main.scene().sigMouseMoved.emit(QtCore.QPointF(-50, -50))   # off every plot
    assert waves._readout.full_text() == ""


def _zoom_into(plot):
    """A user zoom into the middle of whatever the page drew."""
    (x0, x1), (y0, y1) = plot.viewRange()
    _user_zoom(plot, (x0 + 0.3 * (x1 - x0), x0 + 0.5 * (x1 - x0)), (y0, y1))


def test_each_window_restarts_its_history_on_new_geometry_and_keeps_it_on_appearance(
        main_window, qtbot, tmp_path, monkeypatch):
    """The ``reset_plot_history`` wiring of every page: its own redraw (new geometry) starts
    the history over; an appearance edit keeps both the history and the zoom."""
    from nanoscope.gui.psd_panel import VIEW_SPECTROGRAM

    psd, iv, vdep, waves, subpop = _open_every_plot_window(main_window, qtbot, tmp_path)
    def restyle_page(page):
        # Use the appearance widget's edit path so its authoritative quantity/style
        # metadata accompanies the values delivered to the page's changed slot.
        widget = page._appearance_widget
        widget._apply(replace(widget.state(), title="Restyled", line_width=3.0))
    def redraw_density():
        before = np.asarray(waves._density_main.viewRange())
        history = navigation_for(waves._density_main).history()
        waves._logy.toggle()
        assert np.allclose(waves._density_main.viewRange(), before)
        assert navigation_for(waves._density_main).history() == history
        _update_managed_page(main_window, waves)
    cases = [
        ("Noise spectrum", psd._plot, lambda: restyle_page(psd),
         psd._recompute),
        ("I-V", iv._plot, lambda: restyle_page(iv), iv._refresh_from_user),
        ("Voltage dependence", vdep._plot, lambda: restyle_page(vdep),
         vdep._refresh_from_user),
        ("waveform ensemble", waves._ensemble_plot,
         lambda: restyle_page(waves), waves._redraw_ensemble),
        ("density map", waves._density_main, lambda: restyle_page(waves),
         redraw_density),
    ]
    for view in (subpop.embedding_view(), subpop.waveform_view()):
        cases.append((type(view).__name__, view._plot.getPlotItem(),
                      lambda: restyle_page(subpop),
                      lambda view=view: view.set_report(view._report)))
    for name, plot, appearance, redraw in cases:
        nav = navigation_for(plot)
        _settle(qtbot)
        _zoom_into(plot)
        _settle(qtbot)
        assert nav.can_go_back(), name
        zoomed = np.asarray(plot.viewRange())
        appearance()
        _settle(qtbot)
        assert nav.can_go_back(), f"{name}: an appearance edit dropped the zoom history"
        assert np.allclose(plot.viewRange(), zoomed), f"{name}: an appearance edit moved the zoom"
        redraw()
        _settle(qtbot)
        qtbot.wait(150)
        assert not nav.can_go_back() and len(nav.history()) == 1, \
            f"{name}: a redraw kept ranges from the previous geometry"

    # The readout names an axis as it is drawn: an Appearance label typed over the metric's
    # name is the readout's name too.
    w = main_window
    w._open_event_plots()
    waves._d_xlabel.setText("My dwell axis")
    qtbot.wait(50)
    main = waves._density_main
    (x0, x1), (y0, y1) = main.viewRange()
    main.scene().sigMouseMoved.emit(main.getViewBox().mapViewToScene(
        QtCore.QPointF(0.5 * (x0 + x1), 0.5 * (y0 + y1))))
    assert waves._readout.full_text().startswith("My dwell axis "), waves._readout.full_text()

    # Colour bars offer no colour-map menu: it would recolour behind the saved colormap.
    shown = []
    monkeypatch.setattr(QtWidgets.QMenu, "popup", lambda self, *a: shown.append(type(self)))
    waves._colorbar.mouseClickEvent(_RightClick())
    w._open_psd()
    psd.set_view(VIEW_SPECTROGRAM)
    _update_managed_page(w, psd)
    qtbot.wait(300)
    assert psd._spec_bar is not None
    psd._spec_bar.mouseClickEvent(_RightClick())
    assert shown == []
