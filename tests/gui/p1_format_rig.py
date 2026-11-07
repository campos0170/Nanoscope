"""Test support for the Format group's tests (G2): pages to select things on, and real clicks.

Not a test module. A page built here is hosted by a real ``AnalysisExplorer`` through
``tests/gui/p1_hosting.py``; a bare ``FormatPane`` over a plot (``bare_format``) is for the tests
that are about a panel and not about the shell. Every click goes through the QWINDOW of the
view's top-level window (``window().windowHandle()``), a pan is a drag with the button held.
Synthetic data only.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.gui.figure_text import (AppearanceTextHost, EditableLabelItem, TextStyle, TextTarget,
                                       apply_text_style, attach_figure_text, figure_text_for, rich_text, HIDDEN_TEXT)
from nanoscope.gui.fit_label import FitReadout
from nanoscope.gui.voltage_dep_panel import _ProvenanceCaption, _MSG_HEX
from nanoscope.gui.format_inspector import ElementRef, FormatPane, SelectionHub
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget, apply_plot_appearance
from nanoscope.gui.plot_workflow import FormatSpec

from . import p1_hosting as H

L = QtCore.Qt.MouseButton.LeftButton
R = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier

CAPTION = "annotation.provenance"
PANEL = "annotation.panel"


# -- real gestures on a plot ----------------------------------------------------------------------

def view_of(plot) -> QtWidgets.QGraphicsView:
    return plot.scene().views()[0]


def window_pos(plot, scene_pt) -> QtCore.QPoint:
    view = view_of(plot)
    return view.viewport().mapTo(view.window(), view.mapFromScene(scene_pt))


def click_scene(plot, scene_pt, button=L, double=False) -> None:
    """A real click (or double click) at ``scene_pt`` through the window's QWindow."""
    handle = H.window_handle(view_of(plot))
    pos = window_pos(plot, scene_pt)
    QtTest.QTest.mouseMove(handle, pos)
    if double:
        QtTest.QTest.mouseDClick(handle, button, NOMOD, pos)
    else:
        QtTest.QTest.mouseClick(handle, button, NOMOD, pos)
    QtWidgets.QApplication.processEvents()


def drag_scene(plot, scene_pt, delta=(70, 21), steps=7) -> None:
    """A pan: press, moves with the left button HELD, release (``QTest.mouseMove`` holds no
    button, so a drag is sent as real ``QMouseEvent``s)."""
    view = view_of(plot)
    handle = H.window_handle(view)
    start = window_pos(plot, scene_pt)

    def send(kind, pos, buttons, button=QtCore.Qt.MouseButton.NoButton):
        event = QtGui.QMouseEvent(kind, QtCore.QPointF(pos), QtCore.QPointF(view.window().mapToGlobal(pos)),
                                  button, buttons, NOMOD)
        QtWidgets.QApplication.sendEvent(handle, event)
        QtWidgets.QApplication.processEvents()

    send(QtCore.QEvent.Type.MouseButtonPress, start, L, L)
    for i in range(1, steps + 1):
        send(QtCore.QEvent.Type.MouseMove,
             start + QtCore.QPoint(delta[0] * i // steps, delta[1] * i // steps), L)
    send(QtCore.QEvent.Type.MouseButtonRelease, start + QtCore.QPoint(*delta),
         QtCore.Qt.MouseButton.NoButton, L)


def centre_of(item) -> QtCore.QPointF:
    return item.sceneBoundingRect().center()


def point_of(plot, key: str) -> QtCore.QPointF:
    """Where a click lands on element ``key`` of ``plot`` (scene coordinates)."""
    controller = figure_text_for(plot)
    if key == "title":
        return centre_of(plot.titleLabel)
    if key.startswith("axis.") and key.endswith(".label"):
        return centre_of(plot.getAxis(key.split(".")[1]).label)
    if key.startswith("axis.") and key.endswith(".ticks"):
        vb = plot.getViewBox().sceneBoundingRect()
        if key.split(".")[1] == "bottom":
            return QtCore.QPointF(vb.center().x() + 30, vb.bottom() + 9)
        return QtCore.QPointF(vb.left() - 14, vb.center().y() + 30)
    if key.startswith("legend.entry."):
        return centre_of(controller.target(key).item)
    if key == "legend":
        legend = plot.legend
        rect = legend.sceneBoundingRect()
        return QtCore.QPointF(rect.center().x(), rect.bottom() - 2)    # the frame, below the rows
    target = controller.target(key)
    return centre_of(target.item)


def pick_row(list_widget, ref: ElementRef) -> None:
    """A real click on the row of ``ref`` in the element list."""
    item = list_widget.find(ref)
    assert item is not None, f"no row for {ref}"
    # The element list is capped at 150 px; later rows (e.g. Caption) need scrolling
    # before their real viewport coordinates can receive a click.
    list_widget.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
    QtWidgets.QApplication.processEvents()
    rect = list_widget.visualItemRect(item)
    assert list_widget.viewport().rect().contains(rect.center()), f"row is offscreen: {ref}"
    H.click(list_widget.viewport(), pos=rect.center())


# -- a page with everything to select -------------------------------------------------------------

class RichPage(H.SyntheticPlotPage):
    """``SyntheticPlotPage`` with a legend of two named series, a drawn fit label and a
    provenance-style caption, so every kind of text has something on the canvas."""

    def __init__(self, key: str = "iv") -> None:
        super().__init__(key)
        plot = self._plot
        plot.setLabel("bottom", "Voltage", units="V")       # shows the axis labels
        plot.setLabel("left", "Current", units="A")
        self._legend = plot.addLegend(offset=(-10, 10))
        self._legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))
        first = plot.listDataItems()[0]
        self._legend.addItem(first, "series")
        self._second = plot.plot([0, 1, 2, 3], [2.0, 1.0, 2.5, 1.5], name="second",
                                 pen=pg.mkPen("#d55e00"))
        # Use the real caption renderer: its bottom-left/custom placement belongs
        # to its state, so restyling or wrapping cannot move it over the fit label.
        self._caption = _ProvenanceCaption(plot.getViewBox())
        # a panel letter is the other annotation class: an EditableLabelItem (no drag)
        self._panel = EditableLabelItem()
        self._panel.setParentItem(plot.getViewBox())
        self._panel.setText("(a)")
        self._panel.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -10))
        self._text_host.extra_targets = lambda: {
            CAPTION: TextTarget(self._caption, "Pooled points", TextStyle(), "caption",
                                frozenset({"edit", "style"})),
            PANEL: TextTarget(self._panel, "(a)", TextStyle(), "panel letter",
                              frozenset({"edit", "style"}))}
        self._text_host.register_annotation(self._caption, CAPTION)
        self._readouts = (FitReadout(parts={"equation": "y = 2x", "r2": "R² = 0.99",
                                            "n": "n = 4"}),)
        self._draw_extras(self._appearance)
        figure_text_for(plot).refresh()

    def _draw_extras(self, state: PlotAppearance) -> None:
        self._fit_label.apply_state(replace(state.fit_label_state(), in_legend=False),
                                    self._readouts, state.text_style("annotation.fit"),
                                    state.font_family)
        style = state.text_style(CAPTION)
        body = ("Pooled points" if style.text is None else
                "" if style.text == HIDDEN_TEXT else rich_text(style.text))
        self._caption.show_caption(body, style, _MSG_HEX,
                                   state.annotation_offset(CAPTION), state.font_family)
        apply_text_style(self._panel, state.text_style(PANEL), "(a)", state.font_family)

    def _appearance_from_user(self, state: PlotAppearance) -> None:
        super()._appearance_from_user(state)
        self._draw_extras(state)
        figure_text_for(self._plot).refresh()

    @property
    def controller(self):
        return figure_text_for(self._plot)

    def type_words(self, title: str = "A typed title") -> None:
        """A title and both axis labels, so each has something to click."""
        widget = self._appearance_widget
        widget.apply_appearance(replace(widget.state(), title=title, x_label="Voltage (V)",
                                        y_label="Current (pA)"))
        widget.clear_history()
        QtWidgets.QApplication.processEvents()


def hosted_rich(qtbot, key: str = "iv", *, size=(1280, 820), title: str | None = "A typed title"):
    page = RichPage(key)
    explorer = H.hosted(qtbot, page, key, size=size)
    if title is not None:
        page.type_words(title)
    return page, explorer


# -- a bare Format pane over one or two plots ------------------------------------------------------

class Bare(QtWidgets.QWidget):
    """A window holding ``plots`` (each a ``PlotWidget`` with its own text host and
    controller, pin views ``views``), one ``PlotAppearanceWidget`` and a ``FormatPane`` over
    them: the Format tab with no Explorer around it."""

    def __init__(self, views=("",), *, appearance: PlotAppearanceWidget | None = None,
                 defaults=None, extras=None, shared_labels: str = "", **widget_options) -> None:
        super().__init__()
        self.widget = appearance or PlotAppearanceWidget(
            PlotAppearance(), shortcut_parent=self, **widget_options)
        self.views = tuple(views)
        self.plots, self.controllers = [], []
        row = QtWidgets.QHBoxLayout(self)
        for view in views:
            plot_widget = pg.PlotWidget()
            plot_widget.setMinimumSize(360, 260)
            plot_widget.getPlotItem().addLegend()
            plot_widget.plot([0, 1, 2], [1.0, 3.0, 2.0], name=f"s{view}")
            plot_widget.getPlotItem().setLabel("bottom", "Voltage", units="V")
            plot_widget.getPlotItem().setLabel("left", "Current", units="A")
            host = AppearanceTextHost(plot_widget.getPlotItem(), self.widget, pin_view=view,
                                      shared_labels=shared_labels)
            controller = attach_figure_text(plot_widget, host)
            self.plots.append(plot_widget.getPlotItem())
            self.controllers.append(controller)
            row.addWidget(plot_widget, 1)
        self.shown = list(self.controllers)
        self.widget.changed.connect(self._applied)
        self.hub = SelectionHub(self)
        spec = FormatSpec(self.widget, controllers=lambda: list(self.shown),
                          extras=extras or {}, defaults=defaults)
        self.pane = FormatPane(spec, self.hub)
        self.pane.setMinimumWidth(320)
        self.pane.setMaximumWidth(360)
        row.addWidget(self.pane)

    def _applied(self, state: PlotAppearance) -> None:
        for plot_item, view in zip(self.plots, self.views):
            apply_plot_appearance(plot_item, state.for_view(view))
        for controller in self.controllers:
            controller.refresh()

    @property
    def state(self) -> PlotAppearance:
        return self.widget.state()


def bare(qtbot, views=("",), **options) -> Bare:
    window = Bare(views, **options)
    qtbot.addWidget(window)
    window.resize(1100, 560)
    window.show()
    qtbot.waitExposed(window)
    window.pane.show_element("figure")
    QtWidgets.QApplication.processEvents()
    return window


def depth(widget: PlotAppearanceWidget) -> int:
    """Undo steps behind the current appearance (an open run of edits counted)."""
    return widget.history_mark()
