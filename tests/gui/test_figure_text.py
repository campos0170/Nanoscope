"""In-figure text editing (``gui.figure_text``): routing, the inline editor, the menu,
Text style, Axis properties and the crash guard. Synthetic data only.

Clicks go through the QWINDOW (``widget.window().windowHandle()``) or are delivered to the
viewport as the platform would: a ``QTest.mouseDClick`` sent to the viewport WIDGET never
reaches pyqtgraph's items (plotux PLAN §1, D1).
"""
from __future__ import annotations

import json
import re
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.gui import figure_text
from nanoscope.gui.axes import ReadableLinearAxis
from nanoscope.gui.figure_text import (AxisState, InlineTextEditor, TextStyle,
                                       appearance_text_host, figure_text_for)
from nanoscope.gui.fit_label import FitLabelItem, FitReadout
from nanoscope.gui.plot_appearance import (PlotAppearance, PlotAppearanceWidget,
                                           apply_plot_appearance)
from nanoscope.gui.plot_interaction import app_menu_for, install_plot_interaction, navigation_for
from nanoscope.plot.pg_svg import export_svg

L = QtCore.Qt.MouseButton.LeftButton
R = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
QT5 = ("press", "release", "dbl", "release")
QT6 = ("press", "release", "press", "dbl", "release")


@pytest.fixture(autouse=True)
def _fresh_gate():
    # The gate remembers the last click per element for one double-click interval; the
    # tests click faster than that, so each starts clean.
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


class _Window(QtWidgets.QWidget):
    """The smallest PlotAppearance window: a plot, its appearance widget, the app menu and
    the generic text host, wired as a plot page wires them."""

    def __init__(self, axis_items=None, log_hook=None, title="Automatic title"):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.view = pg.PlotWidget(background="w", axisItems=axis_items)
        self.plot = self.view.getPlotItem()
        self.plot.setTitle(title or None)
        self.plot.setLabel("bottom", "Dwell", units="s")
        self.plot.setLabel("left", "Depth (pA)")
        self.legend = self.plot.addLegend(offset=(-12, 12))
        rng = np.random.default_rng(3)
        x = rng.uniform(0.1, 5, 60)
        self.scatter = pg.ScatterPlotItem(x, 2 * x + 1, brush="#0072b2", size=6, name="Apo")
        self.plot.addItem(self.scatter)
        self.appearance = PlotAppearanceWidget()
        self.appearance.changed.connect(self.redraw)
        layout.addWidget(self.view, 3)
        layout.addWidget(self.appearance, 1)
        hooks = {"log": log_hook} if log_hook else None
        self.host = appearance_text_host(self.plot, self.appearance, axis_hooks=hooks)
        self.navigation = install_plot_interaction(
            self.plot, text=self.host,
            exports=[("Export current figure…", lambda: None)])
        self.fit = FitLabelItem(self.plot.getViewBox())
        self.fit.apply_state(self.appearance.state().fit_label_state(),
                             [FitReadout({"equation": "y = 2·x + 1", "r2": "R² = 0.99",
                                          "n": "n = 60"})])
        self.host.register_fit_label(self.fit)
        self.text_item = figure_text.EditableLabelItem("caption", color="#444")
        self.text_item.setParentItem(self.plot.getViewBox())
        self.text_item.anchor(itemPos=(1, 1), parentPos=(1, 1), offset=(-10, -10))
        self.host.extra_targets = lambda: {"annotation.provenance": self.text_item}
        self.controller = figure_text_for(self.plot)
        self.controller.refresh()
        self.resize(1000, 520)

    def redraw(self, state):
        apply_plot_appearance(self.plot, state, plot_widget=self.view)
        figure_text.apply_text_style(self.text_item, state.text_style("annotation.provenance"),
                                     "caption")
        self.fit.apply_state(state.fit_label_state(), self.fit._readouts,
                             state.text_style("annotation.fit"))


@pytest.fixture
def window(qtbot):
    w = _Window()
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    QtWidgets.QApplication.processEvents()
    return w


def _send(view, scene_pt, kinds, button=L):
    """A mouse sequence delivered to the viewport exactly as Qt would (no QTest timing)."""
    vp = view.viewport()
    pos = view.mapFromScene(scene_pt)
    QtTest.QTest.mouseMove(vp, pos)
    gpos = QtCore.QPointF(vp.mapToGlobal(pos))
    lpos = QtCore.QPointF(pos)
    for kind in kinds:
        t = {"press": QtCore.QEvent.Type.MouseButtonPress,
             "release": QtCore.QEvent.Type.MouseButtonRelease,
             "dbl": QtCore.QEvent.Type.MouseButtonDblClick}[kind]
        held = button if kind in ("press", "dbl") else QtCore.Qt.MouseButton.NoButton
        QtWidgets.QApplication.sendEvent(vp, QtGui.QMouseEvent(t, lpos, lpos, gpos, button, held, NOMOD))
        QtWidgets.QApplication.processEvents()


def _platform_double(view, scene_pt):
    vp = view.viewport()
    pos = view.mapFromScene(scene_pt)
    QtTest.QTest.mouseMove(vp, pos)
    QtTest.QTest.mouseDClick(view.window().windowHandle(), L, NOMOD, vp.mapTo(view.window(), pos))
    QtWidgets.QApplication.processEvents()


def _centre(item):
    return item.sceneBoundingRect().center()


def _targets(w):
    plot = w.plot
    return {
        "title": _centre(plot.titleLabel.item),
        "axis.left.label": _centre(plot.getAxis("left").label),
        "axis.bottom.ticks": plot.getAxis("bottom").sceneBoundingRect().center() + QtCore.QPointF(0, -8),
        "legend.entry.Apo": _centre(w.legend.items[0][1]),
        "annotation.fit": _centre(w.fit.item),
        "annotation.provenance": _centre(w.text_item.item),
    }


def _drop_editors(w):
    w.controller.close_editors()
    QtWidgets.QApplication.processEvents()


def _window_pos(view, scene_pt):
    pos = view.mapFromScene(scene_pt)
    return view.viewport().mapTo(view.window(), pos)


def _right_click(view, scene_pt):
    """A right click through the QWindow, as the platform delivers it."""
    handle = view.window().windowHandle()
    pos = _window_pos(view, scene_pt)
    QtTest.QTest.mouseMove(handle, pos)
    QtTest.QTest.mouseClick(handle, R, NOMOD, pos)
    QtWidgets.QApplication.processEvents()


def _names(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def _choose(menu, text):
    """A real click on a menu entry."""
    action = next(a for a in menu.actions() if a.text() == text)
    QtTest.QTest.mouseClick(menu, L, NOMOD, menu.actionGeometry(action).center())
    QtWidgets.QApplication.processEvents()


def _tick_point(plot, side):
    """On the tick NUMBERS of ``side``, clear of the axis label."""
    vb = plot.getViewBox().sceneBoundingRect()
    if side == "bottom":
        return QtCore.QPointF(vb.center().x() + 30, vb.bottom() + 9)
    return QtCore.QPointF(vb.left() - 14, vb.center().y() + 30)


def _background_point(plot):
    """Inside the ViewBox, away from the data, the legend and the caption."""
    vb = plot.getViewBox().sceneBoundingRect()
    return QtCore.QPointF(vb.left() + 0.6 * vb.width(), vb.top() + 0.9 * vb.height())


def _press(widget, sequence):
    """A key through the QWindow (the platform's Undo is Ctrl+Z offscreen, ⌘Z on a Mac)."""
    combo = QtGui.QKeySequence(sequence)[0]
    QtTest.QTest.keyClick(widget.window().windowHandle(), combo.key(), combo.keyboardModifiers())
    QtWidgets.QApplication.processEvents()


@pytest.mark.parametrize("sequence", ["QT5", "QT6", "Platform"])
def test_a_double_click_on_each_text_element_opens_exactly_one_editor(window, sequence):
    opened = []
    window.controller.opened.connect(lambda kind, key: opened.append((kind, key)))
    expected_kind = {"axis.bottom.ticks": "axis", "annotation.provenance": "inline"}
    for key, point in _targets(window).items():
        opened.clear()
        figure_text.GATE.reset()
        if sequence == "Platform":
            _platform_double(window.view, point)
        else:
            _send(window.view, point, QT5 if sequence == "QT5" else QT6)
        assert opened == [(expected_kind.get(key, "inline"), key)], (sequence, key, opened)
        _drop_editors(window)


def test_typed_text_is_literal_and_braces_make_sub_and_superscripts(window):
    """Plan G2 test 1. Before: the title drew as "I 1 ms" and the x label as "a c"."""
    state = PlotAppearance(title="I < 5 pA & τ > 1 ms", x_label="a<b> c")
    window.appearance.apply_appearance(state)
    assert window.plot.titleLabel.item.toPlainText() == "I < 5 pA & τ > 1 ms"
    assert window.plot.getAxis("bottom").label.toPlainText().startswith("a<b> c")
    window.appearance.apply_appearance(PlotAppearance(title="x_{0} and y^{2}"))
    title = window.plot.titleLabel.item
    assert title.toPlainText() == "x0 and y2"
    assert "vertical-align:sub" in title.toHtml() and "vertical-align:super" in title.toHtml()


def test_inline_edit_is_one_undo_step_and_follows_the_appearance_rules(window):
    """Plan G2 test 3: Return commits one step, Undo restores, empty is automatic, a space
    hides the title with no gap."""
    _send(window.view, _centre(window.plot.titleLabel.item), QT5)
    editor = window.controller.editor
    assert isinstance(editor, InlineTextEditor) and editor.text() == "Automatic title"
    editor.selectAll()
    editor.insert("Fig 2")
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert window.appearance.state().title == "Fig 2"
    assert window.plot.titleLabel.item.toPlainText() == "Fig 2"
    assert window.appearance._history.depth() == (1, 0)
    assert window.appearance.undo()
    assert window.plot.titleLabel.item.toPlainText() == "Automatic title"
    # Escape changes nothing
    _send(window.view, _centre(window.plot.titleLabel.item), QT5)
    QtTest.QTest.keyClicks(window.controller.editor, "zzz")
    QtTest.QTest.keyClick(window.controller.editor, QtCore.Qt.Key.Key_Escape)
    QtWidgets.QApplication.processEvents()
    assert window.appearance.state().title is None
    # a space hides it, with no band left where it was
    window.controller._set("title", TextStyle(text=" "))
    assert not window.plot.titleLabel.isVisible()
    assert window.plot.layout.rowMinimumHeight(0) == 0
    window.controller._set("title", TextStyle(text=None))
    assert window.plot.titleLabel.item.toPlainText() == "Automatic title"


def test_right_click_on_the_title_puts_its_entries_on_top_of_the_app_menu(window):
    """Plan G2 test 4: nothing of pyqtgraph's own menu, the page's entries kept below."""
    _send(window.view, _centre(window.plot.titleLabel.item), ("press", "release"), button=R)
    menu = window.controller.last_menu
    assert menu is not None and menu.objectName() == "plotAppContextMenu"
    names = [a.text() for a in menu.actions() if not a.isSeparator()]
    menu.close()
    assert names == ["Edit title…", "Text style…", "Automatic title", "Hide title",
                     "Reset view", "Zoom back", "Zoom forward", "Export current figure…"]
    plain = [a.text() for a in app_menu_for(window.plot).actions() if not a.isSeparator()]
    assert plain[0] == "Reset view", "a plain right-click shows no text entries"
    axis = [a.text() for a in app_menu_for(window.plot, "axis.left.label").actions()]
    assert "Axis properties…" in axis


def test_text_style_survives_apply_a_project_round_trip_and_svg_export(window, tmp_path):
    """Plan G2 test 5: bold and colour on the y label."""
    popover = window.controller.open_style("axis.left.label")
    popover.bold.setChecked(True)
    popover.color.set_value("#d55e00")
    popover.accept()
    QtWidgets.QApplication.processEvents()
    state = window.appearance.state()
    assert state.text_style("axis.left.label") == TextStyle(bold=True, color="#d55e00")
    assert window.appearance._history.depth() == (1, 0)
    css = window.plot.getAxis("left").labelStyle
    assert css["font-weight"] == "bold" and css["color"] == "#d55e00"
    assert PlotAppearance.from_mapping(state.to_dict()) == state
    svg = export_svg(window.plot, tmp_path / "styled.svg").read_text()
    label = next(m.group(0) for m in re.finditer(r"<text[^>]*>[^<]*</text>", svg)
                 if "Depth" in m.group(0))
    assert 'fill="#d55e00"' in label and 'font-weight="700"' in label


def test_axis_properties_pin_the_range_reset_view_returns_to_it_and_log_is_explained(window):
    """Plan G2 test 6, first and third parts."""
    popover = window.controller.open_axis_properties("bottom")
    assert not popover.log.isEnabled() and "no log scale" in popover.log.toolTip()
    popover.auto.setChecked(False)
    popover.lo.setText("0.5")
    popover.hi.setText("4")
    popover.accept()
    QtWidgets.QApplication.processEvents()
    lo, hi = window.plot.getViewBox().viewRange()[0]
    assert (lo, hi) == pytest.approx((0.5, 4))
    window.plot.getViewBox().setXRange(1, 2, padding=0)       # the user zooms
    navigation_for(window.plot).reset_view()
    assert window.plot.getViewBox().viewRange()[0] == pytest.approx([0.5, 4])
    assert "x axis range 0.5–4" in window.appearance.undo_button.toolTip()
    window.appearance.undo()
    navigation_for(window.plot).reset_view()
    assert window.plot.getViewBox().viewRange()[0] != pytest.approx([0.5, 4])


def test_major_tick_spacing_is_honoured_by_the_readable_axis(qtbot):
    """Plan G2 test 6, second part: ReadableLinearAxis chooses its own ticks otherwise."""
    w = _Window(axis_items={"bottom": ReadableLinearAxis(orientation="bottom"),
                            "left": ReadableLinearAxis(orientation="left")})
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    w.host.apply_axis("bottom", AxisState(range=(0.0, 4.0), tick_major=0.5))
    axis = w.plot.getAxis("bottom")
    major = axis.tickValues(0.0, 4.0, 600)[0]
    assert major[0] == pytest.approx(0.5)
    assert np.allclose(np.diff(major[1]), 0.5)
    w.host.apply_axis("bottom", AxisState(range=(0.0, 4.0), tick_major=None))
    assert axis.tickValues(0.0, 4.0, 600)[0][0] != pytest.approx(0.5)


def test_axis_properties_log_goes_to_the_windows_own_control(qtbot):
    calls = []
    w = _Window(log_hook=lambda side, on: calls.append((side, on)))
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    popover = w.controller.open_axis_properties("left")
    assert popover.log.isEnabled()
    popover.log.setChecked(True)
    popover.accept()
    assert calls == [("left", True)]


def test_legend_rename_is_escaped_and_reset_puts_the_name_back(window):
    window.controller._set("legend.entry.Apo", TextStyle(text="Apo <1 mM>"))
    label = window.legend.items[0][1]
    assert label.item.toPlainText() == "Apo <1 mM>"
    names = [title for title, _ in window.controller.menu_entries("legend.entry.Apo")]
    assert names[:3] == ["Rename…", "Text style…", "Reset name"]
    window.controller._set("legend.entry.Apo", TextStyle())
    assert label.item.toPlainText() == "Apo"


def test_a_popover_and_an_editor_survive_the_plot_being_cleared_and_closed(qtbot):
    """Plan G2 test 16: twenty times, no RuntimeError and no abort."""
    for _ in range(20):
        w = _Window()
        qtbot.addWidget(w)
        w.show()
        qtbot.waitExposed(w)
        controller = w.controller
        popover = controller.open_style("title")
        editor = controller.edit_inline("axis.left.label")
        w.plot.clear()
        QtWidgets.QApplication.processEvents()
        if editor is not None and figure_text._alive(editor):
            editor.cancel()
        if figure_text._alive(popover):
            popover.accept()       # commits through a host whose plot was just cleared
        w.close()
        w.deleteLater()
        QtWidgets.QApplication.processEvents()
        controller.close_editors() if figure_text._alive(controller) else None
        QtWidgets.QApplication.processEvents()


# -- build 5: the tick labels' own menu, Add title…, a redraw under a pin, Undo keys ----------

def test_a_right_click_on_the_tick_numbers_styles_the_tick_labels(window):
    """Review intent-1: a right-click on the numbers raised the LABEL's menu, so Text style…
    made the y label bold and orange and left the ticks as they were. Now the numbers have
    their own menu, every style of the tick labels is one Undo step, and the label keeps its."""
    axis = window.plot.getAxis("left")
    point = _tick_point(window.plot, "left")
    assert axis.sceneBoundingRect().contains(point)
    assert not axis.label.sceneBoundingRect().contains(point)
    _right_click(window.view, point)
    menu = window.controller.last_menu
    names = _names(menu)
    assert names[:2] == ["Text style…", "Axis properties…"], names
    assert "Edit label…" not in names and names[2] == "Reset view"
    _choose(menu, "Text style…")
    popover = window.controller.popover
    assert isinstance(popover, figure_text.TextStylePopover) and popover.key == "axis.left.ticks"
    assert not popover.text.isVisibleTo(popover), "tick labels have no text to type"
    QtTest.QTest.mouseClick(popover.use_family, L)
    popover.family.setCurrentText("Courier New")
    popover.size.setValue(13)
    QtTest.QTest.mouseClick(popover.bold, L)
    QtTest.QTest.mouseClick(popover.italic, L)
    QtTest.QTest.keyClicks(popover.color.hex, "#d55e00")
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()
    state = window.appearance.state()
    assert state.text_style("axis.left.ticks") == TextStyle(
        family="Courier New", size_pt=13.0, bold=True, italic=True, color="#d55e00")
    assert state.text_style("axis.left.label") == TextStyle(), "the label is not the ticks"
    font = axis.style["tickFont"]
    assert (font.family(), font.pointSizeF(), font.bold(), font.italic()) == (
        "Courier New", 13.0, True, True)
    assert axis.textPen().color().name() == "#d55e00"
    assert "font-weight" not in axis.labelStyle and axis.labelStyle.get("color") != "#d55e00"
    assert window.appearance._history.depth() == (1, 0)
    assert window.appearance.undo_button.toolTip().startswith("Undo: y axis ticks style")
    # A project keeps it (the appearance round trip a saved project makes).
    reopened = PlotAppearance.from_mapping(json.loads(json.dumps(state.to_dict())))
    assert reopened == state
    window.appearance.undo()
    font = axis.style["tickFont"]                  # the window's own (None: Qt's default)
    assert (font is None or not font.bold()) and axis.textPen().color().name() != "#d55e00"
    # The label keeps its own menu.
    _right_click(window.view, _centre(axis.label))
    assert _names(window.controller.last_menu)[:2] == ["Edit label…", "Text style…"]
    window.controller.last_menu.close()


def test_a_hidden_axis_label_comes_back_from_its_tick_numbers(window):
    """A hidden label leaves nothing to right-click; its numbers offer Show label."""
    axis = window.plot.getAxis("left")
    _right_click(window.view, _centre(axis.label))
    _choose(window.controller.last_menu, "Hide label")
    assert window.appearance.state().y_label == " " and not axis.label.toPlainText().strip()
    _right_click(window.view, _tick_point(window.plot, "left"))
    menu = window.controller.last_menu
    assert _names(menu)[:3] == ["Text style…", "Show label", "Axis properties…"]
    _choose(menu, "Show label")
    assert window.appearance.state().y_label is None
    assert axis.label.toPlainText().strip() == "Depth (pA)"
    assert window.appearance._history.depth() == (2, 0)
    _right_click(window.view, _tick_point(window.plot, "left"))
    assert "Show label" not in _names(window.controller.last_menu), "only while it is hidden"
    window.controller.last_menu.close()


@pytest.fixture
def untitled(qtbot):
    """A window whose automatic title is empty (I-V and Voltage dependence since the fit
    left the title): the title row is 0 px high, so there is nothing to click."""
    w = _Window(title="")
    qtbot.addWidget(w)
    w.show()
    qtbot.waitExposed(w)
    QtWidgets.QApplication.processEvents()
    return w


def _add_title(w):
    _right_click(w.view, _background_point(w.plot))
    menu = w.controller.last_menu
    assert "Add title…" in _names(menu), _names(menu)
    _choose(menu, "Add title…")
    editor = w.controller.editor
    assert isinstance(editor, InlineTextEditor) and editor.isVisible()
    return editor


def test_the_inline_editor_stays_inside_a_view_its_plot_reaches_past(untitled):
    """In the Analysis Explorer the Noise views' ViewBox reached 1-2 px past the viewport's
    right edge, and the editor, centred on it, opened 2 px outside the view (35 + 605 in a
    638 px viewport). It is now kept inside the view's edges."""
    w = untitled
    room = w.view.viewport().width()
    w.plot.setMinimumWidth(room + 2)
    w.resize(w.width() + 1, w.height())
    QtWidgets.QApplication.processEvents()
    assert w.plot.getViewBox().sceneBoundingRect().right() > room, "no overflow to test"
    editor = _add_title(w)
    assert w.view.viewport().rect().contains(editor.geometry()), editor.geometry()
    editor.cancel()


def test_add_title_from_the_plots_menu_when_the_title_draws_nothing(untitled):
    """INTEGRATE §3 (G4): an empty automatic title had no in-figure way to add one."""
    w = untitled
    title = w.plot.titleLabel
    assert not title.isVisible()
    _right_click(w.view, _background_point(w.plot))
    assert _names(w.controller.last_menu) == ["Add title…", "Reset view", "Zoom back",
                                             "Zoom forward", "Export current figure…"]
    w.controller.last_menu.close()
    editor = _add_title(w)
    assert editor.text() == ""
    assert w.view.viewport().rect().contains(editor.geometry()), "opens inside the view"
    QtTest.QTest.keyClicks(editor, "Pore 3: t_{1}")   # QTest types ASCII only
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert w.appearance.state().title == "Pore 3: t_{1}"
    assert title.isVisible() and title.item.toPlainText() == "Pore 3: t1"
    assert w.appearance._history.depth() == (1, 0)
    assert w.appearance.undo_button.toolTip() == "Undo: title automatic → ‘Pore 3: t_{1}’"
    assert "Add title…" not in _names(app_menu_for(w.plot)), "a title that shows needs none"
    assert w.appearance.undo()
    assert not title.isVisible() and w.appearance.state().title is None
    # Esc leaves the figure as it was
    editor = _add_title(w)
    QtTest.QTest.keyClicks(editor, "zzz")
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Escape)
    QtWidgets.QApplication.processEvents()
    assert w.appearance.state().title is None and not title.isVisible()
    assert w.appearance._history.depth() == (0, 1)
    # and so does Return on the box left empty
    editor = _add_title(w)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert w.appearance.state().title is None and w.appearance._history.depth() == (0, 1)


def test_a_hidden_title_offers_add_title_and_show_title(window):
    title = window.plot.titleLabel
    _right_click(window.view, _centre(title.item))
    _choose(window.controller.last_menu, "Hide title")
    assert window.appearance.state().title == " " and not title.isVisible()
    # Return on the empty box does not quietly bring the automatic title back.
    editor = _add_title(window)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert window.appearance.state().title == " " and not title.isVisible()
    assert window.appearance._history.depth() == (1, 0)
    _right_click(window.view, _background_point(window.plot))
    menu = window.controller.last_menu
    assert _names(menu)[:3] == ["Add title…", "Show title", "Reset view"]
    _choose(menu, "Show title")
    assert window.appearance.state().title is None
    assert title.isVisible() and title.item.toPlainText() == "Automatic title"
    assert window.appearance._history.depth() == (2, 0)


def test_a_data_redraw_returns_a_pinned_axis_to_its_pin(window):
    """Review FIG-2: I-V and Voltage dependence redraw with ``autoRange()``, which turns
    auto-range off, so the next apply took the data's framing for a manual zoom and
    Refresh dropped the range typed in Axis properties (while Axis properties showed it)."""
    from nanoscope.gui.plot_appearance import reframe_axis_pins

    view = window.plot.getViewBox()
    window.host.apply_axis("bottom", AxisState(range=(0.5, 4.0)))
    window.host.apply_axis("left", AxisState(range=(3.0, 6.0)))
    assert view.viewRange() == [pytest.approx([0.5, 4.0]), pytest.approx([3.0, 6.0])]

    def redraw(reframe):
        # What the windows' _redraw does: new data, framed by autoRange(), then the apply.
        rng = np.random.default_rng(5)
        x = rng.uniform(0.1, 8, 60)
        window.scatter.setData(x, 3 * x - 2)
        window.plot.autoRange()
        if reframe:
            reframe_axis_pins(window.plot)
        window.redraw(window.appearance.state())

    redraw(reframe=False)
    assert view.viewRange()[0] != pytest.approx([0.5, 4.0]), "the defect this pins"
    redraw(reframe=True)
    assert view.viewRange() == [pytest.approx([0.5, 4.0]), pytest.approx([3.0, 6.0])]
    # An appearance-only step still keeps a manual zoom, and its Undo does too.
    view.setXRange(1.0, 2.0, padding=0)
    window.appearance.apply_appearance(PlotAppearance.from_mapping(
        {**window.appearance.state().to_dict(), "title_size": 16}))
    assert view.viewRange()[0] == pytest.approx([1.0, 2.0])
    window.appearance.undo()
    assert view.viewRange()[0] == pytest.approx([1.0, 2.0])
    redraw(reframe=True)
    assert view.viewRange()[0] == pytest.approx([0.5, 4.0])
    navigation_for(window.plot).reset_view()
    assert view.viewRange()[0] == pytest.approx([0.5, 4.0])


def _page(qtbot, *, on_page):
    page = QtWidgets.QWidget()
    qtbot.addWidget(page)
    layout = QtWidgets.QVBoxLayout(page)
    button = QtWidgets.QPushButton("Refresh")
    layout.addWidget(button)
    widget = PlotAppearanceWidget(shortcut_parent=page if on_page else None)
    layout.addWidget(widget)
    page.show()
    qtbot.waitExposed(page)
    page.activateWindow()
    widget.apply_appearance(PlotAppearance(title="Pore 3"))
    widget.hide()            # the workflow bar folds the Appearance section away
    button.setFocus()
    QtWidgets.QApplication.processEvents()
    return page, widget


def test_the_undo_keys_can_live_on_the_page_so_every_section_answers(qtbot):
    """Review FIG-4: ⌘Z did nothing after an in-figure edit unless Section was Appearance,
    because the keys were children of the folded-away appearance widget."""
    from nanoscope.gui.key_sequences import redo_sequences, undo_sequences

    page, widget = _page(qtbot, on_page=False)
    _press(page, undo_sequences()[0])
    assert widget.state().title == "Pore 3", "default unchanged: a hidden widget's keys"
    page, widget = _page(qtbot, on_page=True)
    _press(page, undo_sequences()[0])
    assert widget.state().title is None and widget._history.depth() == (0, 1)
    _press(page, redo_sequences()[0])
    assert widget.state().title == "Pore 3" and widget._history.depth() == (1, 0)
    # A widget built again on the same page: the old one's keys went with it, so the
    # key is not bound twice (Qt fires neither of an ambiguous pair).
    layout = page.layout()
    layout.removeWidget(widget)
    widget.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    QtWidgets.QApplication.processEvents()
    assert len(page.findChildren(QtGui.QShortcut)) == 0
    again = PlotAppearanceWidget(shortcut_parent=page)
    layout.addWidget(again)
    again.apply_appearance(PlotAppearance(title="Pore 4"))
    again.hide()
    page.activateWindow()        # building a widget offscreen takes the window's activation
    QtWidgets.QApplication.processEvents()
    _press(page, undo_sequences()[0])
    assert again.state().title is None


def test_a_page_behind_another_does_not_answer_the_undo_key(qtbot):
    from nanoscope.gui.key_sequences import undo_sequences

    stack = QtWidgets.QStackedWidget()
    qtbot.addWidget(stack)
    pages, widgets = [], []
    for title in ("Front", "Behind"):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        widget = PlotAppearanceWidget(shortcut_parent=page)
        layout.addWidget(widget)
        stack.addWidget(page)
        widget.apply_appearance(PlotAppearance(title=title))
        widget.hide()        # each page's Appearance section folded away
        pages.append(page)
        widgets.append(widget)
    stack.setCurrentWidget(pages[0])
    stack.show()
    qtbot.waitExposed(stack)
    stack.activateWindow()
    QtWidgets.QApplication.processEvents()
    _press(stack, undo_sequences()[0])
    assert widgets[0].state().title is None
    assert widgets[1].state().title == "Behind"
