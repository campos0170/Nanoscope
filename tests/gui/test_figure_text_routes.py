"""In-figure text through the real windows (build 5): the tick labels' own menu, Add
title…, and a legend name's sub- and superscripts in the fit label.

Every gesture is a real one: clicks and keys go through the QWINDOW
(``widget.window().windowHandle()``), menu entries are clicked, and the save is a real
``_write_project`` reopened by ``_open_project``. Synthetic data only (CSV traces written
to ``tmp_path``, in-memory frames); offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import InlineTextEditor, TextStyle, figure_text_for

L = QtCore.Qt.MouseButton.LeftButton
R = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier
CAPTION = "annotation.provenance"


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


# -- real gestures --------------------------------------------------------------------------------

def _window_pos(plot, scene_pt):
    view = plot.scene().views()[0]
    return view.viewport().mapTo(view.window(), view.mapFromScene(scene_pt))


def _click(plot, scene_pt, button=R, double=False):
    handle = plot.scene().views()[0].window().windowHandle()
    pos = _window_pos(plot, scene_pt)
    QtTest.QTest.mouseMove(handle, pos)
    if double:
        QtTest.QTest.mouseDClick(handle, button, NOMOD, pos)
    else:
        QtTest.QTest.mouseClick(handle, button, NOMOD, pos)
    QtWidgets.QApplication.processEvents()


def _names(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def _choose(menu, text):
    action = next(a for a in menu.actions() if a.text() == text)
    QtTest.QTest.mouseClick(menu, L, NOMOD, menu.actionGeometry(action).center())
    QtWidgets.QApplication.processEvents()


def _tick_point(plot, side):
    # Tick measurement schedules a second layout pass; settle before hit testing.
    for _ in range(6):
        QtWidgets.QApplication.processEvents()
    vb = plot.getViewBox().sceneBoundingRect()
    if side == "bottom":
        return QtCore.QPointF(vb.center().x() + 30, vb.bottom() + 9)
    return QtCore.QPointF(vb.left() - 14, vb.center().y() + 30)


def _style_ticks_bold_orange(plot, side):
    """Right-click the tick numbers, Text style…, B, #d55e00, OK: all real clicks."""
    controller = figure_text_for(plot)
    controller.last_menu = None
    _click(plot, _tick_point(plot, side))
    menu = controller.last_menu
    names = _names(menu)
    assert names[:2] == ["Text style…", "Axis properties…"] and "Edit label…" not in names, names
    _choose(menu, "Text style…")
    popover = controller.popover
    assert popover.key == f"axis.{side}.ticks"
    QtTest.QTest.mouseClick(popover.bold, L)
    popover.color.hex.clear()
    QtTest.QTest.keyClicks(popover.color.hex, "#d55e00")
    ok = popover.findChild(QtWidgets.QDialogButtonBox).button(
        QtWidgets.QDialogButtonBox.StandardButton.Ok)
    QtTest.QTest.mouseClick(ok, L)
    QtWidgets.QApplication.processEvents()


def _ticks_drawn(axis):
    font = axis.style.get("tickFont")
    return (font is not None and font.bold(), axis.textPen().color().name())


# -- I-V ------------------------------------------------------------------------------------------

@pytest.fixture
def iv(qapp, qtbot, tmp_path):
    from nanoscope.gui.iv_panel import IVWindow

    from .test_iv import _ohmic_session

    w = IVWindow(_ohmic_session(tmp_path, G_nS=100.0, leak_pA=-10.0))
    qtbot.addWidget(w)
    w.resize(1000, 800)
    w.show()
    qtbot.waitExposed(w)
    QtWidgets.QApplication.processEvents()
    return w


def test_iv_tick_numbers_have_their_own_text_style(iv):
    """Review intent-1, on I-V: the numbers' menu styled the y LABEL."""
    plot = iv.plot_item()
    widget = iv._appearance_widget
    back = widget._history.depth()[0]
    label_css = dict(plot.getAxis("left").labelStyle)
    _style_ticks_bold_orange(plot, "left")
    state = widget.state()
    assert state.text_style("axis.left.ticks") == TextStyle(bold=True, color="#d55e00")
    assert state.text_style("axis.left.label") == TextStyle()
    assert _ticks_drawn(plot.getAxis("left")) == (True, "#d55e00")
    assert plot.getAxis("left").labelStyle == label_css, "the label is untouched"
    assert widget._history.depth()[0] == back + 1
    assert widget.undo()
    assert _ticks_drawn(plot.getAxis("left")) != (True, "#d55e00")


def test_iv_add_title_from_the_plot_menu(iv):
    """INTEGRATE §3 (G4): I-V's automatic title is empty since the fit left it, and the
    plot menu offered only Reset view, Zoom and Export."""
    plot = iv.plot_item()
    widget = iv._appearance_widget
    controller = figure_text_for(plot)
    assert not plot.titleLabel.isVisible()
    vb = plot.getViewBox().sceneBoundingRect()
    background = QtCore.QPointF(vb.left() + 0.5 * vb.width(), vb.top() + 0.6 * vb.height())
    back = widget._history.depth()[0]

    def add_title():
        controller.last_menu = None
        _click(plot, background)
        menu = controller.last_menu
        assert _names(menu)[:2] == ["Add title…", "Reset view"], _names(menu)
        _choose(menu, "Add title…")
        editor = controller.editor
        assert isinstance(editor, InlineTextEditor) and editor.isVisible() and editor.text() == ""
        return editor

    editor = add_title()
    QtTest.QTest.keyClicks(editor, "zzz")
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Escape)
    QtWidgets.QApplication.processEvents()
    assert widget.state().title is None and not plot.titleLabel.isVisible()
    assert widget._history.depth()[0] == back, "Esc leaves it as it was"
    editor = add_title()
    QtTest.QTest.keyClicks(editor, "Pore 3 I-V")
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    assert widget.state().title == "Pore 3 I-V"
    assert plot.titleLabel.isVisible() and plot.titleLabel.item.toPlainText() == "Pore 3 I-V"
    assert "G (I-V slope)" in iv.fit_label_text(), "the fit stays in its label"
    assert widget._history.depth()[0] == back + 1
    assert widget.undo()
    assert not plot.titleLabel.isVisible()


def _csv(path, value, n=4000):
    x = np.full(n, value, dtype=np.float32)
    pd.DataFrame({"time": np.arange(n) / 50000.0, "current": x}).to_csv(path, index=False)


def _main_window(qtbot):
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_iv_tick_style_and_annotation_places_survive_a_real_save_and_reopen(
        qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.core.project import DocumentRef, ProjectState
    from nanoscope.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    paths = []
    for mv in (-200, -500, -1000):
        path = tmp_path / f"pore_{mv}mV.csv"
        _csv(path, 0.1 * mv - 10.0)
        paths.append(path)
    source = _main_window(qtbot)
    with qtbot.waitSignal(source.projectRestored, timeout=20_000):
        source._begin_restore(ProjectState(documents=[DocumentRef(str(p)) for p in paths]))
    source._open_iv()
    QtWidgets.QApplication.processEvents()
    page = source._iv_window
    plot = page.plot_item()
    _style_ticks_bold_orange(plot, "left")
    widget = page._appearance_widget
    # A dragged annotation's place rides the same appearance (I-V draws no caption: this
    # is the store a window's caption writes, saved and reopened by the project).
    widget.apply_appearance(widget.state().with_annotation_offset(CAPTION, (130.0, 228.5)))
    saved = page.appearance_state()
    assert saved["text"]["axis.left.ticks"] == {"bold": True, "color": "#d55e00"}
    assert saved["annotations"] == {CAPTION: {"offset": [130.0, 228.5]}}
    project = tmp_path / "ticks.nanoscope"
    assert source._write_project(str(project))
    source._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_iv()
    QtWidgets.QApplication.processEvents()
    page = reopened._iv_window
    assert page.appearance_state() == saved
    assert page._appearance_widget.state().annotation_offset(CAPTION) == (130.0, 228.5)
    assert _ticks_drawn(page.plot_item().getAxis("left")) == (True, "#d55e00")
    reopened._dirty = False


# -- Distributions --------------------------------------------------------------------------------

def _df(seed=0, n=300, slope=2.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({
        "xcol": x, "ycol": slope * x + 5.0 + rng.normal(0.0, 0.3, n),
        "dwell_s": rng.exponential(0.002, n) + 1e-4, "depth_pA": rng.normal(50.0, 5.0, n),
        "accepted": np.ones(n, dtype=bool), "filter_limited": np.zeros(n, dtype=bool)})


@pytest.fixture
def scatter(qapp, qtbot):
    from nanoscope.gui.plots_panel import PlotsWindow

    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1200, 820)
    w.show()
    qtbot.waitExposed(w)
    frames = {"Apo": _df(0), "Holo": _df(1, slope=3.0)}
    w.set_data(frames)
    w.set_sources(list(frames))
    w.set_plot_type("Scatter")
    w.set_scatter_columns("xcol", "ycol")
    w.set_scatter_fit("linear")
    QtWidgets.QApplication.processEvents()
    return w


def test_a_legend_name_with_a_subscript_reads_the_same_in_the_fit_label(scatter):
    """Review FIG-3: renamed 'A t_{1}', the legend drew t₁ and the fit label's line
    printed 'A t_{1}: y = …' on screen and in the export."""
    w = scatter
    controller = figure_text_for(w._plot)
    key = next(k for k in controller.targets() if k.startswith("legend.entry.Apo"))
    row = controller.targets()[key].item
    _click(w._plot, row.sceneBoundingRect().center(), button=L, double=True)
    editor = controller.editor
    assert editor is not None, "a double-click on a legend row opened nothing"
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, "A t_{1} <wt>")
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    QtWidgets.QApplication.processEvents()
    rows = [label.item.toPlainText() for _s, label in w._plot.legend.items]
    assert any(text.startswith("A t1 <wt>") for text in rows), rows
    text = w._fit_label_item.plain_text()
    assert text.startswith("A t1 <wt>: y = "), text
    assert "_{" not in text


def test_distributions_tick_numbers_have_their_own_text_style(qapp, qtbot):
    from nanoscope.gui.plots_panel import PlotsWindow

    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1100, 850)
    w.show()
    qtbot.waitExposed(w)
    w.set_data({"Apo": _df()})
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    QtWidgets.QApplication.processEvents()
    _style_ticks_bold_orange(w._plot, "bottom")
    assert w._style.text_style("axis.bottom.ticks") == TextStyle(bold=True, color="#d55e00")
    assert w._style.text_style("axis.bottom.label") == replace(
        w._style.text_style("axis.bottom.label"), bold=None, color=None)
    assert _ticks_drawn(w._plot.getAxis("bottom")) == (True, "#d55e00")
