"""Distributions: every legend row, the detection-limit words and the title are figure text.

The owner's request: "Everything that is text in the figures should be editable, the user
needs freedom to style as needed." In Distributions a source's legend row could only be
renamed or hidden, a fit's row ("Apo fit (exponential)") could not be reached at all, and
the "below detection limit" words on the detection-floor line had no editor.

* A source's row: Rename… (the source's one name), Text style…, Reset name, Hide entry.
* A fit's row: a double-click renames its SOURCE, so "Apo fit (…)" follows; Text style… and
  Hide entry apply to the row. A one-source fit row ("fit (…)") names no source, so its
  double-click opens Text style….
* A row's style is ``text_styles["legend.entry.…"]``: one Undo step, kept by the plot
  configuration's JSON (and so by a project), and left out of templates with the names.
* The detection-limit words: retyped, styled or hidden (``annotation.limit``); the line and
  the shading stay.
* The title is a figure-text target even when it is empty or hidden, so the plot's menu
  can offer to add one.

Clicks go through the QWindow / the viewport, as a user's do. Synthetic data only; offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.core.session import Session
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import HIDDEN_TEXT, TextStyle, figure_text_for
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plots_panel import (
    PlotConfig, PlotsWindow, StyleSettings, plot_config_from_json, plot_config_to_json,
)
from nanoscope.plot.pg_svg import export_svg

LEFT = QtCore.Qt.MouseButton.LeftButton
RIGHT = QtCore.Qt.MouseButton.RightButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier


@pytest.fixture(autouse=True)
def _gate():
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _df(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 5.0, n)
    return pd.DataFrame({
        "xcol": x, "ycol": 2.0 * x + 5.0 + rng.normal(0.0, 0.3, n),
        "dwell_s": rng.exponential(0.002, n) + 1e-4,
        "depth_pA": rng.normal(50.0, 5.0, n),
        "accepted": np.ones(n, dtype=bool), "filter_limited": np.zeros(n, dtype=bool)})


def _frames() -> dict:
    return {"Apo": _df(0), "Holo": _df(4)}


def _window(qtbot, frames=None) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.resize(1200, 820)
    w.show()
    qtbot.waitExposed(w)
    w.set_data(frames or {"probe": _df()})
    if frames:
        w.set_sources(list(frames))
    return w


def _histogram(qtbot, frames=None) -> PlotsWindow:
    w = _window(qtbot, frames)
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    return w


def _rows(w) -> list:
    return [label for _sample, label in w._plot.legend.items]


def _row(w, starts: str):
    return next(label for label in _rows(w) if label.item.toPlainText().startswith(starts))


def _depth(w) -> int:
    return w._history.depth()[0]


def _double_click(w, item) -> None:
    view = w._plot_widget
    at = view.viewport().mapTo(view.window(), view.mapFromScene(item.sceneBoundingRect().center()))
    QTest.mouseDClick(view.window().windowHandle(), LEFT, NOMOD, at)
    QtWidgets.QApplication.processEvents()


def _right_click(w, item) -> QtWidgets.QMenu:
    """A right press and release on the viewport over ``item``: the menu it raised."""
    controller = figure_text_for(w._plot)
    controller.last_menu = None
    view = w._plot_widget
    viewport = view.viewport()
    pos = view.mapFromScene(item.sceneBoundingRect().center())
    QTest.mouseMove(viewport, pos)
    local, screen = QtCore.QPointF(pos), QtCore.QPointF(viewport.mapToGlobal(pos))
    for kind, held in ((QtCore.QEvent.Type.MouseButtonPress, RIGHT),
                       (QtCore.QEvent.Type.MouseButtonRelease, QtCore.Qt.MouseButton.NoButton)):
        QtWidgets.QApplication.sendEvent(
            viewport, QtGui.QMouseEvent(kind, local, local, screen, RIGHT, held, NOMOD))
        QtWidgets.QApplication.processEvents()
    menu = controller.last_menu or QtWidgets.QApplication.activePopupWidget()
    assert menu is not None, "a right-click raised no menu"
    return menu


def _walk_to(w, item) -> None:
    """Move the pointer onto ``item`` in steps, as a hand does: pyqtgraph sends a click to
    the item the last hover found, and a first press with no hover goes to the page."""
    viewport = w._plot_widget.viewport()
    target = w._plot_widget.mapFromScene(item.sceneBoundingRect().center())
    start = QtCore.QPoint(viewport.width() // 2, viewport.height() // 2)
    for step in range(6):
        QTest.mouseMove(viewport, start + (target - start) * (step / 5.0))
        QtWidgets.QApplication.processEvents()


def _titles(menu) -> list[str]:
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def _choose(menu, title: str) -> None:
    action = next(a for a in menu.actions() if a.text() == title)
    menu.close()
    action.trigger()
    QtWidgets.QApplication.processEvents()


def _ok(popover) -> None:
    buttons = popover.findChild(QtWidgets.QDialogButtonBox)
    buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
    QtWidgets.QApplication.processEvents()


# -- a source's row ------------------------------------------------------------------------------

def test_a_source_row_is_styled_from_its_right_click_in_one_step_that_survives_json(qapp, qtbot):
    w = _histogram(qtbot, _frames())
    menu = _right_click(w, _row(w, "Apo (n"))
    assert _titles(menu)[:4] == ["Rename…", "Text style…", "Reset name", "Hide entry"]
    back = _depth(w)
    _choose(menu, "Text style…")
    popover = figure_text_for(w._plot).popover
    assert popover is not None, "Text style… opened nothing"
    popover.bold.click()
    popover.color.set_value("#d55e00")
    _ok(popover)
    apo, holo = _row(w, "Apo (n"), _row(w, "Holo (n")
    assert "font-weight: bold" in apo.text and "#d55e00" in apo.text, apo.text
    assert "font-weight" not in holo.text, holo.text
    assert apo.item.toPlainText().startswith("Apo (n = 300)")
    assert _depth(w) == back + 1
    # Kept by the configuration file, and drawn from it by another window.
    again = plot_config_from_json(plot_config_to_json(w.config()))
    assert again.style.text_style("legend.entry.Apo").bold is True
    other = _histogram(qtbot, _frames())
    other.apply_config(again)
    assert "font-weight: bold" in _row(other, "Apo (n").text
    # ...and one Undo takes it off.
    assert w.undo_config()
    assert "font-weight" not in _row(w, "Apo (n").text


# -- a fit's row ---------------------------------------------------------------------------------

def test_a_fit_row_double_click_renames_its_source_and_the_fit_row_follows(qapp, qtbot):
    w = _histogram(qtbot, _frames())
    controller = figure_text_for(w._plot)
    back = _depth(w)
    _double_click(w, _row(w, "Apo fit (exponential)"))
    assert controller.editor is not None, "a double-click on a fit row opened nothing"
    assert controller.editor.text() == "Apo"
    controller.editor.selectAll()
    QTest.keyClicks(controller.editor, "Wild type")
    QTest.keyClick(controller.editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    shown = [label.item.toPlainText() for label in _rows(w)]
    assert "Wild type fit (exponential)" in shown and "Wild type (n = 300)" in shown, shown
    assert w.config().style.label_overrides.get("Apo") == "Wild type"
    assert w._fit_label_item.plain_text().startswith("Wild type: "), w._fit_label_item.plain_text()
    assert _depth(w) == back + 1


def test_a_fit_row_right_click_offers_text_style_and_hides_that_row_alone(qapp, qtbot):
    w = _histogram(qtbot, _frames())
    menu = _right_click(w, _row(w, "Holo fit (exponential)"))
    titles = _titles(menu)
    assert "Text style…" in titles and "Hide entry" in titles, titles
    back = _depth(w)
    _choose(menu, "Hide entry")
    shown = [label.item.toPlainText() for label in _rows(w)]
    assert "Holo fit (exponential)" not in shown and "Apo fit (exponential)" in shown, shown
    assert any(text.startswith("Holo (n") for text in shown), shown
    assert w.config().style.legend_include.get("fit:Holo") is False
    assert _depth(w) == back + 1
    assert w.undo_config()
    assert "Holo fit (exponential)" in [label.item.toPlainText() for label in _rows(w)]


def test_a_fit_row_text_style_styles_that_row_only(qapp, qtbot):
    w = _histogram(qtbot, _frames())
    _choose(_right_click(w, _row(w, "Apo fit (exponential)")), "Text style…")
    popover = figure_text_for(w._plot).popover
    popover.italic.click()
    _ok(popover)
    assert "font-style: italic" in _row(w, "Apo fit (exponential)").text
    assert "italic" not in _row(w, "Apo (n").text
    assert w.config().style.text_style("legend.entry.fit:Apo").italic is True


def test_a_one_source_fit_row_names_no_source_so_a_double_click_opens_text_style(qapp, qtbot):
    w = _histogram(qtbot)
    controller = figure_text_for(w._plot)
    menu = _right_click(w, _row(w, "fit (exponential)"))
    titles = _titles(menu)
    menu.close()
    assert "Rename…" not in titles and "Text style…" in titles and "Hide entry" in titles
    opened = []
    controller.opened.connect(lambda kind, key: opened.append((kind, key)))
    _double_click(w, _row(w, "fit (exponential)"))
    assert opened == [("style", "legend.entry.fit:probe")], opened
    controller.close_editors()


# -- the detection-limit words -------------------------------------------------------------------

def test_the_detection_limit_words_are_retyped_styled_and_hidden_from_the_figure(
        qapp, qtbot, tmp_path):
    w = _histogram(qtbot)
    controller = figure_text_for(w._plot)
    label = w._limit_line.label
    assert label.toPlainText() == "below detection limit" and label.isVisible()
    back = _depth(w)
    _double_click(w, label)
    assert controller.editor is not None, "a double-click on the limit words opened nothing"
    assert controller.editor.text() == "below detection limit"
    controller.editor.selectAll()
    QTest.keyClicks(controller.editor, "detector floor")
    QTest.keyClick(controller.editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    label = w._limit_line.label
    assert label.toPlainText() == "detector floor"
    assert _depth(w) == back + 1
    assert "detector floor" in export_svg(w._plot, tmp_path / "fig.svg").read_text()

    menu = _right_click(w, label)
    assert _titles(menu)[:4] == ["Edit detection-limit label…", "Text style…",
                                 "Automatic detection-limit label", "Hide detection-limit label"]
    _choose(menu, "Text style…")
    popover = controller.popover
    popover.italic.click()
    _ok(popover)
    assert "italic" in w._limit_line.label.toHtml()
    assert w._limit_line.label.toPlainText() == "detector floor"
    assert _depth(w) == back + 2

    _choose(_right_click(w, w._limit_line.label), "Hide detection-limit label")
    assert not w._limit_line.label.isVisible()
    assert w._limit_line.isVisible(), "hiding the words must leave the detector's floor drawn"
    assert _depth(w) == back + 3
    assert w.undo_config()
    assert w._limit_line.label.isVisible()
    assert w._limit_line.label.toPlainText() == "detector floor"

    # Words and style in the configuration file and in a template; a preset is style only.
    cfg = w.config()
    again = plot_config_from_json(plot_config_to_json(cfg)).style.text_style("annotation.limit")
    assert again.text == "detector floor" and again.italic is True
    assert cfg.as_template().style.text_style("annotation.limit").text == "detector floor"
    preset = PlotAppearance.from_mapping(w.appearance_state()).style_only()
    assert preset.text_style("annotation.limit").text is None


def test_the_detection_limit_words_keep_their_style_through_a_redraw(qapp, qtbot):
    w = _histogram(qtbot)
    host = figure_text_for(w._plot).host
    host.apply_text("annotation.limit", TextStyle(text="floor", bold=True))
    w.set_histogram_column("depth_pA")
    w.set_histogram_column("dwell_s")             # a fresh line and label
    label = w._limit_line.label
    assert label.toPlainText() == "floor" and "font-weight:700" in label.toHtml()
    label.setVisible(True)                        # pyqtgraph's own re-show re-sets plain text
    assert label.toPlainText() == "floor" and "font-weight:700" in label.toHtml()
    # An edit in the Appearance section rewrites the element styles from its own state;
    # the words go through it too.
    widget = w._appearance_widget
    widget.apply_appearance(replace(widget.state(), title_size=16.0))
    assert w.config().style.text_style("annotation.limit").text == "floor"
    assert w._limit_line.label.toPlainText() == "floor"
    host.apply_text("annotation.limit", TextStyle(text=HIDDEN_TEXT))
    assert not w._limit_line.label.isVisible()


def test_hidden_detection_limit_words_come_back_from_the_plots_own_menu(qapp, qtbot):
    """Hidden words leave nothing to right-click, and no control in the window names them:
    a right-click where they were (the plot's own menu) offers Show detection-limit label,
    one Undo step, as Show title does for a hidden title."""
    w = _histogram(qtbot)
    _walk_to(w, w._limit_line.label)
    _choose(_right_click(w, w._limit_line.label), "Hide detection-limit label")
    assert not w._limit_line.label.isVisible() and w._limit_line.isVisible()
    back = _depth(w)
    menu = _right_click(w, w._limit_line.label)
    assert "Show detection-limit label" in _titles(menu), _titles(menu)
    _choose(menu, "Show detection-limit label")
    label = w._limit_line.label
    assert label.isVisible() and label.toPlainText() == "below detection limit"
    assert _depth(w) == back + 1
    entries = [title for title, _cb in figure_text_for(w._plot).background_entries()]
    assert "Show detection-limit label" not in entries, entries


def test_the_kde_overlay_row_is_styled_or_hidden_but_not_renamed(qapp, qtbot):
    w = _window(qtbot, _frames())
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    w._hist_kde.setChecked(True)
    w._plot_btn.click()
    qapp.processEvents()
    menu = _right_click(w, _row(w, "KDE (Gaussian"))
    titles = _titles(menu)
    assert "Rename…" not in titles and "Text style…" in titles and "Hide entry" in titles, titles
    back = _depth(w)
    _choose(menu, "Hide entry")
    assert not any(label.item.toPlainText().startswith("KDE") for label in _rows(w))
    assert w.config().style.legend_include.get("kde:") is False
    assert _depth(w) == back + 1


# -- the configuration seams ---------------------------------------------------------------------

def test_legend_row_styles_go_with_their_source_and_stay_out_of_templates():
    style = (StyleSettings()
             .with_text_style("legend.entry.Apo", TextStyle(bold=True))
             .with_text_style("legend.entry.fit:Apo", TextStyle(italic=True))
             .with_text_style("annotation.limit", TextStyle(text="floor", color="#d55e00"))
             .with_text_style("title", TextStyle(text="Fig 1", bold=True)))
    cfg = PlotConfig(style=style)
    template = cfg.as_template().style
    assert not any(key.startswith("legend.entry.") for key in template.text_styles)
    assert template.text_style("annotation.limit").text == "floor"
    assert template.text_style("title").bold is True and template.title == "Fig 1"
    # Recalled onto another window: that window's legend styles stay, the template's
    # wording and element styles arrive.
    other = PlotConfig(style=StyleSettings().with_text_style(
        "legend.entry.Holo", TextStyle(color="#009e73")))
    recalled = other.with_template(cfg).style
    assert recalled.text_style("legend.entry.Holo").color == "#009e73"
    assert recalled.text_style("legend.entry.Apo").bold is None
    assert recalled.text_style("annotation.limit").text == "floor"
    # A closed source takes its rows' styles with it, as it takes its name.
    pruned = cfg.with_pruned_legend(["Holo"], []).style
    assert "legend.entry.Apo" not in pruned.text_styles
    assert "legend.entry.fit:Apo" not in pruned.text_styles
    assert pruned.text_style("annotation.limit").text == "floor"
    assert cfg.with_pruned_legend(["Apo"], []) is cfg


# -- the title -----------------------------------------------------------------------------------

def test_the_templates_note_names_restyled_legend_entries_among_what_stays_behind(qapp, qtbot):
    """A template drops the legend rows' styles with their names, and the sentence beside
    the templates list says what a template never carries."""
    w = _window(qtbot)
    note = next(label.text() for label in w.findChildren(QtWidgets.QLabel)
                if label.text().startswith("A template carries"))
    assert "your retyped or restyled legend entries" in note, note
    assert "style.text_styles[legend.entry.*]" in PlotConfig.TEMPLATE_DROPS


def test_the_title_is_a_target_when_it_is_empty_or_hidden(qapp, qtbot):
    """The plot's menu offers to add a title through this target; a figure with no
    automatic title (a histogram, a scatter) or a hidden one must still have it."""
    w = _histogram(qtbot)
    controller = figure_text_for(w._plot)
    assert not w._plot.titleLabel.isVisible()
    target = controller.targets().get("title")
    assert target is not None and {"edit", "style", "hide"} <= target.capabilities
    back = _depth(w)
    editor = controller.edit_inline("title")
    assert editor is not None and editor.text() == ""
    QTest.keyClicks(editor, "Fig A")
    QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    assert w._plot.titleLabel.isVisible() and w._plot.titleLabel.item.toPlainText() == "Fig A"
    assert _depth(w) == back + 1
    # Hidden ("Hide title"): still a target, and its editor brings a title back.
    controller.host.apply_text("title", TextStyle(text=HIDDEN_TEXT))
    assert not w._plot.titleLabel.isVisible()
    target = controller.targets().get("title")
    assert target is not None and target.style.text == HIDDEN_TEXT
    editor = controller.edit_inline("title")
    assert editor is not None
    editor.selectAll()
    QTest.keyClicks(editor, "Fig B")
    QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    assert w._plot.titleLabel.isVisible() and w._plot.titleLabel.item.toPlainText() == "Fig B"
    # An empty window (nothing drawn) has the target too.
    empty = PlotsWindow(Session())
    qtbot.addWidget(empty)
    assert figure_text_for(empty._plot).targets().get("title") is not None
