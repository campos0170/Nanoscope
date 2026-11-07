"""Phase 1 Foundation, F1: the page / shell interface in ``plot_workflow``.

``attach_inspector`` hands a bar's Sections to the Explorer's inspector, ``form_rows`` MOVES
widgets, ``PageAdapter.from_widget`` reads a real appearance widget. Real key events go
through the QWindow. Synthetic only; offscreen.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pathlib

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.plot_workflow import (DEFAULT_REVEAL, SECTIONS, ExportEntry, FormatSpec,
                                         HistoryState, InspectorGroup, PageAdapter, PendingState,
                                         PlotWorkflowBar, form_rows, install_plot_workflow)

from . import p1_hosting as H


def _bar_page(qtbot):
    page = QtWidgets.QMainWindow()
    central = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(central)
    labels = {name: QtWidgets.QLabel(name) for name in SECTIONS}
    for label in labels.values():
        layout.addWidget(label)
    page.setCentralWidget(central)
    bar = install_plot_workflow(page, sections={n: (w,) for n, w in labels.items()},
                                actions=(("Run", lambda: None),))
    qtbot.addWidget(page)
    return page, bar, labels


def test_attach_inspector_turns_a_section_into_a_reveal_and_toggles_nothing(qtbot):
    page, bar, labels = _bar_page(qtbot)
    page.show()
    asked: list[str] = []
    heard: list[str] = []
    bar.sectionChanged.connect(heard.append)
    bar.attach_inspector(asked.append)
    before = {name: label.isVisible() for name, label in labels.items()}
    bar.set_section("Appearance")
    bar.set_section("Appearance")           # the same Section again still asks
    assert asked == ["Appearance", "Appearance"] and heard == ["Appearance", "Appearance"]
    assert bar.sections.currentText() == "Appearance"   # the (hidden) combo still follows
    assert {name: label.isVisible() for name, label in labels.items()} == before
    with pytest.raises(ValueError):
        bar.set_section("Nonsense")          # the name is still validated


def test_standalone_bar_still_toggles_widgets(qtbot):
    page, bar, labels = _bar_page(qtbot)
    page.show()
    bar.set_section("Export")
    assert [n for n, w in labels.items() if w.isVisible()] == ["Export"]


def test_the_four_section_keys_reach_nobody_on_a_hosted_bar(qtbot):
    page, bar, labels = _bar_page(qtbot)
    page.show()
    page.activateWindow()
    qtbot.waitActive(page)
    asked: list[str] = []
    bar.attach_inspector(asked.append)
    assert all(not s.isEnabled() for s in bar.section_shortcuts)
    for keycode in (QtCore.Qt.Key.Key_1, QtCore.Qt.Key.Key_2, QtCore.Qt.Key.Key_3,
                    QtCore.Qt.Key.Key_4):
        H.key(page, keycode, H.CTRL)
    assert asked == [] and bar.sections.currentText() == "Data"


def test_unhosted_section_key_still_answers(qtbot):
    """The control: the same real key does reach a bar that is not hosted."""
    page, bar, _labels = _bar_page(qtbot)
    page.show()
    page.activateWindow()
    qtbot.waitActive(page)
    H.key(page, QtCore.Qt.Key.Key_2, H.CTRL)
    assert bar.sections.currentText() == "Analysis"


def test_detach_chrome_hides_the_bar_and_its_toolbar_and_keeps_them_alive(qtbot):
    page, bar, _labels = _bar_page(qtbot)
    page.show()
    toolbar = bar.parentWidget()
    assert isinstance(toolbar, QtWidgets.QToolBar) and toolbar.isVisible()
    bar.detach_chrome()
    assert not bar.isVisible() and not toolbar.isVisible()
    bar.primary_actions[0].click()          # hidden, still wired


def test_from_widget_copies_history_look_and_reset_from_a_real_appearance_widget(qtbot):
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    page = QtWidgets.QWidget()
    qtbot.addWidget(page)
    adapter = PageAdapter.from_widget(
        page, "x", widget, groups=[], format=FormatSpec(widget, lambda: ()),
        update=lambda: None)
    assert adapter.key == "x" and adapter.page is page and adapter.fallback is False
    assert adapter.history() == HistoryState(False, False, "Nothing to undo yet.",
                                             "Nothing to redo.", 0)
    widget.title.textEdited.emit("Hello")        # a user edit, through the box's own signal
    state = adapter.history()
    assert state.can_undo and "title" in state.undo_text and state.mark == 1
    assert adapter.history_changed is widget.historyChanged
    assert adapter.undo() is True and widget.state().title is None
    assert adapter.redo() is True and widget.state().title == "Hello"
    assert adapter.apply_look(PlotAppearance(font_family="Arial", title_size=8), True) is True
    assert widget.state().font_family == "Arial"
    adapter.reset_figure()
    assert widget.state().to_dict() == {}
    # a field the caller passes wins over the widget's own
    custom = PageAdapter.from_widget(page, "x", widget, groups=[], update=lambda: 1,
                                     format=FormatSpec(widget, lambda: ()), undo=lambda: "mine")
    assert custom.undo() == "mine"


def test_adapter_defaults_are_inert():
    adapter = PageAdapter("k", QtWidgets.QWidget(), [], FormatSpec(None, lambda: ()),
                          update=lambda: None)
    assert adapter.pending() == PendingState() and adapter.history() == HistoryState()
    assert adapter.undo() is False and adapter.redo() is False
    assert adapter.exports() == () and adapter.navigation() is None
    assert adapter.reveal_map == DEFAULT_REVEAL and adapter.reveal_map is not DEFAULT_REVEAL
    assert (adapter.chrome, adapter.canvas_actions, adapter.fallback) == ((), (), False)
    assert ExportEntry("t", lambda: None).kind == "figure"
    assert InspectorGroup("k", "K", []).expanded is True


def test_a_hosted_bar_leaves_the_visibility_of_a_late_section_widget_alone(qtbot):
    page, bar, labels = _bar_page(qtbot)
    page.show()
    bar.attach_inspector(lambda name: None)
    late = QtWidgets.QLabel("late")
    labels["Data"].parentWidget().layout().addWidget(late)
    bar.add_section_widget("Export", late)         # standalone this would hide it (Section is Data)
    QtWidgets.QApplication.processEvents()
    assert late.isVisible()
    page2, bar2, _labels = _bar_page(qtbot)
    page2.show()
    other = QtWidgets.QLabel("late")
    page2.centralWidget().layout().addWidget(other)
    bar2.add_section_widget("Export", other)
    QtWidgets.QApplication.processEvents()
    assert not other.isVisible()


def test_form_rows_moves_the_widgets_and_the_old_layout_lets_go(qtbot):
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    old = QtWidgets.QHBoxLayout(host)
    combo, spin = QtWidgets.QComboBox(), QtWidgets.QSpinBox()
    combo.hide()                                   # the old Section logic had hidden it
    for w in (QtWidgets.QLabel("Metric"), combo, QtWidgets.QLabel("Bins"), spin):
        old.addWidget(w)
    assert old.count() == 4
    form = form_rows([("Metric", combo), ("Bins", spin)])
    qtbot.addWidget(form)
    assert old.count() == 2                       # the two labels stay behind in the husk
    assert combo.parentWidget() is form and spin.parentWidget() is form
    layout = form.layout()
    assert isinstance(layout, QtWidgets.QFormLayout) and layout.rowCount() == 2
    assert layout.labelForField(combo).text() == "Metric"
    assert form.minimumWidth() == 0 and combo.minimumWidth() == 0
    form.show()
    assert combo.isVisible() and spin.isVisible()


def test_plot_workflow_imports_only_the_widget_transfer_leaf_from_gui():
    """Ownership transfer needs one leaf helper; pages/shell modules stay forbidden."""
    import ast
    source_path = pathlib.Path(__import__("nanoscope.gui.plot_workflow", fromlist=["x"]).__file__)
    allowed = (1, "widget_transfer", (("prepare_widget_move", None),))
    seen = []
    for node in ast.walk(ast.parse(source_path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.level or
                (node.module or "").startswith("nanoscope.gui")):
            imported = (node.level, node.module, tuple((n.name, n.asname) for n in node.names))
            assert imported == allowed, f"plot_workflow imports another GUI dependency: {ast.dump(node)}"
            seen.append(imported)
        elif isinstance(node, ast.Import):
            assert not any(n.name.startswith("nanoscope.gui") for n in node.names), ast.dump(node)
    assert seen == [allowed]
    # The permitted helper must remain a leaf, not become an indirect page/shell import.
    for node in ast.walk(ast.parse(source_path.with_name("widget_transfer.py").read_text())):
        if isinstance(node, ast.ImportFrom):
            assert not node.level and not (node.module or "").startswith("nanoscope.gui"), ast.dump(node)
        elif isinstance(node, ast.Import):
            assert not any(n.name.startswith("nanoscope.gui") for n in node.names), ast.dump(node)
