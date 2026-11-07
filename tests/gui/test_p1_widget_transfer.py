"""Known-item ownership during inspector moves; no allocation churn or crash loops."""
from __future__ import annotations

import nanoscope  # noqa: F401 -- bootstrap before Qt
import pytest
import shiboken6
from PySide6 import QtWidgets

from nanoscope.gui.analysis_inspector import DataPane, InspectorHost
from nanoscope.gui.format_inspector import FormatPane, SelectionHub
from nanoscope.gui.plot_workflow import FormatSpec, InspectorGroup, PageAdapter, form_rows
from nanoscope.gui.widget_transfer import prepare_widget_move


def _source(qtbot, kind, widget):
    source = QtWidgets.QStackedWidget() if kind == "stacked" else QtWidgets.QWidget()
    qtbot.addWidget(source)
    if kind == "stacked":
        layout = source.layout()
    elif kind == "nested":
        root = QtWidgets.QVBoxLayout(source)
        layout = QtWidgets.QHBoxLayout()
        root.addLayout(layout)
    else:
        layout = {"box": QtWidgets.QVBoxLayout, "grid": QtWidgets.QGridLayout,
                  "form": QtWidgets.QFormLayout}[kind](source)
    other = QtWidgets.QLabel("Keep this control")
    if kind == "grid":
        layout.addWidget(widget, 1, 2)
        layout.addWidget(other, 2, 2)
    elif kind == "form":
        layout.addRow("Move", widget)
        layout.addRow("Keep", other)
    else:
        layout.addWidget(widget)
        layout.addWidget(other)
    held = layout.itemAt(layout.indexOf(widget))
    unrelated = layout.itemAt(layout.indexOf(other))
    return source, layout, held, other, unrelated


@pytest.mark.parametrize("kind", ["box", "nested", "grid", "form", "stacked"])
def test_exact_old_item_is_invalidated_without_changing_the_control_or_other_items(qtbot, kind):
    edit = QtWidgets.QSpinBox()
    edit.setValue(41)
    heard = []
    edit.valueChanged.connect(heard.append)
    source, old, held, other, unrelated = _source(qtbot, kind, edit)
    destination = QtWidgets.QWidget()
    qtbot.addWidget(destination)
    new = QtWidgets.QVBoxLayout(destination)

    prepare_widget_move(edit)
    assert not shiboken6.isValid(held)
    assert old.indexOf(edit) == -1
    assert shiboken6.isValid(edit) and edit.parentWidget() is source
    assert shiboken6.isValid(unrelated) and unrelated.widget() is other
    assert old.indexOf(other) >= 0
    new.addWidget(edit)

    assert edit.parentWidget() is destination and new.indexOf(edit) >= 0
    assert edit.value() == 41 and heard == []
    edit.setValue(42)
    assert heard == [42]


@pytest.mark.parametrize("has_parent", [False, True])
def test_a_control_without_a_layout_is_left_alive_and_can_move(qtbot, has_parent):
    parent = QtWidgets.QWidget() if has_parent else None
    if parent is not None:
        qtbot.addWidget(parent)
    edit = QtWidgets.QLineEdit("Keep the value", parent)
    destination = QtWidgets.QWidget()
    qtbot.addWidget(destination)
    layout = QtWidgets.QVBoxLayout(destination)
    prepare_widget_move(edit)
    assert shiboken6.isValid(edit) and edit.parentWidget() is parent
    layout.addWidget(edit)
    assert edit.text() == "Keep the value" and edit.parentWidget() is destination


def test_format_clears_the_old_scroll_content_pointer_before_moving_its_widget(qtbot):
    area = QtWidgets.QScrollArea()
    qtbot.addWidget(area)
    edit = QtWidgets.QLineEdit("Keep scroll content")
    area.setWidget(edit)
    pane = FormatPane(FormatSpec(edit, lambda: ()), SelectionHub())
    qtbot.addWidget(pane)
    pane._ensure_built()
    assert area.widget() is None
    assert pane.isAncestorOf(edit) and edit.text() == "Keep scroll content"
    area.setWidget(QtWidgets.QLabel("Replacement"))
    assert shiboken6.isValid(edit) and edit.text() == "Keep scroll content"


@pytest.mark.parametrize("boundary", ["data", "format", "form_rows"])
def test_each_new_inspector_move_releases_its_old_item_and_preserves_connections(qtbot, boundary):
    button = QtWidgets.QPushButton("Original action")
    clicked = []
    button.clicked.connect(lambda: clicked.append(True))
    source, old, held, other, unrelated = _source(qtbot, "nested", button)
    if boundary == "data":
        host = DataPane()
        qtbot.addWidget(host)
        host.populate([InspectorGroup("test", "Test", [button])])
    elif boundary == "format":
        host = FormatPane(FormatSpec(None, lambda: (), extras={"figure": [button]}),
                          SelectionHub())
        qtbot.addWidget(host)
        host._ensure_built()
    elif boundary == "form_rows":
        host = form_rows([("Action", button)])
        qtbot.addWidget(host)
    assert not shiboken6.isValid(held)
    assert old.indexOf(button) == -1 and host.isAncestorOf(button)
    assert shiboken6.isValid(unrelated) and unrelated.widget() is other
    assert button.text() == "Original action"
    button.click()
    assert clicked == [True]
