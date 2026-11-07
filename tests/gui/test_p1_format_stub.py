"""Phase 1 Foundation: the frozen names of the Format group's files (``format_inspector``,
``format_toolbar``) and what their stubs promise until the Format group replaces the bodies.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gc

from PySide6 import QtCore, QtWidgets

from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.format_inspector import (ElementList, ElementRef, FormatClipboard, FormatPane,
                                            SelectionHub, SetHereDot, resolve)
from nanoscope.gui.format_toolbar import MiniTextToolbar, SelectionOutline
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plot_workflow import FormatSpec


def test_element_ref_is_a_plain_value():
    ref = ElementRef("", "axis.bottom")
    assert ref.plot == "" and ref.key == "axis.bottom" and ref == ElementRef("", "axis.bottom")
    assert ElementRef("psd", "title") != ref


def test_the_hub_selects_clears_and_emits_once_per_change(qtbot):
    hub = SelectionHub()
    seen = []
    hub.selectionChanged.connect(lambda ref, canvas: seen.append((ref, canvas)))
    assert hub.current() is None
    hub.select(ElementRef("", "title"), from_canvas=True)
    hub.select(ElementRef("", "title"), from_canvas=True)       # no change: no signal
    assert seen == [(ElementRef("", "title"), True)] and hub.current() == ElementRef("", "title")
    hub.clear()
    hub.clear()
    assert seen[-1] == (None, False) and len(seen) == 2 and hub.current() is None
    assert hub.refs() == []


def test_the_hub_keeps_controllers_weakly(qtbot):
    hub = SelectionHub()
    controller = QtCore.QObject()
    hub.register(controller)
    hub.register(controller)
    assert len(hub._controllers) == 1
    del controller
    gc.collect()
    other = QtCore.QObject()
    hub.register(other)                           # a dead reference is swept on the next register
    assert [ref() is other for ref in hub._controllers] == [True]


def test_resolve_reports_where_a_value_comes_from():
    state = PlotAppearance().with_text_style("title", TextStyle(bold=True))
    assert resolve(state, None, ElementRef("", "title"), "bold") == (True, "element")
    assert resolve(state, None, ElementRef("", "title"), "italic") == (None, "default")
    defaults = TextStyle(size_pt=11.0)
    assert resolve(state, defaults, ElementRef("", "title"), "size_pt") == (11.0, "default")


def test_the_clipboard_is_in_memory_only():
    clipboard = FormatClipboard()
    assert clipboard.get() is None
    clipboard.put({"bold": True})
    assert clipboard.get() == {"bold": True}
    clipboard.clear()
    assert clipboard.get() is None


def test_the_stub_widgets_exist_and_do_nothing_on_screen(qtbot):
    outline, toolbar = SelectionOutline(), MiniTextToolbar()
    assert outline.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not outline.isVisible() and not toolbar.isVisible()
    dot = SetHereDot()
    assert (dot.source(), dot.text()) == ("default", "default")
    assert ElementList().accessibleName() == "Figure elements"
    for w in (outline, toolbar):
        w.deleteLater()


def test_the_stub_pane_has_the_frozen_methods_and_shows_the_extras_before_the_appearance(qtbot):
    label, other = QtWidgets.QLabel("extra"), QtWidgets.QLabel("appearance")
    pane = FormatPane(FormatSpec(other, lambda: (), extras={"figure": [label]}), SelectionHub())
    qtbot.addWidget(pane)
    pane.show()
    qtbot.waitExposed(pane)
    assert label.isVisible() and not other.isVisible()
    assert pane.panel_for("figure") is pane._body and pane.elements() == ()
    pane.show_element(ElementRef("", "title"))
    pane.show_element("title")
    pane.refresh()
