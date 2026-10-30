"""Picking a filter stage must not move the Preprocess column (pytest-qt, offscreen).

The "Signal filters" group is one combo over a QStackedWidget, and the Preprocess page's
settings scroll in their own InspectorScrollArea (``page_scrolls``, since c86b119; the
whole column scrolled before). ``QStackedWidget.setCurrentIndex`` hides the outgoing page,
and when the focused widget is ON that page Qt hands focus to the incoming page's
counterpart — inside a scroll area that is a scroll. Measured on a real MainWindow at
1512x900 with the apo ABF loaded: click the Low-pass page's ``enable`` box, then pick
Median from the selector directly above it, and the column went 0 -> 334 of 2020 in a
457 px viewport. The stage combo the pointer was resting on moved from window y=467 to
y=133 and a bare QWidget arrived under the pointer; the same switch with focus off the
page moved 0 px, isolating the cause to the focus transfer rather than the layout.

``DetectionPanel._raise_filter_page`` closes it by handing focus to the combo that raised
the page before the swap, so ``prev->isAncestorOf(fw)`` is false and Qt transfers nothing.
These tests drive all 8 stages in both directions, at two parked scroll positions, with
focus parked on the outgoing page every time, and pin the value at every step.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. -----------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QWidget

from nanoscope.gui.panels import DetectionPanel

N_STAGES = 8
# Short enough that the Preprocess page's own scroller scrolls: ~335 of ~670 px of
# settings in view at this panel size (~590 of ~1335 in the Inspector of a fresh
# 1512x900 window), i.e. every stage change happens while scrolled.
VIEWPORT = (480, 460)


def _scrolled_panel(qtbot) -> tuple[DetectionPanel, QScrollArea]:
    """A DetectionPanel on its Preprocess page, small enough for that page to scroll, and
    the page's own scroller.

    Not an outer QScrollArea around the panel: since c86b119 the panel holds one bounded
    InspectorScrollArea per mode page, whose 0x0 minimum lets the panel shrink to any
    viewport, so an outer area has nothing to scroll (range 0..0).
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.resize(*VIEWPORT)
    panel.show()
    qtbot.wait(60)
    panel.mode_tabs.setCurrentIndex(0)          # Preprocess
    qtbot.wait(60)
    area = panel.page_scrolls[panel._preprocess_page]
    holder = panel.filter_stack.parentWidget()
    while holder is not None and not isinstance(holder, QScrollArea):
        holder = holder.parentWidget()
    assert holder is area, (
        f"the stage editor scrolls in {holder!r}, not the Preprocess page's scroller")
    bar = area.verticalScrollBar()
    assert bar.maximum() >= 300, (
        "the column must be taller than the viewport, by at least the parked value, "
        f"for this test to mean anything (range 0..{bar.maximum()})")
    return panel, area


def _park_focus_on_page(panel: DetectionPanel, qtbot) -> QWidget:
    """Give focus to the first focusable widget on the SHOWN stage page, as a click would."""
    page = panel.filter_stack.currentWidget()
    for child in page.findChildren(QWidget):
        if (child.focusPolicy() & Qt.TabFocus) and child.isVisible() and child.isEnabled():
            child.setFocus(Qt.MouseFocusReason)
            qtbot.wait(10)
            assert panel.window().focusWidget() is child
            return child
    raise AssertionError("no focusable widget on the shown filter page")


@pytest.mark.parametrize("parked", [0, 300])
@pytest.mark.parametrize("order", [list(range(N_STAGES)), list(reversed(range(N_STAGES)))],
                         ids=["forwards", "backwards"])
def test_every_stage_change_leaves_the_column_where_the_user_parked_it(qtbot, parked, order):
    panel, area = _scrolled_panel(qtbot)
    bar = area.verticalScrollBar()
    bar.setValue(parked)
    qtbot.wait(20)
    moves = []
    for idx in order:
        holder = _park_focus_on_page(panel, qtbot)
        before = bar.value()
        panel.filter_selector.setCurrentIndex(idx)
        qtbot.wait(30)
        after = bar.value()
        if after != before:
            moves.append(
                f"stage {idx} ({panel.filter_selector.itemText(idx)}) with focus on "
                f"{holder.objectName() or type(holder).__name__}: {before} -> {after}")
        # the swap itself still happened
        assert panel.filter_stack.currentIndex() == idx
    assert not moves, "the Preprocess column moved under the pointer: " + "; ".join(moves)
    assert bar.value() == parked


def test_focus_goes_to_the_stage_selector_not_to_the_incoming_page(qtbot):
    """The partner widget is the combo the user just used — on screen by construction."""
    panel, _area = _scrolled_panel(qtbot)
    panel.filter_selector.setCurrentIndex(0)
    qtbot.wait(20)
    _park_focus_on_page(panel, qtbot)
    panel.filter_selector.setCurrentIndex(3)          # Median
    qtbot.wait(30)
    focused = panel.window().focusWidget()
    assert focused is panel.filter_selector, (
        f"focus landed on {focused!r}; the incoming page must not take it")
    assert not panel.filter_stack.isAncestorOf(focused)


def test_a_stage_change_with_focus_elsewhere_does_not_steal_it(qtbot):
    """The hand-off is only for focus that is about to be hidden — nothing else moves."""
    panel, area = _scrolled_panel(qtbot)
    bar = area.verticalScrollBar()
    panel.filter_selector.setCurrentIndex(0)
    qtbot.wait(20)
    panel.detector_combo.setFocus(Qt.MouseFocusReason)
    qtbot.wait(10)
    assert panel.window().focusWidget() is panel.detector_combo
    before = bar.value()
    panel.filter_selector.setCurrentIndex(5)          # Running mean
    qtbot.wait(30)
    assert panel.window().focusWidget() is panel.detector_combo
    assert bar.value() == before
    assert panel.filter_stack.currentIndex() == 5
