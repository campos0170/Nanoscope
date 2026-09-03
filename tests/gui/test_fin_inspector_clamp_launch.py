"""The Inspector's width ceiling is reachable FROM THE LAUNCH PATH, and the file says true things.

WHAT WAS WRONG. ``LAY-W1a`` added a clamp so the Inspector cannot open at a width between
its floor and the width its pages need — a dock that clips its own controls AND takes pixels
off the trace for the privilege. The clamp was correct and unreachable: it is applied by
``_apply_default_dock_widths``, which runs from ``_restore_settings``, which runs from
``__init__`` — where ``self.width()`` is the 1440 of the constructor's ``resize`` request
rather than a width any screen has agreed to. So ``_inspector_width_ceiling`` was only ever
asked about 1440 (where it answers 553 and never bites), and a user launching on a 1280 px
display got the pre-clamp behaviour: measured on a fresh profile before this fix, the
Inspector opened at 502 against a ceiling of 473 and left the signal viewer 446 px — the
Inspector the widest column, which is the one thing that ceiling exists to prevent.

AFTER: ``MainWindow.showEvent`` re-derives the opening widths against the width the window
actually got, and the opening layout stays provisional for one further turn of the event loop
so a window manager that shrinks the window as it maps it is still part of the opening rather
than a resize the layout ignores. At 1280x800 the Inspector now opens at its floor (460) and
the viewer keeps 488. Nothing is re-imposed afterwards: the last test here is the guard that
this hook did not turn into furniture moving under the pointer.

The two honesty tests at the end pin claims this module's source makes about numbers and
about sentences another object returns — the kind that are true when written and quietly stop
being true, which is the whole reason they are checked by machine.
"""

from __future__ import annotations

import ast
import inspect
import os
import re

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.main_window import (
    _DOCK_SEPARATOR_WIDTH,
    _FILE_DOCK_DEFAULT_WIDTH,
    _INSPECTOR_DEFAULT_WIDTH,
    MainWindow,
)
from nanoscope.gui.measure_panel import MeasurePanel

#: The display the defect was reported on, and the owner's own.
NARROW = (1280, 800)
OWNERS = (1512, 900)


@pytest.fixture
def fresh_profile(qapp):
    """A private, EMPTY QSettings namespace: a saved ``windowState`` would restore a dragged
    layout and this module would then be measuring the user's drag, not a first launch."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("FinInspectorClampLaunch")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


def _settle() -> None:
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()


def _launch(qtbot, size) -> MainWindow:
    """A first launch on a display of ``size``: construct, size, show — the app's own order
    (``gui/app.py`` does ``MainWindow()`` then ``show()``), with the screen's constraint
    standing in for a window manager's."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(*size)
    w.show()
    _settle()
    return w


# ================================================================================================
# the clamp is reachable where it matters: a launch
# ================================================================================================

def test_a_1280_launch_leaves_the_signal_viewer_the_widest_column(fresh_profile, qtbot):
    """THE PIN. On the display the defect was reported on, launched the way the app launches.

    Before the fix: Inspector 502, viewer 446. The Inspector was 56 px wider than the thing
    this application is for, and it was ALSO too narrow for its own pages (the ceiling at
    1280 is 473, under the 502 they need), so those 502 px bought a horizontal scrollbar
    anyway — the exact both-ways-lose the clamp was written to resolve.
    """
    win = _launch(qtbot, NARROW)
    inspector, viewer = win._inspector.width(), win.centralWidget().width()
    assert inspector == _INSPECTOR_DEFAULT_WIDTH, (
        f"launched at {NARROW[0]}x{NARROW[1]} on a fresh profile, the Inspector opened at "
        f"{inspector} px instead of its floor ({_INSPECTOR_DEFAULT_WIDTH}). The clamp in "
        f"_inspector_default_width is being asked about the constructor's 1440 again.")
    assert viewer > inspector, (
        f"the Inspector ({inspector}) is wider than the signal viewer ({viewer}) at "
        f"{NARROW[0]} px")
    assert viewer > win._file_dock.width()
    assert inspector <= win._inspector_width_ceiling(win.width())


def test_the_owners_display_is_unchanged(fresh_profile, qtbot):
    """The other side of the fix: where the window IS wide enough for both rules, nothing
    moved. The Inspector still opens at the measured knee that shows every page (502 on this
    tree) and the viewer still keeps the rest."""
    win = _launch(qtbot, OWNERS)
    inspector, viewer = win._inspector.width(), win.centralWidget().width()
    assert inspector == win._inspector_default_width()
    assert inspector > _INSPECTOR_DEFAULT_WIDTH, (
        "the derivation collapsed to the floor on a display that has room for the pages")
    assert viewer > inspector > win._file_dock.width()
    assert viewer == OWNERS[0] - 2 * _DOCK_SEPARATOR_WIDTH - _FILE_DOCK_DEFAULT_WIDTH - inspector


def test_the_opening_widths_are_not_re_imposed_on_a_later_resize(fresh_profile, qtbot):
    """THE GUARD ON THE FIX. ``resizeEvent`` now re-derives the opening layout — but only
    while it is still opening. Once that turn of the event loop is over the docks hold their
    absolute widths through any window resize, because a separator that slides because the
    window got 3 px narrower is furniture moving under the pointer."""
    win = _launch(qtbot, OWNERS)
    opened = win._inspector.width()
    assert not win._opening_layout_pending, "the opening layout never stopped being provisional"
    for width, height in ((1440, 900), NARROW, (1152, 800)):
        win.resize(width, height)
        _settle()
        assert win._inspector.width() == opened, (
            f"resized to {width}x{height}: the Inspector moved itself from {opened} to "
            f"{win._inspector.width()} px under the pointer")


# ================================================================================================
# ...and the sentences this module writes about numbers and about other objects' sentences
# ================================================================================================

def test_the_ceiling_docstrings_numbers_are_what_it_returns(fresh_profile, qtbot):
    """Every ``width -> ceiling`` pair stated in ``_inspector_width_ceiling``'s docstring,
    re-measured against the method. It used to open with ``1512 -> 589``, which is the
    arithmetic BEFORE the absolute cap the same docstring then mentions: the method returns
    560 there, and a reader taking 589 to a layout sum gets a window that does not add up."""
    win = _launch(qtbot, OWNERS)
    doc = inspect.getdoc(MainWindow._inspector_width_ceiling) or ""
    pairs = [(int(a), int(b)) for a, b in re.findall(r"(\d{3,5})\s*->\s*(\d{2,4})", doc)]
    assert len(pairs) >= 4, "the measured table vanished from the docstring"
    for width, claimed in pairs:
        assert win._inspector_width_ceiling(width) == claimed, (
            f"the docstring says {width} -> {claimed}; the method returns "
            f"{win._inspector_width_ceiling(width)}")


def test_the_gate_sentences_quoted_here_are_sentences_the_panel_returns():
    """``_sync_processed_export_action``'s docstring QUOTES the panel's refusals, and a quote
    is a promise that those characters are what a user sees. One of them was a sentence
    ``processed_export_gate`` has never returned — it dropped the "and no stored detection on
    this channel ran under one" clause, which is the whole reason the export can be live with
    every stage on the Preprocess form switched off."""
    doc = inspect.getdoc(MainWindow._sync_processed_export_action) or ""
    # A quoted SENTENCE: opens on a capital and closes on a full stop. Prose between two
    # quoted fragments (the docstring also mentions the empty-string success value) is not a
    # claim about what the panel says, and would otherwise be matched as one.
    quoted = [" ".join(q.split()) for q in re.findall(r'"([A-Z][^"]{20,}\.)"', doc)]
    assert quoted, "the docstring no longer quotes the gate at all"
    tree = ast.parse(inspect.getsource(MeasurePanel.processed_export_gate).lstrip())
    returned = {" ".join(node.value.split())
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for sentence in quoted:
        assert sentence in returned, (
            f"this docstring quotes a gate sentence processed_export_gate never returns:\n"
            f"  {sentence}")
