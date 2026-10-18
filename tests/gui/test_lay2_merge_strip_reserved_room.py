"""The merge strip's reserved scrollbar room is room the LAYOUT actually grants.

THE DEFECT, MEASURED BEFORE THE FIX on a real ``MainWindow`` at exactly 1512x900 with a
fresh QSettings profile:

    ``_pin_scroll_height`` set ``area.setMaximumHeight(natural + frame + bar)`` = 115 so
    that an as-needed horizontal scrollbar would "arrive into empty space". A maximum is
    not a grant. The enclosing ``QVBoxLayout`` hands a non-stretch child its ``sizeHint``,
    a ``QScrollArea``'s sizeHint is ``frame + content.sizeHint()``, and the caller's
    trailing ``addStretch(1)`` contributes 0 to that — so the strip was 97 px tall at
    EVERY Inspector width from 860 down to 140 and the reserved 18 px was never taken up.

    When the scrollbar was finally needed it therefore did the one thing the reservation
    existed to prevent: it took its 18 px out of the 95 px viewport.

        dock 280 -> viewport 258x95   hbar.max 0    vbar.max 0
        dock 260 -> viewport 220x77   hbar.max 32   vbar.max 18

    The 95 px of content no longer fitted the 77 px viewport, so a VERTICAL scrollbar
    appeared as well (stealing a further 18 px of width from a right-aligned "Merge all"),
    and "Merge events in selection" — content y = 59..84 — was bottom-clipped by 7 px and
    reachable only by scrolling down. Nothing about that needs a drag to reach:
    ``test_window_layout.py::test_a_saved_window_state_beats_the_defaults`` saves an
    Inspector width of 180, and a relaunch on that state opens straight into it.

THE FIX makes the reserved height and the granted height ONE quantity, derived once in
``_pin_scroll_height``: the same ``bar`` that sizes the cap is appended to the content's
layout as a trailing spacer whose *minimum* is zero. The area's sizeHint is then
``natural + bar + frame`` — equal to its maximum — so the pane grants it. Wide, the spacer
holds the empty room; narrow, the horizontal scrollbar takes the spacer's place and the
viewport is still ``natural`` tall. Measured after, same window: strip 115 px at every
width, viewport 113 -> 95 as the bar arrives, ``vbar.maximum() == 0`` throughout, no
control clipped and not one of them moving by a pixel.

The round's own file (``test_lay_analysis_container.py``) sweeps the same dock widths but
asserts only ``(top, height)`` of the controls in CONTENT coordinates — both of which stay
27/25 through the failure — so it could not see the viewport shrink or either scrollbar.
This file asserts the viewport and the scrollbars, which is where the defect lived.

The last test in this file is not a fix at all: it is the measurement that REJECTED the
sibling report of a "13 px empty" horizontal scrollbar on the event table. See its
docstring — that bar is in ``ScrollPerItem`` units and hides ~1000 px of real columns.

CHANGE A MOVED THE STRIP into the Results dock under the trace, behind the Results toolbar's
``Merge…`` toggle (off by default), and this file was rewritten around that home (it had been
red since the strip went behind a collapsed disclosure). The windows here open it first.
Docked under the trace the Results panel is never narrower than the trace column's own
~365 px floor, where the 252 px group needs no scrollbar at all; the widths at which the bar
arrives are reached the way a user reaches them — with the panel FLOATING.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.gui.main_window import MainWindow

#: The owner's laptop screen — every number in the docstrings above was measured here —
#: followed by the two sizes the round is sanity-checked at, because a fix that holds only
#: at 1512 is what the previous round shipped.
DISPLAY = (1512, 900)
SIZES = (DISPLAY, (1440, 900), (1280, 800))

#: Floating Results-panel widths a user can drag to. 270 and below are the ones that need the
#: horizontal scrollbar.
_DOCK_WIDTHS = (620, 560, 460, 380, 320, 280, 270, 260, 240, 200, 180)


# ================================================================================================
# fixtures
# ================================================================================================

@pytest.fixture
def isolated_settings(qapp):
    """A private, EMPTY QSettings namespace, so "first launch" really means first launch."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("Lay2MergeStrip")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


def _open(qapp, qtbot, size) -> MainWindow:
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(*size)
    win.show()
    qapp.processEvents()
    win._inspector_tabs.setCurrentWidget(win._analysis_panel)
    qapp.processEvents()
    win.resize(*size)
    qapp.processEvents()
    assert (win.width(), win.height()) == tuple(size), (win.width(), win.height())
    win._analysis_panel.merge_toggle.setChecked(True)
    qapp.processEvents()
    return win


@pytest.fixture
def window(qapp, isolated_settings, qtbot):
    """A real, shown MainWindow at exactly 1512x900 with the Analysis tab raised."""
    win = _open(qapp, qtbot, DISPLAY)
    yield win
    win.close()


# ================================================================================================
# helpers
# ================================================================================================

def _strip(win: MainWindow) -> QtWidgets.QScrollArea:
    """The merge strip's scroll area, found through the widget it holds — never by index."""
    node = win._analysis_panel.merge_controls.parentWidget()
    while node is not None and not isinstance(node, QtWidgets.QScrollArea):
        node = node.parentWidget()
    assert node is not None, "the merge group is not inside a QScrollArea any more"
    return node


def _controls(win: MainWindow) -> list[tuple[str, QtWidgets.QWidget]]:
    mp = win._analysis_panel.merge_controls
    return [("window_ms", mp.window_ms),
            ("merge_all_btn", mp.merge_all_btn),
            ("merge_region_btn", mp.merge_region_btn)]


def _rect_in(widget: QtWidgets.QWidget, host: QtWidgets.QWidget) -> QtCore.QRect:
    return QtCore.QRect(widget.mapTo(host, QtCore.QPoint(0, 0)), widget.size())


def _set_dock_width(win: MainWindow, qapp, width: int) -> None:
    """Float the Results panel at ``width`` AND LET THE LAYOUT SETTLE.

    A single ``processEvents`` after a resize is not enough: the scroll area's
    scrollbars are re-evaluated on a later pass, so the viewport still reports its old
    height and a test that measures immediately reads the layout as it was. Measured on the
    pre-fix build at dock 260: without the wait, viewport 238x95 / hbar 14 / vbar 0 (the
    stale frame); with it, 220x77 / hbar 32 / vbar 18 — the defect. ``qWait`` spins the
    event loop, which is what the extra pass needs."""
    dock = win._results_dock
    if not dock.isFloating():
        dock.setFloating(True)
        QTest.qWait(30)
    dock.resize(width, 420)
    QTest.qWait(30)
    qapp.processEvents()


def _bar_height(area: QtWidgets.QScrollArea) -> int:
    return area.horizontalScrollBar().sizeHint().height()


# ================================================================================================
# 1. the reservation is granted, not merely declared
# ================================================================================================

def test_the_strip_is_granted_the_height_it_reserves(window):
    """THE DEFECT. Pre-fix: maximumHeight 115, height 97 — the 18 px existed only as a cap.

    The two numbers are one quantity now, so this compares them rather than re-deriving a
    magic 115: whatever ``natural + frame + bar`` comes to on this platform's style, the
    strip must actually be that tall."""
    area = _strip(window)
    content = area.widget()

    #: what the group actually needs — the spacer the fix appends has minimum 0, so the
    #: content's minimumSizeHint is still the group's own height.
    natural = content.minimumSizeHint().height()
    frame = 2 * area.frameWidth()
    bar = _bar_height(area)

    assert area.maximumHeight() == natural + frame + bar, (
        "the cap is no longer natural + frame + scrollbar")
    assert area.height() == area.maximumHeight(), (
        f"the strip is {area.height()} px tall but reserves {area.maximumHeight()}: the "
        "room a horizontal scrollbar will need is declared and never granted")
    assert area.sizeHint().height() == area.maximumHeight(), (
        "the layout hands a non-stretch child its sizeHint — the reservation has to be IN "
        "that sizeHint or the pane will not hand it over")


def test_the_reserved_room_is_empty_space_below_the_group_not_inside_it(window):
    """The strip is 18 px taller than the control group, and every one of those pixels is
    BELOW it. If the reservation were made by stretching the group, the controls would sit
    at different y's — which is the failure this whole round exists to keep shut."""
    area = _strip(window)
    group = window._analysis_panel.merge_controls
    bar = _bar_height(area)

    assert area.viewport().height() == group.height() + bar
    assert _rect_in(group, area.viewport()).top() == 0
    assert group.height() == group.sizeHint().height(), (
        "the group was stretched to fill the reserved room instead of leaving it empty")


# ================================================================================================
# 2. when the scrollbar finally arrives it spends the reservation
# ================================================================================================

def test_the_horizontal_scrollbar_arrives_into_the_reserved_room(window, qapp):
    """The transition the defect was measured across: dock 280 -> 260.

    Pre-fix the viewport went 258x95 -> 220x77 (the bar ate the content). Post-fix it goes
    258x113 -> 238x95: the bar takes the reserved 18 px and the content still fits."""
    area = _strip(window)
    bar = _bar_height(area)
    hbar = area.horizontalScrollBar()

    _set_dock_width(window, qapp, 320)
    assert hbar.maximum() == 0, "the strip should not need a scrollbar at dock 320"
    wide = area.viewport().height()

    _set_dock_width(window, qapp, 200)
    assert hbar.maximum() > 0, "dock 200 is narrower than the merge group; expected a bar"
    narrow = area.viewport().height()

    assert wide - narrow == bar, (
        f"the scrollbar took {wide - narrow} px where {bar} px were reserved")
    assert narrow == area.widget().minimumSizeHint().height(), (
        "the viewport no longer fits the content once the scrollbar is up")


@pytest.mark.parametrize("width", _DOCK_WIDTHS)
def test_no_vertical_scrollbar_ever_appears_on_the_merge_strip(window, qapp, width):
    """The second half of the defect. A content taller than its viewport gave the strip a
    VERTICAL scrollbar too (max 18 at dock 260 and below), which stole another 18 px of
    width from the right-aligned "Merge all" — a strip 95 px tall scrolling vertically
    inside 77 px. There is never a reason for this strip to scroll vertically."""
    area = _strip(window)
    _set_dock_width(window, qapp, width)
    assert area.verticalScrollBar().maximum() == 0, (
        f"dock {width}: the merge strip scrolls vertically ({area.viewport().height()} px "
        f"of viewport onto {area.widget().height()} px of content)")


@pytest.mark.parametrize("width", _DOCK_WIDTHS)
def test_no_merge_control_is_clipped_at_any_dock_width(window, qapp, width):
    """"Merge events in selection" occupied y=59..84 of a 77 px viewport pre-fix, i.e. its
    bottom 7 px were behind the horizontal scrollbar. Vertical containment is the claim
    here; horizontal is the sibling file's, and is what the horizontal bar is FOR."""
    area = _strip(window)
    _set_dock_width(window, qapp, width)
    for name, widget in _controls(window):
        rect = _rect_in(widget, area.viewport())
        assert rect.top() >= 0 and rect.bottom() < area.viewport().height(), (
            f"dock {width}: {name} at y={rect.top()}..{rect.bottom()} is clipped by a "
            f"{area.viewport().height()} px viewport")


def test_nothing_in_the_strip_moves_as_the_dock_is_dragged(window, qapp):
    """NOTHING MAY MOVE UNDER THE POINTER — asserted where the previous round's sweep could
    not see it: the strip's own outer height (the event table below it must not jump) and
    the controls' position in VIEWPORT coordinates, which is where the eye and the mouse
    are. Content coordinates stayed 27/25 right through the defect."""
    area = _strip(window)
    heights: set[int] = set()
    seen: dict[str, set[tuple[int, int]]] = {name: set() for name, _ in _controls(window)}

    for width in _DOCK_WIDTHS:
        _set_dock_width(window, qapp, width)
        heights.add(area.height())
        for name, widget in _controls(window):
            rect = _rect_in(widget, area.viewport())
            seen[name].add((rect.top(), rect.height()))

    assert heights == {area.maximumHeight()}, (
        f"the strip's height changes with the dock width: {sorted(heights)}")
    for name, places in seen.items():
        assert len(places) == 1, f"{name} moved vertically as the dock was dragged: {places}"


# ================================================================================================
# 3. the reservation adds no floor, and holds on the other two screens
# ================================================================================================

def test_the_reservation_adds_no_height_floor(window):
    """The spacer's minimum is zero. A minimum here would
    propagate into ``QDockWidget::minimumSize()`` and take the whole window's minimum
    height up with it (measured: 363 -> 459 the last time this strip acquired a floor).

    Measured as the strip's CONTRIBUTION, not as an absolute window minimum: the window's
    own number is a sibling module's business and moves when the default dock widths do,
    while what this fix must not do is add to it. Same idiom as
    ``test_lay_analysis_container.py::test_the_merge_strip_imposes_no_window_height_floor``.
    """
    from nanoscope.gui.panels import EventTable

    area = _strip(window)
    pane = area.parentWidget()
    table = pane.findChild(EventTable)
    assert table is not None, "the strip is no longer in the event table's pane"
    # Updated for change A: the open strip holds its height and the Events page scrolls
    # instead of squeezing the table, so the floor stops at that page; pinned at the dock.
    dock = window._results_dock
    opened = (dock.minimumSizeHint().height(), window.minimumSizeHint().height())
    window._analysis_panel.merge_toggle.setChecked(False)
    QtWidgets.QApplication.processEvents()
    closed = (dock.minimumSizeHint().height(), window.minimumSizeHint().height())
    assert opened == closed, (
        f"the open strip moved the dock/window minimum heights {closed} -> {opened}; the "
        "18 px of reserved scrollbar room became a floor")


@pytest.mark.parametrize("size", SIZES)
def test_the_strip_holds_on_the_other_screens_too(qapp, isolated_settings, qtbot, size):
    """1440x900 and 1280x800 as well as the owner's 1512x900 — a fix that holds only at the
    owner's width is what this round is cleaning up after."""
    win = _open(qapp, qtbot, size)
    try:
        area = _strip(win)
        assert area.height() == area.maximumHeight()
        assert area.verticalScrollBar().maximum() == 0
        for name, widget in _controls(win):
            rect = _rect_in(widget, area.viewport())
            assert area.viewport().rect().contains(rect), f"{size}: {name} at {rect}"
            in_window = _rect_in(widget, win)
            assert win.rect().contains(in_window), f"{size}: {name} outside the window"
    finally:
        win.close()


# ================================================================================================
# 4. the event table's horizontal scrollbar — the REJECTED report, kept as a measurement
# ================================================================================================

def test_the_event_tables_horizontal_bar_is_columns_not_a_few_stray_pixels(window):
    """WHY THE "13 px of empty range" REPORT WAS REJECTED, in the form of an assertion.

    A ``QTableView`` scrolls ``ScrollPerItem`` by default, so ``horizontalScrollBar()
    .maximum()`` is a COLUMN INDEX, not a pixel count. Measured at 1512x900, Inspector 560:
    the table's header is 1554 px long over a 520 px viewport — 1034 px and 13 whole
    columns behind that bar, and scrolling it to its maximum puts column 0 at x = -1034.
    (With a detection loaded: header 2145 px, viewport 474, maximum 19.) So the bar is not
    "13 px of empty range that buys nothing": it is the only way to reach two thirds of a
    twenty-column data table inside a 560 px dock.

    This test exists so nobody closes that report the cheap way. Hiding the bar
    (``ScrollBarAlwaysOff``) over content that really is 1034 px wider than its viewport
    does not remove an overhang, it removes the columns; and the only honest way to make
    the overhang small enough to delete would be to stop showing measured columns, which is
    a change to what the application says, not to where it says it."""
    view = window._analysis_panel._table.view
    header = view.horizontalHeader()
    hbar = view.horizontalScrollBar()

    assert view.horizontalScrollMode() == QtWidgets.QAbstractItemView.ScrollPerItem
    overhang = header.length() - view.viewport().width()
    assert overhang > 200, (
        f"only {overhang} px of columns overhang the viewport — if this ever really is a "
        "handful of pixels, close it by sizing the columns, not by hiding the bar")
    assert hbar.maximum() < header.count(), (
        "maximum() is a column index in ScrollPerItem mode; a value at or above the column "
        "count means the mode changed and this file's reasoning has to be re-measured")
    assert view.horizontalScrollBarPolicy() != QtCore.Qt.ScrollBarAlwaysOff, (
        f"the bar was switched off over {overhang} px of columns the user can no longer "
        "reach — a hidden scrollbar over content that really is wider is worse")
