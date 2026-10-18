"""Where the Analysis page puts the merge controls — measured on the owner's 1512x900.

THE DEFECT, MEASURED BEFORE THE FIX on a real ``MainWindow`` resized to exactly 1512x900
with a fresh QSettings profile (the shipped default layout, not a saved one):

    Inspector ▸ Analysis ▸ controls column
        viewport 580 x 529   content 580 x 2540   vertical scrollbar max 2011
        MergePanel was the LAST child of that column
        "Merge all"                 at content y = 2472..2497   -> 1943 px below the fold
        "Merge events in selection" at content y = 2504..2529
        neither had a single pixel inside the viewport

and at the width the Inspector shipped at before this round (460 px, viewport 420) the same
two buttons were off the RIGHT edge as well — "Merge all" at x = 745..825 of an 836 px
content column, i.e. reachable only by dragging the horizontal scrollbar to 393 of 416 AND
the vertical one to its very end, at the same time. There is no menu item and no keyboard
shortcut for either button, so the batch merge — the only one in the application — had no
door at all. That is the shape this file exists to keep shut.

THE FIX IS A CONTAINER FIX, made in ``analysis_panel.py`` and nowhere else: the merge group
comes out of that 2540 px scrolled column and becomes a fixed strip directly above the event
table — the thing it edits — inside the splitter's lower pane. Measured after, same window,
same profile: the strip is 598 x 95 at panel y = 504, both of its scrollbar maxima are 0,
and every control in it is wholly inside its viewport and inside the window.

WHAT EACH TEST HERE IS GUARDING, since none of them is about the merge maths:

* the strip is out of the deep scroll and stays out;
* every merge control fits its viewport at 1512x900 — the definition of reachable — and
  keeps fitting at the narrower Inspector widths the dock can be dragged to;
* the strip imposes NO floor, on either axis. It is scroll-wrapped for the same reason
  ``MeasurePanel`` is (its "Merge events in selection" button cannot shrink below 252 px and
  that number would otherwise land in ``QDockWidget::minimumSize()``), and its minimum
  HEIGHT is taken away too, or ``MainWindow`` could no longer be made shorter than 459 px;
* nothing in it moves when the dock is dragged. The room a horizontal scrollbar will need is
  reserved before the scrollbar exists, so it arrives into empty space instead of re-flowing
  the group under the user's pointer.

CHANGE A MOVED ITS HOME, and this file was rewritten around it (it had been red since the
strip went behind a collapsed disclosure). The results half of the Analysis page — the event
table, its tabs and this strip — is now the Results dock under the trace, and the strip is
shown above the table by the Results toolbar's ``Merge…`` toggle (off by default). So the
fixture checks that toggle, "the pane" is the Events page, and "dragging the dock" means
narrowing the trace column the Results dock sits under, by widening the Inspector.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.merge_panel import MergePanel
from nanoscope.gui.panels import DetectionPanel, EventTable

#: The owner's laptop screen. Every number in this file was measured at exactly this size.
DISPLAY = (1512, 900)

#: The house number for "this widget does not pin the Inspector": the dock floored at 458 px
#: once and the measured value after the scroll-wrapping is ~137. Same constant
#: ``test_window_layout.py`` and ``test_measure_tab.py`` pin.
_SMALL_ENOUGH_TO_SHRINK = 220
_NARROW = 180

#: The viewport the Analysis page had at the Inspector width this application shipped with
#: before this round (dock 460 = frame 2 + vertical scrollbar 18 + viewport 420 + ...).
#: The merge strip must fit THIS, not merely today's wider default, so that reachability
#: does not depend on a sibling module's choice of opening width.
_SHIPPED_VIEWPORT = 420

#: Results-dock widths a user can reach with it docked under the trace, narrowest last (the
#: trace column itself floors at ~365 px; narrower is only reachable floating — see
#: test_lay2_merge_strip_reserved_room.py).
_DOCK_WIDTHS = (720, 620, 560, 460, 400)


# ================================================================================================
# fixtures
# ================================================================================================

@pytest.fixture
def isolated_settings(qapp):
    """A private, EMPTY QSettings namespace, so "first launch" really means first launch.

    ``MainWindow._restore_settings`` reads a saved ``windowState``; a developer who happened
    to float or resize a dock in the real app would otherwise have this file measure her
    layout instead of the shipped default. (conftest has already moved the whole store into
    a temporary directory; this only clears it.)"""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("LayAnalysisContainer")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


@pytest.fixture
def window(qapp, isolated_settings, qtbot):
    """A real, shown MainWindow at exactly 1512x900 with the Analysis tab raised."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(*DISPLAY)
    win.show()
    qapp.processEvents()
    win._inspector_tabs.setCurrentWidget(win._analysis_panel)
    qapp.processEvents()
    win.resize(*DISPLAY)
    qapp.processEvents()
    assert (win.width(), win.height()) == DISPLAY, (win.width(), win.height())
    win._analysis_panel.merge_toggle.setChecked(True)
    qapp.processEvents()
    yield win
    win.close()


# ================================================================================================
# helpers — the measurement vocabulary, spelled once
# ================================================================================================

def _ancestors(widget: QtWidgets.QWidget, stop: QtWidgets.QWidget) -> list[QtWidgets.QWidget]:
    chain, node = [], widget.parentWidget()
    while node is not None and node is not stop:
        chain.append(node)
        node = node.parentWidget()
    return chain


def _scroll_ancestor(widget: QtWidgets.QWidget,
                     stop: QtWidgets.QWidget | None) -> QtWidgets.QScrollArea:
    for node in _ancestors(widget, stop):
        if isinstance(node, QtWidgets.QScrollArea):
            return node
    raise AssertionError(f"{widget!r} is not inside a QScrollArea")


def _merge_scroll(win: MainWindow) -> QtWidgets.QScrollArea:
    return _scroll_ancestor(win._analysis_panel.merge_controls, None)


def _control_columns(win: MainWindow) -> list[QtWidgets.QScrollArea]:
    """The scrolled settings columns — one per mode page since the Inspector redesign —
    the merge group used to be at the bottom of. Found through the detector, never by index."""
    return list(win._analysis_panel.detection_panel.page_scrolls.values())


def _rect_in(widget: QtWidgets.QWidget, host: QtWidgets.QWidget) -> QtCore.QRect:
    return QtCore.QRect(widget.mapTo(host, QtCore.QPoint(0, 0)), widget.size())


def _merge_controls(win: MainWindow) -> list[tuple[str, QtWidgets.QWidget]]:
    mp = win._analysis_panel.merge_controls
    return [("window_ms", mp.window_ms),
            ("merge_all_btn", mp.merge_all_btn),
            ("merge_region_btn", mp.merge_region_btn)]


def _set_results_width(win: MainWindow, qapp, width: int) -> None:
    """Narrow the trace column — and with it the Results dock under it — by widening the
    Inspector."""
    room = win.width() - 2 * 6 - win._file_dock.width()
    win.resizeDocks([win._inspector], [room - width], QtCore.Qt.Horizontal)
    qapp.processEvents()


# ================================================================================================
# 1. the group is out of the deep scroll — and stays out
# ================================================================================================

def test_the_merge_group_is_not_inside_the_scrolled_control_column(window):
    """The regression itself. If someone re-adds ``self._merge`` to a settings column it
    lands far below the fold again and this fails."""
    panel = window._analysis_panel
    merge = panel.merge_controls
    assert isinstance(merge, MergePanel)

    columns = _control_columns(window)
    assert columns, "the detector has no scrolled settings columns at all"
    for column in columns:
        assert column.widget().findChild(MergePanel) is None, (
            "the merge group is back inside a settings column")
        assert _merge_scroll(window) is not column
    assert not window._inspector.isAncestorOf(merge), "the merge group is in the Inspector"


def test_the_merge_group_rides_with_the_event_table_it_edits(window):
    """Not merely "somewhere else" — on the Events page that holds the event table, above
    it. That adjacency is the whole reason the move is not an arbitrary shuffle."""
    panel = window._analysis_panel
    assert window._results_dock.isAncestorOf(panel.merge_controls)

    pane = _merge_scroll(window).parentWidget()
    # The Events tab is a frameless scroll area around the page (it clips a short dock).
    assert pane is panel.results_tabs.widget(0).widget(), "the strip is not on the Events page"
    assert pane.findChild(MergePanel) is panel.merge_controls
    table = pane.findChild(EventTable)
    assert table is panel._table
    assert panel.findChild(DetectionPanel) is panel.detection_panel

    strip = _rect_in(_merge_scroll(window), pane)
    assert strip.top() == 0, "the strip is the top of the page, not floating in it"
    assert strip.bottom() < _rect_in(table, pane).top(), "the strip sits ABOVE the table"


def test_the_merge_toggle_opens_the_strip_without_moving_itself(window, qapp):
    """Merge… is off by default; turning it on shows the strip BELOW the toggle, so the
    toggle itself never moves under the pointer — the table below gives up the height."""
    panel = window._analysis_panel
    toggle = panel.merge_toggle
    table_view = panel._table.view
    on_toggle, on_table = _rect_in(toggle, window), _rect_in(table_view, window)
    toggle.setChecked(False)
    qapp.processEvents()
    assert _merge_scroll(window).isHidden()
    off_toggle, off_table = _rect_in(toggle, window), _rect_in(table_view, window)
    assert off_toggle == on_toggle, "the toggle moved when the strip it opens closed"
    assert off_table.top() < on_table.top(), "the strip did not take its room from the table"
    toggle.setChecked(True)
    qapp.processEvents()
    assert _rect_in(toggle, window) == on_toggle
    assert _rect_in(table_view, window) == on_table


# ================================================================================================
# 2. reachable — the definition, measured
# ================================================================================================

def test_every_merge_control_is_wholly_inside_its_viewport_at_1512x900(window):
    """MEASURED BEFORE: "Merge all" at content y=2472..2497 against a 529 px viewport — zero
    pixels on screen."""
    area = _merge_scroll(window)
    viewport = area.viewport()
    box = QtCore.QRect(QtCore.QPoint(0, 0), viewport.size())

    for name, widget in _merge_controls(window):
        assert widget.isVisible(), name
        rect = _rect_in(widget, viewport)
        assert box.contains(rect), f"{name} {rect} is not inside the viewport {box}"


def test_the_merge_strip_needs_no_scrollbar_at_1512x900(window):
    """Any horizontal scrollbar maximum above 0 means content is cut off horizontally —
    the failure this whole pass is about. Both maxima are 0 here."""
    area = _merge_scroll(window)
    assert area.horizontalScrollBar().maximum() == 0
    assert area.verticalScrollBar().maximum() == 0
    assert not area.horizontalScrollBar().isVisible()
    assert area.widget().sizeHint().width() <= area.viewport().width()


def test_every_merge_control_is_inside_the_1512x900_window(window):
    """The end of the chain: a rectangle on the owner's actual screen. A control inside its
    viewport but outside the window is still a control nobody can click."""
    screen = QtCore.QRect(0, 0, *DISPLAY)
    for name, widget in _merge_controls(window):
        rect = _rect_in(widget, window)
        assert screen.contains(rect), f"{name} {rect} falls outside {screen}"


def test_the_merge_strip_fits_the_narrowest_docked_results_panel(window, qapp):
    """Reachability may not depend on how wide a SIBLING decides to leave the trace column.

    The merge group's own layout minimum is 252 px; it must fit the 420 px viewport the app
    once shipped with, and it does fit the narrowest the docked Results panel can be dragged
    to (the trace column's own floor)."""
    merge = window._analysis_panel.merge_controls
    assert merge.minimumSizeHint().width() <= _SHIPPED_VIEWPORT, (
        merge.minimumSizeHint().width())

    _set_results_width(window, qapp, _NARROW)
    area = _merge_scroll(window)
    assert area.horizontalScrollBar().maximum() == 0, window._results_dock.width()
    box = QtCore.QRect(QtCore.QPoint(0, 0), area.viewport().size())
    for name, widget in _merge_controls(window):
        assert box.contains(_rect_in(widget, area.viewport())), name


# ================================================================================================
# 3. it imposes no floor — neither axis
# ================================================================================================

def test_the_merge_strip_imposes_no_dock_width_floor(window):
    """Out of the scrolled column, ``MergePanel``'s own 252 px layout minimum would have
    become the bound ``QDockAreaLayout`` refuses to drag a separator past. The wrap is what
    stops it, exactly as it does for the measure panel."""
    merge = window._analysis_panel.merge_controls
    assert merge.minimumSizeHint().width() > _SMALL_ENOUGH_TO_SHRINK, (
        "if the group itself became narrow this test stopped proving anything")

    area = _merge_scroll(window)
    assert area.widget() is not None
    assert area.widgetResizable()
    assert area.minimumWidth() == 0
    assert area.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK

    assert window._analysis_panel._results_host.minimumSizeHint().width() \
        <= _SMALL_ENOUGH_TO_SHRINK
    assert window._results_dock.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK


def test_the_results_dock_still_drags_narrow_with_the_merge_strip_open(window, qapp):
    """Floating, nothing but the dock's own minimum bounds it — and the open strip adds
    nothing to that minimum."""
    dock = window._results_dock
    closed = dock.minimumSizeHint().width()
    window._analysis_panel.merge_toggle.setChecked(False)
    qapp.processEvents()
    assert dock.minimumSizeHint().width() == closed
    window._analysis_panel.merge_toggle.setChecked(True)
    dock.setFloating(True)
    dock.resize(_NARROW, 400)
    qapp.processEvents()
    assert dock.width() <= _NARROW + 4, dock.width()


def test_the_merge_strip_imposes_no_window_height_floor(window, qapp):
    """A ``QScrollArea`` reports ``minimumSizeHint() == 90x90`` whatever it holds, and that
    90 would ADD to the event table's own minimum inside this pane: measured, it pushed
    ``MainWindow.minimumSizeHint().height()`` 363 -> 459, so a window asked to be 450 px
    tall came back 459.

    Updated for change A: the open strip now holds its whole height and the Events page
    SCROLLS when the dock is shorter (a squeezed table painted its header over its own
    scrollbar), so the floor stops at that page. The claim is pinned where it matters: the
    dock's and the window's minimum heights do not move when the strip opens."""
    area = _merge_scroll(window)
    assert area.maximumHeight() < 16_777_215, "the strip must be capped, or it stretches"
    toggle = window._analysis_panel.merge_toggle
    opened = (window._results_dock.minimumSizeHint().height(),
              window.minimumSizeHint().height())
    toggle.setChecked(False)
    qapp.processEvents()
    closed = (window._results_dock.minimumSizeHint().height(),
              window.minimumSizeHint().height())
    assert opened == closed, f"the open strip moved the minimum heights {closed} -> {opened}"


def test_a_short_window_is_still_allowed(window, qapp, qtbot):
    """The suite measures geometry at 775x450; the merge strip may not make that impossible.

    Updated for change A: a bottom dock adds its own two-row header to the window's minimum
    height while it is open (the trace column keeps its own), and nothing while closed."""
    dock = window._results_dock
    assert dock.minimumSizeHint().height() <= 80, dock.minimumSizeHint()
    window.resize(775, 450)
    qapp.processEvents()
    assert window.height() <= 460 + dock.minimumSizeHint().height() + 6, window.height()
    dock.close()
    qtbot.wait(30)   # the minimum is recomputed on the layout pass the close posts
    window.resize(775, 450)
    qapp.processEvents()
    assert window.height() <= 460, window.height()


# ================================================================================================
# 4. nothing moves under the pointer
# ================================================================================================

def test_the_merge_controls_do_not_move_vertically_as_the_dock_is_dragged(window, qapp):
    """Three rounds of a previous campaign were lost to controls that shifted while the user
    was aiming at them. Narrowing the Results dock must not re-flow the group."""
    area = _merge_scroll(window)
    seen: dict[str, set[tuple[int, int]]] = {name: set() for name, _ in _merge_controls(window)}
    for width in _DOCK_WIDTHS:
        _set_results_width(window, qapp, width)
        for name, widget in _merge_controls(window):
            rect = _rect_in(widget, area.widget())
            seen[name].add((rect.top(), rect.height()))
    for name, positions in seen.items():
        assert len(positions) == 1, f"{name} moved vertically across dock widths: {positions}"


def test_the_merge_group_never_appears_or_disappears_with_state(window, qtbot):
    """``set_detection_present`` gates the buttons by ENABLING them. A group that showed and
    hid itself would move its neighbours exactly as a re-flow does."""
    merge = window._analysis_panel.merge_controls
    before = _rect_in(merge, window)
    assert merge.isVisible()
    assert not merge.merge_all_btn.isEnabled()      # no detection yet

    merge.set_detection_present(True)
    qtbot.wait(1)
    assert merge.merge_all_btn.isEnabled()
    assert merge.merge_region_btn.isEnabled()
    assert merge.isVisible()
    assert _rect_in(merge, window) == before, "the group moved when a detection arrived"


# ================================================================================================
# 5. the columns it came out of are unchanged in kind
# ================================================================================================

def test_the_control_column_is_still_wrapped_and_still_imposes_nothing(window):
    """The move must not have cost the Analysis page the 2024 round's fix."""
    for column in _control_columns(window):
        assert column.widgetResizable()
        assert column.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
        assert column.minimumWidth() == 0
        assert column.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK
