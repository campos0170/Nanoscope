"""The Inspector opens on the page the work happens on, and wide enough to show it.

Measured on the display this application is used on — **1512 x 900** — with a QSettings
namespace of this module's own, so what is measured is the SHIPPED default layout and not
a geometry the developer happens to have dragged and saved.

WHAT WAS WRONG, measured before the fix (fresh profile, 1512x900, offscreen):

* the Inspector opened on tab 0, ``Calibration``. The Analysis page — the detector, the
  event table, every control this application exists to drive — was behind a tab, and had
  therefore never been laid out at all: ``detect_button.size()`` was Qt's unlaid-out
  ``640x480`` and ``visibleRegion()`` was empty. "Is the Detect button reachable" did not
  have an answer, because the button did not yet have a geometry.
* the dock opened at a flat 460 px, giving the Analysis controls a **420 px viewport for
  an 836 px control column**: ``horizontalScrollBar().maximum() == 416``. The protocol
  ``Save as… / Rename / Delete`` buttons sat at window x = 1624…1876 on a 1512 px screen.
  None of them has a menu item or a shortcut, so a scrollbar drag was the only door.

AFTER: the Inspector opens on Analysis, and its opening width is derived from the widest
page's own minimum rather than guessed (``MainWindow._inspector_default_width``) — 502 px
here and a 462 px viewport (it was 560/520 until the LAY-W1 fix-up traded the size hint for
the minimum and handed the difference back to the trace). Against the control column as it now stands (natural width 646,
minimum 354, after the sibling round capped the preset combo in ``gui/panels.py``) that is
``hbar.maximum() == 0``: the whole column, including every control the audit found
off-screen, inside the viewport with nothing to drag. The last test in this file is the pin
that fails if either half regresses — a narrower default here, or a wider column there.

Nothing here may move under the pointer: the opening tab is chosen at CONSTRUCTION, before
the window is shown and before anything is connected, and the opening widths are applied
only when there is no saved ``windowState`` (``test_window_layout`` pins that half).
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.main_window import (
    _INSPECTOR_CHROME_WIDTH,
    _INSPECTOR_DEFAULT_WIDTH,
    _INSPECTOR_MAX_DEFAULT_WIDTH,
    _INSPECTOR_NESTED_CHROME_WIDTH,
    MainWindow,
)

#: The owner's laptop screen. Every number in this module is measured at this size.
SCREEN = (1512, 900)
#: The width the signal viewer keeps whatever the Inspector asks for — the floor the cap in
#: ``main_window`` defends. 1512 - 12 (separators) - 320 (file manager) - 560 = 620.
VIEWER_FLOOR = 620


@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace: a real saved ``windowState`` would restore a dragged
    layout over the defaults this module exists to measure."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("LayInspectorOpensReachable")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


@pytest.fixture
def win(isolated_settings, qapp, qtbot) -> MainWindow:
    """A first launch on a 1512x900 display."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(*SCREEN)
    w.show()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    return w


def _controls_area(win: MainWindow) -> QtWidgets.QScrollArea:
    """The Analysis page's control column — the scroll area that was cutting controls off.

    Identified by content rather than by index: it is the one whose inner widget is taller
    than the viewport by a mile (~2400 px of detector controls), which no reordering of the
    page can move to a different child.
    """
    areas = [a for a in win._analysis_panel.findChildren(QtWidgets.QScrollArea)
             if a.widget() is not None]
    assert areas, "the Analysis page no longer wraps its sub-panels in scroll areas"
    return max(areas, key=lambda a: a.widget().sizeHint().height())


# ================================================================================================
# the page the work happens on is the page that opens
# ================================================================================================

def test_the_inspector_opens_on_the_analysis_page(win):
    """It opened on Calibration — by insertion order, not by a decision."""
    tabs = win._inspector_tabs
    assert tabs.currentWidget() is win._analysis_panel
    assert tabs.tabText(tabs.currentIndex()) == "Analysis"


def test_the_detector_has_a_real_geometry_on_launch(win):
    """Behind a tab, the whole detector was never laid out: Qt's unlaid-out 640x480 and an
    empty visible region. A control with no geometry cannot be reached or measured."""
    dp = win._analysis_panel.detection_panel
    assert dp.isVisible()
    # Laid out INSIDE the viewport it lives in, not at the default widget size.
    assert dp.width() != 640 or dp.height() != 480
    assert dp.width() <= _controls_area(win).widget().width()
    assert not dp.visibleRegion().isEmpty()


def test_calibration_keeps_its_own_door(win):
    """Choosing a different opening tab may not cost Calibration its way in: its tab is still
    there, and the reveal the workflow chip triggers still lands on it."""
    tabs = win._inspector_tabs
    assert tabs.count() == 3
    assert win._cal_scroll in [tabs.widget(i) for i in range(tabs.count())]
    win._reveal_calibration_panel()
    QtWidgets.QApplication.processEvents()
    assert tabs.currentWidget() is win._cal_scroll


# ================================================================================================
# ...and it opens wide enough to show it
# ================================================================================================

def test_the_opening_width_is_derived_from_the_widest_page(win):
    """Not a guessed constant: the widest page's own MINIMUM plus the dock's chrome, clamped.
    A flat 460 is what handed an 836 px control column a 420 px viewport.

    CHANGED by the LAY-W1 fix-up, and this is the only assertion in this file that moved.
    It read ``sizeHint()`` — the width at which a page is comfortable — which every page
    wanted more of than the ceiling allowed, so the ceiling silently became the width and the
    derivation decided nothing. Whether a control is reachable is decided by the page's
    ``minimumSizeHint()``; asking for that opens the dock at 502 rather than 560 and gives
    the 58 px back to the signal viewer. The number is unchanged in kind — still derived,
    still clamped between the same floor and the same ceiling — so this asserts the same
    thing about the new arithmetic. See tests/gui/test_lay2_inspector_width.py, which pins
    that 502 is the knee: nothing clipped at it, something clipped 12 px under it.
    """
    widest = max(a.widget().minimumSizeHint().width()
                 for a in win._inspector_tabs.findChildren(QtWidgets.QScrollArea)
                 if a.widget() is not None)
    wanted = widest + _INSPECTOR_CHROME_WIDTH + _INSPECTOR_NESTED_CHROME_WIDTH
    expected = max(_INSPECTOR_DEFAULT_WIDTH,
                   min(wanted, win._inspector_width_ceiling(win.width())))
    assert expected <= _INSPECTOR_MAX_DEFAULT_WIDTH
    assert win._inspector_default_width() == expected
    assert win._inspector.width() == pytest.approx(expected, abs=12)


def test_the_inspector_viewport_shows_the_control_column_it_can(win):
    """THE REACHABILITY PIN. The dock must open wide enough to show its control column —
    unless the column has grown past the cap, in which case it shows the cap's worth.

    This fails if anyone drops the opening width back towards 460 (a 420 px viewport against
    the 520 px this cap allows), which is exactly the regression that put five features
    off-screen.
    """
    area = _controls_area(win)
    content = area.widget().minimumSizeHint().width()
    allowance = _INSPECTOR_MAX_DEFAULT_WIDTH - _INSPECTOR_CHROME_WIDTH   # 520
    assert area.viewport().width() >= min(content, allowance), (
        f"the Inspector opened at {win._inspector.width()} px, giving its control column a "
        f"{area.viewport().width()} px viewport for {content} px of controls")


def test_the_signal_viewer_is_never_squeezed_to_a_strip(win):
    """The other end of the same clamp. A control stack that grows a long label may take a
    scrollbar; it may not take the viewer's width."""
    assert win._inspector.width() <= _INSPECTOR_MAX_DEFAULT_WIDTH
    assert win.centralWidget().width() >= VIEWER_FLOOR
    assert win.centralWidget().width() > win._file_dock.width()


def test_the_inspector_still_drags_down_to_its_tab_bar(win):
    """A wider DEFAULT is not a floor. The 2024 round's invariant — no page imposes a width
    on the dock — is what lets the user take it all back for the viewer."""
    assert win._inspector.minimumSizeHint().width() <= 220
    win.resizeDocks([win._inspector], [180], QtCore.Qt.Horizontal)
    QtWidgets.QApplication.processEvents()
    assert win._inspector.width() == pytest.approx(180, abs=12)


def test_the_opening_tab_does_not_move_under_the_pointer(isolated_settings, qapp, qtbot):
    """The opening tab is a construction-time choice, not a reveal: two launches agree, and
    nothing reselects a tab after the window is on screen."""
    seen = []
    for _ in range(2):
        w = MainWindow()
        qtbot.addWidget(w)
        w.resize(*SCREEN)
        w.show()
        QtWidgets.QApplication.processEvents()
        before = w._inspector_tabs.currentIndex()
        w.resize(1200, 800)                  # a resize is not a reveal
        QtWidgets.QApplication.processEvents()
        assert w._inspector_tabs.currentIndex() == before
        seen.append(before)
    assert seen[0] == seen[1]


# ================================================================================================
# the outstanding half, stated as a test rather than as a comment
# ================================================================================================

def test_the_analysis_controls_need_no_horizontal_scrollbar(win):
    """What "reachable" finally means, end to end: the whole control column inside the
    viewport the default layout gives it, with nothing to drag.

    It was 416 px of hidden controls. This is the assertion that fails if the opening width
    is narrowed again OR if a panel grows a control wider than the column can hold — the two
    halves that together put the protocol buttons at window x = 1624 on a 1512 px screen.
    """
    area = _controls_area(win)
    assert area.horizontalScrollBar().maximum() == 0, (
        f"the Analysis control column ({area.widget().width()} px, minimum "
        f"{area.widget().minimumSizeHint().width()}) does not fit its "
        f"{area.viewport().width()} px viewport")
