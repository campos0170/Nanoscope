"""FOLD-2R — the response-plot hint stopped displacing the control it was explaining.

THE REACHABILITY FIX THAT BROKE REACHABILITY. The previous round could not bring the live
``|H(f)|`` plot above the fold, so it named it instead: one line of static text at the
**head** of the "Signal filters" group. That line took a row, and the row it took was
paid for by the group's own first control. Measured on a real ``MainWindow`` on a fresh
QSettings profile, Preprocess (the tab the Analysis page opens on):

    1512x900 / 1440x900   filter_selector  content y=447..470   fold 457   13 px clipped
    1280x800              filter_selector  content y=447..470   fold 396   74 px under,
                                                                ``visibleRegion()`` EMPTY

``filter_selector`` is the control a user needs FIRST — it is how you choose *which* of the
seven filter stages the page below is editing, and the ● markers on its items are the only
place the enabled stages are named. Losing it to a caption about a plot 300 px further down
is the round's own anti-pattern: a disclosure that displaced the thing it disclosed.

The fix costs zero pixels: the caption rides the combo's row instead of taking one above it
(row height = max(combo 23, caption 15) = 23), so the selector is back at the head of the
group at y=366..389 — wholly on screen at all three sizes — and the sentence is still the
first thing read on the filter controls. What is pinned here:

* the stage selector is wholly above the fold, and actually paints, at all three sizes;
* the hint costs it nothing — the two share one row;
* FOLD-2's promise survives: something above the fold still names the plot, and the hint
  still floors the dock at 0 px (it must not be "fixed" by deleting the line);
* the Detect button, re-measured on the same windows, is still above the fold.

Every measurement is on a fresh QSettings profile: a developer who has dragged the
Inspector would otherwise measure THAT layout and see none of this.

2026-11-11 -- WHERE EACH PROMISE LIVES NOW. The numbers above are the layout of the day. Since
then the page lost its outer scroll area (c86b119: every mode page scrolls on its own, its
actions pinned above it), the stage editor moved out of the Inspector into a window of its own
(4a5d4a5, the Signal Conditioning window, since renamed the Preprocess workspace), and the
Preprocess page became an overview whose one door is "Open Preprocess workspace…" (8661d35).
In the shipped window ``filter_selector`` is never on screen: the control that says WHICH
stage is edited is the workspace's ``stage_list``. So:

* the stage-selector test follows the promise to where it lives: the door is whole on the page
  the Analysis tab opens on, with nothing to scroll to, and one press puts the workspace's stage
  list on screen, whole, on the stage the Inspector was on, holding the keyboard;
* the hint test and the FOLD-2 test pin the IN-PAGE editor, which ``set_signal_chain_hosted``
  promises to keep "exactly as before" for a panel that is not hosted. They build a bare
  ``DetectionPanel()`` (never hosted: the panel that promise speaks of) and measure against the
  scroll area of its Preprocess page; no window is built. They were four tests: the hint test
  ran at the three window sizes above on a real window with both hostings switched off, a
  layout nothing in the app ever reaches. A bare panel's rows sit at the same offsets at every
  panel height from 450 to 900 px (measured), so the size parametrisation measured nothing and
  is gone. No user of the shipped app sees that layout today (4a5d4a5, 8661d35), so these two
  can fail only when that in-page editor is edited; they go with it if it is ever removed;
* the Detect test measures the pinned action strip: Detect is whole at rest and does not move
  when the page under it is scrolled to its end. At the three screen sizes that page has nothing
  to scroll (its range is 0), so the scroll leg first takes the window down to its floor, where
  the page does scroll (asserted).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

import nanoscope.analysis.conditioning_recipes as cr
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.main_window import _INSPECTOR_DEFAULT_WIDTH, MainWindow
from nanoscope.gui.panels import DetectionPanel

from .inspector_layout_rig import assert_whole, rectangle, settle

#: The owner's laptop, then the two sizes the round is sanity-checked at.
_SCREENS = [(1512, 900), (1440, 900), (1280, 800)]

#: Words that would tell a reader the response plot exists (FOLD-2's own list).
_NAMES_THE_PLOT = ("response", "h(f)", "frequency")


@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace, so "first launch" really means first launch."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("WireFold2RStageSelector")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


class _Column:
    """The Analysis tab's Inspector on the real window, measured the way a user sees it.

    2026-11-11: there is no outer scroll area around the Analysis page any more (c86b119): each
    mode page scrolls in an area of its own (``detection.page_scrolls``), with its actions
    pinned above it, outside it, so they never scroll.
    """

    def __init__(self, win: MainWindow) -> None:
        self.win = win
        tabs = win._inspector.widget()
        for i in range(tabs.count()):
            if tabs.tabText(i).startswith("Analysis"):
                tabs.setCurrentIndex(i)
        settle(win)
        self.analysis: AnalysisPanel = win.findChild(AnalysisPanel)
        self.detection = self.analysis._detection
        self.mode_tabs = self.detection.findChild(QtWidgets.QTabWidget)

    def raise_page(self, title: str) -> None:
        for i in range(self.mode_tabs.count()):
            if self.mode_tabs.tabText(i) == title:
                self.mode_tabs.setCurrentIndex(i)
                settle(self.win)
                return
        raise AssertionError(f"no {title!r} page")


def _build(qtbot, size) -> _Column:
    """The real window at ``size``, the Analysis tab open, as shipped."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(*size)
    win.show()
    settle(win)
    col = _Column(win)
    col.detection.set_control_level("advanced")
    settle(win)
    return col


#: A bare panel: the Inspector's default width and 450 px of height, short on purpose (its
#: Preprocess page then shows ~325 px of the editor), so "above the fold" is held to a short
#: page. The editor's rows sit at the same offsets at every height from 450 to 900 px.
_BARE_PANEL = (_INSPECTOR_DEFAULT_WIDTH, 450)


class _BareEditor:
    """A bare ``DetectionPanel()`` -- never hosted -- on its Preprocess page: the in-page
    "Signal filters" editor, measured in the page's own scroll area, ``fold`` being the bottom
    of ITS viewport. No window, and no hosting call: the panel as ``set_signal_chain_hosted``
    leaves it before anyone hosts it."""

    def __init__(self, qtbot) -> None:
        self.detection = DetectionPanel()
        qtbot.addWidget(self.detection)
        self.detection.resize(*_BARE_PANEL)
        self.detection.show()
        settle(self.detection)
        self.detection.set_control_level("advanced")
        self.detection.show_signal_chain()
        settle(self.detection)
        self.area = self.detection.page_scrolls[self.detection._preprocess_page]
        self.inner = self.area.widget()

    @property
    def fold(self) -> int:
        return self.area.verticalScrollBar().value() + self.area.viewport().height()

    def yspan(self, w: QtWidgets.QWidget) -> tuple[int, int]:
        y = w.mapTo(self.inner, QtCore.QPoint(0, 0)).y()
        return y, y + w.height()

    def below_the_fold_by(self, w: QtWidgets.QWidget) -> int:
        return self.yspan(w)[1] - self.fold


@pytest.mark.parametrize("size", _SCREENS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_filter_stage_selector_is_above_the_fold(isolated_settings, qtbot, size):
    """Open the app, click Analysis: the control that says WHICH filter you are editing is
    reachable, whole, without a scroll — at 1512x900, 1440x900 and 1280x800.

    2026-11-11: that control is the Preprocess workspace's stage list now (4a5d4a5, 8661d35),
    one press from the page the Analysis tab opens on. Both halves of the promise are pinned:
    the door is whole on that page with nothing to scroll to, and the stage list it opens is
    whole, on the stage the Inspector was on, and holds the keyboard."""
    col = _build(qtbot, size)
    det = col.detection
    col.raise_page("Preprocess")
    # The shipped window: the in-page selector is not what is on screen.
    assert det.is_signal_chain_hosted() and not det.filter_selector.isVisible()
    door = det.preprocess_open_button
    assert door.text() == "Open Preprocess workspace…"
    assert_whole(door, det._preprocess_page)
    overview = det.preprocess_overview_scroll
    assert overview.isVisible()
    assert overview.verticalScrollBar().maximum() == 0 and overview.horizontalScrollBar().maximum() == 0, (
        f"the Preprocess page needs a scroll at {size[0]}x{size[1]}")

    stage = "detrend"          # not the first row, so "opens on the stage you were on" is a claim
    det.select_stage(stage)
    door.click()
    settle(col.win)
    ws = col.win._signal_conditioning
    assert ws is not None and ws.isVisible()
    assert_whole(ws.stage_list, ws)
    assert ws.stage_list.count() == len(cr.STAGE_KEYS)
    assert ws.stage_list.currentRow() == cr.STAGE_KEYS.index(stage)
    assert QtWidgets.QApplication.focusWidget() is ws.stage_list
    ws.hide()


def test_the_hint_costs_the_selector_no_row(isolated_settings, qtbot):
    """The mechanism, not just the outcome.

    The hint and the combo share ONE row, so the caption cannot push the control down
    again — and the selector is the first control in the group (ahead of the protocol
    combo and its three verbs), which is what "the control you need first" means.

    2026-11-11: pins the in-page editor of a bare, never-hosted ``DetectionPanel`` (see the
    module docstring). It ran at the three window sizes; the rows do not depend on the size of
    a bare panel, so it runs once.
    """
    page = _BareEditor(qtbot)
    det = page.detection
    assert page.area.isVisible()
    assert det.filter_selector.isVisible() and det.response_hint.isVisible()
    s_top, s_bottom = page.yspan(det.filter_selector)
    h_top, h_bottom = page.yspan(det.response_hint)
    assert s_top <= h_top and h_bottom <= s_bottom, (
        f"the response hint has a row of its own again: hint y={h_top}..{h_bottom}, "
        f"selector y={s_top}..{s_bottom}")
    assert s_top < page.yspan(det.protocol_combo)[0], (
        "the stage selector is no longer the first control of the Signal filters group")


@pytest.mark.parametrize("size", _SCREENS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_detect_button_is_still_above_the_fold(isolated_settings, qtbot, size):
    """Re-measured on the same three windows: fixing the Preprocess page gave nothing back
    on the Detection page.

    2026-11-11: Detect sits in the page's pinned action strip (c86b119), above the scroll
    area, so "above the fold" is stronger than it was: whole at rest, and not moved by
    scrolling the settings under it to their end. At these three screen sizes the Detection
    page has nothing to scroll (range 0 at all three), so the window is taken down to its
    floor before the page is scrolled, and the test requires that the page then scrolls."""
    col = _build(qtbot, size)
    col.raise_page("Detection")
    det = col.detection
    page, btn = det._detection_page, det.detect_button
    area = det.page_scrolls[page]
    bar = area.verticalScrollBar()
    assert bar.value() == 0
    assert_whole(btn, page)
    assert not area.isAncestorOf(btn) and det.page_actions[page].isAncestorOf(btn), (
        "Detect is in the scrolled settings again, not in the pinned strip above them")
    # 450 asks for less than the window allows; it stops at its floor (521 with Results open).
    col.win.resize(col.win.width(), 450)
    settle(col.win)
    assert bar.maximum() > 0, "the Detection page has nothing to scroll in the shortest window"
    assert_whole(btn, page)
    at_rest = rectangle(btn, col.win)
    bar.setValue(bar.maximum())
    settle(col.win)
    assert rectangle(btn, col.win) == at_rest, "Detect moved when the page scrolled"
    assert not btn.visibleRegion().isEmpty()


def test_fold_2_still_holds_the_plot_is_named_above_the_fold(isolated_settings, qtbot):
    """The previous round's promise is kept, not traded away.

    The hint may not be "fixed" by deleting it: a visible label above the fold on the
    Preprocess page still names the plot, it still elides rather than flooring the dock's
    width, and its full sentence is still in ``text()`` and the tooltip.

    2026-11-11: pins the in-page editor of a bare, never-hosted ``DetectionPanel`` (see the
    module docstring), 450 px tall: the plot is named above the fold of a short page.
    """
    page = _BareEditor(qtbot)
    det = page.detection
    assert page.area.isVisible() and det.response_hint.isVisible()
    seen = []
    for lab in det._preprocess_page.findChildren(QtWidgets.QLabel):
        # A label pinned above the scroll area is outside it: always on screen, never "below".
        if lab.visibleRegion().isEmpty() or not page.inner.isAncestorOf(lab):
            continue
        top, bottom = page.yspan(lab)
        if top >= 0 and bottom <= page.fold:
            seen.append(lab.text())
    assert [t for t in seen if any(w in t.lower() for w in _NAMES_THE_PLOT)], (
        f"nothing above the fold names the |H(f)| plot any more; visible labels: {seen}")
    hint = det.response_hint
    assert hint.minimumSizeHint().width() == 0, (
        f"the response hint floors the dock at {hint.minimumSizeHint().width()} px")
    assert hint.toolTip() == hint.text()
    assert page.below_the_fold_by(hint) <= 0
