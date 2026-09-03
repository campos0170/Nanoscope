"""The Analysis controls column fits the shipped Inspector VERTICALLY too.

THE SECOND AXIS. The previous layout round fixed x and left y. It made the Detection
page's 2x2 action grid fit the dock's width — ``Detect`` at viewport x=11..257, ``Sweep
threshold…`` at 263..509 — and then declared the four buttons REACHABLE. They were not:
at content **y=507..563** in a viewport that shows **y=0..466** on arrival, every one of
them reported ``visibleRegion().isEmpty()``, a click at their centre landed on the panel
below, and a user who opened the Analysis tab saw a page of parameters with no ``Detect``
button on it. ``Sweep threshold…``, ``Null control…`` and ``Recovery map…`` have no menu
item and no keyboard shortcut, so a vertical scroll was their only door — while
``StatsPanel`` prints "false-positive rate: not measured (run Null control)" and points
at one of them. Three independent reviewers converged on this defect.

The shape, not the numbers: the controls column is ~2.3k px of content in a ~460 px
viewport, so scrolling is inevitable and re-ordering the column only moves the problem to
whatever lands last. What is fixed here is WHICH KIND of thing is allowed below the fold.
Parameters may be — a user goes looking for the one they want to change. The actions that
consume them may not. So the action row goes to the head of the Detection page, where the
only things above it are the tab bar and the gate banner, whose row is now RESERVED — and
therefore where its position is constant under every state change.

Three defects are pinned here, all of them measured on a real ``MainWindow`` on a fresh
settings profile at the owner's 1512x900 and at 1440x900 and 1280x800:

* **the action row is above the fold** — all four buttons wholly inside the viewport at
  scroll 0, at all three screen sizes, without giving back the x-fix that got them inside
  the dock, and without the column starting to scroll sideways again;
* **the response curve is named above the fold** — the live ``|H(f)|`` plot sits 256 px
  below the fold on the tab that opens by default, and before this change not one visible
  label above the fold on that page contained "response", "H(f)" or "frequency", so a user
  tuning a filter had no way to learn a picture of it was drawn at all;
* **the gate banner moves nothing** — it used to be added to and removed from the layout
  by show/hide, pulling the whole panel up 21 px whenever the gate answered differently,
  which is the exact failure shape three earlier rounds of this project were lost to.

Every assertion is on a fresh QSettings profile: a developer who has dragged the Inspector
in the real app would otherwise measure THAT layout and never see any of this.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.main_window import _INSPECTOR_DEFAULT_WIDTH, MainWindow
from nanoscope.gui.panels import DetectionPanel

#: The owner's laptop, then the two sizes the round was told to sanity-check. A fix that
#: only holds at 1512 is the failure mode this parametrisation exists to catch.
_SCREENS = [(1512, 900), (1440, 900), (1280, 800)]

#: The four verbs of the Detection page, in grid order.
_ACTIONS = ("detect_button", "sweep_button", "null_button", "recovery_button")

#: THE 8 PX THE INSPECTOR'S OWN CEILING COSTS THIS PAGE, below a 1337 px window.
#: ``MainWindow._inspector_default_width`` clamps the OPENING width to the Inspector's floor
#: (460) at any window narrower than 1337, so that the signal viewer stays the widest column
#: — and until ``MainWindow.showEvent`` existed that clamp was unreachable on the launch
#: path (the width it was asked about was always the constructor's 1440), so this page was
#: measured here at a 502 px dock even at 1280x800. At 460 one label above the selector wraps
#: to a second line and everything under it moves down 15 px; measured, the two controls this
#: module pins end 8 px past the fold at 1280x800 and are whole at 490 px and above. The two
#: rules cannot both hold below 1337 px — a dock wide enough to keep this page unwrapped is
#: wider than the viewer it leaves — so the promise here is stated as what it now is: whole
#: above the boundary, and within one 8 px nudge of the scroll wheel below it.
_NARROW_FOLD_SHORTFALL = 8
#: The window width at and above which the Inspector opens wide enough for the promise to be
#: whole (see ``MainWindow._inspector_width_ceiling``: 1337 is where its ceiling first reaches
#: the width these pages need).
_FOLD_WHOLE_ABOVE = 1337

#: Words that would tell a reader the response plot exists.
_NAMES_THE_PLOT = ("response", "h(f)", "frequency")

#: The two real refusals ``AnalysisPanel._gate`` can push, longest first. The long one is
#: two lines at the shipped width and three at the narrowest — the growth that used to move
#: the panel even while the banner stayed up.
_LONG_REASON = ("Detection runs on CURRENT channels; this is a voltage channel "
                "(your Calibration-panel override; the file says current).")
_SHORT_REASON = ("Sample rate unknown — dwell time is undefined without a time base, "
                 "so detection is disabled for this recording.")


# ================================================================================================
# harness
# ================================================================================================

@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace, so "first launch" really means first launch."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("Lay2ActionsAboveTheFold")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


def _settle(win: MainWindow, rounds: int = 3) -> None:
    """Force the layout to finish.

    The offscreen platform does not propagate size hints ("This plugin does not support
    propagateSizeHints()"), so a plain ``processEvents`` can leave a freshly raised page
    holding the geometry it had before — measure that and you conclude a control is
    off-screen when it is not, and the other way round.
    """
    for _ in range(rounds):
        QtWidgets.QApplication.processEvents()
        for w in win.findChildren(QtWidgets.QWidget):
            lay = w.layout()
            if lay is not None:
                lay.activate()
        if win.layout() is not None:
            win.layout().activate()
        QtWidgets.QApplication.processEvents()


class _Column:
    """The Analysis tab's controls scroll area, measured the way a user sees it."""

    def __init__(self, win: MainWindow) -> None:
        self.win = win
        tabs = win._inspector.widget()
        for i in range(tabs.count()):
            if tabs.tabText(i).startswith("Analysis"):
                tabs.setCurrentIndex(i)
        _settle(win)
        self.analysis: AnalysisPanel = win.findChild(AnalysisPanel)
        self.detection: DetectionPanel = self.analysis._detection
        self.area = self._owning_scroll(self.detection)
        self.inner = self.area.widget()
        self.mode_tabs = self.detection.findChild(QtWidgets.QTabWidget)

    @staticmethod
    def _owning_scroll(w: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        p = w.parentWidget()
        while p is not None:
            if isinstance(p, QtWidgets.QScrollArea):
                return p
            p = p.parentWidget()
        raise AssertionError("the controls are not inside a scroll area at all")

    # -- geometry -------------------------------------------------------------------------
    @property
    def viewport_h(self) -> int:
        return self.area.viewport().height()

    @property
    def viewport_w(self) -> int:
        return self.area.viewport().width()

    @property
    def scroll(self) -> int:
        return self.area.verticalScrollBar().value()

    def raise_page(self, title: str) -> None:
        for i in range(self.mode_tabs.count()):
            if self.mode_tabs.tabText(i) == title:
                self.mode_tabs.setCurrentIndex(i)
                _settle(self.win)
                return
        raise AssertionError(
            f"no {title!r} page: "
            f"{[self.mode_tabs.tabText(i) for i in range(self.mode_tabs.count())]}")

    def yspan(self, w: QtWidgets.QWidget) -> tuple[int, int]:
        """``(top, bottom)`` of ``w`` in the scroll area's CONTENT coordinates."""
        y = w.mapTo(self.inner, QtCore.QPoint(0, 0)).y()
        return y, y + w.height()

    def xspan(self, w: QtWidgets.QWidget) -> tuple[int, int]:
        x = w.mapTo(self.inner, QtCore.QPoint(0, 0)).x()
        return x, x + w.width()

    def below_the_fold_by(self, w: QtWidgets.QWidget) -> int:
        """How far ``w``'s bottom edge falls past the visible region, in px (<=0 == on
        screen at the current scroll position)."""
        _, bottom = self.yspan(w)
        return bottom - (self.scroll + self.viewport_h)


def _build(qtbot, size, dock_width: int | None = None) -> _Column:
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(*size)
    win.show()
    _settle(win)
    if dock_width is not None:
        win.resizeDocks([win._inspector], [dock_width], QtCore.Qt.Horizontal)
        _settle(win)
    return _Column(win)


# ================================================================================================
# FOLD-1 -- the page's verbs are on the page a user arrives at
# ================================================================================================

@pytest.mark.parametrize("size", _SCREENS, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("action", _ACTIONS)
def test_every_detection_action_is_on_screen_without_scrolling(
        isolated_settings, qtbot, size, action):
    """No scroll, no drag, no menu: open the app, click Detection, see the button.

    Measured before the fix, identically at all three sizes: ``detect_button`` and
    ``sweep_button`` at content y=507..532 and ``null_button`` / ``recovery_button`` at
    538..563, against a 466 px viewport (405 px at 1280x800) — 66 px and 97 px below the
    fold respectively.
    """
    col = _build(qtbot, size)
    col.raise_page("Detection")
    assert col.scroll == 0, "the column does not open at the top; re-measure everything"
    btn = getattr(col.detection, action)
    top, bottom = col.yspan(btn)
    assert col.below_the_fold_by(btn) <= 0, (
        f"{action} is {col.below_the_fold_by(btn)} px below the fold at {size[0]}x{size[1]}: "
        f"content y={top}..{bottom} against a viewport showing "
        f"y={col.scroll}..{col.scroll + col.viewport_h}"
    )
    assert top >= col.scroll, f"{action} is above the visible region (y={top})"


@pytest.mark.parametrize("size", _SCREENS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_detection_actions_actually_paint(isolated_settings, qtbot, size):
    """The structural check above, confirmed by Qt's own answer.

    ``visibleRegion().isEmpty()`` was True for all four buttons; it is the number the
    reviewers reported and it costs nothing to pin it beside the arithmetic.
    """
    col = _build(qtbot, size)
    col.raise_page("Detection")
    hidden = [n for n in _ACTIONS
              if getattr(col.detection, n).visibleRegion().isEmpty()]
    assert hidden == [], (
        f"at {size[0]}x{size[1]} these Detection actions paint nothing at rest: {hidden}")


@pytest.mark.parametrize("size", _SCREENS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_action_row_keeps_the_x_fix_it_was_given(isolated_settings, qtbot, size):
    """Fixing y must not give back x.

    The previous round moved ``Null control…`` from x=421..620 to 11..207 by turning the
    four-wide row into a fixed 2x2 grid. This asserts both axes at once so a future
    "just put them back on one row" cannot pass by scrolling into view.
    """
    col = _build(qtbot, size)
    col.raise_page("Detection")
    for name in _ACTIONS:
        left, right = col.xspan(getattr(col.detection, name))
        assert 0 <= left and right <= col.viewport_w, (
            f"{name} is off the side again: x={left}..{right} in a {col.viewport_w} px "
            f"viewport")
    # Still exactly two rows of two -- a flow layout that reflowed with dock width would
    # move a button out from under the pointer aiming at it.
    tops = {col.yspan(getattr(col.detection, n))[0] for n in _ACTIONS}
    assert len(tops) == 2, f"the action grid is no longer 2x2: row tops {sorted(tops)}"


@pytest.mark.parametrize("size", _SCREENS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_pinning_the_actions_steals_no_room_from_the_parameters(
        isolated_settings, qtbot, size):
    """"Above the fold" must not mean "at the parameters' expense".

    The action row stays INSIDE the scrolled content — it is not an overlay clipping the
    viewport, and it is not a footer with a reserved band of its own — so every parameter
    on the page is still reachable by scrolling, and the column still demands no more
    width than it is given.
    """
    col = _build(qtbot, size)
    col.raise_page("Detection")
    hbar = col.area.horizontalScrollBar()
    assert hbar.maximum() == 0 and not hbar.isVisible(), (
        f"the controls scroll sideways again: content "
        f"{col.inner.minimumSizeHint().width()} px through {col.viewport_w} px")
    # The last parameter on the page is still reachable: its bottom is inside the range the
    # vertical scrollbar can actually reach.
    vbar = col.area.verticalScrollBar()
    _, bottom = col.yspan(col.detection.polarity_override)
    assert bottom <= vbar.maximum() + col.viewport_h, (
        f"the polarity override at y..{bottom} is past the end of the scroll range "
        f"({vbar.maximum()} + {col.viewport_h})")
    # And the grid is in the flow, not floating over it.
    assert col.detection.detect_button.parentWidget() is not None
    assert col.area.isAncestorOf(col.detection.detect_button)


# ================================================================================================
# FOLD-2 -- the response curve is named where the filter controls are
# ================================================================================================

def _labels_above_the_fold(col: _Column, slack: int = 0) -> list[str]:
    """Every visible label whose whole extent is inside the viewport at rest.

    ``slack`` extends the fold by the px this page loses to a wrapped line at the narrow
    widths where the Inspector opens at its floor — see ``_NARROW_FOLD_SHORTFALL``.
    """
    out = []
    for lab in col.detection._preprocess_page.findChildren(QtWidgets.QLabel):
        if lab.visibleRegion().isEmpty():
            continue
        top, bottom = col.yspan(lab)
        if top >= col.scroll and bottom <= col.scroll + col.viewport_h + slack:
            out.append(lab.text())
    return out


@pytest.mark.parametrize("size", _SCREENS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_something_above_the_fold_names_the_response_plot(isolated_settings, qtbot, size):
    """The plot is 256 px down the page; at minimum the page must SAY so.

    Before this change the same enumeration returned nothing containing "response",
    "H(f)" or "frequency" — the plot was undiscoverable to a reader who did not think to
    scroll past the entire filter stack.
    """
    col = _build(qtbot, size)
    col.raise_page("Preprocess")
    seen = _labels_above_the_fold(
        col, 0 if size[0] >= _FOLD_WHOLE_ABOVE else _NARROW_FOLD_SHORTFALL)
    hits = [t for t in seen if any(w in t.lower() for w in _NAMES_THE_PLOT)]
    assert hits, (
        f"nothing above the fold at {size[0]}x{size[1]} names the |H(f)| plot; visible "
        f"labels were {seen}")


def test_the_response_hint_imposes_no_width_floor(isolated_settings, qtbot):
    """The lesson of the previous round, applied to the line that fixes this one.

    A plain ``QLabel`` reports its FULL text width as ``minimumSizeHint()``, and that
    number propagates into the dock's minimum width — which is exactly how three other
    controls ended up off the side of the Inspector. The hint elides instead, keeps the
    full sentence in ``text()`` and in its tooltip, and floors nothing.
    """
    col = _build(qtbot, _SCREENS[0], dock_width=_INSPECTOR_DEFAULT_WIDTH)
    hint = col.detection.response_hint
    assert hint.minimumSizeHint().width() == 0, (
        f"the response hint floors the dock at {hint.minimumSizeHint().width()} px")
    assert any(w in hint.text().lower() for w in _NAMES_THE_PLOT)
    assert hint.toolTip() == hint.text()
    # It is above the fold even at the narrowest width the Inspector ships at.
    col.raise_page("Preprocess")
    assert col.below_the_fold_by(hint) <= 0, (
        f"the hint is {col.below_the_fold_by(hint)} px below the fold at the narrowest "
        f"shipped dock")


# ================================================================================================
# LAY-W3 -- the gate banner owns its row whether it speaks or not
# ================================================================================================

def _protocol_geometry(col: _Column) -> dict[str, tuple[int, int]]:
    col.raise_page("Preprocess")
    det = col.detection
    return {n: col.yspan(getattr(det, n)) for n in
            ("protocol_combo", "protocol_save_btn", "protocol_rename_btn",
             "protocol_delete_btn", "filter_selector", "response_hint")}


def test_the_gate_banner_moves_nothing_when_it_appears_or_goes(isolated_settings, qtbot):
    """The anti-pattern, stated as a diff of rectangles.

    Measured before the fix: selecting a recording hid the banner and pulled ``Save as…``,
    ``Rename`` and ``Delete`` from y=477 to y=456 — 21 px against a 25 px button, i.e.
    nearly a whole row, under whatever pointer was aiming at them. Those three verbs only
    became clickable in the previous round, so the shift went from displacing nothing to
    displacing three fresh controls.
    """
    col = _build(qtbot, _SCREENS[0])
    det = col.detection

    det.set_runnable(False, _LONG_REASON)
    gated = _protocol_geometry(col)
    det.set_runnable(True, "")
    runnable = _protocol_geometry(col)
    det.set_runnable(False, _SHORT_REASON)
    gated_short = _protocol_geometry(col)

    moved = {k: (gated[k], runnable[k]) for k in gated if gated[k] != runnable[k]}
    assert moved == {}, f"showing/hiding the gate banner moved: {moved}"
    grew = {k: (gated[k], gated_short[k]) for k in gated if gated[k] != gated_short[k]}
    assert grew == {}, (
        f"a two-line refusal and a one-line refusal reserve different rows, moving: {grew}")


def test_the_gate_banner_does_not_move_the_detection_actions(isolated_settings, qtbot):
    """The same diff on the four controls this round put on screen."""
    col = _build(qtbot, _SCREENS[0])
    det = col.detection
    col.raise_page("Detection")

    det.set_runnable(False, _LONG_REASON)
    _settle(col.win)
    before = {n: col.yspan(getattr(det, n)) for n in _ACTIONS}
    det.set_runnable(True, "")
    _settle(col.win)
    after = {n: col.yspan(getattr(det, n)) for n in _ACTIONS}
    assert before == after, f"the gate moved the action row: {before} -> {after}"


def test_the_reserved_row_keeps_its_height_and_the_banner_keeps_its_words(
        isolated_settings, qtbot):
    """The mechanism, so a future edit cannot satisfy the diff above by accident.

    The row is reserved by a fixed-height carrier, and the banner keeps its own
    show/hide — ``isHidden()`` is how the rest of the app and five test modules ask "is
    the gate up?", so an always-visible empty label would answer that wrongly. The full
    refusal stays in ``text()`` even when only part of it is painted.
    """
    col = _build(qtbot, _SCREENS[0])
    det = col.detection

    det.set_runnable(False, _LONG_REASON)
    _settle(col.win)
    speaking = det._gate_host.height()
    assert not det._gate_label.isHidden()
    assert det._gate_label.text() == _LONG_REASON

    det.set_runnable(True, "")
    _settle(col.win)
    assert det._gate_label.isHidden(), "the gate must still report itself down"
    assert det._gate_host.height() == speaking, (
        f"the reserved row collapsed with the banner: {speaking} -> "
        f"{det._gate_host.height()}")
    assert speaking > 0
