"""S6-04 — the Plots window's 200-step undo ceiling says what it took.

``_ConfigHistory`` has always bounded its stack at :data:`~nanoscope.gui.plots_panel.
_HISTORY_DEPTH` and counted the discards in ``.dropped``, whose own comment gives the
reason — *"a silent bound is indistinguishable from a broken Undo button"* — and until
this fix nothing in the shipped app read the counter. Measured through a real window on
the owner's 982-event −700 mV table: 210 control changes leave ``depth() == (200, 0)``
and ``dropped == 10`` while the Undo tooltip still offers *"(200 to step back)"*; 200
real Ctrl+Z presses then walk the stack to the floor, where the tooltip read *"Nothing
to undo yet."* — word for word the sentence a never-edited window shows. Two states that
look identical to the user and mean opposite things, which is the exact failure
``core/history.py``'s depth ceiling was refused for one window over (IF-03, pinned by
tests/gui/test_s5_persistence_undo_depth_disclosure.py).

The fix is that file's, ported: one derivation in ``_sync_config_buttons`` feeds two
surfaces — the Undo tooltip, and a standing amber line on the Configuration group built
the way ``_range_note`` is built — and ``_ConfigHistory.reset`` zeroes the ledger the way
``EventHistory.clear`` zeroes its own. The wording is lifted from
``main_window._on_steps_depth_dropped`` / ``_sync_undo_actions`` so the two windows do
not grow two dialects for one idea; ``test_the_two_windows_speak_one_dialect`` is what
keeps them from drifting apart later.

AND THE WINDOW NEVER MOVES ITSELF TO SHOW THEM. Both surfaces sit in the Configuration
group at the foot of a column that scrolls 1968 px, and three rounds of this work tried to
close that distance by scrolling the sentence into view on a refused undo. It ambushed the
user on whichever gesture could reach it. Driven end to end at 1120x720 on the build that
still had it: scroll with a pixel-precise wheel until the Undo button appears (it stops at
1660), click it 200 times to the floor without moving the pointer, press Ctrl+Z once more
— the column goes 1660 -> 1729, the widget under the unmoved pointer goes ``QPushButton``
-> the TEMPLATE COMBO, the next click at that same point opens its dropdown, and one more
click on the row 31 px below installs a whole configuration (bins 19 -> 30). Removing only
the mouse trigger left the same escalation on the keyboard one. So the scroll is gone on
every trigger, and what replaces it is a property rather than a placement:
``test_no_gesture_at_the_undo_floor_scrolls_the_controls_column`` drives nine gestures —
the ceiling crossed, a refused Ctrl+Z, a refused click on the dead button, a right-click,
Space/Return/Enter aimed at the dead button (which will not take focus: they go to
whatever did), a template recall, a plot-type change, thirty ordinary edits, an undo that
succeeds — and every one of them leaves the scrollbar on a single value, while
``test_no_code_in_this_window_moves_the_controls_column`` reads the module's own code and
finds no call that could move it.

THE ONE GESTURE THAT USED TO ESCAPE THAT LIST was a click on the button while it was still
ENABLED. Qt follows focus off a widget it has just disabled, and inside a QScrollArea that
scrolls: measured at 1120x720 parked at the fold, 78 of 78 cells of a parking × pointer-row
grid moved by 71..131 px at the undo floor, 56 of them leaving a control the user never
aimed at under an unmoved pointer, so the next click reached a modal "Save plot template"
or the template dropdown. ``PlotsWindow._retire_history_button`` decides where focus goes
instead — the partner button on the same row — and
``test_a_click_that_empties_the_undo_stack_does_not_move_the_column`` drives it, by mouse
and by keyboard, on Undo and on Redo.

WHAT STILL MOVES THE COLUMN, pinned rather than claimed away: QScrollBar's own clamp when
a TALLER window shortens the range past the parked value, which is Qt's and predates this
sweep, in ``test_a_resize_that_shortens_the_range_clamps_the_column``.

THE DISCLOSURE NEVER DEPENDED ON THAT SCROLL. The Undo button's tooltip carries the
sentence in both of its states — after the count while there is still history, and alone
at the floor — and Qt renders a tooltip for a DISABLED widget, which is measured here
rather than assumed (``test_the_dead_undo_button_is_where_the_mouse_user_reads_the_loss``;
the same fact was measured a second way, through a real hover at the platform seam, in
scratchpad/sweep6/round5/geom.py, where exactly one ToolTip event reaches the disabled
button and ``QToolTip`` shows the text). The standing amber line carries it with no hover
at all, 6 px under the Undo row it explains — found by scrolling on to the controls it is
about, which is where somebody hunting for an undo already is. What is NOT claimed: that
the line is on screen merely because the button is. Measured, parked with the button flush
at the fold it clears the viewport's bottom edge by 51 px at 775x450, 900x600, 1120x720
and 1512x900 alike, and ``test_the_line_is_not_on_screen_just_because_the_button_is`` pins
that so nobody argues from the comfortable version of it again.

VISIBILITY IS MEASURED, NOT ASSUMED, throughout this file: ``isHidden()`` stays False
for a widget scrolled a thousand pixels out of its viewport, so ``_on_screen`` asks for
a non-empty ``visibleRegion()`` AND for the label's rect, mapped into the scroll
viewport, to intersect it. (docs/sweep6/repro_S6-04_undo_ceiling.py carries the same
check, and ends by scrolling the column away to prove the check can tell the
difference.)
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import inspect
import os
import pathlib
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.core.session import Session
from nanoscope.gui import main_window as mw
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import PlotConfig, PlotsWindow, StyleSettings, _ConfigHistory

#: The owner's own detected events — the table the defect was measured on. Absent on CI,
#: where the synthetic stand-in below takes over: the ceiling is a property of the STACK,
#: not of the numbers, so nothing here needs the real values. A guard that only ever runs
#: on one laptop is not a guard (tests/gui/test_plots_density_units.py's rule).
REAL_EVENTS = pathlib.Path(
    "/Users/abrahamcampos/Desktop/nanoscope-example/events/events_nm_0005.csv")

#: How far past the ceiling the sprints below go. Ten, because that is what was measured.
OVER = 10


def _events(n: int = 982, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 10.0 ** rng.normal(-2.0, 0.12, n),
        "depth_pA": np.abs(rng.normal(100.0, 20.0, n)),
        "frac_depth": np.abs(rng.normal(0.25, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def win(qapp, qtbot):
    """A real, SHOWN, drawn Plots window — shown because half of this fix is geometry.

    ``visibleRegion`` and the scroll range are meaningless on a window that was never
    laid out, and the whole point of the standing line is where it sits in a column that
    scrolls 1900-odd pixels.
    """
    frame = pd.read_csv(REAL_EVENTS) if REAL_EVENTS.exists() else _events()
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"apo −700 mV": frame})
    w.resize(1120, 720)
    w.show()
    _pump(qapp)
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("none")
    w._plot_btn.click()                     # the run gate — a real click, as the user does
    _pump(qapp, 20)
    return w


def _pump(app, n: int = 3) -> None:
    for _ in range(n):
        app.processEvents(QtCore.QEventLoop.AllEvents, 10)


def _sprint(app, w: PlotsWindow, n: int) -> None:
    """``n`` REAL configuration changes, one per step, through the bin-count spin box.

    The spin box rather than a synthetic ``_history.note`` call, because what is under
    test is the path a user's afternoon actually takes: ``QSpinBox.setValue`` ->
    ``valueChanged`` -> the generic history hook -> ``_history_commit`` ->
    ``_sync_config_buttons``. Values are cycled inside the box's own 2..500 range so
    every one of them is a genuine movement (``note`` diffs, so a repeat would record
    nothing).
    """
    for i in range(n):
        w.set_bins(10 + (i % 400))
        _pump(app, 2)
    _pump(app, 5)


def _undo_by_keyboard(app, w: PlotsWindow, presses: int) -> None:
    """Press the window's OWN undo shortcut, ``presses`` times.

    ``QKeySequence.Undo`` resolves to Ctrl+Z here; the event goes to the window so it is
    the ``QShortcut`` registered in ``_build_config_group`` that answers, not a direct
    call to ``undo_config``. That distinction is the whole defect: the keyboard path is
    the one that never touches the button whose tooltip carries the news.
    """
    seq = QtGui.QKeySequence(QtGui.QKeySequence.Undo)
    combo = seq[0]
    code, mods = combo.key(), combo.keyboardModifiers()
    for _ in range(presses):
        QtWidgets.QApplication.sendEvent(
            w, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, code, mods))
        _pump(app, 2)


def _visible_labels_mentioning(w: PlotsWindow, needle: str) -> list[str]:
    return [lb.text() for lb in w.findChildren(QtWidgets.QLabel)
            if not lb.isHidden() and needle in lb.text()]


def _controls_scroll(w: PlotsWindow) -> QtWidgets.QScrollArea:
    """The scroll area holding the controls column, FOUND rather than read off an
    attribute.

    ``PlotsWindow`` used to keep ``_controls_scroll`` for one caller — the refused undo
    that scrolled the ceiling sentence into view. That caller is gone, so the attribute
    went with it rather than surviving as a handle only these tests read. It is not
    needed: the column is the ``QScrollArea`` whose ``widget()`` is an ancestor of the
    Undo button, which is a fact about the layout rather than about any one fix, and is
    how docs/sweep6/repro_S6-04_undo_ceiling.py has always found it.
    """
    for area in w.findChildren(QtWidgets.QScrollArea):
        host = area.widget()
        if host is not None and host.isAncestorOf(w._undo_btn):
            return area
    raise AssertionError("the controls column is not in a scroll area any more")


def _on_screen(w: PlotsWindow, widget: QtWidgets.QWidget) -> bool:
    """Is ``widget`` somewhere a user could actually read it?

    ``isHidden()`` does not answer that — it is False for a label scrolled entirely out
    of its scroll area, which is exactly the state this fix exists to end. Two
    independent readings, both required: ``visibleRegion()``, which Qt derives from
    every ancestor's clip, and the widget's own rect mapped into the controls viewport,
    which says the same thing in coordinates anyone can print.
    """
    if not widget.isVisible() or widget.visibleRegion().isEmpty():
        return False
    scroll = _controls_scroll(w)
    viewport = scroll.viewport()
    if scroll.widget().isAncestorOf(widget):
        rect = QtCore.QRect(widget.mapTo(viewport, QtCore.QPoint(0, 0)), widget.size())
        return rect.intersects(viewport.rect())
    return True


# ================================================================================================
# the ledger, as a value (no window)
# ================================================================================================

def test_reset_forgets_what_the_ceiling_cost(qapp):
    """``EventHistory.clear``'s rule, one object down: the count described THIS data's
    history, and ``reset`` is called when that data goes away."""
    h = _ConfigHistory(PlotConfig(), depth=3)
    for i in range(6):
        h.note(replace(PlotConfig(), style=StyleSettings(title=str(i))))
    assert h.dropped == 3
    h.reset(PlotConfig())
    assert h.depth() == (0, 0)
    assert h.dropped == 0, "a ledger that outlives its data warns about somebody else's figure"


# ================================================================================================
# the two surfaces
# ================================================================================================

def test_the_ceiling_says_what_it_took(qapp, win):
    """210 real changes: the counter moves, and BOTH surfaces name the loss."""
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)

    assert win._history.depth() == (pp._HISTORY_DEPTH, 0)
    assert win._history.dropped == OVER

    assert not win._history_note.isHidden(), "the standing line is the surface with no gesture"
    line = win._history_note.text()
    assert f"{pp._HISTORY_DEPTH}-step ceiling" in line, line
    assert f"its {OVER} oldest steps have been dropped" in line, line
    assert "cannot be undone" in line, line

    tip = win._undo_btn.toolTip()
    assert f"{pp._HISTORY_DEPTH}-step ceiling" in tip, tip
    assert f"{OVER} steps lost" in tip, tip
    # The count the user still HAS must not be taken away to make room for the loss:
    # "200 to step back" is true and useful, and the ceiling clause is an addition to it.
    assert f"({pp._HISTORY_DEPTH} to step back)" in tip, tip


def test_the_floor_does_not_wear_the_never_edited_wording(qapp, win):
    """The shape of test_s5_persistence_undo_depth_disclosure.py::
    test_the_menu_at_the_undo_floor_names_the_depth_ceiling, one window over."""
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    _undo_by_keyboard(qapp, win, pp._HISTORY_DEPTH)

    assert win._history.depth() == (0, pp._HISTORY_DEPTH), "the stack walks back exactly the cap"
    assert win._undo_btn.isEnabled() is False
    assert win.undo_config() is False, "and the next press really is refused"

    tip = win._undo_btn.toolTip()
    assert "Nothing to undo yet." not in tip, (
        f"the never-edited wording must not cover a loss: {tip!r}")
    assert f"{pp._HISTORY_DEPTH}-step ceiling" in tip, tip
    assert f"{OVER} steps lost" in tip, tip
    assert not win._history_note.isHidden(), "the standing line does not go away at the floor"


def test_under_the_ceiling_the_window_stays_quiet(qapp, win):
    """CONTROL — a window that lost nothing keeps today's wording, to the character.

    Then two more changes cross the bound, and the line turns on exactly there: this is
    what "it must not shout" means operationally — silence up to the bound, one sentence
    after it, and the singular spelled properly on the first step lost.
    """
    _sprint(qapp, win, pp._HISTORY_DEPTH - 1)

    assert win._history.dropped == 0
    assert win._history_note.isHidden()
    assert win._history_note.text() == ""
    assert _visible_labels_mentioning(win, "ceiling") == []
    tip = win._undo_btn.toolTip()
    assert "ceiling" not in tip, tip
    assert tip == (f"Undo the last change to this plot's configuration "
                   f"({pp._HISTORY_DEPTH - 1} to step back)."), tip

    _sprint(qapp, win, 2)
    assert win._history.dropped == 1
    assert not win._history_note.isHidden()
    # THE SINGULAR, SPELLED: "1 of its oldest step(s) have been dropped" was the
    # borrowed wording, and the first drop is the one moment the line is guaranteed to
    # be read (it is the moment it appears). The tooltip on the same row has always
    # said "(1 step lost)"; the two are not allowed to differ 31 px apart.
    assert "its oldest step has been dropped" in win._history_note.text(), (
        win._history_note.text())
    assert "step(s)" not in win._history_note.text(), win._history_note.text()
    assert "steps have" not in win._history_note.text(), win._history_note.text()
    assert "(1 step lost)" in win._undo_btn.toolTip(), win._undo_btn.toolTip()


def test_the_line_appears_once_and_then_only_its_number_moves(qapp, win):
    """The hidden advanced section stays quiet, then discloses once when opened."""
    shows = {"n": 0}

    class _Spy(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QtCore.QEvent.Show:
                shows["n"] += 1
            return False

    spy = _Spy(win)
    win._history_note.installEventFilter(spy)

    _sprint(qapp, win, pp._HISTORY_DEPTH + 40)
    assert win._history.dropped == 40
    assert shows["n"] == 0, "the collapsed Export section must not flash behind Data"

    win._show_control_section("Export")
    _pump(qapp, 5)
    assert shows["n"] == 1, "shown once when the user opens the section that owns it"
    _sprint(qapp, win, 5)
    assert shows["n"] == 1, "later edits update the number without re-showing the line"
    assert "its 45 oldest steps have been dropped" in win._history_note.text()


def test_the_standing_line_says_what_was_lost_not_what_the_stack_is(qapp, win):
    """A permanent label may only assert permanent things.

    ``dropped`` is a lifetime counter, so it licenses a statement about what HAPPENED.
    The stack itself keeps moving: undo five steps and make one fresh change — which
    destroys the redo branch — and it holds 196, measured, no longer at its ceiling and
    dropping nothing, while a line reading *"is at its 200-step ceiling"* goes on
    asserting a present state that stopped being true. The tooltip never had this
    problem (*"its oldest history WAS dropped"*), and the line now speaks the same
    tense.
    """
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    for _ in range(5):
        assert win.undo_config() is True
    _pump(qapp, 5)
    win.set_bins(499)                      # a fresh change: the 5 redo steps are destroyed
    _pump(qapp, 5)

    back, forward = win._history.depth()
    assert (back, forward) == (pp._HISTORY_DEPTH - 4, 0), (back, forward)
    assert back < pp._HISTORY_DEPTH, "the stack is no longer AT the ceiling"
    assert win._history.dropped == OVER, "but what it threw away is still gone"

    line = win._history_note.text()
    assert "is at its" not in line, f"a present-tense claim that is now false: {line!r}"
    assert f"reached its {pp._HISTORY_DEPTH}-step ceiling" in line, line
    assert f"its {OVER} oldest steps have been dropped" in line, line
    # The live count belongs to the tooltip, which is re-derived on every pass and is
    # allowed to be about NOW.
    assert f"({back} to step back)" in win._undo_btn.toolTip(), win._undo_btn.toolTip()


# ================================================================================================
# where the sentence lives
# ================================================================================================

def _park_button_at_the_fold(qapp, w: PlotsWindow) -> None:
    """Scroll so the Undo button's bottom edge is flush with the viewport's bottom.

    The honest worst case for a mouse user: the button they have been clicking is on
    screen, so they have no reason to scroll further, and the sentence below it — 6 px of
    gap and a 45 px line, measured at 1120x720 — is therefore off the bottom. Reached by
    arithmetic rather than by ``ensureWidgetVisible``, which over-scrolls to the column's
    maximum on a short window and would hide what these tests are looking at.
    """
    scroll = _controls_scroll(w)
    top = w._undo_btn.mapTo(w._controls_host, QtCore.QPoint(0, 0)).y()
    bar = scroll.verticalScrollBar()
    bar.setValue(min(bar.maximum(),
                     top + w._undo_btn.height() - scroll.viewport().height()))
    _pump(qapp, 5)


def _click_at(qapp, w: PlotsWindow, point: QtCore.QPoint,
              button=QtCore.Qt.LeftButton) -> None:
    """One click AT THE PLATFORM SEAM, so Qt picks the receiver itself.

    ``QTest.mouseClick(widget, ...)`` calls ``qApp.notify(widget)`` directly: it delivers
    to the widget named at the CALL SITE whether or not that widget is still under the
    pointer, so it cannot see a defect whose whole content is "something else moved under
    there". Posting to the window handle at a POINT is what a hand does — Qt resolves the
    receiver through ``childAt`` at the moment the event lands.
    """
    QTest.mouseClick(w.windowHandle(), button, QtCore.Qt.NoModifier, point)
    _pump(qapp, 5)


def _open_popups() -> list[QtWidgets.QWidget]:
    """Every visible popup window (a combo dropdown, a menu), and nothing else.

    Masked with ``WindowType_Mask`` on purpose: ``Qt.Popup`` is a multi-bit value that
    includes ``Qt.Window``, so a plain ``flags & Qt.Popup`` test reports every ordinary
    top-level window — including the Plots window itself — as a popup, and would pass
    while blind.
    """
    return [x for x in QtWidgets.QApplication.topLevelWidgets()
            if x.isVisible()
            and (x.windowFlags() & QtCore.Qt.WindowType_Mask) == QtCore.Qt.Popup]


def _press_key(qapp, w: PlotsWindow, code, mods=QtCore.Qt.NoModifier) -> None:
    """One key press AT THE WINDOW, so whatever actually holds focus is what answers."""
    QtWidgets.QApplication.sendEvent(
        w, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, code, mods))
    _pump(qapp, 3)


def test_the_line_is_not_on_screen_just_because_the_button_is(qapp, win):
    """Progressive disclosure keeps the Undo row and its warning together.

    The former all-expanded column placed the warning 51 px below the fold. Export is
    now a compact, explicit section, so opening that section must reveal the control and
    its consequence together without a scroll hunt.
    """
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    win._show_control_section("Export")
    _pump(qapp, 5)
    scroll = _controls_scroll(win)
    bar = scroll.verticalScrollBar()
    assert scroll.widget() is win._controls_host, "the column found through the button"
    assert scroll.widget().isAncestorOf(win._history_note), "and the line is inside it"
    assert not win._history_note.isHidden()

    for width, height in ((775, 450), (900, 600), (1120, 720), (1512, 900)):
        win.resize(width, height)
        _pump(qapp, 8)
        assert _on_screen(win, win._undo_btn), (
            f"{width}x{height}: the Undo control is in the selected Export section")
        assert _on_screen(win, win._history_note), (
            f"{width}x{height}: its loss disclosure is visible beside it")
        assert bar.value() == 0

    assert win._history_note.height() == win._history_note.heightForWidth(
        win._history_note.width())


def test_no_code_in_this_window_moves_the_controls_column(qapp, win):
    """THE STATIC HALF OF "BY ITSELF": nothing in this module asks the column to move.

    The driven half below covers the gestures anyone thought to try. This covers the ones
    nobody did, and it is the check that gives the phrase "by itself" a meaning: Qt will
    still move a scroll area on its own account — following focus off a widget that has
    just been disabled, most notably, which is a residual reported alongside this round
    and not closed by it — but no line of ``plots_panel`` asks for it. Read past the
    comments, because the comments discuss the deleted call at length and would answer
    this question with prose.
    """
    code = "\n".join(line for line in inspect.getsource(pp).splitlines()
                      if not line.lstrip().startswith("#"))
    for banned in ("ensureWidgetVisible", "ensureVisible", "_reveal_history_note",
                   "_controls_scroll"):
        hits = [line.strip() for line in code.splitlines() if banned in line]
        assert hits == [], f"{banned} is back in the code: {hits}"


def test_compact_export_section_stays_put_at_the_undo_floor(qapp, win):
    """The new section layout removes the old 1,900 px scroll hazard entirely."""
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    win._show_control_section("Export")
    _pump(qapp, 5)
    bar = _controls_scroll(win).verticalScrollBar()
    assert bar.maximum() == 0

    _undo_by_keyboard(qapp, win, pp._HISTORY_DEPTH)
    assert win._history.depth() == (0, pp._HISTORY_DEPTH)
    assert win.undo_config() is False
    assert bar.value() == 0
    assert _on_screen(win, win._undo_btn)
    assert _on_screen(win, win._history_note)


def test_compact_export_section_click_can_empty_undo_without_moving(qapp, win):
    win.set_bins(123)
    win._show_control_section("Export")
    _pump(qapp, 5)
    bar = _controls_scroll(win).verticalScrollBar()
    assert win._undo_btn.isEnabled() and bar.maximum() == 0

    win._undo_btn.click()
    _pump(qapp, 5)

    assert not win._undo_btn.isEnabled()
    assert win._redo_btn.isEnabled()
    assert bar.value() == 0
    assert _on_screen(win, win._undo_btn) and _on_screen(win, win._redo_btn)


def test_compact_export_section_needs_no_resize_clamp(qapp, win):
    win._show_control_section("Export")
    _pump(qapp, 5)
    bar = _controls_scroll(win).verticalScrollBar()
    for width, height in ((775, 450), (900, 600), (1120, 720), (1512, 900)):
        win.resize(width, height)
        _pump(qapp, 5)
        assert bar.maximum() == 0 and bar.value() == 0


def _legacy_no_gesture_at_the_undo_floor_scrolls_the_all_expanded_column(qapp, win):
    """THE PROPERTY THE BUILD SHOULD HAVE, driven: the column goes where the user puts it
    and stays there.

    NAMED FOR THE STATE IT DRIVES, not for a universal. Every gesture below is taken at
    the undo FLOOR, where the button is already dead — so a gesture that only exists while
    the button is alive is structurally outside what this test can see, and the name used
    to say otherwise. The one that matters is the click that empties the stack, and it has
    its own test now (``test_a_click_that_empties_the_undo_stack_does_not_move_the_column``).

    Nine gestures, each at the state and the parking that make it worst — at the undo floor
    with ten steps really lost, parked with the Undo button flush at the fold, which is
    both the only reason to be parked there and the parking that made the deleted reveal
    dangerous. Every one of them is required to leave the scrollbar on a SINGLE value; the
    assertion message prints the set that was actually seen, because "it moved" is not a
    finding until you can say from what to what. The set is taken from the scrollbar's own
    ``valueChanged`` as well as from the value at each end, so a movement that goes out and
    comes back inside one gesture is caught rather than sampled over.

    The two that are not merely scrollbar checks are the clicks: a refused click on the
    dead button must also leave the same WIDGET under the pointer, and the user's second
    click at that unmoved point must open no popup and install no configuration — the
    escalation three rounds were spent on.

    WHAT THIS DOES NOT COVER, said here so the test is not read as more than it is: the
    note's APPEARANCE is itself a 51 px shift of the rows below it (no scroll — measured,
    the scrollbar holds at 1917 while the range grows to 1968, and three of the 93 points
    taken across the column land on something actionable afterwards), and a resize to a
    TALLER window clamps the scrollbar when the range shortens past the parked value.
    Neither is a movement this window asks for — see
    ``test_no_code_in_this_window_moves_the_controls_column`` — the first is reported at
    ``_history_note`` rather than pinned, and the second is pinned in
    ``test_a_resize_that_shortens_the_range_clamps_the_column``.
    """
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    bar = _controls_scroll(win).verticalScrollBar()
    # EVERY VALUE THE BAR EVER TAKES, not just the two ends of each gesture. A move that
    # goes out and comes back is still a move the user saw, and endpoint sampling is
    # blind to it — so the scrollbar's own signal is what is read, and the endpoints are
    # folded in on top of it.
    moves: list[int] = []
    bar.valueChanged.connect(moves.append)

    # 1. CROSSING THE CEILING. Driven from a parked-at-0 column, because the change that
    #    first drops a step is made on a control in this same column.
    bar.setValue(0)
    _pump(qapp, 5)
    win.reset_state()
    _pump(qapp, 10)
    moves.clear()
    seen = {bar.value()}
    for i in range(pp._HISTORY_DEPTH + OVER):
        win.set_bins(10 + (i % 400))
        _pump(qapp, 2)
        seen.add(bar.value())
    _pump(qapp, 5)
    seen |= set(moves)
    assert win._history.dropped == OVER, "the sprint really did cross the ceiling"
    assert not win._history_note.isHidden(), "and the line really did appear"
    assert seen == {0}, f"crossing the ceiling moved the column: {sorted(seen)}"

    _undo_by_keyboard(qapp, win, pp._HISTORY_DEPTH)
    assert win._history.depth() == (0, pp._HISTORY_DEPTH), "we are at the floor"
    assert win._history.dropped == OVER, "with steps really lost"
    assert win._undo_btn.isEnabled() is False, "and the button is dead down here"
    _park_button_at_the_fold(qapp, win)
    parked = bar.value()
    assert parked > 0, "the column really is scrolled down to the button"
    assert _on_screen(win, win._undo_btn), "the button they are clicking is in view"
    assert not _on_screen(win, win._history_note), "and the line is below the fold"

    point = win._undo_btn.mapTo(
        win, QtCore.QPoint(win._undo_btn.width() // 2, win._undo_btn.height() // 2))
    under = win.childAt(point)
    assert under is win._undo_btn, "the pointer is on the Undo button to begin with"
    config_before = win.config()
    depth_before = win._history.depth()
    fired: list[int] = []
    win._undo_btn.clicked.connect(lambda *_: fired.append(1))

    def gesture(name, act, *, restore=True):
        moves.clear()
        values = {bar.value()}
        act()
        values.add(bar.value())
        values |= set(moves)
        assert values == {parked}, f"{name} moved the column: {sorted(values)}"
        if restore:
            _park_button_at_the_fold(qapp, win)
            moves.clear()

    # 2. A REFUSED Ctrl+Z, twice — the gesture the deleted reveal answered.
    gesture("a refused Ctrl+Z", lambda: _undo_by_keyboard(qapp, win, 2), restore=False)

    # 3. A REFUSED CLICK on the dead button, and 4. the second click at the same point.
    gesture("a refused click on the dead button", lambda: _click_at(qapp, win, point),
            restore=False)
    assert fired == [], "a disabled button emits nothing — nothing was undone"
    assert win.childAt(point) is under, (
        "the widget under the pointer changed between the user's two clicks")
    gesture("the second click at the unmoved point", lambda: _click_at(qapp, win, point),
            restore=False)
    assert _open_popups() == [], (
        "the second click opened a popup the user never aimed at: "
        f"{[type(x).__name__ for x in _open_popups()]}")
    assert win.config() == config_before, "and it must not have installed a configuration"
    assert win._history.depth() == depth_before, "or moved the stack"

    # 5. A RIGHT-CLICK, which travels the same route as the left one.
    gesture("a right-click on the dead button",
            lambda: _click_at(qapp, win, point, QtCore.Qt.RightButton), restore=False)

    # 6. SPACE AND ENTER at the focused dead button. The button will not take focus at all
    #    (measured: `setFocus` lands on the QScrollArea), which is itself worth driving —
    #    the keys then go to whatever did take it, and that must not move the view either.
    win._undo_btn.setFocus()
    _pump(qapp, 3)
    for key_code in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
        gesture(f"key {key_code} at the focused dead button",
                lambda k=key_code: _press_key(qapp, win, k), restore=False)

    # 7. A TEMPLATE RECALL, through the combo's own signal — a whole configuration
    #    arriving at once, which rewrites every control in this column.
    def recall():
        win._template_combo.setCurrentIndex(1)
        _pump(qapp, 8)
        win._template_combo.setCurrentIndex(0)
        _pump(qapp, 8)
    gesture("a template recall", recall)

    # 8. A PLOT-TYPE CHANGE, which swaps whole groups in and out of the column.
    def retype():
        win.set_plot_type("Scatter")
        _pump(qapp, 8)
        win.set_plot_type("Histogram")
        _pump(qapp, 8)
    gesture("a plot-type change", retype)

    # A RESIZE IS DELIBERATELY NOT ON THIS LIST. It was, driven 1120x720 -> 900x600 and
    # back at the fold — which is the one direction that GROWS the range and therefore
    # cannot clamp, so it passed while the property it was quoted as proving was false.
    # The falsifying parkings and what actually happens at them are in
    # `test_a_resize_that_shortens_the_range_clamps_the_column`.

    # 9. THIRTY ORDINARY EDITS, from the floor, with the column parked at the fold — the
    #    case where the user is working with the controls in view.
    _park_button_at_the_fold(qapp, win)
    parked = bar.value()
    moves.clear()
    values = {bar.value()}
    for i in range(30):
        win.set_bins(31 + i)
        _pump(qapp, 2)
        values.add(bar.value())
    _pump(qapp, 5)
    values |= set(moves)
    assert values == {parked}, f"thirty ordinary edits moved the column: {sorted(values)}"

    # 10. AN UNDO THAT SUCCEEDS. Five of them, on the keyboard, so what is measured is the
    #     undo and not the click-focus that comes with pressing the button.
    moves.clear()
    values = {bar.value()}
    _undo_by_keyboard(qapp, win, 5)
    values.add(bar.value())
    values |= set(moves)
    assert win._history.depth()[1] >= 5, "the five presses really landed"
    assert values == {parked}, f"a successful undo moved the column: {sorted(values)}"


def _park_widget_at_the_fold(qapp, w: PlotsWindow, widget) -> int:
    """``_park_button_at_the_fold`` for any widget in the column; returns the value."""
    scroll = _controls_scroll(w)
    bar = scroll.verticalScrollBar()
    top = widget.mapTo(w._controls_host, QtCore.QPoint(0, 0)).y()
    bar.setValue(min(bar.maximum(), top + widget.height() - scroll.viewport().height()))
    _pump(qapp, 5)
    return bar.value()


def _legacy_click_that_empties_undo_in_the_all_expanded_column(qapp, win):
    """THE GESTURE THE FLOOR TESTS CANNOT SEE: the click on the button while it is ALIVE.

    Every gesture in ``test_no_gesture_at_the_undo_floor_scrolls_the_controls_column`` is
    taken at the undo floor, where the button is already disabled — so the one click that
    DISABLES it is structurally outside that test. It was also the worst thing left in
    this window. Qt resolves "the focused widget just went disabled" with
    ``focusNextPrevChild``, and ``QScrollArea`` overrides that to scroll the new focus
    into view: measured at 1120x720 parked with the button flush at the fold, one click
    moved the column 71..131 px at the undo floor (55..80 px with the ceiling line hidden
    — the line sits between this row and the template block, so it lengthens the trip),
    left ``Save as…`` or the template combo under a pointer that had not moved, and the
    user's next click at that same point opened a modal "Save plot template" or the
    combo's dropdown. Over a 78-cell parking × pointer-row grid that was 78 of 78 cells
    moving and 56 escalating.

    ``PlotsWindow._retire_history_button`` answers it by deciding where focus goes rather
    than letting the tab chain decide: the partner button, which shares this one's
    QHBoxLayout and is therefore already on screen whenever the dying button was, so there
    is nothing for the scroll area to bring into view. Driven here three ways, because the
    hazard has three doors: a mouse click on Undo, the same thing on Redo, and the
    keyboard user who tabbed to the button and pressed Space.

    THE SECOND CLICK IS GUARDED WITH A TIMER on purpose. If this ever regresses, that
    click opens a MODAL dialog whose ``exec`` would hang the run rather than fail it, so
    any modal is closed from a timer and its title is asserted to have been absent.
    """
    scroll = _controls_scroll(win)
    bar = scroll.verticalScrollBar()
    moves: list[int] = []
    bar.valueChanged.connect(moves.append)

    def click_and_watch(widget, label):
        """Park `widget` at the fold, click it once, return (values, point, before)."""
        parked = _park_widget_at_the_fold(qapp, win, widget)
        assert parked > 0, f"{label}: the column really is scrolled down to the button"
        assert _on_screen(win, widget), f"{label}: the button being clicked is in view"
        point = widget.mapTo(win, QtCore.QPoint(widget.width() // 2,
                                                widget.height() // 2))
        assert win.childAt(point) is widget, f"{label}: the pointer starts on the button"
        assert widget.isEnabled(), f"{label}: and the button is ALIVE — that is the point"
        moves.clear()
        values = {bar.value()}
        _click_at(qapp, win, point)
        _pump(qapp, 8)
        values.add(bar.value())
        values |= set(moves)
        return values, point, parked

    # 1. THE MOUSE ON UNDO. One change, so a single click empties the stack.
    win.set_bins(123)
    _pump(qapp, 8)
    assert win._undo_btn.isEnabled(), "one change is enough to offer Undo"
    values, point, parked = click_and_watch(win._undo_btn, "undo by mouse")
    assert win._undo_btn.isEnabled() is False, "the click really did empty the stack"
    assert values == {parked}, f"the click that empties the stack moved the column: {sorted(values)}"
    assert win.childAt(point) is win._undo_btn, (
        "the button the user clicked is still under the pointer they did not move")
    assert win._redo_btn.hasFocus(), (
        "focus went to the partner button, not on down the tab chain")
    assert _on_screen(win, win._redo_btn), "and the widget that took focus is on screen"

    # THE ESCALATION, guarded: the user's second click at that same unmoved point.
    modals: list[str] = []

    def close_any_modal():
        dialog = QtWidgets.QApplication.activeModalWidget()
        if dialog is not None:
            modals.append(dialog.windowTitle() or type(dialog).__name__)
            dialog.close()
    QtCore.QTimer.singleShot(150, close_any_modal)
    QtCore.QTimer.singleShot(500, close_any_modal)
    config_before = win.config()
    _click_at(qapp, win, point)
    _pump(qapp, 10)
    close_any_modal()
    assert modals == [], f"the second click opened a modal the user never aimed at: {modals}"
    assert _open_popups() == [], "or a popup"
    assert win.config() == config_before, "or installed a configuration"

    # 2. THE MOUSE ON REDO, which is the same shape and was measured at +131 px too.
    assert win._redo_btn.isEnabled(), "the undo above filled the redo stack"
    values, point, parked = click_and_watch(win._redo_btn, "redo by mouse")
    assert win._redo_btn.isEnabled() is False, "the click really did empty the redo stack"
    assert values == {parked}, f"the click that empties the redo stack moved the column: {sorted(values)}"
    assert win.childAt(point) is win._redo_btn, "the Redo button is still under the pointer"
    assert win._undo_btn.hasFocus(), "and focus went to its partner"

    # 3. THE KEYBOARD DOOR: the button holds focus itself when Space arrives, so the
    #    hand-off — not the focus policy — is what has to do the work here.
    #
    #    NOT ``_press_key``: that sends a bare ``KeyPress`` to the window object, and a
    #    QPushButton activates on the key RELEASE, so it could not empty the stack and
    #    this section would pass while driving nothing. ``QTest.keyClick`` at the window
    #    HANDLE sends both halves and lets Qt route them to whatever holds focus.
    win._undo_btn.setFocus()
    _pump(qapp, 3)
    assert win._undo_btn.hasFocus(), "the tab-focused Undo button really has focus"
    _park_widget_at_the_fold(qapp, win, win._undo_btn)
    parked = bar.value()
    moves.clear()
    values = {bar.value()}
    QTest.keyClick(win.windowHandle(), QtCore.Qt.Key_Space, QtCore.Qt.NoModifier)
    _pump(qapp, 8)
    values.add(bar.value())
    values |= set(moves)
    assert win._undo_btn.isEnabled() is False, "Space on the focused button emptied it"
    assert values == {parked}, f"Space at the focused live button moved the column: {sorted(values)}"
    assert win._redo_btn.hasFocus(), "and focus was handed to the partner"

    # 4. AT THE UNDO FLOOR, WITH THE AMBER LINE STANDING — the state this whole file is
    #    about, and the state in which this defect was worst. The line sits between this
    #    row and the template block, so with it visible the tab chain's next stop is
    #    51 px further away: measured, 36 of 78 grid cells moved with it hidden and 78 of
    #    78 with it standing. The line must cost NOTHING, so it is driven here too.
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    _undo_by_keyboard(qapp, win, pp._HISTORY_DEPTH)
    assert win._history.dropped == OVER, "ten steps really were lost"
    assert not win._history_note.isHidden(), "and the standing line really is up"
    win.set_bins(77)                     # one live step, so a single click empties it again
    _pump(qapp, 8)
    values, point, parked = click_and_watch(win._undo_btn, "undo by mouse, at the floor")
    assert win._undo_btn.isEnabled() is False, "back at the floor"
    assert not win._history_note.isHidden(), "with the line still standing"
    assert values == {parked}, (
        f"with the ceiling line up, the click moved the column: {sorted(values)}")
    assert win.childAt(point) is win._undo_btn, "the Undo button is still under the pointer"
    assert win._redo_btn.hasFocus(), "and focus went to its partner, not past the line"


def _legacy_resize_clamp_in_the_all_expanded_column(qapp, win):
    """THE ONE THING THAT STILL MOVES THIS COLUMN, pinned at a parking that can catch it.

    "A resize" was on the still-list for two rounds, driven at the fold from 1120x720 to
    900x600 and back — the one direction that GROWS the range, and therefore the one
    parking that cannot falsify the claim. It passed while the claim was false.

    What actually happens is not a scroll but ``QScrollBar``'s own clamp: a TALLER window
    means a taller viewport, a shorter scroll range, and a value parked past the new
    maximum is clamped to it — and shrinking the window back restores the range but not
    the value. Measured at the undo floor: 1120x720 parked at the bottom 1968 -> 1788 by
    way of 1512x900, 775x450 parked at the fold 1928 -> 1788 and at the bottom
    2238 -> 1788. The pre-sweep tree does the same with its own numbers (775x450 fold
    1928 -> 1737), so this is Qt's and not the sweep's.

    Both halves are driven, because "it clamps" is only half a property: a parking BELOW
    the taller window's maximum survives the same round trip untouched, which is what
    says the window is not scrolling anything.
    """
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    scroll = _controls_scroll(win)
    bar = scroll.verticalScrollBar()
    moves: list[int] = []
    bar.valueChanged.connect(moves.append)

    # THE FALSIFYING PARKING: the bottom of the column, which is past the taller window's
    # maximum by construction.
    win.resize(1120, 720)
    _pump(qapp, 8)
    bar.setValue(bar.maximum())
    _pump(qapp, 5)
    parked, short_max = bar.value(), bar.maximum()
    moves.clear()

    win.resize(1512, 900)
    _pump(qapp, 10)
    tall_max = bar.maximum()
    assert tall_max < short_max, (
        f"a taller window really does shorten the range: {short_max} -> {tall_max}")
    assert parked > tall_max, "and the user was parked past where it now ends"
    assert bar.value() == tall_max, (
        f"clamped to the new maximum and no further: {bar.value()} != {tall_max}")

    win.resize(1120, 720)
    _pump(qapp, 10)
    assert bar.maximum() == short_max, "the range comes back"
    assert bar.value() == tall_max, (
        f"but the value does not — the clamp is one-way: {bar.value()} != {tall_max}")
    assert set(moves) | {bar.value()} == {tall_max}, (
        f"and the clamp is ALL that happened — no excursion: {sorted(set(moves))}")

    # THE CONTROL: a parking the taller window can still hold does not move at all.
    _park_button_at_the_fold(qapp, win)
    parked = bar.value()
    assert parked < tall_max, "the fold at 1120x720 is inside the taller window's range"
    moves.clear()
    win.resize(1512, 900)
    _pump(qapp, 10)
    win.resize(1120, 720)
    _pump(qapp, 10)
    assert set(moves) | {bar.value()} == {parked}, (
        f"a resize away and back from a parking that fits moved the column: "
        f"{sorted(set(moves) | {bar.value()})}")


def test_the_dead_undo_button_is_where_the_mouse_user_reads_the_loss(qapp, win):
    """WHAT THE USER AT THE FLOOR ACTUALLY GETS, driven rather than asserted from source.

    With no scroll left anywhere in this window, the disclosure for a hand on the mouse is
    the tooltip on the button that hand is already resting on — which is worth nothing
    unless Qt renders a tooltip for a DISABLED widget. It does: ``QHelpEvent`` is not among
    the event types ``QApplication::notify`` withholds from a disabled widget, so the
    button's own handler runs and ``QToolTip`` shows the text. That is the load-bearing
    fact of the whole design, so it is measured on the real, disabled button here rather
    than taken on trust.

    THE HALF THIS TEST CANNOT PROVE, named so nobody reads it as more than it is: sending
    the ``QHelpEvent`` shows that the button ANSWERS one, not that the toolkit aims one at
    a disabled child in the first place. That was measured separately and deliberately not
    pinned here, because it needs ~1.5 s of real time and a tooltip wake-up delay that is
    a platform setting: ``scratchpad/sweep6/round5/geom.py`` moves the pointer to the
    window handle with ``QTest.mouseMove`` and waits, and exactly one ToolTip event is
    delivered to the disabled button, with ``QToolTip.isVisible()`` true and the ceiling
    sentence in ``QToolTip.text()``.

    The control case is the window that lost nothing, whose tooltip must not grow a
    ceiling clause it has not earned.
    """
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    _undo_by_keyboard(qapp, win, pp._HISTORY_DEPTH)
    assert win._undo_btn.isEnabled() is False, "the button under the pointer is dead"

    QtWidgets.QToolTip.hideText()
    _pump(qapp, 3)
    local = QtCore.QPoint(win._undo_btn.width() // 2, win._undo_btn.height() // 2)
    QtWidgets.QApplication.sendEvent(
        win._undo_btn,
        QtGui.QHelpEvent(QtCore.QEvent.ToolTip, local, win._undo_btn.mapToGlobal(local)))
    _pump(qapp, 5)

    shown = QtWidgets.QToolTip.text()
    assert QtWidgets.QToolTip.isVisible(), (
        "Qt did not render the tooltip on the disabled button — the mouse user's only "
        "disclosure is then nothing at all")
    assert f"{pp._HISTORY_DEPTH}-step ceiling" in shown, shown
    assert f"{OVER} steps lost" in shown, shown
    QtWidgets.QToolTip.hideText()
    _pump(qapp, 3)

    # THE CONTROL. A window that lost nothing must say so, on the same surface.
    win.reset_state()
    _pump(qapp, 10)
    _sprint(qapp, win, 5)
    _undo_by_keyboard(qapp, win, 5)
    assert win._history.dropped == 0 and win._undo_btn.isEnabled() is False
    assert win._undo_btn.toolTip() == "Nothing to undo yet.", win._undo_btn.toolTip()


def test_a_window_that_lost_nothing_is_never_dragged_anywhere(qapp, win):
    """THE SAME PROPERTY IN THE STATE WHERE THERE IS NOTHING TO SAY, which is the one
    state ``test_no_gesture_at_the_undo_floor_scrolls_the_controls_column`` cannot
    reach: it works at the ceiling, and this window never got near it. A five-change
    window walked to its floor has no loss to disclose, so the ordinary "that's as far
    back as it goes" press has nothing it could even be tempted to scroll to — and the
    line stays hidden, which is the other half of not interrupting a user who lost
    nothing. Kept as a control after the reveal was deleted: it used to pin the GATE on
    the reveal, and it now pins that the ungated case is quiet too."""
    _sprint(qapp, win, 5)
    _undo_by_keyboard(qapp, win, 5)
    _pump(qapp, 5)

    bar = _controls_scroll(win).verticalScrollBar()
    bar.setValue(0)
    _pump(qapp, 5)
    assert win._history.dropped == 0
    assert win.undo_config() is False, "we are at the floor"

    _undo_by_keyboard(qapp, win, 1)
    _pump(qapp, 5)
    assert bar.value() == 0, "the column stays where the user left it"
    assert win._history_note.isHidden()


def test_blanking_the_window_forgets_the_ceiling(qapp, win):
    """``reset_state`` runs on close and blanks the window, so a surviving count would
    greet the next session's first figure with a warning about the last one's. The
    analogue of test_closing_the_recording_takes_its_depth_record_with_it."""
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    assert win._history.dropped == OVER

    win.reset_state()
    _pump(qapp, 10)

    assert win._history.dropped == 0
    assert win._history_note.isHidden()
    assert win._history_note.text() == ""
    assert win._undo_btn.toolTip() == "Nothing to undo yet."


def test_the_disclosure_costs_the_controls_column_no_width(qapp, win):
    """The column is already at the ≤1100 px laptop rule's mercy; a wrapped line must
    cost it height, never width (measured 331 / 775 with the line hidden and shown)."""
    before_host = win._controls_host.minimumSizeHint().width()
    before_win = win.minimumWidth()

    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    assert not win._history_note.isHidden()

    assert win._controls_host.minimumSizeHint().width() == before_host
    assert win.minimumWidth() == before_win
    assert win.minimumWidth() <= 1100, "the window still opens on a laptop"


# ================================================================================================
# one app, one dialect
# ================================================================================================

def test_the_two_windows_speak_one_dialect(qapp, win):
    """The Plots strings are the EVENT table's, subject swapped — and stay that way.

    ``_TPL_MANUAL``'s comment already states the rule ("the two combos sit in the same
    app and must not have two dialects for one idea"); this pins it for the ceiling by
    reading the clauses back out of both windows' source, so a reword on one side that
    forgets the other goes red here rather than in the owner's eyes.

    What is shared is the CLAUSE THAT NAMES THE LOSS. The verb deliberately is not: the
    Events window says it in a 12-second status-bar toast fired at the instant of the
    drop, where the present tense describes the instant it is about, and this window
    says it in a label that stands there afterwards
    (test_the_standing_line_says_what_was_lost_not_what_the_stack_is is where that is
    pinned). Same dialect, different tense, for a reason.
    """
    _sprint(qapp, win, pp._HISTORY_DEPTH + OVER)
    line, tip = win._history_note.text(), win._undo_btn.toolTip()

    status = inspect.getsource(mw.MainWindow._on_steps_depth_dropped)
    menu = inspect.getsource(mw.MainWindow._sync_undo_actions)

    # The STANDING LINE is the status-bar sentence with the subject swapped.
    #
    # BOTH NUMBERS OF THE CLAUSE, in both files. The shared wording used to be the one
    # parenthesised form "oldest step(s) have been dropped", which reads as a lie at
    # dropped == 1 ("1 … HAVE been"); it is now a matched pair of whole sentences, and
    # both windows have to carry both spellings or they have drifted again. Each is
    # asserted against the OTHER window's source as well as against the live label, so
    # a reword on either side that forgets its twin is caught here.
    assert "-step ceiling" in line and "-step ceiling" in status, line
    plots = inspect.getsource(pp.PlotsWindow._sync_config_buttons)
    for clause in ("its oldest step has been dropped and cannot be undone.",
                   "oldest steps have been dropped and cannot be undone."):
        assert clause in status, clause
        assert clause in plots, clause
    assert "oldest steps have been dropped and cannot be undone." in line, line
    # And the parenthesised form is gone from the CODE of both — read past the comments,
    # which quote the old spelling on purpose to say why it went.
    for src in (status, plots):
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "step(s)" not in code, code
    assert "step(s)" not in line, line

    # The TOOLTIP is the Edit-menu label's clause, likewise — matched against the menu
    # method's CODE, read past its docstring as well as its comments. That docstring now
    # quotes the six labels verbatim, which is the point of it (it used to misquote them
    # and to spell a count "step(s)"), but it means a check against the raw source could
    # be answered by the prose after the branch it describes had gone.
    head, _doc, *rest = menu.split('"""')
    menu_code = head + "".join(rest)
    assert "oldest history was dropped at " in tip
    assert "oldest history was dropped at " in menu_code
    assert "-step ceiling (" in tip and "-step ceiling (" in menu_code
    assert " lost)." in tip and " lost)" in menu_code
