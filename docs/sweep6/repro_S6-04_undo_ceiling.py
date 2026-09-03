"""S6-04 — the Plots window's 200-step undo ceiling throws steps away silently.

Gesture: Plots ▸ tune one figure for an afternoon (here: 210 real bin-count changes) ▸
         then Undo, and keep pressing until it stops.
Expected: the window says it hit its ceiling and how many of the oldest steps it
          discarded — the rule the Events table already follows
          (main_window._on_steps_depth_dropped / _sync_undo_actions) — on the Undo
          button's own tooltip in both of its states, and on a standing line in the
          Configuration group that says it with no gesture at all.
Observed:  `_ConfigHistory.dropped` counts the discards — its own comment says it exists
           "because a silent bound is indistinguishable from a broken Undo button" — and
           NOTHING in the shipped app reads it. The Undo tooltip still offers "(200 to
           step back)" with no mention of the ten that are gone, and at the floor it
           reads "Nothing to undo yet.", word for word the sentence a never-edited
           window shows. Two states that look identical and mean opposite things.

ON SCREEN MEANS ON SCREEN. The first version of this file filtered its labels on
`QLabel.isHidden()`, which is False for a widget scrolled entirely out of its scroll
area — so it printed the amber line as evidence the window "says so" while that line
sat 1729 px below the fold with an EMPTY `visibleRegion()`, and anyone reading the
output would have concluded the disclosure was on screen when it was not. Visibility is
now measured the way the user experiences it: a non-empty `visibleRegion()` AND the
label's own rect, mapped into the scroll viewport, actually intersecting that viewport.
The check ends by scrolling the column back to the top and demanding the detector go
silent — a detector that cannot see the difference is worth nothing here.

AND THE WINDOW DOES NOT MOVE THE VIEW TO CLOSE THAT DISTANCE. Three rounds of this fix
answered a refused undo by scrolling the standing line into view, and it ambushed the
user on whichever gesture reached it: the line sits 6 px under a 25 px button, so the
column travelled ~70 px, the widget under a pointer that had not moved became the
template combo, the next click opened its dropdown and one more installed a whole
configuration. The scroll is gone on every trigger. So this file no longer certifies a
movement; it certifies the opposite, for BOTH hands, and drives every gesture it claims.
The keyboard user's refused Ctrl+Z and the mouse user's refused click on the dead button
are made at the parking each implies, and neither may move the column, swap the widget
under the pointer or open a popup. What the user gets instead is driven too: the tooltip
rendered on the DISABLED button, and the standing line, read off the screen once the
column is scrolled to the controls it explains.

Run:  QT_QPA_PLATFORM=offscreen python docs/sweep6/repro_S6-04_undo_ceiling.py
Exit: 7 = reproduced, 0 = fixed.
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import nanoscope                                     # Qt plugin-path bootstrap
import numpy as np, pandas as pd
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest
from nanoscope.core.session import Session
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import PlotsWindow

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
def pump(n=3):
    for _ in range(n): app.processEvents(QtCore.QEventLoop.AllEvents, 10)

# The owner's own 982 detected events. Synthesised only if this laptop is not the one the
# defect was measured on — the ceiling is a property of the stack, not of the numbers.
EV = Path("/Users/abrahamcampos/Desktop/nanoscope-example/events/events_nm_0005.csv")
if EV.exists():
    frame = pd.read_csv(EV)
else:
    rng = np.random.default_rng(0)
    n = 982
    frame = pd.DataFrame({"dwell_s": 10.0 ** rng.normal(-2.0, .12, n),
                          "depth_pA": np.abs(rng.normal(100, 20, n)),
                          "frac_depth": np.abs(rng.normal(.25, .05, n)),
                          "snr": np.abs(rng.normal(9, 2, n)),
                          "accepted": np.ones(n, bool),
                          "filter_limited": np.zeros(n, bool)})
print(f"data    : {len(frame)} events from {EV.name if EV.exists() else 'a synthetic stand-in'}")

w = PlotsWindow(Session()); w.set_data({"apo −700 mV": frame})
w.resize(1120, 720); w.show(); pump()
w.set_plot_type("Histogram"); w.set_histogram_column("dwell_s"); w.set_fit("none")
for b in w.findChildren(QtWidgets.QPushButton):
    if b.text().strip().lower().startswith("plot"): b.click(); break
pump(20)


def controls_scroll() -> QtWidgets.QScrollArea | None:
    """The scroll area holding the controls column — found through the Undo button.

    Found rather than read off an attribute, and it always was: `_controls_scroll` was a
    handle a middle round of this fix kept for the reveal it has since deleted, so it
    exists on NEITHER the pre-sweep build nor the current one, and this file has to run
    identically on both. The layout answers instead — the column is the QScrollArea whose
    `widget()` is an ancestor of the Undo button.
    """
    for sa in w.findChildren(QtWidgets.QScrollArea):
        host = sa.widget()
        if host is not None and host.isAncestorOf(w._undo_btn):
            return sa
    return None


def on_screen_ceiling_text() -> list[str]:
    """Every label mentioning the ceiling that the user could actually READ right now.

    `isHidden()` is not that question: it stays False for a label scrolled a thousand
    pixels out of its viewport. Two measurements, both required — `visibleRegion()`,
    which Qt computes from the clip of every ancestor, and the label's rect mapped into
    the scroll viewport, which says in plain coordinates whether the thing overlaps the
    hole it is seen through.
    """
    out = []
    sa = controls_scroll()
    for lb in w.findChildren(QtWidgets.QLabel):
        if "ceiling" not in lb.text() or not lb.isVisible():
            continue
        if lb.visibleRegion().isEmpty():
            continue
        if sa is not None and sa.widget() is not None and sa.widget().isAncestorOf(lb):
            vp = sa.viewport()
            if not QtCore.QRect(lb.mapTo(vp, QtCore.QPoint(0, 0)),
                                lb.size()).intersects(vp.rect()):
                continue
        out.append(lb.text())
    return out


def open_popups() -> list:
    """Every visible POPUP window — a combo dropdown, a menu — and nothing else.

    Masked with `WindowType_Mask` deliberately: `Qt.Popup` is a multi-bit value that
    contains `Qt.Window`, so the obvious `flags & Qt.Popup` reports this very window as a
    popup and would report "no dropdown opened" never, or always, but never truthfully.
    """
    return [x for x in QtWidgets.QApplication.topLevelWidgets()
            if x.isVisible()
            and (x.windowFlags() & QtCore.Qt.WindowType_Mask) == QtCore.Qt.Popup]


def press_undo_shortcut() -> None:
    """One real Ctrl+Z at the window, so the window's own QShortcut is what answers.

    The walk-back below calls `undo_config` directly because 200 of them is plumbing;
    the ONE press that matters — the refused one — goes through the keyboard, because
    the keyboard is the path the whole defect is about.
    """
    combo = QtGui.QKeySequence(QtGui.QKeySequence.Undo)[0]
    QtWidgets.QApplication.sendEvent(
        w, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, combo.key(),
                           combo.keyboardModifiers()))
    pump(6)


OVER = 10
for i in range(pp._HISTORY_DEPTH + OVER):                 # 210 real changes, one control
    w.set_bins(10 + (i % 400)); pump(2)
pump(5)

back, forward = w._history.depth()
dropped = w._history.dropped
tip = w._undo_btn.toolTip()
print(f"after {pp._HISTORY_DEPTH + OVER} changes : depth={(back, forward)}  dropped={dropped}")
print(f"undo tip: {tip!r}")
print(f"on screen: {on_screen_ceiling_text()!r}")

silent_while_editing = bool(dropped) and "ceiling" not in tip and not on_screen_ceiling_text()

undos = 0
while w.undo_config():                                    # walk the stack to the floor
    undos += 1
    if undos > pp._HISTORY_DEPTH + OVER: break
pump(5)
floor_tip = w._undo_btn.toolTip()
print(f"after {undos} undos : enabled={w._undo_btn.isEnabled()}  refused={w.undo_config()!r}")
print(f"floor tip: {floor_tip!r}")

floor_lies = bool(dropped) and "ceiling" not in floor_tip

# THE KEYBOARD USER'S OWN MOMENT. Put the column back where someone who only ever
# pressed Ctrl+Z would have left it — the top, untouched — and make the refused press for
# real. It must change nothing at all, the view included: this is the gesture three
# rounds answered with a scroll that ambushed whoever made it.
sa = controls_scroll()
bar = sa.verticalScrollBar() if sa is not None else None
if bar is not None:
    bar.setValue(0); pump(5)
before = bar.value() if bar is not None else -1
print(f"\nscrolled back to the top: scrollbar={before}"
      f"{f' of {bar.maximum()}' if bar is not None else ''}  "
      f"on screen: {on_screen_ceiling_text()!r}")

press_undo_shortcut()                                     # the refused 201st press
after = bar.value() if bar is not None else -1
print(f"after the refused Ctrl+Z: scrollbar={before} -> {after}")

# ...and again from the parking the gesture actually implies: the user has scrolled down
# to the Undo button they are pressing. That is where the deleted reveal did its damage,
# so it is where the absence of it has to be shown.
kbd_moved = after != before
if bar is not None and sa is not None:
    host = sa.widget()
    top = w._undo_btn.mapTo(host, QtCore.QPoint(0, 0)).y()
    bar.setValue(min(bar.maximum(),
                     top + w._undo_btn.height() - sa.viewport().height()))
    pump(6)
    parked_k = bar.value()
    point_k = w._undo_btn.mapTo(
        w, QtCore.QPoint(w._undo_btn.width() // 2, w._undo_btn.height() // 2))
    under_k = w.childAt(point_k)
    press_undo_shortcut()
    kbd_moved = kbd_moved or bar.value() != parked_k or w.childAt(point_k) is not under_k
    print(f"parked at the fold, one more refused Ctrl+Z: scrollbar={parked_k} -> "
          f"{bar.value()}  pointer over {type(under_k).__name__} -> "
          f"{type(w.childAt(point_k)).__name__}")

# WHERE THE STANDING LINE IS, READ OFF THE SCREEN. It is not on screen for free and this
# file does not pretend it is: parked with the Undo button flush at the fold it is still
# below the viewport. It is found by scrolling on to the controls it explains, so that is
# what is driven — and the sentence has to be there, whole, when you get there.
line_at_the_bottom = []
if bar is not None:
    bar.setValue(bar.maximum()); pump(6)
    line_at_the_bottom = on_screen_ceiling_text()
print(f"scrolled to the foot of the controls column (scrollbar={bar.value() if bar else -1}): "
      f"on screen: {line_at_the_bottom!r}")

never_says_it = bool(dropped) and not line_at_the_bottom

# IS THE DETECTOR ABOVE WORTH ANYTHING? Scroll the column away again and demand it go
# quiet. If it still reports the line, it is reading `isHidden()` in some other clothes
# and every "on screen" printed above is worthless.
blind = False
if bar is not None:
    bar.setValue(0); pump(5)
    still = on_screen_ceiling_text()
    blind = bool(still)
    print(f"self-check — scrolled away again: on screen: {still!r} "
          f"({'BLIND' if blind else 'the check can tell the difference'})")

# THE MOUSE USER'S OWN MOMENT, DRIVEN. Their refused undo is the press on the DISABLED
# Undo button — no `clicked` is emitted, so nothing in the window hears it as an undo —
# and they are parked where somebody who scrolled to the button and stopped would be,
# with the button flush at the fold and the sentence just under it. Two things are
# demanded: the window must TELL them (the tooltip on the button their pointer is on,
# which Qt does render for a disabled widget), and it must NOT MOVE. The clicks go to the
# WINDOW HANDLE at a point, not to the button: `QTest.mouseClick(widget, ...)` calls
# `qApp.notify(widget)` and would deliver to the button whether or not the button is
# still there, which is precisely the thing being measured.
mouse_moved = mouse_swapped = mouse_popped = False
mouse_tip = ""
sa = controls_scroll()
bar = sa.verticalScrollBar() if sa is not None else None
if bar is not None and w.windowHandle() is not None:
    host = sa.widget()
    top = w._undo_btn.mapTo(host, QtCore.QPoint(0, 0)).y()
    bar.setValue(min(bar.maximum(),
                     top + w._undo_btn.height() - sa.viewport().height()))
    pump(6)
    parked = bar.value()
    point = w._undo_btn.mapTo(
        w, QtCore.QPoint(w._undo_btn.width() // 2, w._undo_btn.height() // 2))
    under_before = w.childAt(point)
    fired: list[int] = []
    w._undo_btn.clicked.connect(lambda *_: fired.append(1))
    print(f"\nparked at the fold: scrollbar={parked}  pointer over "
          f"{type(under_before).__name__} enabled={w._undo_btn.isEnabled()}  "
          f"on screen: {on_screen_ceiling_text()!r}")

    QTest.mouseClick(w.windowHandle(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
    pump(6)
    after_one = bar.value()
    QTest.mouseClick(w.windowHandle(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
    pump(6)
    after_two = bar.value()
    QTest.mouseClick(w.windowHandle(), QtCore.Qt.RightButton, QtCore.Qt.NoModifier, point)
    pump(6)

    mouse_moved = {parked, after_one, after_two, bar.value()} != {parked}
    mouse_swapped = w.childAt(point) is not under_before
    mouse_popped = bool(open_popups())
    print(f"after the refused click: clicked signals={len(fired)}  scrollbar={parked} -> "
          f"{after_one} -> {after_two} -> {bar.value()} (left, left, right)  pointer now over "
          f"{type(w.childAt(point)).__name__}")
    print(f"second click at the SAME point: popups opened="
          f"{[type(x).__name__ for x in open_popups()]}")
    for x in open_popups():
        x.hide()
    pump(3)

    # And what the click DOES buy them: the tooltip, rendered on the dead button.
    QtWidgets.QToolTip.hideText(); pump(3)
    local = QtCore.QPoint(w._undo_btn.width() // 2, w._undo_btn.height() // 2)
    QtWidgets.QApplication.sendEvent(
        w._undo_btn, QtGui.QHelpEvent(QtCore.QEvent.ToolTip, local,
                                      w._undo_btn.mapToGlobal(local)))
    pump(6)
    mouse_tip = QtWidgets.QToolTip.text() if QtWidgets.QToolTip.isVisible() else ""
    print(f"tooltip rendered on the DISABLED button: {mouse_tip!r}")
    QtWidgets.QToolTip.hideText(); pump(3)

ambushed = mouse_moved or mouse_swapped or mouse_popped or kbd_moved
mouse_told_nothing = bool(dropped) and "ceiling" not in mouse_tip

bad = (silent_while_editing or floor_lies or never_says_it or blind
       or ambushed or mouse_told_nothing)
if bad:
    if blind:
        print("REPRODUCED — this check cannot tell an on-screen sentence from a "
              "scrolled-away one, so it cannot certify anything.")
    elif ambushed:
        also = (" and the user's next click opened a dropdown"
                if mouse_popped else "")
        print("REPRODUCED — a refused undo moved the controls column out from under a "
              "pointer that never moved" + also + ". The user asked to undo and got a "
              "different widget.")
    elif mouse_told_nothing:
        print(f"REPRODUCED — {dropped} step{'s' if dropped != 1 else ''} were discarded "
              "and the mouse user's refused click leaves them nothing at all: the "
              f"tooltip on the dead Undo button reads {mouse_tip!r}.")
    else:
        print(f"REPRODUCED — {dropped} step{'s' if dropped != 1 else ''} were "
              f"discarded at the {pp._HISTORY_DEPTH}-step ceiling and the window never "
              f"said so anywhere the user could read it.")
else:
    # The report names only gestures this file actually made, and claims no movement it
    # did not measure: the parenthesised "step(s)" was the tic an earlier round removed
    # from both windows, and certifying a click nobody clicked was the same fault one
    # level up. Certifying a scroll that no longer happens would be the third.
    print(f"disclosed: the window names its {pp._HISTORY_DEPTH}-step ceiling and the "
          f"{dropped} step{'s' if dropped != 1 else ''} it cost — in the Undo tooltip "
          f"while editing, on the dead Undo button's own tooltip at the floor (rendered "
          f"on a DISABLED widget, driven above), and on a standing line in the "
          f"Configuration group, read off the screen once the column is scrolled to the "
          f"controls it explains. The refused undo was made both ways, keyboard and "
          f"mouse, at the parking each implies, and moved the column zero pixels.")
sys.exit(7 if bad else 0)
