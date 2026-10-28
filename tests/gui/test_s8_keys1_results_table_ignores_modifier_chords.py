"""S8-KEYS1 -- the Results table curates on an UNMODIFIED a / r / f / j / k and on nothing else.

MEASURED DEFECT (sweep 8, owner's -600 mV recording, Schmitt 317 rows, keyboard focus in the
Results table -- where Shift+Cmd+R hands it on purpose "so a/r/f keep working"). The table's
key handler was given ``event.key()`` and nothing else, so a chord on a curation letter WAS the
bare letter: Cmd+A (Qt: Ctrl+A) accepted the selected row, Cmd+R rejected it, Cmd+F flagged it,
and so did Ctrl+A, Option+R, Option+F and Cmd+Shift+A -- every one a silent write ("Undo Curate
event" in the Edit menu, nothing in the status bar) -- while Select All, the gesture the table's
ExtendedSelection exists for, never selected anything (1 row stayed selected of 317). The same
hole sat in the second door the keys come through, the container's ``keyPressEvent``, which every
key a child control ignores climbs to: Cmd+R on the focused "Columns..." button rejected the
selected row.

THE FIX. ``EventTable`` hands its curation handler a key only when no Ctrl / Meta / Alt / Shift
is held (``_handle_curation_event``, the one gate in front of both ``eventFilter`` and
``keyPressEvent``; the keypad flag alone is no modifier, the idiom the gallery's arrow keys
already use). Anything else is left unhandled and goes on to QTableView: Cmd+A selects every row,
the other chords are ignored and keep climbing to the window, where Shift+Cmd+F and Shift+Cmd+R
are menu actions and win before the table sees them, as before.

WHAT THESE TESTS DRIVE.
* A standalone ``EventTable`` on a synthetic frame, with real ``QKeyEvent`` objects sent to the
  view (and to the container): every curation key against every Mac chord of it (Cmd is Qt's
  ControlModifier, Ctrl is MetaModifier), asking the frame whether anything was written and the
  event whether the table kept the key; Cmd+A as Select All; the bare keys still working, with
  the keypad flag and with Caps Lock (Qt reports no modifier for it); auto-repeat; Shift held
  with Caps Lock on; a letter typed into an open cell editor; the container door.
* The wired ``MainWindow`` on a synthetic recording with a Schmitt detection, keys delivered the
  way the platform delivers them (to the active window, so the shortcut map runs first): the
  seven chords the sweep measured, then Shift+Cmd+F and Shift+Cmd+R, which must still reach their
  menu actions from the table.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.events import FLAG_CURATED
from nanoscope.gui import panels as panels_mod

from ._inspector_hosting import (  # noqa: F401  (fixtures)
    build_window, close_window, private_settings, settle)

#: The curation keys, and the Mac chords of them in Qt's names: Cmd is ControlModifier and
#: Ctrl is MetaModifier. Each chord is something a user types for another reason (Select All,
#: Find, Reload, Option-letters, Shift for a capital).
KEYS = {"a": Qt.Key_A, "r": Qt.Key_R, "f": Qt.Key_F, "j": Qt.Key_J, "k": Qt.Key_K}
CHORDS = {
    "Cmd": Qt.ControlModifier,
    "Ctrl": Qt.MetaModifier,
    "Option": Qt.AltModifier,
    "Shift": Qt.ShiftModifier,
    "Cmd+Shift": Qt.ControlModifier | Qt.ShiftModifier,
    "Cmd+Option": Qt.ControlModifier | Qt.AltModifier,
}
N = 12


# ================================================================================================
# builders
# ================================================================================================
def _frame(n: int = N) -> pd.DataFrame:
    """``event_id`` is the source row, so a test can name a row by either."""
    depth = [100.0 - 7.0 * i if i % 2 else 20.0 + 5.0 * i for i in range(n)]
    return pd.DataFrame({
        "event_id": list(range(n)),
        "recording": ["r"] * n,
        "pore_id": ["p"] * n,
        "start_idx": [10 * i for i in range(n)],
        "end_idx": [10 * i + 5 for i in range(n)],
        "t_start_s": [0.1 * i for i in range(n)],
        "dwell_s": [0.001] * n,
        "baseline_pA": [-60.0] * n,
        "depth_pA": depth,
        "frac_depth": [0.2] * n,
        "ecd_pA_s": [0.1] * n,
        "max_dev_pA": [10.0] * n,
        "mean_dev_pA": [9.0] * n,
        "rise_s": [0.0019] * n,
        "fall_s": [0.0028] * n,
        "snr": [3.0] * n,
        "n_levels": [2] * n,
        "filter_limited": [False] * n,
        "accepted": [True] * n,
        "flag": [""] * n,
        "added": [False] * n,
    })


def _table(qtbot, n: int = N) -> panels_mod.EventTable:
    t = panels_mod.EventTable()
    qtbot.addWidget(t)
    t.set_events(_frame(n))
    t.resize(900, 400)
    t.show()
    qtbot.waitExposed(t)
    return t


def _send(widget, key, mods=Qt.NoModifier, *, text=None, autorep=False) -> QtGui.QKeyEvent:
    """One key press handed to ``widget`` as a QKeyEvent -- NOT through the shortcut map, so a
    chord that is also a menu action cannot fire it -- and the event handed back, so a test can
    ask whether the receiver KEPT the key (accepted) or let it go on (ignored)."""
    if text is None:
        letter = chr(int(key))
        text = letter if mods & Qt.ShiftModifier else letter.lower()
    ev = QtGui.QKeyEvent(QEvent.KeyPress, key, mods, text, autorep)
    QtWidgets.QApplication.sendEvent(widget, ev)
    return ev


def _arm(t: panels_mod.EventTable, key, event_id: int = 4) -> int:
    """Select one row and put it where the bare key WOULD change it, so a chord that wrongly
    curates cannot hide behind a no-op write: 'a' needs a rejected row, 'r' an accepted one,
    'f' an unflagged one, j / k a row with a neighbour on each side."""
    t.select_event(event_id)
    if key == Qt.Key_A:
        t._model.set_accepted(event_id, False)
    t.view.setFocus()
    return event_id


def _written(before: pd.DataFrame, t: panels_mod.EventTable) -> list[str]:
    """The columns of the table's frame that differ from the snapshot -- empty when the key
    wrote nothing at all."""
    after = t._model.df
    return [c for c in before.columns if not before[c].equals(after[c])]


def _accepted(t: panels_mod.EventTable) -> list[bool]:
    return [bool(v) for v in t._model.df["accepted"].to_numpy()]


def _flags(t: panels_mod.EventTable) -> list[str]:
    return [str(v) for v in t._model.df["flag"].to_numpy()]


def _row(t: panels_mod.EventTable) -> int:
    return t.view.currentIndex().row()


# ================================================================================================
# 1. the table: a chord on a curation letter is not the bare letter
# ================================================================================================
@pytest.mark.parametrize("chord", list(CHORDS))
@pytest.mark.parametrize("name", list(KEYS))
def test_a_chord_on_a_curation_key_writes_nothing_and_moves_nothing(qtbot, name, chord):
    """Every Ctrl / Meta / Alt / Shift chord on a, r, f, j, k. Measured before the fix: all
    thirty were the bare key -- accepted, rejected, flagged or walked a row."""
    key, mods = KEYS[name], CHORDS[chord]
    t = _table(qtbot)
    row = _arm(t, key)
    before = t._model.df.copy(deep=True)

    _send(t.view, key, mods)

    assert _written(before, t) == [], (
        f"{chord}+{name} was read as the bare key and wrote {_written(before, t)} "
        f"(accepted {before['accepted'].iloc[row]} -> {_accepted(t)[row]}, "
        f"flag {before['flag'].iloc[row]!r} -> {_flags(t)[row]!r})")
    if mods != Qt.ShiftModifier or name in ("j", "k"):
        # (Shift+A / R / F is the one chord QTableView's own type-to-search may answer, by
        # looking for a row whose cell starts with the capital -- navigation, never a write;
        # no cell starts with a J or a K, so those two must stay put.)
        assert _row(t) == row, f"{chord}+{name} walked the current row {row} -> {_row(t)}"
        if not (name == "a" and chord == "Cmd"):           # Cmd+A is Select All: next test
            assert t.selected_source_rows() == [row]


def test_cmd_a_is_select_all_not_accept(qtbot):
    """The sweep's headline: Cmd+A with the focus in the table accepted the selected row and
    selected nothing. It selects every row, says so under the table, and writes nothing."""
    t = _table(qtbot)
    row = _arm(t, Qt.Key_A)
    before = t._model.df.copy(deep=True)
    assert t.selected_row_count() == 1

    ev = _send(t.view, Qt.Key_A, Qt.ControlModifier)

    assert ev.isAccepted(), "nobody handled Cmd+A"
    assert t.selected_row_count() == N, f"Cmd+A selected {t.selected_row_count()} of {N} rows"
    assert t.selection_label.text() == f"{N} of {N} rows selected  ·  A accepts / R rejects all {N}"
    assert _written(before, t) == [], f"Cmd+A wrote {_written(before, t)}"
    assert _accepted(t)[row] is False, "Cmd+A accepted the rejected row"


def test_select_all_then_the_bare_key_curates_all_of_them(qtbot):
    """What Select All is FOR: the caption's own promise, 'A accepts / R rejects all N'. The bare
    key after the chord is still a curation key."""
    t = _table(qtbot)
    _arm(t, Qt.Key_R)
    _send(t.view, Qt.Key_A, Qt.ControlModifier)
    assert t.selected_row_count() == N

    assert _send(t.view, Qt.Key_R).isAccepted()

    assert _accepted(t) == [False] * N, "R over the Select-All selection did not reject every row"
    _send(t.view, Qt.Key_A)
    assert _accepted(t) == [True] * N, "A over the Select-All selection did not accept every row"


@pytest.mark.parametrize("name", ["r", "f", "j", "k"])
def test_a_chord_the_table_does_not_own_goes_on_to_the_window(qtbot, name):
    """'Lets the key go on': Cmd+R / Cmd+F are not consumed, so Qt keeps delivering them up the
    parent chain to the window (and its shortcuts and fallbacks) instead of dying in the table.
    The bare key, by contrast, is consumed -- which also keeps QTableView's type-to-search from
    jumping to a row whose text starts with the letter."""
    t = _table(qtbot)
    _arm(t, KEYS[name])

    kept = _send(t.view, KEYS[name], Qt.ControlModifier)
    assert not kept.isAccepted(), f"Cmd+{name.upper()} was swallowed by the table"

    assert _send(t.view, KEYS[name]).isAccepted(), f"bare {name} is no longer consumed"


# ================================================================================================
# 2. the container's door
# ================================================================================================
def test_a_chord_that_climbs_to_the_container_is_not_the_bare_key_either(qtbot):
    """``EventTable.keyPressEvent`` is the second door: every key a child control ignores climbs
    to it. Measured before the fix: Cmd+R pressed on the focused 'Columns...' button rejected
    the table's selected row."""
    t = _table(qtbot)
    row = _arm(t, Qt.Key_R, event_id=3)
    before = t._model.df.copy(deep=True)
    t.columns_button.setFocus()

    doors = (
        ("Cmd+R on the focused 'Columns...' button",
         lambda: QTest.keyClick(t.columns_button, Qt.Key_R, Qt.ControlModifier)),
        ("Option+F sent to the container", lambda: _send(t, Qt.Key_F, Qt.AltModifier)),
        ("Cmd+Shift+A sent to the container",
         lambda: _send(t, Qt.Key_A, Qt.ControlModifier | Qt.ShiftModifier)),
    )
    for what, send in doors:                                 # one at a time: R then A would cancel
        send()
        assert _written(before, t) == [], f"{what} wrote {_written(before, t)}"

    # ...while a bare key sent to the container (a synthesised event) is still ours.
    _send(t, Qt.Key_R)
    assert _accepted(t)[row] is False


# ================================================================================================
# 3. what must keep working
# ================================================================================================
@pytest.mark.parametrize("variant", ["plain", "keypad flag", "caps lock on"])
def test_the_bare_keys_still_curate_and_walk(qtbot, variant):
    """The control. A bare key is still read, with the keypad flag Qt raises on some keyboards,
    and with Caps Lock on, which Qt reports as NO modifier -- the key is the capital letter's
    text and nothing else, so a gate on ``text()`` or on Shift would have broken it."""
    mods = Qt.KeypadModifier if variant == "keypad flag" else Qt.NoModifier

    def press(key):
        text = chr(int(key)) if variant == "caps lock on" else None
        return _send(t.view, key, mods, text=text)

    t = _table(qtbot)
    t.select_event(4)
    t.view.setFocus()

    assert press(Qt.Key_R).isAccepted()
    assert _accepted(t)[4] is False
    press(Qt.Key_A)
    assert _accepted(t)[4] is True
    press(Qt.Key_F)
    assert FLAG_CURATED in _flags(t)[4]
    press(Qt.Key_J)
    assert _row(t) == 5
    press(Qt.Key_K)
    press(Qt.Key_K)
    assert _row(t) == 3


def test_shift_with_caps_lock_on_is_still_a_chord(qtbot):
    """Shift inverts Caps Lock, so the text is a lower-case 'r' -- and Shift is held, which is
    all the gate looks at."""
    t = _table(qtbot)
    row = _arm(t, Qt.Key_R)
    before = t._model.df.copy(deep=True)
    _send(t.view, Qt.Key_R, Qt.ShiftModifier, text="r")
    assert _written(before, t) == [], f"Shift+R with Caps Lock on rejected row {row}"


def test_a_held_chord_never_curates_and_a_held_j_keeps_walking(qtbot):
    """Auto-repeat. A held Cmd+A repeats the chord, not the letter; a held j still walks one row
    per repeat, which is what holding it is for."""
    t = _table(qtbot)
    _arm(t, Qt.Key_A, event_id=1)
    for i in range(4):
        _send(t.view, Qt.Key_J, autorep=i > 0)
    assert _row(t) == 5, f"a held j walked to row {_row(t)}, expected 5"

    before = t._model.df.copy(deep=True)
    for i in range(4):
        _send(t.view, Qt.Key_A, Qt.ControlModifier, autorep=i > 0)
    assert t.selected_row_count() == N
    assert _written(before, t) == [], f"a held Cmd+A wrote {_written(before, t)}"


def test_a_letter_typed_into_an_open_cell_editor_is_typed_not_curation(qtbot):
    """No editor opens by itself in this table (NoEditTriggers), but if one is open its keys go
    to the editor: the view's filter never sees them, and Cmd+A inside it selects ITS text."""
    t = _table(qtbot)
    t.select_event(3)
    col = list(t._model.df.columns).index("flag")
    index = t._proxy.index(3, col)
    t.view.openPersistentEditor(index)
    editor = t.view.indexWidget(index)
    assert editor is not None and hasattr(editor, "text")
    before = t._model.df.copy(deep=True)

    for key in (Qt.Key_R, Qt.Key_A, Qt.Key_F):
        QTest.keyClick(editor, key)
    QTest.keyClick(editor, Qt.Key_A, Qt.ControlModifier)

    assert editor.text().endswith("raf"), f"the editor holds {editor.text()!r}"
    assert editor.selectedText() == editor.text()
    assert _written(before, t) == [], f"typing in an editor wrote {_written(before, t)}"
    assert t.selected_source_rows() == [3]


# ================================================================================================
# 4. the wired window: the user's own route
# ================================================================================================
def _activate(window, qtbot) -> None:
    window.show()
    window.raise_()
    window.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is window, timeout=3000)


def _press(window, key, mods=Qt.NoModifier) -> None:
    """A key press+release delivered the way the platform delivers it: to the ACTIVE QWindow, so
    the shortcut map and ShortcutOverride run before the focus widget sees the key."""
    QTest.keyClick(window.windowHandle(), key, mods)
    QTest.qWait(20)


def _undo_text(window) -> tuple[str, bool]:
    """Edit > Undo as the menu offers it (opening the menu commits and re-labels)."""
    window._on_edit_menu_about_to_show()
    return window._act_undo.text(), window._act_undo.isEnabled()


@pytest.fixture
def wired(private_settings, qtbot):
    """A MainWindow around a synthetic recording with a six-event Schmitt detection, active,
    its Results table holding the keyboard on a row."""
    window = build_window(qtbot)
    _activate(window, qtbot)
    table = window._analysis_panel._table
    assert window._results_dock.isVisible() and len(table._model.df) == 6
    table.select_event(2)
    table.view.setFocus()
    settle()
    assert QtWidgets.QApplication.focusWidget() is table.view
    yield window
    close_window(window)


#: The seven chords the sweep's repro measured in the wired app, with the label it printed.
SWEEP_CASES = [
    ("Cmd+A", Qt.Key_A, Qt.ControlModifier),
    ("Cmd+R", Qt.Key_R, Qt.ControlModifier),
    ("Cmd+F", Qt.Key_F, Qt.ControlModifier),
    ("Ctrl+A", Qt.Key_A, Qt.MetaModifier),
    ("Option+R", Qt.Key_R, Qt.AltModifier),
    ("Cmd+Shift+A", Qt.Key_A, Qt.ControlModifier | Qt.ShiftModifier),
    ("Option+F", Qt.Key_F, Qt.AltModifier),
]


@pytest.mark.parametrize("label,key,mods", SWEEP_CASES, ids=[c[0] for c in SWEEP_CASES])
def test_the_chords_the_sweep_measured_curate_nothing_in_the_wired_results_table(
        wired, label, key, mods):
    """The sweep's own seven cases, through the platform's key path with the focus in the Results
    table: not one accept flag or flag token changes and Edit > Undo offers nothing new; Cmd+A
    selects all six rows and the others leave the one selected row alone."""
    table = wired._analysis_panel._table
    if key == Qt.Key_A:
        table._model.set_accepted(2, False)               # a state the bare 'a' would change
    settle()
    before = table._model.df.copy(deep=True)
    undo_before = _undo_text(wired)

    _press(wired, key, mods)

    assert _written(before, table) == [], f"{label} wrote {_written(before, table)}"
    assert _undo_text(wired) == undo_before, f"{label}: Edit > Undo says {_undo_text(wired)}"
    assert QtWidgets.QApplication.focusWidget() is table.view
    if label == "Cmd+A":
        assert table.selected_row_count() == 6, "Cmd+A did not select all six rows"
    else:
        assert table.selected_source_rows() == [2]


def test_shift_cmd_f_and_shift_cmd_r_still_reach_their_menu_actions_from_the_table(
        wired, qtbot, monkeypatch):
    """The two chords main_window relies on staying out of the table's hands. Shift+Cmd+F opens
    Signal conditioning and Shift+Cmd+R hides and re-shows the Results dock -- which hands the
    keyboard back to the table, so that the next bare r still rejects the selected row. Neither
    chord curates (a flag on Shift+Cmd+F is what the always-enabled action was guarding)."""
    table = wired._analysis_panel._table
    opened: list[int] = []
    monkeypatch.setattr(wired, "_open_signal_conditioning", lambda *a, **k: opened.append(1))
    before = table._model.df.copy(deep=True)
    cmd_shift = Qt.ControlModifier | Qt.ShiftModifier

    _press(wired, Qt.Key_F, cmd_shift)
    assert opened == [1], "Shift+Cmd+F no longer reaches Analysis > Signal conditioning..."
    assert _written(before, table) == [], f"Shift+Cmd+F wrote {_written(before, table)}"

    dock = wired._results_dock
    _press(wired, Qt.Key_R, cmd_shift)
    qtbot.waitUntil(lambda: not dock.isVisible(), timeout=3000)
    _press(wired, Qt.Key_R, cmd_shift)
    qtbot.waitUntil(lambda: dock.isVisible(), timeout=3000)
    qtbot.waitUntil(lambda: QtWidgets.QApplication.focusWidget() is table.view, timeout=3000)
    assert _written(before, table) == [], f"Shift+Cmd+R wrote {_written(before, table)}"

    _press(wired, Qt.Key_R)                               # "so a/r/f keep working"
    assert _accepted(table)[2] is False, "the bare r no longer rejects after Shift+Cmd+R"
