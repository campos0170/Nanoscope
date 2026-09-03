"""WA2-TABLE — the events table's A / R / F act on the SELECTION and on nothing else,
and the sentence that says so fits the dock without moving the rows.

Three defects the skeptics found inside the wave-A batch-curation feature, all of them
newly reachable because that feature turned the table into an ``ExtendedSelection`` — the
first time "drop the last selected row" was a gesture a user could make here at all.

WHAT WAS MEASURED ON THE PRE-FIX TREE, in the wired application (a real ``MainWindow``
resized to 1512x900 on a throwaway QSettings profile, the owner's 15 s -1000 mV
apoferritin recording, 115 Schmitt events, real ``QTest`` mouse and key events):

* **WA-R1** — click source row 4, then ⌘-click it OFF. ``selected_source_rows()`` is
  ``[]``, the caption is ``''`` and no row is painted selected — but the current index is
  still parked on proxy row 4, and a single ``R`` **rejected source row 4**, with
  Edit ▸ Undo then reading ``'Undo Curate event'``. Worse without the count ever reaching
  zero: click row 2, ⌘-click row 9 on and off again, and the selection is ``[2]`` while
  the current index is 9 — ``R`` changed **rows [9]**, a row nothing on screen showed as
  chosen, while the highlighted row 2 was left alone.
* **WA-R2** — the caption is an ``_ElidingCaption`` using ``Qt.ElideMiddle``. In the
  shipped Inspector dock at 1512x900 the label is **422 px**; the six-row sentence needed
  **453 px** and the sixty-one-row sentence **465 px**, so both were mid-elided and what
  painted was ``'6 of 115 rows selected  ·  A accepts …ts all 6  ·  right-click for the
  same'``. The run Qt swallowed is ``'/ R rejec'`` — the only naming of the R key on this
  surface.
* **WB1** — the caption was shown and hidden with the selection, so the table's viewport
  went **134 px -> 113 px** the instant a row was clicked and back when it cleared: one
  of the five visible rows appearing and vanishing under the pointer.

Every test here fails against the shipped wave-A behaviour, proved by running this file
with the out-of-tree plugin ``wa2_prefix`` (``PYTHONPATH=<scratch>/wa2_prefix
WA2_REVERT=1 ... -p wa2_prefix``), which restores the four pre-fix methods at runtime.
The source is never edited back and forth.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtTest import QTest

from nanoscope.gui import panels as panels_mod

#: The width the events table's caption is given in the shipped Inspector dock with the
#: owner's window at 1512x900 — measured on the wired app, not guessed.
INSPECTOR_CAPTION_WIDTH_PX = 422


# ================================================================================================
# builders
# ================================================================================================
def _frame(n: int = 12) -> pd.DataFrame:
    """A frame whose ``depth_pA`` order is NOT its ``event_id`` order, so the sort proxy
    really permutes and a view row is never accidentally an event row."""
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


def _table(qtbot, df: pd.DataFrame | None = None) -> panels_mod.EventTable:
    t = panels_mod.EventTable()
    qtbot.addWidget(t)
    t.set_events(_frame() if df is None else df)
    t.resize(900, 400)
    t.show()
    qtbot.waitExposed(t)
    return t


def _row_center(t, proxy_row: int) -> QtCore.QPoint:
    # Column 2 on purpose: a click must select a row, not land on a cell that could be
    # mistaken for an editor, and the defect has nothing to do with which column was hit.
    idx = t._proxy.index(proxy_row, 2)
    t.view.scrollTo(idx)
    return t.view.visualRect(idx).center()


def _click(t, proxy_row: int, mods=Qt.NoModifier) -> None:
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, mods, _row_center(t, proxy_row))
    QtWidgets.QApplication.processEvents()


def _accepted(t) -> list[bool]:
    df = t._model.df
    return [bool(v) for v in df["accepted"].to_numpy()]


# ================================================================================================
# 1. WA-R1 — one selection, and the keys read it
# ================================================================================================
def test_r_writes_nothing_after_the_cmd_click_that_dropped_the_only_selected_row(qtbot):
    """The caption teaches "⌘-click to select more" and the same click takes one away.
    Follow it to empty the selection and the pre-fix handler still had a current index to
    fall back on: measured in the wired app, R rejected source row 4 with nothing
    highlighted and nothing on screen naming it."""
    t = _table(qtbot)
    _click(t, 3)
    before = _accepted(t)
    _click(t, 3, Qt.ControlModifier)
    assert t.selected_source_rows() == [], "the ⌘-click did not drop the row"
    assert t.view.currentIndex().isValid(), \
        "this test is pointless unless a current index survives the empty selection"

    t.view.setFocus()
    QTest.keyClick(t.view, Qt.Key_R)
    qtbot.wait(30)

    assert _accepted(t) == before, (
        "R curated a row with nothing highlighted: "
        f"{[i for i, (a, b) in enumerate(zip(before, _accepted(t))) if a != b]}")


def test_r_curates_the_highlighted_row_and_not_the_one_the_cursor_was_left_on(qtbot):
    """The nastier half, because the count never reaches zero and the caption still reads
    "1 of 12 rows selected". Select row 2, ⌘-click row 9 on and then off: the selection is
    [2], the current index is 9. Measured pre-fix in the wired app: R changed row 9."""
    t = _table(qtbot)
    _click(t, 2)
    _click(t, 9, Qt.ControlModifier)
    _click(t, 9, Qt.ControlModifier)
    armed = t.selected_source_rows()
    current = t._current_source_row()
    assert len(armed) == 1 and current != armed[0], (
        f"setup did not produce the divergence: armed={armed} current={current}")

    t.view.setFocus()
    QTest.keyClick(t.view, Qt.Key_R)
    qtbot.wait(30)

    moved = [i for i, v in enumerate(_accepted(t)) if not v]
    assert moved == armed, (
        f"R acted on {moved} while the table was showing {armed} highlighted")


def test_the_keys_and_the_caption_count_the_same_selection(qtbot):
    """One source of truth, stated as an identity: whatever ``armed_rows`` returns is what
    ``selected_row_count`` counted and what the view has highlighted — through every
    gesture, including the one that empties the selection."""
    t = _table(qtbot)
    for gesture in ((3, Qt.NoModifier), (7, Qt.ShiftModifier), (5, Qt.ControlModifier),
                    (5, Qt.ControlModifier), (3, Qt.ControlModifier)):
        _click(t, *gesture)
        painted = sorted({t._proxy.mapToSource(i).row()
                          for i in t.view.selectionModel().selectedRows()})
        assert t.armed_rows() == painted
        assert t.selected_row_count() == len(painted)


def test_f_marks_nothing_when_no_row_is_highlighted(qtbot):
    """The curator's own flag is a write to the shared frame exactly as an accept is, so
    it answers to the same scope."""
    t = _table(qtbot)
    _click(t, 4)
    before = t._model.df["flag"].tolist()
    _click(t, 4, Qt.ControlModifier)
    t.view.setFocus()
    QTest.keyClick(t.view, Qt.Key_F)
    qtbot.wait(30)
    assert t._model.df["flag"].tolist() == before, \
        "F flagged the row the cursor was left on, which nothing on screen showed as chosen"


def test_a_key_into_an_empty_selection_says_so_instead_of_writing(qtbot):
    """Silence would be a second kind of wrong. The gallery answers the same miss with a
    sentence ("R labels the SELECTED tiles and none is selected…"); the table now does the
    same, in the line the caption already reserves."""
    t = _table(qtbot)
    _click(t, 1)
    _click(t, 1, Qt.ControlModifier)
    assert t.selection_label.text() == "", "a standing '0 selected' would be chrome"
    t.view.setFocus()
    QTest.keyClick(t.view, Qt.Key_A)
    qtbot.wait(30)
    note = t.selection_label.text()
    assert "Nothing is selected" in note, note
    # ...and it is a reply to that gesture, not a standing sentence.
    _click(t, 6)
    assert "Nothing is selected" not in t.selection_label.text()


# ================================================================================================
# 2. WA-R2 — the sentence fits the dock it ships in
# ================================================================================================
@pytest.mark.parametrize("n_rows", [2, 6, 11])
def test_the_multi_row_caption_paints_whole_at_the_width_the_inspector_dock_gives_it(
        qtbot, n_rows):
    """422 px is not a guess: it is what this label measures in the shipped Inspector dock
    with the owner's window at 1512x900. Pre-fix the six-row sentence needed 453 px and
    ElideMiddle ate '/ R rejec' — the R key's only appearance on this surface."""
    t = _table(qtbot)
    lab = t.selection_label
    lab.setFixedWidth(INSPECTOR_CAPTION_WIDTH_PX)
    qtbot.wait(20)
    _click(t, 0)
    _click(t, n_rows - 1, Qt.ShiftModifier)
    qtbot.wait(20)
    assert t.selected_row_count() == n_rows

    full = lab.text()
    painted = QtWidgets.QLabel.text(lab)
    needed = QFontMetrics(lab.font()).horizontalAdvance(full)
    assert painted == full, (
        f"the caption is elided at {INSPECTOR_CAPTION_WIDTH_PX} px: it needs {needed} px\n"
        f"  full   : {full!r}\n  painted: {painted!r}")
    assert "R rejects" in painted, f"the R key is not named on screen: {painted!r}"


def test_a_caption_that_still_overflows_loses_its_tail_not_its_middle(qtbot):
    """Belt and braces for a font or a dock this was not measured in: a sentence's
    disposable part is its end, never nine characters out of the middle of a verb."""
    t = _table(qtbot)
    lab = t.selection_label
    lab.setFixedWidth(120)          # far too narrow for any of the sentences
    qtbot.wait(20)
    _click(t, 0)
    _click(t, 5, Qt.ShiftModifier)
    qtbot.wait(20)
    painted = QtWidgets.QLabel.text(lab)
    assert painted != lab.text(), "120 px should not fit the whole sentence"
    assert painted.endswith("…"), (
        f"the cut fell inside the sentence instead of at its end: {painted!r}")


def test_the_right_click_route_survives_in_the_tooltip(qtbot):
    """The clause that had to leave the one painted line is not gone — it moved to the
    tooltip. Nothing was deleted from the disclosure to make it fit."""
    t = _table(qtbot)
    _click(t, 0)
    _click(t, 5, Qt.ShiftModifier)
    qtbot.wait(20)
    assert "right-click" in t.selection_label.toolTip(), t.selection_label.toolTip()


# ================================================================================================
# 3. WB1 — the caption costs the same rows whether or not anything is selected
# ================================================================================================
def test_selecting_a_row_does_not_change_the_height_of_the_table_under_the_pointer(qtbot):
    """Measured in the wired app at 1512x900: the viewport went 134 px -> 113 px on the
    first click and back on Escape — one of five rows appearing and vanishing while the
    mouse sat over the table. The line is now permanent and only its text moves."""
    t = _table(qtbot)
    qtbot.wait(30)
    empty_h = t.view.viewport().height()
    last_row_empty = t.view.rowAt(t.view.viewport().height() - 1)

    _click(t, 2)
    qtbot.wait(30)
    one_h = t.view.viewport().height()

    _click(t, 7, Qt.ShiftModifier)
    qtbot.wait(30)
    many_h = t.view.viewport().height()

    t.view.clearSelection()
    qtbot.wait(30)
    back_h = t.view.viewport().height()
    last_row_back = t.view.rowAt(t.view.viewport().height() - 1)

    assert empty_h == one_h == many_h == back_h, (
        f"the table resized under the pointer: {empty_h} -> {one_h} -> {many_h} -> {back_h}")
    assert last_row_empty == last_row_back, (
        "the last visible row changed with the selection: "
        f"{last_row_empty} -> {last_row_back}")


def test_the_reserved_caption_line_stays_blank_until_it_has_something_to_say(qtbot):
    """Reserving the space is not the same as filling it: a permanent "0 selected" would
    be chrome. The widget stays in the layout; its text does not."""
    t = _table(qtbot)
    assert t.selection_label.text() == ""
    _click(t, 0)
    assert "1 of 12 rows selected" in t.selection_label.text()
    t.view.clearSelection()
    qtbot.wait(20)
    assert t.selection_label.text() == ""


def test_the_caption_never_widens_the_dock_it_lives_in(qtbot):
    """The reason this label is an ``_ElidingCaption`` at all, re-pinned now that it is
    never hidden: a hidden widget could not have held the dock open, a permanent one can.
    108 px is the table's measured floor in the shipped app."""
    t = _table(qtbot)
    _click(t, 0)
    _click(t, 11, Qt.ShiftModifier)
    qtbot.wait(20)
    assert t.selection_label.minimumSizeHint().width() == 0
    assert t.minimumSizeHint().width() <= 1100, t.minimumSizeHint().width()
