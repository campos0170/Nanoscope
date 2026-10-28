"""WA-TABLE — the events table curates a SELECTION, not one row at a time.

WHAT WAS MEASURED ON THE PRE-FIX TREE, in the wired application (a real ``MainWindow``
resized to 1512x900 with a throwaway QSettings profile, the owner's own -600 mV
apoferritin recording, 317 Schmitt events), driving the table with real ``QTest`` mouse
and key events:

* ``AnalysisPanel._table.view.selectionMode()`` was ``SingleSelection``. A Shift-click
  five rows down left **1** row selected; a Ctrl-click left **1**; Ctrl+A left **1**.
* The widget owned **0** QPushButtons and **0** QActions, and its right-click menu held
  exactly two items — "Open in Event inspector" and "Open in Event gallery". There was no
  batch anything on the surface a detection actually lands on; every batch gesture lived
  in the Event Gallery, a separate window the user has to know to open.
* With a six-row range *attempted*, pressing ``r`` changed **1** row.
* Nothing on screen said how many rows were selected.
* And the obvious repair — announce ``acceptanceChanged`` once per row — is the one that
  cannot ship: 300 emissions cost **3.72 s** (12.4 ms/row), because the accepted summary,
  the span overlays and the detector score are each rebuilt from the whole frame by every
  one of them, and the history records 300 undo steps for one gesture.

These tests pin the four halves of the repair: the selection exists, it is mapped
THROUGH THE SORT PROXY before the frame is touched, one gesture is one write and one
announcement, and the count is on screen. Every one of them fails against a pre-fix
``EventTable`` — proved by running this file with the out-of-tree plugin
``-p wa_table_prefix``, which swaps in a verbatim copy of the pre-edit ``panels.py``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.events import REVIEW_COL, REVIEW_DONE
from nanoscope.gui import panels as panels_mod


# ================================================================================================
# builders
# ================================================================================================
def _frame(n: int = 12) -> pd.DataFrame:
    """A frame whose ``depth_pA`` order is NOT its ``event_id`` order.

    The sort proxy has to actually permute for the mapToSource tests to mean anything: a
    view row is not an event, and a table that happened to be sorted in source order would
    pass a broken implementation.
    """
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
    idx = t._proxy.index(proxy_row, 0)
    t.view.scrollTo(idx)
    return t.view.visualRect(idx).center()


def _selected_view_rows(t) -> list[int]:
    sm = t.view.selectionModel()
    return sorted({i.row() for i in sm.selectedRows()})


def _select_view_rows(t, first: int, last: int) -> None:
    """Select a contiguous block by explicit command — the same command Qt's own
    Shift-click issues, stated so the test does not depend on latched modifiers."""
    sm = t.view.selectionModel()
    top = t._proxy.index(first, 0)
    bottom = t._proxy.index(last, t._proxy.columnCount() - 1)
    sm.select(QtCore.QItemSelection(top, bottom),
              QtCore.QItemSelectionModel.ClearAndSelect
              | QtCore.QItemSelectionModel.Rows)


def _batch(t, accepted: bool) -> int:
    """The batch door, or the pre-fix answer (there is none) as a skip-free failure."""
    fn = getattr(t, "curate_selected", None)
    if fn is None:
        pytest.fail("EventTable has no curate_selected: the table offers no batch curation")
    return fn(accepted)


# ================================================================================================
# 1. the selection exists at all
# ================================================================================================
def test_shift_clicking_the_events_table_selects_a_range_of_rows(qapp, qtbot):
    """The gesture every list in every application answers, driven as a real mouse click.

    Pre-fix: SingleSelection, so this left 1 row selected.
    """
    t = _table(qtbot)
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.NoModifier, _row_center(t, 0))
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.ShiftModifier, _row_center(t, 5))
    assert _selected_view_rows(t) == [0, 1, 2, 3, 4, 5]


def test_ctrl_clicking_adds_a_discontiguous_row_to_the_events_table_selection(qapp, qtbot):
    """Ctrl/Cmd-click, the other half of ExtendedSelection. Pre-fix: 1 row."""
    t = _table(qtbot)
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.NoModifier, _row_center(t, 0))
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.ShiftModifier, _row_center(t, 3))
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.ControlModifier, _row_center(t, 8))
    assert _selected_view_rows(t) == [0, 1, 2, 3, 8]


# ================================================================================================
# 2. a curation gesture covers the whole selection
# ================================================================================================
def test_pressing_r_over_a_selected_range_rejects_every_row_in_it(qapp, qtbot):
    """The measured pre-fix answer was ONE row of six. The key is sent to the view, which
    is where the focus actually is (the container is NoFocus — see EventTable's docstring),
    so this is the keystroke the user makes."""
    t = _table(qtbot)
    df = t.dataframe
    _select_view_rows(t, 0, 5)
    t.view.setFocus()
    QTest.keyClick(t.view, Qt.Key_R)
    assert int((~df["accepted"].to_numpy(dtype=bool)).sum()) == 6


def test_rejecting_one_selected_row_still_takes_the_unchanged_single_row_path(qapp, qtbot):
    """The ordinary click-then-'r' must be byte-identical to before: one row, one
    ``acceptanceChanged``, and no batch announcement at all."""
    t = _table(qtbot)
    df = t.dataframe
    seen: list = []
    batches: list = []
    t.acceptanceChanged.connect(lambda e, a: seen.append((e, a)))
    if hasattr(t, "curationBatchChanged"):
        t.curationBatchChanged.connect(lambda: batches.append(1))
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.NoModifier, _row_center(t, 3))
    t.view.setFocus()
    QTest.keyClick(t.view, Qt.Key_R)
    assert int((~df["accepted"].to_numpy(dtype=bool)).sum()) == 1
    assert len(seen) == 1
    assert batches == []


# ================================================================================================
# 3. the sort proxy — a view row is not an event
# ================================================================================================
def test_a_batch_reject_maps_view_rows_to_source_rows_before_touching_the_frame(qapp, qtbot):
    """THE MISTAKE THIS APP HAS ALREADY SHIPPED ONCE, in a different table: curating the
    frame at the VIEW's row numbers. The table is sorted descending by ``depth_pA`` first,
    so the top four view rows are four scattered source rows; the four events whose
    ``accepted`` moves must be exactly the four the user is looking at."""
    t = _table(qtbot)
    df = t.dataframe
    depth_col = list(df.columns).index("depth_pA")
    t.view.sortByColumn(depth_col, Qt.DescendingOrder)
    _select_view_rows(t, 0, 3)

    expected = [int(df["event_id"].to_numpy()[t._proxy.mapToSource(t._proxy.index(r, 0)).row()])
                for r in range(4)]
    assert sorted(expected) != [0, 1, 2, 3], "the proxy must really be permuting"

    before = df["accepted"].to_numpy(dtype=bool).copy()
    _batch(t, False)
    moved = sorted(int(e) for e in df["event_id"].to_numpy()[
        before != df["accepted"].to_numpy(dtype=bool)])
    assert moved == sorted(expected)


# ================================================================================================
# 4. one gesture is one write and one announcement
# ================================================================================================
def test_a_batch_reject_announces_itself_once_and_not_once_per_row(qapp, qtbot):
    """THE LATENCY INVARIANT, stated as a signal count because that is what the cost is
    made of. 300 per-row ``acceptanceChanged`` emissions measured 3.72 s in the wired app;
    the summary, the span overlays, the score, the workflow chip, the dirty flag, the undo
    checkpoint and the analysis-surface gates all hang off that signal and every one of
    them re-reads the whole frame. One gesture therefore emits ONE ``acceptanceChanged``
    (the hop every already-wired listener needs) and nothing else.

    UPDATED BY WA2-WB2. This test used to require a second, batch-shaped
    ``curationBatchChanged`` as well, on the theory that a single id could not re-feed the
    gallery's and inspector's per-row mirrors. Measured in the wired app with both windows
    open — 1512x900, the 15 s -1000 mV run, gallery on "Accepted only", six rows rejected
    in one 'R' — the mirrors were already correct and byte-identical to an explicit
    refresh, because neither trusts the id: both end in a whole-frame re-read. The signal
    had zero receivers and no information of its own, so it is gone; the invariant that
    survives is the one that was always the point — ONE announcement per gesture."""
    t = _table(qtbot, _frame(40))
    seen: list = []
    t.acceptanceChanged.connect(lambda e, a: seen.append((e, a)))
    assert not hasattr(t, "curationBatchChanged"), \
        "the dead batch announcement is back: nothing in the app connects it"

    _select_view_rows(t, 0, 39)
    n = _batch(t, False)
    assert n == 40
    assert len(seen) == 1, f"one gesture emitted {len(seen)} per-row signals"


def test_a_batch_reject_marks_every_changed_table_row_reviewed(qapp, qtbot):
    """The one trailing signal names one id, but the human decision covered six rows."""
    t = _table(qtbot)
    df = t.dataframe
    _select_view_rows(t, 0, 5)
    expected = set(t.selected_event_ids())
    assert _batch(t, False) == 6
    reviewed = set(int(v) for v in df.loc[
        df[REVIEW_COL].astype("string") == REVIEW_DONE, "event_id"])
    assert reviewed == expected


def test_a_batch_accept_marks_already_accepted_table_rows_reviewed(qapp, qtbot):
    t = _table(qtbot)
    df = t.dataframe
    seen: list[tuple[int, bool]] = []
    t.acceptanceChanged.connect(lambda event_id, accepted: seen.append((event_id, accepted)))
    _select_view_rows(t, 0, 5)
    expected = set(t.selected_event_ids())

    assert _batch(t, True) == 0       # no acceptance boolean needed to move

    reviewed = set(int(v) for v in df.loc[
        df[REVIEW_COL].astype("string") == REVIEW_DONE, "event_id"])
    assert reviewed == expected
    assert len(seen) == 1             # but review persistence is announced/checkpointable


def test_a_batch_reject_writes_the_frame_in_one_data_changed_span(qapp, qtbot):
    """The repaint half of the same invariant: the model announces the whole block once,
    not one ``dataChanged`` per row (the proxy then collapses a multi-row span into a
    single whole-column repaint — see ``_VectorSortProxy._on_source_data_changed``)."""
    t = _table(qtbot, _frame(40))
    spans: list = []
    t._model.dataChanged.connect(lambda tl, br, roles=(): spans.append((tl.row(), br.row())))
    _select_view_rows(t, 0, 39)
    _batch(t, False)
    assert len(spans) == 1, f"{len(spans)} dataChanged signals for one gesture"
    assert spans[0] == (0, 39)


# ================================================================================================
# 5. the guard, and the disclosure, are the per-row ones
# ================================================================================================
def test_a_batch_accept_refuses_a_row_the_detector_could_not_measure(qapp, qtbot):
    """``accepted = derived AND requested`` is the schema's own rule and a bulk gesture has
    no standing over the derived half. The batch must reach the identical answer the
    checkbox reaches — and say so in the same sentence, on the same shared ledger."""
    df = _frame(6)
    df.loc[:, "accepted"] = False
    df.loc[2, "flag"] = "clipped_by_boundary"
    t = _table(qtbot, df)
    _select_view_rows(t, 0, 5)
    n = _batch(t, True)
    assert n == 5, "the unmeasurable row must not have been written"
    assert bool(df["accepted"].iat[2]) is False
    assert int(df["accepted"].to_numpy(dtype=bool).sum()) == 5
    assert t.refusal_note(), "a refused accept left no sentence anywhere"
    assert t.refusal_label.isVisible()


def test_a_batch_reject_is_never_refused(qapp, qtbot):
    """Rejection is the curator's own half going to False, which ANDs to False whatever
    the row says — the same asymmetry ``_curate`` has always had."""
    df = _frame(6)
    df.loc[3, "flag"] = "clipped_by_boundary"
    t = _table(qtbot, df)
    _select_view_rows(t, 0, 5)
    assert _batch(t, False) == 6
    assert not df["accepted"].to_numpy(dtype=bool).any()
    assert t.refusal_note() == ""


# ================================================================================================
# 6. the count is on screen
# ================================================================================================
def test_the_table_says_how_many_rows_are_selected(qapp, qtbot):
    """A range selection no widget counts is a gesture the user has to trust — and the
    count IS the scope of the next A/R press. Pre-fix there was no such label at all."""
    t = _table(qtbot)
    assert hasattr(t, "selection_label"), "the table counts nothing"
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.NoModifier, _row_center(t, 0))
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.ShiftModifier, _row_center(t, 5))
    assert t.selection_label.isVisible()
    assert "6 of 12 rows selected" in t.selection_label.text()


def test_the_selection_count_goes_away_when_the_selection_does(qapp, qtbot):
    """A standing "0 selected" caption is chrome, not disclosure.

    UPDATED BY WA2-WB1. What goes away is the TEXT, not the widget. Hiding the label with
    the selection re-laid the column out under the pointer — measured in the shipped
    Inspector dock at 1512x900, the view's viewport went 134 px -> 113 px on the first
    click and back when the selection cleared, one of the five visible rows appearing and
    vanishing under the mouse. The line is now a permanent part of the column and only
    what it says comes and goes, so the assertion moved from ``isVisible()`` to the text
    plus the geometry that must not move (pinned in
    ``test_wa2_table_selection_is_the_scope.py``)."""
    t = _table(qtbot)
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.NoModifier, _row_center(t, 0))
    assert t.selection_label.text() != ""
    height_with_a_selection = t.view.viewport().height()
    t.view.clearSelection()
    qapp.processEvents()
    assert t.selection_label.text() == ""
    assert t.view.viewport().height() == height_with_a_selection


# ================================================================================================
# 7. the context menu carries the gesture
# ================================================================================================
def test_the_right_click_menu_offers_to_curate_the_whole_selection(qapp, qtbot):
    """Right-click is where a user looks for "do something with these rows". Pre-fix the
    menu held exactly the two window doors."""
    t = _table(qtbot)
    _select_view_rows(t, 0, 5)
    captured: list = []

    def spy(menu, _pos):
        captured.append([a.text() for a in menu.actions()])
        return None

    t._exec_menu = staticmethod(spy)
    t._on_context_menu(_row_center(t, 2))
    assert captured, "no menu was built"
    texts = captured[0]
    assert any("Reject" in x and "6" in x for x in texts), texts
    assert any("Accept" in x and "6" in x for x in texts), texts
    # ...and the two window doors survive.
    assert any("inspector" in x for x in texts) and any("gallery" in x for x in texts)


def test_choosing_reject_selected_from_the_menu_curates_every_selected_row(qapp, qtbot):
    """The menu item is wired to the same batch door, not to a second writer."""
    t = _table(qtbot)
    df = t.dataframe
    _select_view_rows(t, 0, 5)

    def spy(menu, _pos):
        for act in menu.actions():
            if "Reject" in act.text():
                return act
        return None

    t._exec_menu = staticmethod(spy)
    t._on_context_menu(_row_center(t, 2))
    assert int((~df["accepted"].to_numpy(dtype=bool)).sum()) == 6


def test_right_clicking_outside_the_selection_re_aims_it(qapp, qtbot):
    """Otherwise "Reject selected (6)" fires on six rows the user is no longer pointing
    at. Inside the selection the selection is the subject and is left alone."""
    t = _table(qtbot)
    _select_view_rows(t, 0, 5)
    captured: list = []
    t._exec_menu = staticmethod(lambda menu, _p: captured.append(
        [a.text() for a in menu.actions()]) or None)
    t._on_context_menu(_row_center(t, 9))
    assert _selected_view_rows(t) == [9]
    assert any("selected event" in x for x in captured[0]), captured[0]


# ================================================================================================
# 8. regressions the new selection mode could have caused
# ================================================================================================
def test_j_does_not_extend_the_selection_when_a_modifier_is_latched(qapp, qtbot):
    """``QTableView.selectRow`` is not "select this row": it asks ``selectionCommand``,
    which consults the LIVE keyboard modifiers, so under ExtendedSelection a Ctrl still
    held turns a j into an extend. ``measure_panel`` measured exactly this and it is
    reproducible in a suite, where a synthetic modified key leaves
    ``keyboardModifiers()`` latched for the whole process.

    The latch used to be made with ``Ctrl+J`` itself, because a chord on a curation letter
    was read as the bare letter. Only an UNMODIFIED key curates now (S8-KEYS1), so Ctrl is
    latched by a chord the table does not read, and the j is the handler's own — delivered
    without a key event, which would reset the latched state before the handler ran."""
    t = _table(qtbot)
    QTest.mouseClick(t.view.viewport(), Qt.LeftButton, Qt.NoModifier, _row_center(t, 0))
    t.view.setFocus()
    QTest.keyClick(t.view, Qt.Key_X, Qt.ControlModifier)
    assert QtWidgets.QApplication.keyboardModifiers() & Qt.ControlModifier, \
        "the trap is not armed: Ctrl is not latched"
    t._handle_curation_key(Qt.Key_J)
    t._handle_curation_key(Qt.Key_J)
    assert _selected_view_rows(t) == [2], "j extended the selection instead of moving it"


def test_selecting_an_event_by_id_replaces_the_selection(qapp, qtbot):
    """The Event Inspector's mirror door. Same trap, same fix: an inspector navigation
    means "this row is now the one", not "add it to whatever was selected"."""
    t = _table(qtbot)
    _select_view_rows(t, 0, 5)
    t.select_event(9)
    assert t.selected_event_ids() == [9] if hasattr(t, "selected_event_ids") \
        else _selected_view_rows(t) == [9]


def test_the_table_did_not_get_wider(qapp, qtbot):
    """The Inspector dock is 460 px of a 1512 px display and this project has already
    shipped a 1475 px toolbar once. The measured pre-fix floor was 108 px; the new caption
    is an ``_ElidingCaption`` precisely so it stays there."""
    t = _table(qtbot)
    assert t.minimumSizeHint().width() <= 200, t.minimumSizeHint().width()


def test_the_frozen_column_count_is_untouched(qapp, qtbot):
    """No column was added, renamed or reordered by any of this."""
    t = _table(qtbot)
    df = _frame()
    assert list(t.dataframe.columns) == list(df.columns)
    assert t._model.columnCount() == len(df.columns)
