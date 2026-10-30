"""S9-14 — Delete of a hand-made event keeps the curation windows' filter and place.

MEASURED DEFECT (sweep 9, the six-event Schmitt rig of ``_inspector_hosting`` with two events
added by hand between the detected ones). A curator works through the new *Hand-made only* filter
in the Event inspector, on the first hand-made event (id 6), and presses Delete. The inspector came
back in *All*, on detected event 3 -- the next row in TABLE order -- not on the remaining hand-made
event, so the filter had to be chosen again and the place found again.
``MainWindow._delete_hand_made_events`` asked the re-feed to keep the windows' place, but
``curation_view.can_keep_place`` refused a frame that is the old one less some rows unless "the
event the window stands on survived", which the inspector's own Delete never satisfies: the event
on screen is the one deleted. ``set_events`` then treated the frame as a new detection (filter back
to All), and MainWindow put both windows on the next survivor in table order, whatever their
filters held. The gallery kept its Show filter but was pointed at that detected event, which
*Hand-made only* excludes, so it selected nothing and said "Event 3 is filtered out of Hand-made
only"; with the filter emptied by the Delete, it said so of an event nobody asked for.

THE FIX. A Delete is the same table less some rows, not a new detection. ``can_keep_place`` keeps
the place for a frame that is the old one less some events, whatever the window stood on, and a
window whose event went with them lands on the next event of ITS OWN list after it that is still
there, else the one before (``curation_view.landing_after_removal``, one rule for both windows):
the inspector's filter runs in table order, the gallery's tiles in the order they are drawn (a sort
by dwell is not time). The landing is quiet, as an add's is. When the filter now holds nothing, the
inspector shows its empty filter note ("No events match ...") and the gallery an empty page, and
neither names an event outside the filter. MainWindow no longer picks the landing; a window that
was not on a deleted event keeps the event it was on. The undo of an add is the same table less a
row, and gets the same rule.

These tests drive a real MainWindow: events added by hand with the Measure region and Ctrl+M, the
filters chosen in each window's own combo, a tile clicked to pick the event, and the Delete made
from each of its three doors -- the inspector's Delete button, the gallery tile's right-click menu
and the Results table's right-click menu, for one row and for two at once (the menus' one blocking
``exec`` answered through their seams) -- and an add taken back with Edit > Undo.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore
from PySide6.QtTest import QTest

from nanoscope.analysis.curation_view import FILTER_HAND_MADE
from nanoscope.gui.event_gallery import EventGalleryWindow
from nanoscope.gui.panels import EventTable

from ._inspector_hosting import FS, build_window, close_window, private_settings, settle  # noqa: F401

#: Between the rig's detected events (5k, 15k, 28k, 43k, 55k, 66k), so the door takes them; three
#: lengths, so a sort by dwell (longest first) draws them B, A, C while time order is A, B, C.
A, B, C = (35_000, 35_300), (72_000, 72_600), (76_000, 76_150)

_EMPTY_NOTE = f"No events match “{FILTER_HAND_MADE}”."


def _frame(window):
    return window._analysis_panel.current_detection().payload.df


def _add_by_hand(window, span) -> int:
    """Bracket ``span`` with the Measure region and press Ctrl+M; the new event's id."""
    window._analysis_panel.measure_panel.set_active(True)
    window._trace_view.set_measure_region(span[0] / FS, span[1] / FS)
    settle()
    before = set(int(i) for i in _frame(window)["event_id"])
    window._act_add_event.trigger()
    settle()
    new = set(int(i) for i in _frame(window)["event_id"]) - before
    assert len(new) == 1, new
    return new.pop()


def _curate_hand_made(window, *spans):
    """The events drawn, both windows open and each switched to Hand-made only in its own combo
    (the gallery's first: the inspector's switch then puts both on the first hand-made event)."""
    ids = [_add_by_hand(window, span) for span in spans]
    window._open_event_inspector()
    window._open_event_gallery()
    settle()
    insp, gal = window._event_inspector, window._event_gallery
    for combo in (gal.filter_box, insp._filter_combo):
        combo.setCurrentIndex(combo.findText(FILTER_HAND_MADE))
        settle()
    assert insp.current_filter() == FILTER_HAND_MADE
    assert gal.filter_box.currentText() == FILTER_HAND_MADE
    assert insp.current_event_id() == gal.current_event_id() == ids[0]
    return insp, gal, ids


def _tile_centre(window, event_id) -> QtCore.QPoint:
    gal = window._event_gallery
    row = int(_frame(window).index[_frame(window)["event_id"] == event_id][0])
    return gal.view.visualRect(gal._model.index_for_df_row(row)).center()


def _on(window, event_id) -> None:
    """Click ``event_id``'s tile, as a person picks an event: the inspector follows it."""
    gal = window._event_gallery
    QTest.mouseClick(gal.view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                     _tile_centre(window, event_id))
    settle()
    assert gal.current_event_id() == event_id and gal.selected_event_ids() == [event_id]
    assert window._event_inspector.current_event_id() == event_id


def _announced(window) -> list[int]:
    heard: list[int] = []
    window._event_inspector.eventSelected.connect(heard.append)
    window._event_gallery.eventSelected.connect(heard.append)
    return heard


def _x_range(window):
    return [round(float(v), 6) for v in window._trace_view.anchor_plot().vb.viewRange()[0]]


def _assert_both_on(window, event_id) -> None:
    insp, gal = window._event_inspector, window._event_gallery
    assert insp.current_filter() == FILTER_HAND_MADE, insp.current_filter()       # was: All
    assert insp.current_event_id() == event_id, insp.current_event_id()            # was: 3, detected
    assert not insp._delete_btn.isHidden()                 # a hand-made event: Delete stays on offer
    assert gal.filter_box.currentText() == FILTER_HAND_MADE
    assert gal.current_event_id() == event_id, gal.current_event_id()              # was: none
    assert gal.selected_event_ids() == [event_id]
    assert "filtered out" not in gal.status.text()


# ================================================================================================
# the inspector's Delete button
# ================================================================================================
def test_delete_in_the_inspector_keeps_hand_made_only_and_lands_on_the_next_hand_made_event(
        private_settings, qtbot):
    window = build_window(qtbot)
    insp, gal, (a, b) = _curate_hand_made(window, A, B)
    _on(window, a)
    view, heard = _x_range(window), _announced(window)

    insp._delete_btn.click()
    settle()

    assert a not in set(_frame(window)["event_id"])
    _assert_both_on(window, b)
    assert heard == [] and _x_range(window) == view          # quietly: the trace stays put
    # ...and the Undo puts the event back without moving either window off its place.
    window._act_undo.trigger()
    settle()
    assert a in set(_frame(window)["event_id"])
    _assert_both_on(window, b)
    close_window(window)


def test_delete_of_the_last_hand_made_event_lands_on_the_one_before(private_settings, qtbot):
    window = build_window(qtbot)
    insp, gal, (a, b) = _curate_hand_made(window, A, B)
    _on(window, b)

    insp._delete_btn.click()
    settle()

    assert b not in set(_frame(window)["event_id"])
    _assert_both_on(window, a)
    close_window(window)


def test_a_delete_that_empties_the_filter_says_so_and_names_no_other_event(private_settings, qtbot):
    window = build_window(qtbot)
    insp, gal, (a,) = _curate_hand_made(window, A)
    _on(window, a)

    insp._delete_btn.click()
    settle()

    assert a not in set(_frame(window)["event_id"]) and len(_frame(window)) == 6
    assert insp.current_filter() == FILTER_HAND_MADE                               # was: All
    assert insp.current_event_id() is None
    assert insp._readout.text() == _EMPTY_NOTE                    # the inspector's empty-filter note
    assert insp._delete_btn.isHidden()
    assert gal.filter_box.currentText() == FILTER_HAND_MADE
    assert gal.current_event_id() is None and gal.selected_event_ids() == []
    status = gal.status.text()
    assert status.startswith("0 of 0 (hand-made only)"), status
    assert "filtered out" not in status, status          # was: "Event 3 is filtered out of ..."
    close_window(window)


def test_the_undo_of_an_add_is_the_same_table_less_a_row_too(private_settings, qtbot):
    """The same rule, reached by Edit > Undo: Ctrl+M lands both windows on the new event, and
    undoing the add takes away the event they stand on."""
    window = build_window(qtbot)
    insp, gal, (a, b) = _curate_hand_made(window, A, B)
    c = _add_by_hand(window, C)
    assert insp.current_event_id() == gal.current_event_id() == c
    heard = _announced(window)

    window._act_undo.trigger()
    settle()

    assert c not in set(_frame(window)["event_id"])
    _assert_both_on(window, b)                     # C was the last in both lists: the one before
    assert heard == []
    close_window(window)


# ================================================================================================
# the gallery tile's menu, and a gallery sorted by dwell
# ================================================================================================
def _delete_from_the_tile_menu(window, monkeypatch, event_id) -> None:
    """Right-click ``event_id``'s tile (which makes it current, as a person's right-click does)
    and choose Delete hand-made event from its menu."""
    gal = window._event_gallery
    pos = _tile_centre(window, event_id)
    QTest.mouseClick(gal.view.viewport(), QtCore.Qt.RightButton, QtCore.Qt.NoModifier, pos)
    settle()
    assert gal.current_event_id() == event_id

    def choose_delete(menu, _pos):
        return next(a for a in menu.actions() if a.text() == "Delete hand-made event")

    monkeypatch.setattr(EventGalleryWindow, "_exec_menu", staticmethod(choose_delete))
    gal._on_context_menu(pos)
    settle()
    assert event_id not in set(_frame(window)["event_id"])


def test_delete_from_the_gallery_tile_menu_keeps_both_filters_and_lands_on_the_next(
        private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    insp, gal, (a, b) = _curate_hand_made(window, A, B)
    _on(window, a)

    _delete_from_the_tile_menu(window, monkeypatch, a)

    _assert_both_on(window, b)
    close_window(window)


def test_a_gallery_sorted_by_dwell_lands_on_the_tile_that_took_the_deleted_ones_place(
        private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    insp, gal, (a, b, c) = _curate_hand_made(window, A, B, C)
    gal.set_sort("Dwell (longest first)")
    settle()
    drawn = [int(_frame(window)["event_id"].iloc[r]) for r in gal._model.df_rows()]
    assert drawn == [b, a, c]                                  # the grid's order is not time order
    _on(window, a)

    _delete_from_the_tile_menu(window, monkeypatch, a)

    assert gal.filter_box.currentText() == FILTER_HAND_MADE
    assert gal.current_event_id() == c and gal.selected_event_ids() == [c]   # next in the grid
    assert insp.current_filter() == FILTER_HAND_MADE
    assert insp.current_event_id() == b                                      # next in time
    close_window(window)


# ================================================================================================
# the Results table's menu
# ================================================================================================
def test_delete_from_the_results_table_menu_keeps_both_filters_and_lands_on_the_next(
        private_settings, qtbot, monkeypatch):
    window = build_window(qtbot)
    insp, gal, (a, b) = _curate_hand_made(window, A, B)
    table = window._analysis_panel._table
    row = int(_frame(window).index[_frame(window)["event_id"] == a][0])
    proxy = table._proxy.mapFromSource(table._model.index(row, 0))
    table.view.selectRow(proxy.row())                    # the click the right-click makes
    settle()
    assert insp.current_event_id() == a and gal.current_event_id() == a

    def choose_delete(menu, _pos):
        return next(act for act in menu.actions() if "Delete" in act.text())

    monkeypatch.setattr(EventTable, "_exec_menu", staticmethod(choose_delete))
    table._on_context_menu(table.view.visualRect(proxy).center())
    settle()

    assert a not in set(_frame(window)["event_id"])
    _assert_both_on(window, b)
    close_window(window)


def test_a_delete_of_two_rows_lands_both_windows_past_every_row_that_went(
        private_settings, qtbot, monkeypatch):
    """The table's menu deletes its whole selection: the windows stood on the first of the two,
    and the next event of their list after it went too, so they land on the one after that."""
    window = build_window(qtbot)
    insp, gal, (a, b, c) = _curate_hand_made(window, A, B, C)
    table = window._analysis_panel._table
    first, second = (
        table._proxy.mapFromSource(table._model.index(
            int(_frame(window).index[_frame(window)["event_id"] == eid][0]), 0))
        for eid in (a, b))
    table.view.selectRow(first.row())                    # a click on A's row...
    table.view.selectionModel().select(                  # ...and the Ctrl-click that adds B's
        second, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    settle()
    assert insp.current_event_id() == a and gal.current_event_id() == a

    def choose_delete(menu, _pos):
        return next(act for act in menu.actions()
                    if act.text().replace("&", "") == "Delete 2 hand-made events")

    monkeypatch.setattr(EventTable, "_exec_menu", staticmethod(choose_delete))
    table._on_context_menu(table.view.visualRect(first).center())
    settle()

    assert not {a, b} & set(_frame(window)["event_id"])
    _assert_both_on(window, c)                   # was: All, on the detected event after A
    close_window(window)
