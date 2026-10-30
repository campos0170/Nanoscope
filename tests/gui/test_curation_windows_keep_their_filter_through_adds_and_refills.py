"""An add, or a re-feed that fills an emptied filter again, keeps each curation window's filter.

MEASURED DEFECTS (the six-event Schmitt rig of ``_inspector_hosting``; found by the cross-group
review of the sweep-9 fixes, after S9-14 made a Delete keep each window's filter).

1. AN ADD THE FILTER HIDES. ``MainWindow._land_on_added_event`` pointed both windows at a new
   hand-made event whatever they showed. With both in *Detected only* on detected event 2,
   Ctrl+M put the inspector on the hand-made event 7 ("filtered out of Detected only (6
   shown)"), so the next A or R ruled on it, and the gallery, which has no tile for it, cleared
   its cursor and selection ("Event 7 is filtered out of Detected only ... nothing is selected
   here"), so A, R and Space acted on nothing. An event cut off by the recording's end is stored
   rejected (``clipped_by_boundary``), so under *Accepted only* every such add did the same.
2. A FILTER EMPTIED, THEN FILLED AGAIN. ``EventInspectorWindow.set_events`` kept its place only
   while it stood on an event. After a Delete emptied *Hand-made only* ("No events match ..."),
   the next re-feed of any kind put it in *All* on the first event: the Undo of that Delete, a
   new Ctrl+M, the undo of a curation step, a reject in the Results table, even the same frame
   re-fed when the recording was selected again. The gallery kept the filter, but left the
   restored tile unselected, and when the Delete had emptied a Manual table it announced tile 0
   as a click.

THE RULE (the owner's, 2026-10-29). After an add, a window moves to the new event only when its
own list holds it (``curation_view.landing_after_add``): the inspector's filter, the gallery's
Show filter with its class or sample. Otherwise it stays where the re-feed that kept its place
left it: its filter, its event, its selection. A re-feed that keeps the place and fills an
emptied filter again keeps the filter and shows the filter's first event, each window in its
own order (``curation_view.landing_after_refill``), quietly; while the filter is still empty,
the window says so. A new detection still starts again in All.

These tests drive a real MainWindow: events added by hand with the Measure region and Ctrl+M,
the filters chosen in each window's own combo, a tile clicked to pick the event, the inspector's
Delete and Accepted controls, the A and R keys, Edit > Undo and Redo, the Results table's door
for a reject, and the recording selected again.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.curation_view import (
    FILTER_ACCEPTED, FILTER_ALL, FILTER_DETECTED, FILTER_HAND_MADE, FILTER_REJECTED)
from nanoscope.analysis.events import MANUAL_MODULE_ID

from ._inspector_hosting import (  # noqa: F401  (``private_settings`` is a fixture)
    FS, N, accepted, build_window, close_window, private_settings, settle)

#: Between the rig's detected events (5k, 15k, 28k, 43k, 55k, 66k), so the door takes them.
A, B = (35_000, 35_300), (72_000, 72_600)
#: An event still in progress when the recording stops: stored rejected, ``clipped_by_boundary``.
TAIL = (N - 300, N)


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
    assert len(new) == 1, (new, window.statusBar().currentMessage())
    return new.pop()


def _open_both(window):
    window._open_event_inspector()
    window._open_event_gallery()
    settle()
    return window._event_inspector, window._event_gallery


def _choose(window, mode, *, gallery=True, inspector=True) -> None:
    """``mode`` in the gallery's Show combo and the inspector's Filter combo (the gallery's
    first, as a person working in both would)."""
    insp, gal = window._event_inspector, window._event_gallery
    combos = ([gal.filter_box] if gallery else []) + ([insp._filter_combo] if inspector else [])
    for combo in combos:
        combo.setCurrentIndex(combo.findText(mode))
        settle()
        assert combo.currentText() == mode


def _on(window, event_id) -> None:
    """Click ``event_id``'s tile, as a person picks an event: the inspector follows it."""
    gal = window._event_gallery
    df = _frame(window)
    row = int(df.index[df["event_id"] == event_id][0])
    QTest.mouseClick(gal.view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                     gal.view.visualRect(gal._model.index_for_df_row(row)).center())
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


def _press_in_the_inspector(insp, key) -> None:
    """A key on the inspector's waveform, where its curation keys act."""
    insp.activateWindow()
    insp._plot_widget.setFocus(Qt.OtherFocusReason)
    settle(5)
    QTest.keyClick(insp._plot_widget, key)
    settle()


def _first_line(insp) -> str:
    return insp._readout.text().splitlines()[0]


def _assert_inspector_on(window, mode, event_id, where) -> None:
    insp = window._event_inspector
    assert insp.current_filter() == mode, insp.current_filter()
    assert insp.current_event_id() == event_id, insp.current_event_id()
    assert _first_line(insp) == f"Event {event_id}    [{where}]", _first_line(insp)


def _assert_gallery_on(window, mode, event_id) -> None:
    gal = window._event_gallery
    assert gal.filter_box.currentText() == mode
    assert gal.current_event_id() == event_id, gal.current_event_id()
    assert gal.selected_event_ids() == [event_id], gal.selected_event_ids()
    assert "filtered out" not in gal.status.text(), gal.status.text()


def _assert_both_empty(window, mode) -> None:
    insp, gal = window._event_inspector, window._event_gallery
    assert insp.current_filter() == mode and insp.current_event_id() is None
    assert insp._readout.text() == f"No events match “{mode}”."
    assert gal.filter_box.currentText() == mode
    assert gal.current_event_id() is None and gal.selected_event_ids() == []


# ================================================================================================
# 1. an add the window's own list does not hold leaves the window where it was
# ================================================================================================
def test_an_add_detected_only_hides_leaves_both_windows_on_their_event(private_settings, qtbot):
    window = build_window(qtbot)
    _add_by_hand(window, A)                       # a hand-made event: the source filters appear
    insp, gal = _open_both(window)
    _choose(window, FILTER_DETECTED)
    _on(window, 2)
    view, heard = _x_range(window), _announced(window)

    b = _add_by_hand(window, B)

    assert b in set(_frame(window)["event_id"]) and f"Added event {b}" in \
        window.statusBar().currentMessage()
    _assert_inspector_on(window, FILTER_DETECTED, 2, "3 of 6 · Detected only")   # was: on b
    _assert_gallery_on(window, FILTER_DETECTED, 2)               # was: nothing current or selected
    assert heard == [] and _x_range(window) == view
    # ...so the next decision is about the event the curator was on: R in the gallery rejects
    # event 2, and A in the inspector accepts it again; the hand-made event is never touched.
    gal.view.setFocus(Qt.OtherFocusReason)
    QTest.keyClick(gal.view, Qt.Key_R)
    settle()
    assert not accepted(window, 2) and accepted(window, b)
    _press_in_the_inspector(insp, Qt.Key_A)
    assert accepted(window, 2) and accepted(window, b)
    close_window(window)


def test_an_add_stored_rejected_leaves_accepted_only_windows_where_they_were(
        private_settings, qtbot):
    window = build_window(qtbot)
    _open_both(window)
    _choose(window, FILTER_ACCEPTED)
    _on(window, 2)
    heard = _announced(window)

    tail = _add_by_hand(window, TAIL)

    row = _frame(window).loc[_frame(window)["event_id"] == tail].iloc[0]
    assert not bool(row["accepted"]) and "clipped_by_boundary" in str(row["flag"])
    assert "rejected automatically (clipped_by_boundary)" in window.statusBar().currentMessage()
    _assert_inspector_on(window, FILTER_ACCEPTED, 2, "3 of 6 · Accepted only")   # was: on tail
    _assert_gallery_on(window, FILTER_ACCEPTED, 2)               # was: nothing current or selected
    assert heard == []
    close_window(window)


def test_an_add_moves_the_window_whose_list_holds_it_and_leaves_the_other(
        private_settings, qtbot):
    """Each window asks its own list: the gallery in All moves to the new event, the inspector
    in Detected only stays on detected event 2."""
    window = build_window(qtbot)
    _add_by_hand(window, A)
    insp, gal = _open_both(window)
    _choose(window, FILTER_DETECTED, gallery=False)
    _on(window, 2)
    view, heard = _x_range(window), _announced(window)

    b = _add_by_hand(window, B)

    _assert_gallery_on(window, FILTER_ALL, b)
    _assert_inspector_on(window, FILTER_DETECTED, 2, "3 of 6 · Detected only")   # was: on b
    assert heard == [] and _x_range(window) == view
    close_window(window)


# ================================================================================================
# 2. a re-feed that keeps the place fills an emptied filter: the filter stays, its first event
# ================================================================================================
def _emptied_by_a_delete(window, span=A):
    """One hand-made event, both windows in Hand-made only on it, and the inspector's Delete:
    the filter now holds nothing, and both windows say so."""
    a = _add_by_hand(window, span)
    insp, gal = _open_both(window)
    _choose(window, FILTER_HAND_MADE)
    _on(window, a)
    insp._delete_btn.click()
    settle()
    assert a not in set(_frame(window)["event_id"])
    return a


def test_the_undo_of_a_delete_that_emptied_the_filter_shows_the_event_again_in_it(
        private_settings, qtbot):
    window = build_window(qtbot)
    a = _emptied_by_a_delete(window)
    _assert_both_empty(window, FILTER_HAND_MADE)
    view, heard = _x_range(window), _announced(window)

    window._act_undo.trigger()
    settle()

    assert a in set(_frame(window)["event_id"])
    _assert_inspector_on(window, FILTER_HAND_MADE, a, "1 of 1 · Hand-made only")  # was: All, 0
    _assert_gallery_on(window, FILTER_HAND_MADE, a)                     # was: the tile unselected
    assert not window._event_inspector._delete_btn.isHidden()
    assert heard == [] and _x_range(window) == view
    # ...and the Redo empties it again, the filter still kept.
    window._act_redo.trigger()
    settle()
    assert a not in set(_frame(window)["event_id"])
    _assert_both_empty(window, FILTER_HAND_MADE)
    assert heard == []
    close_window(window)


def test_an_add_into_an_emptied_filter_keeps_the_filter(private_settings, qtbot):
    window = build_window(qtbot)
    _emptied_by_a_delete(window)
    heard = _announced(window)

    b = _add_by_hand(window, B)

    _assert_inspector_on(window, FILTER_HAND_MADE, b, "1 of 1 · Hand-made only")  # was: All, b
    _assert_gallery_on(window, FILTER_HAND_MADE, b)
    assert heard == []
    close_window(window)


def test_the_undo_of_a_curation_step_fills_an_empty_rejected_only_in_both_windows(
        private_settings, qtbot):
    """Event 2 rejected, then accepted again, with the inspector's Accepted box: Rejected only
    holds nothing. Undoing the accept puts event 2 back in it, and each window shows it there,
    where the inspector used to go back to All on event 0 and the gallery selected nothing."""
    window = build_window(qtbot)
    insp, gal = _open_both(window)
    _on(window, 2)
    insp._accept_check.click()
    settle()
    assert not accepted(window, 2)
    insp._accept_check.click()
    settle()
    assert accepted(window, 2)
    _choose(window, FILTER_REJECTED)
    _assert_both_empty(window, FILTER_REJECTED)
    view, heard = _x_range(window), _announced(window)

    window._act_undo.trigger()
    settle()

    assert not accepted(window, 2)
    _assert_inspector_on(window, FILTER_REJECTED, 2, "1 of 1 · Rejected only")
    _assert_gallery_on(window, FILTER_REJECTED, 2)
    assert heard == [] and _x_range(window) == view
    close_window(window)


def test_a_reject_in_the_results_table_fills_the_inspectors_empty_filter(private_settings, qtbot):
    """The table's reject re-feeds the inspector the same frame: the empty Rejected only stays,
    and shows the event that now fills it (it went back to All on event 0)."""
    window = build_window(qtbot)
    insp, _gal = _open_both(window)
    _choose(window, FILTER_REJECTED, gallery=False)
    assert insp.current_event_id() is None
    heard = _announced(window)

    window._analysis_panel.set_event_accepted(3, False)
    settle()

    assert not accepted(window, 3)
    _assert_inspector_on(window, FILTER_REJECTED, 3, "1 of 1 · Rejected only")
    assert heard == []
    close_window(window)


def test_selecting_the_recording_again_keeps_an_empty_filter(private_settings, qtbot):
    """Every selection re-feeds the inspector the same frame; an empty filter used to fall back
    to All on the first event there."""
    window = build_window(qtbot)
    insp, _gal = _open_both(window)
    _choose(window, FILTER_REJECTED, gallery=False)
    doc_id, channel = window._current_selection

    window._on_selection(doc_id, channel)
    settle()

    assert insp.current_filter() == FILTER_REJECTED, insp.current_filter()       # was: All
    assert insp.current_event_id() is None                                       # was: 0
    assert insp._readout.text() == f"No events match “{FILTER_REJECTED}”."
    close_window(window)


def test_the_undo_of_a_delete_that_emptied_a_manual_table_lands_both_windows_quietly(
        private_settings, qtbot):
    """A Manual table whose only event is deleted holds no event at all, which the gallery keeps
    as no frame; the Undo used to announce tile 0 as a click and put the inspector in All."""
    window = build_window(qtbot)
    selector = window._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData(MANUAL_MODULE_ID))
    settle()
    a = _emptied_by_a_delete(window)
    assert window._analysis_panel.current_detection().module_id == MANUAL_MODULE_ID
    assert len(_frame(window)) == 0
    heard = _announced(window)

    window._act_undo.trigger()
    settle()

    _assert_inspector_on(window, FILTER_HAND_MADE, a, "1 of 1 · Hand-made only")  # was: All
    _assert_gallery_on(window, FILTER_HAND_MADE, a)
    assert heard == []                                                       # was: [a], announced
    close_window(window)
