"""Change B — curation throughput in the Event gallery.

Advance after decision moves a ONE-tile selection to the next tile of the Show subset
(turning pages); the Unreviewed / Unlabelled / Labelled filters; ⌘U; the progress line
in the status bar; keep-place across an undo re-feed. The footer status line and the
Fit page are pinned unchanged.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import FLAG_CLIPPED, EventSet, mark_reviewed, reviewed_mask
from nanoscope.analysis.labels import LabelVocabulary
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_gallery import EventGalleryWindow

FS = 100_000.0
N_EVENTS = 14
_SETTINGS_KEY = "curation/advance_after_decision"


@pytest.fixture(autouse=True)
def _advance_setting_is_clean():
    QtCore.QSettings().remove(_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove(_SETTINGS_KEY)


def _session(tmp_path, n_events: int = N_EVENTS):
    n = 20_000 * (n_events + 1)
    rng = np.random.default_rng(1)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for k in range(n_events):
        s = 10_000 + 20_000 * k
        x[s:s + 200 + 40 * ((k * 5) % n_events)] -= 30.0      # distinct dwells
    arr = np.ascontiguousarray(x, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "g.dat", format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    df, prov = detect_events(arr, FS, min_dwell_s=5e-4)
    assert len(df) == n_events
    return session, doc_id, EventSet(df, prov)


def _vocab(*tokens: str) -> LabelVocabulary:
    v = LabelVocabulary()
    for t in tokens:
        k, _, val = t.partition("=")
        v.add(k, val)
    return v


@pytest.fixture
def gal(qapp, qtbot, tmp_path):
    session, doc_id, events = _session(tmp_path)
    g = EventGalleryWindow(session)
    qtbot.addWidget(g)
    g.set_events(doc_id, 0, events.df)
    g._doc_for_test = doc_id
    return g


def _click(g, page_row: int) -> None:
    """A plain click on a tile: current + the only selected one."""
    g.view.selectionModel().setCurrentIndex(
        g._model.index(page_row, 0), QItemSelectionModel.ClearAndSelect)


def _ids(g) -> list[int]:
    return [int(g._df["event_id"].iloc[r]) for r in g._model.df_rows()]


def _key(g, key, text="", mods=Qt.NoModifier, *, repeat=False) -> bool:
    return g._handle_own_key(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, mods, text, repeat))


# ---------------------------------------------------------------------------------------------
def test_advance_off_a_single_tile_r_keeps_the_selection(gal):
    ids = _ids(gal)
    _click(gal, 2)
    assert gal.advance_after_decision() is False
    assert gal._advance_check.objectName() == "advanceAfterDecision"
    assert _key(gal, Qt.Key_R, "r")
    assert gal.selected_event_ids() == [ids[2]] and gal.current_event_id() == ids[2]


def test_advance_on_a_single_tile_decision_selects_the_next_tile_and_announces_it(gal):
    ids = _ids(gal)
    _click(gal, 2)
    gal.set_advance_after_decision(True)
    seen: list[int] = []
    gal.eventSelected.connect(seen.append)
    assert _key(gal, Qt.Key_R, "r")
    assert gal.selected_event_ids() == [ids[3]] and gal.current_event_id() == ids[3]
    assert seen == [ids[3]]
    assert not bool(gal._df["accepted"].iloc[2])
    assert gal.decide_toggle_selected() is True and gal.selected_event_ids() == [ids[4]]


def test_advance_turns_the_page_from_the_last_tile(gal):
    gal._model.set_page_size(5)
    gal._sync_controls()
    ids = _ids(gal)
    gal.set_page(0)
    _click(gal, 4)
    gal.set_advance_after_decision(True)
    assert gal.decide_selected(False) is True
    assert gal.page() == 1
    assert gal.selected_event_ids() == [ids[5]] and gal.current_event_id() == ids[5]


def test_a_multi_tile_selection_never_advances_even_when_on(gal):
    ids = _ids(gal)
    sel = QItemSelection(gal._model.index(1, 0), gal._model.index(4, 0))
    gal.view.selectionModel().select(sel, QItemSelectionModel.ClearAndSelect)
    gal.view.selectionModel().setCurrentIndex(gal._model.index(1, 0),
                                              QItemSelectionModel.NoUpdate)
    gal.set_advance_after_decision(True)
    assert gal.decide_selected(False) is False
    assert sorted(gal.selected_event_ids()) == ids[1:5]
    assert gal.current_event_id() == ids[1]


def test_advance_at_the_last_tile_stays_and_says_so(gal):
    ids = _ids(gal)
    _click(gal, len(ids) - 1)
    gal.set_advance_after_decision(True)
    assert gal.decide_selected(False) is False
    assert gal.current_event_id() == ids[-1]
    assert gal.progress_text().startswith("End of All")
    # Another surface's write retires the note: the counts it quoted may be stale.
    gal.reflect_acceptance(ids[0], False)
    assert gal.progress_text().startswith("Reviewed")


def test_a_refused_accept_does_not_advance(gal):
    df = gal._df
    df.iat[3, df.columns.get_loc("flag")] = FLAG_CLIPPED
    df.iat[3, df.columns.get_loc("accepted")] = False
    gal.refresh()
    ids = _ids(gal)
    _click(gal, 3)
    gal.set_advance_after_decision(True)
    assert gal.decide_selected(True) is False
    assert gal.current_event_id() == ids[3]


def test_under_unreviewed_the_decided_tile_leaves_and_the_next_is_selected(gal):
    ids = _ids(gal)
    gal.set_filter("Unreviewed")
    assert gal.event_count() == N_EVENTS
    _click(gal, 2)
    gal.set_advance_after_decision(True)
    # Accepting an accepted tile changes no ``accepted`` cell: only the review stamp moves,
    # and that alone must take it out of Unreviewed in a standalone window.
    assert gal.decide_selected(True) is True
    assert gal.event_count() == N_EVENTS - 1
    assert ids[2] not in _ids(gal)
    assert gal.selected_event_ids() == [ids[3]]


def test_labelling_under_unlabelled_refilters_and_keeps_the_rest_selected(gal):
    gal.set_vocabulary(_vocab("type=collision"))
    gal.set_filter("Unlabelled")
    ids = _ids(gal)
    sel = QItemSelection(gal._model.index(0, 0), gal._model.index(2, 0))
    gal.view.selectionModel().select(sel, QItemSelectionModel.ClearAndSelect)
    gal.apply_label("type", "collision")
    assert not set(ids[:3]) & set(_ids(gal))
    assert "filtered out of Unlabelled" in gal.status.text()
    # Under All nothing is reset: the labelled tiles stay selected.
    gal.set_filter("All")
    sel = QItemSelection(gal._model.index(5, 0), gal._model.index(6, 0))
    gal.view.selectionModel().select(sel, QItemSelectionModel.ClearAndSelect)
    before = gal.selected_event_ids()
    gal.apply_label("type", "collision")
    assert gal.selected_event_ids() == before


def test_label_filters_appear_with_the_palette_and_the_combo_keeps_its_width(gal):
    w0 = gal.filter_box.sizeHint().width()
    assert gal.filter_options() == ("All", "Accepted only", "Rejected only", "Unreviewed")
    gal.set_vocabulary(_vocab("type=collision", "molecule=a-very-long-value-indeed"))
    assert gal.filter_options()[-3:] == (
        "Unlabelled", "Labelled: type=collision", "Labelled: molecule=a-very-long-value-indeed")
    assert gal.filter_box.sizeHint().width() == w0
    gal.set_filter("Labelled: type=collision")
    assert gal.event_count() == 0 and gal.filter_box.currentText() == "Labelled: type=collision"


def test_the_status_line_keeps_a_label_filters_case(gal):
    gal.set_vocabulary(_vocab("molecule=ApoF"))
    gal.set_filter("Labelled: molecule=ApoF")
    assert "(labelled: molecule=ApoF)" in gal.status.text()
    gal.set_filter("Accepted only")
    assert "(accepted only)" in gal.status.text()


def test_ctrl_u_focuses_the_first_unreviewed_tile_in_sort_order(gal):
    gal.set_sort("Dwell (longest first)")
    order = _ids(gal)
    mark_reviewed(gal._df, rows=np.array([gal._model.df_rows()[0], gal._model.df_rows()[1]]))
    _click(gal, 5)
    assert _key(gal, Qt.Key_U, "", Qt.ControlModifier)
    assert gal.current_event_id() == order[2] and gal.selected_event_ids() == [order[2]]
    mark_reviewed(gal._df)
    assert gal.first_unreviewed() is None
    assert gal.progress_text().startswith("Every event reviewed")


def test_a_latched_modifier_cannot_turn_an_advance_into_an_extension(qtbot, gal):
    ids = _ids(gal)
    gal.show()
    qtbot.wait(20)
    QTest.keyClick(gal.view, Qt.Key_A, Qt.ControlModifier)    # leaves Ctrl latched in suites
    _click(gal, 1)
    gal.set_advance_after_decision(True)
    gal.decide_selected(False)
    assert gal.selected_event_ids() == [ids[2]]


def test_an_auto_repeated_decision_key_is_consumed_and_ignored(gal):
    ids = _ids(gal)
    _click(gal, 1)
    gal.set_advance_after_decision(True)
    assert _key(gal, Qt.Key_R, "r", repeat=True) is True
    assert bool(gal._df["accepted"].iloc[1]) and gal.current_event_id() == ids[1]


def test_the_gallery_footprint_is_unchanged(qapp, qtbot, tmp_path):
    old = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        session, doc_id, events = _session(tmp_path)
        g = EventGalleryWindow(session)
        qtbot.addWidget(g)
        g.set_events(doc_id, 0, events.df)
        g.resize(940, 700)
        g.show()
        qtbot.wait(50)
        hint = g.minimumSizeHint()
        assert hint.width() <= 862 and hint.height() <= 315
        status_h, page = g.status.height(), g._model.page_size()
        line = g.status.text()
        g.set_advance_after_decision(True)
        _click(g, g._model.rowCount() - 1)
        g.set_page(g.page_count() - 1)
        _click(g, g._model.rowCount() - 1)
        g.decide_selected(False)                     # the end-of-subset note arrives
        qtbot.wait(20)
        assert g.progress_text().startswith("End of All")
        assert (g.status.height(), g._model.page_size()) == (status_h, page)
        assert g.minimumSizeHint() == hint
        # The footer line gained no clause of its own.
        assert "End of" not in g.status.text() and "Reviewed" not in g.status.text()
        assert line.split("   ·   ")[0].endswith("(all)")
        assert g._curation_status.objectName() == "curationStatus"
        assert g._progress.objectName() == "curationProgress"
    finally:
        qapp.setFont(old)


def test_an_undo_refeed_restores_selection_and_cursor_without_emitting(gal):
    ids = _ids(gal)
    gal.set_page(0)
    sel = QItemSelection(gal._model.index(3, 0), gal._model.index(5, 0))
    gal.view.selectionModel().select(sel, QItemSelectionModel.ClearAndSelect)
    gal.view.selectionModel().setCurrentIndex(gal._model.index(4, 0),
                                              QItemSelectionModel.NoUpdate)
    seen: list[int] = []
    gal.eventSelected.connect(seen.append)
    gal.set_events(gal._doc_for_test, 0, gal._df.copy(deep=True), keep_place=True)
    assert sorted(gal.selected_event_ids()) == ids[3:6]
    assert gal.current_event_id() == ids[4] and seen == []
    # A new detection (no hint) gets today's reset: tile 0, announced.
    gal.set_events(gal._doc_for_test, 0, gal._df.copy(deep=True))
    assert gal.current_event_id() == ids[0] and seen == [ids[0]]


def test_keep_place_refuses_renumbered_events(gal):
    _click(gal, 6)
    merged = gal._df.drop(index=gal._df.index[2]).reset_index(drop=True)
    merged["event_id"] = np.arange(len(merged))
    gal.set_events(gal._doc_for_test, 0, merged, keep_place=True)
    assert gal.current_event_id() == 0


def test_the_gallery_undo_returns_to_the_tile_the_decision_advanced_from(gal):
    ids = _ids(gal)
    snapshot = gal._df.copy(deep=True)
    gal.undoRequested.connect(lambda: gal.set_events(gal._doc_for_test, 0,
                                                     snapshot.copy(deep=True), keep_place=True))
    gal.set_undo_state(True, "Undo Curate event")
    _click(gal, 2)
    gal.set_advance_after_decision(True)
    gal.decide_selected(False)
    assert gal.current_event_id() == ids[3]
    gal.undo_btn.click()
    assert gal.current_event_id() == ids[2] and gal.selected_event_ids() == [ids[2]]
    assert bool(gal._df["accepted"].iloc[2])


def test_filter_item_rebuilds_do_not_refilter_spuriously(gal):
    resets: list[int] = []
    gal._model.modelReset.connect(lambda: resets.append(1))
    gal.set_vocabulary(_vocab("type=a", "type=b"))
    assert len(resets) <= 1
    assert gal.filter_box.currentText() == "All"


def test_the_progress_line_follows_writes_and_names_the_unknown_reviews(gal):
    assert gal.progress_text() == f"Reviewed 0 / {N_EVENTS} · accepted {N_EVENTS}"
    assert f"{N_EVENTS} of them have no review record" in gal._progress.toolTip()
    _click(gal, 0)
    gal.decide_selected(False)
    assert gal.progress_text() == f"Reviewed 1 / {N_EVENTS} · accepted {N_EVENTS - 1}"
    assert bool(reviewed_mask(gal._df)[0])


def test_undo_after_a_no_op_decision_stays_where_it_is(gal):
    """R on a tile already rejected writes nothing; the Undo that follows reverts ANOTHER
    tile's decision, so the window must not jump back to the no-op tile."""
    ids = _ids(gal)
    gal.set_undo_state(True, "Undo Curate event")
    gal.set_advance_after_decision(True)
    _click(gal, 2)
    gal.decide_selected(False)                           # a real write on 2
    _click(gal, 6)
    before_6 = gal._df.copy(deep=True)
    gal.decide_selected(False)                           # a real write on 6: the last step
    _click(gal, 2)
    assert gal.decide_selected(False) is True            # no-op on 2, still advances
    assert gal.current_event_id() == ids[3]
    gal.undoRequested.connect(lambda: gal.set_events(gal._doc_for_test, 0,
                                                     before_6.copy(deep=True), keep_place=True))
    gal.undo_btn.click()
    assert bool(gal._df["accepted"].iloc[6]) and not bool(gal._df["accepted"].iloc[2])
    assert gal.current_event_id() == ids[3] and gal.selected_event_ids() == [ids[3]]


def test_a_refused_accept_under_unreviewed_moves_on_instead_of_to_nothing(gal):
    """The refusal still stamps the tile reviewed, which takes it out of Unreviewed; with
    nothing current the next key would write nothing. The cursor goes on to the next tile."""
    df = gal._df
    df.iat[4, df.columns.get_loc("flag")] = FLAG_CLIPPED
    df.iat[4, df.columns.get_loc("accepted")] = False
    gal.refresh()
    gal.set_filter("Unreviewed")
    ids = _ids(gal)
    _click(gal, 4)
    gal.set_advance_after_decision(True)
    assert gal.decide_selected(True) is False            # refused: not an advance...
    assert not bool(gal._df["accepted"].iloc[4])
    assert gal.current_event_id() == ids[5] and gal.selected_event_ids() == [ids[5]]
    assert "kept rejected" in gal.status.text()
    gal.decide_selected(False)                           # ...and the next key lands
    assert not bool(gal._df["accepted"].iloc[5])
    # With Advance off a refusal selects nothing new (as every decision under Unreviewed).
    gal.set_advance_after_decision(False)
    df.iat[7, df.columns.get_loc("flag")] = FLAG_CLIPPED
    df.iat[7, df.columns.get_loc("accepted")] = False
    gal.refresh()
    _click(gal, _ids(gal).index(ids[7]))
    gal.decide_selected(True)
    # Updated (review B-NAV-4): the cursor waits on the next tile, unselected, instead of
    # nowhere (which sent the next → to the page's first tile).
    assert gal.selected_event_ids() == [] and gal.current_event_id() == ids[8]


def test_a_mirrored_move_with_a_modifier_down_replaces_the_selection(qtbot, gal):
    """The inspector's ⌘Z / ⌘U reach show_event with ⌘ still held; the mirror must select
    the one tile, never toggle it into the selection."""
    ids = _ids(gal)
    gal.show()
    qtbot.wait(20)
    _click(gal, 1)
    QTest.keyPress(gal.view, Qt.Key_Control, Qt.ControlModifier)
    try:
        assert QtGui.QGuiApplication.keyboardModifiers() & Qt.ControlModifier
        gal.show_event(ids[4])
        assert gal.selected_event_ids() == [ids[4]] and gal.current_event_id() == ids[4]
        gal.set_advance_after_decision(True)
        gal.decide_selected(False)
        assert gal.selected_event_ids() == [ids[5]]
    finally:
        QTest.keyRelease(gal.view, Qt.Key_Control)
    # A Shift-click now extends from the tile the cursor was MOVED to, not from tile 1.
    rect = gal.view.visualRect(gal._model.index(7, 0))
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, Qt.ShiftModifier, rect.center())
    assert sorted(gal.selected_event_ids()) == ids[5:8]


def test_the_footer_and_pager_hold_still_while_unreviewed_shrinks(qtbot, gal):
    gal.resize(940, 700)
    gal.show()
    qtbot.wait(30)
    gal._model.set_page_size(5)
    gal._sync_controls()
    gal.set_filter("Unreviewed")
    gal.set_advance_after_decision(True)
    _click(gal, 0)

    def spots():
        return [(w.mapTo(gal, QtCore.QPoint(0, 0)).toTuple(), w.width())
                for w in (gal.undo_btn, gal.accept_all_btn, gal.reject_all_btn,
                          gal.page_spin, gal.page_label, gal.next_btn, gal._advance_check)]

    seen = set()
    for _ in range(N_EVENTS - 1):                        # 14 -> 1 unreviewed, 3 pages -> 1
        seen.add(tuple(spots()))
        gal.decide_selected(True)
        qtbot.wait(2)
    seen.add(tuple(spots()))
    assert len(seen) == 1, seen
    assert gal.accept_all_btn.text() == "Accept all 1"


def test_the_progress_line_keeps_its_floor_beside_a_long_project_label(qapp, qtbot, gal):
    old = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        gal.resize(940, 700)
        gal.show()
        qtbot.wait(30)
        hint = gal.minimumSizeHint()
        name = "apoferritin_example_with_a_rather_long_file_name.nanoscope"
        gal.set_project_save_state(f"/x/{name}", True)
        qtbot.wait(30)
        # Updated (review B-PROG-3): the floor is 330 px or the line's own text, whichever
        # is less -- a short line no longer holds room the project label could use.
        assert gal._progress.width() >= min(330, gal._progress.sizeHint().width())
        label = gal._project_save_label
        assert label.text() == f"Project: Unsaved changes — {name}"
        assert QtWidgets.QLabel.text(label).endswith("…")       # the glass gave way
        assert gal.minimumSizeHint() == hint
    finally:
        qapp.setFont(old)


def test_a_label_filter_face_keeps_the_value_and_the_two_windows_share_the_widgets(qtbot, gal):
    import inspect

    from nanoscope.gui import event_gallery, event_inspector
    gal.show()
    qtbot.wait(20)
    gal.set_vocabulary(_vocab("molecule=holo-ferritin-from-horse-spleen"))
    gal.set_filter("Labelled: molecule=holo-ferritin-from-horse-spleen")
    assert gal.filter_box.currentText() == "Labelled: molecule=holo-ferritin-from-horse-spleen"
    # Updated (review B-CLIP-2): too long for the face, the value is cut in the middle.
    face = gal.filter_box.shown_text()
    assert face.startswith("holo-") and face.endswith("spleen") and "…" in face, face
    for name in ("_ElidingCombo", "_ElidedLabel", "_widest_count", "_status_share"):
        assert inspect.getsource(getattr(event_gallery, name)) == \
            inspect.getsource(getattr(event_inspector, name)), name
    assert event_gallery._SHORT_TEXTS_ROLE == event_inspector._SHORT_TEXTS_ROLE
    assert event_gallery._PROGRESS_MIN_W == event_inspector._PROGRESS_MIN_W


# ---------------------------------------------------------------------------------------------
# Review B-NAV-4: a write that takes the cursor's tile out of the subset parks the cursor
# ---------------------------------------------------------------------------------------------
def _shown(qtbot, g):
    g.resize(940, 700)
    g.show()
    qtbot.wait(30)
    g.view.setFocus()
    return _ids(g)


def test_a_decision_that_takes_the_cursor_tile_out_parks_the_cursor_beside_it(qtbot, gal):
    ids = _shown(qtbot, gal)
    gal.set_filter("Unreviewed")
    _click(gal, 4)
    seen: list[int] = []
    gal.eventSelected.connect(seen.append)
    QTest.keyClick(gal.view, Qt.Key_A)                   # Advance off: nothing is chosen
    assert gal.selected_event_ids() == [] and gal.current_event_id() == ids[5]
    assert seen == []
    QTest.keyClick(gal.view, Qt.Key_Right)               # ...the next one, as the inspector
    assert gal.selected_event_ids() == [ids[5]] and seen == [ids[5]]
    QTest.keyClick(gal.view, Qt.Key_R)
    QTest.keyClick(gal.view, Qt.Key_Left)                # ...and back past the decided one
    assert gal.selected_event_ids() == [ids[3]] and seen[-1] == ids[3]
    # Accepted only + R has the same shape (HEAD left nothing current there too).
    gal.set_filter("Accepted only")
    row = _ids(gal).index(ids[8])
    _click(gal, row)
    QTest.keyClick(gal.view, Qt.Key_R)
    assert gal.selected_event_ids() == [] and gal.current_event_id() == ids[9]


def test_a_click_on_the_parked_tile_announces_it(qtbot, gal):
    ids = _shown(qtbot, gal)
    gal.set_filter("Unreviewed")
    _click(gal, 2)
    gal.decide_selected(True)
    assert gal.current_event_id() == ids[3] and gal.selected_event_ids() == []
    seen: list[int] = []
    gal.eventSelected.connect(seen.append)
    rect = gal.view.visualRect(gal.view.currentIndex())
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
    assert gal.selected_event_ids() == [ids[3]] and seen == [ids[3]]


def test_an_echo_of_the_same_write_keeps_the_cursor_parked(qtbot, gal):
    """In the wired app MainWindow echoes the write back (refresh / reflect_acceptance);
    the second refilter must not forget where the cursor waits."""
    ids = _shown(qtbot, gal)
    gal.set_filter("Unreviewed")
    _click(gal, 6)
    gal.decide_selected(False)
    gal.reflect_acceptance(ids[6], False)
    gal.refresh()
    QTest.keyClick(gal.view, Qt.Key_Right)
    assert gal.selected_event_ids() == [ids[7]]


def test_at_the_end_of_the_subset_the_cursor_stays_on_a_tile(qtbot, gal):
    ids = _shown(qtbot, gal)
    gal._model.set_page_size(5)
    gal._sync_controls()
    gal.set_filter("Unreviewed")
    gal.set_advance_after_decision(True)
    gal.set_page(gal.page_count() - 1)
    _click(gal, gal._model.rowCount() - 1)
    assert gal.decide_selected(False) is False
    assert gal.progress_text().startswith("End of Unreviewed")
    assert gal.current_event_id() == ids[-2] and gal.selected_event_ids() == []
    QTest.keyClick(gal.view, Qt.Key_Left)
    assert gal.selected_event_ids() == [ids[-2]]


# ---------------------------------------------------------------------------------------------
# Review B-PROG-3: the status bar is shared by what the two lines say
# ---------------------------------------------------------------------------------------------
def test_status_share_grants_in_turn():
    from nanoscope.gui.event_gallery import _status_share
    # Both fit: each gets all it asks; the floor never exceeds the progress line's text.
    assert _status_share(800, 200, 370, 130, 330) == (200, 370)
    # The progress line's floor first, then the label's save state, then the progress
    # line's text, then the label's name.
    assert _status_share(700, 1000, 370, 130, 330) == (330, 130)
    assert _status_share(700, 500, 370, 130, 330) == (330, 200)
    assert _status_share(400, 1000, 370, 130, 330) == (330, 70)


def test_the_status_bar_is_shared_by_content(qapp, qtbot, tmp_path):
    old = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        session, doc_id, events = _session(tmp_path)
        g = EventGalleryWindow(session)
        qtbot.addWidget(g)
        g.set_events(doc_id, 0, events.df)
        g.resize(940, 700)
        g.show()
        qtbot.wait(40)
        g.set_project_save_state("/x/apoferritin_example.nanoscope", False)
        qtbot.wait(40)
        label = g._project_save_label
        # A short progress line leaves the project label all of its name.
        assert QtWidgets.QLabel.text(label) == label.text()
        hint = g.minimumSizeHint()
        fixed = [(b.mapTo(g, QtCore.QPoint(0, 0)).x(), b.width())
                 for b in (g._first_unreviewed_btn, g._project_save_button)]
        g.set_vocabulary(_vocab("molecule=apo-ferritin", "molecule=holo-ferritin",
                                "type=translocation", "quality=needs-second-look"))
        df = g._df
        for r, tok in enumerate(("molecule=apo-ferritin", "molecule=holo-ferritin",
                                 "type=translocation", "quality=needs-second-look")):
            df.iat[r, df.columns.get_loc("flag")] = tok
        g.refresh()
        qtbot.wait(40)
        # A long one takes the name's room first; the save state stays readable.
        assert g._progress.width() > 330
        assert QtWidgets.QLabel.text(label).startswith("Project: Saved")
        assert QtWidgets.QLabel.text(g._progress).count("=") >= 1      # a count is on view
        assert [(b.mapTo(g, QtCore.QPoint(0, 0)).x(), b.width())
                for b in (g._first_unreviewed_btn, g._project_save_button)] == fixed
        assert g.minimumSizeHint() == hint
    finally:
        qapp.setFont(old)
