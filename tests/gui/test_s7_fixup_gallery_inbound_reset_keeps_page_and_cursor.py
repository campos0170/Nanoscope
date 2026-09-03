"""Sweep-7 fix-up B4 / B5 / B2 — what the gallery's selection-keeping reset does when the
frame is edited on ANOTHER surface: it restores on the page the user is on, it never
re-selects a tile the user deselected, and the sentence about tiles it could not give back
says where they went instead of claiming they were curated.

Measured on the owner's -1000 mV recording (110 events) after the wave-1 fix:

* B4 -- Show = Accepted only, page 2, five tiles swept with the cursor on the FIRST tile of
  the page; the events table rejects an event earlier in the subset. Every page-2 tile
  shifted back one slot, ``_focus_event`` followed the cursor tile to the end of page 1,
  the page turned under the pointer, and the other four could not be restored on a page
  they were not on: selection [24] alone, all five still accepted, and the line read
  "4 selected tiles left this page — still curated, but no longer selected".
* B5 -- click tile 3, Ctrl-click it off (holds [], cursor 3), tick one unrelated checkbox
  in the events table: tile 3 was SELECTED again (``view.setCurrentIndex`` is
  ClearAndSelect on an ExtendedSelection view) and "Reject selected" rejected it.
* B2 -- Representative sample under Accepted only, four tiles swept, one unrelated event
  rejected in the table: the dwell ladder was re-cut, a swept tile (still accepted,
  untouched) fell off the sample, and the line called it "still curated".

Everything below is the real gesture: mouse clicks on the tile view, the table model's
``setData`` on the accepted column (the checkbox), the gallery's own button.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
N_EVENTS = 60
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000
PAGE = 24


def _signal() -> np.ndarray:
    rng = np.random.default_rng(29)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "reset.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == N_EVENTS, len(df)
    assert df["event_id"].tolist() == list(range(N_EVENTS))
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    w._act_event_gallery.trigger()
    qtbot.wait(20)
    gal = w._event_gallery
    gal.show()
    gal.activateWindow()
    gal.resize(1200, 800)                       # room for a whole page of 24 tiles
    gal.page_size_box.setCurrentText(str(PAGE))
    gal.set_filter("All")
    qtbot.wait(60)
    assert gal._model.page_size() == PAGE and gal.page_count() == 3
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _click_pos(gal, pos: int, mod=Qt.NoModifier) -> None:
    """A real click on the tile at grid position ``pos`` of the current page."""
    idx = gal._model.index(pos, 0)
    assert idx.isValid(), f"no tile at position {pos}"
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, mod, gal.view.visualRect(idx).center())


def _click_row(gal, df_row: int, mod=Qt.NoModifier) -> None:
    idx = gal._model.index_for_df_row(df_row)
    assert idx.isValid(), f"row {df_row} is not on this page"
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, mod, gal.view.visualRect(idx).center())


def _tick_table_checkbox(w, row: int, accepted: bool) -> None:
    """The events table's checkbox: ``setData`` on the model's accepted column."""
    model = w._analysis_panel._table._model
    idx = model.index(row, model._accept_col())
    assert idx.isValid()
    assert model.setData(idx, Qt.Checked if accepted else Qt.Unchecked, Qt.CheckStateRole)


def _offpage_clause(gal) -> str:
    hits = [c for c in gal.status.text().split("   ·   ") if "left this page" in c]
    return hits[0] if hits else ""


# ================================================================================================
# B4 -- the page stays where the user is
# ================================================================================================
def test_a_curation_elsewhere_that_shifts_the_subset_does_not_turn_the_page(wired, qtbot):
    """Cursor on the first tile of page 2, five tiles swept; the table rejects the first
    event of the subset. The page stays 2, the four tiles still on it stay selected, and
    the one that shifted onto page 1 is named as such -- not as 'curated'."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_filter("Accepted only")
    gal.set_page(1)
    qtbot.wait(40)
    assert gal.page() == 1
    _click_pos(gal, 4)
    _click_pos(gal, 0, Qt.ShiftModifier)
    qtbot.wait(20)
    swept = sorted(gal.selected_event_ids())
    assert swept == [24, 25, 26, 27, 28], swept
    assert gal.current_event_id() == 24

    _tick_table_checkbox(w, 0, False)
    qtbot.wait(80)
    assert not bool(df["accepted"].iloc[0])
    assert all(bool(df["accepted"].iloc[e]) for e in swept), "nothing else was curated"

    assert gal.page() == 1, f"the inbound edit turned the page to {gal.page() + 1}"
    assert gal._model.page_of_df_row(24) == 0, "event 24 shifted onto page 1"
    assert sorted(gal.selected_event_ids()) == [25, 26, 27, 28], gal.selected_event_ids()
    assert gal.current_event_id() in (25, 26, 27, 28)
    clause = _offpage_clause(gal)
    assert clause == "1 selected tile left this page — now on page 1 — no longer selected", (
        clause or gal.status.text())
    assert "still curated" not in gal.status.text()

    # The sentence is re-derived: turn to page 1 and it goes away by itself.
    gal.set_page(0)
    qtbot.wait(20)
    assert "left this page" not in gal.status.text(), gal.status.text()


def test_a_swept_block_clear_of_the_page_boundary_survives_whole(wired, qtbot):
    """Same edit with the sweep in the middle of the page: every tile shifts one slot and
    every tile is still selected, on the same page, with nothing to admit."""
    w, doc_id, gal = wired
    gal.set_filter("Accepted only")
    gal.set_page(1)
    qtbot.wait(40)
    _click_pos(gal, 5)
    _click_pos(gal, 9, Qt.ShiftModifier)
    qtbot.wait(20)
    swept = sorted(gal.selected_event_ids())
    assert swept == [29, 30, 31, 32, 33], swept
    _tick_table_checkbox(w, 0, False)
    qtbot.wait(80)
    assert gal.page() == 1
    assert sorted(gal.selected_event_ids()) == swept, gal.selected_event_ids()
    assert gal.current_event_id() == 33
    assert "left this page" not in gal.status.text(), gal.status.text()


# ================================================================================================
# B5 -- a tile the user deselected stays deselected
# ================================================================================================
def test_a_curation_elsewhere_does_not_reselect_the_cursor_tile_the_user_deselected(
        wired, qtbot):
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.page_size_box.setCurrentText("200")
    qtbot.wait(40)
    _click_row(gal, 3)
    qtbot.wait(10)
    assert gal.selected_event_ids() == [3] and gal.current_event_id() == 3
    _click_row(gal, 3, Qt.ControlModifier)
    qtbot.wait(10)
    assert gal.selected_event_ids() == [] and gal.current_event_id() == 3, (
        "Ctrl-click must toggle the tile off and leave the cursor on it")

    _tick_table_checkbox(w, 20, False)
    qtbot.wait(80)
    assert gal.selected_event_ids() == [], (
        f"the inbound edit re-selected {gal.selected_event_ids()}")
    assert gal.current_event_id() == 3, "the cursor itself survives the edit"

    gal.reject_selected_btn.click()
    qtbot.wait(60)
    rejected = [i for i in range(N_EVENTS) if not bool(df["accepted"].iloc[i])]
    assert rejected == [20], f"'Reject selected' rejected {rejected}"


# ================================================================================================
# B2 -- the off-page sentence says where the tile went
# ================================================================================================
def test_the_off_page_sentence_names_the_sample_not_a_curation(wired, qtbot):
    """Under the representative sample an unrelated rejection re-cuts the dwell ladder and a
    swept, still-accepted tile can fall off it. The line says that -- 'no longer in the
    representative sample' -- and never 'still curated'."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_filter("Accepted only")
    gal.set_representative(True)
    qtbot.wait(60)
    assert gal.is_representative()
    sample0 = list(gal._model.df_rows())
    assert len(sample0) == PAGE

    vanished: list[int] = []
    for outside in [r for r in range(N_EVENTS) if r not in set(sample0)][:8]:
        sample = list(gal._model.df_rows())
        gal.view.clearSelection()
        _click_row(gal, sample[2])
        _click_row(gal, sample[5], Qt.ShiftModifier)
        qtbot.wait(20)
        swept = sorted(gal.selected_event_ids())
        assert len(swept) == 4, swept
        _tick_table_checkbox(w, outside, False)
        qtbot.wait(80)
        now = set(gal.selected_event_ids())
        vanished = [e for e in swept if e not in now]
        if vanished:
            break
        _tick_table_checkbox(w, outside, True)
        qtbot.wait(60)
    assert vanished, "the fixture's ladder never moved; the case is not exercised"
    assert all(bool(df["accepted"].iloc[e]) for e in vanished), "nothing curated them"
    clause = _offpage_clause(gal)
    assert clause, gal.status.text()
    assert "still curated" not in clause, clause
    assert "no longer in the representative sample" in clause, clause
    assert clause.startswith(f"{len(vanished)} selected tile"), clause

    # Untick the sample: the sentence is RE-DERIVED, so it stops naming the sample. Each
    # tile is back in the unsampled subset, on whatever page that subset puts it, and the
    # line now counts only the ones that are off the page the user is on and names where
    # they went -- nothing at all when they are all back on this page.
    gal.set_representative(False)
    qtbot.wait(40)
    assert "representative sample" not in _offpage_clause(gal), gal.status.text()
    pages = {ev: gal._model.page_of_df_row(ev) for ev in vanished}
    assert all(p is not None for p in pages.values()), (
        f"unticking the sample must bring every tile back into the subset: {pages}")
    away = {ev: p for ev, p in pages.items() if p != gal.page()}
    clause = _offpage_clause(gal)
    if not away:
        assert clause == "", gal.status.text()
    else:
        assert clause.startswith(
            f"{len(away)} selected tile{'' if len(away) == 1 else 's'} left this page"), clause
        for p in sorted(set(away.values())):
            assert f"now on page {p + 1}" in clause, clause
        assert "still curated" not in clause, clause
    # Show the page that holds them and there is nothing left to admit.
    if len(set(pages.values())) == 1:
        gal.set_page(next(iter(pages.values())))
        qtbot.wait(20)
        assert "left this page" not in gal.status.text(), gal.status.text()


def test_a_selection_made_by_hand_retires_the_record_of_the_last_one(wired, qtbot):
    """The off-page record grows with every inbound reset and nothing retired it when the
    user moved on: sweep six under Accepted only, have the table reject one (1 left), click
    ONE new tile, have the table reject that too -- and the line counted "2 selected tiles
    left this page" over a selection that had been one tile. A click, a sweep or a
    Ctrl-click is a new selection, and the record starts again with it."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_filter("Accepted only")
    qtbot.wait(20)
    _click_pos(gal, 3)
    _click_pos(gal, 8, Qt.ShiftModifier)
    qtbot.wait(20)
    assert sorted(gal.selected_event_ids()) == [3, 4, 5, 6, 7, 8]
    _tick_table_checkbox(w, 5, False)
    qtbot.wait(80)
    assert sorted(gal.selected_event_ids()) == [3, 4, 6, 7, 8]
    assert _offpage_clause(gal) == ("1 selected tile left this page — filtered out of "
                                    "Accepted only — no longer selected"), gal.status.text()

    # The user starts over with one tile: the old record is about a selection that is gone.
    _click_pos(gal, 12)
    qtbot.wait(20)
    new_id = gal.current_event_id()
    assert gal.selected_event_ids() == [new_id] and new_id not in (3, 4, 5, 6, 7, 8)
    assert gal._offpage_ids == ()
    assert "left this page" not in gal.status.text(), gal.status.text()

    # ...and the table rejects THAT tile: one left, not two.
    _tick_table_checkbox(w, int(np.nonzero(df["event_id"].to_numpy() == new_id)[0][0]), False)
    qtbot.wait(80)
    assert not bool(df["accepted"].iloc[new_id])
    assert gal.selected_event_ids() == []
    assert _offpage_clause(gal) == ("1 selected tile left this page — filtered out of "
                                    "Accepted only — no longer selected"), gal.status.text()


def test_the_windows_own_reject_under_accepted_only_still_says_the_tiles_left(wired, qtbot):
    """The sentence the wave-1 test pinned, in its new words: the window's own R under
    Accepted only takes the tiles out of the subset, and the line says which subset."""
    w, doc_id, gal = wired
    gal.set_filter("Accepted only")
    qtbot.wait(20)
    _click_pos(gal, 3)
    _click_pos(gal, 8, Qt.ShiftModifier)
    qtbot.wait(20)
    assert len(gal.selected_event_ids()) == 6
    QTest.keyClick(gal, Qt.Key_R)
    qtbot.wait(60)
    clause = _offpage_clause(gal)
    assert clause == ("6 selected tiles left this page — filtered out of Accepted only — "
                      "no longer selected"), clause or gal.status.text()
