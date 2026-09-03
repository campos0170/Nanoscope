"""WAVE-A/fixup WB3 — the Event gallery's curation keys belong to the Event gallery.

The gallery registered A / R / Space / the digits / the page keys / ⌘A / Esc / ⌘I as
``Qt.WindowShortcut`` QActions. That context asks one question — "is my window the one Qt
calls active" — and the answer stops being the right one the moment a SECOND curation
surface is on screen, because activation and focus can disagree. Measured in the wired
app (a real MainWindow at 1512x900, fresh QSettings, the gallery open the way the Analysis
menu opens it), with six rows selected in the events table and 'r' delivered to the
TABLE's view:

  BEFORE: ``EventTable._handle_curation_key`` was never called ("handler calls: []") and
          exactly ONE event was rejected — the gallery's own current tile, an event the
          user had not selected anywhere. Six rows were highlighted; one unrelated event
          was written.
  AFTER:  "handler calls: [82]", six events rejected, and nothing in the gallery moved.

The wave-A round made this worth fixing rather than merely noting: before it the events
table could hold ONE row, so a mis-routed R cost one event either way. It can now hold
three hundred.

The fix narrows the context to ``Qt.WidgetWithChildrenShortcut`` (Qt then walks up from
``QApplication.focusWidget()`` and stops at the first window, so the shortcut matches when
and only when the focused widget lives in the gallery) and adds the fallback that keeps
the narrowing from costing the gallery its own keys — a KeyPress filter on the tile view
plus ``keyPressEvent`` on the window, both routed through the SAME QActions
(:meth:`EventGalleryWindow._handle_own_key`), so a key can never mean two things.

Pre-fix behaviour is provable without editing source back and forth: run this file with
the out-of-tree plugin that restores ``Qt.WindowShortcut`` and removes the fallback at
runtime —

    PYTHONPATH=/private/tmp/claude-501/-Users-abrahamcampos-Desktop/\
6f43f43e-d8d3-4c35-89b0-a20f09cbc171/scratchpad/wave-a-fixup/\
nanoscope_gui_event_gallery_py/prefix \
      pytest tests/gui/test_wa2_gallery_keys_answer_for_their_own_window.py \
      -p wa2_gallery_prefix

which fails the three routing tests (3 failed, 3 passed). The other three pass pre-fix by
construction and say so where they stand: they pin gestures this change must not disturb.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets
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
N_EVENTS = 30
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000


def _signal() -> np.ndarray:
    """A synthetic 50 kHz float32 recording with 30 square blockades."""
    rng = np.random.default_rng(7)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    """A MainWindow at the owner's 1512x900 with a detection and its gallery open, every
    signal MainWindow wires between the two surfaces live."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 900)
    w.show()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa2_keys.dat", format_name="synthetic",
                         n_samples=N, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == N_EVENTS, len(df)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    w._act_event_gallery.trigger()
    qtbot.wait(20)
    gal = w._event_gallery
    gal.page_size_box.setCurrentText("24")
    gal.set_filter("All")
    gal.resize(1200, 860)
    gal.show()
    qtbot.wait(40)
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _accepted(w, doc_id) -> int:
    return int(_frame(w, doc_id)["accepted"].sum())


def _click_table(w, row: int, mods=Qt.NoModifier) -> None:
    tbl = w._analysis_panel._table
    idx = tbl._proxy.index(row, 0)
    tbl.view.scrollTo(idx)
    QTest.mouseClick(tbl.view.viewport(), Qt.LeftButton, mods,
                     tbl.view.visualRect(idx).center())


def _click_tile(gal, tile: int, mods=Qt.NoModifier) -> None:
    idx = gal._model.index(tile, 0)
    assert idx.isValid(), f"tile {tile} is not on this page"
    gal.view.scrollTo(idx)
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, mods,
                     gal.view.visualRect(idx).center())


def _drop_focus_out_of(gal) -> None:
    """Leave no widget of ``gal`` focused — the state where activation and focus disagree.

    Qt drops the focus widget when a window is deactivated and a window can be raised
    without any widget taking it, so this is the state a second window creates, not a
    contrivance: what the assertions below then measure is whether the gallery answers for
    a key it was not given.
    """
    fw = QtWidgets.QApplication.focusWidget()
    if fw is not None and (fw is gal or gal.isAncestorOf(fw)):
        fw.clearFocus()


def test_a_key_sent_to_the_events_table_reaches_the_tables_batch_handler_and_not_the_gallery(
        wired, qtbot):
    w, doc_id, gal = wired
    tbl = w._analysis_panel._table
    gal.activateWindow()
    qtbot.wait(20)
    _drop_focus_out_of(gal)
    qtbot.wait(20)

    seen: list[int] = []
    handler = tbl._handle_curation_key
    tbl._handle_curation_key = lambda key: (seen.append(int(key)), handler(key))[1]

    _click_table(w, 2)
    _click_table(w, 7, Qt.ShiftModifier)
    qtbot.wait(30)
    rows = tbl.selected_source_rows()
    assert len(rows) == 6, rows
    focused = QtWidgets.QApplication.focusWidget()
    assert focused is None or not gal.isAncestorOf(focused), (
        "the point of this test is a key the gallery was not given")

    tiles_before = sorted(gal.selected_event_ids())
    before = _accepted(w, doc_id)
    QTest.keyClick(tbl.view, Qt.Key_R)
    qtbot.wait(60)

    # PRE-FIX: seen == [] and exactly one event flipped -- the gallery's current tile.
    assert seen == [int(Qt.Key_R)], "the focused surface's own handler must get the key"
    assert before - _accepted(w, doc_id) == len(rows)
    assert sorted(gal.selected_event_ids()) == tiles_before


def test_the_gallery_curates_its_own_selection_when_its_view_holds_the_focus(wired, qtbot):
    w, doc_id, gal = wired
    gal.activateWindow()
    gal.view.setFocus(Qt.OtherFocusReason)
    qtbot.wait(20)
    _click_tile(gal, 0)
    _click_tile(gal, 3, Qt.ShiftModifier)
    qtbot.wait(30)
    chosen = sorted(gal.selected_event_ids())
    assert len(chosen) == 4, chosen

    before = _accepted(w, doc_id)
    QTest.keyClick(gal.view, Qt.Key_R)
    qtbot.wait(60)
    assert before - _accepted(w, doc_id) == 4
    QTest.keyClick(gal.view, Qt.Key_A)
    qtbot.wait(60)
    assert _accepted(w, doc_id) == before, "A must put back exactly what R took"
    assert sorted(gal.selected_event_ids()) == chosen


def test_the_gallery_keys_still_work_with_no_widget_of_it_focused(wired, qtbot):
    """The other half of the narrowing: a gallery nobody has clicked into keeps its keys."""
    w, doc_id, gal = wired
    w.activateWindow()
    w._analysis_panel._table.view.setFocus(Qt.OtherFocusReason)
    qtbot.wait(20)
    _click_tile(gal, 0)
    _click_tile(gal, 4, Qt.ShiftModifier)
    qtbot.wait(30)
    chosen = sorted(gal.selected_event_ids())
    assert len(chosen) == 5, chosen
    _drop_focus_out_of(gal)
    qtbot.wait(20)

    before = _accepted(w, doc_id)
    QTest.keyClick(gal.view, Qt.Key_R)
    qtbot.wait(60)
    # PRE-FIX: 0 -- the window shortcut did not match (another window was active) and
    # QListView's type-to-search swallowed the letter.
    assert before - _accepted(w, doc_id) == 5


def test_select_all_and_escape_reach_the_gallery_when_they_are_sent_to_the_window(
        wired, qtbot):
    w, doc_id, gal = wired
    w.activateWindow()
    w._analysis_panel._table.view.setFocus(Qt.OtherFocusReason)
    qtbot.wait(20)
    _drop_focus_out_of(gal)

    QTest.keyClick(gal, Qt.Key_A, Qt.ControlModifier)
    qtbot.wait(40)
    assert len(gal.selected_event_ids()) == gal._model.rowCount()
    QTest.keyClick(gal, Qt.Key_Escape)
    qtbot.wait(40)
    assert gal.selected_event_ids() == []


def test_one_key_is_handled_once_even_when_both_roads_are_open(wired, qtbot):
    """The shortcut and the fallback are the same QAction, and a key takes one of them.

    Space INVERTS the acceptance of the selection, so a double-run would be invisible in
    the count and visible only as an unchanged frame — which is exactly the bug a second
    hand-written key table would produce. Driven with the tile view focused (the shortcut's
    road) over a selection whose events are all accepted.
    """
    w, doc_id, gal = wired
    gal.activateWindow()
    gal.view.setFocus(Qt.OtherFocusReason)
    qtbot.wait(20)
    _click_tile(gal, 0)
    _click_tile(gal, 2, Qt.ShiftModifier)
    qtbot.wait(30)
    assert len(gal.selected_event_ids()) == 3
    before = _accepted(w, doc_id)
    QTest.keyClick(gal.view, Qt.Key_Space)
    qtbot.wait(60)
    assert before - _accepted(w, doc_id) == 3, "one Space, one inversion"


def test_shift_clicking_a_second_tile_selects_the_whole_run_in_the_wired_gallery(
        wired, qtbot):
    """GAL-A2, pinned. Reported as a wired-only collapse to a single tile and left
    unowned; it does not reproduce on this tree — in the wired app, with MainWindow's
    ``eventSelected`` relay live, click tile 2 then Shift-click tile 6 selects the run,
    and a ⌘-click after it still adds one. Pinned here because the relay is what used to
    eat these gestures and nothing else in the suite drives Shift through it."""
    w, doc_id, gal = wired
    gal.activateWindow()
    gal.view.setFocus(Qt.OtherFocusReason)
    qtbot.wait(20)
    _click_tile(gal, 2)
    qtbot.wait(20)
    assert gal._selected_rows() == [2]
    _click_tile(gal, 6, Qt.ShiftModifier)
    qtbot.wait(20)
    assert sorted(gal._selected_rows()) == [2, 3, 4, 5, 6]
    _click_tile(gal, 9, Qt.ControlModifier)
    qtbot.wait(20)
    assert sorted(gal._selected_rows()) == [2, 3, 4, 5, 6, 9]
