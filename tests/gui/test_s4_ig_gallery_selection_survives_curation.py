"""IG-01 — a curation gesture in the Event gallery keeps the block the user swept.

``EventGalleryWindow._set_rows_accepted`` is "THE one place this window writes ``accepted``,
for every curation gesture it has" — A, R, Space and the two "…selected" buttons all end
there. It finished with ``keep_id = current_event_id(); refilter(); show_event(keep_id)``,
and ``refilter()`` is a model reset: a reset clears the selection and ``show_event`` puts
back exactly ONE tile. Measured on six swept tiles, identical by keyboard and by button:

    swept [0,1,2,3,4,5] → R → rows rejected [0,1,2,3,4,5], selection now [0]
                        → A → rows re-accepted [0]        (five stay rejected)

with nothing on the status line, in ``refusal_note()`` or anywhere else saying the scope had
shrunk. The realistic loss is the ordinary one: sweep a block, reject it, see it was the
wrong block, press A to put it back — one tile returns.

The window already states the rule twice. ``refresh()``: *"THE SELECTION SURVIVES.
``refilter()`` is a model reset and a reset clears the selection … a curator holding five
tiles selected lost them the moment anything else in the app announced an edit"*, and
``_after_label_write``: *"the tiles you had just labelled were deselected and the next digit
wrote nothing"*. Somebody else's edit and the label write were fixed; the window's own
curation write was the one path that never got it.

Everything below is driven through a wired MainWindow and the real gallery: a real
click + shift-click sweep on the view's viewport, real key presses, real buttons.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(4_000, 4_400), (10_000, 10_500), (16_000, 16_600), (22_000, 22_500),
          (28_000, 28_400), (34_000, 34_700), (40_000, 40_500), (46_000, 46_400),
          (52_000, 52_600)]
#: The block the user sweeps in every test below.
_SWEEP = [0, 1, 2, 3, 4, 5]


def _signal() -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "ig01.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    w._open_event_gallery()
    qtbot.wait(20)
    gal = w._event_gallery
    gal.page_size_box.setCurrentText("200")
    gal.set_filter("All")
    gal.show()
    gal.activateWindow()
    qtbot.wait(40)
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _sweep(gal, rows: list[int], qtbot) -> None:
    """The gesture itself: click the first tile, shift-click the last. No selection model
    calls — the collapse was invisible to tests precisely because they used those."""
    view = gal.view
    first = gal._model.index_for_df_row(rows[0])
    last = gal._model.index_for_df_row(rows[-1])
    assert first.isValid() and last.isValid(), "the block is not on this page"
    QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                     view.visualRect(first).center())
    QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier,
                     view.visualRect(last).center())
    qtbot.wait(20)
    assert sorted(gal._selected_rows()) == rows, (
        f"the sweep itself did not select {rows}: {sorted(gal._selected_rows())}")


def _selected(gal) -> list[int]:
    return sorted(gal._selected_rows())


def _rejected(df) -> list[int]:
    return [i for i in range(len(df)) if not bool(df["accepted"].iloc[i])]


# ================================================================================================
# The defect, on every gesture that writes
# ================================================================================================
def test_the_keyboard_reject_keeps_the_swept_block_and_A_puts_all_of_it_back(wired, qtbot):
    """THE MEASURED LOSS, end to end: R over six tiles, then the A that undoes it."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    _sweep(gal, _SWEEP, qtbot)

    QTest.keyClick(gal, QtCore.Qt.Key_R)
    qtbot.wait(40)
    assert _rejected(df) == _SWEEP, _rejected(df)
    assert _selected(gal) == _SWEEP, (
        f"the reject narrowed the selection to {_selected(gal)}; the next gesture would land "
        f"on {len(_selected(gal))} tiles instead of {len(_SWEEP)}")

    QTest.keyClick(gal, QtCore.Qt.Key_A)
    qtbot.wait(40)
    assert _rejected(df) == [], (
        f"pressing A after R left {_rejected(df)} rejected — the block did not come back")
    assert _selected(gal) == _SWEEP


def test_space_inverts_the_same_block_twice(wired, qtbot):
    """Space is the destructive-looking one (a mixed block all becomes rejected), so it is
    also the one a user reaches for to undo their own sweep."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    _sweep(gal, _SWEEP, qtbot)

    QTest.keyClick(gal, QtCore.Qt.Key_Space)
    qtbot.wait(40)
    assert _rejected(df) == _SWEEP
    assert _selected(gal) == _SWEEP

    QTest.keyClick(gal, QtCore.Qt.Key_Space)
    qtbot.wait(40)
    assert _rejected(df) == [], f"the second Space put back only {_SWEEP} minus {_rejected(df)}"
    assert _selected(gal) == _SWEEP


def test_the_reject_selected_button_keeps_the_block_too(wired, qtbot):
    """Same write, other gesture — the guard is at the write, not on the keypress."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    _sweep(gal, _SWEEP, qtbot)

    gal.reject_selected_btn.click()
    qtbot.wait(40)
    assert _rejected(df) == _SWEEP
    assert _selected(gal) == _SWEEP

    gal.accept_selected_btn.click()
    qtbot.wait(40)
    assert _rejected(df) == []
    assert _selected(gal) == _SWEEP


def test_the_current_tile_still_follows_the_gesture(wired, qtbot):
    """The tile the keyboard cursor sits on is unchanged — restoring the block must not
    move "where you are" to the top of the sweep, which is what a plain re-select would."""
    w, doc_id, gal = wired
    _sweep(gal, _SWEEP, qtbot)
    before = gal.current_event_id()
    assert before is not None

    QTest.keyClick(gal, QtCore.Qt.Key_R)
    qtbot.wait(40)
    assert gal.current_event_id() == before, (
        f"the current tile moved from {before} to {gal.current_event_id()}")


def test_a_gesture_that_writes_nothing_still_leaves_the_selection_alone(wired, qtbot):
    """The early return above the write was always fine; this pins that the fix did not
    make it worse (an R over rows already rejected writes nothing at all)."""
    w, doc_id, gal = wired
    _sweep(gal, _SWEEP, qtbot)
    QTest.keyClick(gal, QtCore.Qt.Key_R)
    qtbot.wait(40)
    QTest.keyClick(gal, QtCore.Qt.Key_R)          # nothing left to change
    qtbot.wait(40)
    assert _selected(gal) == _SWEEP


# ================================================================================================
# ...and where the tiles really cannot be kept, the line says so
# ================================================================================================
def test_when_the_filter_takes_the_tiles_off_the_page_the_status_line_admits_it(wired, qtbot):
    """Under *Accepted only*, rejecting is the one case where the block genuinely cannot
    stay selected: the tiles leave the subset the filter draws. That is a drop, so it is
    said — and it is re-derived, so putting the filter back to *All* takes the sentence
    away instead of leaving a stale count standing."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_filter("Accepted only")
    qtbot.wait(40)
    assert gal.event_count() == len(df)

    _sweep(gal, _SWEEP, qtbot)
    QTest.keyClick(gal, QtCore.Qt.Key_R)
    qtbot.wait(40)

    assert _rejected(df) == _SWEEP
    assert _selected(gal) == [], "the rejected tiles are not in the 'Accepted only' subset"
    status = gal.status.text()
    assert "6 selected tiles left this page" in status, status
    # Sweep-7 fix-up B2: the sentence names WHERE the tiles went (the control that excludes
    # them) rather than claiming "still curated" -- a claim that was false on the inbound
    # route, where a tile can leave a page for a reason nobody curated.
    assert "6 selected tiles left this page — filtered out of Accepted only — no longer selected" in status, status
    assert "still curated" not in status, status

    gal.set_filter("All")
    qtbot.wait(40)
    assert "left this page" not in gal.status.text(), (
        f"the note outlived the state it describes: {gal.status.text()}")


def test_the_note_is_absent_when_the_block_was_kept(wired, qtbot):
    """No drop, no sentence — the status line does not warn about something that did not
    happen, and the rest of the line is untouched."""
    w, doc_id, gal = wired
    _sweep(gal, _SWEEP, qtbot)
    QTest.keyClick(gal, QtCore.Qt.Key_R)
    qtbot.wait(40)
    status = gal.status.text()
    assert "left this page" not in status, status
    # S7-POP6: the digits named are the ones the palette ARMS, and this fixture has none.
    assert "click to select, R to reject" in status, status
    assert "1–9 to label" not in status, status


def test_a_new_detection_clears_the_note(wired, qtbot):
    """The ids belong to a frame; a different frame is a different detection, and a count
    of tiles that are not there any more is exactly the stale surface this app refuses."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_filter("Accepted only")
    qtbot.wait(40)
    _sweep(gal, _SWEEP, qtbot)
    QTest.keyClick(gal, QtCore.Qt.Key_R)
    qtbot.wait(40)
    assert "left this page" in gal.status.text()

    gal.set_events(doc_id, 0, df.copy(), module_id="detect.schmitt")
    qtbot.wait(20)
    assert gal._offpage_ids == ()
    assert "left this page" not in gal.status.text(), gal.status.text()
