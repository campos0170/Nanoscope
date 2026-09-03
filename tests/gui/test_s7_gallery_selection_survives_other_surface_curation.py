"""Sweep-7 MISSED (major) — the gallery's multi-tile selection survives a curation made on
ANOTHER surface.

``AnalysisPanel.eventAcceptanceChanged`` had two gallery listeners: ``gallery.refresh()``
("THE SELECTION SURVIVES": snapshot by event_id, refilter, restore) and
``gallery.reflect_acceptance``, which called the bare ``_model.refilter()`` — a model reset
with no save and no restore — and, being connected later, ran last and wiped what
``refresh`` had just put back. Measured on the owner's 110-event -1000 mV recording under
Show = All: click tile 3, shift-click 8, tick ONE unrelated checkbox in the events table ->
the gallery held ``[]`` with "Reject selected" still enabled; the click rejected nothing and
said nothing. Same through the Event Inspector's R.

Fixed at the invariant: one selection-keeping model reset
(``_refilter_keeping_selection``) that ``refresh``, ``reflect_acceptance`` and the window's
own write all use. Everything below is the real gesture: mouse clicks on the tile view, the
table MODEL's ``setData`` on the accepted column (the checkbox), the inspector's R key, the
gallery's own button.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore
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
#: The block the user sweeps in every test below (df rows == event ids here).
_SWEEP = [3, 4, 5, 6, 7, 8]
_UNRELATED = 20


def _signal() -> np.ndarray:
    rng = np.random.default_rng(11)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "miss.dat", format_name="synthetic", n_samples=N,
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
    """Click the first tile, shift-click the last — the gesture, not the selection model."""
    view = gal.view
    first = gal._model.index_for_df_row(rows[0])
    last = gal._model.index_for_df_row(rows[-1])
    assert first.isValid() and last.isValid(), "the block is not on this page"
    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     view.visualRect(first).center())
    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.ShiftModifier,
                     view.visualRect(last).center())
    qtbot.wait(20)
    assert sorted(gal._selected_rows()) == rows, sorted(gal._selected_rows())


def _selected(gal) -> list[int]:
    return sorted(gal._selected_rows())


def _rejected(df) -> list[int]:
    return [i for i in range(len(df)) if not bool(df["accepted"].iloc[i])]


def _tick_table_checkbox(w, row: int, accepted: bool) -> None:
    """The events table's checkbox: ``setData`` on the model's accepted column, exactly what
    a click on the box does."""
    model = w._analysis_panel._table._model
    idx = model.index(row, model._accept_col())
    assert idx.isValid()
    assert model.setData(idx, Qt.Checked if accepted else Qt.Unchecked, Qt.CheckStateRole)


# ================================================================================================
def test_control_the_swept_block_is_what_reject_selected_rejects(wired, qtbot):
    """The finding's control: sweep 3..8, click the button, [3..8] are rejected."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    _sweep(gal, _SWEEP, qtbot)
    QTest.mouseClick(gal.reject_selected_btn, Qt.LeftButton)
    qtbot.wait(40)
    assert _rejected(df) == _SWEEP, _rejected(df)


def test_a_checkbox_tick_in_the_events_table_does_not_wipe_the_swept_block(wired, qtbot):
    """THE MEASURED DEFECT: one unrelated checkbox in the table, and the sweep was gone."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    _sweep(gal, _SWEEP, qtbot)
    _tick_table_checkbox(w, _UNRELATED, False)
    qtbot.wait(60)
    assert _rejected(df) == [_UNRELATED]
    assert _selected(gal) == _SWEEP, (
        f"the table's checkbox wiped the gallery's selection: holds {_selected(gal)}")
    assert gal.current_event_id() == _SWEEP[-1]

    QTest.mouseClick(gal.reject_selected_btn, Qt.LeftButton)
    qtbot.wait(40)
    assert _rejected(df) == sorted(_SWEEP + [_UNRELATED]), _rejected(df)


def test_the_inspectors_R_key_does_not_wipe_the_swept_block(wired, qtbot):
    """Same wipe through the other curation window (the finding's section C). Selection is
    mirrored across the three surfaces, so after the sweep the inspector is on tile 8 (the
    sweep's last); its R rejects 8 through the panel, and the gallery's block must still be
    there for the gallery's own R to finish the job."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    w._open_event_inspector()
    qtbot.wait(20)
    insp = w._event_inspector
    insp.show()
    _sweep(gal, _SWEEP, qtbot)
    assert insp.current_event_id() == _SWEEP[-1], "the inspector mirrors the gallery's cursor"
    QTest.keyClick(insp, Qt.Key_R)
    qtbot.wait(60)
    assert _rejected(df) == [_SWEEP[-1]]
    assert _selected(gal) == _SWEEP, (
        f"the inspector's R wiped the gallery's selection: holds {_selected(gal)}")

    # The button, not the R key: the inspector is the active window now and the gallery's
    # R is a window-context shortcut. The button is the gesture the finding measured.
    QTest.mouseClick(gal.reject_selected_btn, Qt.LeftButton)
    qtbot.wait(40)
    assert _rejected(df) == _SWEEP, _rejected(df)


def test_a_curation_elsewhere_that_takes_a_swept_tile_off_the_page_is_admitted(wired, qtbot):
    """Under 'Accepted only' the table rejects a tile INSIDE the sweep: the subset drops it,
    the other five stay selected, and the status line says one tile left — the same
    sentence the window's own gesture uses, instead of a selection that shrank silently."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_filter("Accepted only")
    qtbot.wait(20)
    _sweep(gal, _SWEEP, qtbot)
    _tick_table_checkbox(w, 5, False)
    qtbot.wait(60)
    assert _rejected(df) == [5]
    assert _selected(gal) == [3, 4, 6, 7, 8], _selected(gal)
    status = gal.status.text()
    # Sweep-7 fix-up B2: the sentence says which control excludes the tile now, and never
    # "still curated" -- on this route a tile can leave for a reason nobody curated (a page
    # shift, a re-cut representative sample), so the claim was not the window's to make.
    assert "1 selected tile left this page — filtered out of Accepted only — no longer selected" in status, (
        status)

    gal.set_filter("All")
    qtbot.wait(20)
    assert "left this page" not in gal.status.text(), gal.status.text()
