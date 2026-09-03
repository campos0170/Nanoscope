"""S7-POP2 — an inbound ``show_event`` on an event the gallery's subset excludes leaves NO
tile current, and says which event is filtered out of what.

``MainWindow`` wires ``AnalysisPanel.eventSelected -> gallery.show_event``: it is the door a
click in the events table or on the trace uses. Its own docstring records and fixes the
PAGINATION miss ("the gallery would sit on an unrelated page looking authoritative") and
the three subset filters had the same failure, still silent: ``page_of_df_row`` returned
``None`` and ``show_event`` returned with the PREVIOUS tile still current, still drawn with
the selection ring, so the next R rejected that other event. Measured on the owner's
-1000 mV recording: asked for event 0 -> current 12 -> R rejected [12].

Driven through the wired MainWindow and the real widgets: the table's ``select_event``
door, real mouse clicks on the tile view, real key presses.
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
I0 = -1000.0
N_EVENTS = 30
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(7)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pop2.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
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
    gal.show()
    gal.activateWindow()
    qtbot.wait(40)
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _click_tile(gal, df_row: int) -> None:
    idx = gal._model.index_for_df_row(df_row)
    assert idx.isValid(), f"row {df_row} is not on this page"
    QTest.mouseClick(gal.view.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
                     gal.view.visualRect(idx).center())


def _rejected(df) -> list[int]:
    return [int(df["event_id"].iloc[i]) for i in range(len(df))
            if not bool(df["accepted"].iloc[i])]


# ================================================================================================
# the door the table uses
# ================================================================================================
def test_asking_for_a_rejected_event_under_accepted_only_leaves_no_tile_current_and_says_so(
        wired, qtbot):
    """THE MEASURED MISS, through the wired door. Parked on an accepted tile, the user clicks
    a rejected event in the events table: nothing stays current, nothing stays selected, the
    status line names the event and the subset in the inspector's own words, and R rejects
    nothing."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    ap = w._analysis_panel
    rejected_id = int(df["event_id"].iloc[2])
    df.iloc[2, df.columns.get_loc("accepted")] = False
    gal.refresh()
    gal.set_filter("Accepted only")
    qtbot.wait(20)
    parked_row = gal._model.df_rows()[4]
    _click_tile(gal, parked_row)
    qtbot.wait(20)
    parked_id = int(df["event_id"].iloc[parked_row])
    assert gal.current_event_id() == parked_id

    # The table's select_event emits only on a CHANGE of row, so step onto another row
    # first: the door has to actually fire for this to test the gallery.
    ap.select_event(int(df["event_id"].iloc[8]))
    qtbot.wait(20)
    ap.select_event(rejected_id)
    qtbot.wait(40)

    assert gal.current_event_id() is None, (
        f"a stale tile stayed current: {gal.current_event_id()} (asked for {rejected_id})")
    assert gal.selected_event_ids() == []
    n_shown = gal.event_count()
    status = gal.status.text()
    assert f"Event {rejected_id} is filtered out of Accepted only ({n_shown} shown)" in status, (
        status)
    assert "set Show to All to reach it" in status, status

    before = _rejected(df)
    QTest.keyClick(gal.view, QtCore.Qt.Key_R)
    qtbot.wait(40)
    assert _rejected(df) == before, (
        f"R curated {sorted(set(_rejected(df)) - set(before))} — an event nobody asked for")
    QTest.mouseClick(gal.reject_selected_btn, QtCore.Qt.LeftButton)
    qtbot.wait(40)
    assert _rejected(df) == before


def test_widening_the_subset_drops_the_note_and_a_click_on_a_tile_clears_the_miss(wired, qtbot):
    """RE-DERIVED, NEVER REMEMBERED: the sentence goes away by itself when the event becomes
    reachable (Show = All) — and, separately, when the user focuses a tile, because the
    line then describes that tile."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    missed = int(df["event_id"].iloc[3])
    df.iloc[3, df.columns.get_loc("accepted")] = False
    gal.refresh()
    gal.set_filter("Accepted only")
    qtbot.wait(20)
    gal.show_event(missed)
    qtbot.wait(20)
    assert "filtered out of Accepted only" in gal.status.text()

    gal.set_filter("All")
    qtbot.wait(20)
    assert "filtered out" not in gal.status.text(), gal.status.text()

    gal.set_filter("Accepted only")
    qtbot.wait(20)
    gal.show_event(missed)
    qtbot.wait(20)
    assert "filtered out" in gal.status.text()
    row = gal._model.df_rows()[0]
    _click_tile(gal, row)
    qtbot.wait(20)
    assert gal.current_event_id() == int(df["event_id"].iloc[row])
    assert "filtered out" not in gal.status.text(), gal.status.text()


def test_the_subpopulation_filter_and_the_representative_sample_admit_the_miss_too(wired, qtbot):
    """The other two subsets the finding measured (the report window's own "Event gallery…"
    button turns the representative sample on for you). Same door, same sentence, naming
    the control that reaches the event."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    ids = df["event_id"].to_numpy(dtype=int)
    gal.set_subpopulations({int(e): (0 if i % 2 == 0 else 1) for i, e in enumerate(ids)},
                           source="synthetic", caveats=())
    gal.set_subpop_filter(0)
    qtbot.wait(20)
    _click_tile(gal, gal._model.df_rows()[1])
    qtbot.wait(20)
    other = int(ids[1])                       # class 1
    gal.show_event(other)
    qtbot.wait(20)
    assert gal.current_event_id() is None and gal.selected_event_ids() == []
    status = gal.status.text()
    assert f"Event {other} is filtered out of subpop 0 only ({gal.event_count()} shown)" in status, (
        status)
    assert "set Subpopulation to every subpopulation to reach it" in status, status

    gal.set_subpop_filter(None)
    gal.set_representative(True)
    qtbot.wait(40)
    sample = set(gal._model.df_rows())
    assert len(sample) < len(df), "the sample must leave something out for this to test"
    _click_tile(gal, next(iter(sorted(sample))))
    qtbot.wait(20)
    outside = int(df["event_id"].iloc[next(r for r in range(len(df)) if r not in sample)])
    gal.show_event(outside)
    qtbot.wait(20)
    assert gal.current_event_id() is None and gal.selected_event_ids() == []
    status = gal.status.text()
    assert (f"Event {outside} is filtered out of the representative sample "
            f"({gal.event_count()} shown)") in status, status
    assert "untick Representative sample to reach it" in status, status


def test_the_gallery_and_the_inspector_describe_the_same_miss_in_the_same_words(wired, qtbot):
    """The inspector already printed 'Event 0    [filtered out of Accepted only (100 shown)]';
    the gallery says the same fact with the same words, so the two windows over one frame do
    not tell two stories."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    missed = int(df["event_id"].iloc[5])
    df.iloc[5, df.columns.get_loc("accepted")] = False
    gal.refresh()
    gal.set_filter("Accepted only")
    w._open_event_inspector()
    qtbot.wait(20)
    insp = w._event_inspector
    insp.set_filter("Accepted only")
    insp.show_event(missed)
    gal.show_event(missed)
    qtbot.wait(20)
    n = gal.event_count()
    words = f"filtered out of Accepted only ({n} shown)"
    assert words in insp._readout.text().splitlines()[0], insp._readout.text().splitlines()[0]
    assert words in gal.status.text(), gal.status.text()
