"""S9-05 — Ctrl+M and the Add event button file the row D files, and clear the mask.

MEASURED DEFECT (sweep 9, finding 5; a clean 50 kHz blockade at samples 20 000 - 20 400, caught
with C). ``MainWindow._add_event_from_cursors``, behind Analysis ▸ Add event from cursors (Ctrl+M)
and the Measure tab's Add event button (``addRequested``), called the add door with no edge record,
so the door's default stood in: the caught event was stored as ``start=drawn;end=drawn``, with no
``fit_json`` and the FITTED bounds recorded as the span drawn (``drawn_start_idx`` /
``drawn_end_idx`` 20 000 / 20 400, where D records the placed 20 001 / 20 400). It never discarded
the mask either, so a second Ctrl+M was refused as an overlap with the event just added. Only D,
Return and Enter (``TraceKeyController.add``) passed the record and cleared the mask. The guide
says Ctrl+M and the Add event button "do the same" as D, that "After an add the mask is cleared",
and that each row records how its bounds were made.

THE FIX. After its own four gates (a recording, a shown channel, Measure on, a mask on the
channel in the Analysis panel), ``_add_event_from_cursors`` adds through
``TraceKeyController.add``, the add D makes: the same ``EdgeRecord`` (``bounds_source``, the drawn
span, ``fit_json``), the same second press for an edge that is not fitted (its sentence now says
"Press again", not "Press D again"), and the mask discarded after a successful add with Measure
left on. A plain bracket, which no key fitted, is stored as before: ``start=drawn;end=drawn`` with
the bracket, where its handles were last left, as the span drawn.

These tests drive a real MainWindow on a synthetic recording: the C key's slot with the pointer
stood in (as ``test_trace_keys`` does), then D, the Analysis menu action a Ctrl+M press triggers,
and a click on the Add event button; Edit ▸ Undo between adds.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 120_000
I0 = -1000.0
DEPTH = 300.0
EVENT = (20_000, 20_400)            # a clean blockade no detector was asked about
WAYS = ["D", "Ctrl+M", "Add event button"]


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    x[EVENT[0]:EVENT[1]] += DEPTH
    return x.astype(np.float32)


@pytest.fixture
def rig(qapp, qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "s9_05.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    return win, doc_id


def _point(win, doc_id, sample, px=10.0):
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _table(win, doc_id):
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return None if res is None else res.payload.df


def _catch(win, doc_id) -> tuple[int, int]:
    """Point at the middle of the event and press C: both edges fitted to their steps."""
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()
    bounds = win._trace_view.measure_bounds()
    assert bounds is not None and abs(bounds[0] - EVENT[0]) <= 3 and abs(bounds[1] - EVENT[1]) <= 3
    return bounds


def _add(win, how: str) -> None:
    if how == "D":
        win._trace_keys.add()
    elif how == "Ctrl+M":
        win._act_add_event.trigger()                           # Analysis ▸ Add event from cursors
    else:
        win._analysis_panel.measure_panel._add_button.click()   # the Measure tab's Add event


def _record(row) -> tuple:
    return (row["bounds_source"], int(row["drawn_start_idx"]), int(row["drawn_end_idx"]),
            row["fit_json"], int(row["start_idx"]), int(row["end_idx"]))


# ================================================================================================
# a caught event: the fit's record, the span as caught, and a clean trace after
# ================================================================================================
@pytest.mark.parametrize("how", WAYS)
def test_an_add_after_a_catch_records_the_fitted_edges_and_clears_the_mask(rig, how):
    win, doc_id = rig
    _catch(win, doc_id)

    _add(win, how)

    df = _table(win, doc_id)
    assert df is not None and len(df) == 1, win.statusBar().currentMessage()
    row = df.iloc[0]
    assert row["bounds_source"] == "start=fit:step;end=fit:step", (how, row["bounds_source"])
    assert not pd.isna(row["fit_json"]), (how, "the fit record was dropped")
    record = json.loads(row["fit_json"])
    assert record["edges"]["start"]["fitted"] is True and record["edges"]["end"]["fitted"] is True
    # the span as drawn is the span C placed (its half-peak edges), not the fitted bounds
    assert [int(row["drawn_start_idx"]), int(row["drawn_end_idx"])] == record["placed"], how
    # "After an add the mask is cleared ... so the next event starts clean"; Measure stays on
    assert win._trace_view.measure_bounds() is None and win._trace_view.measure_empty(), how
    assert win._analysis_panel.measure_panel.is_active()
    assert "Added event 0 to Manual" in win.statusBar().currentMessage()


def test_the_three_ways_store_the_same_row_for_the_same_catch(rig):
    win, doc_id = rig
    stored = {}
    for how in WAYS:
        _catch(win, doc_id)
        _add(win, how)
        df = _table(win, doc_id)
        assert len(df) == 1, (how, win.statusBar().currentMessage())
        stored[how] = _record(df.iloc[0])
        win._act_undo.trigger()                                 # take it back for the next way
        df = _table(win, doc_id)
        assert df is None or len(df) == 0
    assert stored["Ctrl+M"] == stored["D"] and stored["Add event button"] == stored["D"], stored


def test_a_second_ctrl_m_after_an_add_does_not_meet_the_event_just_added(rig):
    win, doc_id = rig
    _catch(win, doc_id)
    win._act_add_event.trigger()
    assert len(_table(win, doc_id)) == 1
    win._act_add_event.trigger()                                # no mask now: nothing to add
    assert len(_table(win, doc_id)) == 1
    said = win.statusBar().currentMessage()
    assert "overlaps" not in said and "no region is selected" in said, said


# ================================================================================================
# an unfitted edge asks for a second press, whichever way the add is asked for
# ================================================================================================
@pytest.mark.parametrize("how", ["Ctrl+M", "Add event button"])
def test_an_unfitted_edge_asks_for_a_second_press_and_is_then_added_as_placed(rig, how):
    win, doc_id = rig
    _point(win, doc_id, 90_000)
    win._trace_keys.place_edge("start", fit=True)             # nothing to fit on baseline
    _point(win, doc_id, 90_300)
    win._trace_keys.place_edge("end", fit=True)
    assert win._analysis_panel.measure_panel._add_button.isEnabled()

    _add(win, how)
    assert _table(win, doc_id) is None                          # the first press only asked
    said = win.statusBar().currentMessage()
    assert "not fitted. Press again to add it as placed" in said and "Press D again" not in said
    assert said.startswith("The start and finish edges are not fitted."), said
    _add(win, how)

    row = _table(win, doc_id).iloc[0]
    assert row["bounds_source"] == "start=placed;end=placed"
    assert win._trace_view.measure_bounds() is None


def test_a_draw_release_with_an_unfitted_edge_says_d_twice_adds_it_as_placed(rig):
    """The release's sentence says what D then does: it asks once more and files the unfitted
    edge as placed (design 3.2: "recorded as placed, never drawn"). It used to say "as drawn"."""
    win, doc_id = rig
    tv = win._trace_view
    win._analysis_panel.measure_panel.set_active(True, seed=False)
    lo, hi = 19_990, 25_000             # the start by the event's step, the finish on baseline
    tv.set_measure_region(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, hi))
    win._trace_keys._on_draw_released(doc_id, 0, lo, hi, 10.0)   # the drag's release
    said = win.statusBar().currentMessage()
    assert said.startswith("The finish edge is not fitted"), said
    assert said.endswith("or press D twice to add it as placed."), said
    win._trace_keys.add()
    assert _table(win, doc_id) is None                          # the first D only asked
    win._trace_keys.add()
    assert _table(win, doc_id).iloc[0]["bounds_source"] == "start=fit:step;end=placed"


# ================================================================================================
# a plain bracket, which no key fitted, is stored as it always was
# ================================================================================================
@pytest.mark.parametrize("nudged", [False, True])
@pytest.mark.parametrize("how", WAYS)
def test_a_plain_bracket_is_still_stored_as_drawn(rig, how, nudged):
    win, doc_id = rig
    tv = win._trace_view
    win._analysis_panel.measure_panel.set_active(True)
    tv.set_measure_region(tv.x_for_index(doc_id, 0, 19_990), tv.x_for_index(doc_id, 0, 20_410))
    lo = 19_990
    if nudged:                  # its start handle then dragged one sample in, after it settled
        lo = 19_991
        tv.set_measure_region(tv.x_for_index(doc_id, 0, lo), tv.x_for_index(doc_id, 0, 20_410))
    _add(win, how)
    row = _table(win, doc_id).iloc[0]
    assert (int(row["start_idx"]), int(row["end_idx"])) == (lo, 20_410)
    assert row["bounds_source"] == "start=drawn;end=drawn"
    assert (int(row["drawn_start_idx"]), int(row["drawn_end_idx"])) == (lo, 20_410)
    assert pd.isna(row["fit_json"])
