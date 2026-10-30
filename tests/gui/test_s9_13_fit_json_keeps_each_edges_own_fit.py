"""S9-13 — A row's ``fit_json`` holds the fit of every edge its ``bounds_source`` calls fitted.

MEASURED DEFECT (sweep 9, finding 13; a clean 50 kHz blockade at samples 20 000 - 20 400).
``TraceKeyController`` kept ONE ``_fit_json``. Each S or F replaced it with the record of a fit
asked for that edge only, in which the other edge reads "not requested", and every settle of the
mask the controller had not asked for itself cleared it. So the documented "S then F builds one"
mask was stored with ``bounds_source`` ``start=fit:step;end=fit:step`` and a ``fit_json`` whose
start entry said ``{fitted: false, reason: 'not requested', z: 0, step: 0}``; and C, a drag of the
start, then D stored ``start=drawn;end=fit:step`` with ``fit_json`` NA, the end's fit record lost.
A settle that moved nothing (a rebuild putting the caught mask back) dropped it the same way, and
so did Shift+S, so a caught end lost its record to an exact start. Design 5.5: ``fit_json``
carries the fit's parameters and "per-edge z and step", so a fit can be redone.

THE FIX. Each edge keeps the record of the fit that put it there (``_Edge.fit``). A hand-moved
edge loses only its own (it becomes drawn) and the other edge keeps its record; a settle that
moves no edge changes nothing. At the add, one fit that set both edges is stored as it came; edges
set by two fits, or one moved since, are merged into the latest fit's record, each edge's entry,
window (``W``), rise and placed position from its own fit, and an edge no fit set marked "not
requested" at its position. ``fit_json`` never says an edge was not fitted that ``bounds_source``
calls ``fit:step``.

These tests drive a real MainWindow on a synthetic recording: the C, S, F, Shift+S and D keys'
slots with the pointer stood in (as ``test_trace_keys`` does), the mask moved the way a handle
drag leaves it, and a view-mode change (a rebuild).
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
from PySide6 import QtWidgets

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 120_000
I0 = -1000.0
DEPTH = 300.0
EVENT = (20_000, 20_400)            # a clean blockade no detector was asked about


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
    meta = RecordingMeta(path=tmp_path / "s9_13.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    return win, doc_id


def _point(win, doc_id, sample, px=10.0):
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _added_row(win, doc_id):
    win._trace_keys.add()                                      # D
    res = win._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    assert res is not None and len(res.payload.df) == 1, win.statusBar().currentMessage()
    return res.payload.df.iloc[0]


def _sources(row) -> dict[str, str]:
    return dict(part.split("=", 1) for part in str(row["bounds_source"]).split(";"))


def _assert_honest(row) -> dict:
    """Every edge ``bounds_source`` calls fitted has its fit in ``fit_json``: fitted, z, step."""
    record = json.loads(row["fit_json"])
    for edge, source in _sources(row).items():
        entry = record["edges"][edge]
        if source == "fit:step":
            assert entry["fitted"] is True and entry["z"] > 0 and entry["step"] != 0, (
                f"bounds_source says {edge}=fit:step, fit_json says {entry}")
        else:
            assert entry["fitted"] is False, (edge, source, entry)
    return record


def _catch(win, doc_id) -> tuple[int, int]:
    _point(win, doc_id, 20_200)
    win._trace_keys.catch()                                    # C
    bounds = win._trace_view.measure_bounds()
    assert bounds is not None
    return bounds


def test_s_then_f_stores_the_fit_of_both_edges(rig):
    win, doc_id = rig
    _point(win, doc_id, EVENT[0] + 12)
    win._trace_keys.place_edge("start", fit=True)              # S
    _point(win, doc_id, EVENT[1] - 9)
    win._trace_keys.place_edge("end", fit=True)                # F

    row = _added_row(win, doc_id)

    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    record = _assert_honest(row)
    assert record["edges"]["start"]["pos"] == int(row["start_idx"])
    assert record["edges"]["end"]["pos"] == int(row["end_idx"])
    assert set(record["W"]) == {"start", "end"} and set(record["rise"]) == {"start", "end"}
    assert record["edges_requested"] == ["start", "end"]
    assert record["placed"] == [EVENT[0] + 12, EVENT[1] - 9]  # where each edge was pointed at
    assert record["method"] == "step" and record["tau"] >= 1.0


def test_a_hand_moved_start_keeps_the_fit_of_the_end_it_still_claims(rig):
    win, doc_id = rig
    tv = win._trace_view
    lo, hi = _catch(win, doc_id)
    # the start handle dragged 50 samples out (a drag release ends in exactly this placement)
    tv.set_measure_region(tv.x_for_index(doc_id, 0, lo - 50), tv.x_for_index(doc_id, 0, hi))

    row = _added_row(win, doc_id)

    assert row["bounds_source"] == "start=drawn;end=fit:step"
    assert not pd.isna(row["fit_json"]), "the end's fit record was dropped with the start's"
    record = _assert_honest(row)
    assert record["edges"]["end"]["pos"] == hi
    assert record["edges"]["start"] == {"fitted": False, "pos": lo - 50, "reason": "not requested",
                                        "step": 0.0, "z": 0.0}
    assert set(record["W"]) == {"end"} and record["edges_requested"] == ["end"]


def test_an_exact_start_keeps_the_fit_of_the_caught_end(rig):
    win, doc_id = rig
    _, hi = _catch(win, doc_id)
    _point(win, doc_id, EVENT[0] - 20)
    win._trace_keys.place_edge("start", fit=False)             # Shift+S

    row = _added_row(win, doc_id)

    assert row["bounds_source"] == "start=drawn;end=fit:step"
    assert not pd.isna(row["fit_json"])
    assert _assert_honest(row)["edges"]["end"]["pos"] == hi


def test_a_rebuild_that_moves_nothing_keeps_the_catch_as_it_came(rig):
    win, doc_id = rig
    tv = win._trace_view
    bounds = _catch(win, doc_id)
    caught = win._trace_keys._fit_json
    win._session.set_view_mode(ViewMode.STACKED)              # the trace is rebuilt...
    QtWidgets.QApplication.processEvents()
    assert tv.measure_bounds() == bounds                       # ...and the mask put back

    row = _added_row(win, doc_id)

    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    assert not pd.isna(row["fit_json"]), "the rebuild's settle dropped the catch's fit record"
    assert row["fit_json"] == caught                            # the catch's own record, unmerged
    _assert_honest(row)


def test_a_refit_end_after_a_catch_merges_each_edge_from_its_own_fit(rig):
    win, doc_id = rig
    _catch(win, doc_id)
    _point(win, doc_id, EVENT[1] - 9)
    win._trace_keys.place_edge("end", fit=True)                # F again, on the caught mask

    row = _added_row(win, doc_id)

    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    record = _assert_honest(row)
    assert record["edges_requested"] == ["start", "end"] and set(record["W"]) == {"start", "end"}
