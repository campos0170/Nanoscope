"""A merged event says how each of its edges was made: the start as the first fragment made it, the
end as the fragment that ends last made it.

MEASURED DEFECT (found beside the merge's notes, 2026-10-29; a synthetic recording with one event
split into four fragments 20 samples apart). `merge._merged_row` carried the edge record
(``bounds_source``, ``drawn_start_idx`` / ``drawn_end_idx``, ``fit_json``) as the run's first value,
like any other extra column. The first fragment was caught (start fitted, end placed at 10,200 where
the fit gave up) and the last had its end fitted at 11,000, yet Merge all, Merge events in selection
and the Event inspector's Merge with next each made one 10,000-11,000 event that said
``start=fit:step;end=placed``, with ``drawn_end_idx`` 10,200, inside the span, and a ``fit_json``
whose end entry was the first fragment's unfitted end at 10,200. The fit of the real end was gone.
Cursor design 5.5: "Merge takes the start side from the first fragment and the end side from the
last".

THE FIX. ``manual_events.merged_edge_cells`` builds the four cells side by side: the start part of
``bounds_source``, ``drawn_start_idx`` and the start edge's fit from the first fragment; the end part,
``drawn_end_idx`` and the end edge's fit from the fragment that ends last (not the last to start when
a short piece is nested in a longer one). ``fit_json`` is joined as the trace keys join two fits
(``manual_events.join_edge_fits``): each edge's entry, window, rise and placed position from its own
record. A side whose fragment recorded nothing, a detected row, records nothing: its part of
``bounds_source`` is left out, its drawn bound is ``<NA>`` and its ``fit_json`` entry reads "not
requested".

The first test drives each merge through its own control in a real MainWindow, then Edit ▸ Undo; the
second merges hand-made pieces onto one the detector found; the last pins the nested case on the
pure function.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import manual_events as me
from nanoscope.analysis.baseline import DEFAULT_BASELINE
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.edge_fit import REASON_NOT_REQUESTED, EdgeFit, EdgeResult
from nanoscope.analysis.events import EventSet, ensure_source_status
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.core.analysis import AnalysisResult

from .test_manual_event_door import (  # noqa: F401  (``win`` is a fixture)
    FS, I0, _doc, _signal, _table, win)
from .test_merge_joins_notes_and_origins import (  # noqa: F401  (``modals`` is a fixture)
    SCHMITT, WINDOW_S, _FAR, _FRAGMENTS, _merge_all, _merge_selection, _merge_with_next, _merged,
    modals)

_COLS = ["start_idx", "end_idx", "bounds_source", "drawn_start_idx", "drawn_end_idx", "fit_json"]


def _record(span, placed, *, start=None, end=None) -> str:
    """``fit_json`` as the step fit writes it (`EdgeFit.to_json`) for a fit asked for the edges
    given, each ``(fitted, z, step, reason)``; an edge not given reads "not requested"."""
    results, widths, rises = {}, {}, {}
    for name, pos, got in (("start", span[0], start), ("end", span[1], end)):
        if got is None:
            results[name] = EdgeResult(int(pos), False, reason=REASON_NOT_REQUESTED)
            continue
        fitted, z, step, reason = got
        results[name] = EdgeResult(int(pos), fitted, z, step, reason)
        widths[name] = 24 if name == "start" else 20
        rises[name] = {"samples": 3.0, "s": 3.0 / FS, "source": "estimate"}
    params = {"method": "step", "version": 1, "fs": FS, "noise": 8.0, "tau": 1.0,
              "placed": [int(p) for p in placed], "W": widths, "rise": rises,
              "edges_requested": [n for n, g in (("start", start), ("end", end)) if g is not None]}
    return EdgeFit(results["start"], results["end"], params).to_json()


#: How each fragment's edges were made, in time order: ``(bounds_source, drawn bounds, fit_json)``.
#: The first was caught (C): its start fitted, its end placed where the fit gave up, 20 samples
#: from the next fragment. The second is a plain bracket, the third has no record, and the fourth
#: had its end fitted on its own (F) after its start was set exactly.
_EDGES = [
    ("start=fit:step;end=placed", (10_004, 10_200),
     _record(_FRAGMENTS[0], (10_004, 10_200), start=(True, 11.5, 298.0, ""),
             end=(False, 4.1, 120.0, "the outer side is not at baseline"))),
    ("start=drawn;end=drawn", _FRAGMENTS[1], None),
    (None, (None, None), None),
    ("start=drawn;end=fit:step", (10_720, 10_993),
     _record(_FRAGMENTS[3], (10_720, 10_993), end=(True, 9.8, 301.0, ""))),
]


def _hand_rows(df, prov, signal, spans_edges):
    """``df`` with a hand-made row per ``(span, (bounds_source, drawn, fit_json))``."""
    recording, pore_id, n0 = str(df["recording"].iloc[0]), str(df["pore_id"].iloc[0]), len(df)
    for k, ((s, e), (source, drawn, fit)) in enumerate(spans_edges):
        row = me.measure_hand_row(
            signal, FS, s, e, provenance=prov, units="pA", recording=recording, pore_id=pore_id,
            event_id=n0 + k, edge=me.EdgeRecord(bounds_source=source, drawn_start=drawn[0],
                                                drawn_end=drawn[1], fit_json=fit))
        df = me.insert_hand_row(df, row)
    return df


def _open(win, tmp_path, *, detected=(), hand=None):
    """The recording on screen with its Schmitt table: the far event and the spans in ``detected``
    as the detector found them, and the hand-made fragments ``hand`` (default: all four)."""
    signal = _signal(_FRAGMENTS + [_FAR])
    doc_id = win._session.add_document(_doc(tmp_path, signal, name="edges.dat"))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(list(detected) + [_FAR], seed=1), FS, baseline=I0,
                             min_dwell_s=1e-4)
    assert len(df) == len(detected) + 1, "precondition: the detector found what it was given"
    ensure_source_status(df, source=SCHMITT)          # as the install door stamps a detection
    df = _hand_rows(df, prov, signal, hand if hand is not None else zip(_FRAGMENTS, _EDGES))
    win._session.add_result(AnalysisResult(
        module_id=SCHMITT, module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    return doc_id


def _sources(row) -> dict[str, str]:
    return dict(part.split("=", 1) for part in str(row["bounds_source"]).split(";"))


def _assert_honest(row) -> dict:
    """Every edge ``bounds_source`` names is described in ``fit_json`` at the row's own bound,
    fitted exactly when it says ``fit:step`` (the S9-13 check)."""
    record = json.loads(row["fit_json"])
    bounds = {"start": int(row["start_idx"]), "end": int(row["end_idx"])}
    for edge, entry in record["edges"].items():
        assert entry["pos"] == bounds[edge], (edge, entry, bounds)
    for edge, source in _sources(row).items():
        assert record["edges"][edge]["fitted"] is (source == "fit:step"), (edge, source, record)
    return record


# ------------------------------------------------------------------------------------------------
# the three merges, each through its own control
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("merge", [_merge_all, _merge_selection, _merge_with_next],
                         ids=["merge_all", "merge_events_in_selection", "merge_with_next"])
def test_a_merge_takes_each_edges_record_from_its_own_fragment_and_undo_brings_each_back(
        win, qtbot, tmp_path, modals, merge):
    doc_id = _open(win, tmp_path)
    before = _table(win, doc_id, SCHMITT)[_COLS].copy()

    merge(win, qtbot)

    row = _merged(_table(win, doc_id, SCHMITT))
    assert (int(row["start_idx"]), int(row["end_idx"])) == (_FRAGMENTS[0][0], _FRAGMENTS[-1][1])
    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    assert (int(row["drawn_start_idx"]), int(row["drawn_end_idx"])) == (10_004, 10_993)
    record = _assert_honest(row)
    first, last = json.loads(_EDGES[0][2]), json.loads(_EDGES[3][2])
    assert record["edges"] == {"start": first["edges"]["start"], "end": last["edges"]["end"]}
    assert record["W"] == {"start": 24, "end": 20}
    assert record["rise"] == {"start": first["rise"]["start"], "end": last["rise"]["end"]}
    assert record["placed"] == [10_004, 10_993] and record["edges_requested"] == ["start", "end"]
    assert record["method"] == "step"

    win._act_undo.trigger()
    QtWidgets.QApplication.processEvents()

    restored = _table(win, doc_id, SCHMITT)[_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(restored, before.reset_index(drop=True))
    assert modals == [], modals


def test_an_end_the_detector_made_records_nothing_and_the_start_keeps_its_fit(
        win, qtbot, tmp_path, modals):
    """The detector found the event's last part; the first three were drawn by hand beside it.
    The merged start is the caught one, and the end is the detector's, which no record describes."""
    doc_id = _open(win, tmp_path, detected=[_FRAGMENTS[3]],
                   hand=zip(_FRAGMENTS[:3], _EDGES[:3]))
    df = _table(win, doc_id, SCHMITT)
    end = int(df.loc[(df["start_idx"] > _FRAGMENTS[2][1]) & (df["start_idx"] < _FAR[0]),
                     "end_idx"].iloc[0])

    _merge_all(win, qtbot)

    row = _merged(_table(win, doc_id, SCHMITT))
    assert row["source"] == "mixed" and int(row["end_idx"]) == end
    assert row["bounds_source"] == "start=fit:step"
    assert int(row["drawn_start_idx"]) == 10_004 and pd.isna(row["drawn_end_idx"])
    record = _assert_honest(row)
    assert record["edges"]["end"] == {"pos": end, "fitted": False, "z": 0.0, "step": 0.0,
                                      "reason": REASON_NOT_REQUESTED}
    assert record["W"] == {"start": 24} and record["edges_requested"] == ["start"]
    assert record["placed"] == [10_004, end]
    assert modals == [], modals


# ------------------------------------------------------------------------------------------------
# the rule, on the pure function
# ------------------------------------------------------------------------------------------------
def _manual_table(rows):
    """A Manual table on the synthetic recording holding ``rows``: ``(span, edge cells)``."""
    sig = _signal(_FRAGMENTS + [_FAR])
    contract = me.build_contract(sig, FS, "pA", config=DEFAULT_BASELINE, recording="edges.dat",
                                 doc_id="d", channel=0)
    df = me.empty_manual_events(contract).df
    for k, (span, (source, drawn, fit)) in enumerate(rows):
        row = me.measure_hand_row(sig, FS, *span, provenance=contract.provenance, units="pA",
                                  recording="edges.dat", pore_id="p", event_id=k,
                                  edge=me.EdgeRecord(source, drawn[0], drawn[1], fit))
        df = me.insert_hand_row(df, row)
    return EventSet(df, contract.events_provenance()), sig


def test_the_end_comes_from_the_fragment_that_ends_last_not_the_last_to_start():
    """A short piece nested inside the second fragment starts last; the merged end is still the
    second fragment's, so its record is the one the merged row keeps for the end."""
    a, b, nested = (10_000, 10_500), (10_520, 11_000), (10_600, 10_700)
    b_fit = _record(b, (10_515, 11_006), start=(False, 2.0, 40.0, "the step is too small"),
                    end=(True, 10.2, 305.0, ""))
    a_fit = _record(a, (10_004, 10_500), start=(True, 11.5, 298.0, ""))
    events, sig = _manual_table([
        (a, ("start=fit:step;end=drawn", (10_004, 10_500), a_fit)),
        (b, ("start=placed;end=fit:step", (10_515, 11_006), b_fit)),
        (nested, ("start=drawn;end=drawn", nested, None)),
    ])

    out, n_merged = merge_by_gap(events, sig, FS, max_gap_s=WINDOW_S)

    assert n_merged == 2
    row = _merged(out.df)
    assert row["bounds_source"] == "start=fit:step;end=fit:step"
    assert (int(row["drawn_start_idx"]), int(row["drawn_end_idx"])) == (10_004, 11_006)
    record = _assert_honest(row)
    assert record["edges"]["end"] == json.loads(b_fit)["edges"]["end"]
    assert record["placed"] == [10_004, 11_006]
