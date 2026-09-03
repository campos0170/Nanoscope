"""WA-UNDO-1 — A HAND-ADDED EVENT MADE EVERY LATER CHECKPOINT SEE A MUTATION THAT NEVER HAPPENED.

``EventHistory`` keeps a pristine deep copy of every live result and COMPARES, so a
mutation does not have to announce itself. The comparison bottoms out in
:func:`nanoscope.core.history._column_equal`, whose first documented promise is
"NaN == NaN and ``pd.NA`` == ``pd.NA``".

It broke that promise for exactly one dtype, and that dtype is the one a hand-added
event creates. Measured on the -1000 mV apo-ferritin recording (115 detected events),
through a real ``MainWindow``:

* "Mark selection as missed event" appends a row with no detector bounds, so
  ``orig_start_idx`` / ``orig_end_idx`` are ``pd.NA`` on it. Their dtype is the pandas
  MASKED ``Int64``.
* ``pd.Int64Dtype().kind`` is ``'i'``, so the comparison took the integer fast path —
  while ``Series.to_numpy()`` on a masked column with a null hands back ``float64`` with
  ``NaN`` in that cell. ``NaN == NaN`` under a raw ``==`` is False.
* So ``_frames_equal(pristine, live)`` returned **False on every subsequent checkpoint of
  a table nobody had touched**: ``checkpoint()`` with no edit at all returned True and
  pushed a phantom ``"Edit events"`` step whose "pre-edit" snapshot ALREADY held the added
  row.
* ``EventHistory._move`` checkpoints before it moves (so an unannounced edit is not jumped
  past), which minted a fresh phantom on every press. Ctrl+Z therefore restored the
  post-add table, forever: 116 rows in, 116 rows out, with the real ``"Add event"`` entry
  one place further down and unreachable. The Edit menu said "Undo Add event" the whole
  time.

The invariant these tests hold: **a frame compared against its own snapshot is equal, for
every dtype the event schema can carry** — and therefore an edit that changes the row
count is exactly as undoable as one that flips a boolean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import EventSet
from nanoscope.core import history as H
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.history import EventHistory
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 50_000.0
MODULE = "detect.schmitt"


# ================================================================================================
# fixtures — the smallest frame that carries the masked column a hand-add creates
# ================================================================================================

def _frame(n: int = 3, added_row: bool = False) -> pd.DataFrame:
    """``n`` detector rows, optionally plus one hand-added row with NO detector bounds.

    ``orig_start_idx`` / ``orig_end_idx`` are ``Int64`` — the pandas MASKED integer — for
    the same reason the live frame carries them that way: a detected row knows the bounds
    the detector chose, a hand-added row has none, and ``pd.NA`` is the only honest value.
    """
    from nanoscope.analysis.events import SCHEMA

    idx = np.arange(n, dtype=np.int64)
    cols: dict[str, object] = {}
    for name, (dtype, _u, _d) in SCHEMA.items():
        if name == "event_id":
            cols[name] = idx
        elif name == "start_idx":
            cols[name] = idx * 1000
        elif name == "end_idx":
            cols[name] = idx * 1000 + 200
        elif dtype == "string":
            cols[name] = ["r" if name == "recording" else "p" if name == "pore_id" else ""] * n
        elif dtype == "bool":
            cols[name] = np.full(n, name == "accepted", dtype=bool)
        elif dtype == "int64":
            cols[name] = np.ones(n, dtype=np.int64)
        else:
            cols[name] = np.full(n, 300.0)
    df = pd.DataFrame(cols)
    df["added"] = np.zeros(n, dtype=bool)
    df["orig_start_idx"] = pd.array(idx * 1000, dtype="Int64")
    df["orig_end_idx"] = pd.array(idx * 1000 + 200, dtype="Int64")
    if added_row:
        row = dict(df.iloc[-1])
        row.update(event_id=n, start_idx=90_000, end_idx=90_300, added=True,
                   orig_start_idx=pd.NA, orig_end_idx=pd.NA)
        df = pd.concat([df, pd.DataFrame([row]).reindex(columns=df.columns)],
                       ignore_index=True)
        # The concat widens the integer/bool columns to object/float; restore the
        # schema's own dtypes, exactly as ``add_current_selection_as_event`` does.
        df = df.astype({"orig_start_idx": "Int64", "orig_end_idx": "Int64",
                        "accepted": bool, "added": bool, "filter_limited": bool,
                        "event_id": "int64", "start_idx": "int64",
                        "end_idx": "int64", "n_levels": "int64"})
    return df


def _result(doc_id: str, df: pd.DataFrame) -> AnalysisResult:
    return AnalysisResult(
        module_id=MODULE, module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4}, calibration_fingerprint="cal",
        payload=EventSet(df, {"span_start": 0}))


@pytest.fixture
def session(tmp_path):
    sig = np.zeros(4096, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa.dat", format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    s = Session()
    doc_id = s.add_document(TraceDocument(meta=meta, channels=(sig,)))
    return s, doc_id


# ================================================================================================
# the comparison itself
# ================================================================================================

def test_a_masked_integer_column_compares_equal_to_its_own_copy():
    """The one-line statement of the bug: ``pd.NA`` is not a change."""
    col = pd.Series(pd.array([10, 20, pd.NA], dtype="Int64"))
    assert bool(H._column_equal(col, col.copy(deep=True)).all()), (
        "a masked Int64 column reported a difference against its own deep copy — "
        "to_numpy() materialised the null as NaN and the integer fast path compared "
        "it with a raw ==")


def test_a_frame_with_a_hand_added_row_equals_its_own_deep_copy():
    """``_frames_equal`` is what ``checkpoint`` asks; it must answer True here."""
    df = _frame(added_row=True)
    assert H._frames_equal(df, df.copy(deep=True))


def test_a_masked_null_is_still_a_difference_against_a_real_value():
    """The fix must not make ``pd.NA`` equal to everything — only to ``pd.NA``."""
    a = pd.Series(pd.array([10, 20, pd.NA], dtype="Int64"))
    b = pd.Series(pd.array([10, 20, 30], dtype="Int64"))
    eq = H._column_equal(a, b)
    assert list(eq) == [True, True, False]
    assert not H._frames_equal(_frame(added_row=True), _frame(added_row=False))


def test_an_ordinary_float_nan_is_still_equal_to_itself():
    """The promise this function already kept, kept."""
    col = pd.Series([1.0, np.nan, 3.0])
    assert bool(H._column_equal(col, col.copy(deep=True)).all())


# ================================================================================================
# the consequence: the stack
# ================================================================================================

def test_a_checkpoint_after_a_hand_add_pushes_nothing_when_nothing_changed(session):
    """The phantom, at its source. Two redundant checkpoints must push zero steps.

    ``MainWindow`` wires ``checkpoint`` to four redundant signals precisely because
    "calling it when nothing changed is free". A masked null made it not free.
    """
    s, doc_id = session
    h = EventHistory(s)
    s.add_result(_result(doc_id, _frame()))
    key = (doc_id, 0, MODULE)
    h.checkpoint(key)                                   # adopt the detection silently
    assert h.depth() == 0

    s.add_result(_result(doc_id, _frame(added_row=True)))
    assert h.checkpoint(key) is True                    # the real step: 3 -> 4 rows
    assert h.depth() == 1

    assert h.checkpoint(key) is False, "a checkpoint invented a step for no edit"
    assert h.checkpoint(key) is False
    assert h.depth() == 1


def test_undo_after_a_hand_add_restores_the_table_without_the_added_row(session):
    """116 rows in, 115 rows out — the headline, in miniature."""
    s, doc_id = session
    h = EventHistory(s)
    key = (doc_id, 0, MODULE)
    s.add_result(_result(doc_id, _frame()))
    h.checkpoint(key)
    s.add_result(_result(doc_id, _frame(added_row=True)))
    h.checkpoint(key)

    assert len(s.results.get(doc_id, 0, MODULE).payload.df) == 4
    label = h.undo(key)
    assert label == "Add event", f"the step was named {label!r}"
    df = s.results.get(doc_id, 0, MODULE).payload.df
    assert len(df) == 3, "undo left the hand-added row in the table"
    assert not bool(df["added"].any())


def test_redo_puts_the_added_row_back(session):
    """Redo was reading "nothing to redo" because the phantom consumed the press."""
    s, doc_id = session
    h = EventHistory(s)
    key = (doc_id, 0, MODULE)
    s.add_result(_result(doc_id, _frame()))
    h.checkpoint(key)
    s.add_result(_result(doc_id, _frame(added_row=True)))
    h.checkpoint(key)
    h.undo(key)
    assert h.can_redo(key), "Redo said there was nothing to redo"
    assert h.redo(key) == "Add event"
    assert len(s.results.get(doc_id, 0, MODULE).payload.df) == 4


def test_undo_after_a_merge_that_dropped_rows_restores_every_row(session):
    """The other row-count-changing edit. 4 -> 2 must undo to 4."""
    s, doc_id = session
    h = EventHistory(s)
    key = (doc_id, 0, MODULE)
    s.add_result(_result(doc_id, _frame(added_row=True)))
    h.checkpoint(key)
    merged = _frame(added_row=True).iloc[:2].reset_index(drop=True)
    s.add_result(_result(doc_id, merged))
    assert h.checkpoint(key) is True
    assert len(s.results.get(doc_id, 0, MODULE).payload.df) == 2

    assert h.undo(key) != ""
    assert len(s.results.get(doc_id, 0, MODULE).payload.df) == 4, (
        "undo of a merge left the merged table on screen")


def test_a_second_undo_walks_further_back_instead_of_re_undoing_a_phantom(session):
    """Two real edits, two presses, two steps back — not one phantom twice."""
    s, doc_id = session
    h = EventHistory(s)
    key = (doc_id, 0, MODULE)
    s.add_result(_result(doc_id, _frame()))
    h.checkpoint(key)
    s.add_result(_result(doc_id, _frame(added_row=True)))
    h.checkpoint(key)
    curated = _frame(added_row=True)
    curated.loc[0, "accepted"] = False
    s.add_result(_result(doc_id, curated))
    h.checkpoint(key)

    h.undo(key)
    df = s.results.get(doc_id, 0, MODULE).payload.df
    assert len(df) == 4 and bool(df["accepted"].all()), "first undo missed the rejection"
    h.undo(key)
    assert len(s.results.get(doc_id, 0, MODULE).payload.df) == 3, (
        "second undo did not reach the add")
