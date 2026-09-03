"""E1 — "the user needs the option to revert to the model's start/end regions."

THE DETECTOR'S OWN BOUNDS WERE PRESERVED NOWHERE DURABLE. ``start_idx``/``end_idx`` are
edited IN PLACE: the Event Inspector's mask drag writes them straight onto the shared
frame (that sharing is the design — an edit in the browser is an edit in the table), a
merge rebuilds them, and the undo recorder snapshots the frame after the fact. So the
moment a curator drags an edge, the number the detector produced is gone from the frame,
the result store and the project file at once, and there is nothing to revert TO.

Two columns fix that, and the interesting part is that they are NOT in ``SCHEMA``:

* ``SCHEMA`` is ``REQUIRED`` (``EventSet.__post_init__`` raises on a missing column), so a
  detection saved in an existing project would stop loading;
* ``characterize`` ends with ``df[list(SCHEMA)]``, which DROPS everything outside it — and
  the Inspector's edit path goes through ``recharacterize_span``, so the very first mask
  drag would have erased the record it exists to keep.

Declared alongside instead, they are additive in both directions: a frame that has them
keeps them through an edit, and one that does not reads "unknown". Which is the other half
of this file — the migration is HONEST. A pre-change detection is not backfilled from its
current bounds, because those may already have been edited and the resulting claim would be
"the detector produced whatever the last edit left behind", on the one control whose entire
promise is provenance.

The GUI half (the Revert button, its three states and its tooltip) is in
``tests/gui/test_event_inspector.py``.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.characterize import recharacterize_span
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    ORIG_END_COL,
    ORIG_START_COL,
    ORIGINAL_BOUNDS_SCHEMA,
    REQUIRED,
    SCHEMA,
    EventSet,
    empty_df,
    has_original_bounds,
    original_bounds,
    stamp_original_bounds,
)

FS = 50_000.0
N = 40_000
I0 = -900.0
_SPANS = [(5_000, 5_600), (15_000, 15_500), (25_000, 25_800)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 4.0, N)
    for s, e in _SPANS:
        x[s:e] -= 250.0
    return np.ascontiguousarray(x, dtype=np.float64)


@pytest.fixture
def detected():
    """``(signal, df, provenance)`` from the real Schmitt detector, bounds stamped."""
    sig = _signal()
    df, prov = detect_events(sig, FS, baseline=I0, min_dwell_s=1e-4)
    return sig, stamp_original_bounds(df), prov


# ================================================================================================
# the compatibility mechanism — why these two columns are not in SCHEMA
# ================================================================================================

def test_the_pair_is_declared_outside_the_frozen_schema():
    """If either column ever lands in ``SCHEMA`` this whole design is undone: every existing
    project stops loading and every rebuild erases the record."""
    assert ORIG_START_COL not in SCHEMA and ORIG_END_COL not in SCHEMA
    assert ORIG_START_COL not in REQUIRED and ORIG_END_COL not in REQUIRED
    assert set(ORIGINAL_BOUNDS_SCHEMA) == {ORIG_START_COL, ORIG_END_COL}
    # Same (dtype, unit, description) shape as SCHEMA, so one reader serves both.
    for entry in ORIGINAL_BOUNDS_SCHEMA.values():
        assert len(entry) == 3 and all(isinstance(v, str) for v in entry)


def test_an_eventset_without_the_pair_still_constructs(detected):
    """THE MIGRATION, at the door every producer and every restore goes through. A detection
    already in a project file has no original bounds and must load exactly as before."""
    sig, df, prov = detected
    legacy = df.drop(columns=[ORIG_START_COL, ORIG_END_COL])
    es = EventSet(legacy, prov)                    # must not raise
    assert not has_original_bounds(es.df)
    assert all(original_bounds(es.df.iloc[i]) is None for i in range(len(es.df)))


def test_a_stamped_frame_survives_the_rebuild_the_edit_path_runs(detected):
    """The mask drag re-measures through ``recharacterize_span``, whose output is
    SCHEMA-only. The Inspector writes those measured columns BY NAME onto the shared frame,
    so an untouched column survives — which is what makes a revert possible AFTER an edit,
    the only time anybody wants one.

    Pinned here rather than only in the GUI because the property is about the two functions:
    ``recharacterize_span`` never carries the pair, and it must never need to.
    """
    sig, df, prov = detected
    row = 0
    before = original_bounds(df.iloc[row])
    assert before is not None
    new_start, new_end = before[0], before[1] - 100

    rebuilt = recharacterize_span(sig, FS, new_start, new_end,
                                  provenance=prov, prior=df.iloc[row])
    assert ORIG_START_COL not in rebuilt.columns, (
        "recharacterize_span grew a column outside SCHEMA; the frozen 20-column export "
        "contract says it must not")

    # ...and the Inspector's write is by name, which is why the pair survives it.
    for name in ("start_idx", "end_idx", "dwell_s", "depth_pA"):
        df.iat[row, df.columns.get_loc(name)] = rebuilt.iloc[0][name]
    df.iat[row, df.columns.get_loc("start_idx")] = new_start
    df.iat[row, df.columns.get_loc("end_idx")] = new_end

    assert original_bounds(df.iloc[row]) == before, (
        "the edit rewrote the detector's own bounds, so the record now describes the edit")
    assert int(df.iloc[row]["end_idx"]) == new_end


def test_the_pair_round_trips_through_the_project_writer(detected):
    """A project persists the event frame with ``to_parquet``, which carries every column —
    including a NULLABLE Int64 with its ``pd.NA`` intact. Absence has to survive the trip as
    absence, or a reopened project silently claims provenance for a hand-added event."""
    _sig, df, _prov = detected
    df = df.copy()
    df.loc[df.index[1], ORIG_START_COL] = pd.NA    # e.g. a hand-added event

    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=True)
    back = pd.read_parquet(io.BytesIO(buf.getvalue()), engine="pyarrow")

    assert has_original_bounds(back)
    assert original_bounds(back.iloc[0]) == original_bounds(df.iloc[0])
    assert original_bounds(back.iloc[1]) is None


# ================================================================================================
# the writer — once, at detection, never over an edit
# ================================================================================================

def test_stamping_records_exactly_the_bounds_it_was_handed(detected):
    _sig, df, _prov = detected
    for i in range(len(df)):
        assert original_bounds(df.iloc[i]) == (int(df.iloc[i]["start_idx"]),
                                               int(df.iloc[i]["end_idx"]))


def test_stamping_an_already_stamped_frame_changes_nothing(detected):
    """The guard that stops the writer becoming a backfill. Called a second time — by a
    refresh, a re-install, a defensive call added later — it must not overwrite a record
    with post-edit bounds."""
    _sig, df, _prov = detected
    before = [original_bounds(df.iloc[i]) for i in range(len(df))]
    df.iat[0, df.columns.get_loc("end_idx")] = int(df.iloc[0]["end_idx"]) - 200

    again = stamp_original_bounds(df)
    assert [original_bounds(again.iloc[i]) for i in range(len(again))] == before
    # ...and only an explicit re-detection may say otherwise.
    forced = stamp_original_bounds(df, overwrite=True)
    assert original_bounds(forced.iloc[0]) == (int(df.iloc[0]["start_idx"]),
                                               int(df.iloc[0]["end_idx"]))


def test_stamping_does_not_mutate_the_frame_it_was_given(detected):
    """A copy, not an in-place widening: the caller's frame may already be shared with a
    window, and growing a column under it is how two surfaces come to disagree about the
    schema they are both reading."""
    _sig, df, _prov = detected
    plain = df.drop(columns=[ORIG_START_COL, ORIG_END_COL])
    before = list(plain.columns)
    out = stamp_original_bounds(plain)
    assert list(plain.columns) == before
    assert out is not plain and has_original_bounds(out)


def test_stamping_an_empty_table_is_well_formed():
    out = stamp_original_bounds(empty_df())
    assert has_original_bounds(out) and len(out) == 0
    assert str(out[ORIG_START_COL].dtype) == "Int64"


def test_a_frame_with_no_bounds_at_all_is_returned_untouched():
    """Nothing to record is not an error. ``stamp_original_bounds`` is called at detection,
    and a frame with no ``start_idx`` is not a detection."""
    d = pd.DataFrame({"event_id": [0, 1]})
    assert stamp_original_bounds(d) is d


# ================================================================================================
# the reader — one answer to "is Revert available for this row?"
# ================================================================================================

@pytest.mark.parametrize("value", [pd.NA, None, np.nan, float("inf"), 1.5, "x"])
def test_an_unusable_cell_reads_as_unknown_not_as_a_span(value):
    """Every way the record can be absent or untrustworthy gives ``None``, so the GUI has
    exactly one thing to branch on. A caller that gets ``None`` must DISABLE the revert —
    falling back to the current bounds would be a no-op dressed up as a restoration."""
    row = {ORIG_START_COL: value, ORIG_END_COL: 20}
    assert original_bounds(row) is None
    assert original_bounds({ORIG_START_COL: 10, ORIG_END_COL: value}) is None


def test_a_half_stamped_frame_states_no_span():
    """One column without the other is not half a record; it is no record."""
    d = pd.DataFrame({"start_idx": [10], "end_idx": [20]})
    d[ORIG_START_COL] = pd.array([10], dtype="Int64")
    assert not has_original_bounds(d)
    assert original_bounds(d.iloc[0]) is None


@pytest.mark.parametrize("lo,hi", [(20, 20), (30, 20)])
def test_an_empty_or_reversed_recorded_span_is_refused(lo, hi):
    """A span the detector cannot have produced. Refused rather than repaired, the way
    ``analysed_span`` refuses a stored pair it cannot trust — a revert onto it would collapse
    the event to nothing."""
    assert original_bounds({ORIG_START_COL: lo, ORIG_END_COL: hi}) is None


def test_the_reader_takes_a_series_a_mapping_or_a_row(detected):
    _sig, df, _prov = detected
    want = (int(df.iloc[0]["start_idx"]), int(df.iloc[0]["end_idx"]))
    assert original_bounds(df.iloc[0]) == want
    assert original_bounds(dict(df.iloc[0])) == want
    assert original_bounds({}) is None
    assert original_bounds(object()) is None
