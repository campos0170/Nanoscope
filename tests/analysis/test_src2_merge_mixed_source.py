"""A merged row states the source it actually has, and "added wins" has ONE home.

`merge._combined_provenance_cell` used to compute `events.combine_sources` -- which for
a detector fragment stitched to a hand-added one correctly answers ``mixed``, the token
that vocabulary exists for -- and then throw that answer away for ``manual`` whenever the
run's ``added`` column ORed True. The stated reason was to keep
`events.source_disagrees_with_added` quiet, a checker whose test is one token wide
(``added & known & ~manual``). That inverted the dependency: the data was bent to suit
the checker, the "added wins over source" rule got a SECOND home next to
`events.ensure_source_status`, and `events.source_kind` read the merged row as purely
manual -- dropping a detector-found event out of the ``conventional`` count that the
export sidecar's ``by_kind`` (the owner's conventional-vs-ML split) exists to report.

These tests pin the fixed behaviour:

  * a merge across two authors reads ``mixed``, and still reads ``added=True`` -- both
    recorded facts stay true, and neither is rewritten to flatter the other;
  * the rule that fills a BLANK source on an added row still reaches a merged row, from
    its one home (`ensure_source_status`), which is where a merge of unstamped fragments
    hands it a ``pd.NA`` to fill;
  * an input that genuinely contradicts itself is carried, not repaired;
  * nothing measured moves: the 20 frozen columns of the merged table are identical
    whether the provenance columns are present or not.

They FAIL against the pre-fix module (proved by loading a copy of it under the same
name -- never by editing the source back and forth).
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import merge as merge_mod
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    KIND_MIXED,
    REVIEW_COL,
    REVIEW_NONE,
    SCHEMA,
    SOURCE_COL,
    SOURCE_MANUAL,
    SOURCE_MIXED,
    EventSet,
    ensure_source_status,
    source_kind,
)
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.analysis.scoring import ADDED_COLUMN, ensure_added_column

FS = 50_000.0

#: Three fragments of ONE shattered translocation (20-sample gaps, so they group under
#: any millisecond window) plus one isolated event far away that must pass through.
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000)]
_ISOLATED = (30_000, 30_300)
_GAP_S = 1e-3


def _trace(n: int = 50_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, -1000.0) + rng.normal(0.0, 8.0, n)
    for s, e in [*_FRAGMENTS, _ISOLATED]:
        x[s:e] -= 300.0
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def signal() -> np.ndarray:
    return _trace()


@pytest.fixture(scope="module")
def detection(signal):
    """One real Schmitt detection over the shattered event -- the expensive part."""
    df, prov = detect_events(signal, FS, baseline=-1000.0, min_dwell_s=1e-4)
    assert len(df) == 4, f"fixture drifted: {len(df)} events, expected 4"
    return df, prov


def _table(detection, *, source_of_last: object, added_last: bool) -> EventSet:
    """The detection with its LAST shattered fragment re-attributed / marked added.

    The isolated event at 30 000 is left exactly as the detector produced it, so a test
    can also see that a pass-through row is untouched.
    """
    df, prov = detection
    df = df.copy()
    ensure_added_column(df)
    last = df.index[df["start_idx"] == _FRAGMENTS[-1][0]]
    assert len(last) == 1, "fixture drifted: the hand-added fragment is not there"
    ensure_source_status(df, source="detect.schmitt", review=REVIEW_NONE)
    if added_last:
        df.loc[last, ADDED_COLUMN] = True
    src = df[SOURCE_COL].to_numpy(dtype=object)
    src[df.index.get_indexer(last)[0]] = source_of_last
    df[SOURCE_COL] = pd.array(src, dtype="string")
    return EventSet(df, dict(prov))


def _merged_row(out: EventSet) -> pd.Series:
    """The one row assembled from the three fragments."""
    rows = out.df[out.df["start_idx"] < 20_000]
    assert len(rows) == 1, f"the fragments did not collapse to one row ({len(rows)})"
    return rows.iloc[0]


def test_a_merge_across_a_detector_and_a_person_reads_mixed(detection, signal):
    """``mixed`` is the truthful answer and the one `combine_sources` gives; a merged
    row keeps it instead of being rewritten to ``manual`` to keep a checker quiet."""
    es = _table(detection, source_of_last=SOURCE_MANUAL, added_last=True)
    out, n_merged = merge_by_gap(es, signal, FS, max_gap_s=_GAP_S)
    assert n_merged == 2, "the fixture fragments did not merge; the rest is vacuous"

    row = _merged_row(out)
    assert row[SOURCE_COL] == SOURCE_MIXED, (
        "a row stitched from a detector's fragment and a person's named only the "
        "person, which is a provenance claim the detector's two fragments contradict")
    assert bool(row[ADDED_COLUMN]) is True, (
        "``added`` is the pre-existing recorded fact (score_detection reads it as a "
        "false negative); the fix must keep it, not trade one truth for the other")
    assert source_kind(row[SOURCE_COL]) == KIND_MIXED, (
        "source_kind read the merged row as purely manual, which is how a "
        "detector-found event fell out of the sidecar's conventional-vs-ML split")


def test_the_pass_through_row_and_the_review_rule_are_untouched(detection, signal):
    """The isolated event keeps the detector's own stamp, and the merged row stays
    ``unreviewed`` -- this fix is about ``source`` only."""
    es = _table(detection, source_of_last=SOURCE_MANUAL, added_last=True)
    out, _ = merge_by_gap(es, signal, FS, max_gap_s=_GAP_S)
    passed = out.df[out.df["start_idx"] >= 20_000]
    assert list(passed[SOURCE_COL]) == ["detect.schmitt"]
    assert _merged_row(out)[REVIEW_COL] == REVIEW_NONE


def test_added_wins_over_a_blank_source_from_its_one_home_not_from_merge(
        detection, signal):
    """A run of UNSTAMPED fragments combines to ``pd.NA`` -- and the single home of the
    "added wins over source" rule, `events.ensure_source_status`, fills it with
    ``manual`` at the install door. The rule still reaches a merged row; what it fills
    is a blank, not a recorded, truthful ``mixed``."""
    df, prov = detection
    df = df.copy()
    ensure_added_column(df)                       # a project saved before the columns
    df.loc[df["start_idx"] == _FRAGMENTS[-1][0], ADDED_COLUMN] = True

    out, n_merged = merge_by_gap(EventSet(df, dict(prov)), signal, FS, max_gap_s=_GAP_S)
    assert n_merged == 2
    assert SOURCE_COL not in out.df.columns, "the fixture is meant to be un-stamped"

    ensure_source_status(out.df, source="detect.schmitt")   # the one door
    assert _merged_row(out)[SOURCE_COL] == SOURCE_MANUAL


def test_merge_does_not_re_implement_the_added_wins_rule(detection, signal):
    """One home, checked two ways: a run whose fragments ALL say ``detect.schmitt``
    while one is marked ``added`` keeps ``detect.schmitt`` -- the input's own
    contradiction is carried for `source_disagrees_with_added` to REPORT, not silently
    repaired into ``manual`` -- and the merge module no longer names the rule at all.
    """
    es = _table(detection, source_of_last="detect.schmitt", added_last=True)
    out, _ = merge_by_gap(es, signal, FS, max_gap_s=_GAP_S)
    assert _merged_row(out)[SOURCE_COL] == "detect.schmitt", (
        "merge repaired a contradiction between two recorded facts; repairing loses "
        "the true one, which is why the checker reports instead")

    body = inspect.getsource(merge_mod._combined_provenance_cell)
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith(("*", "#"))).split('"""')
    logic = code[-1] if len(code) > 2 else body
    assert "SOURCE_MANUAL" not in logic and "ADDED_COLUMN" not in logic, (
        "the 'added wins over source' rule grew a second home in merge; it belongs "
        "in events.ensure_source_status alone")


def test_nothing_measured_moves_when_the_provenance_columns_are_there(
        detection, signal):
    """The 20 frozen columns of the merged table are byte-identical with and without
    ``source`` / ``review`` / ``added``. Whatever this function decides about
    provenance, it decides about no number."""
    def frozen20(df: pd.DataFrame) -> str:
        frozen = df[list(SCHEMA)]
        h = hashlib.sha256()
        h.update(("|".join(frozen.columns) + "\n").encode())
        h.update((",".join(str(frozen[c].dtype) for c in frozen.columns) + "\n").encode())
        h.update(frozen.to_csv(index=False, float_format="%.17g").encode())
        return h.hexdigest()

    df, prov = detection
    bare, _ = merge_by_gap(EventSet(df.copy(), dict(prov)), signal, FS,
                           max_gap_s=_GAP_S)
    stamped, _ = merge_by_gap(
        _table(detection, source_of_last=SOURCE_MANUAL, added_last=True),
        signal, FS, max_gap_s=_GAP_S)
    assert frozen20(stamped.df) == frozen20(bare.df)
