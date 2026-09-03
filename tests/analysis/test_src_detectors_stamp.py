"""Every path that PRODUCES or REBUILDS an events frame carries ``source``/``review``.

`nanoscope.analysis.events` declares the pair OUTSIDE ``SCHEMA`` on purpose -- an old
project without the columns still loads, because ``EventSet.__post_init__`` raises only
on a missing ``REQUIRED`` column. The price of that promise is that
:func:`~nanoscope.analysis.characterize.characterize` ends with ``df[list(SCHEMA)]`` and
DROPS them, so every producer and every rebuild has to apply them deliberately. This file
is the census of those paths:

  * the two threshold detectors (`modules.DetectSchmitt`, `modules.DetectCusum`) and the
    ML detector (`ml_modules.MLDetect`) stamp their own ``module_id`` and ``unreviewed``;
  * `merge` COMBINES them across a collapsed run rather than taking the first fragment's;
  * `remeasure` carries them through a re-measure unchanged, and does not discard a
    review made while the job was running;
  * `modules.reconcile_result` -- the one door for a changed frame -- materialises them
    on a table that predates the columns, without claiming anybody reviewed anything.

AND NOTHING MEASURED MOVES. The last test hashes the frozen 20 columns of a detection
with the pair present against the same detection with the pair dropped; on the owner's
15 s -1000 mV recording the same comparison holds for real data (Schmitt 115 events,
sha256 a880ca47…28dc; CUSUM 213 events, 063d6b25…d64c -- identical before and after).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import ml_modules
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.characterize import recharacterize_span
from nanoscope.analysis.events import (
    KIND_MIXED,
    REVIEW_COL,
    REVIEW_DONE,
    REVIEW_NONE,
    SCHEMA,
    SOURCE_COL,
    SOURCE_MANUAL,
    SOURCE_MIXED,
    EventSet,
    has_source_status,
    source_kind,
)
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt, reconcile_result
from nanoscope.analysis.remeasure import (
    curation_snapshot,
    merge_late_curation,
    remeasure_events,
)
from nanoscope.analysis.scoring import ADDED_COLUMN
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import CalibratedChannel
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 50_000.0

#: Five square blockades on a flat noisy open pore, plus a close PAIR at the end whose
#: 200-sample gap (4 ms) is what the merge tests collapse. float32 at 50 kHz, which is
#: the recording shape this app is written for.
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350), (85_550, 85_800)]


def _trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return x.astype(np.float32)


def _context(data: np.ndarray):
    """A minimal `SimpleAnalysisContext` around ``data``, built here rather than through
    the function-scoped ``context_factory`` fixture so a module-scoped detection can use
    it."""
    info = ChannelInfo(index=0, name="synthetic", raw_units="pA",
                       kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=np.ascontiguousarray(data, dtype=np.float32),
                           units="pA", sample_rate_hz=FS, t0_s=0.0)
    meta = RecordingMeta(path=Path("synthetic.dat"), format_name="synthetic",
                         n_samples=int(ch.data.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,), t0_s=0.0)
    return SimpleAnalysisContext(channel=ch, meta=meta, params={})


@pytest.fixture(scope="module")
def signal() -> np.ndarray:
    return _trace()


@pytest.fixture(scope="module")
def schmitt(signal):
    """One real Schmitt detection, shared: the detector is the expensive part here.

    Builds its own context rather than taking the ``context_factory`` fixture, which is
    function-scoped -- and a per-test detection would pay for the detector fourteen times
    to assert things about one table.
    """
    return DetectSchmitt().run(_context(signal))


def _frozen20_sha256(df: pd.DataFrame) -> str:
    """The hash the invariant is stated in: the 20 frozen columns, dtypes included.

    Deliberately NOT ``pd.testing.assert_frame_equal`` -- the claim is that the exported
    bytes are identical, and a hash of the CSV body at full float precision is what an
    exported file and a figure both come from.
    """
    frozen = df[list(SCHEMA)]
    h = hashlib.sha256()
    h.update(("|".join(frozen.columns) + "\n").encode())
    h.update((",".join(str(frozen[c].dtype) for c in frozen.columns) + "\n").encode())
    h.update(frozen.to_csv(index=False, float_format="%.17g").encode())
    return h.hexdigest()


# ===========================================================================
# the producers
# ===========================================================================
def test_a_schmitt_detection_stamps_every_row_with_its_own_module_id(schmitt):
    """The audit's finding was that WHICH detector found a row is knowable only from the
    enclosing result's ``module_id`` -- per TABLE, not per row -- so a table that ever
    mixes detector output with hand-added rows cannot say which is which. After the
    stamp the row itself says, verbatim and with no second vocabulary to keep in step.
    """
    df = schmitt.payload.df
    assert len(df) > 0, "the fixture trace produced no events; the rest proves nothing"
    assert has_source_status(df)
    assert set(df[SOURCE_COL].unique()) == {DetectSchmitt.id}
    assert source_kind(df[SOURCE_COL].iloc[0]) == "conventional"
    assert str(df[SOURCE_COL].dtype) == "string"


def test_a_schmitt_detection_calls_its_rows_unreviewed_because_they_are_candidates(schmitt):
    """Notes §7: "automatic detection should generate candidate events; user curation
    should determine the final dataset." The half that was missing is exactly this: a
    detection arrives with every row ``accepted``, and before this column there was
    nothing to distinguish "I looked at this and kept it" from "nobody has looked".
    ``accepted`` is untouched -- the detection still arrives accepted -- so no export,
    figure or fit sees a different set of rows.
    """
    df = schmitt.payload.df
    assert set(df[REVIEW_COL].unique()) == {REVIEW_NONE}
    assert bool(df["accepted"].to_numpy(dtype=bool).all()), (
        "the stamp changed which rows are in the final dataset; it must only describe")


def test_a_cusum_detection_stamps_its_own_id_so_two_detectors_stay_distinguishable(signal):
    """The point of the identity mapping: pooling a Schmitt run and a CUSUM run gives a
    table whose rows still name their own author, which a per-table ``module_id`` cannot
    do once the two tables are one.
    """
    res = DetectCusum().run(_context(signal))
    df = res.payload.df
    assert set(df[SOURCE_COL].unique()) == {DetectCusum.id}
    assert set(df[REVIEW_COL].unique()) == {REVIEW_NONE}
    assert source_kind(df[SOURCE_COL].iloc[0]) == "conventional"


def test_the_ml_detector_stamps_the_same_pair_at_the_same_moment():
    """`ml_modules.MLDetect` builds its ``EventSet`` through ``stamp_source_status`` with
    its own ``self.id``, exactly as the two threshold detectors do.

    Asserted on the SOURCE rather than by running the model: a real ``MLDetect.run``
    needs a ``.keras`` file and a TensorFlow load, which is minutes and an external
    artefact, and the property under test is structural -- that the ML path has not been
    left as the one producer that forgets. The AST guard is the idiom
    ``tests/analysis/test_flag_invariant.py`` already uses for exactly this shape of
    claim. The behavioural half -- that ``detect.keras`` reads as the owner's "ML
    detector" category -- is checked directly below it.
    """
    src = Path(ml_modules.__file__).read_text()
    tree = ast.parse(src)
    stamped = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "stamp_source_status"
    ]
    assert stamped, (
        "ml_modules no longer stamps source/review, so a Keras detection produces rows "
        "that cannot say which detector authored them")
    assert any(
        isinstance(a, ast.Attribute) and a.attr == "id" for call in stamped
        for a in call.args), (
        "MLDetect must stamp its OWN module id, not a literal: a hard-coded string is a "
        "second spelling of detect.keras to keep in step")
    assert source_kind(ml_modules.MLDetect.id) == "ml"


def test_stamping_moves_no_measured_number(schmitt):
    """THE INVARIANT. The frozen 20 columns of a stamped detection hash identically to
    the same detection with the two columns dropped -- the new columns are description
    beside the measurement, never inside it.

    Measured on the owner's 15 s -1000 mV apoferritin recording as well, through the same
    hash: Schmitt 115 events a880ca47…28dc and CUSUM 213 events 063d6b25…d64c, before and
    after the stamp existed.
    """
    df = schmitt.payload.df
    assert _frozen20_sha256(df) == _frozen20_sha256(df.drop(columns=[SOURCE_COL, REVIEW_COL]))


def test_recharacterize_span_still_returns_exactly_the_frozen_twenty(schmitt, signal):
    """The rebuild product stays SCHEMA-only, which is what forces every rebuild path to
    re-apply the pair deliberately -- and is the same mechanism that lets a project saved
    before the columns existed still load. A rebuild that carried them silently would
    also have to be trusted not to carry ``orig_start_idx``, which must NEVER be written
    after an edit.
    """
    df = schmitt.payload.df
    row = recharacterize_span(
        signal, FS, int(df["start_idx"].iloc[0]), int(df["end_idx"].iloc[0]),
        provenance=schmitt.payload.provenance, prior=df.iloc[[0]])
    assert list(row.columns) == list(SCHEMA)


# ===========================================================================
# the rebuilds
# ===========================================================================
def _two_source_table(schmitt: AnalysisResult) -> EventSet:
    """The Schmitt detection with its LAST row re-attributed to CUSUM.

    Stands in for the table an owner actually builds -- two detections pooled, or one
    detection whose rows came from different runs -- without paying for a second
    detection inside a merge test.
    """
    df = schmitt.payload.df.copy()
    src = df[SOURCE_COL].to_numpy(dtype=object)
    src[-1] = DetectCusum.id
    df[SOURCE_COL] = pd.array(src, dtype="string")
    return EventSet(df, dict(schmitt.payload.provenance))


def test_a_merged_row_from_two_sources_reads_mixed_not_whichever_started_first(
        schmitt, signal):
    """``merge._merged_row`` carries a non-boolean extra column as ``series.iloc[0]``:
    for a measured column that is arbitrary, for a provenance column it is a false
    statement -- the answer would depend only on which fragment is to the left.
    """
    es = _two_source_table(schmitt)
    df = es.df
    # The two events at 85_000 and 85_550 are 200 samples (4 ms) apart; the last row is
    # the one re-attributed above, so their merge spans two sources.
    assert df[SOURCE_COL].iloc[-1] == DetectCusum.id
    assert df[SOURCE_COL].iloc[-2] == DetectSchmitt.id

    merged, n_merged = merge_by_gap(es, signal, FS, max_gap_s=0.006)
    assert n_merged >= 1, "the fixture pair did not merge; the assertion below is vacuous"
    assert SOURCE_MIXED in set(merged.df[SOURCE_COL].dropna()), (
        "a row assembled from a Schmitt fragment and a CUSUM fragment named only one of "
        "them, which is a provenance claim nobody made")


def test_a_merge_that_swallows_a_hand_added_event_says_mixed(schmitt, signal):
    """``added`` is ORed by the merge's existing boolean rule, so the merged row IS a
    hand-added row as far as `scoring.score_detection` and the training export are
    concerned -- AND a detector authored the rest of it, which is what ``mixed`` says.

    This test used to require ``manual`` here, on the argument that ``source`` should
    follow ``added`` because `events.ensure_source_status` states "added wins over
    source" in one place. That inverted the dependency: it put a SECOND copy of that
    rule in `merge`, overwrote the true answer `events.combine_sources` had already
    computed, and made `events.source_kind` read the row as purely manual -- dropping a
    detector-found event out of the ``conventional`` count in the export sidecar's
    ``by_kind``. ``mixed`` is not a claim that no detector authored the row, so it does
    not contradict ``added``; both recorded facts stay true and neither is rewritten to
    flatter the other. The one home of "added wins" still fills a BLANK source on an
    added row, at the install door. See ``test_src2_merge_mixed_source.py``.
    """
    df = schmitt.payload.df.copy()
    added = np.zeros(len(df), dtype=bool)
    added[-1] = True
    df[ADDED_COLUMN] = added
    src = df[SOURCE_COL].to_numpy(dtype=object)
    src[-1] = SOURCE_MANUAL
    df[SOURCE_COL] = pd.array(src, dtype="string")

    merged, n_merged = merge_by_gap(EventSet(df, dict(schmitt.payload.provenance)),
                                    signal, FS, max_gap_s=0.006)
    assert n_merged >= 1, "the fixture pair did not merge; the assertion below is vacuous"
    merged = merged.df
    hand = merged[ADDED_COLUMN].to_numpy(dtype=bool)
    assert hand.any(), "the hand-added fragment did not survive the merge"
    assert set(merged.loc[hand, SOURCE_COL]) == {SOURCE_MIXED}, (
        "a merged row assembled from a detector's fragment and a person's named only "
        "one of them; 'mixed' is the token that vocabulary was invented for")
    assert source_kind(merged.loc[hand, SOURCE_COL].iloc[0]) == KIND_MIXED, (
        "source_kind read the merged row as purely manual, which is how a "
        "detector-found event fell out of the sidecar's conventional-vs-ML split")


def test_a_merged_row_is_reviewed_only_when_every_fragment_was(schmitt, signal):
    """The conservative direction is the load-bearing one: claiming a merged row is
    reviewed because ONE of its fragments was is how a "final dataset" ends up holding
    spans nobody ever looked at.
    """
    df = schmitt.payload.df.copy()
    rev = df[REVIEW_COL].to_numpy(dtype=object)
    rev[-1] = REVIEW_DONE                      # one fragment ruled on, the other not
    df[REVIEW_COL] = pd.array(rev, dtype="string")

    merged, n_merged = merge_by_gap(EventSet(df, dict(schmitt.payload.provenance)),
                                    signal, FS, max_gap_s=0.006)
    assert n_merged >= 1, "the fixture pair did not merge; the assertion below is vacuous"
    assert set(merged.df[REVIEW_COL].dropna()) == {REVIEW_NONE}


def test_a_re_measure_carries_source_and_review_through_unchanged(schmitt, signal):
    """A re-measure rewrites amplitudes; it does not re-detect. Who found the event and
    whether a person ruled on it are therefore exactly what they were -- and they have to
    be re-applied on purpose, because ``recharacterize_span`` hands back the frozen 20.
    """
    df = schmitt.payload.df.copy()
    rev = df[REVIEW_COL].to_numpy(dtype=object)
    rev[0] = REVIEW_DONE
    df[REVIEW_COL] = pd.array(rev, dtype="string")

    out = remeasure_events(EventSet(df, dict(schmitt.payload.provenance)), signal, FS)
    after = out.events.df
    assert has_source_status(after)
    assert list(after[SOURCE_COL]) == list(df[SOURCE_COL])
    assert list(after[REVIEW_COL]) == list(df[REVIEW_COL])
    assert str(after[REVIEW_COL].dtype) == "string"


def test_a_review_marked_while_the_re_measure_ran_is_not_discarded(schmitt, signal):
    """A re-measure of the owner's -600 mV file takes ~3.3 s off the GUI thread and the
    table stays live for all of them. `merge_late_curation` already carries a late accept
    or flag; a late "mark reviewed" is the same gesture in the same seconds and would
    otherwise be silently overwritten by an answer measured from the frozen copy.
    """
    df = schmitt.payload.df.copy()
    before = curation_snapshot(df)
    assert REVIEW_COL in before.columns, (
        "the snapshot does not record reviewedness, so the install cannot tell a late "
        "review from the answer's own")

    out = remeasure_events(EventSet(df.copy(), dict(schmitt.payload.provenance)),
                           signal, FS).events.df

    live = df.copy()                                   # the user marks row 0 reviewed
    rev = live[REVIEW_COL].to_numpy(dtype=object)
    rev[0] = REVIEW_DONE
    live[REVIEW_COL] = pd.array(rev, dtype="string")

    installed = merge_late_curation(out, live, before)
    assert installed[REVIEW_COL].iloc[0] == REVIEW_DONE
    assert list(installed[REVIEW_COL])[1:] == list(out[REVIEW_COL])[1:]
    assert _frozen20_sha256(installed) == _frozen20_sha256(out), (
        "carrying the late review changed a measured column")


# ===========================================================================
# the one door for a changed frame
# ===========================================================================
def test_reconcile_result_materialises_the_pair_on_a_table_that_predates_it(schmitt):
    """`modules.reconcile_result` is where a merge, a hand-add and a re-measure all
    install a changed frame, so it is where the re-application is structural rather than
    remembered by four call sites. ``source`` comes from the result's ``module_id`` --
    the per-table answer this column exists to replace, and for a legacy frame the only
    answer there is.
    """
    legacy = schmitt.payload.df.drop(columns=[SOURCE_COL, REVIEW_COL])
    assert not has_source_status(legacy)

    out = reconcile_result(schmitt, EventSet(legacy, dict(schmitt.payload.provenance)))
    df = out.payload.df
    assert has_source_status(df)
    assert set(df[SOURCE_COL].unique()) == {DetectSchmitt.id}


def test_reconcile_result_leaves_review_unknown_rather_than_claiming_nobody_looked(
        schmitt):
    """``pd.NA`` is the third state and it means "nobody recorded this", not "False". A
    detection saved before the column existed may well have been curated; stamping it
    ``unreviewed`` would tell the owner his old project is a pile of unjudged candidates.
    """
    legacy = schmitt.payload.df.drop(columns=[SOURCE_COL, REVIEW_COL])
    out = reconcile_result(schmitt, EventSet(legacy, dict(schmitt.payload.provenance)))
    assert out.payload.df[REVIEW_COL].isna().all()


def test_reconcile_result_never_overwrites_a_recorded_cell(schmitt):
    """Idempotent, so it is safe on every install. A frame that already says a row was
    reviewed keeps saying so, whatever the enclosing result's module id is.
    """
    df = schmitt.payload.df.copy()
    rev = df[REVIEW_COL].to_numpy(dtype=object)
    rev[0] = REVIEW_DONE
    df[REVIEW_COL] = pd.array(rev, dtype="string")

    out = reconcile_result(schmitt, EventSet(df, dict(schmitt.payload.provenance)))
    assert out.payload.df[REVIEW_COL].iloc[0] == REVIEW_DONE
    assert out.payload.df[SOURCE_COL].iloc[0] == DetectSchmitt.id


def test_a_table_without_the_pair_still_merges_and_still_re_measures(schmitt, signal):
    """THE COMPATIBILITY PROMISE, exercised through the two engines rather than asserted.
    The owner was told he would not have to migrate every saved project; a legacy frame
    must therefore travel every rebuild path without acquiring the columns by accident
    and without raising.
    """
    legacy = schmitt.payload.df.drop(columns=[SOURCE_COL, REVIEW_COL])
    prov = dict(schmitt.payload.provenance)

    merged, _ = merge_by_gap(EventSet(legacy.copy(), prov), signal, FS, max_gap_s=0.006)
    assert not has_source_status(merged.df)

    remeasured = remeasure_events(EventSet(legacy.copy(), prov), signal, FS)
    assert not has_source_status(remeasured.events.df)
    assert _frozen20_sha256(remeasured.events.df[list(SCHEMA)]) == _frozen20_sha256(
        remeasure_events(EventSet(schmitt.payload.df.copy(), prov), signal, FS)
        .events.df)
