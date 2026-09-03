"""DRIFT — the half is the recording's clock, and the sentences match their gate.

Three defects, one arithmetic file. The user-visible half of every one of them is
in ``tests/gui/test_rk_time_half_and_messages_wired.py``, through a real
``MainWindow``; this file pins the numbers underneath.

1. **A split defined by the sample cannot measure the sample's own shape.**
   ``event_features.time_half`` split at each scope's MEDIAN onset, and a median
   halves its own sample by definition — so the pooled first-half fraction was
   **0.50 by construction**, on every recording, for ever. ``early_frac`` was
   therefore a statement about event ORDER printed under a caption about elapsed
   TIME, and ``subpop_gmm._stability_warnings``' recording-level
   non-stationarity caveat, which compares the pooled value against 0.25-0.75,
   could never fire. Three tests added in the previous batch skipped themselves
   saying exactly that (*"pooled is 0.50: both halves round to the same number
   here"*). The app looked like it checked for drift and did not — on recordings
   where voltage and elapsed time are perfectly confounded and drift is the
   confound most likely to be reported as a subpopulation.

2. **A printed number and the band printed beside it were not the same number
   the gate used.** ``population_table`` rounds ``early_frac`` to two decimals
   before ``_stability_warnings`` sees it; the band was then recomputed at full
   precision from those rounded values, so at pooled 0.505 it is
   (0.253769, 0.753731), prints as ``0.25-0.75``, and a class printed at ``0.25``
   was condemned for being *outside* ``0.25-0.75``. Same cause, worse wording: at
   pooled 0.095 a class at exactly **0.50** — one event in each half, the most
   balanced a class can be — was announced as *"at least 3:1 concentrated into
   one half of the recording"*, which is false of it.

3. **The one sentence that travels everywhere carried a number the report's own
   caveat disowns.** ``Separation.one_line()`` ended "; k-means K=3 on the same
   coordinates scores 0.197" while caveat 4, a few lines below, calls a number of
   exactly that kind "a CEILING ... not a rival result" and quotes a different
   one (0.224, at K=2).

WHERE ``nanoscope`` CAME FROM is asserted first, because this file is also run
against an out-of-tree REVERT of these hunks, where every behavioural test below
must fail. A revert stub with a wrong signature once manufactured four fake
failures in this project and was believed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import nanoscope
from nanoscope.analysis import event_features as EF
from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis import subpop_report as SR


# ===========================================================================
# 0. the tree under test
# ===========================================================================


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """The test that must PASS in both trees, so a failure below is behavioural."""
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    for mod in (EF, SG, SR):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} at {p} is not under {root}"
        assert p.stat().st_size > 20_000, f"{p} is {p.stat().st_size} B — a stub?"
    print(f"\n[DRIFT] nanoscope resolved from {root}")


# ===========================================================================
# 1. THE INVARIANT — the half is the clock's, not the sample's
# ===========================================================================


@pytest.mark.parametrize(
    "onsets, duration, expect_early",
    [
        # every event in the first tenth of a 100 s recording: all early.
        (np.linspace(0.5, 9.5, 41), 100.0, 41),
        # every event in the last tenth: none early.
        (np.linspace(90.5, 99.5, 41), 100.0, 0),
        # 9 in 10 late.
        (np.concatenate([np.linspace(1.0, 40.0, 10),
                         np.linspace(60.0, 99.0, 90)]), 100.0, 10),
        # genuinely uniform: about half.
        (np.linspace(0.0, 100.0, 101), 100.0, 51),
    ],
)
def test_the_clock_split_counts_events_in_the_recordings_own_first_half(
        onsets, duration, expect_early):
    """THE DEFECT, at its root: the boundary is ``duration/2``, full stop.

    The median split returns exactly ``ceil(n/2)`` early events for every one of
    these inputs — the same answer for a recording that fired everything in its
    first ten seconds as for one that fired uniformly. That is not a measurement.
    """
    half = EF.time_half(onsets, span=duration)
    got = int((half == "early").sum())
    assert got == expect_early, (got, expect_early)
    # ...and this is what it used to say instead.
    median_split = EF.time_half(onsets)
    assert int((median_split == "early").sum()) == (len(onsets) + 1) // 2


def test_the_pooled_first_half_fraction_is_no_longer_a_constant():
    """The property that made the recording-level caveat unreachable.

    Over a sweep of increasingly lopsided recordings the median split returns
    0.50 every single time (to within one event), while the clock split tracks
    the truth. The caveat's gate is ``pooled outside 0.25-0.75``, so under the
    median split it was arithmetically unable to fire.
    """
    duration = 100.0
    pooled_median, pooled_clock = [], []
    for share_early in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98):
        n = 500
        n_e = int(round(n * share_early))
        onsets = np.concatenate([
            np.linspace(0.5, 49.0, n_e), np.linspace(51.0, 99.5, n - n_e),
        ])
        pooled_median.append(float((EF.time_half(onsets) == "early").mean()))
        pooled_clock.append(float((EF.time_half(onsets, span=duration) == "early").mean()))
    print(f"\n[DRIFT] median split pooled: {pooled_median}")
    print(f"[DRIFT] clock  split pooled: {pooled_clock}")
    assert max(abs(p - 0.5) for p in pooled_median) <= 1.0 / 500
    assert min(pooled_clock) < 0.25 and max(pooled_clock) > 0.75
    for share, p in zip((0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98), pooled_clock):
        assert abs(p - share) < 0.01, (share, p)


def test_a_group_no_longer_decides_where_its_own_boundary_is():
    """``group`` selects the clock, it does not define the split.

    Two scopes in ONE recording, one confined to the first fifth and one to the
    last fifth. Under the median split each scope halved itself and both read
    0.50 — indistinguishable, and both wrong. Under the clock they share one
    boundary and are comparable with each other, which is the whole point of the
    column.
    """
    onsets = np.concatenate([np.linspace(1.0, 19.0, 50), np.linspace(81.0, 99.0, 50)])
    scope = np.array(["a"] * 50 + ["b"] * 50, dtype=object)
    half = EF.time_half(onsets, scope, span=100.0)
    a = half[scope == "a"]
    b = half[scope == "b"]
    assert (a == "early").all() and (b == "late").all()
    old = EF.time_half(onsets, scope)
    assert float((old[scope == "a"] == "early").mean()) == 0.5
    assert float((old[scope == "b"] == "early").mean()) == 0.5


def test_a_pooled_table_gives_every_recording_its_own_clock():
    """Across recordings the mapping form keeps each one's own length."""
    onsets = np.array([1.0, 9.0, 1.0, 9.0])
    grp = np.array(["short", "short", "long", "long"], dtype=object)
    half = EF.time_half(onsets, grp, span={"short": 10.0, "long": 100.0})
    assert list(half) == ["early", "late", "early", "early"]


def test_a_clock_that_cannot_be_a_clock_is_refused_rather_than_guessed():
    """Falling back to the median on a bad span is the defect, not the recovery."""
    with pytest.raises(ValueError, match="positive interval"):
        EF.time_half(np.array([1.0, 2.0]), span=0.0)
    with pytest.raises(ValueError, match="positive interval"):
        EF.time_half(np.array([1.0, 2.0]), span=(5.0, 5.0))
    with pytest.raises(ValueError, match="not finite"):
        EF.time_half(np.array([1.0, 2.0]), span=float("nan"))
    with pytest.raises(ValueError, match="duration in seconds or a"):
        EF.time_half(np.array([1.0, 2.0]), span=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="no clock window for group"):
        EF.time_half(np.array([1.0, 2.0]), np.array(["a", "b"]), span={"a": 10.0})
    with pytest.raises(ValueError, match="no `group` was given"):
        EF.time_half(np.array([1.0, 2.0]), span={"a": 10.0})
    # an empty recording still validates its clock rather than shrugging
    with pytest.raises(ValueError, match="positive interval"):
        EF.time_half(np.array([]), span=-1.0)
    with pytest.raises(ValueError, match="positive interval"):
        EF.time_half(np.array([]), np.array([]), span={"a": -1.0})
    assert EF.time_half(np.array([]), span=10.0).size == 0


def test_the_notebook_fallback_is_still_exactly_the_notebook():
    """``span=None`` is retained ONLY to reproduce the oracle column bit for bit."""
    onsets = np.array([1.0, 2.0, 3.0])
    assert list(EF.time_half(onsets)) == ["early", "early", "late"]
    grp = np.array(["a", "a", "b", "b"], dtype=object)
    assert list(EF.time_half(np.array([1.0, 5.0, 10.0, 40.0]), grp)) == [
        "early", "late", "early", "late"]


# ===========================================================================
# 2. the app's own path supplies the clock
# ===========================================================================


_FS = 50_000.0


def _recording(n_samples: int, event_starts, dwell: int = 60, depth: float = 40.0):
    rng = np.random.default_rng(11)
    x = np.full(n_samples, 200.0) + rng.normal(0.0, 1.2, n_samples)
    rows = []
    for i, s in enumerate(event_starts):
        s = int(s)
        x[s:s + dwell] -= depth
        rows.append({"start_idx": s, "end_idx": s + dwell})
    return (np.ascontiguousarray(x, dtype=np.float32),
            pd.DataFrame(rows))


def test_feature_table_splits_on_the_trace_it_was_handed_not_on_the_events():
    """THE WIRING. The app reaches ``time_half`` only through this function.

    A recording that fires every event in its first fifth: the clock says all of
    them are early, and the median said half of them were.
    """
    n = 2_000_000                                   # 40 s at 50 kHz
    starts = np.linspace(2_000, 395_000, 300).astype(int)     # first 8 s only
    trace, events = _recording(n, starts)
    feats, gate = SR.feature_table(trace, trace, _FS, events, recording="r")
    assert len(feats) > 250, f"fixture guard: only {len(feats)} gated events"
    early = float((feats["time_half"] == "early").mean())
    print(f"\n[DRIFT] feature_table pooled early_frac = {early:.4f} "
          f"over {len(feats)} events on a {n / _FS:.0f} s recording")
    assert early == 1.0, "every event is in the first fifth; all of them are early"
    assert feats["onset_s"].max() < (n / _FS) / 2.0


def test_the_same_events_on_a_longer_trace_move_to_the_other_half():
    """The boundary is a property of the RECORDING, so lengthening the recording
    moves the column while the events stay put. Under the median split the two
    columns are IDENTICAL, which is the tell that it was never about the
    recording at all."""
    starts = np.linspace(2_000, 395_000, 300).astype(int)
    short, events = _recording(500_000, starts)      # events fill the whole 10 s
    long_, _ = _recording(4_000_000, starts)         # same events, 80 s recording
    f_short, _ = SR.feature_table(short, short, _FS, events, recording="r")
    f_long, _ = SR.feature_table(long_, long_, _FS, events, recording="r")
    e_short = float((f_short["time_half"] == "early").mean())
    e_long = float((f_long["time_half"] == "early").mean())
    print(f"\n[DRIFT] same 300 events: 10 s recording -> {e_short:.3f}, "
          f"80 s recording -> {e_long:.3f}")
    assert e_long == 1.0
    assert e_short < e_long
    # and what the median split said about the same pair
    onsets = f_short["onset_s"].to_numpy()
    old_short = EF.time_half(onsets)
    old_long = EF.time_half(f_long["onset_s"].to_numpy())
    assert np.array_equal(old_short, old_long)
    assert float((old_short == "early").mean()) == pytest.approx(0.5, abs=1 / len(onsets))


# ===========================================================================
# 3. the printed number and the band beside it are the gate's own numbers
# ===========================================================================


def _table(early_fracs, ns):
    return pd.DataFrame({
        "subpop": list(range(len(ns))),
        "n": list(ns),
        "early_frac": list(early_fracs),
    })


def _note(notes, prefix):
    for s in notes:
        if s.startswith(prefix):
            return s
    return ""


def test_the_sentence_at_pooled_0_505_no_longer_contradicts_itself():
    """THE DEFECT, in the exact shape the follow-up pass measured it.

    Pooled 0.505 — which ``population_table``'s two-decimal rounding produces all
    the time — puts the real band at (0.253769, 0.753731). It printed as
    ``0.25-0.75``, and a class printed at ``0.25`` was condemned:
    *"subpop 0 (n=100, early_frac 0.25) — outside 0.25-0.75"*. Twenty-five is not
    outside twenty-five.

    THE FIX IS NOT SILENCE, it is self-consistency: the class really is outside
    the band by the odds rule, so it is still named — but every number in the
    sentence is now the printed one, and the band printed beside it recomputes
    from the pooled value printed beside THAT. A reader can check the claim
    against the column it sits under, which is all that was ever asked.
    """
    import re
    ef, ns = [0.25, 0.75, 0.51], [100, 100, 200]
    assert float(np.average(ef, weights=ns)) == 0.505
    raw_lo, raw_hi = SG.stability_band_for(0.505)
    # the arithmetic that made the old sentence unreadable
    assert raw_lo > 0.25 and f"{raw_lo:.2f}" == "0.25", (raw_lo, raw_hi)
    unstable = _note(SG._stability_warnings(_table(ef, ns)), "TIME-UNSTABLE")
    print(f"\n[DRIFT] pooled=0.505 unrounded band=({raw_lo!r}, {raw_hi!r})\n{unstable}")
    assert "subpop 0 " in unstable
    lo, hi = (float(x) for x in
              re.search(r"outside (\d\.\d\d)-(\d\.\d\d)", unstable).groups())
    printed_pooled = float(
        re.search(r"pooled early fraction of (\d\.\d\d)", unstable).group(1))
    # 1. no class printed inside the printed band is named
    assert not (lo <= 0.25 <= hi), (lo, hi)
    # 2. the printed band is the band of the printed pooled — recomputable on paper
    rlo, rhi = SG.stability_band_for(printed_pooled)
    assert (round(rlo, 2), round(rhi, 2)) == (lo, hi), (printed_pooled, lo, hi)


def test_a_class_printed_at_the_band_edge_is_not_condemned_for_being_outside_it():
    """The same defect where the honest answer IS silence.

    An unrounded pooled a hair over 0.50: the band is (0.250021, 0.750021), it
    prints ``0.25-0.75``, and the class printed at ``0.25`` was condemned for
    being outside it — on a difference of two ten-thousandths that the reader is
    never shown and could never have checked. Rounding to a fixed number of
    places is monotone, so gating on the printed values can only withdraw a flag
    like this one; it can never invent one.
    """
    ef, ns = [0.25, 0.75, 0.50, 0.51], [100, 100, 847, 3]
    pooled = float(np.average(ef, weights=ns))
    lo, hi = SG.stability_band_for(pooled)
    assert 0.50 < pooled < 0.5005 and f"{pooled:.2f}" == "0.50", pooled
    assert lo > 0.25 and f"{lo:.2f}" == "0.25", (lo, hi)
    text = "\n".join(SG._stability_warnings(_table(ef, ns)))
    print(f"\n[DRIFT] pooled={pooled!r} unrounded band=({lo!r}, {hi!r})\n{text!r}")
    assert "TIME-UNSTABLE CLASS(ES)" not in text, text


@pytest.mark.parametrize("nudge", [-0.004, -0.002, 0.0, 0.002, 0.004])
def test_every_number_in_the_sentence_is_checkable_against_the_column(nudge):
    """The invariant, swept across the rounding: a class is named IF AND ONLY IF
    the number printed for it lies outside the band printed beside it."""
    ef = [0.25, 0.30, 0.50, 0.70, 0.75]
    ns = [100, 200, 400, 200, 100]
    import re
    ef = [round(min(max(v + nudge, 0.0), 1.0), 2) for v in ef]
    notes = SG._stability_warnings(_table(ef, ns))
    unstable = _note(notes, "TIME-UNSTABLE CLASS(ES)")
    if not unstable:
        return
    named = {int(m) for m in re.findall(r"subpop (\d+) \(n=", unstable)}
    m = re.search(r"outside (\d\.\d\d)-(\d\.\d\d)", unstable)
    assert m, unstable
    lo, hi = float(m.group(1)), float(m.group(2))
    for i, v in enumerate(ef):
        outside = not (lo <= v <= hi)
        assert (i in named) == outside, (
            f"subpop {i} prints {v:.2f}, the band prints {lo:.2f}-{hi:.2f}, "
            f"named={i in named}\n{unstable}")


def test_the_sentence_does_not_make_an_absolute_claim_the_gate_did_not_measure():
    """A class at exactly 0.50 has one event in each half. It cannot be
    "at least 3:1 concentrated into one half of the recording", and on a
    back-loaded recording it was announced as exactly that."""
    notes = SG._stability_warnings(_table([0.09, 0.50, 0.10], [400, 100, 400]))
    unstable = _note(notes, "TIME-UNSTABLE CLASS(ES)")
    assert "subpop 1 " in unstable, unstable
    print(f"\n[DRIFT] {unstable}")
    assert "at least 3:1 concentrated into one half of the recording" not in unstable
    # ...and it says what it DID measure, with the class's own ratio beside it
    assert "odds" in unstable
    assert "early odds" in unstable


def test_a_recording_with_an_empty_half_says_so_once_and_stops():
    """Third defect found inside this pass's own fix.

    Rounding the pooled value to the printed precision made the degenerate case
    reachable: at a printed pooled of 0.00 the band collapses onto (0.00, 0.00)
    — which is what ``stability_band_for`` documents it should do — and the
    class gate then condemned every class with a non-zero ``early_frac`` for
    being "outside 0.00-0.00", with no odds ratio beside it because at pooled 0
    there are no odds. The recording-level sentence already says the whole of
    it, so the class-level one is silent there.
    """
    notes = SG._stability_warnings(_table([0.00, 0.00, 0.01], [400, 400, 200]))
    text = "\n".join(notes)
    print(f"\n[DRIFT] degenerate pooled:\n{text}")
    assert "THE RECORDING IS NOT STATIONARY" in text
    assert "100 % of all gated events fall in the second half" in text
    assert "TIME-UNSTABLE CLASS(ES)" not in text
    assert "0.00-0.00" not in text
    # ...and the mirror image
    text = "\n".join(SG._stability_warnings(_table([1.00, 1.00, 0.99], [400, 400, 200])))
    assert "100 % of all gated events fall in the first half" in text
    assert "TIME-UNSTABLE CLASS(ES)" not in text


def test_the_absolute_claim_is_checked_on_the_class_not_on_the_recording():
    """The same category error one level up, found inside the first fix.

    At pooled 0.34 a class at 0.70 is outside the recording's own odds band and
    is rightly named — but 0.70 is 2.33:1, not 3:1, and gating the absolute
    clause on the RECORDING being stationary announced it as 3:1 anyway. The
    claim is a property of the class, so it is checked on the class.
    """
    ef, ns = [0.30, 0.30, 0.70], [400, 400, 100]
    pooled = round(float(np.average(ef, weights=ns)), 2)
    assert 0.25 <= pooled <= 0.75, pooled          # the recording looks stationary
    unstable = _note(SG._stability_warnings(_table(ef, ns)), "TIME-UNSTABLE")
    print(f"\n[DRIFT] pooled={pooled}\n{unstable}")
    assert "subpop 2 " in unstable, unstable
    assert "at least 3:1 concentrated into one half of the recording" not in unstable


def test_the_absolute_reading_survives_where_it_is_true():
    """No over-correction: on a stationary recording the band IS 0.25-0.75 and
    being outside it IS 3:1 into one half, so the caption's sentence stands."""
    notes = SG._stability_warnings(_table([0.24, 0.50, 0.50, 0.76], [100, 300, 300, 100]))
    unstable = _note(notes, "TIME-UNSTABLE CLASS(ES)")
    assert "outside 0.25-0.75" in unstable
    assert "at least 3:1 concentrated into one half of the recording" in unstable
    assert "subpop 0 " in unstable and "subpop 3 " in unstable
    assert "subpop 1 " not in unstable and "subpop 2 " not in unstable


def test_the_class_ratio_printed_recomputes_from_the_two_printed_fractions():
    """The added clause is checkable arithmetic, not decoration."""
    import re
    ef, ns = [0.09, 0.50, 0.10], [400, 100, 400]
    notes = SG._stability_warnings(_table(ef, ns))
    unstable = _note(notes, "TIME-UNSTABLE CLASS(ES)")
    pooled = float(re.search(r"pooled early fraction of (\d\.\d\d)", unstable).group(1))
    m = re.search(r"subpop 1 \(n=100, early_frac (\d\.\d\d), early odds ([\d.]+)x", unstable)
    assert m, unstable
    v, printed_ratio = float(m.group(1)), float(m.group(2))
    expect = (v / (1 - v)) / (pooled / (1 - pooled))
    assert printed_ratio == pytest.approx(expect, abs=0.05), (printed_ratio, expect)


# ===========================================================================
# 4. the one sentence that travels
# ===========================================================================


def _sep(sil=0.098, ref=0.197, profile=None, **kw):
    return SG.Separation(
        silhouette=sil, neg_fraction=0.27, nearer_other_fraction=0.315,
        per_subpop={i: (100, sil, 0.3) for i in range(7)},
        reference_k=3, reference_silhouette=ref,
        n_used=kw.pop("n_used", 985), n_events=kw.pop("n_events", 985),
        n_pcs=11, reference_profile=dict(profile or {2: 0.224, 3: 0.197}),
        **kw,
    )


def test_the_travelling_sentence_no_longer_carries_the_disowned_rival():
    """THE DEFECT. It is the one sentence that reaches every surface."""
    line = _sep().one_line()
    print(f"\n[DRIFT] one_line(): {line}")
    assert "k-means" not in line
    assert "0.197" not in line
    assert "same coordinates" not in line


def test_the_travelling_sentence_leads_with_the_verdict_so_truncation_keeps_it():
    """WORD ORDER IS THE CORRECTNESS PROPERTY HERE. The status bar elides, and
    what fell off the end used to be the caveat while the rival number stayed."""
    sep = _sep()
    line = sep.one_line()
    assert line.startswith(sep.verdict), line
    assert len(line) < 150, f"{len(line)} chars — it has to fit a status bar"


def test_the_sentence_still_carries_the_numbers_that_qualify_the_count():
    sep = _sep()
    line = sep.one_line()
    assert f"{sep.silhouette:.3f}" in line
    assert f"{sep.n_pcs}-PC" in line
    assert f"{100 * sep.nearer_other_fraction:.1f} %" in line


def test_the_subsample_disclosure_is_not_a_casualty_of_the_edit():
    line = _sep(n_used=4000, n_events=4218, subsampled=True).one_line()
    assert "on 4000 of 4218 events" in line


def test_the_reference_silhouette_is_still_printed_once_where_it_is_framed():
    """Deleted from the travelling sentence, NOT from the report. The document
    prints it under ``separation`` with the ceiling paragraph on the next line —
    the framing the one-liner had no room for. Losing it here would be a
    different defect, so it is pinned rather than assumed."""
    q = SG.SubpopQuantification(
        selection=_FakeSel(), fit=_FakeFit(), ci=None, table=pd.DataFrame(),
        crosstab=None, features=(), n_events=985, separation=_sep(),
    )
    block = "\n".join(SR._separation_lines(q))
    print(f"\n[DRIFT] report separation block:\n{block}")
    assert "k-means K=3 on the same coordinates scores 0.1970" in block
    # and the sentence that disowns it is in the same block, below it
    assert block.index("BOUND on separability") > block.index("k-means K=3")
    assert "not a rival result and not a suggested K" in block


def test_the_claim_puts_the_verdict_next_to_the_count():
    """``claim()`` is the only formatter of the class count in the codebase."""
    fit = _FakeFit()
    q = SG.SubpopQuantification(
        selection=_FakeSel(), fit=fit, ci=None, table=pd.DataFrame(),
        crosstab=None, features=(), n_events=985, separation=_sep(),
    )
    claim = q.claim()
    print(f"\n[DRIFT] claim(): {claim}")
    assert "k-means" not in claim
    i_count, i_verdict = claim.index("7 subpopulation(s)"), claim.index("NO SUBSTANTIAL")
    assert i_verdict - i_count < 40, claim


class _FakeFit:
    k = 7
    n_pcs = 11
    conf_frac = 0.844


class _FakeSel:
    k_grid = (1, 2, 3, 4, 5, 6, 7, 8)
