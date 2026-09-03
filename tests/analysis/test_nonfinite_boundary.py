"""WHERE a NaN/Inf has to be before it taints a row — the boundary, pinned region by region.

``tests/analysis/test_nonfinite_rows.py`` pins the RULE: a row measured from a non-finite
sample is flagged ``non_finite`` and not accepted. This file pins its EDGE, which is the
part that was still wrong: a non-finite sample outside ``[start, end)`` but inside a window
the row's numbers were nevertheless read from.

Two regions, measured on a 600-sample event with one bad sample:

* **the rebuild's padded flanks** (``recharacterize_span``: ``max(event length, 3 ms)`` each
  side), from which the local baseline AND the SNR denominator come and nothing else. The
  behaviour used to split on WHICH non-finite value landed there::

      flank NaN   base=nan  depth=nan   snr=nan          accepted=False  flag='non_finite'
      flank +Inf  base=-100 depth=-20.0 snr=-2.00287e+10 accepted=True   flag=''

  because a median is an order statistic and reads straight through an Inf, while
  ``filters.noise_rms`` refuses the same window, ``span_noise_rms`` walks its whole ladder
  and returns its ``1e-9`` epsilon, and ``snr = depth/1e-9``. This is verification report
  §2.3's "an inf in the pad flank yields a finite plausible-wrong row with snr = 19.8e9",
  which that report marked UNCONFIRMED — it reproduces exactly when provenance is MISSING
  ``noise_rms_pA``, i.e. for a hand-added event or a legacy EventSet.

* **the rise/fall search pad** (``_EDGE_PAD_SAMPLES = 64`` each side of the event, inside
  ``characterize`` and therefore on BOTH detector paths too). A non-finite sample compares
  True against one of the two levels and becomes a false crossing — ``inf >= 0.9*depth``
  on the way in, ``-inf <= 0.1*depth`` on the way out — and ``_cross`` then divides by an
  infinite slope::

      clean                            rise_s=2.3909679e-04  fall_s=2.0959694e-04
      one +inf 40 samples before       rise_s=0.0            fall_s=2.0959694e-04
      one +inf 5 samples into the ramp rise_s=1.6330800e-05  fall_s=2.0959694e-04
      one -inf 5 samples before the end                      fall_s=1.9687800e-04

  ``rise_s = 0`` is an instantaneous transition — physically impossible, finite, accepted
  and unflagged.

  The first shape of the fix refused the WHOLE window (``if not isfinite(w).all():
  continue``), which cost four numbers where one was wrong: measured on the owner's
  recording with one Inf 20 samples before event 100, it blanked ev100's rise AND fall
  AND neighbouring ev99's rise and fall, all four of which the pre-fix code had right —
  the two searches share one array and adjacent events' pads overlap. What is refused now
  is the crossing time whose own 10-90 % interval read the bad sample, and nothing wider.
  On the same 12 poison placements as the table below: pre-fix lied 3 times, the wholesale
  guard blanked 24 numbers, this blanks 2 — the 2 that were read from a non-number.

And the other half of a boundary, which is where over-reach would hide: a sample outside
every window a row reads must leave that row BIT-IDENTICAL. One bad sample in a 15 M-sample
file has to cost the rows around it, not the table.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.baseline import derive_polarity, estimate_baseline
from nanoscope.analysis.characterize import (
    _EDGE_PAD_SAMPLES,
    RECHAR_PAD_MS,
    characterize,
    recharacterize_span,
)
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import SCHEMA, FLAG_NON_FINITE, EventSet, has_flag
from nanoscope.analysis.measure import NonFiniteSelection, measure_selection
from nanoscope.analysis.merge import merge_by_gap
from nanoscope.core.types import nonfinite_issue, nonfinite_reason

FS = 100_000.0
N = 20_000
START, END = 10_000, 10_600
I0 = -100.0
DEPTH = 20.0

ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
_needs_abf = pytest.mark.skipif(not ABF.exists(), reason="owner's -600 mV recording not present")

#: The rebuild pad actually used for this fixture's event: max(event length, 3 ms).
_PAD = max(END - START, int(round(FS * RECHAR_PAD_MS / 1000.0)))


# ===========================================================================
# builders
# ===========================================================================
def _trace(*, seed=7, ramped=False) -> np.ndarray:
    """One 600-sample blockade in 20 k samples of Gaussian noise, positive-going."""
    rng = np.random.default_rng(seed)
    x = I0 + rng.normal(0.0, 1.0, N)
    if ramped:
        ramp = np.linspace(0.0, DEPTH, 30)
        x[START:START + 30] += ramp
        x[START + 30:END - 30] += DEPTH
        x[END - 30:END] += ramp[::-1]
    else:
        x[START:END] += DEPTH
    return x


def _poison(idx: int, value: float, **kw) -> np.ndarray:
    x = _trace(**kw)
    x[idx] = value
    return x


def _rebuilt(sig: np.ndarray, *, noise: bool) -> pd.Series:
    """One rebuilt row. ``noise=False`` is the hand-added-event / legacy case: no
    ``noise_rms_pA`` in provenance, so ``span_noise_rms`` has to estimate one."""
    prov = {"polarity": 1}
    if noise:
        prov["noise_rms_pA"] = 1.0
    return recharacterize_span(sig, FS, START, END, provenance=prov, prior=None,
                               recording="fixture").iloc[0]


def _same(a, b) -> bool:
    """Bit-identical, with NaN == NaN — several SCHEMA columns are legitimately NaN
    (``rise_s``/``fall_s`` whenever the event never reaches 90 %), and a comparison that
    called those "different" would make every bit-identity pin below vacuously fail."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _detected(sig: np.ndarray) -> pd.Series:
    """One row through the DETECTOR path: whole-array characterize, exact baseline."""
    return characterize(sig, np.full(N, I0),
                        np.array([START], dtype=np.int64),
                        np.array([END], dtype=np.int64),
                        FS, 1, noise_rms=1.0).iloc[0]


# ===========================================================================
# THE REBUILD PAD — the row's baseline and SNR denominator are read from it
# ===========================================================================
@pytest.mark.parametrize("side", ["left", "right"])
def test_an_inf_in_the_rebuild_pad_no_longer_reports_an_snr_of_twenty_billion(side):
    """§2.3's unconfirmed sub-claim, reproduced and closed."""
    idx = START - 10 if side == "left" else END + 10
    row = _rebuilt(_poison(idx, np.inf), noise=False)

    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False
    # The number the old code printed was -2.00287e+10. Anything finite here is a lie.
    assert not np.isfinite(row["snr"])
    assert not np.isfinite(row["depth_pA"])


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("noise", [True, False])
def test_nan_and_inf_in_the_pad_are_judged_the_same_way(value, noise):
    """The tell that this was a defect and not a policy: the SAME defect in the SAME
    place produced a rejected row or a silently wrong one depending on which non-finite
    value it was, and on whether provenance happened to carry the detector's noise."""
    row = _rebuilt(_poison(START - 10, value), noise=noise)
    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False


def test_a_poisoned_pad_is_stable_over_repeated_rebuilds():
    """A derived flag must be re-derivable: rebuilding the rebuilt row keeps the verdict."""
    sig = _poison(START - 10, np.inf)
    prov = {"polarity": 1}
    row = None
    for _ in range(5):
        out = recharacterize_span(sig, FS, START, END, provenance=prov, prior=row)
        row = out.iloc[0]
        assert has_flag(row["flag"], FLAG_NON_FINITE)
        assert bool(row["accepted"]) is False


def test_a_poisoned_rebuild_is_silent(recwarn):
    """A merge over a poisoned stretch used to emit a numpy RuntimeWarning PER ROW out of
    a GUI worker (``inf - inf`` in the flank detrend of ``span_noise_rms``), for a number
    that was then thrown away. The row's verdict is what carries the information."""
    sig = _poison(START - 10, np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        row = _rebuilt(sig, noise=False)
    assert has_flag(row["flag"], FLAG_NON_FINITE)


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_a_sample_beyond_the_pad_leaves_the_rebuilt_row_bit_identical(value):
    """The anti-over-reach pin. One bad sample must cost the rows that READ it and no
    others — otherwise a single poisoned sample rejects a 1738-row table."""
    clean = _rebuilt(_trace(), noise=False)
    far = _rebuilt(_poison(START - _PAD - 5, value), noise=False)
    for col in SCHEMA:
        assert _same([clean[col]], [far[col]]), col


# ===========================================================================
# THE RISE/FALL PAD — reaches outside the event on EVERY path
# ===========================================================================
@pytest.mark.parametrize("offset", [-10, -40])
def test_an_inf_before_the_event_no_longer_reports_a_zero_rise_time(offset):
    """``rise_s = 0.0`` is an instantaneous transition: impossible, finite, unflagged.

    A non-finite sample can no longer BE a crossing, so the real 90 % crossing is found
    and the reported rise time is the one the clean trace gives. The bad sample sits
    outside the 10-90 % interval this number is measured over, so it costs nothing --
    which is the whole rule: a row loses what it read from a non-number, not what it
    happened to sit near."""
    clean = _detected(_trace(ramped=True))
    row = _detected(_poison(START + offset, np.inf, ramped=True))

    assert row["rise_s"] != 0.0, "0.0 is the lie this exists to kill"
    assert _same([row["rise_s"]], [clean["rise_s"]])
    assert _same([row["fall_s"]], [clean["fall_s"]])


def test_a_false_ten_percent_crossing_no_longer_shortens_the_fall_time():
    """The exit side of the same defect, with the sign that lies there.

    ``-inf <= 0.1*depth`` is True, so a ``-inf`` inside the exit ramp used to become the
    10 % crossing and the fall time came back 1.96878e-04 against a clean 2.0959694e-04 --
    finite, plausible, 6 % short, accepted and unflagged. (A ``+inf`` in the same place
    already gave NaN, which is why the fall side needs THIS sign to have a failing-before
    state at all.)"""
    clean = _detected(_trace(ramped=True))
    row = _detected(_poison(END - 5, -np.inf, ramped=True))

    assert not np.isfinite(row["fall_s"]), "a fall time read off a -inf is not a measurement"
    assert _same([row["rise_s"]], [clean["rise_s"]]), "the entry side read nothing bad"


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_only_the_side_whose_interval_read_the_bad_sample_loses_its_number(value):
    """The anti-over-reach pin for the guard itself, both signs and both sides.

    The surviving side is compared loosely, not bit-exactly: these two placements are
    INSIDE ``[start, end)``, so the event's own median depth shifts a little and moves
    both level thresholds with it. The claim being pinned is that the other side still
    reports a number at all -- refusing the whole window returned NaN for both."""
    clean = _detected(_trace(ramped=True))

    entry = _detected(_poison(START + 5, value, ramped=True))    # inside the ENTRY ramp
    assert not np.isfinite(entry["rise_s"])
    assert np.isfinite(entry["fall_s"]), "the exit ramp is ~590 samples away"
    assert entry["fall_s"] == pytest.approx(clean["fall_s"], rel=1e-3)

    exit_ = _detected(_poison(END - 5, value, ramped=True))      # inside the EXIT ramp
    assert not np.isfinite(exit_["fall_s"])
    assert np.isfinite(exit_["rise_s"])
    assert exit_["rise_s"] == pytest.approx(clean["rise_s"], rel=1e-3)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_a_neighbours_overlapping_pad_does_not_cost_the_neighbour_its_numbers(value):
    """Two events 80 samples apart: their +/-64-sample pads overlap, so a sample in the
    gap is inside BOTH windows. Refusing the whole window blanked all four crossing
    times; the interval rule blanks only what was measured over the bad sample.

    Measured on the owner's recording (events 99/100, one Inf 20 samples before ev100):
    the wholesale guard moved 2 rows and 4 numbers, this moves 0."""
    a0, a1 = START, START + 200
    b0, b1 = a1 + 80, a1 + 280
    ramp = np.linspace(0.0, DEPTH, 30)

    def two_events(poison_at=None):
        rng = np.random.default_rng(11)
        x = I0 + rng.normal(0.0, 1.0, N)
        for s, e in ((a0, a1), (b0, b1)):
            x[s:s + 30] += ramp
            x[s + 30:e - 30] += DEPTH
            x[e - 30:e] += ramp[::-1]
        if poison_at is not None:
            x[poison_at] = value
        return x

    def rows(sig):
        return characterize(sig, np.full(N, I0),
                            np.array([a0, b0], dtype=np.int64),
                            np.array([a1, b1], dtype=np.int64),
                            FS, 1, noise_rms=1.0)

    clean = rows(two_events())
    got = rows(two_events(poison_at=a1 + 40))      # in the gap: in BOTH pads
    for col in ("rise_s", "fall_s"):
        assert _same(got[col].to_numpy(), clean[col].to_numpy()), (
            f"{col} moved on an event whose own 10-90% interval never read the sample")


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_the_risefall_guard_does_not_reject_the_row(value):
    """Deliberate asymmetry with the pad above: nothing outside ``[start, end)`` enters
    depth, ecd, mean_dev or the baseline, so the ROW is still a measurement. Only the two
    crossing times, which already document NaN as "not measurable in this window", go."""
    clean = _detected(_trace(ramped=True))
    row = _detected(_poison(START - 40, value, ramped=True))

    assert bool(row["accepted"]) is True and row["flag"] == ""
    for col in ("depth_pA", "ecd_pA_s", "mean_dev_pA", "max_dev_pA", "baseline_pA", "snr"):
        assert row[col] == clean[col], col


@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_a_sample_beyond_the_risefall_pad_leaves_the_row_bit_identical(value):
    clean = _detected(_trace(ramped=True))
    far = _detected(_poison(START - _EDGE_PAD_SAMPLES - 40, value, ramped=True))
    for col in SCHEMA:
        assert _same([clean[col]], [far[col]]), col


def test_a_clean_trace_pays_nothing_and_keeps_its_crossing_times():
    """The control for both guards at once."""
    row = _detected(_trace(ramped=True))
    assert np.isfinite(row["rise_s"]) and row["rise_s"] > 0
    assert np.isfinite(row["fall_s"]) and row["fall_s"] > 0
    assert bool(row["accepted"]) is True and row["flag"] == ""


# ===========================================================================
# measure_selection — the SEPARATE 28-column manual schema
# ===========================================================================
def test_a_poisoned_open_pore_window_is_refused_by_its_own_name():
    """It used to refuse only by accident, and only for an Inf: ``filters.noise_rms``
    raised "x contains non-finite samples", naming its own local variable."""
    with pytest.raises(NonFiniteSelection, match="open-pore window beside selection"):
        measure_selection(_poison(START - 50, np.inf), FS, START, END)


def test_a_nan_flank_no_longer_surfaces_as_a_float_conversion_error():
    """A flank NaN reached the level estimator and came out as "cannot convert float NaN
    to integer" — a message that names neither the flank nor the recording."""
    with pytest.raises(NonFiniteSelection) as exc:
        measure_selection(_poison(START - 50, np.nan), FS, START, END)
    assert "cannot convert float NaN" not in str(exc.value)
    assert "open-pore window" in str(exc.value)


def test_the_refusal_is_a_ValueError_so_every_existing_caller_still_works():
    assert issubclass(NonFiniteSelection, ValueError)


def test_an_ordinary_gesture_is_not_dressed_up_as_a_poisoned_selection():
    """A collapsed drag must stay an ordinary ValueError: a GUI that showed a red banner
    on every collapsed drag would train the user to ignore the one that matters."""
    with pytest.raises(ValueError) as exc:
        measure_selection(_trace(), FS, START, START + 1)
    assert not isinstance(exc.value, NonFiniteSelection)


def test_a_supplied_baseline_and_noise_are_not_second_guessed():
    """Only the windows actually READ are checked. A caller that hands over both
    references never reads the flanks, so a poisoned flank is none of its business."""
    stats = measure_selection(_poison(START - 50, np.inf), FS, START, END,
                              baseline=I0, noise_rms=1.0)
    assert np.isfinite(stats["amplitude"])


def test_a_clean_selection_beside_a_poisoned_one_is_untouched():
    x = _poison(START - 50, np.inf)
    assert measure_selection(x, FS, 2_000, 3_000) == measure_selection(_trace(), FS, 2_000, 3_000)


# ===========================================================================
# A DEAD CHANNEL IS NOT A TAINTED ONE
# ===========================================================================
def test_a_dead_channel_and_a_single_bad_sample_do_not_read_the_same():
    """Different conditions, different causes, different remedies: 'every sample is
    non-finite' is a zero/infinite gain or an overflowed conversion and cropping cannot
    help, while 'one sample in 3 M' is a localised taint the user can crop past."""
    dead = nonfinite_reason(3_000, 3_000, 0)
    taint = nonfinite_reason(1, 3_000_000, 1_500_000)

    assert "dead or mis-calibrated" in dead and "localised taint" not in dead
    assert "localised taint" in taint and "dead" in taint     # "not a dead channel"
    assert "1,500,000" in taint and "3,000,000" in taint
    assert dead != taint


def test_the_dead_case_does_not_offer_a_remedy_that_cannot_work():
    assert "crop" not in nonfinite_reason(500, 500, 0)


@pytest.mark.parametrize("fn,kwargs", [
    (lambda x: derive_polarity(x), {}),
    (lambda x: estimate_baseline(x, FS), {}),
])
def test_the_signal_guards_say_which_condition_they_hit(fn, kwargs):
    """Both refusals in ``baseline`` count from a mask they already computed, and both
    go through the one shared writer — so they cannot drift apart in wording."""
    one_bad = _trace()
    one_bad[123] = np.nan
    with pytest.raises(ValueError, match="localised taint"):
        fn(one_bad)
    with pytest.raises(ValueError, match="dead or mis-calibrated"):
        fn(np.full(N, np.nan))


def test_measure_selection_says_which_condition_it_hit():
    with pytest.raises(NonFiniteSelection, match="localised taint"):
        measure_selection(_poison(START + 100, np.nan), FS, START, END)
    with pytest.raises(NonFiniteSelection, match="dead or mis-calibrated"):
        measure_selection(np.full(N, np.nan), FS, START, END)


def test_the_load_boundary_issue_carries_the_same_distinction():
    """``meta.issues`` is where most users meet this first."""
    assert "dead or mis-calibrated" in nonfinite_issue(0, "IN15", 100, 0, 100)
    assert "localised taint" in nonfinite_issue(0, "IN15", 1, 7, 100)


def test_one_writer_for_the_dead_vs_taint_sentence():
    """A second hand-written copy of either phrase is how the two descriptions drift.
    Both are WRITTEN in ``core.types.nonfinite_reason`` and nowhere else — an AST scan
    over non-docstring string literals, so prose about the distinction stays free."""
    import ast

    root = Path(__file__).resolve().parents[2] / "nanoscope"
    hits = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = {id(ast.get_docstring(n, clean=False)) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                        ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node.value) not in docstrings
                    and ("localised taint" in node.value
                         or "dead or mis-calibrated" in node.value)):
                hits.append(path.name)
    assert sorted(set(hits)) == ["types.py"]


# ===========================================================================
# WHERE THE NON-FINITE VALUE CAME FROM IS NOT THE RULE'S BUSINESS
# ===========================================================================
def test_a_calibration_that_overflows_float32_still_reaches_the_row(tmp_path):
    """The load boundary scans the RAW arrays, so a value created AFTER load — a gain
    that overflows float32, report §2.8 — never reaches ``meta.issues`` and the Issues
    box says "(none)". That gap is not fixed here, and it does not have to be: the rule
    is on the array the row is measured from, so the row-level consequence is caught
    whatever produced it. Driven through the real ``Session.calibrated`` path."""
    from nanoscope.core.calibration import ChannelCalibration
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.session import Session
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

    raw = np.ascontiguousarray(_trace(), dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    doc = TraceDocument(
        meta=RecordingMeta(path=tmp_path / "overflow.dat", format_name="synthetic",
                           n_samples=raw.size, sample_rate_hz=FS,
                           rate_provenance=Provenance.FILE_META, channels=(info,)),
        channels=(raw,))

    session = Session()
    doc_id = session.add_document(doc)
    # write_sidecar=False: a harness must never write into the owner's data directory.
    session.set_calibration(doc_id, 0,
                            ChannelCalibration(gain=1e38, offset=0.0, target_units="pA"),
                            write_sidecar=False)
    sig = np.asarray(session.calibrated(doc_id, 0).data, dtype=np.float64)

    assert not np.isfinite(sig).all(), "the overflow is the premise of this test"
    assert doc.meta.issues == (), "...and the load boundary never heard about it (§2.8)"

    row = recharacterize_span(sig, FS, START, END, provenance={"polarity": 1}).iloc[0]
    assert has_flag(row["flag"], FLAG_NON_FINITE)
    assert bool(row["accepted"]) is False


# ===========================================================================
# THE OWNER'S REAL RECORDING
# ===========================================================================
@_needs_abf
def test_one_poisoned_sample_of_the_owners_recording_costs_only_its_neighbours():
    """The whole boundary, on 2,993,798 real samples: poison ONE of them, re-measure
    every detected event through the rebuild path, and require that exactly the rows
    whose padded window contains it are rejected — every other row bit-identical."""
    from nanoscope.core.session import Session
    from nanoscope.io.base import open_document

    session = Session()
    doc_id = session.add_document(open_document(ABF))
    sig = np.asarray(session.calibrated(doc_id, 2).data, dtype=np.float64)
    fs = float(session.document(doc_id).meta.sample_rate_hz)

    df, prov = detect_events(sig, fs, min_dwell_s=2e-5)
    # The documented detection on this file UNDER THE ESTIMATOR IT SHIPS WITH. It was
    # 1738 rows against the 10 ms block median, which stamped 99.7% of those events
    # with a baseline more than 10 pA from the true open pore (-34.8 pA) and reported
    # this event's depth as 32.04 pA against a true 66.3.
    #
    # WHY THE COUNT FALLS, correctly: NOT the 1 ms mask dilation, which an earlier
    # comment here blamed. Measured on this file at min_dwell_s=1e-4 -- block median
    # 10 ms 1641 events at a 0.44 ms median dwell; block median 200 ms (no mask, no
    # dilation at all) 327 events at 17.48 ms; masked default 315 at 17.56 ms. The
    # cause is the LEVEL: a reference that dives into each blockade rises and falls
    # with the event, so the Schmitt trigger re-crosses inside it and one 17 ms
    # blockade is emitted as a string of 0.4 ms fragments. Turning the dilation off
    # does not restore them (263 events, 18.02 ms) -- it removes 52 more.
    #
    # Detection GEOMETRY given a fixed baseline is unchanged (tests/analysis/
    # test_detect.py holds that pin); which events are found is not, and that is the
    # intended effect of the change.
    assert len(df) == 317, "the documented detection on this file"

    spans = [(int(r.start_idx), int(r.end_idx)) for _, r in df.head(200).iterrows()]
    few = int(round(fs * RECHAR_PAD_MS / 1000.0))

    def blast(v):
        return [i for i, (a, b) in enumerate(spans)
                if a - max(b - a, few, 1) <= v < b + max(b - a, few, 1)]

    # The victim is DERIVED, not indexed: the event whose pad the most neighbours
    # share. That is the point of the test -- the blast radius is "every row that read
    # the sample", not "the nearest one" and not "the table" -- and a hard-coded index
    # quietly stops testing it the moment the detection moves.
    victim = max((int(a) - 5 for a, _ in spans), key=lambda v: len(blast(v)))
    expected = blast(victim)
    assert len(expected) >= 2, "pick a victim whose pad more than one row reads"
    poisoned = sig.copy()
    poisoned[victim] = np.inf

    clean_rows = [recharacterize_span(sig, fs, a, b, provenance=prov) for a, b in spans]
    dirty_rows = [recharacterize_span(poisoned, fs, a, b, provenance=prov) for a, b in spans]

    hit = [i for i, r in enumerate(dirty_rows)
           if bool(has_flag(r.iloc[0]["flag"], FLAG_NON_FINITE))]
    assert hit == expected, f"sample {victim}: {hit} vs {expected}"
    assert not any(bool(dirty_rows[i].iloc[0]["accepted"]) for i in hit)
    # A blast radius, not a table. Stated against the rows actually RE-MEASURED
    # rather than against the table's length: the old spelling
    # ``len(hit) * 200 < len(df)`` silently encoded a 1738-row detection, so it
    # tightened from "under 0.5% of the table" to "at most one row" the moment the
    # event count moved, which is not a property of the poisoning rule at all.
    assert len(hit) < 0.05 * len(spans), "a blast radius, not a table"

    for i, (c, d) in enumerate(zip(clean_rows, dirty_rows)):
        if i in hit:
            continue
        for col in SCHEMA:
            assert _same(c[col].to_numpy(), d[col].to_numpy()), (i, col)


@_needs_abf
def test_the_owners_recording_acquires_no_non_finite_flag_anywhere():
    """The control that matters most: the real file is clean, so detection, a declared
    dwell limit and a 5 ms gap merge must all come back with zero ``non_finite`` rows."""
    from nanoscope.core.session import Session
    from nanoscope.io.base import open_document

    session = Session()
    doc_id = session.add_document(open_document(ABF))
    sig = np.asarray(session.calibrated(doc_id, 2).data, dtype=np.float64)
    fs = float(session.document(doc_id).meta.sample_rate_hz)

    df, prov = detect_events(sig, fs, min_dwell_s=2e-5, max_dwell_s=2e-3)
    assert not has_flag(df["flag"], FLAG_NON_FINITE).any()

    merged, _ = merge_by_gap(EventSet(df, prov), sig, fs, max_gap_s=5e-3)
    # 344 under the pre-change 10 ms block median; 245 before `merge_by_gap` grew its
    # occupancy floor. The floor refuses exactly ONE group on this recording, at occupancy
    # 0.213 -- a run of four whose merged span would have been 79% open pore, reported as a
    # single event whose dwell is the whole span and whose median depth the open pore
    # out-votes. The four survive as four, which is why the count moved 245 -> 248.
    assert len(merged.df) == 248
    assert not has_flag(merged.df["flag"], FLAG_NON_FINITE).any()
    assert np.isfinite(merged.df["snr"].to_numpy()).all()
    assert np.isfinite(merged.df["depth_pA"].to_numpy()).all()
