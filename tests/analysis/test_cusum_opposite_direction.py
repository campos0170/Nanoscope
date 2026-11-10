"""A CUSUM run counts its own opposite direction, and says so when it matters.

`detect_cusum.detect_events_cusum` runs the same region finder once more on the
sign-flipped deviation, puts the regions through the same merge, minimum-duration
and exclusion gates, and stamps four provenance keys::

    n_opposite_events     regions left after the gates (comparable with n_events)
    n_opposite_isolated   those not within OPPOSITE_GUARD_S of a KEPT true-direction
                          region (one that became a row)
    n_opposite_reference  how many kept true-direction regions the run had, frozen
    opposite_guard_s      that guard

`modules._build_warnings` turns them into one advisory. The trigger, in order:

1. ``n_opposite_isolated`` is at least `OPPOSITE_ADVISORY_MIN` and at least
   `OPPOSITE_ADVISORY_FRACTION` of ``n_opposite_reference``: that count is shown;
2. otherwise ``n_opposite_events`` is at least `OPPOSITE_ADVISORY_MIN` and at least
   `OPPOSITE_EVENTS_ADVISORY_FRACTION` of the reference: that count is shown (the
   isolated count collapses when noise fires densely enough for the rows' pads to
   cover the trace).

The sentence is one of four: High-pass on (the count is not a noise estimate
there), nothing in the event direction, at least `OPPOSITE_POLARITY_FACTOR` times
the reference (both name the polarity setting), or the noise sentence with its
caveat. Every number is the run's own, so the sentence does not move when the table
is edited afterwards.

What this file holds in place:

* the pass only COUNTS -- no row and no other provenance value moves, compared with
  a run in which the whole pass is removed, the in-place sign flip included;
* the count is the same finder and the same gates (the gates are repeated in the
  helper, not shared, so the mirrored-trace test is what stops the two copies
  drifting apart);
* the advisory fires on smoothed pink noise, on dense noise firings at a low h, on
  real two-way events (with the caveat) and on a wrong polarity override (with its
  own wording, with and without noise rows in the wrong direction); with High-pass
  on it never blames noise; and it stays silent on white noise, on lobes attached
  to fewer than half the rows and on tables that carry no count at all;
* the isolated count agrees with planted truth to within a factor of 3, is anchored
  on the rows and not on raw firings, the guard reaches both sides of a row to the
  sample, and the advisory is frozen against edits.

Synthetic traces only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import detect_cusum as dc
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.analysis.detect_cusum import OPPOSITE_GUARD_S, detect_events_cusum
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.modules import (
    OPPOSITE_ADVISORY_FRACTION,
    OPPOSITE_ADVISORY_MIN,
    OPPOSITE_EVENTS_ADVISORY_FRACTION,
    OPPOSITE_POLARITY_FACTOR,
    DetectCusum,
    _build_warnings,
    opposite_direction_warning,
    reconcile_result,
)
from nanoscope.analysis.preview_capture import capture_detection_trace
from nanoscope.analysis.truth import synth_trace

NEW_KEYS = (
    "n_opposite_events", "n_opposite_isolated", "n_opposite_reference", "opposite_guard_s",
)


def _opposite(warnings) -> list[str]:
    """The opposite-direction advisory among ``warnings`` (zero or one sentence)."""
    return [str(w) for w in warnings if "opposite-direction" in str(w)]


def _events_trace(seed: int, pink_frac: float):
    """4 s at 50 kHz with 40 planted events long enough for CUSUM's 1 ms floor."""
    rec, _truth = synth_trace(
        duration_s=4.0, n_events=40, pink_frac=pink_frac, seed=seed,
        dwell_range_s=(1.5e-3, 6e-3), depth_frac_range=(0.05, 0.3),
    )
    return rec


def _filtered_pink_noise(duration_s: float = 20.0, seed: int = 0):
    """Noise only, 30% pink: with a 5 kHz low-pass CUSUM fires on it in both directions."""
    rec, _truth = synth_trace(n_events=0, pink_frac=0.3, duration_s=duration_s, seed=seed)
    return rec


# ---------------------------------------------------------------------------
# the pass only counts
# ---------------------------------------------------------------------------
_COUNT_ONLY_CASES = {
    "white": (0.0, {}),
    "pink": (0.3, {}),
    "low_passed": (0.3, dict(filter_cutoff_hz=5000.0, filter_kind="bessel")),
    "merge_off": (0.3, dict(merge_gap_s=None)),
    "excluded": (0.3, dict(exclude_regions=[(1.0, 1.5), (2.7, 2.72)])),
    "provided_baseline": (0.3, dict(baseline=-1000.0)),
}


@pytest.mark.parametrize("case", sorted(_COUNT_ONLY_CASES))
def test_the_opposite_pass_changes_no_row_and_no_other_provenance(monkeypatch, case):
    pink_frac, detect_kw = _COUNT_ONLY_CASES[case]
    rec = _events_trace(seed=31, pink_frac=pink_frac)
    x = rec.data
    before = x.copy()

    df, prov = detect_events_cusum(x, rec.fs, polarity=rec.polarity, **detect_kw)
    assert len(df) > 0
    assert set(NEW_KEYS) <= set(prov)

    # The same run with the WHOLE opposite pass removed: the helper owns the in-place
    # sign flip as well as the count, so replacing it removes both. Everything else
    # must be identical.
    monkeypatch.setattr(dc, "_opposite_direction_counts", lambda *args, **kwargs: {})
    plain_df, plain_prov = detect_events_cusum(x, rec.fs, polarity=rec.polarity, **detect_kw)
    assert not set(NEW_KEYS) & set(plain_prov)

    pd.testing.assert_frame_equal(df, plain_df, check_exact=True)
    drop = {*NEW_KEYS, "detected_at"}
    assert ({k: v for k, v in prov.items() if k not in drop}
            == {k: v for k, v in plain_prov.items() if k not in drop})
    np.testing.assert_array_equal(x, before)


def _float64_two_way_events(seed: int = 11) -> np.ndarray:
    """8 s at 10 kHz, float64, zero-centred, 12 events up and 12 down (4 ms, 8 sigma)."""
    y = np.random.default_rng(seed).normal(0.0, 1.0, 80_000)
    for i in range(12):
        s = 3000 + i * 6000
        y[s:s + 40] += 8.0
        y[s + 2500:s + 2540] -= 8.0
    assert y.dtype == np.float64
    return y


def test_the_opposite_pass_leaves_the_detectors_working_array_alone(monkeypatch):
    """The case in which a stray write would reach the rows AND the caller.

    float64 input, no low-pass and ``baseline=0.0``: the detector makes no copy, so
    the array it characterises the rows on IS the caller's array. The opposite pass
    flips the sign of the standardized deviation in place; if that array ever
    aliased the working one (it must be the detector's own temporary), every row
    would be measured on a mirrored trace and the caller's samples would be
    negated. Compared with a run in which the whole pass is removed.
    """
    fs = 10_000.0
    y = _float64_two_way_events()
    before = y.copy()
    kw = dict(polarity=1, baseline=0.0)

    real = dc._opposite_direction_counts
    seen: dict[str, bool] = {}

    def spy(z, *args, **kwargs):
        seen["z_aliases_the_input"] = bool(np.shares_memory(z, y))
        head = np.array(z[:512], copy=True)
        out = real(z, *args, **kwargs)
        seen["flipped_in_place_by_the_helper"] = bool(np.array_equal(z[:512], -head))
        return out

    monkeypatch.setattr(dc, "_opposite_direction_counts", spy)
    with capture_detection_trace() as capture:
        df, prov = detect_events_cusum(y, fs, **kw)
    current, _baseline = capture.require_single()

    # The premise: the array the rows are measured on is the caller's own.
    assert np.shares_memory(current, y)
    # Both directions had work to do, so the opposite pass really ran over events.
    assert len(df) == 12 and prov["n_opposite_events"] == 12
    # The flip happened, inside the helper, on an array that is not the caller's.
    assert seen == {"z_aliases_the_input": False, "flipped_in_place_by_the_helper": True}
    np.testing.assert_array_equal(y, before)

    # The same run with the whole opposite pass removed, flip included.
    monkeypatch.setattr(dc, "_opposite_direction_counts", lambda *args, **kwargs: {})
    plain_df, plain_prov = detect_events_cusum(y, fs, **kw)
    assert not set(NEW_KEYS) & set(plain_prov)

    pd.testing.assert_frame_equal(df, plain_df, check_exact=True)
    drop = {*NEW_KEYS, "detected_at"}
    assert ({k: v for k, v in prov.items() if k not in drop}
            == {k: v for k, v in plain_prov.items() if k not in drop})
    np.testing.assert_array_equal(y, before)
    # The rows are the events that go UP: measured on the trace as given, not on a
    # mirrored one.
    assert (df["depth_pA"] > 0).all()
    assert all(float(y[s:e].mean()) > 4.0 for s, e in zip(df["start_idx"], df["end_idx"]))


# ---------------------------------------------------------------------------
# the count is the same finder and the same gates
# ---------------------------------------------------------------------------
def _two_way_trace(seed: int = 5, n: int = 120_000) -> np.ndarray:
    """Zero-centred correlated noise with planted excursions in BOTH directions.

    Built so every gate has work to do in either direction: excursions of 6 to 120
    samples (some under the 1 ms / 50-sample floor), every third one followed by a
    second piece 3 to 20 samples later (inside the 25-sample merge window), and
    noise correlated enough for CUSUM to fire on it between them.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n + 8)
    x = np.convolve(white, np.ones(8) / np.sqrt(8.0), mode="valid")[:n]
    x += 0.6 * rng.standard_normal(n)
    pos, k = 2000, 0
    while pos < n - 3000:
        sign = 1.0 if k % 2 == 0 else -1.0
        length = int(rng.integers(6, 120))
        depth = float(rng.uniform(4.0, 9.0))
        x[pos:pos + length] += sign * depth
        if k % 3 == 0:
            gap = int(rng.integers(3, 20))
            x[pos + length + gap:pos + 2 * length + gap] += sign * depth
        pos += int(rng.integers(900, 2200))
        k += 1
    return x.astype(np.float64)


_GATES = {
    "merge_on": {},
    "merge_off": dict(merge_gap_s=None),
    "short_min_dwell": dict(min_dwell_s=1e-4),
    "exclusion": dict(exclude_regions=[(0.4, 0.9), (1.5, 1.52)]),
    "all_three": dict(min_dwell_s=2e-4, merge_gap_s=3e-4, exclude_regions=[(0.4, 0.9)]),
    # A 10 ms window proposes merges the occupancy brake REFUSES (the groups would
    # be mostly open pore), so the helper's copy of the brake has work to do.
    "refused_merge": dict(merge_gap_s=1e-2),
}


@pytest.mark.parametrize("gates", sorted(_GATES))
def test_the_opposite_count_is_the_same_finder_and_gates_on_the_mirrored_trace(gates):
    """``n_opposite_events`` of ``x`` is ``n_events`` of ``-x``, gate for gate.

    With a supplied zero baseline and a fixed polarity, mirroring the trace mirrors
    the deviation exactly (the noise scale is a symmetric MAD), so the true-direction
    pass on ``-x`` IS the opposite-direction pass on ``x``. The helper repeats the
    merge, minimum-duration and exclusion rules instead of sharing them; if either
    copy is edited without the other, these counts part.
    """
    fs = 50_000.0
    x = _two_way_trace()
    kw = _GATES[gates]

    _df, prov = detect_events_cusum(x, fs, polarity=1, baseline=0.0, **kw)
    _mdf, mirrored = detect_events_cusum(-x, fs, polarity=1, baseline=0.0, **kw)

    assert prov["n_opposite_events"] == mirrored["n_events"]
    assert mirrored["n_opposite_events"] == prov["n_events"]

    # The comparison is only worth something if the gates really removed regions.
    assert mirrored["n_events"] < mirrored["n_cusum_firings"]
    if gates == "merge_on":
        assert mirrored["n_merged"] > 0 and mirrored["n_rejected_short"] > 0
    if gates in ("exclusion", "all_three"):
        assert mirrored["n_rejected_excluded"] > 0
    if gates == "refused_merge":
        # The mirrored run's forward path refused merges, so the helper met the same
        # proposals on ``x`` and had to refuse them too to reach the same count.
        # (Here the wider window leaves the count where the default has it -- three
        # more fragments merged, three fewer rejected as short -- so the "moves the
        # count" check below does not apply; the refusals are the point.)
        assert mirrored["n_merges_refused"] > 0 and mirrored["n_merged"] > 0
    elif gates != "merge_on":
        # ...and that each setting moves the opposite count, so a gate the helper
        # ignored would show.
        _d, default = detect_events_cusum(x, fs, polarity=1, baseline=0.0)
        assert prov["n_opposite_events"] != default["n_opposite_events"]


# ---------------------------------------------------------------------------
# when the advisory fires, and when it does not
# ---------------------------------------------------------------------------
def test_white_noise_without_a_filter_reports_no_opposite_detections():
    rec, _truth = synth_trace(duration_s=10.0, n_events=0, pink_frac=0.0, seed=4)
    df, prov = detect_events_cusum(rec.data, rec.fs, polarity=rec.polarity)
    assert prov["n_opposite_events"] == 0
    assert prov["n_opposite_isolated"] == 0
    assert not _opposite(_build_warnings(prov, df))

    # The same with real events on it: clean events leave nothing in the other direction.
    rec = _events_trace(seed=3, pink_frac=0.0)
    df, prov = detect_events_cusum(rec.data, rec.fs, polarity=rec.polarity)
    assert len(df) > 0
    assert prov["n_opposite_events"] == 0
    assert not _opposite(_build_warnings(prov, df))


def test_filtered_pink_noise_raises_the_advisory():
    rec = _filtered_pink_noise()
    df, prov = detect_events_cusum(
        rec.data, rec.fs, polarity=rec.polarity,
        filter_cutoff_hz=5000.0, filter_kind="bessel",
    )
    n_isolated = prov["n_opposite_isolated"]
    assert n_isolated >= OPPOSITE_ADVISORY_MIN
    assert n_isolated <= prov["n_opposite_events"]
    assert len(df) > 0          # the true direction fires on the same noise

    # The reference is the run's own kept true-direction regions: at detection time
    # that is the number of rows.
    n_reference = prov["n_opposite_reference"]
    assert n_reference == len(df) == prov["n_events"]

    [sentence] = _opposite(_build_warnings(prov, df))
    assert sentence.startswith(f"{n_isolated} opposite-direction detection(s) ")
    assert f"against {n_reference} in the event direction." in sentence
    assert "low-pass" not in sentence        # no remedy that only fits one cause
    assert sentence == opposite_direction_warning(n_isolated, n_reference)


def test_the_advisory_wording_is_exact():
    expected = (
        "12 opposite-direction detection(s) against 80 in the event direction. "
        "CUSUM fires that often where no events are expected, so unless events "
        "really run both ways many of the shallow rows here may be noise. Raise h "
        "or k and re-run."
    )
    assert opposite_direction_warning(12, 80) == expected
    # ``lowpass`` is still accepted, so an existing call does not raise, and ignored.
    assert opposite_direction_warning(12, 80, lowpass=False) == expected
    assert opposite_direction_warning(12, 80, lowpass=True) == expected

    # It reads correctly when the opposite count is the larger of the two (the
    # owner's recording: 810 isolated against 707 rows).
    assert opposite_direction_warning(810, 707) == (
        "810 opposite-direction detection(s) against 707 in the event direction. "
        "CUSUM fires that often where no events are expected, so unless events "
        "really run both ways many of the shallow rows here may be noise. Raise h "
        "or k and re-run."
    )

    none = (
        "9 opposite-direction detection(s) and none in the event direction: check "
        "the polarity setting. If it is right, noise alone is firing CUSUM at these "
        "settings."
    )
    assert opposite_direction_warning(9, 0) == none
    for lowpass in (False, True):
        assert opposite_direction_warning(9, 0, lowpass=lowpass) == none

    # Three times the reference or more: the polarity setting is named first, with
    # both numbers. One under three times is still the noise sentence.
    assert OPPOSITE_POLARITY_FACTOR == 3
    assert opposite_direction_warning(85, 9) == (
        "85 opposite-direction detection(s) against 9 in the event direction: check "
        "the polarity setting. If it is right, noise alone is firing CUSUM at these "
        "settings."
    )
    assert opposite_direction_warning(30, 10).endswith(
        "against 10 in the event direction: check the polarity setting. If it is "
        "right, noise alone is firing CUSUM at these settings.")
    assert opposite_direction_warning(29, 10).endswith("Raise h or k and re-run.")

    # High-pass on: one sentence whatever the two numbers are, and it names no noise
    # remedy -- the count is not a noise estimate there.
    high_pass = (
        "{n} opposite-direction detection(s) with High-pass on. The far edge of each "
        "event also fires in the opposite direction there, so this count is not a "
        "noise estimate; turn High-pass off to check for noise firings."
    )
    for n, n_reference in ((12, 80), (9, 0), (85, 9), (810, 707)):
        assert opposite_direction_warning(n, n_reference, highpass=True) == high_pass.format(n=n)
        assert opposite_direction_warning(n, n_reference, highpass=False) == (
            opposite_direction_warning(n, n_reference))
    for phrase in ("may be noise", "noise alone", "Raise h or k", "polarity"):
        assert phrase not in high_pass


def test_the_advisory_needs_five_and_a_tenth_of_the_reference():
    assert (OPPOSITE_ADVISORY_MIN, OPPOSITE_ADVISORY_FRACTION) == (5, 0.10)
    rec = _filtered_pink_noise(duration_s=6.0, seed=2)
    df, prov = detect_events_cusum(
        rec.data, rec.fs, polarity=rec.polarity, filter_cutoff_hz=5000.0,
    )
    assert len(df) >= 51

    def fires(n_isolated: int, n_reference: int, frame=df) -> bool:
        # The count before the 2 ms rule is stamped equal to the isolated one, so the
        # second trigger (next test) cannot be what answers here: it needs half the
        # reference where this one needs a tenth.
        stamped = {**prov, "n_opposite_isolated": n_isolated,
                   "n_opposite_events": n_isolated,
                   "n_opposite_reference": n_reference}
        return bool(_opposite(_build_warnings(stamped, frame)))

    assert not fires(4, 10)      # under the minimum count, however large a share
    assert not fires(4, 0)
    assert fires(5, 10)
    assert fires(5, 50)          # exactly a tenth
    assert not fires(5, 51)      # under a tenth
    assert fires(5, 0)           # nothing in the event direction: the polarity wording

    # The rows on screen are not part of the test: the same answers over a table of
    # ten rows and over an empty one.
    for frame in (df.iloc[:10], df.iloc[:0]):
        assert fires(5, 50, frame) and not fires(5, 51, frame)

    # Either number absent, or unreadable, is "not measured" and says nothing --
    # whatever the count before the 2 ms rule says.
    assert prov["n_opposite_events"] >= OPPOSITE_ADVISORY_MIN
    for key in ("n_opposite_isolated", "n_opposite_reference"):
        without = {k: v for k, v in prov.items() if k != key}
        assert not _opposite(_build_warnings(without, df))
        for unreadable in (None, float("nan"), "many"):
            assert not _opposite(_build_warnings({**prov, key: unreadable}, df))


def test_the_second_trigger_reads_the_count_before_the_guard():
    """Five, and half the reference, of ``n_opposite_events`` -- when the isolated
    count did not already answer. The number shown is then that count; when the
    isolated count answers, it is the one shown, as before."""
    assert OPPOSITE_EVENTS_ADVISORY_FRACTION == 0.5
    frame = pd.DataFrame()

    def said(n_isolated, n_events, n_reference, **more) -> list[str]:
        stamped = {"n_opposite_isolated": n_isolated, "n_opposite_events": n_events,
                   "n_opposite_reference": n_reference, "opposite_guard_s": 0.002, **more}
        return _opposite(_build_warnings(stamped, frame))

    noise = "{n} opposite-direction detection(s) against {ref} in the event direction. "

    # The isolated count is small -- none, or under a tenth -- and the count before
    # the guard is at least five and at least half the reference.
    assert said(0, 5, 10)[0].startswith(noise.format(n=5, ref=10))
    assert said(3, 50, 100)[0].startswith(noise.format(n=50, ref=100))
    assert said(114, 3352, 3397)[0].startswith(noise.format(n=3352, ref=3397))
    assert not said(0, 5, 11)          # under half
    assert not said(3, 49, 100)
    assert not said(0, 4, 8)           # under the minimum count
    assert not said(0, 4, 0)

    # When the isolated count meets the first trigger it is the number shown, even
    # where the count before the guard would also have met the second.
    assert said(20, 90, 100)[0].startswith(noise.format(n=20, ref=100))
    assert said(5, 40, 50)[0].startswith(noise.format(n=5, ref=50))

    # The count shown is the one the wording is chosen on: three times the reference
    # names the polarity setting, and High-pass on has its own sentence.
    assert said(1, 30, 10) == [opposite_direction_warning(30, 10)]
    assert "check the polarity setting" in said(1, 30, 10)[0]
    assert said(0, 5, 10, hp_enabled=True) == [
        opposite_direction_warning(5, 10, highpass=True)]

    # That count absent or unreadable: the second trigger is not met, and the first
    # does not need it.
    for unreadable in (None, float("nan"), "many"):
        assert not said(0, unreadable, 10)
        assert said(5, unreadable, 10)[0].startswith(noise.format(n=5, ref=10))
    without = {"n_opposite_isolated": 0, "n_opposite_reference": 10}
    assert not _opposite(_build_warnings(without, frame))


def _events_with_lobes(side: str, offset: int, n_lobes: int, n_rows: int = 12) -> np.ndarray:
    """``n_rows`` events of 3 ms at 8 sigma, the first ``n_lobes`` of them with a
    6 ms lobe of the opposite sign ``offset`` samples AFTER the event ends, or
    ending ``offset`` samples BEFORE it starts. 10 kHz: guard = 20 samples."""
    y = np.random.default_rng(7).normal(0.0, 1.0, 3000 + n_rows * 6500)
    for i in range(n_rows):
        s = 3000 + i * 6500
        y[s:s + 30] -= 8.0
        if i < n_lobes:
            if side == "after":
                y[s + 30 + offset:s + 90 + offset] += 8.0
            else:
                y[s - 60 - offset:s - offset] += 8.0
    return y


@pytest.mark.parametrize("side, attached_offset", [("after", 0), ("before", 5)])
def test_a_lobe_attached_to_an_event_is_not_counted_as_isolated(side, attached_offset):
    """An opposite-sign lobe within the guard of an event is not noise firing on its
    own -- it is overshoot beside the event -- however long it lasts. The guard
    reaches both ways: a lobe that starts within 2 ms AFTER a row, and a lobe that
    ends within 2 ms BEFORE one.

    "An event" is a ROW: the guard is anchored on the regions the run kept. The
    events here are 3 ms at 8 sigma so that all twelve are kept. (At 1.5 ms and 30
    sigma the frozen level closes about one in eight early, the fragment falls
    under the minimum dwell, and the lobe beside that rejected firing is then
    counted -- correctly, since nothing in the table is there to attach it to.)

    Five of the twelve rows carry a lobe: enough for the first trigger if they were
    counted as isolated, and under the half of the reference the second trigger
    needs (see the next test for what happens at half).
    """
    fs = 10_000.0                      # guard = 20 samples, minimum dwell = 10

    attached_df, attached = detect_events_cusum(
        _events_with_lobes(side, attached_offset, n_lobes=5), fs, polarity=-1, baseline=0.0)
    assert len(attached_df) == attached["n_opposite_reference"] == 12
    assert attached["opposite_guard_s"] == OPPOSITE_GUARD_S == 2e-3
    assert attached["n_opposite_events"] == 5 == OPPOSITE_ADVISORY_MIN
    assert attached["n_opposite_isolated"] == 0
    assert not _opposite(_build_warnings(attached, attached_df))

    # The guard is what removed them: the same lobes 5 ms away from each event are
    # outside it, are counted, and raise the advisory.
    apart_df, apart = detect_events_cusum(
        _events_with_lobes(side, 50, n_lobes=5), fs, polarity=-1, baseline=0.0)
    assert len(apart_df) == 12
    assert apart["n_opposite_isolated"] == apart["n_opposite_events"] == 5
    [sentence] = _opposite(_build_warnings(apart, apart_df))
    assert sentence == opposite_direction_warning(5, 12)


@pytest.mark.parametrize("n_lobes", [6, 12])
def test_lobes_beside_half_the_rows_meet_the_second_trigger(n_lobes):
    """What the second trigger costs, stated so it is not a surprise.

    It reads the count BEFORE the 2 ms rule, because the isolated count collapses
    when noise fires densely. Attached lobes are in that count. So a run in which
    at least half the rows carry an opposite lobe longer than the minimum dwell
    raises the noise sentence although every one of those lobes is attached to a
    row and ``n_opposite_isolated`` is 0. The sentence's caveat ("unless events
    really run both ways") is all that covers it. Five lobes beside twelve rows
    stay unsaid (the test above).
    """
    fs = 10_000.0
    df, prov = detect_events_cusum(
        _events_with_lobes("after", 0, n_lobes=n_lobes), fs, polarity=-1, baseline=0.0)
    assert len(df) == prov["n_opposite_reference"] == 12
    assert prov["n_opposite_isolated"] == 0
    assert prov["n_opposite_events"] == n_lobes
    [sentence] = _opposite(_build_warnings(prov, df))
    assert sentence == opposite_direction_warning(n_lobes, 12)
    assert "unless events really run both ways" in sentence


def test_real_events_in_both_directions_are_reported_with_the_caveat():
    fs = 10_000.0
    y = np.random.default_rng(8).normal(100.0, 1.0, 80_000)
    for i in range(12):
        s = 3000 + i * 6000
        y[s:s + 80] -= 12.0              # a blockade ...
        y[s + 3000:s + 3080] += 12.0     # ... and, 300 ms later, an enhancement
    df, prov = detect_events_cusum(y, fs, polarity=-1)
    assert len(df) >= 10
    assert prov["n_opposite_isolated"] >= OPPOSITE_ADVISORY_MIN

    [sentence] = _opposite(_build_warnings(prov, df))
    # It is a count with a caveat, not a verdict: these opposite detections are real.
    assert "unless events really run both ways" in sentence
    assert "may be noise" in sentence
    assert "%" not in sentence                               # a count, never a rate
    assert f"against {prov['n_opposite_reference']} in the event direction" in sentence


def test_a_wrong_polarity_override_says_to_check_polarity():
    rec = _events_trace(seed=3, pink_frac=0.0)
    df, prov = detect_events_cusum(rec.data, rec.fs, polarity=-rec.polarity)
    assert len(df) == 0
    assert prov["n_opposite_isolated"] >= OPPOSITE_ADVISORY_MIN

    assert prov["n_opposite_reference"] == 0

    [sentence] = _opposite(_build_warnings(prov, df))
    assert sentence == (
        f"{prov['n_opposite_isolated']} opposite-direction detection(s) and none in "
        "the event direction: check the polarity setting. If it is right, noise alone "
        "is firing CUSUM at these settings."
    )


@pytest.mark.parametrize("seed, pink_frac", [(5, 0.3), (7, 0.2)])
def test_a_wrong_polarity_override_on_pink_noise_still_says_to_check_polarity(seed, pink_frac):
    """On correlated noise the wrong direction is never empty: noise fires a few rows
    there. The real events are all in the opposite direction, so "raise h or k" is
    the wrong remedy and the polarity setting is what the sentence must name.
    (Measured on these two traces: 85 opposite against 9, and 79 against 4.)
    """
    rec, planted = synth_trace(
        duration_s=8.0, n_events=80, pink_frac=pink_frac, seed=seed,
        dwell_range_s=(1.5e-3, 6e-3), depth_frac_range=(0.05, 0.3),
    )
    df, prov = detect_events_cusum(rec.data, rec.fs, polarity=-rec.polarity)
    n_isolated = prov["n_opposite_isolated"]
    n_reference = prov["n_opposite_reference"]

    # A few noise rows in the (wrong) event direction, so this is not the
    # "none in the event direction" case ...
    assert 0 < n_reference == len(df) < 15
    # ... and the opposite direction holds about what was planted.
    assert n_isolated >= OPPOSITE_POLARITY_FACTOR * n_reference
    assert 0.8 * len(planted) <= n_isolated <= 1.2 * len(planted)

    [sentence] = _opposite(_build_warnings(prov, df))
    assert sentence == (
        f"{n_isolated} opposite-direction detection(s) against {n_reference} in the "
        "event direction: check the polarity setting. If it is right, noise alone is "
        "firing CUSUM at these settings."
    )
    assert "Raise h or k" not in sentence and "shallow rows" not in sentence

    # With the right polarity the same trace gives no polarity sentence.
    right_df, right = detect_events_cusum(rec.data, rec.fs, polarity=rec.polarity)
    assert len(right_df) >= 0.9 * len(planted)
    assert not any("polarity setting" in s for s in _opposite(_build_warnings(right, right_df)))


def test_dense_noise_firings_at_a_low_h_still_raise_the_advisory():
    """Lowering h must not make the advisory disappear while the false positives rise.

    Noise only (30% pink, 5 kHz low-pass) at k 0.5 / h 3 with a 0.1 ms minimum
    dwell: the run keeps over three hundred rows a second, their 2 ms pads cover
    nearly all of the trace, and only a few percent of the opposite-direction
    regions are "isolated" although there are as many of them as rows. The first
    trigger alone was silent here. (Measured: 1366 rows, 1325 opposite, 53 isolated.)
    """
    rec, _truth = synth_trace(duration_s=4.0, n_events=0, pink_frac=0.3, seed=0)
    df, prov = detect_events_cusum(
        rec.data, rec.fs, polarity=rec.polarity,
        cusum_k=0.5, cusum_h=3.0, min_dwell_s=1e-4, filter_cutoff_hz=5000.0,
    )
    n_reference = prov["n_opposite_reference"]
    n_isolated = prov["n_opposite_isolated"]
    n_before_guard = prov["n_opposite_events"]
    assert n_reference == len(df) > 1000

    # The isolated count is far under a tenth of the reference: the first trigger
    # is not met ...
    assert OPPOSITE_ADVISORY_MIN <= n_isolated < OPPOSITE_ADVISORY_FRACTION * n_reference
    # ... while noise fires the opposite direction about as often as it made rows.
    assert 0.9 * n_reference <= n_before_guard <= 1.1 * n_reference

    [sentence] = _opposite(_build_warnings(prov, df))
    assert sentence == opposite_direction_warning(n_before_guard, n_reference)
    assert sentence.startswith(
        f"{n_before_guard} opposite-direction detection(s) against {n_reference} in "
        "the event direction. CUSUM fires that often where no events are expected")

    # At a moderate h on the same trace the isolated count still answers, and is
    # the number shown, as before.
    df, prov = detect_events_cusum(
        rec.data, rec.fs, polarity=rec.polarity,
        cusum_k=0.75, cusum_h=5.0, min_dwell_s=1e-4, filter_cutoff_hz=5000.0,
    )
    assert prov["n_opposite_isolated"] >= OPPOSITE_ADVISORY_FRACTION * prov["n_opposite_reference"]
    [sentence] = _opposite(_build_warnings(prov, df))
    assert sentence == opposite_direction_warning(
        prov["n_opposite_isolated"], prov["n_opposite_reference"])
    assert prov["n_opposite_isolated"] < prov["n_opposite_events"]


# ---------------------------------------------------------------------------
# the keys travel
# ---------------------------------------------------------------------------
def test_the_keys_exist_on_the_empty_path_and_survive_json():
    rec, _truth = synth_trace(duration_s=4.0, n_events=0, pink_frac=0.0, seed=4)
    df, prov = detect_events_cusum(rec.data, rec.fs, polarity=rec.polarity)
    assert df.empty and prov["n_events"] == 0
    assert {k: prov[k] for k in NEW_KEYS} == {
        "n_opposite_events": 0, "n_opposite_isolated": 0, "n_opposite_reference": 0,
        "opposite_guard_s": 0.002,
    }

    busy = _filtered_pink_noise(duration_s=6.0, seed=2)
    _df, busy_prov = detect_events_cusum(
        busy.data, busy.fs, polarity=busy.polarity, filter_cutoff_hz=5000.0,
    )
    for stamped in (prov, busy_prov):
        assert type(stamped["n_opposite_events"]) is int
        assert type(stamped["n_opposite_isolated"]) is int
        assert type(stamped["n_opposite_reference"]) is int
        assert type(stamped["opposite_guard_s"]) is float
        # Frozen at detection time, when it is the number of rows.
        assert stamped["n_opposite_reference"] == stamped["n_events"]
        # None of them may look like a baseline or noise stamp: those prefixes are
        # rewritten wholesale by a re-measure.
        assert not any(k.startswith(("baseline_", "noise_")) for k in NEW_KEYS)
        back = json.loads(json.dumps({k: stamped[k] for k in NEW_KEYS}, allow_nan=False))
        assert back == {k: stamped[k] for k in NEW_KEYS}

    # They are not terms of the audit identity, which still balances without them.
    assert (busy_prov["n_cusum_firings"] - busy_prov["n_merged"]
            - busy_prov["n_rejected_short"] - busy_prov["n_rejected_excluded"]
            == busy_prov["n_events"])


def test_the_advisory_is_absent_without_the_key_and_survives_reconcile(context_factory):
    rec = _filtered_pink_noise(duration_s=6.0, seed=2)
    res = DetectCusum().run(context_factory(
        rec.data, fs=rec.fs,
        params=dict(polarity=int(rec.polarity), filter_cutoff_hz=5000.0),
    ))
    prov = res.payload.provenance
    [sentence] = _opposite(res.warnings)
    assert sentence.startswith(f"{prov['n_opposite_isolated']} opposite-direction detection(s) ")

    # A table from before the count existed (or a Schmitt table) has no key, and an
    # absent key is "not measured", never "zero" or "many".
    older = {k: v for k, v in prov.items() if k not in NEW_KEYS}
    assert not _opposite(_build_warnings(older, res.payload.df))

    # A merge, a hand-added event and a re-measure all rebuild the warnings through
    # reconcile_result; the sentence is re-derived from provenance, so it is kept.
    assert _opposite(reconcile_result(res).warnings) == [sentence]


def test_the_advisory_is_frozen_when_the_table_is_edited(context_factory):
    """The sentence is about the DETECTION, so an edit to the table cannot change it.

    Its trigger and both of its numbers are stamped by the run. Rebuilding the
    warnings over half the rows -- what a merge, a hand edit or a re-measure does
    through `reconcile_result` -- gives the identical sentence; and a run whose
    count was not material stays silent however few rows are left.
    """
    rec = _filtered_pink_noise(duration_s=6.0, seed=2)
    res = DetectCusum().run(context_factory(
        rec.data, fs=rec.fs,
        params=dict(polarity=int(rec.polarity), filter_cutoff_hz=5000.0),
    ))
    prov = res.payload.provenance
    df = res.payload.df
    n_rows = len(df)
    assert prov["n_opposite_reference"] == n_rows >= 20
    [sentence] = _opposite(res.warnings)
    assert sentence == opposite_direction_warning(
        prov["n_opposite_isolated"], prov["n_opposite_reference"])

    half = df.iloc[: n_rows // 2].copy()
    edited = reconcile_result(res, EventSet(half, dict(prov)))
    # The row counts did follow the edit ...
    assert len(edited.payload.df) == n_rows // 2
    assert edited.payload.provenance["n_events"] == n_rows // 2
    # ... the frozen reference did not, and neither did the sentence.
    assert edited.payload.provenance["n_opposite_reference"] == n_rows
    assert _opposite(edited.warnings) == [sentence]
    assert _opposite(_build_warnings(dict(prov), half)) == [sentence]
    assert _opposite(_build_warnings(dict(prov), df.iloc[:0])) == [sentence]
    # Nothing was written back into the caller's result on the way.
    assert res.payload.provenance["n_events"] == n_rows

    # An edit cannot switch the sentence ON either: 5 against 60 is under a tenth
    # (and, before the 2 ms rule, 20 against 60 is under a half), and stays unsaid
    # over a table cut down to 10 rows (5 of 10 would be half).
    quiet = {**prov, "n_opposite_isolated": 5, "n_opposite_events": 20,
             "n_opposite_reference": 60}
    assert not _opposite(_build_warnings(quiet, df))
    assert not _opposite(_build_warnings(quiet, df.iloc[:10]))
    assert not _opposite(
        reconcile_result(res, EventSet(df.iloc[:10].copy(), dict(quiet))).warnings)


# ---------------------------------------------------------------------------
# the count against planted truth, and what the guard is anchored on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [41, 42, 43])
def test_the_isolated_count_tracks_the_rows_that_match_no_planted_event(seed):
    """The estimator against truth, on the case the advisory exists for.

    One-way planted events of 2 to 6 ms on correlated noise (30% pink, 5 kHz Bessel
    low-pass): every row that overlaps no planted event is a noise firing in the
    event direction. The opposite direction holds no events at all, so
    ``n_opposite_isolated`` estimates the same thing from the other side. The two
    agree to within a factor of 3 (measured on these seeds: 0.77x to 1.10x).
    """
    rec, planted = synth_trace(
        duration_s=6.0, n_events=40, pink_frac=0.3, seed=seed,
        dwell_range_s=(2e-3, 6e-3), depth_frac_range=(0.05, 0.3),
    )
    assert planted["dwell_s"].min() >= 2e-3
    df, prov = detect_events_cusum(
        rec.data, rec.fs, polarity=rec.polarity,
        filter_cutoff_hz=5000.0, filter_kind="bessel",
    )
    s = df["start_idx"].to_numpy()[:, None]
    e = df["end_idx"].to_numpy()[:, None]
    overlaps = (s < planted["end_idx"].to_numpy()[None, :]) & (
        e > planted["start_idx"].to_numpy()[None, :])
    assert overlaps.any(axis=0).all()                 # every planted event was found
    n_noise_rows = int((~overlaps.any(axis=1)).sum())
    n_isolated = prov["n_opposite_isolated"]

    assert n_noise_rows >= 5 and n_isolated >= 5
    assert n_noise_rows / 3 <= n_isolated <= 3 * n_noise_rows
    assert _opposite(_build_warnings(prov, df))


def test_short_firings_the_minimum_dwell_rejects_do_not_blank_the_count():
    """The guard is anchored on the KEPT regions -- the rows -- not on raw firings.

    Two traces, identical except that one carries a 0.2 ms spike in the event
    direction every 4 ms. Each spike is a raw CUSUM firing that the 1 ms minimum
    dwell rejects, and with a 2 ms guard on each side they would blank the whole
    trace between them: every excursion the other way starts 0.4 ms after one spike
    and ends 0.4 ms before the next. Anchored on raw firings the isolated count
    would be 0 exactly when noise fires densely. Anchored on the rows it is the
    same with and without the spikes.
    """
    fs = 50_000.0                       # guard = 100 samples, minimum dwell = 50
    period = 2000
    blocks = 60
    kept_blocks = (10, 30, 50)
    guard = int(round(OPPOSITE_GUARD_S * fs))
    assert guard == 100

    def trace(spikes: bool) -> tuple[np.ndarray, int, int]:
        y = np.random.default_rng(17).normal(0.0, 1.0, blocks * period)
        n_spikes = n_free = 0
        for b in range(blocks):
            o = b * period
            kept = b in kept_blocks
            if spikes:
                for off in range(100, period, 200):
                    if kept and off in (300, 500):
                        continue                      # room for the long event below
                    y[o + off:o + off + 10] += 30.0   # 0.2 ms: fires, then too short
                    n_spikes += 1
            if kept:
                y[o + 300:o + 450] += 4.0             # 3 ms: becomes a row ...
                y[o + 470:o + 620] -= 4.0             # ... with a lobe 0.4 ms after it
            else:
                y[o + 330:o + 480] -= 4.0             # 3 ms the other way, on its own
                n_free += 1
        return y, n_spikes, n_free

    quiet, _none, n_free = trace(spikes=False)
    spiky, n_spikes, _same = trace(spikes=True)
    assert n_free == blocks - len(kept_blocks) and n_spikes > 500
    assert 20 < guard        # each free excursion is 20 samples from a spike on each side

    _qdf, without = detect_events_cusum(quiet, fs, polarity=1, baseline=0.0)
    df, with_spikes = detect_events_cusum(spiky, fs, polarity=1, baseline=0.0)

    # Every spike fired and was rejected by the minimum dwell; only the three long
    # events became rows, and they are the reference.
    assert with_spikes["n_rejected_short"] == n_spikes
    assert with_spikes["n_cusum_firings"] == n_spikes + len(kept_blocks)
    assert len(df) == with_spikes["n_events"] == len(kept_blocks)
    assert with_spikes["n_opposite_reference"] == len(kept_blocks)
    assert without["n_opposite_reference"] == len(kept_blocks)

    # The dense raw firings did not reduce the count ...
    assert with_spikes["n_opposite_events"] == without["n_opposite_events"] == n_free + 3
    assert with_spikes["n_opposite_isolated"] == without["n_opposite_isolated"] == n_free
    # ... and the guard still works where it should: the three lobes attached to
    # rows are the only ones left out.
    assert (with_spikes["n_opposite_events"] - with_spikes["n_opposite_isolated"]
            == len(kept_blocks))


def test_the_guard_reaches_two_milliseconds_each_side_of_a_row_to_the_sample(monkeypatch):
    """The overlap test at its edges, with the regions given instead of found.

    The finder is replaced by a list of opposite-direction regions placed against
    five kept regions, so each comparison in the guard is pinned to the sample.
    At 1 kHz the guard is 2 samples: a kept region ``[a, b)`` is padded to
    ``[a - 2, b + 2)`` and an opposite region ``[s, e)`` is attached iff the two
    half-open intervals share a sample.
    """
    fs = 1000.0
    assert int(round(OPPOSITE_GUARD_S * fs)) == 2
    kept_starts = np.array([1000, 3000, 5000, 7000, 9000], dtype=np.int64)
    kept_ends = kept_starts + 100                       # exclusive, as the app has them

    # (start, end EXCLUSIVE, attached?)
    placed = [
        (900, 998, False),      # ends exactly where the pad before a row begins
        (2900, 2999, True),     # one sample into the pad before a row
        (5102, 5200, False),    # starts exactly where the pad after a row ends
        (7101, 7200, True),     # one sample into the pad after a row
        (8000, 8100, False),    # nowhere near a row
        (9010, 9020, True),     # inside a row
    ]
    # The finder's own convention is INCLUSIVE ends; the helper converts.
    found = dc.CusumRegions([(s, e - 1) for s, e, _near in placed], 0)
    monkeypatch.setattr(dc, "cusum_onset_recovery_regions", lambda z, k, h, **kwargs: found)

    z = np.zeros(10_000)
    counts = dc._opposite_direction_counts(
        z, 0.75, 28.0, fs=fs, merge_gap_s=None, min_samples=1, ex=[],
        forward_starts=kept_starts, forward_ends=kept_ends, cancel=None,
    )
    assert counts == {
        "n_opposite_events": len(placed),
        "n_opposite_isolated": sum(not near for _s, _e, near in placed),
        "n_opposite_reference": len(kept_starts),
        "opposite_guard_s": OPPOSITE_GUARD_S,
    }
    assert counts["n_opposite_isolated"] == 3

    # One at a time, so a wrong answer names its region.
    for s, e, near in placed:
        one = dc.CusumRegions([(s, e - 1)], 0)
        monkeypatch.setattr(dc, "cusum_onset_recovery_regions", lambda z, k, h, one=one, **kw: one)
        counts = dc._opposite_direction_counts(
            np.zeros(10_000), 0.75, 28.0, fs=fs, merge_gap_s=None, min_samples=1, ex=[],
            forward_starts=kept_starts, forward_ends=kept_ends, cancel=None,
        )
        assert counts["n_opposite_isolated"] == (0 if near else 1), (s, e)

    # The kept regions were only read.
    np.testing.assert_array_equal(kept_starts, [1000, 3000, 5000, 7000, 9000])
    np.testing.assert_array_equal(kept_ends, kept_starts + 100)


@pytest.mark.parametrize("hp", [False, True])
def test_the_module_adapter_stamps_the_count_with_high_pass_on_and_off(context_factory, hp):
    fs = 20_000.0
    x = np.random.default_rng(923).normal(100, 0.5, 40_000).astype(np.float32)
    for s, e in [(4000, 4200), (12000, 12500), (21000, 21200), (30000, 30100)]:
        x[s:e] -= 30
    params = dict(polarity=-1, min_dwell_s=0.0002,
                  filter_cutoff_hz=3000.0, hp_enabled=hp, hp_cutoff_hz=50.0)

    with capture_detection_trace() as capture:
        res = DetectCusum().run(context_factory(x, fs=fs, params=params))

    # One detector call, so one published trace: the count is a second pass of the
    # region finder, not a second detection.
    assert capture.count == 1
    prov = res.payload.provenance
    assert bool(prov.get("highpass_localization_only")) == hp
    assert prov["hp_enabled"] is hp
    for key in NEW_KEYS:
        assert key in prov
    assert 0 <= prov["n_opposite_isolated"] <= prov["n_opposite_events"]
    # The frozen reference is the run's own kept regions on the split path too: the
    # rows are re-measured there, not re-found, so it is the number of rows.
    assert type(prov["n_opposite_reference"]) is int
    assert prov["n_opposite_reference"] == len(res.payload.df)
    assert prov["opposite_guard_s"] == OPPOSITE_GUARD_S
    assert DetectCusum.version == "1.0"

    said = _opposite(res.warnings)
    if not hp:
        # Four clean events and nothing the other way: nothing is said.
        assert len(res.payload.df) == 4
        assert prov["n_opposite_events"] == 0
        assert not said
    else:
        # The same four events through a 50 Hz high-pass: the table is longer (the
        # tails), the far edges fire the other way, and the isolated count meets the
        # trigger. Nothing here is noise, and the advisory does not say it is.
        assert prov["n_opposite_isolated"] >= OPPOSITE_ADVISORY_MIN
        assert said == [opposite_direction_warning(
            prov["n_opposite_isolated"], prov["n_opposite_reference"], highpass=True)]
        assert "with High-pass on." in said[0] and "not a noise estimate" in said[0]
        assert "may be noise" not in said[0] and "Raise h or k" not in said[0]


def _clean_one_way_events(seed: int = 77, duration_s: float = 6.0, fs: float = 50_000.0):
    """White noise (5 pA) around -1000 pA with one-way events of 3 to 8 ms, 8 sigma.

    Nothing here is noise firing: with High-pass off a CUSUM run finds the planted
    events and nothing at all in the opposite direction.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    x = rng.normal(-1000.0, 5.0, n)
    n_planted = 0
    pos = int(0.2 * fs)
    while pos < n - int(0.3 * fs):
        length = int(rng.uniform(3.0, 8.0) * 1e-3 * fs)
        x[pos:pos + length] += 40.0          # the pore current is negative: an occlusion goes UP
        n_planted += 1
        pos += length + int(rng.uniform(0.15, 0.35) * fs)
    return x.astype(np.float32), n_planted


@pytest.mark.parametrize("polarity", [1, None])
def test_high_pass_on_never_blames_noise_for_the_far_edge_of_real_events(
        context_factory, polarity):
    """With Preprocess > High-pass on, the opposite count is not a noise estimate.

    The count is taken on the high-passed candidate trace. There the far edge of
    each real event is an excursion the other way, and the kept region covers only
    the event's onset, so the 2 ms guard does not attach it. Clean one-way events on
    white noise -- no noise firing anywhere -- therefore stamp a material isolated
    count. The advisory must not call that noise or offer "raise h or k"; it says
    what the count is and how to get a noise check. On Auto polarity the high-pass
    vote is reversed and the run repeats localisation with the physical sign: the
    stamped keys are that second call's.
    """
    fs = 50_000.0
    x, n_planted = _clean_one_way_events(fs=fs)
    assert n_planted >= 20

    off = DetectCusum().run(context_factory(x, fs=fs, params=dict(polarity=polarity)))
    off_prov = off.payload.provenance
    assert off_prov["hp_enabled"] is False
    assert len(off.payload.df) == n_planted == off_prov["n_opposite_reference"]
    assert off_prov["n_opposite_events"] == 0          # clean: nothing the other way
    assert not _opposite(off.warnings)

    on = DetectCusum().run(context_factory(
        x, fs=fs, params=dict(polarity=polarity, hp_enabled=True, hp_cutoff_hz=100.0)))
    prov = on.payload.provenance
    df = on.payload.df
    # How a run's provenance says High-pass was on.
    assert prov["hp_enabled"] is True and prov["highpass_localization_only"] is True
    assert prov["polarity"] == 1
    assert prov["polarity_retry_after_highpass_mismatch"] is (polarity is None)
    # The frozen reference is the number of rows, on the split path and after a retry.
    assert prov["n_opposite_reference"] == len(df) == prov["n_events"]
    assert len(df) >= n_planted

    # The far edges are there, isolated, and would have met the trigger.
    n_isolated = prov["n_opposite_isolated"]
    assert n_isolated >= OPPOSITE_ADVISORY_MIN
    assert n_isolated >= OPPOSITE_ADVISORY_FRACTION * prov["n_opposite_reference"]
    assert n_isolated >= 0.5 * n_planted

    [sentence] = _opposite(on.warnings)
    assert sentence == (
        f"{n_isolated} opposite-direction detection(s) with High-pass on. The far edge "
        "of each event also fires in the opposite direction there, so this count is "
        "not a noise estimate; turn High-pass off to check for noise firings."
    )
    # No noise sentence anywhere among the run's advisories.
    for warning in on.warnings:
        for phrase in ("may be noise", "noise alone is firing", "Raise h or k"):
            assert phrase not in str(warning)

    # It is the stamp that decides: the same numbers without it are the noise
    # sentence (which is what this run used to say), and a table edit keeps the
    # High-pass wording.
    [without_stamp] = _opposite(_build_warnings({**prov, "hp_enabled": False}, df))
    assert without_stamp == opposite_direction_warning(n_isolated, len(df))
    assert "may be noise" in without_stamp
    assert _opposite(reconcile_result(on).warnings) == [sentence]
    assert _opposite(_build_warnings(dict(prov), df.iloc[:3])) == [sentence]


def test_cancel_reaches_the_opposite_pass(monkeypatch):
    rec = _events_trace(seed=3, pink_frac=0.0)
    core = dc.cusum_onset_recovery_regions
    entered: list[tuple[object, np.ndarray]] = []
    finished: list[int] = []

    def recorder(z, k, h, **kwargs):
        entered.append((kwargs.get("cancel"), np.array(z[:64], copy=True)))
        out = core(z, k, h, **kwargs)
        finished.append(len(out.regions))
        return out

    monkeypatch.setattr(dc, "cusum_onset_recovery_regions", recorder)

    def never():
        return False

    detect_events_cusum(rec.data, rec.fs, polarity=rec.polarity, cancel=never)
    assert len(entered) == 2
    assert entered[0][0] is never and entered[1][0] is never
    # The second pass ran on the first one's deviation, sign-flipped.
    np.testing.assert_array_equal(entered[1][1], -entered[0][1])

    with pytest.raises(OperationCancelled):
        detect_events_cusum(rec.data, rec.fs, polarity=rec.polarity, cancel=lambda: True)

    # A cancel that arrives after the true-direction pass stops the run before the
    # opposite pass begins.
    entered.clear()
    finished.clear()
    with pytest.raises(OperationCancelled):
        detect_events_cusum(
            rec.data, rec.fs, polarity=rec.polarity, cancel=lambda: len(finished) >= 1)
    assert len(entered) == 1
