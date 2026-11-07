"""RP+BL through the real application: the two-sided guard, evidence and event table.

The -600 mV recording contains a roughly 521 ms clog near t=37.13 s. The
1 s / 300 ms and 2 s / 500 ms windows tie on median intrusion, but the shorter
window rides into that long state. The test retains the guarded offer, explicit
manual override and visible explanation of both criteria.

Automatic masks, robust flank anchors and the long-state depth gate are versioned.
Compare the displayed ladder with fresh scientific measurements and the event table
with the direct detection entry point, rather than pinning results from an older
policy. Independent quiet raw-signal flanks establish the physical clog depth:
the offered window must preserve it and the rejected window must attenuate it.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.analysis import baseline as B
from .baseline_corpus_contracts import assert_detection_matches_reference

FS = 50_000.0
APO_600 = pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0006_Process_-600 mV.abf")

#: What the median-only two-sided search chose (it "ties at 0.0 and is shorter"),
#: and what the app offers now that the tie is judged on the state the median cannot
#: see. The ridden window stays typable -- rejected as an OFFER, not removed.
RIDDEN = (1.0, 0.3)
OFFERED = (2.0, 0.5)



def api(obj, name):
    """``getattr`` as an ASSERTION, so a package without the fix fails on the CLAIM."""
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: the "
        "guarded two-sided ranking is not reachable from the screen, which is the "
        "defect this file is about")
    return getattr(obj, name)


def where_from():
    """Print AND CHECK the package under test.

    A revert run from inside the repo silently tests the LIVE tree -- PYTHONPATH does
    not beat cwd when cwd holds the package -- so the revert runner sets
    ``NANOSCOPE_EXPECT_TREE`` and this fails rather than passing against the wrong
    package and reporting a green revert.
    """
    import nanoscope
    p = pathlib.Path(nanoscope.__file__).resolve()
    print(f"\nnanoscope resolved from: {p}")
    want = os.environ.get("NANOSCOPE_EXPECT_TREE")
    if want:
        root = pathlib.Path(want).resolve()
        assert str(p).startswith(str(root)), (
            f"nanoscope resolved from {p}, not from {root}: this run is testing a "
            "different tree than it was pointed at")
    return p


def _drain(qapp, qtbot, ms=150):
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)


def _detect(qapp, qtbot, ap, ctrl):
    dp = ap.detection_panel
    dp.show_detection_tab()
    _drain(qapp, qtbot, 100)
    with qtbot.waitSignal(ctrl.finished, timeout=1_800_000):
        dp.detect_button.click()
    _drain(qapp, qtbot, 400)
    return ap.current_events_df(), ap.current_detection().payload.provenance


def _clog_coverage(df):
    """Detected occupancy of the independently identified 37.13–37.65 s episode."""
    original_starts = np.asarray(df["t_start_s"], float)
    original_ends = np.asarray(df["t_start_s"] + df["dwell_s"], float)
    starts = np.maximum(original_starts, 37.13)
    ends = np.minimum(original_ends, 37.65)
    inside = ends > starts
    rows = df.loc[inside]
    starts, ends = starts[inside], ends[inside]
    assert len(rows) > 0, "no detected samples in the known clog interval"
    # Schmitt intervals cannot overlap; summing these clipped durations is their
    # union, not a double-counted event total that could hide lost coverage.
    assert np.all(starts[1:] >= ends[:-1])
    durations = ends - starts
    depth = float(np.average(np.asarray(rows["depth_pA"], float), weights=durations))
    # Preserve the original onset/offset for the separate edge assertions;
    # clipping must not disguise an event that starts too early or ends too late.
    return (float(durations.sum()), float(original_starts[inside][0]),
            float(original_ends[inside][-1]), depth)


@pytest.mark.corpus
@pytest.mark.skipif(not APO_600.exists(), reason="the -600 mV recording is not here")
def test_the_six_hundred_millivolt_one_click_rejects_the_window_that_rides(
        qapp, qtbot):
    """The guard must report the real long-state harm behind a clean median."""
    where_from()
    from nanoscope.core.types import ChannelKind
    from nanoscope.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1600, 900)
    w.show()
    w._inspector.show()
    w._inspector_tabs.setCurrentWidget(w._analysis_panel)
    _drain(qapp, qtbot, 50)

    with qtbot.waitSignal(w._session.documentOpened, timeout=300_000):
        w.open_paths([str(APO_600)])
    _drain(qapp, qtbot, 50)
    doc_id = w._session.doc_id_for_source(str(APO_600.resolve()))
    assert doc_id is not None
    channel = next(c.index for c in w._session.document(doc_id).meta.channels
                   if c.kind is ChannelKind.CURRENT)
    assert channel == 2, "the current channel on these recordings is ch2 (IN15)"
    w._on_selection(doc_id, channel)
    _drain(qapp, qtbot, 50)

    ap = w._analysis_panel
    bp = ap.baseline_panel
    ctrl = w._analysis_controller

    with qtbot.waitSignal(ctrl.finished, timeout=1_800_000):
        bp.probe_button.click()
    _drain(qapp, qtbot, 300)

    pr = bp.proposal()
    assert pr is not None
    got = (round(float(pr.config.rough_s), 6), round(float(pr.config.base_s), 6))
    assert got == OFFERED, (f"the one-click window is {got}", got)
    assert got != RIDDEN, (
        "the proposal is still the window whose baseline rides into the 521 ms "
        "clog: the ranking's tie-break is still judging on the blind median alone")

    # ---- the ladder: the riding rung is ESTIMATED and rejected, not skipped ---
    ranking = api(pr, "ranking")
    windows = [(c.rough_s, c.base_s) for c in ranking]
    assert windows == [RIDDEN, OFFERED], windows
    by_window = {(c.rough_s, c.base_s): c for c in ranking}
    assert by_window[RIDDEN].fraction == 0.0, (
        "1 s / 300 ms no longer ties on the median, so this recording no longer "
        "demonstrates the tie the guard exists to judge", by_window[RIDDEN].to_dict())
    assert api(by_window[RIDDEN], "rides") is True
    # Automatic masks, flank anchors and the separate long-state depth gate are
    # versioned. Preserve the measured guard, not the old 0.5557 scalar.
    x = np.asarray(w._session.calibrated_array(doc_id, channel), np.float64)
    fs = float(w._session.document(doc_id).meta.sample_rate_hz)
    reference = B.reference_states(x, fs, config=B.DEFAULT_BASELINE.replace(
        rough_s=B.reference_rough_s(x.size, fs) / 2.0))
    assert reference is not None
    for candidate in ranking:
        measured = B.config_intrusion(x, fs, config=pr.config.replace(
            rough_s=candidate.rough_s, base_s=candidate.base_s),
            polarity=-1, state_mask=reference[0])
        assert measured is not None
        assert candidate.fraction == measured["fraction"]
        assert candidate.worst_fraction == measured["worst_fraction"]
        assert candidate.worst_dwell_s == measured["worst_dwell_s"]
        assert candidate.worst_t_s == measured["worst_t_s"]
    assert by_window[RIDDEN].worst_fraction > B.LONG_STATE_INTRUSION_MAX
    assert by_window[RIDDEN].chosen is False
    assert by_window[OFFERED].fraction == 0.0
    assert by_window[OFFERED].rides is False
    assert by_window[OFFERED].chosen is True

    # ---- ...and what the SCREEN says, which is where the claims used to be false
    shown = api(bp, "proposal_text")()
    assert ("Nothing can beat a baseline that follows none of the states, so no wider "
            "window was estimated." not in shown), shown
    assert ("nothing WIDER can beat a baseline that follows none of the states"
            not in shown), (
        "the unscoped optimality claim is back on the screen: the wider window on "
        "THIS recording is better than a 'clean' shorter one, because 'follows none' "
        "was only ever the median", shown)
    assert "RANKED BY MEASURED INTRUSION" in shown, shown
    # the criterion, named with its threshold, and the rider named with its number
    assert "50% of any one state's excursion" in shown, (
        "the worst-state criterion the offer now rests on is not named", shown)
    assert "1000 ms / 300 ms follows NONE" in shown, shown
    assert "RIDES INTO its worst single state" in shown, shown
    ridden = by_window[RIDDEN]
    assert (f"{ridden.worst_fraction * 100:.1f}% of a "
            f"{ridden.worst_dwell_s * 1e3:.0f} ms state's own excursion followed") in shown, shown
    assert "cannot move a median" in shown, shown
    # the offer is the RULE's own window: nothing may claim a supersession
    assert "NOT WHAT THE APP OFFERS" not in shown, (
        "the screen announces a supersession while offering the rule's own window",
        shown)
    assert "SUPERSEDES" not in shown, shown
    # every rung estimated is named, and the evidence table carries both criteria
    assert "Ranked: 1000 ms / 300 ms" in shown, shown
    assert "Ranked: 2000 ms / 500 ms" in shown, shown
    assert "CHOSEN" in shown, shown
    assert shown.count("CHOSEN") == 1, shown
    assert "worst single state" in shown, shown
    # the optimality claims are scoped, and the unestimated windows are counted
    assert "under this ranking's objective" in shown, shown
    assert ("2 shorter windows on the grid (300 ms / 100 ms; 500 ms / 100 ms) were "
            "NOT estimated") in shown, (
        "the screen does not say which windows it never tried", shown)
    assert "rather than a claim that they are worse" in shown, shown

    # ---- the health block agrees, and Apply is a BACKSTOP not a step ------
    health = bp.health()
    assert health is not None
    assert bool(health.ok), [c.name for c in health.checks if not c.ok]
    assert bp.condemned_checks() == (), bp.condemned_checks()
    assert bp.apply_button.text() == "Use proposed windows", bp.apply_button.text()
    assert bp.apply_button.styleSheet() == "", bp.apply_button.styleSheet()
    assert bp.gate_label.isVisible() is False
    intrusion = health.check("baseline_intrusion")
    assert intrusion is not None and intrusion.level == "ok"
    assert intrusion.value == 0.0, intrusion.value

    # ---- one press applies it ---------------------------------------------
    bp.apply_button.click()
    _drain(qapp, qtbot, 200)
    assert (round(float(bp.rough_s.value()), 6),
            round(float(bp.base_s.value()), 6)) == OFFERED
    assert bp.param_source() == "proposed"

    # ---- the table preserves the full blocked interval ---------------------
    df, prov = _detect(qapp, qtbot, ap, ctrl)
    assert_detection_matches_reference(w, expected_windows=OFFERED, source="proposed")

    # Robust flank anchors correct the historical overstated event depth. With
    # the unchanged 5/2-sigma detector this permits brief exit crossings,
    # splitting the episode into rows. Its detected occupancy must still retain
    # the original narrow duration and edge bounds, regardless of row count.
    coverage, first, last, measured_depth = _clog_coverage(df)
    assert coverage == pytest.approx(0.5195, abs=1e-3)
    assert first == pytest.approx(37.1302, abs=1e-3)
    assert last == pytest.approx(37.6497, abs=1e-3)
    # Independent physical reference: interpolate between quiet raw-signal
    # flanks on either side of this clog, not another automatic baseline.
    a, b = int(37.13 * fs), int(37.65 * fs)
    left = x[int(37.03 * fs):int(37.11 * fs)]
    right = x[int(37.67 * fs):int(37.75 * fs)]
    quiet_pore = np.interp(np.arange(a, b) / fs, [37.07, 37.71],
                           [np.median(left), np.median(right)])
    true_depth = float(np.median(quiet_pore - x[a:b]))
    assert 20 < true_depth < 26, true_depth
    assert abs(measured_depth - true_depth) < 0.1 * true_depth
    deep = df[df["baseline_pA"] < -45.0]
    assert len(deep) == 0, (
        "the offered window stamps events on a collapsed local open pore",
        len(deep))

    # ======================================================================
    # ONE STEP PAST THE ONE CLICK: the REJECTED window, typed in by hand, in
    # the same window, on the same recording. The guard changes what the app
    # OFFERS, never what an entered window measures -- and detecting with it
    # is what shows, through the app, exactly what the guard is buying.
    # ======================================================================
    bp.rough_s.setValue(RIDDEN[0])
    bp.base_s.setValue(RIDDEN[1])
    _drain(qapp, qtbot, 100)
    assert (round(float(bp.rough_s.value()), 6),
            round(float(bp.base_s.value()), 6)) == RIDDEN
    assert bp.param_source() != "proposed", (
        "hand-typed values are still stamped as the app's proposal, which would make "
        "the rejected window's run indistinguishable from the offered one in "
        "provenance", bp.param_source())

    df2, prov2 = _detect(qapp, qtbot, ap, ctrl)
    assert_detection_matches_reference(w, expected_windows=RIDDEN, source="user")

    # ======================================================================
    # The rejected window must still lose most of the clog and put detected
    # events on a collapsed local baseline. Exact counts depend on the versioned
    # estimator/noise policy and are checked against a fresh direct run above.
    # ======================================================================
    ridden_coverage, _, _, _ = _clog_coverage(df2)
    assert ridden_coverage < 0.1 * coverage, (
        "the rejected window no longer loses most detected clog occupancy",
        ridden_coverage, coverage)
    deep2 = df2[df2["baseline_pA"] < -45.0]
    print(f"\n-600 mV: {OFFERED} -> {len(df)} events, {coverage:.6f} s clog coverage; "
          f"{RIDDEN} -> {len(df2)} events, {len(deep2)} on a local open pore "
          "below -45 pA")
    assert len(deep2) > 0, "the rejected window no longer produces events on a collapsed baseline"
    assert 37.13 < float(deep2["t_start_s"].min()) < 37.65, (
        "the collapsed-baseline events do not originate in the known clog", deep2)

    # Quantify what the guard protects with the same independent raw flanks.
    # The selected window stays within 10% of the physical depth; the rejected
    # window loses more than half of it, rather than pinning obsolete event totals.
    measured_depths = {}
    for window in (OFFERED, RIDDEN):
        result = B.characterisation_baseline(x, fs, config=pr.config.replace(
            rough_s=window[0], base_s=window[1]), polarity=-1, health=False, cache=False)
        measured_depths[window] = float(np.median(result.baseline[a:b] - x[a:b]))
    assert abs(measured_depths[OFFERED] - true_depth) < 0.1 * true_depth
    assert measured_depths[RIDDEN] < 0.5 * true_depth
