"""RP+BL THROUGH THE WIRED APPLICATION -- the -600 mV screen, ladder, guard and table.

WHY THIS FILE EXISTS, AND WHICH "ONE STEP PAST" IT TAKES
=======================================================
The RK batch shipped a wired test for this exact panel and it did not see the RP
defect, because it only ever read the **-700 mV** screen; the RP batch fixed the
one-sided ladder and this file then asserted ITS offer -- 1 s / 300 ms, the shorter
window that "ties at 0.0". The BL batch measured what that tie was worth: the shorter
window's tracked baseline rides into the recording's **521 ms partial clog at
t = 37.13 s** (56% of the one state's own excursion followed, at a MEDIAN intrusion
of exactly 0.0 -- one state among 239 cannot move a median), the clog stops being
detected at all, and 33 shallow events are invented on the collapsed local baseline.
So this file now asserts the guarded offer end to end:

  * **the one click** proposes the rule's own 2 s / 500 ms, with the riding
    1 s / 300 ms ESTIMATED, named on the screen with the number that rejected it,
    and the worst-state criterion (`baseline_probe.WORST_STATE_INTRUSION_MAX`)
    spelled out in the reason text;
  * **the second gesture** -- after the one click is applied and detected, the
    REJECTED 1 s / 300 ms is HAND-ENTERED into the same panel and detected again,
    in the same window, so the harm the guard prevents is a thing the app was
    actually made to do rather than a claim about a library function;
  * **the number the user is actually buying** -- the clog as ONE 519.5 ms event at
    the offered window against NO clog event and 33 events on a sub--45 pA local
    baseline at the rejected one.

Cost ~2 minutes: a 36 MB / 3.0 M-sample ABF, one probe and two detections.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
from PySide6 import QtCore

FS = 50_000.0
APO_600 = pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0006_Process_-600 mV.abf")

#: What the median-only two-sided search chose (it "ties at 0.0 and is shorter"),
#: and what the app offers now that the tie is judged on the state the median cannot
#: see. The ridden window stays typable -- rejected as an OFFER, not removed.
RIDDEN = (1.0, 0.3)
OFFERED = (2.0, 0.5)

#: The app's own event-excluded open pore for this recording: the median of every
#: sample OUTSIDE `baseline.reference_states`' wide-pass mask. Not a new statistic --
#: it is the mask every candidate's intrusion is measured over.
EXCLUDED_OPEN_PORE = -34.48486328125


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


@pytest.mark.corpus
@pytest.mark.skipif(not APO_600.exists(), reason="the -600 mV recording is not here")
def test_the_six_hundred_millivolt_one_click_rejects_the_window_that_rides(
        qapp, qtbot):
    """THE GUARD, END TO END, ON THE RECORDING THAT CARRIES THE STATE IT IS FOR.

    Before (RP): one click offered 1 s / 300 ms because it "ties at 0.0 and is
    shorter", and detection at that offer un-detected a 521 ms clog while inventing
    33 shallow events inside it. After (BL): the descent still estimates that rung
    -- the guard is a measurement, not a skip -- and the screen says which state it
    rides into, by how much, and why the tie the median reports is not a tie.
    """
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
    assert float(by_window[RIDDEN].worst_fraction) == pytest.approx(0.5557, abs=5e-4)
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
    assert "1 s / 300 ms follows NONE" in shown, shown
    assert "RIDES INTO its worst single state" in shown, shown
    assert "55.6% of a 473 ms state's own excursion followed" in shown, shown
    assert "cannot move a median" in shown, shown
    # the offer is the RULE's own window: nothing may claim a supersession
    assert "NOT WHAT THE APP OFFERS" not in shown, (
        "the screen announces a supersession while offering the rule's own window",
        shown)
    assert "SUPERSEDES" not in shown, shown
    # every rung estimated is named, and the evidence table carries both criteria
    assert "Ranked: 1 s / 300 ms" in shown, shown
    assert "Ranked: 2 s / 500 ms" in shown, shown
    assert "CHOSEN" in shown, shown
    assert shown.count("CHOSEN") == 1, shown
    assert "worst single state" in shown, shown
    # the optimality claims are scoped, and the unestimated windows are counted
    assert "under this ranking's objective" in shown, shown
    assert ("2 shorter windows on the grid (0.3 s / 100 ms; 0.5 s / 100 ms) were "
            "NOT estimated") in shown, (
        "the screen does not say which windows it never tried", shown)
    assert "rather than a claim that they are worse" in shown, shown

    # ---- the health block agrees, and Apply is a BACKSTOP not a step ------
    health = bp.health()
    assert health is not None
    assert bool(health.ok), [c.name for c in health.checks if not c.ok]
    assert bp.condemned_checks() == (), bp.condemned_checks()
    assert bp.apply_button.text() == "Use these values", bp.apply_button.text()
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

    # ---- the table it produces: the clog is ONE event ----------------------
    df, prov = _detect(qapp, qtbot, ap, ctrl)
    assert df is not None
    assert str(prov.get("baseline_param_source")) == "proposed"
    assert prov.get("baseline_fingerprint") == "sha256:ce8ae31f10bd", prov.get(
        "baseline_fingerprint")
    assert len(df) == 276, ("the -600 mV detected count at 2 s / 500 ms", len(df))

    clog = df[(df["dwell_s"] > 0.4) & (df["t_start_s"] > 37.0)
              & (df["t_start_s"] < 37.3)]
    assert len(clog) == 1, (
        "the offered window does not detect the 521 ms clog as one event",
        clog.to_dict("records"))
    assert float(clog["dwell_s"].iloc[0]) == pytest.approx(0.5195, abs=1e-3)
    assert float(clog["depth_pA"].iloc[0]) == pytest.approx(28.87, abs=0.05)
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
    assert df2 is not None
    assert prov2.get("baseline_fingerprint") == "sha256:b64a75ebdf29", prov2.get(
        "baseline_fingerprint")
    assert len(df2) == 311, (
        "the rejected window no longer reproduces its measured detected count "
        "through the app", len(df2))

    # ======================================================================
    # WHAT THE 35-EVENT DIFFERENCE ACTUALLY IS, MEASURED THROUGH THE APP.
    # 311 against 276 reads as "more events found". It is not: at 1 s /
    # 300 ms the tracked baseline rides into the clog at t = 37.130159 s to
    # ~-54.5 pA, the clog stops being an event at all, and the region
    # 37.14-38.60 s instead yields 33 shallow events whose stamped local
    # open pore is below -45 pA -- roughly the whole of the difference. This
    # used to be the pinned COST of the RP move; it is now the pinned reason
    # for the BL guard, and `baseline_intrusion`'s median still cannot see
    # it (0.0 over 239 states) -- what sees it is the worst-state reading
    # asserted on the ladder above.
    # ======================================================================
    clog2 = df2[(df2["dwell_s"] > 0.4) & (df2["t_start_s"] > 37.0)
                & (df2["t_start_s"] < 37.3)]
    assert len(clog2) == 0, (
        "the rejected window now detects the clog as one event -- if the estimator "
        "changed, the guard's evidence needs re-measuring",
        clog2.to_dict("records"))
    deep2 = df2[df2["baseline_pA"] < -45.0]
    print(f"\n-600 mV: {OFFERED} -> {len(df)} events, clog as one event; "
          f"{RIDDEN} -> {len(df2)} events, {len(deep2)} on a local open pore "
          "below -45 pA")
    assert len(deep2) == 33, (
        "the count of events the rejected window stamps inside the ridden-into clog "
        "changed; whichever way, it is the number the guard is weighed against",
        len(deep2))
    assert float(deep2["t_start_s"].min()) == pytest.approx(37.14, abs=0.02)

    # ---- the event-excluded open pore, for the record ----------------------
    med_off = float(np.median(np.asarray(df["baseline_pA"], float)))
    med_rid = float(np.median(np.asarray(df2["baseline_pA"], float)))
    print(f"median baseline_pA: {OFFERED} {med_off:.4f} pA, {RIDDEN} "
          f"{med_rid:.4f} pA, event-excluded open pore {EXCLUDED_OPEN_PORE:.4f} pA")
    assert med_off == pytest.approx(-34.7900, abs=5e-4), med_off
