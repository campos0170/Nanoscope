"""The four seams that were REACHABLE BUT INERT, driven in the real window.

Every test here asserts a user-visible EFFECT, never a connection: a rail on the trace, a
status cell that says a person ruled, a merge that survives a review click, a preview
whose array comes from the pure definition. Each one fails against the pre-wiring panel —

* the two baseline boxes ticked and drew nothing (``showBaselineToggled`` /
  ``subtractBaselineToggled`` reached no slot);
* an accepted event read ``unreviewed``, because nothing in the app called
  ``events.mark_reviewed``;
* a "mark reviewed" during a Merge all threw the finished merge away, because the
  install guard diffed the WIDE ``curation_snapshot``;
* ``AnalysisPanel._conditioned_from_form`` was a second copy of
  ``analysis.conditioning.conditioned_trace`` rather than a call to it.

ONE window for the module. Opening MainWindow on the owner's 15 s recording costs seconds,
and these four questions are asked of the same screen a user would be looking at. The
tests that mutate the events frame work on their own rows and say which.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.analysis.events import REVIEW_COL, event_status, mark_reviewed
from nanoscope.analysis.remeasure import curation_snapshot, merge_guard_snapshot

REAL_ABF = ("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0004_"
            "process _ -1000 mV_ 385-400 s time trace.abf")

pytestmark = pytest.mark.skipif(
    not Path(REAL_ABF).exists(),
    reason="the owner's 15 s -1000 mV apo recording is not on this machine")


def _row(win, w) -> tuple[int, int, int]:
    """``(x0, x1, painted)`` of a widget IN WINDOW COORDINATES — see test_pc_baseline_overlay."""
    region = w.visibleRegion()
    if region.isEmpty():
        return 0, 0, 0
    rect = region.boundingRect()
    tl = w.mapTo(win, rect.topLeft())
    x0, x1 = tl.x(), tl.x() + rect.width()
    return x0, x1, max(0, min(x1, win.width()) - max(x0, 0))


@pytest.fixture(scope="module")
def window(qapp):
    """A real 1512x900 MainWindow with his recording loaded and detected (115 events)."""
    from nanoscope.analysis import register_all
    from nanoscope.gui.main_window import MainWindow
    from nanoscope.gui.overlays import register_all_renderers

    register_all()
    register_all_renderers()
    win = MainWindow()
    win.resize(1512, 900)
    win.show()
    _pump(200)
    win.open_paths([Path(REAL_ABF)])
    assert _wait(lambda: len(win._session.layers()) > 0, 30_000), "the recording never loaded"
    _pump(300)
    panel = win._analysis_panel
    win._inspector_tabs.setCurrentWidget(panel)
    panel._detection.mode_tabs.setCurrentIndex(0)
    _pump(100)
    panel._detection.detectRequested.emit(dict(panel._detection.params()))
    assert _wait(lambda: panel.current_detection() is not None, 60_000), "detection never landed"
    _pump(300)
    yield win
    win.close()


def _pump(ms: int = 100) -> None:
    deadline = QtCore.QDeadlineTimer(ms)
    while not deadline.hasExpired():
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 20)


def _wait(fn, timeout: int = 20_000) -> bool:
    deadline = QtCore.QDeadlineTimer(timeout)
    while not deadline.hasExpired():
        if fn():
            return True
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 20)
    return bool(fn())


def _click(box) -> None:
    QTest.mouseClick(box, QtCore.Qt.LeftButton, pos=QtCore.QPoint(8, box.height() // 2))
    _pump(400)


# ================================================================================================
# W-BASELINE — the two boxes draw, and stop drawing
# ================================================================================================
def test_ticking_the_boxes_draws_the_rail_and_the_residual_and_unticking_takes_them_off(window):
    """THE FEATURE, driven by the checkbox: pre-wiring both clicks were silent.

    Measured on the owner's recording: the residual sits at -0.305 pA (the raw trace at
    -57.7 pA) and the y axis relabels itself, so the assertion is on what the pane is
    SHOWING, not on whether a signal found a slot.
    """
    panel = window._analysis_panel
    bp, view = panel.baseline_panel, window._trace_view
    doc_id, channel = panel._sel
    layer = f"{doc_id}#{channel}"
    pane = view.pane_for_layer(layer)

    assert view.has_baseline_track(doc_id, channel) is False
    raw_median = float(np.median(np.asarray(pane._curves[layer].yData)))

    before = {n: _row(window, w) for n, w in (("show", bp.show_track),
                                              ("subtract", bp.subtract_view))}
    _click(bp.show_track)
    assert view.has_baseline_track(doc_id, channel) is True, \
        "the 'Show this baseline' box drew no rail"

    _click(bp.subtract_view)
    assert view.baseline_subtracted_for(doc_id, channel) is not None
    residual = np.asarray(pane._curves[layer].yData)
    assert abs(float(np.median(residual))) < 2.0, \
        f"the curve on screen is not the residual: median {float(np.median(residual)):.3f} pA"
    assert abs(raw_median) > 10.0, "the raw trace was already at zero; the test proves nothing"
    assert "vs baseline" in pane.plot.getAxis("left").labelText

    # NOTHING MOVED UNDER THE POINTER: the boxes are where they were before the click.
    assert {n: _row(window, w) for n, w in (("show", bp.show_track),
                                            ("subtract", bp.subtract_view))} == before

    _click(bp.subtract_view)
    _click(bp.show_track)
    assert view.has_baseline_track(doc_id, channel) is False, "the rail outlived the untick"
    assert view.baseline_subtracted_for(doc_id, channel) is None
    assert float(np.median(np.asarray(pane._curves[layer].yData))) == pytest.approx(
        raw_median, abs=1e-9), "unticking did not put the raw trace back"
    assert "vs baseline" not in pane.plot.getAxis("left").labelText


# ================================================================================================
# W-REVIEW — a person's ruling is recorded
# ================================================================================================
def test_an_accepted_event_reads_accepted_rather_than_unreviewed(window):
    """The status a curated row reports. Pre-wiring every row read ``unreviewed`` forever.

    Both directions of one gesture, on rows this module owns (the last two), because a
    rejection derived correctly WITHOUT the wiring and an acceptance did not — the test
    would pass half-way on the old panel if it only looked at the reject.
    """
    panel = window._analysis_panel
    df = panel.current_detection().payload.df
    ids = df["event_id"].to_numpy(dtype=np.int64)
    reject_id, accept_id = int(ids[-1]), int(ids[-2])

    panel.set_event_accepted(reject_id, False)
    _pump(150)
    panel.set_event_accepted(accept_id, False)
    _pump(150)
    panel.set_event_accepted(accept_id, True)
    _pump(150)

    live = panel.current_detection().payload.df
    status = np.asarray(event_status(live), dtype=object)
    pos = {int(e): i for i, e in enumerate(live["event_id"].to_numpy(dtype=np.int64))}

    assert live[REVIEW_COL].iloc[pos[accept_id]] == "reviewed"
    assert status[pos[accept_id]] == "accepted", (
        "an event the user accepted still reads "
        f"{status[pos[accept_id]]!r} — the ruling was not recorded")
    assert status[pos[reject_id]] == "rejected"
    untouched = [i for i in range(len(live)) if i not in (pos[accept_id], pos[reject_id])]
    assert set(status[untouched]) == {"unreviewed"}, \
        "one gesture marked rows the user never looked at"


def test_a_hand_added_missed_event_counts_as_having_been_ruled_on(window):
    """Drawing a span and calling it a missed event IS the ruling.

    ``ensure_source_status`` deliberately stopped inferring reviewedness from ``added``
    (it may not promote a merged row), so without the panel's own call this row reads
    ``unknown``: source ``manual``, review nothing.
    """
    panel = window._analysis_panel
    before = len(panel.current_detection().payload.df)
    fs = float(window._session.calibrated(*panel._sel).sample_rate_hz)
    panel.measure_panel.set_active(True)
    # The recording's own time base: this file is the 385-400 s excerpt, so a region at
    # t=1 s is off the end of it and the append is (correctly) refused.
    doc_id, channel = panel._sel
    x0, x1 = window._trace_view.pane_for_layer(f"{doc_id}#{channel}").plot.viewRange()[0]
    left = x0 + 0.25 * (x1 - x0)
    window._trace_view.set_measure_region(left, left + 0.002)   # a 2 ms span, in seconds
    _pump(150)
    new_id = panel.add_current_selection_as_event()
    _pump(200)
    assert new_id is not None, "the hand-added event was refused; nothing to judge"

    live = panel.current_detection().payload.df
    assert len(live) == before + 1
    pos = int(np.flatnonzero(live["event_id"].to_numpy(dtype=np.int64) == int(new_id))[0])
    assert live[REVIEW_COL].iloc[pos] == "reviewed"
    assert np.asarray(event_status(live), dtype=object)[pos] == "accepted", \
        "the event the user added by hand does not read as ruled on"


# ================================================================================================
# W-MERGEGUARD — a review click no longer discards a finished merge
# ================================================================================================
def test_marking_reviewed_during_a_merge_all_no_longer_throws_the_merge_away(window):
    """The install guard, driven through the real 'Merge all' on his 115 events.

    The review is written BEFORE the event loop is pumped, so the worker's result cannot
    have been delivered yet — the merge lands on a frame whose ``review`` column moved
    under it, which is exactly the sequence that used to report ``dropped``.
    """
    panel = window._analysis_panel
    live = panel.current_detection().payload.df
    n_before = len(live)

    reports: list = []
    panel.mergeFinished.connect(reports.append)
    try:
        started = panel.merge_all_subevents(1e-3)
        assert started.status == "started", f"the merge did not start: {started.status}"
        # ...a curator marks three rows reviewed while it runs. No accept, no flag.
        assert mark_reviewed(live, rows=np.array([5, 6, 7])) > 0
        assert _wait(lambda: bool(reports), 60_000), "the merge never finished"
    finally:
        panel.mergeFinished.disconnect(reports.append)

    report = reports[-1]
    assert report.status == "ok", (
        f"a review-only click during Merge all reported {report.status!r} — the guard is "
        f"still diffing the wide curation snapshot (pre-wiring: 'dropped')")
    assert len(panel.current_detection().payload.df) < n_before, \
        "the merge reported success but installed nothing"


def test_the_guard_still_drops_a_merge_the_user_actually_curated(window):
    """The other half: the snapshot the panel stores must still notice a VERDICT.

    Same frame, same two calls the panel makes — an accept moves ``merge_guard_snapshot``
    and a review does not, which is the whole distinction the guard exists to draw.
    """
    df = window._analysis_panel.current_detection().payload.df.copy(deep=True)
    guard, wide = merge_guard_snapshot(df), curation_snapshot(df)

    mark_reviewed(df, rows=np.array([0, 1]))
    assert merge_guard_snapshot(df).equals(guard), "a review moved the merge guard"
    assert not wide.equals(curation_snapshot(df)), \
        "the wide snapshot no longer notices a review — this test is measuring nothing"

    df.iloc[2, df.columns.get_loc("accepted")] = not bool(df["accepted"].iloc[2])
    assert not merge_guard_snapshot(df).equals(guard), \
        "an accept/reject during a merge must still drop it"


# ================================================================================================
# W-CONDITION — the panel's chain IS the pure definition
# ================================================================================================
def test_the_live_preview_is_drawn_by_the_pure_conditioning_module(window, monkeypatch):
    """Proved by SUBSTITUTION: replace the definition and the panel's curve follows.

    A copy cannot follow. Pre-wiring the name ``conditioned_trace`` was not even bound in
    the panel's module, so this test errors there rather than passing quietly.
    """
    panel = window._analysis_panel
    calls: list[tuple[int, float]] = []
    marker = None

    from nanoscope.analysis.conditioning import conditioned_trace as real

    def spy(raw, fs, params):
        nonlocal marker
        calls.append((len(raw), float(fs)))
        marker = np.asarray(real(raw, fs, params)) + 1000.0
        return marker

    monkeypatch.setattr("nanoscope.gui.analysis_panel.conditioned_trace", spy)
    panel._detection.filter_enabled.setChecked(True)
    panel._detection.filter_cutoff_hz.setValue(5000.0)
    panel._detection.filter_preview.setChecked(True)
    _pump(500)

    assert calls, "the preview built its processed trace without the pure definition"
    doc_id, channel = panel._sel
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    drawn = [np.asarray(c.yData) for c in pane.plot.listDataItems()
             if c.yData is not None and len(c.yData) == len(marker)]
    assert any(np.array_equal(d, marker) for d in drawn), \
        "the curve on screen is not the array the conditioning module returned"
    panel._detection.filter_preview.setChecked(False)
    _pump(200)
