"""M2 THROUGH THE WIRED APPLICATION -- the strings the -700 mV screen actually shows.

Everything the proposer floor changed that a person can READ is asserted here, on the
owner's own recording, through a real `MainWindow` with the event loop drained, on the
rendered text rather than on the rich-text source: the summary row that carries the
median state, the reason paragraph that says the floor fired, the source line under the
form after ONE Apply click, and -- when a condemned window is typed in by hand and run
-- ``base_vs_state``'s remedy, which used to end *"RE-RUNNING THE PROBE WILL NOT DO
IT"* and now names the probe as the action that works.

WHY THIS FILE EXISTS SEPARATELY FROM `tests/analysis/test_fup_proposer_floor.py`
================================================================================
Four of the six defects the 2026-08-22 integration pass re-measured were in what the
app TELLS you rather than in what it computes, and every one was invisible to a test
that constructs the widget in isolation. The specific one this file guards against is
M2's own: the fix landing in `propose_baseline` while the panel goes on refusing the
first Apply click, or goes on printing a source line that says *"app default (this
recording was not measured)"* over windows that were, in fact, measured.

Cost ~2.5 minutes: a 108 MB / 9.1 M-sample ABF, two probes and two estimator runs.
"""
from __future__ import annotations

import pathlib

import pytest

from PySide6 import QtCore, QtGui

from nanoscope.analysis import baseline as B

APO_700 = pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0005_Process_-700 mV.abf")

#: What the proposer offered before the floor existed, and what it offers now.
WAS = (0.1, 0.03)
NOW = (1.0, 0.3)

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not APO_700.exists(), reason="the -700 mV recording is not here"),
]


def rendered(label) -> str:
    """The characters that reach the pixels, not the rich-text SOURCE.

    The panel's own :meth:`BaselinePanel.health_text` exists for exactly this reason
    (Qt once ate ``<= 0.3`` out of a line on its way to the screen while the source
    still contained it); the summary and reason labels have no such accessor, so the
    same parse is done here rather than asserting markup a reader never sees.
    """
    doc = QtGui.QTextDocument()
    doc.setHtml(label.text())
    return doc.toPlainText()


def _drain(qapp, qtbot, ms=150):
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)


def test_the_seven_hundred_millivolt_screen_after_one_apply_click(qapp, qtbot):
    """THE REPORTED SCREEN, end to end: open -> select ch2 -> Probe -> Apply -> Detect.

    Before this change that sequence left the form on the app default 0.5 / 0.1 s with
    the source line reading *"app default (this recording was not measured)"* and a
    REFUSED banner over the button, because the proposal (0.1 / 0.03) was condemned by
    two health checks and M1's gate would not apply it on one press. The proposal is
    now 0.5 / 0.1, all eight checks pass, and the same one press applies it as
    ``proposed``.
    """
    import nanoscope
    from nanoscope.core.types import ChannelKind
    from nanoscope.gui.main_window import MainWindow

    where = pathlib.Path(nanoscope.__file__).resolve()
    print(f"\nnanoscope resolved from: {where}")

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1600, 900)
    w.show()
    w._inspector.show()
    w._inspector_tabs.setCurrentWidget(w._analysis_panel)
    _drain(qapp, qtbot, 50)

    with qtbot.waitSignal(w._session.documentOpened, timeout=300_000):
        w.open_paths([str(APO_700)])
    _drain(qapp, qtbot, 50)
    doc_id = w._session.doc_id_for_source(str(APO_700.resolve()))
    assert doc_id is not None
    channel = next(c.index for c in w._session.document(doc_id).meta.channels
                   if c.kind is ChannelKind.CURRENT)
    assert channel == 2
    w._on_selection(doc_id, channel)
    _drain(qapp, qtbot, 50)

    ap = w._analysis_panel
    bp = ap.baseline_panel
    ctrl = w._analysis_controller

    # ---- the probe, twice: the proposer is still deterministic ------------
    seen = []
    for _ in range(2):
        with qtbot.waitSignal(ctrl.finished, timeout=900_000):
            bp.probeRequested.emit()
        _drain(qapp, qtbot, 200)
        pr = bp.proposal()
        assert pr is not None
        seen.append((round(float(pr.config.rough_s), 6),
                     round(float(pr.config.base_s), 6)))
    assert seen == [NOW, NOW], (
        f"the proposal on this recording is {seen}, not {NOW}", seen)
    assert seen[0] != WAS, "the condemned proposal is still being offered"

    # ---- the health block agrees, which is the whole invariant ------------
    health = bp.health()
    assert health is not None
    assert bool(health.ok), [c.name for c in health.checks if not c.ok]
    assert bp.condemned_checks() == (), bp.condemned_checks()

    # ---- what the summary table SAYS ------------------------------------
    summary = rendered(bp.summary_label)
    assert "Median state" in summary, summary
    assert "base_s floor" in summary, summary
    assert "21.7 ms" in summary, (
        "the summary does not show the median state base_s is now sized on", summary)
    assert "base_s >= 100 ms" in summary, summary
    assert "31 ms" in summary, ("the longest state must still be shown beside it -- "
                                "the two are different statistics", summary)

    # ---- what the reason paragraph SAYS ---------------------------------
    reason = rendered(bp.reason_label)
    assert "floored" in reason, reason
    assert "0.1 s / 30 ms" in reason, (
        "the reason must name the window the 4:1 rule alone would have proposed, or "
        "the user cannot tell that anything was overridden", reason)
    assert "1.4x the states" in reason, reason
    assert "4.6x" in reason, reason
    assert "21.7 ms" in reason, reason
    assert not reason.startswith("Could not measure"), reason

    # ---- the button is plain, and ONE press applies -----------------------
    assert bp.apply_button.text() == "Use these values", bp.apply_button.text()
    assert bp.apply_button.styleSheet() == "", bp.apply_button.styleSheet()
    assert bp.gate_label.text() == "", bp.gate_label.text()
    assert bp.gate_label.isVisible() is False

    bp.apply_button.click()
    _drain(qapp, qtbot, 200)

    assert (round(float(bp.rough_s.value()), 6),
            round(float(bp.base_s.value()), 6)) == NOW
    assert bp.param_source() == "proposed", bp.param_source()
    src = bp.source_label.text()
    assert "measured on this recording, and confirmed" in src, src
    assert "app default" not in src, (
        "one Apply click still leaves the form on the app default -- the refusal "
        "path, which is what M2 exists to make unnecessary", src)
    assert "ACCEPTED OVER" not in src, src
    assert bp.accepted_condemned() == (), bp.accepted_condemned()
    assert bp.gate_label.text() == "", bp.gate_label.text()

    # ---- and the table it produces --------------------------------------
    dp = ap.detection_panel
    dp.show_detection_tab()
    _drain(qapp, qtbot, 100)
    with qtbot.waitSignal(ctrl.finished, timeout=900_000):
        dp.detect_button.click()
    _drain(qapp, qtbot, 400)
    df = ap.current_events_df()
    assert df is not None and len(df) == 1179, len(df) if df is not None else None
    prov = ap.current_detection().payload.provenance
    assert prov.get("baseline_fingerprint") == "sha256:b64a75ebdf29", prov.get(
        "baseline_fingerprint")
    assert str(prov.get("baseline_param_source")) == "proposed", prov.get(
        "baseline_param_source")


def test_the_condemned_window_still_says_what_to_do_and_names_the_probe(qapp, qtbot):
    """The other side of the same seam, and the one a revert of the message alone
    would leave standing.

    The owner has PUBLISHED from 0.1 / 0.03 s, so that window must stay reachable by
    hand -- and when it is reached, the remedy on screen must be true. It used to end
    *"RE-RUNNING THE PROBE WILL NOT DO IT ... two consecutive probes of the -700 mV
    recording both returned 0.1 / 0.03 s"*, which after the floor is false and would
    send the user the long way round. It now names the probe first and keeps the
    by-hand instruction, because the by-hand route is the one that reached this screen.
    """
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
        w.open_paths([str(APO_700)])
    _drain(qapp, qtbot, 50)
    doc_id = w._session.doc_id_for_source(str(APO_700.resolve()))
    channel = next(c.index for c in w._session.document(doc_id).meta.channels
                   if c.kind is ChannelKind.CURRENT)
    w._on_selection(doc_id, channel)
    _drain(qapp, qtbot, 50)

    ap = w._analysis_panel
    bp = ap.baseline_panel
    ctrl = w._analysis_controller

    bp.rough_s.setValue(WAS[0])
    bp.base_s.setValue(WAS[1])
    _drain(qapp, qtbot, 80)
    assert "set by hand" in bp.source_label.text(), bp.source_label.text()

    ap.detection_panel.show_detection_tab()
    _drain(qapp, qtbot, 100)
    with qtbot.waitSignal(ctrl.finished, timeout=900_000):
        ap.detection_panel.detect_button.click()
    _drain(qapp, qtbot, 400)

    # the published table is still reachable, bit-for-bit in its row count
    df = ap.current_events_df()
    assert df is not None and len(df) == 1179, len(df) if df is not None else None
    assert ap.current_detection().payload.provenance.get(
        "baseline_fingerprint") == "sha256:1a29aabf5fb1"

    said = bp.health_text()
    assert "base window vs state dwell" in said, said[:600]
    assert "WIDEN base_s to at least 0.0651 s BY HAND" in said, said
    assert "OR PRESS PROBE AGAIN, WHICH NOW DOES IT FOR YOU" in said, said
    assert f"floors base_s at {B.BASE_STATE_RATIO_MIN:g}x it" in said, said
    assert "RE-RUNNING THE PROBE WILL NOT DO IT" not in said, (
        "the screen still tells the user the probe cannot repair this, which stopped "
        "being true when the floor landed", said)
    assert "or re-run the probe" not in said.lower().replace("re-running", "")
