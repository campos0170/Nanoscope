"""RK THROUGH THE WIRED APPLICATION -- the ranking as the -700 mV screen shows it.

WHY THIS FILE EXISTS SEPARATELY FROM `tests/analysis/test_rk_intrusion_ranking_core.py`
=======================================================================================
The ranking's whole output to a person is a paragraph and a table of rows under the
proposal, and a widget built in isolation is not evidence that either reaches the
screen: this panel renders the summary as RICH TEXT, and Qt's parser has already eaten
a criterion out of a line here once while the HTML source still contained it
(`BaselinePanel._check_html`). Every assertion below is therefore made on the RENDERED
characters, through a real `MainWindow`, with the event loop drained, after a real
click on the real recording.

Three hunks are asserted here:

  H3  the probe job's answer is the RANKED one (the click, end to end)
  H4  the ranked ladder is rendered where the confirmation click is made
  H5  `BaselinePanel.proposal_text` + the escaping + the PlainText reason label

Cost ~3 minutes for the corpus test: a 108 MB / 9.1 M-sample ABF, one probe (which now
estimates two candidate windows) and one detection. The two panel tests are instant.
"""
from __future__ import annotations

import pathlib

import pytest
from PySide6 import QtCore

from nanoscope.analysis import baseline as B
from nanoscope.analysis import baseline_probe as P
from nanoscope.gui.baseline_panel import BaselinePanel

FS = 50_000.0
APO_700 = pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0005_Process_-700 mV.abf")

#: What the floor stopped at, and what ranking by measured intrusion takes instead.
FLOOR_STOPPED_AT = (0.5, 0.1)
NOW = (1.0, 0.3)


def api(obj, name):
    """``getattr`` as an ASSERTION, so a package without the fix fails on the CLAIM."""
    assert hasattr(obj, name), (
        f"{getattr(obj, '__name__', type(obj).__name__)} has no {name!r}: the ranking "
        "is not reachable from the screen, which is the defect this file is about")
    return getattr(obj, name)


def where_from():
    import nanoscope
    p = pathlib.Path(nanoscope.__file__).resolve()
    print(f"\nnanoscope resolved from: {p}")
    return p


def _drain(qapp, qtbot, ms=150):
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 50)


@pytest.fixture
def panel(qapp, qtbot):
    p = BaselinePanel()
    qtbot.addWidget(p)
    return p


def _proposal(ranking=(), reason="Longest state 31 ms -> rough 1 s."):
    probe = P.BaselineProbe(
        fs=FS, n=9_074_042, polarity=-1, sigma_pA=4.98, noise_pA=4.44,
        longest_state_s=0.0315, n_states=1583, mask_fraction=0.1075, gating=False,
        gating_score=1.65, level_dwell_s=None, probe_rough_s=5.0, usable=True,
        elapsed_s=1.2, state_dwell_s=0.02171)
    kw = {}
    if ranking or "ranking" in P.BaselineProposal.__dataclass_fields__:
        # Built conditionally so the two H5 tests -- which are about the PARSER and not
        # about the ladder -- still reach their own assertion against a package with no
        # ``ranking`` field, instead of dying in the fixture on a TypeError that reads
        # like a broken test.
        kw["ranking"] = ranking
    return P.BaselineProposal(
        config=B.DEFAULT_BASELINE.replace(rough_s=1.0, base_s=0.3, decim=None),
        param_source="proposed", probe=probe, reason=reason, **kw)


# ===========================================================================
# 1. H5 -- what the parser does to the evidence on its way to the pixels
# ===========================================================================
def test_the_summary_rows_survive_the_rich_text_parser(panel, monkeypatch):
    """H5. THE DEFECT THIS PANEL HAS ALREADY SHIPPED ONCE, one label along.

    ``summary_label`` is rich text and its rows were interpolated RAW. Qt reads
    ``passes <= 0.3)</span>`` as a malformed tag and deletes it together with
    everything up to the next ``>`` -- that is exactly how the overshoot criterion
    vanished from the health block while the HTML source still contained it. The
    summary now carries MEASURED values (the intrusion ladder), which is precisely the
    content that grows a comparison operator one day.

    Asserted through :meth:`BaselinePanel.proposal_text`, which is the rendered side of
    the parser; an assertion on ``summary_label.text()`` is an assertion about markup
    the reader never sees, and that is how the last one passed over a live defect.
    """
    where_from()
    monkeypatch.setattr(
        "nanoscope.gui.baseline_panel.proposal_summary",
        lambda proposal, fs=None: (("Ranked: 0.5 s / 100 ms",
                                    "follows <= 0.428% of the state depth"),))
    panel.set_recording(FS)
    panel.set_proposal(_proposal())
    shown = api(panel, "proposal_text")()
    assert "Ranked: 0.5 s / 100 ms" in shown, shown
    assert "<= 0.428% of the state depth" in shown, (
        "the criterion was eaten between the label and the pixel -- the same defect "
        "the health block already carries a fix for", shown)


def test_the_reason_paragraph_reaches_the_screen_exactly_as_written(panel):
    """H5. The reason line is measured prose that now carries percentages and
    comparisons. A QLabel left on ``AutoText`` GUESSES whether that is markup, and the
    guess is what decides whether the user reads the sentence or a silently edited one.
    Declaring it PlainText also makes ``reason_label.text()`` the rendered text, which
    is what :meth:`BaselinePanel.proposal_text` composes."""
    where_from()
    panel.set_recording(FS)
    marked = ("RANKED BY MEASURED INTRUSION: 0.5 s / 100 ms follows 0.428% "
              "<b>of each state's own depth</b>.")
    panel.set_proposal(_proposal(reason=marked))
    assert panel.reason_label.textFormat() == QtCore.Qt.PlainText
    shown = api(panel, "proposal_text")()
    assert marked in shown, shown


def test_the_ladder_is_rendered_under_the_proposal_it_justifies(panel):
    """H4 + H5. The confirmation click decides every ``depth_pA`` in the table, and
    the evidence for it has to be on the SAME screen as the button. A ranking that
    lived only in the object would be a proposal the user still cannot interrogate,
    which is what "an auto-apply with an extra click" means in this panel's docstring.
    """
    where_from()
    candidate = api(P, "BaselineCandidate")
    api(P.BaselineProposal, "ranking")
    ranking = (
        candidate(rough_s=0.5, base_s=0.1, fraction=0.004278771600903435,
                  n_states=982, cost=0.004278771600903435, chosen=False),
        candidate(rough_s=1.0, base_s=0.3, fraction=0.0, n_states=982,
                  cost=0.0, chosen=True),
    )
    panel.set_recording(FS)
    panel.set_proposal(_proposal(ranking=ranking))
    shown = api(panel, "proposal_text")()
    assert "Ranked: 0.5 s / 100 ms" in shown, shown
    assert "0.428%" in shown, shown
    assert "Ranked: 1 s / 300 ms" in shown, shown
    assert "CHOSEN" in shown, shown
    assert shown.index("Ranked: 0.5 s / 100 ms") < shown.index("Ranked: 1 s / 300 ms")


# ===========================================================================
# 2. H3/H4 -- the one click, on the recording the change is about
# ===========================================================================
@pytest.mark.corpus
@pytest.mark.skipif(not APO_700.exists(), reason="the -700 mV recording is not here")
def test_the_seven_hundred_millivolt_one_click_answer_is_the_ranked_window(qapp, qtbot):
    """H3/H4, end to end: open -> select ch2 -> Measure -> Apply -> Detect.

    The floor took this recording from a CONDEMNED 0.1 / 0.03 s to a certified
    0.5 / 0.1 s and stopped, because 0.5 / 0.1 is the shortest window
    ``base_vs_state`` tolerates. Its ``baseline_intrusion`` is 0.00428 -- the baseline
    follows 0.43% of every state's depth -- and the stamped open pore is -48.857 pA
    against the app's own event-excluded estimate of -48.218 (1.33% out). One grid
    point up the same statistic reads exactly 0.0 and the stamped open pore is
    -48.2178 pA (0.0005% out). The click now lands there.

    Everything asserted here is what a person sees or gets: the two windows in the
    form, the ladder rendered under them, a plain (not armed) Apply, one press
    applying as ``proposed``, and the table that comes out.
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

    with qtbot.waitSignal(ctrl.finished, timeout=1_800_000):
        bp.probe_button.click()
    _drain(qapp, qtbot, 300)

    pr = bp.proposal()
    assert pr is not None
    got = (round(float(pr.config.rough_s), 6), round(float(pr.config.base_s), 6))
    assert got == NOW, (f"the one-click window is {got}", got)
    assert got != FLOOR_STOPPED_AT, (
        "the proposal is still the shortest window the floor certifies, not the "
        "least-intruding one on the grid")

    # ---- the ladder the app measured, and what it read on each rung -------
    ranking = api(pr, "ranking")
    assert [(c.rough_s, c.base_s) for c in ranking] == [FLOOR_STOPPED_AT, NOW], ranking
    assert ranking[0].fraction == pytest.approx(0.004278771600903435, abs=1e-9)
    assert ranking[1].fraction == 0.0
    assert ranking[1].chosen is True

    # ---- ...and what the SCREEN says about it -----------------------------
    shown = api(bp, "proposal_text")()
    assert "RANKED BY MEASURED INTRUSION" in shown, shown
    assert "0.5 s / 100 ms follows 0.428%" in shown, shown
    assert "1 s / 300 ms follows NONE" in shown, shown
    assert "Ranked: 0.5 s / 100 ms" in shown, shown
    assert "Ranked: 1 s / 300 ms" in shown, shown
    assert "CHOSEN" in shown, shown
    assert "0.4% shallow" in shown, (
        "the screen does not say what the window it declined would have cost a depth",
        shown)
    assert "SHORTEST" in shown, shown

    # ---- THE FLOOR'S PARAGRAPH IS STILL THERE, AND IT NO LONGER READS AS THE OFFER.
    #
    # THIS ASSERTION USED TO BE ``assert "floored at 3x that median" in shown`` AND
    # NOTHING ELSE, and that is the defect the RP batch exists to end: this screen
    # declares the floor's 0.5 s / 100 ms three times ("the rule's rough window
    # 0.5 s", "the rule's base window 100 ms", "the shortest BASE window on the grid
    # that clears it") BEFORE the ranking paragraph takes 1 s / 300 ms, and a test
    # that pins the floor's sentence while asserting nothing about what supersedes it
    # is a test defending an ambiguity. The floor's derivation is still required --
    # ranking is a second step, not a replacement for the sentence that explains
    # base_s -- but the screen must now also say, in words, that those numbers are the
    # RULE's and not what is being offered.
    assert "floored at 3x that median" in shown, (
        "the sentence that explains base_s is gone; ranking is a second step, not a "
        "replacement for it", shown)
    assert "the rule's rough window 0.5 s" in shown, (
        "the derivation still presents the floor's rough window as the offer", shown)
    assert "the rule's base window 100 ms" in shown, (
        "the derivation still presents the floor's base window as the offer", shown)
    assert "IS THE RULE'S WINDOW AND NOT WHAT THE APP OFFERS" in shown, (
        "nothing on the screen tells the reader the three declarations of "
        "0.5 s / 100 ms above have been superseded by the 1 s / 300 ms below them",
        shown)
    assert "SUPERSEDES it with 1 s / 300 ms" in shown, shown
    assert shown.index("floored at 3x that median") < shown.index(
        "IS THE RULE'S WINDOW AND NOT WHAT THE APP OFFERS"), (
        "the correction has to come AFTER the claim it corrects", shown)

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
            round(float(bp.base_s.value()), 6)) == NOW
    assert bp.param_source() == "proposed"
    src = bp.source_label.text()
    assert "measured on this recording, and confirmed" in src, src
    assert "ACCEPTED OVER" not in src, src

    # ---- and the table it produces ----------------------------------------
    dp = ap.detection_panel
    dp.show_detection_tab()
    _drain(qapp, qtbot, 100)
    with qtbot.waitSignal(ctrl.finished, timeout=1_800_000):
        dp.detect_button.click()
    _drain(qapp, qtbot, 400)
    df = ap.current_events_df()
    assert df is not None and len(df) == 1179, len(df) if df is not None else None
    prov = ap.current_detection().payload.provenance
    assert prov.get("baseline_fingerprint") == "sha256:b64a75ebdf29", prov.get(
        "baseline_fingerprint")
    assert str(prov.get("baseline_param_source")) == "proposed"
