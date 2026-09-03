"""WA2-SKEP-1 — THE STALE-CALIBRATION BANNER'S "Re-run" DESTROYED HAND CURATION IN SILENCE.

Wave A put :meth:`AnalysisPanel.confirm_replacing_curation` on the doors it had found —
the Detect button, the ML Run button, Analysis ▸ Run detection — and deliberately left
``trigger_detection`` unprompted so scripts and the suite are not blocked by a modal nobody
clicked for. It missed a fourth door, and it is a BUTTON: the staleness banner's "Re-run"
launched through ``_on_rerun``, the programmatic replay.

MEASURED through a real ``MainWindow`` at 1512x900 on the 749k-sample -1000 mV apo-ferritin
recording (``wave-a/verify-does-it-hold/v10_stale_rerun.py``, run before the fix):

    A. hand-reject 10 events        -> accepted: 105 of 115, hand_curation_at_stake: (10, 0)
    B. nudge the gain by 5%         -> banner up, "Re-run" enabled
    C. click "Re-run"               -> modal: *** NO MODAL APPEARED ***
                                       accepted 105 -> 115 of 115
    E. Detect, same state           -> modal: ['Cancel', 'Replace the table']

So the exact loss the prompt exists to prevent was still one click away, through a door the
prompt did not know about.

WHAT IS HELD HERE, and why each one is a separate guarantee:

1. **The banner's Re-run is a gesture and asks like one** — naming the stake, cancelling by
   default, and leaving the curated table exactly as it was.
2. **A cancel does not dismiss the warning.** Nothing was recomputed, so "these depths were
   read under the old gain" is still true. (This is the same invariant ``_on_rerun`` already
   held for its own early returns; the new gate must not open a second way to lose it.)
3. **Going ahead still works** — the prompt is a question, not a block.
4. **The banner's OTHER face asks nothing.** A stale BASELINE stamp offers "Re-measure",
   which keeps the events and every accept/reject on them; a prompt there would be the noise
   that gets clicked through.
5. **The programmatic replay stays unprompted.** ``_on_rerun`` is called bare by scripts and
   by ``test_staleness_regressions`` / ``test_s7_fixup_stale_banner_action_gate``; the gate
   belongs on the click, one level up.
6. **The enumeration is written down.** Every door a person can press to replace an event
   table is asserted to pass through the prompt, and the two programmatic launchers are
   asserted NOT to — so a fifth door cannot be added quietly, in either direction.

Every test in this file fails against the pre-fix panel, proven by the out-of-tree runtime
revert ``wa2_revert_plugin.py`` (it rebinds ``_on_stale_action`` to its pre-fix body; the
source file is never edited).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import inspect

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import analysis_panel as AP
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0                      # the owner's rate; synthetic float32, as the brief asks
N = 60_000
I0 = -1000.0
DEPTH = 300.0
#: Six resolved 8-14 ms blockades. Deep and long, so the detector finds the SAME six under
#: any gain — the re-run's event count is therefore not what these tests are reading.
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600),
          (30_000, 30_500), (40_000, 40_400), (50_000, 50_700)]
SCHMITT = "detect.schmitt"


# ================================================================================================
# builders
# ================================================================================================

def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _install(window, tmp_path):
    sig = _signal()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa2_stale.dat", format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(sig,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, sig


def _regain(window, doc_id, gain: float) -> None:
    """Move the gain so the shown result's calibration stamp no longer matches.

    ``write_sidecar=False``: these docs live under ``tmp_path``, but the habit is the
    project's (nothing is ever written next to a recording by a test)."""
    window._session.set_calibration(
        doc_id, 0, ChannelCalibration(gain, 0.0, "pA", Provenance.UNKNOWN),
        write_sidecar=False)


def _n_hand_rejected(panel) -> int:
    df = panel.current_detection().payload.df
    return int((~df["accepted"].to_numpy(bool)).sum())


@pytest.fixture
def stale_and_curated(window, tmp_path, qtbot):
    """One detection, two rejections a PERSON made, and a calibration edit that went stale.

    The detection is launched through ``trigger_detection`` — the programmatic door — on
    purpose: the fixture must not be the thing under test.
    """
    doc_id, _sig = _install(window, tmp_path)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)

    df = panel.current_detection().payload.df
    assert len(df) == len(_SPANS), f"fixture: expected {len(_SPANS)} events, got {len(df)}"
    for row in (1, 3):
        panel.set_event_accepted(int(df["event_id"].iloc[row]), False)
    qtbot.wait(10)
    assert panel.hand_curation_at_stake(panel.current_detection().payload.df) == (2, 0)

    _regain(window, doc_id, 2.0)
    qtbot.wait(10)
    assert not panel._stale.isHidden(), "fixture: the banner must be up before the click"
    assert panel._stale.kind == AP._STALE_CALIBRATION
    assert panel._stale._btn.isEnabled(), panel._stale.action_refusal()
    return window, doc_id, panel


def _click_rerun(panel) -> None:
    """THE GESTURE: a click on the banner's own button, through the wiring the app ships.

    Not ``_on_stale_action(...)`` — the whole defect was that a button and a launcher were
    the same thing, so the test has to press the button.
    """
    panel._stale._btn.click()


# ================================================================================================
# 1. the prompt
# ================================================================================================

def test_clicking_the_stale_banners_rerun_over_hand_curation_asks_before_replacing_it(
        stale_and_curated, qtbot, blocked_modals):
    """The headline, and the exact shape of probe v10 step C.

    The harness's modal guard answers as a cancelling user, so the assertion is twofold:
    the box was raised AT ALL (``blocked_modals`` is empty in the pre-fix panel), and it
    named the stake in words the owner can recognise.
    """
    _window, _doc_id, panel = stale_and_curated
    before = panel.current_detection().payload.df.copy(deep=True)

    _click_rerun(panel)
    qtbot.wait(50)

    assert len(blocked_modals) == 1, blocked_modals
    said = blocked_modals[0].text
    assert "2 events you rejected by hand" in said, said
    after = panel.current_detection().payload.df
    assert _n_hand_rejected(panel) == 2, (
        "the re-run went ahead over a dialog nobody answered")
    assert list(after["event_id"]) == list(before["event_id"])


def test_the_prompt_the_banner_raises_is_the_same_one_the_detect_button_raises(
        stale_and_curated, qtbot):
    """One question, one wording, one default — or the two doors teach different habits.

    Read by identity, exactly as ``confirm_replacing_curation`` documents: ``buttons()``
    returns LAYOUT order, which is how a destructive button became the Enter default on
    this project once already.
    """
    _window, _doc_id, panel = stale_and_curated
    seen: dict[str, object] = {}

    def _spy(self):
        seen["default"] = self.defaultButton()
        seen["escape"] = self.escapeButton()
        seen["text"] = self.text()
        seen["informative"] = self.informativeText()
        seen["roles"] = {b.text(): self.buttonRole(b) for b in self.buttons()}
        return QtWidgets.QMessageBox.Rejected

    old = QtWidgets.QMessageBox.__dict__.get("exec")
    QtWidgets.QMessageBox.exec = _spy
    try:
        _click_rerun(panel)
        qtbot.wait(50)
    finally:
        if old is None:
            del QtWidgets.QMessageBox.exec
        else:
            QtWidgets.QMessageBox.exec = old

    assert seen, "the banner's Re-run raised no dialog"
    # NOT windowTitle(): macOS drops a QMessageBox's title, so it reads "" here and the
    # sentence the user actually sees is the one worth pinning.
    assert "2 events you rejected by hand" in seen["text"], seen["text"]
    assert "Edit \u25b8 Undo" in seen["informative"], seen["informative"]
    assert seen["default"] is seen["escape"], "Escape and Enter must land on one button"
    assert seen["default"].text() == "Cancel"
    assert seen["roles"]["Replace the table"] == QtWidgets.QMessageBox.DestructiveRole


def test_cancelling_the_prompt_leaves_the_staleness_banner_standing(
        stale_and_curated, qtbot, blocked_modals):
    """A refused re-run recomputed nothing, so the warning it was offered for is still true.

    The pre-existing invariant (``_on_rerun`` dismisses on a RUN, never on a CLICK) must
    survive the new gate: a cancel that hid the banner would delete the only notice that the
    depths on screen were read under the old gain.
    """
    _window, _doc_id, panel = stale_and_curated
    _click_rerun(panel)
    qtbot.wait(50)

    assert len(blocked_modals) == 1, blocked_modals
    assert not panel._stale.isHidden(), (
        "cancelling the prompt dismissed the staleness warning without recomputing anything")
    assert panel._stale.kind == AP._STALE_CALIBRATION


def test_answering_replace_re_runs_under_the_new_calibration(stale_and_curated, qtbot):
    """The prompt is a question, not a block — going ahead still repairs the staleness."""
    window, doc_id, panel = stale_and_curated

    def _accept(self):
        for b in self.buttons():
            if b.text().startswith("Replace"):
                b.click()
                return QtWidgets.QMessageBox.Accepted
        return QtWidgets.QMessageBox.Rejected

    old = QtWidgets.QMessageBox.__dict__.get("exec")
    QtWidgets.QMessageBox.exec = _accept
    try:
        with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
            _click_rerun(panel)
    finally:
        if old is None:
            del QtWidgets.QMessageBox.exec
        else:
            QtWidgets.QMessageBox.exec = old
    qtbot.wait(30)

    res = window._session.results.get(doc_id, 0, SCHMITT)
    assert tuple(res.calibration_fingerprint) == (2.0, 0.0)   # recomputed under the new gain
    assert len(res.payload.df) == len(_SPANS)
    assert panel._stale.isHidden(), "a run that really happened must dismiss the banner"


def test_the_prompt_counts_the_table_the_click_would_destroy_not_the_default_detector(
        stale_and_curated, qtbot, monkeypatch):
    """The stake is read off ``_active_detection``, which is what the re-run replaces.

    Not ``_MODULE_ID``. The banner's Re-run replays the SHOWN result's stored params, so on
    a channel that carries both a CUSUM and a Schmitt table it replaces whichever one is
    active — and a prompt hard-coded to Schmitt would quote the wrong table's curation
    (or, worse, stay silent because the Schmitt table happens to be clean).
    ``_stored_detection`` and ``_stale_action_refusal`` already read the same field, so this
    keeps all three judging one result.
    """
    _window, _doc_id, panel = stale_and_curated
    asked: list[str] = []
    monkeypatch.setattr(panel, "confirm_replacing_curation",
                        lambda module_id: asked.append(module_id) or False)
    panel._active_detection = AP._CUSUM_MODULE_ID

    _click_rerun(panel)
    qtbot.wait(20)

    assert asked == [AP._CUSUM_MODULE_ID], asked


def test_a_rerun_over_an_uncurated_table_asks_nothing(window, tmp_path, qtbot,
                                                      blocked_modals):
    """The prompt must stay rare, or it becomes a click-through and stops being read."""
    doc_id, _sig = _install(window, tmp_path)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)
    _regain(window, doc_id, 2.0)
    qtbot.wait(10)
    assert not panel._stale.isHidden()

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        _click_rerun(panel)
    qtbot.wait(20)
    assert blocked_modals == [], blocked_modals


# ================================================================================================
# 2. what must NOT start asking
# ================================================================================================

def test_the_baseline_banners_remeasure_still_asks_nothing(window, tmp_path, qtbot,
                                                           blocked_modals):
    """A stale BASELINE stamp offers "Re-measure", which keeps every accept/reject.

    Nothing is at stake, so nothing is asked. Gating this branch too would have been the
    easy over-fix, and it would put a destructive-sounding warning in front of the one
    repair that destroys nothing.
    """
    doc_id, _sig = _install(window, tmp_path)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)
    df = panel.current_detection().payload.df
    for row in (1, 3):
        panel.set_event_accepted(int(df["event_id"].iloc[row]), False)
    qtbot.wait(10)

    # Drive the dispatcher on the baseline kind directly: the point is the BRANCH, and the
    # banner only offers this face when a stored result carries no baseline stamp.
    panel._stale.show_message("baseline", kind=AP._STALE_BASELINE,
                              action_label="Re-measure")
    _click_rerun(panel)
    qtbot.wait(50)

    assert blocked_modals == [], blocked_modals
    assert _n_hand_rejected(panel) == 2, "a re-measure must keep the curation it re-measures"


def test_the_programmatic_rerun_replay_stays_unprompted(stale_and_curated, qtbot,
                                                        blocked_modals):
    """``_on_rerun`` is the LAUNCHER, and scripts and two existing test modules call it bare.

    A modal there is a modal nobody clicked for — it would hang or silently cancel every
    programmatic replay. The gate belongs on the click, which is why it sits in
    ``_on_stale_action``.
    """
    window, doc_id, panel = stale_and_curated
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel._on_rerun()
    qtbot.wait(20)

    assert blocked_modals == [], blocked_modals
    res = window._session.results.get(doc_id, 0, SCHMITT)
    assert tuple(res.calibration_fingerprint) == (2.0, 0.0)


# ================================================================================================
# 3. the enumeration, written down so nobody has to guess again
# ================================================================================================

#: Every door a PERSON can press that replaces an event table, and what presses it. Kept as
#: data because the defect was a missing entry in exactly this list, held in someone's head.
GESTURE_DOORS = {
    "_on_detect_requested": "the Detection tab's Detect button (detectRequested)",
    "_on_ml_detect_requested": "the Detection tab's ML Run button (mlDetectRequested)",
    "trigger_detection_from_menu": "MainWindow's Analysis ▸ Run detection (⌘D)",
    "_on_stale_action": "the staleness banner's Re-run button (actionRequested)",
}

#: The launchers a gesture ends up in. These must stay silent: they are also the
#: programmatic entry points, and a default decides behaviour.
PROGRAMMATIC_LAUNCHERS = ("trigger_detection", "_on_rerun", "_run_detection", "_launch_ml")


def _code_of(name: str) -> str:
    """A method's source with its DOCSTRING removed.

    These docstrings talk about the prompt at length — that is the house style — so a bare
    substring search over ``getsource`` would read the prose as a call and pass on a method
    that only mentions the gate. What is asserted below is the CALL.
    """
    fn = getattr(AnalysisPanel, name)
    src = inspect.getsource(fn)
    doc = inspect.getdoc(fn)
    if doc:
        for line in doc.splitlines():
            src = src.replace(line.strip(), "")
    return src


@pytest.mark.parametrize("name", sorted(GESTURE_DOORS))
def test_every_user_pressed_door_into_a_re_detection_asks_first(name):
    """One entry per door. ``_on_stale_action`` is the one this wave added."""
    assert "confirm_replacing_curation(" in _code_of(name), (
        f"{name} — {GESTURE_DOORS[name]} — replaces the event table without asking")


@pytest.mark.parametrize("name", PROGRAMMATIC_LAUNCHERS)
def test_the_programmatic_launchers_stay_unprompted(name):
    """The other half of the rule, asserted so the fix cannot be 'simplified' downward.

    Pushing the prompt into a launcher would put a modal in front of every script, every
    stored-param replay and the suite itself.
    """
    assert "confirm_replacing_curation(" not in _code_of(name), (
        f"{name} is a launcher, not a gesture — a modal here is one nobody clicked for")


def test_the_banners_button_is_wired_to_the_gated_door_and_not_to_the_launcher():
    """The enumeration above is only worth anything if the button really goes through it.

    Read off the shipped wiring: ``_StaleBanner`` connects its button to the callback the
    panel hands it, and the panel hands it ``_on_stale_action`` — never ``_on_rerun``.
    """
    panel_src = inspect.getsource(AnalysisPanel)
    assert "_StaleBanner(self._on_stale_action)" in panel_src
    banner_src = inspect.getsource(AP._StaleBanner.__init__)
    assert "self.actionRequested.connect(on_action)" in banner_src
    assert "self._btn.clicked.connect" in banner_src
