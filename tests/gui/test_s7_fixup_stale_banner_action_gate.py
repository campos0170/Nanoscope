"""Sweep-7 fix-up, D2 (minor) + CW-1 (minor): the stale banner's action button takes the
SAME gate as every other door into a run on this target, and shows its reason.

D2, measured on the owner's -1000 mV ABF (115 events on IN15): after "Treat this channel
as: voltage" with gain 2.5, the Detect button greyed out with the override named while the
banner above the table kept a LIVE "Re-run" whose tooltip promised "Run this SAME detection
again under the new gain/offset". Pressing it: no run, no ``runRefused``, status bar '',
banner unchanged, button still enabled. ``_regate`` pushed the new gate to the Detect / ML /
probe buttons only; ``_on_rerun`` -> ``_run_detection`` returned False on the gate with the
comment "the buttons are disabled anyway; belt-and-braces" -- and this button was not one
of those buttons.

CW-1, measured on the owner's 0004 ABF: a stored run whose notch list exceeds the cap was
refused on Re-run with "Fix the list on the Preprocess tab ... and run again"; the user did
exactly that, pressed Re-run again and got the identical 400-character refusal, because
Re-run replays the result's STORED dict and the box's state can never reach it. The door
that runs the box -- the Detect button -- was enabled the whole time and never named.

One resolver (``AnalysisPanel._stale_action_refusal``) now decides whether the banner's
action may be offered, ``_regate`` and ``_judge_staleness`` both push its answer onto the
banner (``_StaleBanner.set_action_gate``), the reason is printed under the message, and the
programmatic doors (``_run_detection`` on the gate, ``_on_rerun``) refuse OUT LOUD through
``runRefused`` instead of returning in silence. Driven through a real MainWindow with a real
Schmitt detection on a synthetic recording; every ``set_calibration`` is ``write_sidecar=False``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtCore, QtTest

from nanoscope.analysis.modules import register_all
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import analysis_panel as AP
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel

FS = 50_000.0
I0 = -1000.0
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]
CAP = getattr(DetectionPanel, "NOTCH_MAX_LINES", 32)   # lazy: the D2 half must run on the pre-cap snapshot
OVER = [60.0 * (i + 1) for i in range(CAP + 1)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _signal(spans, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _as_voltage(gain: float) -> ChannelCalibration:
    """The user's gesture as the Calibration panel commits it: Treat this channel as
    VOLTAGE, with the given gain."""
    return ChannelCalibration(gain, 0.0, "pA", Provenance.MEASURED,
                              "really the command channel", kind=ChannelKind.VOLTAGE)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, qtbot, tmp_path):
    """A real Schmitt detection on screen, its stamps written by the engine."""
    did = window._session.add_document(_doc(tmp_path, _signal(_FRAGMENTS), name="a.dat"))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    qtbot.wait(20)
    panel = window._analysis_panel
    assert panel.can_run()
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    assert panel.current_detection() is not None
    assert not panel._stale.isVisibleTo(panel)
    return window, did


# ================================================================================================
# D2 -- the detection gate reaches the banner's button, in both directions
# ================================================================================================

def test_the_reverse_override_disables_the_banners_rerun_and_prints_the_gates_own_reason(
        detected, qtbot):
    window, did = detected
    panel = window._analysis_panel
    det = panel._detection
    before = panel.current_detection()

    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        window._session.set_calibration(did, 0, _as_voltage(2.5), write_sidecar=False)
    assert not det.detect_button.isEnabled()
    gate_reason = det._gate_label.text()
    assert "your Calibration-panel override" in gate_reason, gate_reason

    banner = panel._stale
    assert banner.isVisibleTo(panel) and banner.kind == "calibration"
    assert banner.action_text() == "Re-run"
    assert not banner._btn.isEnabled(), "a live Re-run on a channel the gate refuses"
    why = banner.action_refusal()
    assert why.startswith("Re-run is unavailable: "), why
    assert gate_reason in why, "ONE gate answer: the banner says what the Detect button says"
    assert banner._why.isVisibleTo(banner), "the reason is on the banner, not only on hover"
    assert banner._btn.toolTip() == why

    # A click on the disabled button is inert, and the stored table is untouched...
    banner._btn.click()
    qtbot.wait(20)
    assert panel._pending is None
    assert panel.current_detection() is before
    # ...and the programmatic door into the same run refuses OUT LOUD.
    with qtbot.waitSignal(panel.runRefused, timeout=1_000) as refused:
        panel._on_rerun()
    assert gate_reason in refused.args[0], refused.args[0]
    assert panel._pending is None
    assert panel.current_detection() is before
    assert banner.isVisibleTo(panel), "the banner is dismissed by a RUN, never by a refusal"
    qtbot.wait(20)
    assert window.statusBar().currentMessage() == refused.args[0]

    # The gate reaches the banner in BOTH directions: take the override back (the gain
    # stays moved, so the banner stays up) and the button is offered again, with its
    # own tooltip back.
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        window._session.set_calibration(
            did, 0, ChannelCalibration(2.5, 0.0, "pA", Provenance.MEASURED),
            write_sidecar=False)
    assert det.detect_button.isEnabled()
    assert banner.isVisibleTo(panel) and banner.kind == "calibration"
    assert banner._btn.isEnabled()
    assert banner.action_refusal() == ""
    assert not banner._why.isVisibleTo(banner)
    assert banner._btn.toolTip() == AP._STALE_ACTION_TIPS["calibration"]


def test_the_menu_route_announces_the_gate_when_it_bypasses_the_buttons(detected, qtbot):
    """MainWindow's Run detection action syncs its enabled state only on a selection
    change, so under an override it can still fire ``trigger_detection``; that path
    used to be the silent ``return False``."""
    window, did = detected
    panel = window._analysis_panel
    window._session.set_calibration(did, 0, _as_voltage(2.5), write_sidecar=False)
    qtbot.wait(20)
    with qtbot.waitSignal(panel.runRefused, timeout=1_000) as refused:
        panel.trigger_detection()
    msg = refused.args[0]
    assert msg.startswith("Detection refused — "), msg
    assert "voltage channel" in msg and "your Calibration-panel override" in msg
    assert panel._pending is None


def test_re_measure_depths_rides_the_same_gate_as_the_probe(detected, qtbot):
    """A kind-only override (gain unchanged, so the calibration stamp still matches)
    followed by a Preprocess knob raises the BASELINE banner; its "Re-measure depths"
    is a baseline job on this channel, and the baseline probe already rides the
    detection gate -- so must this button."""
    window, did = detected
    panel = window._analysis_panel
    window._session.set_calibration(did, 0, _as_voltage(1.0), write_sidecar=False)
    qtbot.wait(20)
    assert not panel._stale.isVisibleTo(panel), "gain 1.0: the calibration stamp matches"
    bl = panel.baseline_panel
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        bl.rough_s.setValue(bl.rough_s.value() * 4)
    banner = panel._stale
    assert banner.isVisibleTo(panel) and banner.kind == "baseline"
    assert banner.action_text() == "Re-measure depths"
    assert not banner._btn.isEnabled()
    why = banner.action_refusal()
    assert why.startswith("Re-measure depths is unavailable: "), why
    assert "your Calibration-panel override" in why
    assert panel._detection._gate_label.text() in why


# ================================================================================================
# CW-1 -- a stored list the box would refuse: Re-run is withheld, and the door is named
# ================================================================================================

def test_a_flooded_stored_list_withholds_rerun_and_names_detect_which_then_runs_the_box(
        detected, qtbot):
    window, did = detected
    panel = window._analysis_panel
    det = panel._detection
    res = panel.current_detection()
    flooded = replace(res, params={**dict(res.params), "notch_enabled": True,
                                   "notch_freqs": OVER})
    window._session.add_result(flooded)
    qtbot.wait(20)
    assert panel.current_detection() is flooded
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        window._session.set_calibration(did, 0, ChannelCalibration(2.0, 0.0, "pA"),
                                        write_sidecar=False)
    banner = panel._stale
    assert banner.isVisibleTo(panel) and banner.kind == "calibration"

    # Refused BEFORE any click, with the count, the cap, and the door that runs the box.
    assert not banner._btn.isEnabled()
    why = banner.action_refusal()
    assert f"{CAP + 1} lines" in why and f"at most {CAP}" in why, why
    assert "press Detect" in why, why
    assert "STORED list" in why and "Preprocess tab" in why, why
    assert det.detect_button.isEnabled(), "the way out must be open while it is named"
    # The programmatic replay says the same sentence.
    with qtbot.waitSignal(panel.runRefused, timeout=1_000) as refused:
        panel._on_rerun()
    assert "press Detect" in refused.args[0]
    assert panel._pending is None and panel.current_detection() is flooded

    # THE USER FOLLOWS THE ADVICE: a legal list typed into the box, then Detect.
    det.notch_use_lines.setChecked(True)
    qtbot.wait(10)
    det.notch_freqs.setFocus()
    det.notch_freqs.selectAll()
    QtTest.QTest.keyClicks(det.notch_freqs, "60, 120")
    QtTest.QTest.keyClick(det.notch_freqs, QtCore.Qt.Key_Tab)
    qtbot.wait(20)
    assert det.params()["notch_freqs"] == [60.0, 120.0]
    assert not banner._btn.isEnabled(), "Re-run still replays the STORED list"
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        det.detect_button.click()
    qtbot.wait(50)
    fresh = panel.current_detection()
    assert fresh is not flooded
    assert fresh.params["notch_freqs"] == [60.0, 120.0]
    assert tuple(fresh.calibration_fingerprint) == \
        tuple(window._session.calibration(did, 0).fingerprint())
    assert not banner.isVisibleTo(panel), "the run the sentence pointed at clears the banner"


def test_a_legal_stored_list_keeps_rerun_offered(detected, qtbot):
    """The control: the gate withholds only what the box would refuse."""
    window, did = detected
    panel = window._analysis_panel
    res = panel.current_detection()
    legal = replace(res, params={**dict(res.params), "notch_enabled": True,
                                 "notch_freqs": [60.0, 120.0]})
    window._session.add_result(legal)
    qtbot.wait(20)
    window._session.set_calibration(did, 0, ChannelCalibration(2.0, 0.0, "pA"),
                                    write_sidecar=False)
    qtbot.wait(20)
    banner = panel._stale
    assert banner.isVisibleTo(panel) and banner.kind == "calibration"
    assert banner._btn.isEnabled() and banner.action_refusal() == ""


def test_the_resolver_is_the_one_reader_of_the_replay_dict():
    """``_on_rerun`` and the banner's gate must judge the SAME dict, so both read it
    through ``_rerun_params``."""
    import inspect
    src = inspect.getsource(AP.AnalysisPanel._on_rerun)
    assert "self._rerun_params(res)" in src
    assert "schmitt_params_from_result(res) or" not in src
    src = inspect.getsource(AP.AnalysisPanel._stale_action_refusal)
    assert "self._rerun_params(res)" in src
