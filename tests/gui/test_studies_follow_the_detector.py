"""The three validation studies follow the detector, and their results are tied to it.

THE DEFECT. The Detection tab has two detectors, Schmitt and sequential CUSUM, and three
study buttons under them: *Sweep threshold…*, *Null control…* and *Recovery map…*. All
three ran the SCHMITT detector whatever the combo said: ``DetectionPanel.surrogate_params``
removed ``"detector"`` and the CUSUM thresholds before the dict left the form, and the
handlers called ``detect_events``. So with CUSUM selected:

* *Null control…* printed a false-positive count for a detector the user was not using,
  and nothing on the readout said which detector it was.
* The result was stored per (recording, channel) only. A Schmitt count measured earlier
  stayed on the Summary tab under a CUSUM table of the same recording, and the Schmitt
  recovery map weighted a dwell fit of CUSUM rows (the fit then printed "measured eta(t)").

THE RULE NOW.

* *Null control…* runs the selected detector. With CUSUM it makes the call a CUSUM Detect
  makes: the form's CUSUM thresholds, ``merge_gap_s=None`` forwarded when the gap is off,
  the Baseline panel's configuration, and the detector's own polarity vote on Auto. The
  Schmitt call is unchanged (no ``detector=`` argument, the same seed line).
* *Null control…* refuses CUSUM while High-pass is enabled: the far edge of each event
  then also fires in the opposite direction, so the detector's own polarity vote can be
  reversed and the opposite-direction count is not a noise count. It also refuses a
  detector id that is neither Schmitt nor CUSUM.
* *Sweep threshold…* and *Recovery map…* are Schmitt-only: disabled with CUSUM selected,
  with the reason under the combo (at the Advanced control level, where the buttons
  are on screen), and their handlers refuse a CUSUM dict.
* A Null control result carries its ``detector`` and ``thresholds``. It is shown only
  while the table on screen is that detector's (with no table, the form's detector
  decides). A result with no tag is a Schmitt result. A result that is on file but not
  shown is not "not measured": the readout says what it was measured for and which
  table shows it. The efficiency correction is on offer, and applied, only for a
  Schmitt table; under another table its box is greyed out and keeps the user's tick,
  and a map that lands there says so on the status line.
* A CUSUM result's readout names the low-pass beside k and h and leads with the
  opposite-direction count.
* Nothing is saved with the project.

Every test here drives the real widgets on synthetic traces. The stubs of
``truth.null_control`` keep the real keyword contract.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import inspect
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from nanoscope.analysis.baseline import DEFAULT_BASELINE
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detect_cusum import detect_events_cusum
from nanoscope.analysis.modules import DETREND_PARAM_KEYS, FILTER_PARAM_KEYS
from nanoscope.analysis.truth import synth_trace
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel, StatsPanel, SweepDialog

FS = 50_000.0
N = 130_000
N_PLANTED = 30

SCHMITT = "detect.schmitt"
CUSUM = "detect.cusum"

NOTE_LINE = "Sweep threshold and Recovery map work with Schmitt only. Null control runs CUSUM."
SWEEP_TIP = "Counts events across a range of the Schmitt enter threshold k_in. Schmitt only."
NULL_TIP = ("Counts what the selected detector finds on this recording's own noise with the "
            "events removed.")
RECOVERY_TIP = ("Measures how many injected events the Schmitt detector recovers, for the "
                "dwell-fit efficiency correction. Schmitt only.")
SWEEP_REFUSAL = ("Sweep threshold works with the Schmitt detector only. Select Schmitt, or "
                 "use Null control with CUSUM.")
RECOVERY_REFUSAL = ("Recovery map works with the Schmitt detector only. Select Schmitt to "
                    "measure it.")
EFFICIENCY_TIP = ("Weights the dwell fit by the detection efficiency the Recovery map "
                  "measured with the Schmitt detector. Used only for fits of a Schmitt table.")
OTHER_TABLE_REASON = ("This recording's recovery map was measured with Schmitt, so it is "
                      "not used for this table.")
NULL_HIGHPASS_REFUSAL = (
    "Null control with CUSUM needs High-pass off: with it on, the far edge of each event "
    "also fires in the opposite direction. Turn High-pass off and run it again.")
NULL_DETECTOR_REFUSAL = "Null control works with the Schmitt and CUSUM detectors only."
RECOVERY_LANDED = "Recovery map finished for Schmitt; it is used for fits of a Schmitt table."
SCHMITT_LABEL = "Schmitt (k_in 5, k_out 2)"
CUSUM_LABEL = "CUSUM (k 0.75, h 28, no low-pass)"

#: A null result that prints a percentage, as `truth.null_control` returns it.
_RATE_SHAPE = {
    "n_real": 200, "n_event_free": 3,
    "per_surrogate": [8, 12, 10, 9, 11],
    "n_surrogate_mean": 10.0, "n_surrogate_std": 1.6,
    "estimated_fdr": 0.05, "notes": (),
}

#: A null result that refuses a rate (the common case on real recordings).
_REFUSAL_SHAPE = {
    "n_real": 200, "n_event_free": 12,
    "per_surrogate": [300, 280, 310, 290, 305],
    "n_surrogate_mean": 297.0, "n_surrogate_std": 12.0,
    "estimated_fdr": float("nan"),
    "notes": ("the AAFT surrogates are not the same noise as this recording.",),
}


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 3) -> np.ndarray:
    """30 blockades of 2-5 ms and 20-40 pA on a 100 pA open pore with 1.5 pA white noise.

    Long and deep enough that both detectors find all of them at their default settings
    (CUSUM keeps events of 1 ms and longer), so each table clears the fitter's n >= 20 gate.
    """
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    pos = 2_000
    for _ in range(N_PLANTED):
        dwell = int(rng.integers(100, 250))
        x[pos:pos + dwell] -= float(rng.uniform(20.0, 40.0))
        pos += 4_000
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray, fs: float = FS) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=int(arr.size),
        sample_rate_hz=fs, rate_provenance=Provenance.FILE_META, t0_s=0.0, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _write_csv(path: Path, arr: np.ndarray) -> None:
    """A time,current CSV the text reader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


def _open(window: MainWindow, tmp_path: Path, name: str = "pore_a.dat",
          arr: np.ndarray | None = None) -> tuple[str, int]:
    """One synthetic recording whose ``meta.path`` exists (the Null control seed reads it)."""
    path = tmp_path / name
    path.write_bytes(b"synthetic " + name.encode() * 64)
    return _install(window, _doc(path, _trace() if arr is None else arr))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._controller.abort()


def _detect(window: MainWindow, qtbot) -> None:
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(30)      # let the queued add_result -> resultsChanged settle


def _choose(window: MainWindow, qtbot, module_id: str) -> None:
    """Move the Detection tab's detector combo, the way the user does."""
    window._analysis_panel.detection_panel.set_detector(module_id)
    qtbot.wait(30)


def _pick_table(window: MainWindow, qtbot, name: str) -> None:
    """Choose a table in the Results toolbar's Table selector."""
    sel = window._analysis_panel._result_selector
    idx = sel.findText(name, Qt.MatchStartsWith)
    assert idx >= 0, [sel.itemText(i) for i in range(sel.count())]
    sel.setCurrentIndex(idx)
    qtbot.wait(30)


def _two_tables(window: MainWindow, qtbot, tmp_path: Path) -> tuple[str, int]:
    """A Schmitt table AND a CUSUM table on one recording, the CUSUM one on screen."""
    sel = _open(window, tmp_path)
    _detect(window, qtbot)
    _choose(window, qtbot, CUSUM)
    _detect(window, qtbot)
    assert _shown(window) == CUSUM, "fixture: the CUSUM table should be on screen"
    for module_id in (SCHMITT, CUSUM):
        rows = len(window._session.results.get(*sel, module_id).payload.df)
        assert rows >= 20, f"fixture: {module_id} found {rows} of {N_PLANTED} events"
    return sel


def _shown(window: MainWindow) -> str | None:
    """The module id of the table on screen, or None."""
    res = window._analysis_panel.current_detection()
    return None if res is None else res.module_id


def _deliver(panel, kind: str, target, payload) -> None:
    """Feed a study answer through the REAL controller sink, as the off-thread job does."""
    panel._pending, panel._pending_sel = kind, target
    panel._on_controller_finished(payload)


def _tagged(shape: dict, detector: str, *, lowpass_hz: float | None = None,
            **extra) -> dict:
    """``shape`` as the panel's worker stamps it: with its seed, detector and thresholds.

    The CUSUM stamp carries the low-pass cutoff too (``None`` when it is off).
    """
    out = dict(shape)
    out["per_surrogate"] = list(shape["per_surrogate"])
    out["seed"] = 20261110
    out["detector"] = detector
    out["thresholds"] = (
        {"cusum_k": 0.75, "cusum_h": 28.0, "filter_cutoff_hz": lowpass_hz}
        if detector == CUSUM else {"k_in": 5.0, "k_out": 2.0})
    out.update(extra)
    return out


def _cusum_lead(label: str, n_opposite: int, n_real: int, n_event_free: int) -> str:
    """What a CUSUM result's readout leads with."""
    return (f"{label}: {n_opposite} opposite-direction detection(s) against {n_real} on "
            f"the recording. {n_event_free} on its event-free noise, a weak lower bound "
            f"for CUSUM.")


def _hidden(label: str, name: str) -> tuple[str, str]:
    """The readout (text, tooltip) for a result that is on file but not shown here."""
    return (f"false-positive rate: measured for {label}; shown with a {name} table.", "")


def _eta_frame(scale: float = 3e-4) -> pd.DataFrame:
    """A recovery_map-shaped efficiency table: eta rises with dwell."""
    t = np.geomspace(2e-5, 5e-3, 24)
    eta = np.clip(1.0 - np.exp(-t / scale), 0.02, 1.0)
    return pd.DataFrame({"dwell_s": t, "efficiency": eta})


def _wait_for_fit(stats, qtbot, timeout_s: float = 60.0) -> None:
    """Wait, bounded, for the Fits tab's background fit to land.

    ``qtbot.wait`` rather than a tight ``QTest.qWait`` loop: it sleeps in an event loop
    with the interpreter lock released, so the worker thread gets to run. The fit
    weighted by an efficiency table bootstraps in Python and starves under the tight
    loop (it does not finish in a minute there, on untouched code as well).
    """
    deadline = time.monotonic() + timeout_s
    while getattr(stats, "_fit_busy", False) and time.monotonic() < deadline:
        qtbot.wait(50)
    assert not getattr(stats, "_fit_busy", False), "the fit worker did not finish"


def _readout(window: MainWindow) -> tuple[str, str]:
    label = window._analysis_panel._stats.fdr_label
    return label.text(), label.toolTip()


def _record_null_calls(monkeypatch, shape: dict = _RATE_SHAPE) -> list[dict]:
    """Replace `truth.null_control` with a recorder that keeps its keyword contract.

    The panel's job imports the function at call time on the worker thread, so patching
    the module attribute is what the worker picks up. The named keywords are the real
    ones; anything else the panel passes lands in ``extra`` and is asserted on, so a
    Schmitt call that grew a ``detector=`` argument is seen, not swallowed.
    """
    from nanoscope.analysis import truth

    calls: list[dict] = []

    def recorder(rec, *, detect_kwargs=None, n_surrogates=5, rng=None, cancel=None,
                 **extra):
        calls.append({"rec": rec, "rng": rng, "n_surrogates": n_surrogates,
                      "detect_kwargs": dict(detect_kwargs or {}), "extra": dict(extra)})
        out = dict(shape)
        out["per_surrogate"] = list(shape["per_surrogate"])
        if "detector" in extra:
            out["n_opposite"] = 7
        return out

    monkeypatch.setattr(truth, "null_control", recorder)
    return calls


def _click_null(window: MainWindow, qtbot) -> tuple[str, str]:
    """Press *Null control…* and drain the loop. Returns what the user then reads."""
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=120_000):
        panel.detection_panel.null_button.click()
    qtbot.wait(40)
    return _readout(window)


def _visible_text(root: QtWidgets.QWidget) -> str:
    """Every string a user could read off ``root``, as one blob (structure-blind).

    ``isVisibleTo(root)`` rather than ``isVisible()``: an offscreen test never realises
    the window, and ``isVisible()`` is False for every child of an unshown top level.
    """
    out: list[str] = []
    for w in root.findChildren(QtWidgets.QWidget):
        if not w.isVisibleTo(root):
            continue
        for attr in ("text", "toPlainText"):
            fn = getattr(w, attr, None)
            if callable(fn):
                try:
                    val = fn()
                except (TypeError, RuntimeError):
                    continue
                if isinstance(val, str) and val:
                    out.append(val)
    return "\n".join(out)


# ================================================================================================
# the form: which dict, which buttons
# ================================================================================================

def test_the_study_dict_is_the_selected_detectors(qtbot):
    """Schmitt: the dict the studies always got. CUSUM: CUSUM's, and it says so."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)

    schmitt = panel.surrogate_params()
    assert "detector" not in schmitt
    assert not {"cusum_k", "cusum_h"} & set(schmitt)
    assert {"k_in", "k_out", "noise_policy", "noise_window_s"} <= set(schmitt)
    # Exactly `params()` minus the three keys it always dropped: not one key more or
    # fewer, because the Null control seed is a hash of this dict.
    assert schmitt == {k: v for k, v in panel.params().items()
                       if k not in {"detector", "cusum_k", "cusum_h"}}

    panel.set_detector(CUSUM)
    cusum = panel.surrogate_params()
    assert cusum["detector"] == CUSUM
    assert {"cusum_k", "cusum_h"} <= set(cusum)
    assert not {"k_in", "k_out", "noise_policy", "noise_window_s"} & set(cusum), sorted(cusum)
    # Everything the two detectors share is still there.
    assert set(cusum) - {"detector", "cusum_k", "cusum_h"} == (
        set(schmitt) - {"k_in", "k_out", "noise_policy", "noise_window_s"})


def test_sweep_and_recovery_are_disabled_with_cusum_and_say_why(qtbot):
    """Two of the three buttons go grey with CUSUM, and the reason is under the combo.

    ``set_runnable(True)`` is re-pushed by the owner whenever the run gate moves; it used
    to enable all four buttons, so it must not bring the two Schmitt-only ones back.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_control_level("advanced")         # where the three study buttons are shown
    tabs = panel.mode_tabs
    tabs.setCurrentIndex([tabs.tabText(i) for i in range(tabs.count())].index("Detection"))
    sweep, null, recovery = panel.sweep_button, panel.null_button, panel.recovery_button

    assert sweep.toolTip() == SWEEP_TIP
    assert null.toolTip() == NULL_TIP
    assert recovery.toolTip() == RECOVERY_TIP
    assert sweep.isEnabled() and null.isEnabled() and recovery.isEnabled()
    assert NOTE_LINE not in panel.detector_note.text()

    panel.set_detector(CUSUM)
    assert not sweep.isEnabled() and not recovery.isEnabled()
    assert null.isEnabled(), "Null control runs CUSUM; it must stay live"
    note = panel.detector_note.text()
    assert NOTE_LINE in note, f"nothing under the combo says why two buttons are grey: {note!r}"
    assert "DIMENSIONLESS" in note, "the existing CUSUM sentence went missing"
    assert panel.detector_note.isVisibleTo(panel), "the reason is not on screen"

    panel.set_runnable(True)
    assert not sweep.isEnabled() and not recovery.isEnabled(), (
        "the owner's run gate re-enabled a Schmitt-only study with CUSUM selected")
    assert null.isEnabled() and panel.detect_button.isEnabled()

    panel.set_runnable(False, "Select a CURRENT channel to run detection.")
    for button in (panel.detect_button, sweep, null, recovery):
        assert not button.isEnabled()
    panel.set_detector(SCHMITT)                 # the detector alone does not open the gate
    for button in (panel.detect_button, sweep, null, recovery):
        assert not button.isEnabled()
    panel.set_detector(CUSUM)

    panel.set_runnable(True)
    assert null.isEnabled() and not sweep.isEnabled() and not recovery.isEnabled()

    panel.set_detector(SCHMITT)
    assert sweep.isEnabled() and null.isEnabled() and recovery.isEnabled()
    assert NOTE_LINE not in panel.detector_note.text()
    # The tooltips are constants of the buttons, not of the detector.
    assert (sweep.toolTip(), null.toolTip(), recovery.toolTip()) == (
        SWEEP_TIP, NULL_TIP, RECOVERY_TIP)


def test_the_schmitt_only_line_is_shown_only_where_the_buttons_are(qtbot):
    """At the Essential control level the three study buttons are hidden, and the note
    under the detector combo must not name them.

    The combo and the note stay on screen at Essential. The line "Sweep threshold and
    Recovery map work with Schmitt only. Null control runs CUSUM." was written whenever
    CUSUM was selected, so an Essential user read a sentence about three controls that
    were not there.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    tabs = panel.mode_tabs
    tabs.setCurrentIndex([tabs.tabText(i) for i in range(tabs.count())].index("Detection"))
    buttons = (panel.sweep_button, panel.null_button, panel.recovery_button)
    note = panel.detector_note

    def on_screen() -> list[bool]:
        return [b.isVisibleTo(panel) for b in buttons]

    panel.set_control_level("advanced")
    panel.set_detector(CUSUM)
    assert on_screen() == [True, True, True]
    assert note.text().endswith("\n" + NOTE_LINE), note.text()

    # The level changes: the line goes with the buttons, the rest of the note stays.
    panel.set_control_level("essential")
    assert on_screen() == [False, False, False], "fixture: Essential hides the study buttons"
    assert note.isVisibleTo(panel) and panel.detector_combo.isVisibleTo(panel)
    assert NOTE_LINE not in note.text(), (
        f"the note names three buttons that are not on screen: {note.text()!r}")
    assert "Sweep threshold" not in note.text() and "Null control" not in note.text()
    assert "DIMENSIONLESS" in note.text() and "\n" not in note.text(), note.text()

    # The detector changes while at Essential: still no line, for either detector.
    panel.set_detector(SCHMITT)
    assert note.text() == "k_in and k_out are DIMENSIONLESS — multiples of the MAD noise sigma."
    panel.set_detector(CUSUM)
    assert NOTE_LINE not in note.text() and "DIMENSIONLESS" in note.text()

    # Back at Advanced the buttons and the line return together, without a detector change.
    panel.set_control_level("advanced")
    assert on_screen() == [True, True, True]
    assert note.text().endswith("\n" + NOTE_LINE), note.text()
    # The level never touched the buttons' gates.
    assert not panel.sweep_button.isEnabled() and not panel.recovery_button.isEnabled()
    assert panel.null_button.isEnabled()

    panel.set_detector(SCHMITT)
    for level in ("essential", "advanced"):
        panel.set_control_level(level)
        assert note.text() == (
            "k_in and k_out are DIMENSIONLESS — multiples of the MAD noise sigma.")


def test_the_sweep_dialog_says_which_threshold_it_sweeps(qtbot):
    df = pd.DataFrame({"k": [1.0, 2.0, 3.0, 4.0, 5.0],
                       "n_events": [200.0, 60.0, 48.0, 47.0, 46.0],
                       "n_surrogate": [5.0, 1.0, 0.0, 0.0, 0.0]})
    df.attrs["k_recommended"] = 3.5
    dlg = SweepDialog(df)
    qtbot.addWidget(dlg)
    assert dlg.windowTitle() == "Threshold sweep (Schmitt k_in)"


# ================================================================================================
# the handlers
# ================================================================================================

def test_the_handlers_refuse_cusum(window, qtbot, tmp_path, monkeypatch):
    """The buttons are disabled, and the handlers refuse a CUSUM dict by any other route."""
    _open(window, tmp_path)
    panel = window._analysis_panel
    _choose(window, qtbot, CUSUM)
    cusum = panel.detection_panel.surrogate_params()
    assert cusum["detector"] == CUSUM

    refused: list[str] = []
    panel.runRefused.connect(refused.append)
    dispatched: list[str] = []
    monkeypatch.setattr(
        panel._controller, "run_callable",
        lambda fn, label="callable", **kw: dispatched.append(label) or True)

    panel._on_sweep_requested(dict(cusum))
    assert refused == [SWEEP_REFUSAL]
    assert panel._pending is None and dispatched == []

    panel._on_recovery_requested(dict(cusum))
    assert refused == [SWEEP_REFUSAL, RECOVERY_REFUSAL]
    assert all("Schmitt" in message for message in refused)
    assert panel._pending is None and dispatched == []

    # Control: the same dict is what Null control runs, and a Schmitt dict is not refused.
    panel._on_null_requested(dict(cusum))
    assert dispatched == ["null_control"] and panel._pending == "null"
    panel._pending = panel._pending_sel = None

    _choose(window, qtbot, SCHMITT)
    schmitt = panel.detection_panel.surrogate_params()
    panel._on_sweep_requested(dict(schmitt))
    panel._pending = panel._pending_sel = None
    panel._on_recovery_requested(dict(schmitt))
    panel._pending = panel._pending_sel = None
    assert dispatched == ["null_control", "sweep_threshold", "recovery_map"]
    assert len(refused) == 2, f"a Schmitt study was refused: {refused[2:]}"


def test_null_control_refuses_cusum_while_high_pass_is_on(window, qtbot, tmp_path, monkeypatch):
    """CUSUM + High-pass: refused through the real button, on Auto and with an override.

    The study detects on the high-passed trace, where the far edge of each event also
    fires in the opposite direction. On Auto the detector's vote there can be reversed
    (a CUSUM Detect re-measures the direction without the High-pass, the study cannot),
    so the study ran the other way from the table beside it; with an override the
    opposite-direction count it leads with is still the far edges, not noise.
    """
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    form = panel.detection_panel
    null = form.null_button
    refused: list[str] = []
    panel.runRefused.connect(refused.append)
    dispatched: list[str] = []
    monkeypatch.setattr(
        panel._controller, "run_callable",
        lambda fn, label="callable", **kw: dispatched.append(label) or True)

    _choose(window, qtbot, CUSUM)
    form.set_params({"hp_enabled": True})
    qtbot.wait(20)
    sent = form.surrogate_params()
    assert sent["detector"] == CUSUM and sent["hp_enabled"] is True and sent["polarity"] is None
    assert null.isEnabled(), "fixture: the button is live; the handler is what refuses"

    null.click()                                # Auto polarity
    qtbot.wait(20)
    assert refused == [NULL_HIGHPASS_REFUSAL], refused
    assert panel._pending is None and panel._pending_sel is None
    assert dispatched == [], f"a refused Null control started a job: {dispatched}"
    assert sel not in panel._fdr

    for polarity in (-1, 1):                    # an override does not lift it
        form.set_params({"polarity": polarity})
        qtbot.wait(20)
        assert form.surrogate_params()["polarity"] == polarity
        null.click()
        qtbot.wait(20)
    assert refused == [NULL_HIGHPASS_REFUSAL] * 3, refused
    assert panel._pending is None and dispatched == [] and sel not in panel._fdr
    assert _readout(window) == (panel._stats._NO_FDR_TEXT, "")

    # ...whatever else is on the dict: it is the High-pass flag that refuses.
    panel._on_null_requested({**sent, "hp_enabled": True, "filter_cutoff_hz": 5000.0})
    assert refused == [NULL_HIGHPASS_REFUSAL] * 4 and dispatched == []

    # The Schmitt study with High-pass on is not refused: it runs as it always did.
    form.set_params({"polarity": None})
    _choose(window, qtbot, SCHMITT)
    assert form.surrogate_params()["hp_enabled"] is True
    null.click()
    qtbot.wait(20)
    assert dispatched == ["null_control"] and panel._pending == "null"
    panel._pending = panel._pending_sel = None

    # ...and CUSUM runs again as soon as High-pass is off.
    _choose(window, qtbot, CUSUM)
    form.set_params({"hp_enabled": False})
    qtbot.wait(20)
    null.click()
    qtbot.wait(20)
    assert dispatched == ["null_control", "null_control"] and panel._pending == "null"
    panel._pending = panel._pending_sel = None
    assert len(refused) == 4, refused[4:]


def test_null_control_refuses_a_detector_it_does_not_run(window, qtbot, tmp_path, monkeypatch):
    """Neither Schmitt nor CUSUM: refused, as the sweep and the recovery map refuse it.

    Such a dict used to fall into the Schmitt branch with its ``detector`` key still
    in the kwargs, and died on the worker with an unexpected-keyword TypeError.
    """
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    schmitt = panel.detection_panel.surrogate_params()
    assert "detector" not in schmitt
    refused: list[str] = []
    panel.runRefused.connect(refused.append)
    dispatched: list[str] = []
    monkeypatch.setattr(
        panel._controller, "run_callable",
        lambda fn, label="callable", **kw: dispatched.append(label) or True)

    for foreign in ("detect.keras", "detect.manual", "no.such.detector"):
        panel._on_null_requested({**schmitt, "detector": foreign})
        assert refused[-1] == NULL_DETECTOR_REFUSAL, refused
        assert panel._pending is None and panel._pending_sel is None
        assert dispatched == [], f"{foreign} was dispatched as a Schmitt study"
    assert len(refused) == 3 and sel not in panel._fdr

    # The two it does run are not refused, named or not.
    for params in (dict(schmitt), {**schmitt, "detector": SCHMITT}):
        panel._on_null_requested(params)
        assert panel._pending == "null"
        panel._pending = panel._pending_sel = None
    assert dispatched == ["null_control", "null_control"] and len(refused) == 3


def test_a_dict_that_names_schmitt_runs_the_schmitt_study(window, qtbot, tmp_path, monkeypatch):
    """``detector: detect.schmitt`` is not a `detect_events` argument and must not reach it.

    The button's own dict carries no ``detector`` key, and its call is the one it always
    was: the kwargs are spelled out here the way the handler spelled them before.
    """
    calls = _record_null_calls(monkeypatch)
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    schmitt = panel.detection_panel.surrogate_params()
    assert "detector" not in schmitt

    _click_null(window, qtbot)                  # the button's own dict
    button = calls[0]
    assert button["extra"] == {}
    assert button["detect_kwargs"] == {
        k: v for k, v in schmitt.items()
        if k != "polarity" and k not in DETREND_PARAM_KEYS
        and k not in FILTER_PARAM_KEYS and v is not None}
    assert isinstance(button["rng"], int)
    assert button["rng"] == panel._null_seed(sel, schmitt), "the Schmitt seed line moved"

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=120_000):
        panel._on_null_requested({**schmitt, "detector": SCHMITT})
    qtbot.wait(40)
    named = calls[1]
    assert named["extra"] == {}, f"the Schmitt call grew an argument: {named['extra']}"
    assert "detector" not in named["detect_kwargs"], (
        "the routing key reached `detect_events` as a keyword")
    assert named["detect_kwargs"] == button["detect_kwargs"]
    assert set(named["detect_kwargs"]) <= set(inspect.signature(detect_events).parameters)
    assert panel._fdr[sel]["detector"] == SCHMITT
    assert _readout(window)[0].startswith("Schmitt (k_in "), _readout(window)


def test_cusum_null_is_called_like_a_cusum_run(window, qtbot, tmp_path, monkeypatch):
    """What reaches `truth.null_control` with CUSUM selected is a CUSUM Detect's call."""
    calls = _record_null_calls(monkeypatch)
    _open(window, tmp_path)
    panel = window._analysis_panel
    form = panel.detection_panel

    _choose(window, qtbot, CUSUM)
    form.merge_gap_ms.setValue(0.0)             # "merge off" -- emitted as None
    form.cusum_k.setValue(0.9)
    form.cusum_h.setValue(31.0)
    # A Baseline panel value the user changed. At its defaults the live configuration
    # IS `DEFAULT_BASELINE`, and a study that ignored the panel would pass unseen.
    assert panel._live_baseline_config() == DEFAULT_BASELINE, "fixture: panel at defaults"
    panel._baseline.set_params({"baseline_base_s": 0.2})
    qtbot.wait(20)
    live = panel._live_baseline_config()
    assert live is not None and live != DEFAULT_BASELINE, (
        "fixture: the Baseline panel edit did not change the live configuration")
    assert form.polarity_choice() is None, "fixture: the form should be on Auto polarity"

    _click_null(window, qtbot)
    assert len(calls) == 1
    call = calls[0]
    kw = call["detect_kwargs"]

    assert call["extra"] == {"detector": detect_events_cusum}, (
        f"Null control did not hand `null_control` the CUSUM detector: {call['extra']}")
    assert kw["cusum_k"] == pytest.approx(0.9) and kw["cusum_h"] == pytest.approx(31.0)
    assert "merge_gap_s" in kw and kw["merge_gap_s"] is None, (
        "merge off must reach CUSUM as None; dropped, CUSUM merges at its own 0.5 ms "
        f"default: {kw}")
    assert kw["baseline_config"] == live == panel._live_baseline_config(), (
        "the study did not use the Baseline panel's configuration")
    assert kw["baseline_config"] != DEFAULT_BASELINE, (
        "the study ran on the default baseline, not on the Baseline panel's edited one")
    for key in ("k_in", "k_out", "noise_policy", "noise_window_s", "detector", "polarity"):
        assert key not in kw, f"{key} reached the CUSUM detector: {sorted(kw)}"
    assert set(kw) <= set(inspect.signature(detect_events_cusum).parameters), sorted(kw)
    assert call["rec"].polarity is None, (
        "on Auto the detector must measure the polarity itself, as a Detect run does")
    assert isinstance(call["rng"], int)

    # A polarity the user chose is passed through, not re-measured.
    form.set_params({"polarity": -1})
    qtbot.wait(20)
    _click_null(window, qtbot)
    assert calls[1]["rec"].polarity == -1

    # ...and the Schmitt call shape is the one it always was: no `detector=`.
    form.set_params({"polarity": None})
    _choose(window, qtbot, SCHMITT)
    _click_null(window, qtbot)
    schmitt = calls[2]
    assert schmitt["extra"] == {}, f"the Schmitt call grew an argument: {schmitt['extra']}"
    assert "k_in" in schmitt["detect_kwargs"] and "cusum_k" not in schmitt["detect_kwargs"]
    assert "baseline_config" not in schmitt["detect_kwargs"]
    assert schmitt["rec"].polarity in (-1, 1)


@pytest.mark.parametrize("lowpass_hz", [None, 5000.0], ids=["no_low_pass", "low_pass_5_kHz"])
def test_cusum_null_counts_what_cusum_detect_counts(window, qtbot, tmp_path, lowpass_hz):
    """Real maths, no stub: the study's real run is the CUSUM table's run.

    Same row count, same direction, same opposite-direction count -- with the low-pass
    off and with it on (the detector applies it, so the study has to hand it over).
    High-pass is the one conditioning stage this cannot hold for, and with it on the
    study is refused (`test_null_control_refuses_cusum_while_high_pass_is_on`).
    """
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    form = panel.detection_panel
    _choose(window, qtbot, CUSUM)
    if lowpass_hz is not None:
        form.set_params({"filter_cutoff_hz": lowpass_hz, "filter_kind": "bessel"})
        qtbot.wait(20)
    sent = form.params()
    assert sent["filter_cutoff_hz"] == (
        None if lowpass_hz is None else pytest.approx(lowpass_hz)), "fixture: the low-pass"
    assert sent["hp_enabled"] is False and sent["polarity"] is None, (
        "fixture: High-pass off, polarity on Auto")
    _detect(window, qtbot)
    table = window._session.results.get(*sel, CUSUM).payload
    rows = len(table.df)
    assert rows >= 20, f"fixture: CUSUM found {rows} of {N_PLANTED} planted events"
    assert table.provenance["filter_cutoff_hz"] == sent["filter_cutoff_hz"], (
        "fixture: the table was not detected with the form's low-pass")

    text, _tip = _click_null(window, qtbot)
    stored = panel._fdr[sel]
    assert stored["detector"] == CUSUM
    assert stored["n_real"] == rows, (
        f"Null control's real run found {stored['n_real']} events where CUSUM Detect, "
        f"with the same form, found {rows}")
    assert int(stored["polarity"]) == int(table.provenance["polarity"]), (
        f"Null control ran CUSUM in direction {stored['polarity']}; the CUSUM table "
        f"beside it was detected in direction {table.provenance['polarity']}")
    assert "n_opposite" in stored and int(stored["n_opposite"]) == int(
        table.provenance["n_opposite_isolated"])
    assert stored["thresholds"]["filter_cutoff_hz"] == sent["filter_cutoff_hz"]
    label = ("CUSUM (k 0.75, h 28, no low-pass)" if lowpass_hz is None
             else "CUSUM (k 0.75, h 28, low-pass 5 kHz)")
    assert text.startswith(_cusum_lead(
        label, stored["n_opposite"], rows, stored["n_event_free"])), text

    # The same press with Schmitt selected counts what the Schmitt detector counts.
    _choose(window, qtbot, SCHMITT)
    _detect(window, qtbot)
    schmitt_rows = len(window._session.results.get(*sel, SCHMITT).payload.df)
    text, _tip = _click_null(window, qtbot)
    stored = panel._fdr[sel]
    assert stored["detector"] == SCHMITT and stored["n_real"] == schmitt_rows
    assert "n_opposite" not in stored
    assert text.startswith("Schmitt (k_in "), text


# ================================================================================================
# the readout
# ================================================================================================

def test_the_readout_names_the_detector_and_thresholds(window, qtbot, tmp_path, monkeypatch):
    _record_null_calls(monkeypatch)
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    _choose(window, qtbot, CUSUM)
    panel.detection_panel.cusum_k.setValue(0.75)
    panel.detection_panel.cusum_h.setValue(28.0)
    qtbot.wait(20)

    text, tip = _click_null(window, qtbot)
    assert text.startswith("CUSUM (k 0.75, h 28, no low-pass): "), text
    assert "Estimated false-positive rate: 5.0% (AAFT, 5 surrogates)" in text, text
    stored = panel._fdr[sel]
    assert stored["detector"] == CUSUM
    assert stored["thresholds"] == {
        "cusum_k": 0.75, "cusum_h": 28.0, "filter_cutoff_hz": None}
    assert isinstance(stored["seed"], int) and f"seed = {stored['seed']}" in tip
    assert ("For comparison, 7 opposite-direction detection(s) on the recording itself, "
            "away from detected events.") in tip, tip

    # The Schmitt stamp is the two thresholds it always was.
    _choose(window, qtbot, SCHMITT)
    text, _tip = _click_null(window, qtbot)
    assert text.startswith("Schmitt (k_in "), text
    assert set(panel._fdr[sel]["thresholds"]) == {"k_in", "k_out"}


def test_the_cusum_readout_leads_with_the_opposite_count_and_names_the_low_pass(
        window, qtbot, tmp_path, monkeypatch):
    """Through the real button, with the low-pass on and with it off.

    The low-pass changes how often CUSUM fires on noise, so the label names it; and
    the count the reader is sent to is the opposite-direction one, so it comes first.
    The study itself is the stub (7 opposite, 200 on the recording, 3 event-free).
    """
    _record_null_calls(monkeypatch)
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    form = panel.detection_panel
    _choose(window, qtbot, CUSUM)
    form.cusum_k.setValue(0.75)
    form.cusum_h.setValue(28.0)
    rate = "Estimated false-positive rate: 5.0% (AAFT, 5 surrogates)"

    # Low-pass on at 5 kHz.
    form.set_params({"filter_cutoff_hz": 5000.0, "filter_kind": "bessel"})
    qtbot.wait(20)
    assert form.params()["filter_cutoff_hz"] == pytest.approx(5000.0)
    text, _tip = _click_null(window, qtbot)
    assert panel._fdr[sel]["thresholds"] == {
        "cusum_k": 0.75, "cusum_h": 28.0, "filter_cutoff_hz": 5000.0}
    lead = _cusum_lead("CUSUM (k 0.75, h 28, low-pass 5 kHz)", 7, 200, 3)
    assert lead == (
        "CUSUM (k 0.75, h 28, low-pass 5 kHz): 7 opposite-direction detection(s) against "
        "200 on the recording. 3 on its event-free noise, a weak lower bound for CUSUM.")
    assert text.startswith(lead + " " + rate), text
    assert "real events" not in text

    # Low-pass off.
    form.set_params({"filter_cutoff_hz": None})
    qtbot.wait(20)
    assert form.params()["filter_cutoff_hz"] is None
    text, _tip = _click_null(window, qtbot)
    assert panel._fdr[sel]["thresholds"]["filter_cutoff_hz"] is None
    assert text.startswith(
        _cusum_lead("CUSUM (k 0.75, h 28, no low-pass)", 7, 200, 3) + " " + rate), text


def test_the_readout_wording_by_branch(qtbot):
    """The renderings, on the panel alone: tagged and untagged, rate and refusal, and
    the third state (a result on file that is not shown with this table)."""
    stats = StatsPanel()
    qtbot.addWidget(stats)

    # Untagged (every result made before the tag existed): not one character changes.
    stats.set_fdr(dict(_RATE_SHAPE))
    assert stats.fdr_label.text().startswith("estimated false-positive rate: 5.0% (AAFT, 5 ")
    stats.set_fdr(dict(_REFUSAL_SHAPE))
    assert stats.fdr_label.text() == (
        "false-positive rate NOT estimated — 12 detection(s) on this recording's own "
        "event-free noise (a lower bound, measured with no surrogate).")
    assert stats.fdr_label.toolTip() == _REFUSAL_SHAPE["notes"][0]

    stats.set_fdr(_tagged(_RATE_SHAPE, SCHMITT))
    assert stats.fdr_label.text().startswith(
        "Schmitt (k_in 5, k_out 2): estimated false-positive rate: 5.0% (AAFT, 5 ")
    assert "opposite-direction" not in stats.fdr_label.toolTip()
    assert stats.fdr_label.toolTip().startswith(
        "The mean of 5 independent AAFT surrogates, over 200 real detections. "
        "Per-surrogate counts: 8, 12, 10, 9, 11.")

    stats.set_fdr(_tagged(_REFUSAL_SHAPE, SCHMITT))
    assert stats.fdr_label.text() == (
        "Schmitt (k_in 5, k_out 2): false-positive rate NOT estimated — 12 detection(s) "
        "on this recording's own event-free noise (a lower bound, measured with no "
        "surrogate).")

    # A CUSUM result with its opposite-direction count LEADS with it. A refusal is
    # then those two sentences and nothing else; the reason stays in the tooltip.
    stats.set_fdr(_tagged(_REFUSAL_SHAPE, CUSUM, n_opposite=41))
    assert stats.fdr_label.text() == (
        "CUSUM (k 0.75, h 28, no low-pass): 41 opposite-direction detection(s) against "
        "200 on the recording. 12 on its event-free noise, a weak lower bound for CUSUM.")
    assert stats.fdr_label.toolTip() == _REFUSAL_SHAPE["notes"][0]
    stats.set_fdr(_tagged(_REFUSAL_SHAPE, CUSUM, lowpass_hz=5000.0, n_opposite=41))
    assert stats.fdr_label.text() == (
        "CUSUM (k 0.75, h 28, low-pass 5 kHz): 41 opposite-direction detection(s) against "
        "200 on the recording. 12 on its event-free noise, a weak lower bound for CUSUM.")

    # With a rate, the existing rate sentence follows the lead.
    stats.set_fdr(_tagged(_RATE_SHAPE, CUSUM, lowpass_hz=5000.0, n_opposite=41))
    assert stats.fdr_label.text() == (
        "CUSUM (k 0.75, h 28, low-pass 5 kHz): 41 opposite-direction detection(s) against "
        "200 on the recording. 3 on its event-free noise, a weak lower bound for CUSUM. "
        "Estimated false-positive rate: 5.0% (AAFT, 5 surrogates) — individual "
        "surrogates gave 4.0–6.0%")
    assert "seed = 20261110" in stats.fdr_label.toolTip()
    # The tooltip of a CUSUM result says "detections on the recording", as its readout
    # does: at these settings many of them may be noise.
    assert stats.fdr_label.toolTip().startswith(
        "The mean of 5 independent AAFT surrogates, over 200 detections on the "
        "recording. Per-surrogate counts: 8, 12, 10, 9, 11.")
    assert "real detections" not in stats.fdr_label.toolTip()

    # A CUSUM result with no opposite-direction count has nothing to lead with.
    stats.set_fdr(_tagged(_RATE_SHAPE, CUSUM))
    assert stats.fdr_label.text().startswith(
        "CUSUM (k 0.75, h 28, no low-pass): estimated false-positive rate: 5.0% (AAFT, 5 ")

    # A count on a result with no tag changes nothing: only a CUSUM result leads.
    stats.set_fdr({**_REFUSAL_SHAPE, "n_opposite": 41})
    assert stats.fdr_label.text() == (
        "false-positive rate NOT estimated — 12 detection(s) on this recording's own "
        "event-free noise (a lower bound, measured with no surrogate).")

    # Thresholds the user typed are printed as typed, not rounded to the defaults; a
    # cutoff below 1 kHz reads in Hz; a stamp with no cutoff says nothing about it.
    odd = _tagged(_RATE_SHAPE, CUSUM)
    odd["thresholds"] = {"cusum_k": 1.25, "cusum_h": 40.5, "filter_cutoff_hz": 800.0}
    stats.set_fdr(odd)
    assert stats.fdr_label.text().startswith("CUSUM (k 1.25, h 40.5, low-pass 800 Hz): ")
    odd["thresholds"] = {"cusum_k": 1.25, "cusum_h": 40.5}
    stats.set_fdr(odd)
    assert stats.fdr_label.text().startswith("CUSUM (k 1.25, h 40.5): ")

    stats.set_fdr(None)
    assert stats.fdr_label.text() == stats._NO_FDR_TEXT and stats.fdr_label.toolTip() == ""
    assert stats._NO_FDR_TEXT == "false-positive rate: not measured (run Null control)"

    # THE THIRD STATE: measured, but for a detector whose table is not the one on screen.
    # It names the stored result's own label and the table that shows it, and it shows
    # no number, so it carries no tooltip.
    stats.set_fdr(_tagged(_RATE_SHAPE, CUSUM, n_opposite=41))       # a tooltip to clear
    assert stats.fdr_label.toolTip()
    stats.set_fdr(None, measured_for=_tagged(_RATE_SHAPE, CUSUM, lowpass_hz=5000.0,
                                             n_opposite=41))
    assert (stats.fdr_label.text(), stats.fdr_label.toolTip()) == (
        "false-positive rate: measured for CUSUM (k 0.75, h 28, low-pass 5 kHz); shown "
        "with a CUSUM table.", "")
    stats.set_fdr(None, measured_for=_tagged(_REFUSAL_SHAPE, SCHMITT))
    assert (stats.fdr_label.text(), stats.fdr_label.toolTip()) == (
        "false-positive rate: measured for Schmitt (k_in 5, k_out 2); shown with a "
        "Schmitt table.", "")
    assert _hidden(SCHMITT_LABEL, "Schmitt")[0] == stats.fdr_label.text()
    # A result with no tag was made by Schmitt and has no label of its own.
    stats.set_fdr(None, measured_for=dict(_RATE_SHAPE))
    assert stats.fdr_label.text() == (
        "false-positive rate: measured for Schmitt; shown with a Schmitt table.")
    # Nothing on file at all is still "not measured", and a result that IS shown wins.
    stats.set_fdr(None, measured_for=None)
    assert stats.fdr_label.text() == stats._NO_FDR_TEXT
    stats.set_fdr(_tagged(_RATE_SHAPE, SCHMITT), measured_for=_tagged(_RATE_SHAPE, CUSUM))
    assert stats.fdr_label.text().startswith("Schmitt (k_in 5, k_out 2): estimated ")


def test_a_result_is_shown_only_under_its_detectors_table(window, qtbot, tmp_path):
    """A combo switch with no CUSUM table, a Table-selector switch, and an ML table.

    Under another detector's table the result is not shown -- and the readout does not
    then say "not measured (run Null control)": it says what the result on file was
    measured for and which table shows it.
    """
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    unmeasured = (panel._stats._NO_FDR_TEXT, "")
    none = _hidden(SCHMITT_LABEL, "Schmitt")
    _detect(window, qtbot)
    assert _shown(window) == SCHMITT
    assert _readout(window) == unmeasured, "nothing is on file yet"
    _deliver(panel, "null", sel, _tagged(_RATE_SHAPE, SCHMITT))
    shown = _readout(window)
    assert shown[0].startswith("Schmitt (k_in 5, k_out 2): ") and shown[1]

    # 1. The combo moves to a detector that has no table here: nothing is on screen, and
    #    the form's detector decides.
    _choose(window, qtbot, CUSUM)
    assert _shown(window) is None
    assert _readout(window) == none, (
        "a Schmitt false-positive count is on screen with CUSUM selected: "
        f"{_readout(window)!r}")
    _choose(window, qtbot, SCHMITT)
    assert _readout(window) == shown, "the Schmitt result did not come back with its table"

    # 2. A CUSUM table arrives; the Table selector switches between the two.
    _choose(window, qtbot, CUSUM)
    _detect(window, qtbot)
    assert _shown(window) == CUSUM
    assert _readout(window) == none, (
        f"the Schmitt count is shown under the CUSUM table: {_readout(window)!r}")
    _pick_table(window, qtbot, "Schmitt")
    assert _readout(window) == shown
    _pick_table(window, qtbot, "CUSUM")
    assert _readout(window) == none

    # 3. An ML table (a real EventSet under the Keras module id) is neither detector's.
    _pick_table(window, qtbot, "Schmitt")
    events = window._session.results.get(*sel, SCHMITT).payload
    window._session.add_result(AnalysisResult(
        module_id="detect.keras", module_version="1.0", doc_id=sel[0], channel=sel[1],
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events))
    qtbot.wait(30)
    assert _shown(window) == "detect.keras", "fixture: the ML table should be on screen"
    assert _readout(window) == none, (
        f"a Schmitt count is shown under an ML table: {_readout(window)!r}")
    _pick_table(window, qtbot, "Schmitt")
    assert _readout(window) == shown

    # A result with no tag was made by Schmitt: shown under Schmitt, hidden under CUSUM.
    panel._fdr[sel] = dict(_RATE_SHAPE)
    _pick_table(window, qtbot, "CUSUM")
    assert _readout(window) == _hidden("Schmitt", "Schmitt")
    _pick_table(window, qtbot, "Schmitt")
    assert _readout(window)[0].startswith("estimated false-positive rate: 5.0%")

    # ...and a CUSUM result is the mirror image.
    panel._fdr[sel] = _tagged(_RATE_SHAPE, CUSUM)
    _pick_table(window, qtbot, "CUSUM")
    assert _readout(window)[0].startswith("CUSUM (k 0.75, h 28, no low-pass): ")
    _pick_table(window, qtbot, "Schmitt")
    assert _readout(window) == _hidden(CUSUM_LABEL, "CUSUM")
    _pick_table(window, qtbot, "Keras")
    assert _shown(window) == "detect.keras"
    assert _readout(window) == _hidden(CUSUM_LABEL, "CUSUM")

    # With nothing on file the readout is "not measured" under every table.
    del panel._fdr[sel]
    for name in ("Schmitt", "CUSUM", "Keras"):
        _pick_table(window, qtbot, name)
        assert _readout(window) == unmeasured, (name, _readout(window))


def test_the_form_decides_when_no_table_is_shown_whatever_was_shown_last(
        window, qtbot, tmp_path):
    """With no table on screen the form's detector decides -- on EVERY combo move.

    A combo move normally ends in a table switch, which re-judges the readout. It does
    not when the last table this panel showed was an ML one (the combo does not name
    that detector, so the panel leaves the table alone): the user then opens a recording
    with no table, moves the combo, and nothing switches. The readout must follow anyway.
    """
    panel = window._analysis_panel
    none = _hidden(SCHMITT_LABEL, "Schmitt")

    # Recording A ends with an ML table on screen.
    a = _open(window, tmp_path, "pore_a.dat")
    _detect(window, qtbot)
    events = window._session.results.get(*a, SCHMITT).payload
    window._session.add_result(AnalysisResult(
        module_id="detect.keras", module_version="1.0", doc_id=a[0], channel=a[1],
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events))
    qtbot.wait(30)
    assert _shown(window) == "detect.keras", "fixture: the ML table should be on screen"

    # Recording B has no table at all; a Schmitt Null control is on file for it.
    b = _open(window, tmp_path, "pore_b.dat", _trace(seed=4))
    qtbot.wait(30)
    assert _shown(window) is None and panel.detection_panel.detector_choice() == SCHMITT
    _deliver(panel, "null", b, _tagged(_RATE_SHAPE, SCHMITT))
    shown = _readout(window)
    assert shown[0].startswith("Schmitt (k_in 5, k_out 2): ")

    _choose(window, qtbot, CUSUM)
    assert _shown(window) is None
    assert _readout(window) == none, (
        f"the Schmitt count is still on screen with CUSUM selected: {_readout(window)!r}")
    _choose(window, qtbot, SCHMITT)
    assert _readout(window) == shown


def test_a_study_that_lands_after_a_detector_switch_is_filed_not_shown(
        window, qtbot, tmp_path):
    """The study takes seconds and the user may move the combo while it runs."""
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    said: list[str] = []
    panel.statusMessage.connect(said.append)

    panel._pending, panel._pending_sel = "null", sel      # launched with Schmitt selected
    _choose(window, qtbot, CUSUM)                         # ...the user switches meanwhile
    said.clear()
    panel._on_controller_finished(_tagged(_RATE_SHAPE, SCHMITT))
    qtbot.wait(10)

    assert panel._fdr[sel]["detector"] == SCHMITT, "the answer was not kept"
    assert _readout(window) == _hidden(SCHMITT_LABEL, "Schmitt"), (
        f"a Schmitt answer was shown with CUSUM selected: {_readout(window)!r}")
    # It says where the answer is shown. It does not say "show the Schmitt table":
    # this recording has none.
    assert said == ["Null control finished for Schmitt; it is shown with a Schmitt table."], (
        f"nothing says where the answer went: {said}")

    _choose(window, qtbot, SCHMITT)
    assert _readout(window)[0].startswith("Schmitt (k_in 5, k_out 2): ")

    # An answer that lands on the detector it was run with says nothing extra.
    said.clear()
    _deliver(panel, "null", sel, _tagged(_RATE_SHAPE, SCHMITT))
    assert not [m for m in said if m.startswith("Null control finished")], said

    # A table that is neither detector's (an ML one) hides the answer the same way.
    _detect(window, qtbot)
    events = window._session.results.get(*sel, SCHMITT).payload
    window._session.add_result(AnalysisResult(
        module_id="detect.keras", module_version="1.0", doc_id=sel[0], channel=sel[1],
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events))
    qtbot.wait(30)
    assert _shown(window) == "detect.keras", "fixture: the ML table should be on screen"
    said.clear()
    _deliver(panel, "null", sel, _tagged(_RATE_SHAPE, CUSUM, n_opposite=7))
    assert _readout(window) == _hidden(CUSUM_LABEL, "CUSUM")
    assert said == ["Null control finished for CUSUM; it is shown with a CUSUM table."], said


def test_null_control_under_a_table_of_neither_detector_says_what_it_measured(
        window, qtbot, tmp_path, monkeypatch):
    """A Keras table on screen, Schmitt on the form: press *Null control…*, twice.

    The button is live and runs the form's detector; the result is filed and is not
    shown with the Keras table. The readout used to keep saying "not measured (run
    Null control)" after each press -- the instruction that had just been followed,
    and that could never fill it.
    """
    _record_null_calls(monkeypatch)
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    unmeasured = (panel._stats._NO_FDR_TEXT, "")
    _detect(window, qtbot)
    events = window._session.results.get(*sel, SCHMITT).payload
    window._session.add_result(AnalysisResult(
        module_id="detect.keras", module_version="1.0", doc_id=sel[0], channel=sel[1],
        params={}, calibration_fingerprint=(1.0, 0.0), payload=events))
    qtbot.wait(30)
    assert _shown(window) == "detect.keras", "fixture: the ML table should be on screen"
    assert panel.detection_panel.detector_choice() == SCHMITT
    assert panel.detection_panel.null_button.isEnabled()
    assert _readout(window) == unmeasured, "nothing has been measured yet"

    for press in (1, 2):
        shown = _click_null(window, qtbot)
        assert panel._fdr[sel]["detector"] == SCHMITT
        assert "run Null control" not in shown[0], (
            f"press {press}: the readout still asks for the click that was just made: "
            f"{shown[0]!r}")
        assert shown == _hidden(SCHMITT_LABEL, "Schmitt"), shown

    # The same press with CUSUM on the form: the readout follows the result on file.
    _choose(window, qtbot, CUSUM)
    assert _shown(window) == "detect.keras", "fixture: the ML table stays on screen"
    assert _click_null(window, qtbot) == _hidden(CUSUM_LABEL, "CUSUM")

    # ...and the result itself is shown as soon as a table of its detector is.
    _detect(window, qtbot)
    assert _shown(window) == CUSUM
    assert _readout(window)[0].startswith(CUSUM_LABEL + ": "), _readout(window)


def test_undo_and_rerun_keep_the_tie(window, qtbot, tmp_path):
    """Undo, Redo and a re-run all put another table object on screen without the user
    touching the Table selector, and the readout is re-judged each time.

    Edit > Undo acts on the table on screen, so it restores an earlier run of the SAME
    detector: a result of that detector stays, and a result of the other detector stays
    hidden. (Before the tie existed the Schmitt count below stood under the CUSUM table
    through all of this.)
    """
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    form = panel.detection_panel
    none = _hidden(SCHMITT_LABEL, "Schmitt")       # the Schmitt result, on file and hidden

    def rows() -> int:
        return len(panel._table.dataframe)

    # Two Schmitt runs, so there is a re-run to undo; the second finds nothing.
    _detect(window, qtbot)
    first = rows()
    assert first >= 20
    form.k_in.setValue(30.0)
    qtbot.wait(30)
    _detect(window, qtbot)
    assert _shown(window) == SCHMITT and rows() < first
    _deliver(panel, "null", sel, _tagged(_RATE_SHAPE, SCHMITT))
    shown = _readout(window)
    assert shown[0].startswith("Schmitt (k_in 5, k_out 2): ")

    window._act_undo.trigger()                  # Edit > Undo "Re-run detection"
    qtbot.wait(100)
    assert _shown(window) == SCHMITT and rows() == first, "fixture: Undo did not restore"
    assert _readout(window) == shown, "Undo dropped the result of the table it restored"
    window._act_redo.trigger()
    qtbot.wait(100)
    assert rows() < first
    assert _readout(window) == shown

    # Two CUSUM runs. The Schmitt result is not CUSUM's, before or after the undo.
    _choose(window, qtbot, CUSUM)
    _detect(window, qtbot)
    cusum_first = rows()
    assert _shown(window) == CUSUM and cusum_first >= 20
    assert _readout(window) == none
    form.cusum_k.setValue(50.0)
    qtbot.wait(30)
    _detect(window, qtbot)
    assert _shown(window) == CUSUM and rows() < cusum_first
    assert _readout(window) == none

    window._act_undo.trigger()
    qtbot.wait(100)
    assert _shown(window) == CUSUM and rows() == cusum_first, "fixture: Undo did not restore"
    assert _readout(window) == none, (
        f"Undo put the Schmitt count under the CUSUM table: {_readout(window)!r}")

    # The undo re-judges, it does not merely leave the label alone: a CUSUM result filed
    # while nothing was looking is on screen as soon as Redo shows the CUSUM table again.
    panel._fdr[sel] = _tagged(_RATE_SHAPE, CUSUM)
    window._act_redo.trigger()
    qtbot.wait(100)
    assert _shown(window) == CUSUM
    cusum_shown = _readout(window)
    assert cusum_shown[0].startswith("CUSUM (k 0.75, h 28, no low-pass): "), cusum_shown

    _detect(window, qtbot)                      # a re-run of the same detector keeps it
    assert _shown(window) == CUSUM
    assert _readout(window) == cusum_shown, "a re-run of the same detector dropped the result"

    _pick_table(window, qtbot, "Schmitt")       # ...and it is still not Schmitt's
    assert _readout(window) == _hidden(CUSUM_LABEL, "CUSUM")


# ================================================================================================
# the recovery map
# ================================================================================================

def test_a_schmitt_recovery_map_never_reaches_a_cusum_fit(window, qtbot, tmp_path):
    """The map is the Schmitt detector's efficiency: offered and applied for a Schmitt
    table only."""
    sel = _two_tables(window, qtbot, tmp_path)
    panel = window._analysis_panel
    stats = panel._stats
    box = stats.use_efficiency
    opts = {"n_components_dwell": 1, "n_components_depth": 1,
            "use_efficiency": True, "use_filter_limited": False}

    _deliver(panel, "recovery", sel, _eta_frame())
    assert _shown(window) == CUSUM
    assert not box.isEnabled(), (
        "the Schmitt recovery map is on offer for a fit of the CUSUM table")
    assert box.toolTip() == EFFICIENCY_TIP + "\n\n" + OTHER_TABLE_REASON

    # Even asked for directly, the map does not reach a fit of CUSUM rows.
    panel._on_fit_requested(dict(opts))
    _wait_for_fit(stats, qtbot)
    qtbot.wait(30)
    on_cusum = stats.results.toPlainText()
    assert "n = " in on_cusum, f"fixture: the CUSUM fit did not run: {on_cusum!r}"
    assert "measured eta(t)" not in on_cusum, (
        "a dwell fit of the CUSUM table was weighted by the Schmitt detector's "
        f"efficiency: {on_cusum!r}")

    # The Schmitt table still gets it, through the box the user ticks.
    _pick_table(window, qtbot, "Schmitt")
    assert box.isEnabled()
    assert box.toolTip() == EFFICIENCY_TIP
    box.setChecked(True)
    stats.fit_button.click()
    _wait_for_fit(stats, qtbot)
    qtbot.wait(30)
    assert "measured eta(t)" in stats.results.toPlainText(), stats.results.toPlainText()

    # Back on CUSUM the box is off again. The tick is the user's and stays.
    _pick_table(window, qtbot, "CUSUM")
    assert not box.isEnabled() and box.isChecked()
    assert box.toolTip() == EFFICIENCY_TIP + "\n\n" + OTHER_TABLE_REASON


def test_a_recovery_map_that_lands_under_another_table_says_so(window, qtbot, tmp_path):
    """The map runs for a long time and the user may be on another table when it lands.

    It is stored and the efficiency box stays grey, which is right -- and used to be
    all: the only trace of the click was the tooltip of a disabled box on another tab.
    """
    sel = _two_tables(window, qtbot, tmp_path)
    panel = window._analysis_panel
    box = panel._stats.use_efficiency
    said: list[str] = []
    panel.statusMessage.connect(said.append)
    refused: list[str] = []
    panel.runRefused.connect(refused.append)

    assert _shown(window) == CUSUM              # launched under Schmitt, lands under CUSUM
    said.clear()
    _deliver(panel, "recovery", sel, _eta_frame())
    assert sel in panel._recovery, "the map was not kept"
    assert not box.isEnabled()
    assert said == [RECOVERY_LANDED], f"nothing says where the map went: {said}"
    assert refused == []

    # Under the Schmitt table the box lights up, and that is the message.
    _pick_table(window, qtbot, "Schmitt")
    assert box.isEnabled()
    said.clear()
    _deliver(panel, "recovery", sel, _eta_frame())
    assert box.isEnabled()
    assert not [m for m in said if m.startswith("Recovery map finished")], said

    # No table, CUSUM on the form: the next fit would not be a Schmitt table's either.
    other = _open(window, tmp_path, "pore_b.dat", _trace(seed=4))
    qtbot.wait(30)
    _choose(window, qtbot, CUSUM)
    assert _shown(window) is None
    said.clear()
    _deliver(panel, "recovery", other, _eta_frame())
    assert said == [RECOVERY_LANDED] and not box.isEnabled()

    # A map for a recording that is no longer selected is filed and nothing is said.
    said.clear()
    _deliver(panel, "recovery", sel, _eta_frame(scale=5e-4))
    assert not [m for m in said if m.startswith("Recovery map finished")], said
    assert panel._recovery[sel]["efficiency"].equals(_eta_frame(scale=5e-4)["efficiency"])


def test_the_efficiency_tick_survives_schmitt_cusum_schmitt(window, qtbot, tmp_path):
    """A table of another detector greys the box out; it does not untick it.

    The user ticks "use measured efficiency" for the Schmitt table, looks at the CUSUM
    table and comes back: the tick is still there. While the CUSUM table is on screen
    the greyed-out tick weights nothing, by the Fit button or by a direct request.
    """
    sel = _open(window, tmp_path)
    panel = window._analysis_panel
    stats = panel._stats
    box = stats.use_efficiency
    other_table = EFFICIENCY_TIP + "\n\n" + OTHER_TABLE_REASON

    _detect(window, qtbot)
    assert _shown(window) == SCHMITT
    _deliver(panel, "recovery", sel, _eta_frame())
    assert box.isEnabled() and not box.isChecked()
    box.setChecked(True)

    # 1. The combo alone (no CUSUM table yet): off with the reason, still ticked.
    _choose(window, qtbot, CUSUM)
    assert _shown(window) is None
    assert not box.isEnabled() and box.isChecked(), "the combo switch unticked the box"
    assert box.toolTip() == other_table
    _choose(window, qtbot, SCHMITT)
    assert box.isEnabled() and box.isChecked() and box.toolTip() == EFFICIENCY_TIP

    # 2. A CUSUM table on screen: the same, and no fit of it is weighted.
    _choose(window, qtbot, CUSUM)
    _detect(window, qtbot)
    assert _shown(window) == CUSUM
    assert len(window._session.results.get(*sel, CUSUM).payload.df) >= 20, "fixture"
    assert not box.isEnabled() and box.isChecked(), "the CUSUM table unticked the box"
    assert box.toolTip() == other_table

    sent: list[dict] = []
    stats.fitRequested.connect(sent.append)
    stats.fit_button.click()                    # the user's route
    _wait_for_fit(stats, qtbot)
    qtbot.wait(30)
    assert sent and sent[-1]["use_efficiency"] is False, (
        f"a greyed-out tick asked for the efficiency correction: {sent}")
    by_button = stats.results.toPlainText()
    assert "n = " in by_button, f"fixture: the CUSUM fit did not run: {by_button!r}"
    assert "measured eta(t)" not in by_button, by_button

    panel._on_fit_requested({"n_components_dwell": 1, "n_components_depth": 1,
                             "use_efficiency": True, "use_filter_limited": False})
    _wait_for_fit(stats, qtbot)
    qtbot.wait(30)
    direct = stats.results.toPlainText()
    assert "n = " in direct and "measured eta(t)" not in direct, (
        f"the Schmitt map weighted a fit of the CUSUM table: {direct!r}")

    # 3. Back on the Schmitt table the tick is still the user's, and it works.
    _pick_table(window, qtbot, "Schmitt")
    assert box.isEnabled() and box.isChecked(), (
        "Schmitt -> CUSUM -> Schmitt lost the user's tick")
    assert box.toolTip() == EFFICIENCY_TIP
    stats.fit_button.click()
    _wait_for_fit(stats, qtbot)
    qtbot.wait(30)
    assert sent[-1]["use_efficiency"] is True
    assert "measured eta(t)" in stats.results.toPlainText(), stats.results.toPlainText()


def test_only_a_missing_map_unticks_the_efficiency_box(qtbot):
    """On the panel alone: no map unticks, as before; a map for another table does not."""
    stats = StatsPanel()
    qtbot.addWidget(stats)
    box = stats.use_efficiency

    stats.set_recovery_available(True)
    box.setChecked(True)
    stats.set_recovery_available(False, OTHER_TABLE_REASON)
    assert not box.isEnabled() and box.isChecked()
    assert box.toolTip() == EFFICIENCY_TIP + "\n\n" + OTHER_TABLE_REASON
    stats.set_recovery_available(True)
    assert box.isEnabled() and box.isChecked() and box.toolTip() == EFFICIENCY_TIP

    stats.set_recovery_available(False)         # the map itself is gone
    assert not box.isEnabled() and not box.isChecked()
    assert box.toolTip() == EFFICIENCY_TIP


# ================================================================================================
# persistence
# ================================================================================================

def _assert_project_holds_no_study_result(proj: Path) -> None:
    """Nothing of a Null control result or of a recovery map is in the saved archive.

    The result dict's own keys are looked for among every key of ``manifest.json``, at
    any depth; the map is a table with an ``efficiency`` column, looked for in every
    parquet member.
    """
    def keys_of(node) -> set[str]:
        if isinstance(node, dict):
            return set(node).union(*(keys_of(v) for v in node.values()))
        if isinstance(node, list):
            return set().union(*(keys_of(v) for v in node))
        return set()

    with zipfile.ZipFile(proj) as zf:
        names = zf.namelist()
        assert "manifest.json" in names, names
        keys = keys_of(json.loads(zf.read("manifest.json").decode("utf-8")))
        assert "results" in keys, "fixture: this is not the project manifest"
        found = keys & {"estimated_fdr", "per_surrogate", "n_event_free",
                        "n_surrogate_mean", "n_surrogate_std", "n_opposite"}
        assert not found, f"the project file carries a study result: {sorted(found)}"
        tables = [name for name in names if name.endswith(".parquet")]
        assert tables, f"fixture: the project holds no event table: {names}"
        for name in tables:
            columns = list(pd.read_parquet(io.BytesIO(zf.read(name))).columns)
            assert "efficiency" not in columns, f"{name} carries a recovery map"


def test_study_results_are_not_saved_with_the_project(qtbot, tmp_path):
    """Both stores are per session. A reopened project reads "not measured"."""
    csv = tmp_path / "pore_a.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = MainWindow()
    qtbot.addWidget(win)
    sel = _install(win, _doc(csv, arr))
    panel = win._analysis_panel
    _detect(win, qtbot)
    _deliver(panel, "null", sel, _tagged(_RATE_SHAPE, SCHMITT))
    _deliver(panel, "recovery", sel, _eta_frame())
    assert _readout(win)[0].startswith("Schmitt (k_in 5, k_out 2): ")
    assert panel._stats.use_efficiency.isEnabled()
    panel._stats.use_efficiency.setChecked(True)

    proj = tmp_path / "studies.nanoscope"
    assert win._write_project(str(proj))
    _assert_project_holds_no_study_result(proj)
    win._analysis_controller.abort()
    win._controller.abort()

    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    try:
        live = w2._session.doc_id_for_source(str(csv.resolve()))
        assert live is not None and _shown(w2) == SCHMITT, "fixture: the table did not reopen"
        p2 = w2._analysis_panel
        assert _readout(w2) == (p2._stats._NO_FDR_TEXT, ""), _readout(w2)
        assert not p2._fdr and not p2._recovery
        assert not p2._stats.use_efficiency.isEnabled()
    finally:
        w2._analysis_controller.abort()
        w2._controller.abort()


def test_study_results_do_not_survive_a_reopen_in_the_same_window(qtbot, tmp_path):
    """The window that MEASURED them reopens its own project: the stores it filled are
    the ones that could outlive the recording they describe.

    A second window starts with empty stores whatever the first one did. Here the same
    panel holds a CUSUM result, a recovery map and the user's tick when the project is
    saved and opened again. The reopened recording is a new load: nothing measured on
    the old one may be on screen, on file, or on offer.

    The TICK is not a study result. It is one of the project's saved fit controls and
    comes back with them, into a box that is greyed out because there is no map. A
    greyed-out tick asks for nothing, and the next repoint with no map clears it, as it
    always did.
    """
    csv = tmp_path / "pore_a.csv"
    arr = _trace()
    _write_csv(csv, arr)
    win = MainWindow()
    qtbot.addWidget(win)
    try:
        sel = _install(win, _doc(csv, arr))
        panel = win._analysis_panel
        box = panel._stats.use_efficiency
        _detect(win, qtbot)
        _choose(win, qtbot, CUSUM)
        _detect(win, qtbot)
        assert _shown(win) == CUSUM
        _deliver(panel, "null", sel, _tagged(_RATE_SHAPE, CUSUM, n_opposite=3))
        _deliver(panel, "recovery", sel, _eta_frame())
        assert _readout(win)[0].startswith(CUSUM_LABEL + ": ")
        _pick_table(win, qtbot, "Schmitt")
        assert box.isEnabled()
        box.setChecked(True)
        assert _readout(win) == _hidden(CUSUM_LABEL, "CUSUM")
        _pick_table(win, qtbot, "CUSUM")
        assert not box.isEnabled() and box.isChecked()
        assert sel in panel._fdr and sel in panel._recovery

        proj = tmp_path / "studies.nanoscope"
        assert win._write_project(str(proj))
        _assert_project_holds_no_study_result(proj)
        with qtbot.waitSignal(win.projectRestored, timeout=30_000):
            win._open_project(str(proj))
        qtbot.wait(80)

        live = win._session.doc_id_for_source(str(csv.resolve()))
        assert live is not None, "fixture: the recording did not reopen"
        assert live != sel[0], "fixture: a reopened recording is a new load"
        assert _shown(win) in (SCHMITT, CUSUM), "fixture: no table reopened"
        assert not panel._fdr and not panel._recovery, (
            f"study results outlived the load they were measured on: "
            f"{list(panel._fdr)} {list(panel._recovery)}")
        assert _readout(win) == (panel._stats._NO_FDR_TEXT, ""), _readout(win)
        assert not box.isEnabled(), "the map of the closed load is still on offer"
        assert box.toolTip() == EFFICIENCY_TIP

        # Whatever the greyed-out box reads, the Fit button asks for no correction.
        sent: list[dict] = []
        panel._stats.fitRequested.connect(sent.append)
        panel._stats.fit_button.click()
        _wait_for_fit(panel._stats, qtbot)
        qtbot.wait(30)
        assert sent and sent[-1]["use_efficiency"] is False, sent
        fitted = panel._stats.results.toPlainText()
        assert "n = " in fitted and "measured eta(t)" not in fitted, fitted

        # Under either table of the reopened recording, with nothing on file.
        for name in ("Schmitt", "CUSUM", "Schmitt"):
            _pick_table(win, qtbot, name)
            assert _readout(win) == (panel._stats._NO_FDR_TEXT, ""), (name, _readout(win))
            assert not box.isEnabled()
        assert not box.isChecked(), "a repoint with no map left the box ticked"

        # A map measured on the reopened recording enables the box, unticked.
        _deliver(panel, "recovery", (live, 0), _eta_frame())
        assert _shown(win) == SCHMITT and box.isEnabled() and not box.isChecked()
    finally:
        win._analysis_controller.abort()
        win._controller.abort()


# ================================================================================================
# the advisory
# ================================================================================================

def test_the_advisory_is_on_the_summary_tab(window, qtbot, tmp_path):
    """A CUSUM run on low-passed 1/f noise fires in both directions, and says so where
    the user reads the table's caveats."""
    fs = 50_000.0
    rec, _truth = synth_trace(fs=fs, duration_s=8.0, n_events=0, pink_frac=0.3, seed=0)
    arr = np.ascontiguousarray(rec.data, dtype=np.float32)
    path = tmp_path / "noise_only.dat"
    path.write_bytes(b"synthetic noise " * 64)
    sel = _install(window, _doc(path, arr, fs=fs))
    panel = window._analysis_panel
    form = panel.detection_panel

    _choose(window, qtbot, CUSUM)
    form.set_params({"filter_cutoff_hz": 5000.0, "filter_kind": "bessel"})
    qtbot.wait(20)
    assert form.params()["filter_cutoff_hz"] == pytest.approx(5000.0)
    _detect(window, qtbot)

    res = window._session.results.get(*sel, CUSUM)
    prov = res.payload.provenance
    assert int(prov["n_opposite_isolated"]) >= 5, (
        f"fixture: too few opposite-direction detections to raise the advisory: "
        f"{prov['n_opposite_isolated']}")
    said = panel._stats.warnings_text()
    assert "opposite-direction" in said, (
        f"the run counted {prov['n_opposite_isolated']} opposite-direction detections "
        f"and the Summary tab does not say so: {said!r}")
    # ...on the Summary tab, where `warnings_label` lives, once that tab is raised.
    tabs = panel.results_tabs
    summary = next(i for i in range(tabs.count())
                   if tabs.widget(i).isAncestorOf(panel._stats.warnings_label))
    assert tabs.tabText(summary) == "Summary", tabs.tabText(summary)
    tabs.setCurrentIndex(summary)
    assert "opposite-direction" in _visible_text(tabs)

    # A Schmitt run of the same trace counts no opposite direction and says nothing.
    _choose(window, qtbot, SCHMITT)
    _detect(window, qtbot)
    assert "opposite-direction" not in panel._stats.warnings_text()
