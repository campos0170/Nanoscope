"""Sweep 10, the Details list and the Revert controls (S10-STAGE1, S10-STAGE2, S10-LIN1,
S10-LIN2), each measured on 7e4e0f6:

* **STAGE1.** A Schmitt run with a 10 kHz low-pass, then a 16 kHz declaration: the form's
  cutoff box clamps to 7999.9 Hz. Every Revert stayed enabled and each press said "Form
  reverted…" with the box still at 7999.9 and the notice at "differs", press after press:
  ``revert_params`` judged a plain dict holding 10 kHz, not what the clamped box can hold
  (design 13.2 says it returns None when Revert cannot put the form back). Reset stage looped
  the same way on a recording slower than 200 Hz, whose Nyquist the 100 Hz high-pass default
  is above.
* **STAGE2.** With every Revert greyed (a run that never recorded the notch Q, or STAGE1's
  clamp), the amber line's Details still ended the Signal chain row "— Revert, or re-run to
  apply".
* **LIN1.** A rate-only declaration listed the open-pore estimator as changed ("decim 66 →
  decim 33", the decimation is derived from the rate) with "Re-measure depths", which the
  panel refuses on a table on another clock (``stale_rate``).
* **LIN2.** The Fits box was not redrawn on a calibration change or its Undo, so its advice
  ("re-run detection" / "press Fit") contradicted the amber line both ways.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path`` (the one calibration
Apply here writes its sidecar beside it).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import conditioning_recipes as cr
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.lineage import LiveInputs, judge_inputs, made_with
from nanoscope.analysis.modules import DetectSchmitt, register_all
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1_000.0
S = "detect.schmitt"
#: The chain the stage tests' run is made with: a low-pass and a notch, every other stage off.
RUN_CHAIN = {"filter_cutoff_hz": 10_000.0, "notch_enabled": True, "notch_freq_hz": 1_000.0,
             "notch_harmonics": 3, "notch_q": 30.0}
#: Short recording for the stage tests (4 blockades); long one for the fits (24, the fits
#: refuse fewer than 20) on a slow drift, so an estimator window changes the depths.
N_SHORT, N_LONG = 60_000, 200_000
SHORT_SPANS = [(5_000, 5_400), (15_000, 15_500), (25_000, 25_600), (40_000, 40_500)]
LONG_SPANS = [(5_000 + 7_500 * i, 5_000 + 7_500 * i + 200 + 40 * i) for i in range(24)]
NYQUIST_8K = "the run's low-pass 10 kHz, which is above this recording's Nyquist (8 kHz)"


# ================================================================================================
# builders
# ================================================================================================

def _signal(long: bool) -> np.ndarray:
    n = N_LONG if long else N_SHORT
    rng = np.random.default_rng(3)
    t = np.arange(n) / FS
    x = I0 + (20.0 * np.sin(2.0 * np.pi * t / 3.0) if long else 0.0) + rng.normal(0.0, 8.0, n)
    for i, (s, e) in enumerate(LONG_SPANS if long else SHORT_SPANS):
        x[s:e] -= (250.0 + 10.0 * i) if long else 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    register_all()
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    yield w
    w._analysis_controller.abort()


def _open(w: MainWindow, tmp_path: Path, *, long: bool = False, rate: float = FS,
          name: str = "pore.dat") -> str:
    data = _signal(long)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=data.size,
                         sample_rate_hz=rate, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(data,))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    return doc_id


def _run(w: MainWindow, doc_id: str):
    """A Schmitt run with the form's params, synchronously; returned, not yet stored."""
    session = w._session
    res = DetectSchmitt().run(SimpleAnalysisContext(
        session.calibrated(doc_id, 0), session.document(doc_id).meta,
        w._analysis_panel._form_params()))
    assert len(res.payload.df), "the synthetic events were found"
    return res


def _detect(w: MainWindow, doc_id: str) -> None:
    w._session.add_result(_run(w, doc_id))
    QtWidgets.QApplication.processEvents()


def _settle(w: MainWindow) -> cr.ChainStatus:
    QtWidgets.QApplication.processEvents()
    status = w._analysis_panel.refresh_chain_status()
    if w._signal_conditioning is not None:
        w._signal_conditioning.flush()
    QtWidgets.QApplication.processEvents()
    return status


def _declare(w: MainWindow, doc_id: str, hz: float) -> None:
    """Inspector ▸ File Info ▸ Sample rate / time unit…"""
    w._session.set_sample_rate_declaration(
        doc_id, SampleRateDeclaration("sample_rate", sample_rate_hz=hz))
    QtWidgets.QApplication.processEvents()


def _details(w: MainWindow) -> str:
    """Open the amber line's Details, as a click does, and read the list of inputs in it."""
    banner = w._analysis_panel._stale
    assert not banner.isHidden(), "precondition: the amber line is up"
    banner._details.click()
    QtWidgets.QApplication.processEvents()
    return banner.inputs_text()


def _row(text: str, title: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"• {title}"))


def _spy(signal) -> list:
    seen: list = []
    signal.connect(seen.append)
    return seen


@pytest.fixture
def clamped(window, tmp_path):
    """A Schmitt run with a 10 kHz low-pass on a 50 kHz recording, the Signal Conditioning
    window open, then a 16 kHz declaration, which clamps the form's cutoff to 7999.9 Hz."""
    doc_id = _open(window, tmp_path)
    ap = window._analysis_panel
    ap.detection_panel.set_params(RUN_CHAIN)
    _detect(window, doc_id)
    window._open_signal_conditioning()
    assert _settle(window).kind == "match"
    _declare(window, doc_id, 16_000.0)
    status = _settle(window)
    assert ap.detection_panel.params()["filter_cutoff_hz"] == pytest.approx(7999.9)
    assert status.kind == "differs", "precondition: the clamp is a difference"
    return window, doc_id


# ================================================================================================
# S10-STAGE1 -- Revert judged on what the clamped form can hold
# ================================================================================================

def test_revert_params_judges_what_the_form_can_hold_at_the_rate():
    """The pure rule: a run's frequency at or above this Nyquist lands as the box's clamp, so
    Revert cannot put it back and says why. Where the clamp changes nothing that differs
    afterwards (the stage is turned off by the same Revert), Revert is still offered."""
    run = cr.complete_recipe(RUN_CHAIN)
    form = dict(run, filter_cutoff_hz=7999.9)
    assert cr.revert_params(form, run, fs=FS) == {"filter_cutoff_hz": 10_000.0,
                                                  "filter_order": 8, "filter_kind": "bessel"}
    assert cr.revert_params(form, run, fs=16_000.0) is None
    assert cr.revert_params(form, run, stages="lowpass", fs=16_000.0) is None

    assert cr.revert_withheld(form, run, fs=FS) == ""
    assert cr.revert_withheld(form, run, fs=16_000.0) == NYQUIST_8K
    status = cr.judge_chain(form, run, module_id=S, detector_label="Schmitt", fs=16_000.0)
    assert status.kind == "differs" and status.revert_params is None
    assert status.sentence.endswith(f"Run detection to apply it — Revert cannot restore "
                                    f"{NYQUIST_8K}.")
    assert status.withheld == NYQUIST_8K
    assert status.stage_withheld == {"lowpass": NYQUIST_8K}

    # The run's 100 Hz high-pass is above a 150 Hz recording's Nyquist, but the Revert turns
    # the stage off, so the clamp leaves nothing differing: still offered.
    run = cr.complete_recipe({"hp_enabled": False, "hp_cutoff_hz": 100.0})
    form = dict(run, hp_enabled=True, hp_cutoff_hz=74.99)
    assert cr.revert_params(form, run, fs=150.0) is not None
    assert cr.revert_withheld(form, run, fs=150.0) == ""

    # A run that never recorded what differs keeps its own reason.
    run = {k: v for k, v in cr.complete_recipe(RUN_CHAIN).items() if k != "notch_q"}
    form = dict(cr.complete_recipe(RUN_CHAIN), notch_q=20.0)
    assert cr.revert_params(form, run, fs=FS) is None
    assert cr.revert_withheld(form, run, fs=FS) == cr.UNRECORDED


def test_every_revert_is_withheld_with_its_reason_after_a_rate_clamp(clamped):
    """S10-STAGE1 through the widgets: the window's Revert, the chip menu's and the Low-pass
    Revert stage are greyed, each says the recorded cutoff is above the declared rate's
    Nyquist, and a press anyway writes nothing and names no step. On 7e4e0f6 each stayed
    enabled and said "Form reverted…" while nothing moved."""
    window, _doc_id = clamped
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    status = ap.chain_status()
    assert status.revert_params is None
    assert NYQUIST_8K in status.sentence and "Run detection to apply it" in status.sentence

    assert not ws.revert_button.isEnabled()
    assert ws.revert_button.toolTip() == (f"Revert cannot restore {NYQUIST_8K}. Run detection "
                                          f"to apply the form.")
    assert not ap._table._notice_actions["revert"].isEnabled()
    assert ws.stage_states() == ({"lowpass"}, frozenset())
    assert ap.can_revert_stage("lowpass") is False
    lowpass_row = ws.stage_list.item(cr.STAGE_KEYS.index("lowpass"))
    assert f"Revert stage cannot restore {NYQUIST_8K}." in lowpass_row.toolTip()
    det.select_stage("lowpass")
    ws.flush()
    assert not ws.stage_revert_button.isEnabled()
    assert ws.stage_revert_button.toolTip().startswith(f"Revert cannot restore {NYQUIST_8K}")

    steps, said = _spy(ap.formStepNamed), _spy(ap.statusMessage)
    assert ap.revert_chain_to_shown_detection() is False
    assert said == [status.sentence]
    assert ap.revert_stage("lowpass") is False
    assert said[-1] == (f"Revert cannot restore {NYQUIST_8K}, so the Low-pass stage cannot be "
                        f"put back. Run detection to apply it.")
    assert steps == [] and det.params()["filter_cutoff_hz"] == pytest.approx(7999.9)


def test_reset_stage_is_judged_on_what_the_box_can_hold(window, tmp_path):
    """The Reset lead: on a 150 Hz recording the 100 Hz high-pass default reads as the box's
    ceiling (74.99 Hz) once set. The first Reset is a step; the second says the stage
    already holds its defaults. On 7e4e0f6 every press reset again, naming a step each time."""
    _open(window, tmp_path, rate=150.0)
    ap = window._analysis_panel
    det = ap.detection_panel
    det.set_params({"hp_cutoff_hz": 20.0, "hp_order": 2})
    _settle(window)
    steps, said = _spy(ap.formStepNamed), _spy(ap.statusMessage)
    assert ap.reset_stage("highpass") is True
    assert det.params()["hp_cutoff_hz"] == pytest.approx(74.99)
    assert det.params()["hp_order"] == 4
    assert ap.reset_stage("highpass") is False
    assert said[-1] == "The High-pass stage already holds its default settings."
    assert steps == ["Reset High-pass stage"]


# ================================================================================================
# S10-STAGE2 -- Details follows the Revert controls' verdict
# ================================================================================================

def test_details_offers_no_revert_that_every_control_withholds(window, tmp_path):
    """S10-STAGE2: a run that did not record its notch Q, the Q edited, the table made stale by
    a 40 kHz declaration (nothing clamps). Every Revert is greyed, and Details' Signal chain
    row says only "Re-run to apply". On 7e4e0f6 it said "— Revert, or re-run to apply"."""
    doc_id = _open(window, tmp_path)
    ap = window._analysis_panel
    det = ap.detection_panel
    det.set_params(RUN_CHAIN)
    res = _run(window, doc_id)
    window._session.add_result(replace(res, params={
        k: v for k, v in res.params.items() if k != "notch_q"}))
    QtWidgets.QApplication.processEvents()
    assert _settle(window).kind == "match"
    det.set_params({"notch_q": 20.0})
    _declare(window, doc_id, 40_000.0)
    status = _settle(window)
    assert status.kind == "differs" and status.revert_params is None
    assert not ap._table._notice_actions["revert"].isEnabled()

    chain = _row(_details(window), "Signal chain:")
    assert chain.endswith("Q 20 — Re-run to apply"), chain
    assert "Revert" not in chain


def test_details_after_a_rate_clamp_offers_only_the_re_run(clamped):
    """STAGE1's table in Details: the low-pass row follows the same verdict."""
    window, _doc_id = clamped
    chain = _row(_details(window), "Signal chain:")
    assert chain.endswith("7.9999 kHz — Re-run to apply"), chain


def test_the_lineage_verdict_follows_revert():
    """The pure list: a chain or threshold row whose Revert is withheld is a re-run, and says
    why; one Revert puts both back, so both rows carry its verdict."""
    run = dict(cr.complete_recipe(RUN_CHAIN), k_in=5.0)
    run.pop("notch_q")

    class _Table:
        module_id, params, payload = S, run, None
        calibration_fingerprint, baseline_fingerprint = (1.0, 0.0), None

    form = dict(cr.complete_recipe(RUN_CHAIN), notch_q=20.0, k_in=6.0)
    verdicts = {v.name: v for v in judge_inputs(made_with(_Table()), LiveInputs(
        rate_hz=FS, chain=form, thresholds=form))}
    for name, it in (("chain", "it"), ("thresholds", "them")):
        v = verdicts[name]
        assert (v.state, v.action) == ("changed", "re-run"), v
        assert v.sentence.endswith(f"Re-run to apply {it} — Revert cannot restore "
                                   f"{cr.UNRECORDED}."), v.sentence

    form = dict(cr.complete_recipe(RUN_CHAIN), filter_cutoff_hz=5_000.0)
    run = dict(cr.complete_recipe(RUN_CHAIN), k_in=5.0)
    _Table.params = run
    verdict = judge_inputs(made_with(_Table()), LiveInputs(rate_hz=FS, chain=form,
                                                           thresholds=form))[3]
    assert verdict.action == "revert" and verdict.sentence.endswith("Revert, or re-run to apply it.")


# ================================================================================================
# S10-LIN1 -- a rate-only declaration is not an estimator change
# ================================================================================================

def test_a_rate_only_declaration_leaves_the_estimator_unchanged_in_details(window, tmp_path):
    """S10-LIN1: the decimation is derived from the rate, so a rate declared since the run moved
    the estimator's stamp with no setting touched. Details now lists the estimator as
    unchanged; on 7e4e0f6 it read "decim 66 → decim 33 — Re-measure depths", which the panel
    refuses on that table."""
    doc_id = _open(window, tmp_path, long=True)
    _detect(window, doc_id)
    panel = window._analysis_panel
    assert panel._stale.isHidden(), "precondition: the table starts current"
    _declare(window, doc_id, 25_000.0)
    assert panel._stale.kind == "sample_rate"
    text = _details(window)
    assert _row(text, "Sample rate:").endswith("— Re-run"), text
    assert _row(text, "Open-pore estimator:") == "• Open-pore estimator: unchanged", text
    assert panel._baseline_staleness(panel.current_detection()) is None


def test_an_estimator_change_on_another_clock_offers_the_re_run(window, tmp_path):
    """A real estimator change with the rate moved too is listed, at the table's own rate (no
    derived decimation moved by the rate), and its action is the Re-run that repairs both:
    Re-measure refuses a table on another clock."""
    doc_id = _open(window, tmp_path, long=True)
    _detect(window, doc_id)
    panel = window._analysis_panel
    panel.baseline_panel.base_s.setValue(0.2)
    QtWidgets.QApplication.processEvents()
    assert panel._stale.kind == "baseline", "precondition: a real estimator change"
    _declare(window, doc_id, 25_000.0)
    assert panel._stale.kind == "sample_rate"
    row = _row(_details(window), "Open-pore estimator:")
    assert "base 100 ms" in row and "base 200 ms" in row, row
    assert "decim 66" in row and "decim 33" not in row, row
    assert row.endswith("— Re-run"), row
    assert panel.remeasure_active_detection() == "stale_rate"


# ================================================================================================
# S10-LIN2 -- the Fits box is redrawn with the calibration
# ================================================================================================

FITS_RERUN = "the calibration has changed since the table was detected — re-run detection"
FITS_PRESS = "press Fit to fit this one"


def _fits_text(w: MainWindow) -> str:
    return w._analysis_panel._stats.results.toPlainText()


def _press_fit(w: MainWindow) -> None:
    w._analysis_panel._stats.fit_button.click()
    QtWidgets.QApplication.processEvents()


def test_the_fits_box_follows_a_calibration_undo(window, qtbot, tmp_path):
    """Route A: a Fit pressed while the table is calibration-stale, then Edit ▸ Undo of the
    calibration. The table is current again and the box says "press Fit"; on 7e4e0f6 it kept
    "re-run detection" until something else redrew it."""
    doc_id = _open(window, tmp_path, long=True)
    _detect(window, doc_id)
    panel = window._analysis_panel
    cal = window._calibration_panel
    cal._confirm_sidecar_removal = lambda *a, **k: True
    cal._confirm_note_loss = lambda *a, **k: True
    cal._gain.setText("2")
    cal._apply()
    QtWidgets.QApplication.processEvents()
    assert panel._stale.kind == "calibration"
    _press_fit(window)
    assert FITS_RERUN in _fits_text(window), "precondition: the fit is withheld"

    assert window._act_undo.text() == "Undo Apply calibration"
    window._act_undo.trigger()
    QtWidgets.QApplication.processEvents()
    assert window._session.calibration(doc_id, 0).gain == 1.0
    assert panel._stale.isHidden(), "the table is current again"
    text = _fits_text(window)
    assert FITS_PRESS in text and "re-run detection" not in text, text


def _set_gain(w: MainWindow, doc_id: str, gain: float) -> None:
    w._session.set_calibration(doc_id, 0, ChannelCalibration(gain=gain, offset=0.0,
                                                             target_units="pA"),
                               write_sidecar=False)
    QtWidgets.QApplication.processEvents()


def test_the_fits_box_follows_a_calibration_change_the_other_way(window, qtbot, tmp_path):
    """Route C: Detect, Fit, gain 2, Detect again (the panel's own run), gain 1. The table
    (gain 2) is now the outdated one and the amber line says so; the box says "re-run
    detection". On 7e4e0f6 it kept "press Fit", drawn while the table was current."""
    doc_id = _open(window, tmp_path, long=True)
    _detect(window, doc_id)
    panel = window._analysis_panel
    _press_fit(window)
    _set_gain(window, doc_id, 2.0)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(60)
    assert FITS_PRESS in _fits_text(window), "precondition: the fit predates the gain-2 table"
    _set_gain(window, doc_id, 1.0)
    assert panel._stale.kind == "calibration"
    text = _fits_text(window)
    assert FITS_RERUN in text and FITS_PRESS not in text, text
