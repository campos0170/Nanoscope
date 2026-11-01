"""The polarity is judged on its own: the owner's answer to design 13.4, question 2.

THE GAP, measured on 3938d32. The polarity was judged only inside the baseline-drift stage, the
one chain stage that reads it. ``detect_events`` reads it too, whatever the chain holds, so a
polarity edited with that stage off left the table on screen looking current: the chain notice
read "match", the Details list said nothing, and Revert had nothing to put back. A run that
never recorded its polarity, with the drift stage on, was called changed against "auto" and
offered no Revert.

THE RULE NOW (``conditioning_recipes.polarity_differences``):

* a polarity that differs from the one the table's run recorded is a detection setting, as a
  threshold is: "⚠ Settings changed: polarity auto → blockade", listed on the Details list's
  Thresholds row with Revert, and put back by Revert as one step that Undo takes back;
* while both the run and the form run the drift stage, the stage names it as before
  ("baseline drift polarity auto → blockade"), and it is not named twice;
* a run that never recorded its polarity is not called changed, drift stage on or off.

Synthetic recordings only, each under ``tmp_path``; the detectors run synchronously through
their modules, so every table carries a real run's record. No sidecar is written: the gain
that raises the amber line is set in the session only.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import conditioning_recipes as cr
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.lineage import LiveInputs, judge_inputs, made_with
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt, register_all
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1_000.0
N = 60_000
S, C = "detect.schmitt", "detect.cusum"
_SPANS = [(5_000, 5_400), (15_000, 15_500), (25_000, 25_600), (40_000, 40_500)]
#: The chain the runs are made with: a low-pass, the drift stage off unless a test turns it on.
RUN_CHAIN = {"filter_cutoff_hz": 10_000.0, "detrend_enabled": False}
#: The polarity combo's rows (``DetectionPanel.polarity_override``).
AUTO, BLOCKADE = 0, 1
SETTINGS = "Revert form to this detection's settings"


# ================================================================================================
# builders
# ================================================================================================

def _signal() -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
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


def _meta(path: Path) -> RecordingMeta:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    return RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, t0_s=0.0, channels=(info,))


def _open(w: MainWindow, path: Path) -> str:
    doc = TraceDocument(meta=_meta(path), channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    return doc_id


def _run(w: MainWindow, doc_id: str, module=None, **params):
    """Run ``module`` (Schmitt by default) with the form's params plus ``params``,
    synchronously; returned, not yet stored."""
    session = w._session
    form = {**w._analysis_panel._form_params(), **params}
    res = (module or DetectSchmitt()).run(SimpleAnalysisContext(
        session.calibrated(doc_id, 0), session.document(doc_id).meta, form))
    assert len(res.payload.df), "the synthetic events were found"
    return res


def _detect(w: MainWindow, doc_id: str, *, drift: bool = False) -> None:
    """A Schmitt table made with ``RUN_CHAIN`` (the drift stage as asked) and auto polarity."""
    det = w._analysis_panel.detection_panel
    det.set_params({**RUN_CHAIN, "detrend_enabled": drift})
    det.polarity_override.setCurrentIndex(AUTO)
    w._session.add_result(_run(w, doc_id))
    QtWidgets.QApplication.processEvents()
    assert _settle(w).kind == "match", "precondition: the table starts current"


def _show_table(w: MainWindow, module_id: str) -> None:
    selector = w._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData(module_id))
    QtWidgets.QApplication.processEvents()
    assert w._analysis_panel.current_detection().module_id == module_id


def _settle(w: MainWindow) -> cr.ChainStatus:
    QtWidgets.QApplication.processEvents()
    status = w._analysis_panel.refresh_chain_status()
    if w._signal_conditioning is not None:
        w._signal_conditioning.flush()
    QtWidgets.QApplication.processEvents()
    return status


def _blockade(w: MainWindow) -> cr.ChainStatus:
    """The user's edit: the polarity combo to "blockade (down, -1)", committed as a step."""
    w._analysis_panel.detection_panel.polarity_override.setCurrentIndex(BLOCKADE)
    w._on_edit_menu_about_to_show()          # one of the moments that commit an edit
    return _settle(w)


def _polarity(w: MainWindow):
    return w._analysis_panel.detection_panel.polarity_choice()


def _details(w: MainWindow) -> str:
    """Raise the amber line (a gain set in the session only, no sidecar) and read Details."""
    doc_id, channel = w._analysis_panel._sel
    w._session.set_calibration(doc_id, channel, ChannelCalibration(
        gain=2.0, offset=0.0, target_units="pA"), write_sidecar=False)
    QtWidgets.QApplication.processEvents()
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


# ================================================================================================
# the rule
# ================================================================================================

def _form(**over):
    return {**cr.complete_recipe(RUN_CHAIN), "k_in": 5.0, "k_out": 3.0, "min_dwell_s": 2e-5,
            "max_dwell_s": None, "merge_gap_s": None, "polarity": None, **over}


def test_the_polarity_is_judged_on_its_own_with_the_drift_stage_off():
    run = _form()
    form = _form(polarity=-1)
    status = cr.judge_chain(form, run, module_id=S, detector_label="Schmitt")
    assert status.kind == "differs", status.sentence
    assert status.thresholds_differ
    assert status.headline == "⚠ Settings changed: polarity auto → blockade"
    assert status.sentence == ("⚠ Settings changed since the Schmitt detection on screen ran: "
                               "polarity auto → blockade. Revert, or run detection to apply "
                               "them.")
    assert status.revert_params == {"polarity": None}
    reverted = dict(form, **status.revert_params)
    assert cr.judge_chain(reverted, run, module_id=S).kind == "match"
    # A CUSUM run reads it too.
    assert cr.judge_chain(form, run, module_id=C).thresholds_differ

    # The Details list's Thresholds row says the same, with Revert as its action.
    verdict = judge_inputs(_made(run), LiveInputs(rate_hz=FS, chain=form, thresholds=form))[4]
    assert (verdict.state, verdict.action) == ("changed", "revert")
    assert (verdict.was, verdict.now) == ("polarity auto", "polarity blockade")
    assert verdict.sentence == ("Thresholds: polarity auto → blockade. Revert, or re-run to "
                                "apply them.")


def test_with_the_drift_stage_on_the_stage_names_it_once():
    on = _form(detrend_enabled=True)
    status = cr.judge_chain(dict(on, polarity=-1), on, module_id=S, detector_label="Schmitt")
    assert [d.text for d in status.diffs] == ["baseline drift polarity auto → blockade"]
    assert not status.thresholds_differ, "the drift stage's own wording, as before"
    assert status.revert_params["polarity"] is None

    # Turned off with the edit, the stage no longer reads it; the detector still does.
    status = cr.judge_chain(dict(on, detrend_enabled=False, polarity=-1), on, module_id=S)
    assert [d.text for d in status.diffs] == ["baseline drift on → off",
                                              "polarity auto → blockade"]
    assert status.thresholds_differ
    assert status.revert_params["polarity"] is None
    assert status.revert_params["detrend_enabled"] is True


def test_a_run_that_never_recorded_its_polarity_is_not_called_changed():
    """As a threshold the run never recorded: not second-guessed with auto. With the drift stage
    on, 3938d32 called it changed against auto and withheld every Revert."""
    for drift in (True, False):
        legacy = {k: v for k, v in _form(detrend_enabled=drift).items() if k != "polarity"}
        form = _form(detrend_enabled=drift, polarity=-1)
        status = cr.judge_chain(form, legacy, module_id=S)
        assert status.kind == "match", (drift, status.sentence)
        assert cr.polarity_differences(form, legacy) == ()
        assert cr.revert_params(form, legacy) == {}
        verdicts = judge_inputs(_made(legacy),
                                LiveInputs(rate_hz=FS, chain=form, thresholds=form))
        assert verdicts[3].state == verdicts[4].state == "same", (drift, verdicts)


def _made(params: dict):
    """What a stored Schmitt table that ran with ``params`` recorded (``made_with`` reads
    only these fields)."""
    return made_with(SimpleNamespace(module_id=S, params=params, payload=None,
                                     calibration_fingerprint=(1.0, 0.0),
                                     baseline_fingerprint=None))


# ================================================================================================
# through the app
# ================================================================================================

def test_a_polarity_edit_with_the_drift_stage_off_is_named_reverted_and_undone(
        window, tmp_path, menu_answer):
    doc_id = _open(window, tmp_path / "pore.dat")
    _detect(window, doc_id)
    ap = window._analysis_panel
    window._open_signal_conditioning()

    status = _blockade(window)
    assert status.kind == "differs", status.sentence
    assert status.headline == "⚠ Settings changed: polarity auto → blockade"
    assert ap._table.notice_text() == "⚠ Settings changed"
    assert ap._table._notice_actions["revert"].isEnabled()
    assert window._signal_conditioning.revert_button.isEnabled()
    assert ap.stages_that_differ() == frozenset(), "no stage of the chain differs"

    steps, said = _spy(ap.formStepNamed), _spy(ap.statusMessage)
    menu_answer(SETTINGS)
    ap._table.notice_chip.click()
    _settle(window)
    assert _polarity(window) is None, "Revert put back the run's auto"
    assert ap.chain_status().kind == "match"
    assert ap._table.notice_text() == ""
    assert steps == ["Revert settings to Schmitt's run"]
    assert said[-1] == "Form reverted to the settings the Schmitt detection on screen ran with."

    assert window._act_undo.text() == "Undo Revert settings to Schmitt's run"
    window._act_undo.trigger()
    status = _settle(window)
    assert _polarity(window) == -1, "Undo of the Revert brings the edit back"
    assert status.headline == "⚠ Settings changed: polarity auto → blockade"
    window._act_redo.trigger()
    assert _settle(window).kind == "match" and _polarity(window) is None


def test_details_names_the_polarity_with_its_revert(window, tmp_path):
    """The Thresholds row lists it; the estimator, which no polarity moves, stays unchanged."""
    doc_id = _open(window, tmp_path / "pore.dat")
    _detect(window, doc_id)
    _blockade(window)

    text = _details(window)

    assert _row(text, "Thresholds:") == (
        "• Thresholds: polarity auto → polarity blockade — Revert, or re-run to apply"), text
    assert _row(text, "Signal chain:") == "• Signal chain: unchanged", text
    assert _row(text, "Open-pore estimator:") == "• Open-pore estimator: unchanged", text


def test_with_the_drift_stage_on_a_polarity_edit_still_reverts(window, tmp_path, menu_answer):
    doc_id = _open(window, tmp_path / "pore.dat")
    _detect(window, doc_id, drift=True)
    ap = window._analysis_panel
    det = ap.detection_panel

    status = _blockade(window)
    assert [d.text for d in status.diffs] == ["baseline drift polarity auto → blockade"]
    assert ap._table.notice_text() == "⚠ Chain changed"
    assert ap.stages_that_differ() == {"detrend"}
    assert ap.revert_chain_to_shown_detection()
    assert _polarity(window) is None and _settle(window).kind == "match"
    window._act_undo.trigger()
    assert _polarity(window) == -1 and _settle(window).kind == "differs"

    # The drift stage unticked with the edit: the stage no longer reads it, the detector does.
    det.set_params({"detrend_enabled": False})
    window._on_edit_menu_about_to_show()
    status = _settle(window)
    assert [d.text for d in status.diffs] == ["baseline drift on → off",
                                              "polarity auto → blockade"], status.diffs
    assert ap._table.notice_text() == "⚠ Settings changed"
    menu_answer(SETTINGS)
    ap._table.notice_chip.click()
    _settle(window)
    assert _polarity(window) is None and det.params()["detrend_enabled"] is True
    assert ap.chain_status().kind == "match"
    window._act_undo.trigger()
    assert _polarity(window) == -1 and det.params()["detrend_enabled"] is False


def test_a_legacy_table_without_a_recorded_polarity_is_not_called_changed(window, tmp_path):
    """A table whose params do not record the polarity (drift off on the Schmitt one, on on the
    CUSUM one): no edit of the polarity marks it, and Revert has nothing to put back."""
    doc_id = _open(window, tmp_path / "pore.dat")
    ap = window._analysis_panel
    det = ap.detection_panel

    def legacy(res):
        return replace(res, params={k: v for k, v in res.params.items() if k != "polarity"})

    det.set_params(RUN_CHAIN)
    window._session.add_result(legacy(_run(window, doc_id)))
    window._session.add_result(legacy(_run(window, doc_id, DetectCusum(), detector=C,
                                           detrend_enabled=True)))
    for module_id in (S, C):
        _show_table(window, module_id)
        run = ap.current_detection().params
        assert "polarity" not in run and run["detrend_enabled"] is (module_id == C)
        det.polarity_override.setCurrentIndex(AUTO)
        assert _settle(window).kind == "match", module_id
        status = _blockade(window)
        assert status.kind == "match", (module_id, status.sentence)
        assert ap._table.notice_text() == "", module_id
        assert ap.revert_chain_to_shown_detection() is False
        assert _polarity(window) == -1, "nothing was put back"
    assert _row(_details(window), "Thresholds:") == "• Thresholds: unchanged"


def test_a_project_restore_keeps_the_polarity_verdict(window, qtbot, tmp_path, menu_answer):
    """The table's run and the edited form go through a real save and reopen; the reopened
    table is still marked, and its Revert still puts the polarity back."""
    csv = tmp_path / "pore.csv"
    arr = _signal()
    pd.DataFrame({"time": np.arange(N) / FS, "current": arr.astype(np.float64)}).to_csv(
        csv, index=False)
    doc_id = _open(window, csv)
    _detect(window, doc_id)
    assert _blockade(window).thresholds_differ
    proj = tmp_path / "polarity.nanoscope"
    assert window._write_project(str(proj))

    w2 = MainWindow()
    qtbot.addWidget(w2)
    try:
        with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
            w2._open_project(str(proj))
        qtbot.wait(50)
        ap = w2._analysis_panel
        assert ap.current_detection().module_id == S
        assert _polarity(w2) == -1, "the edited form came back"
        status = _settle(w2)
        assert status.headline == "⚠ Settings changed: polarity auto → blockade", status.sentence
        assert ap._table.notice_text() == "⚠ Settings changed"
        menu_answer(SETTINGS)
        ap._table.notice_chip.click()
        assert _polarity(w2) is None and _settle(w2).kind == "match"
    finally:
        w2._analysis_controller.abort()
