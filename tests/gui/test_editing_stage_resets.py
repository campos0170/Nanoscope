"""One stage at a time: Revert stage, Reset stage, and a Revert that puts back only what
differs. Design §1.2 defect 7, §6 rows 1-2, §7.2 thresholds, decision 8.

Before this, every reset of the chain was coarse:

* **Revert to detection's chain rewrote all 37 chain keys and the polarity**, and filled
  every key the run never recorded with its default (defect 7). A setting typed into a stage
  that was off, which the judge rightly ignores, was lost to a Revert of the low-pass.
* **A stage had no Revert and no Reset of its own.** Unticking it kept its values; putting
  one stage back meant re-typing it or reverting every other stage with it.
* **The thresholds were never judged** (decision 8): a k_in edited after a run left the
  table on screen looking current, and the chip said "⚠ Chain changed" or nothing.
* **A legacy CUSUM table was never judged on its k and h.** The chain notice read a v1/v2
  table's settings through a Schmitt-only projection of its provenance, which drops
  ``cusum_k`` and ``cusum_h``; ``analysis.lineage`` kept them, so the notice and the
  Details list could disagree on the same table.

What is pinned here, through a real MainWindow and its Signal Conditioning window:

* Revert stage puts back that stage alone and Reset stage restores its defaults with its
  on/off kept; the stage list marks every stage that differs, and the buttons are enabled
  only where they can act;
* the chain notice reads "⚠ Settings changed" once a threshold differs, its slot fits that
  longer text, and its Revert (chip menu or window) puts back only what differs;
* a run that never recorded the setting that differs leaves every Revert disabled, and a
  Revert asked for anyway posts the judge's sentence and writes nothing;
* every action names its step through ``AnalysisPanel.formStepNamed`` (the form history
  turns each into one undo step);
* a legacy CUSUM table (params empty, provenance only) is judged, seeded and replayed on
  its own k and h.

Synthetic recordings only, each with ``meta.path`` under ``tmp_path``; the two detectors
run synchronously through their modules, so every table carries a real run's record.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import conditioning_recipes as cr
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.lineage import made_with
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt, register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import analysis_panel as AP
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1_000.0
N = 60_000
S, C = "detect.schmitt", "detect.cusum"
_SPANS = [(5_000, 5_400), (15_000, 15_500), (25_000, 25_600), (40_000, 40_500)]
#: The chain both runs are made with: a low-pass and a notch, every other stage off.
RUN_CHAIN = {"filter_cutoff_hz": 10_000.0, "notch_enabled": True, "notch_freq_hz": 1_000.0,
             "notch_harmonics": 3, "notch_q": 30.0}
#: The CUSUM run's own h, off the form's default (28), so a seed that forgets it shows.
RUN_CUSUM_H = 20.0
AMBER = "#ffb74d"


def _signal() -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(_signal(),))


@pytest.fixture
def window(qapp, qtbot):
    register_all()
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    yield w
    w._analysis_controller.abort()


def _open(w: MainWindow, tmp_path, name: str = "pore.dat") -> str:
    doc = _doc(tmp_path, name)
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    return doc_id


def _run(w: MainWindow, doc_id: str, module, **params):
    """Run ``module`` on the recording with the form's params (plus ``params``), synchronously,
    and return its result, not yet stored."""
    session = w._session
    form = {**w._analysis_panel._form_params(), **params}
    res = module.run(SimpleAnalysisContext(session.calibrated(doc_id, 0),
                                           session.document(doc_id).meta, form))
    assert len(res.payload.df), "the synthetic events were found"
    return res


def _show_table(w: MainWindow, module_id: str) -> None:
    """What the user does: pick the table in the Results dock's Table selector."""
    selector = w._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData(module_id))
    QtWidgets.QApplication.processEvents()
    assert w._analysis_panel.current_detection().module_id == module_id


@pytest.fixture
def runs(window, tmp_path):
    """A recording with a Schmitt and a CUSUM table made with ``RUN_CHAIN``, the Schmitt
    table on screen and the Signal Conditioning window open."""
    doc_id = _open(window, tmp_path)
    ap = window._analysis_panel
    ap.detection_panel.set_params(RUN_CHAIN)
    window._session.add_result(_run(window, doc_id, DetectSchmitt()))
    window._session.add_result(_run(window, doc_id, DetectCusum(), detector=C,
                                    cusum_h=RUN_CUSUM_H))
    _show_table(window, C)
    _show_table(window, S)
    window._open_signal_conditioning()
    _settle(window)
    assert ap.chain_status().kind == "match", ap.chain_status().sentence
    return window, doc_id


def _settle(w: MainWindow):
    """Judge now and render what the window was pushed: the judge runs on a zero-delay
    timer after each edit, and the window renders its rows once per burst."""
    QtWidgets.QApplication.processEvents()
    status = w._analysis_panel.refresh_chain_status()
    if w._signal_conditioning is not None:
        w._signal_conditioning.flush()
    return status


def _spy(signal) -> list:
    got: list = []
    signal.connect(got.append)
    return got


def _marked(ws) -> set[str]:
    """The stages whose rows are painted marked: bold, in the chain notice's amber."""
    out = set()
    for i, key in enumerate(cr.STAGE_KEYS):
        item = ws.stage_list.item(i)
        bold = item.font().bold()
        amber = item.foreground().color().name().lower() == AMBER
        assert bold == amber, (key, "the mark is bold AND amber, never one alone")
        if bold:
            out.add(key)
    return out


# ================================================================================================
# Revert stage and Reset stage
# ================================================================================================

def test_revert_stage_puts_back_that_stage_only(runs):
    """Two stages differ; the notch's Revert puts the notch back and leaves the low-pass,
    and a setting typed into a stage that is off, exactly as the user left them."""
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    det.set_params({"notch_q": 20.0, "filter_cutoff_hz": 5_000.0, "median_window_s": 5e-4})
    status = _settle(window)
    assert status.kind == "differs"
    assert ws.stage_states() == ({"notch", "lowpass"}, {"notch", "lowpass"})
    steps, said = _spy(ap.formStepNamed), _spy(ap.statusMessage)

    det.select_stage("notch")
    assert ws.stage_revert_button.isEnabled()
    ws.stage_revert_button.click()
    _settle(window)

    p = det.params()
    assert p["notch_q"] == 30.0, "the notch is the run's again"
    assert p["filter_cutoff_hz"] == 5_000.0, "the low-pass was left as edited"
    assert p["median_window_s"] == pytest.approx(5e-4), "an off stage was left as typed"
    assert ap.stages_that_differ() == {"lowpass"}
    assert steps == ["Revert Notch stage"]
    assert said[-1] == ("Notch stage reverted to the settings the Schmitt detection on "
                        "screen ran with.")
    assert not ws.stage_revert_button.isEnabled(), "the notch no longer differs"


def test_reset_stage_restores_the_defaults_and_keeps_the_on_off(runs):
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    det.set_params({"hp_enabled": True, "hp_cutoff_hz": 250.0, "hp_order": 2,
                    "hp_kind": "bessel", "filter_kind": "butter", "filter_order": 4,
                    "filter_cutoff_hz": 5_000.0, "median_window_s": 5e-4})
    _settle(window)
    steps, said = _spy(ap.formStepNamed), _spy(ap.statusMessage)

    det.select_stage("highpass")
    assert ws.stage_reset_button.isEnabled()
    ws.stage_reset_button.click()
    p = det.params()
    assert (p["hp_cutoff_hz"], p["hp_order"], p["hp_kind"]) == (100.0, 4, "butter")
    assert p["hp_enabled"] is True, "a running stage keeps running"
    assert said[-1] == "High-pass stage reset to its default settings; it is still on."

    # The low-pass's on/off IS its cutoff: a Reset restores its design and keeps the cutoff.
    det.select_stage("lowpass")
    ws.stage_reset_button.click()
    p = det.params()
    assert (p["filter_kind"], p["filter_order"]) == ("bessel", 8)
    assert p["filter_cutoff_hz"] == 5_000.0
    assert said[-1] == "Low-pass stage reset to its default settings; its cutoff is unchanged."

    # A stage that is off is reset too, and stays off.
    det.select_stage("median")
    ws.stage_reset_button.click()
    p = det.params()
    assert p["median_window_s"] == pytest.approx(cr.CHAIN_DEFAULTS["median_window_s"])
    assert p["median_enabled"] is False
    assert steps == ["Reset High-pass stage", "Reset Low-pass stage", "Reset Median stage"]

    # Nothing left to reset: nothing written, no step named, and the line says why.
    assert ap.reset_stage("median") is False
    assert steps == ["Reset High-pass stage", "Reset Low-pass stage", "Reset Median stage"]
    assert said[-1] == "The Median stage already holds its default settings."


def test_the_stage_list_marks_follow_the_differences_and_keep_the_rows(runs):
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    row_h = ws.stage_list.sizeHintForRow(0)
    assert _marked(ws) == set()

    det.set_params({"notch_q": 20.0, "filter_cutoff_hz": 5_000.0})
    _settle(window)
    assert _marked(ws) == {"notch", "lowpass"}
    tip = ws.stage_list.item(cr.STAGE_KEYS.index("notch")).toolTip()
    assert tip.startswith("Differs from the Schmitt run: notch Q 30 → 20.")
    # A mark is font and ink only: same height, same words, nothing scrolls.
    assert ws.stage_list.sizeHintForRow(0) == row_h
    assert ws.stage_list.verticalScrollBar().maximum() == 0
    for i, row in enumerate(ws.stage_rows()):
        assert ws.stage_list.item(i).text() == row.text

    det.set_params({"median_enabled": True})
    _settle(window)
    assert _marked(ws) == {"notch", "median", "lowpass"}

    assert ap.revert_stage("notch")
    _settle(window)
    assert _marked(ws) == {"median", "lowpass"}

    assert ap.revert_chain_to_shown_detection()
    _settle(window)
    assert _marked(ws) == set()
    assert ap.chain_status().kind == "match"


def test_the_stage_buttons_are_enabled_only_where_they_can_act(runs, tmp_path):
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel

    # Nothing differs: no stage Revert, and every Reset.
    for key in cr.STAGE_KEYS:
        det.select_stage(key)
        assert not ws.stage_revert_button.isEnabled(), key
        assert ws.stage_reset_button.isEnabled(), key
        assert not ap.can_revert_stage(key), key

    det.set_params({"notch_q": 20.0})
    _settle(window)
    det.select_stage("notch")
    assert ws.stage_revert_button.isEnabled() and ap.can_revert_stage("notch")
    assert "Put back the Notch settings" in ws.stage_revert_button.toolTip()
    det.select_stage("median")
    assert not ws.stage_revert_button.isEnabled(), "the median does not differ"
    assert not ap.can_revert_stage("median")

    # A recording with no table: nothing to revert to, while Reset still works on the form.
    _open(window, tmp_path, "other.dat")
    _settle(window)
    assert ap.chain_status().kind == "none"
    assert ws.stage_states() == (frozenset(), frozenset())
    det.select_stage("notch")
    assert not ws.stage_revert_button.isEnabled() and ws.stage_reset_button.isEnabled()
    said = _spy(ap.statusMessage)
    assert ap.revert_stage("notch") is False
    assert said == ["Nothing to revert the Notch stage to: no Schmitt/CUSUM detection on "
                    "screen recorded the chain it ran with."]
    ws.stage_reset_button.click()
    defaults = cr.reset_stage_params("notch")
    assert {k: det.params()[k] for k in defaults} == defaults


# ================================================================================================
# the chain notice: "Settings changed", and a Revert of only what differs
# ================================================================================================

def test_the_chip_says_settings_for_a_threshold_and_chain_for_the_chain(runs):
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    table = ap._table
    revert_entry = table._notice_actions["revert"]

    det.set_params({"k_in": det.params()["k_in"] + 1.0})
    status = _settle(window)
    assert status.thresholds_differ and not ap.stages_that_differ()
    assert table.notice_text() == "⚠ Settings changed"
    assert revert_entry.text() == "Revert form to this detection's settings"
    assert revert_entry.isEnabled()
    assert ws.revert_button.text() == "Revert to detection's settings"
    assert ws.revert_button.isEnabled()

    # The slot is sized for the longer notice, so neither elides where it has its width.
    chip = table.notice_chip
    for text in ("⚠ Settings changed", "⚠ Chain changed"):
        assert chip._size_for(text).width() <= chip.maximumWidth(), text
    assert chip.isVisible()
    shown = chip.width() - chip._chrome_width()
    assert chip.fontMetrics().horizontalAdvance(chip.text()) <= shown, (
        "at 1512 px the notice is painted whole")

    # The window's Revert is as wide in either wording: the status row never reflows.
    revert_width = ws.revert_button.width()
    assert ap.revert_chain_to_shown_detection()
    det.set_params({"filter_cutoff_hz": 5_000.0})
    status = _settle(window)
    assert status.kind == "differs" and not status.thresholds_differ
    assert table.notice_text() == "⚠ Chain changed"
    assert revert_entry.text() == "Revert form to this detection's chain"
    assert ws.revert_button.text() == "Revert to detection's chain"
    assert ws.revert_button.width() == revert_width


def test_the_notice_revert_puts_back_only_what_differs(runs, menu_answer):
    """Defect 7. At 27ec0eb this Revert wrote every chain key (measured on its export:
    ``revert_params`` held all 37 here), so the median window typed while the median was off
    came back as the run's 0.1 ms; and the k_in edit was not judged at all."""
    window, _ = runs
    ap = window._analysis_panel
    det = ap.detection_panel
    k_in = det.params()["k_in"]
    det.set_params({"median_window_s": 5e-4, "filter_cutoff_hz": 5_000.0,
                    "k_in": k_in + 1.0})
    status = _settle(window)
    assert set(status.revert_params) == {"filter_cutoff_hz", "filter_order", "filter_kind",
                                         "k_in"}
    steps, said = _spy(ap.formStepNamed), _spy(ap.statusMessage)

    menu_answer("Revert form to this detection's settings")
    ap._table.notice_chip.click()
    _settle(window)

    p = det.params()
    assert (p["filter_cutoff_hz"], p["k_in"]) == (10_000.0, k_in)
    assert p["median_window_s"] == pytest.approx(5e-4), "a setting that did not differ"
    assert ap.chain_status().kind == "match"
    assert steps == ["Revert settings to Schmitt's run"]
    assert said[-1] == "Form reverted to the settings the Schmitt detection on screen ran with."

    # A chain-only difference, through the window's own Revert, names the chain.
    det.set_params({"notch_enabled": False})
    _settle(window)
    window._signal_conditioning.revert_button.click()
    _settle(window)
    assert det.params()["notch_enabled"] is True
    assert steps == ["Revert settings to Schmitt's run", "Revert chain to Schmitt's run"]
    assert said[-1] == "Form reverted to the chain the Schmitt detection on screen ran with."


def test_a_run_that_never_recorded_what_differs_leaves_revert_disabled(window, tmp_path):
    """The run recorded its notch without its Q. The judge reads that Q at its default, so
    an edited Q differs, but nothing can put it back: every Revert is disabled, and one asked
    for anyway writes nothing and posts the judge's sentence, which offers a run."""
    doc_id = _open(window, tmp_path)
    ap = window._analysis_panel
    det = ap.detection_panel
    det.set_params(RUN_CHAIN)
    res = _run(window, doc_id, DetectSchmitt())
    window._session.add_result(replace(res, params={
        k: v for k, v in res.params.items() if k != "notch_q"}))
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    assert _settle(window).kind == "match"

    det.set_params({"notch_q": 20.0})
    status = _settle(window)
    assert status.kind == "differs" and status.revert_params is None
    assert status.sentence.endswith(
        "Run detection to apply it — Revert cannot restore a setting the run did not record.")
    assert not ws.revert_button.isEnabled()
    assert ws.revert_button.toolTip().startswith("Revert cannot restore a setting")
    assert not ap._table._notice_actions["revert"].isEnabled()
    assert ws.stage_states() == ({"notch"}, frozenset())
    assert _marked(ws) == {"notch"}
    assert "Revert stage cannot put it back" in ws.stage_list.item(0).toolTip()
    det.select_stage("notch")
    assert not ws.stage_revert_button.isEnabled()
    assert "did not record" in ws.stage_revert_button.toolTip()

    steps, said = _spy(ap.formStepNamed), _spy(ap.statusMessage)
    assert ap.revert_chain_to_shown_detection() is False
    assert said == [status.sentence]
    assert ap.revert_stage("notch") is False
    assert said[-1] == ("The Notch stage differs on a setting the Schmitt run did not record, "
                        "so Revert cannot put it back. Run detection to apply it.")
    assert det.params()["notch_q"] == 20.0 and steps == []


# ================================================================================================
# CUSUM: each label, and a legacy table judged on its own k and h
# ================================================================================================

def test_every_action_names_its_step_on_a_cusum_table_too(runs):
    window, _ = runs
    ap, ws = window._analysis_panel, window._signal_conditioning
    det = ap.detection_panel
    _show_table(window, C)
    assert _settle(window).kind == "match"
    assert det.params()["cusum_h"] == RUN_CUSUM_H, "the form follows the table on screen"
    steps = _spy(ap.formStepNamed)

    det.set_params({"notch_q": 20.0, "filter_cutoff_hz": 5_000.0})
    _settle(window)
    det.select_stage("lowpass")
    ws.stage_revert_button.click()
    det.select_stage("notch")
    ws.stage_reset_button.click()
    det.set_params({"cusum_h": RUN_CUSUM_H + 4.0})
    _settle(window)
    assert ws.revert_button.text() == "Revert to detection's settings"
    ws.revert_button.click()
    _settle(window)
    assert steps == ["Revert Low-pass stage", "Reset Notch stage",
                     "Revert settings to CUSUM's run"]
    p = det.params()
    assert (p["cusum_h"], p["notch_q"], p["filter_cutoff_hz"]) == (RUN_CUSUM_H, 30.0, 10_000.0)


def test_a_legacy_cusum_table_is_judged_on_k_and_h(runs, menu_answer):
    """A v1/v2 project restores a table with ``params={}``: its settings are the detector's
    stamp in the provenance. That stamp holds the run's k and h, and the notice now reads
    them from the record ``analysis.lineage`` reads, instead of a Schmitt-only projection
    that dropped both: the form is seeded with the run's h, an edit of h is judged, and a
    Revert puts it back."""
    window, doc_id = runs
    ap = window._analysis_panel
    det = ap.detection_panel
    legacy = replace(window._session.results.get(doc_id, 0, C), params={})
    stamp = legacy.payload.provenance
    assert (stamp["cusum_k"], stamp["cusum_h"]) == (det.params()["cusum_k"], RUN_CUSUM_H)
    det.set_params({"cusum_h": 28.0})            # what a fresh form would show
    window._session.add_result(legacy)
    _show_table(window, C)
    assert _settle(window).kind == "match"
    assert det.params()["cusum_h"] == RUN_CUSUM_H, "the seed put the run's h on the form"

    # One record for the notice and the Details list.
    read = AP.schmitt_params_from_result(legacy)
    assert {k: read[k] for k in ("cusum_k", "cusum_h")} == {
        k: made_with(legacy).thresholds[k] for k in ("cusum_k", "cusum_h")}
    # ...and the stale banner's Re-run replays the run's h, not the form's.
    det.set_params({"cusum_h": RUN_CUSUM_H + 4.0})
    assert ap._rerun_params(legacy)["cusum_h"] == RUN_CUSUM_H

    status = _settle(window)
    assert status.thresholds_differ
    assert [d.text for d in status.diffs] == ["CUSUM h 20 → 24"]
    assert ap._table.notice_text() == "⚠ Settings changed"
    steps = _spy(ap.formStepNamed)
    menu_answer("Revert form to this detection's settings")
    ap._table.notice_chip.click()
    _settle(window)
    assert det.params()["cusum_h"] == RUN_CUSUM_H
    assert ap.chain_status().kind == "match"
    assert steps == ["Revert settings to CUSUM's run"]
