"""Two gaps the integrated review of the owner's decisions found in the polarity's territory.

* **A Re-measure lost a legacy table's run.** A v1/v2 table stored no params, so its run is
  known from the detector's stamp alone (``schmitt_params_from_result``, ``lineage._run_params``),
  and the readers go to the stamp only while params is empty. Re-measure installed the Baseline
  form's keys over that empty dict, so the table was then judged against a run with no chain,
  thresholds or polarity: "⚠ Differs from the Schmitt run: low-pass off → on", a Revert that
  turned the run's low-pass off, a polarity edit no longer judged, and a later Re-run that
  replaced the table with another detection (no low-pass, k_in 5, auto). The amber line offers
  exactly this Re-measure for such a table ("predates the current estimator").
* **Details was silent on a polarity edit with the drift stage on.** The chain notice named it
  ("baseline drift polarity auto → blockade"), but the Signal chain row printed the stage's
  summary, which shows no polarity, on both sides of its arrow, and the Thresholds row read
  "unchanged" (the stage names it, so it is not named twice). The same held for every drift
  knob the summary leaves out (mask σ, dilate, percentile, ...).

Synthetic recordings only, each under ``tmp_path``. No sidecar is written: the gain that raises
the amber line is set in the session only.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import json
import os
import zipfile

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
N = 60_000
S, C = "detect.schmitt", "detect.cusum"
_SPANS = [(5_000, 5_400), (15_000, 15_500), (25_000, 25_600), (40_000, 40_500)]
#: The run of the legacy table: a 10 kHz low-pass, the drift stage off, k_in 6 and blockade,
#: none of them the form's default.
RUN = {"filter_cutoff_hz": 10_000.0, "detrend_enabled": False, "k_in": 6.0}
#: The polarity combo's rows (``DetectionPanel.polarity_override``).
AUTO, BLOCKADE = 0, 1


# ================================================================================================
# builders
# ================================================================================================

def _signal() -> np.ndarray:
    rng = np.random.default_rng(3)
    x = np.full(N, -1_000.0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
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


def _open(w: MainWindow, path: Path) -> str:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, t0_s=0.0, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    return doc_id


def _settle(w: MainWindow, qtbot=None, ms: int = 0) -> cr.ChainStatus:
    QtWidgets.QApplication.processEvents()
    if qtbot is not None and ms:
        qtbot.wait(ms)
    status = w._analysis_panel.refresh_chain_status()
    QtWidgets.QApplication.processEvents()
    return status


def _gain(w: MainWindow) -> None:
    """Raise the amber line: a gain of 2 set in the session only (no sidecar)."""
    doc_id, channel = w._analysis_panel._sel
    w._session.set_calibration(doc_id, channel, ChannelCalibration(
        gain=2.0, offset=0.0, target_units="pA"), write_sidecar=False)
    QtWidgets.QApplication.processEvents()


def _row(text: str, title: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"• {title}"))


def _legacy_table(w: MainWindow, doc_id: str, module=None, **params):
    """A v1/v2 table as a restore installs one: the run's params gone (``{}``) and no estimator
    stamp, so its run is known from the detector's stamp alone."""
    det = w._analysis_panel.detection_panel
    det.set_params(RUN)
    det.polarity_override.setCurrentIndex(BLOCKADE)
    form = {**w._analysis_panel._form_params(), **params}
    res = (module or DetectSchmitt()).run(SimpleAnalysisContext(
        w._session.calibrated(doc_id, 0), w._session.document(doc_id).meta, form))
    assert len(res.payload.df), "the synthetic events were found"
    w._session.add_result(replace(res, params={}, baseline_fingerprint=None))
    QtWidgets.QApplication.processEvents()
    return res


def _remeasure_from_the_banner(w: MainWindow, qtbot) -> cr.ChainStatus:
    banner = w._analysis_panel._stale
    assert not banner.isHidden() and banner._btn.text() == "Re-measure depths", banner._btn.text()
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        banner._btn.click()
    return _settle(w, qtbot, 100)


# ================================================================================================
# Re-measure keeps a legacy table's run
# ================================================================================================

@pytest.mark.parametrize("module_id", [S, C])
def test_remeasure_of_a_legacy_table_keeps_its_run(window, qtbot, tmp_path, module_id):
    ap = window._analysis_panel
    det = ap.detection_panel
    doc_id = _open(window, tmp_path / "pore.dat")
    cusum = module_id == C
    ran = _legacy_table(window, doc_id, *((DetectCusum(),) if cusum else ()),
                        **({"detector": C} if cusum else {}))
    if cusum:
        selector = ap._result_selector
        selector.setCurrentIndex(selector.findData(C))
        QtWidgets.QApplication.processEvents()
    assert ap.current_detection().module_id == module_id and ap.current_detection().params == {}
    assert _settle(window).kind == "match", "precondition: the form is seeded from the stamp"

    _remeasure_from_the_banner(window, qtbot)
    cur = ap.current_detection()
    assert cur.baseline_fingerprint, "the depths were re-measured"
    # The run is still the whole run: its chain, thresholds and polarity, under the new estimator.
    assert cur.params["filter_cutoff_hz"] == 10_000.0 and cur.params["polarity"] == -1
    assert cur.params["detrend_enabled"] is False
    threshold = "cusum_k" if cusum else "k_in"
    assert cur.params[threshold] == ran.params[threshold]
    assert any(str(k).startswith("baseline_") for k in cur.params)

    status = _settle(window)
    assert status.kind == "match", status.sentence      # was "low-pass off → on"
    made = made_with(cur)
    assert made.chain["filter_cutoff_hz"] == 10_000.0
    assert made.thresholds[threshold] == ran.params[threshold]

    # The polarity is judged as before the Re-measure.
    det.polarity_override.setCurrentIndex(AUTO)
    assert _settle(window).headline == "⚠ Settings changed: polarity blockade → auto"
    det.polarity_override.setCurrentIndex(BLOCKADE)
    assert _settle(window).kind == "match"

    # Revert puts the run's low-pass back (it turned it off).
    det.set_params({"filter_cutoff_hz": 5_000.0})
    assert _settle(window).kind == "differs"
    assert ap.revert_chain_to_shown_detection()
    _settle(window)
    assert det.params()["filter_cutoff_hz"] == 10_000.0

    # The stale banner's Re-run replays the same detection.
    replay = ap._rerun_params(cur)
    assert replay["detector"] == module_id
    assert (replay["filter_cutoff_hz"], replay["polarity"], replay[threshold]) == (
        10_000.0, -1, ran.params[threshold])


def _v2_project(w: MainWindow, qtbot, tmp_path: Path) -> Path:
    """A Schmitt table detected with ``RUN`` and blockade, saved, and its manifest rewritten to
    what a v2 project holds: no params and no estimator stamp."""
    csv = tmp_path / "pore.csv"
    pd.DataFrame({"time": np.arange(N) / FS, "current": _signal().astype(np.float64)}).to_csv(
        csv, index=False)
    w.open_paths([str(csv)])
    qtbot.waitUntil(lambda: w._active_open is None and not w._open_queue, timeout=30_000)
    doc_id = w._session.doc_id_for_source(str(csv.resolve()))
    if w._analysis_panel._sel != (doc_id, 0):
        w._on_selection(doc_id, 0)
    ap = w._analysis_panel
    ap.detection_panel.set_params(RUN)
    ap.detection_panel.polarity_override.setCurrentIndex(BLOCKADE)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=30_000):
        ap.trigger_detection()
    _settle(w, qtbot, 50)
    saved = tmp_path / "saved.nanoscope"
    assert w._write_project(str(saved))

    with zipfile.ZipFile(saved) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["version"] = 2
    for result in manifest["results"]:
        result.pop("params", None)
        result.pop("baseline_fingerprint", None)
        (result.get("provenance") or {}).pop("baseline_fingerprint", None)
    members["manifest.json"] = json.dumps(manifest).encode()
    legacy = tmp_path / "legacy_v2.nanoscope"
    with zipfile.ZipFile(legacy, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return legacy


def test_a_v2_project_re_measured_then_re_run_replays_its_own_detection(
        window, qtbot, tmp_path, monkeypatch):
    """Through the user's doors: File ▸ Open Project of a v2 project, the amber line's
    "Re-measure depths", a gain change, and the amber line's Re-run. The Re-run landed with no
    low-pass, k_in 5 and auto, where the run had 10 kHz, k_in 6 and blockade."""
    legacy = _v2_project(window, qtbot, tmp_path)
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self, *a, **k: None,
                        raising=False)
    w2 = MainWindow()
    qtbot.addWidget(w2)
    try:
        with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
            w2._open_project(str(legacy))
        ap = w2._analysis_panel
        assert ap.current_detection().params == {}
        assert _settle(w2, qtbot, 50).kind == "match"

        assert _remeasure_from_the_banner(w2, qtbot).kind == "match"
        _gain(w2)
        banner = ap._stale
        assert banner._btn.text() == "Re-run" and banner._btn.isEnabled()
        with qtbot.waitSignal(w2._analysis_controller.finished, timeout=30_000):
            banner._btn.click()
        _settle(w2, qtbot, 100)
        prov = ap.current_detection().payload.provenance
        assert (prov.get("filter_cutoff_hz"), prov.get("k_in"), prov.get("polarity_requested")) \
            == (10_000.0, 6.0, -1)
    finally:
        w2._analysis_controller.abort()


# ================================================================================================
# Details names what a stage's summary leaves out
# ================================================================================================

def _form(**over):
    return {**cr.complete_recipe({"filter_cutoff_hz": 10_000.0, "detrend_enabled": True}),
            "k_in": 5.0, "k_out": 3.0, "min_dwell_s": 2e-5, "max_dwell_s": None,
            "merge_gap_s": None, "polarity": None, **over}


def _chain_row(run: dict, form: dict):
    made = made_with(SimpleNamespace(module_id=S, params=run, payload=None,
                                     calibration_fingerprint=(1.0, 0.0),
                                     baseline_fingerprint=None))
    return judge_inputs(made, LiveInputs(rate_hz=FS, chain=form, thresholds=form), fs=FS)[3]


def test_the_chain_row_names_what_the_stage_summary_leaves_out():
    run = _form()
    # The polarity, which the drift stage's "block median · 0.5 s" does not show.
    row = _chain_row(run, _form(polarity=-1))
    assert (row.state, row.action) == ("changed", "revert")
    assert (row.was, row.now) == ("baseline drift polarity auto", "baseline drift polarity blockade")

    # Any knob the summary leaves out: the event-masked median's mask.
    masked = _form(detrend_method="masked_median")
    row = _chain_row(masked, dict(masked, detrend_mask_sigma=2.0))
    assert (row.was, row.now) == ("baseline drift mask 3.5σ", "baseline drift mask 2σ")

    # With a knob the summary does show: the summary, and the polarity after it.
    row = _chain_row(run, _form(detrend_window_s=1.0, polarity=-1))
    assert (row.was, row.now) == (
        "Baseline drift block median · 0.5 s; baseline drift polarity auto",
        "Baseline drift block median · 1 s; baseline drift polarity blockade")
    assert [d.text for d in cr.summary_hidden_differences(
        _form(detrend_window_s=1.0, polarity=-1), run, fs=FS)] == [
        "baseline drift polarity auto → blockade"]

    # A stage whose summary shows what moved reads as it did.
    row = _chain_row(run, _form(filter_cutoff_hz=5_000.0))
    assert (row.was, row.now) == ("Low-pass Bessel 8 · 10 kHz", "Low-pass Bessel 8 · 5 kHz")


def test_details_names_the_polarity_with_the_drift_stage_on(window, tmp_path):
    doc_id = _open(window, tmp_path / "pore.dat")
    ap = window._analysis_panel
    det = ap.detection_panel
    det.set_params({"filter_cutoff_hz": 10_000.0, "detrend_enabled": True})
    det.polarity_override.setCurrentIndex(AUTO)
    res = DetectSchmitt().run(SimpleAnalysisContext(
        window._session.calibrated(doc_id, 0), window._session.document(doc_id).meta,
        ap._form_params()))
    window._session.add_result(res)
    assert _settle(window).kind == "match"
    det.polarity_override.setCurrentIndex(BLOCKADE)
    window._on_edit_menu_about_to_show()
    assert _settle(window).headline == "⚠ Differs from the Schmitt run: baseline drift polarity " \
                                       "auto → blockade"

    _gain(window)
    banner = ap._stale
    assert not banner.isHidden(), "precondition: the amber line is up"
    banner._details.click()
    text = banner.inputs_text()
    assert _row(text, "Signal chain:") == (
        "• Signal chain: baseline drift polarity auto → baseline drift polarity blockade — "
        "Revert, or re-run to apply"), text
    assert _row(text, "Thresholds:") == "• Thresholds: unchanged", "named once, by the stage"


# ================================================================================================
# A legacy table's Re-measure is named a Re-measure; one re-measured before the fix reads its run
# ================================================================================================

_FALSE_LOSS = "Your curation is not carried over"


def _reject_first(w: MainWindow) -> None:
    df = w._analysis_panel.current_detection().payload.df
    assert w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    w._on_edit_menu_about_to_show()


def _rejected(w: MainWindow) -> int:
    df = w._analysis_panel.current_detection().payload.df
    return int((~df["accepted"].astype(bool)).sum())


def _move_the_estimator(w: MainWindow) -> None:
    """A Baseline edit, committed as the Edit menu commits it: the step a Re-measure folds."""
    box = w._analysis_panel.baseline_panel.rough_s
    box.setValue(box.value() * 4.0)
    w._on_edit_menu_about_to_show()


def _undo_text(w: MainWindow) -> str:
    w._on_edit_menu_about_to_show()
    return w._act_undo.text()


@pytest.mark.parametrize("module_id", [S, C])
def test_a_legacy_re_measure_is_named_a_re_measure(window, qtbot, tmp_path, module_id):
    """The stamp written down as params moved no detector setting: the step was named "Re-run
    detection" and the status bar told the user their curation was not carried over."""
    ap = window._analysis_panel
    doc_id = _open(window, tmp_path / "pore.dat")
    cusum = module_id == C
    _legacy_table(window, doc_id, *((DetectCusum(),) if cusum else ()),
                  **({"detector": C} if cusum else {}))
    if cusum:
        ap._result_selector.setCurrentIndex(ap._result_selector.findData(C))
        QtWidgets.QApplication.processEvents()
    assert ap.current_detection().params == {}
    _reject_first(window)
    _move_the_estimator(window)
    seen: list[str] = []
    window.statusBar().messageChanged.connect(seen.append)

    _remeasure_from_the_banner(window, qtbot)
    assert ap.current_detection().params["filter_cutoff_hz"] == 10_000.0, "the run was kept"
    assert window._history.last_step_run, "a run, for the fold of design 3.6"
    assert _undo_text(window) == "Undo Re-measure depths and 1 setting change"
    assert not [m for m in seen if _FALSE_LOSS in m], seen
    assert _rejected(window) == 1


def test_a_project_re_measured_before_the_fix_reads_its_run(window, qtbot, tmp_path, monkeypatch):
    """A v2 project re-measured while the writer stored the estimator's keys alone, saved and
    reopened: the notice read "low-pass off → on", the form was not seeded, Details and Re-run
    read no chain, thresholds or polarity, and a later Re-measure kept the bad dict."""
    from nanoscope.gui import analysis_panel as panel_module

    legacy = _v2_project(window, qtbot, tmp_path)
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self, *a, **k: None,
                        raising=False)
    w2 = MainWindow()
    qtbot.addWidget(w2)
    try:
        with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
            w2._open_project(str(legacy))
        _settle(w2, qtbot, 50)
        # The writer as it stood: a table with no params re-measured from ``{}``.
        with monkeypatch.context() as m:
            m.setattr(panel_module, "schmitt_params_from_result", lambda res: {})
            _remeasure_from_the_banner(w2, qtbot)
        resaved = tmp_path / "re_measured.nanoscope"
        assert w2._write_project(str(resaved))
    finally:
        w2._analysis_controller.abort()

    w3 = MainWindow()
    qtbot.addWidget(w3)
    try:
        with qtbot.waitSignal(w3.projectRestored, timeout=30_000):
            w3._open_project(str(resaved))
        ap = w3._analysis_panel
        cur = ap.current_detection()
        assert cur.params and all(str(k).startswith("baseline_") for k in cur.params), \
            "precondition: the shape such a project holds"
        prov = cur.payload.provenance
        assert (prov["filter_cutoff_hz"], prov["k_in"], prov["polarity_requested"]) == (
            10_000.0, 6.0, -1), "precondition: the stamp is intact"

        assert _settle(w3, qtbot, 50).kind == "match"            # was "low-pass off → on"
        assert ap.detection_panel.params()["filter_cutoff_hz"] == 10_000.0
        assert ap.detection_panel.polarity_override.currentIndex() == BLOCKADE
        made = made_with(cur)
        assert (made.chain["filter_cutoff_hz"], made.chain["polarity"]) == (10_000.0, -1)
        assert made.thresholds["k_in"] == 6.0
        replay = ap._rerun_params(cur)
        assert (replay["filter_cutoff_hz"], replay["polarity"], replay["k_in"]) == (10_000.0, -1, 6.0)

        # A later Re-measure writes the whole run down, and is named a Re-measure.
        _reject_first(w3)
        _move_the_estimator(w3)
        seen: list[str] = []
        w3.statusBar().messageChanged.connect(seen.append)
        assert _remeasure_from_the_banner(w3, qtbot).kind == "match"
        params = ap.current_detection().params
        assert (params["filter_cutoff_hz"], params["polarity"], params["k_in"]) == (10_000.0, -1, 6.0)
        assert _undo_text(w3) == "Undo Re-measure depths and 1 setting change"
        assert not [m for m in seen if _FALSE_LOSS in m], seen
    finally:
        w3._analysis_controller.abort()
