"""Acceptance journeys through Fits, sub-state export, reconciliation and training."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtTest, QtWidgets

from nanoscope.analysis.baseline import BaselineConfig
from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.events import EventSet, REVIEW_COL, REVIEW_DONE
from nanoscope.analysis.labels import LabelVocabulary
from nanoscope.analysis.reconcile import (
    CONSENSUS_MODULE_ID, CONSENSUS_CURATION_CONFLICT_COL, CONSENSUS_LABEL_CONFLICTS_COL,
)
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui.reconcile_dialog import ReconcileDialog

# Collection precedes the session modal guard. Restore native exec only on the
# concrete dialogs exercised here; all other modal cancellation guards stay active.
_NATIVE_DIALOG_EXEC = QtWidgets.QDialog.exec
_artifact_env = os.environ.get("NANOSCOPE_BETA_NEXT_ARTIFACTS")
ARTIFACTS = Path(_artifact_env) if _artifact_env else None
FS = 20000.0


@pytest.fixture
def real_workflow_dialogs(monkeypatch):
    assert not getattr(_NATIVE_DIALOG_EXEC, "_nanoscope_modal_guard", False)
    monkeypatch.setattr(ReconcileDialog, "exec", _NATIVE_DIALOG_EXEC)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _NATIVE_DIALOG_EXEC)


def note(stage, **details):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "stages.jsonl").open("a") as f:
        f.write(json.dumps(dict(pid=os.getpid(), stage=stage, time=time.time(), **details)) + "\n")


@pytest.fixture
def journey(qapp, qtbot, tmp_path, monkeypatch):
    if ARTIFACTS is None:
        monkeypatch.setitem(globals(), "ARTIFACTS", tmp_path / "acceptance-evidence")
    note("window_start")
    rng = np.random.default_rng(919)
    n = 180000
    signal = rng.normal(100, .35, n).astype(np.float32)
    spans = []
    for i, start in enumerate(range(2000, n - 3000, 3000)):
        length = int(100 + min(rng.exponential(220), 900))
        end = start + length
        middle = (start + end) // 2
        signal[start:middle] -= 20 + i % 7
        signal[middle:end] -= 40 + i % 7
        spans.append((start, end))
    path = tmp_path / "acceptance-recording.dat"
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=n,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
        channels=(ChannelInfo(0, "current", "pA", ChannelKind.CURRENT),))
    window = mw.MainWindow()
    qtbot.addWidget(window)
    doc_id = window._session.add_document(TraceDocument(meta, (signal,)))
    window._create_default_layers(window._session.document(doc_id))
    window._on_selection(doc_id, 0)
    window.resize(1400, 950)
    window.show()
    QtWidgets.QApplication.processEvents()
    yield window, doc_id, signal, spans
    window._analysis_controller.abort()
    window._dirty = False
    note("window_finished")


def source(window, doc_id, signal, spans, module="detect.schmitt"):
    df = characterize(signal, np.full(len(signal), 100),
        [s for s, _ in spans], [e for _, e in spans], FS, -1, noise_rms=.35)
    cfg = BaselineConfig(method="constant")
    prov = dict(fs=FS, units="pA", polarity=-1, min_dwell_s=.0002,
                n_samples=len(signal), analyzed_duration_s=len(signal)/FS,
                analysed_start_idx=0, analysed_stop_idx=len(signal),
                **cfg.to_provenance(FS))
    result = AnalysisResult(module, "acceptance", doc_id, 0,
        {"min_dwell_s": .0002, "baseline_config": cfg}, (1., 0.), EventSet(df, prov))
    window._session.add_result(result)
    QtWidgets.QApplication.processEvents()
    return result


def test_fits_tab_and_substate_export(journey, qtbot, monkeypatch):
    window, doc_id, signal, spans = journey
    result = source(window, doc_id, signal, spans)
    df = result.payload.df
    df.loc[0, "accepted"] = False
    df.loc[[1, 2], "filter_limited"] = True
    panel = window._analysis_panel
    panel.resync_curation()
    tabs = panel.results_tabs
    tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == "Fits"))
    stats = panel._stats
    QtWidgets.QApplication.processEvents()
    assert stats.fit_button.isVisible() and stats.fit_button.isEnabled()
    before = df.copy(deep=True)
    note("fits_button_click", events=len(df))
    QtTest.QTest.mouseClick(stats.fit_button, QtCore.Qt.LeftButton)
    dwell = window._session.results.get(doc_id, 0, "fit.dwell")
    depth = window._session.results.get(doc_id, 0, "fit.depth")
    assert dwell is not None and depth is not None, stats.results.toPlainText()
    selected = df.loc[df.accepted & ~df.filter_limited]
    assert dwell.payload.n == depth.payload.n == len(selected)
    expected_tau = selected.dwell_s.mean() - .0002
    assert dwell.payload.params["tau_1"] == pytest.approx(expected_tau, rel=1e-8)
    assert "failed" not in stats.results.toPlainText().lower()
    pd.testing.assert_frame_equal(df, before, check_exact=True)
    window.grab().save(str(ARTIFACTS / "fits-tab.png"))
    note("fits_complete", used=len(selected), tau_s=dwell.payload.params["tau_1"])

    assert window._act_fit_levels.isEnabled()
    worker_failures = []
    window._analysis_controller.failed.connect(worker_failures.append)
    window._act_fit_levels.trigger()
    qtbot.waitUntil(lambda: window._level_fit is not None or bool(worker_failures), timeout=30000)
    assert not worker_failures, worker_failures
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10000)
    levels = window._level_fit.payload
    assert not levels.empty
    assert set(levels.event_id) == set(df.loc[df.accepted, "event_id"])
    for event_id, group in levels.groupby("event_id"):
        parent = df.loc[df.event_id == event_id].iloc[0]
        g = group.sort_values("level_idx")
        assert g.start_idx.iloc[0] == parent.start_idx
        assert g.end_idx.iloc[-1] == parent.end_idx
        np.testing.assert_array_equal(g.start_idx.to_numpy()[1:], g.end_idx.to_numpy()[:-1])
        assert g.dwell_s.sum() == pytest.approx(parent.dwell_s)
    pd.testing.assert_frame_equal(df, before, check_exact=True)
    destination = ARTIFACTS / "substate-levels.csv"
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: destination)
    assert window._act_export_levels.isEnabled()
    window._act_export_levels.trigger()
    saved = pd.read_csv(destination)
    assert len(saved) == len(levels)
    for col in ("event_id", "start_idx", "end_idx", "dwell_s", "depth_pA"):
        np.testing.assert_allclose(saved[col], levels[col], rtol=1e-12, atol=1e-12)
    assert destination.with_suffix(".meta.json").exists()
    note("level_fit_export_complete", parent_events=levels.event_id.nunique(), level_rows=len(levels))


def reconcile_with_buttons(window, monkeypatch, qtbot):
    dialogs = []
    def factory(sources):
        dialog = ReconcileDialog(sources, window)
        dialogs.append(dialog)
        dialog.accepted.connect(lambda: note("reconcile_dialog_accepted"))
        dialog.rejected.connect(lambda: note("reconcile_dialog_rejected"))
        def choose():
            try:
                assert dialog.isVisible()
                QtTest.QTest.mouseClick(dialog.preview_button, QtCore.Qt.LeftButton)
                note("reconcile_preview", output=int(dialog.summary_labels["n_output_rows"].text()))
                assert dialog.create_button.isEnabled()
                QtTest.QTest.mouseClick(dialog.create_button, QtCore.Qt.LeftButton)
            except BaseException:
                dialog.reject()
                raise
        QtCore.QTimer.singleShot(25, choose)
        return dialog
    monkeypatch.setattr(window, "_make_reconcile_dialog", factory)
    assert window._act_reconcile.isEnabled()
    window._act_reconcile.trigger()
    note("reconcile_action_return", result=dialogs[-1].result(),
         current=window._analysis_panel.current_detection().module_id,
         outcome=dialogs[-1].outcome is not None)
    qtbot.waitUntil(lambda: window._analysis_panel.current_detection().module_id == CONSENSUS_MODULE_ID,
                    timeout=3000)
    stored = window._analysis_panel.current_detection()
    assert stored.module_id == CONSENSUS_MODULE_ID
    return stored


def choose_training(button_text, followup=None, observed=None):
    deadline = time.monotonic() + 5
    def choose():
        dialog = QtWidgets.QApplication.activeModalWidget()
        if (not isinstance(dialog, QtWidgets.QMessageBox) or not dialog.isVisible()) and time.monotonic() < deadline:
            QtCore.QTimer.singleShot(25, choose)
            return
        try:
            assert isinstance(dialog, QtWidgets.QMessageBox)
            labels = [b.text() for b in dialog.buttons()]
            note("training_dialog", title=dialog.windowTitle(), buttons=labels,
                 default=dialog.defaultButton().text())
            button = next(b for b in dialog.buttons() if button_text in b.text())
            assert button.isEnabled()
            if followup is not None:
                QtCore.QTimer.singleShot(25, followup)
            QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
            if observed is not None:
                observed.append(button_text)
        except BaseException:
            if isinstance(dialog, QtWidgets.QDialog):
                dialog.reject()
            raise
    return choose


def test_reconcile_reviewed_training_end_to_end(journey, monkeypatch, qtbot, real_workflow_dialogs):
    window, doc_id, signal, spans = journey
    a = source(window, doc_id, signal, spans[:2])
    b = source(window, doc_id, signal, [spans[0], spans[2]], "detect.cusum")
    for res in (a, b):
        res.payload.df[REVIEW_COL] = pd.array([REVIEW_DONE, REVIEW_DONE], dtype="string")
    a.payload.df.loc[0, "flag"] = "molecule=apo"
    b.payload.df.loc[0, "flag"] = "molecule=holo"
    b.payload.df.loc[0, "accepted"] = False
    a.payload.df.loc[1, "accepted"] = False
    a.payload.df.loc[1, "flag"] = "molecule=apo"
    b.payload.df.loc[1, REVIEW_COL] = "unreviewed"
    snapshots = [r.payload.df.copy(deep=True) for r in (a, b)]
    result = reconcile_with_buttons(window, monkeypatch, qtbot)
    df = result.payload.df
    assert len(df) == 3
    for res, before in zip((a, b), snapshots):
        pd.testing.assert_frame_equal(res.payload.df, before, check_exact=True)
    destination = ARTIFACTS / "reviewed-before-resolution.csv"
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: destination)
    assert window._act_export_training.isEnabled()
    QtCore.QTimer.singleShot(25, choose_training("Reviewed decisions"))
    window._act_export_training.trigger()
    first = pd.read_csv(destination)
    assert len(first) == 1 and not bool(first.accepted.iloc[0])
    assert first.label_molecule.tolist() == ["apo"]
    meta = json.loads(destination.with_suffix(".meta.json").read_text())
    assert meta["selection"] == "reviewed" and meta["detector"] == CONSENSUS_MODULE_ID
    assert meta["n_rows"] == 1
    note("training_reviewed_withheld_conflicts", rows=len(first))

    row = int(np.flatnonzero(df[CONSENSUS_CURATION_CONFLICT_COL].fillna("").ne("").to_numpy())[0])
    event_id = int(df.iloc[row].event_id)
    window._analysis_panel.set_event_accepted(event_id, True)
    window._open_event_inspector()
    inspector = window._event_inspector
    inspector.show_event(event_id)
    vocabulary = LabelVocabulary()
    vocabulary.add("molecule", "apo", hotkey="1")
    inspector.set_vocabulary(vocabulary)
    inspector.apply_hotkey("1")
    QtWidgets.QApplication.processEvents()
    assert df.iloc[row][CONSENSUS_CURATION_CONFLICT_COL] == ""
    assert df.iloc[row][CONSENSUS_LABEL_CONFLICTS_COL] == ""
    destination = ARTIFACTS / "reviewed-after-resolution.parquet"
    assert window._act_export_training.isEnabled()
    QtCore.QTimer.singleShot(25, choose_training("Reviewed decisions"))
    window._act_export_training.trigger()
    second = pd.read_parquet(destination)
    assert len(second) == 2 and set(second.accepted) == {False, True}
    assert set(second.label_molecule) == {"apo"}
    assert "|" not in "".join(second.label_molecule)
    meta = json.loads(destination.with_suffix(".meta.json").read_text())
    assert meta["selection"] == "reviewed" and meta["n_rows"] == 2
    note("training_reviewed_resolved", rows=len(second), positives=int(second.accepted.sum()))


def test_training_conflict_cancel_writes_nothing(journey, monkeypatch, qtbot, real_workflow_dialogs):
    window, doc_id, signal, spans = journey
    result = source(window, doc_id, signal, spans[:3])
    result.payload.df.loc[0, "flag"] = "molecule=apo|holo"
    result.payload.df.loc[0, REVIEW_COL] = REVIEW_DONE
    requested = []
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: requested.append(True))
    qtbot.waitUntil(window._act_export_training.isEnabled, timeout=3000)
    note("training_action_ready", enabled=window._act_export_training.isEnabled())
    observed = []
    QtCore.QTimer.singleShot(25, choose_training(
        "All raw candidates", choose_training("Cancel", observed=observed), observed=observed))
    window._act_export_training.trigger()
    assert observed == ["All raw candidates", "Cancel"]
    assert requested == []
    note("training_conflict_cancel_preserved")
