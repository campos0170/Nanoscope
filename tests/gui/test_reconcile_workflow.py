"""GUI contract for non-destructive detector reconciliation."""

from __future__ import annotations

import os

import nanoscope  # noqa: F401 - Qt bootstrap before widget imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6 import QtWidgets

from nanoscope.analysis.events import (
    FLAG_CURATED,
    REVIEW_COL,
    REVIEW_DONE,
    EventSet,
    empty_df,
)
from nanoscope.analysis.reconcile import (
    CONSENSUS_CURATION_CONFLICT_COL,
    CONSENSUS_LABEL_CONFLICTS_COL,
    CONSENSUS_MODULE_ID,
    ReconcileConfig,
    reconcile_detections,
)
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import load_project, save_project
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui import main_window as main_window_module
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.reconcile_dialog import ReconcileDialog

FS = 10_000.0
N = 1_000


def _events(spans: list[tuple[int, int]]) -> EventSet:
    n = len(spans)
    rows = pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "recording": pd.array(["rec"] * n, dtype="string"),
        "pore_id": pd.array(["p0"] * n, dtype="string"),
        "start_idx": np.asarray([s for s, _ in spans], dtype=np.int64),
        "end_idx": np.asarray([e for _, e in spans], dtype=np.int64),
        "t_start_s": np.asarray([s / FS for s, _ in spans], dtype=float),
        "dwell_s": np.asarray([(e - s) / FS for s, e in spans], dtype=float),
        "baseline_pA": np.full(n, -50.0),
        "depth_pA": np.full(n, 20.0),
        "frac_depth": np.full(n, 0.4),
        "ecd_pA_s": np.full(n, 0.02),
        "max_dev_pA": np.full(n, 25.0),
        "mean_dev_pA": np.full(n, 20.0),
        "rise_s": np.full(n, 1e-4),
        "fall_s": np.full(n, 1e-4),
        "snr": np.full(n, 8.0),
        "n_levels": np.ones(n, dtype=np.int64),
        "filter_limited": np.zeros(n, dtype=bool),
        "accepted": np.ones(n, dtype=bool),
        "flag": pd.array([""] * n, dtype="string"),
    })
    frame = pd.concat([empty_df(), rows], ignore_index=True)
    return EventSet(frame, {
        "recording": "rec",
        "fs": FS,
        "units": "pA",
        "analyzed_duration_s": N / FS,
        "analysed_start_idx": 0,
        "analysed_stop_idx": N,
    })


def _result(module_id: str, spans: list[tuple[int, int]], *, doc_id: str = "doc"):
    return AnalysisResult(
        module_id=module_id,
        module_version="1.0",
        doc_id=doc_id,
        channel=0,
        params={},
        calibration_fingerprint=(1.0, 0.0),
        baseline_fingerprint="baseline-a",
        payload=_events(spans),
    )


def _sources(doc_id: str = "doc") -> tuple[AnalysisResult, AnalysisResult]:
    return (
        _result("detect.schmitt", [(10, 30), (100, 120)], doc_id=doc_id),
        _result("detect.cusum", [(12, 28), (200, 220)], doc_id=doc_id),
    )


def _contradictory_sources(
        doc_id: str = "doc") -> tuple[AnalysisResult, AnalysisResult]:
    """Two overlapping, explicitly reviewed rows with opposite human verdicts."""
    sources = list(_sources(doc_id))
    for res in sources:
        res.payload.df[REVIEW_COL] = pd.array(
            [REVIEW_DONE] + [pd.NA] * (len(res.payload.df) - 1), dtype="string")
    sources[0].payload.df.loc[0, "accepted"] = True
    sources[1].payload.df.loc[0, "accepted"] = False
    return tuple(sources)


def _doc(tmp_path) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "rec.dat",
        format_name="synthetic",
        n_samples=N,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(np.zeros(N, dtype=np.float32),))


def _window(qtbot, tmp_path) -> tuple[MainWindow, str, tuple[AnalysisResult, ...]]:
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _doc(tmp_path)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    sources = _sources(doc_id)
    for res in sources:
        win._session.add_result(res)
    QtWidgets.QApplication.processEvents()
    return win, doc_id, sources


def test_dialog_requires_two_sources_and_invalidates_a_stale_preview(qapp, qtbot):
    dialog = ReconcileDialog(_sources())
    qtbot.addWidget(dialog)
    assert dialog.support_spin.value() == 1
    assert dialog.overlap_spin.value() == 50.0
    assert dialog.preview_button.isEnabled()
    assert not dialog.create_button.isEnabled()

    dialog.preview()
    assert dialog.outcome is not None
    assert dialog.outcome.n_input_rows == 4
    assert dialog.outcome.n_clusters == 3
    assert dialog.outcome.n_output_rows == 3
    assert dialog.outcome.n_collapsed == 1
    assert dialog.create_button.isEnabled()
    assert dialog.summary_labels["n_output_rows"].text() == "3"

    dialog.overlap_spin.setValue(75.0)
    assert dialog.outcome is None
    assert not dialog.create_button.isEnabled()

    dialog.source_checks["detect.cusum"].setChecked(False)
    assert not dialog.preview_button.isEnabled()
    assert "at least two" in dialog.preview_message.text().lower()


def test_one_mixed_manual_and_automatic_table_can_be_reconciled(qapp, qtbot):
    source = _result("detect.schmitt", [(10, 30), (11, 29), (100, 120)])
    source.payload.df["added"] = np.array([False, True, False], dtype=bool)
    dialog = ReconcileDialog((source,))
    qtbot.addWidget(dialog)

    assert dialog.preview_button.isEnabled()
    assert dialog.support_spin.maximum() == 1
    dialog.preview()

    assert dialog.outcome is not None
    assert dialog.outcome.input_modules == ("detect.schmitt",)
    assert dialog.outcome.n_input_rows == 3
    assert dialog.outcome.n_output_rows == 2
    assert dialog.outcome.n_collapsed == 1


def test_dialog_support_two_previews_detector_agreement_only(qapp, qtbot):
    dialog = ReconcileDialog(_sources())
    qtbot.addWidget(dialog)
    dialog.support_spin.setValue(2)
    dialog.preview()

    assert dialog.outcome is not None
    assert dialog.outcome.n_output_rows == 1
    assert dialog.outcome.n_filtered == 2
    assert "Intersection" in dialog.support_note.text()


def test_dialog_surfaces_unresolved_label_conflicts(qapp, qtbot):
    sources = _sources()
    sources[0].payload.df.loc[0, "flag"] = "molecule=apo"
    sources[1].payload.df.loc[0, "flag"] = "molecule=holo"
    dialog = ReconcileDialog(sources)
    qtbot.addWidget(dialog)

    dialog.preview()

    assert dialog.outcome is not None
    assert dialog.outcome.n_label_conflicts == 1
    assert dialog.summary_labels["n_label_conflicts"].text() == "1"
    assert "unresolved label conflicts" in dialog.preview_message.text().lower()


def test_dialog_surfaces_contradictory_human_curation(qapp, qtbot):
    dialog = ReconcileDialog(_contradictory_sources())
    qtbot.addWidget(dialog)

    dialog.preview()

    assert dialog.outcome is not None
    assert dialog.outcome.n_curation_conflicts == 1
    assert dialog.summary_labels["n_curation_conflicts"].text() == "1"
    assert "contradictory human" in dialog.preview_message.text().lower()


class _AcceptedPreview:
    def __init__(self, outcome) -> None:
        self.outcome = outcome

    def exec(self) -> int:
        return QtWidgets.QDialog.Accepted


def test_main_window_allows_one_mixed_manual_and_automatic_source(
        qapp, qtbot, tmp_path, monkeypatch):
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = win._session.add_document(_doc(tmp_path))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    source = _result(
        "detect.schmitt", [(10, 30), (11, 29), (100, 120)], doc_id=doc_id)
    source.payload.df["added"] = np.array([False, True, False], dtype=bool)
    win._session.add_result(source)
    QtWidgets.QApplication.processEvents()
    outcome = reconcile_detections((source,))
    monkeypatch.setattr(
        win, "_make_reconcile_dialog", lambda _sources: _AcceptedPreview(outcome))

    assert win._act_reconcile.isEnabled()
    win._reconcile_detections()

    stored = win._session.results.get(doc_id, 0, CONSENSUS_MODULE_ID)
    assert stored is outcome.result
    assert win._analysis_panel.current_detection() is stored


def test_main_window_stores_shows_switches_and_exports_reconciled_result(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, sources = _window(qtbot, tmp_path)
    snapshots = [res.payload.df.copy(deep=True) for res in sources]
    outcome = reconcile_detections(sources)
    monkeypatch.setattr(
        win, "_make_reconcile_dialog", lambda _sources: _AcceptedPreview(outcome))

    assert win._act_reconcile.isEnabled()
    win._reconcile_detections()
    QtWidgets.QApplication.processEvents()

    stored = win._session.results.get(doc_id, 0, CONSENSUS_MODULE_ID)
    assert stored is outcome.result
    assert win._analysis_panel.current_detection() is stored
    assert win._analysis_panel._result_selector.currentData() == CONSENSUS_MODULE_ID
    assert win._act_export_events.isEnabled()
    assert win._act_export_training.isEnabled()
    for res, before in zip(sources, snapshots):
        pd.testing.assert_frame_equal(res.payload.df, before)

    selector = win._analysis_panel._result_selector
    selector.setCurrentIndex(selector.findData("detect.schmitt"))
    QtWidgets.QApplication.processEvents()
    assert win._analysis_panel.current_detection().module_id == "detect.schmitt"
    selector.setCurrentIndex(selector.findData(CONSENSUS_MODULE_ID))
    QtWidgets.QApplication.processEvents()
    assert win._analysis_panel.current_detection().module_id == CONSENSUS_MODULE_ID


def test_reconciled_table_is_curation_only_and_refuses_geometry_changes(
        qapp, qtbot, tmp_path):
    win, _doc_id, sources = _window(qtbot, tmp_path)
    win._session.add_result(reconcile_detections(sources).result)
    QtWidgets.QApplication.processEvents()
    panel = win._analysis_panel
    before = panel.current_events_df().copy(deep=True)

    assert not panel.measure_panel._mark_missed_btn.isEnabled()
    assert not panel.merge_controls.merge_all_btn.isEnabled()
    assert not panel.merge_controls.merge_region_btn.isEnabled()
    assert not win._act_compare_baseline.isEnabled()

    refused: list[str] = []
    panel.runRefused.connect(refused.append)
    assert panel.add_current_selection_as_event() is None
    assert refused and "source detector table" in refused[-1]
    assert panel.merge_all_subevents(0.001).status == "reconciled"
    assert panel.merge_selection_subevents().status == "reconciled"
    event_id = int(before.iloc[0]["event_id"])
    assert panel.merge_event_with_next(event_id, 0.001).status == "reconciled"
    assert panel.remeasure_active_detection() == "reconciled"
    assert panel.compare_baseline_estimators() == "reconciled"
    pd.testing.assert_frame_equal(panel.current_events_df(), before)

    win._open_event_inspector()
    inspector = win._event_inspector
    assert inspector is not None
    assert not inspector._region.movable
    assert not inspector._merge_next_btn.isEnabled()
    assert not inspector._revert_btn.isEnabled()
    inspector._apply_edited_bounds(
        0, int(before.iloc[0]["start_idx"]) + 1,
        int(before.iloc[0]["end_idx"]) - 1, np.zeros(N, dtype=float))
    pd.testing.assert_frame_equal(panel.current_events_df(), before)


def test_mark_missed_action_on_reconciled_table_stays_reachable_and_refuses_out_loud(
        qapp, qtbot, tmp_path):
    win, doc_id, sources = _window(qtbot, tmp_path)
    reconciled = reconcile_detections(sources).result
    win._session.add_result(reconciled)
    panel = win._analysis_panel
    panel.measure_panel.set_active(True)
    axis = win._trace_view.x_axis_for(doc_id, 0).values
    win._trace_view.set_measure_region(float(axis[400]), float(axis[450]))
    qapp.processEvents()
    before = reconciled.payload.df.copy(deep=True)
    writes: list[object] = []
    win._session.resultsChanged.connect(lambda *args: writes.append(args))

    assert panel.current_detection() is reconciled
    assert win._act_mark_missed.isEnabled()
    win._act_mark_missed.trigger()

    assert writes == []
    pd.testing.assert_frame_equal(reconciled.payload.df, before)
    status = win.statusBar().currentMessage().lower()
    assert "source detector table" in status
    assert "reconcile again" in status


def test_reconciled_table_disables_and_defensively_refuses_trace_reconstruction(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, _sources_in = _window(qtbot, tmp_path)
    reconciled = _result(CONSENSUS_MODULE_ID, [(10, 30)], doc_id=doc_id)
    win._session.add_result(reconciled)
    QtWidgets.QApplication.processEvents()

    assert win._analysis_panel.current_detection() is reconciled
    assert not win._act_fit_levels.isEnabled()
    assert not win._act_subpop.isEnabled()
    assert "source detector table" in win._act_fit_levels.toolTip()
    assert "source detector table" in win._act_subpop.toolTip()
    # Frozen event-table analyses and exports remain valid.
    assert win._act_plots.isEnabled()
    assert win._act_export_events.isEnabled()

    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information",
        lambda _parent, title, text, *args: messages.append((title, text)),
    )
    win._fit_levels()
    win._run_subpopulations()

    assert [title for title, _ in messages] == [
        "Fit sub-state levels", "Event subpopulations"]
    assert all("source detector table" in text for _, text in messages)
    assert win._subpop_target is None


def test_stale_reconciled_table_never_offers_or_routes_detector_rerun(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, _sources_in = _window(qtbot, tmp_path)
    reconciled = AnalysisResult(
        module_id=CONSENSUS_MODULE_ID,
        module_version="1.0",
        doc_id=doc_id,
        channel=0,
        params={},
        calibration_fingerprint=(2.0, 0.0),
        baseline_fingerprint="baseline-a",
        payload=_events([(10, 30)]),
    )
    win._session.add_result(reconciled)
    QtWidgets.QApplication.processEvents()
    panel = win._analysis_panel

    assert panel._stale.kind == "calibration"
    assert panel._stale._btn.text() == "Refresh sources first"
    assert not panel._stale._btn.isEnabled()
    assert "source detector" in panel._stale.action_refusal().lower()
    assert "reconcile again" in panel._stale.action_refusal().lower()

    launches: list[str] = []
    monkeypatch.setattr(panel, "_run_detection", lambda *_: launches.append("detect"))
    monkeypatch.setattr(
        panel, "confirm_replacing_curation",
        lambda *_: launches.append("confirm") or True,
    )
    refused: list[str] = []
    panel.runRefused.connect(refused.append)
    panel._on_stale_action("calibration")
    panel._on_rerun()

    assert launches == []
    assert len(refused) == 2
    assert all("refresh the stale source results" in msg.lower() for msg in refused)


def test_baseline_stale_reconciled_table_never_routes_remeasurement(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, _sources_in = _window(qtbot, tmp_path)
    reconciled = _result(CONSENSUS_MODULE_ID, [(10, 30)], doc_id=doc_id)
    win._session.add_result(reconciled)
    QtWidgets.QApplication.processEvents()
    panel = win._analysis_panel
    monkeypatch.setattr(panel, "_baseline_staleness", lambda _res: "Baseline changed.")
    panel._judge_staleness(reconciled)

    assert panel._stale.kind == "baseline"
    assert panel._stale._btn.text() == "Refresh sources first"
    assert not panel._stale._btn.isEnabled()
    assert "reconcile again" in panel._stale._label.text().lower()

    calls: list[str] = []
    monkeypatch.setattr(
        panel, "remeasure_active_detection", lambda: calls.append("remeasure"))
    refused: list[str] = []
    panel.runRefused.connect(refused.append)
    panel._on_stale_action("baseline")

    assert calls == []
    assert refused and "refresh the stale source results" in refused[-1].lower()


def test_reconciled_snapshot_warns_when_sources_change_are_replaced_or_missing(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, sources = _window(qtbot, tmp_path)
    panel = win._analysis_panel
    # Isolate source lineage from the fixture's deliberately synthetic baseline stamp.
    monkeypatch.setattr(panel, "_baseline_staleness", lambda _res: None)
    reconciled = reconcile_detections(sources).result
    win._session.add_result(reconciled)
    QtWidgets.QApplication.processEvents()
    assert panel.current_detection() is reconciled
    assert panel._stale.isHidden()

    # In-place source curation followed by the ResultStore's ordinary refresh signal.
    sources[0].payload.df.loc[0, "accepted"] = False
    win._session.resultsChanged.emit(doc_id, sources[0].module_id)
    QtWidgets.QApplication.processEvents()
    assert panel.current_detection() is reconciled
    assert panel._stale.kind == "reconciliation_source"
    assert "changed:" in panel._stale._label.text().lower()
    assert "reconcile again" in panel._stale._label.text().lower()
    assert not panel._stale._btn.isEnabled()
    assert win._act_export_events.isEnabled()  # warning preserves the exportable snapshot

    # Replacing a source result also refreshes the still-visible snapshot instead of
    # routing it into an ordinary detector re-run.
    replacement = _result("detect.schmitt", [(50, 75)], doc_id=doc_id)
    win._session.add_result(replacement)
    QtWidgets.QApplication.processEvents()
    assert panel.current_detection() is reconciled
    assert "changed:" in panel._stale._label.text().lower()

    # A missing input is distinct and actionable. ResultStore.drop is intentionally a
    # low-level primitive, so emit the same notification a public removal path supplies.
    assert win._session.results.drop(doc_id, 0, "detect.cusum")
    win._session.resultsChanged.emit(doc_id, "detect.cusum")
    QtWidgets.QApplication.processEvents()
    assert panel.current_detection() is reconciled
    assert "missing:" in panel._stale._label.text().lower()
    assert "remains unchanged" in panel._stale._label.text().lower()


def test_replacing_reconciled_result_is_undoable(qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, sources = _window(qtbot, tmp_path)
    first = reconcile_detections(sources, config=ReconcileConfig(min_detector_support=1))
    second = reconcile_detections(sources, config=ReconcileConfig(min_detector_support=2))
    previews = iter((_AcceptedPreview(first), _AcceptedPreview(second)))
    monkeypatch.setattr(win, "_make_reconcile_dialog", lambda _sources: next(previews))
    monkeypatch.setattr(win._analysis_panel, "confirm_replacing_curation", lambda _mid: True)

    win._reconcile_detections()
    assert len(win._analysis_panel.current_events_df()) == 3
    win._reconcile_detections()
    assert len(win._analysis_panel.current_events_df()) == 1
    assert win._history.can_undo((doc_id, 0, CONSENSUS_MODULE_ID))

    win._undo_events()
    assert len(win._analysis_panel.current_events_df()) == 3
    assert win._analysis_panel.current_detection().module_id == CONSENSUS_MODULE_ID


def test_reconciled_result_survives_project_roundtrip_and_restore(qapp, qtbot, tmp_path):
    win, _doc_id, sources = _window(qtbot, tmp_path)
    reconciled = reconcile_detections(sources).result
    # Exercise the v10 fields themselves, not the legacy provenance fallback.
    reconciled.payload.provenance.pop("baseline_fingerprint", None)
    win._session.add_result(reconciled)

    project = save_project(tmp_path / "reconciled.nanoscope", win._gather_project_state())
    state = load_project(project)
    rec = next(r for r in state.results if r.module_id == CONSENSUS_MODULE_ID)
    assert "consensus_contributors" in rec.events.columns
    assert "reconciliation" in rec.provenance
    # These are AnalysisResult fields, not payload-provenance fields. In particular,
    # reconciliation provenance intentionally need not duplicate the baseline stamp.
    assert "baseline_fingerprint" not in rec.provenance
    assert rec.module_version == "1.0"
    assert rec.baseline_fingerprint == "baseline-a"

    restored = MainWindow()
    qtbot.addWidget(restored)
    live_id = restored._session.add_document(_doc(tmp_path))
    restored._create_default_layers(restored._session.document(live_id))
    restored._restore_state = state
    restored._apply_restore_for_doc(live_id)
    restored._on_selection(live_id, 0)
    back = restored._session.results.get(live_id, 0, CONSENSUS_MODULE_ID)
    assert back is not None
    assert len(back.payload.df) == 3
    assert back.module_version == "1.0"
    assert back.baseline_fingerprint == "baseline-a"
    assert restored._analysis_panel._reconciliation_source_staleness(back) is None
    assert restored._analysis_panel._result_selector.findData(CONSENSUS_MODULE_ID) >= 0


def test_replacement_guard_counts_flag_notes_and_review_only_rows():
    df = _events([(10, 20), (30, 40), (50, 60)]).df
    df["added"] = False
    df.loc[0, "flag"] = FLAG_CURATED
    df[REVIEW_COL] = pd.array([pd.NA, REVIEW_DONE, pd.NA], dtype="string")

    work = AnalysisPanel.hand_work_at_stake(df)
    assert work["n_notes"] == 1
    assert work["n_reviewed"] == 1
    assert work["n_hand_rejected"] == 0


def test_workflow_status_treats_reconciled_only_table_as_a_detection(tmp_path):
    from nanoscope.gui.workflow_status import Stage, compute_status

    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    source_a, source_b = _sources(doc_id)
    session.add_result(reconcile_detections((source_a, source_b)).result)
    status = compute_status(session, doc_id, 0)
    assert status.stage(Stage.DETECT).done
    assert status.stage(Stage.CURATE).available


def test_training_export_conflict_warning_defaults_to_cancel_before_path_dialog(
        qapp, qtbot, tmp_path, monkeypatch):
    win, _doc_id, _sources_in = _window(qtbot, tmp_path)
    df = win._analysis_panel.current_events_df()
    df.loc[df.index[0], "flag"] = "molecule=apo|holo"
    df.loc[df.index[0], REVIEW_COL] = REVIEW_DONE
    dialogs: list[tuple[list[str], str]] = []

    def fake_exec(box):
        default = box.defaultButton().text() if box.defaultButton() is not None else ""
        dialogs.append(([b.text() for b in box.buttons()], default))
        return 0

    def fake_clicked(box):
        buttons = box.buttons()
        if any("Reviewed decisions" in button.text() for button in buttons):
            return next(button for button in buttons
                        if "Reviewed decisions" in button.text())
        return next(button for button in buttons if button.text() == "Cancel")

    asked: list[bool] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", fake_clicked)
    monkeypatch.setattr(
        main_window_module, "ask_export_path",
        lambda *args, **kwargs: asked.append(True),
    )

    win._export_training()

    warning = next(
        (item for item in dialogs if "Export with conflicts" in item[0]),
        None,
    )
    assert warning is not None, dialogs
    assert warning[1] == "Cancel"
    assert asked == []


def test_training_export_with_no_reviews_defaults_to_cancel(
        qapp, qtbot, tmp_path, monkeypatch):
    win, _doc_id, _sources_in = _window(qtbot, tmp_path)
    df = win._analysis_panel.current_events_df()
    df[REVIEW_COL] = pd.array(["unreviewed"] * len(df), dtype="string")
    seen: dict[str, object] = {}

    def fake_exec(box):
        seen["default"] = box.defaultButton().text()
        seen["reviewed_enabled"] = next(
            button for button in box.buttons()
            if "Reviewed decisions" in button.text()
        ).isEnabled()
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "clickedButton",
        lambda box: next(button for button in box.buttons()
                         if button.text() == "Cancel"),
    )
    win._export_training()

    assert seen == {"default": "Cancel", "reviewed_enabled": False}


def test_resolving_reconciled_label_clears_live_conflict_and_export_warning(
        qapp, qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis.labels import LabelVocabulary

    win, _doc_id, sources = _window(qtbot, tmp_path)
    sources[0].payload.df.loc[0, "flag"] = "molecule=apo"
    sources[1].payload.df.loc[0, "flag"] = "molecule=holo"
    win._session.add_result(reconcile_detections(sources).result)
    win._open_event_inspector()
    inspector = win._event_inspector
    assert inspector is not None
    df = win._analysis_panel.current_events_df()
    conflict_row = int(np.flatnonzero(
        df[CONSENSUS_LABEL_CONFLICTS_COL].astype("string").to_numpy() != "")[0])
    inspector.show_event(int(df.iloc[conflict_row]["event_id"]))
    vocab = LabelVocabulary()
    vocab.add("molecule", "apo")
    inspector.set_vocabulary(vocab)

    inspector.apply_hotkey("1")
    QtWidgets.QApplication.processEvents()

    assert df.iloc[conflict_row][CONSENSUS_LABEL_CONFLICTS_COL] == ""
    assert str(df.iloc[conflict_row][REVIEW_COL]) == REVIEW_DONE
    button_sets: list[list[str]] = []

    def fake_exec(box):
        button_sets.append([button.text() for button in box.buttons()])
        return 0

    def fake_clicked(box):
        buttons = box.buttons()
        if any("Reviewed decisions" in button.text() for button in buttons):
            return next(button for button in buttons
                        if "Reviewed decisions" in button.text())
        return next(button for button in buttons if button.text() == "Cancel")

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", fake_clicked)
    monkeypatch.setattr(main_window_module, "ask_export_path", lambda *a, **k: None)
    win._export_training()

    assert not any("Export with conflicts" in buttons for buttons in button_sets)


def test_reconcile_commit_status_surfaces_contradictory_curation(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, _old_sources = _window(qtbot, tmp_path)
    sources = _contradictory_sources(doc_id)
    for source in sources:
        win._session.add_result(source)
    outcome = reconcile_detections(sources)
    monkeypatch.setattr(
        win, "_make_reconcile_dialog", lambda _sources: _AcceptedPreview(outcome))

    win._reconcile_detections()

    assert outcome.n_curation_conflicts == 1
    assert "contradictory accept/reject" in win.statusBar().currentMessage().lower()


def test_explicit_curation_clears_only_targeted_reconciliation_conflicts(
        qapp, qtbot, tmp_path):
    win, doc_id, _old_sources = _window(qtbot, tmp_path)
    sources = _contradictory_sources(doc_id)
    reconciled = reconcile_detections(sources).result
    win._session.add_result(reconciled)
    QtWidgets.QApplication.processEvents()
    df = reconciled.payload.df
    conflict_rows = np.flatnonzero(
        df[CONSENSUS_CURATION_CONFLICT_COL].astype("string").fillna("").to_numpy() != "")
    assert conflict_rows.size == 1
    row = int(conflict_rows[0])
    other = next(i for i in range(len(df)) if i != row)
    marker = str(df.iloc[row][CONSENSUS_CURATION_CONFLICT_COL])
    df.iloc[other, df.columns.get_loc(CONSENSUS_CURATION_CONFLICT_COL)] = marker

    # Main event table path. The rejected value is already present: the explicit
    # verdict must still resolve and checkpoint the conflict-only edit.
    event_id = int(df.iloc[row]["event_id"])
    win._analysis_panel.set_event_accepted(event_id, False)
    QtWidgets.QApplication.processEvents()
    assert df.iloc[row][CONSENSUS_CURATION_CONFLICT_COL] == ""
    assert df.iloc[other][CONSENSUS_CURATION_CONFLICT_COL] == marker

    # Event Inspector path.
    df.iloc[row, df.columns.get_loc(CONSENSUS_CURATION_CONFLICT_COL)] = marker
    win._open_event_inspector()
    assert win._event_inspector is not None
    win._event_inspector.show_event(event_id)
    win._event_inspector.set_accepted_current(False)
    QtWidgets.QApplication.processEvents()
    assert df.iloc[row][CONSENSUS_CURATION_CONFLICT_COL] == ""
    assert df.iloc[other][CONSENSUS_CURATION_CONFLICT_COL] == marker

    # Event Gallery path, still scoped to the one tile rather than clearing every
    # unresolved row in the derived table.
    df.iloc[row, df.columns.get_loc(CONSENSUS_CURATION_CONFLICT_COL)] = marker
    win._open_event_gallery()
    assert win._event_gallery is not None
    win._event_gallery._set_rows_accepted([row], False)
    QtWidgets.QApplication.processEvents()
    assert df.iloc[row][CONSENSUS_CURATION_CONFLICT_COL] == ""
    assert df.iloc[other][CONSENSUS_CURATION_CONFLICT_COL] == marker


def test_training_export_curation_conflict_warning_defaults_to_cancel(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, _old_sources = _window(qtbot, tmp_path)
    reconciled = reconcile_detections(_contradictory_sources(doc_id)).result
    win._session.add_result(reconciled)
    dialogs: list[tuple[str, list[str], str, str]] = []

    def fake_exec(box):
        default = box.defaultButton().text() if box.defaultButton() is not None else ""
        dialogs.append((
            box.windowTitle(), [b.text() for b in box.buttons()], default,
            box.informativeText(),
        ))
        return 0

    def fake_clicked(box):
        if any("Reviewed decisions" in button.text() for button in box.buttons()):
            return next(button for button in box.buttons()
                        if "All raw candidates" in button.text())
        return next(button for button in box.buttons() if button.text() == "Cancel")

    asked: list[bool] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", fake_clicked)
    monkeypatch.setattr(
        main_window_module, "ask_export_path",
        lambda *args, **kwargs: asked.append(True),
    )

    win._export_training()

    warning = next(
        (item for item in dialogs if "Export with curation conflicts" in item[1]),
        None,
    )
    assert warning is not None, dialogs
    assert "Export with curation conflicts" in warning[1]
    assert warning[2] == "Cancel"
    chooser = next(item for item in dialogs if any(
        "Reviewed decisions" in text for text in item[1]))
    assert "1 unresolved reconciliation decision excluded from Reviewed" in chooser[3]
    assert asked == []


def test_explicit_all_rows_override_exports_every_curation_conflict_row(
        qapp, qtbot, tmp_path, monkeypatch):
    win, doc_id, _old_sources = _window(qtbot, tmp_path)
    reconciled = reconcile_detections(_contradictory_sources(doc_id)).result
    win._session.add_result(reconciled)
    destination = tmp_path / "with_curation_conflicts.csv"

    def fake_exec(_box):
        return 0

    def fake_clicked(box):
        buttons = box.buttons()
        if any("Reviewed decisions" in button.text() for button in buttons):
            return next(button for button in buttons
                        if "All raw candidates" in button.text())
        return next(button for button in buttons
                    if button.text() == "Export with curation conflicts")

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", fake_clicked)
    monkeypatch.setattr(
        main_window_module, "ask_export_path", lambda *args, **kwargs: destination)

    win._export_training()

    written = pd.read_csv(destination)
    assert len(written) == len(reconciled.payload.df) == 3
    assert "Wrote 3 raw candidates" in win.statusBar().currentMessage()
