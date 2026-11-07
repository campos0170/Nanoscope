"""Explicit UMAP opt-in on stored reports, worker routing, races and durable results."""
from __future__ import annotations

import copy
from dataclasses import replace
import threading

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import subpop_cluster as SC
from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.cancel import OperationCancelled
from nanoscope.core.analysis_workspace import capture_subpopulation, restore_subpopulation
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.subpop_window import ASSIGNMENT_INSTALLED, SubpopReportWindow

from .test_subpop_views import report  # noqa: F401; one real, inexpensive report


@pytest.fixture
def original(report):
    return copy.deepcopy(report)


@pytest.fixture
def ready(qtbot, original):
    window = MainWindow()
    qtbot.addWidget(window)
    owner = window._ensure_figure_workspace()
    key = owner.create("subpopulations")
    page = owner.page(key)
    page.set_report(original, subtitle="stored synthetic report")
    page._nanoscope_workspace_source = {"source_key": "original-source"}
    page._nanoscope_workspace_accepted_event_ids = (4, 8, 16)
    owner.checkpoint(key)
    yield window, owner, key, page
    window._analysis_controller.abort()
    window._on_subpop_stopped()


@pytest.fixture
def embedding(monkeypatch):
    calls = []
    def fit(scores, *, random_state):
        calls.append((np.array(scores, copy=True), random_state, QtCore.QThread.currentThread()))
        return np.column_stack((np.arange(len(scores)), -np.arange(len(scores)))).astype(float)
    monkeypatch.setattr(SC, "umap_embedding", fit)
    return calls


def test_stored_pca_is_the_only_input_and_other_results_are_preserved(original, embedding):
    before = original.embeddings.copy(deep=True)
    original.meta["config"]["random_state"] = 17
    result = SR.add_umap_to_report(original)
    np.testing.assert_array_equal(embedding[0][0], original.clustering.scopes[SC.SCOPE_ALL].pca.scores)
    assert embedding[0][1] == 17
    assert result is not original
    assert result.meta["config"]["with_umap"] is True
    assert original.meta["config"]["with_umap"] is False
    pd.testing.assert_frame_equal(original.embeddings, before)
    pd.testing.assert_frame_equal(result.embeddings[before.columns], before)
    for attr in ("features", "gate", "clustering", "quantification", "labels", "waveforms"):
        assert getattr(result, attr) is getattr(original, attr)
    assert result.embeddings.index.equals(original.embeddings.index)
    assert result.meta["umap_source"] == "stored pooled-scope PCA"


def test_cached_coordinates_never_recompute(original, embedding):
    generated = SR.add_umap_to_report(original)
    assert SR.add_umap_to_report(generated) is generated
    assert len(embedding) == 1


def test_no_clustering_refuses_without_modifying_report(original):
    original.clustering = None
    with pytest.raises(ValueError, match="no stored PCA scores"):
        SR.add_umap_to_report(original)
    assert "UMAP" not in original.embedding_names()


@pytest.mark.parametrize("failure", [ImportError("missing dependency"), ValueError("solver failed")])
def test_solver_failure_preserves_report(original, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(SC, "umap_embedding", fail)
    before = original.embeddings.copy(deep=True)
    with pytest.raises(ValueError, match="NO UMAP"):
        SR.add_umap_to_report(original)
    pd.testing.assert_frame_equal(original.embeddings, before)
    assert original.meta["config"]["with_umap"] is False


def test_cancel_is_not_a_completed_embedding(original, monkeypatch):
    cancel = threading.Event()
    def fit(scores, **kwargs):
        cancel.set()
        return np.zeros((len(scores), 2))
    monkeypatch.setattr(SC, "umap_embedding", fit)
    with pytest.raises(OperationCancelled):
        SR.add_umap_to_report(original, cancel=cancel)
    assert "UMAP" not in original.embedding_names()


def test_nonfinite_coordinates_are_rejected(original, monkeypatch):
    monkeypatch.setattr(SC, "umap_embedding", lambda scores, **kw: np.full((len(scores), 2), np.nan))
    with pytest.raises(ValueError, match="invalid coordinates"):
        SR.add_umap_to_report(original)


def test_button_is_hosted_in_view_and_requires_a_report(qtbot, original):
    page = SubpopReportWindow()
    qtbot.addWidget(page)
    button = page._generate_umap_btn
    assert button.text() == "Generate UMAP" and not button.isEnabled()
    assert button in next(group for group in page.inspector_adapter().groups if group.key == "view").widgets
    heard = []
    page.umapRequested.connect(lambda: heard.append(True))
    page.set_report(original)
    assert button.isEnabled()
    button.click()
    assert heard == [True]
    page.set_umap_busy(True)
    assert not button.isEnabled() and page._pending().busy
    page.set_umap_busy(False)
    assert button.isEnabled() and not page._pending().busy


def test_click_uses_shared_worker_and_selected_stored_report(ready, embedding, qtbot, monkeypatch):
    window, owner, key, page = ready
    # There is no loaded recording/detection. Saved reports must work independently.
    assert window._analysis_panel.current_detection() is None
    old = page.report()
    page.set_assignment_state(ASSIGNMENT_INSTALLED)
    page.set_curation_note("accepted set changed; re-run to update")
    colour = page.embedding_view().colour_mode()
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=10000):
        page._generate_umap_btn.click()
        assert not page._generate_umap_btn.isEnabled()
        assert page.report() is old
    qtbot.waitUntil(lambda: window._umap_figure_request is None)
    result = page.report()
    assert result is not old and result.meta["config"]["with_umap"]
    assert embedding[0][2] is not QtWidgets.QApplication.instance().thread()
    assert page.embedding_view().embedding() == "UMAP"
    assert page.embedding_view().colour_mode() == colour
    assert page.assignment_state()[0] == ASSIGNMENT_INSTALLED
    assert page.curation_note() == "accepted set changed; re-run to update"
    assert page._nanoscope_workspace_source == {"source_key": "original-source"}
    assert page._nanoscope_workspace_accepted_event_ids == (4, 8, 16)
    assert not page._generate_umap_btn.isEnabled()  # stored cache
    assert not owner._entries[key].busy
    assert not page._pending().busy
    assert page._pending().count == 1  # embedding generation did not clear stale curation


def _capture_job(window, monkeypatch):
    jobs = []
    def run(fn, label, wants_progress):
        assert label == "UMAP" and wants_progress
        jobs.append(fn)
        window._on_analysis_started(label)
        return True
    monkeypatch.setattr(window._analysis_controller, "run_callable", run)
    return jobs


def test_refused_start_does_not_leave_busy_state(ready, monkeypatch):
    window, owner, key, page = ready
    original = page.report()
    monkeypatch.setattr(window._analysis_controller, "run_callable", lambda *a, **kw: False)
    page._generate_umap_btn.click()
    assert page.report() is original and window._umap_figure_request is None
    assert page._generate_umap_btn.isEnabled() and not owner._entries[key].busy


@pytest.mark.parametrize("terminal", ["cancelled", "failed", "replacement"])
def test_terminal_paths_clear_claim_and_preserve_report(ready, monkeypatch, terminal):
    window, owner, key, page = ready
    jobs = _capture_job(window, monkeypatch)
    old = page.report()
    page._generate_umap_btn.click()
    assert jobs and window._umap_figure_request is not None
    if terminal == "replacement":
        window._on_analysis_started("other analysis")
    else:
        window._on_subpop_stopped()
    assert window._umap_figure_request is None and page.report() is old
    assert page._generate_umap_btn.isEnabled() and not owner._entries[key].busy


def test_late_result_cannot_replace_a_new_report(ready, embedding, monkeypatch):
    window, owner, key, page = ready
    jobs = _capture_job(window, monkeypatch)
    page._generate_umap_btn.click()
    new = replace(page.report(), meta={**page.report().meta, "recording": "replacement"})
    page.set_report(new)
    payload = jobs[0](lambda *args: None)
    window._on_umap_finished(payload)
    assert page.report() is new and "UMAP" not in new.embedding_names()
    assert window._umap_figure_request is None and not owner._entries[key].busy


def test_foreign_result_cannot_consume_pending_umap(ready, embedding, monkeypatch):
    window, owner, key, page = ready
    jobs = _capture_job(window, monkeypatch)
    original = page.report()
    page._generate_umap_btn.click()
    window._on_umap_finished(original)
    assert window._umap_figure_request is not None and page.report() is original
    window._on_umap_finished(jobs[0](lambda *args: None))
    assert "UMAP" in page.report().embedding_names()


def test_closed_figure_rejects_late_result(ready, embedding, monkeypatch):
    window, owner, key, page = ready
    jobs = _capture_job(window, monkeypatch)
    page._generate_umap_btn.click()
    # Busy figures cannot be deleted; cancellation releases that guard.
    assert not owner.delete(key)
    window._on_subpop_stopped()
    assert owner.delete(key)
    window._on_umap_finished(jobs[0](lambda *args: None))
    assert owner.get(key) is None and window._umap_figure_request is None


def test_saved_report_roundtrip_keeps_umap_and_config(qtbot, original, embedding):
    first, second = SubpopReportWindow(), SubpopReportWindow()
    qtbot.addWidget(first); qtbot.addWidget(second)
    first.set_report(original)
    first.install_umap_report(SR.add_umap_to_report(original))
    restored = restore_subpopulation(second, capture_subpopulation(first))
    assert restored.restored and not restored.warnings
    pd.testing.assert_frame_equal(first.report().embeddings, second.report().embeddings)
    assert second.report().meta["config"]["with_umap"] is True
    assert second.embedding_view().embedding() == "UMAP"
    assert not second._generate_umap_btn.isEnabled()


def test_real_umap_backend_adds_finite_coordinates(original):
    result = SR.add_umap_to_report(original)
    assert result.embeddings[["umap_x", "umap_y"]].shape == (len(original.embeddings), 2)
    assert np.isfinite(result.embeddings[["umap_x", "umap_y"]].to_numpy()).all()
    pd.testing.assert_frame_equal(result.embeddings[["tsne_x", "tsne_y"]], original.embeddings)


def test_real_stop_control_clears_umap_state_and_allows_retry(ready, embedding, monkeypatch):
    window, owner, key, page = ready
    jobs = _capture_job(window, monkeypatch)
    old = page.report()
    page._generate_umap_btn.click()
    window._cancel_load()
    assert window._umap_figure_request is None
    assert page._generate_umap_btn.isEnabled() and not owner._entries[key].busy
    window._on_umap_finished(jobs[0](lambda *args: None))
    assert page.report() is old
    page._generate_umap_btn.click()
    assert len(jobs) == 2 and window._umap_figure_request is not None


def test_project_clear_releases_umap_claim_for_next_project(ready, embedding, monkeypatch):
    window, owner, key, page = ready
    jobs = _capture_job(window, monkeypatch)
    old = page.report()
    page._generate_umap_btn.click()
    window._clear_session()
    assert window._umap_figure_request is None
    replacement_key = owner.create("subpopulations")
    replacement_page = owner.page(replacement_key)
    replacement_page.set_report(old)
    window._on_umap_finished(jobs[0](lambda *args: None))
    assert replacement_page.report() is old
    replacement_page._generate_umap_btn.click()
    assert len(jobs) == 2 and window._umap_figure_request is not None


def test_report_without_pca_disables_generate(qtbot, original):
    page = SubpopReportWindow()
    qtbot.addWidget(page)
    page.set_report(replace(original, clustering=None))
    assert not page._generate_umap_btn.isEnabled()
    assert "stored PCA scores" in page._generate_umap_btn.toolTip()


def test_read_only_workspace_does_not_start_or_mutate(ready, monkeypatch):
    window, owner, key, page = ready
    jobs = _capture_job(window, monkeypatch)
    owner._read_only = True
    old = page.report()
    page._generate_umap_btn.click()
    assert not jobs and page.report() is old
    assert window._umap_figure_request is None and not owner._entries[key].busy
