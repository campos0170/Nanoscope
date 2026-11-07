"""The optional summary page stays lazy, project-owned, and scientifically isolated."""
from copy import deepcopy

import pandas as pd
import pytest

from nanoscope.core.figure_codecs import record_from_artifact, encode_legacy
from nanoscope.core.project import ProjectState, load_project, save_project
from nanoscope.core.sources import PlotSource
from nanoscope.core.figure_tables import decode_figure_table
from nanoscope.gui.analysis_explorer import PAGE_DISTRIBUTION_SUMMARIES as PAGE
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui import distribution_summary_panel as summaries


def _update(window):
    """Run the visible managed-figure verb, including its result/error bookkeeping."""
    button = window._analysis_explorer.toolbar().update_button
    assert button.isVisible() and button.isEnabled()
    button.click()


@pytest.fixture
def window(qtbot, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    source = PlotSource("manual-measurements", "Measured events", pd.DataFrame({
        "dwell_s": [.001, .002, .002, .005, .008],
        "depth_pA": [10., 20., 30., 40., 50.], "accepted": [True, True, True, False, True]}),
        kind="manual", doc_key="manual-measurements", current=True)
    monkeypatch.setattr(window, "_plots_sources", lambda: [source])
    window._test_summary_source = source
    yield window
    window._dirty = False
    if window._figure_library is not None:
        window._figure_library.close()


def test_open_is_lazy_and_generate_is_explicit(window, monkeypatch):
    calls = []
    real = summaries.summarize_distribution
    monkeypatch.setattr(summaries, "summarize_distribution",
                        lambda *args, **kwargs: (calls.append(kwargs["kind"]), real(*args, **kwargs))[1])
    assert window._distribution_summary_window is None
    assert window._plots_window is None
    window._act_distribution_summaries.trigger()
    page = window._distribution_summary_window
    assert not calls and not page.has_data()
    assert window._plots_window is None
    owner = window._figure_workspace
    key = owner.current_id()
    assert window._analysis_explorer.current_page_id() == key
    assert owner.get(key).spec.kind == PAGE
    assert owner.get(key).spec.result.status == "empty"
    _update(window)
    assert calls == ["ecdf"]
    assert page.has_data() and owner.get(key).spec.result.status == "current"
    assert page._snapshot["series"][0]["n"] == 4
    window._activate_analysis_page(key)
    window._open_distribution_summaries()
    assert calls == ["ecdf"] and not page._stale
    assert owner.current_id() == key and len(owner.records()) == 1


def test_library_capture_freezes_summary_and_underlying_observations(window, monkeypatch, qtbot):
    window._open_distribution_summaries()
    _update(window)
    page = window._distribution_summary_window
    qtbot.wait(20)
    values = page._plot_data()
    original = window._test_summary_source.df.copy(deep=True)
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Recomputed summary"))
    page.set_appearance_state({"title": "Paper ECDF", "font_size": 14})
    entry = window._save_analysis_figure(window._figure_workspace.current_id(), "ECDF snapshot")
    assert entry is not None
    frozen = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
    pd.testing.assert_frame_equal(frozen.dataframe(), values)
    assert frozen.config()["analysis_kind"] == "nanoscope.distribution-summaries"
    assert frozen.config()["applied_settings"]["kind"] == "ecdf"
    table_blob = entry.payload["analysis"]["payload"]["table_resources"]["source_0"]
    assert len(decode_figure_table(entry.blobs["analysis:" + table_blob])) == 4
    pd.testing.assert_frame_equal(window._test_summary_source.df, original)


def test_pending_snapshot_survives_source_binding_before_lazy_restore(window, monkeypatch, qtbot, tmp_path):
    window._open_distribution_summaries()
    _update(window)
    before = window._distribution_summary_window._plot_data()
    path = tmp_path / "distributions.nanoscope"
    save_project(path, ProjectState(workspace=window._capture_analytical_workspace()))
    incoming = load_project(path).workspace
    key = window._figure_workspace.current_id()
    artifact_id = window._figure_workspace.get(key).spec.artifact_id
    artifact = next(entry for entry in incoming.artifacts if entry.artifact_id == artifact_id)
    saved = record_from_artifact(artifact)
    expected = deepcopy(encode_legacy(saved.spec, saved.resources).payload)
    other = MainWindow(recovery_enabled=False)
    qtbot.addWidget(other)
    monkeypatch.setattr(other, "_plots_sources", lambda: [])
    other._analytical_workspace = incoming
    owner = other._ensure_figure_workspace()
    assert not owner.resident_ids() and owner.get(key).spec.result.status == "frozen"
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Recomputed on restore"))
    other._open_distribution_summaries()
    assert owner.current_id() == key and owner.get(key).spec.result.status == "frozen"
    restored = owner.get(key)
    assert encode_legacy(restored.spec, restored.resources).payload == expected
    pd.testing.assert_frame_equal(other._distribution_summary_window._plot_data(), before)
    assert other._analysis_explorer_status(key, owner.page(key)) == ("SAVED RESULT", "snapshot")
    other._reset_analytical_workspace()
    assert other._distribution_summary_window is None
    assert not owner.records() and other._analysis_explorer.page(key) is None
    other._dirty = False


def test_source_mutation_marks_stale_without_recomputation(window, monkeypatch):
    window._open_distribution_summaries()
    _update(window)
    page = window._distribution_summary_window
    before = page._plot_data()
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Passive recomputation"))
    window._test_summary_source.df.loc[0, "accepted"] = False
    window._offer_distribution_sources()
    assert page._stale
    assert window._analysis_explorer_status(window._figure_workspace.current_id(), page) == ("STALE", "stale")
    pd.testing.assert_frame_equal(page._plot_data(), before)


def test_empty_explicit_update_reports_error_without_qt_exception(window, monkeypatch):
    from PySide6 import QtWidgets
    messages = []
    monkeypatch.setattr(window, "_plots_sources", lambda: [])
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _p, _t, text: messages.append(text))
    window._open_distribution_summaries()
    owner = window._figure_workspace
    key = owner.current_id()
    _update(window)
    assert messages == ["Select at least one event source."]
    assert not window._distribution_summary_window.has_data()
    assert owner.get(key).spec.result.status == "empty"


@pytest.mark.parametrize("change_draft", [False, True])
def test_failed_generation_does_not_relabel_frozen_result_current(window, monkeypatch, qtbot, tmp_path, change_draft):
    from PySide6 import QtWidgets
    window._open_distribution_summaries()
    _update(window)
    before = window._distribution_summary_window._plot_data()
    path = tmp_path / "failed-update.nanoscope"
    save_project(path, ProjectState(workspace=window._capture_analytical_workspace()))
    key = window._figure_workspace.current_id()
    other = MainWindow(recovery_enabled=False)
    qtbot.addWidget(other)
    monkeypatch.setattr(other, "_plots_sources", lambda: [window._test_summary_source])
    other._analytical_workspace = load_project(path).workspace
    owner = other._ensure_figure_workspace()
    other._open_distribution_summaries()
    page = owner.page(key)
    assert owner.get(key).spec.result.status == "frozen"
    pd.testing.assert_frame_equal(page._plot_data(), before)
    if change_draft:
        # The failed run must retain the pending draft and its original plotted values.
        page._metric_combo.setCurrentIndex(page._metric_combo.findData("depth_pA"))
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _p, _t, text: messages.append(text))
    def fail(*args, **kwargs):
        raise ValueError("Cannot generate this input")
    monkeypatch.setattr(summaries, "summarize_distribution", fail)
    _update(other)
    assert messages == ["Cannot generate this input"]
    assert page._stale is change_draft
    assert owner.get(key).spec.result.status == ("stale" if change_draft else "frozen")
    expected_status = ("STALE", "stale") if change_draft else ("SAVED RESULT", "snapshot")
    assert other._analysis_explorer_status(key, page) == expected_status
    assert other._analysis_explorer._status_label.text() == expected_status[0]
    pd.testing.assert_frame_equal(page._plot_data(), before)
    other._dirty = False


@pytest.mark.parametrize("kind", ["ecdf", "survival", "box", "violin"])
def test_each_summary_kind_captures_and_exports_without_recalculation(window, monkeypatch, qtbot, tmp_path, kind):
    window._open_distribution_summaries()
    page = window._distribution_summary_window
    page._view_combo.setCurrentIndex(page._view_combo.findData(kind))
    _update(window)
    qtbot.wait(20)
    assert page.figure_applied_config()["kind"] == kind
    before = page._plot_data()
    assert not before.empty
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Export recalculated statistics"))
    _analysis, snapshot, _source, _live, _status = window._capture_figure(window._figure_workspace.current_id())
    pd.testing.assert_frame_equal(snapshot.dataframe(), before)
    assert snapshot.png_bytes.startswith(b"\x89PNG")
    assert page.save_image(tmp_path / f"{kind}.svg").is_file()
    assert page.save_csv(tmp_path / f"{kind}.csv").is_file()
