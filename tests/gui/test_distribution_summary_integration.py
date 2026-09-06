"""The optional summary page stays lazy, project-owned, and scientifically isolated."""
from copy import deepcopy

import pandas as pd
import pytest

from nanoscope.core.analysis_workspace import DISTRIBUTION_SUMMARIES_ARTIFACT_ID as ARTIFACT
from nanoscope.core.project import ProjectState, load_project, save_project
from nanoscope.core.sources import PlotSource
from nanoscope.core.figure_tables import decode_figure_table
from nanoscope.gui.analysis_explorer import PAGE_DISTRIBUTION_SUMMARIES as PAGE
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui import distribution_summary_panel as summaries


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
    assert window._analysis_explorer.current_page_id() == PAGE
    assert not window._workspace_has_artifact(ARTIFACT)  # Opening is not a checkpoint.
    page._workflow.primary_actions[0].click()
    assert calls == ["ecdf"]
    assert page.has_data() and window._workspace_has_artifact(ARTIFACT)
    assert page._snapshot["series"][0]["n"] == 4
    window._activate_analysis_page(PAGE)
    window._open_distribution_summaries()
    assert calls == ["ecdf"] and not page._stale


def test_library_capture_freezes_summary_and_underlying_observations(window, monkeypatch, qtbot):
    window._open_distribution_summaries()
    window._generate_distribution_summary()
    page = window._distribution_summary_window
    qtbot.wait(20)
    values = page._plot_data()
    original = window._test_summary_source.df.copy(deep=True)
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Recomputed summary"))
    page.set_appearance_state({"title": "Paper ECDF", "font_size": 14})
    entry = window._save_analysis_figure(PAGE, "ECDF snapshot")
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
    window._generate_distribution_summary()
    before = window._distribution_summary_window._plot_data()
    path = tmp_path / "distributions.nanoscope"
    save_project(path, ProjectState(workspace=window._capture_analytical_workspace()))
    incoming = load_project(path).workspace
    artifact = next(entry for entry in incoming.artifacts if entry.artifact_id == ARTIFACT)
    expected = deepcopy(artifact.payload)
    other = MainWindow(recovery_enabled=False)
    qtbot.addWidget(other)
    monkeypatch.setattr(other, "_plots_sources", lambda: [])
    other._analytical_workspace = incoming
    other._workspace_pending.add(ARTIFACT)
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Recomputed on restore"))
    other._open_distribution_summaries()
    assert ARTIFACT in other._workspace_frozen
    assert ARTIFACT not in other._workspace_pending
    assert other._workspace_artifact(ARTIFACT).payload == expected
    pd.testing.assert_frame_equal(other._distribution_summary_window._plot_data(), before)
    assert other._analysis_explorer_status(PAGE, other._distribution_summary_window) == ("SAVED SNAPSHOT", "snapshot")
    other._reset_analytical_workspace()
    assert other._distribution_summary_window is None
    assert not other._workspace_has_artifact(ARTIFACT)
    other._dirty = False


def test_source_mutation_marks_stale_without_recomputation(window, monkeypatch):
    window._open_distribution_summaries()
    window._generate_distribution_summary()
    page = window._distribution_summary_window
    before = page._plot_data()
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Passive recomputation"))
    window._test_summary_source.df.loc[0, "accepted"] = False
    window._offer_distribution_sources()
    assert page._stale
    assert window._analysis_explorer_status(PAGE, page) == ("STALE", "stale")
    pd.testing.assert_frame_equal(page._plot_data(), before)


def test_empty_explicit_refresh_reports_error_without_qt_exception(window, monkeypatch):
    from PySide6 import QtWidgets
    messages = []
    monkeypatch.setattr(window, "_plots_sources", lambda: [])
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _p, _t, text: messages.append(text))
    window._open_distribution_summaries()
    window._analysis_explorer.refresh_current()
    assert messages == ["Select at least one event source."]
    assert not window._distribution_summary_window.has_data()


def test_failed_generation_does_not_relabel_frozen_result_current(window, monkeypatch):
    from PySide6 import QtWidgets
    window._open_distribution_summaries()
    window._generate_distribution_summary()
    page = window._distribution_summary_window
    before = page._plot_data()
    window._workspace_frozen.add(ARTIFACT)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: None)
    def fail(*args, **kwargs):
        raise ValueError("Cannot generate this input")
    monkeypatch.setattr(summaries, "summarize_distribution", fail)
    page._workflow.primary_actions[0].click()
    assert page._stale
    assert window._analysis_explorer_status(PAGE, page) == ("STALE", "stale")
    assert window._analysis_explorer._status_label.text() == "STALE"
    pd.testing.assert_frame_equal(page._plot_data(), before)


@pytest.mark.parametrize("kind", ["ecdf", "survival", "box", "violin"])
def test_each_summary_kind_captures_and_exports_without_recalculation(window, monkeypatch, qtbot, tmp_path, kind):
    window._open_distribution_summaries()
    page = window._distribution_summary_window
    page._view_combo.setCurrentIndex(page._view_combo.findData(kind))
    window._generate_distribution_summary()
    qtbot.wait(20)
    assert page.figure_applied_config()["kind"] == kind
    before = page._plot_data()
    assert not before.empty
    monkeypatch.setattr(summaries, "summarize_distribution", lambda *a, **k: pytest.fail("Export recalculated statistics"))
    _analysis, snapshot, _source, _live, _status = window._capture_figure(PAGE)
    pd.testing.assert_frame_equal(snapshot.dataframe(), before)
    assert snapshot.png_bytes.startswith(b"\x89PNG")
    assert page.save_image(tmp_path / f"{kind}.svg").is_file()
    assert page.save_csv(tmp_path / f"{kind}.csv").is_file()
