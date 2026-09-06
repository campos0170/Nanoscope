"""The comparison page is lazy, project-owned, and isolated from scientific inputs."""
from copy import deepcopy

import pandas as pd
import pytest

from nanoscope.core.analysis_workspace import COMPARISON_FIGURES_ARTIFACT_ID as ARTIFACT
from nanoscope.core.project import ProjectState, load_project, save_project
from nanoscope.core.sources import PlotSource
from nanoscope.gui.analysis_explorer import PAGE_COMPARISON_FIGURES as PAGE
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow


@pytest.fixture
def window(qtbot, monkeypatch):
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    sources = [PlotSource(f"manual-{i}", f"Recording {i + 1}", pd.DataFrame({
        "dwell_s": [.001, .002, .002, .005, .008],
        "depth_pA": [10. + i, 20., 30., 40., 50.],
        "ecd_pA_s": [.01, .04, .06, .2, .4], "snr": [5., 10., 15., 20., 25.],
        "accepted": [True, True, True, False, True],
    }), kind="manual", doc_key=f"manual-{i}", current=(i == 0)) for i in range(2)]
    monkeypatch.setattr(window, "_plots_sources", lambda: sources)
    window._test_comparison_sources = sources
    yield window
    window._dirty = False
    if window._figure_library is not None:
        window._figure_library.close()


def test_menu_open_is_lazy_and_generation_is_explicit(window, monkeypatch):
    calls = []
    real = ComparisonFiguresWindow.analyze
    def counted(self):
        calls.append(self)
        return real(self)
    monkeypatch.setattr(ComparisonFiguresWindow, "analyze", counted)
    assert window._comparison_figures_window is None
    assert window._plots_window is None
    assert window._act_comparison_figures.isEnabled()
    window._act_comparison_figures.trigger()
    page = window._comparison_figures_window
    assert not calls and not page.has_data()
    assert window._plots_window is None
    assert window._analysis_explorer.current_page_id() == PAGE
    assert not window._workspace_has_artifact(ARTIFACT)
    window._generate_comparison_figures()
    assert calls == [page]
    assert page.has_data() and window._workspace_has_artifact(ARTIFACT)
    window._activate_analysis_page(PAGE)
    window._open_comparison_figures()
    assert calls == [page] and not page._stale


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_library_capture_copies_exact_observations_without_regeneration(window, monkeypatch, qtbot, kind):
    window._open_comparison_figures()
    page = window._comparison_figures_window
    page._view_combo.setCurrentIndex(page._view_combo.findData(kind))
    window._generate_comparison_figures()
    qtbot.wait(20)
    values = page._plot_data()
    originals = [source.df.copy(deep=True) for source in window._test_comparison_sources]
    monkeypatch.setattr(page, "analyze", lambda: pytest.fail("Figure capture regenerated observations"))
    entry = window._save_analysis_figure(PAGE, "Comparison snapshot")
    assert entry is not None
    frozen = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
    pd.testing.assert_frame_equal(frozen.dataframe(), values)
    assert frozen.config()["analysis_kind"] == "nanoscope.comparison-figures"
    assert frozen.config()["applied_settings"] == page.figure_applied_config()
    assert frozen.png_bytes.startswith(b"\x89PNG")
    assert not any(name == "rendering/cartesian.json" for name, _ in frozen.additional_resources)
    for source, original in zip(window._test_comparison_sources, originals):
        pd.testing.assert_frame_equal(source.df, original, check_exact=True)


def test_pending_project_restores_with_no_sources_and_no_generation(window, monkeypatch, qtbot, tmp_path):
    window._open_comparison_figures()
    window._generate_comparison_figures()
    before = window._comparison_figures_window._plot_data()
    path = tmp_path / "comparisons.nanoscope"
    save_project(path, ProjectState(workspace=window._capture_analytical_workspace()))
    incoming = load_project(path).workspace
    artifact = next(a for a in incoming.artifacts if a.artifact_id == ARTIFACT)
    expected = deepcopy(artifact.payload)
    other = MainWindow(recovery_enabled=False)
    qtbot.addWidget(other)
    monkeypatch.setattr(other, "_plots_sources", lambda: [])
    other._analytical_workspace = incoming
    other._workspace_pending.add(ARTIFACT)
    monkeypatch.setattr(ComparisonFiguresWindow, "analyze",
                        lambda self: pytest.fail("Comparison regenerated on project reopen"))
    other._open_comparison_figures()
    assert ARTIFACT in other._workspace_frozen and ARTIFACT not in other._workspace_pending
    assert other._workspace_artifact(ARTIFACT).payload == expected
    pd.testing.assert_frame_equal(other._comparison_figures_window._plot_data(), before)
    assert other._analysis_explorer_status(PAGE, other._comparison_figures_window) == ("SAVED SNAPSHOT", "snapshot")
    other._analysis_explorer.close()
    other._open_comparison_figures()
    pd.testing.assert_frame_equal(other._comparison_figures_window._plot_data(), before)
    other._reset_analytical_workspace()
    assert other._comparison_figures_window is None
    assert not other._workspace_has_artifact(ARTIFACT)
    assert other._analysis_explorer.page(PAGE) is None
    other._dirty = False


def test_source_mutation_marks_stale_without_regeneration(window, monkeypatch):
    window._open_comparison_figures()
    window._generate_comparison_figures()
    page = window._comparison_figures_window
    before = page._plot_data()
    monkeypatch.setattr(page, "analyze", lambda: pytest.fail("Passive regeneration"))
    window._test_comparison_sources[0].df.loc[0, "accepted"] = False
    window._offer_comparison_sources()
    assert page._stale
    assert window._analysis_explorer_status(PAGE, page) == ("STALE", "stale")
    pd.testing.assert_frame_equal(page._plot_data(), before)


def test_failed_explicit_refresh_keeps_old_snapshot_stale(window, monkeypatch):
    from PySide6 import QtWidgets
    window._open_comparison_figures()
    window._generate_comparison_figures()
    page = window._comparison_figures_window
    before = page._plot_data()
    window._workspace_frozen.add(ARTIFACT)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _p, _t, text: warnings.append(text))
    def fail():
        raise ValueError("Cannot generate this input")
    monkeypatch.setattr(page, "analyze", fail)
    window._analysis_explorer.refresh_current()
    assert warnings == ["Cannot generate this input"]
    assert page._stale and ARTIFACT not in window._workspace_frozen
    assert window._analysis_explorer_status(PAGE, page) == ("STALE", "stale")
    pd.testing.assert_frame_equal(page._plot_data(), before)


def test_empty_refresh_reports_error_instead_of_escaping_qt(window, monkeypatch):
    from PySide6 import QtWidgets
    warnings = []
    monkeypatch.setattr(window, "_plots_sources", lambda: [])
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda _p, _t, text: warnings.append(text))
    window._open_comparison_figures()
    window._analysis_explorer.refresh_current()
    assert warnings and not window._comparison_figures_window.has_data()
