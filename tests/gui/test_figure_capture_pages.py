"""The shared library capture adapter covers every analytical page without reruns."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.analysis_workspace import (
    EVENT_PLOTS_ARTIFACT_ID, PSD_ARTIFACT_ID, IV_ARTIFACT_ID,
    VOLTAGE_DEPENDENCE_ARTIFACT_ID, SUBPOPULATION_ARTIFACT_ID,
)
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, save_project, load_project
from nanoscope.core.figure_library import make_figure, figures
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.main_window import MainWindow
from .test_shared_plot_workflow import page, report  # noqa: F401 -- shared scientific fixtures


def test_every_page_captures_exact_tables_and_project_resources(page, monkeypatch, tmp_path):
    kind, panel, plots = page
    artifact_id = {"events": EVENT_PLOTS_ARTIFACT_ID, "noise": PSD_ARTIFACT_ID,
                   "iv": IV_ARTIFACT_ID, "voltage": VOLTAGE_DEPENDENCE_ARTIFACT_ID,
                   "subpopulations": SUBPOPULATION_ARTIFACT_ID}[kind]
    owner = SimpleNamespace(
        _session=getattr(panel, "_session", None),
        _figure_page=lambda _page_id: (artifact_id, panel),
        _workspace_kwarg=MainWindow._workspace_kwarg,
        _analysis_explorer_status=lambda _page_id, _panel: ("CURRENT", "current"))
    before = panel._plot_data().copy(deep=True)
    ranges = [np.asarray(plot.viewRange()).copy() for plot in plots]
    def forbid(*_args, **_kwargs):
        pytest.fail("Figure capture recalculated a numerical result")
    for name in ("_redraw", "_recompute", "refresh", "_redraw_density", "_redraw_ensemble"):
        if hasattr(panel, name):
            monkeypatch.setattr(panel, name, forbid)
    analysis, snapshot, source, live, status = MainWindow._capture_figure(owner, kind)
    pd.testing.assert_frame_equal(snapshot.dataframe(), before, check_exact=True)
    pd.testing.assert_frame_equal(panel._plot_data(), before, check_exact=True)
    for plot, expected in zip(plots, ranges):
        np.testing.assert_array_equal(plot.viewRange(), expected)
    assert snapshot.includes_residual is False  # a Subpop waveform is NOT a residual
    if kind == "subpopulations":
        assert snapshot.provenance()["components"] == ["embedding", "waveform overlay"]
        assert snapshot.height > plots[0].sceneBoundingRect().height()
    metadata, blobs = snapshot.to_storage()
    entry = make_figure(name=kind, page_id=kind, source_label=source, analysis=analysis,
                        snapshot_metadata=metadata, snapshot_blobs=blobs, live_state=live,
                        capture_status=status)
    path = save_project(tmp_path / f"{kind}.nanoscope", ProjectState(
        workspace=AnalyticalWorkspace(artifacts=[entry])))
    back = figures(load_project(path).workspace)[0]
    loaded = FigureSnapshot.from_storage(back.payload["snapshot"], back.blobs)
    assert loaded == snapshot


def test_interactive_panel_forwards_approved_frozen_picture(page, monkeypatch, tmp_path):
    from dataclasses import replace
    import importlib
    from nanoscope.gui.figure_bundle import capture_figure_snapshot
    from nanoscope.gui.plot_export_options import resolve_export_options
    kind, panel, _plots = page
    if kind == "subpopulations":
        return  # The library owns this report's composed-image export.
    snapshot = capture_figure_snapshot(panel.figure_export_target(), data=panel._plot_data())
    options = replace(resolve_export_options((snapshot.width, snapshot.height)), frozen_snapshot=snapshot)
    target = tmp_path / "frozen.png"
    module = importlib.import_module(type(panel).__module__)
    monkeypatch.setattr(module, "choose_live_plot_export", lambda *a, **k: (target, options))
    forwarded = {}
    def write(_target, _item, **kwargs):
        forwarded.update(kwargs)
        return []
    monkeypatch.setattr(module, "write_live_plot_image", write)
    assert panel.save_image() == target
    assert forwarded["frozen_snapshot"] is snapshot
