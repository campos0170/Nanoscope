"""A clean distribution export retains diagnostics and exact scientific state."""
from copy import deepcopy
from types import SimpleNamespace
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.analysis_workspace import PLOTS_ARTIFACT_ID
from nanoscope.gui.figure_bundle import write_figure_bundle
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.plot_export import write_live_plot_image
from .test_s7_plots_rejected_fit_on_figure import rejected  # noqa: F401


def test_warning_free_capture_export_and_bundle_retain_fit_and_details(
        rejected, qtbot, monkeypatch, tmp_path):
    window = rejected
    window._fit_values_check.setChecked(True)
    window._residual_check.setChecked(True)
    window.set_title("Apo-ferritin dwell times")
    window.show()
    qtbot.wait(30)
    window._plot.setRange(xRange=(0.005, 0.035), yRange=(0, 300), padding=0)
    qtbot.wait(10)
    frame = window._plot_data().copy(deep=True)
    fit = window._last_fit
    params, meta = deepcopy(fit.params), deepcopy(fit.meta)
    curves = [(item, item.xData.copy(), item.yData.copy())
              for item in (*window._hist_items, *window._fit_items,
                           *window._resid.listDataItems())]
    ranges = np.asarray(window._plot.viewRange()).copy()
    details = window.details_text()
    assert "KS REJECTED" in details
    assert "REJECTED" not in window._fit_values_item.text

    def forbid(*args, **kwargs):
        pytest.fail("Capturing a warning-free figure must not rerun scientific work")
    for name in ("_redraw", "_overlay_fit", "_goodness_note", "_drawn_frames"):
        monkeypatch.setattr(window, name, forbid)
    owner = SimpleNamespace(
        _session=window._session,
        _figure_page=lambda _page_id: (PLOTS_ARTIFACT_ID, window),
        _workspace_kwarg=MainWindow._workspace_kwarg,
        _analysis_explorer_status=lambda *_args: ("CURRENT", "current"))
    _, snapshot, *_ = MainWindow._capture_figure(owner, "plots", include_residual=True)
    assert snapshot.includes_residual
    assert "KS REJECTED" not in snapshot.svg_bytes.decode()
    assert "Apo-ferritin dwell times" in snapshot.svg_bytes.decode()
    assert snapshot.provenance()["details"] == details
    pd.testing.assert_frame_equal(snapshot.dataframe(), frame, check_exact=True)
    for extension in ("png", "svg", "pdf"):
        path = tmp_path / f"clean-fit.{extension}"
        write_live_plot_image(path, window._plot, frozen_snapshot=snapshot)
        assert path.stat().st_size > 100
    bundle = tmp_path / "clean-fit.zip"
    write_figure_bundle(bundle, snapshot)
    with zipfile.ZipFile(bundle) as archive:
        assert "KS REJECTED" not in archive.read("figure.svg").decode()
        assert json.loads(archive.read("provenance.json"))["details"] == details
    assert window._last_fit is fit
    assert fit.params == params and fit.meta == meta
    assert window.details_text() == details
    pd.testing.assert_frame_equal(window._plot_data(), frame, check_exact=True)
    np.testing.assert_array_equal(window._plot.viewRange(), ranges)
    for item, x, y in curves:
        np.testing.assert_array_equal(item.xData, x)
        np.testing.assert_array_equal(item.yData, y)
    # Deferred advisory workers may finish during Qt teardown; the invariant
    # under test is the synchronous capture/export operation above.
    monkeypatch.undo()


def test_authored_warning_word_is_not_removed(rejected):
    rejected.set_title("Warning threshold comparison")
    assert rejected._plot.titleLabel.text == "Warning threshold comparison"
    assert "KS REJECTED" in rejected.details_text()
