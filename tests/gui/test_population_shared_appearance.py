"""Every Population appearance gesture consumes only the committed picture."""
from __future__ import annotations

from dataclasses import replace
import copy

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

from nanoscope.analysis import populations, stats
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget

from .test_population_plot_presentation_updates import window  # noqa: F401
from .test_plots_panel import _settle


def _forbid(w, monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("Appearance must not run analysis, rebuild a plot or collect sources")
    for name in ("_redraw", "_drawn_frames", "_draw_histogram", "_draw_scatter",
                 "_draw_kde", "_draw_pca", "_overlay_fit", "_overlay_scatter_fit"):
        monkeypatch.setattr(w, name, fail)
    for module, name in ((stats, "ci_band"), (stats, "binned_residuals"),
                         (populations, "linear_fit")):
        monkeypatch.setattr(module, name, fail)


def _snapshot(w):
    return [(item, item.xData, item.yData, item.xData.copy(), item.yData.copy())
            for item in w._plot.listDataItems()
            if isinstance(item, pg.PlotDataItem) and item.xData is not None and item.yData is not None]


def _assert_arrays(snapshot):
    for item, x, y, xc, yc in snapshot:
        assert item.xData is x and item.yData is y
        np.testing.assert_array_equal(x, xc)
        np.testing.assert_array_equal(y, yc)


@pytest.mark.parametrize("kind", ["Histogram", "Scatter", "KDE", "PCA"])
def test_common_controls_presets_and_undo_keep_science_and_view(window, qtbot, monkeypatch, kind):
    w = window
    if kind == "PCA":
        w.set_feature_selection(["dwell_s", "amplitude", "snr"])
        w.run_pca()
    else:
        w.set_plot_type(kind)
        if kind == "Histogram":
            w.set_histogram_column("dwell_s")
            w.set_fit("exponential")
        elif kind == "Scatter":
            w.set_scatter_columns("dwell_s", "amplitude")
        else:
            w.set_kde_columns("dwell_s", "amplitude")
    _settle(w)
    qtbot.wait(10)
    snapshot = _snapshot(w)
    original_data = w._plot_data().copy(deep=True)
    fit = w.fit_result()
    pca = w._pca_result
    source_frames = {key: source.df.copy(deep=True) for key, source in w._sources.items()}
    w._plot.setRange(xRange=(0.0001, 0.0007), yRange=(3, 49), padding=0)
    qtbot.wait(10)
    view = copy.deepcopy(w._plot.viewRange())
    auto = list(w._plot.vb.autoRangeEnabled())
    _forbid(w, monkeypatch)

    w.set_data_color("#236a9a")
    w.set_fit_color("#db7621")
    w._on_style_point_size(9)
    w._on_style_line_width(4)
    w._on_style_show_legend(False)
    w._on_style_show_legend(True)
    w._on_style_legend_counts(False)
    w._on_style_legend_counts(True)
    widget = w._appearance_widget
    widget._apply_preset(1)
    custom = replace(widget.state(), title="Published result", x_label="Horizontal",
                     y_label="Vertical", legend_visible=True, legend_size=12,
                     line_width=3, marker_size=8)
    widget.set_state(custom)
    widget.changed.emit(custom)
    w._history_commit()
    state = w.config()
    assert widget.save_user_preset("Population invariant test", overwrite=True)
    widget._apply_preset(2)
    assert widget.apply_user_preset("Population invariant test")
    assert w.config() == state
    assert w.undo_config()
    assert w.redo_config()
    assert w.config() == state
    assert widget.delete_user_preset("Population invariant test")
    qtbot.wait(10)

    _assert_arrays(snapshot)
    assert w.fit_result() is fit
    assert w._pca_result is pca
    np.testing.assert_array_equal(w._plot.viewRange(), view)
    assert w._plot.vb.autoRangeEnabled() == auto
    pd.testing.assert_frame_equal(w._plot_data(), original_data, check_exact=True)
    for key, frame in source_frames.items():
        pd.testing.assert_frame_equal(w._sources[key].df, frame, check_exact=True)
    assert w._plot.titleLabel.text.endswith("Published result")
    assert w._plot.getAxis("bottom").labelText == "Horizontal"
    assert w.figure_applied_config()["style"]["title"] == "Published result"
    monkeypatch.undo()  # Closing deliberately resets the window, outside this contract.


def test_fit_ornaments_use_cached_geometry_and_keep_fit_and_export(window, qtbot, monkeypatch):
    w = window
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    _settle(w)
    assert w._presentation_residual is not None
    assert w._presentation_bands
    assert not w._band_items
    fit = w.fit_result()
    data = w._plot_data().copy(deep=True)
    snapshot = _snapshot(w)
    w._plot.setRange(xRange=(0.0001, 0.0007), yRange=(0, 50), padding=0)
    qtbot.wait(10)
    view = copy.deepcopy(w._plot.viewRange())
    _forbid(w, monkeypatch)

    for name in ("_fit_values_check", "_r2_check", "_gof_check", "_ci_band_check",
                 "_residual_check", "_prov_check"):
        control = getattr(w, name)
        control.setChecked(False)
        control.setChecked(True)
    assert w._band_items and not w._resid_widget.isHidden()
    assert "POINTWISE" in w.details_text()
    assert "APPROXIMATE" in w.details_text() and "no p-value" in w.details_text()
    residual_items = tuple(w._resid.items)
    band_items = tuple(w._band_items)
    residual_arrays = [(item, item.xData, item.yData, item.xData.copy(), item.yData.copy())
                       for item in w._resid.listDataItems()]
    w.set_fit_color("#ac1234")
    w._on_style_line_width(5)
    w._on_style_point_size(10)
    for item in w._resid.listDataItems():
        if item.opts["pen"] is not None:
            assert item.opts["pen"].color().name() == "#ac1234"
            assert item.opts["pen"].widthF() == 5
    assert w._band_items[0].opts["pen"].color().name() == "#ac1234"
    w._residual_check.setChecked(False)
    w._ci_band_check.setChecked(False)
    w._residual_check.setChecked(True)
    w._ci_band_check.setChecked(True)
    qtbot.wait(10)
    assert tuple(w._resid.items) == residual_items
    assert tuple(w._band_items) == band_items
    _assert_arrays(snapshot + residual_arrays)
    assert w.fit_result() is fit
    pd.testing.assert_frame_equal(w._plot_data(), data, check_exact=True)
    np.testing.assert_array_equal(w._plot.viewRange(), view)
    monkeypatch.undo()


def test_preset_and_undo_do_not_commit_pending_columns(window, monkeypatch):
    w = window
    w.set_histogram_column("dwell_s")
    applied = w.figure_applied_config()
    monkeypatch.setattr(w, "_from_the_user", lambda *_a: True)
    w._hist_col.setCurrentText("amplitude")
    assert w._stale
    w._history_commit()
    item = w._hist_item
    _forbid(w, monkeypatch)
    w._appearance_widget._apply_preset(1)
    assert w.undo_config()
    assert w.redo_config()
    assert w._stale and w._hist_col.currentText() == "amplitude"
    assert w._hist_item is item
    assert w.figure_applied_config()["axes"] == applied["axes"]
    monkeypatch.undo()


def test_shared_user_presets_contain_only_appearance_and_load_silently(qtbot):
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    changes = []
    widget.changed.connect(changes.append)
    widget.set_state({"title": "A title", "legend_visible": False, "legend_size": 12,
                      "bins": 100, "fit": "gamma", "source_keys": ["secret"]})
    assert not changes
    assert widget.save_user_preset("Cross-view test", overwrite=True)
    saved = PlotAppearanceWidget.user_preset_states()["Cross-view test"]
    assert saved == {"title": "A title", "legend_visible": False, "legend_size": 12}
    other = PlotAppearanceWidget()
    qtbot.addWidget(other)
    assert other.apply_user_preset("Cross-view test")
    assert other.state() == widget.state()
    assert widget.delete_user_preset("Cross-view test")


def test_common_appearance_replaces_duplicate_basic_controls(window):
    window._show_control_section("Appearance")
    assert window._appearance_widget.isVisible()
    assert window._title_edit.isHidden() and window._line_width.isHidden()
    assert window._fit_values_check.isVisible() and window._legend_entries.isVisible()


def test_population_export_uses_frozen_picture_after_dialog(window, monkeypatch, tmp_path):
    from dataclasses import replace
    from nanoscope.gui import figure_bundle, plots_panel
    from nanoscope.gui.plot_export_options import resolve_export_options
    snapshot = figure_bundle.capture_figure_snapshot(window._plot, data=window._plot_data())
    options = replace(resolve_export_options((snapshot.width, snapshot.height)), frozen_snapshot=snapshot)
    out = tmp_path / "frozen.png"
    monkeypatch.setattr(plots_panel, "choose_live_plot_export", lambda *a, **k: (out, options))
    monkeypatch.setattr(plots_panel, "ImageExporter", lambda *a, **k: pytest.fail("constructed live exporter"))
    assert window.save_image() == out
    assert out.read_bytes() == snapshot.png_bytes


def test_presets_saved_on_another_open_page_refresh_on_show(qtbot):
    first, second = PlotAppearanceWidget(), PlotAppearanceWidget()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    assert first.save_user_preset("New cross-page preset", overwrite=True)
    assert second.user_presets.findData("New cross-page preset") == -1
    second.show()
    assert second.user_presets.findData("New cross-page preset") >= 0
    assert first.delete_user_preset("New cross-page preset")


def test_residual_export_choice_is_forwarded_without_reanalysis(window, monkeypatch, tmp_path):
    from nanoscope.gui import plots_panel, plot_export
    from nanoscope.gui.plot_export_options import ExportOptions
    window.set_fit("exponential")
    _settle(window)
    window._residual_check.setChecked(True)
    before = window._plot_data().copy(deep=True)
    target = tmp_path / "with_residual.svg"
    def choose(parent, item, suggestion, **kwargs):
        assert item is window._plot and kwargs["residual_item"] is window._resid
        return target, ExportOptions(".svg", include_residual=True)
    writes = []
    monkeypatch.setattr(plots_panel, "choose_live_plot_export", choose)
    monkeypatch.setattr(plot_export, "write_live_plot_image",
                        lambda *args, **kwargs: writes.append((args, kwargs)) or [])
    _forbid(window, monkeypatch)
    assert window.save_image() == target
    assert writes[0][1]["include_residual"] is True
    assert writes[0][1]["residual_item"] is window._resid
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)
    monkeypatch.undo()


def test_shared_histogram_fill_restyles_and_resets_without_replacing_arrays(qtbot):
    from PySide6 import QtGui
    from nanoscope.gui.plot_appearance import apply_plot_appearance
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    tint = QtGui.QColor("#123456")
    tint.setAlpha(70)
    item = widget.plot(x=[0., 1., 2.], y=[4., 3.], stepMode="center",
                       fillLevel=0, brush=pg.mkBrush(tint))
    x, y = item.xData, item.yData
    apply_plot_appearance(widget.getPlotItem(), PlotAppearance(data_color="#006644"),
                          data_items=[item])
    assert item.opts["fillBrush"].color().name() == "#006644"
    assert item.opts["fillBrush"].color().alpha() == 70
    apply_plot_appearance(widget.getPlotItem(), PlotAppearance(), data_items=[item])
    assert item.opts["fillBrush"].color().name() == "#123456"
    assert item.xData is x and item.yData is y


def test_shared_marginal_bar_style_preserves_geometry_and_original_no_border(qtbot):
    from PySide6 import QtGui
    from nanoscope.gui.plot_appearance import apply_plot_appearance
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    positions, heights = np.array([1., 2.]), np.array([7., 3.])
    tint = QtGui.QColor("#334455")
    tint.setAlpha(120)
    bars = pg.BarGraphItem(x=positions, height=heights, width=.8,
                           brush=pg.mkBrush(tint), pen=None)
    widget.addItem(bars)
    apply_plot_appearance(widget.getPlotItem(), PlotAppearance(data_color="#227755", line_width=3),
                          data_items=[bars])
    assert bars.opts["brush"].color().name() == "#227755"
    assert bars.opts["brush"].color().alpha() == 120
    assert bars.opts["pen"].widthF() == 3
    assert bars.opts["x"] is positions and bars.opts["height"] is heights
    assert bars.opts["width"] == .8
    apply_plot_appearance(widget.getPlotItem(), PlotAppearance(), data_items=[bars])
    assert bars.opts["pen"] is None
    assert bars.opts["brush"].color().name() == "#334455"


def test_figure_source_binding_stays_on_drawn_source_with_pending_selection(window, monkeypatch):
    from PySide6.QtCore import Qt
    from nanoscope.core import analysis_workspace
    from nanoscope.core.sources import PlotSource
    frame = next(iter(window._sources.values())).df.copy(deep=True)
    sources = [PlotSource(key=key, doc_key=f"/{key}.abf", label=key, df=frame.copy(deep=True),
                           module_id="detect.threshold", kind="detection")
               for key in ("Source A", "Source B")]
    window.set_data(sources)
    window.set_sources(["Source A"])
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    committed = window.figure_source_recipes()
    assert [item["source_key"] for item in committed] == ["/Source A.abf"]
    expected_stamp = analysis_workspace.fingerprint_frame(sources[0].df)
    assert committed[0]["event_frame_fingerprint"] == expected_stamp
    monkeypatch.setattr(window, "_from_the_user", lambda *_a: True)
    window._source_list.item(1).setCheckState(Qt.Checked)
    window._source_list.item(0).setCheckState(Qt.Unchecked)
    assert window._stale and window.selected_source_keys() == ["Source B"]
    # No rehash is allowed while styling, exporting a recipe, or replaying ink.
    _forbid(window, monkeypatch)
    monkeypatch.setattr(analysis_workspace, "_plot_source_recipe",
                        lambda *_a: pytest.fail("Restamped a figure during presentation"))
    monkeypatch.setattr(analysis_workspace, "fingerprint_frame",
                        lambda *_a: pytest.fail("Rehashed a source during presentation"))
    sources[0].df.iloc[0, 0] = -123
    window._history_commit()
    window.set_data_color("#667788")
    assert window.undo_config() and window.redo_config()
    assert window.figure_source_recipes() == committed
    detached = window.figure_source_recipes()
    detached[0]["source_key"] = "modified caller"
    assert window.figure_source_recipes() == committed
    monkeypatch.undo()
    window._clear_plot()
    assert window.figure_source_recipes() == []
