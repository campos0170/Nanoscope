"""The custom Distributions renderer obeys shared look without changing measurements."""
from dataclasses import replace
import nanoscope  # noqa: F401
import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from nanoscope.core.figure_style import FigureStyle
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plots_panel import PlotsWindow, StyleSettings, PlotConfig, plot_config_to_json, plot_config_from_json
from . import p1_hosting as H


def _look(frame=True):
    style = FigureStyle('plots.palette', 1, 'Palette test', look={
        'palette': ['#aa11bb', '#22cc44'], 'frame': frame, 'line_width_pt': .625,
        'marker_size_pt': 3.125})
    return PlotAppearance.from_mapping(bind_native_style({}, style))


def test_plots_frame_native_reader_and_custom_appearance_bridge_are_lossless():
    for frame in (None, False, True):
        config = replace(PlotConfig(), style=replace(StyleSettings(), frame=frame))
        assert plot_config_from_json(plot_config_to_json(config)) == config
        assert PlotAppearance.from_mapping(PlotsWindow._appearance_of(config.style)).frame == frame
    with pytest.raises(ValueError):
        replace(StyleSettings(), frame=1)


@pytest.mark.parametrize('value', [0, 1, 'true', [], [1, 2], {}])
def test_plots_frame_reader_rejects_non_boolean_values(value):
    from nanoscope.gui.plots_panel import PlotConfigError
    payload = plot_config_to_json(PlotConfig())
    payload['style']['frame'] = value
    with pytest.raises(PlotConfigError, match='Style frame'):
        plot_config_from_json(payload)


def _settle_scientific_callbacks(page, qtbot):
    # Construction/source selection can leave native editing/timer callbacks queued.
    # Establish the committed result before forbidding scientific work during Style.
    if page.focusWidget() is not None:
        page.focusWidget().clearFocus()
    QtWidgets.QApplication.processEvents()
    qtbot.wait(350)  # includes the existing 300 ms stale-mark timer
    page._history_commit()


def test_plots_style_palette_restyles_only_unassigned_ink_and_frame_without_compute(qtbot, monkeypatch):
    page = H.page_builders()['distributions'](); qtbot.addWidget(page)
    page._run_plot()
    _settle_scientific_callbacks(page, qtbot)
    assert page._hist_items
    item = page._hist_items[0]
    before = (item.xData.copy(), item.yData.copy(), page._plot.getViewBox().viewRange())
    monkeypatch.setattr(page, '_redraw', lambda *_a, **_k: pytest.fail('recomputed'))
    monkeypatch.setattr(page, '_run_plot', lambda *_a, **_k: pytest.fail('analysis'))
    assert page.inspector_adapter().apply_look(_look(True), False)
    assert page._data_pen().color().name() == '#aa11bb'
    assert item.opts['pen'].color().name() == '#aa11bb'
    assert page._style.frame is True
    assert page._plot.getViewBox().border.style() != QtCore.Qt.NoPen
    np.testing.assert_array_equal(item.xData, before[0]); np.testing.assert_array_equal(item.yData, before[1])
    assert page._plot.getViewBox().viewRange() == before[2]
    assert page.inspector_adapter().apply_look(_look(False), False)
    assert page._plot.getViewBox().border.style() == QtCore.Qt.NoPen
    QtWidgets.QApplication.processEvents()
    qtbot.wait(350)  # style must not schedule delayed scientific work either
    monkeypatch.undo()  # the guard concerns Style, not QWidget destruction/focus loss


def test_shared_palette_preserves_multiple_source_identity_inks(qtbot):
    page = H.page_builders()['distributions'](); qtbot.addWidget(page)
    page.set_sources(list(page._sources)); page._run_plot()
    assert len(page._hist_items) == 2
    before = [item.opts['pen'].color().name() for item in page._hist_items]
    assert len(set(before)) == 2
    assert page.inspector_adapter().apply_look(_look(True), False)
    assert [item.opts['pen'].color().name() for item in page._hist_items] == before



def test_series_override_is_one_config_step_with_stable_key_and_no_analysis(qtbot, monkeypatch):
    from nanoscope.core.figure_series import validate_series_styles
    from nanoscope.gui.figure_adapter_plots import _restyled_config
    page = H.page_builders()['distributions'](); qtbot.addWidget(page)
    page.set_sources(list(page._sources)); page._run_plot(); page._history_commit()
    _settle_scientific_callbacks(page, qtbot)
    offered = page.figure_series()
    data = [entry for entry in offered if entry.role == 'data']
    assert len(data) == 2 and len({entry.series_id for entry in data}) == 2
    chosen = data[0]
    item = next(item for item in page._hist_items
                if item._figure_series_meta['series_id'] == chosen.series_id)
    original_xy = (item.xData.copy(), item.yData.copy())
    before = page.config()
    monkeypatch.setattr(page, '_redraw', lambda *_a, **_k: pytest.fail('analysis'))
    style = {'color': '#ab12cd', 'line_width_pt': .625, 'visible': False}
    appearance = replace(page._appearance_widget.state(),
        series_styles=validate_series_styles({chosen.series_id: style}))
    page._appearance_widget.apply_appearance(appearance)
    assert page.config().style.series_styles[chosen.series_id]['line_width_pt'] == .625
    assert item.opts['pen'].color().name() == '#ab12cd'
    assert item.opts['pen'].widthF() == pytest.approx(.625 * 96 / 72)
    assert not item.isVisible()
    np.testing.assert_array_equal(item.xData, original_xy[0]); np.testing.assert_array_equal(item.yData, original_xy[1])
    assert page.undo_config() and page.config() == before
    assert item.isVisible() and item.opts['pen'].color() == QtGui.QColor(chosen.color)
    assert page.redo_config() and not item.isVisible()
    state = page.config()
    assert plot_config_from_json(plot_config_to_json(state)) == state
    # Applying another global look preserves explicit series overrides in dormant transforms.
    assert _restyled_config(state, _look(False), True).style.series_styles == state.style.series_styles
    renamed = replace(page._style, label_overrides={chosen.identity_key: 'Renamed display label'})
    page._style = renamed
    assert next(row for row in page.figure_series() if row.series_id == chosen.series_id).label == 'Renamed display label'
    QtWidgets.QApplication.processEvents()
    qtbot.wait(350)
    monkeypatch.undo()


def test_category_override_changes_ink_not_scientific_class_ids_or_point_geometry(qtbot):
    import pandas as pd
    import pyqtgraph as pg
    from types import SimpleNamespace
    from nanoscope.core.figure_series import validate_series_styles
    page = H.page_builders()['distributions'](); qtbot.addWidget(page)
    page._plot_type_combo.setCurrentText('Scatter')
    # A programmatic plot-kind selection intentionally commits a real source draw.
    # Clear that completed draw before installing the frozen category fixture.
    assert page.figure_series()
    page._clear_plot()
    assert not page.figure_series() and not page._presentation_legend
    labels = np.array([0, 1, 0, 1])
    item = pg.ScatterPlotItem(x=[.001, .002, .003, .004], y=[1., 2., 3., 4.], size=7)
    item._presentation_cluster_labels = labels.copy()
    item._presentation_cluster_source = 'source-a'
    item._presentation_role = 'category'
    page._plot.addItem(item); page._scatter_items = [item]
    page._cluster_result = SimpleNamespace(labels=pd.Series(labels))
    page._legend_clusters(labels)
    raw = item.getData()
    offered = page.figure_series()
    assert len(offered) == 2
    first = next(row for row in offered if row.identity_key.endswith(':0'))
    page._style = replace(page._style, series_styles=validate_series_styles({first.series_id: {
        'color': '#102030', 'marker_size_pt': 3.125, 'visible': False}}))
    page._apply_series_styles()
    np.testing.assert_array_equal(page._cluster_of_drawn_points(), labels)
    np.testing.assert_array_equal(item.getData()[0], raw[0]); np.testing.assert_array_equal(item.getData()[1], raw[1])
    assert [point.size() for point in item.points()] == pytest.approx([0., 7., 0., 7.])
    assert item.points()[0].brush().color().name() == '#102030'
    page._style = replace(page._style, series_styles={})
    page._apply_series_styles()
    assert all(point.size() == 7. for point in item.points())
    np.testing.assert_array_equal(page._cluster_of_drawn_points(), labels)


def test_frozen_sampled_fit_keeps_its_series_id_and_override_without_refitting(qtbot, monkeypatch):
    from nanoscope.core.figure_series import validate_series_styles
    from nanoscope.gui.figure_adapter_pages import adapter_for
    from nanoscope.gui import plots_panel
    page = H.page_builders()['distributions'](); qtbot.addWidget(page)
    page.set_plot_type('Scatter'); page.set_scatter_columns('dwell_s', 'frac_depth')
    page.set_scatter_fit('linear'); page._run_plot()
    fit = next(row for row in page.figure_series() if row.role == 'fit')
    page._appearance_widget.apply_appearance(replace(page._appearance_widget.state(),
        series_styles=validate_series_styles({fit.series_id: {'color': '#7733aa', 'line_width_pt': .625}})))
    native = adapter_for(page, 'distributions').capture_native()
    clone = H.page_builders()['distributions'](); qtbot.addWidget(clone)
    monkeypatch.setattr(plots_panel.populations, 'linear_fit', lambda *_a, **_k: pytest.fail('refitted'))
    report = adapter_for(clone, 'distributions').restore_native(native)
    assert not report.skipped, report
    copied = next(row for row in clone.figure_series() if row.series_id == fit.series_id)
    assert copied.color == '#7733aa'
    curve = next(item for item in clone._scatter_fit_items
                 if item._figure_series_meta['series_id'] == fit.series_id)
    assert curve.opts['pen'].widthF() == pytest.approx(.625 * 96 / 72)
    np.testing.assert_array_equal(curve.xData, page._scatter_fit_items[0].xData)
    np.testing.assert_array_equal(curve.yData, page._scatter_fit_items[0].yData)



def test_recipe_drops_source_series_overrides_and_recall_retains_destination_series():
    saved = replace(PlotConfig(), style=replace(StyleSettings(),
        series_styles={'foreign-recording': {'color': '#123456'}}, frame=True))
    template = saved.as_template()
    assert not template.style.series_styles and template.style.frame is True
    destination = replace(PlotConfig(), style=replace(StyleSettings(),
        series_styles={'own-recording': {'color': '#abcdef'}}))
    recalled = destination.with_template(saved)
    assert recalled.style.series_styles == destination.style.series_styles
    assert recalled.style.frame is True



def test_legacy_contour_identity_uses_validated_writer_blocks_not_ink_or_label():
    import pandas as pd
    from copy import deepcopy
    from nanoscope.gui.figure_adapter_plots import _legacy_series_of_render
    grids = {f'grid-{i}': pd.DataFrame([[1., 2.], [3., 4.]]) for i in range(4)}
    render = {'nodes': [{'primitive': 'contour', 'role': 'contour', 'table': f'grid-{i}',
                        'transform': [1., 0., 0., 0., 1., 0., 0., 0., 1.], 'attributes': {}}
                       for i in range(4)],
              'legends': [{'kind': 'series', 'key': 'source-a', 'label': 'Same displayed name'},
                          {'kind': 'series', 'key': 'source-b', 'label': 'Same displayed name'}],
              'drawn_source_keys': ['source-a', 'source-b'], 'applied_config': {'kde': {'levels': 2}}}
    before = deepcopy(render)
    result = _legacy_series_of_render(render, grids)
    assert [result[i][0] for i in range(4)] == ['source-a', 'source-a', 'source-b', 'source-b']
    assert render == before
    grids['grid-1'].iloc[0, 0] = 99.
    assert _legacy_series_of_render(render, grids) == {}  # no guessed association
