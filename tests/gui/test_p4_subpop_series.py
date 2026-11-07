"""A category edit links retained embedding points, waveform and legend only."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

from nanoscope.core.analysis_workspace import capture_subpopulation, restore_subpopulation
from nanoscope.gui.categorical import category_colour
from nanoscope.gui.subpop_window import SubpopReportWindow
from .test_subpop_views import report  # noqa: F401: one small real report fixture


def test_category_ink_links_views_without_reassigning_or_recomputing(qtbot, report, monkeypatch):
    from nanoscope.analysis import subpop_report as module
    page = SubpopReportWindow(); qtbot.addWidget(page); page.set_report(report)
    monkeypatch.setattr(module, 'run_subpopulations', lambda *a, **k: pytest.fail('Style reran clustering'))
    embed = page.embedding_view()._plot.getPlotItem()
    wave = page.waveform_view()._plot.getPlotItem()
    scatter = next(item for item in embed.items if hasattr(item, '_nanoscope_category_ids'))
    curves = [item for item in wave.items if hasattr(item, '_nanoscope_waveform_series')]
    selected = curves[0]._nanoscope_series_descriptor
    descriptor = next(d for d in page.inspector_adapter().format.series_provider()
                      if d.series_id == selected.series_id)
    assert set(descriptor.capabilities) == {'color', 'line_width_pt', 'marker_size_pt', 'visible'}
    cid = int(descriptor.identity_key.rsplit(':', 1)[1])
    mask = scatter._nanoscope_category_ids == cid
    assert mask.any() and (~mask).any()
    raw = (scatter.getData()[0].copy(), scatter.getData()[1].copy(), scatter._nanoscope_category_ids.copy())
    exported = page._plot_data().copy(deep=True)
    original_report = deepcopy(report.meta)
    page._appearance_widget.apply_appearance(replace(page._appearance_widget.state(), series_styles={
        descriptor.series_id: {'color': '#8833aa', 'line_width_pt': 2.25,
                               'marker_size_pt': 6.75, 'visible': False}}))
    assert not curves[0].isVisible()
    assert pg.mkPen(curves[0].opts['pen']).color().name() == '#8833aa'
    assert pg.mkPen(curves[0].opts['pen']).widthF() == pytest.approx(3.)
    for index, point in enumerate(scatter.points()):
        if mask[index]:
            assert point.size() == 0 and point.brush().color().name() == '#8833aa'
        else:
            assert point.size() > 0
            assert point.brush().color().name() == category_colour(int(raw[2][index])).name()
    swatch, sample, label = (ref() for ref in scatter._nanoscope_category_legend[cid])
    assert not sample.isVisible() and not label.isVisible()
    assert pg.mkBrush(swatch.opts['brush']).color().name() == '#8833aa'
    assert descriptor.series_id in {d.series_id for d in page.series_descriptors()}
    current = {d.series_id: d for d in page.series_descriptors()}
    effective = current[descriptor.series_id].extra['effective_style']
    assert effective['color'] == '#8833aa'
    assert effective['line_width_pt'] == 2.25 and effective['marker_size_pt'] == 6.75
    assert 'visible' not in effective
    for other in current.values():
        if other.series_id != descriptor.series_id:
            original_id = int(other.identity_key.rsplit(':', 1)[1])
            assert other.extra['effective_style']['color'] == category_colour(original_id).name()
    pd.testing.assert_frame_equal(page._plot_data(), exported, check_exact=True)
    assert report.meta == original_report
    np.testing.assert_array_equal(scatter.getData()[0], raw[0])
    np.testing.assert_array_equal(scatter.getData()[1], raw[1])
    np.testing.assert_array_equal(scatter._nanoscope_category_ids, raw[2])
    assert page._appearance_widget.undo()
    assert curves[0].isVisible() and sample.isVisible() and label.isVisible()
    assert all(point.size() > 0 for point in scatter.points())
    assert page._appearance_widget.redo()
    artifact = capture_subpopulation(page)
    restored = SubpopReportWindow(); qtbot.addWidget(restored)
    restored_report = restore_subpopulation(restored, artifact)
    assert restored_report.restored and not restored_report.skipped
    assert restored.appearance_state()['series_styles'] == page.appearance_state()['series_styles']
    pd.testing.assert_frame_equal(restored._plot_data(), exported, check_exact=True)
    same = next(d for d in restored.series_descriptors() if d.series_id == descriptor.series_id)
    assert same.identity_key == descriptor.identity_key


def test_embedding_mode_is_part_of_category_identity(qtbot, report):
    page = SubpopReportWindow(); qtbot.addWidget(page); page.set_report(report)
    embedding = page.embedding_view()
    before = {d.series_id for d in page.series_descriptors()}
    alternatives = [mode for mode in embedding.colour_modes() if mode != embedding.colour_mode()]
    assert alternatives
    embedding.set_colour_mode(alternatives[0])
    after = {d.series_id for d in page.series_descriptors()}
    # The waveform's GMM identities remain; other clustering labels get separate IDs.
    wave_ids = {item._nanoscope_series_descriptor.series_id
                for item in page.waveform_view()._plot.getPlotItem().items
                if hasattr(item, '_nanoscope_series_descriptor')}
    assert before & after == wave_ids
    assert after - before


def test_category_outline_tracks_only_visible_mask_footprints_without_moving_data(qtbot):
    from PySide6 import QtCore
    from nanoscope.core.figure_series import stable_series_id
    from nanoscope.gui.categorical import add_category_scatter
    from nanoscope.gui.figure_text import figure_text_for
    from nanoscope.gui.format_toolbar import SelectionOutline
    from nanoscope.gui.subpop_window import _MODE_GMM
    page = SubpopReportWindow(); qtbot.addWidget(page)
    page.resize(1200, 900); page.show()
    plot = page.embedding_view()._plot.getPlotItem()
    item = add_category_scatter(plot, [0., 1., 9., 10.], [0., 1., 9., 10.], [0, 0, 1, 1])
    item._nanoscope_category_ids = np.array([0, 0, 1, 1])
    item._nanoscope_category_mode = _MODE_GMM
    item._nanoscope_category_legend = {}
    plot.setRange(xRange=(-1., 11.), yRange=(-1., 11.), padding=0)
    page._apply_appearance()
    qtbot.wait(10)  # settle the newly shown plot's viewport geometry
    raw = tuple(value.copy() for value in item.getData())
    key0 = stable_series_id('subpopulations', 'data', f'{_MODE_GMM}:0')
    key1 = stable_series_id('subpopulations', 'data', f'{_MODE_GMM}:1')
    bounds0, bounds1 = plot._nanoscope_series_bounds(key0), plot._nanoscope_series_bounds(key1)
    assert bounds0 is not None and bounds1 is not None
    assert bounds0.right() < bounds1.left()
    assert bounds0.width() < item.sceneBoundingRect().width() / 2.
    for index in (0, 1):
        assert bounds0.contains(item.mapToScene(item.points()[index].pos()))
    controller = figure_text_for(plot)
    outline = SelectionOutline(controller.view().viewport()); outline.attach(controller)
    outline.show_key(key0)
    expected = controller.view().mapFromScene(bounds0).boundingRect().adjusted(-2, -2, 2, 2)
    assert outline.geometry() == expected.intersected(controller.view().viewport().rect())
    page._appearance_widget.apply_appearance(replace(page._appearance_widget.state(),
        series_styles={key0: {'visible': False}}))
    assert plot._nanoscope_series_bounds(key0) is None
    assert plot._nanoscope_series_bounds(key1) is not None
    outline.reposition()
    assert not outline.isVisible()
    for actual, original in zip(item.getData(), raw):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(item._nanoscope_category_ids, [0, 0, 1, 1])
