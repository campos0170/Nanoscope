"""Noise source/fit identities and frozen spectra survive presentation-only edits."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import pyqtgraph as pg

from nanoscope.core.analysis_workspace import capture_psd, restore_psd
from nanoscope.core.figure_style import builtin_styles
from nanoscope.core.session import Session
from nanoscope.gui.figure_style_binding import bind_native_style
from nanoscope.gui.psd_panel import PSDWindow
from .test_psd_overlay import _doc


def window(qtbot, tmp_path):
    session = Session()
    first = session.add_document(_doc(tmp_path, 'a.dat', sigma=2., n=8000, seed=1))
    second = session.add_document(_doc(tmp_path, 'b.dat', sigma=7., n=8000, seed=2))
    page = PSDWindow(session); qtbot.addWidget(page)
    page.set_target(first, 0); page.set_compare([second])
    assert len(page.curves()) == 2
    return page


def forbid_kernels(monkeypatch):
    from nanoscope.gui import psd_panel as module
    for name in ('welch_psd', 'fft_magnitude', 'spectrogram_psd'):
        monkeypatch.setattr(module.psd_mod, name, lambda *a, **k: pytest.fail('Style recomputed spectrum'))
    monkeypatch.setattr(module.ap_mod, 'all_points_histogram',
                        lambda *a, **k: pytest.fail('Style recomputed histogram'))


def test_two_psd_source_identities_keep_palette_and_per_series_undo_frozen_restore(qtbot, tmp_path, monkeypatch):
    page = window(qtbot, tmp_path)
    curves = list(page.curves())
    original = [(c.freqs.copy(), c.psd.copy(), deepcopy(c.summary), c.item.xData.copy(), c.item.yData.copy())
                for c in curves]
    page.set_appearance_state(bind_native_style({}, builtin_styles()[0]))
    assert [pg.mkPen(c.item.opts['pen']).color().name() for c in curves] == [c.colour.lower() for c in curves]
    descriptors = [d for d in page.inspector_adapter().format.series_provider() if d.role == 'data']
    assert [d.identity_key for d in descriptors] == [c.key for c in curves]
    assert all(d.identity_kind == 'source' for d in descriptors)
    forbid_kernels(monkeypatch)
    page._plot.setRange(xRange=(1., 3.), yRange=(-3., 0.), padding=0)
    view = deepcopy(page._plot.viewRange())
    widget = page._appearance_widget
    before = page.appearance_state()
    mark = widget.history_mark()
    widget.apply_appearance(replace(widget.state(), series_styles={descriptors[1].series_id: {
        'color': '#8833aa', 'line_width_pt': 2.25, 'visible': False}}))
    assert widget.history_mark() == mark + 1
    assert not curves[1].item.isVisible() and curves[0].item.isVisible()
    assert pg.mkPen(curves[1].item.opts['pen']).widthF() == pytest.approx(3.)
    assert pg.mkPen(curves[1].item.opts['pen']).color().name() == '#8833aa'
    np.testing.assert_allclose(page._plot.viewRange(), view)
    assert widget.undo() and page.appearance_state() == before
    assert curves[1].item.isVisible()
    assert widget.redo() and not curves[1].item.isVisible()
    artifact = capture_psd(page)
    restored = PSDWindow(Session()); qtbot.addWidget(restored)
    report = restore_psd(restored, artifact)
    assert report.restored and not report.skipped
    restored_ids = [d.series_id for d in restored.series_descriptors() if d.role == 'data']
    assert restored_ids == [d.series_id for d in descriptors]
    assert not restored.curves()[1].item.isVisible()
    for curve, saved in zip(curves, original):
        for actual, expected in zip((curve.freqs, curve.psd, curve.item.xData, curve.item.yData),
                                    (saved[0], saved[1], saved[3], saved[4])):
            np.testing.assert_array_equal(actual, expected)
        assert curve.summary == saved[2]


@pytest.mark.parametrize('view', ['fft', 'allpoints'])
def test_auxiliary_noise_view_keeps_exact_source_and_arrays_when_ink_changes(qtbot, tmp_path, monkeypatch, view):
    page = window(qtbot, tmp_path)
    page.set_view(view)
    descriptor, = page.series_descriptors()
    assert descriptor.identity_kind == 'source' and descriptor.identity_key == page.curves()[0].key
    item = page._fft_item if view == 'fft' else page._hist_item
    raw = (item.xData.copy(), item.yData.copy())
    forbid_kernels(monkeypatch)
    page._appearance_widget.apply_appearance(replace(page._appearance_widget.state(),
        series_styles={descriptor.series_id: {'color': '#335577', 'line_width_pt': 1.5}}))
    assert pg.mkPen(item.opts['pen']).color().name() == '#335577'
    assert pg.mkPen(item.opts['pen']).widthF() == pytest.approx(2.)
    np.testing.assert_array_equal(item.xData, raw[0]); np.testing.assert_array_equal(item.yData, raw[1])
