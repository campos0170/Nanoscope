"""Scientific controls remain legible and truthful without changing computed arrays."""
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtGui, QtWidgets

from nanoscope.gui.plot_appearance import PlotAppearance, apply_plot_appearance
from nanoscope.gui.subpop_window import SubpopReportWindow
from .test_subpop_views import report
from .test_p4_noise_series import window as noise_window, forbid_kernels


def _brightness(hexcolor):
    rgb = QtGui.QColor(hexcolor).getRgbF()[:3]
    values = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb]
    return sum(c * w for c, w in zip(values, (.2126, .7152, .0722)))


def _contrast(a, b):
    values = sorted((_brightness(a), _brightness(b)))
    return (values[1] + .05) / (values[0] + .05)


def _rendered_legend_colors(legend):
    colors = []
    for _sample, label in legend.items:
        block = label.item.document().begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid() and fragment.text().strip():
                    colors.append(fragment.charFormat().foreground().color().name())
                iterator += 1
            block = block.next()
    return colors


def test_subpopulation_legends_follow_light_dark_and_restore_original_plate(qtbot, report):
    page = SubpopReportWindow(); qtbot.addWidget(page); page.set_report(report)
    plots = [page.embedding_view()._plot, page.waveform_view()._plot]
    original = [(p.getPlotItem().legend.opts['brush'], p.getPlotItem().legend.opts['pen'],
                 p.getPlotItem().legend.opts['labelTextColor']) for p in plots]
    data = report.embeddings.copy(deep=True)
    for theme, ink, plate in [('light', '#20242b', '#ffffff'), ('dark', '#e6e9ef', '#151a21'),
                              ('light', '#20242b', '#ffffff')]:
        page.set_appearance_state({'background': theme})
        for p in plots:
            legend = p.getPlotItem().legend
            assert QtGui.QColor(legend.opts['labelTextColor']).name() == ink
            assert legend.opts['brush'].color().name() == plate
            assert set(_rendered_legend_colors(legend)) == {ink}
            assert _contrast(ink, plate) >= 4.5
    page.set_appearance_state({})
    for p, (brush, pen, color) in zip(plots, original):
        legend = p.getPlotItem().legend
        assert legend.opts['brush'] == brush
        assert legend.opts['pen'] == pen
        assert legend.opts['labelTextColor'] == color
    import pandas as pd
    pd.testing.assert_frame_equal(report.embeddings, data)


def test_custom_original_legend_plate_restored_after_publication_style(qtbot):
    widget = pg.PlotWidget(); qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    legend = plot.addLegend(brush=pg.mkBrush('#334455'), pen=pg.mkPen('#8899aa'), labelTextColor='#fafafa')
    plot.plot([1, 2], [3, 4], name='series')
    apply_plot_appearance(plot, PlotAppearance(background='light'), plot_widget=widget)
    assert legend.opts['brush'].color().name() == '#ffffff'
    apply_plot_appearance(plot, PlotAppearance(), plot_widget=widget)
    assert legend.opts['brush'].color().name() == '#334455'
    assert legend.opts['pen'].color().name() == '#8899aa'
    assert QtGui.QColor(legend.opts['labelTextColor']).name() == '#fafafa'


def test_noise_fit_quality_visible_with_details_collapsed_and_no_recomputation(qtbot, tmp_path, monkeypatch):
    page = noise_window(qtbot, tmp_path)
    page.set_compare([])
    page.set_nperseg(8192)
    page.show(); qtbot.wait(20)
    assert page._details.isHidden()
    before = [(c.freqs.copy(), c.psd.copy(), deepcopy(c.summary)) for c in page.curves()]
    r2 = page.curves()[0].summary['flicker']['r2']
    text = page.fit_label_text()
    assert f'R² = {r2:.3f}' in text
    assert 'descriptive fit only' in text and 'bins' in text
    forbid_kernels(monkeypatch)
    page.set_appearance_state({'background': 'light'})
    for c, (f, p, summary) in zip(page.curves(), before):
        np.testing.assert_array_equal(c.freqs, f); np.testing.assert_array_equal(c.psd, p)
        assert c.summary == summary


def test_subpop_source_and_action_text_do_not_promise_missing_settings_dialog(qtbot, report):
    page = SubpopReportWindow(); qtbot.addWidget(page)
    page._nanoscope_workspace_source = {'source_key':'/synthetic/a.csv', 'channel':0,
                                      'channel_name':'current', 'module_id':'detect.schmitt'}
    page.set_report(report)
    assert 'a.csv' in page._source_label.text() and 'Pinned' in page._source_label.text()
    tip = page.inspector_adapter().update_tooltip
    assert 'original detection' in tip and 'defaults' in tip and 'dialog opens' not in tip
    page.set_umap_busy(True)
    assert 'Cancel' in page._generate_umap_btn.toolTip() and 'Stop' not in page._generate_umap_btn.toolTip()


def test_noise_legacy_theme_is_restored_after_shared_appearance_override(qtbot, tmp_path, monkeypatch):
    page = noise_window(qtbot, tmp_path)
    original = [c.psd.copy() for c in page.curves()]
    forbid_kernels(monkeypatch)
    page.set_background('light')
    legend = page.plot_item().legend
    assert legend.opts['brush'].color().name() == '#ffffff'
    assert QtGui.QColor(legend.opts['labelTextColor']).name() == '#1a1a1a'
    assert set(_rendered_legend_colors(legend)) == {'#1a1a1a'}
    page.set_appearance_state({'background':'dark'})
    assert legend.opts['brush'].color().name() == '#151a21'
    page.set_background('light')
    assert legend.opts['brush'].color().name() == '#151a21'  # explicit shared override wins
    page.set_appearance_state({})
    assert legend.opts['brush'].color().name() == '#ffffff'
    assert QtGui.QColor(legend.opts['labelTextColor']).name() == '#1a1a1a'
    for curve, values in zip(page.curves(), original):
        np.testing.assert_array_equal(curve.psd, values)
