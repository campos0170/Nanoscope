"""Colour-bar ink and bounded warnings stay readable in the actual shared look."""
from copy import deepcopy

import nanoscope  # noqa: F401
import numpy as np
from PySide6 import QtCore

from nanoscope.gui.figure_text import TextStyle
from nanoscope.gui.plot_workflow import WarningSummary
from .test_event_plot import _fed_window
from .test_p4_noise_series import window as noise_window, forbid_kernels


def _assert_bar_ink(bar, plot, expected):
    assert plot.getAxis('left').pen().color().name() == expected
    assert bar.axis.textPen().color().name() == expected
    assert bar.axis.pen().color().name() == expected
    for side in ('left', 'right'):
        assert bar.getAxis(side).labelStyle['color'] == expected


def test_event_colour_bar_tracks_light_dark_and_keeps_explicit_label_colour(qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import event_shape
    page = _fed_window(qtbot, tmp_path)
    raw = page.density_counts().copy()
    levels = tuple(page._colorbar.levels())
    view = deepcopy(page._density_main.viewRange())
    original_tick_size = page._colorbar.axis.style['tickFont'].pointSizeF()
    monkeypatch.setattr(event_shape, 'density_2d', lambda *a, **k: (_ for _ in ()).throw(AssertionError('recomputed')))
    for theme, color in [('light', '#20242b'), ('dark', '#e6e9ef'), ('light', '#20242b')]:
        page.set_appearance_state({'background': theme, 'tick_size': 7.5})
        _assert_bar_ink(page._colorbar, page._density_main, color)
        assert page._colorbar.axis.style['tickFont'].pointSizeF() == 7.5
        assert page._colorbar.axis.label.toPlainText().strip() == 'Event count'
    own = page._appearance.with_text_style('colorbar.label', TextStyle(text='Events per bin', color='#aa2277'))
    page.set_appearance_state(own.to_dict())
    assert page._colorbar.axis.labelStyle['color'] == '#aa2277'
    assert page._colorbar.axis.label.toPlainText().strip() == 'Events per bin'
    assert page._colorbar.axis.textPen().color().name() == '#20242b'
    page.set_appearance_state({})
    assert page._colorbar.axis.style['tickFont'].pointSizeF() == original_tick_size
    np.testing.assert_array_equal(page.density_counts(), raw)
    assert tuple(page._colorbar.levels()) == levels
    assert page._density_main.viewRange() == view


def test_spectrogram_colour_bar_uses_final_shared_theme_without_changing_spectrum(qtbot, tmp_path, monkeypatch):
    page = noise_window(qtbot, tmp_path)
    page.set_view('spectrogram')
    assert page._spec_bar is not None
    raw = page._spec_grid_z.copy()
    levels = tuple(page._spec_bar.levels())
    view = deepcopy(page._spec_plot.viewRange())
    label = page._spec_bar.getAxis('right').labelText
    forbid_kernels(monkeypatch)
    for theme, color in [('light', '#20242b'), ('dark', '#e6e9ef'), ('light', '#20242b')]:
        page.set_appearance_state({'background': theme, 'tick_size': 7.5})
        _assert_bar_ink(page._spec_bar, page._spec_plot, color)
        assert page._spec_bar.getAxis('right').labelText == label
        np.testing.assert_array_equal(page._spec_grid_z, raw)
        assert tuple(page._spec_bar.levels()) == levels
        assert page._spec_plot.viewRange() == view


def test_narrow_comparison_warning_has_whole_lines_and_complete_tooltip_details(qtbot):
    summary = WarningSummary(title='Comparison notes'); qtbot.addWidget(summary)
    summary.toggle.setText('Details — methods and provenance')
    warnings = ['These are unweighted, descriptive views of caller-selected event observations. '
                'No censoring, missed-event, calibration, recording-length, or biological-replicate correction is applied.',
                'A source-specific disclosure remains available in full.']
    summary.resize(460, 90)
    summary.set_warnings(warnings)
    summary.show(); qtbot.wait(20)
    label = summary.summary
    lines = label.text().splitlines()
    assert 1 <= len(lines) <= 3 and lines[-1].endswith('…')
    assert '(2)' in label.text()
    metrics = label.fontMetrics()
    assert all(metrics.horizontalAdvance(line) <= label.contentsRect().width() for line in lines)
    assert label.contentsRect().height() >= metrics.lineSpacing() * len(lines)
    assert label.toolTip() == summary.full_text() == '\n\n'.join(warnings)
    qtbot.mouseClick(summary.toggle, QtCore.Qt.LeftButton)
    assert summary.details.isVisible()
    assert summary.details.toPlainText() == '\n\n'.join(warnings)
