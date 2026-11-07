"""Ticks, pointer text and typed bounds share an inverse; raw geometry stays raw."""
import nanoscope  # noqa: F401
import numpy as np
import pyqtgraph as pg
import pytest

from nanoscope.core.quantity_display import AxisDisplayBasis, quantity_for, density_quantity
from nanoscope.gui.axes import ReadableLogAxis
from nanoscope.gui.figure_text import AxisPin, _display_to_view, _view_to_display
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget
from nanoscope.gui.quantity_axis import configure_quantity_axis
from nanoscope.gui.plot_interaction import axis_value_text


def test_linear_and_density_ticks_keep_data_and_inverse(qtbot):
    widget = pg.PlotWidget(axisItems={'bottom': ReadableLogAxis('bottom'), 'left': ReadableLogAxis('left')})
    qtbot.addWidget(widget)
    x = np.array([.001, .002, .004]); y = np.array([100., 200., 400.])
    curve = widget.plot(x, y)
    time = quantity_for('dwell_s')
    configure_quantity_axis(widget.getAxis('bottom'), AxisDisplayBasis(time))
    configure_quantity_axis(widget.getAxis('left'), AxisDisplayBasis(density_quantity([time])))
    assert _view_to_display(widget.getAxis('bottom'), .002) == 2.
    assert _display_to_view(widget.getAxis('bottom'), 2.) == .002
    assert axis_value_text(widget.getAxis('bottom'), .002) == '2'
    assert _view_to_display(widget.getAxis('left'), 200.) == .2
    assert _display_to_view(widget.getAxis('left'), .2) == 200.
    np.testing.assert_array_equal(curve.xData, x)
    np.testing.assert_array_equal(curve.yData, y)


def test_log_ticks_shift_decades_and_editor_round_trips(qtbot):
    axis = ReadableLogAxis('bottom')
    widget = pg.PlotWidget(axisItems={'bottom': axis})
    qtbot.addWidget(widget)
    axis.setLogMode(True)
    configure_quantity_axis(axis, AxisDisplayBasis(quantity_for('dwell_s'), 'log10'))
    assert _view_to_display(axis, -3.) == 1.
    assert _display_to_view(axis, 1.) == -3.
    assert axis_value_text(axis, -3.) == '1'


def test_native_witness_round_trip_style_isolation_and_all_history_states(qtbot):
    old = {'bottom': AxisDisplayBasis(quantity_for('dwell_s', display_unit='s'))}
    new = {'bottom': AxisDisplayBasis(quantity_for('dwell_s'))}
    widget = PlotAppearanceWidget()
    qtbot.addWidget(widget)
    widget.apply_appearance(PlotAppearance(title='literal (s)', axes=(('bottom', AxisPin(.001, .005)),)))
    widget.apply_appearance(PlotAppearance(title='typed (s)', axes=(('bottom', AxisPin(.002, .006)),)))
    migrated = widget.migrate_quantity_bases(new, legacy_axes=old)
    assert migrated.axis_pin('bottom').min == 2.
    assert migrated.title == 'typed (s)'
    assert PlotAppearance.from_mapping(migrated.to_dict()) == migrated
    assert migrated.style_only().quantity_display is None
    assert migrated.restyled(PlotAppearance(line_width=2)).quantity_display == migrated.quantity_display
    widget.undo()
    assert widget.state().axis_pin('bottom').min == 1.
    assert widget.state().title == 'literal (s)'
    widget.redo()
    assert widget.state() == migrated


def test_summary_frozen_geometry_and_literal_words_survive_ms_restore(qtbot, monkeypatch):
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
    from .test_p1_summaries_hosted import _source
    from nanoscope.analysis import distribution_summary
    page = DistributionSummaryWindow(); qtbot.addWidget(page)
    page.set_sources([_source()])
    assert page.analyze()
    raw = {key: frame.copy(deep=True) for key, frame in page._tables.items()}
    page._appearance_widget.apply_appearance(PlotAppearance(title='Literal seconds (s)',
        axes=(('bottom', AxisPin(.001, .005)),)))
    # An unmarked old appearance is seconds even though its literal title mentions s.
    page._apply_appearance()
    assert page._appearance.axis_pin('bottom').min == 1.
    assert page._plot.getAxis('bottom').scale == 1000.
    assert page._appearance.title == 'Literal seconds (s)'
    for key, frame in raw.items():
        assert page._tables[key].equals(frame)
    monkeypatch.setattr(distribution_summary, 'summarize_distribution', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('recomputed')))
    page._render_snapshot()
    assert page._appearance.axis_pin('bottom').min == 1.
    assert page._plot.getAxis('bottom').labelText == 'Dwell (ms)'
    for key, frame in raw.items():
        assert page._tables[key].equals(frame)



def test_csv_display_columns_keep_raw_time_and_density_values_and_names():
    import pandas as pd
    from nanoscope.gui.quantity_axis import export_display_columns
    raw = pd.DataFrame({'dwell_s': [.001, .002, np.nan], 'density': [100., 200., np.nan]})
    before = raw.copy(deep=True)
    duration = quantity_for('dwell_s')
    result = export_display_columns(raw, {'dwell_s': duration, 'density': density_quantity([duration])})
    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_frame_equal(result[['dwell_s', 'density']], before)
    np.testing.assert_allclose(result['dwell_s_display'], [1., 2., np.nan], equal_nan=True)
    np.testing.assert_allclose(result['density_display'], [.1, .2, np.nan], equal_nan=True)
    assert set(result['dwell_s_display_unit']) == {'ms'}
    assert set(result['density_display_unit']) == {'1/ms'}
    from nanoscope.core.quantity_display import QuantityError
    with pytest.raises(QuantityError, match='conflicts'):
        export_display_columns(result, {'dwell_s': duration})
