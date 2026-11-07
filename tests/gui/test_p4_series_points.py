"""Category styling keeps exact scientific coordinates and attached row identities."""
import nanoscope  # noqa: F401
import numpy as np
import pyqtgraph as pg
import pytest
from nanoscope.gui.series_points import reset_point_sizes, apply_point_style


def test_masked_marker_style_and_hidden_state_never_replace_coordinates(qtbot):
    widget = pg.PlotWidget(); qtbot.addWidget(widget)
    item = pg.ScatterPlotItem(x=[.001, .002, .003], y=[1., 2., 3.], size=7,
        brush=pg.mkBrush(10, 20, 30, 80), data=['row-a', 'row-b', 'row-c'])
    widget.addItem(item)
    before = tuple(np.array(value, copy=True) for value in item.getData())
    reset_point_sizes(item, 7.)
    apply_point_style(item, [True, False, True], color='#abcdef', size=4.125, visible=False)
    assert [point.size() for point in item.points()] == pytest.approx([0., 7., 0.])
    assert item.points()[0].brush().color().alpha() == 80
    assert item.points()[0].brush().color().name() == '#abcdef'
    assert [point.data() for point in item.points()] == ['row-a', 'row-b', 'row-c']
    for left, right in zip(item.getData(), before): np.testing.assert_array_equal(left, right)
    reset_point_sizes(item, 7.)
    apply_point_style(item, [True, False, True], size=4.125)
    assert [point.size() for point in item.points()] == pytest.approx([4.125, 7., 4.125])
    with pytest.raises(ValueError, match='mask'):
        apply_point_style(item, [True], color='#ffffff')
