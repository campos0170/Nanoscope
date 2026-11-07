"""Frame edits and palette controls preserve literal words, arrays and history."""
from dataclasses import replace

import numpy as np
import pytest
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from nanoscope.core.figure_style import FigureStyle, style_binding
from nanoscope.gui.project_style_editor import ProjectStyleEditor
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget, apply_plot_appearance


def test_style_editor_palette_and_frame_are_real_validated_look_fields(qtbot):
    original = FigureStyle('a', 1, 'A', look={'palette': ['#AABBCC'], 'frame': False,
                                            'future': {'words': ['unchanged']}})
    dialog = ProjectStyleEditor(original)
    qtbot.addWidget(dialog)
    assert dialog.look() == original.to_dict()['look']
    dialog.frame.setCurrentIndex(dialog.frame.findData(True))
    dialog.palette.setText('#112233, #445566')
    assert dialog.look() == {**original.to_dict()['look'], 'frame': True,
                             'palette': ['#112233', '#445566']}
    dialog.palette.setText('#invalid')
    with pytest.raises(ValueError): dialog.look()
    dialog.accept()
    assert 'valid colors' in dialog.error.text()
    dialog.palette.clear()
    dialog.frame.setCurrentIndex(dialog.frame.findData(None))
    assert dialog.look() == {'future': {'words': ['unchanged']}}


def test_native_frame_undo_restores_border_without_changing_axes_or_arrays(qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    plot.setLabel('bottom', 'Literal time (s)')
    plot.setLabel('left', 'Literal current')
    item = plot.plot([1., 2., 3.], [4., 5., 6.])
    plot.setRange(xRange=(1.1, 2.8), yRange=(4.1, 5.8), padding=0)
    old_border = QtGui.QPen(plot.getViewBox().border)
    ranges = [list(value) for value in plot.viewRange()]
    visible = {side: plot.getAxis(side).isVisible() for side in ('top','right','bottom','left')}
    appearance = PlotAppearanceWidget(state=PlotAppearance(title='Literal'))
    qtbot.addWidget(appearance)
    appearance.changed.connect(lambda state: apply_plot_appearance(plot, state, plot_widget=widget,
                                                                   data_items=(item,)))
    appearance.apply_appearance(replace(appearance.state(), frame=True))
    assert plot.getViewBox().border.style() != QtCore.Qt.NoPen
    assert appearance.undo()
    assert appearance.state().frame is None
    assert plot.getViewBox().border == old_border
    appearance.apply_appearance(replace(appearance.state(), frame=False))
    assert plot.getViewBox().border.style() == QtCore.Qt.NoPen
    assert {side: plot.getAxis(side).isVisible() for side in visible} == visible
    np.testing.assert_array_equal(item.xData, [1., 2., 3.])
    np.testing.assert_array_equal(item.yData, [4., 5., 6.])
    np.testing.assert_allclose(plot.viewRange(), ranges, atol=1e-12, rtol=0)
    assert plot.getAxis('bottom').labelText == 'Literal time (s)'
    assert plot.getAxis('left').labelText == 'Literal current'
