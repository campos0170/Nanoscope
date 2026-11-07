"""Series selection shares a point click without replacing exact event selection."""
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.gui.format_inspector import ElementRef
from . import p1_hosting as H
from . import p1_format_rig as F


def test_actual_scatter_click_selects_series_and_original_event_rows(qtbot):
    page=PlotsWindow(Session())
    frame=pd.DataFrame({'dwell_s':[.001,.003,.005],'depth_pA':[10.,40.,20.],
        'accepted':[True]*3},index=['first','second','third'])
    page.set_data({'apo-source':frame});page.set_plot_type('Scatter')
    page.set_scatter_columns('dwell_s','depth_pA');page._run_plot()
    explorer=H.hosted(qtbot,page,'distributions');qtbot.wait(400)
    QtWidgets.QApplication.processEvents()
    pane=explorer.inspector().pane('distributions').format_pane
    item=page._scatter_items[0];points=item.points();raw=tuple(v.copy() for v in item.getData())
    descriptor=next(d for d in page.figure_series() if d.role=='data')
    selected=[];page.eventRowsSelected.connect(lambda key,rows:selected.append((key,rows)))
    point=points[1];location=page._plot.getViewBox().mapViewToScene(point.pos())
    F.click_scene(page._plot,location)
    for _ in range(3):QtWidgets.QApplication.processEvents()
    assert pane._hub.current()==ElementRef('',descriptor.series_id)
    assert selected[-1][1]==('second',)
    assert point.data()=='second'
    for actual,expected in zip(item.getData(),raw):np.testing.assert_array_equal(actual,expected)
