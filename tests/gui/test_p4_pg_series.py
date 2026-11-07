"""A per-series edit changes only ink/visibility while retaining data and viewport."""
from dataclasses import replace
import numpy as np
import pyqtgraph as pg
import pytest
from nanoscope.gui.figure_series import tag_pg_series, pg_series_descriptors
from nanoscope.gui.plot_appearance import PlotAppearance, apply_plot_appearance


def test_series_overrides_are_independent_reversible_and_keep_raw_geometry(qtbot):
    widget=pg.PlotWidget();qtbot.addWidget(widget);plot=widget.getPlotItem()
    a=plot.plot([.001,.002],[1.,2.],pen=pg.mkPen('#0072b2',width=1),symbol='o')
    b=plot.plot([.001,.003],[3.,4.],pen=pg.mkPen('#d55e00',width=2))
    first=tag_pg_series(a,view='time',identity='source-a',identity_kind='source',label='A')
    second=tag_pg_series(b,view='time',identity='source-b',identity_kind='source',label='B')
    state=PlotAppearance(line_width=.75)
    apply_plot_appearance(plot,state,data_items=(a,b))
    original=pg.mkPen(a.opts['pen']);other=pg.mkPen(b.opts['pen'])
    arrays=[tuple(v.copy() for v in item.getData()) for item in (a,b)]
    plot.setXRange(.001,.002,padding=0);plot.setYRange(1,4,padding=0)
    limits=plot.viewRange()
    changed=replace(state,series_styles={first.series_id:{'color':'#aa2277','line_width_pt':2.,'visible':False}})
    apply_plot_appearance(plot,changed,data_items=(a,b))
    assert pg.mkPen(a.opts['pen']).color().name()=='#aa2277'
    assert pg.mkPen(a.opts['pen']).widthF()==pytest.approx(2.*96/72)
    assert not a.isVisible() and pg.mkPen(b.opts['pen'])==other
    assert [d.series_id for d in pg_series_descriptors((plot,))]==[first.series_id,second.series_id]
    assert plot.viewRange()==limits
    for item,old in zip((a,b),arrays):
        for actual,expected in zip(item.getData(),old):np.testing.assert_array_equal(actual,expected)
    apply_plot_appearance(plot,state,data_items=(a,b))
    assert a.isVisible() and pg.mkPen(a.opts['pen'])==original
    assert plot.viewRange()==limits
