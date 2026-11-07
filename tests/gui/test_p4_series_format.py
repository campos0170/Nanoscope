"""Per-series controls edit presentation only, with independent Undo and copy scope."""
from dataclasses import replace
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from nanoscope.core.figure_style import builtin_styles, style_binding
from nanoscope.gui.figure_series import tag_pg_series
from nanoscope.gui.format_inspector import ElementRef
from nanoscope.gui.plot_appearance import apply_plot_appearance
from . import p1_format_rig as F
from . import p1_hosting as H


def settle():
    for _ in range(3):QtWidgets.QApplication.processEvents()


def rig(qtbot):
    window=F.bare(qtbot);plot=window.plots[0];first=plot.listDataItems()[0]
    first.setSymbol('o');first.setPen('#112233');first.setSymbolBrush('#112233')
    second=plot.plot([0,1,2],[2.,1.,3.],pen='#556677',symbol='o',symbolBrush='#556677',name='second')
    ids=[tag_pg_series(item,view='sample',identity=key,identity_kind='source',label=label).series_id
         for item,key,label in ((first,'a','First source'),(second,'b','Second source'))]
    window.widget.changed.disconnect(window._applied)
    def draw(state):
        apply_plot_appearance(plot,state,data_items=(first,second))
        window.controllers[0].refresh()
    window.widget.changed.connect(draw)
    draw(window.state);window.pane.refresh();settle();window.pane._rebuild()
    window.widget.clear_history()
    return window,(first,second),ids


def test_selected_series_controls_change_one_target_and_undo_without_recompute(qtbot):
    window,items,ids=rig(qtbot);pane=window.pane
    arrays=[tuple(v.copy() for v in item.getData()) for item in items]
    F.pick_row(pane.list,ElementRef('',ids[0]));panel=pane.current_panel()
    assert panel.series_box.isVisible() and not panel.global_box.isVisible()
    panel.line_width.setValue(2.);settle()
    assert window.state.series_styles[ids[0]]['line_width_pt']==2.
    assert ids[1] not in window.state.series_styles
    assert pg.mkPen(items[0].opts['pen']).widthF()==2.*96/72
    assert window.controllers[0]._nanoscope_outline.isVisible()
    outline=window.controllers[0]._nanoscope_outline
    assert outline.width()>100 and outline.height()>100  # curve children, not container origin
    assert window.widget.history_mark()==1 and window.widget.undo()
    settle();assert not window.state.series_styles
    for item,expected in zip(items,arrays):
        for actual,old in zip(item.getData(),expected):np.testing.assert_array_equal(actual,old)


def test_inherited_visibility_is_explicit_and_show_hide_remain_undoable(qtbot):
    window,items,ids=rig(qtbot);pane=window.pane;pane.show_element(ids[0]);panel=pane.current_panel()
    assert panel.visible.currentText()=='From figure' and items[0].isVisible()
    panel.visible.setCurrentIndex(panel.visible.findData(False));settle()
    assert panel.visible.currentText()=='Hide' and not items[0].isVisible()
    assert window.widget.undo();settle()
    assert panel.visible.currentText()=='From figure' and items[0].isVisible()


def test_series_copy_paste_and_reset_preserve_identity_visibility_and_words(qtbot):
    window,items,ids=rig(qtbot);pane=window.pane
    window.widget.apply_appearance(replace(window.state,title='Typed title',series_styles={ids[0]:{
        'color':'#aa2277','line_width_pt':2.,'marker_size_pt':5.,'visible':False}}))
    settle();pane.show_element(ids[0]);pane.copy_format();pane.show_element(ids[1]);pane.paste_format();settle()
    own=window.state.series_styles[ids[1]]
    assert dict(own)=={'color':'#aa2277','line_width_pt':2.,'marker_size_pt':5.}
    assert items[1].isVisible() and not items[0].isVisible() and window.state.title=='Typed title'
    pane.reset_element();settle()
    assert ids[1] not in window.state.series_styles and ids[0] in window.state.series_styles
    assert window.widget.undo();settle();assert ids[1] in window.state.series_styles
    pane.reset_look();settle()
    assert dict(window.state.series_styles[ids[0]])=={'visible':False}
    assert ids[1] not in window.state.series_styles and window.state.title=='Typed title'


def test_explicit_palette_is_one_figure_local_step_and_does_not_rewrite_identity_colours(qtbot):
    window,items,ids=rig(qtbot);pane=window.pane
    base=builtin_styles()[0]
    style=replace(base,look={**dict(base.look),'palette':['#00aa00','#aa00aa']})
    window.widget.apply_appearance(replace(window.state,style_binding=style_binding(style)))
    settle();window.widget.clear_history();pane.show_element('series.data')
    before=tuple(d.color for d in pane.series_descriptors())
    pane.current_panel().palette.click();settle()
    assert [window.state.series_styles[k]['color'] for k in ids]==['#00aa00','#aa00aa']
    assert tuple(d.color for d in pane.series_descriptors())==before
    assert window.widget.history_mark()==1 and window.widget.undo()
    settle();assert not window.state.series_styles


def test_curve_click_selects_same_object_row_without_extra_analysis(qtbot):
    window,items,ids=rig(qtbot);pane=window.pane
    plot=window.plots[0];point=plot.getViewBox().mapViewToScene(QtCore.QPointF(.5,2.))
    F.click_scene(plot,point);settle()
    assert window.hub.current()==ElementRef('',ids[0])
    assert pane.list.ref_of(pane.list.currentItem())==ElementRef('',ids[0])
    assert window.widget.history_mark()==0


def test_unbound_figure_copy_between_renderers_preserves_physical_sizes(qtbot):
    from nanoscope.gui.plot_appearance import PlotAppearanceWidget, PlotAppearance
    source,items,ids=rig(qtbot)
    source.widget.apply_appearance(replace(source.state,line_width=2.,marker_size=8.))
    source.pane.show_element('figure');source.pane.copy_format()
    destination=F.bare(qtbot,appearance=PlotAppearanceWidget(PlotAppearance(title='Keep words'),
        style_vocabulary='matplotlib_appearance'))
    destination.pane._clipboard=source.pane._clipboard
    destination.pane.show_element('figure');destination.pane.paste_format()
    assert destination.state.line_width==1.5 and destination.state.marker_size==6.
    assert destination.state.title=='Keep words'
    destination.pane.copy_format();source.pane.paste_format()
    assert source.state.line_width==2. and source.state.marker_size==8.


def test_series_copy_uses_current_ink_and_native_width_instead_of_old_descriptor(qtbot):
    window,items,ids=rig(qtbot);pane=window.pane
    window.widget.apply_appearance(replace(window.state,data_color='#aa1234',line_width=3.))
    settle();pane.show_element(ids[0]);pane.copy_format();pane.show_element(ids[1]);pane.paste_format()
    own=window.state.series_styles[ids[1]]
    assert own['color']=='#aa1234' and own['line_width_pt']==2.25
    assert pg.mkPen(items[1].opts['pen']).widthF()==3.


def test_retired_pane_releases_native_series_receiver(qtbot):
    window,items,ids=rig(qtbot)
    emitter=window.plots[0]._nanoscope_series_selection
    assert emitter.receivers(QtCore.SIGNAL('selected(QString)'))==1
    window.pane.retire();window.pane.retire()
    assert emitter.receivers(QtCore.SIGNAL('selected(QString)'))==0


def test_legend_rows_show_the_display_label_and_keep_the_storage_identity(qtbot):
    from nanoscope.gui.figure_text import TextStyle
    window=F.bare(qtbot);controller=window.controllers[0]
    original=controller.host.text_targets
    key='legend.entry.recording://opaque-source-id/event-table'
    def targets():
        values=original();target=values.pop('legend.entry.s')
        values[key]=replace(target,auto='500 mV',style=TextStyle(text='Apo-ferritin 500 mV'))
        return values
    controller.host.text_targets=targets
    controller.refresh();window.pane._rebuild();settle()
    item=window.pane.list.find(ElementRef('',key))
    assert item is not None and item.text(0)=='Apo-ferritin 500 mV'
    F.pick_row(window.pane.list,ElementRef('',key))
    assert window.hub.current()==ElementRef('',key)
