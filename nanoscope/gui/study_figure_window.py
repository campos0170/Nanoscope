"""Study canvases in NanoScope's shared styling, export and Figure Library tools."""
from copy import deepcopy
from pathlib import Path
import io
import json

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.project import WorkspaceArtifact
from ..core.figure_library import make_figure
from ..plot.study_figure_style import StudyFigureStyle
from .comparison_figures_panel import _ComparisonCanvas
from .matplotlib_figure_item import MatplotlibFigureItem
from .plot_appearance import PlotAppearance, PlotAppearanceWidget
from .figure_bundle import capture_figure_snapshot, write_figure_bundle
from .plot_export import choose_live_plot_export, write_live_plot_image


def _plotted_arrays(figure):
    """Portable coordinate evidence alongside each figure's source tables.

    These are numeric arrays, never executable objects. They include display
    envelopes/KDE paths which need not appear in the source event table.
    """
    arrays={};panels=[]
    def add(name,values):
        values=np.ma.asarray(values)
        if values.dtype.kind not in 'biufc':return None
        arrays[name]=np.asarray(values.filled(np.nan) if np.ma.isMaskedArray(values) and np.any(values.mask) else values)
        return name
    for i,axis in enumerate(figure.axes):
        panel=dict(index=i,x_scale=axis.get_xscale(),y_scale=axis.get_yscale(),
            x_lim=list(axis.get_xlim()),y_lim=list(axis.get_ylim()),lines=[],collections=[],images=[],patches=[])
        for j,line in enumerate(axis.lines):
            panel['lines'].append(dict(label=line.get_label(),x=add(f'a{i}_l{j}_x',line.get_xdata(orig=False)),y=add(f'a{i}_l{j}_y',line.get_ydata(orig=False))))
        for j,collection in enumerate(axis.collections):
            entry=dict(label=collection.get_label(),offsets=add(f'a{i}_c{j}_offsets',collection.get_offsets()))
            if collection.get_array() is not None:entry['values']=add(f'a{i}_c{j}_values',collection.get_array())
            entry['paths']=[add(f'a{i}_c{j}_path{k}',path.vertices) for k,path in enumerate(collection.get_paths())]
            panel['collections'].append(entry)
        for j,image in enumerate(axis.images):
            panel['images'].append(dict(values=add(f'a{i}_image{j}',image.get_array()),extent=list(image.get_extent()),origin=image.origin))
        for j,patch in enumerate(axis.patches):
            panel['patches'].append(dict(path=add(f'a{i}_patch{j}',patch.get_patch_transform().transform(patch.get_path().vertices))))
        panels.append(panel)
    out=io.BytesIO();np.savez_compressed(out,**arrays)
    return {'plotted_arrays.npz':out.getvalue(),'plotted_arrays.json':json.dumps(dict(version=1,panels=panels),default=str).encode()}


class StudyFigureWindow(QtWidgets.QMainWindow):
    """Own a reconstructed Figure; appearance changes never revisit study engines."""

    def __init__(self, figure, data, resources, *, name, provenance, style=None,
                 save_style=None, library_owner=None, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('NanoScope — Style study figure')
        screen=self.screen() or QtGui.QGuiApplication.primaryScreen()
        available=screen.availableGeometry() if screen is not None else QtCore.QRect(0,0,1490,1030)
        self.resize(min(1450,max(600,available.width()-40)),min(950,max(440,available.height()-80)))
        self._figure=figure; self._styler=StudyFigureStyle(figure)
        self._data=data.copy(deep=True)
        # Styling is recorded once, in the snapshot's current config. Imported
        # presentation sidecars may describe an earlier edit of this figure.
        self._resources={key:value for key,value in resources.items()
                         if not Path(key).name.endswith('.style.json')}
        self._resources.update(_plotted_arrays(figure))
        self._name=name; self._provenance=deepcopy(provenance)
        self._save_style=save_style; self._library_owner=library_owner
        self._style=deepcopy(style or {}); self._style.setdefault('appearance',{})
        self._style.setdefault('panels',{}); self._style.setdefault('colors',{})
        self._applied={}; self._item=None; self._loading=False
        central=QtWidgets.QWidget(); self.setCentralWidget(central)
        root=QtWidgets.QVBoxLayout(central)
        self._status=QtWidgets.QLabel(name); self._status.setWordWrap(True); root.addWidget(self._status)
        split=QtWidgets.QSplitter(); root.addWidget(split,1)
        controls=QtWidgets.QWidget(); layout=QtWidgets.QVBoxLayout(controls)
        scroll=QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setMinimumWidth(340); split.addWidget(scroll)
        self._appearance_widget=PlotAppearanceWidget(PlotAppearance.from_mapping(self._style['appearance']),
            compact=True,include_colors=False,include_labels=True)
        self._appearance_widget.x_label.hide();self._appearance_widget.y_label.hide()
        for label in self._appearance_widget.findChildren(QtWidgets.QLabel):
            if label.text() in ('X label','Y label'):label.hide()
        self._appearance_widget.changed.connect(self._appearance_changed); layout.addWidget(self._appearance_widget)
        panel_box=QtWidgets.QGroupBox('Panel titles and axis labels'); form=QtWidgets.QFormLayout(panel_box)
        self._panel=QtWidgets.QComboBox()
        for panel in self._styler.panels():
            title=(panel['title'] or panel['y_label'] or 'untitled').replace('\n',' ')
            self._panel.addItem(f"Panel {int(panel['key'])+1}: {title}",panel['key'])
        form.addRow('Panel',self._panel)
        self._panel_edits={}
        for key,label in [('title','Title'),('x_label','X label'),('y_label','Y label')]:
            edit=QtWidgets.QLineEdit(); self._panel_edits[key]=edit
            edit.textEdited.connect(lambda text,k=key:self._panel_changed(k,text)); form.addRow(label,edit)
        self._panel.currentIndexChanged.connect(self._show_panel); layout.addWidget(panel_box)
        self._title_notice=QtWidgets.QLabel();self._title_notice.setWordWrap(True)
        form.addRow(self._title_notice)
        color_box=QtWidgets.QGroupBox('Linked series colors'); color_layout=QtWidgets.QVBoxLayout(color_box)
        self._color=QtWidgets.QComboBox()
        for group in self._styler.color_groups():
            chip=QtGui.QPixmap(16,16);chip.fill(QtGui.QColor(group['color']))
            self._color.addItem(QtGui.QIcon(chip),group['label'],group['key'])
        color_layout.addWidget(self._color)
        color_btn=QtWidgets.QPushButton('Choose series color…'); color_btn.clicked.connect(self._choose_color); color_layout.addWidget(color_btn)
        note=QtWidgets.QLabel('Matching series, fills and legend keys keep the same color. Color changes do not change class assignments.')
        note.setWordWrap(True);color_layout.addWidget(note);layout.addWidget(color_box)
        for title,callback in [('Apply appearance',self.apply_appearance),('Reset all figure styling',self.reset_style),
                ('Fit figure to view',self.fit),('Save styling to study',self.save_to_study),
                ('Save to Figure Library…',self.save_to_library),('Export image…',self.export_image),
                ('Save figure bundle…',self.export_bundle)]:
            button=QtWidgets.QPushButton(title);button.clicked.connect(callback);layout.addWidget(button)
        layout.addStretch()
        # Long automatic titles belong in the dropdown, not in the column's
        # minimum width; the editor must fit without horizontal scrolling.
        for combo in controls.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        scene=QtWidgets.QGraphicsScene(self);self._view=_ComparisonCanvas(scene);split.addWidget(self._view);split.setSizes([380,1050])
        self._fit_timer=QtCore.QTimer(self);self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self.fit)
        self._show_panel();self.apply_appearance();self._fit_timer.start(0)

    def style(self): return deepcopy(self._style)

    def _draft_changed(self):
        self._status.setText(self._name+' — appearance draft. Apply appearance to preview, then Save styling to study.')

    def _appearance_changed(self,value):
        if self._loading:return
        self._style['appearance']=value.to_dict();self._draft_changed()

    def _show_panel(self,*args):
        self._loading=True
        key=self._panel.currentData(); values=dict(self._style['panels'].get(key,{}))
        original=next((p for p in self._styler.panels() if p['key']==key),{})
        event_identity=original.get('event_identity')
        if 'event_identity' in original:
            values.pop('title',None)
            if event_identity in self._style.get('event_titles',{}):
                values['title']=self._style['event_titles'][event_identity]
        self._panel_edits['title'].setEnabled('event_identity' not in original or bool(event_identity))
        for name,edit in self._panel_edits.items():
            edit.setText(values.get(name,''));edit.setPlaceholderText(original.get(name,'') or 'Original label')
        self._loading=False
        self._update_title_notice()

    def _update_title_notice(self):
        panels=self._styler.panels()
        legacy=any('event_identity' in p and 'title' in self._style['panels'].get(p['key'],{}) for p in panels)
        current={p.get('event_identity') for p in panels}
        stale=set(self._style.get('event_titles',{}))-current
        notices=[]
        if legacy:notices.append('Older event titles were saved without event identity and are not applied. Enter a title again to associate it with the displayed event.')
        if stale:notices.append('Annotations for events that changed or are absent are not applied. The automatic event labels remain visible.')
        self._title_notice.setText(' '.join(notices));self._title_notice.setVisible(bool(notices))

    def _panel_changed(self,key,text):
        if self._loading:return
        panel_key=self._panel.currentData()
        original=next((p for p in self._styler.panels() if p['key']==panel_key),{})
        if key=='title' and 'event_identity' in original:
            if not original['event_identity']:return
            self._style.setdefault('event_titles',{})[original['event_identity']]=text
            self._style['panels'].get(panel_key,{}).pop('title',None)
        else:self._style['panels'].setdefault(panel_key,{})[key]=text
        self._update_title_notice()
        self._draft_changed()

    def _choose_color(self):
        key=self._color.currentData()
        if key is None:return
        color=QtWidgets.QColorDialog.getColor(QtGui.QColor(self._style['colors'].get(key,key)),self,'Series color')
        if color.isValid():self._style['colors'][key]=color.name();self._draft_changed()

    def apply_appearance(self):
        try:
            self._styler.apply(**self._style)
            item=MatplotlibFigureItem(self._figure)
            scene=self._view.scene();scene.clear();scene.addItem(item);scene.setSceneRect(item.boundingRect())
            self._item=item;self._applied=self.style()
            self._status.setText(self._name+' — appearance preview. Analytical values and event selections are unchanged.')
            return True
        except Exception as exc:
            self._status.setText('Appearance could not be applied: '+str(exc));return False

    def reset_style(self):
        self._style={'appearance':{},'panels':{},'colors':{}}
        self._appearance_widget.set_state({});self._show_panel();self.apply_appearance()

    def fit(self):
        if self._item is not None:self._view.fitInView(self._item.boundingRect(),QtCore.Qt.KeepAspectRatio)

    def snapshot(self):
        if self._applied!=self._style and not self.apply_appearance():raise ValueError(self._status.text())
        return capture_figure_snapshot(self._item,data=self._data,
            config={'kind':'manual-study-figure','version':1,'figure':self._name,'style':self.style()},
            provenance={**self._provenance,'appearance_only':True,'analysis_recomputed':False},
            additional_resources={'study_data/'+name:bytes(value) for name,value in self._resources.items()})

    def save_to_study(self):
        try:
            if self._save_style is None:raise ValueError('Open this figure from its saved study workspace.')
            snapshot=self.snapshot(); self._save_style(self.style(),snapshot,self)
        except Exception as exc:self._status.setText('Styling was not saved: '+str(exc))

    def saved(self):self._status.setText(self._name+' — styling saved to the study workspace.')

    def save_to_library(self,checked=False,*,name=None):
        try:
            owner=self._library_owner
            if owner is None or not hasattr(owner,'_append_figure'):
                raise ValueError('Open the study from the main NanoScope window to use its Figure Library.')
            if name is None:
                name,ok=QtWidgets.QInputDialog.getText(self,'Save to Figure Library','Figure name:',text=self._name)
                if not ok:return
            snapshot=self.snapshot();metadata,blobs=snapshot.to_storage()
            analysis=WorkspaceArtifact(artifact_id='manual-study.figure',kind='manual-study.figure',schema_version=1,
                payload={'figure':self._name,'style':self.style(),'provenance':deepcopy(self._provenance)},
                tables={'plotted_data':self._data.copy(deep=True)})
            entry=make_figure(name=name,page_id='manual-study',source_label=str(self._provenance.get('study','Study')),
                analysis=analysis,snapshot_metadata=metadata,snapshot_blobs=blobs,live_state=None,capture_status='saved')
            owner._append_figure(entry);owner._mark_dirty();owner._open_figure_library();owner._figure_library.refresh(entry.artifact_id)
            self._status.setText('Added to Figure Library. Save Project in the main window to persist the library entry.')
            return entry
        except Exception as exc:self._status.setText('Figure was not added: '+str(exc));return None

    def export_image(self,checked=False,*,path=None):
        try:
            if self._applied!=self._style and not self.apply_appearance():return
            if path is None:
                selection=choose_live_plot_export(self,self._item,self._name+'.png')
                if selection is None:return
                path,options=selection
                kwargs=dict(width=options.width,height=options.height,
                    physical_width_mm=options.physical_width_mm,physical_height_mm=options.physical_height_mm,
                    frozen_snapshot=options.frozen_snapshot)
                if options.extension=='.png' and not options.native:kwargs['dpi']=options.dpi
                write_live_plot_image(path,self._item,**kwargs)
            else:write_live_plot_image(path,self._item)
            self._status.setText('Exported '+str(path));return Path(path)
        except Exception as exc:self._status.setText('Image was not exported: '+str(exc));return None

    def export_bundle(self,checked=False,*,path=None):
        try:
            if path is None:
                path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save figure bundle',self._name+'.zip','Figure bundle (*.zip)')
                if not path:return
            result=write_figure_bundle(path,self.snapshot());self._status.setText('Exported '+str(result));return result
        except Exception as exc:self._status.setText('Bundle was not exported: '+str(exc));return None

    def closeEvent(self,event):
        self._fit_timer.stop();self._item=None
        self._figure.clear();self._resources.clear();self._save_style=None
        self._styler=None;self._data=None
        super().closeEvent(event)
