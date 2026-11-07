"""Study canvases in NanoScope's shared styling, export and Figure Library tools."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import io
import json

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.project import WorkspaceArtifact
from ..core.figure_history_state import HistoryCapsule
from ..core.figure_spec import thaw_json
from ..core.figure_style import builtin_styles
from ..core.figure_library import make_figure
from ..plot.study_figure_style import StudyFigureStyle
from .comparison_figures_panel import _ComparisonCanvas
from .export_paths import ask_export_path
from .matplotlib_figure_item import MatplotlibFigureItem
from .plot_appearance import PlotAppearance, PlotAppearanceWidget
from .format_inspector import FormatPane, SelectionHub
from .plot_workflow import FormatSpec
from .matplotlib_text_controller import synchronize_controllers
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
    from ..plot.quantity_axes import figure_quantity_metadata
    display_basis = figure_quantity_metadata(figure)
    out=io.BytesIO();np.savez_compressed(out,**arrays)
    return {'plotted_arrays.npz':out.getvalue(),'plotted_arrays.json':json.dumps(dict(version=1,panels=panels,display_basis=display_basis),default=str).encode()}


class _StudyAppearanceWidget(PlotAppearanceWidget):
    """The shared controls delegate history to the complete Study envelope owner."""
    def __init__(self, owner, state):
        self._study_owner = owner
        super().__init__(state, compact=True, include_colors=False, include_labels=True,
                         history=False, shortcut_parent=owner, style_vocabulary="matplotlib_appearance")
        self.style_vocabulary = 'matplotlib_appearance'
        self.undo_button = QtWidgets.QPushButton('Undo')
        self.redo_button = QtWidgets.QPushButton('Redo')
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)

    def undo(self): return self._study_owner.undo_style()
    def redo(self): return self._study_owner.redo_style()
    def history_mark(self): return self._study_owner.history_mark()
    def _close_burst(self): self._study_owner._commit_style_run()
    def capture_history_capsule(self):
        self._close_burst()
        return self._study_owner._style_history
    def restore_history_capsule(self, capsule):
        history = capsule if isinstance(capsule, HistoryCapsule) else HistoryCapsule.from_dict(capsule)
        if history.codec != 'study-figure-style-v1':
            raise ValueError('This history does not belong to a Study figure')
        if not self._study_owner._install_style(thaw_json(history.base)):
            raise ValueError('Could not restore this figure style history')
        self._study_owner._style_history = history
        self._study_owner._sync_style_history()


class StudyFigureWindow(QtWidgets.QMainWindow):
    """Own a reconstructed Figure; appearance changes never revisit study engines.

    Never ``WA_DeleteOnClose`` (see build_info_dialog.py's docstring for why). Closing
    hides the editor and frees what it held (see ``closeEvent``); the study window that
    opened it disposes of it when the next editor opens. A save still running when the
    user closes the editor reports to it through ``saved()``, which a deleted window
    turned into a failure after a save that had succeeded.
    """

    #: Emitted from ``closeEvent``, so an owner can drop the editor from its open ones.
    closed = QtCore.Signal()
    formatViewsChanged = QtCore.Signal()

    def __init__(self, figure, data, resources, *, name, provenance, style=None,
                 save_style=None, library_owner=None, parent=None, available_styles=None,
                 owner_kind='study'):
        super().__init__(parent)
        self._owner_kind = owner_kind
        self.setWindowTitle('NanoScope — Style study figure' if owner_kind == 'study' else 'NanoScope — Style report figure')
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
        self._style=self._styler.canonical_style(style); self._style.setdefault('appearance',{})
        self._style.setdefault('panels',{}); self._style.setdefault('colors',{})
        self._applied={}; self._item=None; self._loading=False
        self._style_history = HistoryCapsule('study-figure-style-v1', self._style)
        self._text_controllers = []
        self._available_styles = available_styles
        self._style_timer = QtCore.QTimer(self); self._style_timer.setSingleShot(True)
        self._style_timer.setInterval(400); self._style_timer.timeout.connect(self._commit_style_run)
        central=QtWidgets.QWidget(); self.setCentralWidget(central)
        root=QtWidgets.QVBoxLayout(central)
        self._status=QtWidgets.QLabel(name); self._status.setWordWrap(True); root.addWidget(self._status)
        split=QtWidgets.QSplitter(); root.addWidget(split,1)
        controls=QtWidgets.QWidget(); layout=QtWidgets.QVBoxLayout(controls)
        scroll=QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setMinimumWidth(340)
        initial_appearance = {**self._style['appearance']}
        if self._style.get('style_binding'):
            initial_appearance['style_binding'] = self._style['style_binding']
        self._appearance_widget=_StudyAppearanceWidget(self, PlotAppearance.from_mapping(initial_appearance))
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
            edit.textEdited.connect(lambda text,k=key:self._panel_changed(k,text))
            edit.editingFinished.connect(self._commit_style_run); form.addRow(label,edit)
        self._panel.currentIndexChanged.connect(self._commit_style_run)
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
        # Format is the sole visible editing route. Keep legacy widgets detached
        # from the user flow for saved-workspace compatibility and old callers.
        panel_box.hide(); color_box.hide()
        form.removeWidget(self._title_notice); layout.addWidget(self._title_notice)
        format_note = QtWidgets.QLabel('Edit titles, axis labels and series colors in Format. Select an element on the figure or in the Format list.')
        format_note.setWordWrap(True); layout.addWidget(format_note)
        format_button = QtWidgets.QPushButton('Open Format')
        layout.addWidget(format_button)
        for title,callback in [('Apply appearance',self.apply_appearance),('Reset all figure styling',self.reset_style),
                ('Fit figure to view',self.fit),('Save styling to study',self.save_to_study),
                ('Save to Figure Library…',self.save_to_library),('Export image…',self.export_image),
                ('Save figure bundle…',self.export_bundle)]:
            button=QtWidgets.QPushButton(title);button.clicked.connect(callback);layout.addWidget(button)
            if title == 'Save styling to study': button.setVisible(owner_kind == 'study')
        layout.addStretch()
        # Long automatic titles belong in the dropdown, not in the column's
        # minimum width; the editor must fit without horizontal scrolling.
        for combo in controls.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        scene=QtWidgets.QGraphicsScene(self);self._view=_ComparisonCanvas(scene);split.addWidget(self._view)
        self._view.setAccessibleName('Study figure canvas' if owner_kind == 'study' else 'Report figure canvas')
        from .matplotlib_series import MatplotlibSeriesSelection
        self._series_selection = MatplotlibSeriesSelection(self._view, self)
        self._selection_hub = SelectionHub(self)
        self._format_pane = FormatPane(FormatSpec(self._appearance_widget,
            controllers=lambda: self._text_controllers,
            views_changed=self.formatViewsChanged,
            series_provider=self._styler.series_descriptors,
            series_selected=self._series_selection.selected,
            series_select=self._series_selection.select), self._selection_hub)
        self.formatViewsChanged.connect(self._format_pane.refresh)
        inspector = QtWidgets.QTabWidget()
        inspector.addTab(self._format_pane, 'Format')
        inspector.addTab(scroll, 'Study details' if owner_kind == 'study' else 'Figure details')
        format_button.clicked.connect(lambda: inspector.setCurrentIndex(0))
        inspector.setMinimumWidth(320); inspector.setMaximumWidth(430)
        split.addWidget(inspector);split.setSizes([1050,380])
        toolbar = self.addToolBar('Figure')
        for button in (self._appearance_widget.undo_button, self._appearance_widget.redo_button):
            toolbar.addWidget(button)
        for title, shortcut, action in [('Undo',QtGui.QKeySequence.Undo,self.undo_style),
                                         ('Redo',QtGui.QKeySequence.Redo,self.redo_style)]:
            command = QtGui.QAction(title,self); command.setShortcut(shortcut)
            command.triggered.connect(action); self.addAction(command)
        style_button = QtWidgets.QToolButton(); style_button.setText('Style')
        style_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        style_menu = QtWidgets.QMenu(style_button); style_button.setMenu(style_menu)
        style_menu.aboutToShow.connect(lambda: self._fill_style_menu(style_menu))
        toolbar.addWidget(style_button)
        toolbar.addAction('Fit figure',self.fit)
        toolbar.addAction('Export image…',self.export_image)
        toolbar.addAction('Save figure bundle…',self.export_bundle)
        if owner_kind == 'study': toolbar.addAction('Save styling to study',self.save_to_study)
        toolbar.addAction('Save snapshot to project…',self.save_to_library)
        self._fit_timer=QtCore.QTimer(self);self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self.fit)
        self._show_panel();self.apply_appearance();self._sync_style_history();self._fit_timer.start(0)

    def style(self): return deepcopy(self._style)

    def _draft_changed(self):
        if self.apply_appearance():
            self._style_timer.start()
            self._sync_style_history()

    @QtCore.Slot()
    def _commit_style_run(self):
        if not hasattr(self, '_style_history') or self._loading:
            return
        self._style_timer.stop()
        self._style_history = self._style_history.note(self._style)
        self._sync_style_history()

    def history_mark(self):
        history = self._style_history
        return len(history.past) + history.dropped + int(thaw_json(history.base) != self._style)

    def history(self):
        """Shared-toolbar/batch history contract for a retained editor."""
        from .plot_workflow import HistoryState
        pending = thaw_json(self._style_history.base) != self._style
        undo = bool(self._style_history.past) or pending
        redo = bool(self._style_history.future) and not pending
        return HistoryState(undo, redo, 'Undo figure styling' if undo else 'Nothing to undo',
            'Redo figure styling' if redo else 'Nothing to redo', self.history_mark())

    def _sync_style_history(self):
        widget = getattr(self, '_appearance_widget', None)
        if widget is None or widget.undo_button is None:
            return
        pending = thaw_json(self._style_history.base) != self._style
        widget.undo_button.setEnabled(bool(self._style_history.past) or pending)
        widget.redo_button.setEnabled(bool(self._style_history.future) and not pending)
        widget.undo_button.setToolTip('Undo figure styling' if widget.undo_button.isEnabled() else 'Nothing to undo')
        widget.redo_button.setToolTip('Redo figure styling' if widget.redo_button.isEnabled() else 'Nothing to redo')
        widget.historyChanged.emit()

    def _install_style(self, style):
        previous = self._style
        self._style = deepcopy(style)
        if not self.apply_appearance():
            self._style = previous
            self.apply_appearance()
            return False
        native = {**self._style.get('appearance', {})}
        if self._style.get('style_binding'): native['style_binding'] = self._style['style_binding']
        self._loading = True
        try: self._appearance_widget.set_state(native)
        finally: self._loading = False
        self._show_panel()
        return True

    def _move_history(self, direction):
        self._commit_style_run()
        history = getattr(self._style_history, direction)()
        if history == self._style_history or not self._install_style(thaw_json(history.base)):
            return False
        self._style_history = history
        self._sync_style_history()
        return True

    def undo_style(self): return self._move_history('undo')
    def redo_style(self): return self._move_history('redo')

    def _fill_style_menu(self, menu):
        menu.clear()
        offered = list(builtin_styles())
        provider = self._available_styles
        if provider is None:
            workspace = getattr(self._library_owner, '_figure_workspace', None)
            provider = getattr(workspace, 'available_styles', None)
        if callable(provider):
            offered.extend(provider())
        seen = set()
        for style in offered:
            identity = (style.style_id, style.revision)
            if identity in seen: continue
            seen.add(identity)
            menu.addAction(style.name, lambda checked=False, value=style: self.apply_style(value))
        from .computer_styles import styles
        menu.addSection('On this computer')
        try:
            local = styles()
        except ValueError as exc:
            menu.addAction(str(exc)).setEnabled(False)
        else:
            if not local:
                menu.addAction('None saved on this computer').setEnabled(False)
            for style in local:
                menu.addAction(style.name, lambda checked=False, value=style: self.apply_style(value))

    def apply_style(self, style, *, keep_hand=True):
        from .figure_style_binding import bind_native_style
        return self.apply_look(PlotAppearance.from_mapping(bind_native_style({}, style,
            'matplotlib_appearance')), keep_hand=keep_hand)

    def apply_look(self, look, keep_hand=True):
        """Shared batch-style hook over this retained editor's complete Undo owner."""
        if self._figure is None:
            return False
        self._commit_style_run()
        updated = deepcopy(self._style)
        native = self._appearance_widget.state().restyled(look, keep_hand=keep_hand,
            vocabulary='matplotlib_appearance').to_dict()
        binding = native.pop('style_binding', None)
        updated['appearance'] = native
        if binding is not None:
            updated['style_binding'] = binding
        else:
            updated.pop('style_binding', None)
        if updated == self._style:
            return False
        if not self._install_style(updated): return False
        self._commit_style_run()
        return True

    @QtCore.Slot(object)
    def _appearance_changed(self,value):
        if self._loading:return
        native=value.to_dict(); binding=native.pop('style_binding',None)
        self._style['appearance']=native
        if binding is not None: self._style['style_binding']=binding
        else: self._style.pop('style_binding',None)
        self._draft_changed()

    def _show_panel(self,*args):
        self._loading=True
        key=self._panel.currentData(); values=dict(self._style['panels'].get(key,{}))
        original=next((p for p in self._styler.panels() if p['key']==key),{})
        event_identity=original.get('event_identity')
        if 'event_identity' in original:
            values.pop('title',None)
            if event_identity in self._style.get('event_titles',{}):
                values['title']=self._style['event_titles'][event_identity]
        from ..plot.matplotlib_targets import panel_identity, storage_key
        if key is not None and int(key) < len(self._figure.axes):
            panel_id = panel_identity(self._figure.axes[int(key)], int(key))
            for name, role in (('title', 'title'), ('x_label', 'axis.bottom.label'), ('y_label', 'axis.left.label')):
                wording = self._appearance_widget.state().text_style(storage_key(panel_id, role)).text
                if wording is not None:
                    values[name] = wording
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
        from ..plot.matplotlib_targets import artist_targets, storage_key
        current_text = {storage_key(panel, key) for panel, key, *_rest in artist_targets(self._figure)}
        stale.update(key for key in self._style.get('appearance', {}).get('text', {})
                     if key.startswith('annotation.mpl.') and key not in current_text)
        notices=[]
        if legacy:notices.append('Older event titles were saved without event identity and are not applied. Enter a title again to associate it with the displayed event.')
        if stale:notices.append('Annotations for events that changed or are absent are not applied. The automatic event labels remain visible.')
        self._title_notice.setText(' '.join(notices));self._title_notice.setVisible(bool(notices))

    def _panel_changed(self,key,text):
        if self._loading:return
        panel_key=self._panel.currentData()
        if panel_key is None:return
        from ..plot.matplotlib_targets import panel_identity, storage_key
        panel = panel_identity(self._figure.axes[int(panel_key)], int(panel_key))
        role = {'title':'title', 'x_label':'axis.bottom.label', 'y_label':'axis.left.label'}[key]
        state = self._appearance_widget.state()
        target = storage_key(panel, role)
        self._appearance_widget.edit_text_style(target, replace(state.text_style(target), text=text or None))
        self._update_title_notice()

    def _choose_color(self):
        key=self._color.currentData()
        if key is None:return
        self._commit_style_run()
        color=QtWidgets.QColorDialog.getColor(QtGui.QColor(self._style['colors'].get(key,key)),self,'Series color')
        if color.isValid():
            from ..core.figure_series import stable_series_id
            state = self._appearance_widget.state()
            series = thaw_json(state.series_styles or {})
            series.setdefault(stable_series_id('study', 'data', key), {})['color'] = color.name()
            self._appearance_widget.apply_appearance(replace(state, series_styles=series))

    def apply_appearance(self):
        try:
            self._styler.apply(**self._style)
            item=MatplotlibFigureItem(self._figure)
            scene=self._view.scene()
            view = QtGui.QTransform(self._view.transform())
            horizontal = self._view.horizontalScrollBar().value()
            vertical = self._view.verticalScrollBar().value()
            scene.clear();scene.addItem(item);scene.setSceneRect(item.boundingRect())
            self._item=item;self._applied=self.style()
            self._series_selection.set_item(item)
            self._view.setTransform(view)
            self._view.horizontalScrollBar().setValue(horizontal)
            self._view.verticalScrollBar().setValue(vertical)
            self._text_controllers = synchronize_controllers(self, item, self._appearance_widget,
                self._view, self._text_controllers)
            self.formatViewsChanged.emit()
            self._status.setText(self._name+' — appearance preview. Analytical values and event selections are unchanged.')
            return True
        except Exception as exc:
            self._status.setText('Appearance could not be applied: '+str(exc));return False

    def reset_style(self):
        self._commit_style_run()
        reset = deepcopy(self._style)
        for key in ('appearance','panels','colors'):
            reset[key] = {}
        reset.pop('event_titles',None)
        reset.pop('style_binding',None)
        if self._install_style(reset): self._commit_style_run()

    def fit(self):
        if self._item is not None:self._view.fit(self._item.boundingRect())

    def snapshot(self):
        self._commit_style_run()
        if self._applied!=self._style and not self.apply_appearance():raise ValueError(self._status.text())
        return capture_figure_snapshot(self._item,data=self._data,
            config={'kind':'manual-study-figure' if self._owner_kind == 'study' else 'report-figure','version':1,'figure':self._name,'style':self.style()},
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
            epoch = getattr(owner, '_figure_project_epoch', None)
            if name is None:
                name,ok=QtWidgets.QInputDialog.getText(self,'Save to Figure Library','Figure name:',text=self._name)
                if not ok:return
            if getattr(owner, '_figure_project_epoch', None) != epoch:
                raise ValueError('The project changed while the figure name was open; nothing was saved.')
            snapshot=self.snapshot();metadata,blobs=snapshot.to_storage()
            artifact_kind = 'manual-study.figure' if self._owner_kind == 'study' else 'report.figure'
            analysis=WorkspaceArtifact(artifact_id=artifact_kind,kind=artifact_kind,schema_version=1,
                payload={'figure':self._name,'style':self.style(),'provenance':deepcopy(self._provenance)},
                tables={'plotted_data':self._data.copy(deep=True)})
            entry=make_figure(name=name,page_id='manual-study' if self._owner_kind == 'study' else 'report',
                source_label=str(self._provenance.get('study', 'Study' if self._owner_kind == 'study' else 'Report')),
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
                # write_figure_bundle writes exactly the chosen file, so the seam's default
                # output set (that one path) is the honest declaration. None is a cancel.
                path=ask_export_path(self,title='Save figure bundle',suggestion=self._name+'.zip',
                    filters='Figure bundle (*.zip)')
                if path is None:return
            result=write_figure_bundle(path,self.snapshot());self._status.setText('Exported '+str(result));return result
        except Exception as exc:self._status.setText('Bundle was not exported: '+str(exc));return None

    def closeEvent(self,event):
        # The window outlives its close, so free what it rendered too: the scene item
        # keeps a QPicture and SVG, the figure's canvas an Agg buffer, megabytes each for
        # a dense figure. MainWindow's quit closes every window it owns, closed ones
        # included, so a second close must find nothing left to free.
        self._fit_timer.stop();self._style_timer.stop()
        self._format_pane.retire()
        for controller in self._text_controllers: controller.retire()
        self._text_controllers=[]
        self._series_selection.set_item(None)
        self._item=None;self._view.scene().clear()
        if self._figure is not None:self._figure.clear()
        self._figure=None;self._resources.clear();self._save_style=None
        self._styler=None;self._data=None
        super().closeEvent(event)
        self.closed.emit()
