"""One study entry point with explicit methods and saved-workspace routing."""
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .incubation_library import StudyLibrary, StudyLibraryError, default_library_root
from .incubation_profiles import PROFILES, get_profile, peek_workspace, saved_profile


class IncubationStudyLauncher(QtWidgets.QMainWindow):
    def __init__(self, parent, library=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle('Incubation study — NanoScope')
        self.resize(870, 740)
        self._library_error = ''
        try:
            self._library = library if library is not None else StudyLibrary()
            self._library_root = self._library.root
        except StudyLibraryError as exc:
            self._library = None
            self._library_root = default_library_root()
            self._library_error = str(exc)
        self._library_entries = {}
        self._windows = []
        self._opening = {}
        self._pending_close = {}
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        heading = QtWidgets.QLabel('Incubation study')
        font = heading.font(); font.setPointSize(font.pointSize()+6); font.setBold(True)
        heading.setFont(font); layout.addWidget(heading)
        note = QtWidgets.QLabel('Prepare recordings, detect and curate events in the main workspace, then compare conditions and generate study figures.')
        note.setWordWrap(True); layout.addWidget(note)
        saved = QtWidgets.QGroupBox('Saved studies'); saved_layout = QtWidgets.QVBoxLayout(saved)
        self._library_location = QtWidgets.QLabel('Study library: '+str(self._library_root))
        self._library_location.setAccessibleName('Study library location')
        self._library_location.setWordWrap(True)
        self._library_location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        saved_layout.addWidget(self._library_location)
        self._saved = QtWidgets.QTreeWidget()
        self._saved.setAccessibleName('Saved incubation studies')
        self._saved.setHeaderLabels(['Study', 'Location', 'Status'])
        self._saved.setRootIsDecorated(False); self._saved.setUniformRowHeights(True)
        self._saved.setTextElideMode(QtCore.Qt.ElideMiddle)
        self._saved.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._saved.setMinimumHeight(140); self._saved.setMaximumHeight(200)
        self._saved.setColumnWidth(0, 220)
        self._saved.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self._saved.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self._saved.header().setStretchLastSection(False)
        self._saved.itemSelectionChanged.connect(self._saved_selection_changed)
        self._saved.itemDoubleClicked.connect(lambda *_: self._guard(self.open_selected_study))
        saved_layout.addWidget(self._saved)
        self._library_note = QtWidgets.QLabel(); self._library_note.setWordWrap(True)
        saved_layout.addWidget(self._library_note)
        saved_actions = QtWidgets.QHBoxLayout()
        self._open_selected = QtWidgets.QPushButton('Open selected study')
        self._open_selected.clicked.connect(lambda: self._guard(self.open_selected_study))
        self._show_folder = QtWidgets.QPushButton('Show study folder')
        self._show_folder.clicked.connect(lambda: self._guard(self.show_study_folder))
        self._browse = QtWidgets.QPushButton('Open saved study…')
        self._browse.setToolTip('Browse for another study; starts at the last known directory. Its saved method and settings will be restored.')
        self._browse.clicked.connect(lambda: self._guard(self._choose_open))
        for button in (self._open_selected, self._show_folder, self._browse): saved_actions.addWidget(button)
        saved_layout.addLayout(saved_actions); layout.addWidget(saved)
        new = QtWidgets.QGroupBox('New study'); form = QtWidgets.QFormLayout(new)
        self._profile = QtWidgets.QComboBox()
        for profile in PROFILES: self._profile.addItem(profile.label, profile.id)
        self._profile.setAccessibleName('Study profile')
        form.addRow('Profile', self._profile)
        self._description = QtWidgets.QLabel(); self._description.setWordWrap(True)
        form.addRow(self._description)
        self._name = QtWidgets.QLineEdit(); self._name.setAccessibleName('Study name')
        self._sample = QtWidgets.QLineEdit(); self._sample.setPlaceholderText('Optional sample or preparation')
        self._sample.setAccessibleName('Study sample')
        form.addRow('Study name', self._name); form.addRow('Sample', self._sample)
        create = QtWidgets.QPushButton('Create study…')
        create.clicked.connect(lambda: self._guard(self._choose_new)); form.addRow(create)
        layout.addWidget(new)
        self._profile.currentIndexChanged.connect(self._profile_changed)
        self._profile_changed()
        layout.addWidget(QtWidgets.QLabel('Open in this session'))
        self._active = QtWidgets.QComboBox(); self._active.setAccessibleName('Open study workspaces')
        layout.addWidget(self._active)
        self._continue = QtWidgets.QPushButton('Continue selected study')
        self._continue.clicked.connect(lambda: self._guard(self.continue_study)); layout.addWidget(self._continue)
        advanced = QtWidgets.QPushButton('Advanced: original Zn reference recipe…')
        advanced.setToolTip('Reproduce the fixed original Zn notebook workflow or inspect its saved result bundles.')
        advanced.clicked.connect(parent._open_incubation_reference); layout.addWidget(advanced)
        self._status = QtWidgets.QLabel(); self._status.setWordWrap(True); layout.addWidget(self._status)
        layout.addStretch()
        self.refresh()

    def _profile_changed(self):
        profile = get_profile(self._profile.currentData())
        self._description.setText(profile.description)
        self._name.setPlaceholderText(profile.default_name)

    def _guard(self, action):
        try: return action()
        except Exception as exc:
            self._status.setText(str(exc)); return False

    @staticmethod
    def _show(window):
        window.show(); window.raise_(); window.activateWindow()
        return window

    def refresh(self):
        current = self._active.currentData()
        # Include windows opened through older internal links as well: the alias this
        # class itself writes below (_manual_study_window/_acid_study_window), plus
        # MainWindow._open_manual_study()'s own single-window shortcut, which never
        # writes that alias (it must not be repointed by studies it did not create).
        for key in ('_manual_study_window', '_acid_study_window', '_manual_study_shortcut'):
            window = getattr(self.parent(), key, None)
            if window is not None and window not in self._windows and window.folder is not None:
                self._remember(window)
        self._active.clear()
        for window in self._windows:
            if window.folder is not None:
                self._active.addItem(window.state.get('name', 'Study')+' — '+str(window.folder), window)
        if current is not None:
            index = self._active.findData(current)
            if index >= 0: self._active.setCurrentIndex(index)
        self._continue.setEnabled(self._active.count() > 0)
        self._refresh_library()

    def _refresh_library(self, selected=None):
        if selected is None:
            current = self._saved.currentItem()
            selected = current.data(0, QtCore.Qt.UserRole) if current is not None else None
        self._saved.clear(); self._library_entries = {}
        try:
            entries = self._library.entries() if self._library is not None else []
        except StudyLibraryError as exc:
            self._library_error = str(exc); entries = []
        else:
            if self._library is not None: self._library_error = ''
        for entry in entries:
            key = str(entry.path)
            self._library_entries[key] = entry
            item = QtWidgets.QTreeWidgetItem([entry.label, key, 'Ready to open' if entry.available else 'Unavailable'])
            item.setData(0, QtCore.Qt.UserRole, key)
            item.setToolTip(0, entry.label)
            item.setToolTip(1, key)
            item.setToolTip(2, entry.error or 'The complete checkpoint is verified when opened.')
            self._saved.addTopLevelItem(item)
            if key == selected: self._saved.setCurrentItem(item)
        if self._saved.currentItem() is None and self._saved.topLevelItemCount():
            self._saved.setCurrentItem(self._saved.topLevelItem(0))
        self._saved_selection_changed()

    def _selected_entry(self):
        current = self._saved.currentItem()
        return self._library_entries.get(current.data(0, QtCore.Qt.UserRole)) if current is not None else None

    def _saved_selection_changed(self):
        entry = self._selected_entry()
        self._open_selected.setEnabled(entry is not None and entry.available)
        self._show_folder.setEnabled(entry is not None and entry.path.is_dir())
        if self._library_error:
            message = self._library_error+' Use Open saved study… to browse to a study folder.'
        elif entry is not None and not entry.available:
            message = entry.error
        elif not self._library_entries:
            message = 'No saved studies yet. Use Open saved study… to browse, or create one below.'
        else:
            message = 'Open a saved study directly. Its saved method and settings are restored.'
        self._library_note.setText(message)

    def _remember_saved(self, folder, success_message):
        # A registry failure must never turn a successful scientific load into
        # a failed load or discard the newly created/opened study window.
        try:
            if self._library is None: raise StudyLibraryError(self._library_error)
            self._library.remember(folder)
        except StudyLibraryError as exc:
            self._library_error = str(exc)
            self._status.setText(success_message+'. Could not add it to the saved-study list: '+str(exc))
        else:
            self._status.setText(success_message)
        self._refresh_library(str(folder))

    def _remember(self, window):
        if window in self._windows: return
        self._windows.append(window)
        key = '_acid_study_window' if get_profile(saved_profile(window.state)).engine == 'acid' else '_manual_study_window'
        setattr(self.parent(), key, window)
        window.operationFinished.connect(self.refresh)
        window.operationFailed.connect(self._status.setText)

    def _window_for(self, engine):
        if engine == 'acid':
            from .acid_study_window import AcidStudyWindow
            window = AcidStudyWindow(self.parent())
        else:
            from .manual_study_window import ManualStudyWindow
            window = ManualStudyWindow(self.parent())
        return window

    def _existing(self, folder):
        self.refresh()
        return next((w for w in self._windows if w.folder is not None and Path(w.folder).resolve() == folder), None)

    def create_study(self, folder, *, profile='general', name='', sample=''):
        selected = get_profile(profile)
        folder = Path(folder).expanduser().resolve()
        if self._existing(folder) is not None or folder in self._opening:
            raise ValueError('This study is already open. Continue it, or choose a new empty folder.')
        window = self._window_for(selected.engine)
        try:
            window.new_workspace(folder, profile=selected.id, name=name, sample=sample)
        except BaseException:
            window.close(); window.deleteLater(); self._clear_reference(window)
            raise
        self._remember(window); self.refresh()
        self._remember_saved(window.folder, 'Created '+window.state['name'])
        return self._show(window)

    def open_study(self, folder):
        route = peek_workspace(folder)
        existing = self._existing(route.path)
        if existing is not None:
            # This window already completed the engine's verified load. Older
            # internal entry points may have opened it before it was indexed.
            self._remember_saved(existing.folder, 'Continuing '+existing.state['name'])
            return self._show(existing)
        if route.path in self._opening: return self._show(self._opening[route.path])
        window = self._window_for(route.engine)
        self._opening[route.path] = window
        def disconnect():
            window.operationFinished.disconnect(finished)
            window.operationFailed.disconnect(failed)
            window.operationCancelled.disconnect(failed)
            self._pending_close.pop(window, None)
            window.removeEventFilter(self)
        def finished(*_):
            disconnect()
            self._opening.pop(route.path, None)
            if window.folder is not None:
                self._remember(window); self.refresh()
                self._remember_saved(window.folder, 'Opened '+window.state['name'])
        def failed(message='Opening study was cancelled.'):
            disconnect()
            self._opening.pop(route.path, None)
            self._status.setText(message)
            self._clear_reference(window); window.close(); window.deleteLater()
        def closed():
            # Closing a study aborts its worker without emitting cancellation.
            # Release this pending route so the same checkpoint can be reopened.
            disconnect(); self._opening.pop(route.path, None)
            self._status.setText('Opening study was cancelled.')
            window.abort(); self._clear_reference(window); window.deleteLater()
        # Loading verifies the complete checkpoint asynchronously; only a successful
        # load joins the active study list. Other open studies remain untouched.
        window.operationFinished.connect(finished)
        window.operationFailed.connect(failed)
        window.operationCancelled.connect(failed)
        self._pending_close[window] = closed
        window.installEventFilter(self)
        try:
            started = window.open_workspace(route.path)
        except BaseException:
            failed(); raise
        if started is False:
            failed('The study could not be opened.'); return False
        self._status.setText('Opening '+route.name+'…')
        return self._show(window)

    def _clear_reference(self, window):
        for key in ('_manual_study_window', '_acid_study_window'):
            if getattr(self.parent(), key, None) is window: setattr(self.parent(), key, None)

    def continue_study(self):
        window = self._active.currentData()
        if window is not None: return self._show(window)

    def open_selected_study(self):
        entry = self._selected_entry()
        if entry is None: return False
        if not entry.available:
            self._status.setText(entry.error); return False
        return self.open_study(entry.path)

    def show_study_folder(self):
        entry = self._selected_entry()
        if entry is None: return False
        if not entry.path.is_dir():
            raise ValueError('This study folder is unavailable. Browse to its current location to open it.')
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(entry.path))):
            raise ValueError('The study folder could not be shown: '+str(entry.path))
        return True

    def _choose_new(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose a new empty study folder', str(self._library_root))
        if folder:
            return self.create_study(folder, profile=self._profile.currentData(), name=self._name.text().strip(), sample=self._sample.text().strip())

    def _choose_open(self):
        try:
            directory = self._library.picker_directory() if self._library is not None else str(self._library_root)
        except StudyLibraryError:
            directory = str(self._library_root)
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open saved incubation study folder', directory)
        if folder: return self.open_study(folder)

    def showEvent(self, event):
        self.refresh(); super().showEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Close and watched in self._pending_close:
            self._pending_close[watched]()
        return super().eventFilter(watched, event)
