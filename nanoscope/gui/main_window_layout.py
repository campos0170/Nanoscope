"""The layout half of ``MainWindow``: child windows with their remembered size and place, the
Window menu, Reset window layout, the saved settings, the dock widths and corners, and the
Results dock's defaults.

``showEvent``, ``resizeEvent`` and the drag-and-drop overrides stay in ``main_window.py``: they
are Qt overrides.

``_LayoutMixin`` is a plain class listed before ``QMainWindow`` in the bases of ``MainWindow``,
with no ``__init__`` and no state of its own. The method bodies were moved verbatim out of
``main_window.py`` by a mechanical split; nothing in them was edited.

``_on_screen`` is imported from ``main_window_seams``, whose forwarder looks the name up on
``main_window`` each time it is called, so a test's patch of ``main_window._on_screen`` still
reaches these methods.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .analysis_catalog import ANALYSIS_LABELS, PAGE_IDS
from .main_window_shared import (
    _FILE_DOCK_DEFAULT_WIDTH,
    _INSPECTOR_DEFAULT_WIDTH,
    _INSPECTOR_MAX_DEFAULT_WIDTH,
    _INSPECTOR_CHROME_WIDTH,
    _INSPECTOR_NESTED_CHROME_WIDTH,
    _DOCK_SEPARATOR_WIDTH,
    _WINDOW_STATE_VERSION,
    _RESULTS_DEFAULT_ROWS,
    _RESULTS_MAX_HEIGHT_FRACTION,
    _RESULTS_MIN_DEFAULT_HEIGHT,
    _RESULTS_PANEL_PATH,
    _ROLE_INSPECTOR,
    _ROLE_GALLERY,
    _HOST_WINDOW,
    _HOST_DOCK,
    _INSPECTOR_DEFAULT_HOST,
    _CHILD_WINDOW_KEYS,
    _CHILD_WINDOW_LABELS,
    _SETTING_INSPECTOR_HOST,
    _SETTING_DETAILS,
    _geometry_setting,
    _saved_normal_rect,
    _restorable_bytes,
    _settings_store_damaged,
    _string_setting,
    _choice_setting,
    _bool_setting,
)
from .main_window_seams import _on_screen


class _LayoutMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    # -- child windows: remembered size and place ---------------------------------------------
    def _register_child_window(self, window: QtWidgets.QWidget, key: str) -> None:
        """Remember ``window`` as the one ``key`` window: the Window menu lists it and its
        size and place are kept (on hide and at quit) in ``windows/<key>/geometry``."""
        self._child_windows[key] = window
        window.setProperty("nanoscopeGeometryKey", key)
        window.setProperty("nanoscopeDefaultSize", window.size())    # what a Reset goes back to
        window.installEventFilter(self)

    def _adopt_child_window(self, window: QtWidgets.QWidget, key: str) -> None:
        """Register ``window`` and give it its remembered (or default) geometry -- once,
        before its first show."""
        if self._child_windows.get(key) is not window:
            self._register_child_window(window, key)
        self._place_child_window(window, key)

    def _restore_saved_geometry(self, window: QtWidgets.QWidget, blob) -> bool:
        """``restoreGeometry``, but a sound blob Qt declines for a differently sized screen
        gets its saved normal rect, fitted to this screen. False only for unusable bytes."""
        blob = QtCore.QByteArray(bytes(blob))
        if window.restoreGeometry(blob):
            return True
        saved = _saved_normal_rect(blob)
        if saved is None:
            return False
        rect, maximized = saved
        window.setGeometry(rect)
        self._fit_to_available_screen(window)
        if maximized:
            window.setWindowState(window.windowState() | QtCore.Qt.WindowMaximized)
        return True

    def _place_child_window(self, window: QtWidgets.QWidget, key: str) -> None:
        if window.property("nanoscopeGeometryPlaced"):
            return
        window.setProperty("nanoscopeGeometryPlaced", True)
        saved = self._child_geometry.get(key)
        if saved is not None:
            if self._restore_saved_geometry(window, saved):
                self._fit_to_available_screen(window)
                return
            # Right type, contents Qt cannot use: say so once and forget it.
            self._child_geometry.pop(key, None)
            QtCore.QSettings().remove(_geometry_setting(key))
            self.statusBar().showMessage(
                f"The remembered size and place of {window.windowTitle() or 'a window'} "
                "could not be used, so it opened at its default.", 8000)
        if key == _ROLE_INSPECTOR:
            from .event_inspector import _WINDOW_DEFAULT_SIZE
            window.resize(*_WINDOW_DEFAULT_SIZE)
        # Defaults only for a window the user is looking at: a hidden main window (the
        # offscreen suite) keeps every constructor size.
        if not self.isVisible():
            return
        if window.testAttribute(QtCore.Qt.WA_Moved):
            # Already placed (a window that was docked keeps the dock's offsets), so Qt
            # will not centre it: over this window, then inside the screen.
            frame = window.frameGeometry()
            frame.moveCenter(self.frameGeometry().center())
            window.move(frame.topLeft())
            self._fit_to_available_screen(window)
        else:
            # Size only: a move would stop Qt placing it (a dialog centred on this window).
            self._fit_to_available_screen(window, move=False)

    def _fit_to_available_screen(self, window: QtWidgets.QWidget, *, move: bool = True) -> None:
        """Bound ``window``'s size to its screen's available area and (unless ``move`` is
        False) move it fully inside -- a remembered place on a monitor that is gone, or a
        size from a bigger one."""
        screen = window.screen() or self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        frame, inner = window.frameGeometry(), window.geometry()
        if frame == inner and not window.property("nanoscopeGeometryShown"):
            # Never shown, so its frame is not known yet: assume this window's (the title
            # bar a remembered or default size has to leave room for).
            ours, ours_inner = self.frameGeometry(), self.geometry()
            frame = frame.adjusted(ours.left() - ours_inner.left(), ours.top() - ours_inner.top(),
                                   ours.right() - ours_inner.right(),
                                   ours.bottom() - ours_inner.bottom())
        extra_w, extra_h = frame.width() - inner.width(), frame.height() - inner.height()
        width = max(min(inner.width(), avail.width() - extra_w), window.minimumWidth())
        height = max(min(inner.height(), avail.height() - extra_h), window.minimumHeight())
        if (width, height) != (inner.width(), inner.height()):
            window.resize(width, height)
            frame.setSize(QtCore.QSize(width + extra_w, height + extra_h))
        if not move:
            return
        x = max(avail.left(), min(frame.x(), avail.right() + 1 - frame.width()))
        y = max(avail.top(), min(frame.y(), avail.bottom() + 1 - frame.height()))
        if (x, y) != (frame.x(), frame.y()):
            window.move(x, y)              # a window's pos() is its frame's

    def _remember_child_geometry(self, window: QtWidgets.QWidget) -> None:
        key = window.property("nanoscopeGeometryKey")
        if not key or not window.isWindow() or not window.property("nanoscopeGeometryShown"):
            return
        if key not in _CHILD_WINDOW_KEYS:      # Signal Conditioning has keys of its own
            return
        geometry = window.saveGeometry()
        self._child_geometry[str(key)] = geometry
        QtCore.QSettings().setValue(_geometry_setting(str(key)), geometry)

    def _child_window_event(self, obj, event) -> None:
        """Show / Hide / Close of a child window or of the inspector's dock (eventFilter)."""
        kind = event.type()
        dock = self._inspector_dock
        insp = self._event_inspector
        if dock is not None and obj is dock:
            if kind == QtCore.QEvent.Hide and dock.isHidden():
                # Closed (✕): Hide arrives before Qt moves the focus, so park it now.
                self._park_inspector_focus()
            elif (kind == QtCore.QEvent.Show and not event.spontaneous()
                  and not self._presenting_inspector and insp is not None
                  and dock.widget() is insp):
                # Shown by anything but the opener: it was not re-fed while hidden.
                insp.set_vocabulary(self._label_vocab)
                self._refresh_event_inspector()
            return
        if not isinstance(obj, QtWidgets.QWidget) or not obj.property("nanoscopeGeometryKey"):
            return
        if kind == QtCore.QEvent.Show and obj.isWindow():
            obj.setProperty("nanoscopeGeometryShown", True)
        elif kind == QtCore.QEvent.Hide and obj.isWindow() and not self._rehosting:
            self._remember_child_geometry(obj)
        elif (kind == QtCore.QEvent.Close and obj is insp and not obj.isWindow()
              and dock is not None):
            # close() on the docked widget would leave an empty dock behind.
            QtCore.QTimer.singleShot(0, dock, dock.hide)

    def _bring_child_to_front(self, key: str) -> None:
        window = self._child_windows.get(key)
        if window is None:
            return
        if window.isMinimized():
            window.showNormal()
        window.show()
        window.raise_()
        window.activateWindow()

    # -- the Window menu -----------------------------------------------------------------------
    def _build_window_menu(self, menubar: QtWidgets.QMenuBar) -> None:
        """Window: the windows this one opens, and where the Event inspector lives.

        Its entries are built ONCE, on the first open (a window that never opens the menu
        pays for one action, not thirty); every later open re-reads state into the same
        actions, so nothing is created or destroyed per open. No shortcuts: ⌘M is Add
        measurement, and a second Ctrl+I or Ctrl+G here would make the pair ambiguous."""
        menu = menubar.addMenu("&Window")
        menu.setObjectName("windowMenu")
        self._window_menu = menu
        self._act_bring_all_to_front = menu.addAction("Bring All to Front")
        self._act_bring_all_to_front.setToolTip(
            "Raise every open NanoScope window above other applications' windows; the "
            "window you were working in keeps the keyboard.")
        self._act_bring_all_to_front.triggered.connect(self._bring_all_to_front)
        self._window_menu_entries: dict[str, QtGui.QAction] = {}
        self._window_menu_pages: list[tuple[str, QtGui.QAction]] = []
        menu.aboutToShow.connect(self._sync_window_menu)

    def _populate_window_menu(self) -> None:
        """The Window menu's entries, once."""
        if self._window_menu_entries:
            return
        menu = self._window_menu
        menu.addSeparator()

        def entry(key: str, text: str, tip: str, slot) -> QtGui.QAction:
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.triggered.connect(lambda _checked=False, s=slot: s())
            self._window_menu_entries[key] = act
            return act

        entry("main", "NanoScope",
              "Bring the main window — the trace, the files and the panels — to the front.",
              self._bring_main_to_front)
        entry(_ROLE_INSPECTOR, "Event inspector",
              "Open the Event inspector or bring it forward, in its own window or docked — "
              "the same as Analysis ▸ Event inspector… (⌘I).", self._open_event_inspector)
        entry(_ROLE_GALLERY, "Event gallery",
              "Open the Event gallery or bring it forward — the same as Analysis ▸ Event "
              "gallery… (⌘G).", self._open_event_gallery)
        entry("analysis_explorer", "Analysis Explorer",
              "Open the Analysis Explorer or bring it forward on the page it is showing.",
              self._show_analysis_explorer)
        for page_id in PAGE_IDS:
            act = menu.addAction("    " + ANALYSIS_LABELS[page_id])
            act.setCheckable(True)
            act.setVisible(False)
            act.setToolTip(f"Bring the Analysis Explorer forward on its "
                           f"{ANALYSIS_LABELS[page_id]} page.")
            act.triggered.connect(lambda _checked=False, p=page_id: self._raise_explorer_page(p))
            self._window_menu_pages.append((page_id, act))
        entry("figure_library", "Figure Library",
              "Open this project's Figure Library or bring it forward — the same as "
              "Analysis ▸ Export ▸ Figure Library….", self._open_figure_library)
        entry("signal_conditioning", "Preprocess workspace",
              "Open the Preprocess workspace or bring it forward — the same as "
              "Analysis ▸ Preprocess workspace… (⇧⌘F).", self._open_signal_conditioning)
        menu.addSeparator()
        for key, label in _CHILD_WINDOW_LABELS.items():
            act = entry(key, label, f"Bring the {label} window forward.",
                        lambda k=key: self._bring_child_to_front(k))
            act.setVisible(False)
        menu.addSeparator()
        self._act_dock_inspector = menu.addAction("Dock Event inspector beside the trace")
        self._act_dock_inspector.setCheckable(True)
        self._act_dock_inspector.setToolTip(
            "Checked: the Event inspector is a tab in the main window's right column, beside "
            "the trace. Unchecked: it opens as its own window. Remembered between sessions.")
        self._act_dock_inspector.triggered.connect(
            lambda checked: self.set_curation_host(
                _ROLE_INSPECTOR, _HOST_DOCK if checked else _HOST_WINDOW))
        menu.addSeparator()
        self._act_reset_window_layout = menu.addAction("Reset Window Layout…")
        self._act_reset_window_layout.setToolTip(
            "Put the panels back where they open on a first launch, the Event inspector back "
            "in its own window, and forget every remembered window size and place. Asks "
            "first; your analysis is not changed.")
        self._act_reset_window_layout.triggered.connect(self._reset_window_layout)

    def _sync_window_menu(self) -> None:
        """Re-read every entry: its name, whether it is open (checked) and reachable."""
        self._populate_window_menu()
        entries = self._window_menu_entries
        title = self.windowTitle()
        entries["main"].setText(title[2:] if title.startswith("• ") else title)
        entries["main"].setChecked(self.isVisible() and not self.isMinimized())
        insp_entry = entries[_ROLE_INSPECTOR]
        insp_entry.setText("Event inspector — docked" if self._inspector_host == _HOST_DOCK
                           else "Event inspector")
        insp_entry.setChecked(self.curation_surface_on_screen(_ROLE_INSPECTOR))
        insp_entry.setEnabled(self._act_event_inspector.isEnabled()
                              or self._event_inspector is not None)
        gal_entry = entries[_ROLE_GALLERY]
        gal_entry.setChecked(self.curation_surface_on_screen(_ROLE_GALLERY))
        gal_entry.setEnabled(self._act_event_gallery.isEnabled()
                             or self._event_gallery is not None)
        explorer = self._analysis_explorer
        explorer_open = explorer is not None and explorer.isVisible()
        entries["analysis_explorer"].setChecked(explorer_open)
        installed = explorer.page_ids() if explorer is not None else ()
        current = explorer.current_page_id() if explorer is not None else None
        for page_id, act in self._window_menu_pages:
            act.setVisible(page_id in installed)
            act.setChecked(explorer_open and page_id == current)
        entries["figure_library"].setChecked(_on_screen(self._figure_library))
        entries["signal_conditioning"].setChecked(_on_screen(self._signal_conditioning))
        for key in _CHILD_WINDOW_LABELS:
            window = self._child_windows.get(key)
            entries[key].setVisible(window is not None)
            entries[key].setChecked(_on_screen(window))
        self._act_dock_inspector.setChecked(self._inspector_host == _HOST_DOCK)

    def _bring_main_to_front(self) -> None:
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def _show_analysis_explorer(self) -> None:
        if self._analysis_explorer is None:
            self._open_analysis_home()
            return
        explorer = self._analysis_explorer
        if explorer.isMinimized():
            explorer.showNormal()
        explorer.show()
        explorer.raise_()
        explorer.activateWindow()

    def _raise_explorer_page(self, page_id: str) -> None:
        if self._analysis_explorer is not None:
            self._analysis_explorer.raise_page(page_id)

    def _bring_all_to_front(self) -> None:
        """Every visible NanoScope window above other applications'; the active one stays
        active (it is raised last)."""
        active = QtWidgets.QApplication.activeWindow()
        windows = [self, *(w for w in self._child_windows.values()
                           if w.isWindow() and w.isVisible())]
        explorer = self._analysis_explorer
        if explorer is not None and explorer.isVisible() and explorer not in windows:
            windows.append(explorer)
        for window in windows:
            if window is active:
                continue
            if window.isMinimized():
                window.showNormal()
            window.raise_()
        if active is not None:
            active.raise_()
            active.activateWindow()

    # -- Reset window layout ---------------------------------------------------------------------
    def _confirm_reset_window_layout(self) -> bool:
        """Ask first (a test seam, like _confirm_close_choice)."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle("Reset window layout")
        box.setText("Put every panel and window back where it starts?")
        box.setInformativeText(
            "The File manager, Inspector and Results panels return to their opening places "
            "and sizes, the Event inspector to its own window, and every remembered window "
            "size and place is forgotten. Your analysis is not changed.")
        reset = box.addButton("Reset", QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        return box.clickedButton() is reset

    def _reset_window_layout(self) -> None:
        """Window ▸ Reset Window Layout…: the docks as a first launch opens them, the Event
        inspector in its own window, no remembered child geometry. Open windows stay open."""
        if not self._confirm_reset_window_layout():
            return
        insp = self._event_inspector
        insp_open = self.curation_surface_on_screen(_ROLE_INSPECTOR)
        self._inspector_host = _HOST_WINDOW
        s = QtCore.QSettings()
        s.setValue(_SETTING_INSPECTOR_HOST, _HOST_WINDOW)
        for key in (*_SETTING_DETAILS.values(),
                    *(_geometry_setting(k) for k in _CHILD_WINDOW_KEYS),
                    _geometry_setting("signal_conditioning"),    # written by an earlier build
                    self._SC_GEOMETRY_KEY, self._SC_SPLITTER_KEY):
            s.remove(key)
        self._child_geometry.clear()
        self._inspector_window_geometry = None
        self._inspector_details = {}
        if insp is not None:
            insp.reset_details_state()
            if not insp.isWindow():
                self._undock_event_inspector()
                insp.setVisible(insp_open)
        if self._inspector_dock is not None:
            self.removeDockWidget(self._inspector_dock)
        if self._factory_window_state is not None:
            self.restoreState(self._factory_window_state, _WINDOW_STATE_VERSION)
        self._apply_dock_corners()
        self._sync_results_title_bar()
        self._apply_default_dock_widths()
        self._apply_default_results_height()
        for key, window in self._child_windows.items():
            window.setProperty("nanoscopeGeometryPlaced", False)
            if window.isWindow() and window.isVisible():
                self._place_child_window(window, key)
            else:
                # Hidden: it was forgotten above, so the quit-save must not write it back,
                # and it reopens at its default size and place, not the ones it was left in.
                window.setProperty("nanoscopeGeometryShown", False)
                default = window.property("nanoscopeDefaultSize")
                if window.isWindow() and default is not None:
                    window.resize(default)
                    self._place_child_window(window, key)
        self.statusBar().showMessage(
            "Window layout reset to its defaults. Your analysis is unchanged.", 8000)

    # -- settings + teardown ---------------------------------------------------------------------
    def _restore_settings(self) -> None:
        """Re-apply the saved layout and lists — treating the store as external input.

        EVERY value read here is validated before it is used, because the preferences
        store is a file on disk that the app is not the only writer of (see
        :func:`_restorable_bytes`). A value of the wrong kind is dropped, the key it came
        under is remembered, and the info bar names them: a preference the app cannot
        stand behind is worth one default window, not a launch failure and not a silent
        substitution.
        """
        s = QtCore.QSettings()
        # THE STORE ITSELF can be the casualty, not just one value in it. A corrupted
        # preferences file (crash mid-quit, half-synced ~/Library) makes QSettings drop
        # keys wholesale with a FormatError — and this method used to never ask, so the
        # app came up looking factory-fresh (layout gone, recent files gone) with nothing
        # anywhere saying why, then quit and wrote a clean file over the evidence. That
        # silent fallback is exactly what `_restorable_bytes` forbids for a single value
        # ("said out loud rather than swallowing"), one level up.
        store_broken = _settings_store_damaged(s)
        #: Preference keys whose stored value was not something the app could use. Said
        #: out loud at the end of this method — a dropped preference is a change the user
        #: will SEE (their layout is gone), so it is never dropped silently.
        ignored: list[str] = []
        # Before any saved layout: what Window ▸ Reset Window Layout… goes back to.
        self._factory_window_state = self.saveState(_WINDOW_STATE_VERSION)
        geo = _restorable_bytes(s, "geometry", ignored)
        if geo is not None and not self._restore_saved_geometry(self, geo):
            # Right kind, wrong contents — an empty, cut-short or foreign-version byte array.
            # Qt says so by returning False rather than raising, and that answer is worth the
            # same sentence as a wrong-typed value: either way the saved geometry is not applied.
            ignored.append("geometry")
        state = _restorable_bytes(s, "windowState", ignored)
        # Newest version first; a layout saved before the Results dock (version 0) still
        # restores — every width and visibility it holds — and is then migrated.
        restored = None
        if state is not None:
            for version in range(_WINDOW_STATE_VERSION, -1, -1):
                if self.restoreState(state, version):
                    restored = version
                    break
        if restored is None:
            if state is not None:
                ignored.append("windowState")
            # No usable dock layout, for either reason: fall back to the designed widths
            # rather than to whatever the docks' sizeHints happen to add up to.
            #
            # PROVISIONALLY. This runs from ``__init__``, where ``self.width()`` is still
            # the 1440 the constructor asked for rather than a width any screen has agreed
            # to, so the width-dependent half of the derivation (the Inspector's ceiling)
            # cannot be honest yet. :meth:`showEvent` re-derives it against the width the
            # window actually got. Applied here as well, because a window that is never
            # shown — most of the headless tests — still needs the designed widths.
            self._opening_layout_pending = True
            self._results_layout_pending = True
            self._apply_default_dock_widths()
            self._apply_default_results_height()
        else:
            # saveState carries the corners and a version-0 state carries Qt's defaults.
            self._apply_dock_corners()
            if restored < _WINDOW_STATE_VERSION:
                self._migrate_window_state(restored)
        self._sync_results_title_bar()
        raw = s.value("recentFiles", [])
        if isinstance(raw, str):
            raw = [raw] if raw else []
        self._recent = [str(x) for x in (raw or []) if Path(x).exists()]
        raw_proj = s.value("recentProjects", [])
        if isinstance(raw_proj, str):
            raw_proj = [raw_proj] if raw_proj else []
        self._recent_projects = [str(x) for x in (raw_proj or []) if Path(x).exists()]
        # The two remembered directories: a str is what was written, and anything else
        # (a list from a damaged store) would become a nonsense `str([...])` path that
        # the next Open dialog would silently start in. Fall back to home instead.
        self._last_dir = _string_setting(s, "lastDir", str(Path.home()), ignored)
        self._last_project_dir = _string_setting(
            s, "lastProjectDir", str(Path.home()), ignored)
        self._read_window_preferences(s, ignored)
        self._rebuild_recent_menu()
        self._rebuild_recent_projects_menu()
        messages: list[str] = []
        if store_broken:
            messages.append(
                f"The saved-preferences file is damaged and could not be fully read "
                f"({s.fileName()}), so anything it could not give up — window layout, "
                f"recent files, remembered folders — starts at its default. Nothing else "
                f"is affected, and the next quit writes a healthy file over it.")
        if ignored:
            messages.append(
                f"{len(ignored)} saved preference(s) held a value this app did not write "
                f"and could not use, so they were ignored and their defaults used instead: "
                f"{', '.join(ignored)}. Nothing else was affected — the next quit writes "
                f"healthy values over them.")
        if messages:
            # One info bar, one message: the second show_message would replace the first,
            # and a disclosure that overwrites a disclosure is a silent drop with a spare
            # step.
            self._info_bar.show_message(" ".join(messages))

    def _inspector_width_ceiling(self, layout_width: int) -> int:
        """The widest the Inspector may OPEN and still leave the viewer the widest column.

        THE RULE, as arithmetic rather than as a property of one laptop. The three columns
        and the two separators between them fill the window::

            file (320) + separator + viewer + separator + inspector == layout_width

        so the viewer is wider than the Inspector exactly while the Inspector is under half
        of what is left after the file manager — that is the number returned here, taken one
        pixel short of the tie. It was previously a flat 560, chosen because 1512 - 12 - 320
        - 560 = 620 leaves the viewer widest **on a 1512 px display and nowhere narrower**:
        at 1440 the same 560 made the Inspector the widest column (viewer 548), and at 1280
        it was 560 against 388. A signal viewer that stops being the widest column as soon as
        the window is not exactly the developer's is not a rule, it is a coincidence.

        Measured AS THIS METHOD RETURNS IT, cap included: 1512 -> 560, 1440 -> 553,
        1280 -> 473, 1152 -> 409. Only the widest of those is capped — half the room at 1512
        is 589 — and that cap is what stops a big display opening a 900 px control column.

        BELOW 1265 px the two things this method arbitrates — the Inspector showing its own
        pages, and the viewer staying the widest column — cannot both be had, and the clamp
        in :meth:`_inspector_default_width` resolves it towards the floor: the Inspector
        opens at its designed minimum and takes a scrollbar. A scrollbar on a control page
        whose controls are all still whole is a door with a step in front of it; a viewer
        squeezed under the Inspector has no door at all.

        1265 IS MEASURED (offscreen, 12 px text), and it is the width at which this ceiling
        first reaches the width the pages need (466 on this tree, the Calibration page, since
        c86b119 put the Measure page in two columns): ``room = 1265 - 12 - 320 = 933``,
        ``(933-1)//2 == 466``; at 1264 it is 465. With the Mac's 13 px text the pages need 488
        and the boundary is 1309. It was 1337 while the Measure page needed 502. This line
        once said "roughly 1252", which is a
        different boundary — the width at which the ceiling drops below the FLOOR (460) —
        and stating it here left an 84 px band, 1253 to 1336, where the docstring claimed
        both things could be had and the clamp silently gave neither: the Inspector opened
        at the ceiling (460–501, measured 483 at 1300 and 501 at 1336), which is under the
        502 its pages need, so it took a horizontal scrollbar ANYWAY — the Manual-measurement
        panel clipped by 19 px at 1300 — while still spending up to 41 px of the trace on the
        difference. The clamp now resolves the whole band the way this paragraph says it does.
        """
        room = layout_width - 2 * _DOCK_SEPARATOR_WIDTH - _FILE_DOCK_DEFAULT_WIDTH
        return min(_INSPECTOR_MAX_DEFAULT_WIDTH, (room - 1) // 2)

    def _inspector_default_width(self) -> int:
        """How wide the Inspector should OPEN: the narrowest width that shows its pages.

        The pages are all widgetResizable scroll areas with a zero minimum (see
        ``_build_docks``), which is what lets the user drag this dock down to the tab bar —
        and is also why the dock cannot ask for a width. So the ASKING happens here, once,
        from the same number the scroll area would otherwise turn into a horizontal
        scrollbar.

        THE NUMBER TO ASK FOR IS THE PAGE'S MINIMUM, NOT ITS NATURAL WIDTH. This method used
        to read ``sizeHint()`` — the width at which a page is comfortable, 646 px for the
        Analysis control column — which is more than any of these pages needs and more than
        the ceiling allows, so the ceiling became the width and the derivation was theatre.
        ``minimumSizeHint()`` is the width below which the page's own controls start being
        clipped, which is what "reachable" actually means; the widest of those across every
        page, plus the chrome between the dock's edge and the deepest of their viewports, is
        the smallest dock that hides nothing. MEASURED on this tree at 1512x900: 502 px, and
        a 2 px sweep down from there confirms it is the knee — at 500 the Manual-measurement
        panel takes a horizontal scrollbar and clips "Snap to selected event" and "Mark
        selection as missed event"; at 472 the Preprocess column clips "Back to default"; at
        460, the pre-round width, both. Everything above 502 is width taken off the trace for
        nothing, which is why the viewer now opens at 678 rather than 620.

        Clamped at both ends, and both ends are load-bearing:

        * never below ``_INSPECTOR_DEFAULT_WIDTH`` — a page that happens to be narrow today
          must not open a dock so thin the next page looks broken in it;
        * never above :meth:`_inspector_width_ceiling` — the signal viewer stays the widest
          column, at every window width and not just at 1512. A panel that grows a long combo
          item can no longer take the viewer's width with it; it gets a scrollbar instead,
          which is the trade the ceiling exists to make.

        AND WHEN THE CEILING CANNOT REACH ``wanted``, THE FLOOR TAKES IT, not the ceiling.
        Below a 1337 px window (measured; see :meth:`_inspector_width_ceiling`) no width
        satisfies both rules, and a width BETWEEN the two — the ceiling, 460–501 — satisfies
        neither: the pages clip exactly as they do at the floor (the Manual-measurement panel
        by 19 px at a 1300 px window, by 1 px at 1336) and the extra pixels come off the trace
        for a scrollbar the user gets either way. So the answer below the boundary is one
        answer, the one the ceiling's docstring states: the designed floor, with the scrollbar,
        and every pixel the Inspector cannot use handed to the viewer (measured at a 1300 px
        window: Inspector 483 -> 460, viewer 485 -> 508). Above the boundary nothing changes.

        This is a change to the width the dock OPENS at, and only that. It is applied by
        ``_apply_default_dock_widths`` on a first launch — once from ``__init__`` and again
        from :meth:`showEvent`, which is where the window's width stops being the 1440 the
        constructor asked for and becomes the one the screen gave it. The boundary is never
        re-imposed after that opening turn, because a separator that slides because the
        window got 3 px narrower is furniture moving under the pointer, which is a worse bug
        than the scrollbar it would be chasing.

        Read before the window is shown, which is where a first launch applies it:
        ``minimumSizeHint()`` on these pages is layout arithmetic, not painted geometry, and
        measures the same (438 px for the widest, the Manual-measurement panel) before
        ``show()`` as after it.
        """
        widest = 0
        for area in self._inspector_tabs.findChildren(QtWidgets.QScrollArea):
            page = area.widget()
            if page is not None:
                widest = max(widest, page.minimumSizeHint().width())
        wanted = widest + _INSPECTOR_CHROME_WIDTH + _INSPECTOR_NESTED_CHROME_WIDTH
        ceiling = self._inspector_width_ceiling(self.width())
        if wanted > ceiling:
            # The window is too narrow for both rules. One answer for the whole of that
            # range — the floor — rather than a ceiling that clips like the floor and costs
            # like the width that would not. (Under a ~1253 px window the ceiling is already
            # below the floor and this is what the old `max` returned anyway; the change is
            # the band above it.)
            return _INSPECTOR_DEFAULT_WIDTH
        return max(_INSPECTOR_DEFAULT_WIDTH, wanted)

    def _apply_default_dock_widths(self) -> None:
        """Open the two docks at their designed widths — first launch only.

        Neither dock has a meaningful natural width any more: the file manager's tree will
        happily live in 293 px and truncate every recording name, and the Inspector's pages
        are all scroll areas whose combined sizeHint means nothing on its own. So the opening
        layout is stated here rather than inferred: a stated width for the left column, and
        for the right one the content-derived width above. This runs ONLY when there is no
        saved windowState; once the user has dragged the separators, ``_save_settings``
        persists her choice and ``restoreState`` above wins on every later launch — this
        method is never the thing that fights a remembered layout. These are not minimums:
        both docks remain draggable narrower and wider than this.
        """
        self.resizeDocks(
            [self._file_dock, self._inspector],
            [_FILE_DOCK_DEFAULT_WIDTH, self._inspector_default_width()],
            QtCore.Qt.Horizontal)

    def _apply_dock_corners(self) -> None:
        """The side columns run the full height; a top or bottom dock sits under the trace.

        Code-owned, not a preference — but ``saveState`` CARRIES the corners and
        ``restoreState`` overwrites them, and a version-0 state carries Qt's defaults
        (top/bottom own all four corners: measured, the Results dock then spans 1512 px and
        cuts both side docks to 736). So this runs at construction AND after every
        successful restore. Not spanning under the File Manager is deliberate: the file
        tree keeps its whole height for long voltage and incubation series.
        """
        self.setCorner(QtCore.Qt.TopLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.TopRightCorner, QtCore.Qt.RightDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea)

    def _migrate_window_state(self, from_version: int) -> None:
        """Bring a restored older layout up to date; its widths and visibilities stay.

        Says so on the status bar and NOT on the info bar: a migration is not damage."""
        if from_version < 1:
            # A version-0 layout predates the Results dock, which Qt leaves wherever it
            # happens to be at a degenerate height.
            self._results_dock.setFloating(False)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._results_dock)
            self._results_dock.show()
            self._results_layout_pending = True
            self._apply_default_results_height()
            QtCore.QTimer.singleShot(0, self, lambda: self.statusBar().showMessage(
                "The event table now lives in the Results panel under the trace — "
                f"{_RESULTS_PANEL_PATH}.", 10000))

    def _dock_area_height(self) -> int:
        """The height the docks and the central widget share: the window minus an
        in-window menu bar and the status bar."""
        bar = self.menuBar()
        menu = 0 if bar.isNativeMenuBar() else max(bar.height(), bar.sizeHint().height())
        status = max(self.statusBar().height(), self.statusBar().sizeHint().height())
        return max(0, self.height() - menu - status)

    def _results_default_height(self) -> int:
        """How tall the Results dock OPENS: ``_RESULTS_DEFAULT_ROWS`` table rows, capped at
        ``_RESULTS_MAX_HEIGHT_FRACTION`` of the dock area, never below
        ``_RESULTS_MIN_DEFAULT_HEIGHT``. A default, applied like the Inspector's opening
        width — never re-imposed once the layout is the user's."""
        wanted = self._analysis_panel.results_height_for_rows(_RESULTS_DEFAULT_ROWS)
        cap = int(self._dock_area_height() * _RESULTS_MAX_HEIGHT_FRACTION)
        return max(_RESULTS_MIN_DEFAULT_HEIGHT, min(wanted, cap))

    def _apply_default_results_height(self) -> None:
        dock = self._results_dock
        if dock.isFloating() or self.dockWidgetArea(dock) not in (
                QtCore.Qt.TopDockWidgetArea, QtCore.Qt.BottomDockWidgetArea):
            return
        self.resizeDocks([dock], [self._results_default_height()], QtCore.Qt.Vertical)

    def _queue_results_title_sync(self, *_args) -> None:
        # Deferred: never change a dock's features in the middle of Qt's own drag/plug.
        QtCore.QTimer.singleShot(0, self, self._sync_results_title_bar)

    def _sync_results_title_bar(self) -> None:
        """A vertical title bar while docked at the top or bottom (it costs width there,
        not a row of the table); a normal one on the sides or floating. Qt does not save
        this feature, so it is re-derived from where the dock actually is."""
        dock = self._results_dock
        flag = QtWidgets.QDockWidget.DockWidgetVerticalTitleBar
        vertical = not dock.isFloating() and self.dockWidgetArea(dock) in (
            QtCore.Qt.TopDockWidgetArea, QtCore.Qt.BottomDockWidgetArea)
        features = dock.features()
        wanted = (features | flag) if vertical else (features & ~flag)
        if wanted != features:
            dock.setFeatures(wanted)

    def _reveal_results_panel(self) -> None:
        self._results_dock.show()
        self._results_dock.raise_()

    def _hint_hidden_results(self, doc_id: str, module_id: str) -> None:
        """A new table for the selected recording landed while the user has the Results
        panel closed: say where it is, once, and re-open nothing."""
        if (self._restoring or self._history.is_applying()
                or not self._results_dock.isHidden()
                or module_id not in self._detection_module_ids()):
            return
        sel = self._current_selection
        if sel is None or sel[0] != doc_id:
            return
        self.statusBar().showMessage(
            f"New events are in the Results panel, which is closed — {_RESULTS_PANEL_PATH}.",
            8000)

    def _hint_hidden_staleness(self, *_args) -> None:
        """Edge-triggered PER SHOWN TABLE: a table on screen just turned (or was selected)
        stale while the Results panel — the only place its amber line lives — is closed.

        Read from the line's real visibility, which the panel also reports when a repoint
        to an empty recording hides it without re-judging anything."""
        panel = self._analysis_panel
        key = None
        if not panel._stale.isHidden():
            res = panel.current_detection()
            key = ("shown",) if res is None else (res.doc_id, res.channel, res.module_id)
        was, self._results_stale_seen = self._results_stale_seen, key
        if (key is not None and key != was and not self._restoring
                and self._results_dock.isHidden()):
            self.statusBar().showMessage(
                "The shown event table is now stale; the Results panel says why — "
                f"{_RESULTS_PANEL_PATH}.", 10000)

    def _park_results_focus(self) -> None:
        """The Results dock is being hidden: if the focus is inside it, put it on the trace
        and remember where it was. Qt would otherwise hand it to the next widget in the
        chain — the workflow strip's first chip, where Space opens a file."""
        self._results_focus_return = None
        try:
            focus = QtWidgets.QApplication.focusWidget() or self.focusWidget()
            if focus is None or not self._results_dock.isAncestorOf(focus):
                return
            self._trace_view.setFocus(QtCore.Qt.OtherFocusReason)
        except RuntimeError:     # a teardown hide: nothing to park
            return
        self._results_focus_return = (focus, self._trace_view)

    def _return_results_focus(self, shown: bool) -> None:
        """The Results toggle re-showed the dock: if the focus is still where hiding it
        parked it, hand it back (normally the event table, so j/k/a/r/f keep working).
        Showing the dock any other way never takes the focus."""
        if not shown:
            return
        memo, self._results_focus_return = self._results_focus_return, None
        if memo is None:
            return
        widget, parked = memo
        focus = QtWidgets.QApplication.focusWidget() or self.focusWidget()
        try:
            if focus is not parked or not widget.isVisible():
                return
            if self._results_dock.isFloating():
                self._results_dock.activateWindow()
            widget.setFocus(QtCore.Qt.OtherFocusReason)
        except RuntimeError:     # the remembered widget is gone
            return

    @staticmethod
    def _detection_module_ids() -> tuple[str, ...]:
        from ..core.sources import DETECTION_MODULE_IDS
        return tuple(DETECTION_MODULE_IDS)

    def _finish_opening_layout(self) -> None:
        """The layout is the user's from here: the opening sizes are never re-imposed."""
        self._opening_layout_pending = False
        self._results_layout_pending = False

    def _read_window_preferences(self, s: QtCore.QSettings, ignored: list[str]) -> None:
        """The "windows/…" keys: the inspector's host and Details, and each child window's
        geometry. Read once, validated like the rest; applied when each is first built."""
        bad: list[str] = []
        host = _choice_setting(s, _SETTING_INSPECTOR_HOST, (_HOST_WINDOW, _HOST_DOCK), bad)
        self._inspector_host = host or _INSPECTOR_DEFAULT_HOST
        self._inspector_details = {}
        for host_name, key in _SETTING_DETAILS.items():
            shown = _bool_setting(s, key, bad)
            if shown is not None:
                self._inspector_details[host_name] = shown
        self._child_geometry = {}
        for key in _CHILD_WINDOW_KEYS:
            raw = _restorable_bytes(s, _geometry_setting(key), bad)
            if raw is not None:
                self._child_geometry[key] = QtCore.QByteArray(bytes(raw))
        self._ignored_window_keys = bad
        ignored.extend(bad)

    def _write_window_preferences(self, s: QtCore.QSettings) -> None:
        s.setValue(_SETTING_INSPECTOR_HOST, self._inspector_host)
        insp = self._event_inspector
        details = insp.details_state() if insp is not None else self._inspector_details
        written = {_SETTING_INSPECTOR_HOST}
        for host_name, key in _SETTING_DETAILS.items():
            if host_name in details:
                s.setValue(key, bool(details[host_name]))
                written.add(key)
        for window in self._child_windows.values():
            self._remember_child_geometry(window)
        for key, geometry in self._child_geometry.items():
            s.setValue(_geometry_setting(key), geometry)
            written.add(_geometry_setting(key))
        # A value this app could not use is not left for the next launch to trip on.
        for key in self._ignored_window_keys:
            if key not in written:
                s.remove(key)

    def _save_settings(self) -> None:
        s = QtCore.QSettings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState(_WINDOW_STATE_VERSION))
        s.setValue("recentFiles", self._recent)
        s.setValue("recentProjects", self._recent_projects)
        s.setValue("lastDir", self._last_dir)
        s.setValue("lastProjectDir", self._last_project_dir)
        self._write_window_preferences(s)
        ws = getattr(self, "_signal_conditioning", None)
        if ws is not None and self._signal_conditioning_placed:
            if ws.property("nanoscopeGeometryShown"):      # not one a Reset just forgot
                s.setValue(self._SC_GEOMETRY_KEY, ws.saveGeometry())
                s.setValue(self._SC_SPLITTER_KEY, ws.splitter_state())
            s.setValue(self._SC_FOLLOW_KEY, bool(ws.follows_main()))
            for key, getter, _setter in self._SC_VIEW_KEYS:
                s.setValue(key, str(getattr(ws, getter)()))
        # Flush to disk HERE rather than relying on QSettings' own destructor. The quit
        # path may end in ``exit_without_destroying_live_threads`` (a worker that would
        # not stop), which runs no destructors at all — so an unsynced setting would be
        # silently lost exactly on the quits that are already going least well.
        s.sync()
