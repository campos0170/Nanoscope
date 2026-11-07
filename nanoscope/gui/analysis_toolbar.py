"""The Analysis Explorer's one toolbar: Undo, Redo, Update plot, Style, Reset figure, Export.

``Undo | Redo | Update plot (n) ▾ | Style ▾ | … | Reset figure | Export ▾ | Inspector``,
identical on every page and bound to the page in front through its
:class:`~nanoscope.gui.plot_workflow.PageAdapter`. Nothing here analyses: every click lands
on a handler the page already had (its Undo, its verb, its Reset figure button, its export
actions), and every label is read from state the page already keeps.

The only keys set in this module are the two of Update plot (Ctrl+Return and Ctrl+Enter;
Qt treats the keypad key as a different one). The Undo and Redo actions carry NO key: the
pages keep ⌘Z and ⇧⌘Z on their own roots, and a second owner would make Qt fire neither.
"""
from __future__ import annotations

import weakref
from collections.abc import Mapping

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from .plot_workflow import ExportEntry, PageAdapter

__all__ = ["ExplorerToolbar", "UpdateButton"]

_KIND_ORDER = {"figure": 0, "data": 1, "recipe": 2, "page": 3}
_RESET_TIP = ("Put the text you typed on the figure (title, axis labels, fit-label wording, "
              "caption) and its look back to their defaults, as one Undo step. Data, fits and "
              "analysis settings are kept.")
_AMBER = "QToolButton { color:#ffca78; }"


class UpdateButton(QtWidgets.QToolButton):
    """``Update plot`` / ``Update plot (n)`` with a drop-down (``MenuButtonPopup``). The
    dynamic property ``pending`` is true while ``n`` is above zero and drives the amber."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("analysisUpdateButton")
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setProperty("pending", False)

    def show_pending(self, pending: bool) -> None:
        if bool(self.property("pending")) != bool(pending):
            self.setProperty("pending", bool(pending))
            self.setStyleSheet(_AMBER if pending else "")


class ExplorerToolbar(QtWidgets.QToolBar):
    """The toolbar of the Analysis Explorer (objectName ``analysisToolbar``)."""

    refreshFromRecordingsRequested = QtCore.Signal()
    saveFigureRequested = QtCore.Signal()
    saveBundleRequested = QtCore.Signal()
    figureLibraryRequested = QtCore.Signal()
    exportSelectedRequested = QtCore.Signal()
    duplicateRequested = QtCore.Signal()
    #: The buttons were re-read (``sync`` ran): the Explorer's header chip follows the reasons.
    synced = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Analysis", parent)
        self.setObjectName("analysisToolbar")
        self.setAccessibleName("Analysis toolbar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        #: Phase 2 puts New figure ▾ and Duplicate here; nothing else is ever placed there.
        self.leading_actions: list[QtGui.QAction] = []
        self._figure_commands = False
        self.figures_action = QtGui.QAction("Figures", self)
        self.figures_action.setObjectName("analysisFiguresToggle")
        self.figures_action.setCheckable(True)
        self.figures_action.setChecked(True)
        self.figures_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+L"))
        self.figures_action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self.figures_action.setToolTip("Show or hide the Figures list to give the plot more space (⌘⇧L / Ctrl+Shift+L).")
        self.addAction(self.figures_action)
        self.widgetForAction(self.figures_action).setAccessibleName("Show or hide the Figures list")
        self.figures_action.setVisible(False)
        self._key: str | None = None
        self._adapter_ref: weakref.ref | None = None
        self._connections: list[tuple[object, object]] = []
        self._reasons: tuple[str, ...] = ()
        self._count = 0

        self.undo_action = self.addAction("↶ Undo")
        self.undo_action.setObjectName("analysisUndo")
        self.undo_action.triggered.connect(self._undo)
        self.redo_action = self.addAction("↷ Redo")
        self.redo_action.setObjectName("analysisRedo")
        self.redo_action.triggered.connect(self._redo)

        self.update_action = QtGui.QAction("Update plot", self)
        self.update_action.setObjectName("analysisUpdate")
        self.update_action.setShortcuts([QtGui.QKeySequence("Ctrl+Return"),
                                         QtGui.QKeySequence("Ctrl+Enter")])
        self.update_action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        self.update_action.triggered.connect(self.run_update)
        self.update_button = UpdateButton()
        self.update_button.setDefaultAction(self.update_action)
        self._update_menu = QtWidgets.QMenu(self.update_button)
        self._refresh_item = self._update_menu.addAction("Refresh from recordings")
        self._refresh_item.setToolTip("Re-pull this page from the open recordings, as the old "
                                      "Refresh current button did.")
        self._refresh_item.triggered.connect(self.refreshFromRecordingsRequested)
        self._reset_from_menu = self._update_menu.addAction("Reset figure")
        self._reset_from_menu.setToolTip(_RESET_TIP)
        self._reset_from_menu.triggered.connect(self._reset_figure)
        self._reset_from_menu.setVisible(False)
        self.update_button.setMenu(self._update_menu)
        self.addWidget(self.update_button)

        self.style_button = QtWidgets.QToolButton()
        self.style_button.setObjectName("analysisStyleButton")
        self.style_button.setText("Style ▾")
        self.style_button.setAccessibleName("Style")
        self.style_button.setMaximumWidth(260)
        self.style_button.setToolTip("Apply a look to this page or to every open page.")
        self.style_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.addWidget(self.style_button)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        self.reset_figure_button = QtWidgets.QToolButton()
        self.reset_figure_button.setObjectName("analysisResetFigure")
        self.reset_figure_button.setText("Reset figure")
        self.reset_figure_button.setAccessibleName("Reset figure")
        self.reset_figure_button.setToolTip(_RESET_TIP)
        self.reset_figure_button.clicked.connect(self._reset_figure)
        self._reset_widget_action = self.addWidget(self.reset_figure_button)

        self.export_button = QtWidgets.QToolButton()
        self.export_button.setObjectName("analysisExportButton")
        self.export_button.setText("Export ▾")
        self.export_button.setAccessibleName("Export")
        self.export_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._export_menu = QtWidgets.QMenu(self.export_button)
        self._export_menu.aboutToShow.connect(self._fill_export_menu)
        self.export_button.setMenu(self._export_menu)
        self._library_enabled = False
        self._batch_available = self._batch_enabled = False
        self._saved_exports = None
        self.addWidget(self.export_button)

        self.inspector_toggle = QtWidgets.QToolButton()
        self.inspector_toggle.setObjectName("analysisInspectorToggle")
        self.inspector_toggle.setText("Inspector")
        self.inspector_toggle.setAccessibleName("Show or hide the inspector")
        self.inspector_toggle.setToolTip("Show or hide the inspector on the right.")
        self.inspector_toggle.setCheckable(True)
        self.inspector_toggle.setChecked(True)
        self.addWidget(self.inspector_toggle)

        self.bind(None, None)

    # -- the bound page ------------------------------------------------------------------------
    def adapter(self) -> PageAdapter | None:
        """The bound adapter, or None (Home, a stand-in page, a page that has gone)."""
        adapter = self._adapter_ref() if self._adapter_ref is not None else None
        if adapter is None:
            return None
        try:
            if not shiboken6.isValid(adapter.page):
                return None
        except TypeError:
            return None
        return adapter

    def bound_key(self) -> str | None:
        return self._key

    def set_figure_commands(self, new_menu=None) -> None:
        """Project aliases share the Figures list's menu and command, never state."""
        active = new_menu is not None
        if active and not self.leading_actions:
            self.new_figure_button = QtWidgets.QToolButton()
            self.new_figure_button.setText("＋ New")
            self.new_figure_button.setAccessibleName("New figure")
            self.new_figure_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            new_action = self.insertWidget(self.undo_action, self.new_figure_button)
            self.duplicate_action = QtGui.QAction("Duplicate", self)
            self.duplicate_action.setToolTip("Duplicate the selected live figure or saved snapshot, keeping its words and look.")
            self.duplicate_action.triggered.connect(self.duplicateRequested)
            self.insertAction(self.undo_action, self.duplicate_action)
            self.leading_actions = [new_action, self.duplicate_action]
        self._figure_commands = active
        self.figures_action.setVisible(active)
        self.figures_action.setEnabled(active)
        if self.leading_actions:
            self.new_figure_button.setMenu(new_menu)
            for action in self.leading_actions:
                action.setVisible(active)
        # Reset figure was the pre-Figures-list stop-gap. Keep its exact command
        # in the Update menu while the toolbar has explicit New and Duplicate.
        self._reset_widget_action.setVisible(not active)
        self._reset_from_menu.setVisible(active)
        self.style_button.setMaximumWidth(180 if active else 260)

    def set_duplicate_enabled(self, enabled):
        if self.leading_actions:
            self.duplicate_action.setEnabled(bool(enabled))

    def set_style_reference(self, binding=None, *, local_overrides=False, has_figure=False) -> None:
        """Show the pinned identity, never infer a preset by matching rendered values."""
        name, revision = "", None
        if binding and isinstance(binding, Mapping):
            ref, snapshot = binding.get("ref", {}), binding.get("snapshot", {})
            if (isinstance(ref, Mapping) and isinstance(snapshot, Mapping)
                    and ref.get("style_id") == snapshot.get("style_id")
                    and ref.get("revision") == snapshot.get("revision")):
                name = str(snapshot.get("name") or ref.get("style_id") or "")
                revision = ref.get("revision")
                local_overrides = local_overrides or bool(binding.get("overrides"))
        if name and type(revision) is int:
            short = name if len(name) <= 24 else name[:23] + "…"
            suffix = " · ●" if local_overrides else ""
            text = f"{short} · r{revision}{suffix} ▾"
            tip = f"Style: {name}\nPinned revision {revision}."
            tip += ("\n● Figure or element look overrides are set here." if local_overrides
                    else "\nLook inherited from this style.")
        elif has_figure:
            text = "Style: unavailable ▾" if binding else "Style: custom ▾"
            tip = ("The figure's style reference is unavailable; its saved appearance is retained."
                   if binding else "This figure has no named style reference. Its own saved appearance is retained.")
        else:
            text, tip = "Style ▾", "Choose a figure to inspect its style."
        self.style_button.setText(text)
        self.style_button.setToolTip(tip)
        self.style_button.setAccessibleName(tip.replace("\n", " "))

    def bind(self, key: str | None, adapter: PageAdapter | None) -> None:
        """Bind the buttons to ``adapter`` (held weakly). ``None`` is Home or a stand-in
        page: placeholder texts, every figure button disabled."""
        for signal, slot in self._connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._connections.clear()
        self._saved_exports = None
        self._key = key if adapter is not None else None
        self._adapter_ref = weakref.ref(adapter) if adapter is not None else None
        if adapter is not None:
            for signal in (adapter.pending_changed, adapter.history_changed):
                if signal is not None and hasattr(signal, "connect"):
                    signal.connect(self.sync)
                    self._connections.append((signal, self.sync))
        self.sync()

    def bind_saved(self, key: str, exports) -> None:
        """Offer frozen exports without retaining any hidden live editor as a target."""
        self.bind(None, None)
        self._key = key
        self._saved_exports = exports
        self._library_enabled = False
        self.sync()

    def forget(self, key: str) -> None:
        """Drop the bound adapter when its page is being removed, before the page is handed
        back for ``deleteLater``."""
        if self._key == key:
            self.bind(None, None)

    def sync(self) -> None:
        """Re-read ``pending()``, ``history()`` and the exports' enablement."""
        adapter = self.adapter()
        bound = adapter is not None
        pending = None
        history = None
        if bound:
            try:
                pending = adapter.pending()
                history = adapter.history()
            except RuntimeError:                # the page's C++ object is already gone
                bound = False
        self.undo_action.setEnabled(bool(bound and history.can_undo))
        self.redo_action.setEnabled(bool(bound and history.can_redo))
        self.undo_action.setToolTip(history.undo_text if bound else "Nothing to undo")
        self.redo_action.setToolTip(history.redo_text if bound else "Nothing to redo")
        count = pending.count if bound else 0
        self._count = count
        self._reasons = tuple(pending.reasons) if bound and count > 0 else ()
        self.update_action.setText(f"Update plot ({count})" if count > 0 else "Update plot")
        self.update_action.setEnabled(bool(bound and not pending.busy))
        self.update_button.show_pending(count > 0)
        self.update_action.setToolTip(self._update_tooltip(adapter, pending) if bound
                                      else "Run the page's own update.")
        self._refresh_item.setEnabled(bound)
        self.reset_figure_button.setEnabled(bound)
        self._reset_from_menu.setEnabled(bound)
        self.reset_figure_button.setToolTip(self._reset_tooltip(adapter) if bound else _RESET_TIP)
        self.export_button.setEnabled(bound or self._saved_exports is not None or self._batch_enabled)
        self.inspector_toggle.setEnabled(bound)
        self.style_button.setEnabled(self._saved_exports is None)
        self.synced.emit()

    def pending_count(self) -> int:
        """The bound page's pending count as of the last ``sync``."""
        return self._count

    def pending_reasons(self) -> tuple[str, ...]:
        """Why the bound page is out of date, as of the last ``sync`` (empty at count 0)."""
        return self._reasons

    @staticmethod
    def _update_tooltip(adapter: PageAdapter, pending) -> str:
        lines = [adapter.update_tooltip or "Run the page's own update.",
                 "Shortcut: ⌘Return (Ctrl+Return)."]
        if pending.busy:
            lines.append("Busy: a background gather is running.")
        elif pending.reasons:
            lines.append("Out of date: " + "; ".join(pending.reasons) + ".")
        return "\n".join(lines)

    @staticmethod
    def _reset_tooltip(adapter: PageAdapter) -> str:
        """The page's own Reset figure tooltip (it says what the page keeps), else the
        shared one. Read through the appearance widget the adapter names, not the page."""
        button = getattr(adapter.format.appearance, "reset_button", None)
        try:
            text = button.toolTip() if button is not None else ""
        except RuntimeError:
            text = ""
        return text or _RESET_TIP

    # -- the verbs -----------------------------------------------------------------------------
    def commit_focus(self) -> None:
        """``clearFocus()`` on a focused widget inside the inspector. A ``QLineEdit`` commits
        on ``editingFinished`` and ⌘Return is a shortcut, so without this a title typed but
        not yet committed would be left out of the update."""
        focus = QtWidgets.QApplication.focusWidget()
        node = focus
        while node is not None:
            if node.objectName() == "analysisInspector":
                focus.clearFocus()
                return
            node = node.parentWidget()

    def run_update(self) -> None:
        """Update plot: commit the focused edit, then the page's own verb. Does nothing on
        Home, with no page, or while the page is busy."""
        adapter = self.adapter()
        if adapter is None:
            return
        try:
            if adapter.pending().busy:
                return
        except RuntimeError:
            return
        self.commit_focus()
        adapter.update()
        self.sync()

    def _undo(self) -> None:
        adapter = self.adapter()
        if adapter is not None:
            adapter.undo()
            self.sync()

    def _redo(self) -> None:
        adapter = self.adapter()
        if adapter is not None:
            adapter.redo()
            self.sync()

    def _reset_figure(self) -> None:
        adapter = self.adapter()
        if adapter is not None:
            adapter.reset_figure()
            self.sync()

    # -- Style ▾ and Export ▾ ------------------------------------------------------------------
    def set_style_menu(self, menu: QtWidgets.QMenu) -> None:
        self.style_button.setMenu(menu)

    def set_library_enabled(self, enabled: bool) -> None:
        """Whether "Save to Figure Library…" and "Save figure bundle…" are on offer: exactly
        when the old band-2 buttons were (a page in front whose status is not "empty")."""
        self._library_enabled = bool(enabled)

    def set_batch_export_enabled(self, enabled, *, available=True):
        self._batch_available = bool(available)
        self._batch_enabled = bool(available and enabled)

    def library_enabled(self) -> bool:
        return self._library_enabled

    def _fill_export_menu(self) -> None:
        menu = self._export_menu
        menu.clear()
        adapter = self.adapter()
        exports = adapter.exports if adapter is not None else self._saved_exports
        if exports is not None:
            try:
                entries: list[ExportEntry] = sorted(
                    exports(), key=lambda e: _KIND_ORDER.get(e.kind, 9))
            except RuntimeError:
                entries = []
            for entry in entries:
                action = menu.addAction(entry.text)
                action.setEnabled(bool(entry.enabled()) if entry.enabled is not None else True)
                action.triggered.connect(lambda _checked=False, run=entry.run: run())
            if entries:
                menu.addSeparator()
        if self._batch_available:
            selected = menu.addAction("Export selected figures…")
            selected.setEnabled(self._batch_enabled)
            selected.triggered.connect(self.exportSelectedRequested)
        if self._saved_exports is not None:
            return
        library = menu.addAction("Snapshot to Figures…" if self._figure_commands else "Save to Figure Library…")
        library.setEnabled(adapter is not None and self._library_enabled)
        library.triggered.connect(self.saveFigureRequested)
        bundle = menu.addAction("Bundle…" if self._figure_commands else "Save figure bundle…")
        bundle.setEnabled(adapter is not None and self._library_enabled)
        bundle.triggered.connect(self.saveBundleRequested)
        menu.addAction("Snapshot details…" if self._figure_commands else "Open Figure Library").triggered.connect(self.figureLibraryRequested)
