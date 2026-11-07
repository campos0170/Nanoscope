"""The Analysis Explorer's right-hand inspector: Data & analysis and Format, per page.

Architecture ("host and adopt"). A page opts in by providing ``inspector_adapter()`` (a
:class:`~nanoscope.gui.plot_workflow.PageAdapter`: the widgets to show, the page's own
handlers and a few read-only state readers). When the Explorer adds a page,
:meth:`InspectorHost.adopt` MOVES the page's existing control widgets into collapsible
groups, once, and builds the Format tab as a skeleton. A tab switch moves nothing: it flips
a ``QStackedWidget``. Every analytical page supplies its own adapter; a plain Explorer
widget has an empty adapter and leaves the inspector hidden.

This module owns no analysis, reads only the page adapter, imports no page module, and
installs no event filter. Format panels, the mini toolbar and the outline are built at the first gesture
that needs them, never per click.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6 import QtCore, QtWidgets

from .format_inspector import FormatPane, SelectionHub
from .plot_interaction import view_button
from .plot_workflow import InspectorGroup, PageAdapter
from .widget_transfer import prepare_widget_move

__all__ = ["DataPane", "GroupFrame", "InspectorHost", "InspectorSettings", "PagePane"]

_SETTINGS_ROOT = "windows/analysis_explorer/inspector"


# -- settings --------------------------------------------------------------------------------
class InspectorSettings:
    """Per-user view preferences under ``windows/analysis_explorer/inspector/`` in
    ``QSettings``. Nothing here is ever written into a project (format 12 is untouched).

    Every read is validated like ``MainWindow._bool_setting`` and wrapped in ``try``: a
    missing or hostile value gives the default. Expanded flags are per page KIND, not per
    widget instance (Phase 2 will have several figures of one kind); today the kind is the
    Explorer's page id."""

    MIN_WIDTH = 320
    DEFAULT_WIDTH = 360
    MAX_WIDTH = 560

    def __init__(self) -> None:
        self._qsettings: QtCore.QSettings | None = None     # one store, made on first use

    def _store_object(self) -> QtCore.QSettings:
        # One QSettings for every read and write (a new one per read is a Qt object built
        # per group per page). The default-constructed store is shared process-wide, so a
        # second instance elsewhere sees every write.
        if self._qsettings is None:
            self._qsettings = QtCore.QSettings()
        return self._qsettings

    def _raw(self, key: str, default):
        try:
            return self._store_object().value(f"{_SETTINGS_ROOT}/{key}", default)
        except Exception:                       # noqa: BLE001 - a hostile store is the default
            return default

    def _store(self, key: str, value) -> None:
        try:
            self._store_object().setValue(f"{_SETTINGS_ROOT}/{key}", value)
        except Exception:                       # noqa: BLE001 - a preference is never worth a crash
            pass

    def _bool(self, key: str, default: bool) -> bool:
        value = self._raw(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return default

    def _int(self, key: str, default: int, low: int, high: int) -> int:
        value = self._raw(key, default)
        try:
            if isinstance(value, bool):
                return default
            number = int(value)
        except (TypeError, ValueError):
            return default
        return min(max(number, low), high)

    @staticmethod
    def _slug(text: str) -> str:
        return str(text).replace("/", "_").replace("\\", "_") or "_"

    @property
    def tab(self) -> int:
        return self._int("tab", 0, 0, 1)

    @tab.setter
    def tab(self, value: int) -> None:
        self._store("tab", 1 if int(value) == 1 else 0)

    @property
    def width(self) -> int:
        return self._int("width", self.DEFAULT_WIDTH, self.MIN_WIDTH, self.MAX_WIDTH)

    @width.setter
    def width(self, value: int) -> None:
        self._store("width", min(max(int(value), self.MIN_WIDTH), self.MAX_WIDTH))

    @property
    def visible(self) -> bool:
        return self._bool("visible", True)

    @visible.setter
    def visible(self, value: bool) -> None:
        self._store("visible", bool(value))

    @property
    def keep_hand(self) -> bool:
        return self._bool("style/keep_hand", True)

    @keep_hand.setter
    def keep_hand(self, value: bool) -> None:
        self._store("style/keep_hand", bool(value))

    def expanded(self, kind: str, group_key: str, default: bool) -> bool:
        return self._bool(f"groups/{self._slug(kind)}/{self._slug(group_key)}", bool(default))

    def set_expanded(self, kind: str, group_key: str, value: bool) -> None:
        self._store(f"groups/{self._slug(kind)}/{self._slug(group_key)}", bool(value))


# -- one collapsible group -------------------------------------------------------------------
class GroupFrame(QtWidgets.QFrame):
    """A flat checkable header (arrow and title) over a body that holds MOVED widgets.

    Collapsing hides the BODY only, so the moved widgets keep their own visible flags and
    ``isVisible()`` is false for them inside a collapsed group (tests expand first, with
    ``InspectorHost.reveal``)."""

    #: The user clicked the header (not a programmatic ``set_expanded``): persist the flag.
    userToggled = QtCore.Signal(bool)

    def __init__(self, key: str, title: str, *, expanded: bool = True, tooltip: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName(f"inspectorGroup_{key}")
        self.setMinimumWidth(0)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self._header = QtWidgets.QToolButton()
        self._header.setObjectName(f"inspectorGroupHeader_{key}")
        self._header.setText(title.replace("&", "&&"))      # '&' is a mnemonic marker here
        self._header.setCheckable(True)
        self._header.setAutoRaise(True)
        self._header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                   QtWidgets.QSizePolicy.Policy.Fixed)
        self._header.setAccessibleName(f"{title} group")
        self._header.setStyleSheet("QToolButton { text-align:left; font-weight:600; padding:3px 2px; }")
        if tooltip:
            self._header.setToolTip(tooltip)
        column.addWidget(self._header)
        self.body = QtWidgets.QWidget()
        self.body.setMinimumWidth(0)
        inner = QtWidgets.QVBoxLayout(self.body)
        inner.setContentsMargins(4, 2, 4, 8)
        inner.setSpacing(5)
        column.addWidget(self.body)
        self._expanded = True
        self.set_expanded(expanded)
        self._header.toggled.connect(self._header_toggled)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        self._expanded = expanded
        with QtCore.QSignalBlocker(self._header):
            self._header.setChecked(expanded)
        self._header.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded
                                  else QtCore.Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)

    def _header_toggled(self, checked: bool) -> None:
        self.set_expanded(checked)
        self.userToggled.emit(bool(checked))


# -- the Data & analysis tab -----------------------------------------------------------------
class DataPane(QtWidgets.QScrollArea):
    """A vertical column of :class:`GroupFrame`s, with controls fitted to its width."""

    def __init__(self, kind: str = "", settings: InspectorSettings | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind
        self._settings = settings
        self._frames: dict[str, GroupFrame] = {}
        self._visible_when: dict[str, Callable[[], bool]] = {}
        self._forced_hidden: set[str] = set()
        self.setObjectName("analysisDataPane")
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content.setMinimumWidth(0)
        self._column = QtWidgets.QVBoxLayout(content)
        self._column.setContentsMargins(6, 4, 6, 6)
        self._column.setSpacing(2)
        self._column.addStretch(1)
        self.setWidget(content)

    def populate(self, groups: Sequence[InspectorGroup]) -> None:
        """One frame per group, and the group's EXISTING widgets moved into its body (the
        only reparent for plain groups). Every root is made visible explicitly: the old
        Section logic hid the roots of the sections that were not showing, and a widget
        keeps that flag when it moves. Widgets hidden for other reasons are descendants
        and are not touched."""
        for group in groups:
            default = bool(group.expanded)
            expanded = (self._settings.expanded(self._kind, group.key, default)
                        if self._settings is not None else default)
            frame = GroupFrame(group.key, group.title, expanded=expanded, tooltip=group.tooltip)
            frame.userToggled.connect(
                lambda value, key=group.key: self._remember(key, value))
            for widget in group.widgets:
                prepare_widget_move(widget)
                frame.body.layout().addWidget(widget)
                widget.setVisible(True)
            self._column.insertWidget(self._column.count() - 1, frame)
            self._frames[group.key] = frame
            if group.visible_when is not None:
                self._visible_when[group.key] = group.visible_when
        self.sync_visibility()

    def _remember(self, key: str, value: bool) -> None:
        if self._settings is not None:
            self._settings.set_expanded(self._kind, key, value)

    def group(self, key: str) -> GroupFrame | None:
        return self._frames.get(key)

    def group_keys(self) -> tuple[str, ...]:
        return tuple(self._frames)

    def set_group_visible(self, key: str, visible: bool) -> None:
        frame = self._frames.get(key)
        if frame is None:
            return
        if visible:
            self._forced_hidden.discard(key)
        else:
            self._forced_hidden.add(key)
        self._apply_visibility(key)

    def sync_visibility(self) -> None:
        """Re-evaluate every group's ``visible_when`` (a page's own view: the Event tab,
        the Noise view)."""
        for key in self._frames:
            self._apply_visibility(key)

    def _apply_visibility(self, key: str) -> None:
        predicate = self._visible_when.get(key)
        wanted = key not in self._forced_hidden
        if wanted and predicate is not None:
            try:
                wanted = bool(predicate())
            except RuntimeError:            # the page it reads has gone: leave the group alone
                wanted = True
        frame = self._frames[key]
        if frame.isHidden() == wanted:      # shown but unwanted, or hidden but wanted
            frame.setVisible(wanted)


class PagePane(QtWidgets.QStackedWidget):
    """One page's two tabs: ``[DataPane | FormatPane]``."""

    def __init__(self, key: str, adapter: PageAdapter, settings: InspectorSettings,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.page_connections: list[tuple[object, object]] = []
        self.setObjectName(f"analysisPagePane_{key}")
        self.data_pane = DataPane(adapter.kind or key, settings)
        self.data_pane.populate(adapter.groups)
        self.hub = SelectionHub(self)
        self.format_pane = FormatPane(adapter.format, self.hub)
        self.addWidget(self.data_pane)
        self.addWidget(self.format_pane)


# -- the host --------------------------------------------------------------------------------
class InspectorHost(QtWidgets.QWidget):
    """The right-hand column: two tabs over one :class:`PagePane` per page."""

    tabChanged = QtCore.Signal(int)         # 0 Data & analysis, 1 Format

    #: "&&" because a tab's text is a mnemonic string: a bare "&" would be swallowed.
    TAB_TITLES = ("Data && analysis", "Format")

    def __init__(self, parent: QtWidgets.QWidget | None = None, *,
                 settings: InspectorSettings | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("analysisInspector")
        self.setAccessibleName("Plot inspector")
        self.setMinimumWidth(InspectorSettings.MIN_WIDTH)
        self.setMaximumWidth(InspectorSettings.MAX_WIDTH)
        self._settings = settings if settings is not None else InspectorSettings()
        self._panes: dict[str, PagePane] = {}
        self._adapters: dict[str, PageAdapter] = {}
        self._current: str | None = None
        self._user_visible = True
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self._tabs = QtWidgets.QTabBar()
        self._tabs.setObjectName("analysisInspectorTabs")
        self._tabs.setExpanding(True)
        self._tabs.setDrawBase(True)
        for index, title in enumerate(self.TAB_TITLES):
            self._tabs.addTab(title)
            self._tabs.setAccessibleTabName(index, title.replace("&&", "&"))
        self._tabs.setTabToolTip(0, "The page's data, plot, fit and estimator controls (⌘1)")
        self._tabs.setTabToolTip(1, "The figure's text and look (⌘2)")
        self._tabs.setCurrentIndex(self._settings.tab)
        self._tabs.currentChanged.connect(self._tab_changed)
        column.addWidget(self._tabs)
        self._stack = QtWidgets.QStackedWidget()
        column.addWidget(self._stack, 1)
        self.hide()

    # -- queries -------------------------------------------------------------------------------
    def settings(self) -> InspectorSettings:
        return self._settings

    def current_key(self) -> str | None:
        return self._current

    def adapter(self, key: str) -> PageAdapter | None:
        return self._adapters.get(key)

    def pane(self, key: str) -> PagePane | None:
        return self._panes.get(key)

    def selection_hub(self, key: str) -> SelectionHub | None:
        pane = self._panes.get(key)
        return pane.hub if pane is not None else None

    def keys(self) -> tuple[str, ...]:
        return tuple(self._panes)

    # -- the tabs ------------------------------------------------------------------------------
    def set_tab(self, index: int) -> None:
        self._tabs.setCurrentIndex(1 if int(index) == 1 else 0)

    def current_tab(self) -> int:
        return self._tabs.currentIndex()

    def _tab_changed(self, index: int) -> None:
        pane = self._panes.get(self._current or "")
        if pane is not None:
            pane.setCurrentIndex(index)
        self._settings.tab = index
        self.tabChanged.emit(index)

    # -- visibility ----------------------------------------------------------------------------
    def set_user_visible(self, visible: bool) -> None:
        """The toolbar's Inspector toggle: the user's wish, kept in ``QSettings``."""
        self._user_visible = bool(visible)
        self._settings.visible = self._user_visible
        self._sync_shown()

    def user_visible(self) -> bool:
        return self._user_visible

    def has_content(self, key: str | None = None) -> bool:
        """Whether the page's pane shows anything (a bare stand-in page has nothing)."""
        pane = self._panes.get(key if key is not None else (self._current or ""))
        adapter = self._adapters.get(key if key is not None else (self._current or ""))
        if pane is None or adapter is None:
            return False
        spec = adapter.format
        return bool(adapter.groups) or bool(spec.appearance is not None or spec.extras)

    def _sync_shown(self) -> None:
        self.setVisible(self._user_visible and self._current is not None
                        and self.has_content(self._current))

    # -- adopt / retire ------------------------------------------------------------------------
    def adopt(self, key: str, adapter: PageAdapter) -> PagePane:
        """Move the page's control widgets into groups, ONCE, and build the Format skeleton."""
        if key in self._panes:
            self.retire(key)
        pane = PagePane(key, adapter, self._settings)
        self._stack.addWidget(pane)
        self._panes[key] = pane
        self._adapters[key] = adapter
        page = adapter.page
        reveal = lambda target, key=key: self.reveal(target, key=key)       # noqa: E731
        # The page's own hosting hook (Distributions' _show_control_section) reads this.
        page._inspector_reveal = reveal
        adapter.on_adopt()
        for widget in adapter.chrome:
            detach = getattr(widget, "detach_chrome", None)
            if callable(detach):
                detach()
            else:
                widget.hide()
            attach = getattr(widget, "attach_inspector", None)
            if callable(attach):
                attach(reveal)
        self._adopt_status_line(adapter)
        views_changed = adapter.format.views_changed
        if views_changed is not None and hasattr(views_changed, "connect"):
            for slot in (pane.data_pane.sync_visibility, pane.format_pane.refresh):
                views_changed.connect(slot)
                pane.page_connections.append((views_changed, slot))
        # Distributions owns a configuration history separate from its appearance
        # widget. External Undo/Redo must refresh the selected Format controls too.
        history_changed = adapter.history_changed
        if history_changed is not None and hasattr(history_changed, "connect"):
            history_changed.connect(pane.format_pane.refresh)
            pane.page_connections.append((history_changed, pane.format_pane.refresh))
        page.setProperty("nanoscopeInspectorHosted", True)
        # The page dies first, the pane after it: a pane destroyed while its page still
        # holds Python references to the widgets moved into it would turn the page's later
        # attribute reads into "Internal C++ object already deleted".
        page.destroyed.connect(pane.deleteLater)
        if self._current is None or key == self._current:
            pane.setCurrentIndex(self._tabs.currentIndex())
        return pane

    def retire(self, key: str) -> None:
        """Hide the page's pane and let the adapter know; the pane itself is destroyed
        with its page (``page.destroyed``), never earlier: ``QStackedWidget.removeWidget``
        gives the pane to Python and it would be deleted, with every control of the page
        that was moved into it, as soon as this returns. Safe to call twice."""
        pane = self._panes.pop(key, None)
        adapter = self._adapters.pop(key, None)
        if pane is None:
            return
        for signal, slot in pane.page_connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        pane.page_connections.clear()
        pane.format_pane.retire()
        pane.hub.clear()
        pane.hide()
        # The stack owns a QWidgetItem too. Retire its exact Python wrapper before
        # Qt can delete it implicitly and reuse its address for a later QObject.
        prepare_widget_move(pane)
        self._stack.removeWidget(pane)          # hands ownership to Python: take it back, or
        pane.setParent(self._stack)             # the pane and the page's moved controls die here
        if self._current == key:
            self._current = None
            self._sync_shown()
        if adapter is not None:
            try:
                adapter.on_retire()
            except RuntimeError:                # the page is already half gone
                pass

    def show_page(self, key: str | None) -> None:
        """Flip to ``key``'s pane (no reparent, no construction); ``None`` is Analysis home
        and hides the host."""
        pane = self._panes.get(key) if key is not None else None
        if pane is None:
            self._current = None
            self._sync_shown()
            return
        self._current = key
        self._stack.setCurrentWidget(pane)
        pane.setCurrentIndex(self._tabs.currentIndex())
        pane.data_pane.sync_visibility()
        self._sync_shown()

    def refresh_current(self) -> None:
        """Re-read what the front page's panes show (group visibility, the selected
        element's values): the one refresh route for silent paths that emit nothing."""
        pane = self._panes.get(self._current or "")
        if pane is None:
            return
        pane.data_pane.sync_visibility()
        pane.format_pane.refresh()

    # -- navigation ----------------------------------------------------------------------------
    def reveal(self, target: str, *, key: str | None = None,
               focus: QtWidgets.QWidget | None = None) -> bool:
        """Make ``target`` current: a group key, ``"@format"``, ``"@format:<element key>"``
        or an old Section name (through ``adapter.reveal_map``).

        Selects the tab, expands the group and scrolls its frame to the top of the
        viewport (``ensureWidgetVisible`` on the frame, here and only here), and gives
        focus to nothing by default. An explicit page-owned ``focus`` control is scrolled
        into view and focused if enabled. An unknown key falls back to the first visible
        group. Returns whether anything was revealed. ``key`` (default: the front page) names a page
        behind another tab, whose pane is prepared without touching the visible tab."""
        key = key if key is not None else self._current
        pane = self._panes.get(key or "")
        adapter = self._adapters.get(key or "")
        if pane is None or adapter is None:
            return False
        name = adapter.reveal_map.get(target, target)
        front = key == self._current
        if name.startswith("@format"):
            if front:
                self.set_tab(1)
            else:
                pane.setCurrentIndex(1)
            element = name.partition(":")[2]
            if element:
                pane.format_pane.show_element(element)
            if focus is not None and pane.format_pane.isAncestorOf(focus):
                pane.format_pane._scroll.ensureWidgetVisible(focus, 0, 0)
                if front and focus.isEnabled() and focus.isVisible():
                    focus.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            return True
        frame = pane.data_pane.group(name)
        if frame is None or frame.isHidden():
            frame = next((pane.data_pane.group(k) for k in pane.data_pane.group_keys()
                          if not pane.data_pane.group(k).isHidden()), None)
        if frame is None:
            return False
        if front:
            self.set_tab(0)
        else:
            pane.setCurrentIndex(0)
        frame.set_expanded(True)
        area = pane.data_pane
        area.widget().layout().activate()
        if focus is not None and frame.body.isAncestorOf(focus):
            area.ensureWidgetVisible(focus, 0, 0)
            if front and focus.isEnabled() and focus.isVisible():
                focus.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        else:
            area.ensureWidgetVisible(frame, 0, 0)
            area.verticalScrollBar().setValue(frame.y())
        return True

    # -- the page's status line ------------------------------------------------------------------
    def _adopt_status_line(self, adapter: PageAdapter) -> None:
        """Reset view ▾ and the page's canvas actions (Lasso select) on its status line.

        A page that names its navigation gets a fresh ``plot_interaction.view_button`` (the
        hidden toolbar's original is renamed so a lookup by name finds the visible one);
        every analytical page names its own navigation through the adapter."""
        status = adapter.status_bar()
        if status is None:
            return
        page = adapter.page
        for action in adapter.canvas_actions:
            button = QtWidgets.QToolButton()
            button.setObjectName("plotCanvasActionButton")
            button.setDefaultAction(action)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            status.addPermanentWidget(button)
        if adapter.navigation() is not None:
            for old in page.findChildren(QtWidgets.QAbstractButton, "plotResetViewButton"):
                old.setObjectName("plotResetViewButtonLegacy")
            status.addPermanentWidget(view_button(adapter.navigation, adapter.navigations()))
