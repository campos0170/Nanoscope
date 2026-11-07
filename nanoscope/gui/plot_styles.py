"""Style ▾: apply a look to the front page, or to every open page.

A look is a ``PlotAppearance`` that went through ``style_only()``: figure-wide fields and
the title and axis text styles, never words, axis pins, caption places or legend-entry
names. The service is a thin loop over each page's ``adapter.apply_look``, which is
``PlotAppearanceWidget.apply_look``: ONE Undo step per page, so applying a look to N pages
is N steps, one on each page's own history (the two kinds of history are not merged).

``Style ▾`` holds ``This page`` and ``Every open page`` (two submenus, so there is no hidden
mode), the checkbox "Keep text styles I set by hand" (remembered, ticked by default), "Save
this page's look as…", "Manage saved looks…" and "Undo style on N pages", which is on offer
only while every page that batch styled is still at the step the batch left it. A look
reaches the figures that are open now; a figure opened later starts plain.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import weakref

from PySide6 import QtCore, QtWidgets

from .analysis_catalog import ANALYSIS_LABELS
from .plot_appearance import _BUILTIN_PRESETS, PlotAppearance, PlotAppearanceWidget

__all__ = ["StyleBatch", "StyleMenu", "StyleResult", "StyleService", "style_sentence"]

_NOTE = "Applies to figures open now. A figure opened later starts plain."
_REPORT_MS = 15_000


@dataclass(frozen=True)
class StyleResult:
    applied: tuple[str, ...]                    # pages whose appearance moved
    failed: tuple[tuple[str, str], ...]         # (page key, why) for a page that refused
    name: str


@dataclass(frozen=True)
class StyleBatch:
    name: str
    pages: tuple[tuple[str, int], ...]          # (page key, history mark right after)


def style_sentence(result: StyleResult, scope: str = "all", *, unit: str = "page") -> str:
    """What the front page's status line says after a look was applied: how many pages took
    it, that Undo is per page, and any page that refused, with its reason."""
    count = len(result.applied)
    refused = "; ".join(f"{ANALYSIS_LABELS.get(key, key)} ({why})" for key, why in result.failed)
    if count:
        where = (f"this {unit}" if scope == "page" and count == 1
                 else f"{count} {unit}{'s' if count != 1 else ''}")
        text = f"{result.name} applied to {where}; Undo is per {unit}."
        if refused:
            text += f" Not applied to {refused}."
    elif refused:
        text = f"{result.name} was not applied to {refused}."
    else:
        text = f"{result.name}: nothing to change, every {unit} already has it."
    return text


class StyleService(QtCore.QObject):
    """Looks, and the loop that applies one. ``explorer`` is the Analysis Explorer (read
    through its public API only: ``page_ids``, ``current_page_id``, ``adapter``,
    ``statusBar``)."""

    #: The last batch was set, used up or invalidated (the menu's Undo item follows it).
    batchChanged = QtCore.Signal()

    def __init__(self, explorer=None, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._explorer = explorer
        self._last: StyleBatch | None = None
        #: page key -> (mark, undo text, can_redo) right after the batch: any later move of
        #: that page's history (an edit, an Undo, a Redo) makes the batch stale. The mark
        #: alone is not enough: Undo, then a different edit, lands on the same depth.
        self._snapshots: dict[str, tuple[int, str, bool]] = {}
        # A project switch may install a different page under the same key with an
        # identical history description. A batch belongs to the exact original owners.
        self._owners: dict[str, tuple[weakref.ReferenceType, weakref.ReferenceType]] = {}
        self._links: list[tuple[object, object]] = []
        self._record_owner = None
        self._record_tokens: dict[str, object] = {}
        self._imported_owner = None
        self._external_provider = None
        self._external_owners = {}
        self._external_labels = {}
        self._record_epoch = None

    def set_external_provider(self, provider):
        """Include open retained editors in All live figures, without copying ownership.

        The provider returns (stable_id, label, editor) tuples. Editors supply
        apply_look, style, history and undo_style; their own history remains authoritative.
        """
        self._drop_batch()
        self._external_provider = provider

    def _external_targets(self):
        result = {}
        if callable(self._external_provider):
            for key, label, editor in self._external_provider():
                try:
                    if editor is not None and getattr(editor, '_figure', None) is not None:
                        result['retained:' + str(key)] = (str(label), editor)
                except RuntimeError:
                    continue
        return result

    def _controller(self):
        getter = getattr(self._explorer, "figure_workspace", None)
        owner = getter() if callable(getter) else None
        token = (owner, getattr(owner, "_epoch", None))
        if owner is not None and token != self._imported_owner and hasattr(owner, "import_legacy_styles"):
            self._imported_owner = token
            known = {style.name for style in owner.available_styles()}
            legacy = {name: look.style_only().to_dict() for name, look in _BUILTIN_PRESETS.items()
                      if name not in known}
            legacy.update(PlotAppearanceWidget.user_preset_states())
            owner.import_legacy_styles(legacy)
        return owner

    @staticmethod
    def _style_look(style):
        from .figure_style_binding import bind_native_style
        return PlotAppearance.from_mapping(bind_native_style({}, style))

    # -- the looks -----------------------------------------------------------------------------
    def built_in_looks(self) -> list[tuple[str, PlotAppearance]]:
        """The shipped looks in the preset combo's order (the journal looks last)."""
        owner = self._controller()
        if owner is not None:
            return [(style.name, self._style_look(style)) for style in owner.available_styles()
                    if style.provenance.get("builtin")]
        return list(_BUILTIN_PRESETS.items())

    def saved_looks(self) -> list[tuple[str, PlotAppearance]]:
        """The looks saved on this computer, in case-insensitive order."""
        owner = self._controller()
        if owner is not None:
            return [(style.name, self._style_look(style)) for style in sorted(
                owner.available_styles(), key=lambda style: style.name.casefold())
                if not style.provenance.get("builtin")]
        states = PlotAppearanceWidget.user_preset_states()
        return [(name, PlotAppearance.from_mapping(states[name]))
                for name in sorted(states, key=str.casefold)]

    def computer_looks(self):
        from .computer_styles import styles
        return [(style.name, self._style_look(style)) for style in styles()]

    def save_computer_look(self, name, *, overwrite=False):
        from . import computer_styles
        owner = self._controller()
        if owner is not None:
            style = owner.capture_style()
        else:
            from ..core.figure_style import FigureStyle
            from ..core.figure_style_mapping import import_legacy_look
            widget = self._appearance(self._explorer.current_page_id(), "state")
            if widget is None:
                raise ValueError("Choose a figure whose look can be saved.")
            native = widget.state().style_only().to_dict()
            binding = native.get("style_binding")
            if binding:
                from ..core.figure_style import resolve_look
                style = FigureStyle.from_dict(binding["snapshot"])
                from dataclasses import replace
                style = replace(style, look=resolve_look(style, binding.get("overrides", {})).values)
            else:
                style = FigureStyle("captured.local", 1, "Current look",
                    import_legacy_look(native, widget.style_vocabulary).look,
                    provenance={"inherit_native_defaults": True})
        computer_styles.save(style, name, overwrite=overwrite)
        return True

    def name_of(self, look: PlotAppearance) -> str:
        """The shipped or saved name of ``look``, else a plain "The look" (a look handed in
        through ``AnalysisExplorer.apply_look`` has no name of its own)."""
        for name, known in self.built_in_looks() + self.saved_looks():
            if known == look:
                return name
        return "The look"

    def _appearance(self, key: str | None, method: str):
        """The page's appearance widget when it has ``method`` (read from the adapter)."""
        adapter = self._explorer.adapter(key) if self._explorer is not None and key else None
        widget = getattr(getattr(adapter, "format", None), "appearance", None)
        return widget if callable(getattr(widget, method, None)) else None

    def can_save(self) -> bool:
        """Whether the front page has a look to save."""
        owner = self._controller()
        if owner is not None:
            return owner.style_registry() is not None and owner.get(owner.current_id()) is not None
        return self._explorer is not None and \
            self._appearance(self._explorer.current_page_id(), "save_user_preset") is not None

    def can_manage(self) -> bool:
        """Whether any open page can delete a saved look."""
        owner = self._controller()
        if owner is not None:
            return bool(self.saved_looks())
        if self._explorer is None:
            return False
        return any(self._appearance(key, "delete_user_preset") is not None
                   for key in self._explorer.page_ids())

    def save_look(self, name: str, *, overwrite: bool = False) -> bool:
        """Save the front page's look as ``name`` (the appearance widget's own route, which
        keeps the look and drops every typed word)."""
        owner = self._controller()
        if owner is not None:
            owner.save_style(name, overwrite=overwrite)
            return True
        widget = self._appearance(self._explorer.current_page_id() if self._explorer else None,
                                  "save_user_preset")
        return bool(widget is not None and widget.save_user_preset(name, overwrite=overwrite))

    def delete_look(self, name: str) -> bool:
        owner = self._controller()
        if owner is not None:
            style = next((style for style in owner.available_styles() if style.name == name), None)
            if style is None:
                return False
            owner.delete_style(style.style_id)
            return True
        if self._explorer is None:
            return False
        for key in self._explorer.page_ids():
            widget = self._appearance(key, "delete_user_preset")
            if widget is not None:
                return bool(widget.delete_user_preset(name))
        return False

    # -- applying ------------------------------------------------------------------------------
    def _targets(self, scope: str) -> list[str]:
        explorer = self._explorer
        if explorer is None:
            return []
        controller = self._controller()
        if controller is not None:
            front = controller.current_id()
            if scope == "page":
                return [front] if front and controller.get(front) is not None else []
            if scope == "selected":
                return list(controller.selected_ids())
            ordered = [record.spec.figure_id for record in controller.records()]
            if front in ordered:
                ordered.remove(front)
                ordered.insert(0, front)
            ordered.extend(self._external_targets())
            return ordered
        front = explorer.current_page_id()
        if scope == "page":
            return [front] if front in explorer.page_ids() else []
        ordered = list(explorer.page_ids())
        if front in ordered:
            ordered.remove(front)
            ordered.insert(0, front)            # the front page first, then tab order
        if scope == "all":
            ordered.extend(self._external_targets())
        return ordered

    def apply(self, name: str, look: PlotAppearance, *, scope: str = "page",
              keep_hand: bool = True) -> StyleResult:
        """Apply ``look`` to the front page (``scope="page"``) or to every installed page
        (``"all"``). A page that raises is listed in ``failed`` and the others still get the
        look; a page whose appearance did not move (it already had the look) is in neither.
        The front page's status line says what happened."""
        applied: list[str] = []
        failed: list[tuple[str, str]] = []
        marks: list[tuple[str, int]] = []
        snapshots: dict[str, tuple[int, str, bool]] = {}
        controller = self._controller()
        external = self._external_targets() if scope == "all" else {}
        self._external_labels.update({key: label for key, (label, _editor) in external.items()})
        if controller is not None:
            self._explorer.toolbar().commit_focus()
        for key in self._targets(scope):
            adapter = self._explorer.adapter(key) if key not in external else None
            if adapter is None and controller is None and key not in external:
                continue
            try:
                if key in external:
                    _label, editor = external[key]
                    moved = bool(editor.apply_look(look, keep_hand=keep_hand))
                    state = editor.history()
                elif controller is not None:
                    if controller.get(key) is None:
                        raise ValueError("This figure is no longer in the project.")
                    moved = bool(controller.apply_look_to_figure(key, look, keep_hand))
                    state = controller.history(key)
                else:
                    moved = bool(adapter.apply_look(look, keep_hand))
                    state = adapter.history()
            except Exception as exc:            # noqa: BLE001 - one page must not stop the rest
                failed.append((key, str(exc) or type(exc).__name__))
                continue
            if moved:
                applied.append(key)
                marks.append((key, state.mark))
                snapshots[key] = (state.mark, state.undo_text, state.can_redo)
        if marks:
            self._set_batch(StyleBatch(name, tuple(marks)), snapshots, external=external)
        result = StyleResult(tuple(applied), tuple(failed), name)
        self._report(result, scope)
        return result

    def _report(self, result: StyleResult, scope: str) -> None:
        explorer = self._explorer
        if explorer is None:
            return
        try:
            front = explorer.current_page_id()
            adapter = explorer.adapter(front) if front else None
            bar = adapter.status_bar() if adapter is not None else None
            if bar is None:
                bar = explorer.statusBar()      # a page with no status line: the window's own
            controller = self._controller()
            if controller is not None:
                named = tuple((controller.get(key).spec.name if controller.get(key) is not None
                               else self._external_labels.get(key, "Figure"), why) for key, why in result.failed)
                report = StyleResult(result.applied, named, result.name)
                sentence = style_sentence(report, scope, unit="figure")
            else:
                sentence = style_sentence(result, scope)
            bar.showMessage(sentence, _REPORT_MS)
        except RuntimeError:                    # the page or the window is going away
            pass

    # -- the batch -----------------------------------------------------------------------------
    def _set_batch(self, batch: StyleBatch, snapshots: dict[str, tuple[int, str, bool]], *, external=None) -> None:
        self._unlink()
        self._last = batch
        self._snapshots = snapshots
        self._owners = {}
        self._record_owner = self._controller()
        self._record_epoch = getattr(self._record_owner, '_epoch', None)
        self._record_tokens = {}
        external = external or {}
        self._external_owners = {}
        for key, _mark in batch.pages:
            if key not in external:
                continue
            _label, editor = external[key]
            self._external_owners[key] = (weakref.ref(editor), deepcopy(editor.style()), editor.history())
            for signal in (editor._appearance_widget.historyChanged, editor.closed):
                slot = lambda *_args, key=key: self._history_moved(key)
                signal.connect(slot)
                self._links.append((signal, slot))
        if self._record_owner is not None:
            self._record_tokens = {key: self._record_owner.revision_token(key) for key, _ in batch.pages
                                   if key not in self._external_owners}
            for signal, slot in ((self._record_owner.recordChanged, self._history_moved),
                                 (self._record_owner.recordsChanged, self._records_moved)):
                signal.connect(slot)
                self._links.append((signal, slot))
            self.batchChanged.emit()
            return
        for key, _mark in batch.pages:
            if key in self._external_owners:
                continue
            adapter = self._explorer.adapter(key)
            if adapter is not None:
                self._owners[key] = (weakref.ref(adapter), weakref.ref(adapter.page))
            signal = getattr(adapter, "history_changed", None)
            if signal is not None and hasattr(signal, "connect"):
                slot = lambda *_a, key=key: self._history_moved(key)    # noqa: E731
                signal.connect(slot)
                self._links.append((signal, slot))
        self.batchChanged.emit()

    def _unlink(self) -> None:
        for signal, slot in self._links:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._links.clear()

    def _drop_batch(self) -> None:
        self._unlink()
        had = self._last is not None
        self._last = None
        self._snapshots = {}
        self._owners = {}
        self._record_owner = None
        self._record_tokens = {}
        self._external_owners = {}
        self._record_epoch = None
        if had:
            self.batchChanged.emit()

    def _history_moved(self, key: str) -> None:
        """A page of the batch announced a history change: stale if it is not the step the
        batch left (a silent no-op announcement leaves the batch alone)."""
        if self._last is None:
            return
        if key in dict(self._last.pages) and not self._at_snapshot(key):
            self._drop_batch()

    def _records_moved(self) -> None:
        if self._last is not None and not self.batch_intact():
            self._drop_batch()

    def _at_snapshot(self, key: str) -> bool:
        controller = self._controller()
        if key in self._external_owners:
            owner, style, history = self._external_owners[key]
            target = self._external_targets().get(key)
            if (target is None or target[1] is not owner() or controller is not self._record_owner
                    or getattr(controller, '_epoch', None) != self._record_epoch):
                return False
            try:
                return target[1].style() == style and target[1].history() == history
            except RuntimeError:
                return False
        if self._record_owner is not None:
            return (controller is self._record_owner and controller.get(key) is not None
                    and key in self._record_tokens
                    and controller.revision_token(key) == self._record_tokens[key])
        adapter = self._explorer.adapter(key) if self._explorer is not None else None
        snapshot = self._snapshots.get(key)
        owners = self._owners.get(key)
        if (adapter is None or snapshot is None or owners is None
                or owners[0]() is not adapter or owners[1]() is not adapter.page):
            return False
        try:
            state = adapter.history()
        except RuntimeError:
            return False
        return (state.mark, state.undo_text, state.can_redo) == snapshot

    def last_batch(self) -> StyleBatch | None:
        return self._last

    def batch_intact(self) -> bool:
        """Whether every page the last batch styled is still at the step it left."""
        batch = self._last
        return batch is not None and all(self._at_snapshot(key) for key, _ in batch.pages)

    def undo_last_batch(self) -> int:
        """Undo the last batch on every page it styled, once each, while every one of those
        pages is still at the step the batch left it. Returns the pages undone (0 if the
        batch is gone, or any page moved on)."""
        batch = self._last
        if batch is None or self._explorer is None:
            return 0
        if not self.batch_intact():
            self._drop_batch()
            return 0
        controller = self._controller()
        external = self._external_targets()
        self._unlink()                          # our own undos are not "moving on"
        undone = 0
        for key, _ in batch.pages:
            if key in self._external_owners:
                moved = external[key][1].undo_style()
            elif controller is not None:
                moved = controller.undo(key)
            else:
                moved = self._explorer.adapter(key).undo()
            if moved:
                undone += 1
        self._drop_batch()
        return undone


class StyleMenu(QtCore.QObject):
    """The menu behind the toolbar's Style ▾ button.

    The submenus are filled when the menu is first shown (and the saved looks again whenever
    they changed), so nothing is built until someone opens it."""

    def __init__(self, explorer, service: StyleService) -> None:
        super().__init__(explorer)
        self._explorer = explorer
        self._service = service
        self._built = False
        self._builtins = None
        self._saved_looks: tuple[tuple[str, PlotAppearance], ...] | None = None
        self._menu = QtWidgets.QMenu()
        self._menu.setObjectName("analysisStyleMenu")
        self._scopes = {"page": self._menu.addMenu("This page"),
                        "selected": self._menu.addMenu("Selected figures"),
                        "all": self._menu.addMenu("Every open page")}
        self._scopes["selected"].menuAction().setVisible(False)
        self._default_menu = self._menu.addMenu("Project default for new figures")
        self._default_menu.menuAction().setVisible(False)
        self._saved_headers: dict[str, object] = {}
        self._menu.addSeparator()
        self._keep = self._menu.addAction("Keep text styles I set by hand")
        self._keep.setCheckable(True)
        self._keep.toggled.connect(self._keep_toggled)
        self._save = self._menu.addAction("Save this page's look as…")
        self._save.triggered.connect(self._ask_save)
        self._manage = self._menu.addAction("Manage saved looks…")
        self._manage.triggered.connect(self._open_manager)
        self._save_computer = self._menu.addAction("Save this figure's look on this computer…")
        self._save_computer.triggered.connect(lambda: self._ask_save(computer=True))
        self._manage_computer = self._menu.addAction("Manage styles on this computer…")
        self._manage_computer.triggered.connect(lambda: self._open_manager(computer=True))
        self._undo = self._menu.addAction("Undo style")
        self._undo.triggered.connect(self._undo_batch)
        self._menu.addSeparator()
        self._note = self._menu.addAction(_NOTE)
        self._note.setEnabled(False)
        self._manager: QtWidgets.QDialog | None = None
        self._computer_manager: QtWidgets.QDialog | None = None
        self._menu.aboutToShow.connect(self._sync)
        service.batchChanged.connect(self._sync_undo)

    def menu(self) -> QtWidgets.QMenu:
        return self._menu

    def scope_menu(self, scope: str) -> QtWidgets.QMenu:
        """``"page"`` -> This page, ``"all"`` -> Every open page."""
        return self._scopes[scope]

    # -- filling -------------------------------------------------------------------------------
    @staticmethod
    def _label(name: str) -> str:
        return name.replace("&", "&&")          # '&' is a mnemonic marker in a menu text

    def _build(self, looks) -> None:
        self._saved_looks = None
        for scope, submenu in self._scopes.items():
            submenu.clear()
            submenu.addSection("Built in")
            for name, look in looks:
                action = submenu.addAction(self._label(name))
                action.triggered.connect(
                    lambda _checked=False, name=name, look=look, scope=scope:
                    self._apply(name, look, scope))
            self._saved_headers[scope] = submenu.addSection(
                "Project styles" if self._service._controller() is not None else "Saved on this computer")
        self._built = True

    def _fill_saved(self) -> None:
        looks = tuple(self._service.saved_looks())
        try:
            computer = tuple(self._service.computer_looks()) if self._service._controller() is not None else ()
            computer_error = ""
        except ValueError as exc:
            computer, computer_error = (), str(exc)
        signature = (looks, computer, computer_error)
        if signature == self._saved_looks:
            return
        # Actions capture the look itself. Replacing a saved look under its existing
        # name must rebuild those actions just as adding or deleting a name does.
        self._saved_looks = signature
        for scope, submenu in self._scopes.items():
            actions = submenu.actions()
            header = self._saved_headers[scope]
            for stale in actions[actions.index(header) + 1:]:
                submenu.removeAction(stale)
                stale.deleteLater()
            if not looks:
                submenu.addAction("None saved yet").setEnabled(False)
            for name, look in looks:
                action = submenu.addAction(self._label(name))
                action.triggered.connect(
                    lambda _checked=False, name=name, look=look, scope=scope:
                    self._apply(name, look, scope))
            if self._service._controller() is not None:
                submenu.addSection("On this computer")
                if not computer:
                    action = submenu.addAction(computer_error or "None saved on this computer")
                    action.setEnabled(False)
                for name, look in computer:
                    action = submenu.addAction(self._label(name))
                    action.setToolTip("Apply this pinned computer style; existing project style definitions are unchanged.")
                    action.triggered.connect(lambda _checked=False, name=name, look=look, scope=scope:
                                             self._apply(name, look, scope))

    def _sync(self) -> None:
        controller = self._service._controller()
        looks = tuple(self._service.built_in_looks())
        signature = (controller, getattr(controller, "_epoch", None), looks)
        if not self._built or signature != self._builtins:
            self._build(looks)
            self._builtins = signature
        self._fill_saved()
        explorer = self._explorer
        self._default_menu.menuAction().setVisible(controller is not None)
        self._default_menu.clear()
        if controller is not None:
            registry = controller.style_registry()
            for style in controller.available_styles():
                action = self._default_menu.addAction(self._label(style.name))
                action.setCheckable(True)
                action.setChecked(registry is not None and style.ref == registry.default_ref)
                action.triggered.connect(lambda _checked=False, ref=style.ref: self._set_default(ref))
        self._scopes["page"].setTitle("This figure" if controller is not None else "This page")
        self._scopes["all"].setTitle("All live figures" if controller is not None else "Every open page")
        self._scopes["selected"].menuAction().setVisible(controller is not None)
        for scope, menu in self._scopes.items():
            readable = controller is None or (controller.style_registry() is not None and not controller.is_read_only())
            menu.setEnabled(readable and bool(self._service._targets(scope)))
        self._note.setText("All live figures includes dormant figures and open Study/Report editors. New figures use the project default."
                           if controller is not None else _NOTE)
        with QtCore.QSignalBlocker(self._keep):
            self._keep.setChecked(explorer.inspector().settings().keep_hand)
        self._save.setEnabled(self._service.can_save())
        self._save.setText("Save this figure's look as…" if controller is not None
                           else "Save this page's look as…")
        self._manage.setEnabled(self._service.can_manage())
        self._save_computer.setVisible(controller is not None)
        self._manage_computer.setVisible(controller is not None)
        self._save_computer.setEnabled(self._service.can_save())
        self._sync_undo()

    def _sync_undo(self) -> None:
        batch = self._service.last_batch()
        intact = self._service.batch_intact()
        count = len(batch.pages) if batch is not None else 0
        noun = "figure" if self._service._controller() is not None else "page"
        self._undo.setText(f"Undo style on {count} {noun}{'s' if count != 1 else ''}"
                           if intact else "Undo style")
        self._undo.setEnabled(intact)

    # -- the verbs -----------------------------------------------------------------------------
    def _keep_toggled(self, checked: bool) -> None:
        self._explorer.inspector().settings().keep_hand = bool(checked)

    def _apply(self, name: str, look: PlotAppearance, scope: str) -> None:
        keep = self._explorer.inspector().settings().keep_hand
        self._service.apply(name, look, scope=scope, keep_hand=keep)

    def _undo_batch(self) -> None:
        self._service.undo_last_batch()

    def _set_default(self, ref) -> None:
        try:
            self._service._controller().set_default_style(ref)
            self._say("Project default changed. New figures use this style; existing figures keep their look.")
        except ValueError as exc:
            self._say(str(exc))

    def _say(self, text: str) -> None:
        try:
            self._explorer.statusBar().showMessage(text, _REPORT_MS)
        except RuntimeError:
            pass

    def _save_token(self):
        owner = self._service._controller()
        if owner is not None:
            key = owner.current_id()
            return owner, owner._epoch, key, owner.revision_token(key)
        key = self._explorer.current_page_id()
        return self._explorer, key, self._explorer.adapter(key)

    def _ask_save(self, _checked=False, *, computer=False) -> None:
        token = self._save_token()
        name, accepted = QtWidgets.QInputDialog.getText(
            self._explorer, "Save look", "Name for this page's look:")
        name = name.strip()[:100]
        if not accepted or not name:
            return
        if token != self._save_token():
            self._say("The current figure or project changed; open Save look again.")
            return
        try:
            overwrite = name in dict(self._service.computer_looks() if computer else self._service.saved_looks())
        except ValueError as exc:
            self._say(str(exc))
            return
        if overwrite and QtWidgets.QMessageBox.question(
                self._explorer, "Replace saved look?",
                f"Replace the saved look ‘{name}’?") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if token != self._save_token():
            self._say("The current figure or project changed; open Save look again.")
            return
        try:
            save = self._service.save_computer_look if computer else self._service.save_look
            saved = save(name, overwrite=overwrite)
        except ValueError as exc:
            self._say(str(exc))
            return
        if saved:
            place = "in this project" if not computer and self._service._controller() is not None else "on this computer"
            self._say(f"Saved the look ‘{name}’ {place}.")
        else:
            self._say(f"The look ‘{name}’ could not be saved.")

    def _open_manager(self, _checked=False, *, computer=False) -> None:
        field = "_computer_manager" if computer else "_manager"
        manager = getattr(self, field)
        if manager is None:
            manager = _ManageLooks(self._explorer, self._service, computer=computer)
            setattr(self, field, manager)
        manager.refill()
        manager.show()
        manager.raise_()


class _ManageLooks(QtWidgets.QDialog):
    """The saved looks of this computer, with Delete. Closed with ``hide()`` and kept; it
    is a child of the Explorer and goes with it."""

    def __init__(self, explorer, service: StyleService, *, computer=False) -> None:
        super().__init__(explorer)
        self._computer = computer
        self.setObjectName("analysisManageLooks")
        self.setWindowTitle("Saved looks")
        self._service = service
        column = QtWidgets.QVBoxLayout(self)
        self._note = QtWidgets.QLabel()
        self._note.setWordWrap(True)
        column.addWidget(self._note)
        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("analysisManageLooksList")
        self.list.setAccessibleName("Saved looks")
        column.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        self.edit_button = QtWidgets.QPushButton("Edit…")
        self.edit_button.clicked.connect(self._edit)
        row.addWidget(self.edit_button)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.setObjectName("analysisManageLooksDelete")
        self.delete_button.clicked.connect(self._delete)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.hide)
        row.addWidget(self.delete_button)
        row.addStretch(1)
        row.addWidget(self.close_button)
        column.addLayout(row)
        self.list.currentItemChanged.connect(self._sync)

    def refill(self) -> None:
        owner = self._service._controller()
        self._owner_token = (owner, getattr(owner, "_epoch", None))
        project = owner is not None and not self._computer
        self.setWindowTitle("Project styles" if project else "Saved looks")
        self._note.setText("Styles saved in this project. Edits make a new revision; apply it to change existing figures."
                           if project else "Looks saved on this computer. Deleting one does not change any figure.")
        self.edit_button.setVisible(project or self._computer)
        self.list.clear()
        self._definitions = {style.name: style for style in owner.available_styles()} if project else {}
        try:
            looks = self._service.computer_looks() if self._computer else self._service.saved_looks()
        except ValueError as exc:
            looks = ()
            self._note.setText(str(exc))
        self._local_snapshot = dict(looks) if self._computer else {}
        if self._computer and looks:
            from .computer_styles import styles
            self._definitions = {style.name: style for style in styles()}
        for name, _look in looks:
            self.list.addItem(name)
        if self.list.count():
            self.list.setCurrentRow(0)
        self._sync()

    def _sync(self, *_args) -> None:
        self.edit_button.setEnabled(self.list.currentItem() is not None)
        self.delete_button.setEnabled(self.list.currentItem() is not None
                                      and (self._computer or self._service.can_manage()))

    def _valid_selection(self, name):
        owner = self._service._controller()
        token = (owner, getattr(owner, "_epoch", None))
        if self._computer:
            try:
                if self._local_snapshot.get(name) == dict(self._service.computer_looks()).get(name):
                    return True
            except ValueError:
                pass
            self.refill()
            self._note.setText("The saved computer style changed; select it again.")
            return False
        expected = self._definitions.get(name)
        current = next((style for style in owner.available_styles() if style.name == name), None) if owner else None
        if token == self._owner_token and (owner is None or expected == current):
            return True
        self.refill()
        self._note.setText("The project or style changed; select the style again.")
        return False

    def _delete(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        name = item.text()
        if not self._valid_selection(name):
            return
        if QtWidgets.QMessageBox.question(
                self, "Delete saved look?", f"Delete the saved look ‘{name}’?"
        ) == QtWidgets.QMessageBox.StandardButton.Yes:
            if not self._valid_selection(name):
                return
            try:
                if self._computer:
                    from .computer_styles import delete
                    delete(name)
                else:
                    self._service.delete_look(name)
                self.refill()
            except ValueError as exc:
                self._note.setText(str(exc))

    def _edit(self) -> None:
        item, owner = self.list.currentItem(), self._service._controller()
        if item is None or (owner is None and not self._computer):
            return
        name = item.text()
        if not self._valid_selection(name):
            return
        style = self._definitions.get(name)
        if style is None:
            return
        from .project_style_editor import ProjectStyleEditor
        dialog = ProjectStyleEditor(style, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            if not self._valid_selection(name):
                return
            try:
                if self._computer:
                    from dataclasses import replace
                    from .computer_styles import save
                    save(replace(style, look=dialog.look(), columns=dialog.columns()), style.name, overwrite=True)
                else:
                    owner.save_style(style.name, overwrite=True, look=dialog.look(), columns=dialog.columns())
                self.refill()
            except ValueError as exc:
                self._note.setText(str(exc))
