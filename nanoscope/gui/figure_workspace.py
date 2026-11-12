"""Project-owned figures with a bounded cache of native analytical surfaces.

Records own durable state. A page is a temporary editor of one record; listing, saving,
duplicating and styling dormant figures never require a page or run an analysis.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from uuid import uuid4

from PySide6 import QtCore

from ..core.figure_codecs import NATIVE_KINDS, decode_legacy, record_to_artifact
from ..core.figure_migration_v12 import migrate_workspace_v12
from ..core.figure_spec import FIGURE_ARTIFACT_KIND, FigureRecord, thaw_json
from ..core.figure_style import FigureStyle, inherits_native_defaults, resolve_look
from ..core.figure_style_registry import (
    registry_from_workspace, registry_to_workspace, resolve_ref)
from ..core.project import AnalyticalWorkspace, WorkspaceArtifact, WORKSPACE_VERSION
from .analysis_catalog import ANALYSIS_LABELS, PAGE_IDS
from .plot_appearance import _BUILTIN_PRESETS, PlotAppearance
from .plot_workflow import HistoryState

WORKSPACE_KEY = "figure_workspace_v1"
META_KEY = "figure_workspace"
DEFAULT_LOOK = "7 pt Arial · Nature, Elsevier"


@dataclass
class _Entry:
    record: FigureRecord
    history: object = None
    page: object = None
    adapter: object = None
    generation: int = 0
    identity: str = field(default_factory=lambda: str(uuid4()))
    used: int = 0
    links: list = field(default_factory=list)
    request_generation: int = 0
    busy: bool = False
    capturing: bool = False


class FigureWorkspace(QtCore.QObject):
    recordsChanged = QtCore.Signal()
    recordChanged = QtCore.Signal(str)
    currentChanged = QtCore.Signal(str)
    requestFailed = QtCore.Signal(str, str)
    selectionChanged = QtCore.Signal()
    stylesChanged = QtCore.Signal()

    def __init__(self, *, factory, mount, unmount, initialize, bind_sources,
                 current_binding=lambda _kind: {}, dirty=lambda: None,
                 commit_focus=lambda: None, parent=None, resident_limit=6,
                 adapter_factory=None, table_staleness=lambda _record, _page: False):
        super().__init__(parent)
        if resident_limit < 1:
            raise ValueError("A figure workspace needs at least one resident slot.")
        self._factory, self._mount, self._unmount = factory, mount, unmount
        self._initialize, self._bind_sources = initialize, bind_sources
        self._current_binding, self._dirty = current_binding, dirty
        self._commit_focus = commit_focus
        self._adapter_factory = adapter_factory
        self._table_staleness = table_staleness
        self.resident_limit = resident_limit
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._unsupported: OrderedDict[str, tuple[object, str]] = OrderedDict()
        self._deleted: OrderedDict[str, object] = OrderedDict()
        self._current = None
        self._selected = ()
        self._clock = 0
        self._epoch = 0
        self._installing = 0
        self._retiring = []
        self._retire_timer = QtCore.QTimer(self)
        self._retire_timer.setInterval(200)
        self._retire_timer.timeout.connect(self._finish_retiring)
        self._workspace = AnalyticalWorkspace()
        self._read_only = False
        self._default_look = _BUILTIN_PRESETS[DEFAULT_LOOK].style_only().to_dict()
        self._styles = registry_from_workspace(self._workspace).registry
        self._style_warnings = ()

    def style_registry(self):
        return self._styles

    def available_styles(self):
        return self._styles.available() if self._styles is not None else ()

    def default_style(self):
        if self._styles is None:
            raise ValueError("The project's styles could not be read; its original styles are retained.")
        return resolve_ref(self._styles, self._styles.default_ref)

    def _replace_styles(self, registry):
        if self._styles is None or self._read_only:
            raise ValueError("This project's styles are read-only.")
        self._styles = registry
        self.stylesChanged.emit()
        self._dirty()

    def set_default_style(self, ref):
        if self._styles is None:
            raise ValueError("The project's styles are unreadable and cannot be replaced.")
        self._replace_styles(self._styles.with_default(ref))

    def delete_style(self, style_id):
        if self._styles is None:
            raise ValueError("The project's styles are unreadable and cannot be replaced.")
        self._replace_styles(self._styles.hide(style_id))

    def import_legacy_styles(self, states):
        """Copy old looks once into this project; never write the computer's presets."""
        if self._styles is None or self._read_only:
            return
        registry = self._styles
        warnings = []
        for name, mapping in states.items():
            try:
                registry, _ref = registry.import_legacy_style(name, mapping, "plot_appearance")
            except ValueError as exc:
                warnings.append(f"{name}: {exc}")
        self._style_warnings = tuple(dict.fromkeys((*self._style_warnings, *warnings)))
        if registry != self._styles:
            self._replace_styles(registry)

    def capture_style(self, figure_id=None):
        """Return a detached current look without changing either style catalog."""
        provenance = {"project_style": True}
        captured_columns = None
        key = figure_id or self._current
        if key is None or not self.checkpoint(key):
            raise ValueError("Choose a figure whose look can be preserved.")
        record = self._entries[key].record
        from ..core.figure_codecs import encode_legacy, _native_style_path
        from ..core.figure_style_mapping import import_legacy_look
        native = encode_legacy(record.spec, record.resources, materialize_defaults=True)
        path, vocabulary = _native_style_path(record.spec.kind)
        original = native.payload
        for part in path:
            original = original.get(part, {})
        binding = original.get("style_binding")
        if binding:
            style = FigureStyle.from_dict(binding["snapshot"])
            captured_columns = style.columns
            look = resolve_look(style, binding.get("overrides", {})).values
            if inherits_native_defaults(style):
                provenance["inherit_native_defaults"] = True
        else:
            provenance["inherit_native_defaults"] = True
            conversion = import_legacy_look(original, vocabulary)
            if conversion.report["unreadable"]:
                raise ValueError("Some look values cannot be saved losslessly.")
            look = conversion.look
        # A saved look carries reusable text styling, never the figure's words
        # or source-bound legend entries. Explicit title/axis styles become roles.
        look = thaw_json(look)
        roles = dict(look.get("text_roles", {}))
        elements = original.get("text_styles", original.get("text", {}))
        for target, value in elements.items():
            if target == "title" or target.startswith("axis."):
                role = {key: item for key, item in value.items() if key != "text"}
                if role:
                    roles[target] = {**roles.get(target, {}), **role}
        fit = original.get("fit_label", {})
        fit_role = {key: fit[key] for key in ("size_pt", "color") if fit.get(key) is not None}
        if fit_role:
            roles["fit_label"] = {**roles.get("fit_label", {}), **fit_role}
        if roles:
            look["text_roles"] = roles
        return FigureStyle(style_id="captured." + str(uuid4()), revision=1, name="Current figure look",
                           look=look, columns=(captured_columns if captured_columns is not None else self.default_style().columns),
                           provenance=provenance)

    def save_style(self, name, *, figure_id=None, overwrite=False, look=None, columns=None):
        """Save a detached current look or edited definition as an immutable revision."""
        if self._styles is None:
            raise ValueError("The project's styles are unreadable and cannot be replaced.")
        name = str(name).strip()
        if not name or len(name) > 100:
            raise ValueError("A style name must contain 1–100 characters.")
        existing = next((style for style in self.available_styles() if style.name == name), None)
        if existing is not None and not overwrite:
            raise ValueError("A style with this name already exists.")
        if existing is not None and existing.provenance.get("builtin"):
            raise ValueError("Save a built-in look under a new project style name.")
        provenance = {"project_style": True}
        captured_columns = None
        if look is None:
            captured = self.capture_style(figure_id)
            look = captured.look
            captured_columns = captured.columns
            provenance = thaw_json(captured.provenance)
        if columns is None:
            columns = (existing.columns if existing is not None else
                       captured_columns if captured_columns is not None else self.default_style().columns)
        style = (replace(existing, revision=existing.revision + 1, look=look, columns=columns) if existing else
                 FigureStyle(style_id="project." + str(uuid4()), revision=1, name=name, look=look,
                             columns=columns, provenance=provenance))
        self._replace_styles(self._styles.add_revision(style))
        return style

    def records(self):
        return tuple(entry.record for entry in self._entries.values())

    def get(self, figure_id):
        entry = self._entries.get(figure_id)
        return entry.record if entry is not None else None

    def current_id(self): return self._current

    def clear_current(self):
        """Restore an empty navigation position without deleting or editing a figure."""
        if self._current is not None:
            self._current = None
            self.currentChanged.emit("")

    def restore_navigation(self, figure_id):
        """Restore an existing selection identity without materializing its canvas."""
        if figure_id is None:
            self.clear_current()
            return True
        if figure_id not in self._entries:
            return False
        if self._current != figure_id:
            self._current = figure_id
            self.currentChanged.emit(figure_id)
        return True

    def export_ready(self, figure_id):
        """Whether committed artwork can be captured without interrupting its work."""
        entry = self._entries.get(figure_id)
        if entry is None:
            return False, "This figure is unavailable or unsupported by this build."
        if entry.busy or entry.capturing:
            return False, "This figure is busy; export it after its current work finishes."
        if entry.page is None:
            return True, ""
        try:
            return entry.adapter.can_suspend()
        except (RuntimeError, ValueError) as exc:
            return False, f"This figure cannot be preserved for export: {exc}"

    def is_read_only(self): return self._read_only
    def selected_ids(self): return self._selected
    def resident_ids(self): return tuple(key for key, entry in self._entries.items() if entry.page is not None)
    def page(self, figure_id):
        entry = self._entries.get(figure_id)
        return entry.page if entry is not None else None

    def id_for_page(self, page):
        return next((key for key, entry in self._entries.items() if entry.page is page), None)

    def most_recent(self, kind):
        choices = [(entry.used, index, key) for index, (key, entry) in enumerate(self._entries.items())
                   if entry.record.spec.kind == kind]
        return max(choices)[2] if choices else None

    def unsupported_records(self):
        return tuple({"id": key, "name": str((artifact.payload.get("name") if isinstance(artifact.payload, Mapping) else None) or
                     ANALYSIS_LABELS.get(NATIVE_KINDS.get(key, (None, ""))[1], "Unsupported figure")),
                      "kind": artifact.kind, "reason": reason}
                     for key, (artifact, reason) in self._unsupported.items())

    def set_selected(self, ids):
        selected = tuple(dict.fromkeys(key for key in ids if key in self._entries))
        if selected != self._selected:
            self._selected = selected
            self.selectionChanged.emit()

    def _fail(self, key, error):
        self.requestFailed.emit(str(key or ""), str(error) or type(error).__name__)
        return False

    def _metadata(self, entry):
        return thaw_json(entry.record.spec.extra.get(META_KEY, {}))

    def binding(self, figure_id):
        entry = self._entries[figure_id]
        meta = self._metadata(entry)
        if "binding" in meta:
            return meta["binding"]
        # A migrated Event/Noise figure retains v12's explicit follow-selection behavior.
        return {"follow": entry.record.spec.kind in ("event_waveforms", "noise")}

    def set_follow_selection(self, figure_id, follow):
        entry = self._entries.get(figure_id)
        if entry is None:
            return False
        binding = self.binding(figure_id)
        staged = entry.record.spec.kind in ('noise', 'event_waveforms')
        if not follow:
            if not self.checkpoint(figure_id):
                return False
            from ..core.analysis_workspace import artifact_source_recipe
            from ..core.figure_codecs import encode_legacy
            source = (binding.get('source') if staged else
                      artifact_source_recipe(encode_legacy(entry.record.spec, entry.record.resources)))
            binding = {"source": source}
        elif staged:
            binding = {**binding, 'source': self._current_binding(entry.record.spec.kind).get('source')}
        binding = {**binding, "follow": bool(follow)}
        result = self.set_binding(figure_id, binding)
        if result and staged and entry.page is not None:
            self._bind_sources(entry.page, entry.record, binding, True)
            self.checkpoint(figure_id, changed=True)
        return result

    def set_binding(self, figure_id, binding):
        entry = self._entries.get(figure_id)
        if entry is None:
            return False
        self._set_metadata(entry, binding=binding)
        self._changed(figure_id)
        return True

    def _set_metadata(self, entry, **values):
        spec = entry.record.spec
        meta = {**self._metadata(entry), **values}
        entry.record = replace(entry.record, spec=replace(spec, extra={**thaw_json(spec.extra), META_KEY: meta}))

    def _refresh_pending_status(self, entry):
        """Keep list, header and dormant records consistent with analytical drafts.

        The saved base distinguishes a pending edit from source staleness, so
        undoing an edit cannot accidentally certify changed source data.
        """
        if entry.page is None:
            return
        pending = getattr(entry.page.inspector_adapter(), "pending", None)
        if not callable(pending):
            return
        state = pending()
        status = entry.record.spec.result.status
        meta = self._metadata(entry)
        base = meta.get("pending_result_base")
        if state.count and status not in ("empty", "missing"):
            if base is None:
                self._set_metadata(entry, pending_result_base=status)
            status = "stale"
        elif not state.count and base is not None:
            status = base
            self._set_metadata(entry, pending_result_base=None)
        if status != entry.record.spec.result.status:
            entry.record = replace(entry.record, spec=replace(entry.record.spec,
                result=replace(entry.record.spec.result, status=status)))

    def mark_current(self, figure_id):
        """An explicit successful computation committed this result."""
        entry = self._entries.get(figure_id)
        if entry is None:
            return
        self._set_metadata(entry, pending_result_base=None, table_staleness_base=None)
        entry.record = replace(entry.record, spec=replace(entry.record.spec,
            result=replace(entry.record.spec.result, status="current")))
        self._refresh_table_staleness(entry)

    def _refresh_table_staleness(self, entry):
        """Keep a reversible form judgement separate from source and draft changes."""
        status = entry.record.spec.result.status
        if status in ("empty", "missing", "frozen"):
            return False
        meta = self._metadata(entry)
        base = meta.get("table_staleness_base")
        stale = bool(self._table_staleness(entry.record, entry.page))
        if stale:
            if base is None:
                self._set_metadata(entry, table_staleness_base=meta.get("pending_result_base") or status)
            if meta.get("pending_result_base") is not None:
                self._set_metadata(entry, pending_result_base="stale")
            status = "stale"
        elif base is not None:
            self._set_metadata(entry, table_staleness_base=None)
            if meta.get("pending_result_base") is not None:
                self._set_metadata(entry, pending_result_base=base)
            else:
                status = base
        if status == entry.record.spec.result.status:
            return False
        entry.record = replace(entry.record, spec=replace(entry.record.spec,
            result=replace(entry.record.spec.result, status=status)))
        return True

    def refresh_table_staleness(self):
        """Refresh warning badges for resident and dormant figures without fitting."""
        for key, entry in tuple(self._entries.items()):
            if self._refresh_table_staleness(entry):
                self.recordChanged.emit(key)
            schedule = getattr(entry.page, "schedule_stale_redraw", None)
            if callable(schedule):
                schedule()

    def _changed(self, key):
        self._entries[key].generation += 1
        self.recordChanged.emit(key)
        if not self._installing:
            self._dirty()

    def new_figure_look(self, kind, style=None, *, body_size_pt=None):
        """The appearance a NEW figure of ``kind`` starts with in this project: bound to the
        project's current default figure style (or to ``style``), with that style's look.

        What :meth:`create` gives a new figure, and what Reset figure gives a figure that has no
        style (a page's ``PlotAppearanceWidget.set_new_figure_look``), so the two cannot differ.
        ``body_size_pt`` is the Font size (body) a Distributions figure already has, which Reset
        figure leaves as it is: unless the style gives that size anyway, it rides on the binding
        as the figure's own override, as it does on a figure that was made with the style.
        Raises ``ValueError`` when the project's styles cannot be read."""
        from .figure_style_binding import bind_native_style
        default_style = self.default_style() if style is None else style
        vocabulary = "matplotlib_appearance" if kind == "comparison_figures" else "plot_appearance"
        native_look = bind_native_style({}, default_style, vocabulary)
        if kind == "distributions" and inherits_native_defaults(default_style):
            # The old project default used axis/title as the body's fallback. This is
            # native figure state, not an addition to the exact imported definition.
            if "body_size_pt" not in default_style.look:
                native_look["style_binding"]["overrides"]["body_size_pt"] = float(
                    default_style.look.get("axis_size_pt", default_style.look.get("title_size_pt", 7)))
        if kind == "distributions" and body_size_pt is not None:
            overrides = native_look["style_binding"]["overrides"]
            if resolve_look(default_style, overrides).values.get("body_size_pt") != float(body_size_pt):
                overrides["body_size_pt"] = float(body_size_pt)
        return PlotAppearance.from_mapping(native_look)

    def _adopt_new_figure_look(self, figure_id, page, kind):
        """Give a page of this workspace its Reset figure's look: what a new figure of its kind is
        given here, read when the button is pressed (the project's default may have changed since),
        with the Font size (body) a Distributions figure keeps through every Reset figure."""
        adopt = getattr(getattr(page, "_appearance_widget", None), "set_new_figure_look", None)
        if not callable(adopt):
            return

        def look():
            # The page is looked up when asked, not held: this function lives on the page's own
            # widget, and holding the page would make a reference cycle that outlasts its sleep.
            shown = self.page(figure_id)
            body = shown.style().font_size_pt if kind == "distributions" and shown is not None else None
            try:
                return self.new_figure_look(kind, body_size_pt=body)
            except ValueError:
                return None     # the project's styles cannot be read: no look to go back to

        adopt(look)

    def create(self, kind, initial_recipe=None, binding=None):
        if self._read_only:
            self._fail("", "This analytical workspace was written by a newer build and is read-only.")
            return None
        if kind not in PAGE_IDS:
            self._fail("", "Unknown figure kind.")
            return None
        self._commit_focus()
        key = str(uuid4())
        native_id, (native_kind, _kind, label) = next(
            (key, value) for key, value in NATIVE_KINDS.items() if value[1] == kind)
        names = {entry.record.spec.name for entry in self._entries.values()}
        number = 1
        while f"{label} {number}" in names:
            number += 1
        try:
            default_style = self.default_style()
        except ValueError as exc:
            self._fail("", exc)
            return None
        look = self.new_figure_look(kind, default_style)
        payload = {}
        if kind == "distributions":
            from ..core.figure_codecs import V12_PLOT_DEFAULTS
            payload = {"config": thaw_json(V12_PLOT_DEFAULTS)}
            # The legacy body size is separate from title/axis/tick overrides. New
            # figures start it at the project look; existing figures keep their value.
            payload["config"]["fields"]["style"]["fields"]["font_size_pt"] = float(
                default_style.look.get("body_size_pt", default_style.look.get("axis_size_pt",
                    default_style.look.get("title_size_pt", 7))))
        record = decode_legacy(WorkspaceArtifact(native_id, native_kind, payload=payload),
                               figure_id=key, name=f"{label} {number}",
                               created_at=datetime.now(timezone.utc).isoformat()).record
        from .figure_adapter_pages import apply_look_to_record, initial_history
        # Canonical state must already contain the project look while no page exists.
        # This also gives a deferred New figure the correct base for Style and Undo.
        record, _unused_history = apply_look_to_record(record, look, initial_history(record), keep_hand=False)
        if initial_recipe:
            from .figure_recipes import seed_appearance
            try:
                record = seed_appearance(record, initial_recipe)
            except (TypeError, ValueError) as exc:
                self._fail("", "The recipe appearance could not be installed: " + str(exc))
                return None
        entry = _Entry(record)
        self._entries[key] = entry
        chosen_binding = self._current_binding(kind) if binding is None else dict(binding)
        chosen_binding.setdefault("follow", False)
        from .figure_recipes import pending_recipe
        self._set_metadata(entry, uninitialized=True, initial_recipe=pending_recipe(initial_recipe), binding=chosen_binding)
        entry.record = replace(entry.record, spec=replace(entry.record.spec,
            result=replace(entry.record.spec.result, status="empty")))
        self.recordsChanged.emit()
        self._dirty()
        self.activate(key)
        return key

    def checkpoint(self, figure_id, *, changed=False):
        entry = self._entries.get(figure_id)
        if entry is None:
            return False
        if entry.page is None or self._installing or entry.capturing:
            return True
        entry.capturing = True
        try:
            self._refresh_pending_status(entry)
            self._refresh_table_staleness(entry)
            previous = entry.record
            record = entry.adapter.capture_record(previous)
            # Native codecs own result data; freshness is a workspace source-policy state.
            record = replace(record, spec=replace(record.spec,
                result=replace(record.spec.result, status=previous.spec.result.status)))
            history = entry.adapter.capture_transient()
        except Exception as exc:
            self._fail(figure_id, f"Could not preserve this figure: {exc}")
            return False
        finally:
            entry.capturing = False
        entry.record, entry.history = record, history
        if changed or record.spec != previous.spec:
            self._changed(figure_id)
        return True

    def _watch(self, key, entry):
        epoch, identity = self._epoch, entry.identity
        def valid():
            current = self._entries.get(key)
            return (self._epoch == epoch and current is entry and current.identity == identity
                    and not self._installing)
        def edited(*_args):
            if valid():
                # A native edit is still inside its typing/spin burst. Capturing here
                # would close that burst after every character. Transaction boundaries
                # capture the working page before save, switch, duplicate or suspension.
                self._refresh_pending_status(entry)
                self._changed(key)
        def checkpoint_requested(*_args):
            if valid():
                self.checkpoint(key)
        for name, callback in (("workspaceChanged", edited), ("pendingChanged", edited),
                               ("workspaceCheckpointRequested", checkpoint_requested)):
            signal = getattr(entry.page, name, None)
            if signal is not None:
                signal.connect(callback)
                entry.links.append((signal, callback))

    @staticmethod
    def _unlink(entry):
        for signal, slot in entry.links:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        entry.links.clear()

    def _room(self, exclude):
        if len(self.resident_ids()) + len(self._retiring) < self.resident_limit:
            return True
        candidates = sorted(((entry.used, key) for key, entry in self._entries.items()
                             if entry.page is not None and key not in (self._current, exclude)))
        for _used, key in candidates:
            if self.suspend(key):
                return True
        return False

    def activate(self, figure_id):
        entry = self._entries.get(figure_id)
        if entry is None:
            return self._fail(figure_id, "This figure cannot be opened by this build.")
        self._commit_focus()
        if self._current and self._current != figure_id and not self.checkpoint(self._current):
            return False
        if entry.page is None:
            if not self._room(figure_id):
                return self._fail(figure_id, "All available figure slots are busy or have edits that cannot be preserved. The figure remains saved; try opening it after the work finishes.")
            page = None
            saved_record, saved_history = entry.record, entry.history
            self._installing += 1
            try:
                page = self._factory(entry.record.spec.kind, figure_id)
                if self._adapter_factory is None:
                    from .figure_adapter_pages import adapter_for
                    adapter = adapter_for(page, entry.record.spec.kind)
                else:
                    adapter = self._adapter_factory(page, entry.record.spec.kind)
                self._adopt_new_figure_look(figure_id, page, entry.record.spec.kind)
                meta = self._metadata(entry)
                if meta.get("uninitialized"):
                    self._initialize(page, entry.record, meta.get("initial_recipe", {}), self.binding(figure_id))
                    old_spec = entry.record.spec
                    decoded = decode_legacy(adapter.capture_native(), figure_id=figure_id,
                        name=old_spec.name, created_at=old_spec.created_at).record
                    decoded = replace(decoded, spec=replace(decoded.spec,
                        presentation=replace(decoded.spec.presentation, style_snapshot=old_spec.presentation.style_snapshot),
                        lineage=old_spec.lineage,
                        extra={**thaw_json(decoded.spec.extra), META_KEY: {**meta, "uninitialized": False}},
                        result=replace(decoded.spec.result, status=old_spec.result.status)))
                    captured = adapter.capture_record(decoded)
                    entry.record = replace(captured, spec=replace(captured.spec,
                        result=replace(captured.spec.result, status=old_spec.result.status)))
                    from .figure_adapter_pages import initial_history
                    entry.history = initial_history(entry.record)
                    adapter.install_transient(entry.history)
                else:
                    self._bind_sources(page, entry.record, self.binding(figure_id), False)
                    report = adapter.restore_record(entry.record, mode="resume" if entry.history is not None else "project")
                    if getattr(report, "skipped", ()):
                        raise ValueError("The saved figure could not be restored: " + "; ".join(getattr(report, "warnings", ()) or ("unsupported saved content",)))
                    if entry.history is not None:
                        adapter.install_transient(entry.history)
                    if (entry.record.spec.kind in ('noise', 'event_waveforms')
                            and self.binding(figure_id).get('follow')):
                        # A dormant follower may have acquired a new source while
                        # its applied native snapshot stayed safely asleep.
                        self._bind_sources(page, entry.record, self.binding(figure_id), True)
                entry.page, entry.adapter = page, adapter
                self._mount(figure_id, page)
                configure = getattr(page, "configure_figure_binding", None)
                if callable(configure):
                    configure(bool(self.binding(figure_id).get("follow")),
                              lambda follow, key=figure_id: self.set_follow_selection(key, follow))
                self._watch(figure_id, entry)
            except Exception as exc:
                self._unlink(entry)
                entry.record, entry.history = saved_record, saved_history
                if page is not None:
                    if entry.page is page:
                        self._unmount(figure_id, page)
                    page.deleteLater()
                entry.page = entry.adapter = None
                return self._fail(figure_id, exc)
            finally:
                self._installing -= 1
        self._clock += 1
        entry.used = self._clock
        self._current = figure_id
        if figure_id not in self._selected:
            self.set_selected((figure_id,))
        self.currentChanged.emit(figure_id)
        return True

    def suspend(self, figure_id):
        entry = self._entries.get(figure_id)
        if entry is None or entry.page is None:
            return True
        if entry.busy:
            return False
        ready, _reason = entry.adapter.can_suspend()
        if not ready or not self.checkpoint(figure_id):
            return False
        try:
            entry.adapter.suspend()
        except Exception as exc:
            return self._fail(figure_id, exc)
        self._unlink(entry)
        page = entry.page
        self._unmount(figure_id, page)
        entry.page = entry.adapter = None
        page.deleteLater()
        return True

    def duplicate(self, figure_id):
        self._commit_focus()
        if not self.checkpoint(figure_id):
            return None
        source = self._entries[figure_id].record
        key = str(uuid4())
        spec = replace(source.spec, figure_id=key, name=source.spec.name[:193] + " (copy)",
                       created_at=datetime.now(timezone.utc).isoformat(),
                       lineage={**thaw_json(source.spec.lineage), "duplicate_of": figure_id})
        self._entries[key] = _Entry(replace(source, spec=spec))
        self.recordsChanged.emit()
        self._dirty()
        self.activate(key)
        return key

    def rename(self, figure_id, name):
        entry = self._entries.get(figure_id)
        if entry is None or not str(name).strip() or len(str(name).strip()) > 200:
            return self._fail(figure_id, "Figure names must contain 1–200 characters.")
        entry.record = replace(entry.record, spec=entry.record.spec.renamed(str(name).strip()))
        self._changed(figure_id)
        return True

    def delete(self, figure_id):
        if self._read_only:
            return self._fail(figure_id, "This newer analytical workspace is read-only; its original content is preserved.")
        self._commit_focus()
        if figure_id in self._unsupported:
            self._deleted[figure_id] = self._unsupported.pop(figure_id)
        else:
            entry = self._entries.get(figure_id)
            if entry is None:
                return False
            if entry.busy or not self.suspend(figure_id):
                return self._fail(figure_id, "This figure is busy or could not be preserved; it was not deleted.")
            self._deleted[figure_id] = self._entries.pop(figure_id)
        if self._current == figure_id:
            self._current = None
            self.currentChanged.emit("")
        self.set_selected(key for key in self._selected if key != figure_id)
        self.recordsChanged.emit()
        self._dirty()
        if self._current is None and self._entries:
            self.activate(max(self._entries, key=lambda key: self._entries[key].used))
        return True

    def restore_deleted(self, figure_id=None):
        key = figure_id or next(reversed(self._deleted), None)
        if key is None or key not in self._deleted:
            return None
        entry = self._deleted.pop(key)
        if isinstance(entry, _Entry):
            self._entries[key] = entry
        else:
            self._unsupported[key] = entry
        self.recordsChanged.emit()
        self._dirty()
        if isinstance(entry, _Entry):
            self.activate(key)
        return key

    def revision_token(self, figure_id):
        entry = self._entries.get(figure_id)
        return (self._epoch, entry.identity, entry.generation) if entry else None

    def _history(self, entry):
        if entry.history is None:
            from .figure_adapter_pages import initial_history
            entry.history = initial_history(entry.record)
        return entry.history

    def history(self, figure_id):
        entry = self._entries.get(figure_id)
        if entry is None:
            return HistoryState()
        if entry.page is not None:
            return entry.page.inspector_adapter().history()
        from .figure_adapter_pages import history_state
        return history_state(self._history(entry))

    def apply_look_to_figure(self, figure_id, look, keep_hand=True):
        entry = self._entries[figure_id]
        if entry.busy:
            raise RuntimeError("This figure is busy.")
        if entry.page is not None:
            moved = entry.page.inspector_adapter().apply_look(look, keep_hand)
            if moved and not self.checkpoint(figure_id):
                raise RuntimeError("The styled figure could not be preserved.")
            return moved
        from .figure_adapter_pages import apply_look_to_record
        record, history = apply_look_to_record(entry.record, look, self._history(entry), keep_hand=keep_hand)
        if history == entry.history:
            return False
        record = replace(record, spec=replace(record.spec,
            result=replace(record.spec.result, status=entry.record.spec.result.status)))
        entry.record, entry.history = record, history
        self._changed(figure_id)
        return True

    def undo(self, figure_id): return self._move_history(figure_id, "undo")
    def redo(self, figure_id): return self._move_history(figure_id, "redo")

    def _move_history(self, figure_id, direction):
        entry = self._entries[figure_id]
        if entry.page is not None:
            moved = getattr(entry.page.inspector_adapter(), direction)()
            if moved:
                self.checkpoint(figure_id)
            return moved
        from .figure_adapter_pages import NeedsResidentHistory, move_history
        try:
            record, history = move_history(entry.record, self._history(entry), direction)
        except NeedsResidentHistory:
            return self.activate(figure_id) and self._move_history(figure_id, direction)
        if history == entry.history:
            return False
        record = replace(record, spec=replace(record.spec,
            result=replace(record.spec.result, status=entry.record.spec.result.status)))
        entry.record, entry.history = record, history
        self._changed(figure_id)
        return True

    def request_token(self, figure_id):
        entry = self._entries[figure_id]
        entry.request_generation += 1
        entry.busy = True
        return self._epoch, figure_id, entry.identity, entry.request_generation

    def accepts_result(self, token):
        if not isinstance(token, tuple) or len(token) != 4:
            return False
        epoch, key, identity, generation = token
        entry = self._entries.get(key)
        return bool(epoch == self._epoch and entry and entry.busy and entry.identity == identity
                    and entry.request_generation == generation)

    def finish_request(self, token):
        if self.accepts_result(token):
            self._entries[token[1]].busy = False
            return True
        return False

    def sources_changed(self, *, selection_only=False):
        for key, entry in tuple(self._entries.items()):
            binding = self.binding(key)
            follows = bool(binding.get("follow"))
            if selection_only and not follows:
                continue
            if self._metadata(entry).get("table_staleness_base") is not None:
                self._set_metadata(entry, table_staleness_base="stale")
            if follows and entry.record.spec.kind in ('noise', 'event_waveforms'):
                selected = self._current_binding(entry.record.spec.kind)
                binding = {**binding, 'source': selected.get('source')}
                self._set_metadata(entry, binding=binding)
                if entry.page is not None:
                    self._bind_sources(entry.page, entry.record, binding, True)
                    self._refresh_pending_status(entry)
                if (entry.record.spec.result.status != 'empty'
                        and (entry.page is None or not selection_only)):
                    if self._metadata(entry).get('pending_result_base') is not None:
                        self._set_metadata(entry, pending_result_base='stale')
                    entry.record = replace(entry.record, spec=replace(entry.record.spec,
                        result=replace(entry.record.spec.result, status='stale')))
                if entry.page is not None:
                    self.checkpoint(key, changed=True)
                else:
                    self._changed(key)
                continue
            if entry.record.spec.result.status == "frozen":
                continue
            if (entry.page is not None and key == self._current and follows
                    and entry.page.isVisible() and not entry.busy):
                if self._bind_sources(entry.page, entry.record, binding, True) is not False:
                    self.mark_current(key)
                    self.checkpoint(key, changed=True)
            else:
                if entry.record.spec.result.status == "empty":
                    self.recordChanged.emit(key)
                    continue  # There is no committed result to become stale.
                if self._metadata(entry).get("pending_result_base") is not None:
                    self._set_metadata(entry, pending_result_base="stale")
                entry.record = replace(entry.record, spec=replace(entry.record.spec,
                    result=replace(entry.record.spec.result, status="stale")))
                self.recordChanged.emit(key)

    def install_workspace(self, workspace):
        migration = migrate_workspace_v12(workspace)
        records = list(migration.records)
        metadata = workspace.extra.get(WORKSPACE_KEY, {})
        order = metadata.get("order", ()) if isinstance(metadata, dict) else ()
        by_id = {record.spec.figure_id: record for record in records}
        ordered = [by_id.pop(key) for key in order if key in by_id]
        ordered.extend(by_id.values())
        self.clear()
        self._workspace = migration.workspace
        self._read_only = type(workspace.version) is not int or workspace.version != WORKSPACE_VERSION
        self._entries = OrderedDict()
        for record in ordered:
            status = record.spec.result.status
            # A never-generated figure has no retained result to freeze. Pending
            # and missing states are also meaningful after reopening a project.
            if status in ("current", "frozen"):
                status = "frozen"
            entry = _Entry(replace(record, spec=replace(record.spec,
                result=replace(record.spec.result, status=status))))
            self._entries[record.spec.figure_id] = entry
            if self._metadata(entry).get("pending_result_base") == "current":
                self._set_metadata(entry, pending_result_base="frozen")
        failures = {failure.artifact_id: failure.reason for failure in migration.failures}
        for artifact in migration.workspace.artifacts:
            if self._read_only:
                self._unsupported[artifact.artifact_id] = (artifact, "This analytical workspace uses a newer version; its original content is preserved read-only.")
            elif artifact.artifact_id in failures or (artifact.kind == FIGURE_ARTIFACT_KIND and
                    artifact.artifact_id not in {record.spec.artifact_id for record in ordered}):
                self._unsupported[artifact.artifact_id] = (artifact, failures.get(artifact.artifact_id,
                    "This figure uses a newer or unsupported schema; its original data are preserved."))
            elif artifact.artifact_id in NATIVE_KINDS:
                self._unsupported[artifact.artifact_id] = (artifact, "This analytical figure could not be migrated; its original data are preserved.")
        saved_default = metadata.get("default_style") if isinstance(metadata, dict) else None
        self._default_look = dict(saved_default) if isinstance(saved_default, dict) else _BUILTIN_PRESETS[DEFAULT_LOOK].style_only().to_dict()
        styles = registry_from_workspace(workspace)
        self._styles, self._style_warnings = styles.registry, styles.warnings
        self.stylesChanged.emit()
        self.recordsChanged.emit()
        return migration

    def capture_workspace(self, base=None):
        if self._read_only:
            return self._workspace
        self._commit_focus()
        for key in tuple(self._entries):
            if not self.checkpoint(key):
                raise RuntimeError("A figure could not be checkpointed; the previous project was not replaced.")
        workspace = base if base is not None else self._workspace
        replacements = {entry.record.spec.artifact_id: record_to_artifact(entry.record)
                        for entry in self._entries.values()}
        artifacts = []
        for artifact in workspace.artifacts:
            if artifact.artifact_id in replacements:
                artifacts.append(replacements.pop(artifact.artifact_id))
            elif (artifact.kind == FIGURE_ARTIFACT_KIND or artifact.artifact_id in NATIVE_KINDS):
                if artifact.artifact_id in self._unsupported:
                    artifacts.append(self._unsupported[artifact.artifact_id][0])
            else:
                artifacts.append(artifact)
        known = {artifact.artifact_id for artifact in artifacts}
        artifacts.extend(record_to_artifact(entry.record) for entry in self._entries.values()
                         if entry.record.spec.artifact_id not in known)
        artifacts.extend(artifact for key, (artifact, _reason) in self._unsupported.items() if key not in known)
        old = workspace.extra.get(WORKSPACE_KEY, {})
        legacy_default = self._default_look
        if self._styles is not None:
            from ..core.figure_style_mapping import export_legacy_look
            legacy_default = export_legacy_look(resolve_look(self.default_style()).values,
                                                 "plot_appearance", {})
        metadata = {**(old if isinstance(old, dict) else {}), "order": list(self._entries)}
        if self._styles is not None:
            metadata["default_style"] = legacy_default
        # An unreadable old default or future style registry is carried verbatim;
        # saving unrelated figures must not silently replace it with a built-in.
        extra = {**workspace.extra, WORKSPACE_KEY: metadata}
        captured = AnalyticalWorkspace(workspace.version, artifacts, extra)
        return registry_to_workspace(captured, self._styles) if self._styles is not None else captured

    def clear(self):
        self._epoch += 1
        for key, entry in tuple(self._entries.items()):
            if entry.page is None:
                continue
            self._unlink(entry)
            self._unmount(key, entry.page)
            entry.page.hide()
            self._retiring.append(entry)
        self._entries.clear()
        self._unsupported.clear()
        self._deleted.clear()
        self._workspace = AnalyticalWorkspace()
        self._read_only = False
        self._default_look = _BUILTIN_PRESETS[DEFAULT_LOOK].style_only().to_dict()
        self._styles = registry_from_workspace({}).registry
        self._style_warnings = ()
        self.stylesChanged.emit()
        self._current, self._selected = None, ()
        self._finish_retiring()
        self.currentChanged.emit("")
        self.selectionChanged.emit()
        self.recordsChanged.emit()

    def _finish_retiring(self):
        remaining = []
        for entry in self._retiring:
            try:
                ready, _reason = entry.adapter.can_suspend()
                if not ready:
                    remaining.append(entry)
                    continue
                entry.adapter.suspend()
                entry.page.deleteLater()
            except RuntimeError:
                # A close accepted by the native page may already have deleted its Qt owner.
                import shiboken6
                if shiboken6.isValid(entry.page):
                    remaining.append(entry)
        self._retiring = remaining
        if remaining:
            self._retire_timer.start()
        else:
            self._retire_timer.stop()
