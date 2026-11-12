"""The figure half of ``MainWindow``: the Analysis Explorer, the figure surfaces and their
sources (the noise and subpopulation figures among them), recipes and saved figures, the
Figure Library, and figure capture and export. ``_on_umap_finished`` stays in ``main_window.py``: it
is a registered ``@Slot``, and no ``@Slot`` method leaves the class body (see the comment after the
imports there).

``_FiguresMixin`` is a plain class listed before ``QMainWindow`` in the bases of ``MainWindow``,
with no ``__init__`` and no state of its own. The method bodies were moved verbatim out of
``main_window.py`` by a mechanical split; nothing in them was edited.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
from collections.abc import Mapping
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..core.analysis_workspace import (
    DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
    COMPARISON_FIGURES_ARTIFACT_ID,
    EVENT_PLOTS_ARTIFACT_ID,
    IV_ARTIFACT_ID,
    PLOTS_ARTIFACT_ID,
    PSD_ARTIFACT_ID,
    SUBPOPULATION_ARTIFACT_ID,
    VOLTAGE_DEPENDENCE_ARTIFACT_ID,
    capture_analysis_workspace,
)
from ..core.types import ChannelKind
from .analysis_explorer import (
    AnalysisExplorer,
    PAGE_DISTRIBUTIONS,
    PAGE_DISTRIBUTION_SUMMARIES,
    PAGE_COMPARISON_FIGURES,
    PAGE_EVENT_WAVEFORMS,
    PAGE_IV,
    PAGE_NOISE,
    PAGE_SUBPOPULATIONS,
    PAGE_VOLTAGE_SERIES,
)
from .analysis_catalog import PAGE_HOME
from .trace_view import effective_kind


class _FiguresMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    def _ensure_analysis_explorer(self) -> AnalysisExplorer:
        """Build the single analytical shell without constructing any analysis page.

        Pages remain lazy because several of their constructors perform real numerical
        work.  The Explorer is presentation only; every existing page attribute and
        figure record remains the source of truth.
        """
        if self._analysis_explorer is None:
            self._analysis_explorer = AnalysisExplorer(
                self,
                source_session=self._session, source_provider=self._plots_sources,
                context_provider=self._analysis_explorer_context,
                status_provider=self._analysis_explorer_status,
                availability_provider=self._analysis_home_availability,
            )
            self._analysis_explorer.visibilityChanged.connect(
                lambda _visible: self._sync_analysis_open_state())
            self._analysis_explorer.saveFigureRequested.connect(self._save_analysis_figure)
            self._analysis_explorer.saveBundleRequested.connect(self._export_analysis_bundle)
            self._analysis_explorer.figureLibraryRequested.connect(self._open_figure_library)
            self._analysis_explorer.launchRequested.connect(self._launch_analysis_route)
            self._analysis_explorer.snapshotRequested.connect(self._save_analysis_figure)
            self._analysis_explorer.savedFigureActivated.connect(self._show_saved_figure)
            self._analysis_explorer.savedDeleteRequested.connect(self._delete_saved_figures)
            self._analysis_explorer.compositionRequested.connect(self._compose_saved_figures)
            self._analysis_explorer.recipeRequested.connect(self._new_figure_from_recipe)
            self._analysis_explorer.saveRecipeRequested.connect(self._save_figure_recipe)
            self._analysis_explorer.manageRecipesRequested.connect(self._manage_figure_recipes)
            self._analysis_explorer.exportSelectedRequested.connect(self._export_selected_figures)
            self._adopt_child_window(self._analysis_explorer, "analysis_explorer")
            self._analysis_explorer.bind_figure_workspace(self._ensure_figure_workspace())
            self._analysis_explorer.set_saved_figure_provider(
                self._figure_entries, status_provider=self._figure_status)
            self._analysis_explorer._style.set_external_provider(self._retained_style_figures)
        return self._analysis_explorer

    def _retained_style_figures(self):
        """Open retained Study/Report editors share deliberate project-wide styling."""
        from .study_figure_window import StudyFigureWindow
        for editor in self.findChildren(StudyFigureWindow):
            if editor._figure is not None and editor._library_owner is self:
                yield str(id(editor)), editor._name, editor

    def _ensure_figure_workspace(self, *, install=True):
        if self._figure_workspace is None:
            from .figure_workspace import FigureWorkspace
            self._figure_workspace = FigureWorkspace(
                parent=self, factory=self._make_figure_surface, mount=self._mount_figure_surface,
                unmount=self._unmount_figure_surface, initialize=self._initialize_figure_surface,
                bind_sources=self._bind_figure_sources, current_binding=self._figure_current_binding,
                table_staleness=self._figure_table_staleness,
                dirty=self._mark_dirty, commit_focus=lambda: self._analysis_explorer.toolbar().commit_focus()
                if self._analysis_explorer is not None else None)
            self._figure_workspace.currentChanged.connect(self._sync_figure_aliases)
            self._figure_workspace.recordsChanged.connect(self._on_figure_records_changed)
            self._figure_workspace.requestFailed.connect(
                lambda _key, reason: self.statusBar().showMessage(reason, 12000))
            if install:
                self._figure_workspace.install_workspace(self._analytical_workspace)
        return self._figure_workspace

    def _on_figure_records_changed(self):
        owner = self._figure_workspace
        key = getattr(self, "_subpop_assignment_figure_id", None)
        if key is not None and owner.get(key) is None:
            self._drop_subpop_assignment()
        self._sync_saved_subpop_action()

    @staticmethod
    def _figure_aliases():
        return {PAGE_DISTRIBUTIONS: "_plots_window", PAGE_DISTRIBUTION_SUMMARIES: "_distribution_summary_window",
                PAGE_COMPARISON_FIGURES: "_comparison_figures_window", PAGE_EVENT_WAVEFORMS: "_event_plot_window",
                PAGE_NOISE: "_psd_window", PAGE_IV: "_iv_window", PAGE_VOLTAGE_SERIES: "_vdep_window",
                PAGE_SUBPOPULATIONS: "_subpop_window"}

    def _sync_figure_aliases(self, _key=""):
        owner = self._figure_workspace
        if owner is None:
            return
        for kind, name in self._figure_aliases().items():
            current = owner.current_id()
            rec = owner.get(current)
            key = current if rec is not None and rec.spec.kind == kind else owner.most_recent(kind)
            setattr(self, name, owner.page(key))
        self._sync_subpop_window()
        owner.refresh_table_staleness()
        # A figure that just became resident hands its chain judgement to its page.
        self._push_noise_chain()

    def _figure_table_staleness(self, record, page):
        if record.spec.kind == PAGE_NOISE:
            # A resident Noise page says so itself (its pending state); a suspended one
            # is judged from what its record says the spectrum was computed with.
            return page is None and self._noise_record_chain_changed(record)
        if record.spec.kind != PAGE_DISTRIBUTIONS:
            return False
        if page is not None:
            return page.drawn_table_is_stale()
        from .figure_adapter_plots import record_table_is_stale
        return record_table_is_stale(record, self, self._analysis_panel.table_staleness)

    def _push_plot_staleness(self):
        if self._figure_workspace is not None:
            self._figure_workspace.refresh_table_staleness()
        elif self._plots_window is not None:
            self._plots_window.schedule_stale_redraw()

    def _noise_record_chain_changed(self, record) -> bool:
        """Whether a Noise figure's PROCESSED spectrum was computed with a Preprocess chain
        the Detection form no longer names. A raw spectrum never is."""
        signal = record.spec.result.payload.get("signal")
        if not isinstance(signal, Mapping) or signal.get("basis") != "processed":
            return False
        # The page's own signature (Normalize is left out of the processed signal, so its
        # settings are no part of it).
        from .psd_panel import noise_chain_signature
        try:
            live = noise_chain_signature(self._analysis_panel.detection_params())
        except Exception:  # noqa: BLE001 — a form that cannot be read judges nothing
            return False
        return live != signal.get("chain_signature")

    def _push_noise_chain(self) -> None:
        """The Preprocess form may have moved: every Noise figure whose spectrum is of the
        PROCESSED signal is re-judged, and none is recomputed. A resident page turns
        pending (Update plot); a suspended figure's record turns stale. Cheap when no
        Noise figure exists, which is why every form change may call it.

        A JUDGEMENT, routed like table staleness: the figure's status is refreshed and
        ``recordChanged`` is sent, and that is all. The form may have moved only because
        another recording's table came on screen, or because a reopened figure was looked
        at; neither is an edit of the figure, so neither arms the save prompt."""
        owner = self._figure_workspace
        if owner is None:
            return
        for key, entry in tuple(owner._entries.items()):
            if entry.record.spec.kind != PAGE_NOISE:
                continue
            if entry.page is not None:
                if entry.page.processing_chain_changed():
                    owner._refresh_pending_status(entry)
                    owner.recordChanged.emit(key)
            elif owner._refresh_table_staleness(entry):
                owner.recordChanged.emit(key)

    def _drop_noise_traces(self, doc_id: str) -> None:
        """A closed recording leaves no processed trace behind on a Noise page."""
        owner = self._figure_workspace
        if owner is None:
            return
        for entry in tuple(owner._entries.values()):
            if entry.record.spec.kind == PAGE_NOISE and entry.page is not None:
                entry.page.release_processed_traces(doc_id)

    def _open_figure_kind(self, kind):
        owner = self._ensure_figure_workspace()
        explorer = self._ensure_analysis_explorer()
        key = owner.most_recent(kind)
        if key is None:
            key = owner.create(kind)
        else:
            owner.activate(key)
        # A resident figure does not remount; the Analysis menu still raises the
        # hidden/background shell, just as its first creation does.
        explorer.show()
        explorer.raise_()
        explorer.activateWindow()
        return owner.page(key)

    def _make_figure_surface(self, kind, figure_id):
        if kind == PAGE_DISTRIBUTIONS:
            from .plots_panel import PlotsWindow
            page = PlotsWindow(self._session, self)
            page.auto_plot = False
            page._phase2_figure_managed = True
            page.set_staleness_provider(self._analysis_panel.table_staleness)
            page.refreshRequested.connect(lambda: self._refresh_figure(figure_id))
            page.eventRowsSelected.connect(lambda source, rows, page=page:
                self._on_plot_event_rows_selected(source, rows, figure_id=figure_id)
                if self._figure_workspace.page(figure_id) is page else None)
            page.figureResultsPublished.connect(self._record_figure_publication)
        elif kind == PAGE_DISTRIBUTION_SUMMARIES:
            from .distribution_summary_panel import DistributionSummaryWindow
            page = DistributionSummaryWindow(self._session, self)
            page.refreshRequested.connect(lambda: self._prepare_figure_sources(figure_id))
        elif kind == PAGE_COMPARISON_FIGURES:
            from .comparison_figures_panel import ComparisonFiguresWindow
            page = ComparisonFiguresWindow(self._session, self)
            page.refreshRequested.connect(lambda: self._prepare_figure_sources(figure_id))
        elif kind == PAGE_EVENT_WAVEFORMS:
            from .event_plot_panel import EventPlotWindow
            page = EventPlotWindow(self._session, self)
            page.refreshRequested.connect(lambda: self._refresh_figure(figure_id))
        elif kind == PAGE_NOISE:
            from .psd_panel import PSDWindow
            page = PSDWindow(self._session, self, defer_initial_refresh=True)
            page.sourceSelected.connect(lambda doc_id, channel, page=page:
                self._pin_figure_source(figure_id, page, doc_id, channel))
            # Signal ▸ Processed runs the Detection panel's Preprocess form; the page
            # reads it at Update and is told when it moves (:meth:`_push_noise_chain`).
            page.set_processing_provider(self._analysis_panel.detection_params)
        elif kind == PAGE_IV:
            from .iv_panel import IVWindow
            page = IVWindow(self._session, self, defer_initial_refresh=True)
        elif kind == PAGE_VOLTAGE_SERIES:
            from .voltage_dep_panel import VDepWindow
            page = VDepWindow(self._session, self, defer_initial_refresh=True)
        elif kind == PAGE_SUBPOPULATIONS:
            from .subpop_window import SubpopReportWindow
            page = SubpopReportWindow(self)
            page.analyzeRequested.connect(lambda: self._run_subpop_for_figure(figure_id))
            page.umapRequested.connect(lambda: self._generate_umap_for_figure(figure_id))
            page.exportRequested.connect(lambda: self._with_subpop_figure(figure_id, self._export_subpop_report))
            page.galleryRequested.connect(lambda: self._with_subpop_figure(figure_id, self._open_gallery_on_subpops, assignment=True))
            page.traceRequested.connect(lambda: self._with_subpop_figure(figure_id, self._colour_trace_by_subpops, assignment=True))
        else:
            raise ValueError("Unsupported analytical figure kind.")
        page._nanoscope_figure_id = figure_id
        page._nanoscope_figure_managed = True
        adapter = page.inspector_adapter()
        update = adapter.update

        def update_figure():
            if kind == PAGE_DISTRIBUTIONS:
                try:
                    self._stage_changed_distribution_sources(figure_id)
                except Exception as exc:
                    # Said in the status bar like a failure of the update itself (see
                    # `_update_figure`); the figure is not updated and not marked CURRENT.
                    self._figure_workspace._fail(figure_id, exc)
                    return
            self._update_figure(figure_id, update)
        adapter.update = update_figure
        return page

    def _mount_figure_surface(self, figure_id, page):
        owner = self._figure_workspace
        record = owner.get(figure_id)
        self._ensure_analysis_explorer().show_page(figure_id, page,
            kind=record.spec.kind, name=record.spec.name,
            refresh=lambda: self._refresh_figure(figure_id))

    def _unmount_figure_surface(self, figure_id, page):
        # A page that leaves the screen (suspended, or cleared with its project) keeps no
        # whole-recording processed trace while it waits to be deleted.
        release = getattr(page, "release_processed_traces", None)
        if callable(release):
            release()
        if self._analysis_explorer is not None:
            self._analysis_explorer.remove_page(figure_id)
        for name in self._figure_aliases().values():
            if getattr(self, name, None) is page:
                setattr(self, name, None)
        # Suspension does not activate a different record. Refresh after dropping
        # the aliases so the launcher cannot retain the retired page's open tint.
        self._sync_analysis_open_state()

    def _figure_current_binding(self, kind):
        from ..core.analysis_workspace import _stable_source, result_recipe
        if kind == PAGE_EVENT_WAVEFORMS:
            result = self._analysis_panel.current_detection()
            return {"follow": False, "source": result_recipe(self, result) if result is not None else None}
        if kind == PAGE_NOISE and self._current_selection is not None:
            doc_id, channel = self._current_selection
            if self._session.has_document(doc_id):
                meta = self._session.document(doc_id).meta
                channel = next((info.index for info in meta.channels if not info.is_dead and
                                effective_kind(self._session, doc_id, info.index) is ChannelKind.CURRENT), channel)
                return {"follow": False, "source": _stable_source(self, doc_id, channel)}
        return {"follow": False, "source": None}

    def _pin_figure_source(self, figure_id, page, doc_id, channel):
        """An explicit source choice owns the same binding used by Update/save."""
        owner = self._figure_workspace
        if owner is None or owner.page(figure_id) is not page or not self._session.has_document(doc_id):
            return
        from ..core.analysis_workspace import _stable_source
        owner.set_binding(figure_id, {"follow": False,
            "source": _stable_source(self, doc_id, int(channel))})
        follow = getattr(page, "_figure_follow_check", None)
        if follow is not None:
            with QtCore.QSignalBlocker(follow):
                follow.setChecked(False)

    def _bind_figure_sources(self, page, record, binding, compute=False, sources=None):
        kind = record.spec.kind
        if kind == PAGE_DISTRIBUTIONS:
            page.set_active_detection(self.active_detection_key())
            page.set_data(self._plots_sources() if sources is None else sources, catalog_only=True)
        elif kind in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES):
            page.set_sources(self._plots_sources(), analyze=False)
        elif kind in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE) and compute:
            # Source notifications stage inputs. Only the explicit Update route
            # below may replace a managed Noise/Event result.
            page.stage_source(binding.get('source'))
        return True

    def _feed_figure_bound_source(self, page, record, binding, *, compute):
        from ..core.analysis_workspace import _resolve_source, resolve_result_recipe
        kind = record.spec.kind
        selected = self._figure_current_binding(kind) if binding.get("follow") else binding
        source = selected.get("source")
        if not source:
            self._figure_workspace._fail(record.spec.figure_id, "Choose a source recording before updating this figure.")
            return False
        if kind == PAGE_EVENT_WAVEFORMS:
            result, reason = resolve_result_recipe(self, source)
            if result is None:
                self._figure_workspace._fail(record.spec.figure_id, reason or "The pinned detection is unavailable; the saved figure is retained.")
                return False
            page.set_source(result.doc_id, result.channel, result.payload, defer_analysis=not compute)
        elif kind == PAGE_NOISE:
            doc_id, channel, reason = _resolve_source(self, source)
            if doc_id is None:
                self._figure_workspace._fail(record.spec.figure_id, reason or "The pinned recording is unavailable; the saved spectrum is retained.")
                return False
            records = page._gather_recordings()
            if not any(r["doc_id"] == doc_id and r["channel"] == channel for r in records):
                self._figure_workspace._fail(record.spec.figure_id, "The pinned current channel or its sample rate is unavailable; the saved spectrum is retained.")
                return False
            if page.set_target(doc_id, channel, recompute=compute) is False:
                return False
        if compute and not page.has_data():
            return False
        return True

    def _initialize_figure_surface(self, page, record, recipe, binding):
        from .plot_appearance import PlotAppearance
        from ..core.figure_codecs import encode_legacy
        from ..core.figure_spec import thaw_json
        kind = record.spec.kind
        self._bind_figure_sources(page, record, binding, False)
        if kind in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE):
            self._feed_figure_bound_source(page, record, binding, compute=False)
        if kind == PAGE_DISTRIBUTIONS:
            page._updating = True
            try:
                if recipe.get("plot_type"):
                    page.set_plot_type(recipe["plot_type"])
            finally:
                page._updating = False
        elif (kind in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES)
              and recipe and "window_template" not in recipe):
            page.set_config(recipe)
        native_settings = recipe.get("window_template")
        if native_settings is None and kind in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE) and recipe.get("view"):
            view = ("Ensemble" if recipe["view"] == "ensemble" else "Density") if kind == PAGE_EVENT_WAVEFORMS else recipe["view"]
            native_settings = {"view": view}
        if kind == PAGE_DISTRIBUTIONS and recipe.get("plot_template") is not None:
            from .plots_panel import plot_config_from_json
            from dataclasses import replace
            template = plot_config_from_json(recipe["plot_template"])
            template = replace(template, style=page.config().style,
                               axes=replace(template.axes, x_min="", x_max="", y_min="", y_max="",
                                            manual_x=None, manual_y=None))
            page._set_config(page._installable(page.config().with_template(template)), draw=False)
        elif native_settings is not None:
            # The native snapshot installers block compute signals. A saved recipe is
            # a draft on a fresh empty figure, never an implicit statistical run.
            from copy import deepcopy
            from dataclasses import replace
            from .figure_adapter_pages import adapter_for
            native_adapter = adapter_for(page, kind)
            native = native_adapter.capture_native()
            payload = deepcopy(native.payload)
            path = "draft" if kind in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES) else "config"
            allowed = (set(page.template_settings()[0])
                       | set(getattr(page, "TEMPLATE_OPTIONAL_KEYS", ()))) - {"style", "source_keys"}
            payload[path] = {**payload.get(path, {}), **{
                key: value for key, value in native_settings.items() if key in allowed}}
            report = native_adapter.restore_native(replace(native, payload=payload))
            if report.skipped:
                raise ValueError("The recipe could not be installed: " + "; ".join(report.warnings))
        native = encode_legacy(record.spec, record.resources, materialize_defaults=True)
        # The canonical New record already owns its look and shared style binding,
        # including edits made while no resident slot was available.
        if kind == PAGE_DISTRIBUTIONS:
            from dataclasses import replace
            from .figure_adapter_plots import _native_config
            page._set_config(replace(page.config(), style=_native_config(native).style), draw=False)
        else:
            from .figure_adapter_pages import _native_appearance
            page.set_appearance_state(_native_appearance(native, kind).to_dict())
        page._appearance_widget.clear_history()

    def _prepare_figure_sources(self, figure_id):
        owner = self._figure_workspace
        record, page = owner.get(figure_id), owner.page(figure_id)
        if record is not None and page is not None:
            self._bind_figure_sources(page, record, owner.binding(figure_id), False)

    def _update_figure(self, figure_id, update):
        owner = self._figure_workspace
        record, page = owner.get(figure_id), owner.page(figure_id)
        if record is None or page is None:
            return
        if record.spec.kind in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE) and not self._feed_figure_bound_source(
                page, record, owner.binding(figure_id), compute=False):
            return
        from dataclasses import replace
        entry = owner._entries[figure_id]
        try:
            result = update()
        except Exception as exc:
            owner._fail(figure_id, exc)
            return
        if result is False:
            return
        # The page refused (unreadable fit range) before taking its staged tables: status unchanged; the page said why.
        if record.spec.kind == PAGE_DISTRIBUTIONS and getattr(page, "_figure_pending_sources", None) is not None:
            return
        if entry.busy or getattr(page, "_gather_worker", None) is not None:
            return  # The result callback commits freshness when its work finishes.
        kind = record.spec.kind
        if kind in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE, PAGE_DISTRIBUTION_SUMMARIES,
                    PAGE_COMPARISON_FIGURES):
            has_result = page.has_data()
        elif kind in (PAGE_IV, PAGE_VOLTAGE_SERIES):
            has_result = bool(page.points())
        elif kind == PAGE_SUBPOPULATIONS:
            has_result = page.report() is not None
        else:
            has_result = (bool(getattr(page, "_plot_armed", False))
                          and getattr(page, "_message_item", None) is None)
        if not has_result:
            owner._set_metadata(entry, pending_result_base=None)
            entry.record = replace(entry.record, spec=replace(entry.record.spec,
                result=replace(entry.record.spec.result, status="empty")))
            owner.checkpoint(figure_id, changed=True)
            return
        owner.mark_current(figure_id)
        owner.checkpoint(figure_id, changed=True)

    def _refresh_figure(self, figure_id):
        owner = self._figure_workspace
        record, page = owner.get(figure_id), owner.page(figure_id)
        if record is None or page is None:
            return
        kind = record.spec.kind
        if kind in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE):
            self._update_figure(figure_id, lambda: self._feed_figure_bound_source(
                page, record, owner.binding(figure_id), compute=True))
        elif kind == PAGE_DISTRIBUTIONS:
            self._prepare_figure_sources(figure_id)
            self._update_figure(figure_id, page._run_plot)
        elif kind in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES):
            self._prepare_figure_sources(figure_id)
            self._update_figure(figure_id, page.analyze)
        elif kind in (PAGE_IV, PAGE_VOLTAGE_SERIES):
            self._update_figure(figure_id, page.refresh)
        elif kind == PAGE_SUBPOPULATIONS:
            self._run_subpop_for_figure(figure_id)

    def _record_figure_publication(self, receipt):
        if isinstance(receipt, dict) and self._figure_workspace.get(receipt.get("figure_id")) is not None:
            self._analytical_workspace.extra["report_fit_publication"] = dict(receipt)
            self._mark_dirty()

    def _run_subpop_for_figure(self, figure_id):
        owner = self._figure_workspace
        page = owner.page(figure_id) if owner is not None else None
        if page is None:
            return False
        source = None
        if page.report() is not None:
            source, reason = self._subpop_report_source(page)
            if source is None:
                owner._fail(figure_id, reason + " The stored report is unchanged. "
                            "Create a new Subpopulations figure to analyse a different detection.")
                return False
        old = self._subpop_launch_figure_id
        self._subpop_launch_figure_id = figure_id
        try:
            return self._run_subpopulations(source_result=source)
        finally:
            self._subpop_launch_figure_id = old

    def _subpop_result_recipe(self, result):
        """Durable detection identity, with an additional curation-independent stamp."""
        from dataclasses import replace
        from ..core.analysis_workspace import result_recipe

        recipe = result_recipe(self, result)
        frame = result.payload.df.copy()
        # Review state and notes may change. A report update may use newly accepted
        # rows, but never new boundaries, measurements or detector inputs.
        frame["accepted"] = True
        frame["flag"] = ""
        frame = frame.drop(columns=["review", "note"], errors="ignore")
        normalized = replace(result, payload=replace(result.payload, df=frame))
        recipe["subpopulation_detection_fingerprint"] = result_recipe(
            self, normalized)["result_fingerprint"]
        return recipe

    def _subpop_report_source(self, page):
        """Resolve this report's own detection; selection never supplies a fallback."""
        from ..core.analysis_workspace import resolve_result_recipe, _resolve_source
        from ..analysis.events import EventSet

        recipe = getattr(page, "_nanoscope_workspace_source", None)
        original = getattr(page, "_nanoscope_workspace_source_result", None)
        # Older live reports can still capture their original result. A restored report
        # without a recipe has no provable source and remains view/export only.
        if not recipe and original is not None and self._session.has_document(original.doc_id):
            recipe = self._subpop_result_recipe(original)
        if not recipe:
            return None, "This report has no saved detection source."
        result, reason = resolve_result_recipe(self, recipe)
        if result is None and recipe.get("subpopulation_detection_fingerprint"):
            doc_id, channel, _ = _resolve_source(self, recipe)
            candidate = (self._session.results.get(doc_id, channel, recipe.get("module_id"))
                         if doc_id is not None and channel is not None else None)
            if candidate is not None and isinstance(candidate.payload, EventSet):
                stamp = self._subpop_result_recipe(candidate)["subpopulation_detection_fingerprint"]
                if stamp == recipe["subpopulation_detection_fingerprint"]:
                    result = candidate
        if result is None:
            return None, "Cannot update the original detection: " + (reason or "source unavailable.")
        if result.source_fingerprint is not None:
            from ..core.identity import describe_source_change
            from ..core.segment import SegmentDocument, persistent_key
            doc = self._session.document(result.doc_id)
            documents = {persistent_key(item): item for item in self._session.documents()}
            while isinstance(doc, SegmentDocument):
                doc = documents.get(doc.spec.parent_key)
            if doc is None:
                return None, "The report's original recording is not open."
            self._remember_loaded_source(doc.doc_id)
            change = describe_source_change(result.source_fingerprint,
                self._loaded_source_fingerprints.get(doc.doc_id), loaded_n_samples=doc.meta.n_samples)
            if change:
                return None, "The recording changed since this detection was saved: " + change
        if original is not None and result is not original:
            key = (result.doc_id, result.channel, result.module_id)
            replay = getattr(self, "_subpop_history_replays", {}).get(key)
            # An old Curate undo/redo label is not evidence about this result: a
            # byte-identical re-detection pushes no step and leaves that label intact.
            # Only the actual payload installed by history may stand in for the
            # report's original object; ordinary curation retains its payload.
            known_payload = (result.payload is original.payload
                             or (replay is not None and replay() is result.payload))
            same_detection = (recipe.get("subpopulation_detection_fingerprint") ==
                              self._subpop_result_recipe(result)["subpopulation_detection_fingerprint"])
            if not (known_payload and same_detection):
                return None, "The detection used for this report has been replaced."
        stale = self._analysis_panel._statistics_staleness(result)
        if stale is not None:
            return None, "The report's source detection is stale: " + stale[1]
        columns, stale_text = self._analysis_panel.table_staleness(
            result.doc_id, result.channel, result.module_id)
        if columns:
            return None, "The report's source detection is stale: " + stale_text
        return result, None

    def _generate_umap_for_figure(self, figure_id):
        """Add an embedding to the selected stored report, never the current detection."""
        from ..analysis.subpop_report import add_umap_to_report

        owner = self._figure_workspace
        page = owner.page(figure_id) if owner is not None else None
        if page is None or page.report() is None or self._umap_figure_request is not None:
            return
        if owner.is_read_only():
            self.statusBar().showMessage(
                "This newer analytical workspace is read-only; UMAP was not generated.", 10000)
            return
        original = page.report()
        if "UMAP" in page.embedding_view().embeddings():
            page.embedding_view().set_embedding("UMAP")
            return
        if original.clustering is None:
            QtWidgets.QMessageBox.information(
                self, "Generate UMAP",
                "This report has no stored PCA scores. Generate Subpopulations first.")
            return
        token = owner.request_token(figure_id)

        def job(progress):
            result = add_umap_to_report(original, progress=progress)
            result._nanoscope_umap_request = token
            return result

        self._analysis_panel.forget_pending_job()
        if not self._analysis_controller.run_callable(job, label="UMAP", wants_progress=True):
            owner.finish_request(token)
            return
        self._umap_figure_request = (token, original)
        page.set_umap_busy(True)
        self.statusBar().showMessage(
            "Generating UMAP from the stored report. Existing analysis results remain available; "
            "Cancel discards the pending embedding.", 0)

    def _on_umap_stopped(self):
        claim, self._umap_figure_request = self._umap_figure_request, None
        owner = self._figure_workspace
        if claim is not None and owner is not None:
            token, _original = claim
            page = owner.page(token[1])
            if page is not None:
                page.set_umap_busy(False)
            owner.finish_request(token)

    def _with_subpop_figure(self, figure_id, callback, *, assignment=False):
        owner = self._figure_workspace
        page = owner.page(figure_id)
        if page is None:
            return
        old = self._subpop_window
        self._subpop_window = page
        try:
            if assignment and page.report() is not None:
                target, reason = self._subpop_report_source(page)
                if target is None:
                    reason = reason.replace("Cannot update", "Cannot use", 1)
                    owner._fail(figure_id, reason)
                    self.statusBar().showMessage(reason, 10000)
                    return
                if self._analysis_panel.current_detection() is not target:
                    name = self._session.document(target.doc_id).display_name
                    detector = self._detector_display_name(target.module_id)
                    self.statusBar().showMessage(
                        f"This report belongs to “{name}”, channel {target.channel}, {detector}. "
                        "Select that original detection to show its events. "
                        "The stored report and its colouring are unchanged.", 10000)
                    return
                key = (str(target.doc_id), int(target.channel), str(target.module_id))
                held = (self._subpop_assignment_figure_id == figure_id
                        and self._subpop_key == key
                        and self._subpop_frame is target.payload.df
                        and self._subpop_assignment is not None
                        and self._subpop_signal is not None)
                if not held:
                    # A restored/different figure must rebuild only its validated
                    # original detection's trace, never the current selection's.
                    try:
                        signal, _fs = self._analysis_panel._active_signal(
                            dict(target.payload.provenance or {}), strict=True,
                            key=(target.doc_id, target.channel))
                        if signal is None:
                            raise ValueError("the original detection trace is unavailable")
                    except Exception as exc:  # noqa: BLE001 — leave held colouring intact
                        reason = f"Cannot show this report's original detection trace: {exc}"
                        owner._fail(figure_id, reason)
                        self.statusBar().showMessage(reason, 10000)
                        return
                    pending_ids = self._subpop_accepted_ids_pending
                    pending_signal = self._subpop_signal_pending
                    self._subpop_accepted_ids_pending = getattr(
                        page, "_nanoscope_workspace_accepted_event_ids", None)
                    self._subpop_signal_pending = signal
                    try:
                        self._adopt_subpop_assignment(page.report(), target)
                    finally:
                        self._subpop_accepted_ids_pending = pending_ids
                        self._subpop_signal_pending = pending_signal
                    self._subpop_assignment_figure_id = figure_id
                page._nanoscope_workspace_source_result = target
                self._sync_subpop_window()
            return callback()
        finally:
            self._subpop_window = old

    def _install_subpop_figure_report(self, figure_id, report, *, source_result=None,
                                     source_recipe=None, accepted_event_ids=None):
        owner = self._figure_workspace
        page = owner.page(figure_id)
        if page is None or owner.get(figure_id) is None:
            return
        if source_result is not None and self._session.has_document(source_result.doc_id):
            live_recipe = self._subpop_result_recipe(source_result)
            if source_recipe is None:
                source_recipe = live_recipe
            elif source_recipe.get("result_fingerprint") == live_recipe["result_fingerprint"]:
                source_recipe = {**source_recipe, "subpopulation_detection_fingerprint":
                                 live_recipe["subpopulation_detection_fingerprint"]}
        page._nanoscope_workspace_source_result = source_result
        page._nanoscope_workspace_source = source_recipe
        page._nanoscope_workspace_accepted_event_ids = accepted_event_ids
        self._subpop_assignment_figure_id = figure_id
        page.set_report(report, subtitle=str(report.meta.get("recording") or "(unnamed)"))
        owner.mark_current(figure_id)
        owner.checkpoint(figure_id, changed=True)
        self._sync_subpop_window()

    def _show_saved_figure(self, key):
        explorer = self._ensure_analysis_explorer()
        return explorer.show_saved_figure(key, self._ensure_saved_figure_view())

    def _compose_saved_figures(self, ids):
        return self._ensure_saved_figure_view().compose_entries(ids)

    def _delete_saved_figures(self, ids):
        """The Explorer already confirmed the exact selected project entries."""
        allowed = {entry.artifact_id for entry in self._figure_entries()}
        selected = set(ids) & allowed
        if not selected:
            return
        self._analytical_workspace.artifacts[:] = [entry for entry in self._analytical_workspace.artifacts
                                                   if entry.artifact_id not in selected]
        self._mark_dirty()
        if self._analysis_explorer is not None:
            self._analysis_explorer.refresh_figures()

    def _new_figure_from_recipe(self):
        from .figure_recipes import RecipeDialog, initial_recipe
        owner = self._ensure_figure_workspace()
        token = (owner, owner._epoch)
        dialog = RecipeDialog(self._analysis_explorer or self, choose=True)
        try:
            accepted = dialog.exec() == QtWidgets.QDialog.Accepted
            entry, include_words = dialog.selected(), dialog.words.isChecked()
        finally:
            dialog.deleteLater()
        if not accepted:
            return
        if token != (self._figure_workspace, owner._epoch):
            self.statusBar().showMessage("The project changed; open New from recipe again.", 10000)
            return
        if entry is None:
            return
        try:
            recipe = initial_recipe(entry, include_words=include_words)
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self._analysis_explorer or self, "New figure from recipe",
                                          f"{entry.name} could not be recalled.\n\n{exc}")
            return
        owner.create(entry.kind, initial_recipe=recipe)

    def _save_figure_recipe(self, figure_id):
        from .figure_recipes import save_page, entries, SUPPORTED, UNSUPPORTED
        owner = self._figure_workspace
        record = owner.get(figure_id) if owner is not None else None
        page = owner.page(figure_id) if record is not None else None
        if page is None:
            return
        kind = record.spec.kind
        if kind not in SUPPORTED:
            QtWidgets.QMessageBox.information(self, "Figure recipes", UNSUPPORTED[kind])
            return
        self._analysis_explorer.toolbar().commit_focus()
        token = (owner, owner._epoch, page, owner.revision_token(figure_id))
        def unchanged():
            return (self._figure_workspace is owner and owner._epoch == token[1]
                    and owner.page(figure_id) is page and owner.revision_token(figure_id) == token[3])
        name, accepted = QtWidgets.QInputDialog.getText(self._analysis_explorer or self,
            "Save figure recipe", "Recipe name (analysis settings and look; words optional when recalled):")
        name = name.strip()
        if not accepted or not name or not unchanged():
            return
        overwrite = any(entry.kind == kind and entry.name == name and not entry.builtin for entry in entries())
        if overwrite and QtWidgets.QMessageBox.question(self._analysis_explorer or self,
                "Replace recipe?", f"Replace the saved recipe “{name}”?") != QtWidgets.QMessageBox.Yes:
            return
        if not unchanged():
            return
        try:
            save_page(page, kind, name, overwrite=overwrite)
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Save figure recipe", str(exc))
            return
        self.statusBar().showMessage(f"Saved recipe “{name}”. Recall it with New figure ▸ From recipe…", 10000)

    def _manage_figure_recipes(self, kind):
        from .figure_recipes import RecipeDialog
        dialog = RecipeDialog(self._analysis_explorer or self, kind=kind)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _analysis_explorer_context(self, *, for_figure=True) -> str:
        """Describe the figure's own sources, independently of main selection."""
        documents = list(self._session.documents())
        count = len(documents)
        owner = self._figure_workspace if for_figure else None
        key = owner.current_id() if owner is not None else None
        record = owner.get(key) if owner is not None else None
        if record is not None:
            page = owner.page(key)
            if record.spec.kind == PAGE_NOISE and page is not None and page.curves():
                labels = [str(curve.label) for curve in page.curves()]
                return "Displayed: " + ", ".join(labels)
            if record.spec.kind == PAGE_EVENT_WAVEFORMS and page is not None:
                doc_id = getattr(page, "_doc_id", None)
                if doc_id is not None and self._session.has_document(doc_id):
                    return f"{self._session.document(doc_id).display_name} · channel {int(page._channel)}"
            if page is not None and callable(getattr(page, "selected_sources", None)):
                sources = page.selected_sources()
                labels = [str(source.label) for source in sources]
                return "Selected sources: " + (", ".join(labels) if labels else "none")
            binding = owner.binding(key)
            source = (getattr(page, "_nanoscope_workspace_event_source", None)
                      if record.spec.kind == PAGE_EVENT_WAVEFORMS else None) or binding.get("source")
            if source:
                from ..core.analysis_workspace import _resolve_source
                doc_id, channel, _reason = _resolve_source(self, source)
                if doc_id is not None:
                    return f"{self._session.document(doc_id).display_name} · channel {int(channel)}"
                return "Pinned source: " + Path(str(source.get("source_key") or "unavailable")).name
            if record.spec.kind in (PAGE_IV, PAGE_VOLTAGE_SERIES):
                return f"Project recordings · {count} recording{'s' if count != 1 else ''} open"
            return f"{len(record.spec.sources)} figure source{'s' if len(record.spec.sources) != 1 else ''}"
        if (self._current_selection is not None
                and self._session.has_document(self._current_selection[0])):
            doc_id, channel = self._current_selection
            name = self._session.document(doc_id).display_name
            suffix = f" · {count} recording{'s' if count != 1 else ''} open"
            return f"{name} · channel {int(channel)}{suffix}"
        if count:
            return f"{count} recording{'s' if count != 1 else ''} open · none selected"
        return "No recordings open"

    # -- project Figure Library (not the Event Gallery) ------------------------------------
    def _export_selected_figures(self, *, directory=None, formats=("pdf",), sizing="native",
                                 dpi=300, selected_ids=None, saved_ids=None,
                                 interactive=True, overwrite=False):
        """Export a fixed selection of committed live figures and immutable snapshots."""
        from copy import deepcopy
        from .figure_batch_export import build_export_plan, run_export_plan
        from .figure_batch_export_dialog import choose_batch_export_options, show_batch_export_results
        from .figure_bundle import FigureSnapshot
        from .export_paths import confirm_overwrite, existing_of

        explorer = self._ensure_analysis_explorer()
        owner = self._figure_workspace
        epoch = self._figure_project_epoch
        listing = explorer.figures_list()
        selected = tuple(listing.selected_ids() if selected_ids is None else selected_ids)
        saved = tuple(listing.selected_ids("saved") if saved_ids is None else saved_ids)
        previous_rows = listing.selected_ids() + listing.selected_ids("saved")
        previous_row = listing.current_id() or listing.current_id("snapshot") or listing.current_id("composition")
        previous_page = explorer.current_page_id()
        previous_current = owner.current_id()

        def is_current():
            return self._figure_project_epoch == epoch and self._figure_workspace is owner

        try:
            explorer.toolbar().commit_focus()
            for key in selected:
                if owner.export_ready(key)[0]:
                    owner.checkpoint(key)
            tokens = {key: owner.revision_token(key) for key in selected}
            entries = {entry.artifact_id: entry for entry in self._figure_entries()}
            snapshots, saved_values = {}, {}
            for key in saved:
                entry = entries.get(key)
                if entry is None:
                    raise ValueError("A selected snapshot is no longer available. Select it again.")
                snapshots[key] = (entry.payload.get("name") or "Saved figure",
                                  FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs))
                saved_values[key] = (deepcopy(entry.payload), dict(entry.blobs))

            def saved_is_current(key):
                current = next((entry for entry in self._figure_entries() if entry.artifact_id == key), None)
                return current is not None and (current.payload, current.blobs) == saved_values[key]

            names = [owner.get(key).spec.name if owner.get(key) is not None else "Unavailable figure"
                     for key in selected] + [name for name, _snapshot in snapshots.values()]
            if not names:
                raise ValueError("Select live figures or saved snapshots in the Figures list first.")
            if directory is None:
                if not interactive:
                    raise ValueError("An export folder is required.")
                choice = choose_batch_export_options(explorer, names)
                if choice is None:
                    return ()
                directory, formats, sizing, dpi = choice
            if (not is_current()
                    or any(owner.revision_token(key) != token for key, token in tokens.items())
                    or any(not saved_is_current(key) for key in saved)):
                raise ValueError("The project or selected figures changed while choosing export options. Select them again.")
            plan = build_export_plan(owner, selected, directory, formats=formats, sizing=sizing,
                                     dpi=dpi, snapshots=snapshots, ready=owner.export_ready)
            if interactive:
                if not confirm_overwrite(explorer, "Export selected figures", existing_of(plan.paths)):
                    return ()
                overwrite = True
            if not is_current():
                raise ValueError("The project changed before export started.")
            progress = None
            if interactive:
                progress = QtWidgets.QProgressDialog("Preparing committed figures…", "Cancel", 0, len(plan.paths), explorer)
                progress.setWindowTitle("Export selected figures")
                progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.show()

            def advance(result, done, total):
                if progress is not None:
                    progress.setLabelText(f"{result.name}: {result.status}")
                    progress.setValue(done)
                    QtWidgets.QApplication.processEvents()

            try:
                results = run_export_plan(plan, owner=owner,
                    capture=lambda key: self._capture_figure(key)[1], is_current=is_current,
                    saved_is_current=saved_is_current, ready=owner.export_ready,
                    cancelled=(lambda: progress.wasCanceled()) if progress is not None else None,
                    progress=advance, overwrite=overwrite)
            finally:
                if progress is not None:
                    progress.close()
                    progress.deleteLater()
                if is_current():
                    if previous_current is None:
                        owner.clear_current()
                    if previous_page == PAGE_HOME or previous_page is None:
                        explorer.show_home()
                    else:
                        explorer.raise_page(previous_page)
                    explorer.refresh_figures()
                    listing.restore_selection(previous_rows, current_id=previous_row)
            counts = {state: sum(result.status == state for result in results)
                      for state in ("written", "failed", "cancelled")}
            self.statusBar().showMessage(
                f"Figure export: {counts['written']} written, {counts['failed']} failed, "
                f"{counts['cancelled']} cancelled.", 12000)
            if interactive:
                show_batch_export_results(explorer, results)
            return results
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            if not interactive:
                raise
            QtWidgets.QMessageBox.warning(explorer, "Export selected figures", str(exc))
            return ()

    def _figure_entries(self):
        if self._figure_workspace is not None and self._figure_workspace.is_read_only():
            return ()
        from ..core.figure_library import figures
        return figures(self._analytical_workspace)

    def _figure_status(self, entry):
        from ..core.figure_library import figure_status, source_state
        try:
            live = source_state(self, entry.payload.get("source_recipes", []))
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            live = None
        return figure_status(entry, live)

    def _append_figure(self, entry) -> None:
        self._analytical_workspace.artifacts.append(entry)
        if self._analysis_explorer is not None:
            self._analysis_explorer.refresh_figures()

    def _ensure_saved_figure_view(self):
        from .saved_figure_view import SavedFigureView
        if self._figure_library is None:
            self._figure_library = SavedFigureView(
                self._ensure_analysis_explorer(), entries=self._figure_entries, status=self._figure_status,
                append=self._append_figure, project_token=lambda: self._figure_project_epoch)
            self._figure_library.changed.connect(self._mark_dirty)
            self._figure_library.changed.connect(lambda: self._analysis_explorer.refresh_figures()
                if self._analysis_explorer is not None else None)
            self._figure_library.entryCreated.connect(self._show_saved_figure)
        return self._figure_library

    def _open_figure_library(self) -> None:
        """Compatibility entry point: saved figures now use the project's one list."""
        explorer = self._ensure_analysis_explorer()
        explorer.refresh_figures()
        entries = self._figure_entries()
        if entries:
            current = self._figure_library.current_id() if self._figure_library is not None else None
            self._show_saved_figure(current if current in {e.artifact_id for e in entries} else entries[0].artifact_id)
        explorer.show()
        explorer.raise_()
        explorer.activateWindow()

    def _figure_page(self, page_id: str):
        if self._figure_workspace is not None:
            record = self._figure_workspace.get(page_id)
            if record is not None:
                page = self._figure_workspace.page(page_id)
                if page is None:
                    owner = self._figure_workspace
                    failures = []
                    def refused(key, reason):
                        if key == page_id:
                            failures.append(reason)
                    owner.requestFailed.connect(refused)
                    try:
                        if not owner.activate(page_id):
                            raise ValueError(failures[-1] if failures else "This figure could not be opened for capture.")
                        page = owner.page(page_id)
                    finally:
                        owner.requestFailed.disconnect(refused)
                return self._analysis_page_artifact(record.spec.kind), page
        artifact_id = self._analysis_page_artifact(page_id)
        page = self._workspace_surface(artifact_id) if artifact_id else None
        if page is None:
            raise ValueError("Open an analytical figure before saving it.")
        return artifact_id, page

    def _figure_capture_identity(self, page_id: str):
        """Materialize one frozen view and pin its owner across modal capture choices."""
        _artifact_id, page = self._figure_page(page_id)
        owner = self._figure_workspace
        token = owner.revision_token(page_id) if owner is not None else None
        return self._figure_project_epoch, owner, token[:2] if token is not None else page

    def _check_figure_capture_identity(self, page_id: str, expected) -> None:
        epoch, owner, identity = expected
        if self._figure_project_epoch != epoch or self._figure_workspace is not owner:
            raise ValueError("The project changed while choosing capture options; nothing was saved.")
        token = owner.revision_token(page_id) if owner is not None else None
        if isinstance(identity, tuple):
            matches = token is not None and token[:2] == identity
        else:
            artifact_id = self._analysis_page_artifact(page_id)
            matches = artifact_id is not None and self._workspace_surface(artifact_id) is identity
        if not matches:
            raise ValueError("The figure was deleted or replaced while choosing capture options; nothing was saved.")

    def _figure_capture_options(self, page_id: str, *, ask_name: bool):
        """Name/residual inclusion only: no analysis controls or live view mutation."""
        from .figure_library import default_figure_name
        _artifact_id, page = self._figure_page(page_id)
        residual_getter = getattr(page, "figure_residual_target", None)
        residual = residual_getter() if callable(residual_getter) else None
        if not ask_name and residual is None:
            return "Figure", False
        dialog = QtWidgets.QDialog(self._analysis_explorer or self)
        dialog.setWindowTitle("Save figure snapshot" if ask_name else "Figure bundle")
        layout = QtWidgets.QVBoxLayout(dialog)
        record = self._figure_workspace.get(page_id) if self._figure_workspace is not None else None
        name = QtWidgets.QLineEdit(record.spec.name if record is not None else default_figure_name(page_id, page))
        if ask_name:
            name.setMaxLength(200)
            layout.addWidget(QtWidgets.QLabel("Figure name:"))
            layout.addWidget(name)
        include = QtWidgets.QCheckBox("Include the displayed residual panel")
        include.setToolTip("Copies the existing residuals without fitting or recomputing them.")
        include.setVisible(residual is not None)
        layout.addWidget(include)
        note = QtWidgets.QLabel(
            "Saves the visible picture and exact plotted numbers. Pending analysis changes "
            "are recorded separately; this does not apply them.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if ask_name:
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setEnabled(bool(name.text().strip()))
            name.textChanged.connect(lambda text: buttons.button(
                QtWidgets.QDialogButtonBox.StandardButton.Save).setEnabled(bool(text.strip())))
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return name.text().strip(), include.isChecked()

    def _capture_figure(self, page_id: str, *, include_residual: bool = False):
        """Copy one visible result. No checkpoint, navigation, fitting or replotting."""
        from ..core.figure_library import source_recipes, source_state
        from .figure_bundle import capture_figure_snapshot, encode_figure_table

        artifact_id, page = self._figure_page(page_id)
        applied_spec = None
        record = self._figure_workspace.get(page_id) if self._figure_workspace is not None else None
        if record is not None:
            if not self._figure_workspace.checkpoint(page_id):
                raise ValueError("The figure could not be preserved for capture.")
            from .figure_adapter_pages import capture_applied_record
            from ..core.figure_codecs import encode_legacy
            applied_record = capture_applied_record(self._figure_workspace.get(page_id))
            applied_spec = applied_record.spec
            analysis = encode_legacy(applied_record.spec, applied_record.resources)
        else:
            keyword = self._workspace_kwarg(artifact_id)
            captured = capture_analysis_workspace(**{keyword: page})
            analysis = next(a for a in captured.artifacts if a.artifact_id == artifact_id)
        bindings = getattr(page, "figure_source_recipes", None)
        if callable(bindings):
            analysis.payload["figure_sources"] = bindings()
        getter = getattr(page, "figure_export_target", None)
        ordered = getattr(page, "figure_export_items", None)
        residual_getter = getattr(page, "figure_residual_target", None)
        residual = residual_getter() if callable(residual_getter) else None
        components = ["current figure"]
        report_companion = False
        if callable(ordered):
            items = list(ordered())
            if not items or len(items) > 2:
                raise ValueError("This figure does not expose a supported frozen canvas.")
            target = items[0]
            if len(items) == 2:
                residual = items[1]
                include_residual = True
                components = ["embedding", "waveform overlay"]
                report_companion = True
        elif callable(getter):
            target = getter()
        else:
            raise ValueError("This view does not yet expose a figure export target.")
        data = page._plot_data()
        if data.empty:
            raise ValueError("There are no plotted numerical values to save.")
        _label, capture_status = self._analysis_explorer_status(page_id, page)
        recipes = source_recipes(analysis)
        live = source_state(self, recipes) if capture_status == "current" else None
        names = []
        for recipe in recipes:
            label = str(recipe.get("source_key") or "")
            channel = recipe.get("channel")
            if channel is not None:
                label += f" · channel {int(channel)}"
            if label and label not in names:
                names.append(label)
        source_label = "; ".join(names) or "Saved analytical source (see provenance)"
        resources = {}
        table_names = {}
        for index, (key, frame) in enumerate(analysis.tables.items()):
            filename = f"analysis_tables/table_{index}.parquet"
            resources[filename] = encode_figure_table(frame)
            table_names[key] = filename
        blob_names = {}
        for index, (key, value) in enumerate(analysis.blobs.items()):
            extension = ".parquet" if str(key).endswith(".parquet") else ".bin"
            filename = f"analysis_blobs/blob_{index}{extension}"
            resources[filename] = bytes(value)
            blob_names[key] = filename
        if not include_residual:
            # Optional exact Cartesian replay enables true shared axes on detached
            # copies. Unsupported graphics remain usable as frozen composition panels.
            from .figure_composition import capture_cartesian_model
            model = capture_cartesian_model(target)
            if model is not None:
                resources["rendering/cartesian.json"] = model
        applied_getter = getattr(page, "figure_applied_config", None)
        applied = applied_getter() if callable(applied_getter) else None
        if applied is not None:
            # The plot API provides JSON-shaped applied settings, never live objects.
            applied_config = applied
        else:
            applied_config = None
        details_getter = getattr(page, "details_text", None)
        figure_details = str(details_getter()) if callable(details_getter) else ""
        snapshot = capture_figure_snapshot(
            target, data=data,
            config={"analysis_kind": analysis.kind, "schema_version": analysis.schema_version,
                    "workspace_state": analysis.payload,
                    "applied_settings": applied_config,
                    "state_contract": "workspace_state includes current controls; data and saved picture describe the visible result",
                    "tables": table_names, "blobs": blob_names},
            provenance={"source": source_label, "source_recipes": recipes,
                        "capture_status": capture_status, "components": components,
                        "details": figure_details,
                        "analysis_recomputed": False, "source_state": live},
            residual_item=residual, include_residual=include_residual,
            additional_resources=resources, applied_spec=applied_spec)
        if report_companion:
            # Reuse the two-canvas renderer, but never call a waveform a residual
            # in the durable manifest. Its true panel identities are in provenance.
            from dataclasses import replace
            snapshot = replace(snapshot, includes_residual=False)
        return analysis, snapshot, source_label, live, capture_status

    def _save_analysis_figure(self, page_id: str, name: str | None = None, *,
                              include_residual: bool = False):
        from ..core.figure_library import make_figure
        try:
            identity = self._figure_capture_identity(page_id)
            if name is None:
                selected = self._figure_capture_options(page_id, ask_name=True)
                if selected is None:
                    return None
                name, include_residual = selected
            self._check_figure_capture_identity(page_id, identity)
            analysis, snapshot, source_label, live, status = self._capture_figure(
                page_id, include_residual=include_residual)
            metadata, blobs = snapshot.to_storage()
            record = self._figure_workspace.get(page_id) if self._figure_workspace is not None else None
            entry = make_figure(
                name=name, page_id=record.spec.kind if record is not None else page_id,
                source_label=source_label, analysis=analysis,
                snapshot_metadata=metadata, snapshot_blobs=blobs, live_state=live,
                capture_status=status, applied_spec=snapshot.figure_spec(),
                origin_figure_id=page_id if record is not None else None)
            self._check_figure_capture_identity(page_id, identity)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self._analysis_explorer or self,
                                          "Save figure snapshot", str(exc))
            return None
        self._append_figure(entry)
        self._mark_dirty()
        self._show_saved_figure(entry.artifact_id)
        self.statusBar().showMessage("Figure saved in this project. Save Project to persist it to disk.", 10000)
        return entry

    def _export_analysis_bundle(self, page_id: str, path: str | Path | None = None, *,
                                include_residual: bool = False):
        from .export_paths import ask_export_path
        from .figure_bundle import write_figure_bundle
        from ..core.atomic_write import take_identity_notes
        parent = self._analysis_explorer or self
        try:
            identity = self._figure_capture_identity(page_id)
            if path is None:
                selected = self._figure_capture_options(page_id, ask_name=False)
                if selected is None:
                    return None
                _name, include_residual = selected
                path = ask_export_path(
                    parent, title="Save figure bundle", suggestion=f"{page_id}_figure.zip",
                    filters="Figure bundle (*.zip)",
                    outputs=lambda chosen: [chosen if chosen.suffix else chosen.with_suffix(".zip")])
                if path is None:
                    return None
            self._check_figure_capture_identity(page_id, identity)
            out = Path(path)
            if not out.suffix:
                out = out.with_suffix(".zip")
            if out.suffix.lower() != ".zip":
                raise ValueError("Choose a .zip filename for the figure bundle.")
            _analysis, snapshot, _source, _live, _status = self._capture_figure(
                page_id, include_residual=include_residual)
            self._check_figure_capture_identity(page_id, identity)
            write_figure_bundle(out, snapshot)
            notes = take_identity_notes(out)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(parent, "Save figure bundle", str(exc))
            return None
        if notes:
            QtWidgets.QMessageBox.information(parent, "Save figure bundle", "\n\n".join(notes))
        self.statusBar().showMessage(f"Saved figure bundle → {out}", 10000)
        return out

    @staticmethod
    def _analysis_page_artifact(page_id: str) -> str | None:
        return {
            PAGE_DISTRIBUTIONS: PLOTS_ARTIFACT_ID,
            PAGE_DISTRIBUTION_SUMMARIES: DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
            PAGE_COMPARISON_FIGURES: COMPARISON_FIGURES_ARTIFACT_ID,
            PAGE_EVENT_WAVEFORMS: EVENT_PLOTS_ARTIFACT_ID,
            PAGE_NOISE: PSD_ARTIFACT_ID,
            PAGE_IV: IV_ARTIFACT_ID,
            PAGE_VOLTAGE_SERIES: VOLTAGE_DEPENDENCE_ARTIFACT_ID,
            PAGE_SUBPOPULATIONS: SUBPOPULATION_ARTIFACT_ID,
        }.get(page_id)

    def _analysis_explorer_status(
            self, page_id: str, page: QtWidgets.QWidget) -> tuple[str, str]:
        """Return the visible page's freshness label without recomputing anything."""
        if self._figure_workspace is not None:
            record = self._figure_workspace.get(page_id)
            if record is not None:
                if record.spec.result.status == "empty":
                    return "NO DATA", "empty"
                if record.spec.result.status == "frozen":
                    return "SAVED RESULT", "snapshot"
                if record.spec.result.status in ("stale", "missing"):
                    return "STALE", "stale"
                page_id = record.spec.kind
        if page_id in (PAGE_DISTRIBUTIONS, PAGE_DISTRIBUTION_SUMMARIES,
                       PAGE_COMPARISON_FIGURES) and bool(getattr(page, "_stale", False)):
            return "STALE", "stale"
        artifact_id = self._analysis_page_artifact(page_id)
        if artifact_id in self._workspace_frozen:
            return "SAVED RESULT", "snapshot"
        if artifact_id in self._workspace_pending:
            return "SAVED RESULT", "snapshot"

        if page_id == PAGE_DISTRIBUTIONS:
            has_data = bool(page.sources())
        elif page_id in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES):
            has_data = page.has_data()
        elif page_id in (PAGE_EVENT_WAVEFORMS, PAGE_NOISE):
            has_data = bool(page.has_data())
        elif page_id in (PAGE_IV, PAGE_VOLTAGE_SERIES):
            has_data = bool(page.points())
        elif page_id == PAGE_SUBPOPULATIONS:
            has_data = page.report() is not None
        else:  # pragma: no cover - AnalysisExplorer rejects unknown page ids
            has_data = False
        return ("CURRENT", "current") if has_data else ("NO DATA", "empty")

    def _show_analysis_page(
            self, page_id: str, page: QtWidgets.QWidget, *, refresh=None) -> None:
        """Put an existing analytical widget in the one Explorer and focus it."""
        explorer = self._ensure_analysis_explorer()
        if self._figure_workspace is not None:
            figure_id = self._figure_workspace.id_for_page(page)
            if figure_id is not None:
                self._figure_workspace.activate(figure_id)
                explorer.raise_page(figure_id)
                return
        explorer.show_page(
            page_id, page, refresh=refresh,
            activate=lambda _id=page_id: self._activate_analysis_page(_id),
        )
        self._sync_analysis_open_state()

    def _activate_analysis_page(self, page_id: str) -> None:
        """Refresh a live page reached directly through the Explorer's tab bar.

        This is deliberately different from the shared explicit Refresh button: a tab
        switch follows ordinary live session navigation and never replaces a frozen
        project snapshot.  The callback runs synchronously from ``currentChanged``, before
        Qt returns to the event loop to paint the newly-selected page.
        """
        if self._workspace_installing or self._workspace_navigating or self._restoring:
            return
        artifact_id = self._analysis_page_artifact(page_id)
        if artifact_id in self._workspace_frozen or artifact_id in self._workspace_pending:
            self._refresh_analysis_explorer_header()
            return
        if page_id == PAGE_DISTRIBUTIONS and self._plots_window is not None:
            self._refresh_plots()
        elif page_id == PAGE_DISTRIBUTION_SUMMARIES and self._distribution_summary_window is not None:
            self._offer_distribution_sources()
        elif page_id == PAGE_COMPARISON_FIGURES and self._comparison_figures_window is not None:
            self._offer_comparison_sources()
        elif page_id == PAGE_EVENT_WAVEFORMS and self._event_plot_window is not None:
            self._refresh_event_plots()
        elif page_id == PAGE_NOISE and self._psd_window is not None:
            self._point_psd_at_selection()
        elif page_id == PAGE_IV and self._iv_window is not None:
            self._iv_window.refresh()
        elif page_id == PAGE_VOLTAGE_SERIES and self._vdep_window is not None:
            self._vdep_window.refresh()
        elif page_id == PAGE_SUBPOPULATIONS and self._subpop_window is not None:
            self._sync_subpop_window()
        self._refresh_analysis_explorer_header()

    def _refresh_analysis_explorer_header(self) -> None:
        explorer = self._analysis_explorer
        if explorer is not None:
            explorer.refresh_header()
