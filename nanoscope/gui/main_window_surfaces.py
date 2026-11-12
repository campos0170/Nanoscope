"""The analysis surfaces of ``MainWindow``: the gates that the menu and the launcher share, the
analytical workspace (capture, restore, reset), the Analysis home and the guided first
analysis, and the openers of the Plots, Event plots, study, I-V, voltage-dependence and PSD
windows. ``_on_plot_event_rows_selected`` stays in ``main_window.py``: it is a registered ``@Slot``,
and no ``@Slot`` method leaves the class body (see the comment after the imports there).

``_SurfacesMixin`` is a plain class listed before ``QMainWindow`` in the bases of ``MainWindow``,
with no ``__init__`` and no state of its own. The method bodies were moved verbatim out of
``main_window.py`` by a mechanical split; nothing in them was edited.

``_on_screen`` and ``ask_export_path`` are imported from ``main_window_seams``, whose forwarders
look the names up on ``main_window`` each time they are called, so a test's patch of
``main_window._on_screen`` or ``main_window.ask_export_path`` still reaches these methods.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
from pathlib import Path
from typing import Any

from PySide6 import QtWidgets

from ..core.analysis_workspace import (
    ANALYSIS_PANEL_ARTIFACT_ID,
    DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
    COMPARISON_FIGURES_ARTIFACT_ID,
    EVENT_PLOTS_ARTIFACT_ID,
    IV_ARTIFACT_ID,
    PLOTS_ARTIFACT_ID,
    PSD_ARTIFACT_ID,
    SUBPOPULATION_ARTIFACT_ID,
    VOLTAGE_DEPENDENCE_ARTIFACT_ID,
    capture_analysis_workspace,
    restore_analysis_workspace,
)
from ..core.project import AnalyticalWorkspace
from ..core.types import ChannelKind
from .analysis_launcher import (
    SURFACE_EVENT_PLOTS,
    SURFACE_FIGURES,
    SURFACE_GALLERY,
    SURFACE_IV,
    SURFACE_ORDER,
    SURFACE_POPULATIONS,
    SURFACE_PSD,
    SURFACE_VDEP,
)
from .analysis_explorer import (
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
from ..core import clock
from .trace_view import effective_kind
from .workflow_status import Stage, compute_status
from .main_window_seams import _on_screen, ask_export_path


class _SurfacesMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    # ================================================================================================
    # analysis surfaces: ONE gate per surface, feeding BOTH the menu action and the launcher
    # ================================================================================================
    #
    # Each surface's gate returns ``(available, reason)`` from a single evaluation of the SAME
    # predicate that used to drive ``setEnabled`` alone. Everything the user can see about that
    # surface — the menu item's enablement, the launcher button's greyed/live state, its tooltip
    # and the message a locked click posts — is derived from that one tuple, so they cannot drift
    # into disagreeing about reality. Writing the reason was most of the work here: before this,
    # only one of the seven gates stated its condition anywhere the user could read it.

    def _surface_actions(self) -> dict:
        """Surface id -> the EXISTING ``&Analysis`` QAction that opens it.

        The launcher never duplicates an open/gating code path: it triggers these.
        """
        return {
            SURFACE_POPULATIONS: self._act_plots,
            SURFACE_EVENT_PLOTS: self._act_event_plots,
            SURFACE_PSD: self._act_psd,
            SURFACE_IV: self._act_iv,
            SURFACE_VDEP: self._act_vdep,
            SURFACE_FIGURES: self._act_export_figure,
            SURFACE_GALLERY: self._act_event_gallery,
        }

    def _surface_gates(self) -> dict:
        """Surface id -> the callable returning that surface's ``(available, reason)``."""
        return {
            SURFACE_POPULATIONS: self._gate_populations,
            SURFACE_EVENT_PLOTS: self._gate_event_plots,
            SURFACE_PSD: self._gate_psd,
            SURFACE_IV: self._gate_iv,
            SURFACE_VDEP: self._gate_vdep,
            SURFACE_FIGURES: self._gate_figures,
            SURFACE_GALLERY: self._gate_gallery,
        }

    def _apply_surface_gate(self, sid: str) -> None:
        """Evaluate ``sid``'s gate ONCE and push the result to both places it shows up."""
        available, reason = self._surface_gates()[sid]()
        self._surface_actions()[sid].setEnabled(available)
        if sid == SURFACE_FIGURES:
            self._act_edit_report_figure.setEnabled(available)
        self._analysis_launcher.set_gate(sid, available, reason)

    def _sync_analysis_surfaces(self) -> None:
        """Re-derive every analysis surface's gate. Connected to every signal that can change
        one (a document opens/closes, a detection is stored/cleared, an event is curated, a
        measurement is saved, the selection moves) AND to the Analysis menu's aboutToShow."""
        for sid in SURFACE_ORDER:
            self._apply_surface_gate(sid)
        self._sync_analysis_open_state()
        if self._analysis_explorer is not None:
            self._analysis_explorer.home().refresh_availability()
        if self._first_analysis_guide is not None and self._first_analysis_guide.isVisible():
            self._first_analysis_guide.refresh_context()

    def _gate_populations(self) -> tuple[bool, str]:
        """Populations (the Plots window): needs SOME plottable per-event table — any open
        recording's detection, or the saved manual measurements."""
        if self._plots_sources():
            return True, ""
        if self._workspace_has_artifact(PLOTS_ARTIFACT_ID):
            # A frozen project result remains a readable/exportable result even when its
            # source recording or event table cannot currently be resolved (AP-16).
            return True, ""
        if not self._session.documents():
            # Naming the REAL blocker. "Run detection first" with nothing open sends the user
            # to a detector that is itself disabled for want of a recording.
            return False, ("Open a recording first — Distributions plots per-event "
                           "tables (a detection's events, or a saved manual measurement).")
        return False, ("Run detection first — Distributions plots per-event tables "
                       "(a saved manual measurement counts too).")

    def _gate_event_plots(self) -> tuple[bool, str]:
        """Density map / ensemble: needs the ACTIVE detection to have >=1 accepted event —
        there is nothing to bin or overlay otherwise. Each way to fail is a different problem
        with a different fix (nothing open / nothing selected / no detection / nothing
        accepted), so each gets its own sentence naming the blocker that is actually first."""
        from ..analysis.events import EventSet

        if self._workspace_has_artifact(EVENT_PLOTS_ARTIFACT_ID):
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the density map and the ensemble are "
                           "built from a detection's events.")
        if self._workflow_target()[0] is None:
            return False, ("Select a recording in the Files panel first — the density map and "
                           "the ensemble are built from the selected recording's events.")
        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            return False, ("Run detection on the selected recording first — the density map "
                           "and the ensemble are built from its detected events.")
        if res.payload.accepted.df.shape[0] < 1:
            return False, ("No accepted events yet — accept at least one in the Event "
                           "inspector (or tick it in the event table) to fill the density "
                           "map and the ensemble.")
        return True, ""

    def _gate_psd(self) -> tuple[bool, str]:
        """Power spectrum: needs >=1 open recording with a live current channel and a known
        sample rate. It reads the RAW calibrated trace, so it never waits on a detection."""
        if self._has_psd_recording():
            return True, ""
        if self._workspace_has_artifact(PSD_ARTIFACT_ID):
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the noise spectrum needs a current "
                           "channel with a known sample rate (no detection required).")
        return False, ("No open recording has a current channel with a known sample rate — "
                       "the noise spectrum needs both.")

    def _gate_iv(self) -> tuple[bool, str]:
        """I-V / conductance: needs >=1 open recording with a live current channel AND a known
        applied voltage (a live voltage channel, or a voltage recorded in the file's metadata)."""
        if self._has_iv_data():
            return True, ""
        if self._workspace_has_artifact(IV_ARTIFACT_ID):
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the I-V fit needs a current channel and "
                           "a known applied voltage.")
        return False, ("No open recording has both a current channel and a known applied "
                       "voltage (a voltage channel, or a voltage in the file's metadata).")

    def _gate_vdep(self) -> tuple[bool, str]:
        """Voltage dependence: needs >=2 recordings that each have a known applied voltage AND
        a stored detection — one point is not a trend. The reason carries the live count, from
        the very number the predicate thresholds, so it can never overstate progress."""
        n = self._vdep_ready_count()
        if n >= 2:
            return True, ""
        if self._workspace_has_artifact(VOLTAGE_DEPENDENCE_ARTIFACT_ID):
            return True, ""
        return False, (f"Needs at least 2 detected recordings with a known applied voltage — "
                       f"{n} so far. Open the voltage series and run Analysis ▸ Batch detect.")

    def _gate_figures(self) -> tuple[bool, str]:
        """Export figures: needs a detection on the CURRENT selection — the figures are drawn
        from its events (the same gate the other two exports use). Like the density map, it
        names the blocker that is actually first rather than always blaming detection."""
        if self._analysis_panel.current_detection() is not None:
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the figures are drawn from a detection's "
                           "events.")
        if self._workflow_target()[0] is None:
            return False, ("Select a recording in the Files panel first — the figures are "
                           "drawn from the selected recording's events.")
        return False, ("Run detection on the selected recording first — the figures are drawn "
                       "from its events.")

    def _gate_gallery(self) -> tuple[bool, str]:
        """Event gallery: needs a detection on the CURRENT selection — the contact sheet is a
        tile per detected event.

        This is the SAME ``current_detection() is not None`` predicate ``_sync_export_action``
        used to apply straight to ``_act_event_gallery.setEnabled``; that line is gone, so the
        menu item, the launcher button and the reason a locked click posts now all come from
        this one evaluation. Unlike the density map it does NOT ask for an accepted event —
        the gallery is where a curator goes to decide which events those are.
        """
        if self._analysis_panel.current_detection() is not None:
            return True, ""
        if not self._session.documents():
            return False, ("Open a recording first — the gallery lays out a detection's events "
                           "as a grid of waveform tiles.")
        if self._workflow_target()[0] is None:
            return False, ("Select a recording in the Files panel first — the gallery lays out "
                           "the selected recording's detected events.")
        return False, ("Run detection on the selected recording first — the gallery lays out "
                       "its detected events as a grid of waveform tiles.")

    def _on_surface_activated(self, sid: str) -> None:
        """A launcher button was clicked. An AVAILABLE surface triggers its EXISTING menu action
        (never a parallel open path); a LOCKED one posts its gate reason to the status bar and
        opens nothing — exactly what a locked workflow-strip chip does."""
        available, reason = self._surface_gates()[sid]()
        if not available:
            self.statusBar().showMessage(reason, 6000)
            return
        self._surface_actions()[sid].trigger()
        self._sync_analysis_open_state()

    # ============================================================================================
    # Stage 3 analytical workspace — project-owned, lazily installed
    # ============================================================================================

    def _set_analysis_selection_for_navigation(self, doc_id: str, channel: int) -> None:
        """Retarget the live form without turning navigation into a project edit.

        ``AnalysisPanel.set_selection`` legitimately seeds controls from the stored result
        on the new target, and those public setters emit the same aggregate signals as a
        person's edits.  The File Manager selection itself is navigation (AP-13), so guard
        only this synchronous seed.  Detector/result combo changes made by the user still
        arrive outside the guard and remain durable workspace edits.
        """
        installing = self._workspace_installing
        self._workspace_installing = True
        try:
            self._analysis_panel.set_selection(doc_id, channel)
        finally:
            self._workspace_installing = installing

    @staticmethod
    def _workspace_kwarg(artifact_id: str) -> str | None:
        return {
            PLOTS_ARTIFACT_ID: "plots",
            DISTRIBUTION_SUMMARIES_ARTIFACT_ID: "distribution_summaries",
            COMPARISON_FIGURES_ARTIFACT_ID: "comparison_figures",
            SUBPOPULATION_ARTIFACT_ID: "subpopulation",
            PSD_ARTIFACT_ID: "psd",
            EVENT_PLOTS_ARTIFACT_ID: "event_plots",
            IV_ARTIFACT_ID: "iv",
            VOLTAGE_DEPENDENCE_ARTIFACT_ID: "voltage_dependence",
            ANALYSIS_PANEL_ARTIFACT_ID: "analysis_panel",
        }.get(str(artifact_id))

    def _workspace_window_id(self, window) -> str | None:
        if self._figure_workspace is not None:
            return self._figure_workspace.id_for_page(window)
        for artifact_id, candidate in (
                (PLOTS_ARTIFACT_ID, self._plots_window),
                (DISTRIBUTION_SUMMARIES_ARTIFACT_ID, self._distribution_summary_window),
                (COMPARISON_FIGURES_ARTIFACT_ID, self._comparison_figures_window),
                (SUBPOPULATION_ARTIFACT_ID, self._subpop_window),
                (PSD_ARTIFACT_ID, self._psd_window),
                (EVENT_PLOTS_ARTIFACT_ID, self._event_plot_window),
                (IV_ARTIFACT_ID, self._iv_window),
                (VOLTAGE_DEPENDENCE_ARTIFACT_ID, self._vdep_window)):
            if window is candidate:
                return artifact_id
        return None

    def _workspace_surface(self, artifact_id: str):
        if self._figure_workspace is not None and self._figure_workspace.get(artifact_id) is not None:
            return self._figure_workspace.page(artifact_id)
        return {
            PLOTS_ARTIFACT_ID: self._plots_window,
            DISTRIBUTION_SUMMARIES_ARTIFACT_ID: self._distribution_summary_window,
            COMPARISON_FIGURES_ARTIFACT_ID: self._comparison_figures_window,
            SUBPOPULATION_ARTIFACT_ID: self._subpop_window,
            PSD_ARTIFACT_ID: self._psd_window,
            EVENT_PLOTS_ARTIFACT_ID: self._event_plot_window,
            IV_ARTIFACT_ID: self._iv_window,
            VOLTAGE_DEPENDENCE_ARTIFACT_ID: self._vdep_window,
            ANALYSIS_PANEL_ARTIFACT_ID: self._analysis_panel,
        }.get(str(artifact_id))

    def _workspace_artifact(self, artifact_id: str):
        if self._figure_workspace is not None:
            from ..core.figure_codecs import NATIVE_KINDS, encode_legacy, record_to_artifact
            record = self._figure_workspace.get(artifact_id)
            if record is not None:
                return record_to_artifact(record)
            if artifact_id in NATIVE_KINDS:
                key = self._figure_workspace.most_recent(NATIVE_KINDS[artifact_id][1])
                record = self._figure_workspace.get(key)
                if record is not None:
                    return encode_legacy(record.spec, record.resources)
        workspace = self._analytical_workspace
        return next((item for item in workspace.artifacts
                     if item.artifact_id == artifact_id), None)

    def _workspace_has_artifact(self, artifact_id: str) -> bool:
        return self._workspace_artifact(artifact_id) is not None

    def _drop_workspace_artifact(self, artifact_id: str) -> None:
        """Forget one invalidated built-in result without touching opaque siblings.

        A live surface can cease to have a result when one of its scientific inputs is
        edited (for example, re-declaring the only current channel as voltage empties the
        PSD window).  Keeping its last checkpoint would make the corresponding launcher
        gate claim that a readable saved result still exists.  This is deliberately a
        single-artifact operation: plugin blocks, ordering, workspace metadata and every
        other built-in result survive byte-for-byte.
        """
        workspace = self._analytical_workspace
        kept = [item for item in workspace.artifacts
                if item.artifact_id != str(artifact_id)]
        if len(kept) == len(workspace.artifacts):
            return
        self._analytical_workspace = AnalyticalWorkspace(
            version=workspace.version,
            artifacts=kept,
            extra=workspace.extra,
        )
        self._workspace_pending.discard(str(artifact_id))
        self._workspace_frozen.discard(str(artifact_id))
        self._sync_saved_subpop_action()

    def _sync_saved_subpop_action(self) -> None:
        action = getattr(self, "_act_show_saved_subpop", None)
        if action is not None:
            action.setEnabled(self._workspace_has_artifact(SUBPOPULATION_ARTIFACT_ID))

    def _watch_workspace_surface(self, artifact_id: str, window) -> None:
        """Wire one lazily-created surface without making it a project writer."""
        signal = getattr(window, "workspaceChanged", None)
        if signal is not None:
            signal.connect(
                lambda *_args, _id=artifact_id: self._on_workspace_changed(_id))
        checkpoint = getattr(window, "workspaceCheckpointRequested", None)
        if checkpoint is not None:
            checkpoint.connect(
                lambda *_args, _id=artifact_id: self._checkpoint_workspace_surface(_id))

        # Subpopulation view state predates the common signal.  Its two choices and
        # splitter sizes are still durable; user-only combo signals avoid treating adapter
        # replay as an edit, and the install guard covers splitter replay.
        if artifact_id == SUBPOPULATION_ARTIFACT_ID:
            view = window.embedding_view()
            for combo in (getattr(view, "_embed_combo", None),
                          getattr(view, "_colour_combo", None)):
                activated = getattr(combo, "activated", None)
                if activated is not None:
                    activated.connect(
                        lambda *_args: self._on_workspace_changed(
                            SUBPOPULATION_ARTIFACT_ID))
            window.splitter().splitterMoved.connect(
                lambda *_: self._on_workspace_changed(SUBPOPULATION_ARTIFACT_ID))
            window.figures_splitter().splitterMoved.connect(
                lambda *_: self._on_workspace_changed(SUBPOPULATION_ARTIFACT_ID))

    def _checkpoint_workspace_surface(
            self, artifact_id: str, *, capture_frozen: bool = False) -> bool:
        """Synchronously copy one surface into the project-owned cache.

        ``PlotsWindow.closeEvent`` reaches this seam before deliberately resetting itself.
        Closing a window is not project data, so the checkpoint itself is silent; every
        durable control/result already reaches :meth:`_on_workspace_changed`.  A failed
        checkpoint is different: the cache may now be incomplete, so keep the project
        visibly dirty and let Save Project retry/report it.
        """
        if self._workspace_installing or self._workspace_navigating or self._restoring:
            return False
        # An untouched restored snapshot already lives in the cache.  Re-reading its
        # widget at close could restamp it against a ResultStore that changed underneath
        # while frozen; the later Hide event merely makes that same artifact pending for
        # the next lazy install.
        if artifact_id in self._workspace_frozen and not capture_frozen:
            return True
        surface = self._workspace_surface(artifact_id)
        keyword = self._workspace_kwarg(artifact_id)
        if surface is None or keyword is None:
            return False
        try:
            self._analytical_workspace = capture_analysis_workspace(
                self._analytical_workspace, **{keyword: surface})
        except Exception as exc:  # noqa: BLE001 - the save path retries and reports fully
            self._mark_dirty()
            self.statusBar().showMessage(
                f"Could not checkpoint this analytical view yet: {exc}. "
                "Save Project will try again.", 12000)
            return False
        self._workspace_pending.discard(artifact_id)
        # Keep an installed project snapshot insulated from session-navigation refreshes
        # when this was merely a durable style/config edit.  Explicit Refresh routes
        # through ``_replace_workspace_surface`` and clears the bit *before* computing;
        # generic ``workspaceChanged`` cannot tell a title colour from a numerical run and
        # therefore must take the lossless/conservative side of that boundary.
        self._sync_saved_subpop_action()
        return True

    def _on_workspace_changed(self, artifact_id: str) -> None:
        """Checkpoint one durable edit/result and advance the recovery revision."""
        if not self._checkpoint_workspace_surface(artifact_id, capture_frozen=True):
            return
        self._mark_dirty()

    def _capture_analytical_workspace(self) -> AnalyticalWorkspace:
        """Capture safe live surfaces for Save/Recovery, preserving every carried block.

        Hidden PlotsWindow is intentionally excluded: its close contract blanked the
        widget after ``_on_workspace_changed`` cached the real result, and gathering every
        instantiated window here would overwrite that cache with defaults.  A pending
        block is likewise excluded until successfully installed or explicitly edited, so
        an unavailable input cannot be converted into an empty artifact by a later save.
        A frozen restored surface is already represented by that authoritative artifact;
        leaving it out prevents a changed live ResultStore from silently restamping an old
        snapshot merely because the user pressed Save while its window was visible.
        """
        if self._figure_workspace is not None:
            if self._figure_workspace.is_read_only():
                raise ValueError("The analytical workspace was written by a newer NanoScope build. "
                                 "Use that build to save it; the original project has not been changed.")
            supplied = ({"analysis_panel": self._analysis_panel}
                        if ANALYSIS_PANEL_ARTIFACT_ID not in self._workspace_pending else {})
            base = capture_analysis_workspace(self._analytical_workspace, **supplied)
            self._analytical_workspace = self._figure_workspace.capture_workspace(base)
            return self._analytical_workspace
        supplied: dict[str, Any] = {}
        if ANALYSIS_PANEL_ARTIFACT_ID not in self._workspace_pending:
            supplied["analysis_panel"] = self._analysis_panel
        for artifact_id in (
                PLOTS_ARTIFACT_ID, SUBPOPULATION_ARTIFACT_ID, PSD_ARTIFACT_ID,
                DISTRIBUTION_SUMMARIES_ARTIFACT_ID,
                COMPARISON_FIGURES_ARTIFACT_ID,
                EVENT_PLOTS_ARTIFACT_ID, IV_ARTIFACT_ID,
                VOLTAGE_DEPENDENCE_ARTIFACT_ID):
            window = self._workspace_surface(artifact_id)
            # A live PSD whose last eligible current channel disappeared has no result
            # to save.  Its stale checkpoint is removed by ``_refresh_surface_windows``;
            # do not recreate an empty artifact merely because the still-visible window
            # is included in a later Save/Recovery gather.  Frozen saved snapshots take
            # the earlier exclusion branch and remain readable without their source.
            if (artifact_id == PSD_ARTIFACT_ID and window is not None
                    and not window.has_data()):
                continue
            if (artifact_id not in self._workspace_pending
                    and artifact_id not in self._workspace_frozen
                    and _on_screen(window)):
                supplied[self._workspace_kwarg(artifact_id)] = window
        self._analytical_workspace = capture_analysis_workspace(
            self._analytical_workspace, **supplied)
        self._sync_saved_subpop_action()
        return self._analytical_workspace

    def _restore_workspace_surface(self, artifact_id: str, surface) -> bool:
        """Install one pending artifact into an already-built/bound surface, once."""
        if artifact_id not in self._workspace_pending:
            return False
        keyword = self._workspace_kwarg(artifact_id)
        if keyword is None or self._workspace_artifact(artifact_id) is None:
            self._workspace_pending.discard(artifact_id)
            return False
        self._workspace_installing = True
        try:
            report = restore_analysis_workspace(
                self._analytical_workspace, **{keyword: surface})
        finally:
            self._workspace_installing = False
        restored = artifact_id in report.restored
        if restored:
            self._workspace_pending.discard(artifact_id)
            self._workspace_frozen.add(artifact_id)
        if report.warnings:
            sentence = " ".join(str(value) for value in report.warnings)
            if self._restoring:
                self._restore_notes.append(sentence)
            else:
                self.statusBar().showMessage(
                    f"Saved analytical view restored with a caveat: {sentence}", 15000)
        return restored

    def _replace_workspace_surface(self, artifact_id: str, refresh) -> None:
        """Explicit Refresh: leave frozen mode, recompute, then checkpoint the new result."""
        self._workspace_pending.discard(artifact_id)
        self._workspace_frozen.discard(artifact_id)
        refresh()
        self._on_workspace_changed(artifact_id)

    def _reset_analytical_workspace(self) -> None:
        """Drop the outgoing project's analytical state without emitting an edit.

        Plots and the subpopulation report predate the project-owned workspace and have a
        useful established contract: if the user leaves either visible while opening a
        project, that same window stays visible and is re-fed at the end of restore.  Keep
        those two *widget shells* but scrub every source, result and identity sidecar.
        Numerical inspectors are rebuilt lazily because their private arrays have no
        comparable blank-state contract.  The caller has already completed the outgoing
        project's Save/Discard decision, so none of this checkpoints or marks dirty.
        """
        self._on_umap_stopped()
        self._figure_project_epoch += 1
        if self._figure_workspace is not None:
            self._figure_workspace.clear()
            for name in self._figure_aliases().values():
                setattr(self, name, None)
        self._subpop_figure_request = None
        self._level_fit_pending = None
        self._workspace_installing = True
        try:
            if self._plots_window is not None:
                abort = getattr(self._plots_window, "_abort_note_job", None)
                if callable(abort):
                    abort()
                self._plots_window.reset_state()
                for sidecar in ("_nanoscope_unresolved_plot_sources",
                                "_nanoscope_unresolved_plot_results"):
                    if hasattr(self._plots_window, sidecar):
                        delattr(self._plots_window, sidecar)
            if self._subpop_window is not None:
                self._subpop_window.set_report(None)
                # The words typed on the outgoing project's figures, the ranges pinned on them
                # and the look are that project's: the incoming project's first report starts
                # from the defaults, as a restore with no saved appearance does
                # (``_restore_appearance`` hands the window ``{}``).  ``set_report(None)`` ends
                # the appearance HISTORY only; Distributions' ``reset_state`` does the rest.
                self._subpop_window.set_appearance_state({})
                # These are constructor defaults.  Splitter positions are durable view
                # state, so an incoming project without a saved report must not inherit
                # the outgoing project's arrangement.
                self._subpop_window.splitter().setSizes([620, 800])
                self._subpop_window.figures_splitter().setSizes([400, 320, 190])
                for sidecar in ("_nanoscope_workspace_source",
                                "_nanoscope_workspace_source_result",
                                "_nanoscope_workspace_accepted_event_ids"):
                    if hasattr(self._subpop_window, sidecar):
                        delattr(self._subpop_window, sidecar)
            reset_pages = {
                "_distribution_summary_window": PAGE_DISTRIBUTION_SUMMARIES,
                "_comparison_figures_window": PAGE_COMPARISON_FIGURES,
                "_psd_window": PAGE_NOISE,
                "_event_plot_window": PAGE_EVENT_WAVEFORMS,
                "_iv_window": PAGE_IV,
                "_vdep_window": PAGE_VOLTAGE_SERIES,
            }
            for name, page_id in reset_pages.items():
                window = getattr(self, name, None)
                if window is None:
                    continue
                if self._analysis_explorer is not None:
                    self._analysis_explorer.remove_page(page_id)
                try:
                    # Same protocol _stop_holders uses at quit, and the same one
                    # _plots_window's _abort_note_job is given above: a window that
                    # can hold a background worker (IVWindow._gather_worker, B22,
                    # 2026-09-13) must be asked to retire it BEFORE deleteLater()
                    # schedules the C++ object's destruction. Skipping this step
                    # would leave the worker's QThread running with nothing left to
                    # deliver its `done` signal to -- exactly the class of hazard
                    # app.py's own docstring documents at length for a worker that
                    # outlives what it reports back to.
                    abort = getattr(window, "abort", None)
                    if callable(abort):
                        abort()
                    window.hide()
                    window.deleteLater()
                except RuntimeError:
                    pass
                setattr(self, name, None)
            # If the outgoing project had only disposable numerical pages open, do not
            # leave an empty Analysis Explorer floating over the newly reset workspace.
            # The shell object is retained and will reopen with the next lazy page.
            if (self._analysis_explorer is not None
                    and self._analysis_explorer.page_count() == 0):
                self._analysis_explorer.close()
            self._drop_subpop_assignment()
            self._subpop_pending = False
            self._subpop_source_recipe_pending = None
            self._subpop_signal_pending = None
            self._subpop_accepted_ids_pending = None
            self._analytical_workspace = AnalyticalWorkspace()
            if self._figure_library is not None:
                self._figure_library.refresh()
            self._workspace_pending.clear()
            self._workspace_frozen.clear()

            # An old project has no analysis-panel artifact.  Replay the constructor
            # defaults now so its absence means EMPTY, not "whatever the last project
            # happened to leave in the always-live panel".
            default = self._workspace_default_analysis
            if default is not None:
                with self._analysis_panel._writing_form():
                    restore_analysis_workspace(
                        AnalyticalWorkspace(artifacts=[default]),
                        analysis_panel=self._analysis_panel)
        finally:
            self._workspace_installing = False
        self._sync_saved_subpop_action()
        self._sync_analysis_open_state()

    # -- "this window is already open" indication ------------------------------------------------
    def _analysis_home_availability(self, route: str) -> tuple[bool, str]:
        """Read-only navigation gates; inspecting the catalog never constructs a result."""
        gates = {
            PAGE_DISTRIBUTIONS: self._gate_populations,
            PAGE_EVENT_WAVEFORMS: self._gate_event_plots,
            PAGE_NOISE: self._gate_psd,
            PAGE_IV: self._gate_iv,
            PAGE_VOLTAGE_SERIES: self._gate_vdep,
            "event_gallery": self._gate_gallery,
            "report_figures": self._gate_figures,
        }
        if route in gates:
            return gates[route]()
        if route in (PAGE_DISTRIBUTION_SUMMARIES, PAGE_COMPARISON_FIGURES):
            return True, "Open the controls; choose eligible source tables before generating a figure."
        if route == PAGE_SUBPOPULATIONS:
            return True, "Open an existing report or an empty view. Generate explicitly to run an analysis."
        if route in (PAGE_HOME, "figure_library", "guided_first_analysis"):
            return True, ""
        return False, "Unknown analysis destination."

    def _launch_analysis_route(self, route: str) -> None:
        available, reason = self._analysis_home_availability(route)
        if not available:
            # Said in the Explorer, where the tool was chosen: this window's status bar is
            # behind it, so the reason used to land where nobody was looking.
            self._ensure_analysis_explorer().show_refusal(route, reason)
            return
        actions = {
            PAGE_HOME: self._open_analysis_home,
            "guided_first_analysis": self._open_first_analysis_guide,
            PAGE_DISTRIBUTIONS: self._open_plots,
            PAGE_DISTRIBUTION_SUMMARIES: self._open_distribution_summaries,
            PAGE_COMPARISON_FIGURES: self._open_comparison_figures,
            PAGE_EVENT_WAVEFORMS: self._open_event_plots,
            PAGE_NOISE: self._open_psd,
            PAGE_IV: self._open_iv,
            PAGE_VOLTAGE_SERIES: self._open_vdep,
            PAGE_SUBPOPULATIONS: self._open_subpopulations_view,
            "event_gallery": self._open_event_gallery,
            "figure_library": self._open_figure_library,
            "report_figures": self._export_figures,
        }
        actions[route]()

    def _open_subpopulations_view(self) -> None:
        """Opening a destination is not permission to run a mixture analysis."""
        return self._open_figure_kind(PAGE_SUBPOPULATIONS)

    def _open_analysis_home(self) -> None:
        self._ensure_analysis_explorer().show_home()

    def _first_analysis_context(self):
        return self._workflow_target(), self._analysis_explorer_context(for_figure=False)

    def _first_analysis_availability(self, route: str) -> tuple[bool, str]:
        if route in ("open_recording", "practice", "analysis_current", "save_project"):
            return True, ""
        stages = {"calibration": Stage.CALIBRATE, "baseline": Stage.FILTER,
                  "detection": Stage.DETECT, "essential": Stage.DETECT,
                  "advanced": Stage.DETECT, "event_inspector": Stage.CURATE}
        if route in stages:
            status = compute_status(self._session, *self._workflow_target()).stage(stages[route])
            return status.available, status.reason
        return self._analysis_home_availability(route)

    def _open_first_analysis_guide(self) -> None:
        from .first_analysis_guide import FirstAnalysisGuide
        if self._first_analysis_guide is None:
            self._first_analysis_guide = FirstAnalysisGuide(
                self, context_provider=self._first_analysis_context,
                availability_provider=self._first_analysis_availability)
            self._first_analysis_guide.actionRequested.connect(self._first_analysis_action)
            self._first_analysis_guide.controlLevelRequested.connect(
                lambda level: self._analysis_panel.detection_panel.set_control_level(level, persist=True))
            self._adopt_child_window(self._first_analysis_guide, "first_analysis_guide")
        self._first_analysis_guide.refresh_context()
        self._first_analysis_guide.show()
        self._first_analysis_guide.raise_()
        self._first_analysis_guide.activateWindow()

    def _open_build_info(self) -> None:
        from .build_info_dialog import BuildInfoDialog
        if self._build_info_dialog is None:
            self._build_info_dialog = BuildInfoDialog(self)
            self._adopt_child_window(self._build_info_dialog, "about")
        self._build_info_dialog.show()
        self._build_info_dialog.raise_()
        self._build_info_dialog.activateWindow()

    def _first_analysis_action(self, route: str) -> None:
        available, reason = self._first_analysis_availability(route)
        if not available:
            self.statusBar().showMessage(reason, 8000)
            return
        if route == "open_recording":
            self._open_dialog()
        elif route == "practice":
            self._create_practice_recording()
        elif route == "calibration":
            self._reveal_calibration_panel()
        elif route in ("baseline", "detection", "essential", "advanced"):
            self._reveal_detection_panel()
            panel = self._analysis_panel.detection_panel
            if route == "baseline":
                panel.show_open_pore_baseline()
            else:
                panel.show_detection_tab()
                if route in ("essential", "advanced"):
                    panel.set_control_level(route, persist=False)
        elif route == "event_inspector":
            self._open_event_inspector()
        elif route == "analysis_current":
            if self._analysis_explorer is None:
                self._open_analysis_home()
            else:
                self._analysis_explorer.show()
                self._analysis_explorer.raise_()
        elif route == "save_project":
            self._save_project_from(self._first_analysis_guide or self, save_as=False)
        else:
            self._launch_analysis_route(route)

    def _create_practice_recording(self) -> None:
        from .first_analysis_guide import practice_path, write_practice_recording
        parent = self._first_analysis_guide or self
        answer = QtWidgets.QMessageBox.question(
            parent, "Create synthetic practice recording",
            "Create synthetic learning data and add it alongside your open recordings? "
            "This is not experimental data. No analysis settings will be changed. "
            "The saved filename will include SYNTHETIC.",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer != QtWidgets.QMessageBox.StandardButton.Ok:
            return
        chosen = ask_export_path(parent, title="Save synthetic practice recording",
            suggestion="SYNTHETIC-NanoScope-practice.csv", filters="CSV files (*.csv)",
            outputs=lambda path: [practice_path(path)])
        if chosen is None:
            return
        target = practice_path(chosen)
        # "Is this file already open?" is Session.doc_id_for_source, not a raw path
        # comparison (core/identity.py, core/session.py): it compares through
        # same_source rather than `==`, which folds `sub/../` and symlinks but not
        # case, so two spellings of one path on a case-insensitive volume would
        # otherwise both pass this guard and load the same file twice.
        if self._session.doc_id_for_source(str(target.resolve())) is not None:
            QtWidgets.QMessageBox.warning(parent, "Recording is already open",
                "Choose a new filename. A practice recording cannot replace a file currently open in this project.")
            return
        try:
            target = write_practice_recording(target)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(parent, "Practice recording could not be saved", str(exc))
            return
        self.open_paths([target])

    def _sync_plots_action(self) -> None:
        """The Plots action is live whenever there is SOME plottable data — any open
        recording's detection OR any saved manual measurements. Routed through the shared
        surface gate so the action and the launcher button read the same predicate once."""
        self._apply_surface_gate(SURFACE_POPULATIONS)

    def _open_distribution_summaries(self) -> None:
        """Optional new statistics are lazy and run only through explicit Generate."""
        return self._open_figure_kind(PAGE_DISTRIBUTION_SUMMARIES)

    def _generate_distribution_summary(self) -> None:
        page = self._distribution_summary_window
        if page is None:
            return
        if self._figure_workspace is not None:
            figure_id = self._figure_workspace.id_for_page(page)
            if figure_id is not None:
                return self._refresh_figure(figure_id)
        self._prepare_distribution_summary()
        try:
            page.analyze()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(page, "Generate distribution summary", str(exc))

    def _prepare_distribution_summary(self) -> None:
        """Generate intent leaves frozen mode before binding today's event tables."""
        self._workspace_pending.discard(DISTRIBUTION_SUMMARIES_ARTIFACT_ID)
        self._workspace_frozen.discard(DISTRIBUTION_SUMMARIES_ARTIFACT_ID)
        page = self._distribution_summary_window
        if page is not None and page.has_data():
            # A failed explicit generation must not relabel an old, frozen result
            # CURRENT just because today's offered bindings happened to be unchanged.
            page._stale = True
            page._update_status()
        self._offer_distribution_sources()
        self._refresh_analysis_explorer_header()

    def _offer_distribution_sources(self) -> None:
        """Passive rebinding is not a project edit and must not overwrite pending state."""
        if self._distribution_summary_window is None:
            return
        installing = self._workspace_installing
        self._workspace_installing = True
        try:
            self._distribution_summary_window.set_sources(self._plots_sources())
        finally:
            self._workspace_installing = installing

    def _open_comparison_figures(self) -> None:
        """Open the optional publication comparisons without regenerating a result."""
        return self._open_figure_kind(PAGE_COMPARISON_FIGURES)

    def _generate_comparison_figures(self) -> None:
        page = self._comparison_figures_window
        if page is None:
            return
        if self._figure_workspace is not None:
            figure_id = self._figure_workspace.id_for_page(page)
            if figure_id is not None:
                return self._refresh_figure(figure_id)
        self._prepare_comparison_figures()
        try:
            page.analyze()
        except (TypeError, ValueError, OverflowError, ImportError, RuntimeError) as exc:
            QtWidgets.QMessageBox.warning(page, "Generate comparison figures", str(exc))

    def _prepare_comparison_figures(self) -> None:
        """Only explicit generation replaces a restored figure with today's inputs."""
        self._workspace_pending.discard(COMPARISON_FIGURES_ARTIFACT_ID)
        self._workspace_frozen.discard(COMPARISON_FIGURES_ARTIFACT_ID)
        page = self._comparison_figures_window
        if page is not None and page.has_data():
            # Failed generation must leave the previous picture visibly stale.
            page._stale = True
            page._update_status()
        self._offer_comparison_sources()
        self._refresh_analysis_explorer_header()

    def _offer_comparison_sources(self) -> None:
        """Offer current bindings silently, preserving pending saved figure state."""
        if self._comparison_figures_window is None:
            return
        installing = self._workspace_installing
        self._workspace_installing = True
        try:
            self._comparison_figures_window.set_sources(self._plots_sources(), analyze=False)
        finally:
            self._workspace_installing = installing

    def _open_plots(self) -> None:
        """Open the Distributions page of the single Analysis Explorer."""
        return self._open_figure_kind(PAGE_DISTRIBUTIONS)

    def _refresh_plots(self) -> None:
        """(Re-)push the current snapshot into the open PlotsWindow — the target of
        both the initial open and the window's own Refresh button. Each source's
        detection provenance travels WITH its frame so a per-source dwell fit uses
        that detector's own min_dwell_s, exactly as the Fits panel does.

        The ACTIVE DETECTION is pushed first and separately, because the window has to
        answer "which of these is the table the user is looking at" before it decides
        which row to tick — and a source list can carry two detections of one channel.
        ``set_active_detection`` before ``set_data``, in that order, so the tick is
        decided from this answer and never from the stale one (see
        ``PlotsWindow._active_source_key``, which otherwise has to reach back in here)."""
        if self._plots_window is not None:
            self._plots_window.set_active_detection(self.active_detection_key())
            self._plots_window.set_data(self._plots_sources())

    # -- event-plots window (density / ensemble) -------------------------------------------------
    def _event_plots_ready(self) -> bool:
        """True when the active detection has at least one ACCEPTED event — the gate for
        the density / ensemble window (there is nothing to bin or overlay otherwise).

        Reads the answer off :meth:`_gate_event_plots` rather than re-deriving it, so the
        boolean and the reason the user is shown can never come apart."""
        return self._gate_event_plots()[0]

    def _sync_event_plots_action(self) -> None:
        self._apply_surface_gate(SURFACE_EVENT_PLOTS)

    def _open_event_plots(self) -> None:
        """Open the Event waveforms page of the single Analysis Explorer."""
        return self._open_figure_kind(PAGE_EVENT_WAVEFORMS)

    def _refresh_event_plots(self) -> None:
        """(Re-)push the active detection (doc/channel + EventSet) into the open EventPlotWindow,
        or clear it when the selection has no detection. A no-op until the window is built."""
        self._feed_event_plots(defer_analysis=False)

    def _bind_event_plots_for_restore(self) -> None:
        """Bind live source identity before restoring frozen arrays, running no kernel."""
        self._feed_event_plots(defer_analysis=True)

    def _feed_event_plots(self, *, defer_analysis: bool) -> None:
        """Install the current event source with either compute or bind-only semantics."""
        if self._event_plot_window is None:
            return
        from ..analysis.events import EventSet

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            if defer_analysis:
                self._event_plot_window.set_source(
                    None, None, None, defer_analysis=True)
            else:
                self._event_plot_window.set_source(None, None, None)
            return
        if defer_analysis:
            self._event_plot_window.set_source(
                res.doc_id, res.channel, res.payload, defer_analysis=True)
        else:
            self._event_plot_window.set_source(res.doc_id, res.channel, res.payload)

    def _open_incubation_study(self) -> None:
        """One entry for new, saved and currently open incubation studies."""
        from .incubation_launcher import IncubationStudyLauncher

        if self._incubation_study_window is None:
            self._incubation_study_window = IncubationStudyLauncher(self)
            self._adopt_child_window(self._incubation_study_window, "incubation_launcher")
        self._incubation_study_window.show()
        self._incubation_study_window.raise_()
        self._incubation_study_window.activateWindow()
        return self._incubation_study_window

    def _open_incubation_reference(self) -> None:
        """Keep the fixed original Zn recipe available as an explicit reference."""
        from .incubation_window import IncubationStudyWindow
        if self._incubation_reference_window is None:
            self._incubation_reference_window = IncubationStudyWindow(self)
            self._incubation_reference_window.setWindowTitle('Original Zn reference recipe — NanoScope')
            self._adopt_child_window(self._incubation_reference_window, "incubation_reference")
        self._incubation_reference_window.show()
        self._incubation_reference_window.raise_()
        self._incubation_reference_window.activateWindow()
        return self._incubation_reference_window

    def _open_manual_study(self) -> None:
        """The Zn reference recipe's one remaining fallback link (:mod:`incubation_window`).
        Uses its OWN cache (:attr:`_manual_study_shortcut`), never the launcher's alias
        (:attr:`_manual_study_window`) — sharing that slot let the launcher silently
        repoint this shortcut at an unrelated, already-open study (or vice versa)."""
        from .manual_study_window import ManualStudyWindow

        if self._manual_study_shortcut is None:
            self._manual_study_shortcut = ManualStudyWindow(self)
            self._adopt_child_window(self._manual_study_shortcut, "manual_study")
        self._manual_study_shortcut.show()
        self._manual_study_shortcut.raise_()
        self._manual_study_shortcut.activateWindow()

    def open_study_review(self, payload, launch_doc_ids=None):
        """Open an independent prepared study source in the shared analysis tools."""
        from .study_event_bridge import open_review
        return open_review(self, payload, launch_doc_ids)

    def snapshot_study_events(self, record):
        """Freeze the active compatible detections for an explicit study commit."""
        from .study_event_bridge import snapshot_events
        return snapshot_events(self, record)

    def begin_study_curation(self, study, tag, doc_id, *, fresh=False):
        """Keep the exact study and recording beside the app's shared event tools."""
        row = next(r for r in study.state['records'] if r['tag'] == tag)
        previous = self._study_curation_sessions.get(doc_id)
        context = dict(study=study, tag=tag, doc_id=doc_id, session=self._session,
            project_generation=self._study_curation_project_generation,
            recording_id=row['recording_id'], folder=Path(study.folder),
            path=row['main_workspace_bridge']['path'])
        if previous:
            for key in ('message', 'saved_signature', 'module_id', 'error'):
                if key in previous: context[key] = previous[key]
        elif fresh:
            context['saved_signature'] = self._study_curation_signature(self._analysis_panel.current_detection())
        # No prior in-session record AND fresh=False (the fast-reuse path, re-attaching
        # to an already-open bridged document) deliberately leaves 'saved_signature'
        # unset here, not because that path should read as "unsaved forever", but
        # because there is no persisted way to know whether that document's live table
        # already diverged from the study's last save before this session's tracking
        # existed -- test_adopting_existing_project_review_discloses_pending_curation
        # requires exactly this: an edit made before adoption must still surface as
        # pending, not read as "up to date" just because nothing was tracked yet.
        self._study_curation_sessions[doc_id] = context
        self._study_curation_context = context
        self._sync_study_curation_bar()
        self.statusBar().showMessage(
            'Curate '+tag+' with the event table, Gallery and Inspector. Use Save curation to study above to keep these edits in the incubation study.', 20000)

    @staticmethod
    def _study_curation_signature(result):
        from ..analysis.events import EventSet
        if result is None or not isinstance(result.payload, EventSet): return None
        from .study_event_bridge import catalog_signature
        return result.module_id, catalog_signature({'candidates': result.payload.df})

    def _sync_study_curation_bar(self, *_):
        context = self._study_curation_context
        selected = getattr(self, '_current_selection', None)
        if selected and selected[0] in self._study_curation_sessions:
            context = self._study_curation_sessions[selected[0]]
            self._study_curation_context = context
        self._study_curation_bar.setVisible(context is not None)
        if context is None: return
        study = context['study']
        try:
            study._curation_record(context, verify_current=True)
            valid = True
        except (ValueError, RuntimeError):
            valid = False
        available = (context['session'] is self._session
            and context['project_generation'] == self._study_curation_project_generation
            and self._session.has_document(context['doc_id']))
        result = self._analysis_panel.current_detection()
        active = available and result is not None and result.doc_id == context['doc_id']
        if active: context['module_id'] = result.module_id
        bound = (result if active else self._session.results.get(context['doc_id'], 0, context.get('module_id', '')))
        context['pending'] = available and self._study_curation_signature(bound) != context.get('saved_signature')
        busy = study.is_busy()
        if not valid:
            message = 'The study changed. Return to study and open its current events.'
        elif busy:
            message = study._operation_label+'…'
        elif study._curation_closed:
            message = 'Study closed. Return to study to reopen it before saving.'
        elif not available:
            message = 'Review recording closed. Return to study to open its events again.'
        elif not active:
            message = 'Another recording is selected. Show study events to continue this curation.'
        elif context.get('error'):
            message = context['error']
        elif context['pending']:
            message = 'Unsaved study curation. Save curation to study includes these edits in the study; project Save keeps the main workspace.'
        else:
            message = context.get('message', 'Use the event table, Gallery and Inspector; save edits to the study here.')
        self._study_curation_label.setText('Incubation study · '+context['tag']+'\n'+message)
        self._study_curation_save.setEnabled(valid and active and not busy and not study._curation_closed)
        self._study_curation_show.setVisible(not active and available and valid)
        self._study_curation_show.setEnabled(not busy)
        self._study_curation_return.setEnabled(True)

    def _study_curation_action(self, action):
        try:
            return action()
        except (ValueError, RuntimeError) as exc:
            QtWidgets.QMessageBox.warning(self, 'Study curation', str(exc))
            return False

    def show_study_curation(self):
        context = self._study_curation_context
        if context is None: return False
        context['study']._curation_record(context)
        if (context['session'] is not self._session
                or context['project_generation'] != self._study_curation_project_generation
                or not self._session.has_document(context['doc_id'])):
            raise ValueError('Return to the study and open this recording’s events again.')
        from .study_event_bridge import _assert_idle
        _assert_idle(self)
        self._on_selection(context['doc_id'], 0)
        self.show(); self.raise_(); self.activateWindow(); self._open_event_gallery()
        self._sync_study_curation_bar()
        return True

    def _save_study_curation(self):
        context = self._study_curation_context
        if context is None: return False
        return context['study'].save_main_curation(context)

    def _return_to_study(self):
        context = self._study_curation_context
        if context is None: return False
        study = context['study']
        try:
            study.return_to_curation_record(context)
        except ValueError:
            # A replaced study can still be brought forward; it must never be
            # silently switched back to an obsolete folder or recording.
            study.show(); study.raise_(); study.activateWindow()
        self._sync_study_curation_bar()
        return True

    # -- I-V / conductance window ----------------------------------------------------------------
    def _has_iv_data(self) -> bool:
        """True when >=1 open recording could contribute an I-V point — a live current channel plus
        a known voltage (a live voltage channel or a nominal_voltage_mV). Mirrors the conditions in
        analysis.iv.iv_point, but reads META ONLY so gating the menu never computes a median."""
        for doc in self._session.documents():
            meta = doc.meta
            # effective_kind on both tests: the Calibration-panel override moves a
            # channel's kind for the panes and the overlay grouping, so the menu gates
            # must follow it — a corrected recording that draws as current must not
            # stay greyed out here. Still meta+calibration only; nothing is computed.
            has_current = any(effective_kind(self._session, doc.doc_id, c.index)
                              is ChannelKind.CURRENT and not c.is_dead
                              for c in meta.channels)
            has_voltage = (any(effective_kind(self._session, doc.doc_id, c.index)
                               is ChannelKind.VOLTAGE and not c.is_dead
                               for c in meta.channels)
                           or meta.nominal_voltage_mV is not None)
            if has_current and has_voltage:
                return True
        return False

    def _sync_iv_action(self) -> None:
        self._apply_surface_gate(SURFACE_IV)

    def _open_iv(self) -> None:
        """Open the I-V page of the single Analysis Explorer."""
        return self._open_figure_kind(PAGE_IV)

    # -- voltage-dependence window ---------------------------------------------------------------
    def _vdep_ready_count(self) -> int:
        """How many open recordings could place on the voltage-dependence plot: a live
        current channel + a known voltage (like I-V) AND a stored detection of the series' detector.
        Reads META + the ResultStore only, so gating never computes a metric."""
        from ..analysis import voltage_dep
        from ..analysis.events import EventSet

        placeable = []
        for doc in self._session.documents():
            meta = doc.meta
            # effective_kind on both tests: the Calibration-panel override moves a
            # channel's kind for the panes and the overlay grouping, so the menu gates
            # must follow it — a corrected recording that draws as current must not
            # stay greyed out here. Still meta+calibration only; nothing is computed.
            has_current = any(effective_kind(self._session, doc.doc_id, c.index)
                              is ChannelKind.CURRENT and not c.is_dead
                              for c in meta.channels)
            has_voltage = (any(effective_kind(self._session, doc.doc_id, c.index)
                               is ChannelKind.VOLTAGE and not c.is_dead
                               for c in meta.channels)
                           or meta.nominal_voltage_mV is not None)
            if has_current and has_voltage:
                placeable.append(doc.doc_id)
        # The same detector the window will pool (one for the whole series, chosen among the
        # recordings that can become a point), so this gate and the plot agree about which
        # tables count.
        series_module_id = voltage_dep.series_module_id(self._session, placeable)
        return sum(
            1 for doc_id in placeable
            if any(r.module_id == series_module_id and isinstance(r.payload, EventSet)
                   for r in self._session.results.for_doc(doc_id)))

    def _sync_vdep_action(self) -> None:
        """Voltage dependence is live once >=2 recordings have BOTH a voltage AND a
        detection (a single point is not a trend)."""
        self._apply_surface_gate(SURFACE_VDEP)

    def _open_vdep(self) -> None:
        """Open the Voltage series page of the single Analysis Explorer."""
        return self._open_figure_kind(PAGE_VOLTAGE_SERIES)

    # -- power-spectrum / noise window -----------------------------------------------------------
    def _has_psd_recording(self) -> bool:
        """True when >=1 open recording could yield a noise spectrum — a live current
        channel with a known sample rate. Reads META only, so gating never computes a
        PSD. PSD is raw-signal QC, so this never looks at detections."""
        for doc in self._session.documents():
            meta = doc.meta
            if meta.sample_rate_hz is None:
                continue
            if any(effective_kind(self._session, doc.doc_id, c.index)
                   is ChannelKind.CURRENT and not c.is_dead for c in meta.channels):
                return True
        return False

    def _sync_psd_action(self) -> None:
        self._apply_surface_gate(SURFACE_PSD)

    def _open_psd(self) -> None:
        """Open the Noise page of the single Analysis Explorer."""
        return self._open_figure_kind(PAGE_NOISE)

    def _open_psd_for_selection(self) -> None:
        """Analysis ▸ Power spectrum of the measured region — the CLOCK-SAFE transfer.

        The trace and the PSD panel are on two different time origins, both spelled
        "s" (see :mod:`nanoscope.core.clock` and the action's own tooltip). This carries
        the user's on-trace selection across as SAMPLE INDICES and does the one
        conversion — ``index / fs``, into seconds from the start of that recording,
        which is the clock ``PSDWindow.set_segment`` takes — here, in code.

        Both halves have to be right, and the second is the easy one to forget: the
        indices come from :meth:`TraceView.measure_bounds`, and the recording they are
        indices INTO comes from :meth:`TraceView.measure_target`. The Measure region
        does not have to be on the app's currently selected recording, and dividing one
        recording's indices by another's sample rate is exactly the class of error this
        action exists to remove.
        """
        from .psd_panel import PSDWindow

        target = self._trace_view.measure_target()
        bounds = self._trace_view.measure_bounds()
        if target is None or bounds is None:
            self.statusBar().showMessage(
                "Measure is on but no region is selected. Press C over an event, or drag a "
                "region, and this opens the spectrum of exactly that region."
                if self._trace_view.measure_empty() else
                "Turn on the Measure tool and drag a region on the trace first — this "
                "opens the spectrum of exactly that region.", 8000)
            return
        doc_id, channel = target
        if not self._session.has_document(doc_id):
            self.statusBar().showMessage(
                "That selection is on a recording that is no longer open.", 8000)
            return
        meta = self._session.document(doc_id).meta
        fs = meta.sample_rate_hz
        if not fs:
            self.statusBar().showMessage(
                "That recording declares no sample rate, so a window in seconds has no "
                "meaning for it.", 8000)
            return
        # ...and it must be a recording the PSD window can actually point AT. ``set_target``
        # ignores a recording with no live current channel and leaves the combo where it
        # was — so without this the segment would be applied to a DIFFERENT recording, and
        # the status line below would confidently describe the wrong one. The whole action
        # exists to stop a window being analysed on the wrong footing.
        if not any(effective_kind(self._session, doc_id, c.index)
                   is ChannelKind.CURRENT and not c.is_dead for c in meta.channels):
            self.statusBar().showMessage(
                "That selection is on a recording with no live current channel — there is "
                "no noise spectrum to take.", 8000)
            return
        i0, i1 = int(bounds[0]), int(bounds[1])
        start_s, end_s = i0 / float(fs), i1 / float(fs)

        # This menu item is an explicit request for a NEW spectrum, unlike merely opening
        # a saved PSD surface.  It therefore retires the frozen snapshot before any kernel
        # is allowed to run and checkpoints the replacement below.
        self._workspace_pending.discard(PSD_ARTIFACT_ID)
        self._workspace_frozen.discard(PSD_ARTIFACT_ID)
        from ..core.analysis_workspace import _stable_source
        owner = self._ensure_figure_workspace()
        self._open_figure_kind(PAGE_NOISE)
        key = owner.current_id()
        page = owner.page(key)
        if page is None or owner.get(key).spec.kind != PAGE_NOISE:
            return
        owner.set_binding(key, {"follow": False, "source": _stable_source(self, doc_id, channel)})
        page.set_target(doc_id, channel, recompute=False)
        self._update_figure(key, lambda: page.set_segment(start_s, end_s))
        self._show_analysis_page(
            PAGE_NOISE, self._psd_window,
            refresh=lambda: self._replace_workspace_surface(
                PSD_ARTIFACT_ID, self._point_psd_at_selection))
        # The status line states BOTH clocks, so the number the user just saw on the
        # trace and the number now in the Start box are visible together — the one
        # place they were never shown side by side is where the 5.7x error lived.
        offset = self._trace_view.clock_offset_for(doc_id, channel)
        if clock.clocks_agree(offset) or offset is None:
            where = f"{start_s:.4f}–{end_s:.4f} s"
        else:
            where = (f"{start_s:.4f}–{end_s:.4f} s into the recording "
                     f"(= {start_s + offset:.4f}–{end_s + offset:.4f} s on the trace)")
        self.statusBar().showMessage(
            f"Noise spectrum over {where}, {i1 - i0} samples.", 8000)

    def _point_psd_at_selection(self) -> None:
        """Point the open PSD window at the current selection's recording, else just
        refresh its recording list. A no-op until the window is built."""
        if self._psd_window is None:
            return
        if self._current_selection is not None \
                and self._session.has_document(self._current_selection[0]):
            self._psd_window.set_target(self._current_selection[0], self._current_selection[1])
        else:
            self._psd_window.refresh()
