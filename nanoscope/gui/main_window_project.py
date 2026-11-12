"""The project half of ``MainWindow``: gathering the session into a project, Save, Save As and
the write, Open and the restore of every document, and the recent-projects menu.

``_ProjectMixin`` is a plain class listed before ``QMainWindow`` in the bases of ``MainWindow``,
with no ``__init__`` and no state of its own. The method bodies were moved verbatim out of
``main_window.py`` by a mechanical split; nothing in them was edited.

``ask_export_path`` and ``locked_project_read`` are imported from ``main_window_seams``, whose
forwarders look the names up on ``main_window`` each time they are called, so a test's patch
of ``main_window.ask_export_path`` or ``main_window.locked_project_read`` still reaches these
methods.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
import os
from pathlib import Path
from typing import Any

from PySide6 import QtWidgets

from ..core.layer import ColourRegistry
from ..core.analysis_workspace import OWNED_ARTIFACT_IDS
from ..core.project import AnalyticalWorkspace
from ..core.project_guard import (
    ProjectDiskStamp,
    ProjectSaveConflict,
    guarded_project_write,
    project_changed,
    project_disk_stamp,
    project_save_lock,
)
from ..core.types import SampleRange
from .main_window_shared import (
    _MAX_RECENT,
    _CLOSE_SAVE,
    _CLOSE_CANCEL,
    _with_project_suffix,
    _stored_baseline_fingerprint,
)
from .main_window_seams import ask_export_path, locked_project_read


class _ProjectMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    # -- project save/open -----------------------------------------------------------------------
    def _gather_project_state(self):
        """Snapshot the whole Session into a plain :class:`ProjectState` for saving.

        File-backed documents are stored as REFERENCES (resolved path == doc_id, plus the
        loaded SampleRange); SEGMENTS are stored as SPECS (uid + parent key + window +
        name) and re-derived from their parent on load, never as signal. Every channel's
        calibration, every layer's view state, every detection/ML EventSet (curated df +
        provenance + module_id + doc/channel), the DERIVED fits (v6: fit.dwell / fit.depth /
        fit.levels — see FitRec) and the manual measurements table come along. Raw signal
        does NOT: file signal reopens from disk.

        What the live session does NOT know is also written: records from the opened project
        whose recording could not be loaded are carried through verbatim rather than dropped
        (see :meth:`_capture_unrestorable`).

        Calibrations, layers and results are keyed by each document's PERSISTENT KEY (a
        file's path, a segment's uid) — never the volatile in-memory doc_id — so they
        re-attach after the parents reopen and the segments are re-derived."""
        import pandas as pd

        from ..analysis.events import EventSet
        from ..analysis.stats import FitResult, fit_to_json
        from ..core.project import (
            CalibrationRec, DocumentRef, FitRec, LayerRec, ProjectState, ResultRec, SegmentRec,
        )
        from ..core.calibration import channel_identity
        from ..core.identity import describe_source_change, same_source
        from ..core.segment import SegmentDocument, persistent_key

        session = self._session
        documents: list[DocumentRef] = []
        segments: list[SegmentRec] = []
        calibrations: list[CalibrationRec] = []
        pkey: dict[str, str] = {}   # live doc_id -> persistent key (identity across a round-trip)
        # live doc_id -> {channel index: ChannelInfo}. v7: every record files the IDENTITY of
        # its channel (name + kind) alongside the index, because an index is a position and a
        # re-export that reorders the signal columns moves it. See CalibrationRec/ResultRec
        # and `calibration.resolve_channel`.
        chinfo: dict[str, dict[int, Any]] = {}
        by_key = {persistent_key(doc): doc for doc in session.documents()}

        def source_document(doc):
            # None once the parent (or an ancestor) is not open under its saved key: "Close
            # only this recording" keeps a recording's segments open. Each use below skips
            # such a segment, and _write_project's unrestorable-segment warning counts it.
            # A strict lookup here raised KeyError instead, which failed every save and every
            # recovery capture.
            while isinstance(doc, SegmentDocument):
                doc = by_key.get(doc.spec.parent_key)
            return doc

        for doc in session.documents():
            self._remember_loaded_source(doc.doc_id)
        live_sources = dict(self._loaded_source_fingerprints)
        result_sources = {}
        for doc in session.documents():
            root = source_document(doc)
            if root is not None:
                result_sources[doc.doc_id] = live_sources[root.doc_id]

        def evidence(res):
            if res.source_fingerprint_recorded or res.source_fingerprint is not None:
                return res.source_fingerprint
            return result_sources.get(res.doc_id)

        saved_sources = dict(live_sources)
        unknown_roots, mismatched_roots = set(), set()
        carried = self._carried_over or {}
        parents = {spec.segment_uid: spec.parent_key for spec in carried.get("segments", [])}
        evidence_records = []
        for res in session.results.all():
            if session.has_document(res.doc_id):
                root = source_document(session.document(res.doc_id))
                if root is not None:
                    evidence_records.append((res, root))
        for record in [*carried.get("results", []), *carried.get("fits", [])]:
            key, seen = record.doc_id, set()
            while key in parents and key not in seen:
                seen.add(key)
                key = parents[key]
            match = next((doc for doc in session.documents()
                          if same_source(persistent_key(doc), key)), None)
            root = source_document(match) if match is not None else None
            if root is not None:
                evidence_records.append((record, root))
        for res, root_doc in evidence_records:
            if not res.source_fingerprint_recorded and res.source_fingerprint is None:
                continue
            if res.source_fingerprint is None:
                unknown_roots.add(root_doc.doc_id)
                continue
            if describe_source_change(res.source_fingerprint, live_sources[root_doc.doc_id],
                                      loaded_n_samples=root_doc.meta.n_samples):
                saved_sources[root_doc.doc_id] = res.source_fingerprint
                mismatched_roots.add(root_doc.doc_id)
        for doc_id in unknown_roots - mismatched_roots:
            saved_sources[doc_id] = None

        for doc in session.documents():
            doc_id = doc.doc_id
            key = persistent_key(doc)
            pkey[doc_id] = key
            chinfo[doc_id] = {int(i.index): i for i in doc.meta.channels}
            if isinstance(doc, SegmentDocument):
                s = doc.spec
                segments.append(SegmentRec(
                    segment_uid=s.segment_uid, parent_key=s.parent_key,
                    i0=int(s.i0), i1=int(s.i1), name=s.name, t0_abs_s=float(s.t0_abs_s)))
            else:
                lr = doc.meta.loaded_range
                # `key` (== source_key == the resolved path), never `doc_id`: a doc_id
                # carries a per-load token and is meaningless in the next session.
                #
                # The SOURCE FINGERPRINT is what makes the reference checkable. n_samples
                # comes from the DOCUMENT (what was actually analysed), the rest from the
                # file when this load was installed. A save must retain that evidence
                # even if the path now contains replacement bytes. Restored results may
                # require an older stamp until they have all been replaced.
                documents.append(DocumentRef(
                    path=key,
                    loaded_range=(int(lr.start), int(lr.stop)) if lr is not None else None,
                    source=saved_sources[doc_id]))
            for info in doc.meta.channels:
                # The record's OWN translation, never a field list spelled out here: a
                # calibration field that exists on the value object and not in the record is
                # not merely "not persisted", it is silently BLANKED on the next ordinary
                # Apply — which is how `note` used to destroy the user's sidecar. A test
                # compares the two field lists; see CalibrationRec.
                calibrations.append(CalibrationRec.from_calibration(
                    key, info.index, session.calibration(doc_id, info.index), info))

        def _identity(live_doc_id: str, channel: int) -> dict[str, str | None]:
            """``channel_name`` / ``channel_kind`` for a record, or None when the channel is
            not on a live document (nothing to be honest about, so claim nothing)."""
            info = chinfo.get(live_doc_id, {}).get(int(channel))
            if info is None:
                return {"channel_name": None, "channel_kind": None}
            name, kind = channel_identity(info)
            return {"channel_name": name, "channel_kind": kind}

        layers = [LayerRec(doc_id=pkey.get(lay.doc_id, lay.doc_id), channel=lay.channel,
                           color=lay.color, visible=lay.visible,
                           time_offset_s=lay.time_offset_s, excluded=lay.excluded,
                           **_identity(lay.doc_id, lay.channel))
                  for lay in session.layers()]

        results: list[ResultRec] = []
        for res in session.results.all():
            if isinstance(res.payload, EventSet):
                results.append(ResultRec(
                    doc_id=pkey.get(res.doc_id, res.doc_id), channel=res.channel,
                    module_id=res.module_id, events=res.payload.df,
                    provenance=dict(res.payload.provenance or {}),
                    # The settings the run used, and the gain/offset it was computed under.
                    # Dropping either was a silent lie on reload: an empty params made
                    # "Re-run" re-detect with the PANEL'S defaults (wiping every curation),
                    # and a re-derived fingerprint made a genuinely stale result look fresh.
                    params=dict(res.params or {}),
                    calibration_fingerprint=res.calibration_fingerprint,
                    module_version=str(res.module_version or ""),
                    baseline_fingerprint=res.baseline_fingerprint,
                    source_fingerprint=evidence(res),
                    source_fingerprint_recorded=True,
                    # And the caveats those numbers were produced under — the
                    # unprobed-baseline advisory above all. See ResultRec.
                    warnings=tuple(res.warnings or ()),
                    **_identity(res.doc_id, res.channel)))

        # v6 -- DERIVED results (fit.dwell / fit.depth / fit.levels). The loop above
        # collects only EventSet payloads, which is why every fit a user ran was silently
        # dropped on save: the numbers were re-derivable from the events, but a minute of
        # bootstrap vanishing without a word is not a re-derivation, it is a loss. See
        # FitRec. A payload this build does not recognise is skipped rather than guessed at.
        fits: list[FitRec] = []
        for res in self._session.results.all():
            if isinstance(res.payload, EventSet):
                continue
            table = curve = None
            if isinstance(res.payload, pd.DataFrame):
                table = res.payload
            elif isinstance(res.payload, FitResult):
                curve = fit_to_json(res.payload)
            else:
                continue
            fits.append(FitRec(
                doc_id=pkey.get(res.doc_id, res.doc_id), channel=res.channel,
                module_id=res.module_id, table=table, curve=curve,
                params=dict(res.params or {}),
                calibration_fingerprint=res.calibration_fingerprint,
                baseline_fingerprint=res.baseline_fingerprint,
                source_fingerprint=evidence(res),
                source_fingerprint_recorded=True,
                warnings=tuple(res.warnings or ()),
                **_identity(res.doc_id, res.channel)))

        meas = self._analysis_panel.measure_panel.measurements_df
        measurements = meas if (meas is not None and len(meas)) else None

        # Capture after every legacy table has settled.  The cache merge preserves
        # unknown/plugin artifacts and pending unresolved built-ins byte-for-byte; only
        # live, successfully installed surfaces replace the stable id they own.
        workspace = self._capture_analytical_workspace()
        from ..core.acquisition_state import (
            WORKSPACE_KEY, capture_acquisition_overrides, prepare_acquisition_workspace,
        )
        prepare_acquisition_workspace(workspace.extra)
        workspace.extra[WORKSPACE_KEY] = capture_acquisition_overrides(
            session, workspace.extra.get(WORKSPACE_KEY, []))
        from ..core.sample_rate_state import (
            WORKSPACE_KEY as RATE_KEY, capture_sample_rate_declarations,
            prepare_sample_rate_workspace,
        )
        prepare_sample_rate_workspace(workspace.extra)
        workspace.extra[RATE_KEY] = capture_sample_rate_declarations(
            session, workspace.extra.get(RATE_KEY, []))
        from ..core.table_choice_state import (
            WORKSPACE_KEY as CHOICE_KEY, capture_table_choices, prepare_table_choice_workspace,
        )
        prepare_table_choice_workspace(workspace.extra)
        live_choices = []
        for (choice_doc, choice_ch), module_id in self._analysis_panel.chosen_tables().items():
            if not session.has_document(choice_doc):
                continue
            info = chinfo.get(choice_doc, {}).get(int(choice_ch))
            live_choices.append({
                "doc_id": pkey.get(choice_doc, choice_doc), "channel": int(choice_ch),
                "channel_name": None if info is None else channel_identity(info)[0],
                "module_id": module_id})
        if live_choices or CHOICE_KEY in workspace.extra:
            workspace.extra[CHOICE_KEY] = capture_table_choices(
                live_choices, workspace.extra.get(CHOICE_KEY))
        # ...and each table's un-run form edits (design §5.2, the owner's decision 4), keyed
        # the same way. The live session speaks for every channel it holds, so a draft its
        # own run has since settled is not carried on from the file it was opened from.
        from ..core.form_draft_state import (
            WORKSPACE_KEY as DRAFT_KEY, capture_form_drafts, prepare_form_draft_workspace,
        )
        prepare_form_draft_workspace(workspace.extra)
        live_drafts = []
        for (draft_doc, draft_ch, draft_module), form in (
                self._analysis_panel.form_drafts().items()):
            if not session.has_document(draft_doc):
                continue
            info = chinfo.get(draft_doc, {}).get(int(draft_ch))
            live_drafts.append({
                "doc_id": pkey.get(draft_doc, draft_doc), "channel": int(draft_ch),
                "channel_name": None if info is None else channel_identity(info)[0],
                "module_id": draft_module, "state": form})
        if live_drafts or DRAFT_KEY in workspace.extra:
            workspace.extra[DRAFT_KEY] = capture_form_drafts(
                live_drafts, workspace.extra.get(DRAFT_KEY),
                [(pkey[doc_id], {ch: channel_identity(info)[0] for ch, info in infos.items()})
                 for doc_id, infos in chinfo.items()])
        # ...and a copy of this computer's plot presets and templates, which otherwise live
        # in its preferences only: a project opened elsewhere then still offers the looks
        # and recipes its figures were made with. Nothing is written for a computer that
        # has none, so such a project's file is exactly what it was.
        from ..core.plot_preset_state import (
            WORKSPACE_KEY as PRESET_KEY, capture_plot_presets, has_any,
            prepare_plot_preset_workspace,
        )
        from .plot_preset_sync import local_plot_presets, unreadable_preset
        prepare_plot_preset_workspace(workspace.extra)
        local_presets = local_plot_presets()
        if has_any(local_presets) or PRESET_KEY in workspace.extra:
            workspace.extra[PRESET_KEY] = capture_plot_presets(
                local_presets, workspace.extra.get(PRESET_KEY), keep=unreadable_preset)
        state = ProjectState(
            documents=documents, segments=segments, calibrations=calibrations, layers=layers,
            view_mode=self._session.view_mode.value, results=results,
            measurements=measurements, fits=fits,
            label_vocabulary=self._label_vocab.to_json(),
            colours=self._session.colours.to_json(),
            source_labels=self._session.source_labels.to_json(), workspace=workspace)

        # ...and everything the OPENED project held that the restore could not install,
        # written back verbatim. The live session is the wrong authority on what the project
        # CONTAINS: it holds what LOADED. See _capture_unrestorable.
        #
        # Merged BY FIELD NAME off the state itself, not by six hand-written lines: the
        # carry is computed by walking ProjectState's fields, so writing it back must walk
        # them too or a record type added to one side would be collected and then silently
        # dropped on its way to the file — which is the same defect one layer along.
        for name, extra in (self._carried_over or {}).items():
            current = getattr(state, name, None)
            if isinstance(current, list):
                current.extend(extra)
        return state

    def _project_save_parent(self) -> QtWidgets.QWidget:
        """The window in front of which the current save UI must appear."""
        return self._project_save_requester or self

    def _save_project(self) -> bool:
        """File ▸ Save Project (Ctrl+S). Re-saves the current project in place once it
        has a path; otherwise falls through to Save As. Returns True when a file was
        actually written (so the close guard can tell a real save from a cancel/failure)."""
        if self._project_path:
            return self._write_project(self._project_path)
        return self._save_project_as()

    def _save_project_as(self) -> bool:
        """File ▸ Save Project As… — always prompts for a .nanoscope path. Returns True when
        a file was written, False if the dialog was cancelled or the save failed."""
        save_epoch = self._figure_project_epoch
        recovery_target = self._recovery_suggested_project_path
        start = (str(Path(self._project_path).parent) if self._project_path else
                 str(Path(recovery_target).parent) if recovery_target else
                 self._last_project_dir)
        # A recovered snapshot never silently ADOPTS the old project's path: its disk
        # revision may have changed since the crashed process last saw it.  We do retain
        # that path as the chooser's suggestion, so the user can deliberately replace it
        # through Save As (and its existing overwrite confirmation) in one step.
        suggestion = (self._project_path or recovery_target or
                      str(Path(start) / "analysis.nanoscope"))
        # `outputs` applies the SAME .nanoscope normalisation the write does, so the
        # overwrite prompt is about the file that will actually be replaced — typing
        # "study" over an existing study.nanoscope has to warn, not sail past.
        chosen = ask_export_path(
            self._project_save_parent(), title="Save Project", suggestion=suggestion,
            filters="NanoScope project (*.nanoscope);;All files (*)",
            outputs=lambda p: (Path(_with_project_suffix(str(p))),))
        if chosen is None or self._figure_project_epoch != save_epoch:
            return False
        return self._write_project(_with_project_suffix(str(chosen)))

    @staticmethod
    def _unrestorable_segments(state) -> list:
        """The :class:`SegmentRec`s in ``state`` that a reload could NOT re-derive.

        A segment is persisted as a spec and rebuilt from its parent on load, so it is
        restorable only if its ``parent_key`` names a document that is ALSO in this project —
        a saved file reference, or another restorable segment. Anything else is a spec pointing
        at nothing: it will be dropped on reload, along with every result keyed under it.
        Computed by fixpoint so a chain of nested segments resolves in either save order."""
        restorable = {d.path for d in state.documents}
        pending = list(getattr(state, "segments", ()) or ())
        progressed = True
        while pending and progressed:
            progressed = False
            still = []
            for s in pending:
                if s.parent_key in restorable:
                    restorable.add(s.segment_uid)
                    progressed = True
                else:
                    still.append(s)
            pending = still
        return pending

    def _confirm_save_dropping_segments(self, n_segments: int, n_results: int) -> bool:
        """Ask before writing a project that cannot carry ``n_segments`` segment(s). Returns
        True to write it anyway. Injectable so the suite never faces a live modal."""
        box = QtWidgets.QMessageBox(self._project_save_parent())
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Save Project")
        box.setText(f"{n_segments} segment(s) cannot be saved in this project.")
        box.setInformativeText(
            "A segment is stored as a crop of its parent recording and rebuilt from it on "
            "load, and their parent recording is no longer open — so they, and "
            f"{n_results} result(s) on them, would be missing when this project is reopened.\n\n"
            "Reopen the parent recording(s) and save again to keep them.")
        save_anyway = box.addButton("Save without them", QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is save_anyway

    # -- the concurrent-writer guard --------------------------------------------------------------
    @staticmethod
    def _disk_stamp_of(path: str) -> ProjectDiskStamp | None:
        """Content-aware identity of the project at ``path``, or None when absent.

        Size and nanosecond mtime are retained for diagnosis and backward familiarity;
        SHA-256 is the deciding fourth field, so a same-size rewrite that restores the old
        timestamp cannot pass for the revision this window loaded.
        """
        return project_disk_stamp(path)

    def _project_changed_under_us(self, path: str) -> bool:
        """Has the project file changed on disk since this window last read or wrote it?

        True only when all three hold: this window HAS a stamp (it read or wrote a project),
        ``path`` is that same file, and its metadata or content digest no longer matches. A
        file that is simply GONE is not a conflict — there is no newer save there to destroy,
        and the write will recreate it. A different target path is the Save-As overwrite
        prompt's business, not this guard's.
        """
        stamp = self._project_disk_stamp
        if stamp is not None and os.path.realpath(str(path)) != stamp.real_path:
            return False
        return project_changed(stamp, path)

    def _confirm_overwrite_newer_project(self, path: str) -> bool:
        """Ask before replacing a project file that changed under this window. Returns True
        to overwrite it anyway. Injectable so the suite never faces a live modal.

        THE QUESTION IS THE FIX for silent last-writer-wins: two windows on one
        ``.nanoscope`` (a supported shape — the packaged log lock exists so a later
        instance can run) used to revert each other's committed curation with zero
        dialogs, both status bars claiming their save was the file. The atomic layer
        holds (no interleaving, no corruption); what was missing was the honesty. The
        default is Cancel: the destructive answer is never the accident."""
        box = QtWidgets.QMessageBox(self._project_save_parent())
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Save Project")
        box.setText(
            f"{Path(path).name} has changed on disk since this window last read or "
            f"saved it.")
        box.setInformativeText(
            "Another NanoScope window — or another program — has written this project "
            "in the meantime. Saving now replaces that newer file with this window's "
            "state, and any work it holds that this window does not will be lost.\n\n"
            "To keep both, cancel and use File ▸ Save Project As… to save this window's "
            "work under another name, or reopen the project to see the newer version.")
        overwrite = box.addButton("Overwrite the newer file",
                                  QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is overwrite

    def _confirm_overwrite_existing_project(self, path: str) -> bool:
        """Ask before Save As replaces a project this window does not own.

        The native save dialog normally asks too, but its check is outside NanoScope's
        inter-process lock. Two windows can both choose a missing name and one can create
        it before the other's commit. This second, revision-specific question is what
        prevents that newly-created project from being overwritten silently.
        """
        box = QtWidgets.QMessageBox(self._project_save_parent())
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Save Project As")
        box.setText(f"{Path(path).name} now exists.")
        box.setInformativeText(
            "Another NanoScope window — or another program — may have saved this project "
            "after you chose the name. Replacing it will lose any work that file holds.\n\n"
            "To keep both, cancel and choose a different project name.")
        overwrite = box.addButton("Replace the existing file",
                                  QtWidgets.QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.exec()
        return box.clickedButton() is overwrite

    def _confirm_figure_project_conversion(self, path: str) -> str:
        """The first in-place v13 conversion is an explicit product decision."""
        box = QtWidgets.QMessageBox(self._project_save_parent())
        box.setWindowTitle("Save and convert project")
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setText(f"Convert {Path(path).name} to the new figure workspace?")
        box.setInformativeText(
            "Older NanoScope builds cannot open a version 13 project. "
            "Save and convert keeps a verified copy of the original beside this file "
            "before replacing it. Save a copy keeps the original project unchanged.")
        convert = box.addButton("Save and convert", QtWidgets.QMessageBox.AcceptRole)
        copy = box.addButton("Save a copy", QtWidgets.QMessageBox.ActionRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(copy)
        box.setEscapeButton(cancel)
        box.exec()
        return "convert" if box.clickedButton() is convert else "copy" if box.clickedButton() is copy else "cancel"

    def _write_project(self, path: str) -> bool:
        """Serialize the session to ``path`` and adopt it as the current project. Returns
        True when a file was written; on failure the error is reported and False returned
        (the window stays open with the work untouched). A successful write clears the
        unsaved-changes flag and repaints the title.

        A save NEVER reports success while dropping work. A segment whose parent recording is
        closed cannot be re-derived on load, so it (and every result on it) would silently
        vanish — the old code wrote the file, said "Saved project (1 result(s))" and cleared the
        dirty flag while the reload came back empty. Now the user is asked first, and if she
        saves anyway the status line names exactly what was left behind.

        A save NEVER silently reverts somebody else's, either. The file's byte digest is
        compared against the stamp taken at this window's last load/save of it, and a
        mismatch — another window's Ctrl+S, most likely — raises
        :meth:`_confirm_overwrite_newer_project` before anything is replaced. The final
        comparison and atomic replacement share one inter-process lock. Confirmation
        approves one exact conflicting revision; a third write while the question is open
        raises the question again rather than becoming a lost update."""
        from ..core.project import VERSION, read_project_container_version, save_project
        from ..core.project_guard import retain_verified_project_backup

        # Approval is local to this save attempt and NEVER changes the window's remembered
        # revision. Only a successful commit below may do that. The stamp names the exact
        # conflicting bytes the user agreed to replace; if another writer lands after the
        # dialog, guarded_project_write raises with that third revision and we ask again.
        held_stamp = self._project_disk_stamp
        same_held_target = (
            held_stamp is not None
            and os.path.realpath(str(path)) == held_stamp.real_path
        )
        expected_stamp = held_stamp if same_held_target else None
        # ``None`` used to mean both "there was no baseline" and "the baseline was
        # absent". The distinction matters when two first-time Save As operations choose
        # the same missing name: once one creates it, the other must conflict.
        expected_absent = not same_held_target
        confirm_conflict = (
            self._confirm_overwrite_newer_project
            if same_held_target else self._confirm_overwrite_existing_project
        )
        approved_stamp: ProjectDiskStamp | None = None
        approved_dropped: tuple[tuple[str, ...], int] | None = None
        approved_conversion: ProjectDiskStamp | None = None
        conversion_backup = None
        save_epoch = self._figure_project_epoch
        while True:
            if self._figure_project_epoch != save_epoch:
                self.statusBar().showMessage("The project changed; this save was cancelled.", 10000)
                return False
            # Preserve the old user-visible order: if a conflict already exists, ask before
            # checkpointing or presenting any unrelated "save without segments" question.
            # This is only a preflight; correctness comes from the second comparison inside
            # guarded_project_write's lock immediately before the atomic write.
            try:
                with project_save_lock(path):
                    current_stamp = self._disk_stamp_of(path)
                    target_version = (read_project_container_version(path)
                                      if VERSION >= 13 and current_stamp is not None else None)
            except Exception as exc:  # noqa: BLE001 — a save failure is reported, never crashes
                QtWidgets.QMessageBox.warning(
                    self._project_save_parent(), "Save Project",
                    f"Could not save the project:\n{exc}")
                return False
            if project_changed(expected_stamp, path, current_stamp,
                               expected_absent=expected_absent) \
                    and current_stamp != approved_stamp:
                if not confirm_conflict(path):
                    return False
                approved_stamp = current_stamp
                continue
            if target_version is not None and target_version <= 12 and current_stamp != approved_conversion:
                decision = self._confirm_figure_project_conversion(path)
                if self._figure_project_epoch != save_epoch:
                    return False
                if decision == "copy":
                    return self._save_project_as()
                if decision != "convert":
                    return False
                approved_conversion = current_stamp
                # Dialogs run a nested event loop; re-check target bytes and re-gather.
                continue

            # Commit anything edited in place that never announced itself, BEFORE the state
            # is gathered. core/history.py promises that a forgotten announcement costs a
            # menu word and never a row, and this is one of the two moments that promise is
            # about: what goes into the file and what the stack can put back must agree.
            # A conflict dialog runs a nested Qt event loop, so a successful confirmation
            # loops back here and gathers afresh rather than saving a pre-dialog snapshot.
            self._checkpoint_everything()
            write_revision = self._session_revision
            try:
                state = self._gather_project_state()
            except Exception as exc:  # noqa: BLE001 — a save failure is reported, never crashes
                QtWidgets.QMessageBox.warning(
                    self._project_save_parent(), "Save Project",
                    f"Could not save the project:\n{exc}")
                return False

            dropped = self._unrestorable_segments(state)
            n_dropped_results = 0
            if dropped:
                lost_keys = {s.segment_uid for s in dropped}
                # state.fits counts too. A fit on an unrestorable segment is written into the
                # file and can never be re-attached (its segment cannot be re-derived), so it
                # is dropped work — and this dialog exists for the sole purpose of naming the
                # work being left behind. Counting only EventSets under-reported by exactly the
                # artefacts v6 made persistable: a bootstrap dwell fit and a CUSUM level table
                # went unmentioned while the user was told "1 result(s)".
                n_dropped_results = sum(1 for r in state.results if r.doc_id in lost_keys)
                n_dropped_results += sum(1 for f in state.fits if f.doc_id in lost_keys)
                dropped_signature = (
                    tuple(sorted(str(s.segment_uid) for s in dropped)), n_dropped_results)
                if dropped_signature != approved_dropped:
                    if not self._confirm_save_dropping_segments(
                            len(dropped), n_dropped_results):
                        return False
                    approved_dropped = dropped_signature
            if self._figure_project_epoch != save_epoch:
                return False

            # `notes` collects what the write changed about the FILE ITSELF — a symlink
            # followed to its destination, or other hard-linked names that now hold the
            # previous contents. An atomic replace installs a new inode, and that is the one
            # thing it cannot preserve; the alternative is writing in place, which is the
            # destructive path this save exists to avoid. Empty on an ordinary save.
            file_notes: list[str] = []
            def write_verified_project():
                nonlocal conversion_backup
                if VERSION >= 13:
                    conversion_backup = retain_verified_project_backup(path, max_version=12)
                    return save_project(path, state, file_notes, verify_workspace=True)
                return save_project(path, state, file_notes)
            try:
                _, committed_stamp = guarded_project_write(
                    path,
                    expected_stamp,
                    approved_stamp,
                    write_verified_project,
                    expected_absent=expected_absent,
                )
            except ProjectSaveConflict as conflict:
                # The target changed after preflight/gather. Ask outside the lock so another
                # window is never frozen behind a human-facing modal, then loop to re-gather
                # state. Approval applies only to the revision carried by this exception.
                if not confirm_conflict(path):
                    return False
                approved_stamp = conflict.current_stamp
                continue
            except Exception as exc:  # noqa: BLE001 — a save failure is reported, never crashes
                QtWidgets.QMessageBox.warning(
                    self._project_save_parent(), "Save Project",
                    f"Could not save the project:\n{exc}")
                return False
            break

        self._project_path = path
        self._recovery_suggested_project_path = None
        # Use the stamp captured BEFORE the lock was released. Re-reading here would let a
        # fast third writer land in between and be mistaken for this window's own revision.
        self._project_disk_stamp = committed_stamp
        self._last_project_dir = str(Path(path).parent)
        self._add_recent_project(path)
        if dropped:
            # A SAVE THAT DROPPED WORK DOES NOT REPORT SUCCESS. The user was warned and
            # chose "Save without them", so the file is written — but the segments and
            # their results are STILL LIVE in this session and STILL unrecoverable (a
            # segment has no reopen path), and the file on disk does not contain them.
            # Clearing the flag here retired the quit guard on the one session that most
            # needed it: the title lost its `•` and quitting asked nothing, so the work
            # the modal had just named went out with the window. Staying dirty means
            # reopening the parent recording and pressing Save again still costs one
            # click, and closing without doing so still asks.
            # ...AND THE RECOVERY COPY THAT STILL HOLDS THEM STAYS LIVE: the explicit-save
            # fence does not advance, or a capture of this same revision landing after the
            # save is retired at once (_on_recovery_saved), and a crash then loses the
            # segments the file does not contain either.
            self._update_title()
        else:
            self._explicit_saved_revision = max(
                self._explicit_saved_revision, int(write_revision))
            # Retire only snapshots no newer than the state that actually reached disk.
            # An edit arriving through a nested Qt loop after the snapshot was gathered
            # remains dirty and its newer recovery generation remains live.
            self._retire_recovery_through(write_revision)
            if self._session_revision == write_revision:
                self._mark_clean()   # the written revision is still the live revision
            else:
                self._update_title()
        kept = len(state.results) + len(state.fits) - n_dropped_results
        msg = f"Saved project ({kept} result(s)) to {Path(path).name}."
        carried = self._carried_over or {}
        n_carried = len(carried.get("documents", []))
        if n_carried:
            # Say it, every time. The recording is not on screen, so without this line a
            # save looks like it wrote only what is visible — which is precisely the
            # impression under which the old behaviour destroyed the reference.
            # "not open in this session", not "could not be opened": the restore has more
            # than one way of leaving a recording behind, and one of them is the user
            # pressing Cancel — those were never attempted, so naming them as unopenable
            # blamed the disk for something the app was told to do.
            msg += (f" {n_carried} recording(s) that are not open in this session were kept "
                    f"in the project, not dropped.")
        # The same sentence for everything ELSE the restore could not install — a segment
        # whose window no longer fits, a record naming a channel that is gone, a table this
        # build could not read. Counting only documents was how the enumerated carry looked
        # correct while erasing the other six exits: the status line said "0 result(s)" and
        # nothing said the four curated events had just left the file.
        n_other = sum(len(v) for k, v in carried.items() if k != "documents")
        if n_other:
            msg += (f" {n_other} record(s) this project holds but could not be re-attached "
                    f"were kept in it, not dropped.")
        if dropped:
            msg += (f" {len(dropped)} segment(s) and {n_dropped_results} result(s) were NOT "
                    f"saved — their parent recording is closed.")
        if conversion_backup is not None:
            self._last_project_conversion_backup = str(conversion_backup)
            msg += f" Original project retained at {conversion_backup}."
        self.statusBar().showMessage(msg, 12000 if dropped or conversion_backup else 8000)
        fresh = [n for n in file_notes
                 if (str(path), n) not in self._disclosed_file_notes]
        self._disclosed_file_notes.update((str(path), n) for n in file_notes)
        if fresh:
            QtWidgets.QMessageBox.information(
                self._project_save_parent(), "Save Project",
                f"The project was saved.\n\n" + "\n\n".join(fresh))
        return True

    def _disclose_identity_notes(self, title: str, paths) -> None:
        """Tell the user what a finished write changed about the FILE, not its contents.

        THE PROJECT SAVE WAS THE ONLY SURFACE THAT DID THIS, and only because it happens
        to pass a ``notes`` list into :func:`save_project`. Every other write in the app —
        the events CSV and its sidecar, each exported figure, every ``Save image…`` —
        goes through the same atomic replace, breaks hard links and follows symlinks in
        exactly the same way, and said nothing. So a researcher could hard-link an
        exported ``events.csv`` into a manuscript folder, re-export after curating two
        events, and be left with a manuscript copy silently holding the OLD table, while
        the project save in the same session disclosed the identical situation.

        :func:`atomic_write.take_identity_notes` is the seam that makes this cheap: the
        ledger is filled by the write itself, so a surface only has to ask for the notes
        belonging to the files IT wrote — anything pending for another file stays pending,
        and draining means a note is never shown twice. Scoped by ``paths`` for that
        reason; never call it unscoped from an export.
        """
        from ..core import atomic_write

        notes = atomic_write.take_identity_notes(paths)
        fresh = [n for n in notes if (title, n) not in self._disclosed_file_notes]
        self._disclosed_file_notes.update((title, n) for n in notes)
        if fresh:
            QtWidgets.QMessageBox.information(self, title, "\n\n".join(fresh))

    def _open_project_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", self._last_project_dir,
            "NanoScope project (*.nanoscope);;All files (*)")
        if path:
            self._open_project(path)

    def _open_project(self, path: str) -> None:
        """Load a .nanoscope and drive the async restore. A load-time error (not a
        project, corrupt, unsupported version) is reported and aborts before any
        Session mutation.

        Opening a project REPLACES the session; it never merges into it. Two things follow,
        and both used to be missing:

        * unsaved work in the current session is guarded FIRST (Save / Discard / Cancel), so
          the open can never silently discard analysis the user has not written anywhere;
        * the session is then emptied, so the restore starts from nothing. Restoring on top of
          the live session skipped every document that happened to be open — presenting stale
          in-memory calibration/detections AS the project and then marking the whole thing
          clean, so the next Ctrl+S overwrote the good file with the stale state. It also
          merged two projects into one whenever the second was opened over the first.
        """
        from ..core.project import ProjectError, load_project

        try:
            # Read and stamp ONE revision. Without the shared project lock another
            # NanoScope process can commit after load_project returns but before the
            # stamp is taken; this window would then hold the old state while claiming
            # the new revision and could silently overwrite it on its next save.
            state, loaded_stamp = locked_project_read(
                path, lambda: load_project(path))
        except ProjectError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Open Project", f"Could not open the project:\n{exc}")
            return
        except Exception as exc:  # noqa: BLE001 — a corrupt archive must not crash the app
            QtWidgets.QMessageBox.warning(
                self, "Open Project", f"Could not open the project:\n{exc}")
            return

        # The file is readable; only now may the current session be disturbed.
        if not self._offer_to_save_unsaved_work():
            return   # Cancel, or a Save the user cancelled / that failed — nothing is touched.
        self._clear_session()

        self._project_path = path
        # This is the stamp captured under the SAME lock as the read above. A writer that
        # lands after the lock is released must remain different and raise a later save
        # conflict; re-reading here would incorrectly bless that newer revision.
        self._project_disk_stamp = loaded_stamp
        self._last_project_dir = str(Path(path).parent)
        self._add_recent_project(path)
        # The opened project is clean by definition; clear now (before the async restore,
        # whose own mutations are guarded) so the title never flashes a stale marker.
        self._mark_clean()
        self._begin_restore(state)

    def _offer_to_save_unsaved_work(self) -> bool:
        """Guard an action that will DISCARD the current session (today: Open Project).

        Returns True when it is safe to proceed — nothing to lose, the user chose Discard, or
        the user chose Save AND a file was actually written. Returns False when the user
        cancelled or the save did not happen, in which case the caller must leave the session
        exactly as it found it. Routes through the same :meth:`_confirm_close_choice` /
        :meth:`_run_save` seams the close guard uses, so there is ONE unsaved-work dialog.
        """
        if not (self._dirty and self._has_work_to_lose()):
            return True
        choice = self._confirm_close_choice()
        if choice == _CLOSE_CANCEL:
            return False
        if choice == _CLOSE_SAVE:
            return bool(self._run_save())
        # An explicit Discard means the recovery copy must not offer the same work again
        # on the next launch.  Retire it before replacing the session.
        self._discard_recovery(self._recovered_candidate)
        return True

    def _clear_session(self) -> None:
        """Empty the session so a project restore starts from nothing.

        Stops every job in flight (and drops anything queued behind it) so neither a stray
        document nor a late detection can land in the middle of the incoming restore, closes
        every open document — which drops its layers, calibrations and results through the
        normal Session teardown — and clears the manual measurements.

        Two deliberate omissions. It is NOT a dirty-flag operation: the caller has already dealt
        with unsaved work and sets the flag it wants afterwards. And it closes documents
        straight through the Session rather than via :meth:`_close_document`, so the
        parent-has-segments guard does not re-ask per recording — the caller asked once, about
        the whole session.
        """
        self._study_curation_project_generation += 1
        self._study_curation_sessions.clear()
        # Was left unset here: _sync_study_curation_bar only refreshes this pointer
        # from _study_curation_sessions when the CURRENTLY SELECTED doc has an entry
        # there (line below), which can never be true right after the clear() above --
        # so a stale context from before this reset kept pointing at an old study/doc,
        # and the curation bar stayed visible and "valid" for a session that no longer
        # exists.
        self._study_curation_context = None
        recovery_controller = self._recovery_controller
        if recovery_controller is not None:
            reset = getattr(recovery_controller, "reset_session", None)
            if callable(reset):
                reset()
        self._analysis_panel.abort_background_previews()
        # All three emit nothing; we reset the UI ourselves — and we use what they
        # report. A run that did NOT stop is detached: its answer is discarded, but its
        # thread is still burning cores behind the project the user is about to open,
        # and the app has been silent about that. Unlike the quit path there is somebody
        # here to tell, so tell them.
        stopped = self._controller.abort_stopped()
        stopped &= self._analysis_controller.abort_stopped()
        stopped &= self._batch_controller.abort_stopped()
        if not stopped:
            self.statusBar().showMessage(
                "A background run would not stop in time. Its answer is discarded, but it "
                "is still finishing in the background — the new project may open slowly.",
                8000)
        self._open_queue.clear()
        self._active_open = None
        self._active_analysis_label = None
        self._analysis_panel.forget_pending_job()
        self._reset_analytical_workspace()
        # The outgoing project's unloadable records belong to the outgoing project.
        self._carried_over = None
        self._batch_skipped = 0
        self._set_progress_visible(False)
        for doc in self._session.documents():
            self._session.close_document(doc.doc_id)
        self._analysis_panel.measure_panel.set_measurements_df(None)
        # The stack is per-session state: with every document closed there is nothing left
        # for any of its entries to be applied to, and an entry that outlived its recording
        # would be an undo the user could see offered and never see happen.
        self._history.clear()
        self._subpop_history_replays = {}
        self._measure_history.clear()
        self._form_pause.stop()
        self._settings_history.clear()
        self._settings_labels.clear()
        self._sidecar_moves.clear()
        self._sidecar_found.clear()
        self._undo_journal.clear()
        self._redo_journal.clear()
        # ...and the forms' un-run edits, which are the outgoing project's: the incoming one
        # brings its own (see _apply_restore_for_doc), and one never inherits the other's.
        self._analysis_panel.set_form_drafts({})
        # The whole-recording baseline memo is per-trace session state like everything else
        # here, and it had no caller anywhere in `nanoscope/`. It is keyed on a trace's
        # object identity through a weakref, so this is delayed RELEASE rather than a stale
        # hit (95.8 MB of float64 baselines outlived the owner's closed recording) — but a
        # memo that survives a session reset is a wrong number waiting for an id() the
        # allocator hands back, and one line here costs nothing.
        from ..analysis.baseline import clear_baseline_cache
        clear_baseline_cache()
        self._current_selection = None
        # Colour identity is project state, not an application preference. The Session
        # object itself is reused across Open Project, so its registry must not carry the
        # outgoing project's identities into the incoming one.
        self._session.colours = ColourRegistry()
        from ..core.source_style import SourceLabelRegistry
        self._session.source_labels = SourceLabelRegistry()
        self._project_path = None
        self._project_disk_stamp = None   # no project held -> nothing to conflict with
        self._recovery_suggested_project_path = None
        self._recovered_candidate = None
        self._restore_is_recovery = False
        self._info_bar.hide()

    # -- async restore ---------------------------------------------------------------------------
    def _begin_restore(self, state) -> None:
        """Enter restore mode and reopen every referenced document through the SAME
        sequential open pipeline, each at its saved range. Missing sources are skipped
        up front (recorded for a warning); per-doc state is applied as each lands.

        A source that EXISTS is not the same statement as a source that is the one the
        project was saved from, and the difference is not academic: re-exporting from
        Clampex over the same filename used to reattach 5 saved events to a 5,000-sample
        flat trace — all 5 past the end of it, event 0 claiming a 30 pA blockade in a signal
        with a 1.1 pA standard deviation — in complete silence. Each ref's saved fingerprint
        is carried through to :meth:`_apply_restore_for_doc`, which compares it against the
        file and the reopened document (see :mod:`nanoscope.core.identity`). The project
        still opens either way: this warns and explains, and never refuses.
        """
        self._restoring = True
        self._history.set_suspended(True)   # a reconstruction is not an undoable edit
        self._measure_history.set_suspended(True)
        self._settings_history.set_suspended(True)
        from ..core.project import stamp_legacy_result_sources
        stamp_legacy_result_sources(state)
        self._restore_state = state
        workspace = getattr(state, "workspace", None)
        self._analytical_workspace = (
            workspace if isinstance(workspace, AnalyticalWorkspace)
            else AnalyticalWorkspace())
        migrated = self._ensure_figure_workspace(install=False).install_workspace(self._analytical_workspace)
        self._analytical_workspace = migrated.workspace
        self._workspace_pending = {
            item.artifact_id for item in self._analytical_workspace.artifacts
            if item.artifact_id in OWNED_ARTIFACT_IDS
        }
        self._workspace_frozen.clear()
        self._sync_saved_subpop_action()
        # Install the registry BEFORE the first document creates a layer. add_layer asks it
        # for the persistent recording/channel key, so every trace and plot starts in the
        # saved ink instead of flashing or retaining a positional colour.
        self._session.colours = ColourRegistry.from_json(
            getattr(state, "colours", None))
        from ..core.source_style import SourceLabelRegistry
        self._session.source_labels = SourceLabelRegistry.from_json(getattr(state, "source_labels", None))
        self._restore_ranges = {}
        self._restore_sources = {}
        self._restore_missing = []
        self._restore_refusals = {}
        self._restore_lost_segments = []
        self._restore_bad_calibrations = []
        from ..core.acquisition_state import prepare_acquisition_workspace
        from ..core.sample_rate_state import prepare_sample_rate_workspace
        self._restore_notes = [f"A saved figure remains unsupported and was preserved: {failure.reason}"
                               for failure in migrated.failures]
        self._restore_notes.extend(prepare_acquisition_workspace(self._analytical_workspace.extra))
        self._restore_notes.extend(prepare_sample_rate_workspace(self._analytical_workspace.extra))
        # The project's copies of plot presets and templates: added to this computer where it
        # lacks them, never over its own, and said in the banner. A preference is never worth
        # a failed open: this runs before the restore has opened a single recording, so an
        # exception here would leave the window in restore mode with every history suspended.
        from .plot_preset_sync import install_project_presets
        try:
            self._restore_notes.extend(install_project_presets(self._analytical_workspace.extra))
        except Exception as exc:  # noqa: BLE001 -- reported in the banner, never aborts a restore
            import traceback
            traceback.print_exc()
            self._restore_notes.append(
                f"The plot presets and templates saved in this project could not be added to "
                f"this computer ({exc}). They stay in the project; nothing else was affected.")
        self._restore_channel_moved = []
        self._restore_channel_lost = []
        self._restore_path_moved = []
        self._restore_cancelled = []
        self._restored_records = set()
        self._info_bar.hide()

        to_open: list[str] = []
        for d in state.documents:
            p = Path(d.path)
            if not p.exists():
                self._restore_missing.append(d.path)
                continue
            rng = SampleRange(d.loaded_range[0], d.loaded_range[1]) if d.loaded_range else None
            self._restore_ranges[str(p.resolve())] = rng
            self._restore_sources[str(p.resolve())] = getattr(d, "source", None)
            to_open.append(d.path)

        if to_open:
            self.open_paths(to_open)
        # Nothing to open (every source missing, or an empty project): finalise now so the
        # measurements/view/warning still land and projectRestored still fires.
        if self._active_open is None and not self._open_queue:
            self._finalize_restore()

    def _rederive_segments(self, state) -> None:
        """Re-derive every saved SEGMENT from its (now-open) parent, then attach its saved
        calibrations / layers / results by persistent key.

        Parents are re-opened first (file docs, through the normal pipeline); a segment is
        rebuilt with :meth:`Session.create_segment` passing the SAVED uid, so it lands on the
        same identity its records were keyed under. A fixpoint pass handles nesting (a segment
        cut from another segment): each round derives every segment whose parent is now
        available, until no more progress is made. Segments whose parent never appeared (a
        missing source, or a broken chain) are recorded SEPARATELY from missing source files:
        they are not files, they were never on disk, and running them through the
        "source file(s) could not be found" banner (which then took ``Path(...).name`` of a
        sentence) told the user a straightforwardly false story about what went wrong."""
        specs = list(getattr(state, "segments", ()) or ())
        if not specs:
            return
        from ..core.segment import persistent_key, spec_window_refusal

        # persistent key -> live doc_id for everything currently open (parents so far).
        live: dict[str, str] = {persistent_key(d): d.doc_id for d in self._session.documents()}

        pending = specs
        progressed = True
        while pending and progressed:
            progressed = False
            still: list = []
            for spec in pending:
                parent_live = live.get(spec.parent_key)
                if parent_live is None or not self._session.has_document(parent_live):
                    still.append(spec)          # parent not (yet) available — retry next round
                    continue
                # THE PRECONDITION. A saved SegmentRec is an ASSERTION about a particular
                # stretch of a particular recording, not a request to crop; the builder
                # clamps because a live rubber-band drag genuinely is a request. Measured
                # on the sibling path: a "cut" of [40 000, 60 000) whose parent was
                # re-exported at 45 000 samples came back as [40 000, 45 000) — a QUARTER
                # of the analysed window — with meta.issues == (), a note stating only the
                # new window, all 4 saved events reattached and 3 of them lying entirely
                # past the end of the segment they were attached to, their dwell/depth/ECD
                # measured on data no longer on disk. On a project written without a source
                # fingerprint (an older project, which MIN_READ_VERSION = 1 guarantees
                # still loads) there was no banner at all. It refuses and is reported
                # instead; the spec and its results are then carried by
                # `_capture_unrestorable`, so the next save cannot rewrite i1 to 45 000 and
                # cannot drop the four curated events.
                parent_doc = self._session.document(parent_live)
                why = spec_window_refusal(
                    int(parent_doc.meta.n_samples), spec.i0, spec.i1)
                if why is not None:
                    self._restore_lost_segments.append(f"“{spec.name}” ({why})")
                    progressed = True           # consumed (as a refusal) — do not retry
                    continue
                try:
                    seg_id = self._session.create_segment(
                        parent_live, spec.i0, spec.i1, name=spec.name,
                        segment_uid=spec.segment_uid)
                except (ValueError, TypeError, KeyError) as exc:
                    self._restore_lost_segments.append(f"“{spec.name}” ({exc})")
                    progressed = True           # consumed (as a failure) — do not retry
                    continue
                live[spec.segment_uid] = seg_id
                self._restored_records.add(id(spec))    # INSTALLED — see _capture_unrestorable
                self._apply_restore_for_doc(seg_id, match_key=spec.segment_uid)
                progressed = True
            pending = still
        for spec in pending:   # parents never showed up (missing file, or a broken parent chain)
            self._restore_lost_segments.append(
                f"“{spec.name}” (its parent recording is not in this project)")

    def _apply_restore_for_doc(self, doc_id: str, match_key: str | None = None) -> None:
        """Apply the saved calibrations, layers and results for ONE just-landed document.

        Records are matched by the document's PERSISTENT KEY but applied to its LIVE
        ``doc_id``. The key is DERIVED from the document (``segment.persistent_key``: its
        source path for a file, its uid for a segment) and never defaulted to ``doc_id``,
        because a ``doc_id`` now carries this session's per-load token
        (:mod:`nanoscope.core.identity`) and can therefore never equal anything a previous
        session wrote down. ``match_key`` is still accepted for the segment path, which
        knows its uid before the document exists.

        Order matters: calibrations first (so a restored result's calibration fingerprint
        is computed against the SAME gain/offset it had when saved), then the layers, then
        the results.

        WHICH CHANNEL a record belongs to is decided by ``calibration.resolve_channel``,
        never by its stored index alone. An index is a position: re-export a recording with
        its signal columns in the other order and the index-only restore put a
        ``gain=0.5, offset=-1.0, 'pA'`` calibration on the VOLTAGE channel — the -600 mV
        trace read -301.0 pA — left the real current channel at gain 1.0, reattached an
        11-event detection to the voltage trace, and, because the result's saved
        fingerprint matched the misplaced live calibration, positively CERTIFIED those
        events as current-valid. A v7 record names its channel, so it is re-attached by
        identity when the order moved (said out loud) and NOT applied when the channel is
        gone; a pre-v7 record carries no identity and resolves by index exactly as before,
        silently.

        EVERY INSTALL STAMPS ``_restored_records``. That set, not a list of remembered
        failure reasons, is what :meth:`_capture_unrestorable` subtracts from the project
        to decide what the next save must carry — so a record this method skips for ANY
        reason, including one written next year, is kept rather than erased."""
        from ..analysis.events import EventSet
        from ..analysis.stats import fit_from_json
        from ..core.analysis import ANALYSIS_REGISTRY, AnalysisResult
        from ..core.calibration import resolve_channel
        from ..core.segment import persistent_key

        state = self._restore_state
        if state is None or not self._session.has_document(doc_id):
            return
        doc = self._session.document(doc_id)
        key = match_key if match_key is not None else persistent_key(doc)
        channels = doc.meta.channels
        where = doc.display_name
        from ..core.identity import same_source
        source_key = key
        parents = {segment.segment_uid: segment.parent_key for segment in state.segments}
        visited = set()
        while source_key in parents and source_key not in visited:
            visited.add(source_key)
            source_key = parents[source_key]
        inherited_source = next((ref.source for ref in state.documents
                                 if same_source(ref.path, source_key)), None)

        def _belongs(stored_key: object) -> bool:
            """Is this stored record about the recording that just landed?

            *** RECORDS ARE MATCHED BY RECORDING IDENTITY, NEVER BY PATH SPELLING. ***

            The project stores the path as it resolved AT SAVE TIME; ``key`` is how the same
            file resolves NOW. Those two strings stop being equal the moment the path picks
            up a symlink between sessions — move raw data to a NAS or an archive folder and
            leave a stub behind so every old path still works, which is exactly why people
            leave one. The plain ``==`` this replaces then matched nothing, and every
            calibration, layer and detection saved for that recording was ``continue``d in
            silence while the banner said "The rest of the project was restored."

            :func:`~nanoscope.core.identity.same_source` is THE answer to "are these two
            source keys the same recording?" — it is what ``open_paths``/``_already_open``
            already use so two spellings of one file are one recording, and it asks the
            FILESYSTEM (``os.path.samefile``) rather than transforming strings. It costs
            nothing in the common case: identical strings short-circuit before any syscall.

            A match that needed the filesystem is a match under a DIFFERENT spelling than the
            project holds, and that is said out loud (``_restore_path_moved``) — the same
            treatment a record re-attached by channel name gets. Two surfaces must not
            disagree about which path this recording lives at.
            """
            stored = str(stored_key)
            if stored == key:
                return True
            if not same_source(stored, key):
                return False
            said = f"“{where}” (saved as {stored})"
            if said not in self._restore_path_moved:
                self._restore_path_moved.append(said)
            return True

        def _channel_for(rec, what: str) -> int | None:
            """The live channel index ``rec`` belongs to, or None (and say why, once)."""
            idx, moved = resolve_channel(
                channels, rec.channel,
                getattr(rec, "channel_name", None), getattr(rec, "channel_kind", None))
            if moved is not None:
                sink = (self._restore_channel_moved if idx is not None
                        else self._restore_channel_lost)
                # Once per sentence: a document that was already open reaches this method by
                # a second route (see _note_source_change's own POP for the same reason).
                said = f"“{where}” {what} ({moved})"
                if said not in sink:
                    sink.append(said)
            return idx

        for d in getattr(state, "documents", ()) or ():
            if _belongs(d.path):
                self._restored_records.add(id(d))

        for c in state.calibrations:
            if not _belongs(c.doc_id):
                continue
            ch = _channel_for(c, "calibration")
            if ch is None:
                continue
            # A project is a file on disk and can be edited or damaged, so a non-finite or
            # zero gain can arrive here. ChannelCalibration refuses it (that is the point);
            # the restore keeps the channel's seeded default and says so, rather than aborting
            # the whole reopen over one bad number.
            #
            # `to_calibration` is handed the channel's CURRENT calibration — the sidecar's,
            # loaded moments ago by Session.add_document — because a pre-v5 project carries
            # no `note` and must not be read as asserting there is none. See CalibrationRec.
            try:
                cal = c.to_calibration(self._session.calibration(doc_id, ch))
            except ValueError:
                # display_name, not Path(doc_id).name: a doc_id now carries a per-load token
                # (core.identity) and a segment's path is synthetic, so neither is something
                # to show a user.
                self._restore_bad_calibrations.append(
                    f"{self._session.document(doc_id).display_name} ch{ch}")
                continue
            # write_sidecar=False: a RESTORE is reconstruction, not a calibration edit. Letting
            # it fall through to save_sidecar rewrote (or, at default values, DELETED) the
            # user's <datafile>.npcal.json next to their raw data — destroying its `note`, which
            # a project cannot carry. Restoring a project must never touch anything outside it.
            self._session.set_calibration(doc_id, ch, cal, write_sidecar=False)
            self._restored_records.add(id(c))

        from ..core.acquisition_state import WORKSPACE_KEY, restore_acquisition_overrides
        bandwidth_messages = restore_acquisition_overrides(
            self._session, doc_id, state.workspace.extra.get(WORKSPACE_KEY, []), match_key=key,
            clear_existing=WORKSPACE_KEY in state.workspace.extra)
        for message in bandwidth_messages:
            if message not in self._restore_notes:
                self._restore_notes.append(message)
        # The time base next, still BEFORE layers and results: a restored result's stamps
        # are judged against the rate the document has once the project's declaration is
        # back on it, and a segment re-derived later (_rederive_segments) inherits it.
        from ..core.sample_rate_state import (
            WORKSPACE_KEY as RATE_KEY, restore_sample_rate_declarations,
        )
        for message in restore_sample_rate_declarations(
                self._session, doc_id, state.workspace.extra.get(RATE_KEY, []), match_key=key,
                clear_existing=RATE_KEY in state.workspace.extra):
            if message not in self._restore_notes:
                self._restore_notes.append(message)

        for lay in state.layers:
            if not _belongs(lay.doc_id):
                continue
            ch = _channel_for(lay, "layer")
            if ch is None:
                continue
            layer = self._session.add_layer(doc_id, ch)
            self._session.set_layer_color(layer.layer_id, lay.color, update_registry=False)
            # `excluded` first: it drives visibility, and a project that predates it
            # restores False, leaving `visible` exactly as it was written.
            self._session.set_layer_excluded(layer.layer_id, lay.excluded)
            self._session.set_layer_visible(layer.layer_id, lay.visible and not lay.excluded)
            self._session.set_layer_time_offset(layer.layer_id, lay.time_offset_s)
            self._restored_records.add(id(lay))

        for r in state.results:
            if not _belongs(r.doc_id):
                continue
            ch = _channel_for(r, "detection")
            if ch is None:
                continue
            try:
                events = EventSet(r.events, dict(r.provenance or {}))
            except ValueError:
                continue   # a malformed table is skipped, never aborts the restore
            # The fingerprint the result was COMPUTED under, restored from the project —
            # never re-read from the live calibration. That re-derivation silently declared
            # every reopened result fresh: a detection run at gain 1 and then re-gained to 2
            # came back with the "calibration changed" banner GONE while every depth on
            # screen was still the gain-1 number. A pre-v3 project stored no stamp, so it
            # falls back to the (just-restored) calibration, which is what it meant.
            fp = r.calibration_fingerprint
            if fp is None:
                fp = self._session.calibration(doc_id, ch).fingerprint()
            module = ANALYSIS_REGISTRY.get(r.module_id)
            version = str(r.module_version or (
                getattr(module, "version", "") if module is not None else ""))
            # overlays=() — the SPAN is rebuilt from the (curated) df by the AnalysisPanel;
            # the stored EventSet IS the result, so no detection re-runs.
            # params: the settings the run used, so a re-run reproduces THIS detection
            # instead of whatever the panel happens to be showing.
            # THE OTHER STALENESS STAMP, restored the same way and for the same
            # reason. It travels inside the EventSet's own provenance (the sole
            # writer is `BaselineResult.provenance`, and a project persists the
            # provenance verbatim), so no project VERSION bump is needed and a v1/v2
            # project simply has no key -- which is exactly the "absence IS the
            # signal" case: those rows were measured by the 10 ms block median.
            #
            # Dropping it on restore is the same defect C6 fixed for the calibration
            # stamp, one level worse: not re-deriving it from the live config but
            # losing it entirely, so EVERY reopened project read as "no estimator
            # named" -- indistinguishable from a genuinely pre-change table, and the
            # staleness banner could never clear for a project that was in fact
            # freshly measured.
            res = AnalysisResult(
                module_id=r.module_id, module_version=version, doc_id=doc_id,
                channel=ch, params=dict(r.params or {}),
                calibration_fingerprint=tuple(float(v) for v in fp),
                payload=events, overlays=(), scalars={},
                warnings=tuple(r.warnings or ()),
                source_fingerprint=(r.source_fingerprint if r.source_fingerprint_recorded
                                    else r.source_fingerprint or inherited_source),
                source_fingerprint_recorded=True,
                baseline_fingerprint=(
                    r.baseline_fingerprint
                    if r.baseline_fingerprint is not None
                    else _stored_baseline_fingerprint(r.provenance)))
            self._session.add_result(res)
            self._restored_records.add(id(r))

        from ..core.calibration import channel_identity
        from ..core.table_choice_state import (
            WORKSPACE_KEY as CHOICE_KEY, restore_table_choices,
        )
        if CHOICE_KEY in state.workspace.extra:
            doc_info = self._session.document(doc_id).meta.channels
            for message in restore_table_choices(
                    state.workspace.extra.get(CHOICE_KEY), key,
                    lambda ch, module_id: self._analysis_panel.apply_chosen_table(
                        doc_id, ch, module_id),
                    lambda ch: (channel_identity(doc_info[ch])[0]
                                if 0 <= ch < len(doc_info) else None)):
                if message not in self._restore_notes:
                    self._restore_notes.append(message)
        # ...and the un-run form edits each table was saved with (decision 4), handed to the
        # panel NOW, while the restore still holds the selection back: the arrival that
        # `_select_after_restore` causes is what puts the draft for the table on screen into
        # the form, so the draft has to be there before it.
        from ..core.form_draft_state import (
            WORKSPACE_KEY as DRAFT_KEY, restore_form_drafts,
        )
        if DRAFT_KEY in state.workspace.extra:
            drafts, messages = restore_form_drafts(
                state.workspace.extra.get(DRAFT_KEY), key,
                {int(info.index): channel_identity(info)[0]
                 for info in self._session.document(doc_id).meta.channels})
            for message in messages:
                if message not in self._restore_notes:
                    self._restore_notes.append(message)
            if drafts:
                held = self._analysis_panel.form_drafts()
                held.update({(doc_id, ch, module_id): form
                             for (ch, module_id), form in drafts.items()})
                self._analysis_panel.set_form_drafts(held)

        # v6 -- the derived fits. Both staleness stamps come back from the FILE, never from
        # the live calibration/baseline, for the reason the block above spells out: a
        # re-derived stamp makes a stale fit read as fresh. `curve` is rebuilt through
        # stats.fit_from_json, which reconstructs the density with the same function the
        # fitter used, so a reopened fit draws the curve its own tau/AIC describe.
        for fr in getattr(state, "fits", ()):
            if fr.doc_id != key:
                continue
            ch = _channel_for(fr, "fit")
            if ch is None:
                continue
            if fr.table is not None:
                payload: object = fr.table
            elif fr.curve is not None:
                try:
                    payload = fit_from_json(fr.curve)
                except (KeyError, TypeError, ValueError):
                    continue   # a malformed fit is skipped, never aborts the restore
            else:
                continue
            fp = fr.calibration_fingerprint
            if fp is None:
                fp = self._session.calibration(doc_id, ch).fingerprint()
            self._session.add_result(AnalysisResult(
                module_id=fr.module_id, module_version="", doc_id=doc_id,
                channel=ch, params=dict(fr.params or {}),
                calibration_fingerprint=tuple(float(v) for v in fp),
                payload=payload, overlays=(), scalars={},
                warnings=tuple(fr.warnings or ()),
                source_fingerprint=(fr.source_fingerprint if fr.source_fingerprint_recorded
                                    else fr.source_fingerprint or inherited_source),
                source_fingerprint_recorded=True,
                baseline_fingerprint=fr.baseline_fingerprint))
            self._restored_records.add(id(fr))

        self._note_source_change(doc_id)

    #: "This document was never part of the restore (or its sentence is already said)" —
    #: distinct from ``None``, which means "part of the restore, but the project carries
    #: no fingerprint". Conflating the two is exactly how the unstamped case went silent.
    _NO_RESTORE_SOURCE = object()

    def _note_source_change(self, doc_id: str) -> None:
        """Say, once per reopened FILE, when the events no longer fit the recording — and,
        when a fingerprint proves it, that it is not the file the project was saved from.

        THE POINT IS THE CONSEQUENCE, not the metadata. "The file changed" is a sentence a
        user clicks past; "5 of your 5 saved events now lie past the end of this recording"
        is one they act on, and it is the thing that was actually silent. So the note names
        what differs AND counts the restored rows that can no longer be true.

        COMPOSED BY :func:`~nanoscope.core.identity.describe_restored_source` — the seam
        built (and pinned by ``tests/core/test_s4_identity_restored_source.py``) precisely
        so the two halves are gated separately: the "this file changed" half needs
        EVIDENCE (a saved fingerprint; silence still means "no evidence", never "verified
        identical" — a false alarm on every old project is how a real one gets ignored),
        while the events-past-the-end half is a comparison between the project's own saved
        indices and the sample count now in hand, which needs no stamp at all. This method
        used to gate BOTH on the stamp: it early-returned for a stamp-less project
        (every project written before manifest v4 — MIN_READ_VERSION is frozen at 1 so
        they all still open — and any project whose source could not be read at save
        time), so a re-exported, shorter recording came back with its whole event table
        reattached, rows lying past the end of the file, and not one word anywhere.
        """
        from ..core.identity import describe_restored_source, fingerprint_source
        from ..core.segment import SegmentDocument

        if not self._session.has_document(doc_id):
            return
        doc = self._session.document(doc_id)
        if isinstance(doc, SegmentDocument):
            return   # a segment is re-derived from its parent; it has no file of its own
        # POP, not get: one sentence per file however many times the restore touches it
        # (a document that was already open reaches _apply_restore_for_doc by a second route).
        # The sentinel keeps "already said / not restored" apart from "restored, unstamped":
        # only the former is silence.
        saved = self._restore_sources.pop(doc.source_key, self._NO_RESTORE_SOURCE)
        if saved is self._NO_RESTORE_SOURCE:
            return
        # With no stamp there is nothing a re-read could establish (the seam's contract
        # says callers may skip it); with one, the re-read is the evidence.
        now = fingerprint_source(doc.source_key) if saved is not None else None

        n = int(doc.meta.n_samples)
        rows = past = 0
        for res in self._session.results.for_doc(doc_id):
            df = getattr(getattr(res, "payload", None), "df", None)
            if df is None or "end_idx" not in df:
                continue
            ends = df["end_idx"].to_numpy()
            rows += int(len(df))
            past += int((ends > n).sum())
        note = describe_restored_source(
            doc.display_name, saved, now,
            loaded_n_samples=n, restored_events=rows, events_past_end=past)
        if note:
            self._restore_notes.append(note)

    def _capture_unrestorable(self, state) -> None:
        """Remember every record in the OPENED project that the restore did NOT install, so
        that the next save writes it back out instead of dropping it.

        COMPUTED AS A DIFF — the project minus what was actually installed — and that is
        the whole fix, not a tidier spelling of the old one. It used to be
        ``lost = set(self._restore_missing)``: ONE enumerated reason (the file was not on
        disk), while the restore has at least seven exits. Measured on two of them:

        * A segment whose saved window no longer fits its parent. The banner names it; the
          carry did not. Ctrl+S then wrote ``segments=[]``, ``results=[]`` and reported
          *"Saved project (0 result(s))"* — the segment's definition and four curated
          events gone from the file permanently, unrecoverable even by restoring the
          original recording from backup, because the manifest no longer describes them.
        * A record naming a channel the reopened file no longer has. Both the calibration
          and an 11-event detection were ``continue``d in SILENCE, and the next save
          dropped both while the banner mentioned only the file's byte size.

        The others (a malformed EventSet, a non-finite stored calibration, unparseable fit
        JSON, and the four channel guards separately) were the same shape, and enumerating
        reasons is what has come back on this project three rounds running. So the question
        asked here is not "why might a record be missing" but "which records are missing",
        and the answer needs no maintenance: :meth:`_apply_restore_for_doc` and
        :meth:`_rederive_segments` stamp ``_restored_records`` at the line that installs,
        and everything else in the project is carried. A skip path written next year is
        covered on the day it is written; so is a record TYPE nobody here remembered,
        because the fields are walked off the dataclass rather than listed.

        The old transitive fixpoint for nested segments is gone with it, and its
        disappearance is a check on the shape: a segment whose parent never loaded is never
        installed, so it — and its results, and anything nested under it — is carried by
        construction rather than by a rule that had to know about nesting.

        MEASURED BEFORE: a project referencing two recordings, one of whose files was not
        readable at open time, restored one document and warned about the other in the info
        bar — correctly. Then Ctrl+S rewrote the file from the LIVE session, which contains
        only what loaded: documents 2 -> 1, results 2 -> 1, status line "Saved project
        (1 result(s))", no dialog. The second recording's reference, its calibration, its
        layer state and its whole curated detection were gone from the project, and the
        commonest cause of the original failure — an unmounted drive, a network volume, a
        file moved for a moment — is exactly the case where nothing was actually wrong with
        the work. Recovery had been "put the file back and reopen"; after that save there
        was nothing left to reopen.

        A save may not decide that unreadable means unwanted. What could not be loaded is
        carried verbatim: it was read from this very file, so writing it back is lossless,
        and if the recording returns the project restores whole.

        Segments ride along too, and transitively: a segment whose parent could not be
        loaded could not be re-derived either, so it and anything nested under it are kept
        as specs. Carrying the parent reference is also what keeps them RESTORABLE — the
        save-time drop check resolves a segment against the documents in the project, and
        the parent is now still there.
        """
        from dataclasses import fields as _dc_fields, is_dataclass

        from ..core.project import stamp_legacy_result_sources
        stamp_legacy_result_sources(state)
        installed = self._restored_records
        carried: dict[str, list] = {}
        for f in _dc_fields(state):
            values = getattr(state, f.name, None)
            if not isinstance(values, list):
                continue
            # `is_dataclass` is the definition of "a project RECORD", not a shortcut: every
            # persisted record is one, and the only other list on a ProjectState is
            # `label_vocabulary`, which is a plain value the restore adopts WHOLESALE
            # (LabelVocabulary.from_json) rather than record by record. Carrying its entries
            # would duplicate them on the next save.
            missing = [v for v in values if is_dataclass(v) and id(v) not in installed]
            if missing:
                carried[f.name] = missing
        self._carried_over = carried or None

    def _select_after_restore(self, state) -> None:
        """Pick the (doc, channel) to land on after a restore: a restored detection if
        there is one (so its events + spans show immediately), else a document's default
        channel.

        PREFERRING ONE THAT IS ALREADY DRAWN is the whole change here, and it is what keeps
        the saved view intact. The restored layer set is the user's — it may be a deliberate
        multi-recording overlay — so the honest first move is to analyse something the
        project chose to show, not to redecorate. Only when NO drawn recording carries a
        detection does the target have to be brought on screen, and then it is added to the
        view (never swapped in for what is there) and said out loud, because the project's
        saved view genuinely did not include it.

        ``r.doc_id`` here is the LIVE per-load id ``_apply_restore_for_doc`` stamped onto the
        restored result, so ``has_document`` is asking about this very load — see
        :mod:`nanoscope.core.identity`.
        """
        from .file_dock import _default_channel

        shown = self._session.shown_doc_ids()
        live = [r for r in self._session.results.all()
                if self._session.has_document(r.doc_id) and getattr(r.payload, "df", None)
                is not None]
        target: tuple[str, int] | None = None
        has_detection = False
        for r in live:                                 # a detection ON SOMETHING DRAWN
            if r.doc_id in shown:
                target, has_detection = (r.doc_id, r.channel), True
                break
        if target is None and live:                    # ...else any restored detection
            target, has_detection = (live[0].doc_id, live[0].channel), True
        if target is None:                             # ...else a drawn document, else any
            docs = self._session.documents()
            doc = next((d for d in docs if d.doc_id in shown), docs[0] if docs else None)
            if doc is not None:
                target = (doc.doc_id, _default_channel(doc))
        if target is None:
            return
        self._file_dock.focus_document(target[0])
        self._on_selection(*target)      # _ensure_target_drawn runs inside
        if target[0] not in shown and self._session.has_document(target[0]):
            what = ("carries this project's detection but was not"
                    if has_detection else "was not")
            self._restore_notes.append(
                f"“{self._session.document(target[0]).display_name}” {what} among the "
                f"recordings this project was saved showing, so it has been added to the "
                f"view — the app never analyses a recording it is not drawing.")

    # -- recent projects (QSettings, mirroring recent files) -------------------------------------
    def _add_recent_project(self, path: str) -> None:
        p = str(Path(path).resolve())
        if p in self._recent_projects:
            self._recent_projects.remove(p)
        self._recent_projects.insert(0, p)
        del self._recent_projects[_MAX_RECENT:]
        self._rebuild_recent_projects_menu()

    def _rebuild_recent_projects_menu(self) -> None:
        self._recent_projects_menu.clear()
        if not self._recent_projects:
            empty = self._recent_projects_menu.addAction("(none)")
            empty.setEnabled(False)
            return
        for p in self._recent_projects:
            act = self._recent_projects_menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(lambda _checked=False, path=p: self._open_project(path))
        self._recent_projects_menu.addSeparator()
        clear = self._recent_projects_menu.addAction("Clear recent projects")
        clear.triggered.connect(self._clear_recent_projects)

    def _clear_recent_projects(self) -> None:
        self._recent_projects = []
        self._rebuild_recent_projects_menu()
