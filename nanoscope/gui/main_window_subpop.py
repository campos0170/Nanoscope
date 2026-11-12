"""The subpopulation report of ``MainWindow``: its window, the assignment drawn on the trace and
the gallery, the state the report window is told about, and the report export.

The run itself (``_run_subpopulations``, ``_on_subpop_finished`` and ``_on_subpop_stopped``)
stays in ``main_window.py``, where a test reads its source.

``_SubpopMixin`` is a plain class listed before ``QMainWindow`` in the bases of ``MainWindow``,
with no ``__init__`` and no state of its own. The method bodies were moved verbatim out of
``main_window.py`` by a mechanical split; nothing in them was edited.

``ask_export_path`` is imported from ``main_window_seams``, whose forwarder looks the name up on
``main_window`` each time it is called, so a test's patch of ``main_window.ask_export_path``
still reaches these methods.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
from pathlib import Path

from PySide6 import QtWidgets

from ..analysis.reconcile import CONSENSUS_MODULE_ID
from ..core.identity import source_of
from ..core.analysis_workspace import SUBPOPULATION_ARTIFACT_ID
from .analysis_explorer import PAGE_SUBPOPULATIONS
from .main_window_seams import ask_export_path


class _SubpopMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    def _ensure_subpop_window(self):
        """Build the report view without installing a report or showing a window."""
        return self._open_figure_kind(PAGE_SUBPOPULATIONS)

    def _open_subpop_window(
            self, report, *, source_result=None, source_recipe=None,
            accepted_event_ids=None) -> None:
        """Open a newly completed report and checkpoint it as a durable result."""
        if self._figure_workspace is not None:
            token = self._subpop_figure_request
            key = token[1] if token is not None else self._figure_workspace.most_recent(PAGE_SUBPOPULATIONS)
            if key is None:
                key = self._figure_workspace.create(PAGE_SUBPOPULATIONS)
            return self._install_subpop_figure_report(key, report, source_result=source_result,
                source_recipe=source_recipe, accepted_event_ids=accepted_event_ids)
        self._workspace_pending.discard(SUBPOPULATION_ARTIFACT_ID)
        self._workspace_frozen.discard(SUBPOPULATION_ARTIFACT_ID)
        self._ensure_subpop_window()
        # The adapter turns this live result into a persistent source/result recipe.  It
        # never serializes the object itself; keeping it on the widget only bridges the
        # synchronous capture immediately below.
        self._subpop_window._nanoscope_workspace_source_result = source_result
        self._subpop_window._nanoscope_workspace_source = source_recipe
        self._subpop_window._nanoscope_workspace_accepted_event_ids = (
            accepted_event_ids)
        self._subpop_assignment_figure_id = self._figure_workspace.id_for_page(
            self._subpop_window)
        self._subpop_window.set_report(
            report, subtitle=f"{report.meta.get('recording') or '(unnamed)'}")
        # ``set_report`` resets the window's live state (a new document, no claim about
        # any assignment); the truth about THIS report's assignment is this window's to
        # push, and it is pushed right away so the two entry buttons never open live over
        # a colouring that was refused or has already gone (S7-POP3).
        self._sync_subpop_window()
        self._on_workspace_changed(SUBPOPULATION_ARTIFACT_ID)
        self._show_analysis_page(
            PAGE_SUBPOPULATIONS, self._subpop_window,
            refresh=self._run_subpopulations)

    def _show_saved_subpop_report(self) -> None:
        """Show the project-owned report snapshot without dispatching an analysis job."""
        return self._open_figure_kind(PAGE_SUBPOPULATIONS)

    # -- item 13: the report's events, on the trace and in the gallery --------------------------
    def _adopt_subpop_assignment(
            self, report, target, *, require_current_selection: bool = True) -> str:
        """Read the per-event subpopulation ids off a finished report and push them out.

        ``target`` is the detection the run was LAUNCHED on, handed in by the landing slot
        rather than looked up here — see below.

        THE REPORT IS READ ONCE, HERE, AND NOTHING IS RECOMPUTED. ``report.event_table()`` is
        the report's own join of its feature table with the mixture's labels -- the same table
        the export writes -- so the ids the trace and the gallery colour by are, by
        construction, the ids the report printed. Deriving them here from
        ``quantification.fit.labels`` and ``features['event_id']`` would be a second
        implementation of that join living in a window.

        Absent when the mixture was refused: ``event_table`` has no ``subpop`` column at all,
        which is exactly the case the report prints as "NO MIXTURE". The colouring is then
        dropped rather than faked from the exploratory labels -- HDBSCAN's clusters are a
        different question with a different answer (on zn_30h, two clusters and 229 noise
        points where the mixture reports four classes) and are not what "subpop 2" means.
        """
        import numpy as np
        import pandas as pd

        from ..analysis.events import EventSet

        self._drop_subpop_assignment()
        table = report.event_table()
        if "subpop" not in table.columns or "event_id" not in table.columns:
            return ""      # "NO MIXTURE" — the headline already says it; nothing to add
        ids = table["event_id"].to_numpy()
        labels = table["subpop"].to_numpy()
        assignment = {int(e): int(v) for e, v in zip(ids, labels, strict=True)
                      if not pd.isna(v)}
        if not assignment:
            return ""
        # *** THE MAP IS ADOPTED ONTO THE DETECTION IT WAS COMPUTED FROM, OR NOT AT ALL. ***
        #
        # This used to re-read `current_detection()` — whatever happened to be SELECTED when
        # the report landed — and install the report's event ids onto that frame. Click
        # another recording in the File dock during the minute the mixture takes and 120
        # event ids computed on A were attached to B's 40-row frame: 40 of B's events cut
        # into "subpop 0" / "subpop 1" / "not in the report" by event-id NUMBER collision,
        # while the gallery caption and the trace status line still named A. A fabricated
        # per-event class assignment on a recording that was never analysed, attributed in
        # writing to a different file. `_refresh_subpop_views` could not clean it up either,
        # because the frame it compares against had already been overwritten with B's.
        #
        # `_subpop_target` is the detection the run was LAUNCHED on (see
        # :meth:`_run_subpopulations`). It is still valid only if it is still the stored
        # result for its recording — a re-detect or a merge replaces the object, and then
        # the ids describe events that no longer exist under those ids. Anything else and
        # the colouring is dropped WITH A WORD, never silently re-homed.
        if target is None or not isinstance(target.payload, EventSet):
            return ""
        # Every refusal below is ALSO recorded as the reason there is no assignment, so
        # the report window's two buttons can repeat it instead of the only other sentence
        # they used to know ("fitted no mixture" — false for every one of these).
        if not self._session.has_document(target.doc_id):
            self._subpop_absent_reason = (
                f"Nothing was coloured: “{Path(source_of(target.doc_id)).name}” was "
                f"closed while the run was in flight.")
            return self._subpop_absent_reason
        live = self._session.results.get(
            target.doc_id, target.channel, target.module_id)
        if live is not target:
            self._subpop_absent_reason = (
                "Nothing was coloured: the detection this was computed from has been "
                "replaced since the run started, so its event ids no longer name the "
                "same events.")
            return self._subpop_absent_reason
        # ...and the colouring only goes ON SCREEN if the screen is showing that detection.
        # The trace spans and the gallery tiles colour by matching the CURRENT frame's
        # event ids against this map, so pushing it while another recording is selected is
        # the same id-collision, one step later. Same rule the sibling surface keeps.
        if (require_current_selection
                and self._analysis_panel.current_detection() is not target):
            self._subpop_absent_reason = (
                f"Nothing was coloured on the trace or in the gallery: you moved off "
                f"“{Path(source_of(target.doc_id)).name}” while the run was in flight. "
                f"Select it again and re-run to colour its events.")
            return self._subpop_absent_reason
        res = target
        self._subpop_assignment = assignment
        self._subpop_frame = res.payload.df
        # The store key the frame lives under, so a later navigation can ask "is the
        # detection this was fitted on still the stored one?" without re-deriving it from
        # whatever is selected then (see _refresh_subpop_views).
        self._subpop_key = (str(res.doc_id), int(res.channel), str(res.module_id))
        self._subpop_n_fired = int(report.n_fired)
        # ...and the accepted SET behind that count, so the curation note can compare sets
        # and not counts (C2). CONSUMED FROM THE LAUNCH, never rebuilt from the frame here:
        # the frame is edited in place by curation and this is minutes later, so a rebuild
        # would snapshot the rows as they are NOW and call them the rows the run was handed
        # — which is how an in-flight curation used to silence the very line that exists to
        # disclose it (measured: n_fired 110, 109 accepted, note "").
        #
        # And it is installed only if it can be SQUARED with the count the report printed.
        # The two halves are one claim; a set whose size is not ``n_fired`` does not
        # describe the rows the report describes, whatever it does describe, so it is
        # dropped and the note falls back to the count — which still fires on that
        # disagreement. Never a set the line would compare against and read equal.
        _run_ids = self._subpop_accepted_ids_pending
        self._subpop_accepted_ids_pending = None
        self._subpop_accepted_ids = (
            _run_ids if _run_ids is not None and len(_run_ids) == int(report.n_fired)
            else None)
        self._subpop_source = str(report.meta.get("recording", "") or "")
        self._subpop_caveats = tuple(report.warnings)
        # THE REPORT IS READ ONCE, HERE (see above), and its own sentence about what it
        # found is part of what is read: ``_colour_trace_by_subpops`` speaks about this
        # assignment long after the report object has gone, and without this it had no
        # way to say what the classes are worth except by counting them itself — which
        # is exactly the claim ``headline()`` exists to be the only author of.
        self._subpop_headline = report.headline()
        self._subpop_signal = self._subpop_signal_pending
        self._subpop_signal_pending = None
        if self._subpop_signal is not None:
            # A trace of the wrong length would slice tiles at the wrong samples. It cannot
            # normally happen (the run and the channel are the same recording), so it is
            # dropped rather than clipped: a silently shifted picture is worse than none.
            # The same for a channel that has gone away under us -- the assignment is still
            # good, only the alternative trace is not, and losing the whole colouring
            # because a cosmetic extra could not be validated would be the wrong trade.
            try:
                n_channel = len(self._session.calibrated(res.doc_id, res.channel).data)
            except Exception:  # noqa: BLE001 -- cosmetic; never breaks the adoption
                n_channel = -1
            if np.asarray(self._subpop_signal).size != n_channel:
                self._subpop_signal = None
        self._push_subpop_assignment()
        return ""

    def _subpop_installed(self) -> bool:
        """Is the held assignment about the detection ON SCREEN? — frame identity, the
        one test the sibling surfaces already use to tell "the frame was edited" from
        "there is a new frame". False while another recording (or another detector's
        result) is selected; the map is then HELD, not dropped."""
        from ..analysis.events import EventSet

        if self._subpop_assignment is None:
            return False
        res = self._analysis_panel.current_detection()
        df = res.payload.df if (res is not None and isinstance(res.payload, EventSet)) else None
        return df is not None and df is self._subpop_frame

    def _push_subpop_assignment(self) -> None:
        """Hand the assignment to the trace spans and the gallery — or its ABSENCE, which
        is what they get whenever the detection on screen is not the one it was fitted
        on. Both surfaces colour by matching the CURRENT frame's event ids against the
        map, so a held map pushed onto another recording's frame is the id-collision
        `_adopt_subpop_assignment` refuses, one step later. Idempotent on both surfaces."""
        from .event_gallery import TRACE_DETRENDED

        shown = self._subpop_assignment if self._subpop_installed() else None
        if self._analysis_panel.subpop_labels() != shown:
            self._analysis_panel.set_subpop_labels(shown)
        if self._event_gallery is not None:
            gallery = self._event_gallery
            # The detrended trace is offered with the map and withdrawn with it; offering
            # it again re-applies the tile trace, so only the transitions are pushed.
            signal = self._subpop_signal if shown is not None else None
            offered = gallery.trace_box.findText(TRACE_DETRENDED) >= 0
            had_map = gallery.subpopulations() is not None
            # C4: the map is leaving this gallery — take the user's view of it with it,
            # while the widgets still carry it. A push is only the SECOND place the view
            # can be lost (the re-feed in ``_refresh_event_gallery`` is the first and
            # usually gets there earlier), so the remembering is one method used by both
            # and it refuses to read a gallery that has already been reset.
            if shown is None and had_map:
                self._remember_gallery_subpop_view()
            if (signal is not None) != offered:
                gallery.set_detrended_signal(signal)
            if gallery.subpopulations() != shown:
                gallery.set_subpopulations(
                    shown,
                    source=self._subpop_source if shown is not None else "",
                    caveats=self._subpop_caveats if shown is not None else ())
            # C4: and put that view back the moment the map arrives on a gallery that did
            # not have it — the return leg of the round trip. A fresh adoption remembers
            # nothing, so the report window's "Event gallery…" button keeps choosing the
            # trace and the sample itself (see :meth:`_open_gallery_on_subpops`).
            if shown is not None and not had_map:
                self._restore_gallery_subpop_view()

    def _remember_gallery_subpop_view(self) -> None:
        """Remember what the Event gallery is DRAWING the held map with, for the return.

        The one writer of :attr:`_subpop_gallery_trace` /
        :attr:`_subpop_gallery_representative`. Both are the CURATOR'S choices — the report
        window's "Event gallery…" turns on the detrended trace and the representative
        sample, and the curator may change either — and leaving the recording resets them:
        ``set_events`` drops the map (and with it the representative sample) and
        ``set_detrended_signal(None)`` falls the trace box back to "Calibrated (as
        recorded)". Measured on the owner's project before this existed: away to 0005 and
        back to 0006 returned the chips and the classes with the tiles drawn from the
        calibrated trace and the sample off, and the status line stated the new state as
        if it had always been so (C4).

        READ ONLY WHILE THE GALLERY STILL CARRIES THE MAP. The re-feed runs first
        (``_refresh_event_gallery`` before ``_refresh_subpop_views``) and has already put
        the representative box back to False by the time the push withdraws, so a capture
        taken then would remember exactly the reset it exists to undo — measured: the
        trace was remembered correctly and the sample came back off.
        """
        gallery = self._event_gallery
        if (gallery is None or self._subpop_assignment is None
                or gallery.subpopulations() is None):
            return
        self._subpop_gallery_trace = gallery.trace_name()
        self._subpop_gallery_representative = gallery.is_representative()

    def _restore_gallery_subpop_view(self) -> None:
        """Put the remembered trace + representative sample back (see the method above).

        Idempotent and defensive: a trace the gallery is not offering is not forced, and
        neither widget is touched when it already reads the remembered value, so the
        restore cannot bounce the tiles a second time.
        """
        gallery = self._event_gallery
        if gallery is None:
            return
        name = self._subpop_gallery_trace
        if name and gallery.trace_name() != name and gallery.trace_box.findText(name) >= 0:
            gallery.set_trace(name)
        if self._subpop_gallery_representative and not gallery.is_representative():
            gallery.set_representative(True)

    def _drop_subpop_assignment(self) -> None:
        """Forget the assignment everywhere. Idempotent."""
        self._subpop_assignment = None
        self._subpop_assignment_figure_id = None
        self._subpop_retired_figure_id = None
        self._subpop_frame = None
        self._subpop_key = None
        self._subpop_n_fired = None
        self._subpop_accepted_ids = None
        self._subpop_gallery_trace = None
        self._subpop_gallery_representative = False
        self._subpop_absent_reason = ""
        self._subpop_source = ""
        self._subpop_caveats = ()
        self._subpop_headline = ""
        self._subpop_signal = None
        self._push_subpop_assignment()

    def _retire_subpop_assignment(self, why: str) -> None:
        """Drop the assignment because the detection it describes is gone, and SAY SO —
        on the status bar now, and in the report window for as long as that report
        stands (its two buttons then repeat this reason; see :meth:`_subpop_state`)."""
        figure_id = self._subpop_assignment_figure_id
        self._drop_subpop_assignment()
        self._subpop_retired_figure_id = figure_id
        self._subpop_absent_reason = why
        self._sync_subpop_window()
        self.statusBar().showMessage(why, 10000)

    def _refresh_subpop_views(self) -> None:
        """Keep the colouring for as long as the detection it describes still stands.

        Reached from :data:`SURFACE_MUTATION_SIGNALS` and from the retarget, i.e. from every
        edit and every navigation. Curation, labelling and a mask drag all write the SAME
        frame in place, so the map survives them and the colours track the edit. A re-detect,
        a merge, a re-measure or a project open REPLACES the stored frame, and then the map
        is about events that no longer exist under those ids -- so it goes, WITH ITS REASON,
        rather than colouring one detection with another's mixture.

        TWO THINGS THIS DOES NOT RETIRE, both measured as false retirements in sweep 7's
        fix batch. (1) An Edit ▸ Undo/Redo of a plain CURATION: it installs a snapshot copy
        of the same events with one accept flag moved, which the forward edit had already
        left the map installed through, so the map is re-bound to the copy (X6; see
        :meth:`_is_curation_only_change`). (2) Another DETECTOR's run on this channel: it
        is filed under its own module id and never reaches this frame, so nothing here
        fires at all and the map is simply not on screen -- :meth:`_subpop_state` calls
        that "away" and says which detector is showing instead (C1).

        *** NAVIGATION IS NOT A STATEMENT ABOUT THE MIXTURE. *** This used to compare the
        held frame against whatever detection was on screen and drop the map on any
        mismatch -- and nulled the frame anchor with it, so the "same frame" branch below
        could never fire again. Sweep 7 measured the cost on the owner's own project:
        click 0005's row, click straight back to 0006, and a finished 236-event mixture
        was gone -- assignment None, gallery combo empty, both report buttons still live
        and giving a false reason -- although the frame that came back was the SAME
        OBJECT (``k3_frame_identity``: id 13820809488 before and after). The question is
        now asked of the ResultStore, under the key the map was adopted on: is the stored
        detection for (doc, channel, module) still the frame this map describes? While
        it is, the map is held -- pushed onto the surfaces when that frame is on screen,
        withdrawn from them (not forgotten) while another recording is -- exactly as the
        CUSUM level fit is retired only when ITS detection is replaced and re-adopted
        "after switching away and back" (see _sync_export_action).
        """
        from ..analysis.events import EventSet

        if self._subpop_assignment is None:
            return
        key = self._subpop_key
        if key is None or not self._session.has_document(key[0]):
            self._retire_subpop_assignment(
                "This report's assignment was retired when the recording it was fitted "
                "on was closed — re-run Analysis ▸ Generate Subpopulations on the recording "
                "you want coloured.")
            return
        stored = self._session.results.get(*key)
        df = (stored.payload.df
              if stored is not None and isinstance(stored.payload, EventSet) else None)
        if df is not self._subpop_frame:
            # X6: an Edit ▸ Undo/Redo of a plain curation installs a SNAPSHOT COPY of the
            # frame (core.history.EventHistory._move), a new object with the same event
            # ids, the same spans and the same depths — only ``accepted``/``flag`` moved.
            # Frame identity breaks, but the map is event_id → class and is exactly as
            # valid on that copy as on the frame it survived the forward curation on, so
            # it is RE-BOUND to the new object rather than retired. ``describe_change`` is
            # the same author the Edit menu uses to name the step ("Undo Curate event"),
            # so "a curation was undone" and "the map may stay" come from one place and
            # cannot drift. Anything it does not NAME a curation — a merge, a re-detect
            # (even one that returned the same table), a mask resize, a re-measure — still
            # retires the map: those change what the ids mean, and colouring one detection
            # with another's mixture is the hazard this guard exists for.
            if df is not None and self._is_curation_only_change(self._subpop_frame, df):
                self._subpop_frame = df
            else:
                # The wording names what actually retires the map, and nothing else (C1):
                # curation and its undo do not (above), and neither does another
                # DETECTOR's run on this channel — that result is filed under its own
                # module id, leaves this one's frame alone, and parks the map "away"
                # (see :meth:`_subpop_state`). An ML run reaches this line only when the
                # map was fitted on the ML detector's own table and a new one replaced it,
                # which is the same fact as "re-detected".
                self._retire_subpop_assignment(
                    "This report's assignment was retired when the detection changed (a "
                    "re-detect, a merge, a mask edit, a re-measure, or an undo past the "
                    "detection) — re-run Analysis ▸ Generate Subpopulations to colour the "
                    "events on screen.")
                return
        # The detection stands. On screen: offer the map (the gallery may simply have
        # been opened since). Off screen: withdraw it from the surfaces and keep it.
        self._push_subpop_assignment()
        self._sync_subpop_window()

    #: Short human name for each detector, for the report window's "held" sentence when a
    #: DIFFERENT detector's result is on screen. The three the app runs; anything else
    #: reads back its own module id rather than a guess.
    _DETECTOR_DISPLAY_NAMES = {
        "detect.schmitt": "Schmitt",
        "detect.cusum": "CUSUM",
        "detect.keras": "ML",
        "detect.study": "Study candidates",
        CONSENSUS_MODULE_ID: "Reconciled",
    }

    @classmethod
    def _detector_display_name(cls, module_id) -> str:
        """The short name a "held" sentence names a detector by (see :meth:`_subpop_state`)."""
        return cls._DETECTOR_DISPLAY_NAMES.get(str(module_id), str(module_id))

    #: The four things the report window can be told about its assignment. Spelled here
    #: as the window spells them (``subpop_window.ASSIGNMENT_*``; a test pins the two
    #: spellings equal) because this file imports that window lazily, on first use.
    _SUBPOP_NONE, _SUBPOP_INSTALLED, _SUBPOP_AWAY, _SUBPOP_RETIRED = (
        "none", "installed", "away", "retired")
    _SUBPOP_READY = "ready"

    def _subpop_state(self) -> tuple[str, str]:
        """``(state, reason)`` — the ONE resolver of what the report's assignment is doing,
        read by the two report-window buttons, by their gate and by the window's standing
        line, so they cannot disagree about it.

        ``installed``: the map is on the trace and the gallery (reason ""). ``away``: the
        map is held but the detection on screen is another one. ``retired``: the map is
        gone for the reason given (the detection changed, the recording closed, or the
        adoption was refused). ``none``: the report fitted no mixture, so there was never
        anything to colour with — the one case the old sentence was true for.
        """
        if self._subpop_assignment is not None:
            if self._subpop_installed():
                return self._SUBPOP_INSTALLED, ""
            key = self._subpop_key
            name = self._subpop_source or (
                Path(source_of(key[0])).name if key is not None else "its recording")
            where = f"“{name}”" + (f" (channel {key[1]})" if key is not None else "")
            # C1/C-SUB-2: there are TWO ways a map is held-but-not-shown, and they need
            # different words. When the recording+channel it was fitted on IS the one on
            # screen, only the DETECTOR differs: its result was fitted on (say) Schmitt and
            # a CUSUM or ML run is now active on the same channel. The store keeps one
            # result per detector, so this map's own frame is untouched and nothing was
            # retired — but telling the user to "select that recording again in the File
            # Manager" is a dead end, because the recording is already selected and
            # re-selecting shows the same other-detector result. That case gets its own
            # sentence, naming the recovery that actually works.
            res = self._analysis_panel.current_detection()
            on_same_target = (
                key is not None and res is not None
                and str(res.doc_id) == key[0] and int(res.channel) == key[1])
            if on_same_target:
                return self._SUBPOP_AWAY, (
                    f"This report describes the {self._detector_display_name(key[2])} "
                    f"detection of {where}; the "
                    f"{self._detector_display_name(res.module_id)} detection is on screen "
                    f"now, so its colouring is held, not shown. Re-run Analysis ▸ Generate "
                    f"Subpopulations to colour the detection on screen; nothing was retired.")
            # A DIFFERENT recording is on screen: the map comes back the moment its own
            # recording is selected again. The dock the user picks it in is titled "File
            # Manager" (X-SURF-12: there is no widget called "Files").
            return self._SUBPOP_AWAY, (
                f"This report describes {where}, which is not the detection on screen — "
                f"its colouring is held, not shown, until you select that recording again "
                f"in the File Manager; nothing was retired.")
        if self._subpop_absent_reason:
            return self._SUBPOP_RETIRED, self._subpop_absent_reason
        return self._SUBPOP_NONE, (
            "That report fitted no mixture, so there are no subpopulations to colour "
            "the trace with — see the report text.")

    def _subpop_curation_note(self) -> str:
        """S7-POP4: the standing sentence for the report window when the ACCEPTED SET has
        moved under a standing report, "" while it has not.

        The map survives curation by design (the frame is edited in place; see
        :meth:`_refresh_subpop_views`), and the report's fractions, rates and CIs stay
        the run's — measured on the owner's 0006: reject 66 of class 0 and the header,
        the population table and the gallery combo all stood byte-identical while a
        re-run of the same analysis returned three classes instead of two. Nothing here
        recomputes anything; it states which set the numbers describe and how far the
        accepted set is from it now.

        The comparison is of the accepted SET, not its size (C2): rejecting one event and
        accepting another back leaves the count equal while the set is no longer the run's
        (measured on 0004: event 3 accepted now was not in the run, event 7 in the run is
        rejected now), and a count read equal on exactly that swap and cleared the one line
        that exists to catch it. The COUNT is still the fallback — for a frame with no
        ``event_id`` column, and for a run whose set was never captured — because a line
        that is only sometimes right is still better here than one that is silently absent.

        BOTH HALVES ARE THE LAUNCH'S. ``_subpop_accepted_ids`` is snapshotted in
        :meth:`_run_subpopulations` off the ``EventSet`` the job is handed, which is the
        instant ``n_fired`` counts; it was rebuilt from the live frame at adoption, and
        because a mixture takes a minute or many and the app invites the user to keep
        curating while it runs, an in-flight rejection landed inside the snapshot and the
        line compared the current set with itself — measured: the report says 110 accepted,
        109 are, and this returned "" (C2 close).
        """
        return self._subpop_curation_description(
            self._subpop_frame, self._subpop_n_fired, self._subpop_accepted_ids)

    @staticmethod
    def _subpop_curation_description(frame, n_run, run_ids) -> str:
        """Describe the current accepted set against this specific report's run."""
        if frame is None or n_run is None or "accepted" not in frame.columns:
            return ""
        acc = frame["accepted"].to_numpy().astype(bool)
        if "event_id" in frame.columns and run_ids is not None:
            now_ids = frozenset(int(e) for e in frame.loc[acc, "event_id"])
            n_now, unchanged = len(now_ids), now_ids == frozenset(int(e) for e in run_ids)
        else:
            n_now = int(acc.sum())
            unchanged = n_now == n_run
        if unchanged:
            return ""
        return (f"accepted now: {n_now} of {len(frame)}; this run's fractions describe "
                f"the {n_run} accepted at run time — re-run to update")

    def _sync_subpop_window(self) -> None:
        """Push the assignment's state + reason and the curation note onto the report
        window, which gates its two entry buttons on the state and prints both lines."""
        win = self._subpop_window
        if win is None:
            return
        figure_id = getattr(win, "_nanoscope_figure_id", None)
        report = win.report()
        if figure_id is not None and report is not None:
            target, reason = self._subpop_report_source(win)
            if target is None:
                if figure_id == getattr(self, "_subpop_retired_figure_id", None):
                    reason = self._subpop_absent_reason or reason
                win.set_assignment_state(self._SUBPOP_RETIRED, reason)
                win.set_curation_note("")
                return
            # Bind a restored report to this validated live object before a later
            # byte-identical re-detection can masquerade as its original source.
            win._nanoscope_workspace_source_result = target
            win.set_curation_note(self._subpop_curation_description(
                target.payload.df, report.n_fired,
                getattr(win, "_nanoscope_workspace_accepted_event_ids", None)))
            if report.quantification is None:
                win.set_assignment_state(
                    self._SUBPOP_NONE, "This report fitted no mixture, so it has no classes to show.")
                return
            key = (str(target.doc_id), int(target.channel), str(target.module_id))
            held = (self._subpop_assignment_figure_id == figure_id
                    and self._subpop_key == key
                    and self._subpop_frame is target.payload.df
                    and self._subpop_assignment is not None)
            if not held:
                name = self._session.document(target.doc_id).display_name
                detector = self._detector_display_name(target.module_id)
                if self._analysis_panel.current_detection() is target:
                    state = self._SUBPOP_READY
                    reason = ("This report's original detection is selected. Use Event gallery… "
                              "or Colour the trace to apply its stored classes.")
                else:
                    state = self._SUBPOP_AWAY
                    reason = (f"This report belongs to “{name}”, channel {target.channel}, {detector}. "
                              "Select that original detection to show its events.")
                win.set_assignment_state(state, reason)
                return
        state, reason = self._subpop_state()
        win.set_assignment_state(state, reason)
        win.set_curation_note(self._subpop_curation_note())

    def _open_gallery_on_subpops(self) -> None:
        """Report window ▸ "Event gallery…" — open the gallery already carrying the report.

        Only when the assignment is INSTALLED. Any other state posts the true reason and
        opens nothing: a plain gallery with no chips and no sample, opened from a button
        that promised the report's classes, is the silent failure S7-POP3 measured.
        """
        from .event_gallery import TRACE_DETRENDED

        state, reason = self._subpop_state()
        if state != self._SUBPOP_INSTALLED:
            self.statusBar().showMessage(reason, 10000)
            return
        self._open_event_gallery()
        self._push_subpop_assignment()
        if self._subpop_signal is not None:
            self._event_gallery.set_trace(TRACE_DETRENDED)
        self._event_gallery.set_representative(True)

    def _colour_trace_by_subpops(self) -> None:
        """Report window ▸ "Colour the trace" — shade the spans by class and come forward.

        The refusal names the TRUE state (see :meth:`_subpop_state`). It used to print
        "That report fitted no mixture" for every case where the map was absent — after
        a merge, on the recording the report was fitted on, two lines under a header
        saying "1 subpopulation(s)" (S7-POP3).
        """
        state, reason = self._subpop_state()
        if state != self._SUBPOP_INSTALLED:
            self.statusBar().showMessage(reason, 10000)
            return
        self._push_subpop_assignment()
        self.raise_()
        self.activateWindow()
        # The class count is NOT recomputed here. ``{v for v in ... if v >= 0}`` was a
        # second author of the same claim — it printed "coloured by 7 subpopulation(s)"
        # with nothing beside it, on the same run the report calls NO SUBSTANTIAL
        # STRUCTURE. What the run found is said once, by the report, and repeated:
        # this surface only says what has been coloured.
        # The headline goes LAST, and nothing is appended after it. It is another module's
        # sentence and this surface must not assume how it ends — an embedded quotation
        # needed a "." after it, which would read as ".." the day ``headline()`` acquires
        # its own full stop. Found in this fix's own verify pass.
        self.statusBar().showMessage(
            f"Trace spans coloured by subpopulation for "
            f"{len(self._subpop_assignment)} gated event(s); every other detected event keeps "
            f"the grey that means 'in no class'. Zoom in to read the names. The assignment is "
            f"the mixture's and is provisional — the report's caveats say why. "
            f"What the report found: {self._subpop_headline}", 15000)

    def _export_subpop_report(self) -> None:
        """The report window's Export… — the text, the sidecar and every table.

        Several files, so the confirmation covers all of them: the report text is the
        document, the ``.meta.json`` carries the settings and EVERY caveat (a CSV of
        fractions with the label-switching warning stripped off is precisely the
        artefact this pipeline must not emit), and one CSV per table.

        X5: the window shows a standing sentence when the accepted set has moved under this
        report (S7-POP4), and the file is written to be read WITHOUT the app beside it — so
        that same disclosure has to travel into it. It is appended as one more caveat, which
        is exactly where the text's CAVEATS block and the ``.meta.json`` ``warnings`` list
        already carry the reasons the fractions may not describe what is accepted now. The
        numbers are untouched: only the caveat list grows.
        """
        import dataclasses

        from ..analysis.subpop_report import report_paths, write_report

        title = "Export subpopulation report"
        report = None if self._subpop_window is None else self._subpop_window.report()
        if report is None:
            QtWidgets.QMessageBox.information(
                self, title, "There is no subpopulation report to export yet.")
            return
        self._sync_subpop_window()
        curation_note = self._subpop_window.curation_note()
        if curation_note:
            report = dataclasses.replace(
                report,
                warnings=tuple(report.warnings) + (
                    f"CURATION SINCE THE RUN: {curation_note}",))
        names = tuple(report.tables())
        path = ask_export_path(
            self, title=title, suggestion="subpopulations",
            filters="Text report (*.txt);;All files (*)",
            outputs=lambda p: report_paths(p, names))
        if path is None:
            return
        try:
            written = write_report(report, path)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, title, f"Could not write the export:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Wrote {written['text'].name} + {len(written) - 1} sidecar file(s) to "
            f"{written['text'].parent}.", 10000)
