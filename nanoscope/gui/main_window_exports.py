"""The export half of ``MainWindow``: the level-fit reconciliation, and the exports of events,
provenance, levels, density, training data and report figures.

``_ExportsMixin`` is a plain class listed before ``QMainWindow`` in the bases of ``MainWindow``,
with no ``__init__`` and no state of its own. The method bodies were moved verbatim out of
``main_window.py`` by a mechanical split; nothing in them was edited.

``ask_export_path`` is imported from ``main_window_seams``, whose forwarder looks the name up on
``main_window`` each time it is called, so a test's patch of ``main_window.ask_export_path``
still reaches these methods.
"""
from __future__ import annotations

# Imports copied from main_window.py; the unused ones are pruned.
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ..analysis.reconcile import CONSENSUS_CURATION_CONFLICT_COL
from ..core.identity import source_of
from .main_window_seams import ask_export_path


class _ExportsMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    def _reconcile_level_fit(self) -> None:
        """Drop the sub-state levels of every event whose span no longer matches them.

        *** THE LEVEL TABLE HAS ONE OWNER, AND THIS IS IT. ***

        Measured (C-5): fit CUSUM levels, then drag one event's mask in the Inspector from
        6.000 ms to 4.000 ms. The Inspector dropped that event's levels from its own
        private ``self._levels`` — so the window drew no CUSUM line — while THIS window's
        ``_level_fit`` and ``Session.results['fit.levels']`` kept them, Export stayed
        enabled, and the export wrote level 0 starting 0.800 ms before the event now starts
        and level 1 ending 1.200 ms after it ends: 6.0 ms of sub-state inside a 4.0 ms
        event, a 50% overstatement. It then round-tripped through the project file and came
        back exportable. The only signal the user had pointed the wrong way.

        The fix is NOT another connection from the Inspector. Staleness here is DERIVABLE:
        :func:`~nanoscope.analysis.cusum.reconcile_levels` compares the span each level
        group tiles against the span its parent event now has, so a mask drag, a merge, an
        undo, a re-measure and a project restore are all caught by the same rule, including
        the paths written after this one. The Inspector applies that same function to its
        own view (one rule, one implementation) — but the export and the project file read
        THIS table, so this is where the guarantee has to hold.

        Idempotent, and a no-op in the overwhelmingly common case: when nothing has moved,
        ``reconcile_levels`` returns the very frame it was given.

        DROPPING IS NOT THE WHOLE STORY ANY MORE, AND THIS METHOD IS STILL ONLY THE DROP.
        The owner read the deletion as the feature breaking ("when I change the window
        manually, the sublevels disappear"), and :meth:`_refit_levels_for_event` is the
        answer: on a mask edit it re-measures that one event and splices its group back.
        It calls THIS method first, always, so the invariant above holds between the two
        steps and a re-fit that declines or fails leaves exactly the behaviour described
        here. Every other span-mover — a merge, an undo, a re-measure, a restore — still
        gets the drop alone, because they announce nothing to re-fit from.
        """
        from dataclasses import replace

        from ..analysis.cusum import levels_summary, reconcile_levels
        from ..analysis.events import EventSet

        fit = getattr(self, "_level_fit", None)
        if fit is None or getattr(fit, "payload", None) is None:
            return
        det = self._analysis_panel.current_detection()
        events = det.payload.df if (det is not None
                                    and isinstance(det.payload, EventSet)) else None
        # A fit that does not belong to the detection on screen is _sync_export_action's
        # question, not this one; answering it here would drop a fit merely because the
        # user selected another recording for a moment.
        if events is None or det.doc_id != fit.doc_id or int(det.channel) != int(fit.channel):
            return
        kept, dropped = reconcile_levels(fit.payload, events)
        if kept is fit.payload:
            # IDENTITY, not `not dropped`: on the inputs reconcile_levels refuses to
            # certify it returns kept=None while naming no ids, and reading that as
            # "nothing changed" is what left this window exporting a table the
            # Inspector had already discarded.
            return
        if kept is None:
            # Nothing survives: this is exactly _invalidate_level_fit's job — clear the
            # window's hands, the menu, the Inspector AND the store entry (a retired fit
            # left in the store is what project save writes into the file).
            self._invalidate_level_fit()
        else:
            self._level_fit = replace(fit, payload=kept, scalars=levels_summary(kept))
            # Straight into the store, not through add_result: this is a correction to a
            # table already there, and resultsChanged would re-enter this method.
            self._session.results.put(self._level_fit)
            self._push_level_fit(kept)
            self._act_export_levels.setEnabled(True)
        self.statusBar().showMessage(
            (f"Sub-state levels dropped for {len(dropped)} event(s) whose span changed — "
             if dropped else
             "Sub-state levels dropped: the table no longer states which spans it "
             "describes — ")
            + "re-run Analysis ▸ Fit sub-state levels (CUSUM)… to measure the new spans.",
            8000)

    def _refit_levels_for_event(self, event_id: int) -> None:
        """Re-fit ONE edited event's sub-state levels over its NEW span and splice them back.

        *** THIS IS THE SECOND HALF OF THE MASK EDIT, AND IT USED NOT TO EXIST. ***

        Measured: with a level fit standing, event 0 spanning (5000, 5600) had 14 CUSUM plot
        items in the Inspector and 32 level rows in both the Inspector and the result store.
        Drag the mask to (5000, 5300) and it had 0 plot items and 25 rows — the group was
        DELETED from the table, the export and the project file, and nothing put anything
        back. That deletion is correct and deliberate (:meth:`_reconcile_level_fit`: levels
        that no longer tile their parent describe a span that does not exist), but on its own
        it reads to the user as "changing the window destroys the sublevels", which is exactly
        the note this answers. The other half is to MEASURE THE NEW SPAN.

        THE DROP RUNS FIRST, ALWAYS, AND THE SPLICE IS SECOND. Not an ordering convenience:
        the invariant every reader of the level table rests on is "a group tiles its parent
        exactly", and reconciling first means that between the two steps the table holds
        strictly fewer groups rather than a wrong one. If anything below refuses or raises —
        a cold cache, a rejected row, a span too short, a fit that throws — the outcome is
        today's behaviour (dropped, and said so), never a stale group left standing.

        THE RE-FIT IS THE SAME FIT, not a fresh one. ``sigma`` and ``tau`` come from the
        cached inputs, which came from the original result's own ``params`` — so the new group
        is segmented at the significance scale its neighbours were, instead of at whatever a
        second whole-recording estimate would have produced from a table that has since
        changed. It is also what makes it cheap: measured on a 60 k-sample synthetic, one
        event costs 3.4 ms with ``tau`` handed in against 26.3 ms without, because without it
        the call re-walks every event's residual in the recording.

        Guards, each of which leaves the drop standing:
          * no fit, or a fit belonging to another recording/channel;
          * the event is not in the current table, or is REJECTED (the fit ran
            ``accepted_only=True``, so a rejected event has no levels by construction and
            inventing some here would put rows in the export that a re-run would not);
          * a span shorter than :data:`~nanoscope.analysis.cusum.MIN_SAMPLES_FOR_LEVELS`,
            where the fit cannot resolve a boundary and would return one "level" that is
            just a copy of the event;
          * a COLD cache — the fit was restored from a project file, so the baseline track
            behind it is not in memory. Re-deriving it here would run a 644 ms open-pore
            estimate inside a mouse-release handler. The status bar says which, and one
            re-run of the menu item makes every later edit re-fit.

        F2 — EVERY ONE OF THOSE NOW SAYS SO IN THE INSPECTOR, AND SAYS SOMETHING TRUE.
        Two things were wrong with how they ended. The Inspector — the window whose mask
        was just dragged and whose staircase just vanished — said nothing at all; and two
        of the routes (a rejected row, a re-fit that came back empty) returned in silence,
        leaving `_reconcile_level_fit`'s generic "re-run the fit" as the only sentence on
        screen. MEASURED: perform that re-run on a rejected event and it returns 0 level
        rows for it, because the fit is `accepted_only=True` — the prescribed recovery
        could not work, ever. Each route now builds its own sentence (the builders live in
        :mod:`~nanoscope.gui.event_inspector`, next to the surface that renders them) and
        posts it through :meth:`_report_level_gap` to the status bar AND the Inspector,
        naming a remedy that was checked against the app: accept the event, widen the mask,
        or re-run the fit — whichever of those actually puts the levels back from here.
        """
        import numpy as np
        from dataclasses import replace

        import pandas as pd

        from ..analysis.cusum import MIN_SAMPLES_FOR_LEVELS, fit_event_levels, levels_summary
        from ..analysis.events import EventSet
        from ..analysis.modules import detection_signal
        # The sentences themselves live with the window that has to SHOW them (F2): this
        # method posts one to the status bar and hands the same string to the Inspector, so
        # there is one wording per route rather than one per surface.
        from .event_inspector import (
            level_gap_cold,
            level_gap_empty,
            level_gap_failed,
            level_gap_rejected,
            level_gap_too_short,
        )

        fit = getattr(self, "_level_fit", None)
        if fit is None or getattr(fit, "payload", None) is None:
            return
        det = self._analysis_panel.current_detection()
        if det is None or not isinstance(det.payload, EventSet):
            return
        if det.doc_id != fit.doc_id or int(det.channel) != int(fit.channel):
            return
        events = det.payload.df
        rows = events[events["event_id"].astype("int64") == int(event_id)]
        if rows.empty:
            return

        cache = getattr(self, "_level_fit_inputs", None)
        cold = (cache is None
                or cache.get("doc_id") != fit.doc_id
                or int(cache.get("channel", -1)) != int(fit.channel))
        # THE DROP, FIRST AND UNCONDITIONALLY. Idempotent, and a no-op when nothing moved --
        # so calling it here as well as from `_sync_analysis_surfaces` costs nothing and makes
        # this method's guarantee independent of the order Qt happens to run the two slots in.
        self._reconcile_level_fit()
        # WHETHER ANY GROUP IS STILL STANDING, read AFTER the drop, because it decides which
        # remedy each sentence below may name. With a fit still in hand a mask edit re-fits
        # on release; with the whole fit retired (`_invalidate_level_fit`, the single-event
        # case) this method returns at its `fit is None` guard and a mask edit does nothing
        # at all -- measured, twice, on the five-event fixture with the other four rejected.
        standing = getattr(self, "_level_fit", None) is not None

        if cold:
            self._report_level_gap(event_id, level_gap_cold(int(event_id)))
            return

        row = rows.iloc[0]
        if "accepted" in rows.columns and not bool(row["accepted"]):
            # F2-a. This route used to return in silence, leaving `_reconcile_level_fit`'s
            # standing sentence -- "re-run Analysis ▸ Fit sub-state levels (CUSUM)… to
            # measure the new spans" -- as the only thing the user saw. MEASURED: perform
            # that re-run on a rejected event and it comes back with 0 level rows, because
            # the fit runs `accepted_only=True` and always will. The gesture that does work
            # is the accept, so that is what the sentence names now.
            self._report_level_gap(
                event_id, level_gap_rejected(int(event_id), standing_fit=standing))
            return
        start, end = int(row["start_idx"]), int(row["end_idx"])
        if end - start < MIN_SAMPLES_FOR_LEVELS:
            self._report_level_gap(
                event_id,
                level_gap_too_short(int(event_id), n_samples=end - start,
                                    min_samples=MIN_SAMPLES_FOR_LEVELS,
                                    standing_fit=standing),
                msecs=8000)
            return

        try:
            chan = self._session.calibrated(fit.doc_id, int(fit.channel))
            # REBUILT, not cached: 2.1 ms for 2 M samples against a second full-length float64
            # copy of every open recording. The BASELINE is the one that had to be kept.
            signal = np.asarray(
                detection_signal(np.asarray(chan.data), float(cache["fs"]),
                                 cache["provenance"]),
                dtype=np.float64)
            fresh = fit_event_levels(
                signal, float(cache["fs"]), rows.iloc[[0]],
                baseline=cache["baseline"],
                polarity=int(cache["polarity"]),
                sigma=cache["sigma"], tau=cache["tau"],
                provenance=cache["provenance"],
                config=cache["config"],
                # The row's own `accepted` was checked above; asking the fit to filter again
                # on a one-row frame only adds a way for it to come back empty.
                accepted_only=False,
            )
        except Exception as exc:  # noqa: BLE001 — a failed re-fit degrades to the drop
            self._report_level_gap(
                event_id, level_gap_failed(int(event_id), detail=str(exc)))
            return
        if fresh.empty:
            # The literal "a re-fit that yields zero levels" case, and it used to return in
            # silence too — leaving the reconcile's "re-run the fit" standing over it. A
            # re-run cannot help here BY CONSTRUCTION: it measures the same samples against
            # the same cached baseline with the same sigma and tau, so it resolves the same
            # nothing. The span is the only input a second attempt sees differently.
            self._report_level_gap(
                event_id, level_gap_empty(int(event_id), standing_fit=standing))
            return

        # SPLICE. `self._level_fit` may have been cleared entirely by the reconcile above (the
        # edited event was the only one carrying levels), so the surviving rows are read from
        # whatever is in hand NOW while the result's identity comes from the fit that existed
        # BEFORE — same module_id, doc, channel and both staleness stamps.
        current = getattr(self, "_level_fit", None)
        kept = getattr(current, "payload", None) if current is not None else None
        if kept is not None and not kept.empty:
            survivors = kept[kept["event_id"].astype("int64") != int(event_id)]
            merged = pd.concat([survivors, fresh], ignore_index=True)
        else:
            merged = fresh.reset_index(drop=True)
        merged = merged.sort_values(["event_id", "level_idx"], kind="stable") \
                       .reset_index(drop=True)

        result = replace(fit, payload=merged, scalars=levels_summary(merged))
        self._level_fit = result
        self._level_fit_frame = events
        self._level_fit_inputs = cache      # the reconcile may have cleared it; it is still true
        # Straight into the store, as `_reconcile_level_fit` does and for the same reason: this
        # is a correction to a table already there, and `add_result` would re-enter the
        # surface refresh that just called this.
        self._session.results.put(result)
        self._push_level_fit(merged)
        self._act_export_levels.setEnabled(True)
        self._analysis_panel.redraw_overlays()
        # THE OTHER HALF OF THE ROUND TRIP. `_push_level_fit` above already forgets the
        # recorded reason for any event whose rows came back (see `set_levels`), and the
        # Inspector refuses to render one over a live group anyway — this is the third
        # guard, at the one place that KNOWS the re-fit succeeded, because a "no levels"
        # sentence surviving over a restored staircase is a worse defect than the silence
        # F2 set out to fix.
        self._report_level_gap(
            event_id,
            f"Event {int(event_id)} re-fitted over its new span: {int(len(fresh))} "
            f"sub-state level(s) across {end - start} sample(s).",
            msecs=6000, gap=False)

    def _report_level_gap(self, event_id: int, note: str, *, msecs: int = 10000,
                          gap: bool = True) -> None:
        """Say ONCE why an event has no sub-state levels, on BOTH surfaces that can show it.

        The status bar is where every one of these sentences used to end, and it is not
        where the user is looking: they just dragged the Event Inspector's mask, and the
        Inspector's staircase is what vanished. Measured before this — drag event 0 down to
        9 samples and the window's readout, its refusal label and its plot said nothing
        whatsoever about the disappearance, while a status bar behind another window
        carried the whole explanation for eight seconds.

        ONE STRING, both surfaces. The two cannot then prescribe different remedies for the
        same event, which is the failure mode a second wording invites — and it is exactly
        the failure F2-a found between the status bar and the fit's own accepted-only rule.

        ``gap=False`` reports a SUCCESS through the same door: the status bar gets the
        sentence and the Inspector is told to forget whatever reason it was holding for
        that event.
        """
        self.statusBar().showMessage(note, msecs)
        insp = self._event_inspector
        if insp is not None:
            insp.note_level_gap(int(event_id), note if gap else "")

    def _refuse_stale_export(self, res) -> bool:
        """``True`` -- the refusal already said on the status bar -- when ``res``, the table an
        export is about to write, was detected at a sample rate the recording no longer has.

        *** A TABLE THE AMBER LINE CALLS STALE IS NOT WRITTEN BESIDE A CLOCK IT WAS NOT
        MEASURED ON. *** After Inspector ▸ File Info ▸ Sample rate / time unit… the table on
        screen keeps the rate it was detected at (``t_start_s`` and ``dwell_s`` are
        ``start_idx / fs`` at that rate) and wears "Times stale — sample rate changed". The
        scoring exports then built their sidecar from two places: ``fs`` from the table's own
        provenance (the OLD rate) and ``t0_s`` plus the clock sentence from the LIVE recording
        (``scoring.clock_stamp(meta=...)``, the declared one). That sentence says "add t₀ to a
        number here to find it on the trace", which for old-rate seconds lands on another
        sample (2599 samples off for 40 kHz declared over a 50 kHz detection), and a time-unit
        declaration moved ``t0_s`` itself (10000.0 → 10.0) under an unchanged CSV. Neither
        file said the table was stale, that a declaration existed, or what the live rate was.

        The question is the amber line's own -- ``AnalysisPanel._sample_rate_staleness``, the
        ``fs`` the detector stamped in the table's provenance against the recording's live rate
        -- so a door and the screen cannot disagree; a table with no stamp has nothing to be
        judged against and is exported as it always was. Only the RATE is refused here: a
        calibration or estimator change leaves ``fs`` and the clock block on one time base (and
        the sidecar already names the open-pore estimator the depths were measured against).
        And only while the recording still HAS a rate: one handed back to a file that declares
        none has no clock to splice -- the sidecar says its origin is unknown
        (``t0_s_known: false``) rather than stamping another -- and "re-run under the declared
        rate" would send the user to a rate that no longer exists.

        Asked at the TOP of each door, before its file dialog or selection box (the answer does
        not depend on a path, and a refusal after three questions is the worst place to learn
        it), and said on the status bar rather than in a modal: a scripted or batch export has
        no one to press the button. Nothing is repaired or re-scaled -- the one true repair is a
        re-run, which is one click on the banner.
        """
        live = self._session.document(res.doc_id).meta.sample_rate_hz
        if live is None or self._analysis_panel._sample_rate_staleness(res) is None:
            return False
        stamped = float(res.payload.provenance["fs"])
        # The way forward FIRST: the status bar clips a long line at its right edge, and the
        # rates are also on the amber line, which keeps the whole story.
        self.statusBar().showMessage(
            f"Nothing was exported — re-run detection under the declared rate first. This "
            f"table was detected at {stamped:,.6g} Hz and the recording now reads "
            f"{live:,.6g} Hz, so its times are on a clock the trace no longer has.", 12000)
        return True

    def _export_source(self, doc_id: str) -> "tuple[str, object | None]":
        """``(the FILE these rows came from, that document's meta)`` for an export.

        ONE ASK, ONE DOCUMENT. Every scoring export answers two questions about the
        recording behind its rows — which file to name in ``source_file``, and which
        clock its ``t_start_s`` column is on — and they must be answered about the
        SAME document or the sidecar names one recording and stamps another's origin.
        So they are asked together, once, here.

        ``source_of``, not the raw doc_id: the fallback is only reached once the
        recording has been closed, and what belongs in an export's provenance is the
        FILE it came from, not this session's private per-load token (core.identity).

        The meta is the whole ``RecordingMeta``, handed to
        :func:`~nanoscope.analysis.scoring.clock_stamp` unopened on purpose — that seam
        picks ``clock.instrument_origin_s`` (MEASUREMENT_SCHEMA's definition of
        ``t0_s``: a segment's PARENT clock, no layer nudge), which is a choice this
        window must not make for it, because the wrong one of the two clock functions
        differs by 1158.99 s on a cut of the owner's own quick ABF. ``None`` once the
        recording is closed — an honest "the origin is not knowable from here", which
        the sidecar writes as an explicit null rather than a fabricated 0.
        """
        if self._session.has_document(doc_id):
            meta = self._session.document(doc_id).meta
            return str(meta.path), meta
        return source_of(doc_id), None

    def _export_events(self) -> None:
        """Analysis ▸ Export ▸ Export events… — write the current detection's events to a CSV,
        honouring curation (the accepted/curated-in events). This is the plain
        'save these detections for extraction' path, distinct from the figure export."""
        from ..analysis.events import EventSet
        from ..analysis.scoring import events_export_paths, write_events_export

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, "Export events",
                "Run detection on a current channel first — there are no events to export.")
            return
        if self._refuse_stale_export(res):
            return
        events = res.payload

        # TWO files, so the confirmation covers both: the CSV and its
        # `<stem>.meta.json` provenance sidecar. The sidecar is the only place the
        # open-pore estimator's identity (`baseline_fingerprint` and every
        # parameter) reaches a consumer outside this app — the 20-column schema is
        # frozen and must not grow one. Without it, two CSVs exported months apart
        # are not comparable and nothing in either says why.
        path = ask_export_path(
            self, title="Export events", suggestion="events.csv",
            filters="CSV (*.csv);;All files (*)",
            outputs=events_export_paths)
        if path is None:
            return
        accepted = events.accepted.df
        total = len(events.df)
        include = self._include_hand_made() or res.module_id == "detect.manual"
        n_excluded = 0
        if not include:
            from ..analysis.scoring import hand_made_mask

            n_excluded = int(hand_made_mask(accepted).sum())
            total = int((~hand_made_mask(events.df)).sum())
            accepted = accepted[~hand_made_mask(accepted)]
        source_file, doc_meta = self._export_source(res.doc_id)
        try:
            out = write_events_export(
                accepted, str(path),
                provenance=self._export_provenance(events.provenance, accepted),
                source_file=source_file, detector=res.module_id, meta=doc_meta,
                include_added=include, n_added_excluded=n_excluded,
                # How many rows the table this export came FROM had, so the sidecar's
                # `curation.n_withheld` is the exact count rather than the one inferred
                # from `provenance['n_events']` (which a merge or a hand-add can leave
                # stale -- see events.reconcile_row_counts). The frame is in hand here;
                # the sidecar was guessing from a number written before curation.
                n_total=total)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export events", f"Could not write the export:\n{exc}")
            return
        left_out = (f" {n_excluded} hand-made event(s) left out." if n_excluded else "")
        self.statusBar().showMessage(
            f"Wrote {len(accepted)} accepted event(s) of {total} to "
            f"{out['data'].name} (+ provenance {out['meta'].name}).{left_out}", 8000)
        # The recovery copy must never be older than a CSV that just left the app (s4).
        self._flush_recovery_now()
        self._disclose_identity_notes("Export events", list(out.values()))

    @staticmethod
    def _export_provenance(provenance: dict | None, df) -> dict:
        """The detection's provenance plus the note a ``dI/I0`` above 1 must travel with.

        THE OTHER HALF OF THE SUMMARY'S WARNING. ``frac_depth`` is
        ``depth_pA / |baseline_pA|`` and reads as a blockade fraction — a number
        between 0 and 1. On the owner's apoferritin recordings it is 1.81–2.57,
        because the current MAGNITUDE grows during an event, and the Summary table now
        says so with a glyph and a sentence. A CSV leaving the app said nothing at all,
        and the CSV is the copy a co-author actually gets: a column of 2.36 with a
        header spelled ``frac_depth`` is exactly how "2.36" becomes "a 236 % blockade"
        in someone else's notes, with no screen anywhere to contradict it.

        An ADDITIVE PROVENANCE KEY, which is the channel the frozen 20-column schema
        leaves open for precisely this (a new column is forbidden; ``provenance`` and
        the ``flag`` token set are not). It rides in ``<stem>.meta.json`` beside
        ``baseline_fingerprint``, which is where a reader already has to look to find
        out what the depths were measured against.

        The sentence comes from :meth:`nanoscope.gui.panels.StatsPanel.frac_depth_note`
        — the one owner of this wording — so the file and the screen cannot come to
        describe the same number differently. Keyed on the MEDIAN of the rows actually
        being written, so a curated export is judged on what it contains rather than on
        what was detected.
        """
        import numpy as np

        from .panels import StatsPanel

        prov = dict(provenance or {})
        # An EMPTY export (everything rejected) has no median to take, and asking numpy
        # for one is a "Mean of empty slice" warning on a path the user reached on
        # purpose. No rows, no claim about them.
        if df is None or not len(df):
            return prov
        try:
            median = float(np.nanmedian(df["frac_depth"].to_numpy()))
        except (KeyError, TypeError, ValueError):
            return prov
        note = StatsPanel.frac_depth_note(median)
        if note:
            prov["frac_depth_note"] = note
            prov["frac_depth_median"] = median
        return prov

    def _invalidate_level_fit(self) -> None:
        """Drop the stored level fit. Called when the selection's detection changes.

        A level table belongs to the exact event table it was fitted on. Carried across a
        detection change it would be exported under another recording's provenance and
        matched to another recording's event_ids -- which both tables have and which mean
        nothing across recordings. Absence is recoverable in one menu click; a level table
        silently describing the wrong recording is not.
        """
        stale = self._level_fit
        self._level_fit = None
        self._level_fit_frame = None
        # ...and the re-fit inputs with it. They describe the retired fit's recording, its
        # baseline estimate and its two estimators; kept alive past the fit they belong to they
        # would let a later re-fit splice rows measured against ANOTHER recording's open pore
        # into a table matched by event_id, which is the hazard `_invalidate_level_fit` exists
        # for. `_refit_levels_for_event` restores them only when it is itself the caller.
        self._level_fit_inputs = None
        if hasattr(self, "_act_export_levels"):
            self._act_export_levels.setEnabled(False)
        self._push_level_fit(None)
        # ...and out of the RESULT STORE, not just out of this window's hands. A retired
        # level table left in the store was inert for as long as nothing read it — but
        # project save reads the store, so it was written into the file and restored
        # against whatever detection came back, its event_ids naming different events. A
        # fit this method has just declared invalid must not survive anywhere.
        if stale is not None:
            self._session.results.drop(stale.doc_id, stale.channel, stale.module_id)
            # The store drop emits nothing, so tell the panel to take the STEP overlay
            # off the trace now — not on whatever curation edit happens to come next.
            self._analysis_panel.redraw_overlays()

    def _export_levels(self) -> None:
        """Analysis ▸ Export ▸ Export sub-state levels… — the fitted level table plus its sidecar.

        Long layout (one row per event level). The sidecar carries the CUSUM parameters
        for the same reason the events sidecar carries the baseline estimator: those
        numbers decide how many levels an event HAS, and a table that cannot name them
        cannot be compared with one exported next month.
        """
        from ..analysis.scoring import levels_export_paths, write_levels_export

        # PASS THROUGH THE RECONCILER, do not read around it. Every path that moves a
        # span today emits one of SURFACE_MUTATION_SIGNALS and so reaches
        # _reconcile_level_fit before this action can run — but the guarantee "the
        # export never writes a level table describing spans the events no longer
        # have" then rests on the refresher having fired, not on the export. It is
        # idempotent and a no-op when nothing moved (identity return), so this costs
        # nothing and makes the export a passer-through instead of a second reader.
        self._reconcile_level_fit()

        result = getattr(self, "_level_fit", None)
        if result is None or result.payload is None or result.payload.empty:
            QtWidgets.QMessageBox.information(
                self, "Export sub-state levels",
                "Run Analysis ▸ Fit sub-state levels (CUSUM)… first — there are no "
                "fitted levels to export.")
            return
        # The sidecar's ``fs`` is the detection on screen's, its origin the recording's.
        shown = self._analysis_panel.current_detection()
        if shown is not None and self._refuse_stale_export(shown):
            return

        path = ask_export_path(
            self, title="Export sub-state levels", suggestion="levels.csv",
            filters="CSV (*.csv);;All files (*)",
            outputs=levels_export_paths)
        if path is None:
            return
        source_file, doc_meta = self._export_source(result.doc_id)
        det = self._analysis_panel.current_detection()
        prov = {}
        if det is not None and getattr(det.payload, "provenance", None):
            prov = dict(det.payload.provenance)
        try:
            out = write_levels_export(
                result.payload, str(path), provenance=prov,
                cusum_params=dict(result.params),
                source_file=source_file,
                detector=det.module_id if det is not None else "",
                meta=doc_meta)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export sub-state levels", f"Could not write the export:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Wrote {len(result.payload)} level row(s) to {out['data'].name} "
            f"(+ provenance {out['meta'].name}).", 8000)

    def _density_results(self):
        """Exact shown detection owners eligible for the multi-recording density."""
        from ..analysis.events import EventSet
        results = []
        for result in self._session.results.all():
            if not isinstance(result.payload, EventSet) or int(result.payload.df["accepted"].sum()) < 5:
                continue
            if self._session.has_document(result.doc_id):
                shown = self._analysis_panel.stored_to_show(result.doc_id, result.channel)
                if shown is not None and shown.module_id != result.module_id:
                    continue
            results.append(result)
        return tuple(results)

    def _density_groups(self) -> "dict[str, object]":
        """Every open recording that HAS a detection, label -> accepted EventSet.

        Keyed by the recording's own name because that is what the legend must say: the
        figure exists to put several molecules (or several voltages) side by side, and a
        legend reading "Detected events" three times would defeat it. Ordered by descending
        n so the densest group is drawn first and the sparse ones stay visible on top.
        """
        from ..analysis.events import EventSet
        from ..analysis.iv import applied_voltage_mV

        def _common(strings: list[str]) -> int:
            """Length of the longest common prefix. Local so this needs no new import."""
            if not strings:
                return 0
            first, n = strings[0], len(strings[0])
            for other in strings[1:]:
                n = min(n, len(other))
                while n and first[:n] != other[:n]:
                    n -= 1
            return n

        out = []
        for res in self._density_results():
            acc = res.payload.accepted
            stem = (Path(self._session.document(res.doc_id).meta.path).stem
                    if self._session.has_document(res.doc_id) else source_of(res.doc_id))
            # The SHARED resolver, not a local copy. This block used to look only for a
            # VOLTAGE channel, so a current-only recording — every CSV export the owner has —
            # never got a voltage label and fell through to the filename stub, printing
            # "6_Process_-6" in the legend of a three-voltage figure. iv.applied_voltage_mV
            # already prefers the measured channel and falls back to the filename/header
            # nominal, which is exactly the answer the Calibration panel shows.
            volts = None
            if self._session.has_document(res.doc_id):
                try:
                    got = applied_voltage_mV(self._session, res.doc_id)
                    volts = None if got is None else float(got[0])
                except Exception:      # noqa: BLE001 - a label must never break the figure
                    volts = None
            out.append((len(acc.df), stem, volts, acc))
        out.sort(key=lambda t: -t[0])

        # LABEL BY WHAT DISTINGUISHES THEM. A legend is the whole point of this figure and
        # a recording filename is ~60 characters of shared boilerplate -- three of them
        # ran off both sides of the plot and buried the contours. When every recording
        # reports a DIFFERENT applied voltage, that is the axis of the comparison and the
        # label writes itself; otherwise fall back to the filename with the common prefix
        # and suffix stripped, which leaves exactly the part that differs.
        vs = [v for _n, _s, v, _e in out]
        if len(out) > 1 and all(v is not None for v in vs) and len({round(v) for v in vs}) == len(vs):
            return {f"{v:.0f} mV": ev for _n, _s, v, ev in out}
        stems = [s_ for _n, s_, _v, _e in out]
        head = _common(stems) if len(stems) > 1 else 0
        tail = _common([s_[::-1] for s_ in stems]) if len(stems) > 1 else 0
        labels = []
        for s_ in stems:
            core = s_[head: len(s_) - tail] if (head or tail) else s_
            labels.append((core or s_)[:32])
        return {lab: ev for lab, (_n, _s, _v, ev) in zip(labels, out)}

    #: Amplitude measures the density figure can use, as (column, axis label).
    #:
    #: ``frac_depth`` is dI/I0 -- the blockade as a FRACTION of that event's own open-pore
    #: current -- and it is the honest axis whenever the current gain is not calibrated,
    #: which the status bar says outright for most of this owner's recordings. A ratio
    #: divides the gain out, so two recordings taken at different gains are comparable in
    #: dI/I0 and are NOT comparable in pA. It is also the axis that survives a calibration
    #: being revised afterwards.
    #:
    #: ``depth_pA`` stays the default because it is what a reader expects on a blockade
    #: axis and what the event table's headline column reports.
    _DENSITY_Y = {
        "depth (pA)": ("depth_pA", "Amplitude (pA)"),
        "fractional blockade (dI/I0)": ("frac_depth", "Fractional blockade  ΔI/I₀"),
    }

    #: Density-figure style, remembered for the session. Defaults are deliberately modest:
    #: this figure overlays several populations, so its legend sits ON TOP of the contours
    #: it names, and a legend is an index rather than content.
    _DENSITY_STYLE: dict[str, object] = {
        "levels": 8, "line_width": 0.7, "legend_loc": "upper right",
        "legend_fontsize": 6.0, "legend_title": "molecule", "show_counts": True,
        "font_scale": 1.0,
    }

    _DENSITY_LEGEND_LOCS = ("upper right", "upper left", "lower right", "lower left",
                            "center right", "outside right", "none")

    def _edit_density_style(self) -> bool:
        """Small form for the density figure's styling. True when the user accepted.

        Kept as a dialog rather than a menu of presets because the useful knob depends on
        the data: three overlaid recordings need a smaller legend, a single one does not,
        and a figure going into a paper wants the legend off the data entirely.
        """
        st = dict(self._DENSITY_STYLE)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Density figure style")
        form = QtWidgets.QFormLayout(dlg)

        levels = QtWidgets.QSpinBox(); levels.setRange(2, 20); levels.setValue(int(st["levels"]))
        form.addRow("Contour levels", levels)
        lw = QtWidgets.QDoubleSpinBox(); lw.setRange(0.1, 5.0); lw.setSingleStep(0.1)
        lw.setValue(float(st["line_width"])); form.addRow("Line width", lw)
        loc = QtWidgets.QComboBox(); loc.addItems(self._DENSITY_LEGEND_LOCS)
        loc.setCurrentText(str(st["legend_loc"]))
        loc.setToolTip("“outside right” lifts the legend clear of the contours entirely; "
                       "“none” removes it.")
        form.addRow("Legend position", loc)
        lfs = QtWidgets.QDoubleSpinBox(); lfs.setRange(3.0, 20.0); lfs.setSingleStep(0.5)
        lfs.setValue(float(st["legend_fontsize"])); form.addRow("Legend font size", lfs)
        ltitle = QtWidgets.QLineEdit(str(st["legend_title"] or ""))
        ltitle.setPlaceholderText("(no heading)"); form.addRow("Legend heading", ltitle)
        counts = QtWidgets.QCheckBox("show (n = …) in the legend")
        counts.setChecked(bool(st["show_counts"]))
        counts.setToolTip("A legend is sized by its widest entry, and the counts are the "
                          "longest part of one. They stay in the exported source CSV.")
        form.addRow("", counts)
        fs = QtWidgets.QDoubleSpinBox(); fs.setRange(0.5, 3.0); fs.setSingleStep(0.1)
        fs.setValue(float(st["font_scale"])); form.addRow("Font scale", fs)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return False
        self._DENSITY_STYLE = {
            "levels": levels.value(), "line_width": lw.value(),
            "legend_loc": loc.currentText(), "legend_fontsize": lfs.value(),
            "legend_title": ltitle.text().strip() or None,
            "show_counts": counts.isChecked(), "font_scale": fs.value(),
        }
        return True

    def _choose_density_amplitude(self, ycol=None):
        """The same explicit amplitude choice for retained editing and direct export."""
        ylabel = "Amplitude (pA)"
        if ycol is None:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("Amplitude measure")
            box.setText("Which amplitude should the density use?")
            box.setInformativeText(
                "depth (pA) is the absolute blockade.\n\n"
                "Fractional blockade ΔI/I₀ divides by each event's own open-pore current, "
                "so it is independent of the current gain — the comparable choice when the "
                "gain is uncalibrated or differs between recordings.")
            b_pa = box.addButton("depth (pA)", QtWidgets.QMessageBox.AcceptRole)
            b_fr = box.addButton("fractional ΔI/I₀", QtWidgets.QMessageBox.AcceptRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(b_pa)
            box.exec()
            clicked = box.clickedButton()
            if clicked is b_fr:
                ycol, ylabel = self._DENSITY_Y["fractional blockade (dI/I0)"]
            elif clicked is b_pa:
                ycol, ylabel = self._DENSITY_Y["depth (pA)"]
            else:
                return
        else:
            ylabel = next((lab for col, lab in self._DENSITY_Y.values() if col == ycol),
                          "Amplitude (pA)")
        return ycol, ylabel

    def _density_source_token(self):
        from ..core.analysis_workspace import result_recipe
        return (self._figure_project_epoch, self._session, tuple(
            (result, {**result_recipe(self, result), "live_sample_rate_hz":
                self._session.document(result.doc_id).meta.sample_rate_hz})
            for result in self._density_results()))

    def _density_sources_unchanged(self, token):
        epoch, session, previous = token
        if epoch != self._figure_project_epoch or session is not self._session:
            return False
        try:
            current = self._density_source_token()[2]
        except (ValueError, KeyError):
            return False
        return len(current) == len(previous) and all(
            old is now and saved == actual
            for (old, saved), (now, actual) in zip(previous, current))

    def _edit_density_figure(self, *, ycol=None):
        """Generate once and transfer retained artists to the shared report editor."""
        from copy import deepcopy
        from ..core.figure_style import style_binding
        from ..core.figure_style_mapping import project_legacy_style
        from ..plot.figures import joint_density
        from ..plot.matplotlib_targets import bind_panel
        from .report_figure_window import open_report_figure

        try:
            for result in self._density_results():
                if self._session.has_document(result.doc_id) and self._refuse_stale_export(result):
                    return None
            token = self._density_source_token()
            groups = self._density_groups()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Edit density figure", f"Could not preserve the density sources:\n{exc}")
            return None
        if not groups:
            QtWidgets.QMessageBox.information(self, "Edit density figure",
                "Run detection on at least one recording first — a density plot needs detected events.")
            return None
        choice = self._choose_density_amplitude(ycol)
        if choice is None:
            return None
        if not self._density_sources_unchanged(token):
            self.statusBar().showMessage("The project or density sources changed; open the density editor again.", 8000)
            return None
        ycol, ylabel = choice
        figure = None
        try:
            if ycol not in {value[0] for value in self._DENSITY_Y.values()}:
                raise ValueError("Choose depth in pA or fractional blockade for the density axis.")
            owner = self._ensure_figure_workspace()
            initial_style = None
            if owner.style_registry() is not None:
                binding = style_binding(owner.default_style())
                initial_style = project_legacy_style(binding, "study_figure_style", {})
            settings = deepcopy(self._DENSITY_STYLE)
            figure, axis = joint_density(groups, ycol=ycol, ylabel=ylabel,
                title=f"Event density — {len(groups)} recording(s)", **settings)
            # The editor becomes the sole owner; release pyplot's manager reference.
            from matplotlib import pyplot as plt
            plt.close(figure)
            if not self._density_sources_unchanged(token):
                figure.clear()
                self.statusBar().showMessage("The project or density sources changed; the generated figure was not opened.", 8000)
                return None
            identities = (("density.joint", "Density"), ("density.dwell_marginal", "Dwell marginal"),
                          ("density.amplitude_marginal", "Amplitude marginal"))
            for plot, (identity, label) in zip(figure.axes, identities):
                bind_panel(plot, identity, label=label)
            # Raw scientific columns stay canonical; additive display columns name
            # the values shown on the declared quantity axes.
            data = figure._nanoscope_source_data.copy(deep=True)
            display_columns = data.attrs["display_columns"]
            provenance = {
                "kind": "event-density", "builder": "joint_density",
                "source_catalog": [deepcopy(recipe) for _result, recipe in token[2]],
                "groups": {str(label): {"rows": len(value.df),
                    "provenance": deepcopy(value.provenance)} for label, value in groups.items()},
                "x_source_column": "dwell_s", "x_display_column": display_columns["display_x"], "x_scale": 1000.,
                "quantity_display": data.attrs["quantity_display"],
                "column_units": {"dwell_s": "s", display_columns["display_x"]: "ms",
                                 ycol: "1" if ycol == "frac_depth" else "pA"},
                "y_column": ycol, "log_x": True, "generation": settings,
                "analysis_recomputed_by_style": False,
            }
            return open_report_figure(self, figure, data, name="Event density", provenance=provenance,
                                      style=initial_style)
        except Exception as exc:
            if figure is not None:
                figure.clear()
            QtWidgets.QMessageBox.warning(self, "Edit density figure", f"Could not open the density editor:\n{exc}")
            return None

    def _export_density(self, *, ycol: str | None = None) -> None:
        """Analysis ▸ Export ▸ Export density figure… — overlaid KDE contours, one per recording.

        ``ycol`` is KEYWORD-ONLY, and that is load-bearing rather than tidiness.
        ``QAction.triggered`` carries a ``checked`` bool, which Qt passes as the first
        POSITIONAL argument to whatever it is connected to -- so as a positional parameter
        ``ycol`` arrived as ``False`` from the menu, the amplitude prompt was skipped, and
        the export died on "events frame has no column False". Keyword-only makes that
        mis-binding impossible to write; the connection below also drops the bool.
        """
        from ..plot.figures import figure_output_paths, joint_density, save_figure

        groups = self._density_groups()
        if not groups:
            QtWidgets.QMessageBox.information(
                self, "Export density figure",
                "Run detection on at least one recording first — a density plot needs "
                "detected events.")
            return
        choice = self._choose_density_amplitude(ycol)
        if choice is None:
            return
        ycol, ylabel = choice
        path = ask_export_path(
            self, title="Export density figure", suggestion="density.pdf",
            filters="PDF (*.pdf);;PNG (*.png);;All files (*)",
            outputs=lambda p: figure_output_paths(Path(p)))
        if path is None:
            return
        try:
            fig, _ax = joint_density(
                groups, ycol=ycol, ylabel=ylabel,
                title=f"Event density — {len(groups)} recording(s)",
                **self._DENSITY_STYLE)
            out = save_figure(fig, str(path))
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export density figure", f"Could not write the figure:\n{exc}")
            return
        self.statusBar().showMessage(
            f"Wrote a density figure over {len(groups)} recording(s) to "
            f"{out['image'].name if 'image' in out else Path(path).name} "
            f"(+ source data).", 8000)

    def _export_training(self) -> None:
        """Write reviewed decisions (recommended), accepted candidates, or all rows.

        Reviewed is the supervised-learning-safe default: both human positives and human
        negatives, with untouched candidates and automatic rejects withheld. Accepted/all
        remain explicit escape hatches and are described without calling them curated truth.
        """
        import numpy as np

        from ..analysis.events import EventSet, STATUS_ACCEPTED, STATUS_REJECTED, event_status
        from ..analysis.labels import label_conflicts
        from ..analysis.scoring import (
            TRAINING_SELECTION_ACCEPTED,
            TRAINING_SELECTION_ALL,
            TRAINING_SELECTION_REVIEWED,
            build_training_frame,
            training_export_paths,
            write_training_export,
        )

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, "Export events for training",
                "Run detection on a current channel first — there are no events to export.")
            return
        if self._refuse_stale_export(res):
            return

        df = res.payload.df
        n_reviewed = len(build_training_frame(
            df, selection=TRAINING_SELECTION_REVIEWED))
        n_accepted = len(build_training_frame(
            df, selection=TRAINING_SELECTION_ACCEPTED))
        n_all = len(df)
        n_unresolved_curation = 0
        if CONSENSUS_CURATION_CONFLICT_COL in df.columns:
            n_unresolved_curation = int(
                df[CONSENSUS_CURATION_CONFLICT_COL]
                .astype("string").fillna("").str.strip().ne("").sum()
            )
        reviewed_exclusion_note = (
            f"\n{n_unresolved_curation:,} unresolved reconciliation decision"
            f"{'' if n_unresolved_curation == 1 else 's'} excluded from Reviewed until "
            "you explicitly Accept or Reject the row."
            if n_unresolved_curation else ""
        )
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle("Export events for training")
        box.setText("Which events should the training table contain?")
        box.setInformativeText(
            f"Reviewed decisions ({n_reviewed:,}) — recommended: human positives and "
            f"human negatives only.{reviewed_exclusion_note}\n\n"
            f"Accepted candidates ({n_accepted:,}) — includes accepted rows nobody has "
            "reviewed yet.\n\n"
            f"All raw candidates ({n_all:,}) — also includes automatic detector rejects."
        )
        b_reviewed = box.addButton(
            "Reviewed decisions (recommended)", QtWidgets.QMessageBox.AcceptRole)
        b_acc = box.addButton("Accepted candidates", QtWidgets.QMessageBox.ActionRole)
        b_all = box.addButton("All raw candidates", QtWidgets.QMessageBox.ActionRole)
        b_cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setEscapeButton(b_cancel)
        if n_reviewed:
            box.setDefaultButton(b_reviewed)
        else:
            b_reviewed.setEnabled(False)
            b_reviewed.setToolTip(
                (f"{n_unresolved_curation:,} reviewed decision(s) remain contradictory; "
                 "explicitly Accept or Reject them before using Reviewed export."
                 if n_unresolved_curation else
                 "No event in this table records a human review yet. Review both accepted "
                 "and rejected examples before exporting supervised-learning truth.")
            )
            # With no reviewed truth, Return must not quietly downgrade the export to
            # unreviewed accepted candidates. That remains available as an explicit click.
            box.setDefaultButton(b_cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_reviewed:
            selection = TRAINING_SELECTION_REVIEWED
        elif clicked is b_acc:
            selection = TRAINING_SELECTION_ACCEPTED
        elif clicked is b_all:
            selection = TRAINING_SELECTION_ALL
        else:
            return

        # ``|`` in a structured label is reserved for an unresolved merge/reconciliation
        # conflict (for example ``molecule=apo|holo``), not a legitimate combined class.
        # Warn over exactly the rows the chosen export mode will write, defaulting to
        # Cancel so ambiguous labels cannot silently become supervised-learning truth.
        if selection == TRAINING_SELECTION_REVIEWED:
            selected_mask = event_status(df).isin(
                (STATUS_ACCEPTED, STATUS_REJECTED)).to_numpy(dtype=bool)
            if CONSENSUS_CURATION_CONFLICT_COL in df.columns:
                unresolved_curation = (
                    df[CONSENSUS_CURATION_CONFLICT_COL]
                    .astype("string").fillna("").str.strip().ne("")
                    .to_numpy(dtype=bool)
                )
                selected_mask &= ~unresolved_curation
        elif selection == TRAINING_SELECTION_ACCEPTED:
            selected_mask = (df["accepted"].to_numpy(dtype=bool)
                             if "accepted" in df.columns
                             else np.ones(len(df), dtype=bool))
        else:
            selected_mask = np.ones(len(df), dtype=bool)
        n_conflicts = 0
        if "flag" in df.columns and selected_mask.any():
            n_conflicts = sum(
                bool(label_conflicts(flag))
                for flag in df.loc[selected_mask, "flag"].to_numpy(dtype=object)
            )
        if n_conflicts:
            conflict = QtWidgets.QMessageBox(self)
            conflict.setIcon(QtWidgets.QMessageBox.Warning)
            conflict.setWindowTitle("Unresolved label conflicts")
            conflict.setText(
                f"{n_conflicts:,} selected event(s) have unresolved label conflicts."
            )
            conflict.setInformativeText(
                "A vertical bar (|) is the conflict marker, not a combined class. "
                "Resolve these labels in the Event Gallery or Event Inspector before "
                "training, or explicitly export them for external cleanup."
            )
            b_export_conflicts = conflict.addButton(
                "Export with conflicts", QtWidgets.QMessageBox.DestructiveRole)
            b_cancel_conflicts = conflict.addButton(
                "Cancel", QtWidgets.QMessageBox.RejectRole)
            conflict.setDefaultButton(b_cancel_conflicts)
            conflict.setEscapeButton(b_cancel_conflicts)
            conflict.exec()
            if conflict.clickedButton() is not b_export_conflicts:
                return

        # Reconciliation keeps contradictory human rulings visible rather than silently
        # choosing one.  The row remains reviewed but is rejected until a curator makes a
        # fresh explicit decision, so reviewed-only selection alone cannot make it safe.
        n_curation_conflicts = 0
        if CONSENSUS_CURATION_CONFLICT_COL in df.columns and selected_mask.any():
            markers = (
                df.loc[selected_mask, CONSENSUS_CURATION_CONFLICT_COL]
                .astype("string").fillna("").str.strip()
            )
            n_curation_conflicts = int(markers.ne("").sum())
        if n_curation_conflicts:
            conflict = QtWidgets.QMessageBox(self)
            conflict.setIcon(QtWidgets.QMessageBox.Warning)
            conflict.setWindowTitle("Contradictory curation decisions")
            conflict.setText(
                f"{n_curation_conflicts:,} selected event(s) combine contradictory "
                "human accept/reject decisions."
            )
            conflict.setInformativeText(
                "Explicitly Accept or Reject these rows in the event table, Event Gallery, "
                "or Event Inspector before training, or explicitly export them for external "
                "resolution."
            )
            b_export_conflicts = conflict.addButton(
                "Export with curation conflicts", QtWidgets.QMessageBox.DestructiveRole)
            b_cancel_conflicts = conflict.addButton(
                "Cancel", QtWidgets.QMessageBox.RejectRole)
            conflict.setDefaultButton(b_cancel_conflicts)
            conflict.setEscapeButton(b_cancel_conflicts)
            conflict.exec()
            if conflict.clickedButton() is not b_export_conflicts:
                return

        # C18: this command writes a `<stem>.meta.json` provenance sidecar next to the
        # table, so the confirmation has to cover BOTH. training_export_paths is the
        # writer's own enumeration, including its bare-name -> .csv normalisation.
        chosen = ask_export_path(
            self, title="Export events for training", suggestion="events_training.csv",
            filters="CSV (*.csv);;Parquet (*.parquet);;All files (*)",
            outputs=training_export_paths)
        if chosen is None:
            return
        path = str(chosen)

        source_file, doc_meta = self._export_source(res.doc_id)
        export_provenance = dict(res.payload.provenance or {})
        # The vocabulary belongs to the project/session rather than to one detector result.
        # Supply a copy only at this GUI export seam so the sidecar explains each dynamic
        # label column without mutating the stored detector provenance.
        export_provenance["label_vocabulary"] = self._label_vocab.to_json()
        include = self._include_hand_made() or res.module_id == "detect.manual"
        try:
            out = write_training_export(
                res.payload.df, path, provenance=export_provenance,
                source_file=source_file, detector=res.module_id,
                selection=selection, meta=doc_meta, include_added=include)
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Export events for training",
                f"Could not write the training export:\n{exc}")
            return
        n_sel = len(build_training_frame(df, selection=selection, include_added=include))
        selection_label = {
            TRAINING_SELECTION_REVIEWED: "reviewed decision",
            TRAINING_SELECTION_ACCEPTED: "accepted candidate",
            TRAINING_SELECTION_ALL: "raw candidate",
        }[selection]
        self.statusBar().showMessage(
            f"Wrote {n_sel} {selection_label}{'' if n_sel == 1 else 's'} to "
            f"{out['data'].name} "
            f"(+ provenance {out['meta'].name}).", 8000)

    def _edit_report_figure(self, *, kind=None):
        """Build one chosen report once, then transfer its retained artists to an editor."""
        import pandas as pd
        from ..analysis.events import EventSet
        from ..core.analysis_workspace import result_recipe
        from ..core.figure_style import style_binding
        from ..core.figure_style_mapping import project_legacy_style
        from ..plot import figures
        from .report_figure_window import open_report_figure

        title = "Edit standard report figure"
        result = self._analysis_panel.current_detection()
        if result is None or not isinstance(result.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, title, "Run detection on a current channel first.")
            return None
        if self._refuse_stale_export(result):
            return None
        epoch = self._figure_project_epoch
        try:
            recipe = result_recipe(self, result)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, title, str(exc))
            return None
        choices = {"Dwell histogram": "dwell", "Depth histogram": "depth",
                   "Dwell versus depth": "scatter"}
        channel = axis = None
        try:
            channel = self._session.calibrated(result.doc_id, result.channel)
            axis = self._session.x_axis(result.doc_id)
            if axis.is_time:
                choices["Trace preview"] = "trace"
        except Exception:
            pass
        if kind is None:
            selected, accepted = QtWidgets.QInputDialog.getItem(
                self, title, "Figure", list(choices), 0, False)
            if not accepted:
                return None
            kind = choices[selected]
        if kind not in choices.values():
            raise ValueError(f"Unavailable report figure: {kind}")

        def source_is_current():
            if (self._figure_project_epoch != epoch
                    or not self._session.has_document(result.doc_id)
                    or self._analysis_panel.current_detection() is not result):
                return False
            try:
                return result_recipe(self, result) == recipe
            except ValueError:
                return False

        if not source_is_current() or self._refuse_stale_export(result):
            QtWidgets.QMessageBox.warning(
                self, title, "The source changed while choosing the figure. Choose it again.")
            return None
        figure = None
        try:
            events = result.payload
            if kind in ("dwell", "depth"):
                fit, _ = self._stored_fit(result, f"fit.{kind}")
                builder = figures.dwell_histogram if kind == "dwell" else figures.depth_histogram
                figure, _ = builder(events, fit=fit if fit is not None else False)
            elif kind == "scatter":
                figure, _ = figures.scatter_dwell_depth(events)
            else:
                # The chooser runs an event loop. Use the channel's current calibration
                # and time axis, not the preview capability probe captured before it.
                channel = self._session.calibrated(result.doc_id, result.channel)
                axis = self._session.x_axis(result.doc_id)
                if not axis.is_time:
                    raise ValueError("The recording no longer has a time axis.")
                figure, _ = figures.trace_preview(channel, axis, events=events)
            if not source_is_current():
                raise ValueError("The source changed while generating the figure. Choose it again.")
            data = getattr(figure, "_nanoscope_source_data", None)
            if not isinstance(data, pd.DataFrame):
                raise ValueError("The report did not provide its plotted source table.")
            workspace = self._ensure_figure_workspace()
            style = None
            if workspace.style_registry() is not None:
                style = project_legacy_style(style_binding(workspace.default_style()),
                                             "study_figure_style", {})
            provenance = {"kind": f"report.{kind}", "source": recipe,
                          "details": figures.figure_details(figure)}
            editor = open_report_figure(self, figure, data, name=f"Report — {kind}",
                                        provenance=provenance, style=style)
            figure = None  # Ownership transferred; editing/export never calls a builder.
            self.statusBar().showMessage("Report figure ready to edit and export.", 6000)
            return editor
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, title, f"Could not open this report figure: {exc}")
            return None
        finally:
            if figure is not None:
                from matplotlib import pyplot as plt
                plt.close(figure)
                figure.clear()

    def _export_figures(self) -> None:
        """Analysis ▸ Export ▸ Generate Report figures… — write the current detection's publication figures
        (each as PDF + PNG + a source-data CSV) via nanoscope.plot.figures.

        The event-statistics figures are data-model-agnostic (they eat the EventSet); the
        trace preview is threaded from the CalibratedChannel + XAxis so its y-label carries
        the channel's own units instead of a hard-coded pA (PORT_PLAN R9).

        THE TABLE IS JUDGED AS THE EVENT EXPORTS JUDGE IT. A table detected at a rate the
        recording no longer has is refused before anything is built or asked
        (:meth:`_refuse_stale_export`): its dwell histogram would be in seconds the trace no
        longer has. One the amber line calls stale on its calibration or its estimator is
        exported -- a stale table stays "readable, plottable and exportable" -- and the line
        that reports the export names it, since the figures carry no badge of their own.
        Before this only the fit guard ran (``_stored_fit``), and a figure of depths under
        a replaced gain was written under "Wrote 4 figure(s)" and nothing else."""
        from ..analysis.events import EventSet
        from ..plot import figures

        res = self._analysis_panel.current_detection()
        if res is None or not isinstance(res.payload, EventSet):
            QtWidgets.QMessageBox.information(
                self, "Export figure",
                "Run detection on a current channel first — there is no analysis to export.")
            return
        if self._refuse_stale_export(res):
            return
        events = res.payload

        # THE fit the user computed, not a fresh one. figures.dwell_histogram(fit=True)
        # refits from scratch at n_components=1 — so a user who set "dwell components = 2",
        # read tau1/tau2 in the Fits panel and then exported would get a figure annotated
        # with a DIFFERENT, one-component tau. _resolve_fit accepts a FitResult, and the
        # session is already holding the right one; pass it. With no stored fit we keep
        # today's behaviour and say so in the status line.
        dwell_fit, depth_fit = self._stored_fit(res, "fit.dwell"), self._stored_fit(res, "fit.depth")

        # Each figure is best-effort: a fit can fail on sparse data, but the histogram
        # itself must still export. We never let one figure sink the whole export.
        builders = [
            ("dwell", lambda: figures.dwell_histogram(events, fit=dwell_fit[0] or True)),
            ("depth", lambda: figures.depth_histogram(events, fit=depth_fit[0] or True)),
            ("scatter", lambda: figures.scatter_dwell_depth(events)),
        ]
        # The trace preview needs the raw signal + its time axis.
        try:
            ch = self._session.calibrated(res.doc_id, res.channel)
            xax = self._session.x_axis(res.doc_id)
            if xax.is_time:
                builders.append(
                    ("trace", lambda: figures.trace_preview(ch, xax, events=events)))
        except Exception:
            pass

        # The builders are settled BEFORE the dialog because the dialog has to be able to
        # say what will be written. Naming `figures.pdf` — a file this command never writes
        # — while overwriting `{stem}_{name}.{pdf,png,csv}` for every builder was C11.
        base = ask_export_path(
            self, title="Generate standard report figures — choose a base name",
            suggestion="figures",
            filters="PDF (*.pdf);;PNG (*.png);;All files (*)",
            outputs=lambda p: self._figure_export_paths(p, [n for n, _ in builders]))
        if base is None:
            return
        # *** THE FIGURES DESCRIBE THE DETECTION THAT IS CURRENT WHEN THEY ARE WRITTEN. ***
        #
        # Everything above — `res`, `events`, both stored fits, the calibrated channel and
        # the x-axis — was resolved BEFORE `ask_export_path`, and that dialog is a real
        # modal: it spins the event loop, so a detection can land inside it. Nothing gates
        # Detect (or a batch pass) during an export, and `AnalysisController` delivers
        # `finished` queued on the GUI thread, so this is not exotic. Measured: the screen
        # went from 8 events to 2 while the dialog was open, and twelve files were written
        # from the 8-event frame under a status line that said only "Wrote 4 figure(s)".
        # The figure source CSVs carry no provenance columns at all, so nothing on disk or
        # on screen could tell the two runs apart.
        #
        # Re-read, and REFUSE if it moved rather than silently exporting either one: the
        # user consented to a named list of files for the analysis they were looking at,
        # and both "write the old one" and "quietly write the new one" break that. Pressing
        # the action again exports what is on screen now, with the same names.
        # `_stored_fit` validates the fits against `res`, so it cannot catch this — `res`
        # itself is the thing that went stale.
        # The closed-recording case is asked FIRST, because it is also a "the detection
        # moved" case (`current_detection` returns None for a document that is gone) and
        # it has a different, more useful sentence.
        if not self._session.has_document(res.doc_id):
            QtWidgets.QMessageBox.warning(
                self, "Export figure",
                f"“{Path(source_of(res.doc_id)).name}” was closed while the file dialog "
                f"was open, so there is no recording behind these figures any more.\n\n"
                f"Nothing was written.")
            return
        if self._analysis_panel.current_detection() is not res:
            QtWidgets.QMessageBox.warning(
                self, "Export figure",
                "The detection on screen changed while the file dialog was open, so these "
                "figures would have described an analysis you are no longer looking at.\n\n"
                "Nothing was written. Press Generate standard report figures… again to export the detection "
                "that is current now.")
            return
        stem = base.with_suffix("")

        # ONE staged set across the WHOLE export, not one per figure. The user consented to
        # a list of twelve files; three of them landing beside nine survivors of a previous
        # export, under one `figures_*` stem, with no marker and a status line that clears
        # after twelve seconds, is a folder of mixed vintages that is indistinguishable from
        # a consistent one. Everything is staged first and committed together, so the folder
        # goes from one whole export to the next.
        from ..core.atomic_write import StagedSet

        group = StagedSet()
        written: list[str] = []
        # The PATHS as well as the display names: the identity disclosure below is
        # scoped to the files this export actually wrote, so that a pending note about
        # some other file is never printed under this export's title.
        written_paths: list[Path] = []
        failed: list[str] = []
        #: figure name -> the real error, for the figures that BUILT but failed while
        #: being rendered/saved. Build failures have their own retry-without-fit path
        #: and wording; these are disclosed with their cause so the user chases the one
        #: bad label instead of a disk problem that does not exist.
        save_errors: dict[str, str] = {}
        commit_error: Exception | None = None
        write_error: OSError | None = None
        for name, build in builders:
            try:
                fig, _ = build()
            except Exception:
                # Retry the histograms without the fit overlay so the figure still ships.
                if name in ("dwell", "depth"):
                    try:
                        fn = figures.dwell_histogram if name == "dwell" else figures.depth_histogram
                        fig, _ = fn(events, fit=False)
                    except Exception:
                        failed.append(name)
                        continue
                else:
                    failed.append(name)
                    continue
            try:
                out = figures.save_figure(fig, f"{stem}_{name}.pdf", group=group)
                written.append(Path(out["pdf"]).name)
                written_paths.extend(Path(v) for v in out.values())
            except OSError as exc:
                # The FOLDER, not the figure: a permission or disk error at staging
                # hits every figure the same way, so pressing on would only collect
                # the same error again — and the "no figures could be exported from
                # the current detection result" summary would then blame the user's
                # data for a directory they cannot write (EXP-01). Stop and report
                # it below as what it is, the way _export_events already does.
                write_error = exc
                break
            except Exception as exc:
                # THIS figure failed while being rendered/saved (fig.savefig raises at
                # DRAW time — e.g. invalid mathtext in a title carried from the data
                # file), not the folder: save_figure has withdrawn the family from the
                # shared staged set, so the healthy figures still commit below. The
                # figure is disclosed BY NAME with its real error — before this, the
                # orphaned staged entry sank the whole export at commit and the dialog
                # blamed the environment ("could not be put in place") in a perfectly
                # writable folder.
                failed.append(name)
                save_errors[name] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass

        if write_error is not None:
            # Environmental, not per-figure. Nothing was committed (everything was
            # still staged), so the folder is exactly as it was — say so, and name
            # the real cause with the same sentence the events export uses.
            group.abandon()
            QtWidgets.QMessageBox.warning(
                self, "Export figures",
                f"Could not write the export:\n{write_error}\n\n"
                f"Nothing was written — every file already in the folder is exactly "
                f"as it was.")
            return

        if written:
            try:
                group.commit()
            except Exception as exc:      # the commit is where a full disk finally bites
                commit_error = exc
                written = []
                failed = [name for name, _ in builders]
        else:
            group.abandon()

        if commit_error is not None:
            QtWidgets.QMessageBox.critical(
                self, "Export figures",
                f"Nothing was written.\n\nThe figures were prepared but could not be put in "
                f"place next to {stem.name} ({commit_error}). Every file that was already "
                f"there is exactly as it was.")
            return
        if not written:
            msg = "No figures could be exported from the current detection result."
            if save_errors:
                # ...and when a figure built but failed to render, say WHICH and WHY —
                # "no figures could be exported" alone points the user at their data.
                msg += "\n\nFailed while rendering:\n" + "\n".join(
                    f"{stem.name}_{n}: {err}" for n, err in save_errors.items())
            QtWidgets.QMessageBox.warning(self, "Export figure", msg)
            # ...and the leftovers, on THIS path too. The partial-failure path below
            # reports every consented output an export could not replace, with its
            # vintage — the whole point being that a file of that name from a PREVIOUS
            # export now sits under this export's stem and reads as part of this one.
            # Returning here skipped that, so 3-of-4 figures failing got the dated
            # leftover dialog and 4-of-4 — strictly worse, every consented file left
            # unreplaced — got nothing. The user learns least in the case where the
            # most is stale. `failed` is every builder on this branch, which is exactly
            # the set whose names may still be on disk.
            self._report_unreplaced_figures(stem, failed, errors=save_errors)
            return

        msg = f"Wrote {len(written)} figure(s) next to {stem.name} (PDF + PNG + CSV each)."
        # The amber line's own verdict on the table the figures drew, in its own sentence:
        # the columns it badges are the numbers these figures plot as they stand.
        stale_columns, stale = self._analysis_panel.table_staleness(
            res.doc_id, res.channel, res.module_id)
        if stale_columns:
            from ..core.sources import detector_name

            msg += (f" ⚠ They draw the {detector_name(res.module_id)} table of "
                    f"“{Path(source_of(res.doc_id)).name}”, which is stale: {stale}")
        msg += "\n" + self._fit_provenance_note(dwell_fit, depth_fit)
        self.statusBar().showMessage(msg, 12000)
        # A figure that could not be built is not merely "skipped": if a file of that name
        # is already there it is now a survivor of an EARLIER export sitting under this
        # export's stem. Name those files, with the date they are from, in a dialog — a
        # status line the user may not be looking at is not a disclosure.
        if failed:
            self._report_unreplaced_figures(stem, failed, errors=save_errors)
        self._disclose_identity_notes("Export figures", written_paths)

    def _report_unreplaced_figures(self, stem: Path, failed: list[str],
                                   errors: dict[str, str] | None = None) -> None:
        """Name every consented output this export could not replace, and its vintage.

        Two different situations, and the user needs to be told them apart:

        * the figure could not be built and NOTHING of that name is on disk — a plain
          "not exported", which is what the old status line said;
        * the figure could not be built and a file of that name IS on disk — it is from a
          previous export, it now sits under this export's stem, and every tool that reads
          the folder (including the user, next month) will take it for part of this one.

        Only the second needs a dialog, and it gets one, with each file's modification date.

        ``errors`` carries the real cause for the figures that BUILT but failed while
        being rendered/saved (name -> "ExcType: message"). Those are named WITH their
        error — "could not be built" alone sent the user to their data for what was one
        bad label in one figure's title.
        """
        from ..plot import figures

        # Each failed figure, with its real error when one was captured.
        def _named(name: str) -> str:
            return f"{name} — {errors[name]}" if errors and name in errors else name

        stale: list[str] = []
        for name in failed:
            for p in figures.figure_output_paths(f"{stem}_{name}.pdf"):
                try:
                    when = QtCore.QDateTime.fromSecsSinceEpoch(
                        int(p.stat().st_mtime)).toString("dd MMM yyyy HH:mm")
                except OSError:
                    continue
                stale.append(f"{p.name}   ({when})")
        if not stale:
            self.statusBar().showMessage(
                f"{len(failed)} figure(s) could not be built and were not exported: "
                f"{', '.join(_named(n) for n in failed)}.", 12000)
            return
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Export figures")
        box.setText(f"{len(stale)} file(s) under this name are left over from an earlier "
                    f"export.")
        box.setInformativeText(
            f"These figures could not be built from the current detection "
            f"({', '.join(_named(n) for n in failed)}), so the files of that name were "
            f"NOT replaced and still "
            f"hold the previous export:\n\n" + "\n".join(stale) +
            f"\n\nDelete them, or export to a different base name, before treating "
            f"{stem.name}_* as one set.")
        box.exec()

    @staticmethod
    def _figure_export_paths(base: Path, names: list[str]) -> list[Path]:
        """Every file the figure export may write for base name ``base``.

        The same ``f"{stem}_{name}.pdf"`` the write loop hands to ``figures.save_figure``,
        expanded through ``figures.figure_output_paths`` — so the set the user confirms is
        derived from the writer's own declaration rather than a second guess at it.
        """
        from ..plot import figures

        stem = Path(base).with_suffix("")
        return [p for name in names
                for p in figures.figure_output_paths(f"{stem}_{name}.pdf")]

    def _stored_fit(self, res, module_id: str):
        """The FitResult already computed for this selection under ``module_id``.

        Returns ``(fit, reason)`` — ``fit`` is None when the export must fall back
        to refitting, and ``reason`` says why, for the status line. Every guard
        matters because a figure is where a number stops being reviewable:

        * calibration fingerprint — a fit taken against a different gain/offset
          describes different depths;
        * sample size — re-running detection (or curating events) leaves the old
          fit in the store, and stamping a fit of yesterday's events onto today's
          histogram is worse than a documented refit;
        * the column and the table — a fit of another quantity, or one that records
          another table, describes something else.

        ``modules.report_fit_problem`` applies them: the first two are
        ``modules.stored_fit_problem``, the table is the Results dock's Fits tab's
        verdict (``modules.judge_stored_fit``). A Plots-window fit of ``depth_pA`` sits
        in ``fit.depth`` too, and was drawn on the ``frac_depth`` histogram as "your
        saved fit". A fit that records no table is still used, as it always was.
        """
        from ..analysis.modules import report_fit_problem

        try:
            stored = self._session.results.get(res.doc_id, res.channel, module_id)
        except Exception:
            return None, "no saved fit"
        if stored is None or not hasattr(stored.payload, "predict"):
            return None, "no saved fit"
        problem = report_fit_problem(stored, res)
        if problem:
            return None, problem
        return stored.payload, ""

    @staticmethod
    def _fit_provenance_note(dwell, depth) -> str:
        """Say, in the status line, WHICH fit each histogram was annotated with.

        The number on an exported figure is the one that gets published; the user
        has to be able to tell at a glance whether it is the fit they read in the
        panel or a default the exporter invented."""
        parts = []
        for label, (fit, reason) in (("dwell", dwell), ("depth", depth)):
            if fit is None:
                parts.append(f"{label}: {reason} — the figure was refit fresh at k=1")
            else:
                k = int(fit.meta.get("n_components", 1) or 1)
                parts.append(f"{label}: used your saved fit (k={k})")
        return "Fits — " + "; ".join(parts) + "."
