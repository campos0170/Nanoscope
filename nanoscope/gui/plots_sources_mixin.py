"""How data gets into the Plots window and when it is drawn: sources, columns, run state.

``_PlotsSourcesMixin`` is a part of :class:`nanoscope.gui.plots_panel.PlotsWindow`, listed
before ``QtWidgets.QMainWindow`` in its bases. It holds the data-in path (``set_data``,
the staleness hooks, the active detection), the source catalog and its row handlers, the
column pickers and their refill, the control-change handlers, the RUN gate and its
buttons, the fit-apply gate, and the current-frame access (``_frame_of``, ``_dwell_inputs``,
``_depth_inputs``).

It is a plain-Python mixin: no ``__init__``, no signals and no state of its own, so every
attribute its methods read (``self._sources``, ``self._run_btn`` ...) is set by
``PlotsWindow.__init__``. The constants and helpers it uses come from
:mod:`nanoscope.gui.plots_constants`; it never imports ``plots_panel`` or another part.

Every method below was moved out of ``plots_panel.py`` without an edit.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..analysis import modules
from ..core.sources import PlotSource, amplitude_units_label
from .plot_interaction import keep_plot_history
from .plots_constants import (
    _DWELL_COLUMNS,
    _DEPTH_COLUMNS,
    HISTOGRAM,
    SCATTER,
    PCA,
    HEATMAP,
    _PLOT_TYPES,
    _FIT_NONE,
    _MULTI_COMPONENT_FITS,
    _COLOUR_NONE,
    _COLOUR_CLUSTER,
    _NORM_COUNT,
    _NORM_DENSITY,
    _PLOT_PROMPT,
    _METHOD_GMM,
    _K_FIXED,
    _PREFER_DWELL,
    _PREFER_AMP,
    _PREFER_FEATURES,
    _IDENTIFIER_FEATURES,
    _WARN_HEX,
    _plot_column_quantity,
    _numeric_columns,
    _detection_key,
    _normalize_sources,
    _prefer,
)


class _PlotsSourcesMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    # -- data in ---------------------------------------------------------------------------------
    def set_active_detection(self, key: tuple | None) -> None:
        """Tell the window WHICH stored detection the Analysis panel is showing, as
        ``(doc_id, channel, module_id)`` (or None when there is none).

        The owner should call this before (or with) :meth:`set_data`; it costs nothing
        when it is not called, because :meth:`_active_source_key` falls back. See that
        method for what this exists to stop.
        """
        self._active_detection = tuple(key) if key else None

    def set_staleness_provider(self, provider) -> None:
        """Tell the window how the Results panel judges a stored table:
        ``provider(doc_id, channel, module_id) -> (columns, sentence)`` gives the columns
        the event table badges ⚠ on that table and the amber line's sentence for them
        (``((), "")`` when none). MainWindow sets ``AnalysisPanel.table_staleness``.

        Without one (a window built bare, as most tests build it) the window judges the
        calibration alone, as it always did (:meth:`_calibration_superseded`).
        """
        self._staleness_provider = provider

    def schedule_stale_redraw(self) -> None:
        """The Results panel re-judged its tables (``AnalysisPanel.stalenessChanged``).

        A Baseline-form edit does that without touching the session, so no re-feed reaches
        this window and its STALE mark waited for the next draw (design 13.4 Q4). The owner
        chose live, once the knob stops: one look 300 ms after the last step. Any draw in
        between answers it, since :meth:`_redraw` stops the timer."""
        if self.isVisible():
            self._stale_redraw_timer.start()

    def _redraw_stale_marks(self) -> None:
        """Redraw once if a drawn source's mark moved since the last draw.

        The mark is written into the title and Details by the draw itself, so a moved mark
        needs a draw and an unmoved one needs nothing. Not while the window is hidden, has
        drawn nothing, is mid-update, or waits on Plot for a selection the user has not
        committed: drawing would commit it for them.

        The plotted table is the one already drawn (any re-feed draws first, which stops
        the timer), so the user's zoom and pan are put back as :meth:`_presentation_changed`
        puts them back: a mark is not a reason to lose the view."""
        if (not self.isVisible() or self._updating or not self._plot_armed or self._stale
                or self._message_item is not None or self._workspace_snapshot_rendering):
            return
        if self._stale_signature() == self._stale_drawn:
            return
        view = self._plot.getViewBox()
        ranges = [list(pair) for pair in view.viewRange()]
        auto = list(view.autoRangeEnabled())
        # ...and for the same reason the zoom history stays: the draw's clear restarts it
        # as it would for new geometry, but every range in it still means what it meant.
        with keep_plot_history(self._plot):
            self._redraw()
            view = self._plot.getViewBox()
            view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
            for axis, enabled in enumerate(auto):
                if enabled:
                    view.enableAutoRange(axis=axis, enable=enabled)

    def _stale_signature(self) -> tuple:
        """The marks a draw of the current view writes: the title's STALE headline and the
        Details notes, for the DRAWN columns only (what the draw hands :meth:`_stale_headline`
        and :meth:`_column_notes`). A re-judgement of a column this plot does not draw, such
        as a Baseline edit under a ``dwell_s`` histogram, moves neither and redraws nothing.

        A cluster split or cluster colours also write the clustering features' amplitude
        note (:meth:`_cluster_amplitude_warnings`), amplitude_pA's STALE line among them,
        so that note is a mark of those draws whatever column the axes show."""
        try:
            kind = self._plot_type()
            if kind == PCA:
                primary = self.primary_source()
                res = self._pca_result
                cols = list(res.columns) if res is not None else []
                sources = [primary] if primary is not None else []
                clustered = self._cluster_result is not None
            else:
                cols = ([self._hist_col.currentText()] if kind == HISTOGRAM else
                        [self._scatter_x.currentText(), self._scatter_y.currentText()]
                        if kind == SCATTER else
                        [self._heatmap_x.currentText(), self._heatmap_y.currentText()]
                        if kind == HEATMAP else
                        [self._kde_x.currentText(), self._kde_y.currentText()])
                sources = [src for src, _df in self._drawn_frames()]
                if kind == HEATMAP:
                    sources = sources[:1]   # the one source `_draw_heatmap` draws
                colour_by = {HISTOGRAM: self._hist_colour_by,
                             SCATTER: self._colour_by}.get(kind)
                # as _draw_histogram's `split` and _draw_scatter's `by_cluster` decide it
                clustered = (len(sources) == 1 and colour_by is not None
                             and colour_by.currentText() == _COLOUR_CLUSTER
                             and self._cluster_result is not None)
            return (self._stale_headline(cols, sources),
                    tuple(self._column_notes(cols, sources)),
                    self._cluster_amplitude_warnings() if clustered else ())
        except Exception:       # pragma: no cover - a caveat must never break a plot
            return ()

    def _active_source_key(self) -> str:
        """The key of the source that IS the detection on screen in the Analysis panel,
        or ``""`` when it cannot be resolved.

        THE ANSWER TO "the window drew a different detector's table and said nothing".
        A channel can hold several detections at once (``ResultStore`` is keyed
        (doc_id, channel, module_id)), so "the current recording" no longer identifies
        a table. Measured on the owner's -1000 mV recording, 2026-09-02: Schmitt 110
        events and CUSUM 250 both stored, the panel showing CUSUM, and Plots opening
        with Schmitt's 110 rows ticked, drawn and captioned "(n = 110)".

        Three sources of the answer, in order: what the owner told us
        (:meth:`set_active_detection`); a public ``active_detection_key()`` on the
        owner window if it grows one; and -- as a bridge until MainWindow does -- its
        Analysis panel's own public ``current_detection()``. The last one is the only
        reach outside this window and it is deliberately total: every step is a
        ``getattr`` behind a ``try``, so a window with no such parent (every test that
        builds one bare, and the notebook-style callers) simply gets "" and the
        ``PlotSource.current`` flag decides, exactly as before.
        """
        key = self._active_detection
        if key is None:
            owner = self.parent()
            for probe in (lambda w: w.active_detection_key(),
                          lambda w: _detection_key(w._analysis_panel.current_detection())):
                try:
                    key = probe(owner)
                except Exception:
                    key = None
                if key:
                    break
        if not key or len(tuple(key)) < 3:
            return ""
        doc_id, channel, module_id = tuple(key)[:3]
        for k, src in self._sources.items():
            if (src.doc_id, src.channel, src.module_id) == (doc_id, int(channel), module_id):
                return k
        return ""

    def set_data(
        self,
        sources,
        provenance: dict[str, dict] | None = None,
        *,
        catalog_only: bool = False,
    ) -> None:
        """Install the per-event tables to explore, with their detection provenance.

        Args:
            sources: EITHER a sequence of :class:`~nanoscope.core.sources.PlotSource`
                (what :class:`~nanoscope.gui.main_window.MainWindow` pushes -- one
                per open recording/segment with a stored detection, plus the manual
                table), OR the legacy ``{label: DataFrame}`` mapping, which is
                wrapped into single-colour sources keyed by their label. Pass FULL
                event frames (not pre-filtered): the *Use accepted events only*
                check does that filtering, per source, and is enabled only when a
                checked source carries an ``accepted`` column.
            provenance: legacy ``{label: EventSet.provenance}``, applied only to the
                mapping form. A :class:`PlotSource` already carries its own -- and
                that pairing is load-bearing: the dwell fit uses the detector's real
                ``min_dwell_s`` instead of the shortest observed dwell (the
                difference is a ~25% bias in tau at a 50 us floor against a 200 us
                tau, and it is invisible on the plot), and with an overlay up EACH
                source must be fit with ITS OWN floor. A source without provenance
                still fits, but the window says out loud what it fell back to.

        Sources still present after a refresh keep their checked state, so
        re-pulling data doesn't yank the user off their view. Any previously
        computed clustering/PCA is dropped -- the row set may have changed
        underneath it.

        ``catalog_only`` prepares sources without drawing. An already drawn figure
        keeps its exact inputs, caches and controls until the next explicit Plot
        consumes the prepared catalog. This also leaves the old result intact if
        the owner stops between preparing sources and requesting that Plot. A fresh
        page installs the catalog immediately so a frozen snapshot can resolve it.
        """
        normalized = _normalize_sources(sources, provenance)
        if catalog_only and self._plot_armed:
            self._figure_pending_sources = normalized
            return
        self._figure_pending_sources = None
        self._install_source_catalog(normalized, draw=not catalog_only)

    def _install_source_catalog(self, sources: dict[str, PlotSource], *, draw: bool) -> None:
        """Install normalized sources; the explicit run owns the optional draw."""
        # NEW ROWS SUPERSEDE ANY ADVISORY WORK IN FLIGHT, IMMEDIATELY.
        #
        # This is the entry point a curation click reaches, and every deferred note is
        # keyed on the bytes of the sample it was computed from -- so a batch that was
        # running when this was called is, by construction, about events that are no
        # longer what the user is looking at. Waiting for the settle timer to notice
        # (400 ms) is not free: the jobs are `scipy.optimize.minimize(Nelder-Mead)`,
        # which is PYTHON, so a stale batch does not merely waste a core, it holds the
        # GIL against the next click. Measured on the owner's -700 mV run with
        # `exponential + gamma (bimodal)` on screen -- the family whose own point fit
        # (~250 ms) makes its click period brush the 400 ms settle -- a batch left to
        # run alongside the burst took the median click from 258 ms to 1014 ms. Stopping
        # it here takes it back.
        #
        # Nothing is lost: the redraw at the end of this method asks for whatever the
        # new table needs, and the timer starts again from the user's last action.
        self._abort_note_job()

        prev_checked = self.selected_source_keys()
        prev_primary = self._primary_key()

        self._sources = dict(sources)
        self._cluster_result = None
        self._pca_result = None

        want = [k for k in prev_checked if k in self._sources]
        if not want:
            want = [k for k in (prev_primary,) if k and k in self._sources]
        if not want:
            # First push (or every previous source closed): follow the app's active
            # DETECTION so opening Plots shows the table the user was already looking
            # at -- not merely the active recording, which since one-source-per-detector
            # discovery can offer several tables. `_active_source_key` says what that
            # distinction cost. The `PlotSource.current` flag is the fallback: it is
            # right whenever the channel carries one detection, and it is a GUESS
            # (the channel's first detector) when it carries more and nobody named the
            # active one.
            current = self._active_source_key() or next(
                (k for k, s in self._sources.items() if s.current), None)
            want = [current or next(iter(self._sources), "")]
            want = [k for k in want if k]

        self._updating = True
        # NOT `clear()`: this can be reached from inside a source row's own
        # `itemChanged` (a tick, or a detection result landing mid-click), where
        # destroying the rows would leave Qt marshalling a freed one. See `_retire_row`.
        self._empty_list_widget(self._source_list)
        for src in self._sources.values():
            item = QtWidgets.QListWidgetItem(self._legend_label(src))
            item.setData(Qt.UserRole, src.key)
            # EDITABLE, because this row IS the legend key (the hint above says so) and
            # owner note #15 is that the legend was not editable. Double-clicking a row
            # renames the series everywhere it is named; the source's real label is kept
            # on the PlotSource and restored by clearing the edit back to empty.
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Checked if src.key in want else Qt.Unchecked)
            item.setForeground(QtGui.QBrush(QtGui.QColor(src.colour)))
            item.setToolTip(self._source_tooltip(src))
            self._source_list.addItem(item)
        if self._source_list.count():
            self._source_list.setCurrentRow(
                max(0, next((i for i in range(self._source_list.count())
                             if self._source_list.item(i).data(Qt.UserRole) in want), 0)))
        self._updating = False
        # THE ONE PLACE ``auto_plot`` IS READ. A push of data is a programmatic commit
        # by default -- MainWindow's Refresh, the test corpus and every caller that
        # expects a plot after ``set_data`` keep exactly the behaviour they had. Set
        # ``auto_plot = False`` before the first push and the window comes up on the
        # prompt instead, which is the "opens showing an empty plot" half of Q1; the
        # gate on the user's own gestures is on either way.
        # ARRIVING DATA IS NOT A CONFIGURATION CHANGE, so it is not an undo step --
        # `EventHistory.set_suspended`'s rule ("a reconstruction never becomes an offer to
        # undo the project you just opened") one object down. Which columns the pickers
        # end up on and which rows are ticked are a CONSEQUENCE of the push, not something
        # the user did, and MainWindow re-pushes on every detection; recording each one
        # would put "the window before your data existed" at the bottom of every stack and
        # make Undo untick everything. The stack SURVIVES -- the styling steps on it are
        # still applicable, and `_installable` resolves away a column that has gone --
        # only the base moves.
        was, self._history_suspended = self._history_suspended, True
        try:
            self._on_sources_changed(draw=draw and (self.auto_plot or self._plot_armed))
            if draw and not self._plot_armed:
                self._show_plot_prompt()
        finally:
            self._history_suspended = was
        self._history.rebase(self.config())
        self._sync_config_buttons()
        self._sync_template_selection()

    @staticmethod
    def _source_tooltip(src: PlotSource) -> str:
        bits = [f"{src.n_rows:,} rows"]
        if src.is_segment:
            bits.append("derived segment")
        if src.module_id:
            # NAME, then id. The row may be one of two tables from the same recording
            # (one per detector), so which detector produced these rows is the first
            # thing a hovering user needs; the id stays for the provenance-minded.
            bits.append(f"{src.detector_label} ({src.module_id})")
        md = src.provenance.get("min_dwell_s") if src.provenance else None
        if md:
            bits.append(f"Minimum dwell time = {float(md) * 1000:g} ms")
        elif src.kind == "manual":
            bits.append("no detector provenance (hand measurements)")
        # ...and the edit hint on the row itself, not only in the group's label: the
        # tooltip is what a user who is already hovering the row will read (note Q3).
        return (f"{src.display_name or src.label}\n" + "  ·  ".join(bits)
                + "\n✎ Double-click to retype this source's legend entry.")

    # -- source / column plumbing ----------------------------------------------------------------
    def _on_source_item_changed(self, item) -> None:
        """A tick changed: the drawn set changed, so everything tied to rows is stale.

        ...OR the row was RENAMED, which is how the legend is edited (owner note #15).
        One signal carries both, so the rename is resolved by reading the row's text
        AGAINST THE STYLE rather than against the source's label: whatever the row now
        says is what the override becomes, and a row that says the source's own name --
        or nothing -- removes the override rather than legending a blank entry.

        WHY THERE IS NO ``text_now != src.label`` GUARD ANY MORE (S7-FIX2 / S7-SK2).
        That guard returned before either branch below could run whenever the row was
        retyped to the source's ORIGINAL name -- the gesture a person uses to undo a
        rename by hand -- so the override stayed. MEASURED with real double-click and
        per-character key events on the owner's -700 mV source: rename to "Apo" (row,
        override, canvas all "Apo"); type "-700 mV" back -> row "-700 mV", override
        STILL "Apo", canvas STILL "Apo  (n = 982)"; a re-Plot left it; the next Refresh
        (``set_data`` rebuilds the rows from ``_legend_label``) put "Apo" back on the
        row, and the typed correction was gone. Worse, giving "Apo" to the -600 mV
        source next drew ['Apo  (n = 982)', 'Apo  (n = 236)'] on the canvas and in the
        exported SVG -- two voltages under one name, with the Sources list reading
        ['-700 mV', 'Apo']. Clearing the row to EMPTY removed the override cleanly, so
        the two ways of saying "put it back" disagreed. The branch below already did the
        right thing for exactly this text; only the guard stopped it reaching there. A
        TICK on a row whose text equals its label now costs one idempotent
        ``without_label_override`` (a no-op on a missing key), which is why the guard was
        never load-bearing."""
        if self._updating:
            return
        previous_labels = self._style.label_overrides
        try:
            key, text_now = item.data(Qt.UserRole), item.text()
        except RuntimeError:
            return   # see `_on_legend_entry_toggled`; belt and braces
        src = self._sources.get(key or "")
        if src is not None:
            text = text_now.strip()
            if text and text != src.label:
                self._style = self._style.with_label_override(src.key, text)
            else:
                self._style = self._style.without_label_override(src.key)
            # ...AND THEN THE ROW IS WRITTEN BACK FROM THE STYLE, on this path too.
            # `_set_config` states the rule (see the comment over its
            # `_sync_source_row_labels` call): a row's TEXT is a projection of the style
            # exactly as the Title box is. That held on a config RESTORE and not here,
            # because the two branches above store `text_now.strip()` while the row keeps
            # whatever the item editor committed. MEASURED: typing "  Apo, -700 mV  "
            # into a row left the Sources list reading '  Apo, -700 mV  ' while the
            # legend checklist and the drawn legend read 'Apo, -700 mV' -- one source
            # under two names -- and a re-Plot did NOT reconcile them, because nothing on
            # the draw path rewrites a row; only an undo/redo did, because that goes
            # through `_set_config`. The stripping itself is right (a legend entry with a
            # trailing space is not what anyone typed on purpose); showing the user
            # something other than what was stored is what was wrong.
            #
            # `_legend_label` RATHER THAN `text`, because it is the one reader that
            # answers "what is this source called now" from the style, and it answers for
            # both branches above in one line: the override just stored, or `src.label`
            # when the override was just removed. That second case is why the blank-name
            # write that used to sit under `if not text` is gone rather than merely
            # joined -- it was this same write, spelled out for one branch of two. The
            # third case it now also covers is the one that has no branch at all: a name
            # that differs from the label ONLY by padding ("  Apo  " over "Apo") stores
            # no override and used to leave the padding standing on the row.
            #
            # UNDER `_updating`, AND IN PLACE. `_updating` because `setText` re-emits
            # `itemChanged` synchronously and this handler is one of that signal's own
            # receivers: unguarded, the write re-enters here, reads the window's own text
            # back as a fresh user gesture and pushes a SECOND history step for one
            # rename. The flag is the first thing this method tests, so the re-entry
            # returns immediately -- the same guard the blank-name write already used.
            # In place because this can run from INSIDE Qt's delivery of that same
            # `itemChanged`, which still holds `item` and re-marshals it to every
            # receiver it has not reached yet: `setText` on an existing row frees
            # nothing, while a `clear()`, a `takeItem` or a fresh `QListWidgetItem` here
            # would rebuild the EXC_BAD_ACCESS the shipped build died on (the rule is
            # stated at length in `_sync_source_row_labels`).
            #
            # ONLY WHAT DIFFERS IS WRITTEN, for `_write_legend_row`'s reason: the
            # ordinary rename needs no stripping, and that common case should cost one
            # string comparison and emit nothing at all.
            shown = self._legend_label(src)
            if item.text() != shown:
                was, self._updating = self._updating, True
                try:
                    item.setText(shown)
                finally:
                    self._updating = was
        # WHICH OF THE TWO IT WAS decides which side of the run gate this falls on. A
        # TICK changes the rows that are drawn, so it makes the plot stale; a RENAME is
        # legend text, which is as cosmetic as a colour and redraws live. One Qt signal
        # carries both, so the checked set is compared with the one the last draw used.
        if self._style.label_overrides != previous_labels:
            self._presentation_changed()
        elif tuple(self.selected_source_keys()) != self._checked_snapshot:
            self._on_sources_changed(draw=False)
            self._mark_stale("the data selection")
        else:
            self._presentation_changed()
        # THE HISTORY HOOK, BY HAND -- see `_on_legend_entry_toggled`.
        self._history_touch()

    def _on_source_current_changed(self, *_a) -> None:
        """Highlighting a row without ticking it does not change the plot; it only
        moves the keyboard focus. Redrawing here would fight the user's tick."""
        return

    def _highlighted_source(self):
        item = self._source_list.currentItem()
        return self._sources.get(item.data(Qt.UserRole)) if item is not None else None

    def _save_project_source_label(self):
        source = self._highlighted_source()
        setter = getattr(self._session, "set_source_appearance", None)
        if source is None or not callable(setter):
            return
        try:
            setter(source.key, label=self._legend_label(source))
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Project source name", str(exc))
            return
        self.statusBar().showMessage(
            "Source name saved for new figures. Existing saved figures retain their own labels.", 6000)

    def _pick_project_source_color(self):
        source = self._highlighted_source()
        setter = getattr(self._session, "set_source_appearance", None)
        if source is None or not callable(setter):
            return
        chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(source.colour), self, "Project source color")
        if not chosen.isValid():
            return
        labels = getattr(self._session, "source_labels", None)
        setter(source.key, label=labels.label(source.key) if labels is not None else "", color=chosen.name())
        # Only the next explicit Plot consumes this source appearance. Cached
        # bars/points and fitted results remain exactly as they were drawn.
        self._sources[source.key] = _dc_replace(source, colour=chosen.name())
        item = self._source_list.currentItem()
        was, self._updating = self._updating, True
        try:
            item.setForeground(QtGui.QBrush(chosen))
        finally:
            self._updating = was
        self.statusBar().showMessage(
            "Project source color saved. Use Plot/Generate to apply it to a source-colored figure. "
            "Current numerical results and figure are unchanged.", 8000)

    def _on_sources_changed(self, *, draw: bool = True) -> None:
        """Re-sync everything that depends on WHICH sources are checked.

        ``draw=False`` does the syncing and stops short of the picture: the run gate's
        caller has already marked the plot stale and the user has not pressed Plot yet.
        The syncing itself is never skipped -- the column pickers must offer the new
        source's columns before the user can choose one to plot.
        """
        # A new row set invalidates labels/scores tied to the old rows.
        self._cluster_result = None
        self._pca_result = None
        # NO BLANKET UNPIN HERE ANY MORE. This used to clear the pinned axis range on
        # every source change, INCLUDING a plain Refresh -- while leaving the typed title
        # standing, which is the same bug in mirror image and was the evidence the two
        # were being cleared by different ad-hoc rules. Re-pulling the same columns is
        # not a change of analysis at all (the owner, asked directly: "no need to clear
        # the data... the user selects the data from the file manager as is done now"),
        # so nothing is cleared for it. When the refill below genuinely DOES move a
        # column -- a refreshed source that no longer carries it -- `_repopulate_columns`
        # ends in `_rebase`, and the pin, the fit window and the caption go together.
        self._repopulate_columns()
        frames = [s.df for s in self.selected_sources()]
        has_accepted = any(df is not None and "accepted" in df.columns for df in frames)
        self._accepted_check.setEnabled(bool(has_accepted))
        self._sync_norm_default()
        self._sync_cluster_split_toggle()
        self._sync_filter_limited_toggle()
        self._checked_snapshot = tuple(self.selected_source_keys())
        # A source that is GONE takes its legend decisions with it; one merely unticked
        # keeps them (see `PlotConfig.with_pruned_legend`).
        self._prune_legend_memory()
        self._sync_legend_entry_list()
        if draw:
            self._armed_redraw()

    def _sync_norm_default(self) -> None:
        """Default to DENSITY the moment a second source is drawn (and back to count
        when it is alone), unless the user has chosen a normalization themselves.

        Overlaying a 5000-event and a 200-event recording as raw counts hides the
        second one under the axis; density is the only default that makes
        the comparison the user just asked for actually readable."""
        if self._norm_user_set or not hasattr(self, "_norm_combo"):
            return
        want = _NORM_DENSITY if len(self.selected_sources()) > 1 else _NORM_COUNT
        if self._norm_combo.currentText() != want:
            was, self._updating = self._updating, True
            self._norm_combo.setCurrentText(want)
            self._updating = was

    def _on_norm_moved(self, _text: str) -> None:
        """The y quantity moved by SOMETHING OTHER THAN THE ONE WRITER, so whatever an
        install could not honour is no longer the reason for what is on screen.

        ``_updating`` is what separates the two, and it is the same flag every other
        handler in this window reads: :meth:`_set_config` and :meth:`_sync_norm_default`
        both write this combo under it. The writer therefore does not clear its own
        record through here -- it decides the record itself, in the same statement that
        writes the combo -- and the automatic density default does not clear it either,
        which is deliberate: nobody re-decided anything when the window swung to density
        because a second source was ticked, and the refusal is still the true account of
        why the requested quantity is not on screen. See :class:`_NormRefusal`.

        A PICK is a change of what the y axis counts, though, so on the histogram it is a
        move of the y quantity (owner's reset rule, beyond his list): the typed words and the
        y range go back to automatic through :meth:`_rebase`, in the step of the change itself
        (this slot runs ahead of ``_changed``, which draws and records it), and the sentence says
        what the histogram is now set to show. The automatic swing runs under ``_updating`` and
        takes nothing; neither does a Normalize set on a page that does not draw a histogram.
        """
        if not self._updating:
            self._norm_refusal.clear()
            if self._plot_type_combo.currentText() == HISTOGRAM:
                self._rebase(y_moved=True)

    def _sync_cluster_split_toggle(self) -> None:
        """One owner for the colour channel: SOURCE while an overlay is up, CLUSTER
        only when a single source is drawn. Rather than let the two fight silently,
        the cluster-split control is disabled (and explains itself) during an
        overlay, and any selection it held is reset."""
        if not hasattr(self, "_hist_colour_by"):
            return   # may fire during construction, before the histogram page exists
        multi = len(self.selected_sources()) > 1
        self._hist_colour_by.setEnabled(not multi)
        self._hist_colour_by.setToolTip(
            "Colour already means SOURCE while more than one source is drawn — untick "
            "all but one to split by cluster instead."
            if multi else
            "Split the histogram by cluster label after Run clustering — the other half "
            "of the clustering workflow.")
        if multi and self._hist_colour_by.currentText() != _COLOUR_NONE:
            was, self._updating = self._updating, True
            self._hist_colour_by.setCurrentText(_COLOUR_NONE)
            self._updating = was

    def _on_accepted_changed(self, _checked: bool) -> None:
        if self._updating:
            return
        # Filtering to accepted-only changes which rows exist; drop stale fits.
        self._cluster_result = None
        self._pca_result = None
        # ...and it is a DATA SELECTION change, so a CLICK waits for Plot like a tick
        # does. Code moving the check is the same confirmation `set_fit` is (see
        # :meth:`_from_the_user`): this box has no `set_*` of its own, so owner and test
        # code drive the widget itself and must still take effect at once.
        if self._from_the_user(self._accepted_check):
            self._mark_stale("the accepted-events filter")
        else:
            self._armed_redraw()

    def _repopulate_columns(self) -> None:
        """Refill every column picker from the checked sources' numeric columns.

        With ONE source that is simply its columns. With an overlay it is the
        INTERSECTION: a column only one source carries cannot be overlaid, and
        offering it would draw a single curve while the legend promised two."""
        selected = self.selected_sources()
        if not selected:
            cols: list[str] = []
        else:
            cols = _numeric_columns(selected[0].df)
            for src in selected[1:]:
                other = set(_numeric_columns(src.df))
                cols = [c for c in cols if c in other]

        self._updating = True
        try:
            x_prev, y_prev = self._scatter_x.currentText(), self._scatter_y.currentText()
            kx_prev, ky_prev = self._kde_x.currentText(), self._kde_y.currentText()
            hx_prev, hy_prev = self._heatmap_x.currentText(), self._heatmap_y.currentText()
            h_prev = self._hist_col.currentText()
            feat_prev = set(self.feature_selection())

            for combo in (self._hist_col, self._scatter_x, self._scatter_y,
                          self._kde_x, self._kde_y, self._heatmap_x, self._heatmap_y):
                combo.clear()
                combo.addItems(cols)
                for i, name in enumerate(cols):
                    if name in {"depth_pA", "amplitude_pA"}:
                        combo.setItemData(i, (
                            name + " — Original whole-event median depth; unchanged by Curate amplitude choices."
                            if name == "depth_pA" else
                            name + " — Selected amplitude applied in Curate. Recorded units: "
                            + amplitude_units_label(src.df for src in selected)
                            + ". Review method, interval scope and stale-value warnings before pooling."),
                            QtCore.Qt.ToolTipRole)

            measured = [c for c in cols if c not in _IDENTIFIER_FEATURES]
            identifiers = [c for c in _IDENTIFIER_FEATURES if c in cols]
            self._features.set_separator_before(
                identifiers[0] if identifiers else _IDENTIFIER_FEATURES[0], "Identifiers")
            from ..core.quantity_display import column_entry
            self._features.set_entries(
                [column_entry(c, _plot_column_quantity(c)) for c in measured + identifiers], (),
                [c for c in _PREFER_FEATURES if c in cols])

            if not cols:
                return

            self._hist_col.setCurrentText(h_prev if h_prev in cols
                                          else (_prefer(cols, _PREFER_DWELL) or cols[0]))
            self._scatter_x.setCurrentText(x_prev if x_prev in cols
                                           else (_prefer(cols, _PREFER_DWELL) or cols[0]))
            self._scatter_y.setCurrentText(y_prev if y_prev in cols
                                           else (_prefer(cols, _PREFER_AMP, fallback=1) or cols[-1]))
            # KDE defaults follow the signature translocation figure: dwell on X,
            # fractional blockade on Y when the table has it (it survives a gain
            # revision; see _DENSITY_Y's rationale in main_window), amplitude otherwise.
            self._kde_x.setCurrentText(kx_prev if kx_prev in cols
                                       else (_prefer(cols, _PREFER_DWELL) or cols[0]))
            self._kde_y.setCurrentText(
                ky_prev if ky_prev in cols
                else ("frac_depth" if "frac_depth" in cols
                      else (_prefer(cols, _PREFER_AMP, fallback=1) or cols[-1])))
            # The heatmap defaults to Event waveforms' own pair -- dwell against the
            # median depth -- so the two count images open on the same axes.
            self._heatmap_x.setCurrentText(hx_prev if hx_prev in cols
                                           else (_prefer(cols, _PREFER_DWELL) or cols[0]))
            self._heatmap_y.setCurrentText(
                hy_prev if hy_prev in cols
                else (_prefer(cols, _PREFER_AMP, fallback=1) or cols[-1]))

            keep = feat_prev & set(cols)
            if not keep:
                keep = ({c for c in _PREFER_FEATURES if c in cols}
                        or set([c for c in cols if c not in _IDENTIFIER_FEATURES][:4]))
            self._features.set_checked(keep)
        finally:
            self._updating = False
        # The pickers may have moved on their own -- a refreshed source that no longer
        # carries the old column -- and a move nobody typed invalidates exactly what a
        # typed one does. One rebase after the whole refill, not one per combo.
        self._rebase()

    def _changed(self, *_args) -> None:
        """Numerical/display geometry controls use the existing redraw path.

        Text, typography, grid and background changes are isolated in
        :meth:`_presentation_changed`; they do not rebuild this numerical picture.
        Changes that need reconstruction (for example bin count) retain this path
        and its history behavior. All appearance controls use cached items instead.
        Source and axis-column proposals keep their separate :meth:`_mark_stale` gate.

        With nothing committed yet there is nothing to restyle, so the prompt stays up.
        """
        if self._updating:
            return
        if not self._plot_armed:
            self._show_plot_prompt()
        else:
            self._redraw()
        # THE UNDO HOOK. `_wire_history`'s walk covers everything the user MOVES; this
        # covers what changes without a control moving -- a colour picked in a modal, a
        # legend dragged on the canvas, a source row renamed into `label_overrides`.
        # AFTER the draw, not before: a draw can write a config value back (Auto-BIC's
        # chosen k lands in the spin and in `_applied_fit_k`), and a step taken first
        # would name a state that was never on screen.
        self._history_settle()

    # -- the RUN gate (round-4 Q1) ---------------------------------------------------------------
    def _run_plot(self) -> None:
        """Commit every pending selection and draw. THE one recompute in this window.

        Wired to the Plot button and to BOTH "Apply fit" buttons, which is the answer
        to "should this and Apply fit be one gate": they are one gate with three
        doors. Two buttons that each commit a different half of the same picture is
        how a user ends up reading a gamma's k off a plot still showing an
        exponential -- the exact failure owner note #12 was written about, re-created
        one level up.
        """
        errors = self._fit_range_input_notes() if self._plot_type() == HISTOGRAM else []
        if errors and self._fit_combo.currentText() != _FIT_NONE:
            self._validate_fit_range_fields()
            self.statusBar().showMessage(" ".join(errors), 15000)
            self._set_details(self.details_text() + "\n\nFit not run: " + " ".join(errors))
            return
        pending = getattr(self, "_figure_pending_sources", None)
        if pending is not None:
            self._install_source_catalog(pending, draw=False)
            self._figure_pending_sources = None
        self._commit_fit_selection()
        self._commit_scatter_fit_selection()
        self._armed_redraw()
        if getattr(self, "_phase2_figure_managed", False):
            # An explicit Update/Apply is the publication boundary. Closing or evicting
            # one of several figures must never publish whichever fit happened to close last.
            published = self._persist_results()
            self.figureResultsPublished.emit({
                "figure_id": str(getattr(self, "_nanoscope_figure_id", "")),
                "results": published, "sources": self.figure_source_recipes()})

    def _armed_redraw(self) -> None:
        """Draw, and record that what is on screen now matches the controls.

        Every programmatic entry point ends here rather than at :meth:`_redraw`, so
        ``set_data``/``set_fit``/``set_plot_type`` keep drawing on the spot AND clear
        any staleness the same click would have cleared by hand.
        """
        self._plot_armed = True
        self._stale = False
        self._stale_reason = ""
        self._stale_reasons = []
        # ...which, for the KDE overlay, is the moment the proposal becomes the picture:
        # here rather than in `_run_plot` because every run gesture -- Plot, a setter,
        # `apply_config`, an undo -- ends here, and one that cleared the ● without
        # committing the box would leave a ticked overlay silently undrawn.
        self._applied_kde_overlay = bool(self._hist_kde.isChecked())
        # The applied fit has just become the selected one, so both Apply buttons and
        # the Plot button lose their ● together — they are one gate and they must not
        # be able to disagree about whether anything is pending.
        self._sync_fit_apply_buttons()
        self._sync_run_state()
        self._redraw()
        # The APPLIED fit is config and `_commit_fit_selection` moves it with no control
        # having changed, so the run gesture needs the hook of its own. See `_changed`.
        self._history_settle()

    def _mark_stale(self, reason: str) -> None:
        """The drawn plot no longer matches the controls; say so and wait for Plot.

        ``reason`` is shown to the user verbatim, because "press Plot again" without
        saying WHAT changed is indistinguishable from the window having glitched.
        """
        if not self._plot_armed:
            self._show_plot_prompt()
        else:
            self._stale = True
            self._stale_reason = reason
            if reason not in self._stale_reasons:
                self._stale_reasons.append(reason)
            self._sync_run_state()
        self._history_settle()  # see `_changed`; free, because the hook ends in a diff

    def _show_plot_prompt(self) -> None:
        """The empty canvas a never-run window carries, with the instruction on it."""
        self._stale = False
        self._stale_reason = ""
        self._stale_reasons = []
        self._sync_run_state()
        self._message(_PLOT_PROMPT)
        # `_message` bypasses `_redraw`, so the legend checklist has to be emptied here
        # or it would go on listing the rows of a plot that is no longer on screen.
        self._sync_legend_entry_list()

    def _sync_run_state(self) -> None:
        """Put the gate's state on the Plot button and on the banner beneath it.

        ``pending`` deliberately includes an un-applied FIT as well as a stale
        selection: they are one gate, so the Plot button has to badge everything
        pressing it would commit, not only the half it was named after."""
        if not hasattr(self, "_plot_btn"):
            return   # fires during construction, before the Plot group exists
        pending = self._stale or self._pending_fit_change()
        self._plot_btn.setText("Plot ●" if pending else "Plot")
        self._plot_btn.setToolTip(
            f"Redraw — {self._stale_reason} changed since the plot on screen was drawn."
            if self._stale else
            "Draw the checked sources with the plot type and columns chosen above."
            if not self._plot_armed else
            "Redraw with the current selection.")
        if hasattr(self, "_run_action"):
            self._run_action.setText("Plot / Apply ●" if pending else "Plot / Apply")
            self._run_action.setToolTip(self._plot_btn.toolTip())
        self._plot_state.setVisible(bool(self._stale))
        if self._stale:
            # Hosted in the Explorer the Plot button is hidden and the verb is the
            # toolbar's Update plot (⌘Return); the sentence names the one the user sees.
            verb = "Update plot (⌘Return)" if self._hosted else "Plot"
            self._plot_state.setText(
                f"⚠ {self._stale_reason} changed — the plot below is the PREVIOUS "
                f"selection. Press {verb} to redraw.")
            self._plot_state.setStyleSheet(f"color:{_WARN_HEX};")
        self.pendingChanged.emit()

    def _on_plot_type_changed(self, name: str) -> None:
        if name in _PLOT_TYPES:
            self._pages.setCurrentIndex(_PLOT_TYPES.index(name))
        # THE ONE RULE: a different plot has different axes, so the pins, the typed
        # limits, the fit window and the caption are settled by `rebased` rather than by
        # a list of clears here. Run BEFORE the `_updating` gate below because
        # `run_pca`/`reset_state` move this combo themselves and their new analysis is
        # every bit as incompatible with the old pins as a click's would be.
        self._rebase()
        if self._updating:
            # A BULK WIDGET UPDATE, not a choice: `run_pca` moves this combo to PCA
            # itself and `reset_state` moves it back to Histogram, and in both cases the
            # caller owns the draw that follows -- drawing here as well would cost a
            # whole extra render per Run PCA. (The caption such a caller owns is kept by
            # `_rebase` above, which is where that exception is stated.)
            return
        if self._from_the_user(self._plot_type_combo):
            self._mark_stale("the plot type")
        else:
            self._armed_redraw()   # code named the plot type; naming it IS the run

    # -- the fit APPLY gate (owner note #12) -----------------------------------------------------
    @staticmethod
    def _from_the_user(widget) -> bool:
        """Whether the PERSON just moved ``widget``, as opposed to code moving it.

        Qt's value signals (``currentTextChanged``, ``toggled``, ``valueChanged``) fire
        identically for a click and for ``setCurrentText``/``setChecked``, and the Apply
        gate has to tell the two apart. Owner note #12 is about the GESTURE: picking a
        model out of the dropdown must not commit it, while the programmatic contract --
        :meth:`set_fit`, :meth:`set_fit_components`, :meth:`set_scatter_fit`, and the
        owner/test code that drives these widgets directly -- IS itself the confirmation
        and must still take effect at once. Gating both would mean "apply the fit"
        having no caller at all outside a mouse.

        A real gesture always leaves a fingerprint on the widget: a combo the user is
        picking from has its popup on screen, and a control they are typing into or
        clicking has the keyboard focus (Qt hands it over on mouse PRESS, before the
        value moves). Code touches neither, so code reads as "already confirmed".
        """
        view = widget.view() if isinstance(widget, QtWidgets.QComboBox) else None
        if view is not None and view.isVisible():
            return True
        return bool(widget.hasFocus())

    def _fit_selection_moved(self, widget) -> None:
        """A histogram-fit control moved: arm Apply, or commit when code moved it."""
        if self._updating:
            return
        if self._from_the_user(widget):
            self._sync_fit_apply_buttons()
        else:
            self._apply_fit()

    def _commit_fit_selection(self) -> None:
        """Copy the histogram-fit PROPOSAL into the applied state, without drawing.

        Everything a draw reads about the fit lives in ``_applied_*``; the combo and
        the spin are only a proposal until this runs. Split out from
        :meth:`_apply_fit` so :meth:`_run_plot` can commit the fit and the data
        selection in one pass and draw ONCE, rather than twice for one click."""
        self._applied_fit = self._fit_combo.currentText()
        self._applied_fit_k = int(self._fit_k.value())
        self._applied_fit_k_auto = bool(self._fit_k_auto.isEnabled()
                                        and self._fit_k_auto.isChecked())
        self._rebase()      # a new applied model takes the fit label's wording with it

    def _commit_scatter_fit_selection(self) -> None:
        self._applied_scatter_fit = self._scatter_fit.currentText()
        self._rebase()

    def _apply_fit(self) -> None:
        """Commit the selected histogram fit and draw it.

        The PROGRAMMATIC commit: ``set_fit``/``set_fit_components`` reach it through
        :meth:`_fit_selection_moved`. The buttons do not -- they run
        :meth:`_run_plot`, which commits this and the rest of the selection together."""
        self._commit_fit_selection()
        self._armed_redraw()

    def _apply_scatter_fit(self) -> None:
        """Commit the selected scatter best-fit line and draw it (programmatic path)."""
        self._commit_scatter_fit_selection()
        self._armed_redraw()

    def _on_scatter_fit_changed(self, _name: str) -> None:
        if self._updating:
            return
        if self._from_the_user(self._scatter_fit):
            self._sync_fit_apply_buttons()
        else:
            self._apply_scatter_fit()

    def _sync_fit_apply_buttons(self) -> None:
        """Show, on the buttons themselves, that a chosen fit is NOT the drawn one.

        A gate the user cannot see is a control that stopped working: the whole failure
        mode of note #12's fix is a scientist picking `gamma`, reading the still-visible
        exponential's tau off the plot and never noticing. So a pending selection both
        ENABLES the button and marks it, and the tooltip names the model that is
        actually drawn."""
        if not hasattr(self, "_scatter_fit_apply_btn"):
            return   # fires during construction, before the scatter page exists
        pending = (self._fit_combo.currentText() != self._applied_fit
                   or int(self._fit_k.value()) != int(self._applied_fit_k)
                   or bool(self._fit_k_auto.isEnabled() and self._fit_k_auto.isChecked())
                   != bool(self._applied_fit_k_auto))
        self._fit_apply_btn.setEnabled(pending)
        self._fit_apply_btn.setText("Apply fit ●" if pending else "Apply fit")
        self._fit_apply_btn.setToolTip(
            f"Draw “{self._fit_combo.currentText()}” — the plot still shows "
            f"“{self._applied_fit}”." if pending else
            f"“{self._applied_fit}” is the fit on the plot.")

        s_pending = self._scatter_fit.currentText() != self._applied_scatter_fit
        self._scatter_fit_apply_btn.setEnabled(s_pending)
        self._scatter_fit_apply_btn.setText("Apply fit ●" if s_pending else "Apply fit")
        self._scatter_fit_apply_btn.setToolTip(
            f"Draw “{self._scatter_fit.currentText()}” — the plot still shows "
            f"“{self._applied_scatter_fit}”." if s_pending else
            f"“{self._applied_scatter_fit}” is the line on the plot.")
        self._scatter_fit_status.setText(
            f"Quick fit applied: {self._applied_scatter_fit}." +
            (f" Pending: {self._scatter_fit.currentText()} — Update plot to apply." if s_pending else ""))
        # The Plot button badges the SAME pending state, because it is the same gate.
        self._sync_run_state()

    def _pending_fit_change(self) -> bool:
        """Whether either fit combo proposes something the plot is not showing."""
        if not hasattr(self, "_scatter_fit"):
            return False
        return (self._fit_combo.currentText() != self._applied_fit
                or int(self._fit_k.value()) != int(self._applied_fit_k)
                or bool(self._fit_k_auto.isEnabled() and self._fit_k_auto.isChecked())
                != bool(self._applied_fit_k_auto)
                or self._scatter_fit.currentText() != self._applied_scatter_fit)

    def _on_fit_changed(self, name: str) -> None:
        on = name != _FIT_NONE
        self._fit_k_auto.setEnabled(on and name in _MULTI_COMPONENT_FITS)
        self._fit_k.setEnabled(on and not (self._fit_k_auto.isEnabled()
                                           and self._fit_k_auto.isChecked()))
        self._sync_filter_limited_toggle()
        # ...and the fit itself waits for Apply. The enable/disable above is about the
        # PROPOSAL — the user has to be able to set k before committing to it.
        self._fit_selection_moved(self._fit_combo)

    def _on_kde_overlay_toggled(self, _on: bool) -> None:
        """The KDE overlay box moved: a hand tick waits for Plot, code commits at once.

        The plot-type combo's rule (:meth:`_on_plot_type_changed`), because this is the
        same kind of choice -- WHAT is estimated, not how it looks. Nothing is drawn by a
        click, so the curve can never appear over bars it was not asked for -- and,
        unlike the plot type, not by the NEXT live redraw either: the draw reads
        ``_applied_kde_overlay``, which only :meth:`_armed_redraw` moves, as the fit's
        draw reads ``_applied_fit``. Without that, a bin-count spin after a hand tick
        drew the curve while the banner still said the plot was the previous selection.
        """
        if self._updating:
            return
        if self._from_the_user(self._hist_kde):
            self._mark_stale("the KDE overlay")
        else:
            self._armed_redraw()

    def _on_fit_k_changed(self, _value: int) -> None:
        self._fit_selection_moved(self._fit_k)

    def _on_fit_k_auto_toggled(self, on: bool) -> None:
        """Auto-BIC owns the spin while it is on, exactly as the clustering group's does —
        a number the user can still type but the fit ignores is a lie about who is in
        charge. The chosen k is written BACK into the spin after each draw, so the value
        on screen is always the one that was fitted."""
        self._fit_k.setEnabled(not (on and self._fit_k_auto.isEnabled()))
        self._fit_selection_moved(self._fit_k_auto)

    def _sync_filter_limited_toggle(self) -> None:
        """The include/exclude toggle only means something when a fit is on and some
        checked source actually carries the ``filter_limited`` flag."""
        if not hasattr(self, "_use_filter_limited"):
            return   # may fire during construction, before the histogram page exists
        has_flag = any("filter_limited" in s.df.columns for s in self.selected_sources())
        self._use_filter_limited.setEnabled(
            bool(has_flag and self._fit_combo.currentText() != _FIT_NONE))

    def _on_features_checked(self, _keys: list) -> None:
        self._on_features_changed()
        self._history_touch()

    def _on_features_changed(self) -> None:
        if self._updating:
            return
        # PCA axes are defined by the feature set; a change makes any run stale.
        self._pca_result = None
        if self._plot_type() == PCA:
            # THE FEATURE SET IS THE PCA'S QUANTITY (AnalysisIdentity.features): other
            # features are other scores, so the words and pins typed for the old ones go
            # now, as a column's do, and Run PCA then draws the new scores over automatic
            # text. On any other plot type the set is a clustering setting, not a quantity.
            self._rebase()
            # Through `_changed`, not `_redraw`, so an un-run window stays on its prompt
            # instead of being drawn into by a control it has not been told to plot yet.
            # The PCA view is gated by its own Run button either way.
            self._changed()

    def _on_method_changed(self, name: str) -> None:
        is_gmm = name == _METHOD_GMM
        self._k_mode.setEnabled(is_gmm)
        if not is_gmm:
            self._updating = True
            self._k_mode.setCurrentText(_K_FIXED)   # k-means needs an explicit k
            self._updating = False
        self._k_spin.setEnabled(self._k_mode.currentText() == _K_FIXED or not is_gmm)

    def _on_kmode_changed(self, mode: str) -> None:
        self._k_spin.setEnabled(mode == _K_FIXED)

    # -- current frame ---------------------------------------------------------------------------
    def _plot_type(self) -> str:
        return self._plot_type_combo.currentText()

    def _frame_of(self, src: PlotSource) -> pd.DataFrame | None:
        """One source's frame, filtered to accepted rows when that is asked for.

        The original index is preserved through the filter so a clustering's
        labels (indexed by the source ``df.index``) still align onto whatever
        subset is on screen.
        """
        df = src.df
        if df is None:
            return None
        if (self._accepted_check.isEnabled() and self._accepted_check.isChecked()
                and "accepted" in df.columns):
            df = df[df["accepted"].astype(bool)]
        return df

    def _current_frame(self) -> pd.DataFrame | None:
        """The PRIMARY source's frame (clustering / PCA operate on one source)."""
        src = self.primary_source()
        return None if src is None else self._frame_of(src)

    def _drawn_frames(self) -> list[tuple[PlotSource, pd.DataFrame]]:
        """``(source, filtered frame)`` for every checked, non-empty source."""
        out = []
        for src in self.selected_sources():
            df = self._frame_of(src)
            if df is not None and df.shape[0]:
                out.append((src, df))
        return out

    def _source_provenance(self) -> dict:
        """The PRIMARY source's detection provenance ({} when there is none)."""
        src = self.primary_source()
        return dict(src.provenance) if src is not None else {}

    def _accepted_only(self) -> bool:
        return bool(self._accepted_check.isEnabled() and self._accepted_check.isChecked())

    def _dwell_inputs(self, col: str, src: PlotSource | None = None):
        """The shared dwell-fit resolution for ONE source + column, or None.

        This is :func:`nanoscope.analysis.modules.dwell_fit_inputs` -- the very
        function ``modules.fit_dwell`` (and therefore the Fits panel) calls -- so
        the sample, the detection floor ``t_min`` and any modelled right
        truncation are identical in both windows by construction, not by
        coincidence. Returns None for a column the detector's dwell provenance
        does not describe (``rise_s``, ``ecd_pA_s``, ...): borrowing the dwell
        floor for those would be worse than not having one.

        ``src`` defaults to the PRIMARY source. It is a parameter precisely so an
        overlay resolves EACH source against ITS OWN provenance: two recordings
        detected at different ``min_dwell_s`` have different detection floors, and
        applying one's floor to the other's dwell times is the exact bias the
        provenance plumbing exists to remove.
        """
        if col not in _DWELL_COLUMNS:
            return None
        src = src if src is not None else self.primary_source()
        if src is None:
            return None
        df = src.df
        if df is None or col not in df.columns:
            return None
        frame = df if col == "dwell_s" else df.assign(dwell_s=df[col])
        try:
            return modules.dwell_fit_inputs(
                (frame, dict(src.provenance)),
                accepted_only=self._accepted_only(),
                use_filter_limited=self._use_filter_limited.isChecked(),
            )
        except (ValueError, KeyError, TypeError):
            return None

    def _depth_inputs(self, col: str, src: PlotSource | None = None):
        """The shared depth-fit resolution for ONE source (sample + that detector's
        own depth floor), or None. See :meth:`_dwell_inputs` on why the source is a
        parameter rather than an implicit 'the' source."""
        if col not in _DEPTH_COLUMNS:
            return None
        src = src if src is not None else self.primary_source()
        if src is None:
            return None
        df = src.df
        if df is None or col not in df.columns:
            return None
        try:
            return modules.depth_fit_inputs(
                (df, dict(src.provenance)), column=col,
                accepted_only=self._accepted_only(),
                use_filter_limited=self._use_filter_limited.isChecked(),
            )
        except (ValueError, KeyError, TypeError):
            return None
