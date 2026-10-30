"""The trace keys: catch an event, place an edge, add it, move on (docs/CURSOR_ERGONOMICS_DESIGN.md).

A manual pass over a recording is one gesture per event: point at it, ``C`` (catch: find the
event under the pointer and fit both edges to the steps), ``D`` (add). The keys sit under the
left hand, the pointer is the right hand's, and they are live whenever the TRACE has keyboard
focus:

====================  ==============================================================
``C``                 catch the event under the pointer: find it and fit both edges
``S`` / ``F``         put the start / finish edge at the pointer, fitted to the step
``Shift+S`` / ``F``   put that edge exactly at the pointer
``D``                 add the mask as a hand-made event
``Shift+D``           add, then go to the next window
``G`` / ``Shift+G``   next / previous window
``X``                 discard the mask
``E``                 draw mode: a plain left-drag draws a span instead of box-zooming
====================  ==============================================================

Return and Enter add like ``D``, Shift+Return like Shift+``D``, Esc cancels a draw in progress
and otherwise discards the mask. Those aliases act only when the trace got focus from a click
or the pointer arriving (``MouseFocusReason``): the window parks focus on the trace when a dock
closes, and a habitual Return or Esc after that must not add or discard.

This object owns the keys, the half-built mask's record (which edge came from a fit and which
was placed by hand), and the label; everything about the mechanics of a drag or the pointer is
`TraceView`'s, and everything about measuring is `AnalysisPanel`'s (`fit_span_edges`,
`catch_event_at`, `add_manual_event`). It talks to MainWindow only through the handful of
members it is handed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PySide6 import QtCore, QtGui

from ..analysis.manual_events import EdgeRecord, join_edge_fits

__all__ = ["TraceKeyController"]

#: How long a "press again to add it as placed" stays armed.
_CONFIRM_WINDOW_S = 10.0
#: A one-sided press with no mask gives the unknown edge this many pixels of room to be fitted in.
_PROVISIONAL_PX = 40


@dataclass
class _Edge:
    """One edge of the half-built mask."""

    pos: int | None = None
    #: ``unset`` | ``drawn`` (exact, by hand) | ``placed`` (a fit was tried and failed, or a
    #: catch's half-peak guess) | ``fit:step``.
    source: str = "unset"
    #: Where the edge stood before any fit moved it.
    drawn: int | None = None
    #: The record (``EdgeFit.to_json``) of the fit that put the edge here; None when no fit did.
    fit: str | None = None


class TraceKeyController(QtCore.QObject):
    """Wires the trace keys, draw mode and the pointer gestures to the add door."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self._w = window
        self._tv = window._trace_view
        self._start = _Edge()
        self._end = _Edge()
        #: The latest fit's record: the shared parameters of a ``fit_json`` merged from two.
        self._fit_json: str | None = None
        self._target: tuple[str, int] | None = None
        self._expect: tuple[int, int] | None = None
        self._confirm_until = 0.0
        #: The mask (target and bounds) the armed "press again" was asked about.
        self._confirm_for: tuple | None = None
        #: Measure as the trace last reported it, and the bounds of a seed it showed while off.
        self._measure_on = False
        self._seed: tuple[int, int] | None = None
        self.shortcuts: list[QtGui.QShortcut] = []
        self._install()
        tv = self._tv
        tv.sigDrawReleased.connect(self._on_draw_released)
        tv.sigCatchRequested.connect(self._on_catch_requested)
        tv.sigDrawRefused.connect(self._say)
        tv.sigMeasureCleared.connect(self._reset)
        tv.sigMeasureFinished.connect(self._on_mask_settled)
        tv.sigMeasureActiveChanged.connect(self._on_measure_active_changed)
        tv.sigDrawModeChanged.connect(lambda _on: self.refresh_label())
        self._label_timer = QtCore.QTimer(self)
        self._label_timer.setSingleShot(True)
        self._label_timer.setInterval(120)
        self._label_timer.timeout.connect(self._paint_label)
        tv.sigVisibleRangeChanged.connect(lambda *_: self.refresh_label())
        window._session.resultsChanged.connect(lambda *_: self.refresh_label())

    # -- installation ------------------------------------------------------------------------------
    def _install(self) -> None:
        def add(sequence: str, slot, *, repeat: bool = False, alias: bool = False) -> None:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self._tv)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(repeat)
            shortcut.activated.connect(
                (lambda s=slot: s()) if not alias else (lambda s=slot: self._alias(s)))
            self.shortcuts.append(shortcut)

        add("C", self.catch)
        add("S", lambda: self.place_edge("start", fit=True))
        add("F", lambda: self.place_edge("end", fit=True))
        add("Shift+S", lambda: self.place_edge("start", fit=False))
        add("Shift+F", lambda: self.place_edge("end", fit=False))
        add("D", self.add)
        add("Shift+D", self.add_and_next)
        add("G", lambda: self.step_window(+1), repeat=True)
        add("Shift+G", lambda: self.step_window(-1), repeat=True)
        add("X", self.discard)
        add("E", self.toggle_draw_mode)
        # Return and the keypad's Enter are different keys to Qt; each needs its own sequence.
        for key in ("Return", "Enter"):
            add(key, self.add, alias=True)
            add(f"Shift+{key}", self.add_and_next, alias=True)
        add("Esc", self.escape, alias=True)

    def _alias(self, slot) -> None:
        """Return / Enter / Esc act only when the user pointed at the trace (see the module
        docstring); after the window parked focus here they do nothing until a click on the
        trace wakes them (`TraceView.mousePressEvent`)."""
        if self._tv.last_focus_reason() == QtCore.Qt.FocusReason.MouseFocusReason:
            slot()

    # -- small helpers -----------------------------------------------------------------------------
    def _say(self, text: str, ms: int = 8000) -> None:
        self._w.statusBar().showMessage(str(text), ms)

    def _panel(self):
        return self._w._analysis_panel

    def _current(self) -> tuple[str, int] | None:
        return self._panel().current_target()

    def _reset(self) -> None:
        self._start, self._end = _Edge(), _Edge()
        self._fit_json = None
        self._target = None
        self._expect = None
        self._confirm_until = 0.0
        self._confirm_for = None
        self._seed = None
        self.refresh_label()

    def _on_measure_active_changed(self, on: bool) -> None:
        self._measure_on = bool(on)
        if not on:
            self._reset()

    def _ensure_measure(self) -> tuple[str, int] | None:
        """Turn Measure on (without seeding a mask) and return the target, or say why not."""
        target = self._current()
        if target is None:
            self._say("Select a recording first.")
            return None
        measure = self._panel().measure_panel
        if not measure.is_active():
            if not measure.can_measure():
                self._say("The selected current channel is not shown. Tick that channel in "
                          "the File Manager first.")
                return None
            measure.set_active(True, seed=False)
        if self._tv.measure_target() != target:
            self._say("Measure is on another channel. Turn it off and on over this one.")
            return None
        return target

    def _x(self, target: tuple[str, int], index: int) -> float | None:
        """The data-x of a mask edge at sample ``index``. ``x_for_index`` clamps to the last
        sample, so an end at ``n`` (an event the recording's end cuts off) would sit on ``n - 1``
        and be filed one sample short; it goes one sample past the last x instead, where
        ``MeasurePanel._x_span`` puts it and ``measure_bounds`` reads ``n`` back."""
        doc_id, channel = target
        x = self._tv.x_for_index(doc_id, channel, int(index))
        if x is not None and int(index) >= self._tv.x_axis_for(doc_id, channel).values.size:
            x += self._tv.sample_dx(doc_id, channel)
        return x

    def _show_mask(self, target: tuple[str, int], start: int, end: int) -> None:
        x0, x1 = self._x(target, start), self._x(target, end)
        if x0 is None or x1 is None:
            return
        # The region's signals fire synchronously inside this call; the expectation covers
        # exactly that, and must not outlive it (the call emits nothing at all when the mask is
        # already there, and a stale expectation would swallow the user's next real drag).
        self._expect = (int(start), int(end))
        try:
            self._tv.set_measure_region(x0, x1)
        finally:
            self._expect = None

    def _visible_samples(self, target: tuple[str, int], n_hint: int = 0) -> tuple[int, int]:
        rng = self._panel().visible_sample_range()
        if rng is not None:
            return int(rng[0]), int(rng[1])
        return 0, int(n_hint)

    # -- the mask's record ----------------------------------------------------------------------
    def _on_mask_settled(self, lo: int, hi: int) -> None:
        """The mask came to rest. If it is where this controller put it, nothing changes; an edge
        that moved (a handle or the body dragged, a double-click, a snap) was placed by hand and
        loses its fit record, and the other edge keeps its own. A mask this controller has no
        record of is what the person sees, so both its edges are recorded as drawn and ``S`` or
        ``F`` then moves only one of them. Measure's own seed (the middle fifth, which settles
        before Measure reports itself on) is nobody's aim: until something moves it, it is not
        recorded, so ``S`` then ``F`` still builds a mask over it, and a rebuild putting it
        back changes nothing. Snap to view and a double-click turn Measure on without a seed, so
        their mask is recorded even on the seed's own middle fifth; only Snap to view over a seed
        nothing has moved lands on it exactly and leaves it the seed (the trace says nothing of a
        mask placed where one already is)."""
        if self._expect is not None:
            elo, ehi = self._expect
            self._expect = None
            if abs(lo - elo) <= 1 and abs(hi - ehi) <= 1:
                return
        if self._start.pos is None and self._end.pos is None:
            seed = self._seed
            if not self._measure_on or (
                    seed is not None and abs(lo - seed[0]) <= 1 and abs(hi - seed[1]) <= 1):
                self._seed = (int(lo), int(hi))
            else:
                # Quietly: two drawn edges put nothing on the label, so there is no repaint
                # (and no second measurement of the span Measure has just measured).
                self._target = self._tv.measure_target()
                self._start = _Edge(int(lo), "drawn", int(lo))
                self._end = _Edge(int(hi), "drawn", int(hi))
            return
        before = (self._start.source, self._end.source)
        stand_in = hi - lo <= 1
        if self._moved(self._start, lo, stand_in):
            self._start = _Edge(int(lo), "drawn", int(lo))
        if self._moved(self._end, hi, stand_in):
            self._end = _Edge(int(hi), "drawn", int(hi))
        # The label says how the edges were made, not where they are, so only a change of how
        # repaints it (a repaint measures the span again, and Measure has just measured it).
        if (self._start.source, self._end.source) != before:
            self.refresh_label()

    @staticmethod
    def _moved(edge: _Edge, at: int, stand_in: bool) -> bool:
        """Whether a settle with this edge of the mask at ``at`` put the edge somewhere new. A
        drawn edge is wherever the mask's edge is; a fitted or placed one keeps its record
        through a one-sample difference. The edge still to come after S or F alone is put
        somewhere only by dragging the one-sample stand-in wider: a rebuild putting the
        stand-in back puts it nowhere."""
        if edge.pos is None:
            return not stand_in
        if edge.source == "drawn":
            return int(at) != edge.pos
        return abs(int(at) - edge.pos) > 1

    def _bounds(self) -> tuple[int, int] | None:
        b = self._tv.measure_bounds()
        return None if b is None else (int(b[0]), int(b[1]))

    # -- the keys ---------------------------------------------------------------------------------
    def catch(self, *, at: tuple[str, int, float, float] | None = None) -> None:
        """``C``: find the event under the pointer, fit both edges, show the mask."""
        target = self._ensure_measure()
        if target is None:
            return
        ps = at if at is not None else self._tv.pointer_sample()
        if ps is None or (ps[0], ps[1]) != target:
            self._say("Point at the recording trace, then press C.")
            return
        _doc, _ch, sample, px = ps
        n = self._sample_count(target)
        view_lo, view_hi = self._visible_samples(target, n)
        catch, why = self._panel().catch_event_at(
            target[0], target[1], sample, px_samples=px, view_lo=view_lo, view_hi=view_hi)
        if catch is None:
            self._say(why)
            return
        if not catch.ok:
            self._say(catch.reason)
            return
        fit = catch.fit
        s_fit = fit is not None and fit.start.fitted
        e_fit = fit is not None and fit.end.fitted
        start = int(fit.start.position) if s_fit else int(catch.start)
        end = int(fit.end.position) if e_fit else int(catch.end)
        record = fit.to_json() if fit is not None else None
        self._start = _Edge(start, "fit:step" if s_fit else "placed", int(catch.start), record)
        self._end = _Edge(end, "fit:step" if e_fit else "placed", int(catch.end), record)
        self._fit_json = record
        self._target = target
        self._show_mask(target, start, end)
        if s_fit and e_fit:
            self._say(f"Caught: {end - start:,} samples. D adds it.")
        else:
            which = "start" if not s_fit else "finish"
            self._say(f"Caught, but the {which} edge is not fitted"
                      f"{' (' + self._reason(fit, which) + ')' if fit is not None else ''}. "
                      f"Point at it and press {'S' if which == 'start' else 'F'}, "
                      f"Shift+{'S' if which == 'start' else 'F'} to set it exactly, or D twice to add it "
                      "as placed.")
        self.refresh_label()

    @staticmethod
    def _reason(fit, which: str) -> str:
        edge = fit.start if which == "start" else fit.end
        return edge.reason

    def _sample_count(self, target: tuple[str, int]) -> int:
        try:
            return int(self._w._session.calibrated(target[0], target[1]).data.size)
        except Exception:  # noqa: BLE001
            return 0

    def place_edge(self, which: str, *, fit: bool) -> None:
        """``S`` / ``F`` (fitted) and Shift+``S`` / Shift+``F`` (exact): put one edge at the
        pointer. With no mask yet, ``S`` then ``F`` builds one."""
        target = self._ensure_measure()
        if target is None:
            return
        ps = self._tv.pointer_sample()
        if ps is None or (ps[0], ps[1]) != target:
            self._say("Point at the recording trace, then press the key.")
            return
        _doc, _ch, sample, px = ps
        p = int(round(sample))
        n = self._sample_count(target)
        if self._target != target:
            self._start, self._end = _Edge(), _Edge()
        # The other edge is the mask's own (a mask placed some other way was recorded when it
        # settled; only Measure's seed is not an instruction); with none, give the fit room.
        other_edge = self._end if which == "start" else self._start
        provisional = other_edge.pos is None
        if provisional:
            other = p + (_PROVISIONAL_PX * max(px, 1.0) * (1 if which == "start" else -1))
            other = int(max(0, min(n, round(other))))
        else:
            other = int(other_edge.pos)
        lo_s, hi_s = (p, other) if which == "start" else (other, p)
        if hi_s <= lo_s:
            self._say("The start edge has to stay before the finish edge: point earlier.")
            return
        if fit:
            fitted, why = self._panel().fit_span_edges(
                target[0], target[1], lo_s, hi_s, px_samples=px, edges=(which,))
            if fitted is None:
                self._say(why)
                return
            edge_result = fitted.start if which == "start" else fitted.end
            position = int(edge_result.position)
            source = "fit:step" if edge_result.fitted else "placed"
            reason = edge_result.reason
            fit_json = self._fit_json = fitted.to_json()
        else:
            position, source, reason, fit_json = p, "drawn", "", None
        new = _Edge(position, source, p, fit_json)
        if which == "start":
            self._start = new
            if provisional:
                self._end = _Edge(None, "unset", None)
        else:
            self._end = new
            if provisional:
                self._start = _Edge(None, "unset", None)
        self._target = target
        a = self._start.pos if self._start.pos is not None else (
            max(0, position - 1) if which == "end" else position)
        b = self._end.pos if self._end.pos is not None else (
            min(n, position + 1) if which == "start" else position)
        if b <= a:
            b = a + 1
        self._show_mask(target, a, b)
        if source == "placed":
            self._say(f"The {which} edge could not be fitted ({reason}); it stays where you "
                      "pointed. Shift+" + ("S" if which == "start" else "F")
                      + " sets it exactly.")
        elif provisional:
            self._say("Start edge set. Point at the finish and press F." if which == "start"
                      else "Finish edge set. Point at the start and press S.")
        else:
            self._say(f"The {which} edge is {'fitted' if source == 'fit:step' else 'set'}. "
                      "D adds the event.")
        self.refresh_label()

    def add(self, *, then_next: bool = False) -> None:
        """``D``: add the mask as a hand-made event; the mask is discarded and the view stays.
        Return, Enter, Ctrl+M and the Add button add through here too, so each files the same
        row for the same mask."""
        target = self._current()
        if target is None:
            self._say("Select a recording first.")
            return
        bounds = self._bounds()
        if bounds is None:
            self._say("No mask: press C over an event, or drag in draw mode (E).")
            return
        # The table's refusal first, as the Measure tab's preview gives it: no edge to set and no
        # second press is asked for a row the table on screen cannot take.
        if self._panel().refuse_add_for_the_table(target[0], target[1]):
            return                       # the door's own sentence; the mask is kept
        hold = self._hold()
        if hold:
            self._say(f"Nothing was added: {hold}")
            return
        record = self._edge_record(bounds)
        unfitted = [name for name, edge in (("start", self._start), ("finish", self._end))
                    if edge.source == "placed"]
        # The second press confirms the mask it was asked about, not whatever is on screen next.
        asked = (target, bounds)
        if unfitted and (time.monotonic() > self._confirm_until or self._confirm_for != asked):
            self._confirm_until = time.monotonic() + _CONFIRM_WINDOW_S
            self._confirm_for = asked
            edges = (f"{unfitted[0]} edge is" if len(unfitted) == 1
                     else "start and finish edges are")
            self._say(f"The {edges} not fitted. Press again to add it as placed, or fit it first "
                      "(S / F) or set it exactly (Shift+S / Shift+F).")
            return
        outcome = self._w._file_hand_made_event(target[0], target[1], bounds[0], bounds[1],
                                                edge=record)
        if not outcome.ok:
            return                       # a refusal keeps the mask and says why
        self._tv.discard_measure_region()
        self._reset()
        if then_next and not self._w._pan_scrollbar.step_window(+1):
            # The add's own sentence stays (where the event went, how to undo it).
            self._say(f"{outcome.message} There is no next window: that is the end of the "
                      "recording.", 10000)

    def add_and_next(self) -> None:
        self.add(then_next=True)

    def _hold(self) -> str:
        """Why the mask on screen cannot be added yet, or "". S or F alone shows a one-sample
        stand-in for the other edge, which nobody has put anywhere. The add says it on a press
        and the Measure tab's preview says it before one."""
        unset = [(name, key) for name, key, edge in (("start", "S", self._start),
                                                     ("finish", "F", self._end))
                 if edge.pos is None]
        if len(unset) != 1:
            return ""
        name, key = unset[0]
        return (f"the {name} edge is not set yet. Point at it and press {key}, or Shift+{key} "
                "to set it exactly.")

    def _edge_record(self, bounds: tuple[int, int]) -> EdgeRecord:
        """How each edge of the mask being added was made. An edge a person moved after the fit
        is ``drawn``; an unset or unknown one is ``drawn`` too, since the mask is what they see,
        and a drawn edge was drawn where it stands (a plain bracket's row is the one the door
        stores for it on its own)."""
        s, e = self._start, self._end
        s_src = s.source if s.pos is not None and abs(s.pos - bounds[0]) <= 1 else "drawn"
        e_src = e.source if e.pos is not None and abs(e.pos - bounds[1]) <= 1 else "drawn"
        return EdgeRecord(
            bounds_source=f"start={s_src};end={e_src}",
            drawn_start=int(bounds[0] if s_src == "drawn" or s.drawn is None else s.drawn),
            drawn_end=int(bounds[1] if e_src == "drawn" or e.drawn is None else e.drawn),
            fit_json=self._fit_record(bounds, s_src, e_src))

    def _fit_record(self, bounds: tuple[int, int], s_src: str, e_src: str) -> str | None:
        """The row's ``fit_json`` (None when no edge is fitted): each edge's part of the fit that
        put it there (cursor design 5.5). One fit that set both edges (``C``, a drawn span) is
        stored as it came. Edges set by two fits (``S`` then ``F``), or one moved by hand since
        its fit, are merged into the latest fit's record by `join_edge_fits`, the join a merge
        uses: each edge's entry, window, rise and placed position come from its own fit, and an
        edge no fit set reads "not requested", as in a fit asked for the other edge only. So the
        record never calls an edge unfitted that ``bounds_source`` calls ``fit:step``."""
        if "fit:step" not in (s_src, e_src):
            return None
        own = {name: edge.fit for name, edge, src in (("start", self._start, s_src),
                                                      ("end", self._end, e_src))
               if src != "drawn" and edge.fit}
        return join_edge_fits(own, bounds, shared=self._fit_json)

    def step_window(self, direction: int) -> None:
        """``G`` / Shift+``G``: the next / previous window at the same zoom. With no mask nothing
        is carried; with one, the existing scan step carries it as it always did."""
        if not self._w._pan_scrollbar.step_window(int(direction)):
            self._say("That is the end of the recording." if direction > 0
                      else "That is the start of the recording.", 3000)

    def discard(self) -> None:
        """``X``: throw the mask away; Measure stays on."""
        if self._tv.discard_measure_region():
            self._reset()

    def escape(self) -> None:
        """Esc cancels a draw in progress, else discards the mask. It never leaves draw mode: a
        habitual second Esc must not change what the next drag does."""
        if self._tv.cancel_draw():
            return
        self.discard()

    def toggle_draw_mode(self) -> None:
        """``E``: draw mode on or off. On needs Measure; it turns Measure on without a mask."""
        self.set_draw_mode(not self._tv.draw_mode())

    def set_draw_mode(self, on: bool) -> None:
        """Draw mode on or off (the key's and the menu item's one door)."""
        on = bool(on)
        if on and not self._tv.draw_mode() and self._ensure_measure() is None:
            return
        self._tv.set_draw_mode(on)
        self.refresh_label()

    # -- the pointer gestures ---------------------------------------------------------------------
    def _on_draw_released(self, doc_id: str, channel: int, lo: int, hi: int, px: float) -> None:
        """A drag in draw mode ended as a real span: fit both edges to the steps it was drawn
        near and show the result. The settle is ours (we move the mask), so Measure recomputes
        once, on the fitted bounds."""
        target = (doc_id, int(channel))
        if self._current() != target:
            return
        fit, why = self._panel().fit_span_edges(doc_id, channel, lo, hi, px_samples=px)
        if fit is None:
            self._start = _Edge(int(lo), "drawn", int(lo))
            self._end = _Edge(int(hi), "drawn", int(hi))
            self._fit_json = None
            self._target = target
            self._tv.settle_measure_region()
            self._say(why)
            self.refresh_label()
            return
        s_fit, e_fit = fit.start.fitted, fit.end.fitted
        start = int(fit.start.position) if s_fit else int(lo)
        end = int(fit.end.position) if e_fit else int(hi)
        record = self._fit_json = fit.to_json()
        self._start = _Edge(start, "fit:step" if s_fit else "placed", int(lo), record)
        self._end = _Edge(end, "fit:step" if e_fit else "placed", int(hi), record)
        self._target = target
        if (start, end) != (int(lo), int(hi)):
            self._show_mask(target, start, end)
        else:
            self._tv.settle_measure_region()
        if s_fit and e_fit:
            self._say(f"Fitted: {end - start:,} samples. D adds it.")
        else:
            which = "start" if not s_fit else "finish"
            self._say(f"The {which} edge is not fitted ({self._reason(fit, which)}). Point at it "
                      f"and press {'S' if which == 'start' else 'F'}, or press D twice to add it "
                      "as placed.")
        self.refresh_label()

    def _on_catch_requested(self, doc_id: str, channel: int, sample: int, px: float) -> None:
        """A double-click in draw mode catches the event there, as ``C`` does at that x."""
        self.catch(at=(doc_id, int(channel), float(sample), float(px)))

    # -- the label on the trace ---------------------------------------------------------------------
    def refresh_label(self) -> None:
        """Re-text the label and the Add button's destination, soon (a burst becomes one)."""
        self._label_timer.start()

    def _paint_label(self) -> None:
        panel = self._panel()
        target = self._current()
        if target is None or not self._w._session.has_document(target[0]):
            self._tv.set_target_label("")
            panel.measure_panel.set_add_target_text("")
            panel.measure_panel.set_add_hold("")
            return
        name = panel.target_table_name(*target)
        # The Measure tab's preview names the same table and counts the same rows: whatever
        # moved the label (a new table, a chosen table, a hand-made row, an edge set or unset)
        # moved the preview.
        panel.measure_panel.set_add_hold(self._hold())
        panel.measure_panel.refresh_preview()
        text = f"→ {name}"
        res = panel._session.results.get(target[0], target[1], panel.target_module_id(*target))
        if res is not None and hasattr(res.payload, "df"):
            from ..analysis.scoring import hand_made_mask

            df = res.payload.df
            hand = hand_made_mask(df)
            n_hand = int(hand.sum())
            text += f" · {n_hand} hand-made"
            rng = panel.visible_sample_range()
            if rng is not None and n_hand:
                s = df["start_idx"].to_numpy()
                e = df["end_idx"].to_numpy()
                n_view = int((hand & (s < rng[1]) & (e > rng[0])).sum())
                text += f" · {n_view} in view"
        if self._tv.draw_mode():
            text += " · draw"
        for label, edge in (("start", self._start), ("finish", self._end)):
            if edge.source == "placed":
                text += f" · {label} not fitted"
        self._tv.set_target_label(text)
        panel.measure_panel.set_add_target_text(f"→ {name}")
