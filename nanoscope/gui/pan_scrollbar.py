"""The zoomed-in pan bar: slide the trace left/right at the current zoom to centre an event.

WHY IT EXISTS
-------------
Diving into a blockade is a two-gesture job: frame it (box-zoom / +) and then *centre* it.
The framing gesture almost never lands the event in the middle — you box a little left, or the
zoom is centred a little away from the event — so at 200 µs of visible trace it sits
half off the pane and there is no cheap way to nudge the view sideways. The Navigator can pan,
but at ~36 ms/pixel one pixel of drag is 180 screens at that zoom: it is a *where am I* map,
not a fine control. This bar is the fine control, and it is deliberately the plainest one
possible — a QScrollBar whose trough IS the recording and whose handle IS the visible window.

THE SECOND CONTROLLER PROBLEM
-----------------------------
The x-range now has two widgets bound to it (this bar and the Navigator's region) plus the plot
itself. All three must agree at all times and none may drive another into a feedback loop, so
this follows the SAME discipline the Navigator uses: a single ``_syncing`` flag, set around every
programmatic write, and checked at the top of every inbound slot. The view is the single source
of truth; the bar only ever *proposes* a new x-range through ``TraceView.set_x_range`` and then
learns the result the same way the Navigator does — through ``sigVisibleRangeChanged``.

INTEGER BAR, FLOAT AXIS — WHY THE WIDTH CANNOT DRIFT
----------------------------------------------------
A QScrollBar is integer; the time axis is float64 seconds. The mapping is a FIXED resolution of
``PAN_STEPS`` steps across the full extent (never a per-view rescaling), so a step is a constant
slice of the recording: 0.3 µs on a 300 s file — finer than one sample at 50 kHz, i.e. below the
resolution of the data itself.

The anti-drift rule is the important half: a pan reads the *start* from the bar and takes the
*width* from ``self._width``, the float the view last reported. It never reconstructs the width
from ``pageStep``. So a pan is a pure translation — ``x1 = x0 + w`` with the same ``w`` every
time — and twenty successive pans cannot accumulate a single quantisation error into the zoom
level. ``_width`` is refreshed ONLY from a genuine external range change (a zoom), never from
the bar's own round trip.

LAYOUT: THE HEIGHT IS RESERVED, THE BAR IS HIDDEN
-------------------------------------------------
The bar must "appear only once zoomed", but zooming across the full-extent threshold would
then add/remove a scrollbar's height on every notch: the plot resizes, pyqtgraph re-runs its
auto-downsampling for the new pixel width, and the trace visibly twitches mid-gesture. So the
CONTAINER keeps a fixed height forever and only the QScrollBar inside it is shown/hidden. The user
gets "it appears when I zoom"; the trace geometry never moves — measured, the trace pane loses
exactly ZERO pixels when the bar appears.

The price is paid once, up front: the reserved strip is a permanent ~18 px (the style's scrollbar
height, floored at 14), so the WINDOW's minimum height rises by that plus the layout spacing —
about 20 px, measured 436 -> 456. Minimum WIDTH is untouched (the bar's minimumSizeHint is 0x0 and
its policy is Expanding, so it never pushes the window wider). Twenty pixels of minimum height is
a cheap price for a viewer whose whole job is looking at the trace.

Imports nothing from ``nanoscope.gui`` — MainWindow wires it, like every other panel.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core.session import Session

__all__ = [
    "PanScrollBar", "PAN_STEPS", "FULL_EXTENT_TOL", "SCAN_STEP_FRACTION",
    "LOCAL_ZOOM_FACTOR",
]

# Fixed resolution of the integer<->float mapping: the number of scrollbar steps spanning the
# WHOLE recording, constant at every zoom level. 1e9 keeps a step below one sample for any
# realistic file (300 s / 1e9 = 0.3 µs vs a 20 µs sample period at 50 kHz) while staying an
# order of magnitude clear of the int32 ceiling QScrollBar is bounded by.
PAN_STEPS = 1_000_000_000

# "The view covers (approximately) the whole recording" — the not-zoomed test, relative to the
# extent so it is scale-free. 0.1% of the extent is far below one pixel of a 1200 px plot, so
# the bar cannot flicker on a rounding-sized difference, yet any real zoom trips it instantly.
FULL_EXTENT_TOL = 1e-3

# Previous/Next view leave a visible seam between adjacent scan windows.  With a selection in the
# middle of the old view, a 90% step also leaves that event inside the new window before the cursor
# itself is carried forward, which makes the direction of travel visually obvious.
SCAN_STEP_FRACTION = 0.90

# One local zoom button press changes scale by one octave. It is deliberately incremental: unlike
# Reset, Zoom out does not throw an event-scale annotator back to the entire recording.
LOCAL_ZOOM_FACTOR = 2.0


class PanScrollBar(QtWidgets.QWidget):
    """Horizontal pan control for the trace's x-axis. Visible only while zoomed in.

    Duck-typed against the view exactly as the Navigator is (``full_x_extent()``,
    ``anchor_plot()``, ``set_x_range()``, ``sigVisibleRangeChanged``) so this module never
    imports a sibling gui panel.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._session: Session | None = None
        self._view = None
        self._target: tuple[str, int] | None = None
        self._syncing = False              # the Navigator's discipline; see the module docstring
        self._extent: tuple[float, float] | None = None
        self._x0 = 0.0
        self._width = 0.0                  # THE authoritative zoom width; never rebuilt from pageStep
        self._active = False

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        self._scan_label = QtWidgets.QLabel("Scan")
        self._scan_label.setStyleSheet("color:#8b9199; font-size:10px;")
        self._scan_label.setToolTip(
            "Move through a recording at the current event-scale zoom. Previous/Next leave "
            "10% overlap and carry active Measure cursors into the new window.")
        self._scan_label.setFixedWidth(self._scan_label.sizeHint().width())
        lay.addWidget(self._scan_label)

        def tool_button(text: str, name: str, tip: str, slot) -> QtWidgets.QToolButton:
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setObjectName(name)
            button.setAutoRaise(True)
            button.setToolTip(tip)
            button.setFixedWidth(max(22, button.sizeHint().width()))
            # `clicked` carries its `checked` flag, and PySide hands it to a slot that has room
            # for it. `zoom_in` / `zoom_out` now take an optional anchor, so a bare connect would
            # read "checked=False" as "zoom about x = 0". The Scan strip's buttons keep the centre.
            button.clicked.connect(lambda _checked=False, _slot=slot: _slot())
            lay.addWidget(button)
            return button

        self._previous_button = tool_button(
            "◀ Prev", "previousScanWindow",
            "Previous view (Shift+Left): move back by 90% of this view, preserving zoom and "
            "leaving 10% overlap.", self.previous_view)
        self._next_button = tool_button(
            "Next ▶", "nextScanWindow",
            "Next view (Shift+Right): move forward by 90% of this view, preserving zoom and "
            "leaving 10% overlap.", self.next_view)

        self._bar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self._bar.setRange(0, 0)
        self._bar.setPageStep(PAN_STEPS)
        self._bar.setToolTip(
            "Slide the view along the recording at the current zoom — drag the handle to bring an "
            "off-centre event into the middle. Click the trough to page by one screenful. "
            "Appears only while zoomed in.")
        lay.addWidget(self._bar)
        self._bar.hide()

        self._zoom_out_button = tool_button(
            "−", "localZoomOut",
            "Zoom out one step (Shift+Down) around the centre; keeps local context instead of "
            "resetting to the full recording.", self.zoom_out)
        self._zoom_in_button = tool_button(
            "+", "localZoomIn",
            "Zoom in one step (Shift+Up) around the centre.", self.zoom_in)
        self._full_trace_button = tool_button(
            "Full", "fullTrace",
            "Show the complete trace. Ctrl+0 does the same anywhere; Home does it while "
            "the trace has focus.", self.show_full_trace)

        # Reserve the height for good (see the module docstring): show/hide must never reflow
        # the trace. sizeHint can be 0 under some offscreen styles, hence the floor.
        self.setFixedHeight(max(self._bar.sizeHint().height(),
                                self._previous_button.sizeHint().height(), 18))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._bar.valueChanged.connect(self._bar_to_view)

    # -- wiring ----------------------------------------------------------------------------------
    def bind(self, session: Session, view) -> None:
        """Wire to Session (for the signals that can change the extent) and to the TraceView (for
        the two-way visible-range binding). ``view`` is duck-typed — never imported."""
        self._session = session
        self._view = view
        session.layersChanged.connect(self.refresh)
        session.viewModeChanged.connect(self.refresh)
        session.documentClosed.connect(lambda _doc_id: self.refresh())
        session.calibrationChanged.connect(lambda _d, _c: self.refresh())
        view.sigVisibleRangeChanged.connect(self._view_to_bar)
        if hasattr(view, "sigMeasureActiveChanged"):
            # measure_target takes precedence over the selected fallback in _movement_extent;
            # activation/deactivation can therefore change the correct boundary without moving
            # the x view at all, so visible-range signals alone are insufficient.
            view.sigMeasureActiveChanged.connect(lambda *_a: self.refresh())
        self.refresh()

    # -- public API ------------------------------------------------------------------------------
    def is_active(self) -> bool:
        """True when the bar is showing, i.e. the view is zoomed in on a real extent."""
        return self._active

    def scrollbar(self) -> QtWidgets.QScrollBar:
        """The underlying QScrollBar (for tests and for anyone who needs its live geometry)."""
        return self._bar

    def set_target(self, doc_id: str, channel: int) -> None:
        """Use this selected channel's plotted extent for scan boundaries.

        A multi-recording overlay can contain traces with different starts/ends. Manual scanning
        must stop at the boundary of the recording being labelled, not at a neighbor's union.
        """
        target = (str(doc_id), int(channel))
        if target == self._target:
            return
        self._target = target
        self.refresh()

    def previous_view(self) -> bool:
        """Move one overlapping annotation window left, preserving zoom exactly."""
        return self._scan(-1)

    def next_view(self) -> bool:
        """Move one overlapping annotation window right, preserving zoom exactly."""
        return self._scan(1)

    def step_window(self, direction: int) -> bool:
        """Step one window along the recording: ``direction`` >= 0 is forward, < 0 backward.

        The same 90% step with 10% overlap, at the same zoom, clamped to the annotation target,
        as :meth:`next_view` / :meth:`previous_view` (which are this with the direction fixed).
        A live Measure mask travels with the window and is moved by the same clamped delta;
        with NO mask (Measure off, or on and empty) nothing is carried -- there is nothing to
        move and nothing is created. False when the window cannot move that way."""
        return self._scan(1 if direction >= 0 else -1)

    def zoom_out(self, anchor: float | None = None) -> bool:
        """Double the visible width, clamped to this recording. Around the centre, or around the
        data x ``anchor`` when one is given (and lies inside the view): that x stays where it is on
        screen, as long as the recording's ends do not force the window to move."""
        return self._zoom(LOCAL_ZOOM_FACTOR, anchor)

    def zoom_in(self, anchor: float | None = None) -> bool:
        """Halve the visible width, never below two sample intervals. Around the centre, or
        around the data x ``anchor`` (see :meth:`zoom_out`)."""
        return self._zoom(1.0 / LOCAL_ZOOM_FACTOR, anchor)

    def show_full_trace(self) -> bool:
        """Restore the complete trace through the same path as View ▸ Reset zoom."""
        if self._view is None or self._view.anchor_plot() is None:
            return False
        extent = self._view.full_x_extent()
        if extent is None or extent[1] <= extent[0]:
            return False
        self._view.reset_zoom()
        return True

    def refresh(self, *_args) -> None:
        """Re-derive extent + visible range from the view and restate the bar. Cheap enough to call
        from any session signal: ``full_x_extent`` is a few cached-array lookups."""
        rng = self._current_range()
        self._extent = self._movement_extent()
        if rng is None:
            self._set_active(False)
            self._sync_buttons(0.0, 0.0)
            return
        self._apply(rng[0], rng[1])

    def centre_on(self, x: float) -> bool:
        """Pan — never zoom — so data-x ``x`` sits in the middle of the view.

        This is the user's actual goal ("centre the event") reduced to its one line: the width is
        the width you already chose, only the start moves, clamped to the extent. Returns False
        when there is nothing to pan (no document, degenerate view).

        The extent check is not a formality: with no extent there is nothing to clamp AGAINST, so
        moving the view would fling it to wherever ``x`` happens to be — off the data, showing
        nothing, while still reporting success. A degenerate recording (one sample, no x-span)
        therefore leaves the view exactly where it was."""
        rng = self._current_range()
        if self._view is None or rng is None:
            return False
        lo, hi = rng
        width = hi - lo
        if width <= 0.0:
            return False
        self._extent = self._movement_extent()
        if self._extent is None:
            self._apply(lo, hi)               # restate (this hides the bar); do NOT move the view
            return False
        e0, e1 = self._extent
        span = e1 - e0
        eps = max(width, span, 1.0) * 1e-14
        if span <= 0.0 or width >= span - eps:
            # A width-preserving window cannot be centred on an arbitrary point AND remain inside
            # a shorter target. Refuse instead of returning True for a view whose centre is wrong.
            self._apply(lo, hi)
            return False
        self._width = width
        x0 = float(x) - width / 2.0
        x0 = self._clamp_start(x0, width)
        self._set_view_range(x0, width)
        self._apply(x0, x0 + width)
        return True

    def _movement_extent(self) -> tuple[float, float] | None:
        """Extent of the selected/Measure target, falling back to the visible-layer union."""
        if self._view is None:
            return None
        target = self._movement_target()
        if target is not None and hasattr(self._view, "x_axis_for"):
            try:
                vals = self._view.x_axis_for(*target).values
            except (KeyError, ValueError, IndexError, AttributeError):
                vals = ()
            if len(vals) >= 2:
                lo, hi = float(vals[0]), float(vals[-1])
                if hi > lo:
                    return lo, hi
        return self._view.full_x_extent()

    def _movement_target(self) -> tuple[str, int] | None:
        """Live/drawn target whose extent and sample floor this control may enforce."""
        if self._view is None:
            return None
        target = self._view.measure_target() if hasattr(self._view, "measure_target") else None
        if target is None:
            target = self._target
        if target is not None and hasattr(self._view, "visible_x_range_for"):
            try:
                drawn = self._view.visible_x_range_for(*target) is not None
            except (KeyError, ValueError, IndexError, AttributeError):
                drawn = False
            if not drawn:
                # MainWindow deliberately permits an unticked channel to remain selected. Its
                # aligned x-axis is not a movement boundary for a plot that does not draw it.
                target = None
        return target

    def _minimum_zoom_width(self) -> float:
        """Two sample intervals for the live/drawn target, or zero when none defines one."""
        target = self._movement_target()
        if target is not None and hasattr(self._view, "sample_dx"):
            try:
                dx = abs(float(self._view.sample_dx(*target)))
            except (KeyError, ValueError, IndexError, TypeError, AttributeError):
                dx = 0.0
            if dx > 0.0:
                return 2.0 * dx
        return 0.0

    def _scan(self, direction: int) -> bool:
        """Translate view and live Measure selection by the same clamped delta."""
        rng = self._current_range()
        if self._view is None or rng is None:
            return False
        lo, hi = rng
        width = hi - lo
        if width <= 0.0:
            return False
        self._extent = self._movement_extent()
        if self._extent is None:
            self._apply(lo, hi)
            return False
        e0, e1 = self._extent
        span = e1 - e0
        # Preserving the chosen zoom is the scan contract. If the old (usually overlay-union)
        # window is as wide as, or wider than, a newly selected short recording, no translated
        # copy can fit inside that target. Leave the view alone; Zoom in remains the escape hatch.
        if span <= 0.0 or width >= span * (1.0 - FULL_EXTENT_TOL):
            self._apply(lo, hi)
            return False
        requested = lo + (1 if direction >= 0 else -1) * SCAN_STEP_FRACTION * width
        x0 = self._clamp_start(requested, width)
        delta = x0 - lo
        eps = max(width, 1.0) * 1e-14
        # A padded event jump may legitimately leave part of the view beyond the recording
        # (for example [9.5, 10.5] around an event ending at 10 s).  Clamping a further-right
        # proposal then produces x0=9.0: without the directional check, Next moves BACKWARDS even
        # though its disabled button says there is nowhere further right to go.  Shortcuts call
        # this method directly, so the button's disabled state is not a sufficient guard.
        if ((direction >= 0 and delta <= eps)
                or (direction < 0 and delta >= -eps)):
            self._apply(lo, hi)
            return False

        self._width = width
        self._set_view_range(x0, width)
        # Moving on means moving the live annotation cursors too. This changes no saved row; it
        # only preserves the cursor's screen position so the next event can be bracketed at once.
        if hasattr(self._view, "translate_measure_region"):
            self._view.translate_measure_region(delta)
        self._apply(x0, x0 + width)
        return True

    def _zoom(self, multiplier: float, anchor: float | None = None) -> bool:
        rng = self._current_range()
        if self._view is None or rng is None or multiplier <= 0.0:
            return False
        lo, hi = rng
        width = hi - lo
        self._extent = self._movement_extent()
        if width <= 0.0 or self._extent is None:
            return False
        e0, e1 = self._extent
        full = e1 - e0
        if full <= 0.0:
            return False
        wanted = min(full, width * float(multiplier))
        if multiplier < 1.0:
            minimum = self._minimum_zoom_width()
            wanted = max(wanted, minimum) if minimum > 0.0 else wanted
        eps = max(width, wanted, 1.0) * 1e-14
        # A target switch can leave a union-sized view wider than the new target. min(full, ...)
        # then makes a nominal Zoom out *smaller*. Buttons are disabled in that state, but the
        # Shift+Down shortcut calls this method directly, so direction is enforced here too.
        if ((multiplier > 1.0 and wanted <= width + eps)
                or (multiplier < 1.0 and wanted >= width - eps)
                or abs(multiplier - 1.0) <= 1e-14):
            self._apply(lo, hi)
            return False
        # The point that stays put: the pointer's x when one was given and it is INSIDE the view
        # (an x off the view -- the pointer over an axis, say -- would fling the window), else the
        # centre. ``m`` is the multiplier actually achieved after the extent and sample-floor
        # clamps above, so the anchor holds even where the request was cut short.
        px = 0.5 * (lo + hi)
        if anchor is not None and not isinstance(anchor, bool):   # a bool is a signal's `checked`
            try:
                candidate = float(anchor)
            except (TypeError, ValueError):
                candidate = float("nan")
            if lo <= candidate <= hi:
                px = candidate
        m = wanted / width
        x0 = self._clamp_start(px - (px - lo) * m, wanted)
        if hasattr(self._view, "zoom_x_to"):
            self._view.zoom_x_to(x0, x0 + wanted)
        else:
            self._view.set_x_range(x0, x0 + wanted)
        self._apply(x0, x0 + wanted)
        return True

    # -- internals -------------------------------------------------------------------------------
    def _current_range(self) -> tuple[float, float] | None:
        if self._view is None:
            return None
        anchor = self._view.anchor_plot()
        if anchor is None:
            return None                       # no document / no visible layer: nothing to pan
        lo, hi = anchor.vb.viewRange()[0]
        return float(lo), float(hi)

    def _clamp_start(self, x0: float, width: float) -> float:
        """Keep [x0, x0+width] inside the extent. Order matters: pull the right edge in first, then
        the left, so a view WIDER than the extent lands flush at the start instead of off the end."""
        if self._extent is None:
            return x0
        e0, e1 = self._extent
        if x0 + width > e1:
            x0 = e1 - width
        if x0 < e0:
            x0 = e0
        return x0

    def _set_active(self, on: bool) -> None:
        if on == self._active:
            return
        self._active = on
        self._bar.setVisible(on)              # the CONTAINER keeps its height; only this hides
        if not on:
            self._syncing = True
            self._bar.setRange(0, 0)          # nothing to pan: the handle fills the trough
            self._syncing = False

    @QtCore.Slot(float, float)
    def _view_to_bar(self, xmin: float, xmax: float) -> None:
        """The view moved (box zoom, navigator drag, Full reset, zoom back/forward,
        event-table jump, measurement Go-to — every one of them lands here). Mirror it."""
        if self._syncing:
            return                            # our own pan bouncing back; the Navigator's rule
        self._extent = self._movement_extent()
        self._apply(float(xmin), float(xmax))

    def _apply(self, xmin: float, xmax: float) -> None:
        """Restate visibility + bar geometry for a visible range of [xmin, xmax]. Never touches
        the view."""
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        width = hi - lo
        self._x0, self._width = lo, width

        extent = self._extent
        if extent is None or width <= 0.0:
            self._set_active(False)           # no document, or a degenerate/collapsed view
            self._sync_buttons(lo, hi)
            return
        e0, e1 = extent
        span = e1 - e0
        if span <= 0.0 or width >= span * (1.0 - FULL_EXTENT_TOL):
            self._set_active(False)           # at (or wider than) full extent: nothing to pan
            self._sync_buttons(lo, hi)
            return

        self._set_active(True)
        page = min(PAN_STEPS, max(1, int(round(width / span * PAN_STEPS))))
        value = int(round((lo - e0) / span * PAN_STEPS))
        value = max(0, min(PAN_STEPS - page, value))

        self._syncing = True
        # pageStep BEFORE the range: the maximum is PAN_STEPS - pageStep, so clicking the trough
        # pages by exactly one screenful and the handle's length reads as the visible fraction.
        self._bar.setPageStep(page)
        self._bar.setSingleStep(max(1, page // 10))    # an arrow nudge = a tenth of a screen
        self._bar.setRange(0, PAN_STEPS - page)
        self._bar.setValue(value)
        self._syncing = False
        self._sync_buttons(lo, hi)

    def _sync_buttons(self, lo: float, hi: float) -> None:
        extent = self._extent
        valid = extent is not None and hi > lo
        if not valid:
            self._previous_button.setEnabled(False)
            self._next_button.setEnabled(False)
            self._zoom_out_button.setEnabled(False)
            self._zoom_in_button.setEnabled(False)
            self._full_trace_button.setEnabled(False)
            return
        e0, e1 = extent
        width = hi - lo
        span = e1 - e0
        eps = max(width, e1 - e0, 1.0) * 1e-12
        scannable = span > 0.0 and width < span * (1.0 - FULL_EXTENT_TOL)
        self._previous_button.setEnabled(scannable and lo > e0 + eps)
        self._next_button.setEnabled(scannable and hi < e1 - eps)
        self._zoom_out_button.setEnabled(width < span * (1.0 - FULL_EXTENT_TOL))
        minimum = self._minimum_zoom_width()
        self._zoom_in_button.setEnabled(
            minimum <= 0.0 or width > minimum + max(width, minimum, 1.0) * 1e-14)
        full_extent = self._view.full_x_extent() if self._view is not None else None
        if full_extent is None:
            self._full_trace_button.setEnabled(False)
        else:
            f0, f1 = full_extent
            full_span = f1 - f0
            full_eps = max(abs(full_span) * FULL_EXTENT_TOL, 1e-12)
            at_full = (full_span > 0.0 and abs(lo - f0) <= full_eps
                       and abs(hi - f1) <= full_eps)
            self._full_trace_button.setEnabled(not at_full)

    def _set_view_range(self, x0: float, width: float) -> None:
        """Push [x0, x0+width] onto the view under the loop guard."""
        self._syncing = True
        try:
            self._view.set_x_range(x0, x0 + width)
        finally:
            self._syncing = False
        self._x0 = x0

    @QtCore.Slot(int)
    def _bar_to_view(self, value: int) -> None:
        """The user moved the handle: translate the view, preserving the zoom width EXACTLY."""
        if self._syncing or not self._active or self._view is None:
            return                            # our own programmatic writes never drive the view
        if self._extent is None or self._width <= 0.0:
            return
        e0, e1 = self._extent
        span = e1 - e0
        if span <= 0.0:
            return
        width = self._width                   # the float the VIEW reported — not page-derived
        x0 = self._clamp_start(e0 + (float(value) / PAN_STEPS) * span, width)
        self._set_view_range(x0, width)
        self._apply(x0, x0 + width)
