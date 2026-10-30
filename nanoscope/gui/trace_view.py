from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np
import pyqtgraph as pg
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.analysis import OverlayHint, OverlayKind
from ..core.layer import LAYER_INKS, Layer, ViewMode
from ..core.session import Session
from ..core import clock
from ..core.types import MIN_Y_SPAN_BY_UNITS, ChannelKind
from ..plot.style import BASELINE_TRACK_INK
from .axes import ReadableLinearAxis
from .display_residual import DisplayResidualCache

__all__ = ["PaneSpec", "plan_panes", "effective_kind", "ChannelPane", "TraceView", "Crosshair", "TraceViewBox",
           "PREVIEW_CHANNEL_NOT_SHOWN", "MAX_X_SPAN_FACTOR", "MEASURE_MAX_VIEW_FRACTION",
           "MEASURE_HANDLE_SIZE_PX", "WHEEL_Y_PAN_FRACTION", "WHEEL_X_PAN_FRACTION",
           "DRAW_CLICK_MAX_PX", "SWIPE_LOCK_PX", "layer_ink", "FILTER_PREVIEW_HEX"]

#: The filter-preview curve's ink. NOT a layer colour and deliberately outside
#: :data:`~nanoscope.core.layer.LAYER_INKS`: it is a "what this filter would do to the
#: trace you are looking at" ghost drawn ON TOP of a real layer, so reading it as an
#: overlaid recording is the one misreading that matters. Measured at >= 15 CIEDE2000
#: from every one of the eight layer inks (and from the level overlay and the event-span
#: amber) under normal vision AND under all three simulated deficiencies -- that
#: separation is a CONSTRAINT the layer palette was solved under, not a coincidence, so
#: this hex is one of the four reserved inks named in the LAYER_INKS note. Spelled as a
#: constant because ``tests/gui/test_s3_repoint_preview_rebuild.py`` identifies the
#: preview item on the plot by this exact colour.
FILTER_PREVIEW_HEX = "#e69f00"

#: *** THE ZOOM-OUT CEILING. *** The widest x-span any pane may show, as a multiple of the
#: recording's full extent. The y-side has had a floor since v1 (``minYRange`` in
#: :meth:`ChannelPane.apply_y_policy`) but the x zoom-out direction was completely
#: unguarded: repeated scale-out gestures past full extent drove the visible span to
#: ~6e16 s, at which point pyqtgraph's auto-downsampler
#: computes a per-bin reshape sized by ``span / dx`` and raises
#: ``ValueError: Maximum allowed dimension exceeded`` on EVERY repaint — a dead,
#: non-updating curve with nothing on screen saying why, recoverable only by knowing
#: Ctrl+0 / the Full button. 2x the extent keeps the whole recording (plus honest
#: empty margin either side) reachable while staying about seven orders of magnitude
#: under the failure, so a zoom-out storm coasts to a stop instead of crashing the
#: renderer. Enforced with ``setLimits(maxXRange=...)``, i.e. inside
#: ``ViewBox.setRange`` itself, so right-drag pan-mode scaling and any programmatic
#: ``setXRange`` both hit the same ceiling — one rule, every horizontal zoom path.
MAX_X_SPAN_FACTOR = 2.0

#: A newly placed manual-annotation mask occupies at most one fifth of the visible
#: trace. The user may still drag either edge farther afterwards: this is a placement
#: rule, not a restriction on what can be annotated.
MEASURE_MAX_VIEW_FRACTION = 0.20

#: Circular grips on the two mask boundaries. Unlike a thicker cosmetic pen, an
#: InfiniteLine marker expands the real device-pixel mouse hit box too.
MEASURE_HANDLE_SIZE_PX = 16.0

#: One conventional 120-delta wheel notch translates the waveform by one tenth of the
#: current pane height. Translation (rather than scaling) preserves amplitude span.
WHEEL_Y_PAN_FRACTION = 0.10

#: ...and a horizontal notch (or Shift plus a vertical one, which a Mac delivers as
#: horizontal) slides the view along TIME by one tenth of the visible span.
WHEEL_X_PAN_FRACTION = 0.10

#: A drag that ends narrower than this many screen pixels is a click, not a drawn span.
DRAW_CLICK_MAX_PX = 3.0

#: A trackpad swipe picks its axis (time or amplitude) once either accumulated ``pixelDelta``
#: component reaches this many pixels. The first event of a gesture (``ScrollBegin``) carries
#: no movement, so the choice cannot be made there; until it is made nothing moves, and what
#: was held back is applied on the chosen axis in one go so no travel is lost.
SWIPE_LOCK_PX = 6.0

_QT = QtCore.Qt
_SCROLL_BEGIN = _QT.ScrollPhase.ScrollBegin
_NO_SCROLL_PHASE = _QT.ScrollPhase.NoScrollPhase


def _wheel_phase(ev):
    """The scroll phase of a wheel event; ``NoScrollPhase`` for a mouse wheel (and for a test
    double that has no such method)."""
    try:
        return ev.phase()
    except AttributeError:
        return _NO_SCROLL_PHASE


def _wheel_pixel_delta(ev) -> tuple[float, float]:
    """``(x, y)`` of the event's ``pixelDelta`` in logical pixels; zero for a mouse wheel."""
    try:
        pd = ev.pixelDelta()
    except AttributeError:
        return 0.0, 0.0
    return float(pd.x()), float(pd.y())


def _wheel_is_horizontal(ev) -> bool:
    try:
        return ev.orientation() == _QT.Orientation.Horizontal
    except AttributeError:
        return False            # a bare test double: the vertical wheel the trace always had


def _wheel_has_shift(ev) -> bool:
    try:
        return bool(ev.modifiers() & _QT.KeyboardModifier.ShiftModifier)
    except AttributeError:
        return False


class _ScrollGesture:
    """The axis one trackpad scroll gesture is locked to, shared by every pane's ViewBox.

    Qt reports each wheel event along whichever component dominates THAT event, so a diagonal
    swipe alternates between time and amplitude from one event to the next. The lock is made
    once, from the accumulated ``pixelDelta`` since the gesture began, and kept until the next
    ``ScrollBegin`` (momentum events continue on it). Two failsafes for hardware that does not
    frame its gestures: a pause longer than :attr:`IDLE_S` ends a gesture too, and an event with
    a pixel delta but no phase at all decides for itself and never locks (see
    :meth:`TraceViewBox.wheelEvent`)."""

    IDLE_S = 0.6

    def __init__(self) -> None:
        self.axis: str | None = None
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.last_t: float | None = None

    def reset(self) -> None:
        self.axis = None
        self.acc_x = 0.0
        self.acc_y = 0.0


def _alive(obj) -> bool:
    """Is the C++ object behind this wrapper still there? Calling into one that is not can
    SEGFAULT instead of raising: ``GraphicsView.close()`` clears the scene (deleting every item)
    and only then detaches the viewport, and that last step sends the view a focus-out while the
    panes it lists are already gone."""
    try:
        return bool(shiboken6.isValid(obj))
    except Exception:  # noqa: BLE001
        return False


def _draw_wanted(draw_mode, ev) -> bool:
    """Does this left drag draw a span? Draw mode on XOR Option (Alt) held at its start.

    Option inverts the drag in BOTH states, so a one-off span needs no mode and a quick dive
    into an event (box zoom) needs no mode change. Control is avoided on purpose: on a Mac,
    Control+click is a right click."""
    on = bool(draw_mode()) if draw_mode is not None else False
    option = bool(ev.modifiers() & _QT.KeyboardModifier.AltModifier)
    return on != option


def _drag_phase(ev) -> str:
    return "start" if ev.isStart() else ("finish" if ev.isFinish() else "move")


class TraceViewBox(pg.ViewBox):
    """A ViewBox tuned for trace inspection, so the user can get *inside* a millisecond event.

    - Mouse wheel PANS amplitude vertically without changing either axis span. Wheel up moves the
      drawn signal up; wheel down moves it down. Horizontal scale belongs to box zoom and the
      persistent −/+ controls, so an accidental scroll cannot lose the event in time.
    - A horizontal two-finger swipe (or a horizontal wheel, or Shift plus the wheel) slides the
      view along TIME, with the axis locked for the whole gesture (:class:`_ScrollGesture`).
    - Mouse mode is RectMode by default: left-drag draws a zoom box, so you frame an event and dive
      straight in — the gesture the coarse navigator strip physically cannot do (~36 ms/pixel).
      Right-drag still scales, and View ▸ Drag mode flips back to panning.
    - In DRAW MODE (or with Option held) a left-drag draws a Measure span instead: the drag is
      handed, phase by phase, to ``on_draw`` and never reaches the box zoom. Option inverts it
      in both states.
    - Double-click requests a manual mask centred at that data-x. The Measure panel owns the active
      annotation target and therefore performs the actual placement. In draw mode it asks to
      CATCH the event there (``on_catch``) instead.
    - ``hoverEvent`` reports the pointer ENTERING this box (``on_hover_enter``): pyqtgraph's
      ViewBox has no hover handling of its own.

    Every hook is an injected callable, like ``on_double_click`` always was, so this class
    knows nothing of the view that owns it.
    """

    def __init__(self, *args, on_double_click=None, on_draw=None, draw_mode=None,
                 on_catch=None, on_hover_enter=None, scroll=None, extent=None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_double_click = on_double_click
        #: ``on_draw(source, phase, ev, handle) -> handle``. ``phase`` is "start", "move",
        #: "finish" or "cancel"; ``handle`` is whatever the previous call returned (None at
        #: the start), so the owner can hold the gesture's token and the box need not.
        self._on_draw = on_draw
        self._draw_mode = draw_mode
        self._on_catch = on_catch
        self._on_hover_enter = on_hover_enter
        self._scroll = scroll if scroll is not None else _ScrollGesture()
        self._extent = extent
        self._drawing = False          # THIS box took the left drag now in progress
        self._draw_handle = None

    # -- wheel / trackpad ------------------------------------------------------------------------
    def wheelEvent(self, ev, axis=None):  # noqa: N802  (Qt/pyqtgraph signature)
        del axis  # the trace answers every wheel the same way, wherever on the pane it lands
        gesture = self._scroll
        phase = _wheel_phase(ev)
        now = time.monotonic()
        # THE LOCK IS RESET BEFORE ANY EARLY RETURN. A gesture's first event (ScrollBegin) carries
        # no movement, so the zero-delta return below would otherwise swallow the one event that
        # says "a new swipe starts here" and the previous swipe's axis would outlive it.
        if phase == _SCROLL_BEGIN or (gesture.last_t is not None
                                      and now - gesture.last_t > gesture.IDLE_S):
            gesture.reset()
        gesture.last_t = now

        px, py = _wheel_pixel_delta(ev)
        if px != 0.0 or py != 0.0:
            self._precise_wheel(ev, phase, px, py)
        else:
            self._notched_wheel(ev)

    def _precise_wheel(self, ev, phase, px: float, py: float) -> None:
        """A trackpad (or any device that reports pixels). Locked per gesture when it frames
        its gestures with scroll phases; decided event by event when it does not."""
        gesture = self._scroll
        if phase == _NO_SCROLL_PHASE:
            axis = "x" if abs(px) > abs(py) else "y"
            use_x, use_y = px, py
        else:
            gesture.acc_x += px
            gesture.acc_y += py
            if gesture.axis is None:
                if max(abs(gesture.acc_x), abs(gesture.acc_y)) < SWIPE_LOCK_PX:
                    ev.accept()           # still undecided: hold the travel back
                    return
                gesture.axis = "x" if abs(gesture.acc_x) >= abs(gesture.acc_y) else "y"
                use_x, use_y = gesture.acc_x, gesture.acc_y     # replay what was held back
            else:
                use_x, use_y = px, py
            axis = gesture.axis
        if axis == "x":
            self._pan_time_pixels(ev, use_x)
        else:
            # Qt gives a pixel-precise device an angle delta of 2 per pixel; the amplitude pan
            # is defined on that scale, so a vertical swipe moves exactly as it always did.
            self._pan_amplitude(ev, 2.0 * use_y)

    def _notched_wheel(self, ev) -> None:
        """A mouse wheel: no pixels, no phase, so every event decides from its orientation.
        Horizontal (or Shift plus vertical, which a Mac delivers as horizontal) pans time."""
        delta = float(ev.delta())
        if not np.isfinite(delta) or delta == 0.0:
            ev.ignore()
            return
        if _wheel_is_horizontal(ev) or _wheel_has_shift(ev):
            self._pan_time_notches(ev, delta)
        else:
            self._pan_amplitude(ev, delta)

    def _pan_amplitude(self, ev, delta: float) -> None:
        if not self.state["mouseEnabled"][1]:
            ev.ignore()
            return
        y0, y1 = (float(v) for v in self.viewRange()[1])
        span = y1 - y0
        if not np.isfinite(delta) or delta == 0.0 or not np.isfinite(span) or span <= 0.0:
            ev.ignore()
            return

        # QGraphics scene-y grows downward. On a normal (non-inverted) data axis, moving the
        # viewed numerical range DOWN makes a fixed waveform render UP on screen — exactly the
        # literal gesture requested. Account for the rare inverted-y view so the visual direction
        # remains stable. Cap one event's fling while retaining smooth fractional trackpad deltas.
        steps = float(np.clip(delta / 120.0, -4.0, 4.0))
        visual_sign = 1.0 if self.state.get("yInverted", False) else -1.0
        shift = visual_sign * span * WHEEL_Y_PAN_FRACTION * steps
        self._resetTarget()
        self.translateBy(y=shift)
        ev.accept()
        self.sigRangeChangedManually.emit([False, True])

    def _pan_time_pixels(self, ev, px: float) -> None:
        """Slide the view so the trace follows the fingers: ``px`` logical pixels of content
        travel, converted through this pane's own pixels-per-second."""
        x0, x1 = (float(v) for v in self.viewRange()[0])
        width = float(self.width())
        if not (np.isfinite(px) and width > 0.0 and x1 > x0):
            ev.ignore()
            return
        self._slide_time(ev, -px * (x1 - x0) / width)

    def _pan_time_notches(self, ev, delta: float) -> None:
        x0, x1 = (float(v) for v in self.viewRange()[0])
        if not (np.isfinite(delta) and x1 > x0):
            ev.ignore()
            return
        steps = float(np.clip(delta / 120.0, -4.0, 4.0))
        self._slide_time(ev, -steps * (x1 - x0) * WHEEL_X_PAN_FRACTION)

    def _slide_time(self, ev, dt: float) -> None:
        """Translate x by ``dt`` data units without leaving the recording any further than the
        view already was (a padded event jump legitimately shows blank margin; a fling must
        not carry the window off into nothing, and momentum keeps flinging)."""
        if not self.state["mouseEnabled"][0]:
            ev.ignore()
            return
        x0, x1 = (float(v) for v in self.viewRange()[0])
        extent = self._extent() if self._extent is not None else None
        if extent is not None:
            e0, e1 = extent
            if dt > 0.0:
                dt = min(dt, max(0.0, max(float(e1), x1) - x1))
            elif dt < 0.0:
                dt = max(dt, min(0.0, min(float(e0), x0) - x0))
        ev.accept()
        if dt == 0.0 or not np.isfinite(dt):
            return
        self._resetTarget()
        self.translateBy(x=dt)
        self.sigRangeChangedManually.emit([True, False])

    # -- left drag: box zoom, or draw -----------------------------------------------------------
    def mouseDragEvent(self, ev, axis=None):  # noqa: N802  (pyqtgraph MouseDragEvent)
        # Only an un-axised LEFT drag can be a draw: a drag started on an axis scales that axis,
        # and the right and middle buttons keep their pyqtgraph meaning.
        if (self._on_draw is not None and axis is None
                and ev.button() == QtCore.Qt.LeftButton):
            if ev.isStart():
                # Decided ONCE, at the start, from the modifier then: releasing Option halfway
                # must not turn a draw into a box zoom under the user's hand.
                self._drawing = _draw_wanted(self._draw_mode, ev)
                self._draw_handle = None
            if self._drawing:
                ev.accept()           # accepted WITHOUT super(): that is what stops the box zoom
                self._draw_handle = self._on_draw(self, _drag_phase(ev), ev, self._draw_handle)
                if ev.isFinish():
                    self._drawing = False
                    self._draw_handle = None
                return
        super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):  # noqa: N802  (pyqtgraph MouseClickEvent)
        # A right-click while this box owns a draw cancels it. While the box owns a drag the
        # scene routes clicks of OTHER buttons to it, and there is no context menu on a trace
        # pane to lose.
        if ev.button() == QtCore.Qt.RightButton and self._drawing:
            ev.accept()
            if self._on_draw is not None and self._draw_handle is not None:
                self._on_draw(self, "cancel", ev, self._draw_handle)
            return
        # Left double-click -> request a mask at the clicked data-x. pyqtgraph flags
        # double-clicks via ev.double(); the owner decides whether a valid target exists.
        if ev.double() and ev.button() == QtCore.Qt.LeftButton:
            if (self._on_catch is not None and self._draw_mode is not None
                    and self._draw_mode()):
                point = self.mapSceneToView(ev.scenePos())
                ev.accept()
                self._on_catch(self, float(point.x()))
                return
            if self._on_double_click is not None:
                point = self.mapSceneToView(ev.scenePos())
                ev.accept()
                self._on_double_click(float(point.x()))
                return
        super().mouseClickEvent(ev)

    # -- hover -----------------------------------------------------------------------------------
    def hoverEvent(self, ev):  # noqa: N802  (pyqtgraph HoverEvent)
        # pg.ViewBox has no hoverEvent of its own, so the scene never told anyone the pointer
        # had arrived. ``isEnter`` fires once per entry (not per move), which is exactly the
        # "the pointer came to the trace" moment the focus rule wants.
        if self._on_hover_enter is not None and not ev.isExit() and ev.isEnter():
            self._on_hover_enter(self)


class _MeasureRegionItem(pg.LinearRegionItem):
    """The Measure mask. Identical to a ``LinearRegionItem`` except that, in draw mode or with
    Option held, a drag that starts on its BODY draws a new span instead of moving the mask.

    The scene gives a drag to the item that claimed it at the last hover, and failing that to
    the first item near the press that accepts it in z order, and a movable region accepts any
    left drag — so merely not claiming the hover is not enough (probed: the mask still moved).
    The body therefore accepts the drag and forwards it to the same callback the pane's
    ViewBox uses. Its two edge LINES are separate items and keep resizing exactly as before.

    A right-click during a forwarded draw reaches THIS item (it owns the drag), whose base
    ``mouseClickEvent`` only acts on a move it started itself, so the cancel is handled here."""

    def __init__(self, *args, on_draw=None, draw_mode=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_draw = on_draw
        self._draw_mode = draw_mode
        self._forwarding = False
        self._draw_handle = None

    def mouseDragEvent(self, ev):  # noqa: N802
        if self._on_draw is not None and ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                self._forwarding = _draw_wanted(self._draw_mode, ev)
                self._draw_handle = None
            if self._forwarding:
                ev.accept()
                self._draw_handle = self._on_draw(self, _drag_phase(ev), ev, self._draw_handle)
                if ev.isFinish():
                    self._forwarding = False
                    self._draw_handle = None
                return
        super().mouseDragEvent(ev)

    def mouseClickEvent(self, ev):  # noqa: N802
        if ev.button() == QtCore.Qt.RightButton and self._forwarding:
            ev.accept()
            if self._on_draw is not None and self._draw_handle is not None:
                self._on_draw(self, "cancel", ev, self._draw_handle)
            return
        super().mouseClickEvent(ev)


class _PaneLabel(QtWidgets.QGraphicsObject):
    """A small caption pinned to the bottom-left of one pane, like a legend.

    TRANSPARENT TO THE MOUSE, TWICE OVER: it accepts no mouse button (so the scene never makes
    it the grabber of a press) and its ``shape`` is empty (so pyqtgraph's own item search for
    drags, clicks and hovers, which tests shapes, never finds it). A banner that was a real
    widget blocked every drag beneath it; this one cannot. It is a child of the ViewBox, not
    of its data group, so it stays put in pixels while the data moves, and it is not an
    ``addItem``'d child either, so autorange never sees it."""

    _MARGIN = 8.0
    _PAD_X = 6.0
    _PAD_Y = 3.0

    def __init__(self, view_box: pg.ViewBox) -> None:
        super().__init__(view_box)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)
        self.setZValue(30)               # above the curves, the mask (8) and the crosshair (20/21)
        self._view_box = view_box
        self._text = ""
        font = QtGui.QFont()
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1.0))
        self._font = font
        self._rect = QtCore.QRectF()
        view_box.sigResized.connect(self._reposition)

    def text(self) -> str:
        return self._text

    def detach(self) -> None:
        """Stop following the ViewBox's size (it outlives this caption)."""
        try:
            self._view_box.sigResized.disconnect(self._reposition)
        except (RuntimeError, TypeError):
            pass

    def set_text(self, text: str) -> None:
        text = str(text)
        if text == self._text and not self._rect.isNull():
            return
        self.prepareGeometryChange()
        self._text = text
        metrics = QtGui.QFontMetricsF(self._font)
        w = metrics.horizontalAdvance(text) + 2.0 * self._PAD_X
        h = metrics.height() + 2.0 * self._PAD_Y
        self._rect = QtCore.QRectF(0.0, 0.0, w, h)
        self._reposition()
        self.update()

    def _reposition(self, *_args) -> None:
        try:
            height = float(self._view_box.height())
        except RuntimeError:             # the ViewBox is being torn down with its plot
            return
        self.setPos(self._MARGIN, max(0.0, height - self._rect.height() - self._MARGIN))

    def boundingRect(self) -> QtCore.QRectF:  # noqa: N802
        return self._rect

    def shape(self) -> QtGui.QPainterPath:
        return QtGui.QPainterPath()

    def paint(self, painter, option=None, widget=None) -> None:
        if not self._text:
            return
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(20, 24, 30, 175))
        painter.drawRoundedRect(self._rect, 3.0, 3.0)
        painter.setPen(QtGui.QColor("#d7dde5"))
        painter.setFont(self._font)
        painter.drawText(self._rect, int(QtCore.Qt.AlignmentFlag.AlignCenter), self._text)


class _DrawGesture:
    """One left-drag draw, from its first accepted frame to its release.

    ``alive`` IS THE TOKEN. A rebuild, a cancel or a release clears it, and every later frame the
    scene sends is dropped on seeing it clear: the scene keeps delivering a drag to the item that
    started it until the button comes up, and after a rebuild that item is a ViewBox that has
    been removed from the plot. The ``prev_*`` fields are what the mask was before the gesture,
    so a cancel can put it back exactly (including "there was no mask")."""

    __slots__ = ("alive", "start_x", "doc_id", "channel", "prev_range", "prev_xrange",
                 "prev_empty")

    def __init__(self, start_x: float, doc_id: str, channel: int,
                 prev_range: tuple[float, float] | None,
                 prev_xrange: tuple[float, float] | None, prev_empty: bool) -> None:
        self.alive = True
        self.start_x = start_x
        self.doc_id = doc_id
        self.channel = channel
        self.prev_range = prev_range
        self.prev_xrange = prev_xrange
        self.prev_empty = prev_empty


# ------------------------------------------------------------------------------------------------
# Pane planning — the whole resolution of "stacked panes" vs "overlay".
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PaneSpec:
    kind: ChannelKind
    units: str
    layer_ids: tuple[str, ...]


def _effective_units(session: Session, doc_id: str, ch: int) -> str:
    """calibration.target_units if set, else the channel's raw_units; "" == unknown. Computed
    without materialising the calibrated array (grouping needs the label, not the data)."""
    info = session.document(doc_id).meta.channels[ch]
    cal = session.calibration(doc_id, ch)
    return cal.target_units or info.raw_units


def layer_ink(session: Session, layer: Layer) -> str:
    """*** THE ONE PLACE THE TRACE VIEW DECIDES WHAT COLOUR A LAYER IS. ***

    ``Layer.color`` is the concrete dark-pane ink persisted in ``layers[].color`` and is
    therefore the trace's visual truth.  New layers derive that value from the project's
    identity-keyed registry, and a live colour edit updates both representations, but a
    project restore must not derive one persisted fact from the other: doing so repaints an
    older project after a palette change.  Plot sources independently render the registry's
    saved slot (or user ink) through their light-background palette.

    ``session`` remains in the signature because this is the stable trace-view seam used by
    callers and tests; colour resolution itself deliberately needs no session lookup.
    """
    del session
    return layer.color


def effective_kind(session: Session, doc_id: str, ch: int) -> ChannelKind:
    """*** WHAT THE VIEW TREATS THIS CHANNEL AS. *** The user's calibration override when they
    set one (``ChannelCalibration.kind``), else the header's own declaration.

    THE SAME SOURCE OF TRUTH AS :func:`_effective_units`, and that is the whole point. The two
    are read side by side — one becomes the y-axis label, the other the pane's rank and (in
    OVERLAY) its grouping key — and until this read the calibration too they came from
    different places, which is the owner's opening question about this app: *"why does the app
    place the pA on the top panel for some traces, and the voltage on the bottom yet reverses
    in others"*. The ordering rule below is deterministic and always was; what was not was the
    KIND it sorted on. Correcting an undeclared channel to 'pA' in the Calibration panel
    relabelled the axis and left the pane ranked UNKNOWN — i.e. below the voltage — so two
    recordings of one experiment came up in opposite orders depending on which of them
    happened to declare its channels.

    Because this feeds BOTH :func:`_current_first` AND the OVERLAY grouping key, the same one
    answer also un-merges the other half of the same complaint: on a TDMS whose channels are
    both (RAW_VOLTAGE, 'V') the two curves grouped into a single pane sharing one y-axis, with
    a -49 pA current and its command voltage on one scale and no way to separate them.

    A DELEGATE, NOT AN IMPLEMENTATION. The answer lives on
    :meth:`~nanoscope.core.session.Session.effective_kind` now, because while it lived here
    the surfaces that could not import a GUI module -- the pure workflow-status derivation
    behind the status-bar chip and the strip, the Calibration panel's partner lookup, the
    analysis panel's gate -- each read the header instead, and the app said "No current
    channel" about a channel it had just calibrated and detected on. One home, reachable from
    core, is what makes "count as current EVERYWHERE" a property rather than a hope.
    """
    return session.effective_kind(doc_id, ch)


#: The old private name, kept as an alias: three internal callers and a handful of
#: comments/tests refer to ``_kind``; the public name exists because the navigator and
#: MainWindow's surface gates need the SAME kind resolution the panes use (a channel
#: corrected to pA in the Calibration panel must count as current EVERYWHERE, not just
#: in pane rank — the owner's opening complaint was two surfaces disagreeing).
_kind = effective_kind


def plan_panes(session: Session) -> list[PaneSpec]:
    """*** THE PANE RULE. *** Visible layers only, in layer order.

    STACKED: one PaneSpec per layer.
    OVERLAY: group layers by (ChannelKind, effective units), first-appearance order — NEVER across
    them. That single rule makes the comparison workflow the default data path (the four voltage
    runs' IN15 are all (CURRENT, "pA") -> one pane, overlaid, x-linked) while structurally forbidding
    a shared y-axis between a -49 pA current and a -700 mV command voltage.
    """
    visible = [lay for lay in session.layers() if lay.visible]

    if session.view_mode is ViewMode.STACKED:
        return _current_first([
            PaneSpec(
                kind=_kind(session, lay.doc_id, lay.channel),
                units=_effective_units(session, lay.doc_id, lay.channel),
                layer_ids=(lay.layer_id,),
            )
            for lay in visible
        ])

    groups: dict[tuple[ChannelKind, str], list[str]] = {}
    order: list[tuple[ChannelKind, str]] = []
    for lay in visible:
        key = (_kind(session, lay.doc_id, lay.channel),
               _effective_units(session, lay.doc_id, lay.channel))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(lay.layer_id)
    return _current_first(
        [PaneSpec(kind=k[0], units=k[1], layer_ids=tuple(groups[k])) for k in order])


#: Pane order by kind: the signal being analysed on top, its drive below it, the rest
#: after. STABLE — files disagree about channel order (owner note #1: one recording put
#: current above voltage and another the reverse, so the eye had to re-find the trace on
#: every open), and this is the one place that disagreement is normalised away. Within a
#: kind, layer order still rules, so a deliberately arranged overlay keeps its ordering.
_KIND_RANK = {ChannelKind.CURRENT: 0, ChannelKind.CONDUCTANCE: 1,
              ChannelKind.VOLTAGE: 2, ChannelKind.RAW_VOLTAGE: 3, ChannelKind.UNKNOWN: 4}

#: Relative height of a pane's row. With equal rows a constant command voltage -- a flat
#: line -- took ~37% of the plot height away from the current trace the user is judging.
_PANE_STRETCH_CURRENT = 5
_PANE_STRETCH_OTHER = 2


def _current_first(specs: list[PaneSpec]) -> list[PaneSpec]:
    return sorted(specs, key=lambda s: _KIND_RANK.get(s.kind, len(_KIND_RANK)))


def _pane_y_label(spec: PaneSpec) -> str:
    if spec.units:
        return spec.units
    if spec.kind is not ChannelKind.UNKNOWN:
        return spec.kind.value
    return "signal"


# ------------------------------------------------------------------------------------------------
# One pyqtgraph PlotItem and the curves assigned to it.
# ------------------------------------------------------------------------------------------------
class ChannelPane(QtCore.QObject):
    """One PlotItem holding one PlotDataItem per layer assigned to it."""

    def __init__(self, plot: pg.PlotItem, spec: PaneSpec, session: Session,
                 parent: QtCore.QObject | None = None,
                 baselines: Mapping[str, np.ndarray] | None = None,
                 residual_cache: DisplayResidualCache | None = None) -> None:
        super().__init__(parent)
        self.plot = plot
        self.spec = spec
        self.session = session
        self._curves: dict[str, pg.PlotDataItem] = {}
        #: layer_id -> the baseline that layer's curve is drawn MINUS, when the
        #: baseline-subtracted view is on for it. THE VIEW'S OWN DICT, held by
        #: reference and never copied: the request outlives every pane (a rebuild
        #: constructs new panes with the same mapping), which is what stops a
        #: Stacked/Overlay switch from silently putting the raw trace back under a
        #: y axis still labelled "Δ vs baseline". See `TraceView.set_baseline_subtracted`.
        self._baselines: Mapping[str, np.ndarray] = (
            {} if baselines is None else baselines)
        self._residual_cache = residual_cache if residual_cache is not None else DisplayResidualCache()

        plot.showGrid(x=True, y=True, alpha=0.15)

        for layer_id in spec.layer_ids:
            layer = session.layer(layer_id)
            if layer is None:
                continue
            curve = self.make_curve(plot, layer_ink(session, layer))
            self._curves[layer_id] = curve
            self.set_layer_data(layer)

        self.refresh_y_label()
        self.apply_y_policy()

    @staticmethod
    def make_curve(plot: pg.PlotItem, color: str) -> pg.PlotDataItem:
        """*** MANDATORY FOUR-STEP ORDER (contract §15). Any other order is a hard crash or silent
        data loss. Encapsulated here so it can never be got wrong at a call site. ***

        1. create an EMPTY PlotDataItem
        2. plot.addItem(item)
        3. item.setDownsampling(auto=True, method="peak"); item.setClipToView(True)
        4. item.setData(x=…, y=…)   <- data LAST, by the caller (set_layer_data)

        method="peak" is passed EXPLICITLY (though it is the 0.14.0 default): peak keeps per-bin min
        AND max, so a one-sample blockade survives decimation. subsample would silently hide it —
        the worst possible failure mode for this app.
        """
        item = pg.PlotDataItem()                                   # 1
        plot.addItem(item)                                         # 2
        item.setDownsampling(auto=True, method="peak")            # 3
        item.setClipToView(True)
        item.setPen(pg.mkPen(color, width=1))
        return item                                                # 4 done by caller

    def set_layer_data(self, layer: Layer) -> None:
        """x = float64 shared time axis; y = float32 calibrated signal. Both come from Session.

        ...MINUS ITS BASELINE when this layer is in the subtracted view, and that is the
        ONLY place the subtraction happens. It is a DISPLAY transform: the Session's
        samples are untouched, so every measurement, every column and every export is
        the same array it always was — a curve is drawn from a different number, nothing
        is measured from one.
        """
        curve = self._curves.get(layer.layer_id)
        if curve is None:
            return
        xaxis = self.session.x_axis(layer.doc_id, layer.time_offset_s)
        chan = self.session.calibrated(layer.doc_id, layer.channel)
        y = chan.data
        base = self._baselines.get(layer.layer_id)
        if base is not None and np.asarray(base).shape == np.asarray(y).shape:
            # Imported here, not at module scope: `analysis.baseline` is 5.5k lines and
            # pulls SciPy, and the trace view is constructed on every app start whether
            # or not anybody asks for a residual. The ARITHMETIC lives there and not
            # here on purpose — a view may not own a definition of what "minus its
            # baseline" means (house rule: kernel in analysis/, thin widget in gui/).
            y = self._residual_cache.residual(layer.layer_id, y, base)
        curve.setData(x=xaxis.values, y=y)

    def refresh_y_label(self) -> None:
        """Re-derive the y-axis label, which SAYS when the pane is showing residuals.

        THE ONE THING THAT KEEPS A MODE HONEST. A baseline-subtracted trace looks like a
        perfectly ordinary recording of a pore with no drift — that is the whole point of
        it — so a user who leaves the box ticked, walks away, and comes back reads
        "-2 pA open pore" off a recording whose open pore is -566 pA. The axis is the one
        surface that is always on screen beside the curve, so the axis is where it says
        so; nothing about this is decoration.

        A pane mixing subtracted and raw layers says so too ("mixed"), because on that
        pane neither answer is true of every curve.
        """
        units = _pane_y_label(self.spec)
        subtracted = [lid for lid in self.spec.layer_ids if lid in self._baselines]
        if not subtracted:
            self.plot.setLabel("left", units)
        elif len(subtracted) == len(self.spec.layer_ids):
            self.plot.setLabel("left", f"Δ vs baseline ({units})")
        else:
            self.plot.setLabel("left", f"{units} — mixed: {len(subtracted)} of "
                                       f"{len(self.spec.layer_ids)} baseline-subtracted")

    def apply_y_policy(self) -> None:
        """Impose a physical minimum y-span ONLY where we actually know the physical unit. Keyed by
        effective units, so a pA channel gets a 5 pA floor (which rescues the constant-voltage pane
        from rendering as full-height quantisation noise) while an undeclared-unit ("") trace
        autoranges freely. The limit is SET unconditionally (None == no floor): refresh_units()
        re-keys this after a calibration edit, and a units change must replace or release the old
        unit's floor — a 5 pA floor left behind on nA data forbids any window tighter than
        5000 pA, rendering a ~26 pA signal as an unzoomable flat line."""
        units = self.spec.units.strip().lower()
        self.plot.vb.setLimits(minYRange=MIN_Y_SPAN_BY_UNITS.get(units))

    def refresh_units(self) -> bool:
        """Calibration changed on a channel this pane draws: re-derive the effective units and
        refresh every surface that was baked from them at construction — the y-axis label
        (set in __init__) and the minimum-y-span floor (apply_y_policy). Without this the axis
        keeps claiming the OLD unit while the curves and the crosshair readout beside it are
        already in the new one — two on-screen surfaces disagreeing 1000x about the same signal.

        Overlay panes group layers by identical effective units (the pane rule); an edit can
        break that agreement in place, and regrouping is the NEXT rebuild's job — until then
        this pane stops claiming a unit it can no longer vouch for (units "", so the label
        falls back to the ChannelKind and the y-span floor is released).

        Returns True when the pane's units actually changed, False for a same-unit edit.
        The caller needs the distinction: a units change also invalidates any NUMERIC
        y-window a box-zoom left behind (see remap_y_window), while a same-unit gain
        tweak keeps the user's window exactly as before."""
        units = {
            _effective_units(self.session, layer.doc_id, layer.channel)
            for layer in (self.session.layer(lid) for lid in self.spec.layer_ids)
            if layer is not None
        }
        new_units = units.pop() if len(units) == 1 else ""
        if new_units == self.spec.units:
            return False
        self.spec = replace(self.spec, units=new_units)
        # Through `refresh_y_label`, not `setLabel` directly: a recalibration must not be
        # the gesture that quietly drops the "Δ vs baseline" marker off a pane that is
        # still showing residuals. One writer per fact — that fact is the label.
        self.refresh_y_label()
        self.apply_y_policy()
        return True

    def layer_y_data(self, layer_id: str) -> np.ndarray | None:
        """The y-array currently set on this layer's curve (the ORIGINAL data handed to
        setData, not the display-decimated copy), or None if the layer has no curve.
        Exists so refresh_channel can snapshot a curve's pre-recalibration values and
        derive the old->new mapping from the data itself — the old calibration is
        already gone from the Session by the time calibrationChanged fires."""
        curve = self._curves.get(layer_id)
        return None if curve is None else curve.yData

    def remap_y_window(self, y_before: np.ndarray | None, y_after: np.ndarray | None) -> None:
        """Carry a box-zoom's kept y-window through a units-changing recalibration.

        A RectMode drag turns y-autorange OFF, so after a pA -> nA edit the viewbox
        would keep the OLD numeric window under the NEW unit label: a blank pane whose
        axis reads numbers the signal never had, while the crosshair beside it
        disagrees 1000x — the exact two-surfaces failure refresh_units() fixed for the
        label, left open for the window. Calibration is affine (calibrated =
        raw*gain + offset), so the same raw samples give an exact old->new affine map;
        it is derived from the curve data itself (two samples of distinct value)
        because the Session no longer holds the old gain/offset when the signal fires.

        Falls back to re-enabling y-autorange — re-derive, never guess — when no map
        can be derived (constant trace, missing/mismatched snapshots): autorange
        reframes the signal correctly, it merely loses the exact zoom tightness.
        A pane whose y-autorange is still ON is left alone; it refits by itself."""
        vb = self.plot.vb
        if bool(vb.autoRangeEnabled()[1]):
            return   # autorange will refit to the new-unit data on its own
        y0, y1 = vb.viewRange()[1]

        def _autorange_fallback() -> None:
            vb.enableAutoRange(y=True)

        if (
            y_before is None or y_after is None
            or len(y_before) == 0 or len(y_before) != len(y_after)
        ):
            _autorange_fallback()
            return
        i = int(np.argmin(y_before))
        j = int(np.argmax(y_before))
        vi, vj = float(y_before[i]), float(y_before[j])
        if vi == vj:   # constant trace: the affine map is underdetermined
            _autorange_fallback()
            return
        # calibrated_new = a * calibrated_old + b, exact for an affine recalibration
        # (up to the float32 rounding of the stored samples, far below framing scale).
        a = (float(y_after[j]) - float(y_after[i])) / (vj - vi)
        b = float(y_after[i]) - a * vi
        lo, hi = a * y0 + b, a * y1 + b
        if lo > hi:   # a negative relative gain flips the window; keep it ordered
            lo, hi = hi, lo
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            _autorange_fallback()
            return
        vb.setYRange(lo, hi, padding=0)

    def first_layer(self) -> Layer | None:
        for layer_id in self.spec.layer_ids:
            layer = self.session.layer(layer_id)
            if layer is not None:
                return layer
        return None


# ------------------------------------------------------------------------------------------------
# v2 SEAM — overlay renderers. Declarative hints in; graphics items out. EMPTY registry in v1.
# ------------------------------------------------------------------------------------------------
# A v2 module returns OverlayHint arrays (never pg items); the VIEW turns them into graphics via a
# renderer registered here, keyed by OverlayKind. This is the thing a reviewer points at: to render
# v2 events, register a callable for OverlayKind.SPAN. No renderer imports leak into analysis; no
# per-module special case accumulates in the view.
# The keyword-only ``time_offset_s`` is the layer's alignment nudge, so index->x lands where
# the CURVE is drawn; it defaults to 0.0 in every renderer, which is the pre-existing behaviour.
OverlayRenderer = Callable[..., list[pg.GraphicsObject]]
_OVERLAY_RENDERERS: dict[OverlayKind, OverlayRenderer] = {}   # populated in v2. EMPTY in v1.


def register_overlay_renderer(kind: OverlayKind, renderer: OverlayRenderer) -> None:
    _OVERLAY_RENDERERS[kind] = renderer


# Overlay ids used to end in ``id(hint)``, i.e. a CPython memory ADDRESS. Callers build the hint as
# a temporary and drop it the moment add_overlay returns, so the allocator hands the very next hint
# the same address and two distinct overlays on the same (doc, channel, kind) collide — the second
# one silently evicts the first from ``_overlays`` and its old id becomes a stranded dead reference.
# A process-wide monotonic counter cannot repeat: it never reuses a value, however fast the calls
# come or however promptly the hints are freed. ``next()`` on an itertools counter is a single
# atomic bytecode under the GIL, so concurrent callers still get distinct values.
_overlay_seq = itertools.count(1)


#: Why a STANDING filter-preview request is not on the plot: its channel is not layered
#: anywhere (hidden in the File Manager, or in a view that does not include it). Written
#: as a clause because it is read after `DetectionPanel.set_preview_error`'s fixed prefix
#: ("processed-trace preview unavailable: ..."), and it says how to get the curve back
#: because the request is still standing — showing the channel redraws it.
PREVIEW_CHANNEL_NOT_SHOWN = (
    "that channel is not on the plot right now (hidden in the File Manager, or not part "
    "of the current view) — show it again and the preview comes back with it.")


# ------------------------------------------------------------------------------------------------
# The viewer widget.
# ------------------------------------------------------------------------------------------------
class TraceView(pg.GraphicsLayoutWidget):
    """Renders Layers. Knows about Session and nothing else in gui/."""

    sigVisibleRangeChanged = QtCore.Signal(float, float)   # v2 lazy-overlay hook; also drives Navigator
    # A left double-click on any trace pane, mapped to that pane's data-x. TraceView deliberately
    # does not know which recording/channel is the annotation target; MeasurePanel owns that fact.
    sigMeasurePlacementRequested = QtCore.Signal(float)
    sigMeasureChanged = QtCore.Signal(int, int)            # manual-selection edges -> (start_idx, stop_idx)
    # ...and the SETTLED edges: the drag ended (mouse released) or the region was placed
    # programmatically. sigMeasureChanged fires on EVERY intermediate frame of a drag and is
    # what a listener must coalesce; this fires once, at rest, so a listener can also settle
    # immediately instead of waiting out its own debounce. Same payload, so a listener that
    # only wants "the selection is now X" can connect to either.
    sigMeasureFinished = QtCore.Signal(int, int)           # settled selection -> (start_idx, stop_idx)
    # The adaptive navigator mirrors the selection as a read-only blue band. Bounds signals cannot
    # announce the one state in which there ARE no bounds (Measure was turned off), so activation
    # has its own small signal rather than leaving a stale annotation in the context strip.
    sigMeasureActiveChanged = QtCore.Signal(bool)
    #: Measure is ON and the mask is GONE (:meth:`discard_measure_region`, or a cancelled draw
    #: that put "no mask" back). The bounds signals cannot say this either: there ARE no bounds
    #: to report. The Navigator drops its band on it and the Measure panel drops the readout
    #: and the statistics it would otherwise go on offering to save.
    sigMeasureCleared = QtCore.Signal()
    #: DRAW MODE (a plain left-drag draws a span instead of box-zooming) was switched. Emitted
    #: for every change, including the one the view makes itself when the Measure target pane
    #: goes away (``False``).
    sigDrawModeChanged = QtCore.Signal(bool)
    #: A draw's first accepted frame.
    sigDrawStarted = QtCore.Signal()
    #: A draw was released as a real span: ``(doc_id, channel, lo_sample, hi_sample,
    #: px_samples)``. ``(doc_id, channel)`` is exactly what :meth:`measure_target` returns;
    #: ``lo``/``hi`` are half-open sample indices clipped to the loaded data (the same numbers
    #: :meth:`measure_bounds` now returns); ``px_samples`` is samples per screen pixel at the
    #: current zoom. A receiver that moves the mask itself or calls
    #: :meth:`settle_measure_region` takes over the settle; if none does, the view settles.
    sigDrawReleased = QtCore.Signal(str, int, int, int, float)
    #: A draw ended without a span: right-click, a rebuild, :meth:`cancel_draw`, or a release
    #: narrower than :data:`DRAW_CLICK_MAX_PX` pixels (a click).
    sigDrawCancelled = QtCore.Signal()
    #: A draw (or a catch) was refused, with the sentence to show: the drag was on another
    #: pane ("Draw on the IN15 pane"), or Measure is off.
    sigDrawRefused = QtCore.Signal(str)
    #: A double-click in draw mode on the target pane asks to catch the event there:
    #: ``(doc_id, channel, sample, px_samples)``. No mask is placed.
    sigCatchRequested = QtCore.Signal(str, int, int, float)
    #: The managed filter-preview curve's ON-SCREEN state changed: ``""`` when the standing
    #: request is drawn (or there is no request), otherwise WHY it is not drawn. The view
    #: rebuilds itself on gestures its owner never hears — View ▸ Stacked/Overlay and every
    #: channel show/hide both land on `rebuild` — so the owner cannot know from its own
    #: signals whether the curve it asked for is still there. It is emitted on every
    #: attempt, never coalesced: a listener that renders it (AnalysisPanel puts it on the
    #: preview error line) must not be starved of the one emission that would have told the
    #: user why a ticked box draws nothing.
    sigFilterPreviewChanged = QtCore.Signal(str)

    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._panes: list[ChannelPane] = []
        self._overlays: dict[str, list[pg.GraphicsObject]] = {}
        self._overlay_plots: dict[str, pg.PlotItem] = {}       # overlay_id -> owning plot (for removal)
        #: overlay_id -> the REQUEST behind it, ``(doc_id, channel, hint)``. The rendered
        #: items above die with their pane on every `rebuild`; this outlives one, so the
        #: id an owner is holding still names something on the plot afterwards
        #: (`_reattach_overlays`). Ended by `remove_overlay`, and only by it.
        self._overlay_reqs: dict[str, tuple[str, int, OverlayHint]] = {}
        #: THE BASELINE-SUBTRACTED VIEW's standing request: ``layer_id -> baseline``,
        #: the full-length array that layer's curve is drawn minus. Handed to every
        #: `ChannelPane` BY REFERENCE, so a rebuild's new panes read the same live
        #: mapping and the mode survives a Stacked/Overlay switch — the same rule as
        #: the measure region, the overlays and the filter preview. Empty is the
        #: unchanged app.
        self._subtracted: dict[str, np.ndarray] = {}
        self._residual_cache = DisplayResidualCache()
        #: ``layer_id -> (overlay_id, idx, ABSOLUTE values, ink)`` for the
        #: estimated-baseline rail on that layer, so a second tick REPLACES the line
        #: instead of stacking a second one on it. Rebuild survival is `add_overlay`'s
        #: already; what is kept HERE beyond the id is the track as MEASURED, because
        #: what gets drawn depends on whether the layer is in the subtracted view and
        #: that can change under a standing rail. Storing only the drawn numbers made
        #: the second toggle of "subtract" draw a rail 566 pA off the curve.
        self._baseline_tracks: dict[str, tuple[str, np.ndarray, np.ndarray, str]] = {}
        self._filter_preview: pg.PlotDataItem | None = None    # D4 managed low-pass preview curve
        self._filter_preview_plot: pg.PlotItem | None = None
        #: THE STANDING REQUEST behind that curve — ``(doc_id, channel, data)``, or None
        #: when nobody is asking for a preview. The curve and its plot die with the pane on
        #: every `rebuild`; the REQUEST does not, exactly as `_measure_active` /
        #: `_measure_layer_id` / `_measure_xrange` survive one so the measure region can be
        #: reattached. Without it a rebuild silently withdrew a request the checkbox on the
        #: Analysis panel was still showing as ON. It holds a reference to the processed
        #: array (not a copy — the curve holds the same one) rather than asking the owner to
        #: recompute the whole preprocessing chain on every channel toggle.
        self._filter_preview_req: tuple[str, int, np.ndarray] | None = None
        #: Why that request is not on screen (``""`` when it is, or when there is none).
        #: Mirrors what was last emitted on `sigFilterPreviewChanged`.
        self._filter_preview_reason: str = ""
        self._placeholder: pg.LabelItem | None = None
        # Manual-measurement region (the Measure tool). A single movable LinearRegionItem whose two
        # edges ARE the boundary cursors; lives on the current-channel pane. _measure_active /
        # _measure_layer_id / _measure_xrange survive a rebuild so the region can be reattached; the
        # region + plot refs die with the plot in clear() and are recreated by _reattach_measure().
        self._measure_region: pg.LinearRegionItem | None = None
        self._measure_plot: pg.PlotItem | None = None
        self._measure_layer_id: str | None = None
        self._measure_active: bool = False
        self._measure_xrange: tuple[float, float] | None = None
        #: MEASURE ON, NO MASK. A missing stored range already means "seed a mask", so the empty
        #: state cannot be "no range": it is this explicit flag. While it is set a rebuild
        #: re-creates nothing, and only :meth:`set_measure_region`, a draw, or an explicit
        #: re-seeding turns it off again. Never True while ``_measure_region`` exists.
        self._measure_empty: bool = False
        #: Bumped each time the mask SETTLES (a real drag's release, a programmatic placement,
        #: :meth:`settle_measure_region`), so a draw can tell whether a receiver of its release
        #: already did the settling.
        self._settled_count: int = 0
        # Draw mode: a plain left-drag draws a Measure span instead of box-zooming.
        self._draw_mode: bool = False
        self._draw: _DrawGesture | None = None
        # Focus follows the pointer onto the trace -- but only from holders with no keys of
        # their own. ``_key_owners`` are widgets (the docked Event inspector) that own bare keys.
        self._focus_follows: bool = True
        self._closing: bool = False
        self._key_owners: list[QtWidgets.QWidget] = []
        self._last_focus_reason: QtCore.Qt.FocusReason | None = None
        self._focus_lost_to_window: bool = False
        # The target-pane caption and the optional fallback target used when Measure is off.
        self._target_label_text: str = ""
        self._target_label: _PaneLabel | None = None
        self._annotation_layer_id: str | None = None
        # One axis lock for the whole trace: a swipe that wanders across panes is one gesture.
        self._scroll_gesture = _ScrollGesture()
        # RectMode = left-drag box-zoom (frame an event, dive in). Flipped by set_drag_mode().
        self._drag_mode: int = pg.ViewBox.RectMode

        # X-range history for Back/Forward: dive into an event, reject it, step back to where you
        # were browsing. Settled views are recorded on a debounce so a zoom animation is one entry,
        # not fifty. _restoring guards against recording our own restores.
        self._view_hist: list[tuple[float, float]] = []
        self._view_ptr: int = -1
        self._restoring: bool = False
        self._prev_doc_ids: frozenset[str] = frozenset()   # to tell a toggle from a new document
        self._settle = QtCore.QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(250)
        self._settle.timeout.connect(self._commit_view)

        self.setMinimumHeight(240)
        self.setBackground(None)   # inherit the pg global background

        self.crosshair = Crosshair(self, session)
        # One scene for the whole layout; connect mouse-move once. Panes are recreated per rebuild,
        # but the scene persists, so this connection survives rebuilds.
        self.scene().sigMouseMoved.connect(self.crosshair.on_mouse_moved)

        session.layersChanged.connect(self.rebuild)
        session.viewModeChanged.connect(self.rebuild)
        session.documentClosed.connect(lambda _doc_id: self.rebuild())
        session.calibrationChanged.connect(self.refresh_channel)
        session.sampleRateChanged.connect(self._on_sample_rate_changed)

        self.rebuild()

    def _on_sample_rate_changed(self, doc_id: str) -> None:
        """A document's time base was re-declared (``Session.set_sample_rate_declaration``).

        Every x this view remembers is on the OLD clock: the anchor's range that
        :meth:`rebuild` restores across a same-document rebuild, the Back/Forward trail,
        and the Measure region. Measured before this slot existed, on a 40,000-sample CSV
        read as 100 Hz (0.05–400 s): zoom to [119, 124] s, declare the column is in
        milliseconds, and the rebuild restored [121.1, 121.9] s over data that now span
        0.00005–0.4 s — a blank pane under a valid-looking seconds axis, the Measure region
        300x off the end of the recording, and a Zoom Back that led to 400 s. A rateless
        recording given a rate is the same defect at 10,000x (samples became seconds). So
        when the document is on screen the old-clock state is forgotten first and the
        view frames fresh; the region is re-seeded on the new axis by ``_reattach_measure``.
        """
        if doc_id not in self.session.shown_doc_ids():
            return                              # nothing drawn changed; File Info and the
                                                # status bar follow the signal themselves
        # The Measure region's x-range is on the old clock, but the SPAN it covers is a fact
        # about samples (the panel's own rate-stale tooltip says so): read the indices off
        # the curve still on screen — its x data IS the old axis — and re-place the region
        # from them on the new axis after the rebuild. Re-seeding would move the cursors off
        # the blockade the user put them on and quietly measure a baseline stretch instead.
        carry: tuple[str, int, int] | None = None
        if (self._measure_active and self._measure_layer_id is not None
                and self._measure_xrange is not None):
            pane = self.pane_for_layer(self._measure_layer_id)
            curve = pane._curves.get(self._measure_layer_id) if pane is not None else None
            old_x = getattr(curve, "xData", None)
            if old_x is not None and len(old_x):
                lo, hi = sorted(self._measure_xrange)
                # The same nearest-sample rule measure_bounds() applies, so the carried
                # span IS the span the panel reported just before the signal.
                i0 = self._nearest_index(old_x, float(lo))
                i1 = self._nearest_index(old_x, float(hi))
                carry = (self._measure_layer_id, i0, max(i0, i1))
        self._prev_doc_ids = frozenset()       # not a toggle: never restore an old-clock range
        self._view_hist = []                   # the trail is in the old unit; rebuild reseeds
        self._view_ptr = 0
        self._measure_xrange = None            # placed below from the carried indices
        self.rebuild()
        if carry is not None and self._measure_region is not None:
            layer_id, i0, i1 = carry
            layer = next((lay for lay in self.session.layers() if lay.layer_id == layer_id), None)
            if layer is not None and self._measure_layer_id == layer_id:
                new_x = self.session.x_axis(layer.doc_id, layer.time_offset_s).values
                if len(new_x):
                    x0, x1 = float(new_x[min(i0, len(new_x) - 1)]), float(new_x[min(i1, len(new_x) - 1)])
                    if x1 <= x0:
                        x1 = x0 + (float(new_x[1] - new_x[0]) if len(new_x) > 1 else 1.0)
                    self._measure_region.setRegion((x0, x1))   # emits: the panel re-measures

    # -- construction ----------------------------------------------------------------------------
    def rebuild(self) -> None:
        """Recompute panes via plan_panes() and recreate them. Capture the anchor's x-range first
        and restore it after, so a layer toggle does not throw away the user's zoom. Pane 0 is the
        anchor; every other pane setXLink(anchor). Only the bottom pane shows its x-axis values and
        a label (the honest XAxis.unit, "s" or "sample" — never a fabricated "Time (s)")."""
        # Preserve the user's x-zoom ONLY across a channel toggle (same recordings on screen). When
        # the set of SHOWN documents changes — a file opened or closed, or the viewer switched to
        # another recording — frame the new view fresh instead of inheriting a stale range from
        # the recording that was there before. Note "shown", not "layered": a document switched
        # away from keeps its (now invisible) layers, so counting those would have made a switch
        # between two recordings look like a mere channel toggle and hand the incoming recording
        # the outgoing one's zoom window — which, on a shorter recording, is off the end of it.
        doc_ids = self.session.shown_doc_ids()
        live_layers = {layer.layer_id for layer in self.session.layers()}
        self._residual_cache.retain(live_layers)
        for layer_id in tuple(self._subtracted):
            if layer_id not in live_layers:
                self._subtracted.pop(layer_id, None)
        prev_x: tuple[float, float] | None = None
        if self._panes and doc_ids == self._prev_doc_ids:
            prev_x = tuple(self._panes[0].plot.vb.viewRange()[0])  # type: ignore[assignment]
        self._prev_doc_ids = doc_ids

        # A draw in flight dies with the panes it was drawn on. Its token is cleared HERE, before
        # clear(): the scene goes on delivering the drag to the removed ViewBox until the button
        # comes up, and every one of those frames must find the gesture dead. The mask's state is
        # put back to what it was before the gesture, so the re-attach below cannot resurrect the
        # half-drawn span (or a seeded mask where there was none).
        draw_aborted = self._abort_draw_for_rebuild()

        self.clear()
        self._panes = []
        # The rendered items and their plots died in clear(); the REQUESTS behind them
        # (`_overlay_reqs`) deliberately do not, and are re-rendered onto the new panes
        # at the end of this method (`_reattach_overlays`).
        self._overlays.clear()
        self._overlay_plots.clear()
        # Died with its plot in clear(), like the measure region below — and, like the
        # measure region, re-applied at the end of this method from the request that
        # OUTLIVES the pane (`_reattach_filter_preview`). This comment used to say "the
        # panel re-applies", which was simply false: the owner connects to none of the
        # signals that reach `rebuild` (layersChanged, viewModeChanged), so View ▸
        # Stacked/Overlay and every channel show/hide dropped the curve while the
        # "show processed trace on the plot" checkbox stayed ticked, with nothing drawn
        # and nothing said.
        self._filter_preview = None
        self._filter_preview_plot = None
        self._measure_region = None          # died with its plot in clear(); reattached below if active
        self._measure_plot = None
        self._target_label = None            # a child of the old ViewBox: gone with it
        self._placeholder = None

        specs = plan_panes(self.session)
        if not specs:
            self._placeholder = self.addLabel(
                "Open a recording  ·  File ▸ Open…  or drag a file here", color="#5f6874")
            self.crosshair.attach([])
            # Nothing is layered at all, so a standing preview request cannot be drawn —
            # and this exit must say so as loudly as the one below, or the placeholder
            # screen becomes the one place the ticked checkbox goes quiet again.
            self._reattach_filter_preview()
            self._finish_rebuild(draw_aborted)
            return

        anchor: pg.PlotItem | None = None
        x_unit = "s"
        x_offset: float | None = None
        for row, spec in enumerate(specs):
            # Both axes are ReadableLinearAxis (gui/axes.py), not pyqtgraph's default:
            # the default's "%g" labels collapse to SIX IDENTICAL "86400"s on a zoomed
            # window a day into a recording (precision from magnitude, NE-05) and mix
            # "800000" with "1e+06" on one 10⁶ pA autorange (two notations, NE-06).
            # The replacement derives precision from the tick SPACING and picks one
            # notation for the whole axis — each pane gets its own instances, because
            # pyqtgraph axis items are owned by exactly one PlotItem.
            plot = self.addPlot(row=row, col=0,
                                viewBox=TraceViewBox(
                                    on_double_click=self._request_measure_placement,
                                    on_draw=self._on_draw_event, draw_mode=self.draw_mode,
                                    on_catch=self._request_catch,
                                    on_hover_enter=self._on_pane_hover_enter,
                                    scroll=self._scroll_gesture, extent=self.full_x_extent),
                                axisItems={"bottom": ReadableLinearAxis(orientation="bottom"),
                                           "left": ReadableLinearAxis(orientation="left")})
            self.ci.layout.setRowStretchFactor(
                row, _PANE_STRETCH_CURRENT if spec.kind is ChannelKind.CURRENT
                else _PANE_STRETCH_OTHER)
            plot.vb.setMouseMode(self._drag_mode)   # box-zoom (or pan) + vertical wheel-pan
            # No pyqtgraph context menu on the trace panes — same call the navigator and
            # the subpop embedding already make, and here it is load-bearing: the default
            # menu's Transforms ▸ Log X flips the pane into log mode, where the level
            # overlay's pixel floor (StepOverlayItem, linear data units) goes sub-pixel
            # again — the exact invisibility it exists to prevent — and Log Y on a
            # negative-current trace blanks the curve outright. Every gesture the menu
            # offered that the app means to support (zoom reset, export, drag mode) has
            # a first-class home in the View menu / Scan strip already.
            plot.setMenuEnabled(False)
            pane = ChannelPane(plot, spec, self.session,
                               baselines=self._subtracted, residual_cache=self._residual_cache)
            self._panes.append(pane)

            if anchor is None:
                anchor = plot
                first = pane.first_layer()
                if first is not None:
                    x_unit = self.session.x_axis(first.doc_id, first.time_offset_s).unit
                    x_offset = clock.clock_offset_s(
                        self.session.document(first.doc_id).meta, first.time_offset_s)
            else:
                plot.setXLink(anchor)

            # Only the bottom pane shows tick values + the x label; the rest suppress values so the
            # x-linked stack reads as one time span with a single axis at the bottom.
            is_bottom = row == len(specs) - 1
            axis = plot.getAxis("bottom")
            axis.setStyle(showValues=is_bottom)
            # THE TICKS ARE IN THE UNIT THE LABEL CLAIMS. pyqtgraph's automatic SI
            # prefix multiplies every tick VALUE and appends "(x0.001)" to the label the
            # moment the visible span drops below 1 -- so zooming into a blockade turned
            # this axis into "s (x0.001)" reading 884.3 for 0.8843 s, an invitation to a
            # 1000x error on a surface whose whole job is telling the user WHERE in the
            # recording they are. The dwell histogram and the PSD panel carry the same
            # fix; here it also has to be off before the clock note below is attached,
            # because a label pyqtgraph rewrites is a label that loses the note.
            axis.enableAutoSIPrefix(False)
            # ...and it says WHICH CLOCK it is on. See nanoscope.core.clock: this axis is
            # the ONLY surface in the app drawn on the instrument clock, and it used to be
            # labelled "s" exactly like the recording-clock numbers everywhere else.
            plot.setLabel(
                "bottom",
                clock.axis_label(x_unit, clock.INSTRUMENT, x_offset) if is_bottom else "")

        # The zoom-out ceiling needs the panes in place (full_x_extent walks them) and must
        # be up BEFORE the framing below, so even a restored range is under the same rule.
        self._apply_x_zoom_ceiling()

        if anchor is not None:
            anchor.vb.sigXRangeChanged.connect(self._emit_visible_range)
            if prev_x is not None:
                anchor.vb.setXRange(prev_x[0], prev_x[1], padding=0)
            else:
                # Fresh load: show the WHOLE trace. autoRange()+clipToView under-shows (it fit only
                # ~half the recording), so set the full data extent explicitly via reset_zoom().
                self.reset_zoom()
            self._emit_visible_range()
            # Connect history recording AFTER the initial framing, so it isn't itself recorded.
            anchor.vb.sigXRangeChanged.connect(self._on_x_changed)
            # The Back/Forward trail survives a rebuild the same way the zoom does, and on the
            # same test: prev_x is non-None exactly when the SHOWN documents are unchanged (a
            # layer toggle, an Overlay/Stacked switch), and then every recorded x-range is still
            # about the recordings on screen — wiping it would leave a still-enabled Zoom Back
            # that silently does nothing. A new/changed document set reseeds: the old trail's
            # framings belong to a view that no longer exists. (A zoom still mid-debounce keeps
            # its pending _settle across the rebuild, so it commits — or is flushed by
            # view_back/_forward — exactly as it would have without the rebuild.)
            if prev_x is None or not self._view_hist:
                self._seed_history()

        self.crosshair.attach(self._panes)
        self._reattach_measure()
        # ...and the other two managed items, on the same rule as the first: a managed
        # item whose REQUEST outlives the pane is re-applied by the view that tore it
        # down, never by a listener that was never told the pane went away.
        self._reattach_overlays()
        self._reattach_filter_preview()
        self._finish_rebuild(draw_aborted)

    def refresh_channel(self, doc_id: str, channel: int) -> None:
        """Calibration changed: re-set data on the affected curves ONLY. MUST NOT rebuild —
        rebuilding would discard the zoom the user is inspecting a blockade at. Re-applying a gain
        to a 15M channel is ~8.5 ms: no reload, no thread. A touched pane also re-derives its
        units surfaces (y-axis label + min-y-span floor, see refresh_units): the curve data are
        now in the calibration's target units, and an axis still labelled in the old ones would
        disagree with the crosshair readout right beside it. When the units DID change, any
        numeric y-window a box-zoom left behind (y-autorange off) is carried through the same
        old->new mapping the data went through (remap_y_window) — the old numbers are
        meaningless in the new unit, and keeping them frames a window the signal is no longer
        in: a blank pane under a valid-looking axis. A same-unit gain change keeps the window
        untouched, exactly as before.

        THE ONE EXCEPTION IS A CHANGE THAT MOVES THE PANES THEMSELVES. A calibration now also
        carries the user's answer to *what is this channel* (``ChannelCalibration.kind``, read
        by :func:`_kind`), and that answer is an input to ``plan_panes`` — it sets the pane's
        rank and, in OVERLAY, which pane the layer is even in. Re-setting curve data cannot
        express "this layer belongs in a different pane now", so a commit that changes the PLAN
        is the one calibration edit that has to rebuild, and only when the LAYOUT actually
        moved: which layers sit in which pane, and in what order. Not "the plan differs" —
        the units are an input to the same plan, and a pane whose units changed while it kept
        exactly its own layers is precisely the case ``refresh_units``/``remap_y_window``
        below handle better than a rebuild can (they carry the box-zoomed y-window through
        the unit change; a rebuild would throw it away). ``rebuild`` preserves the x-zoom
        across a same-documents rebuild, so the blockade the user is inside stays framed.

        THE TEST IS THE KIND, NOT THE LAYOUT. Comparing whole layouts looks equivalent and
        is not: in OVERLAY the units are part of the GROUPING key, so a plain pA -> nA edit
        splits one pane into two and moves the layout — and rebuilding on that is the very
        thing the paragraph above says not to do (it discards the remapped y-window, and
        the pane object the caller is holding). Asking the narrower question the exception
        is actually about — did THIS channel's kind stop matching the pane it is sitting
        in? — rebuilds for the override and nothing else."""
        self._residual_cache.invalidate(f"{doc_id}#{int(channel)}")
        if self._kind_moved(doc_id, channel):
            self.rebuild()
            return
        for pane in self._panes:
            touched_layers = []
            for layer_id in pane.spec.layer_ids:
                layer = self.session.layer(layer_id)
                if layer is not None and layer.doc_id == doc_id and layer.channel == channel:
                    touched_layers.append(layer)
            if not touched_layers:
                continue
            # Snapshot ONE touched curve's y-data BEFORE re-setting it: paired with the
            # same curve's data after, it yields the exact old->new affine map for the
            # y-window remap (the old calibration itself is already gone from Session).
            ref_id = touched_layers[0].layer_id
            y_before = pane.layer_y_data(ref_id)
            for layer in touched_layers:
                pane.set_layer_data(layer)
            if pane.refresh_units():
                pane.remap_y_window(y_before, pane.layer_y_data(ref_id))
        # Refresh the crosshair readout in place (its y-value may have moved with the new gain).
        self.crosshair.refresh()

    def _kind_moved(self, doc_id: str, channel: int) -> bool:
        """Did this channel's kind stop agreeing with the pane it is currently in?

        The only calibration edit a no-rebuild refresh cannot express. ``_kind`` now reads
        the user's ``ChannelCalibration.kind`` override, and that feeds both the pane RANK
        and — in OVERLAY — which pane the layer is grouped into, neither of which
        ``set_layer_data`` can move.

        Answered against the panes ON SCREEN rather than by re-planning, so it cannot be
        fooled by the other inputs to the same plan: a units edit regroups OVERLAY panes
        too, and that case is handled in place, better, by ``refresh_units`` +
        ``remap_y_window``. False when the channel has no pane (nothing to move).
        """
        want = _kind(self.session, doc_id, channel)
        for pane in self._panes:
            for layer_id in pane.spec.layer_ids:
                layer = self.session.layer(layer_id)
                if layer is not None and layer.doc_id == doc_id and layer.channel == channel:
                    if pane.spec.kind is not want:
                        return True
        return False

    def full_x_extent(self) -> tuple[float, float] | None:
        """True [t0, t_end] across all visible layers, straight from the data — never from the
        viewbox (a clipToView curve mis-reports its bounds when deeply zoomed)."""
        lo = hi = None
        for pane in self._panes:
            for layer_id in pane.spec.layer_ids:
                layer = self.session.layer(layer_id)
                if layer is None:
                    continue
                vals = self.session.x_axis(layer.doc_id, layer.time_offset_s).values
                if len(vals) == 0:
                    continue
                x0, x1 = float(vals[0]), float(vals[-1])
                lo = x0 if lo is None else min(lo, x0)
                hi = x1 if hi is None else max(hi, x1)
        return (lo, hi) if lo is not None and hi is not None and hi > lo else None

    def _apply_x_zoom_ceiling(self) -> None:
        """Clamp every pane's widest reachable x-span to ``MAX_X_SPAN_FACTOR`` x the data
        extent (see that constant for the measured failure this prevents). The mirror of
        the ``minYRange`` floor in :meth:`ChannelPane.apply_y_policy`, and re-derived the
        same way: called from every :meth:`rebuild` (the panes — and their ViewBoxes,
        where the limit lives — are recreated there, so a stale ceiling cannot survive a
        data change). Only the SPAN is limited, never the position: panning past the end
        of the recording is a legitimate, reversible gesture that other surfaces already
        answer honestly (the "view is panned off the recording" run refusal), and an
        ``xMin``/``xMax`` clamp would silently forbid it. ``setLimits`` updates only the
        keys it is handed, so this coexists with the per-pane ``minYRange`` floor.
        With no drawable extent (no layers, or a degenerate single-point one) the new
        ViewBoxes simply keep pyqtgraph's default of no limit — there is no data whose
        rendering the ceiling would be protecting."""
        extent = self.full_x_extent()
        if extent is None:
            return
        max_span = (extent[1] - extent[0]) * MAX_X_SPAN_FACTOR
        for pane in self._panes:
            pane.plot.vb.setLimits(maxXRange=max_span)

    def reset_zoom(self) -> None:
        """Zoom all the way back out to the full trace. Sets x from the true data extent rather than
        autoRange(), because setClipToView(True) makes a deeply-zoomed curve report bounds only within
        the current window — so autoRange() alone cannot expand back out (it would barely move). Every
        pane's x-autorange is disabled first so a linked pane can't re-fit x to its clipped bounds."""
        anchor = self.anchor_plot()
        extent = self.full_x_extent()
        if anchor is not None and extent is not None:
            for pane in self._panes:
                pane.plot.vb.disableAutoRange(axis=pg.ViewBox.XAxis)
            anchor.vb.setXRange(extent[0], extent[1], padding=0)
            for pane in self._panes:
                pane.plot.vb.enableAutoRange(y=True)
            return
        for pane in self._panes:   # fallback (no layers/x-axis): best-effort autorange
            pane.plot.vb.enableAutoRange(x=True, y=True)
            pane.plot.vb.autoRange()

    def _request_measure_placement(self, x: float) -> None:
        """Relay a trace double-click as a data-x annotation request.

        The view has no concept of the currently selected annotation channel, so it emits rather
        than placing the region itself. A finite guard keeps malformed synthetic/scene events from
        arming a mask at NaN or infinity.
        """
        x = float(x)
        if np.isfinite(x):
            self.sigMeasurePlacementRequested.emit(x)

    def set_drag_mode(self, box_zoom: bool) -> None:
        """True -> left-drag draws a zoom box (dive into events); False -> left-drag pans.
        Applies to every current pane and is remembered for panes built on the next rebuild."""
        self._drag_mode = pg.ViewBox.RectMode if box_zoom else pg.ViewBox.PanMode
        for pane in self._panes:
            pane.plot.vb.setMouseMode(self._drag_mode)

    def zoom_x_to(self, xmin: float, xmax: float) -> None:
        """Zoom the linked x-axis to [xmin, xmax] and autorange y to fit — used by 'zoom to cursor
        window' and any programmatic dive-in."""
        anchor = self.anchor_plot()
        if anchor is not None and xmax > xmin:
            anchor.vb.setXRange(xmin, xmax, padding=0)
            for pane in self._panes:
                pane.plot.vb.enableAutoRange(y=True)

    #: *** THE DIVE-IN PAD, AND THE GALLERY TILE'S PAD, ARE ONE NUMBER. *** Half the span on each
    #: side, so the thing jumped to occupies HALF the picture. It was 1.5 — a 4x-span window, with
    #: the event 25% of the pane — while ``event_gallery._PAD_FRACTION`` has always been 0.5, and
    #: the owner reported the consequence as *"the mask on the gallery does not match the mask on
    #: the main panel"*: one event, clicked in the gallery and landed on the trace, is drawn at
    #: two different sizes on two screens that are meant to be the same event. The tile is the one
    #: that is right — it is the picture the curation decision is actually made from, and it was
    #: sized against real events — so the trace matches the tile rather than the other way round.
    #: Half a span of open pore either side is still context: the framing this replaces was chosen
    #: for legibility, and 0.5 keeps that property while making the two surfaces agree.
    ZOOM_PAD_FRACTION = 0.5

    def zoom_to_span(self, x0: float, x1: float, *,
                     pad_frac: float | None = None, min_pad: float = 0.0) -> None:
        """*** THE dive-in framing. *** Zoom to the span [x0, x1] with SYMMETRIC padding, so the
        thing you jumped to sits in the middle of some context instead of filling the pane edge to
        edge (a blockade with no open pore either side is unreadable).

        The padding is ``max(width*pad_frac, min_pad)`` on each side: proportional for a normal
        event, with ``min_pad`` as the floor that keeps a one-sample span from zooming to nothing.
        ``pad_frac`` defaults to :data:`ZOOM_PAD_FRACTION`, which is the gallery tile's own pad —
        see the note on it for why the two have to be the same number.

        ONE implementation, shared by every 'take me to that span' gesture — the event table's
        zoom-to-event and the manual-measurement table's return-to-measurement both call it, so the
        two can never drift apart. Neither passes ``pad_frac``, so both take the shared default;
        an explicit 0.0 is honoured (it is a number, not a missing argument)."""
        if pad_frac is None:
            pad_frac = self.ZOOM_PAD_FRACTION
        lo, hi = (float(x0), float(x1)) if x0 <= x1 else (float(x1), float(x0))
        pad = max((hi - lo) * float(pad_frac), float(min_pad))
        if pad <= 0.0 and hi <= lo:
            return
        self.zoom_x_to(lo - pad, hi + pad)

    def layer_time_offset(self, doc_id: str, channel: int) -> float:
        """The user's alignment nudge for (doc_id, channel), in seconds — 0.0 when that channel
        has no layer.

        THE one place this lookup lives. Every on-screen x for a channel is the document's
        native axis shifted by this number, so anything converting between an x the user can
        see (a view range, a cursor, a click) and a SAMPLE INDEX has to go through it. Reading
        the native axis instead silently answers about a different stretch of the recording —
        which is exactly what "detect over the visible range" did on an aligned layer."""
        layer = self.session.layer(f"{doc_id}#{channel}")
        return float(layer.time_offset_s) if layer is not None else 0.0

    def x_axis_for(self, doc_id: str, channel: int):
        """The :class:`XAxis` this channel is actually PLOTTED against, i.e. the document's axis
        through :meth:`layer_time_offset`. The axis every visible x for this channel lives in."""
        return self.session.x_axis(doc_id, self.layer_time_offset(doc_id, channel))

    def clock_offset_for(self, doc_id: str, channel: int) -> float | None:
        """``instrument_seconds - recording_seconds`` for what this view DRAWS.

        Public because MainWindow -- the sole wirer -- is where a selection made on
        this (instrument-clock) axis is handed to a panel that wants recording
        seconds, and the conversion must happen in code with the same offset the
        curve was drawn through, ``t0_s + time_offset_s``. None when the document has
        no sample rate, i.e. the axis is samples and neither clock exists.
        """
        return clock.clock_offset_s(self.session.document(doc_id).meta,
                                    self.layer_time_offset(doc_id, channel))

    def x_axis_label(self) -> str:
        """The text currently on the bottom axis of the bottom pane ("" when empty).

        The clock note lives IN that label, so this is how a caller (or a test) reads
        back what the axis is telling the user about its own clock.
        """
        if not self._panes:
            return ""
        item = self._panes[-1].plot.getAxis("bottom")
        return str(item.labelText or "")

    def sample_dx(self, doc_id: str, channel: int = 0) -> float:
        """The x-axis step between two adjacent samples of (doc_id, channel), or 0.0 when the
        document has fewer than two samples. The natural unit for a minimum zoom pad."""
        vals = self.x_axis_for(doc_id, channel).values
        return float(vals[1] - vals[0]) if vals.size > 1 else 0.0

    def x_for_index(self, doc_id: str, channel: int, index: int) -> float | None:
        """The data-x of sample ``index`` on (doc_id, channel), through the SAME layer time offset
        the region->index mapping uses (:meth:`_x_to_index`).

        This is the exact inverse of ``measure_bounds()``: place the region with these values and
        reading the bounds back gives the index you started from. Using the document's native axis
        instead (offset 0) puts the cursors ``time_offset_s`` away from the sample the caller asked
        for — silently measuring the wrong span on any aligned layer. None when the document is
        gone or empty."""
        if not self.session.has_document(doc_id):
            return None
        vals = self.x_axis_for(doc_id, channel).values
        if vals.size == 0:
            return None
        return float(vals[int(np.clip(index, 0, vals.size - 1))])

    # -- zoom history (Back / Forward) -----------------------------------------------------------
    @staticmethod
    def _same_range(a: tuple[float, float], b: tuple[float, float]) -> bool:
        span = max(abs(a[1] - a[0]), abs(b[1] - b[0]), 1e-12)
        return abs(a[0] - b[0]) < span * 1e-4 and abs(a[1] - b[1]) < span * 1e-4

    def _seed_history(self) -> None:
        anchor = self.anchor_plot()
        if anchor is None:
            return
        self._view_hist = [tuple(anchor.vb.viewRange()[0])]  # type: ignore[list-item]
        self._view_ptr = 0

    def _on_x_changed(self, *_args) -> None:
        # Debounce: only record once the view stops moving (one entry per settled zoom, not per frame).
        if not self._restoring:
            self._settle.start()

    def _commit_view(self) -> None:
        anchor = self.anchor_plot()
        if anchor is None:
            return
        cur = tuple(anchor.vb.viewRange()[0])
        if self._view_hist and self._same_range(cur, self._view_hist[self._view_ptr]):
            return
        del self._view_hist[self._view_ptr + 1:]      # a new zoom prunes any forward branch
        self._view_hist.append(cur)                    # type: ignore[arg-type]
        self._view_ptr = len(self._view_hist) - 1

    def _restore_view(self, rng: tuple[float, float]) -> None:
        anchor = self.anchor_plot()
        if anchor is None:
            return
        self._restoring = True
        anchor.vb.setXRange(rng[0], rng[1], padding=0)
        for pane in self._panes:
            pane.plot.vb.enableAutoRange(y=True)
        self._restoring = False

    def can_go_back(self) -> bool:
        return self._view_ptr > 0

    def can_go_forward(self) -> bool:
        return 0 <= self._view_ptr < len(self._view_hist) - 1

    def _commit_pending(self) -> None:
        """Flush an in-flight (still-debouncing) zoom into the history NOW. The 250 ms settle is
        a recording optimisation, not a semantic one: the framing on screen when the user presses
        Back/Forward is real, and discarding it made a quick Back land TWO framings back — and
        silently dropped the dive from history, so Forward could never return to it."""
        if self._settle.isActive():
            self._settle.stop()
            self._commit_view()

    def view_back(self) -> None:
        """Step back to the previous settled view — the framing you had before you dove in."""
        self._commit_pending()   # the view mid-flight is the framing Back steps back FROM
        if self.can_go_back():
            self._view_ptr -= 1
            self._restore_view(self._view_hist[self._view_ptr])

    def view_forward(self) -> None:
        # Same flush: an uncommitted zoom prunes the forward branch exactly as it would have
        # 250 ms later, so a quick Forward cannot behave differently from a slow one (restoring
        # a branch the user's own new zoom had already invalidated).
        self._commit_pending()
        if self.can_go_forward():
            self._view_ptr += 1
            self._restore_view(self._view_hist[self._view_ptr])

    # -- accessors -------------------------------------------------------------------------------
    def anchor_plot(self) -> pg.PlotItem | None:
        return self._panes[0].plot if self._panes else None

    def visible_x_range_for(self, doc_id: str, channel: int) -> tuple[float, float] | None:
        """The visible x window OF THE PANE THAT DRAWS (doc_id, channel), or None when
        nothing draws it.

        THE point is the None. Every pane is ``setXLink``ed to the anchor, so when the
        channel IS drawn this returns the same numbers ``anchor_plot().vb.viewRange()[0]``
        does — the difference is entirely that this one refuses to answer about a recording
        that is not on screen, and the old code could not tell the two cases apart.

        That is the third time an offset/target mix-up in this app has produced "analysed a
        different stretch than displayed", and the first two were fixed one call site at a
        time. Measured on the restore that motivated this: with ``short_b.dat`` (4,000
        samples) drawn and ``long_a.dat`` (40,000) selected, "Detect over the visible view"
        read the anchor's x window, searched it into the TARGET's axis, and resolved to
        samples (0, 4,000) of 40,000 — the first 10 % of a recording the user was not
        looking at, reported as "the visible range". With no answer available, the caller
        can say so; with a plausible wrong one, it cannot.

        (:meth:`layer_time_offset` handles the other half — an x the user can see is the
        document's native axis SHIFTED — and is applied by the caller through
        :meth:`x_axis_for`, which is the axis these numbers live in.)
        """
        pane = self.pane_for_layer(f"{doc_id}#{int(channel)}")
        if pane is None:
            return None
        layer = self.session.layer(f"{doc_id}#{int(channel)}")
        if layer is not None and not layer.visible:
            return None
        xmin, xmax = pane.plot.vb.viewRange()[0]
        return float(xmin), float(xmax)

    def panes(self) -> tuple[ChannelPane, ...]:
        return tuple(self._panes)

    def pane_for_layer(self, layer_id: str) -> ChannelPane | None:
        for pane in self._panes:
            if layer_id in pane.spec.layer_ids:
                return pane
        return None

    def _emit_visible_range(self, *_args) -> None:
        anchor = self.anchor_plot()
        if anchor is not None:
            xmin, xmax = anchor.vb.viewRange()[0]
            self.sigVisibleRangeChanged.emit(float(xmin), float(xmax))

    def set_x_range(self, xmin: float, xmax: float) -> None:
        anchor = self.anchor_plot()
        if anchor is not None:
            anchor.vb.setXRange(xmin, xmax, padding=0)

    # -- v2 overlay seam -------------------------------------------------------------------------
    def add_overlay(self, doc_id: str, channel: int, hint: OverlayHint) -> str:
        """v2 SEAM. Render an OverlayHint onto the pane showing (doc_id, channel), dispatching by
        hint.kind through the renderer registry. v1 has NO callers and NO registered renderers; the
        plumbing (locate pane, track ids, dispatch, remove) exists so v2 drops in a renderer and
        nothing else changes."""
        overlay_id = f"{doc_id}#{channel}:{hint.kind.value}:{next(_overlay_seq)}"
        # THE REQUEST, kept for exactly as long as the id is (see `_overlay_reqs`): an
        # overlay is a standing annotation on a channel, and the pane it is drawn on is
        # recreated by every rebuild. Recorded here and re-rendered there, so the id a
        # caller is holding keeps naming something on the plot.
        self._overlay_reqs[overlay_id] = (doc_id, channel, hint)
        try:
            self._draw_overlay(overlay_id)
        except NotImplementedError:
            # An unrenderable kind is a programming error, not a standing annotation:
            # the caller never receives this id, so nothing may keep asking for it.
            self._overlay_reqs.pop(overlay_id, None)
            raise
        return overlay_id

    def _draw_overlay(self, overlay_id: str) -> bool:
        """Render one recorded overlay request onto the pane that shows its channel.

        THE ONE PLACE AN OVERLAY IS RENDERED, so the first call and every rebuild's
        re-application take the identical path. Returns whether it is on the plot;
        False means the channel is not layered right now (hidden in the File Manager,
        or its document was closed) and the request is simply waiting.
        """
        req = self._overlay_reqs.get(overlay_id)
        if req is None:
            return False
        doc_id, channel, hint = req
        # THE RENDERER IS CHECKED BEFORE THE PANE, which is the one behaviour this
        # refactor deliberately changes: an unrenderable kind now raises whether or not
        # the channel happens to be on screen. It used to depend on visibility (the
        # pane-is-None exit returned first), which meant a request that can NEVER be
        # drawn could be stored — and the rebuild that later gave it a pane would raise
        # from inside `rebuild`, taking the whole viewer down with it. Refused at the
        # door instead, so re-application is total.
        renderer = _OVERLAY_RENDERERS.get(hint.kind)
        if renderer is None:
            raise NotImplementedError(
                f"no renderer registered for {hint.kind}; register one via "
                f"trace_view.register_overlay_renderer (this is the v2 rendering seam).")
        pane = self.pane_for_layer(f"{doc_id}#{channel}")
        if pane is None:
            return False
        # The renderer maps int64 SAMPLE INDICES onto x, and the curve beside it is drawn
        # through the layer's alignment offset — so the offset has to travel with the hint or
        # every span/marker lands `time_offset_s` away from the event it annotates. Passed
        # keyword-only with a 0.0 default, so a renderer written against the original
        # four-argument signature is unaffected.
        items = renderer(pane.plot, self.session, doc_id, hint,
                         time_offset_s=self.layer_time_offset(doc_id, channel))
        self._overlays[overlay_id] = items
        self._overlay_plots[overlay_id] = pane.plot
        return True

    def _reattach_overlays(self) -> None:
        """After a rebuild: re-render every standing overlay request on the new panes.

        The third managed item under the same rule as the measure region and the filter
        preview — **a managed item torn down by a rebuild is put back by the rebuild.**
        Without it, View ▸ Stacked/Overlay or any channel show/hide left the event SPANS
        (and the baseline track, and the selected-event marker) off the trace while the
        owner still held live-looking ids for them: measured, one detection of 20 events,
        one SpanOverlayItem on the pane before the switch and zero after, with
        ``AnalysisPanel._overlay_ids`` still naming it. The events stayed in the table,
        so the trace and the table disagreed about which stretches are events.
        """
        for overlay_id in list(self._overlay_reqs):
            self._draw_overlay(overlay_id)

    def remove_overlay(self, overlay_id: str) -> None:
        """Detach the overlay's items from their pane, and END the request.

        Uses ``PlotItem.removeItem`` so the item leaves both the plot's item list AND the
        scene (a bare ``scene.removeItem`` would strand a dead reference in
        ``plot.items``). Dropping the request is what makes this the opposite of
        :meth:`add_overlay`: a rebuild must never resurrect an overlay its owner removed."""
        self._overlay_reqs.pop(overlay_id, None)
        plot = self._overlay_plots.pop(overlay_id, None)
        for item in self._overlays.pop(overlay_id, []):
            if plot is not None:
                try:
                    plot.removeItem(item)          # removes from plot.items AND the scene
                    continue
                except (RuntimeError, ValueError):
                    pass                            # plot torn down / item already gone
            scene = item.scene()
            if scene is not None:
                scene.removeItem(item)

    # -- the baseline made visible (owner: "processing should be visible") -----------------------
    #
    # TWO CONTROLS, ONE MEASUREMENT, AND THEY COMPOSE. `set_baseline_track` draws the
    # estimator's answer ON the raw trace ("here is the level your depths are measured
    # against"); `set_baseline_subtracted` redraws the trace MINUS that answer ("here is
    # what is left once it is taken away"). Ticking both is not a contradiction: the track
    # then lies flat at zero, which is the picture that says the subtraction did what it
    # claimed.
    #
    # WHY THE RESIDUAL IS A MODE ON THIS PANE AND NOT A PANE OF ITS OWN. The reason a
    # subtracted view is worth having at all is that a -566 pA recording and a -1000 pA one
    # cannot otherwise be read on ONE y axis; a second pane gives them a second axis and
    # loses exactly that. It also costs vertical space the owner's 900 px display does not
    # have (the trace already shares it with a dock and a table), and it would double the
    # panes on a stacked four-run comparison. The one real hazard of a mode — forgetting you
    # are in it — is answered by `ChannelPane.refresh_y_label`, which puts "Δ vs baseline"
    # on the axis for as long as the mode is on. A mode you can read off the screen is not
    # a hidden mode.
    def set_baseline_track(self, doc_id: str, channel: int,
                           idx: np.ndarray | None, values: np.ndarray | None,
                           *, color: str | None = None) -> bool:
        """Show the estimated baseline on (doc_id, channel) as ONE cased dashed rail.

        ``idx``/``values`` are `analysis.baseline.baseline_track`'s knots. Passing
        ``values=None`` removes the track. Idempotent per layer: a second call replaces
        the standing rail rather than laying another on top of it, which is what makes
        this safe to call from a checkbox that also fires on re-selection.

        WHEN THE LAYER IS IN THE SUBTRACTED VIEW THE TRACK IS SUBTRACTED TOO, so it lies
        along zero instead of hanging 566 pA off the top of the pane where the residual
        cannot see it. The producer therefore always hands over the SAME numbers — the
        baseline as measured — and the view, which is the only thing that knows what its
        own y axis currently means, puts them where the curve is.

        Returns whether the rail is on the plot now (False = its channel is not layered).
        """
        layer_id = f"{doc_id}#{int(channel)}"
        if values is None or idx is None:
            old = self._baseline_tracks.pop(layer_id, None)
            if old is not None:
                self.remove_overlay(old[0])
            return False
        starts = np.ascontiguousarray(np.asarray(idx, dtype=np.int64))
        absolute = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        if color is None:
            color = BASELINE_TRACK_INK
        return self._draw_baseline_track(doc_id, int(channel), starts, absolute, str(color))

    def _draw_baseline_track(self, doc_id: str, channel: int, starts: np.ndarray,
                             absolute: np.ndarray, color: str) -> bool:
        """THE ONE PLACE THE RAIL IS ISSUED, so a first tick and a later mode change take
        the identical path — including the shift onto whatever the y axis now means."""
        layer_id = f"{doc_id}#{channel}"
        # THE REPLACEMENT IS HERE, not at the two call sites, because both of them
        # replace: a re-tick and a mode change each end with exactly one rail on the
        # layer. Doing it in `set_baseline_track` alone left the mode change stacking a
        # second rail on the first, which is invisible until the two disagree.
        previous = self._baseline_tracks.pop(layer_id, None)
        if previous is not None:
            self.remove_overlay(previous[0])
        vals = absolute
        base = self._subtracted.get(layer_id)
        if base is not None and base.size:
            b = np.asarray(base, dtype=np.float64)
            vals = absolute - b[np.clip(starts, 0, b.size - 1)]
        hint = OverlayHint(kind=OverlayKind.HLINE, starts=starts, values=vals,
                           label="baseline", color=color, name="estimated baseline")
        overlay_id = self.add_overlay(doc_id, channel, hint)
        self._baseline_tracks[layer_id] = (overlay_id, starts, absolute, color)
        return overlay_id in self._overlays

    def has_baseline_track(self, doc_id: str, channel: int) -> bool:
        """Is a baseline rail STANDING on this layer (drawn or waiting for its pane)?"""
        return f"{doc_id}#{int(channel)}" in self._baseline_tracks

    def set_baseline_subtracted(self, doc_id: str, channel: int,
                                baseline: np.ndarray | None) -> bool:
        """Draw (doc_id, channel) as ``signal - baseline``, centred on zero. Or stop.

        ``baseline`` is the full-length estimate (one value per sample, from
        `analysis.baseline.characterisation_baseline`); ``None`` returns the layer to
        its raw samples. A DISPLAY transform and nothing else — `ChannelPane.set_layer_data`
        does the arithmetic on the way to the curve, the Session's array is never touched,
        and no measured column can move because nothing measures a curve.

        THE Y WINDOW IS RE-DERIVED, NOT KEPT. The residual sits at 0 and the raw trace at
        -566 pA, so a pane whose y-autorange the user turned off with a box-zoom would
        keep a window the new curve is nowhere near — a blank pane, which is the same
        "drawn and invisible" failure the overlay's z-order was. Autorange is switched
        back on for the affected pane on every toggle: the exact zoom tightness is lost,
        the signal is on screen. (`remap_y_window` makes the same trade for the same
        reason after a recalibration.)

        Returns whether the pane was redrawn now (False = its channel is not layered;
        the request still stands and the next `rebuild` honours it).

        Raises:
            ValueError: the baseline's length is not the channel's. A residual against
                another recording's baseline is a picture of nothing.
        """
        layer_id = f"{doc_id}#{int(channel)}"
        if baseline is None:
            self._subtracted.pop(layer_id, None)
        else:
            b = np.ascontiguousarray(baseline)
            n = int(np.asarray(self.session.calibrated(doc_id, int(channel)).data).size)
            if b.size != n:
                raise ValueError(
                    f"baseline of {b.size} samples cannot be subtracted from a "
                    f"{n}-sample channel ({doc_id}#{channel})")
            self._subtracted[layer_id] = b
        self._residual_cache.invalidate(layer_id)
        # A STANDING RAIL MOVES WITH THE CURVE IT ANNOTATES. Re-issued from the values as
        # MEASURED, never from the values last drawn, so toggling the mode twice returns
        # the line to exactly where it started instead of subtracting the baseline again.
        standing = self._baseline_tracks.get(layer_id)
        if standing is not None:
            _, starts, absolute, ink = standing
            self._draw_baseline_track(doc_id, int(channel), starts, absolute, ink)
        layer = self.session.layer(layer_id)
        pane = self.pane_for_layer(layer_id)
        if layer is None or pane is None:
            return False
        pane.set_layer_data(layer)
        pane.refresh_y_label()
        pane.plot.vb.enableAutoRange(y=True)
        return True

    def baseline_subtracted_for(self, doc_id: str, channel: int) -> np.ndarray | None:
        """The baseline this layer's curve is currently drawn minus, or None."""
        return self._subtracted.get(f"{doc_id}#{int(channel)}")

    def clear_baseline_views(self) -> None:
        """Take every baseline rail and every residual off every layer.

        THE ONE GESTURE A CHECKBOX OWNER NEEDS, and the reason it exists is a lie a
        checkbox would otherwise tell. These are per-LAYER requests and the box that
        asks for them is per-SELECTION: tick it on the -600 mV run, select the -1000 mV
        run, untick — and the -600 mV rail stays up, claimed by nothing. So the owner
        clears everything and re-applies to the current selection on every change; one
        writer, and the screen can only ever show what the box currently says.
        """
        for layer_id in list(self._baseline_tracks):
            doc_id, _, ch = layer_id.rpartition("#")
            self.set_baseline_track(doc_id, int(ch), None, None)
        for layer_id in list(self._subtracted):
            doc_id, _, ch = layer_id.rpartition("#")
            self.set_baseline_subtracted(doc_id, int(ch), None)

    # -- D4 filtered-trace live preview ----------------------------------------------------------
    def set_filter_preview(self, doc_id: str, channel: int,
                           data: np.ndarray | None) -> bool:
        """Show a low-pass PREVIEW of (doc_id, channel) as a managed curve (PORT_PLAN D4).

        A filtered trace is a full-length float32 SIGNAL, not a span/marker array, so an
        OverlayHint (int64 index arrays) is the wrong vehicle. It is instead a managed
        pg.PlotDataItem on the pane that already shows this channel — peak-downsampled by the
        same MANDATORY four-step order every trace curve uses, so a microsecond feature survives
        decimation — owned here and toggled by the AnalysisPanel. Passing data=None clears it.
        Only ONE preview exists at a time (a second call replaces the first).

        THE CALL PLACES A STANDING REQUEST, not a one-shot draw. The request survives
        every `rebuild` (see `_filter_preview_req`) and is re-applied there, so the
        curve is on the plot for as long as its channel is — which is what the ticked
        checkbox on the Analysis panel claims.

        Returns whether the curve is ON THE PLOT now. False means the request stands but
        cannot be drawn (its channel is not layered); `sigFilterPreviewChanged` has
        already carried the reason to whoever renders it, so a caller may ignore the
        return value without the failure going silent.
        """
        if data is None:
            self.clear_filter_preview()
            return False
        self._filter_preview_req = (str(doc_id), int(channel), np.asarray(data))
        return self._draw_filter_preview()

    def clear_filter_preview(self) -> None:
        """WITHDRAW the request and remove the curve. Safe to call when none exists and safe
        after a rebuild (the refs were dropped when the plot was torn down).

        The withdrawal is the point: this is how the owner says "nobody is asking for a
        preview any more" (the box was unticked, every stage was switched off, the
        selection moved, a stage refused to compute). A rebuild must NOT reach this — it
        destroys the curve, not the request."""
        self._filter_preview_req = None
        self._remove_preview_curve()
        # No request left, so there is nothing for a reason to be about. Every caller
        # that clears because something went wrong puts its own words on the error line
        # immediately after this returns; this only makes sure a reason about a request
        # that no longer exists cannot outlive it.
        self._say_filter_preview("")

    def _remove_preview_curve(self) -> None:
        """Detach the drawn curve, leaving the request alone."""
        curve, plot = self._filter_preview, self._filter_preview_plot
        self._filter_preview = None
        self._filter_preview_plot = None
        if curve is None:
            return
        try:
            if plot is not None:
                plot.removeItem(curve)
            else:
                scene = curve.scene()
                if scene is not None:
                    scene.removeItem(curve)
        except RuntimeError:
            pass   # the underlying C++ item was already destroyed (e.g. a concurrent rebuild)

    def _draw_filter_preview(self) -> bool:
        """Draw the standing request on the pane that shows its channel, or say why not.

        THE ONE PLACE THE PREVIEW CURVE IS CREATED, so the first call and every rebuild's
        re-application take the identical path and cannot drift. It always ends by
        announcing the on-screen state (`_say_filter_preview`): a request that cannot be
        drawn must produce a sentence somewhere, because the surface that asked for it
        goes on claiming it is on.
        """
        self._remove_preview_curve()
        req = self._filter_preview_req
        if req is None:
            self._say_filter_preview("")
            return False
        doc_id, channel, data = req
        pane = self.pane_for_layer(f"{doc_id}#{channel}")
        if pane is None:
            # Channel not currently layered; nothing to draw onto. It used to return here
            # in silence, in a workflow whose whole contract is that a preview it cannot
            # draw says why.
            self._say_filter_preview(PREVIEW_CHANNEL_NOT_SHOWN)
            return False
        layer = self.session.layer(f"{doc_id}#{channel}")
        offset = layer.time_offset_s if layer is not None else 0.0
        xax = self.session.x_axis(doc_id, offset)
        y = np.asarray(data)
        curve = ChannelPane.make_curve(pane.plot, FILTER_PREVIEW_HEX)  # see the constant
        curve.setZValue(5)                                     # over the raw trace, under the crosshair
        curve.setData(x=xax.values, y=y)                       # data LAST (four-step order)
        self._filter_preview = curve
        self._filter_preview_plot = pane.plot
        self._say_filter_preview("")
        return True

    def _reattach_filter_preview(self) -> None:
        """After a rebuild: redraw the standing preview request on the pane that now shows
        its channel, or say why it cannot be drawn. The exact counterpart of
        `_reattach_measure`, and for the same reason — a managed item torn down by a
        rebuild is put back by the rebuild, never by a listener that was never told.

        Unlike the measure region this does NOT deactivate itself when the channel has
        gone: hiding a channel is a reversible view gesture, the request is still the
        user's, and showing the channel again brings the curve straight back. The
        checkbox therefore stays honest either way — drawn, or ticked with a sentence
        on the error line saying what is in the way."""
        if self._filter_preview_req is None:
            return
        self._draw_filter_preview()

    def _say_filter_preview(self, reason: str) -> None:
        """THE ONLY WRITER of the preview's status, text and signal together.

        Emitted on EVERY attempt rather than only on a change: the owner writes the same
        error line for its own reasons (a cutoff above Nyquist, a detrend window longer
        than the recording) and clears it on the way through, so a suppressed "nothing has
        changed" emission is exactly how the line would end up blank under a ticked box
        with no curve — the failure this signal exists to make impossible."""
        self._filter_preview_reason = reason
        self.sigFilterPreviewChanged.emit(reason)

    def has_filter_preview(self) -> bool:
        """Whether the preview curve is ON THE PLOT (not merely requested)."""
        return self._filter_preview is not None

    def filter_preview_requested(self) -> bool:
        """Whether a preview request stands, drawn or not."""
        return self._filter_preview_req is not None

    def filter_preview_reason(self) -> str:
        """Why the standing request is not drawn ("" when it is, or when there is none)."""
        return self._filter_preview_reason

    # -- manual measurement region (the Measure tool) --------------------------------------------
    def set_measure_active(self, doc_id: str, channel: int, on: bool, *, seed: bool = True) -> None:
        """Show/hide the manual-selection region on the (doc_id, channel) pane.

        When ``on``: create a single movable LinearRegionItem on that pane (added
        ignoreBounds=True so it never drives y-autorange), seeded to the middle ~20% of the current
        x-view, and start emitting sigMeasureChanged as it moves (plus sigMeasureFinished once it
        comes to rest — see those signals). When ``off``: remove it and forget
        the selection. Idempotent; switching channel just re-targets and reseeds. The region is
        rebuild-safe: it is torn down with its plot on rebuild and recreated by _reattach_measure()
        as long as the same channel is still layered.

        ``seed=False`` turns Measure on WITHOUT a mask: the empty state (see
        :meth:`measure_empty`). Nothing is seeded and nothing is emitted but the activation; the
        mask appears when something gives it a range (:meth:`set_measure_region`, a draw). A mask
        already standing on this very target is left alone; a stored range from before the
        activation is put back; a mask on another target is removed (``sigMeasureCleared``).

        ``sigMeasureActiveChanged`` reports whether Measure is on AND its target pane exists. It
        used to read "is there a region", which would uncheck Measure the moment the mask was
        discarded."""
        layer_id = f"{doc_id}#{channel}"
        if on:
            reseed = layer_id != self._measure_layer_id  # a new target seeds fresh, not stale x
            self._measure_active = True
            self._measure_layer_id = layer_id
            if reseed:
                self._measure_xrange = None
                self._abandon_draw()          # the gesture was on the old target's pane
            if seed:
                self._measure_empty = False
                self._create_measure_region(seed=self._measure_xrange is None)
            elif self._measure_region is not None and not reseed:
                pass                          # already on, with a mask: nothing to do
            elif self._measure_xrange is not None and not reseed:
                self._measure_empty = False
                self._create_measure_region(seed=False)   # a range stored before: put it back
            else:
                had_mask = self._measure_region is not None
                self._remove_measure_region()
                self._measure_empty = True
                if had_mask:
                    self.sigMeasureCleared.emit()
        else:
            self._abandon_draw()
            self._measure_active = False
            self._measure_empty = False
            self._measure_layer_id = None
            self._measure_xrange = None
            self._remove_measure_region()
        self.sigMeasureActiveChanged.emit(
            bool(on and self.pane_for_layer(layer_id) is not None))
        self._sync_draw_mode()
        self._refresh_pane_decorations()

    def measure_bounds(self) -> tuple[int, int] | None:
        """Current selection as half-open sample indices (start_idx, stop_idx), or None when the
        Measure tool is inactive / unmapped, or ON WITH NO MASK (:meth:`measure_empty`). Edges
        map through the target layer's x-axis and are clamped to [0, n].

        INDICES, NOT SECONDS, and that is the point: this view's x-axis is on the
        instrument clock and most panels' time inputs are on the recording clock (see
        :mod:`nanoscope.core.clock`). A sample index is on neither, so handing one across
        a surface boundary cannot be misread — which is why every consumer of a
        selection takes it from here rather than from the numbers on the axis.
        """
        return self._measure_indices()

    def measure_target(self) -> tuple[str, int] | None:
        """``(doc_id, channel)`` the Measure region currently sits on, or None.

        A caller converting :meth:`measure_bounds` into that recording's own seconds
        needs to know WHICH recording, and the region does not have to be on the
        recording the app has selected. Still the target while Measure is on with no mask:
        pair it with :meth:`measure_bounds`, which is None then.
        """
        layer_id = self._measure_layer_id
        if not self._measure_active or layer_id is None:
            return None
        doc_id, _, channel = layer_id.rpartition("#")
        if not doc_id or not channel.isdigit():
            return None
        return doc_id, int(channel)

    def set_measure_region(self, x0: float, x1: float) -> None:
        """Place the region programmatically in x (data) coords — e.g. snap to a detected event or
        the current view. No-op when Measure is off (the range is still recorded so a later
        rebuild reattaches at the same place). With Measure ON and no mask (the empty state) it
        CREATES the mask at exactly this range: no seed, and no seeded emission before it."""
        lo, hi = (float(x0), float(x1)) if x0 <= x1 else (float(x1), float(x0))
        if self._measure_region is not None:
            self._measure_xrange = (lo, hi)
            # setRegion emits sigRegionChanged AND sigRegionChangeFinished (pyqtgraph calls
            # lineMoved then lineMoveFinished), i.e. sigMeasureChanged followed by
            # sigMeasureFinished — a programmatic placement is a settled one. It emits
            # NOTHING at all when both edges already sit here.
            self._measure_region.setRegion((lo, hi))
        elif self._measure_active and self._measure_empty:
            # Measure on, no mask: this range BECOMES the mask (and is remembered by the
            # creation, only if there is a pane to hold it -- an unplaced range must not linger
            # and be mistaken for a mask to put back).
            self._create_measure_region(seed=False, initial=(lo, hi), announce="settled")
        else:
            self._measure_xrange = (lo, hi)      # remembered for a later activation / rebuild

    def measure_x_range(self) -> tuple[float, float] | None:
        """Current Measure edges in plotted x coordinates, or None while Measure is off or has
        no mask.

        This complements :meth:`measure_bounds`: navigation widgets need to translate the live
        cursor pair without a lossy x -> sample -> x round trip, while analysis code should keep
        using sample indices.
        """
        if not self._measure_active or self._measure_region is None:
            return None
        lo, hi = self._measure_region.getRegion()
        return float(lo), float(hi)

    def translate_measure_region(self, delta_x: float) -> bool:
        """Translate the live Measure cursors, preserving their width and clamping to the target.

        Used by Previous/Next view scanning: saved measurements are untouched, but the working
        cursor pair follows the view so the annotator can immediately bracket the next event.
        """
        rng = self.measure_x_range()
        target = self.measure_target()
        if rng is None or target is None or not np.isfinite(float(delta_x)):
            return False
        extent = self.measure_cursor_extent(*target)
        if extent is None:
            return False
        e0, e1 = extent
        lo, hi = rng
        width = hi - lo
        span = e1 - e0
        if (not np.isfinite(lo) or not np.isfinite(hi)
                or width <= 0.0 or width > span):
            # There is no width-preserving translation that can put an over-wide selection inside
            # the recording. Refuse instead of claiming success with one edge still out of bounds.
            return False
        requested_delta = float(delta_x)
        new_lo = lo + requested_delta
        if new_lo + width > e1:
            new_lo = e1 - width
        if new_lo < e0:
            new_lo = e0
        new_hi = new_lo + width
        actual_delta = new_lo - lo
        eps = max(width, span, 1.0) * 1e-14
        # A clamp must never turn a forward cursor carry into a backward one (or vice versa).
        if ((requested_delta >= 0.0 and actual_delta <= eps)
                or (requested_delta < 0.0 and actual_delta >= -eps)):
            return False
        self.set_measure_region(new_lo, new_hi)
        return True

    def measure_cursor_extent(self, doc_id: str, channel: int) -> tuple[float, float] | None:
        """Legal plotted-x extent for half-open Measure edges on one target.

        The right cursor may represent ``stop_idx == n``. Its x is one sample interval beyond the
        last sample (the same contract used when MeasurePanel restores such a row), not the last
        sample's own x; clamping to ``values[-1]`` silently drops that final sample.
        """
        vals = self.x_axis_for(doc_id, channel).values
        if vals.size < 2:
            return None
        dx = float(vals[1] - vals[0])
        e0, e1 = float(vals[0]), float(vals[-1]) + dx
        if not (np.isfinite(e0) and np.isfinite(e1) and dx > 0.0 and e1 > e0):
            return None
        return e0, e1

    # -- Measure ON with NO mask -----------------------------------------------------------------
    def measure_empty(self) -> bool:
        """True while Measure is ON and there is no mask: the state after an add (the next event
        starts clean, and Next window carries nothing), after :meth:`discard_measure_region`, or
        after Measure was armed with ``seed=False``.

        An explicit flag and not "no stored range", because a missing range already means "seed
        a mask": read that way, the first rebuild brought a seeded mask back, placement did
        nothing, and Ctrl+M saved the event just added a second time from the stale readout.
        :meth:`measure_bounds` and :meth:`measure_x_range` are None here and
        :meth:`measure_target` still names the target."""
        return bool(self._measure_active and self._measure_empty
                    and self._measure_region is None)

    def discard_measure_region(self) -> bool:
        """Remove the mask but KEEP Measure on. True when there was a mask to remove.

        Emits :attr:`sigMeasureCleared` (and nothing else: no bounds, no deactivation). A draw in
        flight is cancelled with it. Turn the mask back on with :meth:`set_measure_region`, a
        draw, or ``set_measure_active(..., seed=True)``."""
        if not self._measure_active or self._measure_region is None:
            return False
        self._abandon_draw()
        self._remove_measure_region()
        self._measure_xrange = None
        self._measure_empty = True
        self.sigMeasureCleared.emit()
        return True

    def preview_measure_region(self, lo_x: float, hi_x: float) -> bool:
        """Move the mask to ``[lo_x, hi_x]`` (data x, clamped to the target's legal extent) as one
        FRAME of a drag: the two lines move with ``blockLineSignal``, exactly as
        ``LinearRegionItem.mouseDragEvent`` itself does, and only :attr:`sigMeasureChanged`
        goes out, so the Measure panel's debounce coalesces the stream. :meth:`set_measure_region`
        emits a finished signal on every call and the panel recomputes at once on it, about
        8 ms at 200,000 samples per frame. Creates the mask when there is none (the empty state).
        Pair the last frame with :meth:`settle_measure_region`."""
        lo, hi = (float(lo_x), float(hi_x)) if lo_x <= hi_x else (float(hi_x), float(lo_x))
        target = self.measure_target()
        if target is None or not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        extent = self.measure_cursor_extent(*target)
        if extent is not None:
            lo = min(max(lo, extent[0]), extent[1])
            hi = min(max(hi, extent[0]), extent[1])
        region = self._measure_region
        if region is None:
            if self.pane_for_layer(self._measure_layer_id) is None:
                return False
            return self._create_measure_region(seed=False, initial=(lo, hi), announce="changed")
        region.blockLineSignal = True
        try:
            region.lines[0].setValue(lo)
            region.lines[1].setValue(hi)
            region.prepareGeometryChange()
        finally:
            region.blockLineSignal = False
        region.sigRegionChanged.emit(region)      # -> _on_measure_region_changed -> sigMeasureChanged
        return True

    def settle_measure_region(self) -> bool:
        """Announce the mask AT REST: one :attr:`sigMeasureFinished` with its current bounds. On a
        draw's release and after a fit. False when there is no mask."""
        bounds = self._settled_measure_bounds()
        if bounds is None:
            return False
        self._settled_count += 1
        self.sigMeasureFinished.emit(bounds[0], bounds[1])
        return True

    # -- draw mode -------------------------------------------------------------------------------
    def draw_mode(self) -> bool:
        """Whether a plain left-drag draws a Measure span (True) or box-zooms (False). Option
        inverts it for the one drag."""
        return self._draw_mode

    def set_draw_mode(self, on: bool) -> bool:
        """Switch draw mode. Returns the mode now in force.

        Draw mode draws on the Measure TARGET, so it needs one: asked to turn on with Measure off
        (or its pane hidden) it stays off and says why on :attr:`sigDrawRefused`. It then survives
        rebuilds, view-mode changes and table switches for as long as the target pane exists, and
        drops itself (with ``sigDrawModeChanged(False)``) the moment the pane is gone. Turning it
        on places no mask."""
        on = bool(on)
        if on and self._measure_pane() is None:
            self.sigDrawRefused.emit("Turn on Measure first: draw mode draws on the Measure target")
            return self._draw_mode
        if on != self._draw_mode:
            self._draw_mode = on
            self.sigDrawModeChanged.emit(on)
        return self._draw_mode

    def draw_in_progress(self) -> bool:
        g = self._draw
        return g is not None and g.alive

    def cancel_draw(self) -> bool:
        """Cancel the draw in flight, putting the mask back as it was. True when there was one.
        (A right-click does the same from the gesture itself; Esc is the owner's.)"""
        g = self._draw
        if g is None or not g.alive:
            return False
        self._cancel_gesture(g, restore=True)
        return True

    def _measure_pane(self) -> ChannelPane | None:
        """The Measure target's pane, or None when Measure is off or the pane is not drawn."""
        if not self._measure_active or self._measure_layer_id is None:
            return None
        return self.pane_for_layer(self._measure_layer_id)

    def _sync_draw_mode(self) -> None:
        """Draw mode needs its target: drop it, loudly, when the pane is gone."""
        if self._draw_mode and self._measure_pane() is None:
            self._draw_mode = False
            self.sigDrawModeChanged.emit(False)

    def _target_channel_name(self) -> str:
        target = self.measure_target()
        if target is not None and self.session.has_document(target[0]):
            channels = self.session.document(target[0]).meta.channels
            if 0 <= target[1] < len(channels):
                return str(channels[target[1]].name)
        return "target"

    def _target_x(self, scene_pos) -> float | None:
        """Data x of a scene position THROUGH THE TARGET PANE'S OWN ViewBox, clamped to the
        target's legal cursor extent. Never through whichever pane received the event (panes
        with different axis widths map one scene x to different data x) and never through a
        pane's first layer (an overlay's first layer can sit on another time offset)."""
        pane = self._measure_pane()
        target = self.measure_target()
        if pane is None or target is None:
            return None
        x = float(pane.plot.vb.mapSceneToView(scene_pos).x())
        if not np.isfinite(x):
            return None
        extent = self.measure_cursor_extent(*target)
        if extent is not None:
            x = min(max(x, extent[0]), extent[1])
        return x

    def _data_per_pixel(self) -> float | None:
        pane = self._measure_pane()
        if pane is None:
            return None
        vb = pane.plot.vb
        x0, x1 = (float(v) for v in vb.viewRange()[0])
        width = float(vb.width())
        if width <= 0.0 or not x1 > x0:
            return None
        return (x1 - x0) / width

    def _on_draw_event(self, source, phase: str, ev, handle):
        """THE one callback every drag-drawing item reports to (the panes' ViewBoxes and the
        mask's body). ``phase`` is "start", "move", "finish" or "cancel"; ``handle`` is the
        gesture that "start" returned (None when the draw was refused), handed back by the item
        on every later call. A gesture the view has since killed (rebuild, cancel, release) is
        simply not advanced, which is how the frames the scene keeps sending to a removed
        ViewBox are dropped."""
        if phase == "start":
            return self._begin_draw(source, ev)
        g = handle
        if g is None or not g.alive or g is not self._draw:
            return g
        if phase == "cancel":
            self._cancel_gesture(g, restore=True)
            return g
        self._advance_draw(g, ev, finish=(phase == "finish"))
        return g

    def _begin_draw(self, source, ev) -> _DrawGesture | None:
        pane = self._measure_pane()
        target = self.measure_target()
        if pane is None or target is None:
            self.sigDrawRefused.emit("Turn on Measure, then draw a span on the trace")
            return None
        if isinstance(source, pg.ViewBox) and source is not pane.plot.vb:
            self.sigDrawRefused.emit(f"Draw on the {self._target_channel_name()} pane")
            return None
        # The START is the press, not the first frame: the scene only calls a press a drag
        # after the pointer has travelled, and the span must begin where the user pressed.
        start_x = self._target_x(ev.buttonDownScenePos())
        if start_x is None:
            return None
        prior = self._draw
        if prior is not None and prior.alive:
            self._cancel_gesture(prior, restore=True)
        region = self._measure_region
        g = _DrawGesture(start_x, target[0], target[1],
                         tuple(float(v) for v in region.getRegion()) if region is not None else None,
                         self._measure_xrange, self._measure_empty)
        self._draw = g
        self.sigDrawStarted.emit()
        if region is None:
            # Measure on, no mask: the draw makes one, at the press, with nothing seeded and
            # nothing announced -- the first frame below announces it.
            self._create_measure_region(seed=False, initial=(start_x, start_x), announce="none")
        self._advance_draw(g, ev, finish=False)
        return g

    def _advance_draw(self, g: _DrawGesture, ev, *, finish: bool) -> None:
        x = self._target_x(ev.scenePos())
        if x is None:
            return
        lo, hi = (g.start_x, x) if g.start_x <= x else (x, g.start_x)
        if not finish:
            self.preview_measure_region(lo, hi)
            return
        dpp = self._data_per_pixel()
        if dpp is None or (hi - lo) < DRAW_CLICK_MAX_PX * dpp:
            self._cancel_gesture(g, restore=True)         # narrower than 3 px: a click
            return
        self.preview_measure_region(lo, hi)               # the release position, always
        g.alive = False
        self._draw = None
        bounds = self._measure_indices()
        px = self.target_px_samples()
        if bounds is None or px is None:
            self.settle_measure_region()
            return
        settled = self._settled_count
        self.sigDrawReleased.emit(g.doc_id, g.channel, int(bounds[0]), int(bounds[1]), float(px))
        if self._settled_count == settled:
            # Nobody moved the mask to a fitted position or settled it: do it here, once.
            self.settle_measure_region()

    def _cancel_gesture(self, g: _DrawGesture, *, restore: bool) -> None:
        if not g.alive:
            return
        g.alive = False
        if self._draw is g:
            self._draw = None
        if restore:
            self._restore_before_draw(g)
        self.sigDrawCancelled.emit()

    def _abandon_draw(self) -> None:
        """Cancel a draw in flight WITHOUT putting the mask back: the caller is removing or
        re-targeting the mask anyway."""
        g = self._draw
        if g is not None and g.alive:
            self._cancel_gesture(g, restore=False)

    def _restore_before_draw(self, g: _DrawGesture) -> None:
        """Put the mask back as it was before the gesture, "no mask" included."""
        if g.prev_range is not None:
            if self._measure_region is None:
                self._create_measure_region(seed=False, initial=g.prev_range, announce="settled")
            else:
                self.preview_measure_region(*g.prev_range)
                self.settle_measure_region()
        elif self._measure_region is not None and self._measure_active:
            self._remove_measure_region()
            self._measure_xrange = None
            self._measure_empty = True
            self.sigMeasureCleared.emit()

    def _abort_draw_for_rebuild(self) -> bool:
        """A rebuild kills the draw: clear its token and put the stored mask state back, so the
        re-attach that follows recreates the OLD mask (or none)."""
        g = self._draw
        if g is None or not g.alive:
            return False
        g.alive = False
        self._draw = None
        self._measure_xrange = g.prev_xrange
        self._measure_empty = g.prev_empty
        return True

    def _finish_rebuild(self, draw_aborted: bool) -> None:
        """The end of every rebuild, including the one with nothing to draw."""
        self._sync_draw_mode()
        self._refresh_pane_decorations()
        if draw_aborted:
            if self.measure_empty():
                self.sigMeasureCleared.emit()    # the half-drawn span the readout still shows
            self.sigDrawCancelled.emit()

    def _request_catch(self, view_box, x: float) -> None:
        """A double-click in draw mode: ask the owner to catch the event at that x, on the
        target pane, instead of placing a 20% mask."""
        x = float(x)
        if not np.isfinite(x):
            return
        pane = self._measure_pane()
        target = self.measure_target()
        if pane is None or target is None:
            self.sigDrawRefused.emit("Turn on Measure, then double-click the trace to catch an event")
            return
        if view_box is not pane.plot.vb:
            self.sigDrawRefused.emit(f"Draw on the {self._target_channel_name()} pane")
            return
        vals = self.x_axis_for(*target).values
        px = self.target_px_samples()
        if vals.size < 2 or px is None:
            return
        sample = int(min(max(self._nearest_index(vals, x), 0), vals.size - 1))
        self.sigCatchRequested.emit(target[0], target[1], sample, float(px))

    # -- where the pointer is --------------------------------------------------------------------
    def set_annotation_target(self, doc_id: str | None, channel: int | None = None) -> None:
        """Name the (doc_id, channel) the annotation keys will act on while Measure is OFF, so the
        pointer helpers, the focus rule, the cursor and the caption already work before the
        first mask. ``None`` clears it. While Measure is on, its own target wins."""
        new = None if doc_id is None else f"{doc_id}#{int(channel if channel is not None else 0)}"
        if new == self._annotation_layer_id:
            return
        self._annotation_layer_id = new
        self._refresh_pane_decorations()

    def annotation_target(self) -> tuple[str, int] | None:
        layer_id = self._annotation_layer_id
        if layer_id is None:
            return None
        doc_id, _, channel = layer_id.rpartition("#")
        return (doc_id, int(channel)) if doc_id and channel.isdigit() else None

    def _target_ref(self) -> tuple[str, int, ChannelPane] | None:
        """``(doc_id, channel, pane)`` of the pane the keys act on, or None: the Measure target
        when Measure is on, else the annotation target, and only if its pane is drawn."""
        layer_id = (self._measure_layer_id if (self._measure_active and self._measure_layer_id)
                    else self._annotation_layer_id)
        if layer_id is None:
            return None
        doc_id, _, channel = layer_id.rpartition("#")
        if not doc_id or not channel.isdigit():
            return None
        pane = self.pane_for_layer(layer_id)
        return None if pane is None else (doc_id, int(channel), pane)

    def _pointer_scene_pos(self) -> QtCore.QPointF | None:
        """The pointer in scene coordinates, or None unless it is really over THIS trace.

        ``QCursor.pos()`` mapped to the viewport, and ``QApplication.widgetAt`` must be that
        viewport: the position test alone is fooled by a window lying over the trace, which
        still maps into it."""
        global_pos = QtGui.QCursor.pos()
        viewport = self.viewport()
        local = viewport.mapFromGlobal(global_pos)
        if not viewport.rect().contains(local):
            return None
        if QtWidgets.QApplication.widgetAt(global_pos) is not viewport:
            return None
        return self.mapToScene(local)

    def pointer_data_x(self) -> float | None:
        """The data x under the pointer, or None unless the pointer is inside the PLOT AREA (the
        ViewBox, not its axes) of the target pane."""
        ref = self._target_ref()
        scene_pos = self._pointer_scene_pos() if ref is not None else None
        if ref is None or scene_pos is None:
            return None
        vb = ref[2].plot.vb
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        x = float(vb.mapSceneToView(scene_pos).x())
        return x if np.isfinite(x) else None

    def target_px_samples(self) -> float | None:
        """Samples per screen pixel of the target pane at the current zoom, or None."""
        ref = self._target_ref()
        if ref is None:
            return None
        doc_id, channel, pane = ref
        vb = pane.plot.vb
        x0, x1 = (float(v) for v in vb.viewRange()[0])
        width = float(vb.width())
        dx = self.sample_dx(doc_id, channel)
        if width <= 0.0 or dx <= 0.0 or not x1 > x0:
            return None
        return (x1 - x0) / width / dx

    def pointer_sample(self) -> tuple[str, int, float, float] | None:
        """``(doc_id, channel, sample_index, px_samples)`` under the pointer, or None.

        ``sample_index`` is fractional and clamped to ``[0, n]``, through the target layer's own
        x axis (its time offset included). None unless the pointer is over the target pane's
        plot area (:meth:`pointer_data_x`)."""
        ref = self._target_ref()
        x = self.pointer_data_x()
        if ref is None or x is None:
            return None
        doc_id, channel, _pane = ref
        vals = self.x_axis_for(doc_id, channel).values
        dx = self.sample_dx(doc_id, channel)
        px = self.target_px_samples()
        if vals.size < 2 or dx <= 0.0 or px is None:
            return None
        index = float(np.clip((x - float(vals[0])) / dx, 0.0, float(vals.size)))
        return doc_id, channel, index, px

    # -- focus, cursor, caption ------------------------------------------------------------------
    def last_focus_reason(self) -> QtCore.Qt.FocusReason | None:
        """Why the trace last took keyboard focus (None if it never has). A click or the
        pointer arriving (``MouseFocusReason``) means the user pointed at the trace; the window
        parking focus here when a dock closes (``OtherFocusReason``) does not. A click on a
        trace that already has focus counts as that click (:meth:`mousePressEvent`); the pointer
        arriving on it does not, since the trace grows into a closing dock's place under a
        pointer that never moved toward it. A window that loses and regains activation gets its
        focus back for ``ActiveWindowFocusReason``; that is not a new reason, and does not
        overwrite a ``MouseFocusReason``."""
        return self._last_focus_reason

    def add_key_owner(self, widget: QtWidgets.QWidget) -> None:
        """Register a widget (the dock that holds the docked Event inspector) whose descendants
        own bare keys: the pointer entering the trace never takes focus from anything inside it."""
        if widget is not None and all(w is not widget for w in self._key_owners):
            self._key_owners.append(widget)

    def remove_key_owner(self, widget: QtWidgets.QWidget) -> None:
        self._key_owners = [w for w in self._key_owners if w is not widget]

    def set_focus_follows_pointer(self, on: bool) -> None:
        self._focus_follows = bool(on)

    def focus_follows_pointer(self) -> bool:
        return self._focus_follows

    def _focus_holder_is_free(self, holder: QtWidgets.QWidget | None) -> bool:
        """May the trace take focus from ``holder``? Only from a holder with no keys of its own.

        NEVER from an item view, a text or spin field, a slider, a menu, an open or editable
        combo box, or anything inside a registered key owner (:meth:`add_key_owner`): those own
        bare keys (the inspector's Left/Right/A/R/Space and its label hotkeys, the tables'
        j/k/a/r/f, the tree's arrows, a field's typing) and a pointer crossing the trace must not
        silently kill them -- a label hotkey on ``d`` would turn into an add after the crossing.

        From everything else, yes: nothing, the trace itself, buttons (and the custom clickable
        chips that are buttons in all but class), scroll bars, a closed combo box, docks, tab
        bars, the navigator strip. The window starts with focus on such a chip, so a rule that
        left unknown widgets alone would never engage in a fresh window until the first click."""
        if holder is None or holder is self or self.isAncestorOf(holder):
            return True
        for owner in tuple(self._key_owners):
            try:
                if owner is holder or owner.isAncestorOf(holder):
                    return False
            except RuntimeError:
                self._key_owners = [w for w in self._key_owners if w is not owner]
        if isinstance(holder, (QtWidgets.QAbstractItemView, QtWidgets.QLineEdit,
                               QtWidgets.QAbstractSpinBox, QtWidgets.QTextEdit,
                               QtWidgets.QPlainTextEdit, QtWidgets.QKeySequenceEdit,
                               QtWidgets.QMenu, QtWidgets.QMenuBar)):
            return False
        if isinstance(holder, QtWidgets.QAbstractSlider):
            return isinstance(holder, QtWidgets.QScrollBar)       # a slider's arrows are its keys
        if isinstance(holder, QtWidgets.QComboBox):
            return not holder.isEditable() and not holder.view().isVisible()
        return True

    def _on_pane_hover_enter(self, view_box) -> None:
        """The pointer entered a pane: if it is the target pane and the window is active, the
        trace takes keyboard focus (``MouseFocusReason``) -- from a holder with no keys only."""
        # Not even the reason changes on a trace that already has focus: after a park it may have
        # grown under a pointer that never moved toward it (see last_focus_reason).
        if not self._focus_follows or self.hasFocus():
            return
        ref = self._target_ref()
        if ref is None or view_box is not ref[2].plot.vb:
            return
        window = self.window()
        if window is None or not window.isActiveWindow():
            return
        if self._focus_holder_is_free(QtWidgets.QApplication.focusWidget()):
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)

    def close(self):  # noqa: A003  (pyqtgraph's GraphicsView.close, kept)
        """pyqtgraph tears the scene down in here; from this moment the pane decorations must not
        touch an item (see :func:`_alive`)."""
        self._closing = True
        return super().close()

    def focusInEvent(self, ev) -> None:  # noqa: N802
        reason = ev.reason()
        restored = (reason == QtCore.Qt.FocusReason.ActiveWindowFocusReason
                    and self._focus_lost_to_window
                    and self._last_focus_reason == QtCore.Qt.FocusReason.MouseFocusReason)
        if not restored:
            self._last_focus_reason = reason
        self._focus_lost_to_window = False
        super().focusInEvent(ev)
        self._apply_target_cursor()

    def focusOutEvent(self, ev) -> None:  # noqa: N802
        self._focus_lost_to_window = ev.reason() == QtCore.Qt.FocusReason.ActiveWindowFocusReason
        super().focusOutEvent(ev)
        self._apply_target_cursor()

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        # Qt gives focus for a press before delivering it, so focusInEvent has already recorded
        # MouseFocusReason for a press on a trace without focus. A trace that already had it
        # (parked here when a dock closed) gets no focus-in, and the click would leave Return,
        # Enter and Esc asleep.
        if self.hasFocus():
            self._last_focus_reason = QtCore.Qt.FocusReason.MouseFocusReason
        super().mousePressEvent(ev)

    def _apply_target_cursor(self) -> None:
        """A cross on the TARGET pane's ViewBox while the trace has keyboard focus (the keys are
        live), an arrow while focus is elsewhere. On the ViewBox and not the viewport, so the
        other panes and the axes keep their own cursor."""
        if self._closing:
            return
        ref = self._target_ref()
        target_vb = ref[2].plot.vb if ref is not None else None
        keys_live = self.hasFocus()
        for pane in self._panes:
            vb = pane.plot.vb
            if not _alive(vb):
                continue
            if vb is target_vb:
                vb.setCursor(QtCore.Qt.CursorShape.CrossCursor if keys_live
                             else QtCore.Qt.CursorShape.ArrowCursor)
            else:
                vb.unsetCursor()

    def set_target_label(self, text: str) -> None:
        """Caption the target pane (bottom-left, over the data, transparent to the mouse). Empty
        hides it. Survives rebuilds. The crosshair readout owns the top edge."""
        self._target_label_text = str(text or "")
        self._place_target_label()

    def target_label_text(self) -> str:
        return self._target_label_text

    def _place_target_label(self) -> None:
        if self._closing:
            return
        ref = self._target_ref()
        want = bool(self._target_label_text) and ref is not None
        label = self._target_label
        if label is not None and not _alive(label):
            label = self._target_label = None
        if label is not None and (not want or label.parentItem() is not ref[2].plot.vb):
            self._drop_target_label()
            label = None
        if not want:
            return
        if label is None:
            label = _PaneLabel(ref[2].plot.vb)
            self._target_label = label
        label.set_text(self._target_label_text)
        label.show()

    def _drop_target_label(self) -> None:
        label, self._target_label = self._target_label, None
        if label is None:
            return
        try:
            label.detach()
            scene = label.scene()
            if scene is not None:
                scene.removeItem(label)
            label.setParentItem(None)
        except RuntimeError:
            pass                        # torn down with its ViewBox

    def _refresh_pane_decorations(self) -> None:
        """Cursor and caption follow the target: re-derived whenever it can have changed."""
        self._apply_target_cursor()
        self._place_target_label()

    # -- measure internals -----------------------------------------------------------------------
    def _seed_measure_range(self) -> tuple[float, float]:
        """The initial region: the middle ~20% of the current x-view (span/5, centred). Falls back
        to a few ms (or ~200 samples) from the layer start when no view range is available."""
        layer = self.session.layer(self._measure_layer_id) if self._measure_layer_id else None
        target_extent = (self.measure_cursor_extent(layer.doc_id, layer.channel)
                         if layer is not None else None)
        anchor = self.anchor_plot()
        if anchor is not None:
            x0, x1 = anchor.vb.viewRange()[0]
            vlo, vhi = ((float(x0), float(x1)) if x0 <= x1
                        else (float(x1), float(x0)))
            view_span = vhi - vlo
            if view_span > 0 and target_extent is not None:
                e0, e1 = target_extent
                overlap0, overlap1 = max(vlo, e0), min(vhi, e1)
                if overlap1 > overlap0:
                    # Seed from what this target actually contributes to the current view, not the
                    # overlay union. This retains the familiar middle-fifth gesture locally.
                    centre = 0.5 * (overlap0 + overlap1)
                    width = MEASURE_MAX_VIEW_FRACTION * (overlap1 - overlap0)
                else:
                    # The selected target is wholly off screen. A region on the union view would
                    # map to (n,n); put a target-local seed in its centre instead, at the current
                    # zoom scale when that can fit.
                    centre = 0.5 * (e0 + e1)
                    width = MEASURE_MAX_VIEW_FRACTION * min(view_span, e1 - e0)
                width = min(width, e1 - e0)
                if width > 0.0:
                    lo = centre - 0.5 * width
                    lo = min(max(lo, e0), e1 - width)
                    return lo, lo + width
            if view_span > 0:
                centre = 0.5 * (vlo + vhi)
                half = 0.5 * MEASURE_MAX_VIEW_FRACTION * view_span
                return (centre - half, centre + half)
        if layer is not None:
            xax = self.session.x_axis(layer.doc_id, layer.time_offset_s)
            vals = xax.values
            if vals.size >= 2:
                lo = float(vals[0])
                full = float(vals[-1]) - lo
                width = min(full, 3e-3 if xax.is_time else 200.0)
                return (lo, lo + width)
        return (0.0, 1.0)

    def _create_measure_region(self, *, seed: bool,
                               initial: tuple[float, float] | None = None,
                               announce: str = "settled") -> bool:
        """(Re)build the region on the current target pane. ``initial`` places it at exactly that
        range; otherwise ``seed`` picks a fresh window and otherwise again the stored
        _measure_xrange is restored (used on rebuild). Does nothing if the target channel is not
        currently layered -- or if Measure is ON WITH NO MASK and nobody gave it a range, which
        is the whole point of that state: a rebuild must not put a seeded mask back.

        ``announce`` is what is emitted for the range it now holds: "settled" (a changed AND a
        finished emit: a programmatic placement is at rest), "changed" (a drag preview: only the
        debounced stream) or "none". Returns whether a region now exists."""
        self._remove_measure_region()
        pane = self.pane_for_layer(self._measure_layer_id) if self._measure_layer_id else None
        if pane is None:
            return False
        if initial is None and not seed and self._measure_empty:
            return False
        region = _MeasureRegionItem(
            orientation="vertical",
            movable=True,
            brush=pg.mkBrush(33, 150, 243, 40),        # translucent blue — distinct from event spans
            hoverBrush=pg.mkBrush(33, 150, 243, 70),
            pen=pg.mkPen("#1565C0", width=1.5),        # the two edges ARE the boundary cursors
            hoverPen=pg.mkPen("#1E88E5", width=3),
            on_draw=self._on_draw_event,               # a body drag draws, in draw mode or with Option
            draw_mode=self.draw_mode,
        )
        region.setToolTip(
            "Drag the shaded mask to move it. Drag either circular edge handle to resize it.")
        for line in region.lines:
            line.addMarker("o", position=0.5, size=MEASURE_HANDLE_SIZE_PX)
            line.setToolTip("Drag this handle to resize the annotation mask.")
        region.setZValue(8)                            # above the trace, distinct from the crosshair (20)
        pane.plot.addItem(region, ignoreBounds=True)   # ignoreBounds: never ratchet the y-autorange
        self._measure_region = region
        self._measure_plot = pane.plot

        if initial is not None:
            x0, x1 = initial
        elif seed or self._measure_xrange is None:
            x0, x1 = self._seed_measure_range()
        else:
            x0, x1 = self._measure_xrange
        self._measure_empty = False
        region.setRegion((x0, x1))                     # set BEFORE connecting -> no spurious emit here
        self._measure_xrange = (float(x0), float(x1))
        region.sigRegionChanged.connect(self._on_measure_region_changed)
        region.sigRegionChangeFinished.connect(self._on_measure_region_finished)
        if announce in ("settled", "changed"):
            self._on_measure_region_changed()          # one deliberate initial emit of the bounds
        # ...and the range is already AT REST, so announce that too: a listener that debounces
        # sigMeasureChanged would otherwise show nothing for its whole debounce interval
        # every time the tool is armed.
        if announce == "settled":
            self._on_measure_region_finished()
        return True

    def _remove_measure_region(self) -> None:
        """Detach the region from its pane (and the scene). Safe when none exists or after a rebuild
        already destroyed the underlying item."""
        region, plot = self._measure_region, self._measure_plot
        self._measure_region = None
        self._measure_plot = None
        if region is None:
            return
        for signal, slot in ((region.sigRegionChanged, self._on_measure_region_changed),
                             (region.sigRegionChangeFinished, self._on_measure_region_finished)):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        try:
            if plot is not None:
                plot.removeItem(region)
            else:
                scene = region.scene()
                if scene is not None:
                    scene.removeItem(region)
        except RuntimeError:
            pass   # the underlying C++ item was already destroyed (e.g. a concurrent rebuild)

    def _reattach_measure(self) -> None:
        """After a rebuild: recreate the region on the target pane if the Measure tool is active and
        that channel is still layered; otherwise deactivate (its doc/channel went away). With
        Measure on and NO mask there is nothing to recreate: the empty state survives the rebuild."""
        if not self._measure_active or self._measure_layer_id is None:
            return
        if self.pane_for_layer(self._measure_layer_id) is None:
            self._measure_active = False
            self._measure_empty = False
            self._measure_layer_id = None
            self._measure_xrange = None
            # A layer hide can remove the pane without anybody calling set_measure_active(False).
            # Announce the real state so the owning panel/menu cannot stay checked with no cursors.
            self.sigMeasureActiveChanged.emit(False)
            return
        self._create_measure_region(seed=False)   # restore at the remembered x-range (or stay empty)

    def _x_to_index(self, layer: Layer, x: float) -> int:
        """Nearest sample index for a data-x on ``layer``, clamped to [0, n]."""
        xax = self.session.x_axis(layer.doc_id, layer.time_offset_s)
        return self._nearest_index(xax.values, x)

    @staticmethod
    def _nearest_index(vals: np.ndarray, x: float) -> int:
        """THE ONE nearest-sample rule for a data-x against an axis array, clamped to
        [0, n] — ``_x_to_index`` on the live axis, and the Measure-region carry on the axis
        a curve was drawn with, so the two agree to the sample."""
        n = vals.size
        if n == 0:
            return 0
        i = int(np.searchsorted(vals, x))
        if i <= 0:
            return 0
        if i >= n:
            return n
        return i - 1 if (x - vals[i - 1]) <= (vals[i] - x) else i

    def _measure_indices(self) -> tuple[int, int] | None:
        if self._measure_region is None or self._measure_layer_id is None:
            return None
        layer = self.session.layer(self._measure_layer_id)
        if layer is None:
            return None
        x0, x1 = self._measure_region.getRegion()      # already sorted low->high
        start = self._x_to_index(layer, float(x0))
        stop = self._x_to_index(layer, float(x1))
        return (start, stop) if start <= stop else (stop, start)

    def _settled_measure_bounds(self) -> tuple[int, int] | None:
        """Current bounds, remembering the x-range so a rebuild reattaches in place."""
        bounds = self._measure_indices()
        if bounds is None:
            return None
        if self._measure_region is not None:
            self._measure_xrange = tuple(self._measure_region.getRegion())  # type: ignore[assignment]
        return bounds

    def _on_measure_region_changed(self, *_args) -> None:
        """Every intermediate frame of a drag. Listeners MUST treat this as a stream."""
        bounds = self._settled_measure_bounds()
        if bounds is not None:
            self.sigMeasureChanged.emit(bounds[0], bounds[1])

    def _on_measure_region_finished(self, *_args) -> None:
        """The selection came to rest: mouse released, or setRegion() placed it.

        pyqtgraph fires ``sigRegionChangeFinished`` after the last ``sigRegionChanged`` of a
        drag AND at the end of a programmatic ``setRegion``, so this is exactly "the edges
        are now where the user left them" — one emit, not a stream."""
        bounds = self._settled_measure_bounds()
        if bounds is not None:
            self._settled_count += 1
            self.sigMeasureFinished.emit(bounds[0], bounds[1])


# ------------------------------------------------------------------------------------------------
# Crosshair — the live blockade-depth readout.
# ------------------------------------------------------------------------------------------------
class Crosshair(QtCore.QObject):
    """One vertical InfiniteLine + one TextItem per pane, added ignoreBounds=True so they stay out
    of autorange. Driven by the shared scene's sigMouseMoved. Snaps to the true sample via
    Session.time_to_index() — O(1) arithmetic, not an array search (the other reason the time axis
    must be float64: at 300 s a float32 axis maps ~35% of samples onto a neighbour).

    Emits Session.set_cursor(x); positioning happens in the cursorMoved handler so an externally set
    cursor is mirrored identically.
    """

    def __init__(self, view: pg.GraphicsLayoutWidget, session: Session,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._view = view
        self._session = session
        self._items: list[tuple[ChannelPane, pg.InfiniteLine, pg.TextItem]] = []
        session.cursorMoved.connect(self._position)

    def attach(self, panes: list[ChannelPane]) -> None:
        """(Re)create the line+label pair on each pane after a rebuild. Old items were destroyed
        with their plots by TraceView.clear()."""
        self._items = []
        pen = pg.mkPen("#9aa2ab", width=1, style=QtCore.Qt.DashLine)
        for pane in panes:
            vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            vline.setZValue(20)
            pane.plot.addItem(vline, ignoreBounds=True)
            label = pg.TextItem(anchor=(0, 1), color="#e6eaef")
            label.setZValue(21)
            pane.plot.addItem(label, ignoreBounds=True)
            self._items.append((pane, vline, label))

    def on_mouse_moved(self, scene_pos) -> None:
        """Find the pane under the cursor, read x in data coords, snap to that pane's nearest true
        sample, and publish it. All positioning is done by _position via cursorMoved."""
        for pane, _vline, _label in self._items:
            if not pane.plot.sceneBoundingRect().contains(scene_pos):
                continue
            x = float(pane.plot.vb.mapSceneToView(scene_pos).x())
            layer = pane.first_layer()
            if layer is not None:
                idx = self._session.time_to_index(layer.doc_id, layer.channel, x, layer.time_offset_s)
                if idx is not None:
                    xax = self._session.x_axis(layer.doc_id, layer.time_offset_s)
                    x = float(xax.values[idx])
            self._session.set_cursor(x)
            return

    def refresh(self) -> None:
        """Re-render labels at the current cursor (e.g. after a calibration change)."""
        self._position(self._session.cursor_x)

    def _x_text(self, layer: Layer, xax, idx: int) -> str:
        """The cursor's x, stated on BOTH clocks when they differ.

        This readout is where the 5.7x PSD error was born: the user reads a quiet
        stretch off the trace -- an INSTRUMENT-clock number -- and types it into a
        panel that wanted recording seconds. Printing both here means the number the
        other surface wants is already on screen, beside the one that is not.
        ``nanoscope.core.clock`` collapses it back to a single figure on a recording
        whose t₀ is 0, so the common case gains no clutter.
        """
        x = float(xax.values[idx])
        if not xax.is_time:
            return f"{x:.6g} {xax.unit}"
        offset = clock.clock_offset_s(
            self._session.document(layer.doc_id).meta, layer.time_offset_s)
        if clock.clocks_agree(offset) or offset is None:
            return f"{x:.6g} {xax.unit}"
        return clock.both_clocks(offset, instrument_s=x)

    def _position(self, x: float) -> None:
        for pane, vline, label in self._items:
            vline.setPos(x)
            layer = pane.first_layer()
            text = ""
            if layer is not None:
                idx = self._session.time_to_index(layer.doc_id, layer.channel, x, layer.time_offset_s)
                if idx is not None:
                    chan = self._session.calibrated(layer.doc_id, layer.channel)
                    xax = self._session.x_axis(layer.doc_id, layer.time_offset_s)
                    yval = float(chan.data[idx])
                    units = f" {chan.units}" if chan.units else ""
                    text = f"{self._x_text(layer, xax, idx)}\n{yval:.6g}{units}"
            label.setText(text)
            try:
                ymax = pane.plot.vb.viewRange()[1][1]
            except Exception:
                ymax = 0.0
            label.setPos(x, ymax)
