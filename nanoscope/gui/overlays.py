"""Overlay renderers — the concrete fill for the v2 rendering seam in ``trace_view.py``.

A v2 analysis module never constructs graphics items; it DECLARES overlays as
:class:`~nanoscope.core.analysis.OverlayHint` arrays of int64 sample indices. The
VIEW turns those hints into pyqtgraph items through a renderer registered by
:class:`~nanoscope.core.analysis.OverlayKind`. This module supplies those renderers
and :func:`register_all_renderers` to install them.

THE ANTI-``10k-LinearRegionItem`` DESIGN (contract §, OverlayHint docstring):
a detection run yields 1e3-1e5 events. One :class:`pg.LinearRegionItem` each would
be 1e5 full ``QGraphicsObject``s carrying hover/drag machinery — it would destroy
exactly the frame rate the no-pyramid decision exists to protect. Instead
:class:`SpanOverlayItem` is a SINGLE ``pg.GraphicsObject`` that paints every event
span as one ``QRectF`` per *visible* span inside a single ``paint()`` clipped to the
current x-range. N events => 1 graphics item, not N.

...AND ONE ITEM PER SUBPOPULATION WHEN THE SPANS ARE COLOURED (item 13's full-recording
map). A hint carries ONE colour, so a mixture's four classes are four hints and therefore
four items — the same order as the single amber overlay they replace, and not the 1076 that
colouring each event separately would cost. Every one of them keeps ``label="events"``,
because every one of them is still the detection's span overlay; the class travels in
``color`` and in ``name``. See :class:`~nanoscope.core.analysis.OverlayHint`.

Index -> x mapping uses ``session.x_axis(doc_id, time_offset_s).values[idx]``, i.e.
the SAME axis the trace curve on that pane is drawn against (PORT_PLAN §10 R3, now
closed). ``TraceView.add_overlay`` knows the channel and therefore the layer, so it
passes the alignment nudge in keyword-only; the renderers default it to 0.0, the
document's native t0, which is the unchanged answer for an unaligned layer. Mapping at
the native axis unconditionally — as this did — drew every span and marker
``time_offset_s`` away from the event it annotates.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from ..core.analysis import OverlayHint, OverlayKind
from ..core.session import Session
from ..plot.style import BASELINE_TRACK_INK, LEVEL_INK
from .trace_view import register_overlay_renderer

__all__ = [
    "SpanOverlayItem",
    "StepOverlayItem",
    "STEP_LEVEL_HEX",
    "render_spans",
    "render_hline",
    "BASELINE_TRACK_HEX",
    "render_step",
    "render_marker",
    "register_all_renderers",
]

_DEFAULT_SPAN_ALPHA = 60   # 0-255; semi-transparent so the trace reads through the shade

# -- naming the spans, and the two thresholds that keep it honest ---------------------------------
# A span carrying a NAME (item 13: "subpop 2", from a finished subpopulation report) writes that
# name over each event -- but only in a ZOOM WINDOW, never on the full-recording map. Both floors
# below exist because the failure they prevent is the same one the pixel-widening above prevents,
# in the opposite direction: at the full 250 s extent of the owner's recording the 1076 spans are
# each ~8e-5 of the width, so 1076 copies of "subpop 2" would be painted on top of one another and
# the map would be a grey smear where the signal is. The name is information about ONE event, so it
# is drawn only when the picture is about one event.
#   _NAME_MIN_PX      -- a span narrower than this cannot hold even the shortest name, and a name
#                        wider than its own span points at the wrong event.
#   _NAME_MAX_VISIBLE -- above this many visible spans the labels collide however wide they are.
# The COLOUR still carries the class at every zoom; only the words wait for the zoom.
_NAME_MIN_PX = 30
_NAME_MAX_VISIBLE = 40
_NAME_ALPHA = 235          # the ink, near-opaque: the shade under it is deliberately not
_NAME_DY_PX = 4            # baseline offset below the top of the pane

# -- the CUSUM sub-state level overlay, and why it is drawn the way it is ------------------------
# Reported twice: "the Cusum sublevels show on the curate window but not on the main panel", then
# "the sublevels still don't show on the main panel, and cannot be read easily on the curate panel.
# Either the color has to be boldened/changed or another way of representing the sublevels is
# needed." The draw path was never broken — two PlotDataItems really are added to the pane. What
# was broken is that neither of them is VISIBLE, for two independent reasons:
#
#   * NO PIXEL FLOOR. ``render_step`` built a static PlotDataItem at the levels' own x extent, so a
#     1 ms level inside a 250 s recording is 4e-6 of the pane width — a zero-length stroke, which
#     Qt paints as the pen's CAP and nothing else: a ~6 px dark speck with a ~3 px green core
#     inside it, sitting on top of the trace it annotates. Drawn, and invisible, which from the
#     user's chair is indistinguishable from not drawn. :class:`SpanOverlayItem` has had a
#     one-device-pixel floor since exactly this complaint was made about the event spans; the
#     level segments now get the same treatment, computed from the same ``viewPixelSize`` at the
#     same zoom-time, in :class:`StepOverlayItem`.
#   * TOO DARK, AND TOO THIN. ``#009E73`` is the Okabe-Ito bluish green the Event Inspector draws
#     a level in — a good choice THERE, where the level has a pane to itself and one event in it.
#     On the main trace it lies ON the curve inside a dark pane at HSV value 0.62, which is the
#     second half of the complaint. Raising the SAME HUE to full value keeps the inspector's
#     identity and Okabe-Ito's colour-vision separation (which is a hue relationship, not a
#     luminance one) while buying the contrast the trace needs, and it collides with neither the
#     amber event-span shade (``#FFB74D``) nor the vermillion selected-event marker.
#
#: The level segments' ink. PUBLIC so the two surfaces that also draw a fitted level — the Event
#: Inspector's ``_CUSUM_HEX`` and the panel that produces the STEP hint — can be pointed at this
#: one name rather than re-spelling a hex, which is how the inspector and the trace drifted apart
#: about what a level looks like in the first place. The VALUE now lives in
#: ``plot.style.LEVEL_INK`` (the Event Gallery draws levels too and may not import GUI
#: siblings, so the spelling had to move below the GUI layer); this re-export keeps every
#: existing importer working.
STEP_LEVEL_HEX = LEVEL_INK
#: The dark casing stroked UNDER the ink, so the segment reads as a line laid ON the trace rather
#: than as a piece of it. Near-black rather than the pane background: a level drawn over a bright
#: subpopulation shade needs the separation too.
_STEP_CASING_HEX = "#10151b"
_STEP_LEVEL_WIDTH = 4      # up from 3 — "boldened", the user's own word for what was missing
_STEP_CASING_WIDTH = 7     # always > _STEP_LEVEL_WIDTH: the casing is a border, not a second line
#: THE MINIMUM SEGMENT LENGTH, in device pixels — three, where a span's floor is one, and the
#: difference is the difference between a fill and a stroke. One pixel of a span is a visible
#: mark; a stroke shorter than its own pen is thick is drawn as the pen's cap alone, i.e. as a
#: dot sitting at one arbitrary instant of the event rather than as a level held over a time.
#: Three pixels is about the least that still reads as horizontal, and few enough that a whole
#: recording's levels stay a row of separate marks at the blockade depth instead of merging into
#: one continuous band across the pane.
_STEP_MIN_PX = 3.0

# -- the estimated-baseline track, and why it is drawn ON the trace rather than under it ----------
#: The baseline track's default ink, and the choice is MEASURED rather than tasteful.
#:
#: No fixed hue can be "the contrasting one": the trace under it is one of eight layer inks
#: and the pane may also carry an amber span wash, ``#24F0B8`` levels and a vermillion
#: marker. The first ink tried here was ``#FF8A3D``, and a grab of the owner's own -1000 mV
#: recording says why it was wrong — his current channel draws in layer slot 1 (``#E44802``,
#: vermillion), so the pane held 42,157 px of near-vermillion trace and 1,878 px of
#: near-orange rail: legible only because of the casing, and the same hue family as the
#: signal it annotates.
#:
#: White is the one ink measured to collide with NOTHING. On that same real pane, before
#: any rail was drawn, the count of pixels within 24/255 of ``#FFFFFF`` was **zero** — the
#: axis chrome is ``#b9bfc7``, the grid dimmer still, and no layer ink comes near. It is
#: also the maximum-luminance choice (18.6:1 against the ``#111316`` pane), which paired
#: with the near-black casing is the highest-contrast pair available anywhere in this
#: palette. The one collision that remains is layer slot 7, which is white because
#: Okabe-Ito's eighth member is black and a dark pane cannot show it — an EIGHTH overlaid
#: layer, where the casing is still the separation. A producer with a better idea for its
#: own pane passes ``hint.color``.
BASELINE_TRACK_HEX = BASELINE_TRACK_INK
_BASELINE_TRACK_WIDTH = 2   # 1 px of dash inside a noisy 1-2 px trace is speckle, not a line
_BASELINE_CASING_WIDTH = 5  # always > the ink: the rail is a border, not a second line


# ------------------------------------------------------------------------------------------------
# The batched span item — N event spans drawn in ONE paint().
# ------------------------------------------------------------------------------------------------
class SpanOverlayItem(pg.GraphicsObject):
    """A single graphics item that shades every event span.

    Holds two float64 x arrays (start, stop) in DATA coordinates. ``boundingRect``
    tracks the current viewbox (so full-height rects always lie inside it) and
    ``paint`` draws one ``QRectF`` per span whose x-extent intersects the visible
    x-range — everything off-screen is skipped, so the cost is O(visible spans),
    not O(total spans). Repaint is driven by ``viewRangeChanged`` (pyqtgraph calls
    it on every zoom/pan), so there is no per-frame Python work while the view is
    still.
    """

    def __init__(self, x_starts: np.ndarray, x_stops: np.ndarray, *,
                 color: str = "#FFB74D", alpha: int = _DEFAULT_SPAN_ALPHA,
                 label: str = "", name: str = "") -> None:
        super().__init__()
        self._x0 = np.ascontiguousarray(np.asarray(x_starts, dtype=np.float64))
        self._x1 = np.ascontiguousarray(np.asarray(x_stops, dtype=np.float64))
        if self._x0.shape != self._x1.shape:
            raise ValueError(
                f"span starts/stops differ in length: {self._x0.shape} vs {self._x1.shape}")
        self.label = label
        #: The name painted over each span once the view is zoomed in far enough -- see
        #: ``_NAME_MIN_PX``. Distinct from ``label``, which is this overlay's identity to the
        #: panel that owns it (see ``OverlayHint.name``). Empty means "draw no words", which
        #: is every overlay that predates the subpopulation report.
        self.name = str(name)
        qcolor = QtGui.QColor(color)
        qcolor.setAlpha(int(alpha))
        self._brush = pg.mkBrush(qcolor)
        self._pen = QtGui.QPen(QtCore.Qt.NoPen)
        # The words are drawn in the span's OWN hue at near-full alpha. The shade is 60/255 by
        # design (the trace has to read through it); text at that alpha is unreadable, and text
        # in a neutral grey would stop saying WHICH class the span belongs to, which is the only
        # thing the name is for.
        self._name_color = QtGui.QColor(color)
        self._name_color.setAlpha(_NAME_ALPHA)
        self.setZValue(-5)   # behind the trace curve (z 0) but in front of the grid

    @property
    def n_spans(self) -> int:
        return int(self._x0.size)

    def x_bounds(self) -> tuple[float, float] | None:
        """Data-space [min start, max stop] across ALL spans (not just visible)."""
        if self._x0.size == 0:
            return None
        return float(self._x0.min()), float(self._x1.max())

    def boundingRect(self) -> QtCore.QRectF:  # noqa: N802  (Qt signature)
        vb = self.getViewBox()
        if vb is None:
            b = self.x_bounds()
            if b is None:
                return QtCore.QRectF()
            return QtCore.QRectF(b[0], 0.0, b[1] - b[0], 1.0)
        (x0, x1), (y0, y1) = vb.viewRange()
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:  # noqa: N802
        if self._x0.size == 0:
            return
        vb = self.getViewBox()
        if vb is None:
            return
        (vx0, vx1), (vy0, vy1) = vb.viewRange()
        # Clip to the visible x-range: a span is visible iff it overlaps [vx0, vx1].
        visible = (self._x1 >= vx0) & (self._x0 <= vx1)
        idx = np.nonzero(visible)[0]
        if idx.size == 0:
            return
        painter.setPen(self._pen)
        painter.setBrush(self._brush)
        h = vy1 - vy0
        # A SPAN NARROWER THAN A PIXEL MUST STILL BE A PIXEL. At the full 250 s extent of
        # the owner's -500 mV recording a 20 ms event is 8e-5 of the width: every one of
        # the 1196 detected spans rounded to a sub-pixel rect, and "detection finished"
        # looked identical whether 0 events or 1196 had been found -- on the one screen
        # whose job is to show you that they were found. Widening to one device pixel is
        # not decoration: it is the difference between an overlay that is drawn and an
        # overlay that is drawn and invisible. It changes no number and no geometry the
        # user can zoom into: as soon as a span is wider than a pixel this is a no-op.
        px = 0.0
        try:
            ps = vb.viewPixelSize()
            px = float(ps[0]) if ps is not None else 0.0
        except Exception:  # noqa: BLE001 — no viewport yet; fall back to exact widths
            px = 0.0
        if not np.isfinite(px) or px < 0.0:
            px = 0.0
        # THE GEOMETRY IS COMPUTED VECTORISED AND MERGED AT PIXEL RESOLUTION, so the
        # number of rects actually DRAWN is O(min(visible spans, viewport pixels)) —
        # never one Python QRectF + drawRect per span. The per-span loop this replaces
        # honoured the docstring's O(visible spans) but with a ~70 µs Python constant:
        # at the 1e5 top of this module's own declared event range, EVERY full-window
        # repaint (each expose, resize, pan — no cache absorbs it) blocked the GUI
        # thread 4-8 s. Merging is the same decimation idea the trace curve already
        # uses: two spans separated by less than one device pixel cannot be told apart
        # on screen, so they are painted as one run of shade. Any gap the user could
        # SEE (>= 1 px) is preserved exactly, which also means a zoomed-in view — where
        # gaps are wide — paints the same distinct rects it always did.
        xs = self._x0[idx].astype(np.float64, copy=True)
        widths = self._x1[idx] - xs
        np.maximum(widths, 0.0, out=widths)
        if px > 0.0:
            # A SPAN NARROWER THAN A PIXEL MUST STILL BE A PIXEL (see the block comment
            # above) — grown about its own CENTRE, so a comb of events stays where the
            # events are instead of drifting right by half a pixel each.
            narrow = widths < px
            xs[narrow] -= 0.5 * (px - widths[narrow])
            widths[narrow] = px
        ends = xs + widths
        if idx.size > 1:
            order = np.argsort(xs, kind="stable")
            xs = xs[order]
            # Running right edge: a span wholly inside an earlier one must not END a run.
            ends = np.maximum.accumulate(ends[order])
            # A new run starts where the gap to everything before it is a visible pixel
            # (px == 0.0 — no viewport yet — merges only true overlaps: no visual change).
            first = np.empty(idx.size, dtype=bool)
            first[0] = True
            np.greater_equal(xs[1:] - ends[:-1], px, out=first[1:])
            starts_at = np.nonzero(first)[0]
            run_x0 = xs[starts_at]
            run_x1 = ends[np.append(starts_at[1:] - 1, idx.size - 1)]
        else:
            run_x0, run_x1 = xs, ends
        # One drawRect PER MERGED RUN, not per span. The run count is bounded by the
        # viewport width (every preserved gap costs >= 1 px), so this loop is
        # O(min(visible spans, viewport pixels)) — milliseconds even at 1e5 spans.
        # drawRect (not one drawRects(list)) deliberately: the readout-honesty tests
        # record the painter through the same drawRect call the per-span loop used,
        # so the one-pixel-minimum and leave-wide-spans-alone invariants stay pinned
        # by the tests that already guard them.
        for a, b in zip(run_x0.tolist(), run_x1.tolist()):
            painter.drawRect(QtCore.QRectF(float(a), vy0, float(b - a), h))

        self._paint_names(painter, idx, px, vy1)

    def visible_names(self, *, n_visible: int, pixel_width: float) -> bool:
        """Whether the name is painted over a span ``pixel_width`` px wide in a view showing
        ``n_visible`` spans.

        THE RULE LIVES HERE AND :meth:`_paint_names` CALLS IT, rather than each stating the
        same two thresholds. Split out so it can be asserted without a real paint device --
        a test that had to grab a pixmap to find out whether the words appear would be
        testing the compositor -- but a predicate a test asks and the painter re-implements
        is two rules that agree until they do not, which is the shape of defect this tree
        keeps paying for.
        """
        return bool(self.name) and n_visible <= _NAME_MAX_VISIBLE and pixel_width >= _NAME_MIN_PX

    def _paint_names(self, painter: QtGui.QPainter, idx: np.ndarray,
                     px: float, vy1: float) -> None:
        """Write :attr:`name` over each visible span wide enough to hold it.

        THE PAINTER TRANSFORM IS RESET FIRST. This item paints in DATA coordinates, where one
        unit of x is a second and one unit of y is a picoamp; text drawn through that transform
        is stretched by the axis scales and flipped by the y axis, i.e. illegible in a way that
        changes with the zoom. So the anchor is mapped to device space and the glyphs are drawn
        unscaled, which is also why the width test below is in PIXELS and not in seconds.
        """
        if not self.name or px <= 0.0:
            return
        # THE COUNT HALF OF THE RULE, ASKED ONCE. ``visible_names`` is per-span only in
        # its width test; the n_visible half is the same for every span, so when it says
        # no at infinite width nothing below could draw — without this, a full-window
        # paint at 1e5 spans walked all 1e5 of them deciding "no" one at a time.
        if not self.visible_names(n_visible=int(idx.size), pixel_width=np.inf):
            return
        font = painter.font()
        metrics = QtGui.QFontMetricsF(font)
        text_w = metrics.horizontalAdvance(self.name)
        painter.save()
        painter.resetTransform()
        painter.setPen(QtGui.QPen(self._name_color))
        for i in idx:
            x0 = float(self._x0[i])
            x1 = float(self._x1[i])
            if not self.visible_names(n_visible=int(idx.size),
                                      pixel_width=(x1 - x0) / px):
                continue
            anchor = self.mapToDevice(QtCore.QPointF(0.5 * (x0 + x1), vy1))
            if anchor is None:
                continue
            painter.drawText(
                QtCore.QPointF(float(anchor.x()) - 0.5 * text_w,
                               float(anchor.y()) + metrics.ascent() + _NAME_DY_PX),
                self.name)
        painter.restore()

    def viewRangeChanged(self) -> None:  # noqa: N802  (pyqtgraph hook)
        """pyqtgraph calls this on every view zoom/pan. The bounding rect follows the
        viewbox, so declare the geometry stale and repaint the newly-visible spans."""
        self.prepareGeometryChange()
        self.update()


# ------------------------------------------------------------------------------------------------
# The fitted-level item — one batched curve, with a pixel floor on each segment.
# ------------------------------------------------------------------------------------------------
class StepOverlayItem(pg.PlotDataItem):
    """The CUSUM level table as flat segments, each at least :data:`_STEP_MIN_PX` px long.

    A :class:`pg.PlotDataItem` subclass and NOT a bare ``GraphicsObject`` like
    :class:`SpanOverlayItem`: the levels really are a curve (NaN-separated triplets drawn with
    ``connect="finite"``), pyqtgraph already strokes and caches that correctly, and every
    surface that goes looking for this overlay — the analysis panel that retires it, the suite
    that asserts the drawn y values are the stored ``level_pA`` — finds it through
    ``getData()``. Batching is unchanged: ONE item for the whole table, never one per level.

    What it adds is the one thing a static PlotDataItem cannot do: re-derive its own geometry
    when the ZOOM changes. The native extents are kept in ``_x0``/``_x1`` and a level narrower
    than the floor is grown to it ABOUT ITS OWN CENTRE — the same rule, and the same reason, as
    the widening in :meth:`SpanOverlayItem.paint`: a comb of events must stay where the events
    are instead of drifting right by half a floor each. The y values are never touched, so the
    segment stays at the level's own measured current, which is the number the picture is about.
    """

    # Class-level defaults so the pyqtgraph hooks below are safe if Qt calls one of them from
    # inside PlotDataItem.__init__, i.e. before the instance attributes exist.
    _x0: np.ndarray | None = None
    _x1: np.ndarray | None = None
    _y: np.ndarray | None = None
    _drawn_px: float | None = None
    _rewidening: bool = False

    def __init__(self, x_starts: np.ndarray, x_stops: np.ndarray, y: np.ndarray,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._x0 = np.ascontiguousarray(np.asarray(x_starts, dtype=np.float64))
        self._x1 = np.ascontiguousarray(np.asarray(x_stops, dtype=np.float64))
        self._y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if not (self._x0.shape == self._x1.shape == self._y.shape):
            raise ValueError(
                f"level starts/stops/values differ in length: {self._x0.shape} vs "
                f"{self._x1.shape} vs {self._y.shape}")
        self.refresh_widths()

    @property
    def n_levels(self) -> int:
        return 0 if self._y is None else int(self._y.size)

    def pixel_width(self) -> float:
        """One device pixel in DATA units on the x axis, or 0.0 when there is no viewport yet.

        Same source and same fallback as :meth:`SpanOverlayItem.paint` — with no viewbox the
        exact extents are drawn, which is what this always did and is never worse."""
        vb = self.getViewBox()
        if vb is None:
            return 0.0
        try:
            ps = vb.viewPixelSize()
        except Exception:  # noqa: BLE001 — no viewport yet; fall back to exact widths
            return 0.0
        px = float(ps[0]) if ps is not None else 0.0
        return px if np.isfinite(px) and px > 0.0 else 0.0

    def refresh_widths(self) -> None:
        """Re-derive the drawn geometry for the CURRENT zoom. Idempotent and cheap.

        Public because the pixel floor is a function of the viewport, and a caller that has
        just resized or re-ranged a pane (or a test that wants to ask what the current zoom
        draws) must be able to say so without faking a Qt signal. Does nothing when the pixel
        size has not moved, so the zoom/pan signal storm costs one float comparison.
        """
        if self._rewidening or self._y is None:
            return
        px = self.pixel_width()
        if self._drawn_px is not None and px == self._drawn_px:
            return
        n = int(self._y.size)
        xs = self._x0.astype(np.float64, copy=True)
        widths = self._x1 - self._x0
        np.maximum(widths, 0.0, out=widths)
        floor = _STEP_MIN_PX * px
        if floor > 0.0:
            # A LEVEL SHORTER THAN THE FLOOR MUST STILL BE A BAR — grown about its own centre.
            narrow = widths < floor
            xs[narrow] -= 0.5 * (floor - widths[narrow])
            widths[narrow] = floor
        out_x = np.empty(3 * n, dtype=np.float64)
        out_y = np.empty(3 * n, dtype=np.float64)
        out_x[0::3], out_x[1::3], out_x[2::3] = xs, xs + widths, np.nan
        out_y[0::3], out_y[1::3], out_y[2::3] = self._y, self._y, np.nan
        # The re-entrancy guard is not belt-and-braces: setData informs the view its bounds
        # changed, which is one of the things that lands back here.
        self._rewidening = True
        try:
            self.setData(x=out_x, y=out_y, connect="finite")
        finally:
            self._rewidening = False
        self._drawn_px = px

    def viewRangeChanged(self, *args, **kwargs) -> None:  # noqa: N802  (pyqtgraph hook)
        """Every zoom/pan changes how many data units a pixel is worth, so the floor moves."""
        super().viewRangeChanged(*args, **kwargs)
        self.refresh_widths()

    def viewTransformChanged(self) -> None:  # noqa: N802  (pyqtgraph hook)
        """A RESIZE is not a range change but it is a pixel-size change, and a pane that grows
        from nothing to its real extent (which is what happens on the first layout, and what
        every offscreen test does) never emits the range signal at all."""
        super().viewTransformChanged()
        self.refresh_widths()


# ------------------------------------------------------------------------------------------------
# Index -> x helpers.
# ------------------------------------------------------------------------------------------------
def _x_values(session: Session, doc_id: str, time_offset_s: float = 0.0) -> np.ndarray:
    """The x axis values an overlay on this document must be mapped through. float64 (Rule 2).

    ``time_offset_s`` is the LAYER's alignment nudge, which is also the shift the trace curve
    beside the overlay is drawn with (``TraceView.x_axis_for``). Mapping at the document's
    native axis instead puts every span and marker ``time_offset_s`` away from the event it
    annotates — shaded blockades over quiet baseline. Defaults to 0.0, which is the native
    axis and therefore the unchanged answer for every unaligned layer."""
    return session.x_axis(doc_id, float(time_offset_s)).values


def _map_indices(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Map int64 sample indices onto x, clipped into the valid range so an end_idx
    equal to n (half-open) or any stray out-of-range index lands on a real sample
    rather than raising."""
    if values.size == 0:
        return np.zeros(np.asarray(idx).shape, dtype=np.float64)
    clipped = np.clip(np.asarray(idx, dtype=np.int64), 0, values.size - 1)
    return values[clipped].astype(np.float64, copy=True)


# ------------------------------------------------------------------------------------------------
# Renderers — signature (plot, session, doc_id, hint, *, time_offset_s=0.0)
#             -> list[pg.GraphicsObject].
# ------------------------------------------------------------------------------------------------
def render_spans(plot: pg.PlotItem, session: Session, doc_id: str,
                 hint: OverlayHint, *,
                 time_offset_s: float = 0.0) -> list[pg.GraphicsObject]:
    """Render event spans as ONE batched :class:`SpanOverlayItem` (never N items)."""
    values = _x_values(session, doc_id, time_offset_s)
    starts = np.asarray(hint.starts, dtype=np.int64)
    stops = hint.stops if hint.stops is not None else starts + 1
    stops = np.asarray(stops, dtype=np.int64)
    x_starts = _map_indices(values, starts)
    x_stops = _map_indices(values, stops)
    item = SpanOverlayItem(x_starts, x_stops, color=hint.color, label=hint.label,
                           name=hint.name)
    # ignoreBounds=True: the item paints full-pane-height spans that TRACK the view, so its
    # boundingRect equals the current view range. Left in autorange it ratchets the y-axis wider on
    # every autorange pass (union of curve bounds and an ever-growing view rect), crushing the signal
    # to a flat line. It is an annotation, not data — keep it out of bounds like the crosshair.
    plot.addItem(item, ignoreBounds=True)
    return [item]


def render_hline(plot: pg.PlotItem, session: Session, doc_id: str,
                 hint: OverlayHint, *,
                 time_offset_s: float = 0.0) -> list[pg.GraphicsObject]:
    """Render the estimated baseline track ON the raw trace — a cased dashed rail.

    The producer emits it DECIMATED (`analysis.baseline.baseline_track`): ``starts``
    are the int64 sample indices of the knots, ``values`` the baseline there.

    THIS RENDERER SHIPPED UNCALLED FOR A REASON THAT WOULD HAVE COST A ROUND. It drew
    ``width=1``, dashed, at ``zValue=-4`` — a hairline UNDERNEATH the trace curve
    (z 0), which is where the signal's own ink is. Handed a correct track against the
    owner's own recording it produced a screenshot with the overlay present in the
    scene, the values right, and nothing visible on screen: this project's oldest
    failure mode, and its third instance (a toolbar past 1475 px, a level overlay at
    sub-pixel width, a wheel guard that exempted focused widgets). Three things had to
    change together, and none of them is cosmetic:

    * **Z-ORDER.** Above the curve (3/4), not below it — but still below the fitted
      levels (5/6) and the selected-event marker (10), because those are about ONE
      event and this is about the whole recording. ``SpanOverlayItem`` stays at -5.
    * **THE CASING.** No ink is legible over every trace, and the numbers say so.
      Measured WCAG contrast of the default white rail against the eight layer inks it
      may be laid on: 2.20 (blue), 4.02 (vermillion), 2.90, 2.40, 3.31, 4.06, **1.29**
      (yellow), 1.00 (white). Four of those are below the 3:1 floor for a graphical
      object and two are effectively invisible — and swapping the hue only moves which
      layer loses (the first ink tried, ``#FF8A3D``, reads 1.07:1 against the layer-0
      blue). The near-black rail stroked underneath is 8.34, 4.56, 6.32, 7.63, 5.54,
      4.52, 14.21 and 18.34 against those same eight, so the COMPOSITE — dark border,
      bright core — reads over any trace ink, over a subpopulation shade and over
      another layer. Same device as `render_step`'s casing, for the same measured reason.
    * **WIDTH.** 2 px of ink on a 5 px rail. A 1 px dash inside a 1-2 px-thick noisy
      trace is not a line, it is speckle.

    Dashed, not solid, and that is the one thing kept from the original: it is how the
    Event Inspector already draws a baseline on a waveform, and it is what says the
    line is an annotation rather than a second signal.

    ``hint.color`` IS RESPECTED (unlike `render_step`, which owns its ink): two layers
    overlaid in one pane have two different baselines, and the producer is the only
    thing that knows which trace each belongs to. `BASELINE_TRACK_HEX` is the default
    to pass.
    """
    values = _x_values(session, doc_id, time_offset_s)
    x = _map_indices(values, np.asarray(hint.starts, dtype=np.int64))
    y = (np.asarray(hint.values, dtype=np.float64)
         if hint.values is not None else np.zeros_like(x))
    items: list[pg.GraphicsObject] = []
    for pen, z in ((pg.mkPen(_STEP_CASING_HEX, width=_BASELINE_CASING_WIDTH), 3),
                   (pg.mkPen(hint.color, width=_BASELINE_TRACK_WIDTH,
                             style=QtCore.Qt.DashLine), 4)):
        curve = pg.PlotDataItem()                                  # 1  four-step order
        plot.addItem(curve, ignoreBounds=True)                     # 2  annotation, not data
        # SUBSAMPLE, NOT PEAK — the one place in this app that decimation rule is
        # inverted, and deliberately. Peak keeps a min AND a max per bin because a
        # one-sample blockade must survive; a baseline has no such spike to lose, and
        # peak-decimating it would draw a BAND where the estimate is a line.
        curve.setDownsampling(auto=True, method="subsample")       # 3
        curve.setClipToView(True)
        curve.setPen(pen)
        curve.setZValue(z)
        curve.setData(x=x, y=y)                                    # 4  data LAST
        items.append(curve)
    return items


def render_step(plot: pg.PlotItem, session: Session, doc_id: str,
                hint: OverlayHint, *,
                time_offset_s: float = 0.0) -> list[pg.GraphicsObject]:
    """Render fitted sub-state levels as flat horizontal segments — v4's STEP kind.

    ONE batched :class:`StepOverlayItem` for the whole level table (NaN breaks between
    segments, ``connect='finite'``), never one item per level — the same
    anti-10k-items design as :func:`render_spans`. Each segment runs ``[start, stop)``
    at that level's OWN mean current (``LEVEL_SCHEMA['level_pA']``, absolute, so it
    sits on the trace with no polarity reasoning) — exactly the picture the Event
    Inspector draws per event. Until this renderer existed, ``OverlayKind.STEP`` was
    an enum value with no renderer and no producer: sub-state levels showed in the
    inspector and NOWHERE on the main trace (owner note #18).

    THE INK IS THIS MODULE'S, NOT THE HINT'S — the one place a renderer here overrides
    ``hint.color``, and the reason is in the constants block at the top of this file. How a
    level reads is a property of the SURFACE it is drawn on: the same green that is right in
    the inspector, where a level has a pane to itself, disappears into a live trace inside a
    dark pane, and the hint is declared by an analysis panel that cannot see either. STEP has
    exactly one producer and one meaning (a CUSUM level fit), so there is no second colour to
    respect; a future producer that genuinely needs its own ink should be given
    ``hint.color`` back here at that point, and :data:`STEP_LEVEL_HEX` becomes its default.
    """
    values = _x_values(session, doc_id, time_offset_s)
    starts = np.asarray(hint.starts, dtype=np.int64)
    stops = (np.asarray(hint.stops, dtype=np.int64) if hint.stops is not None
             else starts + 1)
    y = (np.asarray(hint.values, dtype=np.float64) if hint.values is not None
         else np.zeros(starts.shape, dtype=np.float64))
    n = min(starts.size, stops.size, y.size)
    if n == 0:
        return []
    x0 = _map_indices(values, starts[:n])
    x1 = _map_indices(values, stops[:n])
    # Cased like the inspector's segments (a wider dark underlay beneath the colour),
    # because a 2-px line at trace zoom dissolved into the waveform it annotates. BOTH
    # strokes are StepOverlayItems so the casing follows the ink through every zoom — a
    # static underlay would sit a whole pixel floor short of the bar it is meant to border.
    under = StepOverlayItem(x0, x1, y[:n],
                            pen=pg.mkPen(_STEP_CASING_HEX, width=_STEP_CASING_WIDTH),
                            connect="finite")
    item = StepOverlayItem(x0, x1, y[:n],
                           pen=pg.mkPen(STEP_LEVEL_HEX, width=_STEP_LEVEL_WIDTH),
                           connect="finite")
    under.setZValue(5)
    item.setZValue(6)   # above the shaded spans, below the selected-event marker (10)
    # ignoreBounds: an annotation, not data — levels lie inside the curve's own range,
    # so excluding them from autorange loses nothing and can never widen it.
    plot.addItem(under, ignoreBounds=True)
    plot.addItem(item, ignoreBounds=True)
    # Both re-derived once more AFTER both are in the pane. Adding the first item can itself
    # move the view (a range signal, a relayout), and the second is then built against a pixel
    # size the first has not seen — measured: a casing 1.4% shorter than the bar it borders,
    # invisible but real, and the sort of ordering dependence that stops being invisible the
    # day a pane is created at one size and shown at another.
    under.refresh_widths()
    item.refresh_widths()
    return [under, item]


def render_marker(plot: pg.PlotItem, session: Session, doc_id: str,
                  hint: OverlayHint, *,
                  time_offset_s: float = 0.0) -> list[pg.GraphicsObject]:
    """Render the SELECTED event as the single allowed interactive item — one
    non-movable :class:`pg.LinearRegionItem` spanning [start, stop] (or one
    :class:`pg.InfiniteLine` if the hint carries no stop)."""
    values = _x_values(session, doc_id, time_offset_s)
    starts = np.asarray(hint.starts, dtype=np.int64)
    x_start = _map_indices(values, starts)
    qcolor = QtGui.QColor(hint.color)
    if hint.stops is not None and np.asarray(hint.stops).size:
        x_stop = _map_indices(values, np.asarray(hint.stops, dtype=np.int64))
        x0 = float(x_start[0])
        x1 = float(x_stop[0])
        qcolor.setAlpha(90)
        region = pg.LinearRegionItem(values=(x0, x1), movable=False,
                                     brush=pg.mkBrush(qcolor))
        region.setZValue(10)   # the selected event sits ON TOP
        plot.addItem(region, ignoreBounds=True)   # vertical marker: must not drive y-autorange
        return [region]
    line = pg.InfiniteLine(pos=float(x_start[0]), angle=90, movable=False,
                           pen=pg.mkPen(qcolor, width=2))
    line.setZValue(10)
    plot.addItem(line, ignoreBounds=True)         # vertical marker: must not drive y-autorange
    return [line]


# ------------------------------------------------------------------------------------------------
# Registration — installs the renderers into trace_view's registry.
# ------------------------------------------------------------------------------------------------
def register_all_renderers() -> None:
    """Register every overlay renderer into ``trace_view``. Idempotent — a second
    call simply re-binds the same callables, so calling it from both ``app.main()``
    and the test ``qapp`` fixture is safe."""
    register_overlay_renderer(OverlayKind.SPAN, render_spans)
    register_overlay_renderer(OverlayKind.HLINE, render_hline)
    register_overlay_renderer(OverlayKind.STEP, render_step)
    register_overlay_renderer(OverlayKind.MARKER, render_marker)
