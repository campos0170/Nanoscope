"""Reset view, zoom history, the app right-click menu and the hover readout of a plot window.

ONE HELPER, INSTALLED PER PLOT. The plot windows (Distributions, Noise spectrum, I-V, Voltage
dependence, Event waveforms, Distribution summaries, Subpopulations) all draw into pyqtgraph
PlotItems, and until this module each kept pyqtgraph's generic right-click menu: View All, X
axis, Y axis, Mouse Mode, "Plot Options" with log / FFT / derivative / invert transforms, and the
scene's "Export..." dialog. Those transforms change the picture OUTSIDE the page's saved
configuration (a log toggle the page never hears about, an FFT of a histogram), and the export
dialog bypasses the page's own export seam (provenance, physical size, staged writes). So the
menu is replaced, here, once, with the page's own verbs.

Everything in this module is VIEW STATE. Nothing here recomputes an analysis, reads source data,
changes a saved setting, restyles anything, or touches either Cmd+Z stack (Distributions'
PlotConfig history, the appearance history of the other pages) or Edit ▸ Undo:

* :class:`PlotNavigation` — Reset view and the Back / Forward history of one PlotItem.
* :func:`install_plot_interaction` — the app menu and the hover readout on one PlotItem.
* :class:`PlotReadout` / :func:`window_readout` — the one-line pointer readout of a window.
* :func:`view_button` — a "Reset view" tool button whose drop-down holds Zoom back / forward.
"""
from __future__ import annotations

import contextlib
import math
import re
import weakref
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

__all__ = ["SETTLE_MS", "PlotNavigation", "PlotReadout", "app_menu_for", "axis_name",
           "install_plot_interaction", "keep_plot_history", "navigation_for", "readout_text",
           "reset_plot_history", "view_button", "window_readout"]

#: The trace's settle (``trace_view.py``): a zoom animation or a drag is one history entry,
#: not fifty, and a view is only "a place you were" once it has stopped moving.
SETTLE_MS = 250

#: How far (screen pixels) the pointer may be from a point for the readout to name it.
NEAREST_PX = 12.0

_TAG = re.compile(r"<[^>]+>")


def _same_view(a, b) -> bool:
    """Two (x, y) view ranges are one framing when every edge agrees to 1e-4 of its span."""
    for (a0, a1), (b0, b1) in zip(a, b):
        span = max(abs(a1 - a0), abs(b1 - b0), 1e-12)
        if abs(a0 - b0) >= span * 1e-4 or abs(a1 - b1) >= span * 1e-4:
            return False
    return True


def _as_view(ranges) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((float(ranges[0][0]), float(ranges[0][1])),
            (float(ranges[1][0]), float(ranges[1][1])))


class PlotNavigation(QtCore.QObject):
    """Reset view and the settled-view history (Back / Forward) of one PlotItem.

    Entries are ``((x0, x1), (y0, y1))`` in VIEWBOX coordinates, i.e. log10 on a log axis,
    because that is what ``setRange`` takes back; they are never shown to anyone as numbers.

    TWO KINDS OF RANGE CHANGE, AND ONLY ONE OF THEM IS A PLACE YOU WENT. The page moves the
    view itself (a draw auto-ranges, a box re-pins its typed limits, a late auto-range settles
    once the items report their bounds); the user moves it (wheel, drag, box zoom — pyqtgraph
    says so with ``sigRangeChangedManually``) or asks the app to (Reset view). Until the user
    has navigated, a settled page-made change REPLACES the current entry instead of
    appending: otherwise every redraw would leave a trail of near-identical framings to step
    back through, and the first Back after a plot would land on half-drawn geometry. Once the
    user has navigated, every settled change appends -- Distributions' Auto buttons and an
    applied Axis range included -- and a new one prunes the forward branch exactly as the
    trace's history does.

    THE HISTORY BELONGS TO THE GEOMETRY ON SCREEN. :meth:`reset_history` is called where the
    page clears a plot to draw something new (a Plot / Apply, new data, another plot type, a
    log toggle — all of which change what a range MEANS). Appearance edits never clear a plot
    (``apply_plot_appearance`` "styles existing items, without clear"), so they keep it. The
    history is never saved with a project: it is where you looked, not what you analysed.
    A redraw of the SAME geometry that puts the user's view back (Distributions' stale-mark
    redraw) wraps itself in :meth:`keep_history`, because its clear is not new geometry.

    A LINKED SIDE PLOT NAVIGATES THIS VIEW TOO. Distributions' residual strip is x-linked
    to the plot above it; a wheel there moves this ViewBox through the link, which pyqtgraph
    reports here as a plain range change. :meth:`listen_to` connects that plot's "manually"
    as well, so the gesture is a user step wherever it was made.
    """

    historyChanged = QtCore.Signal()

    def __init__(self, plot: pg.PlotItem, *, reset: Callable[[], None] | None = None,
                 parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.plot = plot
        self._vb: pg.ViewBox = plot.getViewBox()
        self._reset = reset if reset is not None else plot.autoRange
        self._hist: list = []
        self._ptr = -1
        #: True while the current entry is the page's own framing (no user navigation since
        #: the last draw or Reset view): a settled page-made change then replaces it.
        self._tail_auto = True
        self._restoring = False
        current = _as_view(self._vb.viewRange())
        self._before = current   # the range before the latest change (see _on_manual)
        self._now = current
        #: True from a range change until control returns to the event loop. A gesture's
        #: own move and its ``sigRangeChangedManually`` arrive in ONE handler, so this is
        #: what tells "the zoom just moved the view from ``_before``" apart from a gesture
        #: that moved nothing here (a y-only drag on an x-linked strip), whose starting
        #: framing is the view on screen, not the stale ``_before`` of some older change.
        self._moved_now = False
        self._moved_clear = QtCore.QTimer(self)
        self._moved_clear.setSingleShot(True)
        self._moved_clear.setInterval(0)
        self._moved_clear.timeout.connect(self._clear_moved)
        self._settle = QtCore.QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self._commit_view)
        self._vb.sigRangeChanged.connect(self._on_range_changed)
        self.listen_to(plot)
        self.reset_history()

    def listen_to(self, plot: pg.PlotItem) -> None:
        """Count ``plot``'s user gestures (wheel, drag, box zoom) as navigation of this view:
        the plot itself, or a side plot linked to it that shares this history."""
        plot.getViewBox().sigRangeChangedManually.connect(self._on_manual)

    # -- recording -------------------------------------------------------------------------------
    def _current(self):
        return _as_view(self._vb.viewRange())

    def _clear_moved(self) -> None:
        self._moved_now = False

    def _on_range_changed(self, *_args) -> None:
        if self._restoring:
            return
        self._before, self._now = self._now, self._current()
        self._moved_now = True
        self._moved_clear.start()
        self._settle.start()

    def _on_manual(self, *_args) -> None:
        if self._restoring:
            return
        if self._tail_auto:
            # The first user step after a draw: the framing it starts FROM is the drawn view,
            # even when the draw's own settle has not fired yet (a zoom inside 250 ms of a
            # Plot). Record that as the current entry, then let the zoom append on settle.
            # If this gesture did not move this view (a y-only zoom on a linked strip), the
            # drawn view is the one still on screen.
            self._record(self._before if self._moved_now else self._current(), replace=True)
            self._tail_auto = False
        self._settle.start()

    def _record(self, view, *, replace: bool) -> None:
        view = _as_view(view)
        if self._hist and _same_view(view, self._hist[self._ptr]):
            return
        if replace and self._hist:
            del self._hist[self._ptr + 1:]
            self._hist[self._ptr] = view
        else:
            del self._hist[self._ptr + 1:]          # a new place prunes any forward branch
            self._hist.append(view)
            self._ptr = len(self._hist) - 1
        self.historyChanged.emit()

    def _commit_view(self) -> None:
        self._record(self._current(), replace=self._tail_auto)

    def _commit_pending(self) -> None:
        """Flush a still-settling view into the history now (``trace_view._commit_pending``):
        the framing on screen when Back is pressed is real, and dropping it would make a quick
        Back land two framings back and lose the zoom from Forward."""
        if self._settle.isActive():
            self._settle.stop()
            self._commit_view()

    def reset_history(self) -> None:
        """Forget every recorded view: the page is drawing new geometry. The current view is
        the provisional first entry, replaced as the new drawing settles."""
        self._settle.stop()
        self._hist = [self._current()]
        self._ptr = 0
        self._tail_auto = True
        self._before = self._now = self._hist[0]
        self._settle.start()
        self.historyChanged.emit()

    @contextlib.contextmanager
    def keep_history(self):
        """Keep the recorded views across a redraw of the SAME geometry that puts the
        user's view back afterwards (Distributions' stale-mark redraw: the plotted table is
        the one already drawn, so every recorded range still means what it meant). The
        redraw's clear calls :meth:`reset_history` like any other; this undoes that."""
        self._commit_pending()
        saved = (list(self._hist), self._ptr, self._tail_auto)
        try:
            yield
        finally:
            self._settle.stop()
            self._hist, self._ptr, self._tail_auto = saved
            self._before = self._now = self._current()
            if not _same_view(self._now, self._hist[self._ptr]):
                self._settle.start()   # the view did not come back exactly: record it
            self.historyChanged.emit()

    # -- navigation ------------------------------------------------------------------------------
    def history(self) -> list:
        """The recorded views, oldest first (tests and diagnostics)."""
        self._commit_pending()
        return list(self._hist)

    def position(self) -> int:
        self._commit_pending()
        return self._ptr

    def can_go_back(self) -> bool:
        if self._settle.isActive() and not self._tail_auto and self._hist and not _same_view(
                self._current(), self._hist[self._ptr]):
            return True    # an unsettled zoom: Back would flush it, then step back from it
        return self._ptr > 0

    def can_go_forward(self) -> bool:
        if self._settle.isActive() and not self._tail_auto and self._hist and not _same_view(
                self._current(), self._hist[self._ptr]):
            return False   # that zoom prunes the forward branch the moment it is recorded
        return 0 <= self._ptr < len(self._hist) - 1

    def _restore(self, view) -> None:
        self._restoring = True
        try:
            self._vb.setRange(xRange=view[0], yRange=view[1], padding=0)
        finally:
            self._restoring = False
        self._before = self._now = self._current()
        self._tail_auto = False
        self.historyChanged.emit()

    def view_back(self) -> None:
        """Step back to the previous settled view."""
        self._commit_pending()
        if self._ptr > 0:
            self._ptr -= 1
            self._restore(self._hist[self._ptr])

    def view_forward(self) -> None:
        """Step forward again after Back."""
        self._commit_pending()
        if 0 <= self._ptr < len(self._hist) - 1:
            self._ptr += 1
            self._restore(self._hist[self._ptr])

    def reset_view(self) -> None:
        """The page's configured framing: zoom and pan only, through the page's own reset.

        The page's callable repeats what its last draw did to frame the plot (auto-range, a
        category slot range, Distributions' typed Axis range). It is recorded as a new entry
        so Back returns to the zoom you reset from, and then treated as the page's own
        framing again, so an auto-range that settles a moment later replaces it."""
        self._commit_pending()
        self._tail_auto = False
        if all(isinstance(item, pg.TextItem) for item in self.plot.items):
            # Nothing but text: a page's message frame ("No usable recordings", drawn on a
            # fixed 0..1 frame). ``autoRange()`` would fit the text's anchor POINT and zoom
            # in on it, so return to the framing the page drew the message with instead.
            self._vb.setRange(xRange=self._hist[0][0], yRange=self._hist[0][1], padding=0)
        else:
            self._reset()
            # A reset that only ENABLES auto-range (Distribution summaries' ECDF framing)
            # has moved nothing yet: pyqtgraph fits at the next paint. Fit now, so the
            # framing recorded below is the reset one, not the zoom it reset from -- which
            # would make the late fit replace the zoom's entry and lose it from Back.
            if any(self._vb.state["autoRange"]) and getattr(self._vb, "_autoRangeNeedsUpdate",
                                                            True):
                self._vb.updateAutoRange()
        self._settle.stop()
        self._before = self._now = self._current()
        self._record(self._now, replace=False)
        self._tail_auto = True
        self._settle.start()
        self.historyChanged.emit()


# -- the readout ---------------------------------------------------------------------------------
class PlotReadout(QtWidgets.QLabel):
    """One fixed-height line of pointer values. It never sizes the window or moves a plot.

    Its width is whatever the layout gives it (``Ignored`` horizontally) and long text is
    elided, so a long axis name cannot widen a status bar, and its height is one line, so a
    readout that empties when the pointer leaves cannot shift anything either. It lives OUTSIDE
    every graphics scene: no figure export, Figure Library capture or frozen snapshot reads a
    QLabel, so the readout can never end up in artwork.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("plotPointerReadout")
        self.setAccessibleName("Pointer readout")
        self.setTextFormat(QtCore.Qt.PlainText)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(1)
        self.setFixedHeight(self.fontMetrics().height() + 4)
        self.setToolTip("Values under the pointer, in the units the axes show. Shown on screen "
                        "only; never part of an exported figure.")
        self._full = ""

    def set_readout(self, text: str) -> None:
        self._full = str(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self) -> None:
        width = max(1, self.width() - 4)
        super().setText(self.fontMetrics().elidedText(self._full, QtCore.Qt.ElideRight, width))

    def resizeEvent(self, event) -> None:   # noqa: N802 (Qt)
        super().resizeEvent(event)
        self._elide()


def window_readout(window: QtWidgets.QWidget, layout: QtWidgets.QBoxLayout | None = None,
                   index: int = -1) -> PlotReadout:
    """The window's readout: in a QMainWindow's status bar, else inserted into ``layout``.

    A NORMAL (not permanent) status-bar widget with the stretch: it takes the whole line, and
    a temporary status message (Distributions' lasso notes) hides it for the message's few
    seconds instead of being squeezed into a sliver beside it.
    """
    readout = PlotReadout()
    if isinstance(window, QtWidgets.QMainWindow) and layout is None:
        window.statusBar().addWidget(readout, 1)
    elif layout is not None:
        layout.insertWidget(index if index >= 0 else layout.count(), readout)
    else:
        raise ValueError("A plain widget needs a layout for its readout.")
    return readout


def axis_name(axis: pg.AxisItem, fallback: str = "") -> str:
    """The axis title as drawn (HTML stripped): a user's typed label wins, as on screen."""
    text = " ".join(_TAG.sub(" ", str(axis.labelText or "")).split())
    return text or fallback


def _category(axis: pg.AxisItem, v: float) -> str | None:
    """The tick label of a category axis (box / violin sources) under ``v``, if any."""
    levels = getattr(axis, "_tickLevels", None)
    if not levels:
        return None
    best = None
    for level in levels:
        for pos, label in level:
            d = abs(float(pos) - v)
            if d <= 0.5 and (best is None or d < best[0]):
                best = (d, str(label))
    return None if best is None else best[1]


def _number(v: float) -> str:
    if not math.isfinite(v):
        return "—"
    return f"{v:.5g}"


def axis_value_text(axis: pg.AxisItem, v: float, *, log: bool | None = None,
                    category: bool = True) -> str:
    """``v`` (a ViewBox coordinate) as the number the axis' ticks would show there.

    Log: pyqtgraph's log mode holds log10 values, so the real value is 10**v. A page that
    draws log10 values on a linear axis (the Event-waveforms density) says so with ``log``.
    Scale: the ticks are multiplied by ``setScale`` (Distributions' Time unit on a linear
    axis) or, on a log axis, by the window's ``display_factor`` (``ReadableLogAxis``). The
    same multiplication here is what makes the readout agree with the tick labels.
    """
    if category:
        name = _category(axis, v)
        if name is not None:
            return name
    is_log = bool(axis.logMode) if log is None else bool(log)
    try:
        real = 10.0 ** v if is_log else float(v)
    except OverflowError:
        real = math.inf
    factor = float(getattr(axis, "scale", 1.0) or 1.0)
    if axis.logMode:
        factor *= float(getattr(axis, "display_factor", 1.0) or 1.0)
    shown = real * factor
    units = str(axis.labelUnits or "")
    if units and math.isfinite(shown):
        if axis.autoSIPrefix:
            return pg.siFormat(shown, precision=5, suffix=units)
        return f"{_number(shown)} {units}"
    return _number(shown)


def readout_text(plot: pg.PlotItem, view_pos: QtCore.QPointF, *, x_name=None, y_name=None,
                 x_log=None, y_log=None, extra=None, points=None) -> str:
    """The readout for the pointer at ``view_pos`` (ViewBox coordinates) over ``plot``."""
    bottom, left = plot.getAxis("bottom"), plot.getAxis("left")
    vx, vy = float(view_pos.x()), float(view_pos.y())
    xn = x_name() if callable(x_name) else (x_name or axis_name(bottom, "x"))
    yn = y_name() if callable(y_name) else (y_name or axis_name(left, "y"))
    xl = x_log() if callable(x_log) else x_log
    yl = y_log() if callable(y_log) else y_log
    parts = [f"{xn} {axis_value_text(bottom, vx, log=xl)}",
             f"{yn} {axis_value_text(left, vy, log=yl)}"]
    if extra is not None:
        more = extra(vx, vy)
        if more:
            parts.append(str(more))
    if points is not None:
        near = _nearest_point(plot, vx, vy, points(), xl, yl)
        if near:
            parts.append(near)
    return "   ·   ".join(parts)


def _nearest_point(plot, vx, vy, providers: Iterable, xl, yl) -> str:
    """The nearest visible point within :data:`NEAREST_PX` screen pixels, with its identity.

    Measured in PIXELS, not data units: on a dwell (ms) × depth (pA) scatter one data unit
    means wildly different screen distances on the two axes. Vectorised over each item's
    record array, so a 100 k-point cloud costs about a millisecond per pointer move.
    """
    px, py = plot.getViewBox().viewPixelSize()
    if not (px > 0 and py > 0):
        return ""
    best = None
    for item, describe in providers:
        if item is None or not item.isVisible():
            continue
        data = getattr(item, "data", None)
        if data is None or not len(data):
            continue
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
        d2 = ((x - vx) / px) ** 2 + ((y - vy) / py) ** 2
        visible = np.asarray(data["visible"], dtype=bool) if "visible" in data.dtype.names else None
        if visible is not None:
            d2 = np.where(visible, d2, np.inf)
        d2 = np.where(np.isfinite(d2), d2, np.inf)
        i = int(np.argmin(d2))
        if math.isfinite(d2[i]) and (best is None or d2[i] < best[0]):
            best = (float(d2[i]), item, describe, i, float(x[i]), float(y[i]))
    if best is None or best[0] > NEAREST_PX ** 2:
        return ""
    _d2, item, describe, i, x, y = best
    identity = describe(item.data[i]["data"]) if describe is not None else ""
    where = (f"({axis_value_text(plot.getAxis('bottom'), x, log=xl, category=False)}, "
             f"{axis_value_text(plot.getAxis('left'), y, log=yl, category=False)})")
    return f"nearest {identity or 'point'} {where}"


def image_value(image: pg.ImageItem | None, vx: float, vy: float):
    """The value of the image pixel under view position (vx, vy), or None outside it."""
    if image is None or not image.isVisible() or image.image is None:
        return None
    p = image.mapFromView(QtCore.QPointF(vx, vy))
    i, j = int(math.floor(p.x())), int(math.floor(p.y()))
    arr = image.image
    # ImageItem maps x to axis 0 for col-major images and to axis 1 for row-major ones.
    if getattr(image, "axisOrder", "col-major") == "row-major":
        i, j = j, i
    if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
        return float(arr[i, j])
    return None


class _SceneHover(QtCore.QObject):
    """One mouse-move dispatcher per graphics scene, so the plots sharing a scene (the
    density map and its marginals) do not take turns clearing each other's text."""

    def __init__(self, scene) -> None:
        super().__init__(scene)
        self._entries: list[tuple[pg.PlotItem, PlotReadout, dict]] = []
        #: The plot whose values the readout shows now, so a hover EXIT of another plot in
        #: this scene cannot blank it (see :meth:`watch_exit`).
        self._owner: pg.PlotItem | None = None
        scene.sigMouseMoved.connect(self._moved)

    def add(self, plot, readout, options) -> None:
        self._entries.append((plot, readout, options))
        self.watch_exit(plot)

    def _moved(self, scene_pos) -> None:
        hit = None
        for plot, readout, options in self._entries:
            try:
                vb = plot.getViewBox()
                if plot.isVisible() and vb.sceneBoundingRect().contains(scene_pos):
                    hit = (plot, readout, options, vb.mapSceneToView(scene_pos))
                    break
            except RuntimeError:     # a plot deleted with its page
                continue
        for plot, readout, options in self._entries:
            if hit is None or readout is not hit[1]:
                readout.set_readout("")
        self._owner = hit[0] if hit is not None else None
        if hit is not None:
            plot, readout, options, view_pos = hit
            readout.set_readout(readout_text(plot, view_pos, **options))

    def watch_exit(self, plot) -> None:
        """Empty the readout when the pointer leaves ``plot``'s view.

        NOT A PYTHON EVENT FILTER ON THE VIEW. That was the first design (a ``Leave`` in
        ``eventFilter``), and it made the PySide construction segfault family about four
        times as frequent: every event the GraphicsView gets -- resize, layout, polish,
        child and timer events, thousands per redraw -- went through Python. Measured on
        one synthetic five-window redraw loop: 8 crashes in 24 runs with the filter, 1 in
        16 without it, 2 in 24 on 650403e. The scene already tells the PlotItem when the
        pointer leaves (pyqtgraph's ``sendHoverEvents(exitOnly=True)`` from the view's
        ``leaveEvent``), from Python, so the PlotItem's own ``hoverEvent`` is wrapped,
        on this instance only, and only for plots that have a readout.

        pyqtgraph emits ``sigMouseMoved`` BEFORE it sends the hover events, so moving
        from one plot of this scene to another has already handed the readout to the new
        plot when the old one's exit arrives: an exit clears only the plot that owns it.
        Weak references keep the wrapper from tying the plot and this object together.
        """
        original = plot.hoverEvent
        plot_ref, hover_ref = weakref.ref(plot), weakref.ref(self)

        def hover_event(ev, original=original):
            original(ev)
            hover, owner = hover_ref(), plot_ref()
            if ev.exit and hover is not None and owner is not None and hover._owner is owner:
                hover._owner = None
                for entry_plot, readout, _options in hover._entries:
                    if entry_plot is owner:
                        readout.set_readout("")

        plot.hoverEvent = hover_event


def navigation_for(plot) -> PlotNavigation | None:
    """The navigation installed on ``plot`` (a PlotItem or a PlotWidget), if any."""
    if isinstance(plot, pg.PlotWidget):
        plot = plot.getPlotItem()
    return getattr(plot, "_nanoscope_plot_navigation", None)


def reset_plot_history(*plots) -> None:
    """Restart the zoom history of each plot that has one: the page is clearing it to draw
    new geometry. A no-op for a plot without a navigation, so a page's clear helper may
    call it before (or without) :func:`install_plot_interaction`."""
    for plot in plots:
        navigation = navigation_for(plot)
        if navigation is not None:
            navigation.reset_history()


@contextlib.contextmanager
def keep_plot_history(plot):
    """:meth:`PlotNavigation.keep_history` for ``plot``, or nothing if it has no navigation."""
    navigation = navigation_for(plot)
    if navigation is None:
        yield
    else:
        with navigation.keep_history():
            yield


# -- the menu ------------------------------------------------------------------------------------
def _populate_menu(menu: QtWidgets.QMenu, navigation: PlotNavigation | None, exports) -> None:
    menu.clear()
    if navigation is not None:
        reset = menu.addAction("Reset view", navigation.reset_view)
        reset.setToolTip("Zoom and pan back to the page's own framing (auto-range or your "
                         "typed axis range). Nothing is recomputed or restyled.")
        back = menu.addAction("Zoom back", navigation.view_back)
        back.setEnabled(navigation.can_go_back())
        forward = menu.addAction("Zoom forward", navigation.view_forward)
        forward.setEnabled(navigation.can_go_forward())
    entries = list(exports() if callable(exports) else exports)
    if entries and navigation is not None:
        menu.addSeparator()
    for entry in entries:
        title, callback = entry[0], entry[1]
        enabled = entry[2]() if len(entry) > 2 and callable(entry[2]) else True
        action = menu.addAction(title, lambda fn=callback: fn())
        action.setEnabled(bool(enabled))
    menu.setToolTipsVisible(True)


def _app_menu(plot: pg.PlotItem, navigation, exports) -> QtWidgets.QMenu:
    menu = getattr(plot, "_nanoscope_app_menu", None)
    if menu is None:
        views = plot.scene().views() if plot.scene() is not None else []
        menu = QtWidgets.QMenu(views[0] if views else None)
        menu.setObjectName("plotAppContextMenu")
        menu.setAccessibleName("Plot actions")
        plot._nanoscope_app_menu = menu
    _populate_menu(menu, navigation, exports)
    return menu


def install_plot_interaction(plot, *, readout: PlotReadout | None = None,
                             reset: Callable[[], None] | None = None,
                             exports: Sequence | Callable[[], Sequence] = (),
                             navigation: PlotNavigation | None = None,
                             x_name=None, y_name=None, x_log=None, y_log=None,
                             extra=None, points=None) -> PlotNavigation:
    """Give ``plot`` the app's Reset view / zoom history / menu / readout. Returns its navigation.

    ``navigation`` shares another plot's history instead of making one: the density
    marginals and Distributions' residual strip are linked to a main plot and mean the same
    view. ``exports`` is the page's own ``(title, callback[, enabled])`` list — the same verbs
    as its Export menu. The readout options are documented on :func:`readout_text`.
    """
    if isinstance(plot, pg.PlotWidget):
        plot = plot.getPlotItem()
    if navigation is None:
        navigation = PlotNavigation(plot, reset=reset, parent=plot.getViewBox())
    elif navigation.plot is not plot:
        # A linked side plot: a wheel or drag on it is a user step of the shared view.
        navigation.listen_to(plot)
    # Held ON the plot (not in a module registry, which would keep every closed page's
    # plot alive): the navigation lives and dies with the plot it navigates.
    plot._nanoscope_plot_navigation = navigation
    vb = plot.getViewBox()
    # THE GENERIC MENU GOES, ON THIS INSTANCE ONLY. "Plot Options" is the PlotItem's menu;
    # the ViewBox keeps its right-click handling (``mouseClickEvent`` needs menuEnabled) but
    # raises the app menu instead, and offers nothing to a child item's own menu, so neither
    # View All, the axis/mouse-mode entries, the log/FFT/invert transforms nor the scene's
    # Export... dialog can be reached any more.
    plot.setMenuEnabled(False, enableViewBoxMenu=True)

    def raise_menu(ev, plot=plot, navigation=navigation, exports=exports):
        menu = _app_menu(plot, navigation, exports)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    vb.raiseContextMenu = raise_menu
    vb.getContextMenus = lambda _event: []
    plot._nanoscope_menu_factory = lambda: _app_menu(plot, navigation, exports)
    if readout is not None and plot.scene() is not None:
        scene = plot.scene()
        hover = getattr(scene, "_nanoscope_hover", None)
        if hover is None:
            hover = _SceneHover(scene)
            scene._nanoscope_hover = hover
        hover.add(plot, readout, {"x_name": x_name, "y_name": y_name, "x_log": x_log,
                                  "y_log": y_log, "extra": extra, "points": points})
    return navigation


def app_menu_for(plot) -> QtWidgets.QMenu | None:
    """The app menu a right-click on ``plot`` would show now, freshly populated (tests)."""
    if isinstance(plot, pg.PlotWidget):
        plot = plot.getPlotItem()
    factory = getattr(plot, "_nanoscope_menu_factory", None)
    return factory() if factory is not None else None


# -- the toolbar button --------------------------------------------------------------------------
def view_button(resolve: Callable[[], PlotNavigation | None],
                navigations: Iterable[PlotNavigation] = (),
                parent: QtWidgets.QWidget | None = None) -> QtWidgets.QToolButton:
    """"Reset view" on a click; Zoom back / Zoom forward on its drop-down.

    ``resolve`` names the navigation of the plot on screen (the PSD's current view, the
    Event-waveforms tab), so one button serves a page with several plots. The drop-down's
    items are greyed out when there is nowhere to go, re-read every time it opens and
    whenever any of ``navigations`` changes.
    """
    button = QtWidgets.QToolButton(parent)
    button.setObjectName("plotResetViewButton")
    button.setText("Reset view")
    button.setAccessibleName("Reset plot view")
    button.setToolTip("Zoom and pan back to the page's own framing. The arrow holds Zoom back "
                      "and Zoom forward. View only: nothing is recomputed or restyled.")
    button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
    menu = QtWidgets.QMenu(button)
    back = menu.addAction("Zoom back")
    forward = menu.addAction("Zoom forward")
    button.setMenu(menu)

    def sync() -> None:
        nav = resolve()
        back.setEnabled(nav is not None and nav.can_go_back())
        forward.setEnabled(nav is not None and nav.can_go_forward())
        button.setEnabled(nav is not None)

    def run(name: str) -> None:
        nav = resolve()
        if nav is not None:
            getattr(nav, name)()
        sync()

    button.clicked.connect(lambda _checked=False: run("reset_view"))
    back.triggered.connect(lambda _checked=False: run("view_back"))
    forward.triggered.connect(lambda _checked=False: run("view_forward"))
    menu.aboutToShow.connect(sync)
    for nav in navigations:
        nav.historyChanged.connect(sync)
    button._nanoscope_sync = sync
    button._nanoscope_back = back
    button._nanoscope_forward = forward
    sync()
    return button
