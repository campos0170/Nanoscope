from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.session import Session
from ..core.types import ChannelKind
from .trace_view import effective_kind

__all__ = ["Navigator", "LOCAL_CONTEXT_FACTOR"]


# At a deep zoom the full-recording strip is not a controller any more: on a 300 s trace a
# 2 ms view is a few thousandths of a pixel wide.  Nine view-widths keeps the draggable window
# at 1/9 of the strip (about 50--100 px in the supported layouts) while leaving four neighboring
# windows on either side.  Once nine windows cover the recording the navigator automatically
# becomes the ordinary full-recording overview again.
LOCAL_CONTEXT_FACTOR = 9.0


class Navigator(pg.PlotWidget):
    """Adaptive overview/local-context strip + a draggable main-view region.

    At broad zooms this is the original full-recording map.  At event-scale zooms it becomes a
    local lens spanning :data:`LOCAL_CONTEXT_FACTOR` main-view widths, which keeps the shaded
    viewport physically grabbable no matter how long the recording is.  Drag the region body to
    pan at the current zoom or drag an edge to zoom; the lens freezes while that gesture is in
    flight and recentres only when it finishes.

    Binds to the selected annotation target when it is visible. Standalone use falls back to the
    first visible CURRENT-kind layer, then the first visible layer of any kind (see
    :meth:`_overview_layer`). The curve uses peak downsampling plus clip-to-view; clipping is a
    no-op in full mode and avoids painting the rest of a multi-million-sample trace outside the
    bounded lens in local mode.

    The region<->view binding IS guarded (self._syncing): a programmatic view->region update must not
    bounce back and drive the view. Without the guard, a view change propagates to the region, and if
    the region's bounds are momentarily stale (e.g. the instant a new file replaces one with a smaller
    time extent) setRegion CLAMPS the value and feeds the clamped range back, silently truncating the
    main view to the previous file's extent. Only a genuine user drag of the region drives the view.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._syncing = False
        self.setMaximumHeight(90)
        self.setMinimumHeight(64)
        self.setMenuEnabled(False)
        plot = self.getPlotItem()
        plot.hideAxis("left")
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setLabel("bottom", "")

        self._session: Session | None = None
        self._view = None
        self._curve: pg.PlotDataItem | None = None
        self._target: tuple[str, int] | None = None
        self._extent: tuple[float, float] | None = None
        self._context: tuple[float, float] | None = None
        self._local_mode = False
        self._view_width: float | None = None
        self._dragging = False
        self._drag_resized = False

        # A child label overlays the plot instead of taking another layout row.  It is both the
        # mode disclosure and the beginner-facing explanation for why the scale changes after a
        # deep zoom.  Transparent mouse handling leaves every pixel available to the region.
        self._mode_label: QtWidgets.QLabel

        self._region = pg.LinearRegionItem(brush=(80, 130, 170, 60),
                                            hoverBrush=(80, 130, 170, 90))
        self._region.setZValue(10)
        self.addItem(self._region)
        self._region.hide()
        self._region.sigRegionChanged.connect(self._region_to_view)
        self._region.sigRegionChangeFinished.connect(self._region_finished)

        # Read-only mirror of the working Measure cursors. The amber/grey region controls the
        # VIEW; this blue band describes the EVENT being annotated. Keeping it non-movable avoids
        # two overlapping range controllers while still showing where the saved/working span sits
        # as the user scans to its neighbors.
        self._measure_band = pg.LinearRegionItem(
            movable=False,
            brush=pg.mkBrush(33, 150, 243, 34),
            pen=pg.mkPen("#42A5F5", width=1),
        )
        self._measure_band.setZValue(15)
        self.addItem(self._measure_band, ignoreBounds=True)
        self._measure_band.hide()

        # A direct child of the PlotWidget cannot affect its size hint.
        self._mode_label = QtWidgets.QLabel("Full recording", self)
        self._mode_label.setObjectName("navigatorModeLabel")
        self._mode_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._mode_label.setStyleSheet(
            "QLabel { color:#d7dde5; background:rgba(20,24,30,175); "
            "border-radius:3px; padding:1px 5px; font-size:10px; }")
        self._mode_label.adjustSize()
        self._mode_label.move(6, 4)
        self.setToolTip(
            "Navigation strip. At broad zooms it shows the full recording. At event scale it "
            "automatically shows nearby waveform context so the shaded main-view window remains "
            "wide enough to grab. Drag the shaded body to pan; drag either edge to zoom.")

    def bind(self, session: Session, view) -> None:
        """Wire to Session (for data + change signals) and to the TraceView (for the visible-range
        two-way binding). `view` is duck-typed — Navigator never imports the TraceView class."""
        self._session = session
        self._view = view
        session.layersChanged.connect(self._refresh)
        session.viewModeChanged.connect(self._refresh)
        session.documentClosed.connect(lambda _doc_id: self._refresh())
        session.calibrationChanged.connect(lambda _d, _c: self._refresh())
        session.sampleRateChanged.connect(self._on_sample_rate_changed)
        view.sigVisibleRangeChanged.connect(self._view_to_region)
        if hasattr(view, "sigMeasureChanged"):
            view.sigMeasureChanged.connect(lambda *_a: self._sync_measure_band())
        if hasattr(view, "sigMeasureFinished"):
            view.sigMeasureFinished.connect(lambda *_a: self._sync_measure_band())
        if hasattr(view, "sigMeasureActiveChanged"):
            view.sigMeasureActiveChanged.connect(lambda *_a: self._sync_measure_band())
        if hasattr(view, "sigMeasureCleared"):
            # Measure is ON with no mask: no bounds signal can say that, and the band would
            # otherwise stay drawn over the event that was just added or discarded.
            view.sigMeasureCleared.connect(lambda *_a: self._sync_measure_band())
        self._refresh()

    def set_target(self, doc_id: str, channel: int) -> None:
        """Prefer the recording/channel the application is currently annotating.

        The old first-visible-current fallback remains for standalone use and for a target that is
        not drawn.  In an overlay this explicit target prevents the local lens from showing a
        neighboring recording while the Measure cursors belong to this one.
        """
        target = (str(doc_id), int(channel))
        if target == self._target:
            return
        self._target = target
        self._refresh()

    def context_range(self) -> tuple[float, float] | None:
        """The x-range currently displayed by this strip (test/export-friendly accessor)."""
        return self._context

    def is_local_mode(self) -> bool:
        """Whether the strip is showing bounded neighboring context rather than the full trace."""
        return self._local_mode

    # -- internals -------------------------------------------------------------------------------
    def _overview_layer(self):
        """The layer the overview draws: the visible annotation target when one is named,
        otherwise the first visible CURRENT-kind layer, then the first visible layer of any kind.

        Plain first-visible was wrong on the primary format: ABF channel order puts the command
        voltage BEFORE the current, and a -1000 mV command line is flat — a "where am I" map with
        no events on it. Preferring current keeps the standalone strip useful there; the fallback
        keeps voltage-only recordings from losing the strip.
        """
        if self._session is None:
            return None
        if self._target is not None:
            doc_id, channel = self._target
            layer = self._session.layer(f"{doc_id}#{channel}")
            if (layer is not None and layer.visible
                    and self._session.has_document(layer.doc_id)):
                return layer
        fallback = None
        for layer in self._session.layers():
            if not (layer.visible and self._session.has_document(layer.doc_id)):
                continue
            # The SAME kind resolution the panes rank by (trace_view.effective_kind):
            # a channel corrected to pA in the Calibration panel must be the one this
            # strip prefers, or the navigator summarises the command voltage while the
            # top pane shows the current — the two-surfaces-disagreeing complaint again.
            kind = effective_kind(self._session, layer.doc_id, layer.channel)
            if kind is ChannelKind.CURRENT:
                return layer
            if fallback is None:
                fallback = layer
        return fallback

    def _make_curve(self, color: str) -> pg.PlotDataItem:
        item = pg.PlotDataItem()
        self.addItem(item)
        item.setDownsampling(auto=True, method="peak")
        # In local mode this prevents a multi-million-point full trace being painted outside the
        # nine-window lens. At full extent clipping is a no-op, so one setting covers both modes.
        item.setClipToView(True)
        item.setPen(pg.mkPen(color, width=1))
        return item

    def _on_sample_rate_changed(self, doc_id: str) -> None:
        """A document's time base was re-declared (``Session.set_sample_rate_declaration``).

        The strip keeps a copy of the recording's x-axis that only :meth:`_refresh` re-reads: its
        curve is plotted against the axis it was handed, and its region's bounds and its lens
        are cut from the extent read off the same axis. The trace re-frames on this signal and
        the scan bar, File Info and the status bar follow it; the strip stayed on the old clock,
        so the view it mirrors and the strip it is mirrored on disagreed. Measured on a CSV
        whose column is in milliseconds (read as 50 Hz, from t = 10,000 s) after declaring the
        milliseconds: 'Local context · 9× view' over a 2 s recording, none of its 100,000
        samples inside the strip's own x-range, and a drag of the shaded window to 13–15 s took
        the trace there too — past the last sample, a blank pane. 'Use the file's own time
        base', a rate-only declaration and an axis in samples declared in microseconds are the
        same defect.

        Re-fed from the Session, not from the trace view: the curve and the extent come off
        ``Session.x_axis``, and the view's range reaches the strip through
        ``sigVisibleRangeChanged``, so it does not matter whether this slot runs before the
        trace view's own (which forgets the old-clock zoom and re-frames) or after it. Run
        first, the region is seeded from a range that is about to be replaced and the re-frame
        replaces it; run second, it reads the new one. A recording that is not drawn changes
        nothing the strip shows (the trace view makes the same call), and with nothing open
        there is nothing to re-feed.
        """
        if self._session is None or doc_id not in self._session.shown_doc_ids():
            return
        self._refresh()

    def _refresh(self) -> None:
        if self._curve is not None:
            self.removeItem(self._curve)
            self._curve = None

        layer = self._overview_layer()
        if layer is None or self._session is None:
            self._extent = None
            self._context = None
            self._local_mode = False
            self._view_width = None
            self._set_mode_label("No recording")
            self._region.hide()
            self._measure_band.hide()
            return

        xaxis = self._session.x_axis(layer.doc_id, layer.time_offset_s)
        chan = self._session.calibrated(layer.doc_id, layer.channel)
        if len(xaxis.values) == 0:
            self._extent = None
            self._context = None
            self._local_mode = False
            self._view_width = None
            self._set_mode_label("No samples")
            self._region.hide()
            self._measure_band.hide()
            return

        self._curve = self._make_curve(layer.color)
        self._curve.setData(x=xaxis.values, y=chan.data)

        # Once MainWindow has named an annotation target, every navigation control uses THAT
        # recording's plotted extent. Otherwise a longer overlay lets this region move the main
        # trace into a range where the selected recording (and its Measure cursors) do not exist,
        # while the scan bar correctly stops earlier. Standalone Navigator use retains the visible
        # union fallback.
        selected_layer = (self._session.layer(f"{self._target[0]}#{self._target[1]}")
                          if self._target is not None else None)
        if selected_layer is not None and selected_layer.visible:
            selected_x = self._session.x_axis(
                selected_layer.doc_id, selected_layer.time_offset_s).values
            self._extent = ((float(selected_x[0]), float(selected_x[-1]))
                            if len(selected_x) >= 2 and selected_x[-1] > selected_x[0] else None)
        else:
            self._extent = self._view.full_x_extent() if self._view is not None else None
        if self._extent is None:
            x0, x1 = float(xaxis.values[0]), float(xaxis.values[-1])
            self._extent = (x0, x1) if x1 > x0 else None
        if self._extent is None:
            self._region.hide()
            self._set_mode_label("No range")
            return
        self._syncing = True                 # bounds+seed are programmatic: must not drive the view
        # Seed the region from the main view's current x-range so the two start in agreement.
        if self._view is not None:
            anchor = self._view.anchor_plot()
            if anchor is not None:
                lo, hi = anchor.vb.viewRange()[0]
                self._set_controller_bounds(float(lo), float(hi))
                self._set_context(float(lo), float(hi), force=True)
                self._region.setRegion((lo, hi))
            else:
                self._region.setBounds(list(self._extent))
        self._syncing = False
        self._region.show()
        self._sync_measure_band()

    @QtCore.Slot(float, float)
    def _view_to_region(self, xmin: float, xmax: float) -> None:
        # The view is the source of truth; mirror it onto the region WITHOUT bouncing back.
        if self._dragging:
            # This emission is the synchronous echo of the region gesture already under the
            # pointer. Reframing the lens here would make the target chase the mouse.
            return
        self._syncing = True
        self._set_controller_bounds(float(xmin), float(xmax))
        self._set_context(float(xmin), float(xmax))
        self._region.setRegion((xmin, xmax))
        self._syncing = False

    def _region_to_view(self) -> None:
        if self._view is None or self._syncing:
            return                            # ignore our own programmatic updates; only user drags pass
        self._dragging = True
        lo, hi = self._region.getRegion()
        lo, hi = float(lo), float(hi)
        anchor = self._view.anchor_plot()
        current = tuple(anchor.vb.viewRange()[0]) if anchor is not None else (lo, hi)
        old_width = float(current[1]) - float(current[0])
        new_width = hi - lo
        resized = abs(new_width - old_width) > max(abs(old_width), abs(new_width), 1e-15) * 1e-6
        self._drag_resized = self._drag_resized or resized
        # A body drag is a pure pan. Edge dragging is a zoom and should re-fit y just like the
        # main plot's box/wheel routes do.
        if resized and hasattr(self._view, "zoom_x_to"):
            self._view.zoom_x_to(lo, hi)
        else:
            self._view.set_x_range(lo, hi)

    def _region_finished(self) -> None:
        if self._syncing or self._view is None:
            return
        self._dragging = False
        anchor = self._view.anchor_plot()
        if anchor is None:
            return
        lo, hi = anchor.vb.viewRange()[0]
        # Re-size of an edge always re-derives a nine-window lens. A body pan keeps the existing
        # local neighborhood until the viewport approaches its edge, then recentres once.
        self._syncing = True
        self._set_controller_bounds(float(lo), float(hi))
        self._set_context(float(lo), float(hi), force=self._drag_resized)
        self._region.setRegion((float(lo), float(hi)))
        self._syncing = False
        self._drag_resized = False

    def _controller_extent(self, xmin: float, xmax: float) -> tuple[float, float] | None:
        """Target data extent plus only the padding already present in the main viewport.

        Event jumps deliberately use symmetric context, so a first/last event can put half of the
        main view beyond the first/last sample. The region must mirror that view exactly. This
        envelope admits the existing padding without granting a Navigator drag any NEW off-data
        room; after a drag lands inside the target it contracts to the target again.
        """
        if self._extent is None:
            return None
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        e0, e1 = self._extent
        return min(e0, float(lo)), max(e1, float(hi))

    def _set_controller_bounds(self, xmin: float, xmax: float) -> None:
        extent = self._controller_extent(xmin, xmax)
        if extent is not None:
            self._region.setBounds(list(extent))

    def _set_context(self, xmin: float, xmax: float, *, force: bool = False) -> None:
        """Choose full overview or a stable nine-view local lens around ``[xmin, xmax]``."""
        extent = self._controller_extent(xmin, xmax)
        if extent is None:
            return
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        width = hi - lo
        e0, e1 = extent
        full = e1 - e0
        if width <= 0.0 or full <= 0.0:
            return

        # Wheel, box, +/- and event jumps are zoom changes, not pans. Rebuild the lens around the
        # new scale even when the new viewport still happens to fit inside the old context.
        previous_width = self._view_width
        if (previous_width is not None
                and abs(width - previous_width)
                > max(abs(width), abs(previous_width), 1e-15) * 1e-6):
            force = True
        self._view_width = width

        desired = min(full, width * LOCAL_CONTEXT_FACTOR)
        local = desired < full * (1.0 - 1e-6)

        keep = False
        if local and not force and self._local_mode and self._context is not None:
            c0, c1 = self._context
            # Keep the lens still while ordinary pans stay comfortably inside it. A half-view
            # margin means the viewport is never left with an ungrabbable edge at the strip edge.
            # The old lens must also fit the CURRENT controller extent: once a padded view moves
            # back over data, neither its blank context nor its drag allowance may linger.
            keep = (c0 >= e0 and c1 <= e1
                    and lo >= c0 + 0.5 * width and hi <= c1 - 0.5 * width)

        if not local:
            c0, c1 = e0, e1
        elif keep:
            c0, c1 = self._context  # type: ignore[misc]
        else:
            centre = 0.5 * (lo + hi)
            c0 = centre - 0.5 * desired
            c1 = c0 + desired
            if c0 < e0:
                c0, c1 = e0, e0 + desired
            if c1 > e1:
                c0, c1 = e1 - desired, e1

        self._context = (float(c0), float(c1))
        self._local_mode = bool(local)
        self.getPlotItem().setXRange(float(c0), float(c1), padding=0)
        self.getPlotItem().vb.enableAutoRange(y=True)
        self._set_mode_label(
            f"Local context · {LOCAL_CONTEXT_FACTOR:g}× view" if local else "Full recording")

    def _set_mode_label(self, text: str) -> None:
        label = self._mode_label
        if hasattr(label, "setText"):
            label.setText(text)
            label.adjustSize()
            label.raise_()

    def _sync_measure_band(self) -> None:
        if self._view is None or not hasattr(self._view, "measure_x_range"):
            self._measure_band.hide()
            return
        rng = self._view.measure_x_range()
        if rng is None or rng[1] <= rng[0]:
            self._measure_band.hide()
            return
        self._measure_band.setRegion((float(rng[0]), float(rng[1])))
        self._measure_band.show()
