"""What the Plots window draws on and beside the data: legend, annotations, selection.

``_PlotsAnnotationMixin`` is a part of :class:`nanoscope.gui.plots_panel.PlotsWindow`,
listed before ``QtWidgets.QMainWindow`` in its bases. It holds the fit label and the
legend (rows, names, placement and drag), the detection provenance caption, the fit's
confidence band and residual strip, the shaded detection floor and its label, the
message and Details writers, and the click and lasso selection of event points.

It is a plain-Python mixin: no ``__init__``, no signals and no state of its own, so every
attribute its methods read (``self._plot``, ``self._style`` ...) is set by
``PlotsWindow.__init__``. The constants and helpers it uses come from
:mod:`nanoscope.gui.plots_constants`; it never imports ``plots_panel`` or another part.

Every method below was moved out of ``plots_panel.py`` without an edit.
"""

from __future__ import annotations

import html as _html
import math
from collections.abc import Mapping
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..analysis import stats
from ..core.sources import PlotSource, disambiguate_labels
from .figure_text import css_args, figure_text_for, plain_text, rich_text, shown_text
from .fit_label import FitReadout, format_fit_label
from .font_picker import html_font_family
from .plot_appearance import apply_legend_family
from .plot_lasso import RowIdentityIndex, toggled_rows, unique_rows
from .plots_constants import (
    _LEGEND_DEFAULT_OFFSET,
    _WARN_HEX,
    _EXCLUDED_HEX,
    LIMIT_LABEL_KEY,
    _LIMIT_LABEL_AUTO,
    _legend_style_key,
    _LegendText,
    _FitLegendText,
    _LimitLabel,
)


class _PlotsAnnotationMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    def _shown_readout(self, key: str, readout: FitReadout, names: dict) -> FitReadout:
        """``readout`` as this figure shows it: the residual strip's number joined to the
        equation, the legend name as its source, and the fit colour where the source has
        none of its own."""
        parts = dict(readout.parts)
        resid = parts.pop("resid", "")
        if resid and self._style.show_residuals and parts.get("equation"):
            # The strip's number, on the figure, because the strip itself is not exported.
            parts["equation"] = f"{parts['equation']}   {resid}"
        return FitReadout(parts, readout.refusals, names.get(key) or None,
                          readout.colour or self._style.fit_color)

    def _fit_rows_in_legend(self) -> set:
        """Source keys whose fit row is in the drawn legend (In legend instead puts their
        numbers there; a fit with no drawn row keeps them in the label)."""
        if not self._style.show_legend:
            return set()
        return {key for entry_id, (_item, _label, kind, key)
                in getattr(self, "_presentation_legend", {}).items()
                if kind == "fit" and self._legend_included(entry_id)}

    def _fit_legend_label(self, item, name: str, key: str) -> str:
        """A fit's legend row: "Apo fit (linear)", with no numbers -- they are in the fit
        label, and a middle-elided "Apo fit: y = 1.96….971" helped nobody. With In legend
        instead the numbers come back, as a second line that is never elided."""
        family = getattr(item, "_presentation_fit_family", "")
        head = f"{name} fit".strip() if name else "fit"
        head = f"{head} ({family})" if family else head
        state = self._style.fit_label_state()
        if not state.in_legend:
            return head
        for rkey, readout in getattr(self, "_presentation_readouts", []):
            if rkey == key:
                lines = format_fit_label([self._shown_readout(rkey, readout, {})],
                                         _dc_replace(state, in_legend=False))
                if lines:
                    return _FitLegendText(head, plain_text(lines[0][0]))
        return head

    # -- E2: what made these events, ON the figure -----------------------------------------------
    #: How a detector id is written for a human. `provenance['detector']` is a module
    #: string ("schmitt_hysteresis", "cusum_sequential", "ml_keras"); an unknown one is
    #: printed VERBATIM rather than dropped, because a detector this table does not know
    #: is exactly the case where the reader most needs to be told which one ran.
    _DETECTOR_NAMES: "dict[str, str]" = {
        "schmitt_hysteresis": "Schmitt threshold",
        "cusum_sequential": "CUSUM",
        "ml_keras": "ML (Keras)",
        "reconciled": "Reconciled detectors",
    }

    @staticmethod
    def _preprocessing_chain(prov: Mapping) -> str:
        """The preprocessing stages that RAN, in the order the signal met them.

        THE STAGES THAT RAN, not the ones that were offered: every ``*_enabled`` key is a
        boolean the module stamped after deciding, so a chain printed from them describes
        this recording rather than the panel's defaults. An empty string means the raw
        signal went to the detector, which is itself worth being able to read off a figure.

        Each stage carries the ONE number that changes what it does -- a notch without its
        frequency and a high-pass without its corner are not reproducible descriptions --
        and nothing else, because this line has to fit under a plot.
        """
        parts: list[str] = []
        if prov.get("detrend_enabled"):
            win = prov.get("detrend_window_s")
            parts.append(f"detrend {prov.get('detrend_method', 'block_median')}"
                         + (f" {float(win):g} s" if win else ""))
        if prov.get("notch_enabled"):
            freqs = prov.get("notch_freqs")
            if freqs:
                parts.append(f"notch {', '.join(f'{float(f):g}' for f in freqs)} Hz")
            else:
                n_h = int(prov.get("notch_harmonics") or 1)
                parts.append(f"notch {float(prov.get('notch_freq_hz') or 0):g} Hz"
                             + (f" ×{n_h}" if n_h > 1 else ""))
        if prov.get("median_enabled"):
            parts.append(f"median {float(prov.get('median_window_s') or 0) * 1e3:g} ms")
        if prov.get("hp_enabled"):
            parts.append(f"high-pass {float(prov.get('hp_cutoff_hz') or 0):g} Hz")
        if prov.get("rmean_enabled"):
            parts.append(f"rolling mean {float(prov.get('rmean_window_s') or 0) * 1e3:g} ms")
        if prov.get("savgol_enabled"):
            parts.append(f"Savitzky–Golay {float(prov.get('savgol_window_s') or 0) * 1e3:g} ms")
        if prov.get("normalize_enabled"):
            parts.append(f"normalize {prov.get('normalize_mode', 'zscore')}")
        return " → ".join(parts)

    def _provenance_line(self, src: PlotSource) -> str:
        """One source's caption: detector, model, preprocessing. ``""`` when unknown.

        WHY THIS AND NOT ``_note_cache``. That cache holds deferred STATISTICS -- keyed by
        a sample digest and a censoring window, invalidated when the numbers move -- and
        provenance moves for none of those reasons: it is a property of the RUN that made
        the rows, already sitting on :attr:`PlotSource.provenance`, and it is stale only
        when the source is. Routing it through the notes machinery would give it an
        expiry rule it does not have and a worker it does not need.

        THE MODEL IS NAMED BY ITS FILE, not by its preset alone. The owner runs three
        differently-trained detectors over the same chemistry (HG, Zn, Cu), and they are
        distinguished by which ``.keras`` file was loaded -- the preset names the decode
        thresholds, not the weights. The basename is used because the full path is a home
        directory and does not belong in a published figure.
        """
        prov = dict(getattr(src, "provenance", None) or {})
        if not prov:
            return ("manual measurement table (no detector provenance)"
                    if getattr(src, "kind", "") == "manual" else "")
        det = str(prov.get("detector") or "")
        parts = [self._DETECTOR_NAMES.get(det, det) if det else "detector not recorded"]
        model = prov.get("model_path")
        if model:
            parts.append(f"model {Path(str(model)).name}")
        preset = prov.get("ml_preset")
        if preset:
            parts.append(f"preset {preset}")
        decode = prov.get("ml_decode_preset")
        if decode and str(decode) != "legacy":
            parts.append(f"decode {decode}")
        chain = self._preprocessing_chain(prov)
        parts.append(chain if chain else "no preprocessing")
        return " · ".join(p for p in parts if p)

    def _draw_provenance_caption(self) -> None:
        """Write each drawn source's provenance under the plot, INSIDE the figure.

        Bottom-left, because the other two corners are taken: the fit label owns the
        top-left and the legend the top-right. Parented to the ViewBox exactly as those
        are, which is what puts it in the exported image and keeps it still under pan and
        zoom -- and, like them, means it must be detached by hand in :meth:`_clear_plot`,
        since ``PlotItem.clear()`` cannot reach a child of the ViewBox.

        Silent when the plot is showing a MESSAGE rather than data: "No data yet" with a
        preprocessing chain under it describes nothing.

        Each line is led by the name the FIGURE calls its source (:meth:`_legend_name`,
        the one resolver), not the raw override: this caption is a plot item Save image
        keeps, and with two rows typed to "Apo" it read "Apo: manual measurement table"
        twice under a legend that had just told the two apart (A1-N2).
        """
        if self._message_item is not None:
            return
        drawn = self._drawn_frames()
        multi = len(drawn) > 1
        lines: list[str] = []
        for src, _df in drawn:
            line = self._provenance_line(src)
            if not line:
                continue
            self._presentation_provenance_by_source.append((src.key, line))
            lines.append(f"{self._legend_name(src)}: {line}" if multi else line)
        self._presentation_provenance = list(lines)
        if not self._style.show_provenance:
            return
        if not lines:
            return
        text = "<br>".join(_html.escape(l) for l in lines)
        item = pg.LabelItem(text, color=self._msg_hex(),
                            size=f"{max(6, self._style.font_size_pt - 2)}pt",
                            family=html_font_family(self._style.font_family),
                            justify="left")
        item.setParentItem(self._plot.vb)
        item.anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(8, -6))
        item.setZValue(30)
        self._prov_item = item

    # -- the fit's own uncertainty, and where it misses ------------------------------------------
    def _clear_residual_strip(self) -> None:
        """Empty the strip and take it off screen. Every draw starts here, so a strip can
        only ever show the residuals of the picture currently above it."""
        if not hasattr(self, "_resid"):
            return      # may run during construction, before _build_ui
        self._resid.clear()
        self._resid_widget.setVisible(False)

    def _draw_ci_band(self, fit, grid: np.ndarray, at: np.ndarray, scale, colour,
                      role="fit") -> str:
        """Shade the bootstrap's 95% pointwise band around the drawn fit curve.

        ``""`` when there is nothing to draw, which is the common case and is NOT an
        error: :func:`stats.ci_band` declines for a fit restored from a project, a fit
        made under ``point_fits_only``, a family with no bootstrap, and a bootstrap with
        too few usable replicates. A note is returned only when a band WAS drawn, because
        the alternative -- explaining the absence on every unfitted redraw -- is the
        clutter the switch exists to avoid.

        The band is put through the SAME ``expected`` arithmetic as the curve
        (``fit.n * density * width * _fit_scale``) rather than being scaled separately:
        the two must agree bin for bin, and one expression they both use is the only way
        to guarantee it under three normalizations and a log x axis.
        """
        band = stats.ci_band(fit, at)
        if band is None:
            return ""
        lo = np.asarray(band["lo"], dtype=np.float64) * scale
        hi = np.asarray(band["hi"], dtype=np.float64) * scale
        metadata = getattr(self._fit_items[-1], "_figure_series_meta", None) if self._fit_items else None
        self._presentation_bands.append((grid.copy(), lo.copy(), hi.copy(), colour,
                                         str(band["note"]), role, metadata))
        if not self._style.show_ci_band:
            return ""
        self._add_cached_band(grid, lo, hi, colour, role, metadata)
        return str(band["note"])

    def _add_cached_band(self, grid, lo, hi, colour, role="fit", metadata=None) -> None:
        """Materialize graphical ornaments from an already computed interval."""
        pen = pg.mkPen(QtGui.QColor(colour), width=1, style=Qt.DotLine)
        c = QtGui.QColor(colour)
        c.setAlpha(60)      # readable over the bars without hiding them
        top = pg.PlotDataItem(x=grid, y=hi, pen=pen)
        bot = pg.PlotDataItem(x=grid, y=lo, pen=pen)
        fill = pg.FillBetweenItem(top, bot, brush=pg.mkBrush(c))
        for it in (top, bot, fill):
            it._presentation_role = role
            it._presentation_band = True
            if metadata is not None:
                it._figure_series_meta = dict(metadata)
            self._plot.addItem(it, ignoreBounds=True)
        # NOT a legend row. The band is an attribute of the fit curve that is already
        # legended, and a second entry for one object is the note-Q2 complaint the legend
        # checklist exists to fix.
        self._band_items.extend([top, bot, fill])

    def _draw_residual_strip(self, fit, sample, edges: np.ndarray,
                             log_x: bool) -> tuple[str, str]:
        """Standardised residuals per bin under the main axes, and their reduced χ².

        WHAT THIS ANSWERS THAT KS DOES NOT. The KS verdict is one number over the whole
        curve: it says the fit is or is not consistent, never where it fails. A run of
        same-sign bars here is the finding -- a gamma fit that is high through the peak
        and low through the tail is a wrong FAMILY, and looks nothing like the scatter of
        a right one, whatever either one's D happens to be.

        Returns ``(sentence, short)`` -- the full χ² paragraph for Details and the
        one-token form for the fit label, which is the half an exported PNG
        keeps. Both ``""`` when nothing could be computed: a family whose density will not
        integrate over these bins loses its strip, exactly as it loses its KS verdict, and
        never raises into a draw.
        """
        self._clear_residual_strip()
        try:
            res = stats.binned_residuals(fit, sample, edges)
        except (ValueError, RuntimeError, TypeError, ZeroDivisionError,
                FloatingPointError):
            return "", ""
        self._presentation_residual = (res, bool(log_x))
        fit.meta["saved_residual_geometry"] = {
            "result": {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in res.items()}, "log_x": bool(log_x)}
        self._presentation_residual_note = str(res["note"])
        self._presentation_residual_ready = True
        short = ("" if res["reduced_chi2"] is None
                 else f"χ²/dof = {float(res['reduced_chi2']):.2f} (approx. dof)")
        if not self._style.show_residuals:
            return "", short
        self._render_cached_residual(res, log_x)
        return str(res["note"]), short

    def _render_cached_residual(self, res, log_x) -> None:
        """Draw existing binned residuals; no sample or fit is re-evaluated."""
        r = np.asarray(res["residual"], dtype=np.float64)
        x = np.asarray(res["centers"], dtype=np.float64)
        colour = QtGui.QColor(self._style.fit_color)
        # Stems, not a connected line: each residual belongs to ONE bin and joining them
        # draws a curve through numbers that have no order beyond their bins.
        raw = res.get("mode") == "raw"
        if not raw:
            for xi, ri in zip(x, r):
                stem = pg.PlotDataItem(x=[xi, xi], y=[0.0, float(ri)],
                                      pen=pg.mkPen(colour, width=self._style.line_width))
                stem._presentation_role = "fit"
                self._resid.addItem(stem)
        points = pg.PlotDataItem(
            x=x, y=r, pen=None, symbol="o", symbolSize=max(4, self._style.point_size - 2),
            symbolBrush=pg.mkBrush(colour), symbolPen=None)
        points._presentation_role = "fit"
        self._resid.addItem(points)
        # ±2 is where a standardised residual stops being ordinary counting noise; the
        # guides are what turn "a tall bar" into "a tall bar for this many counts".
        guides = ((0.0, Qt.SolidLine),) if raw else ((0.0, Qt.SolidLine), (2.0, Qt.DashLine), (-2.0, Qt.DashLine))
        for level, style in guides:
            self._resid.addItem(pg.InfiniteLine(
                pos=level, angle=0,
                pen=pg.mkPen(self._msg_hex(), width=1, style=style)))
        self._resid.setLogMode(x=bool(log_x), y=False)
        residual_label = (f"y−fit ({res.get('unit') or 'unitless'})" if raw else "(obs−exp)/√exp")
        self._resid.setLabel("left", residual_label, **self._label_style())
        self._resid.getAxis("bottom").setStyle(showValues=False)   # the plot above labels it
        self._style_residual_axes()
        # Symmetric about zero and never tighter than ±3, so the guides are always in
        # frame and two strips of the same data are comparable by eye.
        span = max(1e-12 if raw else 3.0, float(np.max(np.abs(r))) * 1.15) if r.size else 3.0
        self._resid_view_span = span
        self._resid.setYRange(-span, span, padding=0)
        self._resid_widget.setVisible(True)
        # "approx." is not decoration: the dof came from an unbinned MLE, and a bare
        # "χ²/dof = 1.3" on an exported figure is read as the textbook quantity.

    def _style_residual_axes(self) -> None:
        """Apply the saved appearance whenever cached residuals are displayed.

        Native restoration does not replay a Style gesture. Keep axis ink/fonts
        and guide colors identical there, without refitting or resetting views.
        """
        self._resid_widget.setBackground(self._bg_spec())
        for side in ("bottom", "left"):
            axis = self._resid.getAxis(side)
            axis.setPen(pg.mkPen(self._fg_hex()))
            axis.setTextPen(pg.mkPen(self._fg_hex()))
            axis.setStyle(tickFont=self._qfont(self._tick_pt()))
            axis.setLabel(axis.labelText, **self._label_style())
        for item in self._resid.items:
            if isinstance(item, pg.InfiniteLine):
                pen = QtGui.QPen(item.pen)
                pen.setColor(QtGui.QColor(self._msg_hex()))
                item.setPen(pen)

    def _drop_legend(self) -> None:
        """Empty the legend and hide it.

        ``PlotItem.clear()`` empties the ViewBox but leaves ``self.legend`` in place
        holding its stale rows, and ``addLegend`` then returns that same item -- so
        without this the entries of every previous draw accumulate forever. The item
        is emptied rather than destroyed: ``LegendItem.setParentItem(None)`` re-runs
        its anchoring and raises "Cannot anchor; parent is not set", so detaching one
        is a trap. Hidden while empty, because an empty box floating over a
        single-series plot is noise.
        """
        legend = getattr(self._plot, "legend", None)
        if legend is None:
            return
        legend.clear()
        legend.setVisible(False)

    def _ensure_legend(self):
        """The legend, created if needed and re-inked for the CURRENT theme.

        Restyled on every draw because its text colour and translucent backing have
        to follow the dark/light switch the Style group offers -- a legend that stays
        light-grey on the white publication canvas is not a legend, it is a smudge.
        """
        fg = QtGui.QColor(self._fg_hex())
        backing = QtGui.QColor(self._bg_spec())
        backing.setAlpha(200)     # readable over bars/points without hiding them
        border = QtGui.QColor(fg)
        border.setAlpha(90)
        existing = getattr(self._plot, "legend", None)
        legend = self._plot.addLegend(offset=_LEGEND_DEFAULT_OFFSET)
        if existing is None:
            # First legend of this window's life. ``PlotItem.addLegend`` returns the
            # EXISTING item untouched from here on, so this is the one chance to hook
            # the drag -- and the hook is how a dragged position gets remembered.
            self._install_legend_drag(legend)
            # The in-figure text controller was installed before any legend existed;
            # its rows become editable (double-click, right-click) once it is told.
            controller = figure_text_for(self._plot)
            if controller is not None:
                controller.refresh()
        self._apply_legend_position(legend)
        legend.setLabelTextColor(fg)
        legend.setBrush(pg.mkBrush(backing))
        legend.setPen(pg.mkPen(border))
        # Three points BELOW the body font, floored at 6, with tight padding. A legend
        # is an INDEX, not content: at font-1 with the default 10 pt body it rendered
        # large enough to cover a meaningful part of the plot area, and what it covers
        # is the data it names. That rule is the DEFAULT (legend_font_size_pt = 0), not
        # a law: a figure going into a talk needs a legend that reads from the back of
        # the room, and one going into a dense multi-panel needs it smaller still.
        pt = (float(self._style.legend_font_size_pt) or
              max(6, self._style.font_size_pt - 3))
        legend.setLabelTextSize(f"{max(1, pt)}pt")
        try:
            legend.layout.setContentsMargins(2, 2, 2, 2)
            legend.layout.setHorizontalSpacing(4)
            legend.layout.setVerticalSpacing(0)
        except Exception:      # pragma: no cover - pyqtgraph layout internals
            pass
        legend.setVisible(True)
        return legend

    def _legend_row_added(self, legend) -> None:
        """A row was just added: draw the legend's rows in the figure family, then put the
        legend where the user left it. The second step is needed because the family changes
        the width the row was measured at, and a legend anchored by its far edge (a dragged
        one is, past the middle of the plot) keeps that edge, not the left one."""
        apply_legend_family(legend, self._style.font_family)
        self._apply_legend_position(legend)

    # -- legend PLACEMENT: the user's, dragged, and kept (round-4 Q4) -----------------------------
    def _install_legend_drag(self, legend) -> None:
        """Record where the user drops the legend, every time they drop it.

        pyqtgraph's ``LegendItem`` has been draggable since long before this window
        existed -- ``LegendItem.mouseDragEvent`` calls ``autoAnchor`` on each mouse
        move -- so the FEATURE was never missing; what was missing was memory of it,
        and what actively destroyed it was a ``setOffset`` on every draw (see
        :data:`_LEGEND_DEFAULT_OFFSET`). This wraps the real handler rather than
        replacing it: pyqtgraph still does the dragging, and the wrapper's only job is
        to read the result back afterwards.

        Wrapped on the INSTANCE, which works because pyqtgraph dispatches these itself
        in Python (``GraphicsScene`` calls ``item.mouseDragEvent(ev)``), so the
        instance attribute shadows the class method the way a normal attribute lookup
        does.
        """
        base = legend.mouseDragEvent

        def _dragged(ev, _base=base):
            _base(ev)
            self._remember_legend_position()

        legend.mouseDragEvent = _dragged

    def _remember_legend_position(self) -> None:
        """Store the legend's current spot as a fraction of the plot area.

        Read off the live item rather than out of the drag event, so it is the position
        pyqtgraph actually settled on -- ``autoAnchor`` picks the nearest corner and
        may land a pixel away from the raw pointer delta.
        """
        legend = getattr(self._plot, "legend", None)
        parent = legend.parentItem() if legend is not None else None
        if parent is None:
            return
        br = parent.boundingRect()
        if br.width() <= 0 or br.height() <= 0:
            return
        pos = legend.pos()
        self._style = _dc_replace(
            self._style, legend_pos=(pos.x() / br.width(), pos.y() / br.height()))
        self._sync_legend_reset_button()

    def _apply_legend_position(self, legend) -> None:
        """Put the legend where the user left it -- or, if they never moved it, where
        this window has always started it.

        THE RE-APPLY IS SAFE TO RUN ON EVERY DRAW, which is the fact the four-corner
        combo got wrong. ``autoAnchor(pos, relative=True)`` is idempotent: measured on
        the owner's two-source dwell overlay, five successive re-applies of a dragged
        (287.0187, 192.0) all landed on exactly (287.0187, 192.0). ``setOffset`` is
        not -- it re-anchors to a CORNER, which is why it threw the drag away (measured:
        drag to (287, 192), next redraw put it back at (486.7, 12)).
        """
        parent = legend.parentItem()
        if parent is None:      # pragma: no cover - addLegend always parents to the vb
            return
        pos = self._style.legend_pos
        if pos is None:
            # Never dragged: the historical top-right corner. Applied every draw is
            # harmless here precisely because there is no user position to destroy.
            legend.setOffset(_LEGEND_DEFAULT_OFFSET)
            return
        br = parent.boundingRect()
        if br.width() <= 0 or br.height() <= 0:   # pragma: no cover - un-laid-out view
            return
        legend.autoAnchor(pg.Point(pos[0] * br.width(), pos[1] * br.height()),
                          relative=True)

    def _reset_legend_position(self) -> None:
        """Send the legend back to the default corner (the Style group's button).

        The four-position combo this replaces is what the owner rejected -- "the
        location should not be limited to four fixed positions ... it should be
        interactive". A RESET is a different thing from a position setting: it does not
        run on every draw, so it cannot fight the drag, and it answers the one real
        need a free drag creates, which is getting back from a legend dropped somewhere
        useless.
        """
        self._style = _dc_replace(self._style, legend_pos=None)
        self._sync_legend_reset_button()
        self._presentation_changed()

    def _sync_legend_reset_button(self) -> None:
        """Enable Reset only when there is something to reset -- the button doubles as
        the only on-screen statement that the legend has been moved at all."""
        if not hasattr(self, "_legend_reset_btn"):
            return   # fires during construction, before the Style group exists
        moved = self._style.legend_pos is not None
        self._legend_reset_btn.setEnabled(moved)
        self._legend_reset_btn.setToolTip(
            "Put the legend back in the top-right corner." if moved else
            "The legend is where it started — drag it on the plot to move it.")

    #: Longest legend entry drawn in full. A legend is sized by its widest label, and a
    #: source label is a RECORDING FILENAME -- the owner's run to a hundred characters
    #: ("500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.csv"), which made the
    #: legend span most of the plot width no matter how small the font was set. Shrinking
    #: the text alone could never fix that; the width comes from the string.
    _LEGEND_LABEL_CHARS = 34

    @classmethod
    def _elide_legend(cls, label: str) -> str:
        """Middle-elide a long legend label. The middle goes because the ends carry the
        identity -- the leading concentration/sample and the trailing voltage/`n` are what
        tell two of this owner's recordings apart; the centre is boilerplate they share.
        The full string stays in Details, which is where the unabridged provenance lives."""
        if len(label) <= cls._LEGEND_LABEL_CHARS:
            return label
        keep = cls._LEGEND_LABEL_CHARS - 1
        head, tail = (keep + 1) // 2, keep // 2
        return f"{label[:head]}…{label[-tail:]}"

    def _legend_label(self, src: PlotSource | None) -> str:
        """What the legend calls ``src``: the user's own text if they retyped it.

        Owner note #15, "the legend is not editable". A source's label is a RECORDING
        FILENAME — "500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.csv" —
        and no figure caption in a paper says that; it says "apo, −500 mV". The
        override is edited in place on the source list (double-click a row) and stored
        per source KEY, so it survives a Refresh that re-pulls the same recordings.
        """
        if src is None:
            return ""
        return self._style.label_overrides.get(src.key, src.label) or src.label

    def _legend_names(self) -> dict[str, str]:
        """What the DRAWN legend calls each ticked source: :meth:`_legend_label`, made
        unique among the sources on the figure.

        THE ONE RESOLVER FOR THE FIGURE'S NAMES, and the reason it is not
        :meth:`_legend_label` itself. Source LABELS are unique by construction
        (:func:`nanoscope.core.sources.disambiguate_labels` runs in discovery), but an
        OVERRIDE is free text, and two rows can be typed to one name -- deliberately, or
        by the stranded-override route S7-SK2 measured: rename -700 mV to "Apo", type
        the original back (the row obeyed, the override did not -- fixed in
        :meth:`_on_source_item_changed`), then give "Apo" to -600 mV, and the canvas and
        the exported SVG both read ['Apo (n = 982)', 'Apo (n = 236)'] with no warning
        anywhere. A figure with two curves under one name cannot be read, and Save
        image exports exactly the legend drawn here.

        SO A COLLISION IS TOLD APART ON THE FIGURE, by the same function the source
        list uses, with the source's own label as the distinguishing name ("Apo —
        -700 mV" / "Apo — -600 mV"): it is the one name guaranteed unique per source, it
        is what the row's tooltip already shows, and it lets a reader of the exported
        figure see which recording is which without the window. The ROWS keep showing
        what was typed (``_legend_label``): they are the rename affordance, and writing
        the disambiguated form back onto them would be read as a new override by the
        next tick and would move the user's text under them. That the figure differs
        from the rows is said in Details (:meth:`_legend_collision_notes`).

        Cheap: a handful of ticked sources, a Counter and a dict, per legend entry.
        """
        sources = self.selected_sources()
        entries = [{"label": self._legend_label(src), "channel_name": src.label,
                    "is_segment": bool(src.is_segment)} for src in sources]
        disambiguate_labels(entries)
        return {src.key: entry["label"] for src, entry in zip(sources, entries)}

    def _legend_name(self, src: PlotSource | None) -> str:
        """The legend entry's name for ``src`` on THIS figure -- see :meth:`_legend_names`.
        Falls back to :meth:`_legend_label` for a source that is not ticked (a fit drawn
        for the primary source on a single-series plot)."""
        if src is None:
            return ""
        return self._legend_names().get(src.key) or self._legend_label(src)

    def _legend_collision_notes(self) -> list[str]:
        """One sentence per legend name shared by two or more ticked sources, or ``[]``.

        The disclosure half of :meth:`_legend_names`: the figure has been drawn with the
        names told apart, and the user has to be told that it differs from what they
        typed and which row to retype -- in Details, the pane every other refusal is
        said in and the one that is on screen without scrolling.
        """
        sources = self.selected_sources()
        if len(sources) < 2:
            return []
        drawn = self._legend_names()
        groups: dict[str, list[PlotSource]] = {}
        for src in sources:
            groups.setdefault(self._legend_label(src), []).append(src)
        out: list[str] = []
        for label, group in groups.items():
            if len(group) < 2:
                continue
            which = ", ".join(src.label for src in group)
            told_apart = ", ".join(f"“{drawn[src.key]}”" for src in group)
            out.append(
                f"LEGEND: {len(group)} of the drawn sources are both called “{label}” "
                f"({which}), so the legend and any saved image name them {told_apart} "
                f"— two curves under one name cannot be read. Retype one of them in "
                f"the Sources list to name them apart.")
        return out

    def _legend_entry(self, label: str, n: int | None = None) -> str:
        """A legend entry, with the automatic ``(n = N)`` only when it is wanted.

        The count is genuinely useful on an overlay and it is today's default, but it
        must be defeatable: silently welding "  (n = 986)" onto a label the user has
        just RETYPED overrides the edit they made in the same breath as honouring it.
        """
        text = label if n is None or not self._style.legend_counts else f"{label}  (n = {n})"
        return _LegendText(text, label, n)

    @staticmethod
    def _legend_id(kind: str, key: str = "") -> str:
        """The stable id one legend entry is remembered by: ``"<kind>:<key>"``.

        KIND, NOT LABEL, because the label is the volatile part -- it carries the fitted
        tau, the n, and whatever the user retyped, so keying inclusion on it would lose
        the setting the moment a number moved. ``series`` and ``fit`` are keyed by the
        SOURCE key for the same reason ``label_overrides`` is: a refresh re-identifies
        sources by key, and an inclusion decision must not slide onto another curve.
        """
        return f"{kind}:{key}"

    def _legend_included(self, entry_id: str) -> bool:
        """Whether this entry is in the legend. Absent means yes -- an empty
        ``legend_include`` is today's look."""
        return bool(self._style.legend_include.get(entry_id, True))

    def _legend_add(self, item, label: str, *, kind: str = "series", key: str = "") -> None:
        """Offer one entry to the legend, subject to the two inclusion gates.

        THE OFFER IS RECORDED WHETHER OR NOT IT IS DRAWN. ``_legend_offered`` is what
        the Style group's *Legend entries* checklist is rebuilt from, and a checklist
        that only listed the entries currently shown would be a list you could untick
        rows out of and never get back.

        Owner note Q2 -- "when the fit is performed and the legend is selected, the app
        includes icons for both the histogram and the lines. This is incorrect; the
        user needs control over which icons to add to the legend."
        """
        entry_id = self._legend_id(kind, key)
        self._presentation_legend[entry_id] = (item, label, kind, key)
        self._legend_offered[entry_id] = label
        # The one gate for "Show legend" (owner request): every entry flows through
        # here, so unticking the box empties the next draw's legend everywhere at once.
        if not self._style.show_legend or not self._legend_included(entry_id):
            return
        legend = self._ensure_legend()
        legend.addItem(item, self._legend_markup(label, _legend_style_key(kind, key)))
        self._legend_row_added(legend)
        legend.setVisible(True)

    def _legend_markup(self, label: str, element: str = "") -> str:
        """A legend row as Qt rich text: elided, then drawn LITERALLY (a source renamed
        "a<b>" lost its middle), with ``_{…}``/``^{…}`` as sub/superscripts. A fit row
        carrying its numbers (In legend instead) puts them on a second line, unelided.

        ``element`` is the row's Text style key (``legend.entry.…``): its font, size,
        weight and colour wrap the row, over the legend's own ink and size. An unstyled
        row stays plain text."""
        if isinstance(label, _FitLegendText):
            text = f"{rich_text(self._elide_legend(label.head))}<br>{rich_text(label.numbers)}"
        else:
            text = rich_text(self._elide_legend(str(label)))
        css = css_args(self._style.effective_text_style(element)) if element else {}
        if not css:
            return text
        return "<span style='{}'>{}</span>".format(
            "; ".join(f"{name}: {value}" for name, value in css.items()), text)

    def _shade_below_limit(self, t_min: float, lo: float, log_x: bool) -> None:
        """Shade the region the detector could never report, as figures.py does.

        ``figures.dwell_histogram`` shades ``[left, t_min]`` and drops a dashed
        line on the limit because the histogram's left edge is an artefact of the
        detector, not of the physics -- and that is exactly why the fit is a
        censored MLE. The interactive histogram must not be the one view that
        hides it. Coordinates are converted to log10 by hand: ``setLogMode``
        transforms plot DATA items, not items added straight to the scene.
        """
        left = min(t_min, lo) * 0.5
        if not (t_min > left > 0):
            return
        to_view = (lambda x: float(np.log10(x))) if log_x else float
        shade = QtGui.QColor(_EXCLUDED_HEX)
        shade.setAlphaF(0.30)   # the alpha figures.py's axvspan uses
        self._limit_region = pg.LinearRegionItem(
            values=(to_view(left), to_view(t_min)), movable=False,
            brush=pg.mkBrush(shade), pen=pg.mkPen(None))
        self._limit_region.setZValue(-10)
        self._plot.addItem(self._limit_region)
        self._limit_line = pg.InfiniteLine(
            pos=to_view(t_min), angle=90,
            pen=pg.mkPen(_EXCLUDED_HEX, width=1, style=Qt.DashLine))
        # The words are figure text (``annotation.limit``): retyped, styled or hidden from
        # the figure, so the label is this window's own InfLineLabel, made here rather
        # than by ``InfiniteLine(label=…)``. The line and the shading stay whatever the
        # words say: they are the detector's floor, not a caption.
        self._limit_line.label = _LimitLabel(self._limit_line, text=_LIMIT_LABEL_AUTO,
                                             position=0.06, color=_EXCLUDED_HEX,
                                             movable=False)
        self._plot.addItem(self._limit_line)
        self._draw_limit_label()
        controller = figure_text_for(self._plot)
        if controller is not None:
            controller.refresh()        # a new label each draw: hook its double-click

    def _draw_limit_label(self) -> None:
        """The detection-floor line's words as the figure has them: automatic, typed (drawn
        literally, ``_{…}``/``^{…}`` as sub/superscripts) or hidden, in their Text style
        over the excluded-region grey."""
        line = getattr(self, "_limit_line", None)
        label = getattr(line, "label", None) if line is not None else None
        if not isinstance(label, _LimitLabel):
            return
        style = self._style.effective_text_style(LIMIT_LABEL_KEY)
        text = shown_text(style, _LIMIT_LABEL_AUTO)
        css = {"color": _EXCLUDED_HEX, "font-family": html_font_family(self._style.font_family),
               **css_args(style)}
        label.show_rich("<span style='{}'>{}</span>".format(
            "; ".join(f"{name}: {value}" for name, value in css.items()), rich_text(text))
            if text.strip() else "")

    def _message(self, text: str) -> None:
        """Blank the plot and show a friendly centred note (empty/invalid state).

        DETAILS CARRIES THE REFUSALS TOO, and the centred note does not. This method is
        the ONE funnel for every early return in every draw path -- "no data yet", "tick
        a source", "pick a numeric column", "pick numeric X and Y", "no points for this
        pair", "select ≥2 feature columns, then click Run PCA" -- and it replaced Details
        wholesale, so a window in any of those states said nothing about a configuration
        it had just been handed and could not honour. Fixing it HERE rather than at the
        fourteen call sites is the point: an early return that skips the disclosure was
        the bug, not any one page, and a fifteenth empty state arrives disclosed.

        MEASURED BOTH WAYS on all four pages, by unticking every source so that each one
        takes the shared "tick a source" early return with a refused config installed:
        before this line the sentence was absent on 4 of 4, after it present on 4 of 4.

        THE CENTRED TEXT ITEM IS LEFT ALONE deliberately. It is drawn INTO the figure and
        goes into an exported image; the instruction it carries is what the user has to do
        next, and a paragraph about somebody's file printed across the canvas is neither
        that nor something anyone wants in a saved PNG. Details is where the window's own
        account of itself already lives and it is on screen without scrolling even with
        both sentences in it -- measured on a 1200x780 window: the pane is 803x180 px, its
        ``visibleRegion()`` is non-empty at rest and its vertical scrollbar range is 0..0,
        where the same probe over the same text plus 40 filler lines reports 0..37.
        """
        self._clear_plot()
        self._apply_background()
        self._plot.setLogMode(x=False, y=False)
        self._plot.setLabel("bottom", "")
        self._plot.setLabel("left", "")
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=self._msg_hex())
        self._plot.addItem(item)
        self._plot.setXRange(0, 1, padding=0)
        self._plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)
        self._message_item = item
        self._apply_fonts()
        self._set_details("\n".join([text, *self._incoming_notes()]))

    def _set_details(self, text: str) -> None:
        # Keep the short verdict as well as the full explanation outside the
        # figure. This same Details text is retained by figure-bundle provenance.
        headline = getattr(self, "_title_presentation", (None, None, None))[2]
        if headline and headline not in text:
            text = f"{text}\n\n{headline}"
        self._details.setPlainText(text)
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        self._result_summary.setText(lines[0] if lines else "No result yet.")

        # One actionable warning, not a second copy of the report. Explicit warning
        # markers win; otherwise take the first line that describes a material omission
        # or invalidation. The complete wording remains one click away in Details.
        warning = next((line for line in lines
                        if line.startswith(("!", "⚠")) or "NOT COMPARABLE" in line.upper()), "")
        if not warning:
            markers = (" failed", " unavailable", " dropped", " ignored", " refused",
                       " exact by construction", " could not", " cannot ",
                       " not certified")
            warning = next((line for line in lines
                            if any(marker in f" {line.lower()}" for marker in markers)), "")
        if len(warning) > 280:
            warning = warning[:277].rstrip() + "…"
        self._result_warning.setText(warning)
        self._result_warning.setVisible(bool(warning))

    @staticmethod
    def _same_row_identity(left: object, right: object) -> bool:
        """Equality for pandas index scalars without leaking array-valued comparisons."""
        try:
            equal = left == right
            return bool(equal) if np.ndim(equal) == 0 else bool(np.all(equal))
        except (TypeError, ValueError):
            return left is right

    def _make_event_points_clickable(
        self, item: pg.ScatterPlotItem, source_key: str, row_ids: Sequence[object],
    ) -> None:
        """Attach exact source-row identities to a plotted Scatter/PCA point cloud."""
        identities = list(row_ids)
        if len(identities) != len(item.points()):
            raise RuntimeError("point/event identity count mismatch")
        # ``setData`` would replace x/y; assigning through the spots preserves their
        # coordinates and lets ``SpotItem.data()`` be the sole identity seam.
        for point, row_id in zip(item.points(), identities):
            point.setData(row_id)
        item.setCursor(Qt.PointingHandCursor)
        item.setToolTip(
            "Click to inspect this event; Ctrl/Cmd/Shift-click to select several.")
        item.sigClicked.connect(
            lambda clicked, points, event, key=source_key:
                self._on_event_points_clicked(key, clicked, points, event))
        self._event_point_items.setdefault(source_key, []).append(item)

    @staticmethod
    def _point_identities(src: PlotSource, df: pd.DataFrame,
                          mask: np.ndarray | None = None) -> list[object]:
        """Stable identities for plotted rows, in their exact plotted order.

        Detector ``event_id`` is the table/trace navigation key. A dataframe index is
        only a label and is explicitly allowed to be duplicated or NaN by EventSet, so
        using it could make one point select another row. Bare/manual frames keep their
        index because they have no detector event to navigate to.
        """
        if src.kind == "detection" and "event_id" in df.columns:
            values = df["event_id"].to_numpy(copy=False)
        else:
            values = df.index.to_numpy(copy=False)
        if mask is not None:
            values = values[np.asarray(mask, dtype=bool)]
        return list(values)

    def _on_event_points_clicked(self, source_key: str, _item: pg.ScatterPlotItem,
                                 points: Sequence[object], event: object | None) -> None:
        """Maintain an additive visual selection and emit its exact dataframe rows."""
        clicked = [point.data() for point in points]
        if not clicked:
            return
        modifiers = (event.modifiers() if event is not None
                     and hasattr(event, "modifiers") else Qt.NoModifier)
        self._apply_event_row_selection(source_key, clicked, modifiers)

    def _apply_event_row_selection(self, source_key: str, rows: Sequence[object],
                                    modifiers=Qt.NoModifier) -> None:
        """The same one-source selection semantics for point clicks and lasso."""
        additive = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier | Qt.ShiftModifier))
        if additive:
            # MainWindow can display one detector table at a time. Leaving source A
            # ringed while emitting only source B would show a selection that was not
            # represented in the table or signal, so additive selection is deliberately
            # scoped to one source.
            selected = toggled_rows(self._selected_event_rows.get(source_key, ()), rows,
                                     self._same_row_identity)
        else:
            selected = unique_rows(rows, self._same_row_identity)
        self._selected_event_rows = {source_key: selected}
        self._paint_event_point_selection()
        self.eventRowsSelected.emit(source_key, tuple(selected))

    def _on_lasso_mode_changed(self, enabled: bool) -> None:
        if hasattr(self, "_lasso_action"):
            with QtCore.QSignalBlocker(self._lasso_action):
                self._lasso_action.setChecked(enabled)
            self._selection_mode_label.setText(
                "Mode: Lasso selection" if enabled else "Mode: Pan/zoom")
        if enabled:
            self.statusBar().showMessage(
                "Drag an outline to select events. Ctrl/Cmd/Shift toggles; Esc/right-click cancels. "
                "No event curation or analysis is changed.", 9000)

    def _lasso_hits(self, polygon: QtGui.QPolygonF) -> dict[str, list[object]]:
        """Read exact displayed point identities, including already-log-transformed axes."""
        if len(polygon) < 3:
            return {}
        bounds = polygon.boundingRect()
        if not all(math.isfinite(value) for value in
                   (bounds.left(), bounds.top(), bounds.right(), bounds.bottom())):
            return {}
        hit_bounds = bounds.adjusted(-.5, -.5, .5, .5)
        path = QtGui.QPainterPath()
        path.addPolygon(polygon)
        path.closeSubpath()
        border = QtGui.QPainterPathStroker()
        border.setWidth(1.0)
        boundary = border.createStroke(path)
        clip = self._plot.getViewBox().sceneBoundingRect()
        hits = {}
        for key, items in self._event_point_items.items():
            rows = []
            for item in items:
                if not item.isVisible():
                    continue
                for point in item.points():
                    if not point.isVisible() or point.size() <= 0:
                        continue
                    position = item.mapToScene(point.pos())
                    if (math.isfinite(position.x()) and math.isfinite(position.y())
                            and clip.contains(position)
                            and hit_bounds.contains(position)
                            and (path.contains(position) or boundary.contains(position))):
                        rows.append(point.data())
            if rows:
                hits[key] = unique_rows(rows, self._same_row_identity)
        return hits

    def _choose_lasso_source(self, hits: Mapping[str, Sequence[object]]) -> str | None:
        """The event table can represent one source, so mixed regions ask explicitly."""
        keys = list(hits)
        if len(keys) == 1:
            return keys[0]
        names = self._committed_legend_names()
        labels = [f"{names.get(key, key)} — {len(hits[key]):,} events" for key in keys]
        label, accepted = QtWidgets.QInputDialog.getItem(
            self, "Choose lasso source",
            "The outline contains multiple sources. Select one source for the event table:",
            labels, 0, False)
        return keys[labels.index(label)] if accepted and label in labels else None

    def _on_lasso_completed(self, polygon, modifiers=Qt.NoModifier) -> None:
        generation = self._lasso_generation
        hits = self._lasso_hits(polygon)
        if not hits:
            additive = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier | Qt.ShiftModifier))
            if not additive:
                old_sources = tuple(self._selected_event_rows)
                self._selected_event_rows.clear()
                self._paint_event_point_selection()
                for key in old_sources:
                    self.eventRowsSelected.emit(key, ())
            self.statusBar().showMessage("No plotted events inside the lasso.", 6000)
            return
        key = self._choose_lasso_source(hits)
        if key is None:
            self.statusBar().showMessage("Lasso selection cancelled; selection is unchanged.", 6000)
            return
        if self._lasso_generation != generation:
            self.statusBar().showMessage(
                "The plot changed while choosing a source. Draw a new lasso.", 7000)
            return
        self._apply_event_row_selection(key, hits[key], modifiers)
        self.statusBar().showMessage(
            f"Selected {len(self._selected_event_rows.get(key, ())):,} events from one source. "
            "Selection only; curation and scientific results are unchanged.", 7000)

    def _paint_event_point_selection(self) -> None:
        """Ring inside the existing spot footprint, preserving auto-range padding."""
        base_size = self._style.point_size
        ring_width = min(2, base_size)
        ring = pg.mkPen(_WARN_HEX, width=ring_width)
        selected_size = base_size - (max(math.ceil(ring_width), 1) - 1)
        for source_key, items in self._event_point_items.items():
            selected = RowIdentityIndex(self._selected_event_rows.get(source_key, ()),
                                         self._same_row_identity)
            for item in items:
                sizes = getattr(item, "_presentation_series_sizes", None)
                for number, point in enumerate(item.points()):
                    size = base_size if sizes is None else float(sizes[number])
                    on = size > 0 and selected.contains(point.data())
                    # Order matters: pyqtgraph's symbol atlas retains its largest
                    # footprint. Never transiently grow it when toggling a ring.
                    if on:
                        point.setSize(max(0., size - (max(math.ceil(ring_width), 1) - 1)))
                        point.setPen(ring)
                    else:
                        point.setPen(pg.mkPen(None))
                        point.setSize(size)
