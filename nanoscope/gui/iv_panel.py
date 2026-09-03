"""The standalone I-V / conductance window: the classic pore-characterisation plot.

Opened from ``Analysis ▸ I-V / Conductance…``, this window gathers one
``(V, I_open)`` point from every open recording that has a current channel and a
known voltage (:func:`nanoscope.analysis.iv.iv_points`), scatters open-pore
current against applied voltage, overlays the least-squares I-V line, and reports
the fitted **conductance** (the slope, in nS), the **leak** (the intercept), R²
and the number of recordings. It is the aggregate counterpart to the Calibration
panel's per-recording read-out: dial each recording's current gain there, then
come here to see the whole -1000..-200 mV series fall on one line.

Design mirrors :class:`~nanoscope.gui.plots_panel.PlotsWindow`: a single,
lazily-built top-level window owned by :class:`~nanoscope.gui.main_window.MainWindow`,
rendered with pyqtgraph so it inherits the app's dark theme. Unlike the plots
window it needs no per-event tables pushed in -- its data is the open documents
themselves -- so :meth:`refresh` re-gathers straight from the
:class:`~nanoscope.core.session.Session` (documents may have been opened since),
and a *Refresh* button and *Save image* export sit alongside the plot.

The fit runs in **volts** (so the slope is a conductance in ``current_units/V``
that converts cleanly to nS). Recordings that share a current unit are fit in it
directly; a mixed-unit set is converted to nA first -- through
:func:`nanoscope.analysis.iv.current_scale`, the module's own table, never a copy
of it kept here -- with a note, and anything genuinely unconvertible is dropped
and named as dropped everywhere it is listed.

UNITS ARE ONLY EVER REPORTED, NEVER INVENTED, and a refusal names the thing that
is actually blocking: the exclusion list's gate is the VOLTAGE units alone
(:func:`~nanoscope.analysis.iv.voltage_units_note`), so the row and the headline
say the same word, and the current units -- undeclared on every fresh HDF5/CSV --
cost the nS conductance, not the point.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..analysis import iv, stats
from ..core.session import Session

#: Free parameters in the I-V line: (slope, intercept). What
#: :func:`nanoscope.analysis.stats.r2_dof_tag` subtracts from the recording count to
#: decide whether the printed R² has any residual degrees of freedom left to be
#: evidence over -- the SAME constant, through the SAME writer, that the
#: voltage-dependence panel and the Plots scatter already use for their own lines.
#: Two recordings is the minimum this window asks for ("open another at a different
#: voltage to fit a conductance"), and at two points a straight line passes through
#: both whatever they are: R² = 1 is arithmetic, and identical for any two numbers
#: whatsoever, noise included. The plot title is what the PNG export rasterises, so
#: an unqualified "R² = 1.0000" is what was leaving the app on a figure.
_IV_FIT_PARAMS = 2

#: THE RECIPROCAL HALF OF Medium-8. The event table's Summary row and its ``baseline_pA``
#: column tip both point AT this window and say which number to quote where; this window
#: said nothing back, so a reader who starts here still calls its I₀ "the open pore" with
#: no indication that the depths in the event table were measured against a different one.
#: A cross-reference that runs one way is half a cross-reference.
#:
#: The two are computed differently on purpose: this is ONE I₀ for the whole recording
#: from `analysis.iv`'s own estimator, and the table's is the median of the per-event
#: stamped baseline under that detection's `baseline_fingerprint`. On the owner's
#: apoferritin recordings they differ by up to 5.1%.
_I0_CROSS_REFERENCE = (
    "The event table reports an open-pore current too, and it is NOT this number: its "
    "`baseline_pA` column is the per-event open pore stamped by the baseline estimator "
    "under that detection's baseline_fingerprint, and the Summary shows its median. On "
    "the owner's apoferritin recordings the two differ by up to 5.1% (−49.13 here vs "
    "−51.78 pA stamped, at −700 mV). Quote THIS one beside a conductance; quote the "
    "table's beside a depth from that table."
)
from ..plot.style import OKABE_ITO
from .export_paths import ask_export_path

__all__ = ["IVWindow"]

_DATA_HEX = OKABE_ITO[0]        # blue -- the (V, I) points
_FIT_HEX = OKABE_ITO[1]         # vermillion -- the I-V line
_MSG_HEX = "#8b9199"
_POINT_ALPHA = 200


def _fmt_q(value: float, units: str) -> str:
    """A measured quantity with the units it ACTUALLY has.

    ``""`` means the file never declared them and the user has not set any (the
    state every fresh HDF5/CSV channel is in) -- say so, in the Calibration panel's
    own words (``_fmt_voltage`` there), instead of printing a bare number the reader
    will read in whatever unit the rest of the line happens to be in. One wording
    for every quantity this window prints, current and voltage alike.
    """
    return f"{value:.4g} {units}" if units else f"{value:.4g} (units unknown)"


class IVWindow(QtWidgets.QMainWindow):
    """Scatter open-pore current vs applied voltage and fit the pore conductance.

    Fed nothing directly: it reads the session's open documents on every
    :meth:`refresh`. The owner builds it once, shows it, and calls
    :meth:`refresh`; the window's own *Refresh* button does the same, so opening
    another recording and clicking Refresh brings it into the fit.
    """

    def __init__(self, session: Session, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._session = session
        self.setWindowTitle("NanoScope — I-V / Conductance")
        self.resize(820, 640)

        self._points: list[iv.IVPoint] = []
        self._exclusions: list[iv.IVExclusion] = []
        self._fit: iv.IVFit | None = None
        self._scatter_item: pg.ScatterPlotItem | None = None
        self._fit_item: pg.PlotDataItem | None = None
        self._message_item: pg.TextItem | None = None

        self._build_ui()
        self.refresh()

    # -- construction ----------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        bar = QtWidgets.QHBoxLayout()
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip(
            "Re-gather (V, I₀) from every open recording — pick up any opened since.")
        self._refresh_btn.clicked.connect(self.refresh)
        self._save_btn = QtWidgets.QPushButton("Save image…")
        self._save_btn.setToolTip("Export the I-V plot as a PNG.")
        self._save_btn.clicked.connect(lambda: self.save_image())
        bar.addWidget(self._refresh_btn)
        bar.addWidget(self._save_btn)
        bar.addStretch(1)
        self._headline = QtWidgets.QLabel("")
        self._headline.setStyleSheet("font-weight:600; font-size:14px;")
        bar.addWidget(self._headline)
        root.addLayout(bar)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(Qt.Vertical)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumSize(480, 340)
        self._plot: pg.PlotItem = self._plot_widget.getPlotItem()
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        splitter.addWidget(self._plot_widget)

        details_box = QtWidgets.QGroupBox("Fit details")
        dlay = QtWidgets.QVBoxLayout(details_box)
        dlay.setContentsMargins(6, 6, 6, 6)
        self._details = QtWidgets.QPlainTextEdit()
        self._details.setReadOnly(True)
        self._details.setMaximumHeight(180)
        self._details.setStyleSheet("font-family: Menlo, monospace; font-size: 11px;")
        dlay.addWidget(self._details)
        splitter.addWidget(details_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)

    # -- data ------------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-gather the I-V points from the session's open documents and redraw.

        Also gathers the refused recordings -- a measured voltage whose units the
        file never declared cannot go on a volts axis, but the refusal must be
        shown with its value and reason, not swallowed (the Calibration panel
        refuses the same claim for the same document; CF-5)."""
        self._points = iv.iv_points(self._session)
        self._exclusions = iv.iv_exclusions(self._session)
        self._redraw()

    def points(self) -> list[iv.IVPoint]:
        return list(self._points)

    def exclusions(self) -> list[iv.IVExclusion]:
        return list(self._exclusions)

    def fit(self) -> iv.IVFit | None:
        return self._fit

    def plot_item(self) -> pg.PlotItem:
        return self._plot

    def details_text(self) -> str:
        return self._details.toPlainText()

    # -- unit resolution -------------------------------------------------------------------------
    @staticmethod
    def _resolve_units(
        points: list[iv.IVPoint],
    ) -> tuple[np.ndarray, np.ndarray, str, list[str], list[iv.IVPoint]]:
        """(V_volts, I_current, units, notes, dropped) for a fit: one current unit.

        Recordings sharing a current unit fit in it directly; a mixed set is
        converted to nA so the points live on one physical axis (a point whose
        units are unrecognised is dropped, with a note, and comes back in
        ``dropped`` so every surface downstream can say which recordings the fit is
        actually standing on).

        THE CONVERSION IS :func:`iv.current_scale`, not a table of this window's own.
        This method used to carry ``{"pA": 1e-3, "nA": 1.0, "uA": 1e3}``, which is
        ``iv._I_TO_A`` minus ``"A"`` -- so a channel the user had declared in amperes
        (a free-text Target-units field; the Calibration panel prints "A / mV -> nS"
        and a conductance for it, and ``iv.to_nS`` converts it) was dropped here as
        "unknown", but ONLY when some other open recording happened to be in pA. One
        declaration must not get two verdicts from two surfaces of one app (JHT-03).
        """
        notes: list[str] = []
        unit_set = {p.current_units for p in points}
        if len(unit_set) == 1:
            u = points[0].current_units
            V = np.array([p.v_volts for p in points], dtype=np.float64)
            I = np.array([p.i_open for p in points], dtype=np.float64)
            return V, I, u, notes, []

        # Convert every point's open-pore current to nA so mixed units share one physical axis.
        V_list, I_list = [], []
        dropped: list[iv.IVPoint] = []
        for p in points:
            factor = iv.current_scale(p.current_units, "nA")
            if factor is None:
                notes.append(f"{p.label}: current units “{p.current_units or '?'}” unknown — dropped")
                dropped.append(p)
                continue
            V_list.append(p.v_volts)
            I_list.append(p.i_open * factor)
        notes.append("mixed current units — converted to nA for the fit")
        return (np.array(V_list, dtype=np.float64), np.array(I_list, dtype=np.float64),
                "nA", notes, dropped)

    # -- rendering -------------------------------------------------------------------------------
    def _clear(self) -> None:
        self._plot.clear()
        self._plot.setTitle(None)
        self._scatter_item = self._fit_item = self._message_item = None

    def _message(self, text: str) -> None:
        self._clear()
        self._fit = None
        self._headline.setText("")
        item = pg.TextItem(text, anchor=(0.5, 0.5), color=_MSG_HEX)
        self._plot.addItem(item)
        self._plot.setLabel("bottom", "")
        self._plot.setLabel("left", "")
        self._plot.setXRange(0, 1, padding=0)
        self._plot.setYRange(0, 1, padding=0)
        item.setPos(0.5, 0.5)
        self._message_item = item
        self._details.setPlainText(text)

    @staticmethod
    def _exclusion_line(e: "iv.IVExclusion") -> str:
        """One refused recording, value first and reason attached.

        Same shape as a Points line, but the voltage keeps its honest units --
        "(units unknown)" when the file declared none, exactly the wording the
        Calibration panel uses for the same read-out -- and the reason is
        `iv.voltage_units_note`'s, phrased as the Calibration panel's voltage
        clause, so the two surfaces cannot drift apart AND the row names the same
        box as the headline above it."""
        return (f"  {e.label}:  V = {_fmt_q(e.v_value, e.v_units)},  "
                f"I₀ = {_fmt_q(e.i_open, e.current_units)}"
                f"  [measured] — {e.reason}")

    def _redraw(self) -> None:
        pts = self._points
        if not pts:
            if self._exclusions:
                # Not "no usable recordings" -- there ARE recordings, and the real
                # cause is that their measured voltage carries no declared units.
                # Name it, show the values, and say where the fix lives.
                lines = ["No recording has a voltage in declared units — cannot place "
                         "points on a volts axis or fit a conductance in nS."]
                lines += [self._exclusion_line(e) for e in self._exclusions]
                lines.append("Set the voltage units in Calibration, then Refresh.")
                self._message("\n".join(lines))
                return
            self._message("No usable recordings.\nOpen a recording with a current channel and a "
                          "known voltage, then Refresh.")
            return

        V, I, units, notes, dropped = self._resolve_units(pts)
        finite = np.isfinite(V) & np.isfinite(I)
        V, I = V[finite], I[finite]
        if V.size == 0:
            self._message("No finite (V, I₀) points to plot.")
            return

        self._clear()
        point_color = QtGui.QColor(_DATA_HEX)
        point_color.setAlpha(_POINT_ALPHA)
        self._scatter_item = pg.ScatterPlotItem(
            x=V, y=I, size=11, pen=pg.mkPen("#0d0f12"), brush=pg.mkBrush(point_color))
        self._plot.addItem(self._scatter_item)
        self._plot.setLabel("bottom", "Applied voltage (V)")
        # An axis whose unit is blank reads "I₀ ()" and invites the reader to supply
        # the one they expect. Name the state instead -- the same words the read-outs use.
        self._plot.setLabel("left", f"Open-pore current I₀ ({units or 'units unknown'})")
        # THE OTHER OPEN PORE. Report Medium-8: two surfaces print an "open-pore
        # current" for one recording and they are not the same number -- this window's
        # per-recording I₀ against the event table's median stamped `baseline_pA`
        # (-49.13 vs -51.78 pA at -700 mV, 5.1%; identical at -600 mV, 4.6% apart at
        # -500 mV). The event table's tooltips were made to point here in the first
        # pass; this is the return direction, without which a reader who starts on this
        # window still has no signal that a second number exists.
        self._plot.getViewBox().setToolTip(_I0_CROSS_REFERENCE)

        # ``units or '(units unknown)'`` like every other line in this pane: a fresh
        # HDF5/CSV declares no current units, and this header read "current in " —
        # a sentence that stopped after "in" beside five lines that say the named state.
        details: list[str] = [f"I-V over {V.size} recording(s), current in "
                              f"{units or '(units unknown)'}"]
        self._fit = None
        if V.size >= 2:
            try:
                fit = iv.iv_fit(V, I, current_units=units)
            except ValueError as exc:
                details.append(f"Fit unavailable: {exc}")
                self._headline.setText("")
                self._plot.autoRange()   # the scatter still stands; frame it
                self._finish_details(details, notes, pts, units, dropped)
                return
            self._fit = fit
            self._draw_fit_line(V, fit)
            details.extend(self._fit_summary(fit))
            details.extend(self._chord_lines(pts, dropped, fit))
        else:
            # "Open another recording" is advice the user cannot act on when they
            # HAVE another one open and this window threw it out: the count that
            # reached the fit and the count on screen must be reconciled by naming
            # what was dropped and what would clear it.
            details.append(self._single_point_advice(dropped, len(pts)))
            self._headline.setText("")

        self._plot.autoRange()
        self._finish_details(details, notes, pts, units, dropped)

    def _draw_fit_line(self, V: np.ndarray, fit: iv.IVFit) -> None:
        vmin, vmax = float(np.min(V)), float(np.max(V))
        # Extend the line to the origin so the leak (intercept at V=0) is visible.
        lo, hi = min(vmin, 0.0), max(vmax, 0.0)
        xs = np.linspace(lo, hi, 200)
        self._fit_item = pg.PlotDataItem(x=xs, y=fit.predict(xs), pen=pg.mkPen(_FIT_HEX, width=2))
        self._plot.addItem(self._fit_item)

        # The degrees-of-freedom qualifier rides WITH the number, in the title the PNG
        # keeps -- the same short tag the voltage-dependence window puts beside its own
        # two-parameter fit, from the same writer, so one session cannot give two
        # verdicts to two windows about the identical n = 2 fit.
        dof_tag = stats.r2_dof_tag(fit.n, _IV_FIT_PARAMS)
        r2_text = f"R² = {fit.r2:.4f}" + (f" ({dof_tag})" if dof_tag else "")
        if fit.conductance_nS is not None:
            # "G" ALONE IS AMBIGUOUS ON THIS SCREEN. The Calibration panel prints an
            # "Open-pore conductance" for one recording at a time (G = I₀ / V, the
            # chord through the origin) and this one is the SLOPE of the line through
            # the whole series, which absorbs the leak; on the owner's three
            # recordings they differ by 1.71x. Say which one this is, here and in the
            # headline, and reconcile them in the details pane (_CHORD_CROSS_REFERENCE).
            self._plot.setTitle(
                f"G (I-V slope) = {fit.conductance_nS:.4g} nS   ·   {r2_text}",
                color=_FIT_HEX)
            self._headline.setText(f"G (I-V slope) = {fit.conductance_nS:.4g} nS")
        else:
            # No nS is available because the current units are unknown -- so the
            # slope that IS shown must not read as a bare "/V" either.
            u = fit.current_units or "(units unknown)"
            self._plot.setTitle(
                f"slope = {fit.slope:.4g} {u}/V   ·   {r2_text}", color=_FIT_HEX)
            self._headline.setText(f"slope = {fit.slope:.4g} {u}/V")

    @staticmethod
    def _fit_summary(fit: iv.IVFit) -> list[str]:
        # "current" reads like a unit; the state has a name and every other read-out
        # in this window uses it.
        u = fit.current_units or "(units unknown)"
        g = ("conductance not in nS (unknown current units)" if fit.conductance_nS is None
             else f"G = {fit.conductance_nS:.5g} nS")
        # ...and the full sentence goes on the Goodness line, exactly as the
        # voltage-dependence panel does it: the title carries the short tag (its space
        # is real), the pane carries the reason.
        dof_note = stats.r2_dof_caveat(fit.n, _IV_FIT_PARAMS, model="line",
                                       noun="recording")
        return [
            f"Conductance (slope):  {g}",
            f"                      slope = {fit.slope:.5g} {u}/V",
            f"Leak (intercept):     {fit.intercept:.5g} {u} at 0 V",
            f"Goodness of fit:      R² = {fit.r2:.5f}   ·   n = {fit.n} recording(s)"
            + (f"\n                      {dof_note}" if dof_note else ""),
        ]

    @staticmethod
    def _chord_lines(pts: list[iv.IVPoint], dropped: list[iv.IVPoint],
                     fit: iv.IVFit) -> list[str]:
        """Reconcile THIS window's conductance with the Calibration panel's.

        THE OTHER CONDUCTANCE (VS-07), and the exact twin of ``_I0_CROSS_REFERENCE``
        one line up the same screen. For one pore the app prints two numbers both
        called an open-pore conductance:

        * the Calibration panel's ``G = I₀ / V`` box -- the CHORD conductance of ONE
          recording, which assumes the I-V line passes through the origin; and
        * this window's fitted SLOPE -- one number for the whole series, which lets
          the line have a leak at 0 V and absorbs that leak into nothing.

        On the owner's own three apoferritin recordings they are 0.0562 / 0.0580 /
        0.0702 nS against 0.1053 nS -- 1.71x apart, driven by a +25.8 pA intercept
        that is 92 % of the smallest open-pore current in the series and of the
        opposite sign. Neither surface named the other, so a reader took whichever
        they saw first. This block prints the per-bias numbers the Calibration panel
        would show, their spread, and the leak that separates them.
        """
        used = [p for p in pts if p.doc_id not in {d.doc_id for d in dropped}]
        chords = [(p, iv.conductance_in_nS(p.i_open, p.current_units,
                                           p.v_volts * 1000.0)) for p in used]
        known = [(p, g) for p, g in chords if g is not None and np.isfinite(g)]
        lines = ["", "The OTHER conductance — G = I₀ / V, one per recording:"]
        if not known:
            lines.append("  not in nS for these recordings (current or voltage units "
                         "undeclared); the Calibration panel's box says the same.")
            return lines
        lines.append("  " + ",  ".join(f"{p.label}: {g:.5g} nS" for p, g in known))
        vals = np.array([g for _, g in known], dtype=np.float64)
        mean = float(np.mean(vals))
        spread = (float(vals.max() - vals.min()) / abs(mean) * 100.0) if mean else float("nan")
        lines.append(
            f"  mean {mean:.5g} nS, spread {spread:.1f}% of the mean across "
            f"{vals.size} recording(s).")
        # WHY THE TWO DISAGREE, in the numbers of this session rather than in general.
        u = fit.current_units or "(units unknown)"
        why = (f"  This window's headline is the SLOPE of the line through all "
               f"{fit.n} recording(s), not any of those: the fit is allowed a leak of "
               f"{fit.intercept:.5g} {u} at 0 V, while G = I₀ / V assumes the line "
               f"passes through the origin.")
        smallest = min((abs(p.i_open) for p, _ in known), default=0.0)
        if smallest > 0:
            # MAGNITUDES, said as magnitudes: the currents here are negative at a
            # negative bias and the leak may not be, so comparing the two as signed
            # numbers would read as a bigger or smaller claim than the one meant.
            signs = {np.sign(p.i_open) for p, _ in known}
            opposed = (len(signs) == 1 and fit.intercept != 0.0
                       and np.sign(fit.intercept) not in signs)
            why += (f" That leak is {abs(fit.intercept) / smallest * 100:.0f}% of the "
                    f"smallest open-pore current MAGNITUDE in the series "
                    f"({smallest:.5g} {u})"
                    + (", and of the opposite sign." if opposed else "."))
        lines.append(why)
        if fit.conductance_nS is not None and mean:
            lines.append(
                f"  Slope G = {fit.conductance_nS:.5g} nS is "
                f"{fit.conductance_nS / mean:.2f}x that mean. Quote the SLOPE for the "
                f"pore's conductance over the series; quote a G = I₀ / V beside the "
                f"one recording it came from.")
        return lines

    @staticmethod
    def _single_point_advice(dropped: list[iv.IVPoint], n_points: int) -> str:
        """The one-point message, naming a DROP when a drop is what caused it.

        With no drop this is the plain "open another recording". With one it is not:
        the recordings ARE open, at different voltages, and the Points list four
        lines below says so -- what the user has to change is the units of the ones
        this window could not convert, not the number of recordings."""
        if not dropped:
            return ("Only one recording — open another at a different voltage to fit a "
                    "conductance.")
        names = ", ".join(f"{p.label} ({p.current_units or 'no units'})" for p in dropped)
        return (f"Only one recording reached the fit: {len(dropped)} of the {n_points} "
                f"recordings with a (V, I₀) point were dropped for current units this window "
                f"cannot convert — {names}. Declare their units in Calibration "
                f"(pA / nA / uA / A), or open another recording at a different voltage.")

    def _finish_details(self, details: list[str], notes: list[str],
                        pts: list[iv.IVPoint], units: str,
                        dropped: list[iv.IVPoint]) -> None:
        details.append("")
        details.append("Points:")
        dropped_ids = {p.doc_id for p in dropped}
        for p in pts:
            # A Points list that reads identically for a fitted point and a discarded
            # one makes "I-V over N recording(s)" a count of something the reader
            # cannot see. Mark the discarded ones where they are listed.
            mark = "  — DROPPED, not in the fit" if p.doc_id in dropped_ids else ""
            details.append(
                f"  {p.label}:  V = {p.v_volts * 1000.0:.4g} mV,  "
                f"I₀ = {_fmt_q(p.i_open, p.current_units)}  [{p.voltage_source}]{mark}")
        if self._exclusions:
            details.append("")
            details.append("Excluded — voltage units undeclared (set them in Calibration):")
            details.extend(self._exclusion_line(e) for e in self._exclusions)
        details.append("")
        details.extend(textwrap.wrap(_I0_CROSS_REFERENCE, 88))
        if notes:
            details.append("")
            details.extend(notes)
        self._details.setPlainText("\n".join(details))

    # -- export ----------------------------------------------------------------------------------
    def save_image(self, path: str | None = None) -> Path | None:
        """Export the current I-V plot as a PNG (dialog when ``path`` is None).

        Returns the written path, or None when the user cancelled OR the write
        failed (a failed write is reported in a dialog, never returned as a path).
        """
        if path is None:
            chosen = ask_export_path(
                self, title="Save I-V image", suggestion="iv.png",
                filters="PNG image (*.png);;All files (*)")   # writes exactly this file
            if chosen is None:
                return None
            path = str(chosen)
        out = Path(path)
        # STAGED, like every other export (USER_GUIDE §12). pyqtgraph's ImageExporter
        # ends in ``QImage.save``, which OPENS AND TRUNCATES its target and reports
        # failure by RETURNING False — writing ``out`` directly destroyed the previous
        # export for any format QImage cannot write and then said "The file was not
        # created". Rendering into a staged sibling costs the user nothing on failure,
        # which is what lets the message below say so truthfully (EXP3-06).
        from ..plot.figures import image_write_failure_text, save_plot_image

        exporter = ImageExporter(self._plot)
        try:
            notes = save_plot_image(out, lambda staged: exporter.export(str(staged)))
        except Exception as exc:  # noqa: BLE001 — an export failure is reported, never crashes
            QtWidgets.QMessageBox.warning(
                self, "Save I-V image", image_write_failure_text(out, exc))
            return None
        # What the replace changed about the FILE (symlink followed, hard-linked names
        # left holding the previous contents) — unrecoverable after the fact, so it is
        # disclosed here the way File ▸ Save Project discloses it (EXP3-02).
        if notes:
            QtWidgets.QMessageBox.information(
                self, "Save I-V image", "\n\n".join(notes))
        self._details.setPlainText(self.details_text() + f"\n\nSaved image → {out.name}")
        return out
