"""The Plots window's histogram helpers and the notes it prints about its columns and sources.

``_PlotsNotesMixin`` is a part of :class:`nanoscope.gui.plots_panel.PlotsWindow`, listed
before ``QtWidgets.QMainWindow`` in its bases. It holds the histogram's shared bin edges,
normalisation and y-axis wording, the normalisation-refusal, incoming-config and
density-unit notes, the KDE overlay, and the notes about what the window drew: the
KS headline, cluster groups, source comparability, detection notes, superseded and stale
columns, the clock offset and the column notes.

It is a plain-Python mixin: no ``__init__``, no signals and no state of its own, so every
attribute its methods read (``self._style``, ``self._norm_combo`` ...) is set by
``PlotsWindow.__init__``. The constants and helpers it uses come from
:mod:`nanoscope.gui.plots_constants`; it never imports ``plots_panel`` or another part.

Every method below was moved out of ``plots_panel.py`` without an edit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt

from ..analysis import populations
from ..analysis.remeasure import REMEASURED_COLUMNS
from ..core import clock
from ..core.sources import PlotSource, SourceVerdict, source_comparability
from .plots_constants import (
    _ORIGIN_COLUMNS,
    HISTOGRAM,
    _NORM_COUNT,
    _NORM_DENSITY,
    _NORM_PROB,
    _TIME_COLUMNS,
    _TIME_UNITS,
    _SCHEMA_UNITS,
    _MSG_HEX,
    _plot_density_quantity,
    _KS_REJECTED_TAG,
    _cluster_color,
    _column_quantity,
)


class _PlotsNotesMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    # -- histogram -------------------------------------------------------------------------------
    @staticmethod
    def _shared_edges(samples: Sequence[np.ndarray], nbins: int, log_x: bool) -> np.ndarray:
        """ONE bin-edge set spanning every drawn sample.

        Binning each source over its own min/max would put the bars of two sources
        on different grids: the eye reads two shapes that were never measured
        against the same ruler, and the taller-looking one may simply have narrower
        bins. Shared edges are what makes an overlay a comparison.

        The outermost edges are pinned back onto the exact ``lo``/``hi`` the samples
        actually contain. ``np.logspace`` round-trips through ``log10`` and back, so
        its end edges can land an ulp *inside* the data range (e.g. ``lo =
        5.001781298025952e-05`` came back as ``5.0017812980259555e-05``) -- and
        ``np.histogram`` then silently drops the smallest and/or largest event. That
        is not cosmetic on an overlay: the loss lands on whichever source owns each
        extreme, so under *density* the two series integrate to DIFFERENT
        values (measured: 0.9998 vs 0.9950 for a 5000- and a 200-event source) and
        the small-n source is penalised ~25x harder -- destroying exactly the
        comparability shared edges exist to provide. It also made the drawn bars
        disagree with the ``n = ...`` the legend prints.
        """
        lo = min(float(np.min(s)) for s in samples)
        hi = max(float(np.max(s)) for s in samples)
        if log_x:
            if hi <= lo:
                hi = lo * 10.0 if lo > 0 else 1.0
            edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
        else:
            if hi <= lo:
                hi = lo + 1.0     # a single-valued column still draws one bar
            edges = np.linspace(lo, hi, nbins + 1)
        edges[0], edges[-1] = lo, hi
        return edges

    def _normalized(self, counts: np.ndarray, edges: np.ndarray, n: int) -> np.ndarray:
        """Bin heights in the chosen y quantity.

        * ``count``: raw counts.
        * ``density``: ``counts / (n * width_i)`` — ``sum(y_i * w_i) = 1``,
          so a 5000-event and a 200-event recording become the same kind of number.
        * ``probability per bin``: ``counts / n`` — ``sum(y_i) = 1``.

        Widths are per-bin, so this is correct for the unequal widths of log bins
        as well as for uniform linear ones.

        THE WIDTHS ARE IN THE COLUMN'S STORED UNIT, which is what makes ``sum(y_i *
        w_i) = 1`` true and is the ONE thing this method must never learn about the
        display. ``_axis_label_for`` re-scales the x ticks for reading (ms, µs) and
        must not re-scale the sample -- the exponential fit takes its detection floor
        from the detector's ``min_dwell_s`` in seconds -- so ``y`` is per second for a
        dwell whatever the ticks say, and the drawn area is 1000 on the app's default
        ms axis and 23.3 on a log10 abscissa. This mode was called "density (area = 1)"
        until report S6-03; the values here were correct and the NAME was not, so the
        name went and :meth:`_y_axis_label` now puts the unit on the picture instead.
        """
        mode = self._norm_combo.currentText()
        y = counts.astype(np.float64)
        if mode == _NORM_COUNT or n <= 0:
            return y
        if mode == _NORM_PROB:
            return y / float(n)
        widths = np.diff(edges).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = y / (float(n) * widths)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_scale(self, edges: np.ndarray, grid_width, n: int) -> np.ndarray | float:
        """The factor turning an expected-COUNT fit curve into the drawn y quantity.

        Same three cases as :meth:`_normalized`, evaluated on the fit grid: for a
        density the local bin width divides straight back out (which is why a
        density curve is scale-free), for a probability only ``n`` does.
        """
        mode = self._norm_combo.currentText()
        if mode == _NORM_COUNT or n <= 0:
            return 1.0
        if mode == _NORM_PROB:
            return 1.0 / float(n)
        return 1.0 / (float(n) * np.asarray(grid_width, dtype=np.float64))

    def _y_label(self) -> str:
        """The y QUANTITY, with no unit on it.

        Read by :meth:`_y_axis_label` below AND by :meth:`_plot_data`, and it does not
        carry a unit because of the second reader -- see the comment there.
        """
        mode = self._norm_combo.currentText()
        return {_NORM_COUNT: "count", _NORM_DENSITY: "density", _NORM_PROB: "probability"}[mode]

    def _y_axis_label(self, col: str, log_y: bool) -> str:
        """Tick-only Jacobian: raw bars, fits and KDE samples share one density unit."""
        from ..core.quantity_display import AxisDisplayBasis, density_quantity, quantity_for
        from .quantity_axis import configure_quantity_axis
        if self._norm_combo.currentText() == _NORM_DENSITY:
            quantity = _plot_density_quantity(col, self._time_unit.currentText())
        else:
            quantity = quantity_for("histogram." + self._norm_combo.currentText(),
                declared_unit="-", quantity="count" if self._norm_combo.currentText() == _NORM_COUNT else "dimensionless")
        configure_quantity_axis(self._plot.getAxis("left"),
                                AxisDisplayBasis(quantity, "log10" if log_y else "linear"))
        parts = [quantity.display_unit] if self._norm_combo.currentText() == _NORM_DENSITY and quantity.display_unit not in ("", "-") else []
        if log_y:
            parts.append("log")
        return self._y_label() + (f" ({', '.join(parts)})" if parts else "")

    def _norm_refusal_notes(self) -> list[str]:
        """The refused-y-quantity sentence for the page being drawn, or ``[]``.

        THE ONE READING OF :class:`_NormRefusal`, and the reason it is a method rather
        than two lines inside the histogram. Its callers reach it through
        :meth:`_incoming_notes`, which is the one SURFACE -- four draw paths and
        :meth:`_message` -- so that this record and :class:`_ConfigLosses` cannot be
        half-applied to a page. The refusal was
        rendered into the histogram's notes list only, so a configuration that is not a
        histogram was refused in SILENCE -- measured end to end on the owner's 982-event
        table with a v2 sidecar naming ``plot_type`` "Scatter" and ``density (1/s)``:
        ``import_config_json`` returned True, the window landed on Scatter, the record
        was set, Details said nothing and no dialog appeared. The one tell left was the
        template combo dropping to "— manual —" with no reason attached, and a re-export
        then wrote ``count`` over the sender's name on a path where nothing had ever said
        it was refused. The user has to be told BEFORE they save it back, and the page
        they are on when they save it back is whichever one their colleague's file chose.

        NOT MOVED TO THE DOOR INSTEAD, measured rather than argued: the door is a draw
        (:meth:`apply_config` is ``_set_config(cfg, draw=True)``), so a sentence printed
        there is wiped by the same call -- see :class:`_NormRefusal` for the marker
        measurement. And the sidecar is NOT rewritten to carry the refused name forward:
        ``export_config_json`` writes :meth:`config`, the settings actually on screen,
        and a sidecar saved beside an image ("the same gesture on two halves of one
        figure", in that method's own words) that named a y quantity the figure is not
        drawn in would be a second lie in place of the first. Disclosure, not a
        substitution, is what the whole of S6-03 is.

        WHERE IN EACH PAGE'S DETAILS: directly under that page's header lines and ahead
        of its NOTES -- the drops, the column caveats, the pooled verdict -- for the
        reason :meth:`_draw_histogram` gives at its own call: what the window was asked
        for and could not give outranks anything said about the quantity it drew
        instead.
        """
        note = self._norm_refusal.note(self._norm_combo.currentText(),
                                       on_axis=self._plot_type() == HISTOGRAM)
        return [note] if note else []

    def _incoming_notes(self) -> list[str]:
        """Everything a configuration handed this window and this window did not take.

        THE ONE SURFACE, and the reason it exists beside :meth:`_norm_refusal_notes`
        rather than as five pairs of calls: there are now TWO records saying what an
        incoming configuration lost (:class:`_NormRefusal` for the y quantity,
        :class:`_ConfigLosses` for the other six kinds), they belong in the same place in
        the same order on every page, and a page that asked for one and forgot the other
        would be a half-disclosure that reads like a whole one. Every draw path and the
        empty state ask THIS.

        AND THE EMPTY STATE ASKS IT TOO -- the fix for the hole round 4 left. Every page's
        early return goes through :meth:`_message`, which replaces Details wholesale, so
        a window that cannot draw said why it could not draw and nothing about what it was
        asked for. That is not a corner: a configuration cannot carry a PCA RESULT, so
        EVERY import onto the PCA page lands in exactly that state until Run PCA is
        pressed. Measured before the fix, on the owner's 982-event table with a v2 sidecar
        whose ``plot_type`` is "PCA": import returned True, the record read
        ``density (1/s)``, Details read the eleven-word instruction and nothing else, and
        a re-save then wrote ``count`` over the sender's file with 0 dialogs -- the
        round-3 defect verbatim, in the one page-state round 4's placement stepped over.
        The ORDER is the same everywhere: what the window was asked for and could not give
        comes under the header (or under the instruction) and ahead of that page's notes.

        AND THE THIRD ONE, from the window's OWN configuration rather than a file: two
        ticked sources whose ``label_overrides`` resolve to one legend name. The legend
        does not draw that literally (:meth:`_legend_names` tells them apart), and a
        figure that differs from what the user typed has to say so where the other two
        refusals are said -- see :meth:`_legend_collision_notes` for the figure this
        closes.
        """
        return self._norm_refusal_notes() + (
            [note] if (note := self._config_losses.note()) else []
        ) + self._legend_collision_notes()

    @staticmethod
    def _units_column_reading(frames: Sequence[pd.DataFrame]) -> tuple[str, ...] | None:
        """What the drawn tables' own ``units`` column says about the channel their rows
        were measured on: the distinct non-blank labels, sorted -- ``()`` when the column
        is there and blank everywhere, ``None`` when any drawn frame has no such column
        (a sentence about "this table's units column" has to be true of every table on
        the axis). Cheap: one ``unique`` per frame over a string column."""
        if not frames or any("units" not in df.columns for df in frames):
            return None
        seen: set[str] = set()
        for df in frames:
            labels = df["units"].dropna().astype(str).str.strip().unique()
            seen.update(str(v) for v in labels if v)
        return tuple(sorted(seen))

    def _density_unit_notes(self, col: str, log_x: bool,
                            frames: Sequence[pd.DataFrame] | None = None) -> list[str]:
        """What the bar heights are IN, and -- when it is not 1 -- what the DRAWN area is.

        The axis label states the unit; this states the CONSEQUENCE, which is the sentence a
        reader needs before they trust an area they measured off the screen. Measured on the
        owner's 982 events: the same bars integrate to 1 over seconds, 1000 over the ms
        ticks, 1e6 over us, and 23.3011485 over a log10 axis.

        THE LOG CASE IS A DIFFERENT STATEMENT, NOT A BIGGER FACTOR, and that is why it gets
        its own sentence rather than a number. On a log axis the area under the curve is not
        the integral of the density by any constant -- the abscissa is a logarithm -- so no
        rescaling could ever make ``area = 1`` readable there, and printing "multiply by
        23.3" would be a number true only for this dataset's range and would invite exactly
        the wrong correction.

        Placed directly under the ``y = ...`` header rather than in the notes list at the
        bottom, because it is a statement ABOUT that header and that is where the reader is
        looking when they ask what y is. The notes list is for drops and refusals.

        "UNIT UNKNOWN" IS NOT "DIMENSIONLESS" (S7-FIX3). :func:`_column_quantity` answers
        ``""`` for two different things -- a column the schema DECLARES dimensionless
        (``frac_depth``, ``snr``, ``n_levels``, spelled ``"-"`` there) and a column
        nobody has declared at all -- and this note used to collapse both into "a
        dimensionless density". Measured on the manual-measurement table (Analysis ▸
        Measure ▸ Add to list), whose ``units`` column read ``pA``: 14 of 14 of its
        ``signal``/``signal*s``/``sample`` columns (amplitude, level, baseline, mean,
        median, std, var, min, max, ptp, rms, cal_offset, area_ecd, n_samples) were told
        "bin width in the units of amplitude — a dimensionless density", i.e. the first
        half says the unit is unknown and the second asserts there is none, over heights
        that are per picoamp; the same quantity off the events table (``depth_pA``) read
        "a density in 1/pA". The dimensionless clause is now kept ONLY for a declared
        ``"-"``; an undeclared unit says so. NOT resolved from
        ``measure.MEASUREMENT_SCHEMA`` on purpose: its ``"signal"`` is a placeholder for
        whatever the channel was calibrated to, not a unit, and it declares ``var`` as
        ``signal`` where the variance of a picoamp signal is pA², so a second lookup
        would trade a vague falsehood for a precise one.

        AND "THIS TABLE DOES NOT DECLARE WHAT THAT UNIT IS" WAS THE NEXT LIE (A1-N3 /
        X-SURF-7), on the very table the finding was about: the manual-measurement
        table carries a ``units`` column -- ``measure.MEASUREMENT_SCHEMA``: "the
        channel's units label when this row was measured" -- and it read ``pA`` on
        every row while amplitude, level, baseline and area_ecd were each told the
        table declared nothing. So the note now asks THE TABLE ITSELF
        (:meth:`_units_column_reading`, ``frames``): where every drawn frame carries a
        ``units`` column that agrees on one label, the sentence says the axis cannot
        name the unit and what that column reads -- a fact about the CHANNEL the rows
        were measured on, deliberately not "a density in 1/pA", which is the schema
        trap above (``var`` is pA², ``n_samples`` is a count, ``area_ecd`` is pA·s);
        where there is no such column, or it is blank, or it disagrees with itself,
        the sentence says exactly that instead of "does not declare".
        """
        if self._norm_combo.currentText() != _NORM_DENSITY:
            return []
        unit = _column_quantity(col)[1]
        per = ((f"1/({unit})" if any(c in unit for c in "·*/ ") else f"1/{unit}")
               if unit else "")
        # The schema's own "-" (translated to "" by `_SCHEMA_UNITS`), and the one column
        # `_column_quantity` special-cases to a dimensionless reading.
        declared_dimensionless = col == "frac_depth" or _SCHEMA_UNITS.get(col) == ""
        if per:
            tail = f" — a density in {per}."
        elif declared_dimensionless:
            tail = " — a dimensionless density."
        else:
            if frames is None:
                frames = [df for _src, df in self._drawn_frames()]
            reads = self._units_column_reading(frames)
            if reads is None:
                said = "this table has no units column and does not declare what that unit is"
            elif not reads:
                said = ("this table's units column is blank, so it does not declare what "
                        "that unit is")
            elif len(reads) == 1:
                said = (f"the axis cannot name it, but the channel these rows were "
                        f"measured on reads “{reads[0]}” in this table's units column")
            else:
                said = (f"this table's units column reads more than one label here "
                        f"({', '.join(reads)}), so no single unit can be named")
            tail = f" — a density per unit of {col}; {said}."
        out = [f"  y = counts / (n × bin width), bin width in "
               f"{unit or 'the units of ' + col}" + tail]
        factor, name = _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))
        if log_x:
            out.append(
                f"  x is log10 here, so the AREA UNDER THE DRAWN CURVE is not the integral "
                f"of the density at all. ∫ y dx = 1 holds over {col} in "
                f"{unit or 'its own unit'}, not over the log axis on screen.")
        if col in _TIME_COLUMNS and factor != 1.0:
            out.append(f"  x ticks read {name}; density ticks apply the inverse Jacobian "
                       f"1/{factor:g} and read per {name}. Raw curves and CSV values stay per s. "
                       "The integral over linear displayed coordinates is 1; log-axis visual area is not probability.")
        return out

    def _hist_sample(self, df: pd.DataFrame, col: str, log_x: bool,
                     notes: list[str], who: str = "") -> np.ndarray | None:
        """One source's finite (and, on a log axis, positive) values for ``col``.

        BOTH DROPS ARE DISCLOSED, in the same voice, with their counts. The log-x
        exclusion below has said so since it was written; the non-finite filter above
        it did not, so a histogram of the owner's real ``rise_s`` column printed
        "apo_700: n = 1016" for a 1164-row table and nothing anywhere said the other
        148 rows had gone -- while the CLUSTERING pane of this same window, over the
        identical rows, said "1960 kept, 40 dropped (non-finite features)". Four
        surfaces of one window gave four different accounts of one drop. This method's
        own sibling :meth:`_finite` states the rule ("Drop non-finite entries, SAYING
        how many ... dropping silently would change n behind the caller's back"); the
        row TOTAL travels with the count because the shrunken n is the only number the
        pane shows and the user has nothing in the window to compare it against.
        """
        if col not in df.columns:
            return None
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = v[np.isfinite(v)]
        if finite.size < v.size:
            notes.append(
                f"{who}{v.size - finite.size} of {v.size} row(s) dropped: “{col}” is "
                f"not a finite number there.")
        v = finite
        if v.size == 0:
            return None
        if log_x:
            pos = v[v > 0]
            if pos.size == 0:
                return None
            if pos.size < v.size:
                notes.append(f"{who}{v.size - pos.size} non-positive value(s) dropped for log x.")
            v = pos
        return v

    @staticmethod
    def _nothing_to_draw(frames: Sequence[pd.DataFrame], col: str, log_x: bool) -> str:
        """WHY the histogram is empty, said truthfully. A refusal names its real cause.

        :meth:`_hist_sample` returns ``None`` for three different reasons -- the column
        is absent, every value is non-finite, or the log axis has no positive value to
        put on it -- and the caller printed the SECOND one for all three. On the
        owner's own -700 mV detection, ticking "log x" over ``baseline_pA`` (1164 of
        1164 values finite, all about -49 pA) replaced the plot with

            Column “baseline_pA” has no finite values.

        which is false about a column whose every value is a finite number, and sends
        the reader looking for a corrupt table instead of unticking a checkbox. The
        window already words the PARTIAL case correctly on the same code path ("300
        non-positive value(s) dropped for log x."); this is the all-or-nothing branch
        of the same sentence.
        """
        vals = [pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=np.float64)
                for d in frames if col in d.columns]
        if not vals:
            return (f"Column “{col}” is not in these events.")
        v = np.concatenate(vals) if len(vals) > 1 else vals[0]
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return (f"Column “{col}” has no finite values"
                    + (f" (all {v.size} row(s) are NaN, infinite or non-numeric)."
                       if v.size else "."))
        if log_x and finite[finite > 0].size == 0:
            return (f"All {finite.size} value(s) of “{col}” are non-positive "
                    f"({float(finite.min()):g} … {float(finite.max()):g}), and a log x "
                    f"axis cannot show them. Untick “log x” to plot this column.")
        return f"Column “{col}” has nothing plottable here."

    def _draw_kde_overlay(self, groups, col: str, edges: np.ndarray, log_x: bool,
                          log_y: bool, multi: bool, split: bool) -> list[str]:
        """One dotted Gaussian KDE per drawn histogram series, in the BARS' y quantity.

        THE SCALING IS THE FIT OVERLAY'S, LINE FOR LINE (:meth:`_overlay_workspace_fit`):
        ``n · f(x) · width(x) · _fit_scale``. So count is ``n · bin width · f``, density
        is ``f`` per the column's STORED unit (``_fit_scale`` divides the width back
        out -- the bars' own unit, which is why ``_density_unit_notes`` describes the
        curve too), and probability per bin is ``bin width · f``. ``n`` is the series'
        own count and the scale's ``n`` is its normalising total -- equal for a source,
        the whole source for a cluster split, whose bars normalise the same way, so the
        clusters' curves add up to the source's.

        ON LOG X THE KDE IS FITTED TO log10 OF THE COLUMN, because the bins are uniform
        there: ``width(x) = x · ln10 · Δu`` is the same model of a log bin the fit
        overlay uses, and ``f = g(u) / (x ln 10)``, so ``f · width = g(u) · Δu`` --
        the curve sits on the bars it smooths at every decade.

        DRAWN DOTTED so it cannot be read as a fit (solid, or dashed per source), and
        never estimated on project open: a restored snapshot is a picture, not a run --
        the 2-D KDE page's rule -- so the caller (:meth:`_draw_histogram`) does not call
        this then, and the gate is left showing Plot ●.
        """
        notes: list[str] = []
        if log_x:
            du = float(np.log10(edges[1]) - np.log10(edges[0]))
        lines: list[str] = []
        proxy = None
        factor, unit_name = self._display_time(col)
        for i, (label, colour, v, src, norm_n) in enumerate(groups):
            who = label if (multi or split) else (src.label if src is not None else col)
            try:
                curve = self._kde_for(v, log_x)
            except ValueError as exc:
                notes.append(f"{who}: KDE overlay skipped — {exc}.")
                continue
            x = np.asarray(curve.x, dtype=np.float64)
            width = (x * np.log(10.0) * du) if log_x else float(edges[1] - edges[0])
            y = (float(v.size) * np.asarray(curve.density) * width
                 * self._fit_scale(edges, width, int(norm_n)))
            if log_y:
                y = np.where(y > 0, y, np.nan)   # log10(0) has no place on the axis
            pen_colour = self._series_colour(index=i, identity=colour if multi else None)
            pen = pg.mkPen(pen_colour, width=self._style.line_width + 1, style=Qt.DotLine)
            item = pg.PlotDataItem(x=x, y=y, pen=pen, connect="finite")
            item._presentation_role = "category" if multi else "data"
            item._presentation_width_extra = 1
            # Exported as `kde` rows, never as a fit: `plot_source_data` reads this tag,
            # and `_plot_data` names the row by the series index it carries.
            item._plot_data_part = "kde"
            item._kde_series = i
            self._tag_series(item, "kde", label if split else (src.key if src else "data"),
                             f"{label} KDE", identity_kind="category" if split else "source",
                             colour=colour if multi or split else None)
            self._plot.addItem(item)
            proxy = proxy or item
            if curve.log:
                bw = f"{curve.bandwidth:.4g} decades of {col}"
            elif col in _TIME_COLUMNS:
                bw = f"{curve.bandwidth * factor:.4g} {unit_name}"
            else:
                quantity_unit = _column_quantity(col)[1]
                bw = f"{curve.bandwidth:.4g}" + (f" {quantity_unit}" if quantity_unit else "")
            lines.append(f"  {who}: n = {curve.n}, bandwidth h = {bw} "
                         f"(Scott factor {curve.scott_factor:.3f})")
        if proxy is not None:
            # ONE legend row, so a saved figure says what the dotted lines are; the
            # colour already names each source through its bars' row.
            if multi:
                proxy = pg.PlotDataItem(pen=pg.mkPen(_MSG_HEX, width=self._style.line_width + 1,
                                                     style=Qt.DotLine))
            self._legend_add(proxy, "KDE (Gaussian, Scott)", kind="kde", key="")
        out = []
        if lines:
            out.append("KDE overlay (Gaussian kernel, Scott bandwidth"
                       + (", estimated on log10 of the column" if log_x else "")
                       + "), drawn dotted over each series' observed range:")
            out.extend(lines)
            out.append(
                "  the KDE curves are in the bars' own y quantity (count: n × bin width × f; "
                "density: f per the column's stored unit; probability per bin: bin width × "
                "f). A Gaussian KDE spreads mass past the data's edges"
                + ("" if log_x else " (below zero for a dwell on a linear axis)")
                + ", so beside a hard edge such as the detection limit it reads below the "
                "bars.")
        return out + notes

    @staticmethod
    def _ks_headline(rejected: Sequence[str]) -> str:
        """A concise rejected-fit headline for Details and the warning summary.

        Entries are source display names, or an empty string for a single fit.
        This annotation is never added to the plot title or exported artwork.
        """
        if not rejected:
            return ""
        named = [name for name in rejected if name]
        return f"⚠ {_KS_REJECTED_TAG}" + (f": {', '.join(named)}" if named else "")

    def _cluster_groups(self, df: pd.DataFrame, col: str, log_x: bool,
                        notes: list[str], src: PlotSource):
        """One ``(label, colour, values, source, norm_n)`` per cluster of the source.

        The missing half of the clustering workflow: ``run_clustering`` could find
        two populations and the histogram had no way to show their distributions
        apart. Labels are reindexed onto the frame actually on screen, so the
        accepted-only filter and a clustering computed before it still agree; rows
        the clustering dropped (non-finite features) land in the grey ``-1`` group
        rather than vanishing without a word.

        Every cluster normalizes against the SAME total (all clusters together),
        unlike two independent sources: these are PARTS OF ONE distribution, so
        their densities must add up to the whole rather than each be rescaled to
        area 1 — which would erase exactly the "one population is rare" finding the
        clustering just made.
        """
        res = self._cluster_result
        if res is None or col not in df.columns:
            return []
        labels = res.labels.reindex(df.index, fill_value=populations.DROPPED_LABEL)
        codes = labels.to_numpy(dtype=int)
        total = self._hist_sample(df, col, log_x, [])
        norm_n = int(total.size) if total is not None else int(len(df))
        out = []
        for cid in sorted({int(c) for c in codes}, key=lambda c: (c < 0, c)):
            sub = df[codes == cid]
            v = self._hist_sample(sub, col, log_x, notes, f"cluster {cid}: ")
            if v is None:
                continue
            name = "dropped (non-finite features)" if cid < 0 else f"cluster {cid}"
            out.append((name, _cluster_color(cid).name(), v, src, norm_n))
        return out

    def _comparability(self, *columns: str) -> SourceVerdict:
        """MAY these sources be pooled for these columns? The ONE verdict, per metric.

        :func:`nanoscope.core.sources.source_comparability` — the same baseline /
        merge rule ``Analysis ▸ Voltage dependence`` enforces, plus the gain rule
        this window already had, reconciled into one answer.

        Two things this replaced. The window had NO baseline check at all: overlay
        ``frac_depth`` from two recordings the probe had proposed different windows
        for and Details listed the two sources, their n, and nothing else — although
        dI/I0 is ``depth_pA / |baseline_pA|`` and moves with the window (measured on
        the owner's ladder: 11.3 % on the −700 mV recording). And the gain guard,
        firing on ``depth_pA``, actively redirected the user onto ``frac_depth`` —
        onto the unguarded column. One verdict, built by one function that knows
        about both guards, is what stops the advice contradicting the warning.

        Judged over the DRAWN sources (ticked and non-empty), not merely the ticked
        ones, so the verdict describes what is on screen.
        """
        # A single source cannot disagree with anything, and the rule says so
        # itself — no early return here, so there is one code path and one verdict.
        # (A source's OWN detection warnings show regardless: _detection_notes.)
        drawn = self._drawn_frames()
        return source_comparability([src for src, _df in drawn], columns,
                                    frames=[df for _src, df in drawn])

    def _detection_notes(self, sources=None) -> list[str]:
        """Every DRAWN detection's own caveats — ``AnalysisResult.warnings``, whole.

        The pane already surfaces a fit's ``meta["warnings"]`` and a clustering's
        ``ClusterResult.warnings``; the detection's were the one channel with no
        outlet here, so a table measured at the unprobed default baseline plotted
        with no caveat while the Summary panel said so about the same numbers.

        Prefixed with the source label because an overlay draws several, and an
        un-attributed caveat on a two-source plot is a caveat about nothing in
        particular. Taken whole and unfiltered, exactly as
        :meth:`nanoscope.gui.panels.StatsPanel.set_warnings` takes its list (its label
        shows one headline per item, the full text behind its switch): this must not
        become a place that decides which of a detector's advisories a plot inherits.
        """
        if sources is None:
            sources = [src for src, _df in self._drawn_frames()]
        srcs = list(sources)
        multi = len(srcs) > 1
        out: list[str] = []
        for src in srcs:
            for w in getattr(src, "warnings", ()) or ():
                text = str(w).strip()
                if text:
                    out.append(f"{src.label}: {text}" if multi else text)
        return out

    # -- what the WINDOW must say about the COLUMNS it drew ----------------------------------------
    #
    # `_detection_notes` above carries a DETECTION's own caveats, whole, and must stay
    # that way. These two carry the facts that belong to a (source, column) pair
    # instead -- whether the column is still measured the way a fresh run would measure
    # it, and which clock a column of instants is on -- and they are collected by ONE
    # method that every draw path calls, because "the plot says less about these
    # numbers than the table they came from" is the defect, not the surface it was
    # found on.

    def _superseded_columns(self, src: PlotSource | None) -> tuple[str, ...]:
        """Every column of ``src`` the event table badges ⚠: measured under a calibration
        that has since been replaced (:meth:`_calibration_superseded`), and -- through the
        Results panel's own judge -- on a clock the recording no longer has, or by an
        open-pore estimator the table on screen no longer has (:meth:`_table_staleness`).

        The app's doctrine for a superseded measurement is "never silently re-measure":
        the numbers stay and every surface that shows them says so (USER_GUIDE §9a --
        such a project stays "marked, badged, and otherwise fully readable, plottable
        and exportable"). *Plottable* is where the mark stopped: two panels away from a
        ⚠ header and amber cells, this window drew the same ``depth_pA`` and printed its
        fitted mean in the title, the legend and Details with nothing to say the
        calibration behind it had been superseded -- and, until the judge was asked here,
        nothing at all for a re-declared sample rate or a changed estimator.
        """
        return tuple(dict.fromkeys((*self._calibration_superseded(src),
                                    *self._table_staleness(src)[0])))

    def _calibration_superseded(self, src: PlotSource | None) -> tuple[str, ...]:
        """Columns of ``src`` measured under a calibration that has since been replaced.

        The SAME test the event table's badge is driven by
        (``AnalysisPanel._calibration_moved``): the detection's stored
        ``calibration_fingerprint`` against the channel's live one. Same columns too --
        :data:`nanoscope.analysis.remeasure.REMEASURED_COLUMNS`, ``baseline_pA``
        included, because it is the column the other six are computed FROM.

        Empty for a source with no document behind it (the manual table), for one whose
        detection is gone, and whenever the stamps match. Asked of the session directly,
        so a window with no provider still marks a superseded calibration.
        """
        if src is None or src.doc_id is None or src.channel is None or not src.module_id:
            return ()
        try:
            res = self._session.results.get(src.doc_id, int(src.channel), src.module_id)
            live = self._session.calibration(src.doc_id, int(src.channel)).fingerprint()
        except Exception:       # pragma: no cover - a caveat must never break a plot
            return ()
        stamped = getattr(res, "calibration_fingerprint", None) if res is not None else None
        if stamped is None or live is None:
            return ()
        try:
            same = (tuple(float(x) for x in stamped) == tuple(float(x) for x in live))
        except (TypeError, ValueError):     # pragma: no cover - malformed stamp
            return ()
        if same:
            return ()
        optional = ("amplitude_pA",) if src.df is not None and "amplitude_pA" in src.df else ()
        return (*REMEASURED_COLUMNS, *optional)

    def _table_staleness(self, src: PlotSource | None) -> tuple[tuple[str, ...], str]:
        """``(columns, sentence)``: what the Results panel badges on ``src``'s table BEYOND a
        superseded calibration, and the amber line's sentence for it -- ``((), "")`` without
        a provider (:meth:`set_staleness_provider`), for a source with no stored detection
        behind it, and when the table is current.

        The provider is the panel's own judge, so a re-declared sample rate (the time
        columns) is marked on every source, and a changed open-pore estimator (the
        amplitude columns) on the table on screen -- the one whose form it is judged
        against -- exactly where the event table would badge them. The columns a
        superseded calibration explains are left to :meth:`_calibration_superseded`, whose
        note already says so; ``amplitude_pA`` rides with ``depth_pA`` here as it does there
        and in the event table's own badge.
        """
        provider = self._staleness_provider
        if (provider is None or src is None or src.doc_id is None or src.channel is None
                or not src.module_id):
            return (), ""
        try:
            columns, sentence = provider(src.doc_id, int(src.channel), str(src.module_id))
        except Exception:       # pragma: no cover - a caveat must never break a plot
            return (), ""
        calibration = set(self._calibration_superseded(src))
        cols = [str(c) for c in columns if str(c) not in calibration]
        if ("depth_pA" in cols and "amplitude_pA" not in cols and src.df is not None
                and "amplitude_pA" in src.df):
            cols.append("amplitude_pA")
        return (tuple(cols), str(sentence or "")) if cols else ((), "")

    def _clock_offset(self, src: PlotSource | None) -> float | None:
        """``t0_s`` of the recording behind ``src``, or None when there is no clock."""
        if src is None or src.doc_id is None:
            return None
        try:
            doc = self._session.document(src.doc_id)
        except Exception:       # pragma: no cover - a closed document
            return None
        return clock.clock_offset_s(getattr(doc, "meta", None))

    def _column_notes(self, columns: Sequence[str], sources=None) -> list[str]:
        """Everything this window owes the reader about the COLUMNS it just drew.

        One collector, called by the histogram, the scatter and the PCA alike, so a
        caveat cannot be true in one view of the same numbers and absent from another.
        """
        if sources is None:
            sources = [src for src, _df in self._drawn_frames()]
        srcs = [s for s in sources if s is not None]
        cols = [c for c in dict.fromkeys(str(c) for c in columns if c)]
        out: list[str] = []
        multi = len(srcs) > 1

        # 1. STALE COLUMNS -- per source, naming the drawn columns each mark touches: a
        #    SUPERSEDED CALIBRATION first, then whatever else the Results panel's judge
        #    badges on the same table (a re-declared rate, a changed estimator), in the
        #    amber line's own sentence.
        for src in srcs:
            hit = [c for c in cols if c in self._superseded_columns(src)]
            if not hit:
                continue
            who = f"{src.label}: " if multi else ""
            calibration = self._calibration_superseded(src)
            old_gain = [c for c in hit if c in calibration]
            if old_gain:
                out.append(
                    f"{who}SUPERSEDED CALIBRATION: {', '.join(old_gain)} on this plot "
                    f"{'was' if len(old_gain) == 1 else 'were'} measured under a gain/offset "
                    f"that has since been changed, and "
                    f"{'it has' if len(old_gain) == 1 else 'they have'} NOT been recomputed — "
                    f"the event table badges the same column(s) ⚠. The numbers drawn, and "
                    f"every fitted parameter printed from them, are in the OLD calibration. "
                    f"Re-run detection on this recording before reporting them.")
            rest = [c for c in hit if c not in calibration]
            if rest:
                one = len(rest) == 1
                out.append(
                    f"{who}STALE: {', '.join(rest)} on this plot {'is' if one else 'are'} not "
                    f"what a fresh run would measure now, and {'it has' if one else 'they have'} "
                    f"NOT been recomputed — the event table badges the same column(s) ⚠, and "
                    f"every fitted parameter printed from {'it' if one else 'them'} is as "
                    f"stale. {self._table_staleness(src)[1]}".rstrip())

        # 2. WHICH CLOCK a column of instants is on. `t_start_s` is seconds since THIS
        #    recording's first loaded sample, so pooling it across recordings puts two
        #    different origins on one axis -- and a segment's rows are rebased onto the
        #    segment's own start, which is how ONE physical event lands at two different
        #    x positions here. Every other surface that shows the column says which
        #    clock it is on (the event table header, the measurement table, the PSD
        #    segment spin box); this is the only one that POOLS it, and it said nothing.
        #    core/clock.py's rule: "One clock, or every control that shows or takes a
        #    time says which clock it is in."
        origin_cols = [c for c in cols if c in _ORIGIN_COLUMNS]
        if origin_cols:
            named = ", ".join(origin_cols)
            if multi:
                where = []
                for src in srcs:
                    off = self._clock_offset(src)
                    where.append(f"{src.label} (t₀ = {clock.format_t0(off)} s"
                                 + (", a segment, rebased onto its own start"
                                    if src.is_segment else "") + ")")
                out.append(
                    f"CLOCK: {named} is measured from EACH recording's own first loaded "
                    f"sample, not from a shared origin, and these {len(srcs)} sources are "
                    "pooled on one axis: " + "; ".join(where) + ". Two points at the same "
                    "x are not simultaneous; add each recording's t₀ to put them on the "
                    "instrument clock.")
            elif srcs:
                off = self._clock_offset(srcs[0])
                if off is None:
                    # No open recording behind this source (a pushed frame, the manual
                    # table, a closed document). `clock.field_note` answers None with
                    # "this recording declares no sample rate", which would be a
                    # different -- and here false -- reason for the same silence.
                    out.append(
                        f"CLOCK: {named} is measured from the recording's OWN first "
                        f"loaded sample. No open recording stands behind these rows in "
                        f"this window, so their t₀ on the instrument clock cannot be "
                        f"stated here.")
                else:
                    # `clock.field_note` is a whole sentence and the app's ONE wording
                    # for this fact -- restating it here is how the surfaces drifted
                    # apart in the first place -- so it is joined, not paraphrased.
                    out.append(f"CLOCK: {named} — "
                               + clock.field_note(clock.RECORDING, off))
        return out

    def _stale_headline(self, columns: Sequence[str], sources=None) -> str:
        """The title's own mark for a drawn column the event table badges stale:
        ``⚠ SUPERSEDED CALIBRATION`` under a replaced gain/offset, ``⚠ STALE: <columns>``
        for what the Results panel's judge badges besides (a re-declared rate, a changed
        estimator; :meth:`_table_staleness`), and both when both hold.

        In the title because the title is the only thing ``Save image`` keeps -- the
        same reason the comparability verdict goes there (see :meth:`_apply_title`).
        An exported figure of a superseded depth must not look like an exported figure
        of a current one.
        """
        if sources is None:
            sources = [src for src, _df in self._drawn_frames()]
        cols = [c for c in dict.fromkeys(str(c) for c in columns if c)]
        old_gain = False
        rest: list[str] = []
        for src in sources:
            hit = [c for c in cols if c in self._superseded_columns(src)]
            calibration = self._calibration_superseded(src)
            old_gain = old_gain or any(c in calibration for c in hit)
            rest += [c for c in hit if c not in calibration]
        marks = ["⚠ SUPERSEDED CALIBRATION"] if old_gain else []
        if rest:
            marks.append("⚠ STALE: " + ", ".join(dict.fromkeys(rest)))
        return self._warn_headline(*marks) or ""
