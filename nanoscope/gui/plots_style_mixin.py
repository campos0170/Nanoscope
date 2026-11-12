"""The Plots window's style handlers, style helpers and axis-range controls.

``_PlotsStyleMixin`` is a part of :class:`nanoscope.gui.plots_panel.PlotsWindow`, listed
before ``QtWidgets.QMainWindow`` in its bases. It holds the Style group's change handlers
(title, fonts, sizes, colours, background, grid, legend rows), the helpers that turn the
current style into pens, brushes, fonts and colours at each render site, the axis labels
and the title, and the axis-range boxes with their unit labels, notes and reset.

It is a plain-Python mixin: no ``__init__``, no signals and no state of its own, so every
attribute its methods read (``self._style``, ``self._x_min`` ...) is set by
``PlotsWindow.__init__``. The constants and helpers it uses come from
:mod:`nanoscope.gui.plots_constants`; it never imports ``plots_panel`` or another part.

Every method below was moved out of ``plots_panel.py`` without an edit.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..core import clock
from ..core.sources import amplitude_units_label
from .figure_text import css_args, rich_text
from .font_picker import figure_family, figure_qfont, html_font_family
from .plot_appearance import PlotAppearance
from .plots_constants import (
    _ORIGIN_COLUMNS,
    HISTOGRAM,
    SCATTER,
    KDE,
    HEATMAP,
    _HEATMAP_COLORMAPS,
    _NORM_DENSITY,
    _TIME_COLUMNS,
    _TIME_UNITS,
    _SCHEMA_UNITS,
    _MSG_HEX,
    _MSG_LIGHT_HEX,
    _HIST_FILL_ALPHA,
    _POINT_ALPHA,
    _BG_DARK,
    _FG_DARK,
    _BG_LIGHT,
    _FG_LIGHT,
    _plot_column_quantity,
    _plot_density_quantity,
    _swatch_text_color,
)


class _PlotsStyleMixin:
    """Methods of the main class that live in this file. No ``__init__``, no state of its own."""

    @staticmethod
    def _pt_override_spin(value: float, tip: str) -> QtWidgets.QDoubleSpinBox:
        """A point-size spin whose 0 reads "follow the body font", spelled out on itself.

        The same shape (and the same 0-means-automatic rule) as the legend's own size
        spin, so the four text controls in this group behave alike rather than each
        having its own idea of what "unset" looks like."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(.25)
        spin.setRange(0, 72)
        spin.setSpecialValueText("auto (body)")
        spin.setValue(float(value))
        spin.setToolTip(tip)
        return spin

    def _make_swatch(self, hex_color: str, on_click, tip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton()
        btn.setToolTip(tip)
        btn.setFixedHeight(24)
        btn.clicked.connect(on_click)
        self._paint_swatch(btn, hex_color)
        return btn

    @staticmethod
    def _paint_swatch(btn: QtWidgets.QPushButton, hex_color: str) -> None:
        """Show the chosen colour ON the button, with legible ink and its hex as the label."""
        btn.setText(hex_color)
        btn.setStyleSheet(
            f"background-color: {hex_color}; color: {_swatch_text_color(hex_color)}; "
            "border: 1px solid #555; padding: 2px;")

    # -- style-change handlers -------------------------------------------------------------------
    @staticmethod
    def _label_override(text: str) -> str:
        """A label box as the style keeps it, by ``PlotAppearance``'s one rule (design 13.4
        Q1): "" is the automatic label, a box of spaces hides it, anything else is the text."""
        return PlotAppearance.from_mapping({"title": text}).title or ""

    def _on_style_title(self, text: str) -> None:
        self._style = _dc_replace(self._style, title=self._label_override(text))
        self._presentation_changed()

    def _on_style_font_family(self, family: str) -> None:
        """The Font row was picked. ``family`` is what the row STORES: ``""`` for its first
        entry ("Arial (default)", the blank this style has always meant by "no family
        chosen"), else the family. Only a pick gets here -- :meth:`_select_font_family`
        shows a style on the row without saying anything."""
        self._style = _dc_replace(self._style, font_family=str(family))
        self._presentation_changed()

    def _on_style_font_size(self, value: float) -> None:
        self._style = _dc_replace(self._style, font_size_pt=float(value))
        self._presentation_changed()

    def _on_style_point_size(self, value: float) -> None:
        # Not while a style is written onto the controls: the box keeps three decimals and
        # says its size back rounded, and a line of 1 pt (1.3333 px) would come back 1.333 and
        # read, at the next edit, as the figure's own override of the style's line.
        if self._updating:
            return
        self._style = _dc_replace(self._style, point_size=float(value))
        self._presentation_changed()

    def _on_style_line_width(self, value: float) -> None:
        if self._updating:      # see _on_style_point_size
            return
        self._style = _dc_replace(self._style, line_width=float(value))
        self._presentation_changed()

    def _on_style_background(self, text: str) -> None:
        self._style = _dc_replace(
            self._style, background="light" if text == "Light" else "dark")
        self._presentation_changed()

    def _on_style_ci_band(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_ci_band=bool(on))
        self._presentation_changed()

    def _on_style_residuals(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_residuals=bool(on))
        self._presentation_changed()

    def _on_style_provenance(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_provenance=bool(on))
        self._presentation_changed()

    def _on_style_x_label(self, text: str) -> None:
        self._style = _dc_replace(self._style, x_label=self._label_override(text))
        self._presentation_changed()

    def _on_style_y_label(self, text: str) -> None:
        self._style = _dc_replace(self._style, y_label=self._label_override(text))
        self._presentation_changed()

    def _on_style_show_legend(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_legend=bool(on))
        self._presentation_changed()

    def _on_style_title_size(self, value: float) -> None:
        self._style = _dc_replace(self._style, title_font_size_pt=float(value))
        self._presentation_changed()

    def _on_style_axis_label_size(self, value: float) -> None:
        self._style = _dc_replace(self._style, axis_label_font_size_pt=float(value))
        self._presentation_changed()

    def _on_style_tick_size(self, value: float) -> None:
        self._style = _dc_replace(self._style, tick_font_size_pt=float(value))
        self._presentation_changed()

    def _on_style_grid(self, on: bool) -> None:
        self._style = _dc_replace(self._style, show_grid=bool(on))
        self._presentation_changed()

    def _on_style_heatmap_colormap(self, name: str) -> None:
        if name not in _HEATMAP_COLORMAPS:
            return
        self._style = _dc_replace(self._style, heatmap_colormap=str(name))
        self._presentation_changed()

    def _on_legend_entry_toggled(self, item) -> None:
        """One row of the Legend-entries checklist moved: remember it and restyle.

        Cosmetic by construction — nothing here re-reads a row of data — so it redraws
        live rather than going behind the run gate."""
        if self._updating:
            return
        try:
            entry_id = item.data(Qt.UserRole)
            checked = item.checkState() == Qt.Checked
        except RuntimeError:
            # Belt and braces. Shiboken raises this for a row whose C++ half is already
            # gone; the ORDERING fix (nothing this window frees is freed during its own
            # signal) is the real protection, because a genuinely dangling pointer never
            # reaches Python at all -- it crashes in the converter.
            return
        if not entry_id:
            return
        self._style = self._style.with_legend_include(str(entry_id), checked)
        self._presentation_changed()
        # THE HISTORY HOOK, BY HAND. `_wire_history`'s walk deliberately skips
        # `itemChanged` (see the note there), so this handler calls the hook itself, in
        # the position -- after the change is applied -- the generic connection held.
        self._history_touch()

    def _sync_legend_entry_list(self) -> None:
        """Re-offer the Legend-entries checklist from what the last draw offered.

        RECONCILED IN PLACE, NEVER CLEARED AND REFILLED -- and that is a CRASH FIX, not
        an optimisation. This method runs inside :meth:`_redraw`'s ``finally``, which a
        legend tick reaches through :meth:`_on_legend_entry_toggled` -> :meth:`_changed`:
        i.e. from INSIDE Qt's delivery of ``itemChanged(QListWidgetItem*)``, while Qt is
        still holding the ticked row as that emission's argument. ``clear()`` DELETED
        that C++ row on the spot, and every receiver the emission had not reached yet
        then had the same argument marshalled from a freed pointer -- PySide reads the
        dead object's vtable to build a wrapper, which is the EXC_BAD_ACCESS the shipped
        build died on. ``blockSignals`` (kept, for the re-entrant emission this method's
        own writes would otherwise cause) stops the SIGNAL and does nothing whatever
        about the POINTER.

        Reconciling makes the crashing path free-nothing: a tick changes whether an
        entry is INCLUDED, never which entries are OFFERED (``_legend_add`` records the
        offer either way), so every row is reused. When the offered set genuinely does
        change, the surplus rows are ``takeItem``-ed and parked by :meth:`_retire_row`
        so that their deletion happens after the event that caused it rather than inside
        it. Writing only what differs also ends the full model reset this list used to do
        on every restyle. That is the same OPERATION the session's
        ``qt.accessibility.table: ... Resizing table model`` warning complains about (an
        item view repopulated under something holding its old shape), though that line
        names a five-row view and this list had one -- it was a witness to the churn, not
        to this crash.
        """
        if not hasattr(self, "_legend_entries"):
            return   # fires during construction, before the Style group exists
        widget = self._legend_entries
        blocked = widget.blockSignals(True)
        try:
            offered = list(self._legend_offered.items())
            while widget.count() > len(offered):
                self._retire_row(widget.takeItem(widget.count() - 1))
            for row, (entry_id, label) in enumerate(offered):
                item = widget.item(row)
                if item is None:
                    item = QtWidgets.QListWidgetItem()
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    # EXPLICIT, even though `checkState()` already answers Unchecked: a
                    # row with no CheckStateRole *data* is drawn without a checkbox at
                    # all, and `_write_legend_row` below writes only what differs.
                    item.setCheckState(Qt.Unchecked)
                    widget.addItem(item)
                self._write_legend_row(item, entry_id, label)
        finally:
            widget.blockSignals(blocked)
        widget.setEnabled(bool(self._legend_offered) and self._style.show_legend)
        if hasattr(self, "_appearance_widget"):
            self._appearance_widget.set_legend_available(bool(self._legend_offered))
            markers = bool(getattr(self, "_scatter_items", []))
            self._appearance_widget.marker_size.setEnabled(markers)
            self._appearance_widget.marker_size.setToolTip(
                "Size of the existing scatter/PCA markers." if markers else
                "The current figure has no scatter/PCA markers.")
        if hasattr(self, "_lasso_action"):
            available = bool(self._event_point_items) and self._message_item is None
            self._lasso_action.setEnabled(available)
            if not available:
                self._lasso.set_enabled(False)

    def _write_legend_row(self, item, entry_id: str, label: str) -> None:
        """Put one offered entry onto an EXISTING row, touching only what differs.

        Every setter here goes out as a ``dataChanged``, so writing back a value the row
        already carries would re-create exactly the model churn reconciling removed.
        """
        text = self._elide_legend(label)
        tip = f"{label}\n\nUntick to leave this row out of the legend."
        state = Qt.Checked if self._legend_included(entry_id) else Qt.Unchecked
        if item.text() != text:
            item.setText(text)
        if item.data(Qt.UserRole) != entry_id:
            item.setData(Qt.UserRole, entry_id)
        if item.checkState() != state:
            item.setCheckState(state)
        if item.toolTip() != tip:
            item.setToolTip(tip)

    def _retire_row(self, item) -> None:
        """Park a row taken out of a list widget until the current event has unwound.

        ``takeItem`` hands the C++ row's ownership to Python, so holding its wrapper
        here keeps the object ALIVE -- which is the entire point. A row can be removed
        from inside the delivery of a signal that carries a row of the same list, and Qt
        marshals that argument again for every receiver it has not reached yet; a row
        freed underneath that is a dangling pointer, and PySide dereferences it. One
        zero-timer later the event is over and letting go is safe.
        """
        if item is None:
            return
        self._retired_rows.append(item)
        if not self._retire_timer.isActive():
            self._retire_timer.start(0)

    def _drop_retired_rows(self) -> None:
        """Let go of every parked row: the event that removed them is long over."""
        self._retired_rows.clear()

    def _empty_list_widget(self, widget) -> None:
        """``clear()`` for a list whose rows may be in flight -- see :meth:`_retire_row`.

        Same end state as ``QListWidget.clear()``; the rows are handed to the parking
        lot instead of being destroyed inside whatever event asked for the rebuild.
        """
        while widget.count():
            self._retire_row(widget.takeItem(widget.count() - 1))

    def _on_style_legend_font_size(self, value: float) -> None:
        self._style = _dc_replace(self._style, legend_font_size_pt=float(value))
        self._presentation_changed()

    def _on_style_legend_counts(self, on: bool) -> None:
        self._style = _dc_replace(self._style, legend_counts=bool(on))
        self._presentation_changed()

    def _on_style_export_size(self, *_args) -> None:
        # Export size only affects save_image; no redraw needed.
        self._style = _dc_replace(self._style,
                                  export_w=int(self._export_w.value()),
                                  export_h=int(self._export_h.value()))

    def _pick_data_color(self) -> None:
        self._pick_color("data_color", self._data_swatch)

    def _pick_fit_color(self) -> None:
        self._pick_color("fit_color", self._fit_swatch)

    def _pick_color(self, attr: str, swatch: QtWidgets.QPushButton) -> None:
        col = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(getattr(self._style, attr)), self, "Pick colour")
        if col.isValid():
            hex_color = col.name()
            # The field is chosen by NAME by the two one-line callers below, so the
            # frozen-copy spelling has to be by name too -- `replace` takes kwargs.
            self._style = _dc_replace(self._style, **{attr: hex_color})
            self._paint_swatch(swatch, hex_color)
            self._presentation_changed()

    # -- style helpers (colours / fonts / theme applied at each render site) ---------------------
    def _qfont(self, pt: int | None = None) -> QtGui.QFont:
        """The chosen family at ``pt``, defaulting to the BODY size.

        ``pt`` is the whole of owner note Q5 at this level: one family, four sizes.
        Every caller passes the size that belongs to the text it is about to set, and
        each of those sizes falls back to the body size when its own override is 0, so
        an untouched style still produces one number in four places."""
        qf = figure_qfont(self._style.font_family)         # unset: Arial; missing: a substitute
        qf.setPointSizeF(float(self._style.font_size_pt if pt is None else pt))
        return qf

    def _title_pt(self) -> float:
        """Title size in pt: its own override, or the body size."""
        return float(self._style.title_font_size_pt or self._style.font_size_pt)

    def _axis_label_pt(self) -> float:
        """Axis-caption size in pt: its own override, or the body size."""
        return float(self._style.axis_label_font_size_pt or self._style.font_size_pt)

    def _tick_pt(self) -> float:
        """Tick-number size in pt: its own override, or the body size."""
        return float(self._style.tick_font_size_pt or self._style.font_size_pt)

    def _fg_hex(self) -> str:
        return _FG_LIGHT if self._style.background == "light" else _FG_DARK

    def _bg_spec(self) -> str:
        return _BG_LIGHT if self._style.background == "light" else _BG_DARK

    def _msg_hex(self) -> str:
        return _MSG_LIGHT_HEX if self._style.background == "light" else _MSG_HEX

    def _series_colour(self, role="data", *, index=0, identity=None, explicit=None):
        """Shared palette affects unassigned ink; source/category identity stays owned."""
        from ..core.figure_style import resolve_series_color
        native = self._style.fit_color if role == "fit" else self._style.data_color
        return resolve_series_color(self._style.style_binding, role=role, index=index,
            native_color=native, identity_color=identity, explicit_color=explicit)

    def _hist_brush(self) -> QtGui.QBrush:
        c = QtGui.QColor(self._series_colour())
        c.setAlpha(_HIST_FILL_ALPHA)
        return pg.mkBrush(c)

    def _point_brush(self) -> QtGui.QBrush:
        c = QtGui.QColor(self._series_colour())
        c.setAlpha(_POINT_ALPHA)
        return pg.mkBrush(c)

    def _data_pen(self) -> QtGui.QPen:
        return pg.mkPen(self._series_colour(), width=self._style.line_width)

    def _fit_pen(self) -> QtGui.QPen:
        # The fit curve reads one px heavier than the bar outline (default 2 px, as before).
        return pg.mkPen(self._series_colour("fit"), width=self._style.line_width + 1)

    def _scatter_fit_pen(self) -> QtGui.QPen:
        # The scatter best-fit line uses the fit colour at (at least) the chosen line width,
        # so it reads clearly over the marker cloud.
        return pg.mkPen(self._series_colour("fit"), width=max(self._style.line_width, 1))

    def _label_style(self) -> dict:
        """CSS-ish kwargs for an axis label: theme-aware colour + the chosen font.

        Sized by :meth:`_axis_label_pt` rather than by the body size — the axis
        captions are the second of the four independent text sizes owner note Q5 asks
        for, and they are the ones a poster usually wants larger than the ticks."""
        return {"color": self._fg_hex(), "font-size": f"{self._axis_label_pt()}pt",
                # AxisItem merges styles; explicitly reset the family when the user returns
                # to the default (Arial), instead of retaining it. The fallback stack, in the
                # double quotes a ``style='…'`` attribute allows, so a missing family still
                # draws in a face of its own kind.
                "font-family": html_font_family(self._style.font_family)}

    def _apply_grid(self) -> None:
        """Show or hide the gridlines, from the style, on every draw (owner note Q7).

        The alpha stays at the 0.15 the window has always used: whether there is a grid
        is an editorial choice, how faint it is is a house style, and offering both
        would let a user produce a grid that is technically on and invisible.
        """
        show = bool(self._style.show_grid)
        self._plot.showGrid(x=show, y=show, alpha=0.15)

    def _apply_background(self) -> None:
        """Paint the canvas + axis lines/ticks for the current theme so light mode stays legible.

        The grid rides along: it is part of the canvas, it has to be re-read on every
        draw for the same reason the ink does, and every render site in this file
        already calls this one (owner note Q7)."""
        self._apply_grid()
        from .plot_appearance import apply_plot_frame
        apply_plot_frame(self._plot, self._style.frame, foreground=self._fg_hex())
        self._plot_widget.setBackground(self._bg_spec())
        pen = pg.mkPen(self._fg_hex())
        for name in ("bottom", "left"):
            ax = self._plot.getAxis(name)
            ax.setPen(pen)
            ax.setTextPen(pen)

    def _apply_fonts(self) -> None:
        """Tick-label font. Its OWN size (owner note Q5): the ticks are the text a
        larger title used to drag up with it until the numbers collided."""
        qf = self._qfont(self._tick_pt())
        for name in ("bottom", "left"):
            self._plot.getAxis(name).setStyle(tickFont=qf)
        message = getattr(self, "_message_item", None)
        if message is not None:
            message.setFont(self._qfont(self._style.font_size_pt))
        self._apply_element_styles()

    def _apply_element_styles(self) -> None:
        """The per-axis styles set from the figure (Text style… on a label or the ticks),
        over the figure-wide fonts. Run after every ``setLabel``: the label CSS that call
        installs is the figure-wide one, and an element override sits on top of it."""
        for side in ("bottom", "left"):
            axis = self._plot.getAxis(side)
            label = self._style.effective_text_style(f"axis.{side}.label")
            if not _dc_replace(label, text=None).is_default():
                axis.labelStyle = {**axis.labelStyle, **css_args(label)}
                axis._updateLabel()
            ticks = self._style.effective_text_style(f"axis.{side}.ticks")
            if _dc_replace(ticks, text=None).is_default():
                continue
            qf = self._qfont(ticks.size_pt or self._tick_pt())
            if ticks.size_pt:
                qf.setPointSizeF(float(ticks.size_pt))
            if ticks.family:
                qf.setFamily(figure_family(ticks.family))
            if ticks.bold is not None:
                qf.setBold(ticks.bold)
            if ticks.italic is not None:
                qf.setItalic(ticks.italic)
            axis.setStyle(tickFont=qf)
            if ticks.color:
                axis.setTextPen(pg.mkPen(ticks.color))

    def _axis_label_for(self, side: str, col: str, log: bool) -> str:
        """Label for ``side`` ('bottom'/'left'), rescaling the axis for time columns.

        ONE PLACE DECIDES THE UNIT NAME AND THE NUMBER THE TICKS SHOW. That is the whole
        contract of this method, and it is why the tick multiplier is set here as a side
        effect rather than anywhere the label is not also being written. Both are reset
        on every draw — a stale x1000 left on a pA axis would be a silent
        factor-of-1000 lie.

        The two modes reach the same place by different mechanics:

        * **linear** — ``AxisItem.setScale(factor)`` multiplies the tick labels, which is
          what it is for.
        * **log** — ``setScale`` is unusable, because pyqtgraph's ``tickValues``
          multiplies the log10 RANGE by it, i.e. scales an exponent. So the axis stays at
          ``setScale(1.0)`` and :attr:`_DecadeLabelAxis.display_factor` multiplies the
          decade VALUES instead, which is the same arithmetic in the one place it is
          meaningful.

        The log branch used to give up here and label the axis ``dwell (s, log)`` while
        leaving the ticks alone. pyqtgraph then supplied its own automatic SI prefix,
        multiplied the tick values by 1000 anyway and appended ``(x0.001)`` — so with
        Time unit = ms an 18 ms event was drawn at ``x = 10^1`` under a label saying
        seconds (report High-7). The prefix is now off (see :class:`_DecadeLabelAxis`)
        and the factor is this method's to set, so the label and the ticks cannot
        disagree.
        """
        axis = self._plot.getAxis(side)
        from ..core.quantity_display import AxisDisplayBasis
        from .quantity_axis import configure_quantity_axis
        configure_quantity_axis(axis, AxisDisplayBasis(
            _plot_column_quantity(col, self._time_unit.currentText()), "log10" if log else "linear"))
        factor, name = _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))
        is_time = col in _TIME_COLUMNS
        setter = getattr(axis, "set_display_factor", None)
        if setter is not None:
            setter(factor if (is_time and log) else 1.0)
        # A DURATION HAS NO ORIGIN; AN INSTANT DOES. `dwell_s`/`rise_s`/`fall_s` mean
        # the same thing on every recording, but `t_start_s` is "seconds since THIS
        # recording's first loaded sample" -- so the axis has to name its clock, and
        # say when it is showing more than one of them at once. The axis, not only
        # Details, because the axis label is what an exported image keeps.
        clock_tag = self._axis_clock_tag(col)
        # The user's override, if any, REPLACES THE TEXT ONLY: the tick scaling above has
        # already run, so a custom label sits over correctly scaled numbers. Overriding
        # both text and scale would let a label say "ms" over ticks showing seconds.
        if is_time and not log:
            axis.setScale(factor)
            return self._remember_axis_label(side, f"{col.removesuffix('_s')} ({name}{clock_tag})")
        axis.setScale(1.0)
        if is_time and log:
            return self._remember_axis_label(
                side, f"{col.removesuffix('_s')} ({name}{clock_tag}, log)")
        label = col
        if col == "depth_pA":
            label = "Median depth (depth_pA)"
        elif col == "amplitude_pA":
            units = amplitude_units_label(df for _src, df in self._drawn_frames())
            label = f"Selected amplitude ({units})"
        return self._remember_axis_label(side, label + (" (log)" if log else ""))

    def _remember_axis_label(self, side: str, auto: str) -> str:
        """Retain the rendered unit/clock caption, including when it is overridden."""
        self._axis_auto_labels[side] = auto
        override = self._style.x_label if side == "bottom" else self._style.y_label
        # Typed text is drawn literally (see _apply_title); blank: hidden (13.4 Q1).
        return rich_text(override.strip()) if override else auto

    def _axis_clock_tag(self, col: str) -> str:
        """The clock clause a time axis needs, or ``""`` for a duration.

        Short by design: :meth:`_column_notes` carries the t₀ values and the arithmetic
        in Details, and this is the half that has to survive inside an axis label.
        """
        if col not in _ORIGIN_COLUMNS:
            return ""
        try:
            n_sources = len([src for src, _df in self._drawn_frames()])
        except Exception:       # pragma: no cover - a label must never break a draw
            return ""
        if n_sources > 1:
            return ", from EACH recording's own start — origins differ"
        # One source: state ITS origin rather than assume zero. On the owner's own
        # files t₀ is 0 s, 10.2526 s and 61.3034 s, and it is the two non-zero ones
        # that make a number read off the trace disagree with this axis.
        srcs = [src for src, _df in self._drawn_frames()]
        off = self._clock_offset(srcs[0]) if srcs else None
        if off is None:
            return ", from recording start"
        return f", from recording start (t₀ = {clock.format_t0(off)} s)"

    def _plot_title(self, auto: str | None) -> str | None:
        """The title to show: the user override if any, else the automatic title. A blank
        override is no title (design 13.4 Q1)."""
        return self._style.title.strip() if self._style.title else auto

    def _apply_title(self, auto: str | None, warn: str | None = None) -> None:
        """Set the title from the override-or-auto text, in the style font size and colour.

        A FIT NEVER REACHES THIS. A fit's equation and numbers used to be the automatic
        title, drawn in the fit colour -- so typing a title took the result off the figure
        and clearing it put the equation back as the title. They are in the fit label now
        (:meth:`_draw_fit_label`), and the automatic title is what the plot shows without
        a fit (none, for a scatter or a histogram).

        The user's own text is drawn LITERALLY (``figure_text.rich_text``): "I < 5 pA &
        τ > 1 ms" used to lose everything between the two angle brackets, and ``_{…}`` /
        ``^{…}`` are the one markup, for sub- and superscripts. The element style set from
        the figure (Text style…) wins over the figure-wide title fields.

        ``warn`` is retained with the committed presentation for Details, but is
        not artwork. Comparability, calibration and fit verdicts belong in the
        warning summary, Details and saved provenance, never in the plot title.
        """
        # The middle slot was the fit accent; kept so the tuple's shape (and every reader
        # of its warn slot) is unchanged.
        self._title_presentation = (auto, None, warn)
        text = self._plot_title(auto)
        if not text:
            # setTitle(None) only HIDES pyqtgraph's title label -- the previous
            # draw's text stays on the item. Harmless in the exported PNG (a hidden,
            # zero-height row is not rendered), but it means "what does the title
            # say" stops being a function of the current draw: switch from an
            # incomparable frac_depth overlay to dwell_s and the item still reads
            # NOT COMPARABLE. Cleared explicitly so the verdict cannot outlive the
            # plot that earned it.
            self._plot.setTitle(None)
            self._plot.titleLabel.setText("")
            return
        style = self._style.effective_text_style("title")
        shown = rich_text(text) if self._style.title else text
        size = style.size_pt or self._title_pt()
        self._plot.setTitle(shown, color=style.color or self._fg_hex(), size=f"{size:g}pt",
                            justify=style.align or "center")
        # Set both positive and negative choices: an in-place restyle must also
        # remove a formerly selected family, bold weight, or italic slant.
        qf = self._qfont(int(round(size)))
        qf.setPointSizeF(float(size))
        if style.family:
            qf.setFamily(figure_family(style.family))
        qf.setBold(bool(style.bold))
        qf.setItalic(bool(style.italic))
        self._plot.titleLabel.item.setFont(qf)

    # -- axis range ------------------------------------------------------------------------------
    @staticmethod
    def _to_log(rng: tuple[float, float] | None, is_log: bool) -> tuple[float, float] | None:
        """A user range is typed in DATA units; pyqtgraph's setRange wants log10 on a log axis."""
        if rng is None:
            return None
        lo, hi = rng
        if is_log:
            if lo <= 0 or hi <= 0:
                return None   # a non-positive bound is meaningless on a log axis; treat as auto
            lo, hi = float(np.log10(lo)), float(np.log10(hi))
        return (lo, hi) if hi > lo else None

    def _read_range_pair(self, lo_key: str, hi_key: str,
                         is_log: bool = False) -> tuple[float, float] | None:
        """The typed pair, or ``None`` when it is not a usable window.

        REFUSED HERE RATHER THAN AT PIN TIME. ``_to_log`` already dropped a reversed pair
        and a non-positive bound on a log axis, but only on the way to the ViewBox -- so
        ``_manual_y`` was left holding ``(50.0, 1.0)``, a pin that is CONFIG (it is saved,
        restored and handed to an undo stack) and that nothing can ever apply. One rule,
        one place: a pair this declines is not stored, and
        :meth:`_axis_range_input_notes` says why in the same words.
        """
        lo_t = self._range_edits[lo_key].text().strip()
        hi_t = self._range_edits[hi_key].text().strip()
        if not lo_t or not hi_t:
            return None
        try:
            lo, hi = float(lo_t), float(hi_t)
        except ValueError:
            return None
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return None
        if is_log and lo <= 0:
            return None
        return lo, hi

    def _range_axis_columns(self) -> tuple[str, str]:
        """The ``(x, y)`` columns the CURRENT plot type puts on its two axes.

        One resolution, three readers: the unit factors the range boxes are read in,
        the unit names printed on their row labels, and the rescale that runs when
        either moves. Written once because the failure it prevents is precisely two
        of them disagreeing about which column the y axis is showing.
        ``""`` means "no column on that axis" — the histogram's y is a count, and
        PCA's axes are scores.
        """
        kind = self._plot_type()
        if kind == HISTOGRAM:
            return self._hist_col.currentText(), ""
        if kind == SCATTER:
            return self._scatter_x.currentText(), self._scatter_y.currentText()
        if kind == KDE:
            return self._kde_x.currentText(), self._kde_y.currentText()
        if kind == HEATMAP:
            return self._heatmap_x.currentText(), self._heatmap_y.currentText()
        return "", ""

    def _axis_unit_factors(self) -> tuple[float, float]:
        """(x, y) factors converting the DISPLAYED axis unit back to data units.

        The ticks on a time axis show the Time-unit combo's choice (ms by default),
        so the range boxes must be read in that same unit — the number the user just
        read off the axis (owner: "Axis range defaults to microseconds. The user
        needs control"). A non-time column's factor is 1.0: its axis shows data units.
        """
        return self.config().identity.axis_factors

    def _axis_unit_names(self) -> tuple[str, str]:
        """The unit each range box is read in, as text for its row label.

        From the SCHEMA that declares the column (:data:`_SCHEMA_UNITS`) for everything
        but a time column, which follows the Time-unit combo like its axis does.
        ``""`` for a dimensionless column, an axis with no column (a count, a PCA
        score), or a column the schema does not know."""
        unit = _TIME_UNITS.get(self._time_unit.currentText(), (1.0, "s"))[1]
        out = []
        for col in self._range_axis_columns():
            if not col:
                out.append("")
            elif col in _TIME_COLUMNS:
                out.append(unit)
            else:
                out.append(_SCHEMA_UNITS.get(col, ""))
        if self._plot_type_combo.currentText() == HISTOGRAM and self._norm_combo.currentText() == _NORM_DENSITY:
            from ..core.quantity_display import density_quantity
            out[1] = _plot_density_quantity(self._hist_col.currentText(), unit).display_unit
        return out[0], out[1]

    def _sync_range_unit_labels(self) -> None:
        """Write the unit the boxes are READ IN onto their own row labels.

        Owner note #4, second half: the Axis-range tooltip already said a time axis
        follows the Time-unit combo, but a tooltip is not on screen while you are
        typing into the box it describes — so "X min" over a box wanting milliseconds
        is where a µs/ms mix-up starts. The label says which."""
        if not hasattr(self, "_range_labels"):
            return   # may fire during construction, before the group exists
        x_unit, y_unit = self._axis_unit_names()
        for key, base, unit in (("xmin", "X min", x_unit), ("xmax", "X max", x_unit),
                                ("ymin", "Y min", y_unit), ("ymax", "Y max", y_unit)):
            self._range_labels[key].setText(f"{base} ({unit})" if unit else base)

    def _retune_range_boxes(self) -> None:
        """Rewrite the typed limits into the unit the axis NOW shows, then relabel.

        THE 1000x HAZARD THIS CLOSES. The boxes are read in the DISPLAY unit and stored
        in data units, so their text is only meaningful against the unit that was
        selected when it was typed. Switching Time unit from ms to µs, or the histogram
        column from ``dwell_s`` to ``depth_pA``, left "0.5" sitting in the box -- and the
        next Apply read it as 0.5 µs (a 1000x smaller window than the user pinned) or as
        0.5 s of picoamps. Nothing on screen said the number had changed meaning.

        NOW ONE CLAUSE OF ONE RULE, not a rule of its own. The retune lived here and
        covered the axis-range boxes only; the Fit-range boxes two rows above them are
        read in the SAME display unit and were left out, which is precisely the bug this
        round is fixing. Both are handled by :meth:`PlotConfig.rebased` now, and this
        name is kept as the door onto it because it is what the axis-range group's own
        code and comments have always called the operation.
        """
        self._rebase()

    def _on_time_unit_changed(self, _unit: str) -> None:
        """The axis unit moved: carry every typed limit with it, then redraw.

        EVERY typed limit -- the four axis-range boxes AND the two Fit-range boxes. The
        second half is the fix: the unit is not part of what the plot is OF, so a window
        the user typed keeps the physical span it described and only its text moves.
        A rescale, not a stale plot: the redraw below is safe because nothing about
        WHICH rows or columns are drawn has changed.
        """
        self._rebase()
        self._changed()

    def _on_axis_column_changed(self, _col: str, widget=None) -> None:
        """A column picker moved: retune the range boxes, blank the analysis-specific
        text, and put the plot behind the run gate.

        ``widget`` is the combo that moved, passed in rather than read off
        ``sender()`` because the gate has to tell a person's pick from
        ``set_histogram_column``'s -- see :meth:`_from_the_user`.
        """
        if self._updating:
            return   # `_repopulate_columns` rebases once, after the whole refill
        # THE ONE RULE. An axis column IS the analysis, so this one call unpins that
        # axis's range, drops a fit window typed against the old column, and blanks the
        # caption (owner note Q6, second half: "Dwell time (ms)" over a frac_depth axis
        # is a mislabelled figure, and the label is the half a saved PNG keeps).
        self._rebase()
        if widget is not None and self._from_the_user(widget):
            self._mark_stale("an axis column")
        else:
            self._armed_redraw()

    def _apply_axis_range(self) -> None:
        """Store the typed limits and pin them on the current plot, and SAY what was
        declined. Typed in the unit the axis DISPLAYS (see _axis_unit_factors); stored in
        data units."""
        fx, fy = self._axis_unit_factors()
        log_x, log_y = self._current_log_state()
        x = self._read_range_pair("xmin", "xmax", log_x)
        y = self._read_range_pair("ymin", "ymax", log_y)
        self._manual_x = None if x is None else (x[0] / fx, x[1] / fx)
        self._manual_y = None if y is None else (y[0] / fy, y[1] / fy)
        self._apply_stored_ranges(*self._current_log_state())
        self._sync_range_notes()
        # The PIN is config and this is the only place it moves, so the undo hook is
        # called by hand: `_wire_history`'s walk hooks CONTROLS, and pressing Apply
        # changes `_manual_x`/`_manual_y` without any control's own value having moved.
        self._history_touch()

    def _reset_axis_range(self, *, axis: str = "both") -> None:
        """Unpin ``"x"``, ``"y"`` or ``"both"``: clear those boxes and auto-fit that axis.

        ONE IMPLEMENTATION, THREE SCOPES. The all-or-nothing version threw the other
        axis's pin away as a side effect of wanting this one back, which is a silent
        edit to a setting the user did not touch.
        """
        do_x = axis in ("x", "both")
        do_y = axis in ("y", "both")
        if do_x:
            self._manual_x = None
            self._range_edits["xmin"].clear()
            self._range_edits["xmax"].clear()
        if do_y:
            self._manual_y = None
            self._range_edits["ymin"].clear()
            self._range_edits["ymax"].clear()
        vb = self._plot.getViewBox()
        vb.enableAutoRange(x=do_x, y=do_y)
        self._plot.autoRange()
        # An axis that is still pinned has just been auto-ranged along with the other
        # one by `autoRange()`, which fits BOTH; putting the surviving pin back is what
        # makes "Auto X" mean "auto x, and leave y alone" rather than "auto x, and quietly
        # unpin y as well".
        self._apply_stored_ranges(*self._current_log_state())
        self._sync_range_notes()
        self._history_touch()   # unpinning is a change; see `_apply_axis_range`

    def _axis_range_input_notes(self) -> list[str]:
        """What the Axis range boxes CONTAIN but did not do. The axis half of the rule
        :meth:`_fit_range_input_notes` states for the Fit range boxes.

        THREE SILENT NO-OPS, each a typed instruction the window declined:

        1. **Reversed limits.** ``_to_log`` ends ``return (lo, hi) if hi > lo else None``,
           so "5" over "1" pinned nothing at all — the same empty-interval refusal the
           Fit range boxes make, minus the sentence.
        2. **A non-positive bound on a log axis.** ``_to_log`` returns None for ``lo <= 0``
           because ``log10`` of it is not a number, and 0 is exactly what a person types
           for "start at the origin". The refusal is right; the silence is not, because
           the fix (type a small positive number, or untick Log x) is not guessable from
           an axis that simply did not move.
        3. **A half-filled pair.** ``_read_range_pair`` requires BOTH edges and returns
           None otherwise, so clearing one box unpins the whole axis. Stated rather than
           changed: a one-edged pin is a different feature (``manual_x`` is a pair, and
           the view has to be given both bounds), and inventing the missing edge from the
           data would be a limit the user never typed.

        Worded per axis and with the unit the box is read in, because "min ≥ max" over a
        pair of boxes labelled in µs is only actionable if you know which pair.
        """
        out: list[str] = []
        log_x, log_y = self._current_log_state()
        units = self._axis_unit_names()
        for name, lo_key, hi_key, is_log, unit in (
                ("X", "xmin", "xmax", log_x, units[0]),
                ("Y", "ymin", "ymax", log_y, units[1])):
            lo_t = self._range_edits[lo_key].text().strip()
            hi_t = self._range_edits[hi_key].text().strip()
            if not lo_t and not hi_t:
                continue
            tag = f" ({unit})" if unit else ""
            if not lo_t or not hi_t:
                filled, blank = ((f"{name} min", f"{name} max") if lo_t
                                 else (f"{name} max", f"{name} min"))
                out.append(
                    f"{blank}{tag} is blank, so {filled} was IGNORED and the {name} axis "
                    f"is auto-fitting the data. An axis is pinned by BOTH of its edges — "
                    f"type the other one, or clear this one to stop the amber.")
                continue
            try:
                lo, hi = float(lo_t), float(hi_t)
            except ValueError:
                out.append(
                    f"{name} range [{lo_t}, {hi_t}]{tag} is not a pair of numbers and was "
                    f"IGNORED; the {name} axis is auto-fitting the data.")
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                out.append(
                    f"{name} range [{lo_t}, {hi_t}]{tag} contains a non-finite bound and "
                    f"was IGNORED; the {name} axis is auto-fitting the data.")
            elif hi <= lo:
                out.append(
                    f"{name} range [{lo_t}, {hi_t}]{tag} is empty (min ≥ max), so BOTH "
                    f"bounds were IGNORED and the {name} axis is auto-fitting the data.")
            elif is_log and lo <= 0:
                out.append(
                    f"{name} min {lo_t}{tag} is not positive and the {name} axis is "
                    f"logarithmic, so the whole {name} range was IGNORED — log10 of it is "
                    f"not a number. Type a small positive bound, or untick Log "
                    f"{name.lower()}.")
        return out

    def _sync_range_notes(self) -> None:
        """Show or hide the Axis-range group's amber line. Idempotent; safe before the
        group exists (``set_axis_range`` can be called from a script at any time)."""
        if not hasattr(self, "_range_note"):
            return
        notes = self._axis_range_input_notes()
        self._range_note.setText("\n".join(notes))
        self._range_note.setVisible(bool(notes))

    def _current_log_state(self) -> tuple[bool, bool]:
        """Is each axis of the CURRENT plot logarithmic? One question, one answer.

        Delegates to :attr:`PlotConfig.log_state` rather than switching on the plot type
        a second time. The switch is written once, in the config, and this is the door
        onto it that the draw path and the range pin already call by this name.
        """
        return self.config().log_state

    def _apply_stored_ranges(self, log_x: bool, log_y: bool) -> None:
        vb = self._plot.getViewBox()
        xr = self._to_log(self._manual_x, log_x)
        yr = self._to_log(self._manual_y, log_y)
        if xr is not None:
            vb.setXRange(xr[0], xr[1], padding=0)
        if yr is not None:
            vb.setYRange(yr[0], yr[1], padding=0)

    def _finalize_view(self, log_x: bool, log_y: bool) -> None:
        """End every draw by auto-fitting, then re-pinning any manual limits the user set."""
        self._plot.autoRange()
        self._apply_stored_ranges(log_x, log_y)
