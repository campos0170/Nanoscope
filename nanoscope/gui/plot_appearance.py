"""Shared, opt-in styling of existing plots, with no analysis callbacks.

Empty state preserves the original appearance. Clearing overrides restores the
original graphical properties. Text overrides never change units, arrays or fits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import json
from typing import Any, Mapping, Sequence

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .key_sequences import redo_sequences, undo_sequences

#: How many appearance values one widget's history keeps, and how long a run of edits
#: may stay open before it becomes one undo step. EQUAL to ``plots_panel``'s
#: ``_HISTORY_DEPTH`` and ``_HISTORY_COALESCE_MS``, on purpose: design §4.2 gives every
#: plot window the Plots window's depth and pause, and two windows of one app must not
#: disagree about how far back Undo reaches or how long a pause ends a run. Only those
#: two: what joins a run is this widget's own rule (any edit, a combo pick included),
#: where the Plots window commits every change but typing at once and ends typing when
#: the box loses focus (``PlotsWindow._history_touch``). Restated rather than imported,
#: because ``plots_panel`` hosts this widget and so imports THIS module.
#: tests/gui/test_plot_appearance_history.py holds the two pairs equal. No byte ceiling:
#: an entry is fifteen scalars, where the Plots window's measured 1.5 kB entries already
#: cost only ~320 kB at this depth.
APPEARANCE_HISTORY_DEPTH = 200
APPEARANCE_COALESCE_MS = 700

#: A Title / X label / Y label that draws nothing: what a box holding only spaces stores.
#: Empty (or never touched) means the window's automatic label.
HIDDEN_LABEL = " "

#: The sizes a figure draws, per field: ``from_mapping`` drops anything outside, and each
#: size spin box runs 0 ("Default") then this range, so the two cannot disagree.
_SIZE_RANGES: dict[str, tuple[float, float]] = {
    "title_size": (6, 48), "axis_size": (6, 36), "tick_size": (6, 30),
    "line_width": (.25, 10), "marker_size": (1, 30), "legend_size": (6, 30),
}


@dataclass(frozen=True)
class PlotAppearance:
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend_visible: bool | None = None
    legend_size: float | None = None
    font_family: str | None = None
    title_size: float | None = None
    axis_size: float | None = None
    tick_size: float | None = None
    grid: bool | None = None
    background: str | None = None
    data_color: str | None = None
    fit_color: str | None = None
    line_width: float | None = None
    marker_size: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe overrides only; an empty mapping means the original style."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "PlotAppearance":
        """Ignore unknown/invalid saved fields without changing analysis defaults."""
        values = values if isinstance(values, Mapping) else {}
        valid: dict[str, Any] = {}
        # One rule in every plot window (owner, design 13.4 Q1): an emptied box is the
        # automatic label, as one never touched, and a box holding only spaces hides the
        # label (kept as one space, :data:`HIDDEN_LABEL`).
        for key in ("title", "x_label", "y_label"):
            text = values.get(key)
            if isinstance(text, str) and text.strip():
                valid[key] = text.strip()[:1000]
            elif isinstance(text, str) and text:
                valid[key] = HIDDEN_LABEL
        family = values.get("font_family")
        if isinstance(family, str) and family.strip() and not any(c in family for c in ";{}<>"):
            valid["font_family"] = family.strip()[:200]
        for key, (low, high) in _SIZE_RANGES.items():
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value) and low <= value <= high:
                    valid[key] = float(value)
        if isinstance(values.get("grid"), bool):
            valid["grid"] = values["grid"]
        if isinstance(values.get("legend_visible"), bool):
            valid["legend_visible"] = values["legend_visible"]
        if values.get("background") in ("dark", "light"):
            valid["background"] = values["background"]
        for key in ("data_color", "fit_color"):
            color = values.get(key)
            if isinstance(color, str) and QtGui.QColor(color).isValid():
                valid[key] = QtGui.QColor(color).name()
        return cls(**valid)


#: The shipped looks, by their entry in the preset combo. Values, so that a history step
#: which lands exactly on one can be named after it ("Undo: Publication · light preset").
_BUILTIN_PRESETS: dict[str, PlotAppearance] = {
    "Publication · light": PlotAppearance(
        font_family="Arial", title_size=14, axis_size=12, tick_size=10, grid=False,
        background="light", data_color="#0072b2", fit_color="#d55e00", line_width=2,
        marker_size=8),
    "Presentation · dark": PlotAppearance(
        font_family="Arial", title_size=14, axis_size=12, tick_size=10, grid=True,
        background="dark", data_color="#0072b2", fit_color="#d55e00", line_width=2,
        marker_size=11),
}

#: How a step names each field, and the word for "no override" in that control's own
#: vocabulary: the size spins and colour buttons say "Default", the three-way combos
#: "Original", the label boxes' placeholder "Automatic". An emptied label box is that
#: too; a box of spaces (:data:`HIDDEN_LABEL`) draws no label, so a step names it "hidden".
_STEP_WORDS: dict[str, tuple[str, str]] = {
    "title": ("title", "automatic"), "x_label": ("x label", "automatic"),
    "y_label": ("y label", "automatic"), "legend_visible": ("legend", "original"),
    "legend_size": ("legend text", "default"), "font_family": ("font", "default"),
    "title_size": ("title size", "default"), "axis_size": ("axis label size", "default"),
    "tick_size": ("tick size", "default"), "grid": ("grid", "original"),
    "background": ("background", "original"), "data_color": ("data color", "default"),
    "fit_color": ("fit color", "default"), "line_width": ("line width", "default"),
    "marker_size": ("marker size", "default"),
}
_POINT_FIELDS = frozenset({"title_size", "axis_size", "tick_size", "legend_size"})
_TEXT_FIELDS = ("title", "x_label", "y_label")


def _spell(key: str, value) -> str:
    if value is None:
        return _STEP_WORDS[key][1]
    if key in _TEXT_FIELDS and not value.strip():
        return "hidden"
    if isinstance(value, bool):
        return "shown" if value else "hidden"
    if isinstance(value, (int, float)):
        return f"{value:g}"
    if key in _TEXT_FIELDS:
        return f"‘{value if len(value) <= 24 else value[:23] + '…'}’"
    return str(value)


def _describe_step(before: PlotAppearance, after: PlotAppearance) -> str:
    """Name the step from ``before`` to ``after`` for its Undo or Redo tooltip.

    Read off the two VALUES, so the stack holds nothing but appearances. A single field is
    named with both of its values, whatever moved it: a size typed back to Default is
    that, not a reset. Several fields that land exactly on a shipped preset are that
    preset, several that land on no override at all are a reset, and any other several
    are counted.
    """
    moved = [f.name for f in fields(PlotAppearance)
             if getattr(before, f.name) != getattr(after, f.name)]
    if len(moved) == 1:
        key = moved[0]
        old, new = getattr(before, key), getattr(after, key)
        said_old, said_new = _spell(key, old), _spell(key, new)
        if key in _POINT_FIELDS:    # "12 → 14 pt": the unit once, on a side with a number
            if new is not None:
                said_new += " pt"
            elif old is not None:
                said_old += " pt"
        return f"{_STEP_WORDS[key][0]} {said_old} → {said_new}"
    for name, preset in _BUILTIN_PRESETS.items():
        if after == preset:
            return f"{name} preset"
    if not after.to_dict():
        return "reset appearance"
    return f"{len(moved)} appearance changes"


class _AppearanceHistory:
    """A bounded undo/redo stack of :class:`PlotAppearance` VALUES, local to one widget.

    ``plots_panel._ConfigHistory``'s shape, and its arguments, without the origin
    machinery its import disclosures need: ``base`` is the value the controls are believed
    to show, ``_past`` what came before it (oldest first), ``_future`` what an undo
    stepped off. :meth:`note` records by DIFF, so a caller that says "something may have
    moved" too often costs one comparison, and walking a combo away and back pushes
    nothing. Qt-free.
    """

    __slots__ = ("base", "_past", "_future", "dropped")

    def __init__(self, base: PlotAppearance) -> None:
        self.base = base
        self._past: list[PlotAppearance] = []
        self._future: list[PlotAppearance] = []
        #: Steps thrown away at :data:`APPEARANCE_HISTORY_DEPTH`. Counted rather than
        #: merely dropped, because a silent bound looks exactly like a broken Undo button.
        self.dropped = 0

    def depth(self) -> tuple[int, int]:
        """``(steps back, steps forward)``."""
        return len(self._past), len(self._future)

    def last_step(self) -> tuple[PlotAppearance, PlotAppearance] | None:
        """``(before, after)`` of the step an undo would take back."""
        return (self._past[-1], self.base) if self._past else None

    def next_step(self) -> tuple[PlotAppearance, PlotAppearance] | None:
        """``(before, after)`` of the step a redo would replay."""
        return (self.base, self._future[-1]) if self._future else None

    def note(self, now: PlotAppearance) -> bool:
        """Record that the controls moved from :attr:`base` to ``now``. True if they had."""
        if now == self.base:
            return False
        self._past.append(self.base)
        self.base = now
        # A new step ends the redo branch: "forward" would lead somewhere no longer
        # reachable from here.
        self._future.clear()
        while len(self._past) > APPEARANCE_HISTORY_DEPTH:
            del self._past[0]
            self.dropped += 1
        return True

    def undo(self) -> PlotAppearance | None:
        if not self._past:
            return None
        self._future.append(self.base)
        self.base = self._past.pop()
        return self.base

    def redo(self) -> PlotAppearance | None:
        if not self._future:
            return None
        self._past.append(self.base)
        self.base = self._future.pop()
        return self.base

    def rebase(self, state: PlotAppearance) -> None:
        """Move the base WITHOUT recording a step."""
        self.base = state

    def reset(self, base: PlotAppearance) -> None:
        """Forget every step, and the ceiling count with them (``_ConfigHistory.reset``)."""
        self.base = base
        self._past.clear()
        self._future.clear()
        self.dropped = 0


class _SizeSpinBox(QtWidgets.QDoubleSpinBox):
    """A size whose 0 is Default and whose next value is the smallest size a figure draws.

    Run from 0 by single steps, the title, axis, tick and legend sizes showed 1-5 pt while
    ``from_mapping`` dropped them, so the figure and the history did nothing (sweep 10,
    S10-PLOT2). A step up from below ``floor`` lands on it and a step down below it lands on
    Default; a typed size below it is not taken until it grows (``"1"`` on the way to
    ``"12"``), and Return or leaving the box puts the last size back.
    """

    def __init__(self, floor: float, maximum: float) -> None:
        super().__init__()
        self._floor = floor
        self.setRange(0, maximum)

    def stepBy(self, steps: int) -> None:  # noqa: N802 -- Qt API
        target = self.value() + steps * self.singleStep()
        if not 0 < target < self._floor:
            super().stepBy(steps)
            return
        self.setValue(self._floor if steps > 0 else self.minimum())
        if self.style().styleHint(QtWidgets.QStyle.SH_SpinBox_SelectOnStep, None, self):
            self.selectAll()

    def validate(self, text: str, pos: int):
        state, text, pos = super().validate(text, pos)
        if (state == QtGui.QValidator.State.Acceptable
                and 0 < self.valueFromText(text) < self._floor):
            state = QtGui.QValidator.State.Intermediate
        return state, text, pos


class PlotAppearanceWidget(QtWidgets.QWidget):
    """Compact common controls. ``set_state`` is silent; user edits emit once.

    With ``history`` (the default) the widget keeps its own Undo and Redo of its
    appearance, apart from Edit ▸ Undo (design §4.2): every user change is one step, a run
    of edits closer together than :data:`APPEARANCE_COALESCE_MS` is one, and an undo or
    redo emits ``changed`` exactly as the user's own edit would, so the host re-applies it
    the same way. ``set_state`` is never a step. A host whose figure is replaced from
    outside -- a restore, a window blanked for an incoming project, a reset of everything
    -- calls :meth:`clear_history` after it.

    ``shortcuts`` binds the platform's Undo and Redo keys to that history, scoped to the
    widget's window. A host whose window already binds those keys passes False: the same
    key bound twice in one window is ambiguous, and Qt then fires NEITHER. The Plots window
    passes ``history=False`` -- its own configuration stack already holds appearance, and
    one window must not have two stacks.
    """

    changed = QtCore.Signal(object)

    def __init__(self, state: PlotAppearance | None = None, parent=None, *,
                 include_colors=True, compact=False, include_markers=True,
                 include_labels=True, include_legend=True, multi_view_labels=False,
                 history=True, shortcuts=True):
        super().__init__(parent)
        self._state = state or PlotAppearance()
        self._loading = False
        #: How deep this widget is inside its own ``changed`` emission. A ``set_state``
        #: arriving then is the host answering the value it was just handed, not an
        #: install from outside (see :meth:`set_state`).
        self._emitting = 0
        self._history = _AppearanceHistory(self._state) if history else None
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(["Choose appearance preset…", *_BUILTIN_PRESETS])
        self.preset.setToolTip("Appearance only: never changes data, fitting, axis scales, or analysis settings.")
        self.preset.activated.connect(self._apply_preset)
        self.reset_button = QtWidgets.QPushButton("Reset appearance")
        self.reset_button.setToolTip("Remove appearance overrides; keep all data and analytical settings.")
        self.reset_button.clicked.connect(self._reset)
        # Undo and Redo sit beside Reset, the only other buttons here that move the whole
        # appearance. Without a history the cell is the bare button, as it always was.
        self.undo_button = self.redo_button = self._burst = None
        reset_cell = self.reset_button
        if self._history is not None:
            self.undo_button = QtWidgets.QPushButton("Undo")
            self.undo_button.setAccessibleName("Undo appearance change")
            self.undo_button.clicked.connect(self.undo)
            self.redo_button = QtWidgets.QPushButton("Redo")
            self.redo_button.setAccessibleName("Redo appearance change")
            self.redo_button.clicked.connect(self.redo)
            for button in (self.undo_button, self.redo_button):
                # The text's own width, not the style's 80 px push-button floor. At the
                # floor the compact row wanted 308 px where the rest of the column wants
                # 287, so it would have widened every sidebar it sits in (measured
                # offscreen); at this width the column is as wide as it always was.
                button.setFixedWidth(button.fontMetrics().horizontalAdvance(button.text()) + 20)
            reset_cell = QtWidgets.QWidget()
            reset_cell.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
            row_layout = QtWidgets.QHBoxLayout(reset_cell)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for button in (self.reset_button, self.undo_button, self.redo_button):
                row_layout.addWidget(button)
            self._burst = QtCore.QTimer(self)
            self._burst.setSingleShot(True)
            self._burst.setInterval(APPEARANCE_COALESCE_MS)
            self._burst.timeout.connect(self._close_burst)
            if shortcuts:
                # Window-scoped, and live only while this widget is on screen: Qt matches
                # a WindowShortcut whose parent is visible and enabled in the active
                # window, so a page behind another Explorer tab, or an Appearance section
                # the workflow bar has folded away, does not answer. Each resolved key
                # once (key_sequences says why). A focused text box keeps the key for its
                # own typing, as everywhere else in the app.
                for keys, slot in ((undo_sequences(), self.undo), (redo_sequences(), self.redo)):
                    for key in keys:
                        shortcut = QtGui.QShortcut(key, self)
                        shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
                        shortcut.activated.connect(slot)
        # Three buttons need two columns of the four; the preset gives up the third, which
        # costs the grid nothing where the extra column would have cost it the row's width.
        wide = reset_cell is not self.reset_button
        layout.addWidget(self.preset, 0, 0, 1, 2 if wide else 3)
        layout.addWidget(reset_cell, 0, 2 if wide else 3, 1, 2 if wide else 1)
        self.font_family = QtWidgets.QComboBox()
        self.font_family.setEditable(True)
        self.font_family.addItems(["Default", *QtGui.QFontDatabase.families()])
        self.font_family.setMaxVisibleItems(12)
        self.font_family.setMinimumContentsLength(12)
        self.font_family.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.font_family.currentTextChanged.connect(
            lambda text: self._edit("font_family", None if text == "Default" else text))
        self._add_control(layout, 1, 0, "Font", self.font_family)
        self.background = QtWidgets.QComboBox()
        for label, value in (("Original", None), ("Light", "light"), ("Dark", "dark")):
            self.background.addItem(label, value)
        self.background.currentIndexChanged.connect(
            lambda: self._edit("background", self.background.currentData()))
        self._add_control(layout, 1, 2, "Background", self.background)
        for key, label, row, col in (
                ("title_size", "Title (pt)", 2, 0),
                ("axis_size", "Axis labels (pt)", 2, 2),
                ("tick_size", "Ticks (pt)", 3, 0),
                ("line_width", "Line width", 4, 0),
                ("marker_size", "Marker size", 4, 2)):
            spin = _SizeSpinBox(*_SIZE_RANGES[key])
            spin.setDecimals(1)
            spin.setSingleStep(.5 if key == "line_width" else 1)
            spin.setSpecialValueText("Default")
            spin.valueChanged.connect(lambda value, field=key: self._edit(field, value or None))
            setattr(self, key, spin)
            self._add_control(layout, row, col, label, spin)
        self.grid = QtWidgets.QComboBox()
        for label, value in (("Original", None), ("Show", True), ("Hide", False)):
            self.grid.addItem(label, value)
        self.grid.currentIndexChanged.connect(lambda: self._edit("grid", self.grid.currentData()))
        self._add_control(layout, 3, 2, "Grid", self.grid)
        for key, label, col in (("data_color", "Data color", 0), ("fit_color", "Fit color", 2)):
            button = QtWidgets.QPushButton("Default…")
            button.clicked.connect(lambda checked=False, field=key: self._choose_color(field))
            setattr(self, key, button)
            self._add_control(layout, 5, col, label, button)
        note = QtWidgets.QLabel("Appearance only. Axis scales, numerical values and scientific settings are preserved.")
        note.setWordWrap(True)
        layout.addWidget(note, 6, 0, 1, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        if not include_colors:
            for row, column in ((5, 0), (5, 1), (5, 2), (5, 3)):
                layout.itemAtPosition(row, column).widget().hide()
            # Category-colored views still share typography and canvas presets.
            # Their host ignores data/fit overrides so category identities stay intact.
        if not include_markers:
            layout.itemAtPosition(4, 2).widget().hide()
            self.marker_size.hide()
        if compact:
            pairs = [(layout.itemAtPosition(row, col).widget(),
                      layout.itemAtPosition(row, col + 1).widget())
                     for row, col in ((1, 0), (1, 2), (2, 0), (2, 2), (3, 0),
                                      (3, 2), (4, 0), (4, 2), (5, 0), (5, 2))]
            while layout.count():
                layout.takeAt(0)
            row = 0
            layout.addWidget(self.preset, row, 0, 1, 2)
            row += 1
            for label, control in pairs:
                if control.isHidden():
                    continue
                layout.addWidget(label, row, 0)
                layout.addWidget(control, row, 1)
                row += 1
            layout.addWidget(reset_cell, row, 0, 1, 2)
            layout.addWidget(note, row + 1, 0, 1, 2)
            layout.setColumnStretch(3, 0)
        columns = 2 if compact else 4
        row = layout.rowCount()
        for key, caption in (("title", "Title"), ("x_label", "X label"), ("y_label", "Y label")):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText("Automatic / per-view label")
            edit.setToolTip("Text only: axis scaling and numerical values stay unchanged. "
                            "Include the scientific unit in a custom axis label.")
            edit.textEdited.connect(lambda text, field=key: self._edit(field, text))
            setattr(self, key, edit)
            label = QtWidgets.QLabel(caption + (" (all views)" if multi_view_labels else ""))
            layout.addWidget(label, row, 0)
            layout.addWidget(edit, row, 1, 1, columns - 1)
            label.setVisible(include_labels)
            edit.setVisible(include_labels)
            row += 1
        self.legend_visible = QtWidgets.QComboBox()
        for label, value in (("Original", None), ("Show", True), ("Hide", False)):
            self.legend_visible.addItem(label, value)
        self.legend_visible.currentIndexChanged.connect(
            lambda: self._edit("legend_visible", self.legend_visible.currentData()))
        self.legend_size = _SizeSpinBox(*_SIZE_RANGES["legend_size"])
        self.legend_size.setSpecialValueText("Default")
        self.legend_size.valueChanged.connect(lambda value: self._edit("legend_size", value or None))
        for label, control in (("Legend", self.legend_visible), ("Legend text (pt)", self.legend_size)):
            caption = QtWidgets.QLabel(label)
            layout.addWidget(caption, row, 0)
            layout.addWidget(control, row, 1, 1, columns - 1)
            caption.setVisible(include_legend)
            control.setVisible(include_legend)
            row += 1
        self.user_presets = QtWidgets.QComboBox()
        self.user_presets.setToolTip("Reusable appearance only — no source, column, fit, bin or analysis settings.")
        self.user_presets.activated.connect(self._activate_user_preset)
        layout.addWidget(self.user_presets, row, 0, 1, columns)
        row += 1
        self.save_preset_button = QtWidgets.QPushButton("Save appearance preset…")
        self.delete_preset_button = QtWidgets.QPushButton("Delete selected preset")
        self.save_preset_button.clicked.connect(self._ask_save_preset)
        self.delete_preset_button.clicked.connect(self._delete_selected_preset)
        layout.addWidget(self.save_preset_button, row, 0, 1, columns)
        layout.addWidget(self.delete_preset_button, row + 1, 0, 1, columns)
        self.refresh_user_presets()
        self.set_state(self._state)

    @staticmethod
    def _add_control(layout, row, column, label, control):
        label_widget = QtWidgets.QLabel(label)
        label_widget.setBuddy(control)
        layout.addWidget(label_widget, row, column)
        layout.addWidget(control, row, column + 1)

    def state(self) -> PlotAppearance:
        return self._state

    def set_state(self, state: PlotAppearance | Mapping[str, Any]):
        """Adopt ``state`` silently: a restore, a recipe load, a host's reset or its answer.

        Never an undo step. From outside, it first closes any open run of edits -- that
        run was the user's and stays a step -- then moves the history's base without one.
        From INSIDE this widget's own ``changed``, where a host normalises or refuses the
        value it was just handed, it only moves the controls: the step being recorded is
        the user's, and it records what the host left on screen, so a refused edit records
        nothing.
        """
        state = state if isinstance(state, PlotAppearance) else PlotAppearance.from_mapping(state)
        outside = self._history is not None and not self._emitting
        if outside:
            self._close_burst()
        self._write(state)
        if outside:
            self._history.rebase(self._state)
            self._sync_history_buttons()

    def _write(self, state: PlotAppearance) -> None:
        """Put ``state`` on the controls without any of them reporting an edit."""
        self._state = state
        self._loading = True
        try:
            self.font_family.setCurrentText(self._state.font_family or "Default")
            for key in ("title_size", "axis_size", "tick_size", "line_width", "marker_size"):
                getattr(self, key).setValue(getattr(self._state, key) or 0)
            for key in ("title", "x_label", "y_label"):
                getattr(self, key).setText(getattr(self._state, key) or "")
            self.legend_size.setValue(self._state.legend_size or 0)
            self.legend_visible.setCurrentIndex(self.legend_visible.findData(self._state.legend_visible))
            for key in ("background", "grid"):
                combo = getattr(self, key)
                combo.setCurrentIndex(combo.findData(getattr(self._state, key)))
            for key in ("data_color", "fit_color"):
                button = getattr(self, key)
                value = getattr(self._state, key)
                button.setText(f"{value}…" if value else "Default…")
                button.setStyleSheet(f"border-bottom: 3px solid {value};" if value else "")
        finally:
            self._loading = False

    def set_legend_available(self, available: bool) -> None:
        for widget in (self.legend_visible, self.legend_size):
            widget.setEnabled(available)
            widget.setToolTip("Appearance of existing legend entries." if available else
                              "This view has no legend entries; no data or legend is invented.")

    _PRESETS_KEY = "plot_appearance/user_presets_v1"

    @classmethod
    def user_preset_states(cls) -> dict[str, dict]:
        raw = QtCore.QSettings().value(cls._PRESETS_KEY, "{}")
        try:
            values = json.loads(raw) if isinstance(raw, str) else {}
            if not isinstance(values, dict):
                return {}
            return {name: PlotAppearance.from_mapping(state).to_dict()
                    for name, state in values.items()
                    if isinstance(name, str) and name.strip() and isinstance(state, dict)}
        except (TypeError, ValueError):
            return {}

    def refresh_user_presets(self) -> None:
        selected = self.user_presets.currentData()
        with QtCore.QSignalBlocker(self.user_presets):
            self.user_presets.clear()
            self.user_presets.addItem("Choose saved appearance…", None)
            for name in sorted(self.user_preset_states(), key=str.casefold):
                self.user_presets.addItem(name, name)
            self.user_presets.setCurrentIndex(max(0, self.user_presets.findData(selected)))

    def showEvent(self, event) -> None:
        # Another open analytical page may have saved a preset since construction.
        self.refresh_user_presets()
        super().showEvent(event)

    def save_user_preset(self, name: str, *, overwrite: bool = False) -> bool:
        name = str(name).strip()[:100]
        if not name:
            return False
        values = self.user_preset_states()
        if name in values and not overwrite:
            return False
        values[name] = self.state().to_dict()
        settings = QtCore.QSettings()
        settings.setValue(self._PRESETS_KEY, json.dumps(values, sort_keys=True, allow_nan=False))
        settings.sync()
        self.refresh_user_presets()
        self.user_presets.setCurrentIndex(self.user_presets.findData(name))
        return settings.status() == QtCore.QSettings.NoError

    def apply_user_preset(self, name: str) -> bool:
        values = self.user_preset_states()
        if name not in values:
            return False
        self._apply(PlotAppearance.from_mapping(values[name]))
        return True

    def delete_user_preset(self, name: str) -> bool:
        values = self.user_preset_states()
        if name not in values:
            return False
        del values[name]
        settings = QtCore.QSettings()
        settings.setValue(self._PRESETS_KEY, json.dumps(values, sort_keys=True, allow_nan=False))
        settings.sync()
        self.refresh_user_presets()
        return settings.status() == QtCore.QSettings.NoError

    def _activate_user_preset(self, index: int) -> None:
        name = self.user_presets.itemData(index)
        if name:
            self.apply_user_preset(name)

    def _ask_save_preset(self) -> None:
        name, accepted = QtWidgets.QInputDialog.getText(self, "Save appearance", "Preset name:")
        if not accepted or not name.strip():
            return
        overwrite = name.strip()[:100] in self.user_preset_states()
        if overwrite and QtWidgets.QMessageBox.question(
                self, "Replace appearance preset?", f"Replace the saved appearance ‘{name.strip()[:100]}’?") != QtWidgets.QMessageBox.Yes:
            return
        self.save_user_preset(name, overwrite=overwrite)

    def _delete_selected_preset(self) -> None:
        name = self.user_presets.currentData()
        if name and QtWidgets.QMessageBox.question(
                self, "Delete appearance preset?", f"Delete the saved appearance ‘{name}’?") == QtWidgets.QMessageBox.Yes:
            self.delete_user_preset(name)

    def _edit(self, key, value):
        """A control moved: one property, held open as part of a run of edits.

        Spin boxes, the editable font box and the label boxes report every keystroke, so
        "14" typed into Title (pt) is two values a few hundred milliseconds apart. The run
        becomes ONE step once :data:`APPEARANCE_COALESCE_MS` passes without another edit,
        or sooner when Undo, Redo, a one-gesture change or ``set_state`` closes it.
        """
        if self._loading:
            return
        updated = PlotAppearance.from_mapping({**self._state.to_dict(), key: value})
        if updated != self._state:
            self._state = updated
            self._announce(updated)
            if self._history is not None:
                self._burst.start()
                self._sync_history_buttons()

    def _apply(self, state: PlotAppearance, *, always: bool = False) -> None:
        """A one-gesture change -- a preset, a saved appearance, a colour, Reset: one step.

        Recorded after the host has answered ``changed``, so the step holds what the host
        left on screen. ``always`` announces even a state these controls already show, as
        a built-in preset always has: the Plots window installs whole configurations
        without writing them back here, so its controls can lag the window, and the preset
        must still reach it. Announcing an unchanged state records no step.
        """
        self._close_burst()      # a run of edits still open is its own, earlier step
        if state == self._state and not always:
            return
        self._write(state)
        self._announce(state)
        self._close_burst()

    def _announce(self, state: PlotAppearance) -> None:
        self._emitting += 1
        try:
            self.changed.emit(state)
        finally:
            self._emitting -= 1

    def _choose_color(self, field):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(getattr(self._state, field) or "#0072B2"), self,
            "Choose data color" if field == "data_color" else "Choose fit color")
        if color.isValid():
            self._apply(PlotAppearance.from_mapping({**self._state.to_dict(), field: color.name()}))

    def _reset(self):
        self._apply(PlotAppearance())

    def _apply_preset(self, index):
        presets = tuple(_BUILTIN_PRESETS.values())
        if not 1 <= index <= len(presets):
            return
        self.preset.setCurrentIndex(0)
        self._apply(presets[index - 1], always=True)

    # -- history ------------------------------------------------------------------------
    def undo(self) -> bool:
        """Step the appearance back one change. True if it moved.

        A run of edits still open is committed first, so Undo right after typing takes
        back the typing rather than the step before it.
        """
        if self._history is None:
            return False
        self._close_burst()
        return self._step(self._history.undo, self._history.redo)

    def redo(self) -> bool:
        """Step the appearance forward one change. True if it moved."""
        if self._history is None:
            return False
        self._close_burst()
        return self._step(self._history.redo, self._history.undo)

    def clear_history(self) -> None:
        """Forget every step, for a host whose figure was just replaced.

        A step back would restyle what the window no longer shows, so the history ends
        with the data it was made on, as the Plots window's does
        (``PlotsWindow.reset_state``). Call it after the ``set_state`` that installs the
        replacement.
        """
        if self._history is None:
            return
        self._burst.stop()
        self._history.reset(self._state)
        self._sync_history_buttons()

    def _step(self, move, unmove) -> bool:
        """Move the stack one step with ``move`` and hand the host the state it lands on.

        ``unmove`` is ``move``'s inverse, for a host that refuses the step.
        """
        left = self._state
        state = move()
        if state is None:
            return False
        self._write(state)
        # ``changed`` exactly as after a user edit: the host re-applies and re-saves the
        # value the way it always does, so an undo cannot be styled differently from the
        # edit it takes back.
        self._announce(state)
        if state != left and self._state == left:
            # The host REFUSED it and put back what it showed, as Comparison figures does
            # when the figure will not render. Nothing moved, so the stack moves back too:
            # re-basing here would lose the step, and leave a Redo that replays a step to
            # where the plot already is.
            unmove()
            self._sync_history_buttons()
            return False
        # Re-based onto what is on the controls once the host has answered, as the Plots
        # window re-bases after its own installs: a base that disagreed with the controls
        # would make the next edit push a phantom step.
        self._history.rebase(self._state)
        self._sync_history_buttons()
        return True

    def _close_burst(self) -> None:
        """Close any open run of edits: one step, if the controls moved from the base."""
        if self._history is None:
            return
        self._burst.stop()
        self._history.note(self._state)
        self._sync_history_buttons()

    def _sync_history_buttons(self) -> None:
        """Enable and name Undo and Redo from the stack, counting an OPEN run of edits.

        An open run is offered to Undo (committing it is the first thing :meth:`undo`
        does) and withdraws Redo, which committing it will clear -- the Plots window's
        rule (``PlotsWindow._sync_config_buttons``), in its words.
        """
        history = self._history
        if history is None or self.undo_button is None:
            return
        pending = history.base != self._state
        back, forward = history.depth()
        want_undo, want_redo = bool(back or pending), bool(forward and not pending)
        # A focused button about to be disabled hands focus to its partner on the same
        # row first. Left to Qt, focus walks the tab chain, and the scroll area these
        # controls sit in scrolls after it -- measured in the Plots window, where it put a
        # different control under a pointer that had not moved
        # (``PlotsWindow._retire_history_button``).
        for dying, wanted, partner, partner_wanted in (
                (self.undo_button, want_undo, self.redo_button, want_redo),
                (self.redo_button, want_redo, self.undo_button, want_undo)):
            if not wanted and dying.hasFocus():
                if partner_wanted:
                    partner.setEnabled(True)
                    partner.setFocus()
                else:
                    dying.clearFocus()
        self.undo_button.setEnabled(want_undo)
        self.redo_button.setEnabled(want_redo)
        lost = history.dropped
        plural = "s" if lost != 1 else ""
        ceiling = (f". Its oldest history was dropped at the {APPEARANCE_HISTORY_DEPTH}-step "
                   f"ceiling ({lost} step{plural} lost).") if lost else ""
        step = (history.base, self._state) if pending else history.last_step()
        if step is not None:
            self.undo_button.setToolTip(f"Undo: {_describe_step(*step)}{ceiling}")
        elif lost:
            self.undo_button.setToolTip(
                f"Nothing left to undo — this plot's oldest appearance history was dropped "
                f"at the {APPEARANCE_HISTORY_DEPTH}-step ceiling ({lost} step{plural} lost).")
        else:
            self.undo_button.setToolTip("Nothing to undo yet.")
        step = history.next_step()
        self.redo_button.setToolTip(
            "Redo is cleared by a new change." if pending and step else
            f"Redo: {_describe_step(*step)}" if step else "Nothing to redo.")


def _item_defaults(item):
    defaults = getattr(item, "_nanoscope_appearance_defaults", None)
    if defaults is None:
        defaults = {key: item.opts.get(key) for key in
                    ("pen", "brush", "fillBrush", "size", "symbolPen", "symbolBrush", "symbolSize")}
        item._nanoscope_appearance_defaults = defaults
    return defaults


def _style_data_item(item, state, color):
    if not isinstance(item, (pg.PlotDataItem, pg.ScatterPlotItem, pg.BarGraphItem)):
        return
    defaults = _item_defaults(item)
    pen = defaults["pen"]
    if isinstance(item, pg.BarGraphItem):
        brush = defaults["brush"]
        if color:
            tint = QtGui.QColor(color)
            if brush is not None:
                tint.setAlpha(pg.mkBrush(brush).color().alpha())
            brush = pg.mkBrush(tint)
        if pen is not None or state.line_width is not None:
            pen = pg.mkPen(pen if pen is not None else
                           (color or pg.mkBrush(brush).color()))
            if color:
                pen.setColor(QtGui.QColor(color))
            if state.line_width is not None:
                pen.setWidthF(state.line_width)
        # Geometry fields (x/y/width/height) are never passed through this seam.
        item.setOpts(brush=brush, pen=pen)
    elif isinstance(item, pg.PlotDataItem):
        if pen is not None:
            pen = pg.mkPen(pen)
            if color:
                pen.setColor(QtGui.QColor(color))
            if state.line_width is not None:
                pen.setWidthF(state.line_width)
        item.setPen(pen)
        fill = defaults["fillBrush"]
        if fill is not None:
            brush = pg.mkBrush(fill)
            if color:
                tint = QtGui.QColor(color)
                tint.setAlpha(brush.color().alpha())
                brush = pg.mkBrush(tint)
            item.setFillBrush(brush)
        brush = defaults["symbolBrush"]
        if color:
            tint = QtGui.QColor(color)
            if brush is not None:
                tint.setAlpha(pg.mkBrush(brush).color().alpha())
            brush = pg.mkBrush(tint)
        item.setSymbolBrush(brush)
        item.setSymbolSize(state.marker_size if state.marker_size is not None else defaults["symbolSize"])
    else:
        brush = defaults["brush"]
        if color:
            tint = QtGui.QColor(color)
            if brush is not None:
                tint.setAlpha(pg.mkBrush(brush).color().alpha())
            brush = pg.mkBrush(tint)
        item.setBrush(brush)
        item.setSize(state.marker_size if state.marker_size is not None else defaults["size"])


def apply_plot_appearance(plot: pg.PlotItem, state: PlotAppearance, *,
                          plot_widget=None, data_items: Sequence = (),
                          fit_items: Sequence = (), title_prefix: str = "") -> None:
    """Style existing items, without ``clear``, ``setData``, fitting, or autoRange.

    Original graphical properties are recorded on first opt-in so Reset restores
    them. Data/fit items replaced by explicit analysis refresh acquire their own
    original styles. Existing manual zoom is retained.
    """
    defaults = getattr(plot, "_nanoscope_appearance_defaults", None)
    if defaults is None and not state.to_dict():
        return
    if defaults is None:
        defaults = {
            "title_opts": dict(plot.titleLabel.opts),
            "title_text": plot.titleLabel.text,
            "applied_title_text": plot.titleLabel.text,
            "axes": {side: {"label": dict(plot.getAxis(side).labelStyle),
                            "text": plot.getAxis(side).labelText,
                            "applied_text": plot.getAxis(side).labelText,
                            "font": plot.getAxis(side).style.get("tickFont"),
                            "pen": plot.getAxis(side).pen(),
                            "text_pen": plot.getAxis(side).textPen(),
                            "grid": plot.getAxis(side).grid}
                     for side in ("bottom", "left", "top", "right")},
            "background": plot_widget.backgroundBrush() if plot_widget is not None else None,
        }
        plot._nanoscope_appearance_defaults = defaults
    view = plot.getViewBox()
    ranges = [list(axis) for axis in view.viewRange()]
    auto = list(view.autoRangeEnabled())
    view.disableAutoRange()
    try:
        foreground = {"light": "#20242b", "dark": "#e6e9ef"}.get(state.background)
        if plot_widget is not None:
            background = {"light": "#ffffff", "dark": "#151a21"}.get(state.background)
            plot_widget.setBackground(background if background else defaults["background"])
        for side, original in defaults["axes"].items():
            axis = plot.getAxis(side)
            if axis.labelText != original["applied_text"]:
                original["text"] = axis.labelText
            override = state.x_label if side == "bottom" else state.y_label if side == "left" else None
            axis.labelText = override or original["text"]
            original["applied_text"] = axis.labelText
            label_style = dict(original["label"])
            if state.font_family:
                label_style["font-family"] = state.font_family
            if state.axis_size is not None:
                label_style["font-size"] = f"{state.axis_size:g}pt"
            if foreground:
                label_style["color"] = foreground
            axis.labelStyle = label_style
            axis._updateLabel()  # preserve text, units and automatic SI-prefix state
            font = original["font"]
            if state.font_family or state.tick_size is not None:
                font = QtGui.QFont(font or QtWidgets.QApplication.font())
                if state.font_family:
                    font.setFamily(state.font_family)
                if state.tick_size is not None:
                    font.setPointSizeF(state.tick_size)
            axis.setStyle(tickFont=font)
            axis.setPen(foreground or original["pen"])
            axis.setTextPen(foreground or original["text_pen"])
            axis.setGrid(original["grid"] if state.grid is None else
                         (0.15 * 255 if state.grid and side in ("bottom", "left") else False))
        title = plot.titleLabel
        if title.text != defaults["applied_title_text"]:
            defaults["title_text"] = title.text
        last_title_color = defaults.get("applied_title_color", defaults["title_opts"].get("color"))
        if title.opts.get("color") != last_title_color:
            # A fresh scientific render can deliberately recolor its fit/warning
            # title. Keep that original color available for palette reset.
            defaults["title_opts"]["color"] = title.opts.get("color")
        title.opts = dict(defaults["title_opts"])
        if state.font_family:
            title.opts["family"] = state.font_family
        if state.title_size is not None:
            title.opts["size"] = f"{state.title_size:g}pt"
        if foreground:
            title.opts["color"] = foreground
        title.setText(title_prefix + state.title if state.title else defaults["title_text"])
        # setTitle(None) hides the row. An authored title must make it visible
        # again; Reset to an empty automatic title, or a hidden one, must hide it
        # without a gap.
        shown = bool(title.text.strip())
        title.setVisible(shown)
        defaults["applied_title_text"] = title.text
        defaults["applied_title_color"] = title.opts.get("color")
        if shown:
            height = max(30.0, float(title.item.boundingRect().height()) + 6)
            title.setMaximumHeight(height)
            plot.layout.setRowFixedHeight(0, height)
        else:
            # Both, as pyqtgraph's own setTitle(None): the label's minimum height (its
            # text's, 23 px for a space) otherwise wins over the row's 0 and leaves a band.
            title.setMaximumHeight(0)
            plot.layout.setRowFixedHeight(0, 0)
        for item in data_items:
            _style_data_item(item, state, state.data_color)
        for item in fit_items:
            _style_data_item(item, state, state.fit_color)
        legend = getattr(plot, "legend", None)
        if legend is not None:
            if "legend" not in defaults:
                defaults["legend"] = {"visible": legend.isVisible(),
                                      "size": legend.opts.get("labelTextSize", "9pt"),
                                      "color": legend.opts.get("labelTextColor")}
            original = defaults["legend"]
            legend.setVisible(original["visible"] if state.legend_visible is None else
                              bool(state.legend_visible and legend.items))
            legend.setLabelTextSize(f"{state.legend_size:g}pt" if state.legend_size else original["size"])
            # Axis text already follows the canvas theme; legend text must do
            # the same or a white publication preset leaves pale labels on white.
            legend.setLabelTextColor(foreground or original["color"] or pg.getConfigOption("foreground"))
    finally:
        view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)
        for axis, enabled in enumerate(auto):
            if enabled:
                view.enableAutoRange(axis=axis, enable=enabled)
