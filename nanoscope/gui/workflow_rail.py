"""The workflow strip — a thin horizontal breadcrumb that answers "where am I / what's next".

The app has grown many features with no sense of place. This strip sits directly above the
signal viewer and renders the fixed pipeline
``Open › Calibrate › Filter › Detect › Curate › Analyze`` from a pure
:class:`~nanoscope.gui.workflow_status.PipelineStatus`, one chip per stage:

* ``✓`` done (green) · ``●`` current (accent, emphasised) · ``○`` available-to-do (neutral)
  · ``–`` unavailable (dim);
* FILTER is shown as an optional ``○`` that never blocks and is never the current step.
* ANALYZE always opens a metadata-only home. Individual tools retain their own data gates.
  A CURATE tick means the whole table has recorded human review, not merely automatic acceptance.

Each chip's tooltip carries the stage's reason (what it does / why it is locked). Clicking a
chip emits :attr:`stageActivated` with the stage id; ``main_window`` routes an available stage
to its existing reveal action and shows a locked stage's reason in the status bar.

This replaces the old fixed-width **vertical** rail, which spent a 172 px left column on six
short words. Laid out horizontally the same information costs one ~28 px row, and the column
it used to own now belongs to the file manager. Because the strip spans the window it must
never be the thing that stops the window getting narrower, so it degrades through four
densities — full labels, labels without the "(optional)" note, the current stage's label
only, then glyphs only — and reports the glyphs-only width as its minimum.

This widget imports only Qt and the pure status module — never another panel.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .workflow_status import PIPELINE_ORDER, PipelineStatus, Stage, StageStatus

__all__ = ["WorkflowStrip"]

# Glyphs per state (kept as constants so tests read them symbolically).
GLYPH_DONE = "✓"
GLYPH_CURRENT = "●"
GLYPH_TODO = "○"
GLYPH_LOCKED = "–"

# Density levels, widest first — `_relayout` picks the widest one that fits. The "you are
# here" label is the last thing to go, because it is the one the strip exists to say.
DENSITY_FULL = "full"        # "✓ Open  ›  ○ Filter (optional)  › …"
DENSITY_COMPACT = "compact"  # "✓ Open  ›  ○ Filter  › …"
DENSITY_CURRENT = "current"  # "✓  ›  ○  ›  ● Detect  ›  –  › …"
DENSITY_MINIMAL = "minimal"  # "✓  ›  ○  › …"   (labels live in the tooltips)
DENSITIES: tuple[str, ...] = (
    DENSITY_FULL, DENSITY_COMPACT, DENSITY_CURRENT, DENSITY_MINIMAL)

# Colours consistent with the app's dark instrument theme (see gui/app.py).
_GREEN = "#81C784"       # done
_ACCENT = "#4FC3F7"      # current
_TODO = "#c3c9d1"        # available, to-do
_DIM = "#6b7178"         # unavailable / optional-idle
_CURRENT_BG = "#1d2a33"  # subtle accent-tinted highlight for the current chip
_SEP_COLOR = "#4a5058"   # the "›" between chips

_SEPARATOR = "›"
_OPTIONAL_SUFFIX = " (optional)"

# Geometry. The strip is chrome: one short row, never taller than a toolbar.
_MIN_HEIGHT = 24
_MAX_HEIGHT = 32
_GLYPH_W = 14
_CHIP_PAD = 6      # per-side content margin inside a chip
_CHIP_GAP = 6      # glyph-to-label gap inside a chip
_SEP_W = 14        # width budget for one "›" cell
_EDGE_PAD = 6      # per-side content margin of the strip itself


class _StageChip(QtWidgets.QFrame):
    """One clickable breadcrumb chip: a status glyph + the stage label."""

    clicked = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("StageChip")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._glyph_char = GLYPH_LOCKED
        self._label = ""
        self._optional = False
        self._current = False
        self._density = DENSITY_FULL

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(_CHIP_PAD, 0, _CHIP_PAD, 0)
        lay.setSpacing(_CHIP_GAP)
        self._glyph = QtWidgets.QLabel(GLYPH_LOCKED)
        self._glyph.setFixedWidth(_GLYPH_W)
        self._glyph.setAlignment(QtCore.Qt.AlignCenter)
        self._text = QtWidgets.QLabel("")
        lay.addWidget(self._glyph)
        lay.addWidget(self._text)

    # -- state -----------------------------------------------------------------------------
    def set_state(self, *, glyph: str, glyph_color: str, text: str, text_color: str,
                  bold: bool, current: bool, tooltip: str, optional: bool) -> None:
        self._glyph_char = glyph
        self._label = text
        self._optional = optional
        self._current = current
        self._glyph.setText(glyph)
        weight = "700" if bold else "500"
        self._glyph.setStyleSheet(f"color:{glyph_color}; background:transparent;")
        self._text.setStyleSheet(
            f"color:{text_color}; background:transparent; font-weight:{weight};")
        bg = _CURRENT_BG if current else "transparent"
        self.setStyleSheet(
            f"#StageChip {{ background:{bg}; border:1px solid transparent; border-radius:5px; }}"
            "#StageChip:focus { border:1px solid #4FC3F7; }")
        self.setToolTip(f"{text} — {tooltip}")
        self.setAccessibleName(f"{text}: {tooltip}")
        self._render_text()

    def set_density(self, density: str) -> None:
        self._density = density
        self._render_text()

    def _render_text(self) -> None:
        text = self.text_for(self._density)
        self._text.setVisible(bool(text))
        self._text.setText(text)

    # -- measurement (so the strip can choose a density before it is laid out) ---------------
    def text_for(self, density: str) -> str:
        """The label this chip shows at ``density`` ("" when only the glyph is shown)."""
        if density == DENSITY_MINIMAL:
            return ""
        if density == DENSITY_CURRENT:
            return self._label if self._current else ""
        if density == DENSITY_FULL and self._optional:
            return self._label + _OPTIONAL_SUFFIX
        return self._label

    def width_for(self, density: str) -> int:
        """The width this chip needs at ``density``, from font metrics — no layout required."""
        text = self.text_for(density)
        w = 2 * _CHIP_PAD + _GLYPH_W
        if text:
            w += _CHIP_GAP + self._text.fontMetrics().horizontalAdvance(text)
        return w

    # -- introspection ----------------------------------------------------------------------
    def glyph(self) -> str:
        return self._glyph_char

    def label(self) -> str:
        """The stage's name, independent of the density currently rendered."""
        return self._label

    def shown_text(self) -> str:
        """What the chip is actually rendering right now ("" at minimal density)."""
        return self._text.text() if self._text.isVisible() else ""

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            if not event.isAutoRepeat():
                self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class WorkflowStrip(QtWidgets.QFrame):
    """A thin, non-closable horizontal stepper over the analysis pipeline."""

    stageActivated = QtCore.Signal(str)   # a stage chip was clicked (Stage.value)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("WorkflowStrip")
        self.setStyleSheet(
            "#WorkflowStrip{background:#22272e; border:1px solid #333941; border-radius:5px;}")
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(
            max(_MIN_HEIGHT, min(_MAX_HEIGHT, self.fontMetrics().height() + 10)))

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(_EDGE_PAD, 0, _EDGE_PAD, 0)
        root.setSpacing(0)

        self._chips: dict[Stage, _StageChip] = {}
        self._separators: list[QtWidgets.QLabel] = []
        for i, stage in enumerate(PIPELINE_ORDER):
            if i:
                sep = QtWidgets.QLabel(_SEPARATOR)
                sep.setFixedWidth(_SEP_W)
                sep.setAlignment(QtCore.Qt.AlignCenter)
                sep.setStyleSheet(f"color:{_SEP_COLOR}; background:transparent;")
                self._separators.append(sep)
                root.addWidget(sep)
            chip = _StageChip()
            chip.clicked.connect(lambda s=stage: self.stageActivated.emit(s.value))
            self._chips[stage] = chip
            root.addWidget(chip)

        # Trailing stretch keeps the breadcrumb left-aligned instead of smeared across a wide
        # window; the chips are then free to keep their natural widths.
        root.addStretch(1)
        self._density = DENSITY_FULL
        self._status: PipelineStatus | None = None
        self._apply_density(DENSITY_FULL)

    # -- state -------------------------------------------------------------------------------
    def set_status(self, status: PipelineStatus) -> None:
        """Re-render every chip from a :class:`PipelineStatus`."""
        self._status = status
        for stage in PIPELINE_ORDER:
            self._apply(self._chips[stage], status.stage(stage))
        self._relayout()

    @staticmethod
    def _apply(chip: _StageChip, st: StageStatus) -> None:
        if st.current:
            glyph, gcolor, tcolor, bold = GLYPH_CURRENT, _ACCENT, _ACCENT, True
        elif st.done:
            glyph, gcolor, tcolor, bold = GLYPH_DONE, _GREEN, _TODO, False
        elif st.available:
            # An available step is a neutral to-do; the optional FILTER stays dim so it
            # reads as "you may" rather than "you must".
            glyph = GLYPH_TODO
            gcolor = _DIM if st.optional else _TODO
            tcolor = _DIM if st.optional else _TODO
            bold = False
        else:
            glyph, gcolor, tcolor, bold = GLYPH_LOCKED, _DIM, _DIM, False

        chip.set_state(glyph=glyph, glyph_color=gcolor, text=st.label, text_color=tcolor,
                       bold=bold, current=st.current, tooltip=st.reason,
                       optional=st.optional)

    # -- density: degrade instead of forcing the window wider ---------------------------------
    def width_for(self, density: str) -> int:
        """Total width the breadcrumb needs at ``density``."""
        return (2 * _EDGE_PAD
                + sum(c.width_for(density) for c in self._chips.values())
                + _SEP_W * len(self._separators))

    def density_for(self, width: int) -> str:
        """The widest density that fits ``width`` — glyphs-only is always the fallback."""
        for density in DENSITIES:
            if self.width_for(density) <= width:
                return density
        return DENSITIES[-1]

    def density(self) -> str:
        """Which of :data:`DENSITIES` is currently rendered."""
        return self._density

    def _apply_density(self, density: str) -> None:
        self._density = density
        for chip in self._chips.values():
            chip.set_density(density)

    def _relayout(self) -> None:
        chosen = self.density_for(self.width())
        if chosen != self._density:
            self._apply_density(chosen)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._relayout()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        return QtCore.QSize(self.width_for(DENSITY_FULL), self.height())

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802 (Qt override)
        # The glyphs-only width, NOT whatever is rendered right now — otherwise a wide strip
        # would floor the whole main window at its full-label width and the user could not
        # make the window (or the panes either side of it) narrower.
        return QtCore.QSize(self.width_for(DENSITY_MINIMAL), self.height())

    # -- introspection (widgets never reach in; tests + main_window read through these) -------
    def glyph(self, stage: Stage) -> str:
        """The glyph char currently shown for ``stage`` (``✓ ● ○ –``)."""
        return self._chips[stage].glyph()

    def label(self, stage: Stage) -> str:
        """``stage``'s name, whether or not the current density is rendering it."""
        return self._chips[stage].label()

    def shown_text(self, stage: Stage) -> str:
        """What ``stage``'s chip renders at the current density ("" when glyph-only)."""
        return self._chips[stage].shown_text()

    def tooltip(self, stage: Stage) -> str:
        """``stage``'s tooltip — its label followed by its reason."""
        return self._chips[stage].toolTip()

    def stage_chip(self, stage: Stage) -> _StageChip:
        """The chip widget for ``stage`` (exposed so a click can be simulated in tests)."""
        return self._chips[stage]
