"""The analysis launcher — a thin, always-visible row that says which analysis surfaces exist.

The five plot windows, the figure export and the event gallery were reachable only through
``&Analysis``, a flat ~14-item menu that on macOS lives at the top of the SCREEN, detached from
the window. Nothing in the visible layout said the density map, the ensemble overlay, the noise
spectrum, the I-V fit, the voltage-dependence series or the event gallery existed at all, and
the explanatory copy written for exactly that
purpose was set on ``QAction.setToolTip`` — where a native menu never renders it. This widget is
where that copy finally shows up, on a control the user can actually see.

It is the sibling of :class:`~nanoscope.gui.workflow_rail.WorkflowStrip` and sits directly under
it: the strip says *where you are* in ``Open › Calibrate › Filter › Detect › Curate › Analyze`;
this row expands that last stage into the seven surfaces it actually means. (CURATE keeps its
own chip one row up, which is why the Event INSPECTOR is deliberately NOT duplicated here — the
event GALLERY is, because the chip does not route to it and nothing else in the layout does
either.)

Three rules make it an accessibility fix rather than another place to get stuck:

* **A gated button stays visible.** It is greyed, never hidden and never removed — a dim
  "V-dependence" button still advertises that voltage dependence exists, which is the entire
  point. Discoverability cannot depend on the feature already being usable.
* **A gated button is still clickable, and clicking it explains itself.** Qt never delivers a
  click to a widget that is ``setEnabled(False)``, so a genuinely disabled button could not tell
  the user why it is disabled. The locked state is therefore styling, exactly as it is for a
  locked :class:`WorkflowStrip` chip: the button emits :attr:`surfaceActivated` either way and
  ``main_window`` decides between "trigger the action" and "post the reason to the status bar".
* **The reason is pushed in, never invented here.** ``main_window`` derives ``(enabled, reason)``
  from the ONE predicate that also drives the menu action's ``setEnabled``, so the greyed button,
  its tooltip and the menu can never disagree about reality.

Width: the central column can be ~820 px with both docks open, so like the strip this row must
never be the thing that stops the window getting narrower. It degrades through three densities —
full labels, short labels, glyphs only — and reports the glyphs-only width as its minimum. It
never scrolls and never clips below that minimum.

This widget imports Qt and nothing else — no sibling gui panel (AST-guarded); ``main_window``
is the sole wirer, as it is for every other panel.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets

__all__ = [
    "AnalysisLauncher",
    "Surface",
    "SURFACES",
    "SURFACE_ORDER",
    "SURFACE_POPULATIONS",
    "SURFACE_EVENT_PLOTS",
    "SURFACE_PSD",
    "SURFACE_IV",
    "SURFACE_VDEP",
    "SURFACE_FIGURES",
    "SURFACE_GALLERY",
    "DENSITIES",
    "DENSITY_FULL",
    "DENSITY_SHORT",
    "DENSITY_MINIMAL",
]

# -- surface ids (stable strings; main_window maps each to its EXISTING QAction) -----------------
SURFACE_POPULATIONS = "populations"    # PlotsWindow
SURFACE_EVENT_PLOTS = "event_plots"    # EventPlotWindow (density map / ensemble)
SURFACE_PSD = "psd"                    # PSDWindow
SURFACE_IV = "iv"                      # IVWindow
SURFACE_VDEP = "vdep"                  # VDepWindow
SURFACE_FIGURES = "figures"            # Export figure…
SURFACE_GALLERY = "gallery"            # EventGalleryWindow (the contact sheet)


@dataclass(frozen=True)
class Surface:
    """One launchable analysis surface.

    ``label`` is what the button says when there is room; ``short`` is the abbreviation it falls
    back to; ``glyph`` is what survives all the way down (the name then lives in the tooltip).
    The explanatory prose is NOT here — it is copied off the surface's existing QAction by
    :meth:`AnalysisLauncher.set_tooltips`, so there is exactly one copy of it in the app.
    """

    sid: str
    label: str
    short: str
    glyph: str


SURFACES: tuple[Surface, ...] = (
    # "Populations" has no honest abbreviation, so it does not get one — the short density
    # exists to save the width the OTHER six can spare, not to mangle every label.
    Surface(SURFACE_POPULATIONS, "Populations", "Populations", "◉"),
    Surface(SURFACE_EVENT_PLOTS, "Count heatmap / ensemble", "Count map", "▦"),
    Surface(SURFACE_PSD, "Power spectrum", "Noise", "∿"),
    Surface(SURFACE_IV, "I-V / conductance", "I-V", "╱"),
    Surface(SURFACE_VDEP, "V-dependence", "V-dep", "↗"),
    Surface(SURFACE_FIGURES, "Report figures", "Report", "↓"),
    # The seventh, and the only one that CURATES rather than plots. It is here for the
    # reason the other six are: it had no visible affordance at all — item 13 of the
    # ``&Analysis`` menu, no shortcut, and otherwise reachable only from inside another
    # window. The Event INSPECTOR stays out (the strip's CURATE chip is its always-visible
    # door); the gallery has no chip of its own, which is exactly the gap this row fills.
    Surface(SURFACE_GALLERY, "Event gallery", "Gallery", "▤"),
)
SURFACE_ORDER: tuple[str, ...] = tuple(s.sid for s in SURFACES)
_BY_ID: dict[str, Surface] = {s.sid: s for s in SURFACES}

# Density levels, widest first — `_relayout` picks the widest one that fits.
DENSITY_FULL = "full"        # "◉ Populations   ▦ Density / ensemble  …"
DENSITY_SHORT = "short"      # "◉ Pops   ▦ Density   ∿ Noise  …"
DENSITY_MINIMAL = "minimal"  # "◉  ▦  ∿  ╱  ↗  ↓  ▤"   (names live in the tooltips)
DENSITIES: tuple[str, ...] = (DENSITY_FULL, DENSITY_SHORT, DENSITY_MINIMAL)

# Colours consistent with the app's dark instrument theme and with the workflow strip above.
_ACCENT = "#4FC3F7"       # an available surface (and, brighter, an open one)
_TEXT = "#c3c9d1"         # an available surface's label
_DIM = "#6b7178"          # a locked surface — visible, plainly not usable yet
_OPEN_BG = "#1d2a33"      # the same accent-tinted highlight the strip's CURRENT chip uses
_HOVER_BG = "#2b323a"

# Geometry. One short row of chrome, never taller than the strip above it.
_MIN_HEIGHT = 24
_MAX_HEIGHT = 32
_GLYPH_W = 14
_BTN_PAD = 8       # per-side content margin inside a button
_BTN_GAP = 6       # glyph-to-label gap inside a button
_BTN_SPACING = 4   # gap between buttons
_EDGE_PAD = 6      # per-side content margin of the row itself

_LOCKED_PREFIX = "Unavailable: "


class _SurfaceButton(QtWidgets.QFrame):
    """One flat launcher button: a glyph + the surface's name.

    A QFrame rather than a QPushButton for the same reason the workflow strip's chip is one: it
    must keep receiving clicks while it is LOCKED, so it can answer "why not?" instead of
    swallowing the click. Keyboard activation is wired explicitly (Space / Enter) so the row is
    reachable by Tab as well as by mouse.
    """

    clicked = QtCore.Signal()

    def __init__(self, surface: Surface, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("SurfaceButton")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._surface = surface
        self._enabled = False
        self._reason = ""
        self._prose = ""
        self._open = False
        self._density = DENSITY_FULL

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(_BTN_PAD, 0, _BTN_PAD, 0)
        lay.setSpacing(_BTN_GAP)
        self._glyph = QtWidgets.QLabel(surface.glyph)
        self._glyph.setFixedWidth(_GLYPH_W)
        self._glyph.setAlignment(QtCore.Qt.AlignCenter)
        self._text = QtWidgets.QLabel(surface.label)
        lay.addWidget(self._glyph)
        lay.addWidget(self._text)

        self._restyle()
        self._render_text()

    # -- state ------------------------------------------------------------------------------
    def set_gate(self, enabled: bool, reason: str) -> None:
        """Grey/ungrey the button and record why it is locked. Never hides it."""
        self._enabled = bool(enabled)
        self._reason = reason if not self._enabled else ""
        self._restyle()

    def set_prose(self, prose: str) -> None:
        """The surface's explanatory copy, lifted off its existing QAction."""
        self._prose = prose or ""
        self._restyle()

    def set_open(self, is_open: bool) -> None:
        """Tint the button while its window is open (it re-focuses rather than re-opening)."""
        self._open = bool(is_open)
        self._restyle()

    def set_density(self, density: str) -> None:
        self._density = density
        self._render_text()

    # -- rendering --------------------------------------------------------------------------
    def _restyle(self) -> None:
        colour = _ACCENT if (self._enabled and self._open) else (
            _TEXT if self._enabled else _DIM)
        glyph_colour = _ACCENT if self._enabled else _DIM
        weight = "600" if self._open else "500"
        self._glyph.setStyleSheet(f"color:{glyph_colour}; background:transparent;")
        self._text.setStyleSheet(
            f"color:{colour}; background:transparent; font-weight:{weight};")
        bg = _OPEN_BG if self._open else "transparent"
        self.setStyleSheet(
            f"#SurfaceButton {{ background:{bg}; border:none; border-radius:5px; }}"
            f"#SurfaceButton:hover {{ background:{_HOVER_BG}; }}")
        tip = self.tooltip_text()
        self.setToolTip(tip)
        self.setAccessibleName(self._surface.label)
        self.setAccessibleDescription(tip)

    def tooltip_text(self) -> str:
        """The surface's prose, plus the gate reason when it is locked (both, never one)."""
        parts = [self._surface.label]
        if self._prose:
            parts.append(self._prose)
        if not self._enabled and self._reason:
            parts.append(_LOCKED_PREFIX + self._reason)
        return "\n\n".join(parts)

    def _render_text(self) -> None:
        text = self.text_for(self._density)
        self._text.setVisible(bool(text))
        self._text.setText(text)

    # -- measurement (so the row can choose a density before it is laid out) -----------------
    def text_for(self, density: str) -> str:
        """The label this button shows at ``density`` ("" when only the glyph is shown)."""
        if density == DENSITY_MINIMAL:
            return ""
        if density == DENSITY_SHORT:
            return self._surface.short
        return self._surface.label

    def width_for(self, density: str) -> int:
        """The width this button needs at ``density``, from font metrics — no layout required."""
        text = self.text_for(density)
        w = 2 * _BTN_PAD + _GLYPH_W
        if text:
            w += _BTN_GAP + self._text.fontMetrics().horizontalAdvance(text)
        return w

    # -- introspection ----------------------------------------------------------------------
    def surface(self) -> Surface:
        return self._surface

    def is_enabled(self) -> bool:
        """Whether the surface is available. NOT ``QWidget.isEnabled`` — see the class docstring:
        the widget stays Qt-enabled while locked so the click can post the reason."""
        return self._enabled

    def reason(self) -> str:
        return self._reason

    def shown_text(self) -> str:
        """What the button is actually rendering right now ("" at minimal density)."""
        return self._text.text() if self._text.isVisible() else ""

    # -- activation -------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AnalysisLauncher(QtWidgets.QFrame):
    """A thin, always-visible row of flat buttons — one per analysis surface."""

    surfaceActivated = QtCore.Signal(str)   # a button was clicked (surface id), locked or not

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("AnalysisLauncher")
        self.setStyleSheet(
            "#AnalysisLauncher{background:#22272e; border:1px solid #333941; "
            "border-radius:5px;}")
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(
            max(_MIN_HEIGHT, min(_MAX_HEIGHT, self.fontMetrics().height() + 10)))

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(_EDGE_PAD, 0, _EDGE_PAD, 0)
        root.setSpacing(_BTN_SPACING)

        self._buttons: dict[str, _SurfaceButton] = {}
        for surface in SURFACES:
            btn = _SurfaceButton(surface)
            btn.clicked.connect(lambda s=surface.sid: self.surfaceActivated.emit(s))
            self._buttons[surface.sid] = btn
            root.addWidget(btn)

        # Trailing stretch keeps the row left-aligned under the workflow strip instead of
        # smeared across a wide window; the buttons keep their natural widths.
        root.addStretch(1)
        self._density = DENSITY_FULL
        self._apply_density(DENSITY_FULL)

    # -- state pushed in by main_window (the sole wirer) --------------------------------------
    def set_tooltips(self, prose: dict[str, str]) -> None:
        """Adopt each surface's existing explanatory copy, keyed by surface id.

        ``main_window`` passes each surface's ``QAction.toolTip()`` straight through, so the
        prose written for the menu is REUSED rather than rewritten — and it renders here,
        which on a native macOS menu it never did.
        """
        for sid, text in prose.items():
            btn = self._buttons.get(sid)
            if btn is not None:
                btn.set_prose(text)

    def set_gate(self, sid: str, enabled: bool, reason: str = "") -> None:
        """Set one surface's availability and, when locked, the reason a click will post."""
        btn = self._buttons.get(sid)
        if btn is not None:
            btn.set_gate(enabled, reason)

    def set_open(self, sid: str, is_open: bool) -> None:
        """Indicate that this surface's (single, re-focusing) window is currently open."""
        btn = self._buttons.get(sid)
        if btn is not None:
            btn.set_open(is_open)

    # -- density: degrade instead of forcing the window wider ---------------------------------
    def width_for(self, density: str) -> int:
        """Total width the row needs at ``density``."""
        n = len(self._buttons)
        return (2 * _EDGE_PAD
                + sum(b.width_for(density) for b in self._buttons.values())
                + _BTN_SPACING * max(0, n - 1))

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
        for btn in self._buttons.values():
            btn.set_density(density)

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
        # The glyphs-only width, NOT whatever is rendered right now — otherwise a wide row
        # would floor the whole main window at its full-label width and the user could not
        # make the window (or the docks either side of it) narrower.
        return QtCore.QSize(self.width_for(DENSITY_MINIMAL), self.height())

    # -- introspection (main_window + tests read through these; nothing reaches inside) -------
    def surface_ids(self) -> tuple[str, ...]:
        """Every surface this row launches, in display order."""
        return SURFACE_ORDER

    def label(self, sid: str) -> str:
        """``sid``'s full name, whether or not the current density is rendering it."""
        return _BY_ID[sid].label

    def shown_text(self, sid: str) -> str:
        """What ``sid``'s button renders at the current density ("" when glyph-only)."""
        return self._buttons[sid].shown_text()

    def is_enabled(self, sid: str) -> bool:
        """Whether ``sid`` is available. A False here means GREYED, never hidden."""
        return self._buttons[sid].is_enabled()

    def reason(self, sid: str) -> str:
        """Why ``sid`` is locked ("" when it is available)."""
        return self._buttons[sid].reason()

    def tooltip(self, sid: str) -> str:
        """``sid``'s rendered tooltip — its name, its prose and (when locked) its reason."""
        return self._buttons[sid].toolTip()

    def is_open(self, sid: str) -> bool:
        """Whether ``sid``'s window is currently marked open."""
        return self._buttons[sid]._open

    def surface_button(self, sid: str) -> _SurfaceButton:
        """The button widget for ``sid`` (exposed so a click can be simulated in tests)."""
        return self._buttons[sid]
