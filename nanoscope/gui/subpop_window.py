"""The subpopulation report window — a read-and-export surface, not a panel.

``Analysis > Event subpopulations…`` runs the whole ported pipeline
(:mod:`nanoscope.analysis.subpop_report`) off the GUI thread and lands here. What
the user gets is the report the owner's notebook printed: the gate, the per-scope
PCA and clusterings, the cross-validated ``(n_PCs, K)`` choice, the population
table, the crosstab, the per-subpopulation waveform summary — and, at the bottom,
every caveat every stage raised.

THE CAVEATS ARE NOT AN APPENDIX. On the owner's own data the bootstrap intervals
carry the label-switching signature and are not calibrated; the BIC disagrees with
the cross-validation about K; below a few hundred events the mixture is fitting
more parameters than it has events. A window that showed the population table and
put the warnings behind a disclosure would be presenting numbers this pipeline
already knows are not measurements. So the count is on the header, the text is one
scrollable document, and Export writes the warnings into the sidecar too.

THE FIGURES (items 6-7 and the substance of item 13)
====================================================
The report already carries every number a figure needs — ``embeddings`` holds the
t-SNE coordinates the pipeline computes and deliberately never prints, ``labels``
and ``quantification.fit.labels`` hold the two independent label sets whose
crosstab the text prints, ``waveforms`` holds the per-subpopulation median
waveforms "as a table a plot can be drawn from". Nothing here computes; both views
read that object and draw it. If a view wants a number the report does not carry,
the view goes without it.

Two views, because the scientific question is two questions:

* :class:`EmbeddingScatterView` — the t-SNE cloud, coloured by GMM subpopulation
  or by either exploratory labelling. Recolouring ONE embedding by two methods
  that share no machinery is the whole point: "do the four GMM subpopulations look
  like four things?" is answered by whether the second method agrees, and on the
  owner's zn_30h it visibly does not (HDBSCAN finds two clusters and 229 noise
  points where the mixture reports four classes).
* :class:`WaveformOverlayView` — item 13's median waveforms on one axis, in the
  same colours, so a subpopulation's identity is one ink everywhere in the window.

THEY SIT BESIDE THE TEXT, NOT IN A TAB. A tabbed layout would put four tidy
coloured blobs on the tab the reader opens and the reasons they may not be four
things on a tab they do not, which is exactly the figure this pipeline should not
produce. So the window is a splitter: the whole report document on one side, the
figures on the other, and sharing the figures' own splitter — as a third pane, not
as a strip bolted underneath — a caveat box that repeats every warning verbatim.
The picture and the reasons to distrust it are on screen at the same time, at every
window size, and the sizes below are the measured floors that make "at every window
size" true rather than aspirational.

UMAP IS SURFACED ONLY IF THE RUN COMPUTED IT, and never turned on from here.
``random_state`` forces it single-threaded but the result is still not
bit-reproducible across umap-learn / numba / platform versions, so it is an
explicit opt-in at the analysis layer (``SubpopConfig(with_umap=True)``) and the
view says the coordinates are not stable whenever they are on screen.

Deliberately NOT a panel:

* it owns no session state and reads nothing — ``main_window`` hands it a finished
  :class:`~nanoscope.analysis.subpop_report.SubpopReport` and takes it away again;
  it persists nothing, so there is no QSettings key here to restore stale;
* it imports Qt, pyqtgraph, the app's palette (:mod:`nanoscope.plot.style`, which
  every window already asks for its ink by role) and ONE shared gui leaf
  (:mod:`.categorical`, the single categorical colour map, which imports no gui
  module itself) — no sibling PANEL and no window, and not the analysis layer
  either, so the report it renders is entirely the report's own words. AST-guarded,
  like every other window here: the sibling allow-list is a list of one and the next
  addition to it has to argue for itself in the test;
* Export announces intent (:attr:`SubpopReportWindow.exportRequested`) rather than
  running a file dialog itself, so the export goes through the one
  ``ask_export_path`` + overwrite-confirmation path every other export in the app
  uses, and the multi-file output set is confirmed before anything is written.

THE REST OF ITEM 13 IS ELSEWHERE, AND ON PURPOSE
================================================
The per-subpopulation event galleries and the coloured trace overlays are not built
here. They are views of the RECORDING -- tiles sliced out of the channel, spans over
the trace -- and this window owns no session state and holds no document; building
them here would mean either importing a panel or growing a second event renderer
beside the one :mod:`.event_gallery` already is. So this window offers the two
ENTRY POINTS (:attr:`SubpopReportWindow.galleryRequested`,
:attr:`SubpopReportWindow.traceRequested`) and ``main_window`` -- which has the
report, the session and both surfaces -- does the wiring, exactly as it does for
every other cross-surface move in the app. What travels is an ``event_id ->
subpopulation id`` map and the run's warnings; neither surface learns what a report
is, and neither writes the assignment onto the events frame.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..plot.style import PALETTE
from .categorical import (
    add_category_scatter,
    category_colour,
    category_counts,
    colours_repeat,
    legend_swatch,
)

__all__ = ["EmbeddingScatterView", "SubpopReportWindow", "WaveformOverlayView"]

#: Monospace, because the report is columnar: the population table, the loading
#: tables and the crosstab are all aligned by spaces and a proportional font
#: turns every one of them into a ragged block.
_FONT_FAMILIES = ("Menlo", "DejaVu Sans Mono", "Consolas", "Courier New")

#: Wide enough for the 100-column report AND a figure column beside it. The old
#: 960 was sized for the text alone; at that width the splitter would open with
#: figures too narrow to read, which is how a view becomes a view nobody uses.
#:
#: BOTH NUMBERS WERE MEASURED, not guessed, and both moved for a stated reason.
#: The height: at 860 the scatter's DATA rectangle came out 675x133 -- a 2-D
#: embedding drawn at five to one, in which a round blob is a streak and two
#: neighbouring classes are one arm. 980 is the aspect the owner's own figures are
#: read at. The width went 1440 -> 1600 when the legend moved out of the ViewBox
#: (see :func:`_legend`): the legend's ~164 px now costs plot WIDTH rather than
#: costing 62 hidden points, and 1600 buys that width back. Measured here at 980
#: high: data rectangle 670x213, aspect 3.15, 0 of 1076 points occluded.
#:
#: IT IS A PREFERRED SIZE AND NOT AN OPENING SIZE. See :func:`fitted_window_size`:
#: ``resize(*_DEFAULT_SIZE)`` was unconditional, and on the owner's 1512x900 display
#: it opened a window 88 px wider and 80 px taller than the screen, which put every
#: one of this window's five buttons off it. The pair below is what the geometry is
#: now clamped against; this number is only what the window asks for first.
_DEFAULT_SIZE = (1600, 980)
#: The floor is what the three stacked panes plus their captions actually need
#: (:data:`_MIN_PLOT_H` / :data:`_MIN_PLOT_W`, twice, the caveat strip, and the
#: labels between them). The old (520, 320) was a number that predated the figures:
#: dragged to 1100x700 the scatter rendered a 535x12 data rectangle -- a picture of
#: nothing, still captioned as though it were a picture of something. Qt now
#: refuses the size instead. The width went 760 -> 900 with the legend column: at
#: 760 the data rectangle was 197 px wide beside a 164 px legend, i.e. the legend
#: took 45 % of the figure. Measured at this floor: data 276x110, aspect 2.51.
_MIN_SIZE = (900, 620)
#: Initial splitter split: the text gets the width its widest table needs, the
#: figures get the rest. The user can drag it anywhere, including all the way over.
_SPLIT = (620, 800)

#: What a top-level window costs BEYOND the client area Qt sizes here, and how much
#: screen to leave around it. ``QWidget.resize`` and ``QWidget.move`` address the
#: client rectangle; the title bar sits above it and is part of what has to fit on
#: the screen. 28 px is the macOS title bar the app ships on; the 8 px margin keeps
#: the window off the screen edge, where a window is awkward to grab. Both are
#: budgeted rather than measured because the frame does not exist until the window
#: is first shown, and the size has to be right at the FIRST paint -- which is the
#: whole defect.
_FRAME_ALLOWANCE = (0, 28)
_SCREEN_MARGIN = 8


def fitted_window_size(available, preferred=_DEFAULT_SIZE,
                       minimum=_MIN_SIZE) -> QtCore.QSize:
    """The opening size for a window that would LIKE ``preferred``, on ``available``.

    ``available`` is a screen's available rectangle (a ``QRect``) or its size; the
    result is ``preferred`` shrunk until the window, title bar and margins included,
    fits inside it -- and never below ``minimum``, which is a floor this window's
    figures are measured against (see :data:`_MIN_SIZE`) and which Qt enforces
    anyway. A screen smaller than the floor therefore still gets the floor: a window
    that must be dragged is a lesser failure than a window drawn as a strip.

    WHAT IT FIXES, MEASURED. The shipped default is 1600x980 and it was applied
    unconditionally. On the owner's 1512x900 display that window opens with:

    ==========================  ================================================
    Copy report      y 947-972  72 px BELOW the bottom of the screen
    Export...        y 947-972  72 px below
    Event gallery... y 947-972  72 px below
    Colour the trace y 947-972  72 px below
    Close  x 1512-1592, y 947   72 px below AND 80 px past the right edge
    the standing note (live_note) 41 px below the bottom, 80 px past the right
    ==========================  ================================================

    -- i.e. zero pixels of any of the five buttons on the screen, before the macOS
    menu bar and this window's own title bar are subtracted. Everything this window
    can DO is on that row. At 1512x900 available this returns 1496x856, which puts
    the row at y 823-848 and Close at x 1408-1488: all five wholly on screen.
    """
    if isinstance(available, QtCore.QRect):
        available = available.size()
    pref_w, pref_h = int(preferred[0]), int(preferred[1])
    min_w, min_h = int(minimum[0]), int(minimum[1])
    room_w = available.width() - 2 * _SCREEN_MARGIN - _FRAME_ALLOWANCE[0]
    room_h = available.height() - 2 * _SCREEN_MARGIN - _FRAME_ALLOWANCE[1]
    return QtCore.QSize(max(min_w, min(pref_w, room_w)),
                        max(min_h, min(pref_h, room_h)))


def available_screen_rect(widget: QtWidgets.QWidget) -> QtCore.QRect:
    """The available rectangle of the screen ``widget`` will open on.

    A seam, and the reason it is one: the offscreen platform the tests run under
    reports an 800x800 screen, so a test that wants to know what this window does on
    the owner's 1512x900 display has to be able to say so. Patch this, not Qt.
    """
    screen = widget.screen() or QtGui.QGuiApplication.primaryScreen()
    if screen is None:                     # no screen at all: assume the preferred
        return QtCore.QRect(0, 0, _DEFAULT_SIZE[0], _DEFAULT_SIZE[1])
    return screen.availableGeometry()

#: Below this a plot is not a small plot, it is a strip. Enforced on the plot
#: widget itself so no arrangement of splitters can quietly go under it.
_MIN_PLOT_H = 150
#: Most legend rows in one column before the legend wraps into another column.
#:
#: THE FAILURE THIS BOUNDS, measured. Moving the legend into the PlotItem layout
#: (see :func:`_legend`) made its height a LAYOUT DEMAND rather than a decoration
#: floating over the data. With 25 HDBSCAN clusters the legend asked for 593 px
#: inside a 238 px widget and the ViewBox grew to match: 595 px of plot rendered
#: into a 238 px viewport, i.e. the bottom two thirds of the cloud clipped away
#: while the caption still read "1076 of 1076 event(s) drawn". At 40 classes the
#: ViewBox reached 940 px. That is the same lie the opaque plate told, rotated
#: onto the other axis, so it is bounded the same way: the legend wraps into
#: columns and grows into the width the figure has, never past the height it has.
#:
#: EIGHT IS THE CUT BECAUSE EIGHT IS WHAT THE MIXTURE CAN PRODUCE. The CV grid is
#: K=1-8, so every mixture colouring stays one column and keeps the full 675 px of
#: width; only HDBSCAN, whose cluster count is data-dependent and uncapped, ever
#: reaches a second column. Measured across the range at the default size (data
#: rectangle width x height, widget height 238-253): 2/4/8 classes 675x214 — one
#: column, nothing given up; 11 -> 538x199; 16 -> 532x204; 25 -> 234x199;
#: 40 -> 85x204. Nothing clips at any of them. A 40-cluster legend leaves a figure
#: too narrow to read, but it is narrow ON SCREEN and admits it, which is the whole
#: difference from cropping the cloud and captioning it as complete.
#:
#: IT IS A CAP AND NOT THE CAP. "The default size" above was 1600x980, which does not
#: fit the owner's 1512x900 screen; at the size that does, the embedding's plot is
#: 168 px and eight rows demand 202, so the row cap alone cropped the figure again on
#: the only display this app is used on. :func:`_fit_legend` therefore takes the
#: height it actually has as well, and this stays the ceiling on rows per column
#: rather than the whole rule.
_LEGEND_ROWS = 8
#: The same argument on the other axis, and it became load-bearing when the legend
#: moved OUT of the ViewBox into its own layout column (see :func:`_legend`): the
#: legend's ~164 px now comes off the plot's width instead of off the visible
#: points. A t-SNE squeezed narrow is the 5:1 streak again, rotated. Measured: at
#: this floor the data rectangle is ~200 px wide beside a ~164 px legend, which is
#: the narrowest the cloud stays round at.
_MIN_PLOT_W = 470
#: Enough for two caveats without scrolling; the strip is in the splitter with the
#: figures, so the reader can drag it larger and can never drag it away.
_MIN_CAVEAT_H = 104

_POINT_SIZE = 6
_WAVE_WIDTH = 2

#: The GMM assignment is not a column of ``report.labels`` — it lives on
#: ``quantification.fit`` — so it gets its own entry rather than being joined in.
_MODE_GMM = "GMM subpopulation (the mixture)"

#: The claim is argued in ``subpop_report.scope_embeddings``' docstring, which is
#: where it belongs — the analysis layer knows why the embedding never sees a label.
#: It is restated here as a STRING rather than imported because this window does not
#: import the analysis layer, and the drawn picture is the one place a reader is most
#: likely to over-read the coordinates. A test pins that the load-bearing clause is
#: still present in that docstring, so this sentence cannot outlive the argument
#: behind it.
_EMBED_HONESTY = (
    "The embedding never saw a cluster label, so the picture cannot be accused of "
    "drawing the clusters it is used to display — and by the same token nothing may be "
    "concluded from the coordinates. Axes are arbitrary units; distance, density and "
    "blob size in this picture are not measurements. Colour is the only information here."
)

_UMAP_OFF = (
    "UMAP is OFF for this run and that is not a slow-path excuse: `random_state` forces it "
    "single-threaded but the result is still not bit-reproducible across umap-learn / numba / "
    "platform versions, so a run and a re-run can disagree about a column that looks like data. "
    "Turn it on deliberately at the analysis layer — SubpopConfig(with_umap=True)."
)
_UMAP_ON = (
    "THESE UMAP COORDINATES ARE NOT STABLE. They are not bit-reproducible across "
    "umap-learn / numba / platform versions: a re-run of this same report can place the same "
    "events elsewhere. Read the t-SNE view before quoting this one."
)

#: The one caveat that is about the PICTURE rather than the pipeline, and the
#: reason it has to be said: the two colourings share a palette but not a meaning.
_COLOUR_CAVEAT = (
    "VIEW: colour is a class id WITHIN ONE COLOURING. Subpopulation 2 and HDBSCAN cluster 2 "
    "share an ink and are not the same set of events — the crosstab in the report text is the "
    "correspondence, not the colour. Grey is always \"in no class\"."
)

_NO_REPORT = "No report yet — run Analysis \u25b8 Event subpopulations (GMM)\u2026"

#: What ``main_window`` can tell this window about the report's per-event ASSIGNMENT —
#: the ``event_id -> subpopulation`` map it read off the report and pushed onto the trace
#: and the gallery. The two entry buttons are gated on it (:meth:`SubpopReportWindow.
#: set_assignment_state`), NOT on ``report.quantification``: that is a property of the
#: stored document, and gating on it kept the buttons live after a merge had retired
#: the map, so "Colour the trace" then reported "That report fitted no mixture" two
#: lines under a header saying "1 subpopulation(s)" (S7-POP3, measured). The window does
#: not know what an assignment is; it repeats the state and the reason it is given.
ASSIGNMENT_NONE = "none"            # the report fitted no mixture: nothing to colour with
ASSIGNMENT_INSTALLED = "installed"  # on the trace and in the gallery right now
ASSIGNMENT_AWAY = "away"            # held; another detection is on screen
ASSIGNMENT_RETIRED = "retired"      # gone, for the reason given (detection changed, …)
ASSIGNMENT_STATES = (ASSIGNMENT_NONE, ASSIGNMENT_INSTALLED, ASSIGNMENT_AWAY,
                     ASSIGNMENT_RETIRED)

#: The standing line under the document never MOVES anything: it is one label that is
#: always in the layout, of a height that does not depend on its text, and only its text
#: changes. A label that appeared when there was something to say would shift the
#: splitter (and every figure in it) under the pointer the moment a curation made
#: elsewhere changed the accepted count — this app's rule is that nothing scrolls or
#: moves a widget under the pointer.
#:
#: THREE LINES, NOT TWO, AND THE REASON IS MEASURED. The line carries up to two
#: sentences: the AWAY/RETIRED reason, which names the recording BY FILE NAME, and the
#: S7-POP4 curation sentence under it. With the owner's own file name
#: ("500 pM Apo ferritin_1 x PBS_22 nm_0004_process _ -1000 mV_ 385-400 s time trace.abf")
#: that pair is 377 characters and wraps to three lines — 45 px at 15 px/line — at the
#: window's default 1600 px, at 1440 px and at the 900 px minimum alike; the two-line
#: budget was 34 px, so the whole "accepted now: 109 of 110 …" sentence was drawn as the
#: tops of its glyphs and the reader never saw it (sweep-7 fix-up C3, repro
#: repro_C3_livenote_clipped.py, C3_label_1600.png). A budget is not a guarantee on its
#: own — a file name can be arbitrarily long — so :meth:`SubpopReportWindow._fit_live_note`
#: elides anything past it at a word and puts the whole sentence in the tooltip. The
#: height still never changes, so nothing moves.
_LIVE_NOTE_LINES = 3

_CAPTION_CSS = "color: #8b9199;"
#: The app's own warning ink, ASKED FOR BY ROLE. This used to be the literal
#: ``#E69F00`` — which is ``PALETTE["warn"]`` spelled out by hand, i.e. a second
#: copy of a palette entry sitting in a caller, which is the exact shape of the
#: duplication ``gui.categorical`` exists to prevent. A test pins that neither this
#: file nor ``plots_panel`` writes a palette colour as a literal.
_NOTE_CSS = f"color: {PALETTE['warn']};"


def _mono_font() -> QtGui.QFont:
    font = QtGui.QFont()
    font.setFamilies(list(_FONT_FAMILIES))
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


def _caption(text: str = "") -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet(_CAPTION_CSS)
    lab.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def _plot_widget(x_label: str, y_label: str) -> pg.PlotWidget:
    w = pg.PlotWidget()
    pi = w.getPlotItem()
    pi.setLabel("bottom", x_label)
    pi.setLabel("left", y_label)
    pi.showGrid(x=True, y=True, alpha=0.15)
    pi.setMenuEnabled(False)
    # See _MIN_PLOT_H: a figure squeezed below this is not a smaller figure, it is
    # a misleading one, and it arrives silently -- the caption still says "1076 of
    # 1076 event(s) drawn" over a twelve-pixel band.
    w.setMinimumHeight(_MIN_PLOT_H)
    w.setMinimumWidth(_MIN_PLOT_W)
    return w


def _legend(plot: pg.PlotWidget) -> pg.LegendItem:
    """A legend in its OWN COLUMN, beside the data rather than on top of it.

    pyqtgraph's default legend is transparent and anchored inside the ViewBox, and
    both of these figures fill their own bounding box by construction -- a t-SNE
    spreads to its axes and the waveforms cross the whole axis. The default
    printed "subpop 2 (n = 269)" across the green arm of the cloud, so an earlier
    pass put an opaque plate under it. That made the legend legible by making the
    points beneath it invisible, and the caption underneath went on claiming
    "1076 of 1076 event(s) drawn". Measured on the owner's zn_30h at this window's
    own default size, the plate covered 62 of 1076 points -- and not evenly: 43 of
    subpop 2 (16 % of it) and 19 of subpop 3 (15 %), against 0 of subpops 0 and 1.
    Subpop 3 is the class the crosstab already shows is the least well supported,
    i.e. exactly the one a reader needs to see whole.

    So the legend goes in a layout column to the RIGHT of the ViewBox instead.
    ``PlotItem.layout`` is a QGraphicsGridLayout with the ViewBox at (2, 1); an
    item added at column 5 is laid out past the right axis and cannot overlap the
    data at any window size or aspect. It costs ~164 px of plot width, which is
    width the figures have (see :data:`_MIN_PLOT_W`) and points are not.

    The opaque plate stays: the legend now sits on the plot background rather than
    on the data, where the plate reads as a panel and separates it from the axis.
    """
    legend = pg.LegendItem(
        offset=None,
        brush=pg.mkBrush(17, 19, 22, 216),
        pen=pg.mkPen("#3a3f46"),
        labelTextColor="#d7dbe0",
    )
    plot.getPlotItem().layout.addItem(legend, 2, 5)
    # addLegend() would have done this; we bypass it to control the anchoring, so
    # the back-reference the PlotItem keeps has to be set by hand or a later
    # addLegend() elsewhere would silently build a SECOND legend.
    plot.getPlotItem().legend = legend
    return legend


def _legend_room(plot: pg.PlotWidget) -> float:
    """The height the legend's own layout row actually has, in pixels.

    The legend sits at ``PlotItem.layout`` (2, 5) — the ViewBox's row — so what it
    has is the widget minus the axes above and below it. Measured rather than
    assumed, because it is the number :func:`_fit_legend` has to respect and it
    changes every time the window does.
    """
    item = plot.getPlotItem()
    used = 0.0
    for name in ("top", "bottom"):
        axis = item.getAxis(name)
        if axis is not None and axis.isVisible():
            used += float(axis.height())
    return max(0.0, float(plot.height()) - used)


def _fit_legend(legend: pg.LegendItem, n_entries: int, room: float = 0.0) -> None:
    """Wrap ``legend`` into as many columns as it needs to stay inside its own row.

    Called AFTER the entries are added, because the row count is what decides the
    column count. See :data:`_LEGEND_ROWS` for the clipping this prevents: in a
    layout column the legend's height is a demand on the plot, so an unbounded
    list of classes silently crops the figure it is labelling.

    ``room`` is :func:`_legend_room`, and it is why :data:`_LEGEND_ROWS` alone is not
    enough. MEASURED, with 8 HDBSCAN classes: eight rows demand 202 px, which the
    223 px plot at this window's old 1600x980 default had and the 168 px plot at the
    size that FITS the owner's 1512x900 screen does not — so the ViewBox grew to
    202 px inside a 168 px widget and the bottom of the cloud was cropped away while
    the caption underneath still read "220 of 220 event(s) drawn". Two figures and a
    caveat strip cannot each keep a 205 px plot inside a 900 px screen, so on that
    display the row cap could never have held; the height has to be asked for.

    The remedy is the one :data:`_LEGEND_ROWS` already chose on the other axis: a
    legend that will not fit its height wraps into another column and spends the
    figure's WIDTH instead. A narrow cloud is narrow on screen and admits it; a
    cropped one is captioned as complete. ``room <= 0`` (a plot with no size yet)
    means "cap by rows alone", which is what the caller gets before the first layout
    and what the resize below then corrects.
    """
    rows = max(1, int(_LEGEND_ROWS))
    n = max(1, int(n_entries))
    cols = max(1, -(-n // rows))
    _set_legend_columns(legend, cols)
    if room <= 0.0:
        return
    # Empirical, not modelled: ask the legend how tall it came out and give it another
    # column until it fits. LegendItem.setColumnCount repacks synchronously, so the
    # bounding rectangle read here is the one the layout will use.
    while cols < n and legend.boundingRect().height() > room:
        cols += 1
        _set_legend_columns(legend, cols)


def _set_legend_columns(legend: pg.LegendItem, cols: int) -> None:
    """``setColumnCount`` only when it would change something.

    The fit runs from a resize, and a repack that changed nothing would still be a
    relayout inside a relayout.
    """
    if int(getattr(legend, "columnCount", 0) or 0) != int(cols):
        legend.setColumnCount(int(cols))


def _class_name(mode: str, cid: int) -> str:
    """Legend text for one class id under one colouring.

    The two sentinels are NOT the same thing and are not given the same word:
    HDBSCAN's ``-1`` is a density decision it made about that event, and the
    mixture has no ``-1`` at all.
    """
    if cid < 0:
        return "noise (−1)" if "hdbscan" in mode.lower() else "unassigned (−1)"
    if mode == _MODE_GMM:
        return f"subpop {cid}"
    return f"cluster {cid}"


class EmbeddingScatterView(QtWidgets.QWidget):
    """Items 6-7: one 2-D embedding of the pooled PCA space, recoloured on demand.

    Reads ``report.embeddings`` for the coordinates and ``report.labels`` /
    ``report.quantification.fit.labels`` for the colouring. Computes nothing.

    The colour combo is the view's whole argument. The same 1076 points under the
    mixture's four classes and under HDBSCAN's two-plus-noise is the picture that
    answers "are these four things?"; a single fixed colouring would be a picture
    that only agrees with itself.

    Public API (what the window and the tests use)::

        set_report(report)            # install, or clear with None
        colour_modes()                # the offered colourings, in order
        colour_mode() / set_colour_mode(name)
        embeddings() / embedding() / set_embedding(name)
        drawn()                       # (x, y, labels) actually on screen
        legend_entries()              # [(text, "#rrggbb"), …] in legend order
        caption() / message()
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._report = None
        self._drawn: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
        self._legend_entries: list[tuple[str, str]] = []
        self._message = ""

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QtWidgets.QLabel("Embedding"))
        self._embed_combo = QtWidgets.QComboBox()
        self._embed_combo.setToolTip(
            "Which 2-D embedding of the pooled PCA space to draw. Only the ones this "
            "run actually computed are offered.")
        self._embed_combo.currentTextChanged.connect(self._redraw)
        row.addWidget(self._embed_combo)
        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel("Colour by"))
        self._colour_combo = QtWidgets.QComboBox()
        self._colour_combo.setToolTip(
            "Which labelling paints the points. The mixture and the exploratory "
            "clusterings share no machinery, so recolouring the SAME cloud by both is "
            "the honest test of whether the classes are there.")
        self._colour_combo.currentTextChanged.connect(self._redraw)
        row.addWidget(self._colour_combo, 1)
        root.addLayout(row)

        self._plot = _plot_widget("embedding 1 (arb. units)", "embedding 2 (arb. units)")
        self._legend = _legend(self._plot)
        # THE LEGEND IS REFITTED WHEN THE PLOT IS RESIZED, and it has to be: the height
        # it must fit inside is not known at draw time. ``main_window`` installs the
        # report BEFORE it shows the window, so the first fit runs against a plot that
        # has not been laid out yet; without this the default view -- the mixture's own
        # colouring, up to eight classes -- opens cropped on any screen too short to
        # give the plot 205 px. Filtered on the PLOT and not on self, so the height read
        # is the one the plot already has rather than the one it is about to get.
        self._legend_n = 0
        self._plot.installEventFilter(self)
        root.addWidget(self._plot, 1)

        self._caption = _caption()
        root.addWidget(self._caption)
        self._honesty = _caption(_EMBED_HONESTY)
        root.addWidget(self._honesty)
        self._umap_note = QtWidgets.QLabel()
        self._umap_note.setWordWrap(True)
        self._umap_note.setStyleSheet(_NOTE_CSS)
        root.addWidget(self._umap_note)

        # A view that has never been given a report still says what it is waiting
        # for. Blank axes with no caption read as "there is nothing in this data".
        self._redraw()

    def eventFilter(self, obj, event) -> bool:      # noqa: N802 (Qt)
        """Refit the legend to the height the plot HAS, every time that changes.

        See :func:`_fit_legend`: the legend's height is a demand on the ViewBox, so
        one that no longer fits crops the figure it is labelling while the caption
        goes on claiming every point is drawn. The count is re-read from
        ``_legend_n`` rather than re-derived: this must not redraw anything, only
        re-column what is already there.
        """
        if obj is self._plot and event.type() == QtCore.QEvent.Type.Resize:
            _fit_legend(self._legend, self._legend_n, _legend_room(self._plot))
        return super().eventFilter(obj, event)

    # -- state ---------------------------------------------------------------
    def set_report(self, report) -> None:
        """Install a report (or ``None``), rebuild the combos, redraw."""
        self._report = report
        for combo, items in ((self._embed_combo, self._embedding_options(report)),
                             (self._colour_combo, self._colour_options(report))):
            prev = combo.currentText()
            blocked = combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            if prev in items:
                combo.setCurrentText(prev)
            combo.blockSignals(blocked)
            combo.setEnabled(bool(items))
        self._redraw()

    def embeddings(self) -> tuple[str, ...]:
        """The embeddings this report actually carries, in offer order."""
        return tuple(self._embed_combo.itemText(i)
                     for i in range(self._embed_combo.count()))

    def embedding(self) -> str:
        return self._embed_combo.currentText()

    def set_embedding(self, name: str) -> None:
        """Select an embedding by exact name.

        Raises:
            ValueError: naming one this report does not carry — a silent no-op
                would leave the caller reading t-SNE under a UMAP heading.
        """
        options = self.embeddings()
        if name not in options:
            raise ValueError(f"no embedding {name!r} in this report; have {list(options)}")
        self._embed_combo.setCurrentText(name)

    def colour_modes(self) -> tuple[str, ...]:
        return tuple(self._colour_combo.itemText(i)
                     for i in range(self._colour_combo.count()))

    def colour_mode(self) -> str:
        return self._colour_combo.currentText()

    def set_colour_mode(self, name: str) -> None:
        """Select a colouring by exact name.

        Raises:
            ValueError: naming one this report does not carry.
        """
        options = self.colour_modes()
        if name not in options:
            raise ValueError(f"no colouring {name!r} in this report; have {list(options)}")
        self._colour_combo.setCurrentText(name)

    def drawn(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(x, y, labels)`` for the points actually on screen."""
        return self._drawn

    def legend_entries(self) -> list[tuple[str, str]]:
        """``[(text, "#rrggbb"), …]`` in the order the legend lists them."""
        return list(self._legend_entries)

    def caption(self) -> str:
        return self._caption.text()

    def message(self) -> str:
        """Why nothing is drawn, or ``""`` when something is."""
        return self._message

    # -- options -------------------------------------------------------------
    @staticmethod
    def _embedding_options(report) -> list[str]:
        if report is None or report.embeddings is None:
            return []
        cols = list(report.embeddings.columns)
        return [name for name, prefix in (("t-SNE", "tsne_"), ("UMAP", "umap_"))
                if f"{prefix}x" in cols and f"{prefix}y" in cols]

    @staticmethod
    def _colour_options(report) -> list[str]:
        if report is None:
            return []
        out: list[str] = []
        if report.quantification is not None:
            out.append(_MODE_GMM)
        if report.labels is not None:
            for col in report.labels.columns:
                out.append(_mode_for_column(str(col)))
        return out

    def _label_series(self, mode: str):
        """The label series behind one colour mode, or ``None``."""
        rep = self._report
        if rep is None:
            return None
        if mode == _MODE_GMM:
            return None if rep.quantification is None else rep.quantification.fit.labels
        if rep.labels is None:
            return None
        for col in rep.labels.columns:
            if _mode_for_column(str(col)) == mode:
                return rep.labels[col]
        return None

    # -- drawing -------------------------------------------------------------
    def _redraw(self, *_args) -> None:
        # THE LEGEND MUST BE CLEARED BY HAND. ``PlotItem.clear()`` un-registers only
        # the items it owns, and the legend samples here are free-standing swatches
        # that were never added to the plot -- so on the second colouring the
        # legend listed the mixture's four subpopulations AND HDBSCAN's clusters at
        # once, which is a picture claiming seven classes.
        self._legend.clear()
        self._plot.getPlotItem().clear()
        self._legend_entries = []
        self._legend_n = 0
        self._drawn = (np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
        rep = self._report
        name = self.embedding()
        # Three states, and the DEFAULT one still speaks: a reader who never sees
        # a UMAP option should learn that it exists and why it is off, rather than
        # concluding this pipeline has one embedding.
        if rep is None:
            self._umap_note.setText("")
        elif name == "UMAP":
            self._umap_note.setText(_UMAP_ON)
        elif "UMAP" not in self.embeddings():
            self._umap_note.setText(_UMAP_OFF)
        else:
            self._umap_note.setText(
                "This run also computed UMAP — its coordinates are not stable across "
                "umap-learn / numba / platform versions. Read this view first.")

        if rep is None:
            self._set_empty(_NO_REPORT)
            return
        if not name:
            self._set_empty(
                "This run computed no embedding — the report text says why (the scope was "
                "not clustered, or it is above the event ceiling). Everything else in the "
                "report stands without it.")
            return

        mode = self.colour_mode()
        labels = self._label_series(mode)
        emb = rep.embeddings
        prefix = "tsne_" if name == "t-SNE" else "umap_"
        x_all = emb[f"{prefix}x"].to_numpy(dtype=np.float64)
        y_all = emb[f"{prefix}y"].to_numpy(dtype=np.float64)
        if labels is None:
            self._set_empty("No labelling is available to colour this embedding by.")
            return

        # Join BY INDEX LABEL, never by position: the mixture is fitted on one
        # scope and the embedding covers the pooled scope, so the two need not be
        # the same rows or the same order.
        joined = labels.reindex(emb.index)
        values = joined.to_numpy(dtype=np.float64)
        keep = np.isfinite(values)
        # NaN means "this event is not in this scope" and -1 means "this event IS
        # in this scope and the method put it in no cluster". Collapsing them (a
        # fillna(-1), say) would silently repaint every out-of-scope row as noise.
        n_unlabelled = int(np.count_nonzero(~keep))
        if not keep.any():
            self._set_empty(f"No event in this embedding carries a {mode} label.")
            return
        x, y = x_all[keep], y_all[keep]
        ids = values[keep].astype(np.int64)

        item = add_category_scatter(
            self._plot.getPlotItem(), x, y, ids, size=_POINT_SIZE)
        item.setZValue(1)
        counts = category_counts(ids)
        for cid, n in counts:
            swatch = legend_swatch(cid, size=_POINT_SIZE)
            self._legend.addItem(swatch, f"{_class_name(mode, cid)}  (n = {n})")
            self._legend_entries.append(
                (f"{_class_name(mode, cid)}  (n = {n})", category_colour(cid).name()))
        self._legend_n = len(counts)
        _fit_legend(self._legend, self._legend_n, _legend_room(self._plot))

        self._plot.getPlotItem().setLabel("bottom", f"{name} 1 (arb. units)")
        self._plot.getPlotItem().setLabel("left", f"{name} 2 (arb. units)")
        n_classes = sum(1 for cid, _n in counts if cid >= 0)
        bits = [
            f"{name} of the pooled PCA space · {x.size} of {x_all.size} event(s) drawn",
            f"colour = {mode} · {n_classes} class(es)"
            + (" + grey" if any(cid < 0 for cid, _n in counts) else ""),
        ]
        if n_unlabelled:
            bits.append(
                f"{n_unlabelled} event(s) carry no {mode} label and are NOT drawn "
                f"(out of that stage's scope — not the same thing as noise)")
        # More classes than the palette has inks. The legend still lists them all,
        # but two of its rows now carry the SAME swatch, so the picture stops being
        # readable without being told. HDBSCAN is the one that can get here.
        if colours_repeat(ids):
            bits.append(
                f"more classes than the palette has colours — SOME COLOURS REPEAT, "
                f"so two classes here share an ink; read the class off the legend "
                f"order, not off the colour")
        self._caption.setText(" · ".join(bits))
        self._message = ""
        self._honesty.setVisible(True)
        self._drawn = (x, y, ids)

    def _set_empty(self, why: str) -> None:
        self._message = why
        self._caption.setText(why)
        self._honesty.setVisible(False)


def _mode_for_column(col: str) -> str:
    """Combo text for one ``report.labels`` column.

    The column names are the pipeline's (``hdbscan``, ``kmeans3``, plus a
    ``_<scope>`` suffix per scope), and they are kept IN the label so a reader can
    match the combo against the crosstab and the exported table without guessing.
    """
    base, _, scope = col.partition("_")
    pretty = {"hdbscan": "HDBSCAN (exploratory)", "kmeans3": "k-means, k=3 (exploratory)"}
    head = pretty.get(base, base)
    return f"{head} — scope {scope}" if scope else head


class WaveformOverlayView(QtWidgets.QWidget):
    """Item 13's substance: every subpopulation's median waveform on one axis.

    Reads ``report.waveforms`` — the common-axis table the pipeline already builds
    — and its ``attrs["n_events"]``. Computes nothing, not even a mean.

    Colour is :func:`~.categorical.category_colour` of the subpopulation id, which
    is the same ink the scatter paints that subpopulation with, so a class has one
    identity in this window rather than one per figure.

    Public API::

        set_report(report)
        series()      # [(column, "#rrggbb", n_events), …] in id order
        caption() / message()
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._report = None
        self._series: list[tuple[str, str, int]] = []
        self._message = ""

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        self._plot = _plot_widget("normalised time T  (0 = onset, 1 = end)", "current (pA)")
        self._legend = _legend(self._plot)
        self._legend_n = 0            # see EmbeddingScatterView.__init__
        self._plot.installEventFilter(self)
        root.addWidget(self._plot, 1)
        self._caption = _caption()
        root.addWidget(self._caption)
        self._redraw()

    def eventFilter(self, obj, event) -> bool:      # noqa: N802 (Qt)
        """Refit the legend to the height the plot HAS, every time that changes.

        See :func:`_fit_legend`: the legend's height is a demand on the ViewBox, so
        one that no longer fits crops the figure it is labelling while the caption
        goes on claiming every point is drawn. The count is re-read from
        ``_legend_n`` rather than re-derived: this must not redraw anything, only
        re-column what is already there.
        """
        if obj is self._plot and event.type() == QtCore.QEvent.Type.Resize:
            _fit_legend(self._legend, self._legend_n, _legend_room(self._plot))
        return super().eventFilter(obj, event)

    def set_report(self, report) -> None:
        self._report = report
        self._redraw()

    def series(self) -> list[tuple[str, str, int]]:
        """``[(column_name, "#rrggbb", n_events), …]``, in subpopulation id order."""
        return list(self._series)

    def caption(self) -> str:
        return self._caption.text()

    def message(self) -> str:
        return self._message

    def _redraw(self) -> None:
        self._legend.clear()          # see EmbeddingScatterView._redraw
        self._plot.getPlotItem().clear()
        self._series = []
        self._legend_n = 0
        rep = self._report
        if rep is None:
            self._set_empty(_NO_REPORT)
            return
        wf = rep.waveforms
        if wf is None or not len(wf) or "T" not in wf.columns:
            self._set_empty(
                "No per-subpopulation waveforms in this run (no mixture was fitted, or "
                "waveforms were disabled). The report text says which.")
            return

        pi = self._plot.getPlotItem()
        # The event's own edges, so the padding either side reads as padding. Both
        # are axis furniture from the table's own coordinate, not new numbers.
        for t in (0.0, 1.0):
            pi.addItem(pg.InfiniteLine(
                pos=t, angle=90,
                pen=pg.mkPen("#5a6068", width=1, style=QtCore.Qt.PenStyle.DashLine)))

        T = wf["T"].to_numpy(dtype=np.float64)
        counts = wf.attrs.get("n_events", {}) or {}
        drawn = 0
        flat: list[str] = []
        for col in wf.columns:
            if col == "T":
                continue
            cid = _subpop_id(str(col))
            y = wf[col].to_numpy(dtype=np.float64)
            colour = category_colour(cid)
            if not np.any(np.isfinite(y)):
                # A column that could not be averaged is NOT drawn as a flat line
                # at zero; it is named in the caption instead.
                flat.append(str(col))
                self._series.append((str(col), colour.name(), int(counts.get(col, 0))))
                continue
            curve = pg.PlotDataItem(x=T, y=y, pen=pg.mkPen(colour, width=_WAVE_WIDTH))
            pi.addItem(curve)
            n = int(counts.get(col, 0))
            self._legend.addItem(curve, f"subpop {cid}  (n = {n})")
            self._series.append((str(col), colour.name(), n))
            drawn += 1

        if not drawn:
            self._set_empty("Every subpopulation's window fell off the trace; nothing to draw.")
            return
        self._legend_n = drawn
        _fit_legend(self._legend, self._legend_n, _legend_room(self._plot))
        bits = [
            f"{drawn} median waveform(s), duration-normalised on a common axis "
            f"({len(wf)} points, T from {T[0]:.2f} to {T[-1]:.2f})",
            "MEDIAN, on the RAW trace — not the detrended signal the features were "
            "computed on, so this is the current the recording actually carried",
            "normalised in time, so classes whose dwells differ by a factor of "
            "eighteen still overlay BY SHAPE — the x axis is not seconds",
        ]
        if flat:
            bits.append(f"not drawable (no event could be averaged): {', '.join(flat)}")
        self._caption.setText(" · ".join(bits))
        self._message = ""

    def _set_empty(self, why: str) -> None:
        self._message = why
        self._caption.setText(why)


def _subpop_id(col: str) -> int:
    """``"subpop_3"`` -> ``3``; anything unparseable -> ``-1`` (grey, unnamed)."""
    tail = col.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


class SubpopReportWindow(QtWidgets.QWidget):
    """A scrollable, exportable rendering of one :class:`SubpopReport`, with figures.

    Signals:
        exportRequested: The Export button. ``main_window`` runs the save dialog
            and writes the files; this window never touches the filesystem.

    Public API (what ``main_window`` and the tests use)::

        set_report(report, subtitle="")   # install (or clear, with None)
        report()                          # the installed report, or None
        text()                            # exactly what is on screen
        header()                          # the one-line summary above the text
        can_export()                      # whether Export is live
        embedding_view()                  # the t-SNE/UMAP scatter (items 6-7)
        waveform_view()                   # the median-waveform overlay (item 13)
        caveats()                         # the strip beside the figures
        can_show_events()                 # whether the two entry-point buttons are live
        set_assignment_state(state, reason="")   # what the owner says the map is doing
        assignment_state()                # (state, reason) as last pushed
        set_curation_note(text)           # "accepted now: … — re-run to update", or ""
        curation_note()                   # that sentence, "" when the set is the run's
        live_note()                       # the standing line under the document, as DRAWN
        live_note_full()                  # ...and its sentences verbatim, elided or not
    """

    exportRequested = QtCore.Signal()
    #: "Show me these events." Both are ANNOUNCEMENTS, like Export: this window does not
    #: know what a gallery or a trace pane is, and ``main_window`` is the sole wirer.
    galleryRequested = QtCore.Signal()
    traceRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle("Event subpopulations")
        # -- THE OPENING GEOMETRY, ONCE, HERE. --------------------------------------
        # It was ``resize(*_DEFAULT_SIZE)``, unclamped, and on the owner's 1512x900
        # screen that opened a 1600x980 window whose entire button row -- Copy report,
        # Export..., Event gallery..., Colour the trace, Close -- sat 72 px below the
        # bottom of the display, with Close a further 80 px past the right edge. The
        # window was resizable, so the report was readable and nothing it can DO was
        # reachable. :func:`fitted_window_size` shrinks the ask to the screen.
        #
        # AND ONLY ONCE. This window is built lazily and re-shown for every later
        # report (``main_window._open_subpop_window``), so placing it here rather than
        # in ``showEvent`` means a window the user has dragged or resized stays where
        # they put it: re-running the mixture must not make the window jump out from
        # under the pointer that just chose the menu item.
        avail = available_screen_rect(self)
        size = fitted_window_size(avail)
        self.resize(size)
        self.setMinimumSize(*_MIN_SIZE)
        # A size that fits is not a position that fits: a window system that opens a
        # child near its parent can still put the bottom-right of a screen-sized window
        # off the screen. Centred in the available rectangle, with the title bar's own
        # height kept clear of the top so the bar itself is grabbable.
        self.move(avail.x() + max(_SCREEN_MARGIN,
                                  (avail.width() - size.width()) // 2),
                  avail.y() + max(_SCREEN_MARGIN + _FRAME_ALLOWANCE[1],
                                  (avail.height() - size.height()) // 2))
        self._report = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self._header = QtWidgets.QLabel(self._empty_header())
        self._header.setWordWrap(True)
        self._header.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._header)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        # THE DOCUMENT WRAPS, AND IT IS A SOFT WRAP. `NoWrap` laid the report out at
        # its natural width and let the reader chase it sideways: measured on the
        # owner's 1512x900 display, with the window fitted to it, the report pane's
        # horizontal scrollbar ran to 2906 px inside a 624 px viewport -- 3530 px of
        # document in a pane five and a half times narrower. What is out there is
        # PROSE, not columns: the four numbered caveats are 245, 290, 326 and 488
        # characters long -- the honesty text this whole window exists to put in
        # front of a reader, its sentences ending four screens to the right.
        #
        # WHAT IT COSTS, MEASURED, because the argument for `NoWrap` was real: of the
        # 140 lines in a report, 9 wrap at the fitted width and only 3 of those are
        # columnar (the population table's header and its k rows, 160 chars against
        # the 89 the pane holds). The other six are prose. A table that wraps is the
        # price of prose that does not need a scrollbar, and it is the same trade the
        # caveat strip below already makes (:attr:`_caveats`, `WidgetWidth`).
        #
        # NOTHING ABOUT THE TEXT CHANGES -- this is a layout mode, not a re-flow of
        # the string. `toPlainText()` stays byte-identical to ``report.text()``, so
        # `Copy report`, `Export…` and every test that compares the two are untouched,
        # and no widget moves: the pane's own geometry and the button row below it are
        # identical to the pixel with the scrollbar gone.
        #
        # `WrapAtWordBoundaryOrAnywhere` AND NOT `WordWrap`: a report carries file
        # paths and feature lists, and a single token wider than the pane would put
        # the scrollbar straight back. Breaking a long token is the lesser failure.
        self._text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._text.setWordWrapMode(
            QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self._text.setFont(_mono_font())
        # A read-only document still wants a text cursor: the columnar tables are
        # meant to be selected and pasted into a notebook.
        self._text.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)

        # A SPLITTER AND NOT A TAB WIDGET. See the module docstring: the figures and
        # the reasons not to believe them have to be on screen together, and a tab
        # is a place to put the second one where it is not.
        self._split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._split.setChildrenCollapsible(False)
        self._split.addWidget(self._text)
        self._split.addWidget(self._build_figures())
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 1)
        self._split.setSizes(list(_SPLIT))
        root.addWidget(self._split, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(6)
        self._copy_btn = QtWidgets.QPushButton("Copy report")
        self._copy_btn.setToolTip(
            "Copy the whole report to the clipboard, caveats included.")
        self._copy_btn.clicked.connect(self._copy)
        self._export_btn = QtWidgets.QPushButton("Export…")
        self._export_btn.setToolTip(
            "Write the report text, a JSON sidecar of the settings and every caveat, "
            "and one CSV per table (events with their subpopulation and posterior, "
            "the population table, the PCA loadings, the cross-validation grid, the "
            "crosstab and the per-subpopulation median waveforms — including the "
            "embedding coordinates and the waveforms these figures are drawn from).")
        self._export_btn.clicked.connect(self.exportRequested)
        # -- the two ways out of the report and back to the events it is about ------------
        self._gallery_btn = QtWidgets.QPushButton("Event gallery…")
        self._gallery_btn.setToolTip(
            "Open the Event gallery on this report: every tile chipped with the "
            "subpopulation the mixture assigned it, filterable to one class, and offering a "
            "sample spaced through that class's dwell distribution rather than its first "
            "tiles. The caveats travel with it.")
        self._gallery_btn.clicked.connect(self.galleryRequested)
        self._trace_btn = QtWidgets.QPushButton("Colour the trace")
        self._trace_btn.setToolTip(
            "Shade the detected events on the main trace in these same colours, so the "
            "full recording becomes a map of where each subpopulation occurs. Zoom in and "
            "each span is named.")
        self._trace_btn.clicked.connect(self.traceRequested)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(self._copy_btn)
        buttons.addWidget(self._export_btn)
        buttons.addWidget(self._gallery_btn)
        buttons.addWidget(self._trace_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)

        # -- the standing line: what the assignment is doing, and whether the accepted
        #    set has moved under this report. Always present, fixed height (see
        #    _LIVE_NOTE_LINES); built like the window's other notes.
        self._assignment_state: str = ASSIGNMENT_NONE
        self._assignment_reason: str = ""
        self._curation_note_text: str = ""
        self._live_note_text: str = ""     # the sentences in full, whatever is drawn
        self._live_note_fitting = False    # re-entry guard: setText inside a resize
        self._live_note = QtWidgets.QLabel("")
        self._live_note.setWordWrap(True)
        self._live_note.setStyleSheet(_NOTE_CSS)
        self._live_note.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        # TOP, not the QLabel default centre. In a fixed box that is taller than most of
        # its texts, centring makes the first sentence drift down the box as the second
        # arrives, and it clips BOTH ends rather than one if a text ever did overrun
        # (the repro measured alignment 129 = AlignLeft|AlignVCenter). Anchored at the
        # top the sentence starts at the same y whether the line carries one, two or
        # three lines of it.
        self._live_note.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft
                                     | QtCore.Qt.AlignmentFlag.AlignTop)
        fm = self._live_note.fontMetrics()
        self._live_note.setFixedHeight(fm.lineSpacing() * _LIVE_NOTE_LINES + 4)
        # The fit depends on the label's WIDTH, which the layout owns, so it is redone
        # whenever that width changes (drag the window narrower and a three-line text
        # becomes four) — not only when the text changes.
        self._live_note.installEventFilter(self)
        #: A hidden twin used only to ask "how tall would THIS text be at THIS width?".
        #: Measuring on the real label would mean setting text on it once per binary
        #: search step, i.e. a repaint per step of something the reader is looking at.
        self._live_note_probe = QtWidgets.QLabel(self)
        self._live_note_probe.setWordWrap(True)
        self._live_note_probe.hide()
        root.addWidget(self._live_note)
        root.addLayout(buttons)

        self._sync_buttons()

    def _build_figures(self) -> QtWidgets.QWidget:
        """The figure column: scatter, waveforms and caveats, all three in one splitter.

        Returns the splitter itself. It used to be wrapped in a QWidget + QVBoxLayout
        that held the splitter and, beneath it, the caveat box; with the box moved
        into the splitter that wrapper had one child and no purpose.
        """
        self._embed_view = EmbeddingScatterView()
        self._wave_view = WaveformOverlayView()
        figures = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        figures.setChildrenCollapsible(False)
        figures.addWidget(self._embed_view)
        figures.addWidget(self._wave_view)

        # THE CAVEAT STRIP IS NOT COLLAPSIBLE AND HAS NO TAB. It is the second half
        # of every figure above it.
        self._caveat_box = QtWidgets.QGroupBox("Caveats")
        box = QtWidgets.QVBoxLayout(self._caveat_box)
        box.setContentsMargins(6, 6, 6, 6)
        self._caveats = QtWidgets.QPlainTextEdit()
        self._caveats.setReadOnly(True)
        self._caveats.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._caveats.setMinimumHeight(_MIN_CAVEAT_H)
        box.addWidget(self._caveats)

        # IT IS A PANE OF THE SAME SPLITTER AS THE FIGURES, and that is the whole
        # design in one line. As a fixed-height widget below the splitter it took
        # the SAME ~226 px at every window size, so shrinking the window came out of
        # the pictures alone and the scatter reached a twelve-pixel data rectangle
        # while the caveats sat unchanged. Sharing the splitter makes the three
        # panes trade proportionally, and `setChildrenCollapsible(False)` plus the
        # minimum height above mean the strip can be made bigger, made smaller, and
        # never dragged to nothing. There is still no tab and no disclosure arrow:
        # the reader cannot put the warnings away, only resize them.
        figures.addWidget(self._caveat_box)
        figures.setStretchFactor(0, 3)   # the embedding grows first
        figures.setStretchFactor(1, 2)
        figures.setStretchFactor(2, 1)
        figures.setSizes([400, 320, 190])
        self._figures = figures
        return figures

    # -- state ---------------------------------------------------------------
    def set_report(self, report, subtitle: str = "") -> None:
        """Install a report (or ``None`` to clear). Idempotent; re-entrant-safe."""
        self._report = report
        if report is None:
            self._header.setText(self._empty_header())
            self._text.setPlainText("")
        else:
            self._header.setText(self._describe(report, subtitle))
            self._text.setPlainText(report.text())
            # Top of the document, not wherever the previous report was scrolled
            # to — a second run silently landing mid-table reads as the first one.
            self._text.moveCursor(QtGui.QTextCursor.MoveOperation.Start)
        self._embed_view.set_report(report)
        self._wave_view.set_report(report)
        self._caveats.setPlainText(self._caveat_text(report))
        self._caveat_box.setTitle(self._caveat_title(report))
        # A new document makes no claim about any assignment: the owner pushes the
        # truth about THIS report's map right after installing it. Until then the
        # entry buttons are dead, never live over a colouring that may not exist.
        self._assignment_state = ASSIGNMENT_NONE
        self._assignment_reason = ""
        self._curation_note_text = ""
        self._render_live_note()
        self._sync_buttons()

    def report(self):
        """The installed :class:`SubpopReport`, or ``None``."""
        return self._report

    def text(self) -> str:
        """Exactly what is rendered — the tests read this, not the report object."""
        return self._text.toPlainText()

    def header(self) -> str:
        """The one-line summary above the document."""
        return self._header.text()

    def can_export(self) -> bool:
        """Whether Export is live. False with nothing installed."""
        return self._export_btn.isEnabled()

    def embedding_view(self) -> EmbeddingScatterView:
        """Items 6-7's scatter."""
        return self._embed_view

    def waveform_view(self) -> WaveformOverlayView:
        """Item 13's median-waveform overlay."""
        return self._wave_view

    def caveats(self) -> str:
        """The caveat strip that sits under the figures, verbatim."""
        return self._caveats.toPlainText()

    def splitter(self) -> QtWidgets.QSplitter:
        """The text/figures splitter, so a caller can size it."""
        return self._split

    def figures_splitter(self) -> QtWidgets.QSplitter:
        """The vertical splitter holding scatter, waveforms AND the caveat strip.

        The strip is a pane of it, not a widget below it — see :meth:`_build_figures`.
        """
        return self._figures

    # -- rendering -----------------------------------------------------------
    @staticmethod
    def _empty_header() -> str:
        return "No subpopulation report yet — run Analysis ▸ Event subpopulations…"

    @staticmethod
    def _caveat_title(report) -> str:
        n = 0 if report is None else len(report.warnings)
        return f"Caveats — {n} from the run, and one about these figures"

    @staticmethod
    def _caveat_text(report) -> str:
        """Every warning verbatim, plus the one that is about the picture.

        VERBATIM AND NUMBERED, not summarised: a caveat paraphrased into a
        reassuring sentence is worse than no caveat. This is the same text the
        report document carries at its bottom and the export writes to the
        sidecar; it is repeated here because the reader looking at the figures is
        not, at that moment, looking at the bottom of the document.
        """
        if report is None:
            return ""
        lines = [f"1. {_COLOUR_CAVEAT}", ""]
        for i, w in enumerate(report.warnings, start=2):
            lines.append(f"{i}. {w}")
            lines.append("")
        if not report.warnings:
            lines.append("The run itself raised none.")
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _describe(report, subtitle: str) -> str:
        """The header: what was analysed, what came out, and how well it holds up.

        THIS WINDOW NO LONGER WRITES THE CLAIM. It used to compose its own
        sentence out of ``quantification.fit.k`` and ``conf_frac``, which is how
        a header came to read "7 subpopulation(s) at n_PCs=11 (84.4 %
        confidently assigned)" over a partition whose classes score a 0.098
        silhouette and on which the independent density clusterer finds no
        cluster at all. Both of those facts were computed, and neither was in
        either of the two strings a reader carries away.

        So the sentence is ``report.headline()``'s, written once in the analysis
        layer beside the statistics it has to be honest about, and this method
        only decides where it goes. That is the same rule this window already
        follows for the caveats: repeated VERBATIM, never paraphrased.

        THE CAVEAT COUNT IS STILL ON THE HEADER, and still for its own reason:
        every one of them is a reason not to quote a number below at face value,
        and they sit at the bottom of a long document.
        """
        head = report.headline()
        n_warn = len(report.warnings)
        if n_warn:
            head += " — beside the figures and at the bottom of the text"
        return f"{subtitle}\n{head}" if subtitle else head

    def can_show_events(self) -> bool:
        """Whether the two "show me these events" buttons are live."""
        return self._gallery_btn.isEnabled()

    # -- the live state: what the assignment is doing, and what the accepted set did ----
    def set_assignment_state(self, state: str, reason: str = "") -> None:
        """What the report's per-event assignment is doing right now — one of
        :data:`ASSIGNMENT_STATES` — and, for every state but ``installed``, WHY in one
        sentence. The owner (``main_window``) is the only caller; it resolves the state
        from the session and this window repeats it: the two entry buttons are live
        exactly when the map is on the trace and in the gallery, and the standing line
        under the document carries the reason whenever it is not (S7-POP3)."""
        if state not in ASSIGNMENT_STATES:
            raise ValueError(f"unknown assignment state {state!r}; one of {ASSIGNMENT_STATES}")
        self._assignment_state = state
        self._assignment_reason = str(reason)
        self._render_live_note()
        self._sync_buttons()

    def assignment_state(self) -> tuple[str, str]:
        """``(state, reason)`` as last pushed by the owner."""
        return self._assignment_state, self._assignment_reason

    def set_curation_note(self, text: str) -> None:
        """S7-POP4: the sentence saying the accepted set has moved under this report
        ("accepted now: 170 of 236; this run's fractions describe the 236 accepted at run
        time — re-run to update"), or "" to clear it. Written by the owner on every
        curation, cleared by a re-run (``set_report``). The report's own numbers are NOT
        touched: they are the run's, and this line says so."""
        self._curation_note_text = str(text)
        self._render_live_note()

    def curation_note(self) -> str:
        """The standing curation sentence, "" when the accepted set is the run's."""
        return self._curation_note_text

    def live_note(self) -> str:
        """Exactly what the standing line under the document DRAWS — the whole of both
        sentences at every size this window has been measured at, and a text elided at a
        word with "…" in the rare case one runs past the three-line budget (see
        :data:`_LIVE_NOTE_LINES`). :meth:`live_note_full` is the verbatim text."""
        return self._live_note.text()

    def live_note_full(self) -> str:
        """The standing line's sentences VERBATIM, whether or not the label had room to
        draw all of them. This is what the tooltip carries when they are elided."""
        return self._live_note_text

    def _render_live_note(self) -> None:
        lines = []
        if self._assignment_state != ASSIGNMENT_INSTALLED and self._assignment_reason:
            lines.append(self._assignment_reason)
        if self._curation_note_text:
            lines.append(self._curation_note_text)
        self._live_note_text = "\n".join(lines)
        self._fit_live_note()

    # -- one place decides what the fixed-height line DRAWS ----------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt)
        """Re-fit the standing line when its width changes. The label is word-wrapped in
        a box whose height is fixed, so how many lines a sentence needs is a function of
        the width the layout hands it — dragging the window narrower is exactly as able
        to overrun the budget as a longer file name is."""
        if obj is self._live_note and event.type() in (
                QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show,
                QtCore.QEvent.Type.FontChange):
            self._fit_live_note()
        return super().eventFilter(obj, event)

    def _live_note_fits(self, text: str, width: int, budget: int) -> bool:
        """Would ``text`` wrap into ``budget`` px at ``width`` px in the standing line?
        Asked of a hidden twin carrying the real label's font and box metrics, because
        ``QLabel.heightForWidth`` can only answer for the text it is currently holding."""
        probe = self._live_note_probe
        probe.setFont(self._live_note.font())
        probe.setContentsMargins(self._live_note.contentsMargins())
        probe.setIndent(self._live_note.indent())
        probe.setMargin(self._live_note.margin())
        probe.setText(text)
        return probe.heightForWidth(width) <= budget

    def _fit_live_note(self) -> None:
        """Put the sentences on the label, elided at a word if they do not fit.

        THE FIXED HEIGHT IS THE ONE THING THAT NEVER MOVES (see :data:`_LIVE_NOTE_LINES`),
        so when a text is too long for it the text gives, never the box: the longest
        prefix ending at a word boundary that still fits, plus "…", with the whole of it
        in the tooltip and in :meth:`live_note_full`. Nothing is ever drawn as a row of
        half-glyphs, which is what the reader was given before (fix-up C3).

        A label that is not on screen draws nothing, so there is nothing to elide and the
        text stands in full; the fit is redone the moment it is shown or resized
        (:meth:`eventFilter`), which is also why the guard below exists — ``setText``
        inside a resize can invalidate the layout and come straight back here.
        """
        if self._live_note_fitting:
            return
        self._live_note_fitting = True
        try:
            lab = self._live_note
            full = self._live_note_text
            if not full:
                lab.setText("")
                lab.setToolTip("")
                return
            width, budget = lab.width(), lab.height()
            if not lab.isVisible() or width <= 0 or self._live_note_fits(full, width, budget):
                lab.setText(full)
                lab.setToolTip("")
                return
            lab.setText(self._elided_live_note(full, width, budget))
            lab.setToolTip(full)
        finally:
            self._live_note_fitting = False

    def _elided_live_note(self, full: str, width: int, budget: int) -> str:
        """The longest prefix of ``full`` that ends at a word boundary and still fits in
        ``budget`` px at ``width`` px, plus "…". Cutting mid-word would invent a word;
        cutting at whitespace (the newline between the two sentences included) never
        does. A text with no whitespace at all cannot be cut that way, so it falls back
        to Qt's own single-line elision, which fits by construction."""
        cuts = [i for i, ch in enumerate(full) if ch.isspace()]
        if not cuts:
            return self._live_note.fontMetrics().elidedText(
                full, QtCore.Qt.TextElideMode.ElideRight, width)
        # fits() is monotone in the prefix length, so bisect it.
        lo, hi, best = 0, len(cuts) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._live_note_fits(full[:cuts[mid]] + "…", width, budget):
                best, lo = cuts[mid], mid + 1
            else:
                hi = mid - 1
        if best is None:      # not even the first word fits: draw that word alone
            best = cuts[0]    # (one word + "…" is one line; the budget is three)
        return full[:best] + "…"

    def _sync_buttons(self) -> None:
        live = self._report is not None
        self._export_btn.setEnabled(live)
        self._copy_btn.setEnabled(live)
        # The two entry points need MORE than a report: they act on the ASSIGNMENT the
        # owner read off it and installed on the trace and the gallery, and that map has
        # a life of its own -- refused at adoption, held while another recording is on
        # screen, retired when the detection it describes is replaced. So they are gated
        # on the owner's word about the map (:meth:`set_assignment_state`), never on
        # ``quantification is not None``: that is a property of the stored document, and
        # a button gated on it stayed live after a merge had retired the map and then
        # opened a plain gallery, or reported "fitted no mixture" under a header saying
        # otherwise (S7-POP3). A report with no mixture never reaches ``installed``, so
        # the old "NO MIXTURE" case is still dead, for the same reason as before.
        has_classes = (live and self._report.quantification is not None
                       and self._assignment_state == ASSIGNMENT_INSTALLED)
        self._gallery_btn.setEnabled(has_classes)
        self._trace_btn.setEnabled(has_classes)

    def _copy(self) -> None:
        clip = QtWidgets.QApplication.clipboard()
        if clip is not None:
            clip.setText(self.text())
