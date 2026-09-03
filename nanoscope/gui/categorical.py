"""Label-coloured points, ONCE. The categorical colour map and the drawing that uses it.

WHAT IS SHARED HERE, AND WHY IT IS THIS AND NOT A WIDGET
========================================================
Two windows now draw a cloud of points coloured by an integer class id:
``plots_panel`` (PCA scores and the X/Y scatter, coloured by ``populations``
cluster) and ``subpop_window`` (the t-SNE embedding, coloured by GMM
subpopulation or by HDBSCAN). A second, independently-written colour map is this
project's signature failure -- the log axis and the combo setter both came back in
the file the first fix had not reached -- so the map lives in exactly one place
and both callers import it.

What is NOT shared is the *window*. ``plots_panel``'s scatter is not a general
widget and must not be bent into one: it is welded to ``PlotSource`` overlays,
the X/Y column combos, the two log toggles, the best-fit line, the comparability
verdict and the Details pane. None of those exist for an embedding -- the axes
are t-SNE 1 and 2 (unitless, never logged, never fitted, and a regression through
them would be meaningless). Reusing it would also make the report window import a
panel, which the tree forbids. So the *kernel* is extracted and the two callers
keep their own layout:

* :func:`category_colour` -- id -> colour, the ONE mapping;
* :func:`category_brushes` -- the per-point brush list;
* :func:`category_counts` -- present ids with their n, in legend order;
* :func:`add_category_scatter` / :func:`legend_swatch` -- the pyqtgraph plumbing
  both windows would otherwise write twice.

THE PALETTE IS OKABE-ITO AND INDEX 1 IS NOT SKIPPED.
``plot.style.PALETTE["fit"]`` is Okabe-Ito index 1 (vermillion) and every window
that draws a *fitted overlay* uses it for that line. That is a ROLE reservation
and it binds a view that draws a fit. It does not reindex the categorical cycle:
class 1 has been vermillion in ``plots_panel`` since the cluster colours landed,
and shifting the cycle here to dodge a fit line that neither embedding nor
waveform view draws would move every existing cluster colour in the app and leave
two mappings behind -- which is the failure this module exists to prevent.

THE MAPPING IS BY VALUE, NOT BY ORDER. ``category_colour(2)`` is the same colour
whoever asks, in whatever view, however many classes are present and in whatever
order they were encountered. A palette walked in enumeration order would recolour
a subpopulation the moment an empty class was filtered out of one of the two
views, which is precisely how a reader concludes that two pictures disagree.

Negative ids are the "not in a class" sentinel in both callers -- ``populations``
spells it ``DROPPED_LABEL`` (non-finite features) and HDBSCAN spells it noise --
and both draw grey. They are NOT given a palette slot: a noise cloud in a
categorical colour reads as a fifth class. What the grey MEANS differs per caller,
so the *name* is deliberately not this module's business; only the ink and the
ordering are.

THIS MAP IS NOT THE RECORDING-IDENTITY MAP, AND MUST NOT BE MERGED WITH IT.
``core.layer.ColourRegistry`` (D1) answers "what colour is THIS RECORDING", keyed by
its persistent key and shared by the trace view and the Plots panel. This module
answers "what colour is CLASS 2", keyed by an integer whose meaning is local to one
clustering. They are two different axes and a recording drawn in its identity colour
alongside a cluster drawn in class 2's colour is correct even when the two inks
coincide. Wiring the registry in here would make a cluster id name a recording, which
is not a thing. What the two DO share is the palette and the by-value rule, and that
is already all they should.

ONE MEASURED CAVEAT, WHICH IS NOT THIS MODULE'S TO FIX. ``CATEGORY_COLOURS[7]`` is
Okabe-Ito's black, and the plot panes default to the app's dark background: measured
against ``plots_panel``'s dark ``_BG_DARK`` it is a 1.13:1 contrast ratio, below the
3:1 WCAG floor for a graphical object and darker than the pane, so an EIGHTH class is
drawn in ink nobody can see. The same is true of ``core.sources.SOURCE_COLOURS[7]``
for an eighth source. It is not fixed here because the fix is a decision about the
PANE (``plots_panel`` already offers a Light/Dark background toggle, under which the
same ink is perfect), not about the palette: see ``core.layer.LAYER_INKS``, which had
to solve exactly this for the trace pane and put white in slot 7. A caller that can
reach eight classes on a dark pane should say so the way :func:`colours_repeat`'s
answer is said.

Imports nothing from ``gui``: numpy, pyqtgraph, Qt and the palette. It is a leaf,
and a test pins that it stays one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

from ..plot.style import OKABE_ITO, PALETTE

__all__ = [
    "CATEGORY_COLOURS",
    "NOISE_HEX",
    "add_category_scatter",
    "category_brushes",
    "category_colour",
    "category_counts",
    "colours_repeat",
    "legend_swatch",
]

#: The categorical cycle. Okabe-Ito, separable under all three common forms of
#: colour-vision deficiency and in greyscale print.
CATEGORY_COLOURS: tuple[str, ...] = OKABE_ITO

#: Ink for a negative id -- "this point is in no class". The same grey the
#: below-detection shading uses, because both mean "present, but not counted".
NOISE_HEX: str = PALETTE["excluded"]


def category_colour(label: int) -> QtGui.QColor:
    """Colour for class ``label``; any negative id renders :data:`NOISE_HEX` grey.

    By VALUE and not by position -- see the module docstring. Ids beyond the
    palette wrap, which is a real collision at 8 classes and not one this map can
    fix. A legend does not fix it either -- two legend rows with the SAME swatch
    tell the reader the ambiguity exists but not which point is which -- so a
    caller that can produce that many classes must SAY SO in words. Ask
    :func:`colours_repeat` and put the answer where the reader is already looking.
    """
    if int(label) < 0:
        return QtGui.QColor(NOISE_HEX)
    return QtGui.QColor(CATEGORY_COLOURS[int(label) % len(CATEGORY_COLOURS)])


def colours_repeat(labels: Iterable[int]) -> bool:
    """True when two DISTINCT non-negative ids in ``labels`` share an ink.

    The cycle is 8 long. A Gaussian mixture here cannot reach 9 (the CV grid is
    K=1-8), but HDBSCAN's cluster count is data-dependent and uncapped, so a
    denser recording can hand a view 11 clusters and get three ambiguous pairs.
    The answer lives here rather than in the caller because a caller that spells
    ``len(CATEGORY_COLOURS)`` itself is a second copy of the palette's size, which
    is the class of duplication this module exists to end.
    """
    ids = {int(v) for v in labels if int(v) >= 0}
    return len({i % len(CATEGORY_COLOURS) for i in ids}) < len(ids)


def category_brushes(
    labels: Iterable[int], alpha: int | None = None
) -> list[QtGui.QBrush]:
    """One brush per point, in the order given.

    Args:
        labels: Class id per point.
        alpha: 0-255 fill alpha, or ``None`` for opaque. ``None`` is what the
            cluster-coloured scatter has always used: a translucent categorical
            point in an overlapping cloud reads as an intermediate colour, i.e.
            as a class that is not there.
    """
    out: list[QtGui.QBrush] = []
    for v in labels:
        c = category_colour(int(v))
        if alpha is not None:
            c.setAlpha(int(alpha))
        out.append(pg.mkBrush(c))
    return out


def category_counts(labels: Iterable[int]) -> list[tuple[int, int]]:
    """``[(class_id, n), …]`` for the classes actually present, in legend order.

    Ascending by id with every negative id LAST, so the legend reads as the
    classes and then the leftovers -- and so the two views of one report list
    their classes in the same order.
    """
    arr = np.asarray(list(labels), dtype=np.int64)
    ids = sorted({int(v) for v in arr.tolist()}, key=lambda c: (c < 0, c))
    return [(cid, int(np.count_nonzero(arr == cid))) for cid in ids]


def add_category_scatter(
    plot: pg.PlotItem,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    size: int = 7,
    alpha: int | None = None,
    pen: object = None,
) -> pg.ScatterPlotItem:
    """Draw ``(x, y)`` into ``plot``, one :func:`category_colour` per point.

    Returns the item so the caller can track it for its own clear/export path;
    this module never keeps a reference to a widget.

    Raises:
        ValueError: if the three arrays are not the same length. Silently
            truncating would draw a picture in which some points are simply the
            wrong colour, which is worse than not drawing it.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    ls = np.asarray(labels)
    if not (xs.size == ys.size == ls.size):
        raise ValueError(
            f"x, y and labels must be the same length; got {xs.size}, {ys.size}, {ls.size}"
        )
    item = pg.ScatterPlotItem(
        x=xs, y=ys, brush=category_brushes(ls.astype(np.int64).tolist(), alpha),
        pen=pen, size=int(size),
    )
    plot.addItem(item)
    return item


def legend_swatch(label: int, *, size: int = 7, alpha: int | None = None) -> pg.ScatterPlotItem:
    """An empty scatter item whose only job is to be a legend sample for ``label``."""
    c = category_colour(int(label))
    if alpha is not None:
        c.setAlpha(int(alpha))
    return pg.ScatterPlotItem(x=[], y=[], brush=pg.mkBrush(c), pen=None, size=int(size))
