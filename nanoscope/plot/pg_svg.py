"""Vector (SVG) export for the live pyqtgraph views -- and the workaround that makes it possible.

THIS FILE CARRIES A PATCH TO SOMEBODY ELSE'S CODE. It is written against
**pyqtgraph 0.14.0 / PySide6 6.11.1 / Qt 6.11.1** and it exists only because
``pyqtgraph.exporters.SVGExporter`` cannot export *any* plot on this stack. When
pyqtgraph fixes the bug the patch takes itself out of the way (see `install`);
nothing in the closepath patch should outlive the upstream defect.

It also carries one thing that is NOT a workaround and stays: the font stack written into
every ``<text>`` of an SVG (see "THE FONT STACK" below and `install_font_stack`).

WHY (the bug, measured this session)
------------------------------------
``SVGExporter(plot_item).export(path)`` raises::

    ValueError: not enough values to unpack (expected 2, got 1)
    .../pyqtgraph/exporters/SVGExporter.py:427 in correctCoordinates -> x,y = c.split(',')

on ALL TEN cases probed -- plain line, filled step histogram, scatter, isocurve,
legend, TextItem, LinearRegionItem, InfiniteLine, log axis, **and a completely
empty PlotItem**. It is not an item-type limitation; it is every plot, because the
offending path is the ViewBox border that every plot has::

    d='M0,0 L603.5,0 L603.5,458.5 L0,458.5 L0,0 Z'

``correctCoordinates`` flattens each ``<g transform="matrix(...)">`` by applying the
matrix to the coordinates inside, and its token loop assumes **every** whitespace
token of a path's ``d`` is an ``x,y`` pair. The SVG closepath command ``Z`` has no
comma. Qt6's ``QSvgGenerator`` emits ``Z``; pyqtgraph's parser predates it. The app's
PNG-only file filter is the only reason nobody had hit this yet.

HOW IT IS FIXED, AND WHY NOT BY FORKING THE FUNCTION
----------------------------------------------------
The rest of ``correctCoordinates`` is correct -- the gradient rewrite, the text/non-text
group splitting, the stroke-width rescaling -- and it is ~120 lines. Copying those 120
lines here would mean that the next pyqtgraph release could improve any of them and we
would silently keep running the 0.14.0 text. So the patch is a **wrapper** that leaves
upstream owning all of the logic and only takes the comma-less tokens out of its way:

  * before upstream runs, every ``<path>`` under the node has its comma-less tokens
    lifted out of ``d``, with their token indices remembered;
  * upstream transforms the remaining coordinate tokens -- it emits exactly one output
    token per input token, space separated, in order, which is what makes step 3 exact;
  * afterwards each lifted token is re-inserted **verbatim at its original index**.

That is the same output as fixing the token loop in place to "pass comma-less tokens
through verbatim" -- ``tests/plot/test_pg_svg.py`` asserts the two agree character for
character on the real ViewBox-border path. The second half of the upstream fix (guard
``newCoords[0] != 'M'`` against an empty string) falls out for free: a path made only of
comma-less tokens is handed to upstream as ``d=""``, which its own ``if oldCoords == ''``
guard skips before ``newCoords[0]`` is ever evaluated, and the original ``d`` is put back.

Verified after patching: all eight probe plots export, and every file parses as XML.

NOTE ON THE NAME. ``pyqtgraph.exporters.SVGExporter`` is the *class*, re-exported by
``pyqtgraph/exporters/__init__.py`` over the module of the same name. The module has to
be reached through ``importlib`` / ``sys.modules``, and the module global is what the
``correctCoordinates(g1, defs, item, options)`` call site at SVGExporter.py:288 resolves
against -- so rebinding it there is what actually takes effect.

WHERE THE IMPORT GOES -- DELIBERATELY *NOT* ``nanoscope/plot/__init__.py``
---------------------------------------------------------------------------
The patch installs on ``import nanoscope.plot.pg_svg``, and that import is left to the
caller. ``nanoscope.plot`` is the matplotlib half of the app and today imports no
pyqtgraph at all (verified: ``import nanoscope.plot`` leaves ``sys.modules`` free of it);
its own docstring says so. Hanging this module off the package ``__init__`` would pull
pyqtgraph -- and therefore Qt -- into every notebook and headless caller that only wanted
``figures``/``style``. The GUI, which has Qt loaded anyway, imports this module instead;
one such import anywhere fixes the exporter for the whole process, including pyqtgraph's
own right-click "Export... > SVG" dialog.

TWO KNOWN LIMITATIONS OF PYQTGRAPH'S SVG EXPORT
-----------------------------------------------
(a) GEOMETRY IS STALE UNLESS SOMETHING PAINTED FIRST -- fixed here, see `export_svg`.
    ``ImageExporter`` renders through ``GraphicsScene.render()``, whose first line is
    ``self.prepareForPaint()``; that is what runs each ViewBox's deferred
    ``updateAutoRange()`` + ``updateMatrix()``. ``SVGExporter`` calls ``item.paint()``
    directly (SVGExporter.py:267) and is the one render path in pyqtgraph that never
    reaches ``prepareForPaint``. On a plot that has never been shown the ViewBox's
    item->scene matrix is therefore still the one for the default [0,1]x[0,1] range,
    while ``viewRange()`` may already report something else. Measured on a KDE overlay
    (256-bin grid, IsocurveItem in grid-index coordinates, 640x480 canvas):

        vb.updateAutoRange() only ............ max |coord| = 113675.6 px   (~1e5, 5 orders
                                                                            off-canvas)
        + vb.updateMatrix() .................. max |coord| =    639.5 px
        ground truth: show() + real paint .... max |coord| =    639.5 px, byte-identical

    So it is not "never shown" per se, it is "no paint has reconciled the matrix with the
    range". `export_svg` calls ``scene.prepareForPaint()`` -- pyqtgraph's own pre-paint
    pass, the same one a real repaint would run -- so the exporter sees what the screen
    would. This never changes a range the user chose; it only applies work already queued.

(b) ONE ``<path>`` PER SCATTER POINT -- reported, never silently degraded. The exporter
    emits a separate ``<g>``/``<path>`` pair for every symbol, so a dwell-vs-blockade
    scatter grows linearly and fast. Measured (600x450 canvas, size-4 symbols):

        n=   200 ->   0.12 MB,    202 paths,  0.04 s
        n=  1000 ->   0.55 MB,   1002 paths,  0.17 s
        n=  5000 ->   2.66 MB,   5002 paths,  1.12 s
        n= 20000 ->  10.59 MB,  20002 paths,  9.40 s
        n= 50000 ->  26.41 MB,  50002 paths, 52.15 s

    i.e. ~528 bytes and ~1 ms per point, with the time superlinear beyond ~20k. A real
    event census of 50k events is a 26 MB SVG that Illustrator and Inkscape will both
    struggle to open. `svg_size_warning` turns a plot into the sentence a panel should
    show BEFORE it exports, so the user can choose PNG instead. Nothing here subsamples
    the data or swaps the format on its own -- a figure that silently lost points would
    be worse than a big file.
"""

from __future__ import annotations

import html
import importlib
import re
import sys
import xml.dom.minidom as _xml
from pathlib import Path
from typing import Any
from xml.sax import saxutils

from . import figure_fonts

__all__ = [
    "BYTES_PER_VECTOR_POINT",
    "FONT_PATCH_STATUS",
    "PATCH_STATUS",
    "SVG_POINT_WARN_THRESHOLD",
    "count_vector_points",
    "estimate_svg_bytes",
    "estimate_svg_bytes_for_points",
    "export_svg",
    "install",
    "install_font_stack",
    "refresh_for_export",
    "set_font_stack",
    "svg_size_warning",
]

# The module, not the class of the same name that shadows it on the package. See the
# NOTE ON THE NAME above.
_SVG_MODULE_NAME = "pyqtgraph.exporters.SVGExporter"

# Stamped on the replacement so a second import, a reload, or a second explicit
# ``install()`` cannot wrap the wrapper.
_PATCH_MARK = "_nanoscope_closepath_fix"

# The path that breaks pyqtgraph 0.14.0: the ViewBox border, whose trailing ``Z`` has no
# comma. Used both as the version probe below and as the regression fixture in the tests.
VIEWBOX_BORDER_PATH = "M0,0 L603.5,0 L603.5,458.5 L0,458.5 L0,0 Z"


# ==================================================================================================
# The patch
# ==================================================================================================

def _lift_commaless(node: Any) -> dict:
    """Take the comma-less tokens out of every ``<path d=...>`` beneath *node*.

    Returns the plan `_restore_commaless` needs: for each touched element, the
    ``(index, token)`` pairs that were lifted, how many coordinate tokens were left
    behind, and the original ``d`` in case upstream declines the path entirely.
    """
    plan: dict[Any, tuple[list[tuple[int, str]], int, str]] = {}
    # Collected BEFORE upstream runs. It re-parents these elements between groups but
    # never clones them, so the identities in this dict stay valid across the call.
    for ch in node.getElementsByTagName("path"):
        d = ch.getAttribute("d").strip()
        if not d:
            continue
        tokens = d.split(" ")
        lifted = [(i, t) for i, t in enumerate(tokens) if "," not in t]
        if not lifted:
            continue                      # nothing upstream would choke on
        kept = [t for t in tokens if "," in t]
        plan[ch] = (lifted, len(kept), d)
        # An all-comma-less path becomes ``d=""``; upstream's own ``if oldCoords == ''``
        # skips it, which is how we keep it away from ``newCoords[0]`` on an empty string.
        ch.setAttribute("d", " ".join(kept))
    return plan


def _restore_commaless(plan: dict) -> None:
    """Put each lifted token back verbatim, at the index it came from."""
    for ch, (lifted, n_kept, original) in plan.items():
        d = ch.getAttribute("d").strip()
        if not d:
            # Upstream skipped this path (it had no coordinate tokens at all), so there is
            # nothing transformed to merge with -- restore what Qt wrote. Such a path draws
            # nothing; it loses its group transform, which is why that is harmless.
            ch.setAttribute("d", original)
            continue
        out = d.split()
        if len(out) != n_kept:
            # Upstream stopped emitting one token per input token. Our index mapping is no
            # longer meaningful, so leave its transformed output alone rather than corrupt
            # it: the path still renders, just without its closepath. Loud enough to find --
            # `install`'s probe would normally have caught a change this large first.
            continue
        for i, token in lifted:           # ascending, so each insert lands correctly
            out.insert(i, token)
        ch.setAttribute("d", " ".join(out) + " ")


def _make_patched(upstream):
    """Wrap *upstream* ``correctCoordinates`` so comma-less path tokens survive it."""

    def correctCoordinates(node, defs, item, options):  # noqa: N802 (pyqtgraph's spelling)
        plan = _lift_commaless(node)
        try:
            return upstream(node, defs, item, options)
        finally:
            # ``finally`` so a failure deeper in upstream still leaves the document with
            # paths that at least draw, instead of half-stripped ones.
            _restore_commaless(plan)

    setattr(correctCoordinates, _PATCH_MARK, True)
    correctCoordinates.__wrapped__ = upstream
    correctCoordinates.__doc__ = (
        "nanoscope wrapper around pyqtgraph's correctCoordinates: preserves SVG "
        "commands that carry no coordinates (notably closepath 'Z'), which "
        "pyqtgraph 0.14.0's token loop cannot parse. See nanoscope/plot/pg_svg.py."
    )
    return correctCoordinates


def _is_broken(fn) -> bool | None:
    """Does *fn* still choke on a comma-less path token?

    THIS IS THE VERSION CHECK, and it is deliberately behavioural rather than a
    ``pyqtgraph.__version__`` comparison: the fix could arrive in any release, or in a
    distribution's backport, and a version table would then keep patching a function that
    is already correct. The probe runs the real ``correctCoordinates`` over a two-element
    DOM holding the exact ViewBox-border path that fails, with ``defs=[]`` (so the
    gradient branch, the only user of *item*, never runs) and ``options={}``. No Qt
    widget, no QApplication, no scene -- it is safe at import time.

    Returns True (still broken), False (upstream handles it -> do not patch), or None
    (it failed in a way this file does not recognise -> do not patch, and say so).
    """
    doc = _xml.parseString(
        '<g id="probe"><g transform="matrix(1,0,0,1,0,0)">'
        f'<path d="{VIEWBOX_BORDER_PATH}"/></g></g>'
    )
    try:
        fn(doc.documentElement, [], None, {})
    except ValueError:
        return True                        # the known defect
    except Exception:                      # noqa: BLE001 - anything else is unknown territory
        return None
    return False


def install() -> str:
    """Install the closepath workaround if -- and only if -- it is needed. Idempotent.

    Returns one of:
      ``"patched"``        the wrapper was installed just now;
      ``"already-patched"`` a wrapper from this module was already in place (a second
                           import, a reload, or a second explicit call) -- nothing done,
                           so the wrapper can never wrap itself;
      ``"upstream-fixed"``  this pyqtgraph parses comma-less tokens on its own; the patch
                           stands aside entirely;
      ``"unknown-upstream"`` the probe failed in an unrecognised way; nothing patched,
                           because guessing at somebody else's rewrite is worse than an
                           SVG export that fails visibly.
    """
    module = sys.modules.get(_SVG_MODULE_NAME) or importlib.import_module(_SVG_MODULE_NAME)
    current = module.correctCoordinates

    if getattr(current, _PATCH_MARK, False):
        return "already-patched"

    broken = _is_broken(current)
    if broken is False:
        return "upstream-fixed"
    if broken is None:
        return "unknown-upstream"

    module.correctCoordinates = _make_patched(current)
    return "patched"


#: Result of the import-time `install()` call -- readable by a panel that wants to say
#: why SVG export is unavailable, and asserted on by the tests.
PATCH_STATUS: str = install()


# ==================================================================================================
# The font stack
# ==================================================================================================
#
# THE FONT STACK. Qt's SVG writer puts ONE ``font-family`` on each ``<text>`` -- the name the
# QFont was given, even for a font that is not installed, and with no generic behind it (pyqtgraph
# adds ``sans-serif`` only for a font whose style hint says so, which a family set by name never
# does) -- and embeds nothing. A viewer without that one family silently picks its own. And text
# that was never given a font (a plot's grey no-data message) is written with the APPLICATION's
# face, which on macOS is the private ``.AppleSystemUIFont`` that no other machine has. (A figure
# drawn by the windows is in Arial unless a family was picked: ``gui.font_picker``.)
#
# So every ``<text>`` is rewritten, as the SVG is produced, to the family's stack from
# :mod:`nanoscope.plot.figure_fonts` (``Arial, Helvetica, 'Liberation Sans', Arimo, ...,
# sans-serif``): the family Qt named first, then its metric-compatible stand-ins, then the
# generic; a family nobody can name (empty, or a private ``.``-prefixed one) becomes Arial. A
# family someone picked is named whatever the application's own font is (Segoe UI on Windows
# can be picked in Other font…), so the SVG says what the screen, the PNG and the PDF show. The
# text stays text -- nothing is outlined.
#
# THE GROUPS' ``font-family`` GOES. Qt also writes the painter's face on every enclosing ``<g>``
# (410 of 418 groups in a 400-point scatter, text or no text). It governs nothing -- the text
# under a group carries its own -- and for an untouched figure it is the private application
# font again, so it is dropped rather than rewritten (which would add ~70 bytes to every group
# of a file that already holds one per scatter point).
#
# WHERE IT HOOKS IN. At pyqtgraph's ``generateSvg``, which every ``SVGExporter`` export ends in:
# ``export_svg`` below, the frozen snapshot the Figure Library / bundle / Export dialog take
# (``gui/figure_bundle.py`` calls ``SVGExporter(...).export(toBytes=True)`` itself), and
# pyqtgraph's own right-click Export all pass through it, so no caller has to remember to ask.

#: Stamped on the replacement so a reload or a second explicit call cannot wrap the wrapper.
_FONT_MARK = "_nanoscope_font_stack"

#: A ``<text>``, or the ``<tspan>`` inside one (group 1 says which): the two elements that can
#: carry a face of their own.
_TEXT_TAG = re.compile(r"<(text|tspan)\b[^>]*>")
_FONT_FAMILY_ATTR = re.compile(r'(\sfont-family=")([^"]*)(")')
#: A ``font-family`` on any element but ``<text>`` and ``<tspan>``: group 1 is everything before it
#: in the tag.
_OTHER_FONT_FAMILY = re.compile(r'(<(?!(?:text|tspan)\b)[A-Za-z][^<>]*?)\sfont-family="[^"]*"')
#: Characters a family name cannot carry into a CSS ``font-family`` without escaping.
_UNWRITABLE = re.compile(r"""['"\\;,<>{}]""")


def _first_family(value: str) -> str:
    """The family a ``font-family`` attribute value starts with: ``"Times New Roman", serif``
    (as pyqtgraph writes it) -> ``Times New Roman``."""
    text = html.unescape(value).strip()
    if text[:1] in ("'", '"'):
        end = text.find(text[0], 1)
        return text[1:end] if end > 0 else text[1:]
    return text.split(",", 1)[0].strip()


def _stack_for(family: str) -> str:
    """The CSS ``font-family`` value for text Qt wrote in ``family``.

    No family (empty), a private one (leading ``.``: on macOS the application's own face) or
    one that cannot be written safely all mean "the figure default", Arial: the SVG never names
    a font that only this computer has. Any other family is named first, the application's own
    included -- a person can pick it, and then the SVG must agree with the screen.
    """
    unset = not family or family.startswith(".") or _UNWRITABLE.search(family) is not None
    return figure_fonts.css_font_family(None if unset else family)


def set_font_stack(svg: str) -> str:
    """``svg`` with every ``<text>`` element's ``font-family`` replaced by the family's stack, and
    the ``font-family`` Qt also writes on the enclosing groups dropped.

    A ``<text>`` with no ``font-family`` gets the default (Arial) stack. A ``<tspan>`` that names a
    family of its own gets that family's stack too (it is never dropped, which would hand it the
    ``<text>``'s face); one that names none keeps inheriting. Nothing else changes: not
    the text, its position, size, weight or colour, and not the number of elements. Applying it to
    its own output changes nothing.
    """
    stacks: dict[str, str] = {}

    def restack(match: "re.Match[str]") -> str:
        tag = match.group(0)
        found = _FONT_FAMILY_ATTR.search(tag)
        if found is None and match.group(1) == "tspan":
            return tag      # no face of its own: it takes the stack of the <text> around it
        family = _first_family(found.group(2)) if found else ""
        if family not in stacks:
            stacks[family] = saxutils.escape(_stack_for(family), {'"': "&quot;"})
        attribute = f' font-family="{stacks[family]}"'
        if found:
            return tag[:found.start()] + attribute + tag[found.end():]
        cut = -2 if tag.endswith("/>") else -1
        return tag[:cut] + attribute + tag[cut:]

    return _OTHER_FONT_FAMILY.sub(r"\1", _TEXT_TAG.sub(restack, svg))


def _make_font_stacked(upstream):
    """Wrap *upstream* ``generateSvg`` so the SVG it returns carries the font stack."""

    def generateSvg(item, options=None):  # noqa: N802 (pyqtgraph's spelling)
        return set_font_stack(upstream(item, options))

    setattr(generateSvg, _FONT_MARK, True)
    generateSvg.__wrapped__ = upstream
    generateSvg.__doc__ = (
        "pyqtgraph's generateSvg, then nanoscope's font stack on every <text>. "
        "See nanoscope/plot/pg_svg.py."
    )
    return generateSvg


def install_font_stack() -> str:
    """Put the font stack on every SVG pyqtgraph writes in this process. Idempotent.

    Returns ``"patched"`` (installed just now), ``"already-patched"`` (a wrapper from this module
    is in place -- a second import, a reload or a second call) or ``"unavailable"`` (this
    pyqtgraph has no module-level ``generateSvg`` to wrap: nothing is done, and an SVG then keeps
    Qt's one family per element).
    """
    module = sys.modules.get(_SVG_MODULE_NAME) or importlib.import_module(_SVG_MODULE_NAME)
    current = getattr(module, "generateSvg", None)
    if current is None:
        return "unavailable"
    if getattr(current, _FONT_MARK, False):
        return "already-patched"
    module.generateSvg = _make_font_stacked(current)
    return "patched"


#: Result of the import-time `install_font_stack()` call -- asserted on by the tests.
FONT_PATCH_STATUS: str = install_font_stack()


# ==================================================================================================
# What a panel calls
# ==================================================================================================

# Marginal cost per vector point, from the measurement table in the module docstring:
# (26_413_340 - 123_444) / (50_000 - 200) = 527.9 bytes/point, on a ~17.8 kB fixed base
# (axes, grid, border, tick labels).
BYTES_PER_VECTOR_POINT = 528
_SVG_FIXED_BYTES = 17_800

#: Above this many individually-drawn points, warn the user before exporting. 5 000 points
#: is already 2.7 MB and about a second; 20 000 is 10.6 MB and nine seconds.
SVG_POINT_WARN_THRESHOLD = 5_000


def _descendants(item) -> list:
    """Every QGraphicsItem at or below *item*, itself included."""
    seen: list = []
    stack = [item]
    while stack:
        node = stack.pop()
        seen.append(node)
        children = getattr(node, "childItems", None)
        if children is not None:
            stack.extend(children())
    return seen


def count_vector_points(item) -> int:
    """How many individually-drawn symbols *item* will emit as separate ``<path>`` elements.

    Counts every ``ScatterPlotItem`` at or below *item*, which covers both a bare scatter
    and the ``.scatter`` a ``PlotDataItem`` carries when a symbol is set. Curves are not
    counted: a whole curve is one polyline, however many samples it has -- it is the
    per-point symbols that make the file big (limitation (b)).
    """
    import pyqtgraph as pg

    total = 0
    for node in _descendants(item):
        if isinstance(node, pg.ScatterPlotItem):
            data = getattr(node, "data", None)
            total += 0 if data is None else len(data)
    return total


def estimate_svg_bytes_for_points(n: int) -> int:
    """The measured size law itself: ~17.8 kB of axes plus ~528 bytes per drawn point.

    Good to a few percent over 200..50 000 points on the probe geometry; it is an order-
    of-magnitude aid for the warning text, not a promise about the byte count.
    """
    return _SVG_FIXED_BYTES + BYTES_PER_VECTOR_POINT * int(n)


def estimate_svg_bytes(item) -> int:
    """Predicted SVG size for *item*. See `estimate_svg_bytes_for_points`."""
    return estimate_svg_bytes_for_points(count_vector_points(item))


def svg_size_warning(item) -> str | None:
    """The sentence to show the user before exporting *item*, or None if it is small.

    Returned rather than raised, and the export still goes ahead if they say yes: the
    caller must not quietly subsample or fall back to raster, because a figure that lost
    points without saying so is a worse outcome than a large file.
    """
    n = count_vector_points(item)
    if n < SVG_POINT_WARN_THRESHOLD:
        return None
    mb = estimate_svg_bytes(item) / 1e6
    seconds = max(1, round(n / 1000))
    return (
        f"This plot draws {n:,} points, and SVG stores every one of them as its own "
        f"element. The file will be roughly {mb:.0f} MB and take about {seconds} s to "
        f"write; some vector editors are slow to open a file that size. Exporting as PNG "
        f"instead keeps the same picture at a fixed size. Continue with SVG?"
    )


def refresh_for_export(item) -> None:
    """Run pyqtgraph's own pre-paint pass over *item*'s scene. See limitation (a).

    ``GraphicsScene.render()`` -- the entry point ``ImageExporter`` uses -- starts with
    ``prepareForPaint()``, which is what flushes each ViewBox's deferred
    ``updateAutoRange()`` and ``updateMatrix()``. ``SVGExporter`` paints items directly
    and skips it, so on a plot no repaint has reconciled, coordinates land up to ~1e5 px
    off a 640 px canvas. Calling the same pass costs nothing on a plot that is already
    on screen (it recomputes the identical matrix) and changes no range the user set.
    """
    scene = item.scene() if hasattr(item, "scene") else None
    prepare = getattr(scene, "prepareForPaint", None)
    if prepare is not None:
        prepare()
        return
    # Plain QGraphicsScene (or none): fall back to the ViewBoxes we can reach ourselves.
    import pyqtgraph as pg

    for node in _descendants(item):
        if isinstance(node, pg.ViewBox):
            node.prepareForPaint()


def export_svg(item, path: str | Path, *, refresh: bool = True) -> Path:
    """Write *item* to *path* as SVG. This is the call a panel makes.

    Args:
        item: the ``PlotItem`` (or any exportable pyqtgraph item) to write.
        path: destination. Pass the STAGED path when going through
            ``nanoscope.plot.figures.save_plot_image``, exactly as the PNG export does.
        refresh: run `refresh_for_export` first. Leave it True unless you have a reason;
            see limitation (a) in the module docstring.

    Returns:
        ``Path(path)``.

    Raises:
        Whatever the exporter raises. Nothing is caught here: a panel needs the exception
        to tell the user the file was not written, and `save_plot_image` keeps the
        previous export intact when the render declines.
    """
    from pyqtgraph.exporters import SVGExporter

    if refresh:
        refresh_for_export(item)
    out = Path(path)
    SVGExporter(item).export(str(out))
    return out
