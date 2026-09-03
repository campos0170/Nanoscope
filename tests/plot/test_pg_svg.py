"""Regression tests for :mod:`nanoscope.plot.pg_svg` -- the pyqtgraph SVG-export workaround.

Written from the DEFECT, not from the patch. Each test states the user-visible outcome:
the export succeeds, the file is real SVG, the elements the user drew are in it, and the
curve lands on the canvas rather than 1e5 px off it.

The bug (pyqtgraph 0.14.0 + Qt 6.11.1): ``correctCoordinates`` assumes every whitespace
token of a path's ``d`` is an ``x,y`` pair, so the closepath ``Z`` that Qt6's
``QSvgGenerator`` writes at the end of the ViewBox border raises
``ValueError: not enough values to unpack``. Every plot has a ViewBox border, so every
plot failed -- including an empty one.

`test_the_unpatched_function_still_fails` is the "can this test fail?" anchor: it runs the
ORIGINAL function this module wrapped, so if a pyqtgraph upgrade ever makes the defect
disappear it announces itself here rather than leaving a patch nobody can justify.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib
import re
import xml.dom.minidom as xml
import xml.etree.ElementTree as ET
from contextlib import contextmanager

import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph import functions as fn

from nanoscope.plot import pg_svg

SVG_NS = "http://www.w3.org/2000/svg"
_SVG_MODULE = importlib.import_module("pyqtgraph.exporters.SVGExporter")

# A non-identity matrix, so a test that compared transformed coordinates against the
# untransformed input would not accidentally pass.
_MATRIX = "matrix(2,0,0,3,10,20)"
_TR = np.array([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0]])


# ================================================================================================
# helpers
# ================================================================================================

def _dom(d: str, matrix: str = _MATRIX):
    """``<g><g transform=...><path d=.../></g></g>`` -- the shape correctCoordinates expects."""
    doc = xml.parseString(
        f'<g id="root"><g transform="{matrix}"><path d="{d}"/></g></g>')
    return doc.documentElement


def _path_d(node) -> str:
    return node.getElementsByTagName("path")[0].getAttribute("d")


def _reference_in_loop_fix(d: str, tr: np.ndarray) -> str:
    """What pyqtgraph's token loop WOULD emit if it were fixed in place.

    A transcription of SVGExporter.py:421-437 with the two-line fix the defect calls for:
    comma-less tokens pass through verbatim, and the trailing ``newCoords[0]`` guard is
    protected against an empty string. `pg_svg` reaches the same output by a wrapper
    instead of a fork; this is what proves the two agree.
    """
    new_coords = ""
    for c in d.strip().split(" "):
        if "," not in c:
            new_coords += c + " "
            continue
        x, y = c.split(",")
        if x[0].isalpha():
            t, x = x[0], x[1:]
        else:
            t = ""
        nc = fn.transformCoordinates(tr, np.array([[float(x), float(y)]]), transpose=True)
        new_coords += t + str(nc[0, 0]) + "," + str(nc[0, 1]) + " "
    if new_coords and new_coords[0] != "M":
        new_coords = f"M{new_coords[1:]}"
    return new_coords


@contextmanager
def _plot(width: int = 600, height: int = 450):
    """A PlotWidget that is always disposed of -- the suite may not leak Qt objects."""
    w = pg.PlotWidget()
    w.resize(width, height)
    try:
        yield w, w.getPlotItem()
    finally:
        w.close()
        w.setParent(None)


def _svg_coordinate_extremes(path) -> float:
    """Largest |coordinate| appearing in any ``d`` of the written file, in SVG px."""
    text = path.read_text()
    worst = 0.0
    for d in re.findall(r'd="([^"]+)"', text):
        for token in d.split():
            if "," not in token:
                continue
            a, b = token.split(",")
            if a and a[0].isalpha():
                a = a[1:]
            try:
                worst = max(worst, abs(float(a)), abs(float(b)))
            except ValueError:
                continue
    return worst


def _texts(root) -> list[str]:
    return ["".join(e.itertext()).strip() for e in root.iter(f"{{{SVG_NS}}}text")]


# ================================================================================================
# the patch itself
# ================================================================================================

def test_the_unpatched_function_still_fails_on_a_closepath():
    """The defect is real: pyqtgraph's own correctCoordinates cannot parse ``Z``.

    If this ever passes, pyqtgraph fixed the bug -- delete `nanoscope.plot.pg_svg`'s patch
    half rather than leaving a wrapper around working code. `install` already detects that
    case at runtime (see ``test_install_stands_aside_...``); this makes it visible.
    """
    if pg_svg.PATCH_STATUS == "upstream-fixed":
        pytest.skip("this pyqtgraph parses comma-less path tokens; the patch stood aside")
    assert pg_svg.PATCH_STATUS == "patched"
    original = _SVG_MODULE.correctCoordinates.__wrapped__
    with pytest.raises(ValueError, match="not enough values to unpack"):
        original(_dom(pg_svg.VIEWBOX_BORDER_PATH), [], None, {})


def test_the_viewbox_border_path_parses_and_keeps_its_closepath():
    """The exact ``d`` that broke every export goes through, transformed, with ``Z`` intact."""
    node = _dom(pg_svg.VIEWBOX_BORDER_PATH)
    _SVG_MODULE.correctCoordinates(node, [], None, {})
    out = _path_d(node)

    tokens = out.split()
    assert tokens[-1] == "Z", f"closepath lost: {out!r}"
    assert tokens[0].startswith("M")
    # 6 tokens in, 6 tokens out -- nothing dropped to make the parse succeed.
    assert len(tokens) == len(pg_svg.VIEWBOX_BORDER_PATH.split())
    # And the coordinates really went through the matrix: M0,0 -> M(2*0+10),(3*0+20).
    assert tokens[0] == "M10.0,20.0"


def test_the_wrapper_agrees_with_an_in_loop_fix_character_for_character():
    """Wrapping instead of forking must not change the output. This is what says so.

    Covers the plain border, a closepath in the MIDDLE of a path (two subpaths), and a
    path made only of comma-less tokens.
    """
    for d in (
        pg_svg.VIEWBOX_BORDER_PATH,
        "M0,0 L10,0 Z M20,20 L30,20 L30,30 Z",
        "M5,5 L6,6",                      # no comma-less token at all: pure pass-through
    ):
        node = _dom(d)
        _SVG_MODULE.correctCoordinates(node, [], None, {})
        assert _path_d(node) == _reference_in_loop_fix(d, _TR), f"disagreement on {d!r}"


def test_a_path_of_only_commandless_tokens_does_not_crash_and_is_left_alone():
    """The empty-string guard: upstream's ``newCoords[0]`` must never see ``''``."""
    node = _dom("Z")
    _SVG_MODULE.correctCoordinates(node, [], None, {})     # must not raise IndexError
    assert _path_d(node) == "Z"


def test_install_is_idempotent_and_never_wraps_its_own_wrapper():
    before = _SVG_MODULE.correctCoordinates
    assert pg_svg.install() == "already-patched"
    assert pg_svg.install() == "already-patched"
    assert _SVG_MODULE.correctCoordinates is before, "the wrapper was re-wrapped"


def test_install_stands_aside_when_upstream_already_handles_closepaths(monkeypatch):
    """Version awareness is BEHAVIOURAL, not a version-string table.

    `install` probes the installed function with the real failing path; a pyqtgraph (or a
    distribution backport) that parses it is left completely alone, so this file cannot
    keep shimming code that no longer needs it.
    """
    def already_fixed(node, defs, item, options):
        return None                       # parses anything, patches nothing

    monkeypatch.setattr(_SVG_MODULE, "correctCoordinates", already_fixed)
    assert pg_svg.install() == "upstream-fixed"
    assert _SVG_MODULE.correctCoordinates is already_fixed

    def broken_in_a_new_way(node, defs, item, options):
        raise RuntimeError("upstream rewrote this")

    monkeypatch.setattr(_SVG_MODULE, "correctCoordinates", broken_in_a_new_way)
    assert pg_svg.install() == "unknown-upstream"
    assert _SVG_MODULE.correctCoordinates is broken_in_a_new_way


# ================================================================================================
# end-to-end: real plots, real files, valid SVG
# ================================================================================================

def test_a_filled_step_histogram_exports_as_valid_svg_with_its_fill(qapp, tmp_path):
    """A dwell histogram -- filled ``stepMode`` curve -- is the commonest thing to export."""
    with _plot() as (w, pi):
        edges = np.linspace(0.0, 5.0, 21)
        counts = np.array([2, 5, 11, 24, 40, 61, 78, 71, 55, 38,
                           25, 17, 11, 7, 4, 3, 2, 1, 1, 1], dtype=float)
        pi.plot(edges, counts, stepMode="center", fillLevel=0,
                brush=(0, 114, 178, 150), pen="k")
        out = pg_svg.export_svg(pi, tmp_path / "hist.svg")

    root = ET.parse(out).getroot()                        # raises on malformed XML
    assert root.tag == f"{{{SVG_NS}}}svg"
    paths = [p for p in root.iter(f"{{{SVG_NS}}}path")]
    assert len(paths) >= 2, "expected at least the border and the histogram outline"
    # The fill under the steps is there, in the colour asked for. Qt puts paint on the
    # enclosing <g>, not on the <path>, so that is where the brush shows up.
    filled = [g for g in root.iter(f"{{{SVG_NS}}}g")
              if (g.get("fill") or "none").lower() == "#0072b2"]
    assert filled, "the fill under the histogram did not survive export"
    assert any(child.get("d", "").strip() for g in filled
               for child in g.iter(f"{{{SVG_NS}}}path")), "the filled region has no geometry"
    # Every closepath our patch preserved is still a legal token.
    for p in paths:
        for token in p.get("d", "").split():
            assert "," in token or token in {"M", "L", "Z", "z", "H", "V", "C"}, token


def test_a_plot_with_a_legend_and_a_textitem_keeps_both_labels(qapp, tmp_path):
    """Legend entries and a TextItem annotation must reach the file as real ``<text>``."""
    with _plot() as (w, pi):
        pi.addLegend()
        pi.plot([0, 1, 2, 3], [1.0, 2.0, 1.5, 2.5], pen="r", name="ferritin")
        pi.plot([0, 1, 2, 3], [0.5, 0.9, 0.7, 1.1], pen="b", name="buffer")
        label = pg.TextItem("pore 22 nm")
        label.setPos(1.0, 2.0)
        pi.addItem(label)
        out = pg_svg.export_svg(pi, tmp_path / "legend.svg")

    root = ET.parse(out).getroot()
    assert root.tag == f"{{{SVG_NS}}}svg"
    found = _texts(root)
    for expected in ("ferritin", "buffer", "pore 22 nm"):
        assert any(expected in t for t in found), f"{expected!r} missing from {found}"


def test_an_empty_plotitem_exports_too(qapp, tmp_path):
    """The recon's sharpest case: even a plot with NOTHING in it used to fail, because the
    ViewBox border alone is enough to trip the parser."""
    with _plot() as (w, pi):
        out = pg_svg.export_svg(pi, tmp_path / "empty.svg")
    root = ET.parse(out).getroot()
    assert root.tag == f"{{{SVG_NS}}}svg"
    assert list(root.iter(f"{{{SVG_NS}}}path")), "not even the axes were written"


# ================================================================================================
# limitation (a): geometry the exporter never reconciled
# ================================================================================================

def _kde_overlay(pi):
    """A KDE overlay in the app's idiom: an IsocurveItem whose coordinates are GRID INDICES
    (0..255) sharing a plot with a scatter in data coordinates (~0..5). That mismatch is
    what turns a stale ViewBox matrix into coordinates 1e5 px off the canvas."""
    gy, gx = np.mgrid[0:256, 0:256]
    density = np.exp(-(((gx - 128) / 40.0) ** 2 + ((gy - 120) / 30.0) ** 2))
    rng = np.random.default_rng(0)
    pi.addItem(pg.ScatterPlotItem(x=rng.normal(2.5, 0.6, 200),
                                  y=rng.normal(0.35, 0.08, 200), size=3))
    pi.addItem(pg.IsocurveItem(data=density, level=0.5, pen="g"))


def test_without_the_pre_paint_pass_the_isocurve_lands_far_off_the_canvas(qapp, tmp_path):
    """The defect, reproduced: ``updateAutoRange()`` moves the range but nothing recomputes
    the item->scene matrix until a paint, and SVGExporter never triggers one."""
    with _plot() as (w, pi):
        _kde_overlay(pi)
        pi.getViewBox().updateAutoRange()          # what a panel naturally does before export
        out = pg_svg.export_svg(pi, tmp_path / "stale.svg", refresh=False)
    assert _svg_coordinate_extremes(out) > 10_000, (
        "expected the stale-matrix blow-up; if this is gone, pyqtgraph now runs its "
        "pre-paint pass inside SVGExporter and refresh_for_export can be retired")


def test_export_svg_runs_the_pre_paint_pass_and_keeps_the_isocurve_on_the_canvas(qapp, tmp_path):
    """The fix: `export_svg` calls the same ``prepareForPaint`` that ``scene.render()`` --
    and therefore ImageExporter -- has always called."""
    with _plot() as (w, pi):
        _kde_overlay(pi)
        pi.getViewBox().updateAutoRange()
        out = pg_svg.export_svg(pi, tmp_path / "fresh.svg")   # refresh=True by default
    # Canvas is ~640x480 scene units; allow generous slack for stroke overshoot.
    assert _svg_coordinate_extremes(out) < 2_000


def test_the_pre_paint_pass_does_not_move_a_range_the_user_chose(qapp, tmp_path):
    """`refresh_for_export` may only flush deferred work -- never re-frame a manual zoom."""
    with _plot() as (w, pi):
        pi.plot([0, 1, 2, 3], [0.0, 1.0, 4.0, 9.0])
        vb = pi.getViewBox()
        vb.setRange(xRange=(0.5, 1.5), yRange=(0.0, 2.0), padding=0)
        before = [v for axis in vb.viewRange() for v in axis]
        pg_svg.refresh_for_export(pi)
        after = [v for axis in vb.viewRange() for v in axis]
    assert after == pytest.approx(before)
    assert before == pytest.approx([0.5, 1.5, 0.0, 2.0])


# ================================================================================================
# limitation (b): one <path> per scatter point
# ================================================================================================

def test_a_scatter_really_does_emit_one_path_per_point(qapp, tmp_path):
    """The measured size law, at a size a test can afford: ~1 path and ~528 bytes per point."""
    n = 400
    rng = np.random.default_rng(2)
    with _plot() as (w, pi):
        pi.addItem(pg.ScatterPlotItem(x=rng.normal(2.5, 0.6, n), y=rng.normal(0.35, 0.08, n),
                                      size=4, pen=None, brush=(0, 114, 178, 120)))
        pi.getViewBox().updateAutoRange()
        out = pg_svg.export_svg(pi, tmp_path / "scatter.svg")

    text = out.read_text()
    assert text.count("<path") >= n, "the exporter stopped drawing one element per point"
    # Prediction within a factor of two of reality is all the warning text needs.
    predicted = pg_svg.estimate_svg_bytes_for_points(n)
    assert 0.5 * predicted < out.stat().st_size < 2.0 * predicted


def test_the_size_warning_counts_the_points_and_fires_only_when_it_should(qapp):
    rng = np.random.default_rng(3)
    with _plot() as (w, pi):
        pi.plot([0, 1, 2], [1, 2, 3])                      # a curve is ONE path, not three
        assert pg_svg.count_vector_points(pi) == 0
        assert pg_svg.svg_size_warning(pi) is None

        small = pg_svg.SVG_POINT_WARN_THRESHOLD - 1
        s1 = pg.ScatterPlotItem(x=rng.random(small), y=rng.random(small), size=3)
        pi.addItem(s1)
        assert pg_svg.count_vector_points(pi) == small
        assert pg_svg.svg_size_warning(pi) is None, "warned below the threshold"

        s2 = pg.ScatterPlotItem(x=rng.random(50), y=rng.random(50), size=3)
        pi.addItem(s2)
        assert pg_svg.count_vector_points(pi) == small + 50
        warning = pg_svg.svg_size_warning(pi)

    assert warning is not None
    assert f"{small + 50:,}" in warning
    assert "PNG" in warning, "the warning must name the cheaper alternative"


def test_symbols_on_a_plotdataitem_are_counted_too(qapp):
    """``pi.plot(..., symbol='o')`` hides a ScatterPlotItem inside a PlotDataItem; it costs
    exactly as much and must not slip past the count."""
    with _plot() as (w, pi):
        pi.plot(np.arange(300.0), np.arange(300.0), symbol="o", symbolSize=4, pen=None)
        assert pg_svg.count_vector_points(pi) == 300
