"""B1 — vector export (SVG + PDF) from the interactive Plots window.

Written from the DEFECTS, not from the patch:

* the panel's *Save image…* filter was ``PNG image (*.png);;All files (*)``, so the one
  vector format the app appeared to offer was reachable only by typing a ``.svg`` name
  through "All files" — and pyqtgraph's SVG exporter raised on EVERY plot on this Qt
  (``nanoscope/plot/pg_svg.py``), so it never worked;
* the patch that fixes that exporter installs on IMPORT, and pyqtgraph's own right-click
  "Export… ▸ SVG" dialog runs no code of ours — so a lazily-imported workaround would
  leave that menu broken. ``test_the_panel_imports_the_svg_patch_at_module_scope`` is the
  guard for that and is an AST check on purpose: an import moved into a function would
  still make every test in this file pass;
* pg_svg limitation (a) — a plot no repaint has reconciled exports its curves up to five
  orders of magnitude off the canvas. ``test_a_never_shown_kde_exports_onto_the_canvas``
  pins the fix, and its ``refresh=False`` twin is the "can this test fail?" anchor;
* pg_svg limitation (b) — one ``<path>`` per scatter symbol. The panel must SAY so before
  a nine-second write, and must never quietly subsample or swap the format.

PDF is here because measurement said it should be: the same 10 000-point scatter is
0.151 MB / 0.10 s as PDF against 5.28 MB / 3.06 s as SVG (see ``figures.export_plot_pdf``).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui import plots_panel
from nanoscope.gui.plots_panel import PlotsWindow
from nanoscope.plot import figures, pg_svg

PANEL_SOURCE = pathlib.Path(plots_panel.__file__)


# ================================================================================================
# builders
# ================================================================================================
def _events(n: int = 200, *, seed: int = 0) -> pd.DataFrame:
    """Two separable blobs in (dwell_s, amplitude) — the panel's usual input."""
    rng = np.random.default_rng(seed)
    half = n // 2
    return pd.DataFrame({
        "dwell_s": np.abs(np.r_[rng.normal(1e-4, 1.5e-5, half),
                                rng.normal(6e-4, 6e-5, n - half)]),
        "amplitude": np.abs(np.r_[rng.normal(20.0, 2.0, half),
                                  rng.normal(48.0, 3.0, n - half)]),
        "frac_depth": np.abs(rng.normal(0.25, 0.05, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def window(qapp, qtbot):
    """A Plots window that is NEVER SHOWN — which is the interesting case for (a)."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"apo": _events()})
    return w


def _svg_coords(path: pathlib.Path) -> np.ndarray:
    """Every numeric coordinate in every ``<path d=…>`` of an SVG file."""
    text = path.read_text()
    ds = re.findall(r'\sd="([^"]*)"', text)
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", " ".join(ds))
    return np.abs(np.asarray([float(n) for n in nums], dtype=float))


# ================================================================================================
# the patch is installed before ANY export, including pyqtgraph's own dialog
# ================================================================================================
def test_the_panel_imports_the_svg_patch_at_module_scope():
    """The import that installs the closepath fix may not be deferred into a function.

    pyqtgraph's right-click *Export… ▸ SVG* on this plot calls ``SVGExporter`` directly.
    Nothing of ours runs first, so the ONLY thing that can have fixed the exporter by
    then is an import that happened when the module was loaded. An import moved inside
    ``save_image`` would fix our button and leave that menu raising.
    """
    tree = ast.parse(PANEL_SOURCE.read_text())
    # ``tree.body`` is the MODULE body and nothing nested, which is the whole point:
    # ``from ..plot import pg_svg`` parses as ImportFrom(module="plot", level=2).
    module_level = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "plot"
        for alias in node.names
    } | {
        alias.name.split(".")[-1]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "pg_svg" in module_level, (
        "nanoscope/gui/plots_panel.py must import nanoscope.plot.pg_svg at MODULE scope; "
        "importing it inside the export method leaves pyqtgraph's own right-click "
        "Export dialog raising on every plot")


def test_the_patch_is_actually_in_place_once_the_panel_is_imported():
    import sys

    module = sys.modules["pyqtgraph.exporters.SVGExporter"]
    assert pg_svg.PATCH_STATUS in ("patched", "already-patched", "upstream-fixed")
    if pg_svg.PATCH_STATUS != "upstream-fixed":
        assert getattr(module.correctCoordinates, "_nanoscope_closepath_fix", False)


# ================================================================================================
# the dialog offers the formats that work
# ================================================================================================
def test_the_save_dialog_offers_png_svg_and_pdf():
    filters = PlotsWindow._IMAGE_FILTERS
    assert "*.png" in filters and "*.svg" in filters and "*.pdf" in filters
    # PNG stays the first (default) filter: a slide is still the commonest destination.
    assert filters.index("*.png") < filters.index("*.svg")


def test_the_failure_advice_names_the_formats_the_panel_really_renders(tmp_path):
    """"try a .png name" is right for four panels and wrong for this one.

    The shared sentence takes the list from its caller rather than growing a second
    copy in one panel: PSD / I-V / voltage-dependence / event-plot go straight to
    ``ImageExporter`` and PNG is genuinely their way out, while a user sent to PNG from
    the Plots window would be talked out of the two formats a journal asks for.
    """
    exc = figures.ImageNotRendered("no")
    default = figures.image_write_failure_text(tmp_path / "nope.eps", exc)
    assert "try a .png name" in default
    assert ".svg" not in default

    plots = figures.image_write_failure_text(
        tmp_path / "nope.eps", exc, formats=(".png", ".svg", ".pdf"))
    assert ".png" in plots and ".svg" in plots and ".pdf" in plots


def test_the_panel_asks_for_that_wording(window, tmp_path, monkeypatch):
    """The panel must actually PASS its formats — a default it forgets is silent."""
    shown: list[str] = []
    monkeypatch.setattr(plots_panel.QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a: shown.append(a[-1])))
    monkeypatch.setattr(
        pg_svg, "export_svg",
        lambda *a, **k: (_ for _ in ()).throw(figures.ImageNotRendered("nope")))
    assert window.save_image(str(tmp_path / "x.svg")) is None
    assert shown and ".svg" in shown[0] and ".pdf" in shown[0]


# ================================================================================================
# SVG
# ================================================================================================
def test_saving_an_svg_writes_parseable_svg(window, tmp_path):
    out = window.save_image(str(tmp_path / "plot.svg"))
    assert out == tmp_path / "plot.svg"
    assert out.stat().st_size > 0
    root = ET.parse(out).getroot()
    assert root.tag.endswith("svg")


def test_a_never_shown_kde_exports_onto_the_canvas(window, tmp_path):
    """pg_svg limitation (a): the deferred ViewBox matrix, and the pre-paint that fixes it.

    ``SVGExporter`` paints items directly and is the one render path in pyqtgraph that
    skips ``GraphicsScene.prepareForPaint``. On a window driven by ``set_plot_type`` and
    never shown — a scripted or batch export — the ViewBox matrix still describes the
    default [0, 1] range while ``viewRange()`` reports the KDE's grid, and the isocurves
    land ~1e6 px off a 640 px canvas.

    ``save_image`` leaves ``export_svg``'s ``refresh`` at its default, so this passes;
    the anchor below proves the assertion is not vacuous.
    """
    window.set_plot_type("KDE (density)")
    window.set_kde_columns("dwell_s", "amplitude")
    assert not window.isVisible()

    out = window.save_image(str(tmp_path / "kde.svg"))
    coords = _svg_coords(out)
    assert coords.size > 0
    # The canvas is a few hundred px; anything beyond 5 000 is off it by orders of
    # magnitude, which is the failure this pins.
    assert coords.max() < 5_000, f"max |coord| = {coords.max()}"


def test_the_off_canvas_defect_is_real_without_the_pre_paint(window, tmp_path):
    """The "can this test fail?" anchor for the test above.

    Exports the SAME never-shown KDE with the pre-paint pass switched off. If a
    pyqtgraph upgrade ever makes this pass, the refresh has become unnecessary and this
    announces it here rather than leaving a workaround nobody can justify.
    """
    window.set_plot_type("KDE (density)")
    window.set_kde_columns("dwell_s", "amplitude")
    out = tmp_path / "kde_raw.svg"
    pg_svg.export_svg(window.plot_item(), out, refresh=False)
    assert _svg_coords(out).max() > 50_000


# ================================================================================================
# SVG limitation (b) — said out loud, never worked around silently
# ================================================================================================
def test_a_small_plot_is_never_interrupted(window, tmp_path, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(plots_panel, "confirm_large_svg",
                        lambda parent, msg: asked.append(msg) or True)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    window.save_image(str(tmp_path / "small.svg"))
    assert asked == []


def test_a_big_scatter_warns_with_the_size_and_the_alternative(window, tmp_path,
                                                              monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(plots_panel, "confirm_large_svg",
                        lambda parent, msg: asked.append(msg) or True)
    window.set_data({"big": _events(pg_svg.SVG_POINT_WARN_THRESHOLD + 500, seed=3)})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")

    out = window.save_image(str(tmp_path / "big.svg"))
    assert out is not None and out.is_file()
    assert len(asked) == 1
    # The count and the size, from pg_svg's measured law; and this panel's own advice.
    assert "5,500 points" in asked[0]
    assert "MB" in asked[0]


def test_declining_the_warning_writes_nothing(window, tmp_path, monkeypatch):
    monkeypatch.setattr(plots_panel, "confirm_large_svg", lambda parent, msg: False)
    window.set_data({"big": _events(pg_svg.SVG_POINT_WARN_THRESHOLD + 500, seed=3)})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")

    target = tmp_path / "declined.svg"
    assert window.save_image(str(target)) is None
    assert not target.exists()


def test_the_warning_never_degrades_the_figure_behind_the_users_back(window, tmp_path,
                                                                    monkeypatch):
    """Saying yes writes SVG — not a PNG under the .svg name, and not fewer points.

    A raster fallback chosen for the user would write a file the overwrite prompt never
    named, and a subsampled one would be a figure that lost events without saying so.
    """
    monkeypatch.setattr(plots_panel, "confirm_large_svg", lambda parent, msg: True)
    n = pg_svg.SVG_POINT_WARN_THRESHOLD + 500
    window.set_data({"big": _events(n, seed=3)})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")

    out = window.save_image(str(tmp_path / "kept.svg"))
    assert ET.parse(out).getroot().tag.endswith("svg")     # really SVG, not a PNG
    assert pg_svg.count_vector_points(window.plot_item()) == n


# ================================================================================================
# PDF
# ================================================================================================
def test_saving_a_pdf_writes_a_real_pdf(window, tmp_path):
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    out = window.save_image(str(tmp_path / "plot.pdf"))
    assert out == tmp_path / "plot.pdf"
    assert out.read_bytes()[:5] == b"%PDF-"
    assert out.stat().st_size > 5_000     # a blank page is ~1 kB; this has a plot on it


def test_the_pdf_carries_the_points_and_not_just_the_axes(window, tmp_path):
    """A blank page is also a valid PDF, so the file must be shown to grow with the data."""
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    small = window.save_image(str(tmp_path / "small.pdf")).stat().st_size

    window.set_data({"more": _events(4_000, seed=4)})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")
    big = window.save_image(str(tmp_path / "big.pdf")).stat().st_size
    assert big > small * 2


def test_the_pdf_stays_far_smaller_than_the_same_plot_as_svg(window, tmp_path,
                                                             monkeypatch):
    """The measurement the decision to build PDF rests on, pinned as a property.

    Not a byte count — those move with Qt — but the ORDER: pyqtgraph's SVG writes one
    ``<path>`` per symbol while Qt's PDF backend compresses its content stream, so on a
    scatter the PDF must be several times the smaller. If that ever inverts, "PDF is the
    better vector format for this panel" stops being true and should be re-decided.
    """
    monkeypatch.setattr(plots_panel, "confirm_large_svg", lambda parent, msg: True)
    window.set_data({"big": _events(6_000, seed=5)})
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "amplitude")

    pdf = window.save_image(str(tmp_path / "a.pdf")).stat().st_size
    svg = window.save_image(str(tmp_path / "a.svg")).stat().st_size
    assert pdf * 5 < svg, f"pdf={pdf} svg={svg}"


def test_the_pdf_page_is_the_canvas_by_default(window, tmp_path):
    out = window.save_image(str(tmp_path / "native.pdf"))
    rect = window.plot_item().sceneBoundingRect()
    w_pt, h_pt = _pdf_page_size(out)
    assert w_pt == pytest.approx(rect.width(), abs=2)
    assert h_pt == pytest.approx(rect.height(), abs=2)


def test_the_export_size_boxes_pin_the_pdf_page(window, tmp_path):
    """*Export size* means the same thing for the PDF as it does for the PNG.

    A width alone follows the aspect ratio, exactly as ``ImageExporter`` does — the
    alternative (ignoring the boxes for PDF) is a control that silently applies to two
    of the three formats.
    """
    rect = window.plot_item().sceneBoundingRect()
    window.set_export_size(800, 0)
    w_pt, h_pt = _pdf_page_size(window.save_image(str(tmp_path / "wide.pdf")))
    assert w_pt == pytest.approx(800, abs=2)
    assert h_pt == pytest.approx(800 * rect.height() / rect.width(), abs=3)

    window.set_export_size(640, 480)
    w_pt, h_pt = _pdf_page_size(window.save_image(str(tmp_path / "both.pdf")))
    assert (round(w_pt), round(h_pt)) == (640, 480)


def _pdf_page_size(path: pathlib.Path) -> tuple[float, float]:
    """(width, height) in points, from the first /MediaBox in the file."""
    raw = path.read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", raw)
    assert m, "no /MediaBox in the PDF"
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return x1 - x0, y1 - y0


def test_a_pdf_of_an_item_with_no_canvas_is_refused(tmp_path):
    """The refusal a PNG of the same item would give, so one sentence can cover both."""
    import pyqtgraph as pg

    with pytest.raises(figures.ImageNotRendered):
        figures.export_plot_pdf(pg.PlotItem(), tmp_path / "orphan.pdf")


# ================================================================================================
# every vector write is staged, like the raster one
# ================================================================================================
@pytest.mark.parametrize("suffix", [".svg", ".pdf"])
def test_a_failed_vector_write_leaves_the_previous_export_whole(window, tmp_path,
                                                               monkeypatch, suffix):
    """USER_GUIDE §12: a failure leaves the previous file whole rather than truncated."""
    target = tmp_path / f"prev{suffix}"
    target.write_bytes(b"yesterday's export")

    def boom(*a, **k):
        raise RuntimeError("renderer said no")

    monkeypatch.setattr(pg_svg, "export_svg", boom)
    monkeypatch.setattr(figures, "export_plot_pdf", boom)
    shown: list[str] = []
    monkeypatch.setattr(plots_panel.QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a: shown.append(a[-1])))

    assert window.save_image(str(target)) is None
    assert target.read_bytes() == b"yesterday's export"
    assert shown and "renderer said no" in shown[0]
