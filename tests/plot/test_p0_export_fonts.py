"""Phase 0 / G4 -- the fonts an export carries (plotting redesign, design section 4).

What is pinned here is read off the FILE the Export button's call writes (or off the font
matplotlib really draws with), not off a state field:

* the standard report figures (``figures.save_figure``, behind Analysis > Generate report
  figures and Export density figure) write their PDF text as TrueType (Type 42), not
  matplotlib's default Type 3 glyph drawings, and in Arial -- the face the figure was laid
  out in -- instead of laying out in Helvetica and printing in DejaVu Sans;
* the style's default sans-serif list is Arial first, with DejaVu Sans last so Greek still
  draws on a machine without the others;
* the style's ``font.family`` is a LIST (Arial, its stand-ins, the generic, DejaVu Sans), not the
  word ``sans-serif`` -- which names one face -- so a character Arial lacks (the subscript zero of
  ΔI/I₀, the superscript minus of s⁻¹) is drawn from the next face instead of as a missing-glyph box;
* the live plots' SVG (``pg_svg``, whichever caller asks pyqtgraph for it) names a
  fallback stack on every ``<text>`` instead of Qt's one bare family, writes Arial for text
  that was given no font when the application's own face is a private one (macOS), and never
  names that private face; a family someone picked is named first, the application's own
  (Segoe UI on Windows) included;
* the matplotlib figures with a font of their own (Comparison figures, the Study figure
  editor) draw in the CHOSEN family's list, or in Arial when none is chosen, and a family
  that is not installed falls back quietly;
* the live plots' PDF embeds TrueType and no Type 3 (a guard: nothing changed there).

Synthetic data only: in-memory frames and documents, files under ``tmp_path``.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging
import re
import warnings
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import replace
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter
from PySide6 import QtGui, QtWidgets

from nanoscope.plot import figure_fonts as ff
from nanoscope.plot import figures, pg_svg, style
from nanoscope.plot import study_figure_style
from nanoscope.plot.comparison_figures import render_comparison

from .test_comparison_figures import prepare
from .test_figures import make_events

SVG_NS = "http://www.w3.org/2000/svg"
ARIAL_STACK = ff.css_font_family(None)
TIMES_STACK = ff.css_font_family("Times New Roman")


# ================================================================================================
# helpers
# ================================================================================================
def _installed(name: str) -> bool:
    return name.casefold() in {e.name.casefold() for e in font_manager.fontManager.ttflist}


needs_arial = pytest.mark.skipif(not _installed("Arial"), reason="Arial is not installed here")
needs_times = pytest.mark.skipif(not _installed("Times New Roman"),
                                 reason="Times New Roman is not installed here")


@pytest.fixture
def application_font(qapp):
    """``use(family)`` sets the family Qt draws text in when it was given none (the application
    font) for one test; the old font is put back after it. ``QFont.family()`` reports the family
    as requested, installed or not, which is all the SVG writer reads."""
    saved = QtGui.QGuiApplication.font()

    def use(family: str) -> None:
        font = QtGui.QFont(saved)
        font.setFamily(family)
        QtGui.QGuiApplication.setFont(font)

    yield use
    QtGui.QGuiApplication.setFont(saved)


@pytest.fixture
def private_application_font(application_font):
    """The application font as it is on macOS under the native platform: a private face."""
    application_font(".AppleSystemUIFont")


def _pdf_fonts(path) -> dict:
    """The font objects in a PDF file, read off its bytes (no PDF library: none is a dependency)."""
    raw = open(path, "rb").read()

    def count(subtype: bytes) -> int:
        return len(re.findall(rb"/Subtype\s*/" + subtype + rb"\b", raw))

    return {
        "type3": count(b"Type3"), "truetype": count(b"TrueType"), "type0": count(b"Type0"),
        "cid_truetype": count(b"CIDFontType2"), "font_file2": len(re.findall(rb"/FontFile2", raw)),
        "names": sorted({m.decode().split("+")[-1]
                         for m in re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-_,.]+)", raw)}),
    }


def _assert_embedded_truetype(path) -> dict:
    fonts = _pdf_fonts(path)
    assert fonts["type3"] == 0, f"{path.name}: Type 3 glyph drawings in the PDF: {fonts}"
    assert fonts["cid_truetype"] + fonts["truetype"] >= 1 and fonts["font_file2"] >= 1, \
        f"{path.name}: no embedded TrueType font program: {fonts}"
    return fonts


def _svg_text_families(svg_text: str) -> list[str]:
    """The ``font-family`` of every ``<text>`` element, as the XML parser reads it."""
    root = ET.fromstring(svg_text)
    return [e.get("font-family", "") for e in root.iter(f"{{{SVG_NS}}}text")]


def _all_font_families(svg_text: str) -> set[str]:
    """Every ``font-family`` value anywhere in the file (a text element or a group)."""
    root = ET.fromstring(svg_text)
    return {e.get("font-family") for e in root.iter() if e.get("font-family") is not None}


@contextmanager
def _fonts_drawn():
    """Every text request matplotlib resolved while this is open: ``(requested family list,
    font files)`` in request order. The first file is the face the text is drawn in, the rest
    are per-glyph fallbacks. Text drawn as outlines (the canvas item) and text drawn by a raster
    or PDF backend resolve through this one call.
    """
    seen: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    original = font_manager.FontManager._find_fonts_by_props

    def spy(self, prop, *args, **kwargs):
        paths = original(self, prop, *args, **kwargs)
        seen.append((tuple(FontProperties._from_any(prop).get_family()), tuple(paths)))
        return paths

    with mock.patch.object(font_manager.FontManager, "_find_fonts_by_props", spy):
        yield seen


@contextmanager
def _fonts_drawn_by_savefig():
    """:func:`_fonts_drawn`, but only for what ``Figure.savefig`` itself resolves -- not the draws a
    caller makes first (the study's styler draws the figure once before it styles it)."""
    seen: list = []
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        with _fonts_drawn() as inner:
            try:
                return original(self, *args, **kwargs)
            finally:
                seen.extend(inner)

    with mock.patch.object(Figure, "savefig", savefig):
        yield seen


def _first_faces(seen, requested=None) -> set[str]:
    """The family names of the first font of every request (of those that asked for the
    family list ``requested``, when given)."""
    return {font_manager.get_font(paths[0]).family_name for family, paths in seen
            if paths and (requested is None or list(family) == list(requested))}


def _fresh_font_caches():
    """matplotlib caches ``findfont`` answers per rc, not per installed-font list."""
    font_manager.FontManager._findfont_cached.cache_clear()


@contextmanager
def _only_installed(*families):
    """A computer that has only ``families`` (and DejaVu, which ships with matplotlib)."""
    kept = [e for e in font_manager.fontManager.ttflist if e.name in families or e.name.startswith("DejaVu")]
    with mock.patch.object(font_manager.fontManager, "ttflist", kept):
        _fresh_font_caches()
        yield
    _fresh_font_caches()


def _missing_glyphs(caught) -> list[str]:
    """The 'Glyph … missing from font(s) …' warnings matplotlib raised: it draws a box for each."""
    return [str(w.message) for w in caught if "missing from font" in str(w.message)]


def _has_glyph(family: str, char: str) -> bool:
    path = next(e.fname for e in font_manager.fontManager.ttflist if e.name == family)
    return font_manager.get_font(path).get_char_index(ord(char)) != 0


def _events_for_page(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": np.abs(rng.normal(3e-4, 1e-4, n)) + 1e-5,
        "amplitude": np.abs(rng.normal(30.0, 5.0, n)),
        "frac_depth": np.abs(rng.normal(0.25, 0.05, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


# ================================================================================================
# 1. save_figure: the report figures' PDFs
# ================================================================================================
_BUILDERS = {
    "dwell": lambda ev: figures.dwell_histogram(ev, fit=True),
    "depth": lambda ev: figures.depth_histogram(ev, fit=True),
    "scatter": lambda ev: figures.scatter_dwell_depth(ev),
    "density": lambda ev: figures.joint_density({"pore": ev}),
}


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_report_figure_pdfs_hold_truetype_and_no_type3(name, tmp_path):
    """Before: all three font objects of every report PDF were Type 3 (Nature asks for 42),
    because ``save_figure`` runs outside the style that says 42."""
    fig, _ax = _BUILDERS[name](make_events())
    out = figures.save_figure(fig, tmp_path / f"fig_{name}.pdf")
    plt.close(fig)
    _assert_embedded_truetype(out["pdf"])


@needs_arial
def test_report_pdf_text_is_arial_the_face_it_was_laid_out_in(tmp_path):
    """No math in this figure, so every glyph is body text: the PDF must hold Arial alone.
    Before: DejaVu Sans (matplotlib's own default list, applied at save time)."""
    with style.nanoscope_style():
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [3, 1, 2])
        ax.set_xlabel("Dwell (ms)")
        ax.set_ylabel("Counts")
        ax.set_title("Dwell histogram")
    out = figures.save_figure(fig, tmp_path / "plain.pdf", data={"x": [1, 2, 3], "y": [3, 1, 2]})
    plt.close(fig)
    fonts = _assert_embedded_truetype(out["pdf"])
    assert fonts["names"] == ["ArialMT"], fonts


@needs_arial
def test_the_annotation_box_and_the_text_around_it_share_one_face(tmp_path):
    """The box is pinned to a font FILE at build time; the rest is resolved at save time.
    Both must now be Arial (before: Helvetica for the box, DejaVu Sans for the rest)."""
    fig, ax = figures.dwell_histogram(make_events(), fit=True)
    boxes = [t for t in ax.texts if t.get_fontproperties().get_file() is not None]
    assert boxes, "no annotation box was built"
    assert {font_manager.get_font(t.get_fontproperties().get_file()).family_name for t in boxes} == {"Arial"}
    with _fonts_drawn() as seen:
        figures.save_figure(fig, tmp_path / "box")
    plt.close(fig)
    # every Text that carries the style's family list (the labels, the ticks) is printed in Arial;
    # what is not Arial is the math text, which asks for its own fonts
    assert _first_faces(seen, requested=style.font_family_list()) == {"Arial"}
    assert "Helvetica" not in _first_faces(seen)


@pytest.fixture
def detected_window(qapp, qtbot, tmp_path):
    """A MainWindow holding one synthetic recording (fourteen blockades) detected on its
    current channel -- what Generate report figures… and Export density figure… work from."""
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    from nanoscope.gui.main_window import MainWindow

    fs, n = 100_000.0, 80_000
    rng = np.random.default_rng(0)
    trace = 100.0 + rng.normal(0.0, 1.5, n)
    for k in range(14):
        start = 4_000 + k * 5_000
        trace[start:start + 250 + 20 * k] -= 30.0
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic", n_samples=n,
                         sample_rate_hz=fs, rate_provenance=Provenance.FILE_META, channels=(info,))
    document = TraceDocument(meta=meta, channels=(np.ascontiguousarray(trace, dtype=np.float32),))
    window = MainWindow()
    qtbot.addWidget(window)
    doc_id = window._session.add_document(document)
    window._create_default_layers(document)
    window._on_selection(doc_id, 0)
    window._analysis_panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert window._analysis_panel.current_detection() is not None
    yield window
    window._analysis_controller.abort()


def _choose_file(monkeypatch, target):
    """Answer the Save dialog the export commands open (the same patch the export tests use)."""
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "PDF (*.pdf)")))


@needs_arial
def test_generate_report_figures_writes_truetype_pdfs_in_arial(detected_window, monkeypatch, tmp_path):
    """The command itself (``MainWindow._export_figures``, behind Analysis > Generate report
    figures…), not a hand-built figure: every PDF it writes is TrueType, no Type 3."""
    out = tmp_path / "figs"
    out.mkdir()
    _choose_file(monkeypatch, out / "figures.pdf")
    detected_window._export_figures()
    pdfs = sorted(out.glob("figures_*.pdf"))
    assert {p.stem for p in pdfs} >= {"figures_dwell", "figures_depth", "figures_scatter"}
    for pdf in pdfs:
        fonts = _assert_embedded_truetype(pdf)
        assert "ArialMT" in fonts["names"], (pdf.name, fonts)
    # the dwell histogram writes tau and the microsecond sign as math text, which matplotlib sets in
    # DejaVu Sans whatever the font list says -- and the PDF carries that face too (USER_GUIDE)
    dwell = _pdf_fonts(out / "figures_dwell.pdf")["names"]
    assert any(name.startswith("DejaVuSans") for name in dwell), dwell


@needs_arial
def test_export_density_figure_writes_a_truetype_pdf(detected_window, monkeypatch, tmp_path):
    out = tmp_path / "density.pdf"
    _choose_file(monkeypatch, out)
    detected_window._export_density(ycol="depth_pA")
    fonts = _assert_embedded_truetype(out)
    assert "ArialMT" in fonts["names"], fonts


# ------------------------------------------------------------------------------------------------
# 1b. characters Arial does not have (the report figures used to be printed in DejaVu Sans, which does)
# ------------------------------------------------------------------------------------------------
# Subscript digits and the superscript minus and plus are not in Arial, Times New Roman or Courier
# New on a Mac, and are in Helvetica and DejaVu Sans. ``s⁻¹`` and ``k₁`` are what a rate constant
# looks like.


@needs_arial
def test_the_fractional_blockade_density_figure_draws_every_glyph_in_its_png_and_pdf(
        detected_window, monkeypatch, tmp_path):
    """Analysis > Export density figure… > 'fractional ΔI/I₀' writes the y label 'Fractional
    blockade  ΔI/I₀', which ends in SUBSCRIPT ZERO (U+2080), a glyph Arial does not have. The figure
    was printed in DejaVu Sans, which does; printed in Arial alone the zero was a missing-glyph box
    in the PNG and in the PDF (and a warning for every one the file drew)."""
    window = detected_window
    out = tmp_path / "fractional.pdf"
    _choose_file(monkeypatch, out)
    assert window._act_export_density.isEnabled()
    with monkeypatch.context() as patched:
        # the question "Which amplitude should the density use?", answered with its second button
        patched.setattr(QtWidgets.QMessageBox, "exec", lambda box: 0)
        patched.setattr(QtWidgets.QMessageBox, "clickedButton",
                        lambda box: next(b for b in box.buttons() if b.text() == "fractional ΔI/I₀"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            window._act_export_density.trigger()
    assert out.is_file() and out.with_suffix(".png").is_file()
    assert not _missing_glyphs(caught), _missing_glyphs(caught)
    fonts = _assert_embedded_truetype(out)
    assert "ArialMT" in fonts["names"]
    if not _has_glyph("Arial", "₀"):
        assert set(fonts["names"]) - {"ArialMT"}, f"the zero was drawn from no other face: {fonts}"


def test_text_that_arial_lacks_still_draws_in_a_report_figure(tmp_path):
    """The title, the axis labels and the source names (the legend) are the user's own words, so
    subscripts and superscript signs reach them: none may come out as a missing-glyph box."""
    with warnings.catch_warnings(record=True) as caught, _capture_font_logs() as logs:
        warnings.simplefilter("always")
        fig, _ax = figures.joint_density({"Sample₁ (k₂)": make_events()}, xlabel="k₀ (s⁻¹)",
                                         ylabel="ΔI/I₀ (s⁻¹)", title="n₃ = 10⁺ and 10⁻")
        out = figures.save_figure(fig, tmp_path / "unicode", data={"x": [1], "y": [1]})
    plt.close(fig)
    assert not _missing_glyphs(caught), _missing_glyphs(caught)
    # a list in font.family is a warning per name nobody has, on every text drawn: it names only
    # what is installed
    assert not logs, [r.getMessage() for r in logs]
    fonts = _assert_embedded_truetype(out["pdf"])
    if _installed("Arial") and not _has_glyph("Arial", "₀"):
        assert set(fonts["names"]) - {"ArialMT"}, f"drawn from no other face: {fonts}"


@needs_arial
def test_a_computer_with_arial_but_no_helvetica_draws_what_arial_lacks_in_dejavu_sans(tmp_path):
    """Windows has Arial and not Helvetica, and Liberation Sans, Arimo, TeX Gyre Heros and Nimbus
    Sans are not there either: the only face left to take a subscript from is DejaVu Sans, which
    ships with matplotlib. It has to be in the list, or the zero is a box."""
    with _only_installed("Arial"):
        if _has_glyph("Arial", "₀"):
            pytest.skip("this Arial has the subscripts")
        fig, _ax = figures.joint_density({"S₁": make_events()}, ylabel="ΔI/I₀ (s⁻¹)")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = figures.save_figure(fig, tmp_path / "windows", data={"x": [1], "y": [1]})
        plt.close(fig)
    assert not _missing_glyphs(caught), _missing_glyphs(caught)
    fonts = _assert_embedded_truetype(out["pdf"])
    assert "ArialMT" in fonts["names"] and any(n.startswith("DejaVuSans") for n in fonts["names"]), fonts


@needs_arial
def test_a_text_that_still_says_sans_serif_is_printed_in_the_styles_arial_by_save_figure(tmp_path):
    """A figure not built by the report builders (a script's own, drawn outside any style) has Texts
    that carry the WORD sans-serif; the word is resolved when the file is drawn, and it is the style's
    list that answers, as it does for the figures the builders make -- not matplotlib's DejaVu Sans."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3, 1, 2])
    ax.set_xlabel("Dwell (ms)")
    assert ax.xaxis.label.get_fontfamily() == ["sans-serif"]
    with _fonts_drawn() as seen:
        figures.save_figure(fig, tmp_path / "script", data={"x": [1, 2, 3], "y": [3, 1, 2]})
    plt.close(fig)
    assert _first_faces(seen, requested=["sans-serif"]) == {"Arial"}


def test_export_style_is_scoped_and_touches_only_the_font_keys():
    before = dict(matplotlib.rcParams)
    with style.export_style():
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42
        assert matplotlib.rcParams["font.sans-serif"][0] == "Arial"
        # the page layout of a save is what it always was
        assert matplotlib.rcParams["savefig.pad_inches"] == before["savefig.pad_inches"]
        assert matplotlib.rcParams["figure.autolayout"] == before["figure.autolayout"]
    assert matplotlib.rcParams["pdf.fonttype"] == before["pdf.fonttype"]
    assert matplotlib.rcParams["font.sans-serif"] == before["font.sans-serif"]


# ================================================================================================
# 2. the style's default list
# ================================================================================================
def test_the_default_sans_serif_list_is_arial_first_with_dejavu_as_the_last_resort():
    sans = style.rc_params()["font.sans-serif"]
    assert sans[:len(ff.matplotlib_font_list("Arial"))] == ff.matplotlib_font_list("Arial")
    assert sans[0] == "Arial" and sans[1] == "Helvetica"
    assert sans[-1] == "DejaVu Sans"


@needs_arial
def test_an_untouched_figure_is_laid_out_in_arial_inside_the_style():
    with style.nanoscope_style():
        face = font_manager.get_font(findfont(FontProperties(family=["sans-serif"]))).family_name
        box = figures._box_font().get_file()
    assert face == "Arial"
    assert font_manager.get_font(box).family_name == "Arial"


def test_greek_still_draws_on_a_machine_without_arial(monkeypatch, tmp_path):
    """Only DejaVu (which ships with matplotlib) is 'installed': tau, mu and Delta must still be
    real glyphs, in the PDF and the PNG, with no missing-glyph warning and no findfont warning."""
    dejavu = [e for e in font_manager.fontManager.ttflist if e.name.startswith("DejaVu")]
    assert dejavu, "matplotlib ships DejaVu"
    monkeypatch.setattr(font_manager.fontManager, "ttflist", dejavu)
    _fresh_font_caches()
    try:
        with warnings.catch_warnings(record=True) as caught, \
                _capture_font_logs() as logs:
            warnings.simplefilter("always")
            with style.nanoscope_style():
                fig, ax = plt.subplots()
                ax.set_xlabel("τ (µs)")
                ax.set_title("ΔI / I₀  τ = 0.92 µs")
                ax.plot([1, 2], [1, 2])
                face = font_manager.get_font(findfont(FontProperties(family=["sans-serif"])))
            out = figures.save_figure(fig, tmp_path / "greek", data={"x": [1, 2], "y": [1, 2]})
            plt.close(fig)
        assert face.family_name == "DejaVu Sans"
        for ch in "τµΔ":
            assert face.get_char_index(ord(ch)) != 0, f"{ch!r} has no glyph in {face.family_name}"
        assert not [w for w in caught if "missing from font" in str(w.message)], \
            [str(w.message) for w in caught]
        assert not [r for r in logs if "not found" in r.getMessage()], [r.getMessage() for r in logs]
        _assert_embedded_truetype(out["pdf"])
    finally:
        _fresh_font_caches()


@contextmanager
def _capture_font_logs():
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Keep(level=logging.WARNING)
    logger = logging.getLogger("matplotlib.font_manager")
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def test_font_family_list_is_the_stack_cut_to_installed_names_then_the_generic_then_dejavu(monkeypatch):
    fake = [font_manager.FontEntry(fname="a.ttf", name=n) for n in
            ("Arial", "Liberation Sans", "Times New Roman", "DejaVu Sans")]
    monkeypatch.setattr(font_manager.fontManager, "ttflist", fake)
    assert style.font_family_list() == ["Arial", "Liberation Sans", "sans-serif", "DejaVu Sans"]
    assert style.font_family_list("") == style.font_family_list(None)
    assert style.font_family_list("times new roman") == ["Times New Roman", "serif", "DejaVu Sans"]
    # a family nobody has: its stack is Arial's, with the generic and DejaVu Sans behind it
    assert style.font_family_list("Calibri") == ["Arial", "Liberation Sans", "sans-serif", "DejaVu Sans"]
    assert style.font_family_list("Courier New") == ["monospace", "DejaVu Sans"]
    # DejaVu Sans is cut like the rest: a list that names a face nobody has is a warning per text
    monkeypatch.setattr(font_manager.fontManager, "ttflist", fake[:2])
    assert style.font_family_list() == ["Arial", "Liberation Sans", "sans-serif"]


def test_font_family_list_finds_an_installed_family_whatever_the_case_it_was_stored_in(monkeypatch):
    """matplotlib compares family names without regard to case, so the list has to as well: a
    family stored as ``menlo`` (typed under Other font…, or saved on another system) is Menlo."""
    fake = [font_manager.FontEntry(fname="a.ttf", name=n) for n in ("Menlo", "Arial", "DejaVu Sans")]
    monkeypatch.setattr(font_manager.fontManager, "ttflist", fake)
    assert style.font_family_list("menlo") == ["menlo", "Arial", "sans-serif", "DejaVu Sans"]
    assert style.font_family_list("ARIAL") == style.font_family_list("Arial")


# ================================================================================================
# 3. the SVG font stack: the unit (a string in, a string out)
# ================================================================================================
def _svg(*elements: str) -> str:
    return ('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            + "".join(elements) + "</svg>")


def test_a_text_element_gets_the_stack_of_the_family_qt_named():
    out = pg_svg.set_font_stack(_svg(
        '<text x="1" font-family="Arial" font-size="12">a</text>',
        '<text x="2" font-family="&quot;Times New Roman&quot;, serif">b</text>',
        '<text x="3" font-family="Calibri">c</text>'))
    families = _svg_text_families(out)
    assert families[0] == ARIAL_STACK and families[0].startswith("Arial, Helvetica, ")
    assert families[1] == TIMES_STACK and families[1].endswith(", serif")
    # a family that is not installed anywhere still gets the stack, its own name first
    assert families[2].startswith("Calibri, Arial, Helvetica, ") and families[2].endswith(", sans-serif")
    assert _texts(out) == ["a", "b", "c"]


def _texts(svg_text: str) -> list[str]:
    return ["".join(e.itertext()) for e in ET.fromstring(svg_text).iter(f"{{{SVG_NS}}}text")]


@pytest.mark.parametrize("private", [".AppleSystemUIFont", ".SF NS Text", ".SFNS-Regular", ""])
def test_the_application_font_and_private_faces_are_written_as_arial(private):
    """Text drawn in the application's face -- on macOS the private ``.AppleSystemUIFont`` -- has
    no family a viewer has, and the SVG must not name it."""
    out = pg_svg.set_font_stack(_svg(f'<text font-family="{private}">t</text>'))
    assert _svg_text_families(out) == [ARIAL_STACK]
    assert private == "" or private not in out


@pytest.mark.parametrize("family", ["Segoe UI", "Noto Sans", "Sans Serif"])
def test_the_platform_default_family_is_named_when_a_figure_is_drawn_in_it(family, application_font):
    """On Windows the application's font is Segoe UI, and a person can pick it in Other font…: the
    screen, the PNG and the PDF are then drawn in it, so the SVG names it first (Arial's list
    behind it) -- it was written as Arial because the family was the application's own."""
    application_font(family)
    out = pg_svg.set_font_stack(_svg(f'<text font-family="{family}">t</text>',
                                     '<text font-family="Garamond">u</text>'))
    assert _svg_text_families(out) == [ff.css_font_family(family), ff.css_font_family("Garamond")]
    assert _svg_text_families(out)[0].split(",")[0].strip("'\" ") == family


def test_a_text_without_a_family_gets_the_default_stack_and_the_rest_is_left_alone():
    out = pg_svg.set_font_stack(_svg('<text x="1" y="2" font-size="9" fill="#123456">t</text>',
                                     '<text x="3"/>'))
    root = ET.fromstring(out)
    texts = list(root.iter(f"{{{SVG_NS}}}text"))
    assert [t.get("font-family") for t in texts] == [ARIAL_STACK, ARIAL_STACK]
    assert (texts[0].get("x"), texts[0].get("y"), texts[0].get("font-size"), texts[0].get("fill")) == \
        ("1", "2", "9", "#123456")


def test_the_groups_font_family_is_dropped_and_nothing_else_about_them():
    out = pg_svg.set_font_stack(_svg(
        '<g fill="none" font-family=".AppleSystemUIFont" font-size="16" font-weight="400">'
        '<text font-family="Arial">t</text><path d="M0,0 L1,1"/></g>'))
    root = ET.fromstring(out)
    group = next(root.iter(f"{{{SVG_NS}}}g"))
    assert group.get("font-family") is None
    assert (group.get("fill"), group.get("font-size"), group.get("font-weight")) == ("none", "16", "400")
    assert ".AppleSystemUIFont" not in out
    assert len(list(root.iter(f"{{{SVG_NS}}}path"))) == 1 and _texts(out) == ["t"]


def test_set_font_stack_is_idempotent_and_survives_awkward_names():
    once = pg_svg.set_font_stack(_svg(
        '<text font-family="Foo&quot;Bar">a</text>', '<text font-family="Fo&amp;o">b</text>',
        '<text font-family="Garamond">c</text>'))
    assert pg_svg.set_font_stack(once) == once
    families = _svg_text_families(once)            # still well-formed XML
    assert families[0] == ARIAL_STACK              # a name CSS cannot carry unescaped is not written
    assert families[1].startswith("Fo&o, Arial, ") # an ampersand round-trips through the escape
    assert families[2].startswith("Garamond, Arial, ")


def test_the_patch_is_installed_once_and_never_wraps_itself():
    import importlib

    module = importlib.import_module("pyqtgraph.exporters.SVGExporter")
    assert pg_svg.FONT_PATCH_STATUS in ("patched", "already-patched")
    before = module.generateSvg
    assert getattr(before, "_nanoscope_font_stack", False)
    assert pg_svg.install_font_stack() == "already-patched"
    assert module.generateSvg is before


def test_install_says_so_when_pyqtgraph_has_no_generateSvg(monkeypatch):
    import importlib

    module = importlib.import_module("pyqtgraph.exporters.SVGExporter")
    monkeypatch.delattr(module, "generateSvg")
    assert pg_svg.install_font_stack() == "unavailable"


# ================================================================================================
# 4. the SVG font stack: real pyqtgraph plots, written by the calls the app makes
# ================================================================================================
def _plot_with_text(qtbot, families=("Times New Roman", "Calibri")):
    """``(widget, plot item)`` -- the caller keeps the widget alive, or the item is deleted.

    Besides the axis labels, ticks and title the figure was never styled for, it carries one
    note per entry of ``families`` (``note 0``, ``note 1``...) drawn in exactly that family.
    """
    import pyqtgraph as pg

    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(520, 360)
    item = widget.getPlotItem()
    item.plot([0, 1, 2, 3], [1.0, 2.0, 1.5, 3.0])
    item.setLabel("bottom", "Dwell (ms)")
    item.setLabel("left", "Counts")
    item.setTitle("Dwell histogram")
    for i, family in enumerate(families):
        note = pg.TextItem(f"note {i}")
        font = QtGui.QFont()
        font.setFamily(family)
        note.setFont(font)
        note.setPos(0.5, 2.0 + i * 0.4)
        item.addItem(note)
    return widget, item


def _family_by_text(svg_text: str) -> dict[str, str]:
    """``{text: font-family}`` for every ``<text>`` element."""
    return dict(zip(_texts(svg_text), _svg_text_families(svg_text)))


def test_export_svg_names_the_stack_on_every_text_and_keeps_text_as_text(
        private_application_font, qtbot, tmp_path):
    widget, item = _plot_with_text(qtbot)
    text = pg_svg.export_svg(item, tmp_path / "stack.svg").read_text()
    by_text = _family_by_text(text)
    # text stays text: the labels are characters in <text> elements, not outlines
    joined = " ".join(by_text)
    assert "Dwell (ms)" in joined and "Counts" in joined and "Dwell histogram" in joined
    assert len(re.findall(r"<text\b", text)) == len(_texts(text)) == len(_svg_text_families(text))
    # a family that is installed: the stack of that family
    assert by_text["note 0"] == TIMES_STACK
    # one that is not installed anywhere: its own name first, the stack behind it
    assert by_text["note 1"] == ff.css_font_family("Calibri")
    assert by_text["note 1"].startswith("Calibri, Arial, Helvetica, ")
    # everything the figure was never styled for is the default (Arial's stack): the labels, the
    # title and the tick numbers, which Qt wrote in the application's own (private) face
    plain = {t: f for t, f in by_text.items() if not t.startswith("note")}
    assert {"Dwell (ms)", "Counts", "Dwell histogram", "0", "1", "2"} <= {t.strip() for t in plain}
    assert set(plain.values()) == {ARIAL_STACK}
    for family in by_text.values():
        assert family.endswith(("sans-serif", "serif", "monospace")), family


def test_the_svg_never_names_the_private_application_face(private_application_font, qtbot, tmp_path):
    """Whatever the QFont says -- here explicit private families on two notes, and (offscreen)
    Qt's own placeholder for everything the figure never styled -- the file carries none of it."""
    widget, item = _plot_with_text(qtbot, families=(".AppleSystemUIFont", ".SF NS Text"))
    text = pg_svg.export_svg(item, tmp_path / "private.svg").read_text()
    assert "AppleSystemUIFont" not in text and ".SF NS" not in text
    # no font-family anywhere (a text, or the group around it) but the default stack
    assert _all_font_families(text) == {ARIAL_STACK}


def test_the_frozen_snapshot_svg_the_export_dialog_writes_has_the_stack_too(
        private_application_font, qtbot):
    """``figure_bundle`` asks pyqtgraph for the SVG itself (``SVGExporter(...).export(toBytes=True)``),
    not through ``export_svg``: the Figure Library, the bundle and the Export dialog take this one."""
    from nanoscope.gui.figure_bundle import capture_figure_snapshot

    widget, item = _plot_with_text(qtbot)
    snapshot = capture_figure_snapshot(item, data=pd.DataFrame())
    by_text = _family_by_text(snapshot.svg_bytes.decode("utf-8"))
    assert by_text["note 0"] == TIMES_STACK
    assert by_text["note 1"] == ff.css_font_family("Calibri")
    assert {f for t, f in by_text.items() if not t.startswith("note")} == {ARIAL_STACK}


def test_a_snapshot_stored_before_this_exports_the_svg_text_it_was_captured_with(qapp, qtbot, tmp_path):
    """Only a figure captured now carries the lists. One already in a project or the Figure Library
    is written back byte for byte (``write_frozen_figure_image``: "Native PNG/SVG exports retain the
    captured bytes exactly"), so it keeps the single family Qt wrote for each text."""
    from nanoscope.gui.figure_bundle import capture_figure_snapshot, write_frozen_figure_image
    from nanoscope.gui.plot_export_options import resolve_export_options

    widget, item = _plot_with_text(qtbot, families=("Times New Roman",))
    fresh = capture_figure_snapshot(item, data=pd.DataFrame())
    options = resolve_export_options((fresh.width, fresh.height), extension=".svg")
    write_frozen_figure_image(tmp_path / "fresh.svg", fresh, options)
    assert TIMES_STACK in set(_svg_text_families((tmp_path / "fresh.svg").read_text()))
    old_bytes = re.sub(rb'font-family="[^"]*"', b'font-family="Times New Roman"', fresh.svg_bytes)
    stored = replace(fresh, svg_bytes=old_bytes)
    write_frozen_figure_image(tmp_path / "stored.svg", stored, options)
    assert (tmp_path / "stored.svg").read_bytes() == old_bytes
    assert set(_svg_text_families((tmp_path / "stored.svg").read_text())) == {"Times New Roman"}
    # at a physical size the file is rewritten (its width and height), the text is not
    sized = resolve_export_options((fresh.width, fresh.height), extension=".svg",
                                   native=False, width=85, height=60)
    write_frozen_figure_image(tmp_path / "sized.svg", stored, sized)
    assert ET.parse(tmp_path / "sized.svg").getroot().get("width") == "85mm"
    assert set(_svg_text_families((tmp_path / "sized.svg").read_text())) == {"Times New Roman"}


def test_pyqtgraphs_own_exporter_gets_the_stack_with_no_help_from_our_callers(qapp, qtbot):
    from pyqtgraph.exporters import SVGExporter

    widget, item = _plot_with_text(qtbot, families=("Times New Roman",))
    pg_svg.refresh_for_export(item)
    text = bytes(SVGExporter(item).export(toBytes=True)).decode("utf-8")
    assert _family_by_text(text)["note 0"] == TIMES_STACK


# ================================================================================================
# 5. a live page in the main window: PDF, SVG, and the Export current figure… route
# ================================================================================================
@pytest.fixture
def distributions(qapp, qtbot):
    """The Distributions page inside a real MainWindow's Analysis Explorer, with synthetic events."""
    from nanoscope.gui.main_window import MainWindow

    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window._open_plots()
    page = window._plots_window
    page.set_data({"apo": _events_for_page()})
    qtbot.wait(20)
    yield page
    window._analysis_controller.abort()


def _export_current_figure(page, monkeypatch, target, extension, **size):
    """Trigger the page's own Export ▾ > Export current figure… entry, with the two modal
    dialogs it opens answered the way a user would (this extension, this size, this file)."""
    from nanoscope.gui import plot_export
    from nanoscope.gui.plot_export_options import resolve_export_options

    monkeypatch.setattr(plot_export, "choose_export_options",
                        lambda parent, native_size, **kw: resolve_export_options(
                            native_size, extension=extension, **size))
    monkeypatch.setattr(plot_export, "ask_export_path", lambda *a, **k: str(target))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    button = page.findChild(QtWidgets.QToolButton, "populationPlotExportMenu")
    action = next(a for a in button.menu().actions() if a.text() == "Export current figure…")
    action.trigger()
    assert target.is_file() and target.stat().st_size > 0, f"{target.name} was not written"


def test_a_live_pages_svg_has_the_stack_by_every_route(distributions, monkeypatch, tmp_path):
    page = distributions
    assert page.save_image(str(tmp_path / "direct.svg")) is not None
    _export_current_figure(page, monkeypatch, tmp_path / "dialog.svg", ".svg")
    # at a physical size the staged file is parsed and written again: the stack must survive it
    _export_current_figure(page, monkeypatch, tmp_path / "sized.svg", ".svg",
                           native=False, width=85, height=60)
    assert ET.parse(tmp_path / "sized.svg").getroot().get("width") == "85mm"
    for name in ("direct.svg", "dialog.svg", "sized.svg"):
        text = (tmp_path / name).read_text()
        families = _svg_text_families(text)
        assert families, name
        assert set(families) == {ARIAL_STACK}, (name, set(families))
        assert _all_font_families(text) == {ARIAL_STACK}, name
        assert "Sans Serif" not in text and "AppleSystemUIFont" not in text


def test_a_live_pages_chosen_font_reaches_the_svg_as_its_stack(distributions, monkeypatch, tmp_path):
    page = distributions
    page.set_font_family("Times New Roman")
    page.save_image(str(tmp_path / "times.svg"))
    _export_current_figure(page, monkeypatch, tmp_path / "times_dialog.svg", ".svg")
    for name in ("times.svg", "times_dialog.svg"):
        families = set(_svg_text_families((tmp_path / name).read_text()))
        assert TIMES_STACK in families, (name, families)
    # a family that is not installed: still the stack behind it, so a viewer never picks its own
    page.set_font_family("Calibri")
    page.save_image(str(tmp_path / "calibri.svg"))
    families = set(_svg_text_families((tmp_path / "calibri.svg").read_text()))
    assert families
    for family in families:
        assert "Arial" in family and family.endswith("sans-serif"), family


def test_a_live_pages_pdf_embeds_truetype_and_no_type3(distributions, monkeypatch, tmp_path):
    """A guard, not a fix: the live-plot PDF (QPdfWriter) was already TrueType. Both routes."""
    page = distributions
    assert page.save_image(str(tmp_path / "direct.pdf")) is not None
    _export_current_figure(page, monkeypatch, tmp_path / "dialog.pdf", ".pdf")
    for name in ("direct.pdf", "dialog.pdf"):
        _assert_embedded_truetype(tmp_path / name)


def _synthetic_main_window(qtbot, tmp_path, monkeypatch):
    """A MainWindow holding one synthetic recording (an in-memory-style CSV under ``tmp_path``)
    with its own synthetic event table, and its Distributions page open on them."""
    from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
    from nanoscope.core.analysis import AnalysisResult
    from nanoscope.core.project import DocumentRef, ProjectState
    from nanoscope.gui.main_window import MainWindow

    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    path = tmp_path / "rec.csv"
    pd.DataFrame({"time": np.arange(20000) / 10000.0,
                  "current": np.full(20000, 100.0, dtype=np.float32)}).to_csv(path, index=False)
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    with qtbot.waitSignal(window.projectRestored, timeout=20000):
        window._begin_restore(ProjectState(documents=[DocumentRef(str(path))]))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        start = 100 + i * 150
        dwell = float(rng.exponential(0.002) + 1e-4)
        depth = float(rng.normal(40, 5))
        rows.append({"event_id": i, "recording": "r.csv", "pore_id": "P", "start_idx": start,
                     "end_idx": start + max(2, int(dwell * 10000)), "t_start_s": start / 10000.0,
                     "dwell_s": dwell, "baseline_pA": 100.0, "depth_pA": depth,
                     "frac_depth": depth / 100.0, "ecd_pA_s": depth * dwell, "max_dev_pA": depth,
                     "mean_dev_pA": depth, "rise_s": 1e-4, "fall_s": 1e-4, "snr": 10.0, "n_levels": 1,
                     "filter_limited": False, "accepted": True, "flag": ""})
    frame = pd.concat([empty_df(), pd.DataFrame(rows)], ignore_index=True)
    frame = frame.astype({name: spec[0] for name, spec in SCHEMA.items()})
    doc_id = window._session.documents()[0].doc_id
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=tuple(window._session.calibration(doc_id, 0).fingerprint()),
        payload=EventSet(frame, {})))
    window._on_selection(doc_id, 0)
    window._open_plots()
    return window


def test_a_saved_font_choice_still_exports_as_its_stack_after_the_project_is_reopened(
        qapp, qtbot, tmp_path, monkeypatch):
    """The project stores the family the user chose; the stack is derived when the file is
    written, so a reopened project exports the same SVG list as the session that saved it."""
    from nanoscope.gui.main_window import MainWindow

    window = _synthetic_main_window(qtbot, tmp_path, monkeypatch)
    window._plots_window.set_font_family("Times New Roman")
    before = window._plots_window.save_image(str(tmp_path / "before.svg"))
    assert TIMES_STACK in set(_svg_text_families(before.read_text()))

    project = tmp_path / "fonts.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False
    again = MainWindow(recovery_enabled=False)
    qtbot.addWidget(again)
    again.resize(1200, 900)
    again.show()
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(project))
    again._open_plots()
    for _ in range(5):
        QtWidgets.QApplication.processEvents()
    after = again._plots_window.save_image(str(tmp_path / "after.svg"))
    assert after is not None
    assert TIMES_STACK in set(_svg_text_families(after.read_text()))
    again._dirty = False


# ================================================================================================
# 6. Comparison figures
# ================================================================================================
def _texts_of(figure) -> list[Text]:
    return [t for t in figure.findobj(Text) if t.get_text().strip()]


def test_a_comparison_figure_names_the_stack_of_its_family_in_every_text():
    snapshot, tables = prepare("pairs", features=["dwell_s", "depth_pA"])
    for family, generic in (("Times New Roman", "serif"), ("Courier New", "monospace")):
        figure = render_comparison(snapshot, tables, {"font_family": family})
        texts = _texts_of(figure)
        assert texts
        for text in texts:
            names = text.get_fontfamily()
            assert len(names) >= 3 and names[-2:] == [generic, "DejaVu Sans"], (family, names)
            if _installed(family):
                assert names[0] == family
        figure.clear()


@needs_arial
def test_a_comparison_figure_with_no_family_is_arial_not_dejavu():
    snapshot, tables = prepare("facets")
    figure = render_comparison(snapshot, tables, {})
    for text in _texts_of(figure):
        assert text.get_fontfamily()[0] == "Arial", text.get_fontfamily()
    figure.clear()


@needs_arial
def test_comparison_text_that_the_chosen_font_lacks_is_drawn_from_the_next_face():
    """The axis labels are the user's words: Arial has no subscript one and no superscript minus,
    and the figure is drawn in the chosen family followed by its stand-ins (USER_GUIDE)."""
    snapshot, tables = prepare("facets", labels={"dwell_s": "k₁ (s⁻¹)", "depth_pA": "ΔI/I₀"})
    figure = render_comparison(snapshot, tables, {"font_family": "Arial"})
    FigureCanvasAgg(figure)
    assert any("k₁ (s⁻¹)" in text.get_text() for text in _texts_of(figure))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure.canvas.draw()
    assert not _missing_glyphs(caught), _missing_glyphs(caught)
    figure.clear()


def _comparison_page(qtbot):
    from nanoscope.core.sources import PlotSource
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow

    frame = pd.DataFrame({"event_id": [5, 5, 9, 11, 14, 17],
                          "dwell_s": [.001, .002, .003, .004, .005, .006],
                          "depth_pA": [10., 20., 30., 40., 50., 60.],
                          "amplitude_pA": [12., 25., 35., 42., 55., 64.],
                          "ecd_pA_s": [.01, .04, .09, .16, .25, .36], "snr": [5., 8., 10., 12., 15., 16.],
                          "accepted": [True, True, False, True, True, True]}, index=[2, 2, 4, 9, 10, 10])
    page = ComparisonFiguresWindow()
    qtbot.addWidget(page)
    page.resize(1100, 800)
    page.show()
    page.set_sources([PlotSource("first", "Recording A", frame, kind="manual", doc_key="first", current=True)])
    page.set_config({"kind": "facets"})
    assert page.analyze() and page.has_data()
    return page


@needs_arial
@needs_times
def test_the_comparison_page_draws_and_exports_in_the_chosen_family(qtbot, tmp_path):
    page = _comparison_page(qtbot)
    for state, expected in (({}, "Arial"), ({"font_family": "Times New Roman"}, "Times New Roman"),
                            ({"font_family": "Arial"}, "Arial")):
        with _fonts_drawn() as seen:
            page.set_appearance_state(state)             # re-renders the canvas item
        assert _first_faces(seen) == {expected}, (state, _first_faces(seen))
    # …and the three exports the page offers all work on that canvas
    for extension in (".svg", ".pdf", ".png"):
        assert page.save_image(str(tmp_path / f"comparison{extension}")) is not None
        assert (tmp_path / f"comparison{extension}").stat().st_size > 100


def test_comparison_figure_text_is_exported_as_outlines_in_svg_and_pdf(qtbot, tmp_path):
    """Documented as it is (USER_GUIDE, Comparison figures): the canvas item records text as
    glyph paths, so the SVG has no <text> and the PDF no font object. Nature asks for editable
    text; whether that changes is the owner's call, and this test is where it will say so."""
    page = _comparison_page(qtbot)
    svg = page.save_image(str(tmp_path / "outlines.svg"))
    pdf = page.save_image(str(tmp_path / "outlines.pdf"))
    assert b"<text" not in svg.read_bytes()
    fonts = _pdf_fonts(pdf)
    assert fonts["names"] == [] and fonts["type3"] == 0


@needs_arial
def test_a_family_that_is_not_installed_falls_back_quietly_on_the_comparison_page(qtbot):
    """Before: the bare name reached matplotlib, which logged 'findfont: Font family … not
    found' for it on every text it drew and used DejaVu Sans."""
    page = _comparison_page(qtbot)
    assert not _installed("Zapfino Imaginary")
    with _capture_font_logs() as logs, _fonts_drawn() as seen:
        page.set_appearance_state({"font_family": "Zapfino Imaginary"})
    assert _first_faces(seen) == {"Arial"}
    assert not [r for r in logs if "not found" in r.getMessage()], [r.getMessage() for r in logs]


def test_a_family_stored_in_another_case_is_still_the_family_drawn_on_the_comparison_page(qtbot):
    """DejaVu Sans Mono ships with matplotlib, so it is installed everywhere. The state holds it
    the way it was typed under Other font…, in lower case: cut as a different name it would be
    dropped, and the figure would be drawn in Arial without a word."""
    page = _comparison_page(qtbot)
    with _fonts_drawn() as seen:
        page.set_appearance_state({"font_family": "dejavu sans mono"})
    assert _first_faces(seen) == {"DejaVu Sans Mono"}


# ================================================================================================
# 7. the Study figure editor
# ================================================================================================
def _study_figure():
    figure = Figure(figsize=(6, 3), dpi=100, layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    left.plot([1., 2., 3.], [.3, .5, .2], "o-", color="#147d92", label="2 hours")
    left.set(xlabel="Dwell (ms)", ylabel="Current (pA)", title="Accepted events")
    left.legend()
    right.scatter([1., 2., 3.], [2., 1., 3.], color="#ca7627", label="30 hours")
    right.set(title="Other recording", xlabel="Time (s)", ylabel="Current (pA)")
    return figure


def test_the_study_style_puts_the_chosen_stack_in_every_text_and_arial_when_none_is_chosen():
    figure = _study_figure()
    styler = study_figure_style.StudyFigureStyle(figure)
    styler.apply(appearance={"font_family": "Times New Roman", "title": "A title"})
    for text in _texts_of(figure):
        names = text.get_fontfamily()
        assert names[-2:] == ["serif", "DejaVu Sans"] and len(names) >= 3, names
        if _installed("Times New Roman"):
            assert names[0] == "Times New Roman"
    styler.apply(appearance={"title": "A title"})
    for text in _texts_of(figure):
        names = text.get_fontfamily()
        assert names[-2:] == ["sans-serif", "DejaVu Sans"], names
        if _installed("Arial"):
            assert names[0] == "Arial", names
    figure.clear()


def test_a_saved_study_style_that_names_no_font_still_draws_in_arial_when_the_group_is_regenerated():
    """The hook the study figure code calls after drawing each figure applies the styling saved for
    THAT figure, and whatever else it holds (a title size, nothing at all) the figure is then drawn
    in Arial, not in the DejaVu Sans the study forces on the figures it draws itself. A sibling
    with no saved styling is left as the study drew it, so a regenerated study can hold both."""
    def drawn_by_the_study():
        with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
            figure = _study_figure()
            figure.canvas.draw()
        return figure

    sized, empty, plain = drawn_by_the_study(), drawn_by_the_study(), drawn_by_the_study()
    saved = {"sized": {"appearance": {"title_size": 12}, "panels": {}, "colors": {}},
             "empty": {"appearance": {}, "panels": {}, "colors": {}}}
    with study_figure_style.styled_figures(saved):
        for figure, name in ((sized, "sized.png"), (empty, "empty.png"), (plain, "plain.png")):
            assert study_figure_style.capture_saved_figure(figure, name) is False
    for figure in (sized, empty):
        assert {tuple(t.get_fontfamily()) for t in _texts_of(figure)} == {tuple(style.font_family_list())}
    assert {tuple(t.get_fontfamily()) for t in _texts_of(plain)} == {("DejaVu Sans",)}
    for figure in (sized, empty, plain):
        figure.clear()


@needs_arial
def test_study_text_that_the_chosen_font_lacks_is_drawn_from_the_next_face():
    """Arial has no subscript digits and no superscript minus: a title and a panel label that use
    them must not come out as boxes."""
    figure = _study_figure()
    styler = study_figure_style.StudyFigureStyle(figure)
    styler.apply(appearance={"font_family": "Arial", "title": "k₁ (s⁻¹)"},
                 panels={"0": {"title": "n₂", "y_label": "ΔI/I₀"}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure.canvas.draw()
    assert not _missing_glyphs(caught), _missing_glyphs(caught)
    figure.clear()


@needs_arial
def test_the_study_png_is_written_in_arial_for_a_figure_with_saved_styling_and_not_for_one_without(tmp_path):
    """The same regeneration, through the study's own save helper (``incubation_figures._save``,
    which calls the hook and then writes the PNG): the face the PNG is drawn in."""
    from nanoscope.plot import incubation_figures

    def drawn_by_the_study():
        with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
            figure = _study_figure()
            figure.canvas.draw()
        return figure

    saved = {"sized": {"appearance": {"title_size": 12}, "panels": {}, "colors": {}},
             "empty": {"appearance": {}, "panels": {}, "colors": {}}}
    faces = {}
    with study_figure_style.styled_figures(saved):
        for name in ("sized", "empty", "plain"):
            with _fonts_drawn_by_savefig() as seen:
                incubation_figures._save(drawn_by_the_study(), tmp_path, name)
            faces[name] = _first_faces(seen)
    assert faces == {"sized": {"Arial"}, "empty": {"Arial"}, "plain": {"DejaVu Sans"}}
    assert all((tmp_path / f"{name}.png").stat().st_size > 1000 for name in faces)


@pytest.fixture
def study_page(qtbot):
    from nanoscope.gui.study_figure_window import StudyFigureWindow

    frame = pd.DataFrame({"event_id": ["a", "b", "c"], "dwell_ms": [1., 2., 3.],
                          "accepted": [True, False, True]})
    page = StudyFigureWindow(_study_figure(), frame, {}, name="study_fonts",
                             provenance={"study": "fonts"})
    qtbot.addWidget(page)
    page.show()
    qtbot.wait(1)
    yield page
    if page._figure is not None:
        page._figure.clear()


@needs_arial
@needs_times
def test_the_study_editor_draws_and_exports_in_the_chosen_family(study_page, tmp_path):
    page = study_page
    widget = page._appearance_widget
    with _fonts_drawn() as seen:
        assert page.apply_appearance()
    assert _first_faces(seen) == {"Arial"}                  # no font chosen: Arial
    widget.apply_appearance(replace(widget.state(), font_family="Times New Roman"))
    assert page.style()["appearance"]["font_family"] == "Times New Roman"
    with _fonts_drawn() as seen:
        assert page.apply_appearance()
    assert _first_faces(seen) == {"Times New Roman"}
    for extension in (".svg", ".pdf", ".png"):
        out = tmp_path / f"study{extension}"
        assert page.export_image(path=out) == out, page._status.text()
    # documented (USER_GUIDE): the editor's SVG holds its text as outlines, like Comparison figures
    assert b"<text" not in (tmp_path / "study.svg").read_bytes()
    assert _pdf_fonts(tmp_path / "study.pdf")["names"] == []


@needs_arial
@needs_times
def test_reset_all_figure_styling_leaves_the_study_figure_in_arial(study_page):
    """Not in the font the study drew it in: with no font chosen the figure is Arial (USER_GUIDE)."""
    page = study_page
    widget = page._appearance_widget
    widget.apply_appearance(replace(widget.state(), font_family="Times New Roman"))
    with _fonts_drawn() as seen:
        assert page.apply_appearance()
    assert _first_faces(seen) == {"Times New Roman"}
    with _fonts_drawn() as seen:
        page.reset_style()
    assert page.style()["appearance"] == {}
    assert _first_faces(seen) == {"Arial"}


@needs_arial
def test_a_study_family_that_is_not_installed_falls_back_quietly(study_page):
    widget = study_page._appearance_widget
    widget.apply_appearance(replace(widget.state(), font_family="Zapfino Imaginary"))
    with _capture_font_logs() as logs, _fonts_drawn() as seen:
        assert study_page.apply_appearance()
    assert _first_faces(seen) == {"Arial"}
    assert not [r for r in logs if "not found" in r.getMessage()], [r.getMessage() for r in logs]
