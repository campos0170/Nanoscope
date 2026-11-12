"""Typed ``_{...}`` / ``^{...}`` words on the matplotlib pages: drawn, saved, reopened and exported.

The inline text editor of the Comparison figures, the study figure editor and the report figure
editor promises sub- and superscripts in its tooltip. What is pinned here is read off what the page
DRAWS and WRITES, not off a state field:

* the title, an axis label and a legend entry typed with the markup are drawn as scripts (the
  frozen SVG), and what is stored is exactly what was typed;
* a saved page opens with the same words and draws the same;
* the SVG export keeps the text as text, and so does the PDF: every glyph of a mathtext line is a
  text command of an embedded font, in reading order, not an outline (the PDF is read off its
  bytes: no PDF library is a dependency);
* the canvas item draws mathtext the way matplotlib's own renderer does: where its outlines were,
  and on the baseline a plain label beside it is on, at any size, resolution and export scale;
* a line with a script never draws a character as the currency sign, in the SVG, the PDF or the
  picture, also when Format gives the element its own font after the words were typed in the
  Title, X label or Y label box, and a figure title typed in the editors has no "automatic" text to
  go back to.

Synthetic data only: in-memory frames and figures, files under ``tmp_path``.
"""
import json
import re
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.gui import figure_bundle as B
from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.matplotlib_figure_item import MatplotlibFigureItem
from nanoscope.gui.plot_export import write_live_plot_image
from nanoscope.gui.plot_export_options import resolve_export_options
from nanoscope.gui.report_figure_window import open_report_figure
from nanoscope.gui.study_figure_window import StudyFigureWindow

from .test_comparison_figures_panel import source

SVG = "{http://www.w3.org/2000/svg}"
TYPED = "ΔI/I_{0} (%) at 10^{3} Hz"
SEEN = "ΔI/I0 (%) at 103 Hz"
MARKUP = re.compile(r"[_^{}]")


# ================================================================================================
# helpers: what a page draws and writes
# ================================================================================================
def _size(node) -> float:
    return float(re.search(r"font-size:\s*([\d.]+)px", node.get("style", "")).group(1))


def svg_lines(svg: bytes) -> list[list[tuple[str, float, float]]]:
    """For every Text of a frozen SVG, ``(characters, size, y)`` runs in reading order (a mathtext
    line is one tspan per glyph; matplotlib writes its spaces as no-break spaces)."""
    out = []
    for text in ET.fromstring(svg).iter(SVG + "text"):
        spans = text.findall(SVG + "tspan")
        if spans:
            out.append([((span.text or "").replace("\xa0", " "), _size(span), float(span.get("y")))
                        for span in sorted(spans, key=lambda s: float(s.get("x")))])
        elif text.text:
            out.append([(text.text, _size(text), float(text.get("y", 0)))])
    return out


def svg_words(svg: bytes) -> list[str]:
    return ["".join(chars for chars, _size_, _y in runs) for runs in svg_lines(svg)]


def pdf_glyph_text(path) -> str:
    """The text a PDF shows, glyph by glyph in drawing order, read off its bytes (no PDF library is
    a dependency): the page's text-showing operators, decoded through each font's ToUnicode map.
    (Qt's map reads the space glyph as a tab, in every text it writes; it is read as a space.)"""
    raw = Path(path).read_bytes()
    objects = {int(number): body for number, body in re.findall(rb"(?s)(\d+) 0 obj(.*?)endobj", raw)}

    def stream(body: bytes) -> bytes:
        found = re.search(rb"(?s)stream\r?\n(.*?)\r?\nendstream", body)
        if found is None:
            return b""
        return zlib.decompress(found.group(1)) if b"FlateDecode" in body[:found.start()] else found.group(1)

    def unicode_map(data: bytes) -> dict[int, str]:
        table = {}
        for block in re.findall(rb"(?s)beginbfchar(.*?)endbfchar", data):
            for code, value in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                table[int(code, 16)] = bytes.fromhex(value.decode()).decode("utf-16-be")
        for block in re.findall(rb"(?s)beginbfrange(.*?)endbfrange", data):
            for low, high, target in re.findall(
                    rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[.*?\]|<[0-9A-Fa-f]+>)", block):
                codes = range(int(low, 16), int(high, 16) + 1)
                if target.startswith(b"["):
                    for code, value in zip(codes, re.findall(rb"<([0-9A-Fa-f]+)>", target)):
                        table[code] = bytes.fromhex(value.decode()).decode("utf-16-be")
                else:
                    first = int(target.strip(b"<>"), 16)
                    for offset, code in enumerate(codes):
                        table[code] = chr(first + offset)
        return table

    fonts = {}
    for body in objects.values():
        resources = re.search(rb"/Font\s*<<(.*?)>>", body, re.S)
        if resources is None or b"/Type /Font" in body:
            continue
        for name, number in re.findall(rb"/(F\d+)\s+(\d+)\s+0\s+R", resources.group(1)):
            font = objects.get(int(number), b"")
            to_unicode = re.search(rb"/ToUnicode\s+(\d+)\s+0\s+R", font)
            fonts[name.decode()] = unicode_map(stream(objects[int(to_unicode.group(1))])) if to_unicode else {}
    shown, current = [], {}
    for body in objects.values():
        content = stream(body).decode("latin-1")
        if "BT" not in content:
            continue
        for token in re.finditer(r"/(F\d+)\s+[\d.\-]+\s+Tf|<([0-9A-Fa-f]+)>\s*Tj|\[(.*?)\]\s*TJ", content, re.S):
            if token.group(1):
                current = fonts.get(token.group(1), {})
                continue
            groups = [token.group(2)] if token.group(2) else re.findall(r"<([0-9A-Fa-f]+)>", token.group(3))
            shown.extend(current.get(int(group[i:i + 4], 16), "?") for group in groups
                         for i in range(0, len(group), 4))
    return "".join(shown).replace("\t", " ")


def comparison_page(qtbot, kind="events"):
    page = ComparisonFiguresWindow()
    qtbot.addWidget(page)
    page.resize(1100, 800)
    page.show()
    page.set_sources([source("a"), source("b", "Recording B")], selected_keys=["a", "b"])
    page.set_config({"kind": kind})
    assert page.analyze()
    page.fit_canvas()
    return page


def controllers(page):
    figure = next(c for c in page._text_controllers if c.panel == "figure")
    panel = next(c for c in page._text_controllers if c.panel != "figure")
    return figure, panel


def type_into(controller, key, typed_words):
    """Open the inline editor on ``key``, type, press Return: the user's own route."""
    editor = controller.edit_inline(key)
    assert editor is not None and editor.isVisible()
    assert "_{" in editor.toolTip() and "^{" in editor.toolTip()     # the promise it keeps
    editor.setText(typed_words)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)


def scripts(runs) -> tuple[str, str]:
    base = max(size for _chars, size, _y in runs)
    level = next(y for _chars, size, y in runs if size == base)
    return ("".join(c for c, size, y in runs if size < base and y > level + .5),
            "".join(c for c, size, y in runs if size < base and y < level - .5))


# ================================================================================================
# the Comparison page
# ================================================================================================
def test_comparison_title_axis_label_and_legend_entry_draw_their_scripts_and_store_the_typed_words(qtbot):
    page = comparison_page(qtbot)
    figure, panel = controllers(page)
    entry = next(key for key in figure.targets() if key.startswith("legend.entry."))
    type_into(figure, "title", TYPED)
    type_into(panel, "axis.left.label", "Dwell t_{d} (ms^{-1})")
    type_into(figure, entry, "Apo n^{2}")
    # what is stored is what was typed
    assert figure.target("title").style.text == TYPED
    assert panel.target("axis.left.label").style.text == "Dwell t_{d} (ms^{-1})"
    assert figure.target(entry).style.text == "Apo n^{2}"
    # what is drawn is the scripts, with none of the markup left
    svg = page._figure_item.native_figure_svg()
    shown = svg_words(svg)
    for expected in (SEEN, "Dwell td (ms−1)", "Apo n2"):
        assert expected in shown, (expected, shown)
    assert not [text for text in shown if MARKUP.search(text)], shown
    by_words = {"".join(c for c, _s, _y in runs): runs for runs in svg_lines(svg)}
    assert scripts(by_words[SEEN]) == ("0", "3")
    assert scripts(by_words["Dwell td (ms−1)"]) == ("d", "−1")
    assert scripts(by_words["Apo n2"]) == ("", "2")


def test_the_axis_properties_label_draws_its_scripts(qtbot):
    from dataclasses import replace
    page = comparison_page(qtbot)
    _figure, panel = controllers(page)
    state = panel.host.axis_state("bottom")
    panel.host.apply_axis("bottom", replace(state, label="Time t_{d} (s^{-1})"))
    assert panel.target("axis.bottom.label").style.text == "Time t_{d} (s^{-1})"
    assert "Time td (s−1)" in svg_words(page._figure_item.native_figure_svg())


def test_plain_words_with_dollars_and_percent_are_still_drawn_as_typed(qtbot):
    page = comparison_page(qtbot)
    figure, _panel = controllers(page)
    plain = "Costs $5 or $6 at 50% (a_b, x^2, {all}, C:\\dir)"
    type_into(figure, "title", plain)
    assert figure.target("title").style.text == plain
    assert plain in svg_words(page._figure_item.native_figure_svg())


def test_a_saved_comparison_reopens_with_the_typed_words_and_draws_the_same(qtbot):
    page = comparison_page(qtbot)
    figure, panel = controllers(page)
    type_into(figure, "title", TYPED)
    type_into(panel, "axis.left.label", "I^{2} (pA^{2})")
    before = svg_words(page._figure_item.native_figure_svg())
    payload, tables = page.snapshot_state()
    payload = json.loads(json.dumps(payload))            # through the project's JSON
    reopened = ComparisonFiguresWindow()
    qtbot.addWidget(reopened)
    reopened.restore_state(payload, tables)
    figure2, panel2 = controllers(reopened)
    assert figure2.target("title").style.text == TYPED
    assert panel2.target("axis.left.label").style.text == "I^{2} (pA^{2})"
    after = svg_words(reopened._figure_item.native_figure_svg())
    assert SEEN in after and "I2 (pA2)" in after and not [t for t in after if MARKUP.search(t)]
    assert sorted(after) == sorted(before)


def test_comparison_svg_export_keeps_the_scripts_as_text(qtbot, tmp_path):
    page = comparison_page(qtbot)
    figure, _panel = controllers(page)
    type_into(figure, "title", TYPED)
    target = tmp_path / "scripts.svg"
    write_live_plot_image(target, page._figure_item)
    assert SEEN in svg_words(target.read_bytes())
    glyphs = [node for node in ET.fromstring(target.read_bytes()).iter(SVG + "tspan")]
    assert {node.text for node in glyphs} >= {"Δ", "I", "0", "3", "H", "z"}
    for node in glyphs:     # every glyph names the figure's font, then its stand-ins
        assert "Arial" in node.get("font-family", "") and "sans-serif" in node.get("font-family", "")


def test_comparison_pdf_export_keeps_the_scripts_as_text(qtbot, tmp_path):
    page = comparison_page(qtbot)
    figure, panel = controllers(page)
    type_into(figure, "title", TYPED)
    type_into(panel, "axis.left.label", "I^{2} (pA^{2})")
    live = tmp_path / "live.pdf"
    write_live_plot_image(live, page._figure_item)
    snapshot = B.capture_figure_snapshot(page._figure_item, data=pd.DataFrame({"x": [1.]}), config={})
    frozen = tmp_path / "frozen.pdf"
    B.write_frozen_figure_image(frozen, snapshot, resolve_export_options(
        (snapshot.width, snapshot.height), extension="pdf", native=False, width=180, height=120))
    for path in (live, frozen):
        shown = pdf_glyph_text(path)
        assert SEEN in shown, (path.name, shown)
        assert "I2 (pA2)" in shown, (path.name, shown)
        payload = path.read_bytes()
        assert b"/FontFile2" in payload and b"/Subtype /Type3" not in payload


DEPTH = "Fractional depth ΔI/I₀ (s⁻¹, ℃) at 10^{3} Hz"
DEPTH_SEEN = "Fractional depth ΔI/I₀ (s⁻¹, ℃) at 103 Hz"


def test_a_line_with_a_script_keeps_every_character_the_first_face_lacks_in_the_svg_and_the_pdf(qtbot, tmp_path):
    """The app's own automatic labels carry ₀ and ⁻¹, and the inline editor opens filled with them:
    one script added to such a label must not turn those characters into the currency sign."""
    page = comparison_page(qtbot)
    _figure, panel = controllers(page)
    type_into(panel, "axis.left.label", DEPTH)
    assert panel.target("axis.left.label").style.text == DEPTH
    svg = svg_words(page._figure_item.native_figure_svg())
    assert DEPTH_SEEN in svg, svg
    live = tmp_path / "depth.pdf"
    write_live_plot_image(live, page._figure_item)
    snapshot = B.capture_figure_snapshot(page._figure_item, data=pd.DataFrame({"x": [1.]}), config={})
    frozen = tmp_path / "depth-frozen.pdf"
    B.write_frozen_figure_image(frozen, snapshot, resolve_export_options(
        (snapshot.width, snapshot.height), extension="pdf", native=False, width=180, height=120))
    assert "¤" not in "".join(svg)
    for path in (live, frozen):
        pdf = pdf_glyph_text(path)
        assert DEPTH_SEEN in pdf, (path.name, pdf)
        assert "¤" not in pdf


def test_words_with_a_character_no_font_has_are_drawn_as_typed_in_the_svg_and_the_pdf(qtbot, tmp_path):
    page = comparison_page(qtbot)
    _figure, panel = controllers(page)
    type_into(panel, "axis.left.label", "温度 T_{m}")
    assert panel.target("axis.left.label").style.text == "温度 T_{m}"
    svg = svg_words(page._figure_item.native_figure_svg())
    assert "温度 T_{m}" in svg, svg                       # as typed, the markup visible: no stand-in character
    target = tmp_path / "typed.pdf"
    write_live_plot_image(target, page._figure_item)
    assert "¤" not in "".join(svg) and "¤" not in pdf_glyph_text(target)


def test_the_label_boxes_keep_every_character_when_the_element_font_lacks_it_in_the_svg_and_the_pdf(qtbot, tmp_path):
    """Format's Title, X label and Y label boxes are typed first and the element's own font (the Format pane's
    font for that one text) is chosen after: a character that font lacks (cmss10, which matplotlib ships, has no
    Greek) must not become the currency sign, in the SVG, the PDF or the picture."""
    page = comparison_page(qtbot)
    boxes = page._appearance_widget
    typed = {"title": ("Δ I_{0} at 10^{3} Hz", "Δ I0 at 103 Hz"), "x_label": ("τ_{d} Δ (ms)", "τd Δ (ms)"),
             "y_label": ("Dwell τ_{off} at Δ (ms)", "Dwell τoff at Δ (ms)")}
    for name, (words, _seen) in typed.items():
        box = getattr(boxes, name)
        box.setText(words)
        box.textEdited.emit(words)              # the signal the box listens to
        assert getattr(boxes.state(), name) == words
    before = svg_words(page._figure_item.native_figure_svg())
    assert all(seen in before for _words, seen in typed.values()), before
    for owner, key in (("figure", "title"), ("panel", "axis.bottom.label"), ("panel", "axis.left.label")):
        figure, panel = controllers(page)       # (a font change builds the figure again, and its controllers)
        (figure if owner == "figure" else panel).set_style_property(key, "family", "cmss10")
    shown = svg_words(page._figure_item.native_figure_svg())
    assert all(seen in shown for _words, seen in typed.values()), shown
    assert "¤" not in "".join(shown), shown
    target = tmp_path / "poor.pdf"
    write_live_plot_image(target, page._figure_item)
    pdf = pdf_glyph_text(target)
    assert all(seen in pdf for _words, seen in typed.values()), pdf
    assert "¤" not in pdf, pdf
    figure, panel = controllers(page)
    assert panel.target("axis.left.label").style.family == "cmss10"
    assert boxes.state().y_label == typed["y_label"][0]         # what is stored is what was typed


# ================================================================================================
# the report and study figure editor
# ================================================================================================
def report_figure():
    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot([1., 2.], [3., 4.], label="Recording")
    axis.set_title("Generated figure")
    axis.set_xlabel("Time (s)")
    axis.legend()
    return figure


def open_editor(qtbot, kind, style=None):
    """The editor of a report figure (the route Analysis uses) or of a study figure."""
    figure, data = report_figure(), pd.DataFrame({"x": [1., 2.]})
    if kind == "report":
        owner = QtWidgets.QWidget()
        qtbot.addWidget(owner)
        return open_report_figure(owner, figure, data, name="amplitude", provenance={}, style=style)
    editor = StudyFigureWindow(figure, data, {}, name="amplitude", provenance={}, style=style, owner_kind="study")
    qtbot.addWidget(editor)
    editor.show()
    return editor


@pytest.mark.parametrize("kind", ["report", "study"])
def test_study_and_report_editors_draw_store_export_and_reopen_the_typed_scripts(qtbot, tmp_path, kind):
    editor = open_editor(qtbot, kind)
    controller = next(c for c in editor._text_controllers if c.panel != "figure")
    entry = next(key for key in controller.targets() if key.startswith("legend.entry."))
    type_into(controller, "title", TYPED)
    type_into(controller, "axis.bottom.label", "Dwell t_{d} (ms)")
    type_into(controller, entry, "Apo n^{2}")
    assert controller.target("title").style.text == TYPED
    shown = svg_words(editor._item.native_figure_svg())
    for expected in (SEEN, "Dwell td (ms)", "Apo n2"):
        assert expected in shown, (expected, shown)
    assert not [text for text in shown if MARKUP.search(text)], shown
    # the SVG and the PDF of the editor keep real text
    svg_target, pdf_target = tmp_path / "figure.svg", tmp_path / "figure.pdf"
    assert editor.export_image(path=svg_target) == svg_target and editor.export_image(path=pdf_target) == pdf_target
    assert SEEN in svg_words(svg_target.read_bytes())
    assert SEEN in pdf_glyph_text(pdf_target) and "Dwell td (ms)" in pdf_glyph_text(pdf_target)
    # saved with the figure, reopened on a fresh figure: the same words, the same drawing
    saved = json.loads(json.dumps(editor.style()))
    assert any(value.get("text") == TYPED for value in saved["appearance"]["text"].values())
    reopened = open_editor(qtbot, kind, style=saved)
    again = next(c for c in reopened._text_controllers if c.panel != "figure")
    assert again.target("title").style.text == TYPED
    assert SEEN in svg_words(reopened._item.native_figure_svg())
    editor.close()
    reopened.close()


@pytest.mark.parametrize("kind", ["report", "study"])
def test_a_figure_title_typed_in_the_editor_has_no_automatic_text(qtbot, kind):
    """The Format words box shows ``Automatic: <that text>`` as its placeholder: for a title the
    figure did not have there is none, and it is never the mathtext made from the typed words."""
    editor = open_editor(qtbot, kind)
    figure = next(c for c in editor._text_controllers if c.panel == "figure")
    assert figure.target("title").auto == ""
    type_into(figure, "title", TYPED)
    assert figure.target("title").style.text == TYPED
    assert SEEN in svg_words(editor._item.native_figure_svg())
    assert figure.target("title").auto == ""
    panel = next(c for c in editor._text_controllers if c.panel != "figure")
    assert panel.target("title").auto == "Generated figure"          # a title the figure had keeps its own
    editor.close()


# ================================================================================================
# the canvas item draws mathtext as text
# ================================================================================================
def attach(qtbot, item):
    view = QtWidgets.QGraphicsView()
    scene = QtWidgets.QGraphicsScene(view)
    scene.addItem(item)
    scene.setSceneRect(item.boundingRect())
    view.setScene(scene)
    view.setBackgroundBrush(QtGui.QColor("white"))
    view.resize(450, 300)
    qtbot.addWidget(view)
    view.show()
    return view, scene


def rendered(png: bytes) -> np.ndarray:
    image = QtGui.QImage.fromData(png).convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[:, :, :3] / 255.


def math_figure(title: str, ylabel: str = "", *, weight="normal", style="normal", rotation=90):
    figure = Figure(figsize=(4, 2.5), dpi=100, facecolor="white")
    axis = figure.add_axes((.22, .22, .72, .52))
    axis.plot([0, 1], [0, 1], linewidth=.75, color="black")
    axis.set_title(title, fontsize=14, fontweight=weight, fontstyle=style, family=["Arial", "DejaVu Sans"])
    axis.set_ylabel(ylabel, rotation=rotation, fontsize=12, family=["Arial", "DejaVu Sans"])
    return figure


@pytest.mark.parametrize("weight, style", [("normal", "normal"), ("bold", "normal"), ("normal", "italic")])
def test_mathtext_is_text_in_the_pdf_whatever_the_face(qtbot, tmp_path, weight, style):
    item = MatplotlibFigureItem(math_figure(r"ΔI/I$_{\mathregular{0}}$ (%) at 10$^{\mathregular{3}}$ Hz",
                                            weight=weight, style=style))
    view, scene = attach(qtbot, item)
    target = tmp_path / f"{weight}-{style}.pdf"
    write_live_plot_image(target, item)
    assert SEEN in pdf_glyph_text(target)
    metadata = item.capture_rendering_metadata()
    assert any(face["weight"] in (700, "bold") for face in metadata["retained_text_faces"]) == (weight == "bold")
    assert "mathematical text as glyph text commands" in metadata["picture_text_policy"]


def test_a_fraction_bar_is_not_text_and_keeps_its_outlines(qtbot, tmp_path):
    figure = Figure(figsize=(2, 1.5), dpi=100, facecolor="white")
    figure.text(.3, .4, r"$\frac{a}{b}$", fontsize=20)
    item = MatplotlibFigureItem(figure)
    view, scene = attach(qtbot, item)
    target = tmp_path / "fraction.pdf"
    write_live_plot_image(target, item)
    assert pdf_glyph_text(target) == ""
    png, _svg, _width, _height = B.capture_figure_render(item)
    assert (np.min(rendered(png), axis=2) < .5).sum() > 40


def ink_of(qtbot, title: str, ylabel: str, rotation: int) -> np.ndarray:
    """How dark each pixel of the canvas item's PNG is."""
    item = MatplotlibFigureItem(math_figure(title, ylabel, rotation=rotation))
    view, scene = attach(qtbot, item)
    png, _svg, _width, _height = B.capture_figure_render(item)
    view.close()
    return 1. - np.min(rendered(png), axis=2)


def agreement(one: np.ndarray, other: np.ndarray, region) -> float:
    """The correlation of two renderings' ink in a region, a little blurred."""
    from scipy.ndimage import gaussian_filter
    return float(np.corrcoef(gaussian_filter(one[region], .8).ravel(), gaussian_filter(other[region], .8).ravel())[0, 1])


def test_a_new_capture_of_mathtext_records_its_faces_and_replays_them_from_a_bundle(qtbot, tmp_path):
    import zipfile
    item = MatplotlibFigureItem(math_figure(r"$\mu$ ΔI/I$_{\mathregular{0}}$ $x^{2}_{i}$", r"$\tau_{off}$"))
    view, scene = attach(qtbot, item)
    snapshot = B.capture_figure_snapshot(item, data=pd.DataFrame({"x": [1.]}), config={})
    faces = snapshot.provenance()["nanoscope_capture_rendering"]["renderers"][0]["retained_text_faces"]
    assert any(face["style"] in ("italic", "oblique") for face in faces)     # mathtext's own italic letters
    bundle = B.write_figure_bundle(tmp_path / "math.zip", snapshot)
    with zipfile.ZipFile(bundle) as archive:
        metadata = json.loads(archive.read("manifest.json"))
        restored = B.FigureSnapshot.from_storage(
            metadata, {key: archive.read(path) for key, path in metadata["files"].items()})
    target = tmp_path / "math.pdf"
    B.write_frozen_figure_image(target, restored, resolve_export_options(
        (restored.width, restored.height), extension="pdf", native=False, width=160, height=100))
    shown = pdf_glyph_text(target)
    assert "μ ΔI/I0" in shown and "τoff" in shown.replace(" ", ""), shown


def test_a_capture_made_when_mathtext_was_outlines_still_bundles_and_exports(qtbot, monkeypatch, tmp_path):
    import zipfile
    from nanoscope.gui import matplotlib_figure_item as item_module
    with monkeypatch.context() as patch:          # the picture the app recorded before: outlines
        patch.setattr(item_module._PictureRenderer, "_math_glyphs", lambda self, text, prop: None)
        item = MatplotlibFigureItem(math_figure(r"ΔI/I$_{\mathregular{0}}$ (%) at 10$^{\mathregular{3}}$ Hz"))
    view, scene = attach(qtbot, item)
    snapshot = B.capture_figure_snapshot(item, data=pd.DataFrame({"x": [1.]}), config={})
    bundle = B.write_figure_bundle(tmp_path / "before.zip", snapshot)
    with zipfile.ZipFile(bundle) as archive:
        metadata = json.loads(archive.read("manifest.json"))
        restored = B.FigureSnapshot.from_storage(
            metadata, {key: archive.read(path) for key, path in metadata["files"].items()})
    for extension in ("pdf", "png"):
        target = tmp_path / f"before.{extension}"
        B.write_frozen_figure_image(target, restored, resolve_export_options(
            (restored.width, restored.height), extension=extension, native=False, width=160, height=100))
        assert target.stat().st_size > 100
    shown = pdf_glyph_text(tmp_path / "before.pdf")
    assert shown and SEEN not in shown          # the tick numbers are text; the title keeps its outlines


def test_a_figures_own_mathtext_annotation_box_draws_as_text_where_its_outlines_were(qtbot, monkeypatch, tmp_path):
    """The words the app itself writes (a fit's ``$t_{min}$ = 1.2 ms``), in a boxed, right-aligned,
    three-line annotation: some lines are mathtext and one is not."""
    from nanoscope.gui import matplotlib_figure_item as item_module

    def boxed():
        figure = Figure(figsize=(4, 2.5), dpi=100, facecolor="white")
        axis = figure.add_axes((.14, .16, .8, .76))
        axis.plot([0, 1], [0, 1], linewidth=.75, color="black")
        axis.text(.96, .94, "fit REJECTED\n" r"$t_{min}$ = 1.2 ms" "\n" r"$\mu_1$ = 3.4 pA", ha="right", va="top",
                  transform=axis.transAxes, fontsize=9, bbox={"facecolor": "white", "edgecolor": "#888888", "pad": 3})
        return figure

    def ink(pdf=None):
        item = MatplotlibFigureItem(boxed())
        view, scene = attach(qtbot, item)
        png, _svg, _width, _height = B.capture_figure_render(item)
        if pdf is not None:
            write_live_plot_image(pdf, item)
        view.close()
        return 1. - np.min(rendered(png), axis=2)

    as_text = ink(tmp_path / "boxed.pdf")
    with monkeypatch.context() as patch:
        patch.setattr(item_module._PictureRenderer, "_math_glyphs", lambda self, text, prop: None)
        as_outlines = ink()
    assert agreement(as_text, as_outlines, (slice(20, 110), slice(240, 395))) > .93
    assert agreement(as_text, as_outlines, (slice(None), slice(None))) > .98
    shown = pdf_glyph_text(tmp_path / "boxed.pdf").replace("ﬁ", "fi")     # Qt joins an "fi" of plain text
    assert "fit REJECTED" in shown and "tmin = 1.2 ms" in shown and "μ1 = 3.4 pA" in shown, shown


@pytest.mark.parametrize("rotation", [90, 0])
def test_mathtext_drawn_as_text_lands_where_its_outlines_were_drawn(qtbot, monkeypatch, rotation):
    from nanoscope.gui import matplotlib_figure_item as item_module
    title = r"$\mu$ ΔI/I$_{\mathregular{0}}$ $x^{2}_{i}$ at 10$^{\mathregular{3}}$ Hz"
    ylabel = r"$I^{2}$ ($\mu A^{2}$) $\tau_{off}$"
    # (The floors sit just under what a faithful layout reaches -- the label 0.985 and more, the whole
    # figure 0.987, the title, which is thin text, 0.965 -- and over what one pixel off reaches.)
    regions = {"title": ((slice(28, 64), slice(40, 360)), .955), "label": ((slice(40, 200), slice(0, 40)), .965),
               "figure": ((slice(None), slice(None)), .975)}
    as_text = ink_of(qtbot, title, ylabel, rotation)
    layout = item_module._PictureRenderer._math_glyphs
    with monkeypatch.context() as patch:        # what these glyphs were drawn as before: outlines
        patch.setattr(item_module._PictureRenderer, "_math_glyphs", lambda self, text, prop: None)
        as_outlines = ink_of(qtbot, title, ylabel, rotation)
    for name, (region, floor) in regions.items():
        assert agreement(as_text, as_outlines, region) > floor, name
    # The measure does see a one-pixel error: with every glyph moved one pixel up, or right, the
    # title no longer agrees.
    for dx, dy in ((0., -1.), (1., 0.)):
        with monkeypatch.context() as patch:
            patch.setattr(item_module._PictureRenderer, "_math_glyphs", lambda self, text, prop, dx=dx, dy=dy: [
                (font, char, x + dx, y + dy) for font, char, x, y in layout(self, text, prop)])
            moved = ink_of(qtbot, title, ylabel, rotation)
        for name in ("title", "label"):
            assert agreement(moved, as_outlines, regions[name][0]) < regions[name][1], (name, dx, dy)


# ================================================================================================
# where mathtext is drawn: as exact as its outlines were, on the baseline of the plain text beside it
# ================================================================================================
MATH_LINES = [r"$\mathdefault{10^{-3}}$", r"$t_{min}$ = 1.2 ms", r"ΔI/I$_{\mathregular{0}}$ (%) at 10$^{\mathregular{3}}$ Hz",
              r"$\mu$A", r"$x^{2}_{i}$", r"$\tau_{off}$", r"$\mathdefault{10^{1}}$", r"blockade depth $\Delta I / I_0$",
              r"$\mu_1$ = 3.4 pA"]


@pytest.mark.parametrize("dpi", [72, 100, 300, 600])
def test_mathtext_glyphs_are_laid_out_as_exactly_as_the_outlines_were_and_sit_on_the_baseline(qtbot, dpi):
    from matplotlib.font_manager import FontProperties
    from nanoscope.gui import matplotlib_figure_item as item_module
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    try:
        renderer = item_module._PictureRenderer(painter, 1600, 1200, dpi)
        text2path = renderer._text2path
        for size in (7, 9, 10, 12, 14):
            prop = FontProperties(family=["Arial", "DejaVu Sans"], size=size)
            unit = renderer.points_to_pixels(size) / text2path.FONT_SCALE      # figure pixels per unit of the layout
            for line in MATH_LINES:
                runs = renderer._math_glyphs(line, prop)
                outlines, _glyph_map, _rects = text2path.get_glyphs_mathtext(prop, line)
                assert runs is not None and len(runs) == len(outlines), line
                # the same layout, glyph for glyph: x exactly, y up to the one rounding mathtext makes
                for (_font, _char, dx, _dy), (_glyph, x, _y, _scale) in zip(runs, outlines):
                    assert dx == pytest.approx(x * unit, abs=1e-9), (line, size, dpi)
                raised = {round(y * unit + dy, 9) for (_f, _c, _dx, dy), (_g, _x, y, _s) in zip(runs, outlines)}
                assert len(raised) == 1 and 0. <= raised.pop() <= .0101 * renderer.points_to_pixels(size), (line, size, dpi)
                # the line's own-size glyphs stand ON the baseline (the outlines floated up to a hundredth of an em)
                body = max(font.pointSizeF() for font, *_rest in runs)
                assert all(abs(dy) < 1e-9 for font, _c, _dx, dy in runs if font.pointSizeF() == body), (line, size, dpi)
    finally:
        painter.end()


FIGURE_FACES = ("Arial", "DejaVu Sans")          # the app's face, and the one that ships with matplotlib
PAIR_CASES = [(8, 100, "normal", FIGURE_FACES), (9, 100, "normal", FIGURE_FACES), (12, 100, "normal", FIGURE_FACES),
              (12, 100, "bold", FIGURE_FACES), (10, 300, "normal", FIGURE_FACES), (10, 72, "normal", ("DejaVu Sans",)),
              (9, 100, "normal", ("DejaVu Sans",)), (14, 200, "normal", ("DejaVu Sans",))]


def window_ink(item, scale, left, top, width, height):
    """How dark each pixel of a window of the item's picture is, when it is replayed ``scale`` times larger as an
    export does; the window (figure pixels) starts on a whole device pixel, so it is what that part of the whole
    replay holds."""
    x0, y0 = int(left * scale), int(top * scale)
    image = QtGui.QImage(int(width * scale), int(height * scale), QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
    painter.translate(-x0, -y0)
    painter.scale(scale, scale)
    item.paint(painter, QtWidgets.QStyleOptionGraphicsItem())
    painter.end()
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    ink = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[:, :, :3]
    return 1. - ink.min(axis=2) / 255., y0


def glyph_centre(ink, y0):
    """Where, down the figure in device pixels, the ink of the one glyph in a window has its centre of mass: its
    whole shape counts (a stem alone moves with where the glyph falls across the pixel columns)."""
    profile = ink.sum(axis=1)
    assert profile.sum() > .02 * ink.shape[0] * ink.shape[1], "the window holds no glyph"
    return y0 + float((profile * (np.arange(len(profile)) + .5)).sum() / profile.sum())


def pair_gaps(size, dpi, scale, family=FIGURE_FACES, weight="normal"):
    """Ten pairs of labels, each pair on one baseline, the baselines a tenth of a pixel apart: a plain ``HIH`` at the
    left and a mathtext ``HIH`` with a subscript at the right. For each pair, how far lower the first letter of the
    mathtext label has its centre of mass than the first letter of the plain one (figure pixels; negative: the
    mathtext label is higher): the same letter in the same face on the same baseline."""
    pixels = size * dpi / 72.
    width, height = int(pixels * 20), int(pixels * 34)
    figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
    bases = [pixels * (2.2 + 3 * k) + k / 10. for k in range(10)]
    plain_x, math_x = pixels * .7, pixels * 11
    for base in bases:
        for x, words in ((plain_x, "HIH"), (math_x, r"HIH$_{\mathregular{0}}$")):
            figure.text(x / width, 1 - base / height, words, va="baseline", fontsize=size, family=list(family), fontweight=weight)
    item = MatplotlibFigureItem(figure)
    gaps = []
    for base in bases:
        centres = []
        for x in (plain_x, math_x):          # a window around the first letter: its whole width, not the next letter
            ink, y0 = window_ink(item, scale, x - .05 * pixels, base - 1.2 * pixels, .8 * pixels, 1.6 * pixels)
            centres.append(glyph_centre(ink, y0))
        gaps.append((centres[1] - centres[0]) / scale)
    return gaps


@pytest.mark.parametrize("scale", [1, 2, 6])
@pytest.mark.parametrize("size, dpi, weight, family", PAIR_CASES)
def test_a_plain_label_and_a_mathtext_label_on_one_baseline_stand_on_one_line(qtbot, size, dpi, weight, family, scale):
    """Qt puts a text's baseline on a whole pixel, so a mathtext glyph even a hair above its baseline lands a whole
    pixel above the plain label's: a 600 dpi export (six times the figure) shows six."""
    gaps = pair_gaps(size, dpi, scale, family, weight)
    assert max(abs(gap) for gap in gaps) < .05, [round(gap, 3) for gap in gaps]


@pytest.mark.parametrize("scale", [1, 6])
@pytest.mark.parametrize("up", [.5, -.5])
def test_the_pair_measure_sees_half_a_pixel_between_the_labels(qtbot, monkeypatch, scale, up):
    """The test above would pass vacuously if the measure could not see a mathtext label that is off its baseline."""
    from nanoscope.gui import matplotlib_figure_item as item_module
    layout = item_module._PictureRenderer._math_glyphs
    with monkeypatch.context() as patch:
        patch.setattr(item_module._PictureRenderer, "_math_glyphs", lambda self, text, prop: [
            (font, char, x, y - up) for font, char, x, y in layout(self, text, prop)])
        gaps = pair_gaps(10, 100, scale)
    seen = [abs(gap) for gap in gaps]
    # six times larger, half a figure pixel is three whole pixels of the replay on every baseline; at one times it is
    # rounded to a whole pixel, or to none, according to where the baseline falls
    assert (min(seen) if scale == 6 else max(seen)) > .4, [round(gap, 3) for gap in gaps]
