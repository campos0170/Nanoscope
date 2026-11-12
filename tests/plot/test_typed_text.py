"""Typed words on matplotlib figures: ``_{...}`` and ``^{...}`` draw as sub- and superscripts.

The text editors of the Comparison, study and report figures promise it in their tooltip, and the
pyqtgraph pages keep the promise. Here it is read off what the figure DRAWS (its SVG: one
``<text>`` per Text, one ``<tspan>`` per glyph of a mathtext line), never off a state field:

* a title, an axis label and a legend entry draw the script smaller and off the baseline, and
  nothing of the markup (``_``, ``^``, braces) is left on the figure;
* words with no markup are drawn exactly as typed, whatever mathtext would have made of them
  (``$``, ``%``, a lone ``_`` or ``^``, braces, a backslash);
* a dollar sign, a brace or a backslash typed BESIDE markup is still literal;
* a script mathtext cannot draw stays as typed, and nothing a user can type raises when drawn;
* a line that holds a script never draws a character as the currency sign ``¤`` (mathtext's stand-in
  for a character its first face lacks): each such character is drawn as itself, from the face
  mathtext can reach that has it, and when none has it the whole Text is drawn as typed; that holds
  for the face the Text ENDS UP in, also when Format gives it its own font after the words were typed;
* a script is set as text: a hyphen between two letters is a hyphen, a minus stays a minus, and a
  ``+``, ``,``, ``:``, ``<``, ``>`` or apostrophe has no mathematical spacing or meaning;
* the stored words stay as typed, the automatic text a Text had is never the converted string, and a
  restyle restores what the figure drew before.

Synthetic data only: in-memory frames and figures.
"""
import io
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import pytest
from matplotlib import cbook, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.mathtext import MathTextParser
from matplotlib.text import Text

from nanoscope.plot import comparison_figures as comparison
from nanoscope.plot.matplotlib_targets import (
    apply_text_overrides, artist_targets, bind_panel, panel_identity, remember_automatic_text, storage_key,
)
from nanoscope.plot.study_figure_style import StudyFigureStyle, bind_event_panel
from nanoscope.plot.style import font_family_list

from .test_comparison_figures import prepare

SVG = "{http://www.w3.org/2000/svg}"
TYPED = "ΔI/I_{0} (%) at 10^{3} Hz"
SEEN = "ΔI/I0 (%) at 103 Hz"
MARKUP = re.compile(r"[_^{}]")


# ================================================================================================
# helpers: what a figure draws
# ================================================================================================
def _size(node) -> float:
    return float(re.search(r"font-size:\s*([\d.]+)px", node.get("style", "")).group(1))


def drawn(figure) -> list[list[tuple[str, float, float, float]]]:
    """For every Text the figure's SVG draws: ``(characters, size, y, x)`` runs in reading order.

    A plain Text is one run. A mathtext line is one run per glyph, whose y is relative to the
    line's baseline (SVG's y grows downward): a subscript has a larger y than its neighbours.
    (matplotlib writes the spaces of a mathtext line as no-break spaces; they read as spaces.)
    """
    buffer = io.BytesIO()
    with rc_context({"svg.fonttype": "none", "svg.hashsalt": "typed-text"}):
        figure.savefig(buffer, format="svg")
    out = []
    for text in ET.fromstring(buffer.getvalue()).iter(SVG + "text"):
        spans = text.findall(SVG + "tspan")
        if spans:
            out.append([((span.text or "").replace("\xa0", " "), _size(span), float(span.get("y")), float(span.get("x")))
                        for span in sorted(spans, key=lambda s: float(s.get("x")))])
        elif text.text:
            out.append([(text.text, _size(text), float(text.get("y", 0)), float(text.get("x", 0)))])
    return out


def words(figure) -> list[str]:
    """What each Text of the figure reads, as one string."""
    return ["".join(run[0] for run in runs) for runs in drawn(figure)]


def line_of(figure, seen: str):
    """The runs of the one Text that reads ``seen``."""
    found = [runs for runs in drawn(figure) if "".join(run[0] for run in runs) == seen]
    assert len(found) == 1, f"{seen!r} is not drawn once: {words(figure)}"
    return found[0]


def scripts(runs) -> tuple[str, str]:
    """``(subscripts, superscripts)`` of a drawn line: the glyphs that are smaller than the text
    and lower, or higher, than its baseline."""
    base = max(run[1] for run in runs)
    level = next(run[2] for run in runs if run[1] == base)
    low = "".join(c for c, size, y, _x in runs if size < base and y > level + .5)
    high = "".join(c for c, size, y, _x in runs if size < base and y < level - .5)
    return low, high


def canvas():
    """A figure with one axes, a title, labels and a legend, bound the way the app binds one."""
    figure = Figure(figsize=(5, 3.5), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    bind_panel(axis, "panel-a", label="Panel A")
    axis.plot([1., 2., 3.], [3., 2., 4.], label="Recording")
    axis.set_title("Automatic title")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Current (pA)")
    legend = axis.legend()
    legend._nanoscope_source_keys = ["rec-a"]
    return figure, axis


def type_words(figure, by_key):
    """Type ``{element key: words}`` the way the Format editor stores them, and draw them."""
    apply_text_overrides(figure, {"text": {storage_key("panel-a", key): {"text": value}
                                           for key, value in by_key.items()}})


# ================================================================================================
# the defect: the markup was drawn as typed
# ================================================================================================
def test_title_axis_label_and_legend_entry_draw_their_scripts():
    figure, axis = canvas()
    type_words(figure, {"title": TYPED, "axis.bottom.label": "Dwell τ_{d} (ms^{-1})",
                        "axis.left.label": "I^{2} (pA^{2})", "legend.entry.rec-a": "Apo Fe_{3}^{+}"})
    seen = words(figure)
    for expected in (SEEN, "Dwell τd (ms−1)", "I2 (pA2)", "Apo Fe3+"):
        assert expected in seen, (expected, seen)
    assert not [text for text in seen if MARKUP.search(text)], seen
    # The scripts are smaller than the words around them and sit off its baseline.
    assert scripts(line_of(figure, SEEN)) == ("0", "3")
    assert scripts(line_of(figure, "Dwell τd (ms−1)")) == ("d", "−1")
    assert scripts(line_of(figure, "I2 (pA2)")) == ("", "22")
    assert scripts(line_of(figure, "Apo Fe3+")) == ("3", "+")
    runs = line_of(figure, SEEN)
    base = max(run[1] for run in runs)
    assert base == pytest.approx(axis.title.get_fontsize())         # the words keep the Text's own size
    assert [run[1] for run in runs if run[1] < base] == [pytest.approx(.7 * base)] * 2


def test_the_script_is_drawn_in_the_figures_own_face_weight_and_slant():
    figure, axis = canvas()
    axis.title.set_fontweight("bold")
    axis.title.set_fontstyle("italic")
    axis.title.set_fontfamily(["DejaVu Sans"])
    type_words(figure, {"title": "Gap I_{0}"})
    buffer = io.BytesIO()
    with rc_context({"svg.fonttype": "none"}):
        figure.savefig(buffer, format="svg")
    title = [span for span in ET.fromstring(buffer.getvalue()).iter(SVG + "tspan")
             if (span.text or "") in ("G", "a", "p", "I", "0")]
    assert {span.text for span in title} == {"G", "a", "p", "I", "0"}
    for span in title:
        style = span.get("style")
        # (DejaVu Sans has an oblique, not an italic, face.)
        assert "font-weight: 700" in style and re.search(r"font-style: (italic|oblique)", style), style
        assert "DejaVu Sans" in style, style


def test_legacy_title_and_labels_of_a_comparison_figure_draw_their_scripts():
    snapshot, tables = prepare("events")
    figure = comparison.render_comparison(snapshot, tables,
                                          {"title": TYPED, "x_label": "t_{d}", "y_label": "I^{2}"})
    seen = words(figure)
    for expected in (SEEN, "td", "I2"):
        assert expected in seen, (expected, seen)
    assert not [text for text in seen if MARKUP.search(text)], seen
    assert scripts(line_of(figure, "td")) == ("d", "")
    assert scripts(line_of(figure, "I2")) == ("", "2")


def test_format_words_of_a_comparison_figure_draw_their_scripts():
    snapshot, tables = prepare("events")
    plain = comparison.render_comparison(snapshot, tables)
    panel = panel_identity(plain.axes[0], 0)
    entry = next(key for owner, key, *_rest in artist_targets(plain)
                 if owner == "figure" and key.startswith("legend.entry."))
    state = {"text": {storage_key("figure", "title"): {"text": TYPED},
                      storage_key(panel, "title"): {"text": "Event I_{0}"},
                      storage_key(panel, "axis.left.label"): {"text": "Dwell t_{d} (ms)"},
                      entry: {"text": "Apo n^{2}"}}}
    figure = comparison.render_comparison(snapshot, tables, state)
    seen = words(figure)
    for expected in (SEEN, "Event I0", "Dwell td (ms)", "Apo n2"):
        assert expected in seen, (expected, seen)
    assert not [text for text in seen if MARKUP.search(text)], seen
    assert scripts(line_of(figure, "Apo n2")) == ("", "2")


def test_study_figure_style_draws_scripts_in_panel_overrides_and_the_title_then_restores():
    figure, axis = canvas()
    original = (axis.title.get_text(), axis.title.get_parse_math(), axis.xaxis.label.get_text(),
                axis.xaxis.label.get_parse_math())
    styler = StudyFigureStyle(figure)
    styler.apply(appearance={"title": TYPED}, panels={"0": {"title": "Panel I_{0}", "x_label": "t_{d}"}})
    seen = words(figure)
    for expected in (SEEN, "Panel I0", "td"):
        assert expected in seen, (expected, seen)
    assert not [text for text in seen if MARKUP.search(text)], seen
    styler.apply()
    assert (axis.title.get_text(), axis.title.get_parse_math(), axis.xaxis.label.get_text(),
            axis.xaxis.label.get_parse_math()) == original
    seen = words(figure)
    assert "Automatic title" in seen and not [text for text in seen if "Hz" in text]


# ================================================================================================
# words with no markup are exactly what they were
# ================================================================================================
PLAIN = ["Price $5 and $6", "$", "a_b", "x^2", "50% of {all}", r"C:\temp\_x", r"a\$b", "x_{0", "I_{0 open",
         "^{", "}{", "snake_case_name", "2^10", "a $b_{", "_ ^", "% $ _ ^ { } \\", "ΔI/I0 (%)", ""]


@pytest.mark.parametrize("typed_words", PLAIN)
def test_words_without_the_markup_are_drawn_exactly_as_typed(typed_words):
    figure, axis = canvas()
    type_words(figure, {"title": typed_words})
    assert axis.title.get_text() == typed_words and axis.title.get_parse_math() is False
    if typed_words:
        assert typed_words in words(figure)


@pytest.mark.parametrize("typed_words", PLAIN)
def test_words_without_the_markup_are_handed_over_as_typed_with_math_off(typed_words):
    from nanoscope.plot.typed_text import drawn_text, set_typed_text
    assert drawn_text(typed_words) == (typed_words, False)
    _figure, axis = canvas()
    set_typed_text(axis.title, typed_words)
    assert axis.title.get_text() == typed_words and axis.title.get_parse_math() is False


# ================================================================================================
# literal characters beside markup, and what a script may hold
# ================================================================================================
BESIDE = [
    ("Price $5 at I_{0}", "Price $5 at I0", ("0", "")),
    ("$_{0}$", "$0$", ("0", "")),
    ("50% x^{2} {all}", "50% x2 {all}", ("", "2")),
    ("a_b I_{0} c^d", "a_b I0 c^d", ("0", "")),
    ("{x} y_{1} {z}", "{x} y1 {z}", ("1", "")),
    (r"C:\dir I_{0}", r"C:\dir I0", ("0", "")),
    ("x\\_{0}", "x\\0", ("0", "")),                  # a backslash typed right before the markup
    ("x\\\\^{2}", "x\\\\2", ("", "2")),
    ("x^{-1}", "x−1", ("", "−1")),
    ("τ_{off} (ms^{-1})", "τoff (ms−1)", ("off", "−1")),
    ("A_{1}^{2}", "A12", ("1", "2")),
    ("é_{0}", "é0", ("0", "")),
    ("y_{%}", "y%", ("%", "")),
    ("y_{#}", "y#", ("#", "")),
    ("y_{$}", "y$", ("$", "")),
    ("y_{a_b}", "ya_b", ("a_b", "")),
    ("z_{\\}", "z\\", ("\\", "")),
    ("a_{b c}", "abc", ("bc", "")),
    ("n_{}x", "nx", ("", "")),
    ("^{3}He", "3He", ("", "3")),
]


@pytest.mark.parametrize("typed_words, seen, expected_scripts", BESIDE)
def test_dollars_braces_and_backslashes_beside_markup_stay_literal(typed_words, seen, expected_scripts):
    figure, _axis = canvas()
    type_words(figure, {"title": typed_words})
    assert seen in words(figure), (seen, words(figure))
    assert scripts(line_of(figure, seen)) == expected_scripts


UNDRAWABLE = ["x_{a^b}", 'x_{"}', "x_{`}", "x_{~}", "x^{^}"]


@pytest.mark.parametrize("typed_words", UNDRAWABLE)
def test_a_script_mathtext_cannot_draw_stays_as_typed(typed_words):
    figure, _axis = canvas()
    type_words(figure, {"title": typed_words})
    assert typed_words in words(figure)
    from nanoscope.plot.typed_text import drawn_text
    assert drawn_text(typed_words) == (typed_words, False)


def test_a_character_this_matplotlib_refuses_leaves_the_markup_as_typed():
    from nanoscope.plot.typed_text import drawn_text
    for typed_words in ("x_{a\tb}", "x_{\x00}", "x^{\U00020000}"):     # a control character, one beyond mathtext's range
        assert drawn_text(typed_words) == (typed_words, False)


def test_a_space_inside_a_script_is_a_gap_in_the_script():
    figure, _axis = canvas()
    type_words(figure, {"title": "a_{b c} a_{bc}"})
    runs = line_of(figure, "abc abc")
    base = max(run[1] for run in runs)
    scripted = [run for run in runs if run[1] < base]
    assert [run[0] for run in scripted] == ["b", "c", "b", "c"]
    spaced = scripted[1][3] - scripted[0][3]          # b to c, with a space between them
    closed = scripted[3][3] - scripted[2][3]          # b to c, without
    assert spaced > closed + 1.


def test_a_script_this_matplotlib_refuses_is_left_as_typed_and_the_others_still_draw():
    from nanoscope.plot.typed_text import drawn_text
    text, converted = drawn_text("I_{0} y^{a^b} t^{2}")
    assert converted is True
    assert text == r"I$_{\mathregular{0}}$ y^{a^b} t$^{\mathregular{2}}$"
    # A script that stays as typed puts its characters among the plain words, and one that no face can
    # draw there (a character beyond what mathtext reads) would be drawn as the currency sign: the
    # whole Text is drawn as typed.
    assert drawn_text("I_{0} y^{\U00020000} t^{2}") == ("I_{0} y^{\U00020000} t^{2}", False)


def test_one_undrawable_script_does_not_spoil_the_others():
    figure, _axis = canvas()
    type_words(figure, {"title": "I_{0} then y_{a^b} then t^{2}"})
    assert scripts(line_of(figure, "I0 then y_{a^b} then t2")) == ("0", "2")


def test_every_ascii_character_in_a_script_draws_or_stays_as_typed_and_never_raises():
    from nanoscope.plot.typed_text import drawn_text
    for kind in "_^":
        for code in range(32, 127):
            character = chr(code)
            if character in "{}":        # braces end a script: `x_{}}` is an empty script and a brace
                continue
            typed_words = f"x{kind}{{{character}}}"
            _text, converted = drawn_text(typed_words)
            figure, _axis = canvas()
            type_words(figure, {"title": typed_words})
            seen = words(figure)                      # an invalid span would raise while drawing
            if not converted:
                assert typed_words in seen, (typed_words, seen)
            elif character.isalnum():
                assert "x" + character in seen, (typed_words, seen)


TOKENS = ["a", "b", " ", "0", "τ", "_{x}", "^{2}", "_{ }", "^{-1}", "_{a b}", "_{}", "$", "\\", "%", "_", "^", "{", "}",
          "\n", "#", "'", "_{$}", "_{\\}", "^{%}", "_{#}", "_{a_b}", "_{{}", "^{a^b}", '_{"}', "_{~}", "x_{0}", "$_{1}$",
          "\\_{k}", "\\\\^{2}", "50%", "é", "ΔI/I", "_{'}", "^{*}", "_{(a)}", "^{+}", "_{é}", "_{τ_1}", "^{a,b}",
          "_{n−1}", "^{−1}", "_{a−b}", "_{a-b}"]


def reads(typed_words: str) -> list[str]:
    """What typed words read after the markup is applied, line by line: a drawable script is its
    characters (no space), a hyphen between two letters stays a hyphen and any other hyphen is a minus
    sign, and an apostrophe stays one."""
    from nanoscope.plot.typed_text import SUBSUP
    lines = []
    for line in typed_words.split("\n"):
        out, position = [], 0
        for found in SUBSUP.finditer(line):
            content = found.group(2)
            if any(c in '^"`~' or ord(c) < 32 or ord(c) == 127 for c in content):
                continue
            out.append(line[position:found.start()])
            position = found.end()
            content = content.replace(" ", "")
            out.append("".join("−" if c == "-" and not (0 < i < len(content) - 1 and content[i - 1].isalpha()
                                                        and content[i + 1].isalpha()) else c
                               for i, c in enumerate(content)))
        out.append(line[position:])
        lines.append("".join(out))
    return [line for line in lines if line]


def test_random_typed_words_never_raise_and_read_as_typed_less_the_markup():
    import random
    from nanoscope.plot.typed_text import drawn_text, set_typed_text
    rng = random.Random(20261111)
    checked = 0
    for _ in range(160):
        typed_words = "".join(rng.choice(TOKENS) for _ in range(rng.randint(1, 8))).strip()
        if not typed_words.strip():
            continue
        figure = Figure(figsize=(5, 3), dpi=100)
        FigureCanvasAgg(figure)
        set_typed_text(figure.text(.05, .5, ""), typed_words)
        assert words(figure) == reads(typed_words), (typed_words, drawn_text(typed_words))
        checked += 1
    assert checked > 100


def test_the_conversion_is_the_documented_mathtext_and_is_valid_for_this_matplotlib():
    from nanoscope.plot.typed_text import drawn_text
    text, converted = drawn_text(TYPED)
    assert converted is True
    assert text == r"ΔI/I$_{\mathregular{0}}$ (%) at 10$^{\mathregular{3}}$ Hz"
    assert cbook.is_math_text(text)
    assert drawn_text("Price $5 at I_{0}") == (r"Price \$5 at I$_{\mathregular{0}}$", True)
    # A backslash typed before a span must not escape the dollar sign that opens the span.
    assert drawn_text("x\\_{0}") == (r"x$\mathregular{{\backslash}}$$_{\mathregular{0}}$", True)


def test_lines_are_read_one_at_a_time():
    from nanoscope.plot.typed_text import drawn_text
    typed_words = "first I_{0}\nsecond costs $5 or $6\nthird"
    figure, _axis = canvas()
    type_words(figure, {"title": typed_words})
    seen = words(figure)
    for expected in ("first I0", "second costs $5 or $6", "third"):
        assert expected in seen, (expected, seen)
    text, converted = drawn_text(typed_words)
    assert converted is True
    assert [cbook.is_math_text(line) for line in text.split("\n")] == [True, False, False]
    assert text.split("\n")[1] == r"second costs \$5 or \$6"


def automatic_of(figure, panel, key):
    """The automatic text a target reports, as the Format pane reads it (after a draw)."""
    from nanoscope.plot.matplotlib_targets import capture_targets
    targets, _axes = capture_targets(figure, figure.canvas.get_renderer())
    return next(target.automatic for target in targets if target.panel == panel and target.key == key)


def test_the_stored_words_and_the_remembered_automatic_text_are_not_touched():
    figure, axis = canvas()
    remember_automatic_text(figure)
    state = {"text": {storage_key("panel-a", "title"): {"text": TYPED}}}
    apply_text_overrides(figure, state)
    assert state == {"text": {storage_key("panel-a", "title"): {"text": TYPED}}}
    assert axis.title._nanoscope_automatic_text == "Automatic title"
    assert SEEN in words(figure)
    assert automatic_of(figure, "panel-a", "title") == "Automatic title"


@pytest.mark.parametrize("route", ["study", "text overrides"])
def test_the_automatic_text_of_a_figure_title_the_figure_did_not_have_is_empty(route):
    """A title typed for a figure that had none: there is nothing automatic to go back to, and the
    Format words box must not show the converted mathtext string as its placeholder."""
    figure, _axis = canvas()
    assert figure._suptitle is None
    if route == "study":
        StudyFigureStyle(figure).apply(appearance={"title": TYPED})
    else:
        apply_text_overrides(figure, {"title": TYPED})
    assert SEEN in words(figure)
    assert figure._suptitle.get_text() != TYPED                  # (it is drawn as mathtext made from the words)
    assert automatic_of(figure, "figure", "title") == ""
    # the panel's own title keeps the automatic words it was drawn with
    assert automatic_of(figure, "panel-a", "title") == "Automatic title"


def test_a_text_typed_over_remembers_the_words_it_had_and_never_the_converted_string():
    from nanoscope.plot.typed_text import set_typed_text
    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    text = figure.text(.1, .5, "Automatic words")
    set_typed_text(text, TYPED)
    assert remember_automatic_text(figure) is None
    assert text._nanoscope_automatic_text == "Automatic words"
    set_typed_text(text, "Other I_{0}")                         # a second edit does not replace it
    assert text._nanoscope_automatic_text == "Automatic words"
    assert "Other" in words(figure)[0]


def test_the_pyqtgraph_pages_read_the_same_markup():
    from nanoscope.gui import figure_text
    from nanoscope.plot.typed_text import SUBSUP
    assert figure_text._SUBSUP is SUBSUP
    assert figure_text.rich_text(TYPED) == "ΔI/I<sub>0</sub> (%) at 10<sup>3</sup> Hz"


# ================================================================================================
# a character the first face lacks: drawn as itself, or the whole Text as typed (never the currency sign)
# ================================================================================================
POOR = ["cmss10"]        # ships with matplotlib: no Greek, no subscript zero, no superscript minus, no ℃
FACES = {"app": font_family_list(None), "poor": POOR}
LACKING = [
    ("Fractional depth ΔI/I₀ (s⁻¹, ℃) at 10^{3} Hz", "Fractional depth ΔI/I₀ (s⁻¹, ℃) at 103 Hz"),
    ("ΔI/I₀ ^{3}", "ΔI/I₀ 3"),
    ("℃ T_{m}", "℃ Tm"),
    ("x_{a₀} y^{s⁻¹}", "xa₀ ys⁻¹"),
    ("τ_{off} Δ^{2} (µs)", "τoff Δ2 (µs)"),
    ("Δ_{℃}", "Δ℃"),
    ("T_{m} 😀", "Tm 😀"),                      # no Arial or cmss10 glyph, but DejaVu Sans has one
    ("a\\₀ x_{1}", "a\\₀ x1"),                 # a backslash typed right before a character that is spelled for mathtext
]
NOWHERE = ["温度 T_{m}", "x_{温}", "T_{m} ก", "I_{0} y^{a^b} 温", "I_{0} 温度\nsecond line",
           "I_{0} 温度\nsecond y_{1}"]      # no face has them (and the markup of the other line of the label is shown, too)


def in_face(figure, family):
    """Put every Text of the figure in ``family`` -- before the words are typed, as the app does."""
    for text in figure.findobj(Text):
        text.set_fontfamily(family)


def lacks(family, char) -> bool:
    """Whether the first face matplotlib finds for ``family`` has no glyph for ``char``."""
    return not get_font(findfont(FontProperties(family=family))).get_char_index(ord(char))


def glyph_styles(figure, seen):
    """``[(character, {css property: value})]`` of the one mathtext Text of the SVG that reads ``seen``."""
    buffer = io.BytesIO()
    with rc_context({"svg.fonttype": "none", "svg.hashsalt": "typed-text"}):
        figure.savefig(buffer, format="svg")
    found = []
    for text in ET.fromstring(buffer.getvalue()).iter(SVG + "text"):
        spans = sorted(text.findall(SVG + "tspan"), key=lambda span: float(span.get("x")))
        glyphs = [((span.text or "").replace("\xa0", " "),
                   {key.strip(): value.strip().strip("'\"") for key, _, value in
                    (part.partition(":") for part in span.get("style", "").split(";")) if key.strip()})
                  for span in spans]
        if "".join(char for char, _style in glyphs) == seen:
            found.append(glyphs)
    assert len(found) == 1, f"{seen!r} is not drawn once"
    return found[0]


@pytest.mark.parametrize("face", sorted(FACES))
@pytest.mark.parametrize("typed_words, seen", LACKING)
def test_a_character_the_first_face_lacks_is_drawn_as_itself_never_as_the_currency_sign(face, typed_words, seen):
    figure, _axis = canvas()
    in_face(figure, FACES[face])
    type_words(figure, {"title": typed_words})
    shown = words(figure)
    assert seen in shown, (seen, shown)
    assert "¤" not in "".join(shown), shown


@pytest.mark.parametrize("face", sorted(FACES))
def test_a_currency_sign_the_user_typed_is_drawn_and_is_not_mistaken_for_a_stand_in(face):
    figure, axis = canvas()
    in_face(figure, FACES[face])
    type_words(figure, {"title": "¤ x_{a} ¤₀"})
    assert axis.title.get_parse_math() is True
    assert "¤ xa ¤₀" in words(figure), words(figure)


def test_the_poor_face_lacks_what_the_runs_above_need_it_to_lack_on_every_machine():
    assert all(lacks(POOR, char) for char in "Δτ₀⁻℃")
    assert not any(lacks(POOR, char) for char in "I/a(), 1µ²")


@pytest.mark.parametrize("face", sorted(FACES))
def test_a_character_the_first_face_lacks_comes_from_dejavu_sans_not_a_serif_stand_in(face):
    figure, _axis = canvas()
    in_face(figure, FACES[face])
    type_words(figure, {"title": "ΔI/I₀ (s⁻¹, ℃) at 10^{3} Hz"})
    glyphs = glyph_styles(figure, "ΔI/I₀ (s⁻¹, ℃) at 103 Hz")
    first = get_font(findfont(FontProperties(family=FACES[face]))).family_name
    for char, style in glyphs:
        # what the first face has it draws, as the same words without markup are; the rest is DejaVu Sans
        assert style["font-family"] == ("DejaVu Sans" if lacks(FACES[face], char) else first), (char, style)


def test_a_character_the_face_lacks_keeps_the_weight_of_the_text():
    figure, axis = canvas()
    in_face(figure, ["DejaVu Sans Mono"])                # (has no ℃; DejaVu Sans, in every weight and slant, does)
    axis.title.set_fontweight("bold")
    type_words(figure, {"title": "℃ T_{m}"})
    glyphs = dict(glyph_styles(figure, "℃ Tm"))
    assert glyphs["℃"]["font-family"] == "DejaVu Sans" and glyphs["℃"]["font-weight"] == "700"
    assert glyphs["T"]["font-family"] == "DejaVu Sans Mono" and glyphs["T"]["font-weight"] == "700"


@pytest.mark.parametrize("typed_words", NOWHERE)
@pytest.mark.parametrize("face", sorted(FACES))
def test_words_with_a_character_no_face_can_draw_are_drawn_as_typed_with_the_markup_visible(face, typed_words):
    figure, axis = canvas()
    in_face(figure, FACES[face])
    type_words(figure, {"title": typed_words})
    assert axis.title.get_text() == typed_words and axis.title.get_parse_math() is False
    shown = words(figure)
    for line in typed_words.split("\n"):
        assert line in shown, (line, shown)
    assert "¤" not in "".join(shown), shown


@pytest.mark.parametrize("face", sorted(FACES))
def test_a_line_of_its_own_with_no_script_is_drawn_as_typed_beside_a_line_that_has_one(face):
    """Only a line that mathtext lays out is at risk: a line with no script is drawn the way it always was."""
    figure, axis = canvas()
    in_face(figure, FACES[face])
    type_words(figure, {"title": "I_{0}\n温度 ΔI/I₀"})
    assert axis.title.get_parse_math() is True
    shown = words(figure)
    assert "I0" in shown and "温度 ΔI/I₀" in shown, shown
    assert "¤" not in "".join(shown)


def test_the_decision_is_made_from_the_face_the_text_is_drawn_in():
    from nanoscope.plot.typed_text import drawn_text
    typed_words = "ΔI/I₀ at 10^{3} Hz"
    # DejaVu Sans has every character of it: nothing is re-spelled, the first face draws them all.
    assert drawn_text(typed_words, FontProperties(family=["DejaVu Sans"])) == (
        r"ΔI/I₀ at 10$^{\mathregular{3}}$ Hz", True)
    # The face matplotlib ships as cmss10 has neither the capital delta nor the subscript zero.
    assert drawn_text(typed_words, FontProperties(family=POOR)) == (
        r"$\mathrm{Δ}$I/I$\mathrm{₀}$ at 10$^{\mathregular{3}}$ Hz", True)
    # Inside a script too, and neighbouring characters the face lacks share one span.
    assert drawn_text("x_{a₀₁}b⁻¹", FontProperties(family=POOR)) == (
        r"x$_{\mathregular{a\mathrm{₀₁}}}$b$\mathrm{⁻}$¹", True)


@pytest.mark.parametrize("keywords, command", [({}, r"\mathrm"), ({"weight": "bold"}, r"\mathbf"),
                                               ({"style": "italic"}, r"\mathit"),
                                               ({"weight": "bold", "style": "italic"}, r"\mathbfit")])
def test_a_character_the_face_lacks_is_spelled_in_the_mathtext_font_of_its_weight_and_slant(keywords, command):
    from nanoscope.plot.typed_text import drawn_text
    prop = FontProperties(family=["DejaVu Sans Mono"], **keywords)         # no ℃ in any of its four faces
    assert drawn_text("℃ T_{m}", prop) == (f"${command}{{℃}}$ T$_{{\\mathregular{{m}}}}$", True)


def test_a_line_without_a_script_is_never_re_spelled_whatever_the_face_lacks():
    from nanoscope.plot.typed_text import drawn_text
    for family in (POOR, ["DejaVu Sans"]):
        assert drawn_text("ΔI/I₀ (s⁻¹, ℃) 温度", FontProperties(family=family)) == ("ΔI/I₀ (s⁻¹, ℃) 温度", False)
        # ... nor a line of its own beside a line that has one
        assert drawn_text("I_{0}\nΔI/I₀ costs $5", FontProperties(family=family))[0].endswith(
            "\nΔI/I₀ costs \\$5")


# ================================================================================================
# the element's own font is applied AFTER the words were typed: they are spelled for the face it ends up in
# ================================================================================================
POOR_NAME = "cmss10"           # as Format's family box names it (font_family_list() makes the list of it)
TYPED_IN = "DejaVu Sans"        # the figure's font when the words are typed: ships with matplotlib, has every character below
GREEK = "Dwell τ_{off} at Δ (ms)"
GREEK_SEEN = "Dwell τoff at Δ (ms)"


def comparison_panel():
    """The events page of a Comparison figure and the identity of its one panel, as Format keys it."""
    snapshot, tables = prepare("events")
    return snapshot, tables, panel_identity(comparison.render_comparison(snapshot, tables).axes[0], 0)


def test_the_comparison_label_boxes_are_spelled_for_the_font_the_element_ends_up_with():
    """The Title, X label and Y label boxes of a Comparison page type their words when the figure is built; Format's
    per-element font comes after. Spelled for the first face the Text had, a character the element's own font lacks
    (the Greek in a face with none) was drawn as the currency sign."""
    snapshot, tables, panel = comparison_panel()
    state = {"font_family": TYPED_IN, "title": "Δ I_{0} at 10^{3} Hz", "x_label": "τ_{d} Δ (ms)", "y_label": GREEK,
             "text": {storage_key("figure", "title"): {"family": POOR_NAME},
                      storage_key(panel, "axis.bottom.label"): {"family": POOR_NAME},
                      storage_key(panel, "axis.left.label"): {"family": POOR_NAME}}}
    figure = comparison.render_comparison(snapshot, tables, state)
    shown = words(figure)
    for expected in ("Δ I0 at 103 Hz", "τd Δ (ms)", GREEK_SEEN):
        assert expected in shown, (expected, shown)
    assert "¤" not in "".join(shown), shown
    # each character from the element's own face where it has it, from DejaVu Sans where it does not
    for char, style in glyph_styles(figure, GREEK_SEEN):
        assert style["font-family"] == ("DejaVu Sans" if lacks(POOR, char) else POOR_NAME), (char, style)


def test_the_comparison_label_boxes_follow_the_weight_the_element_ends_up_with():
    """No face of DejaVu Sans Mono has ℃: it comes from DejaVu Sans, in the weight the element ends up in."""
    snapshot, tables, panel = comparison_panel()
    state = {"font_family": TYPED_IN, "y_label": "℃ T_{m}",
             "text": {storage_key(panel, "axis.left.label"): {"family": "DejaVu Sans Mono", "bold": True}}}
    glyphs = dict(glyph_styles(comparison.render_comparison(snapshot, tables, state), "℃ Tm"))
    assert glyphs["℃"]["font-family"] == "DejaVu Sans" and glyphs["℃"]["font-weight"] == "700", glyphs["℃"]
    assert glyphs["T"]["font-family"] == "DejaVu Sans Mono" and glyphs["T"]["font-weight"] == "700", glyphs["T"]


def test_words_no_face_can_draw_stay_as_typed_whatever_font_the_element_ends_up_with():
    snapshot, tables, panel = comparison_panel()
    state = {"font_family": TYPED_IN, "y_label": "温度 T_{m}", "text": {storage_key(panel, "axis.left.label"): {"family": POOR_NAME}}}
    shown = words(comparison.render_comparison(snapshot, tables, state))
    assert "温度 T_{m}" in shown and "¤" not in "".join(shown), shown


def test_the_study_panel_labels_and_gallery_event_titles_are_spelled_for_the_font_the_element_ends_up_with():
    """The study and report editors' older per-panel words (a panel's title and axis labels, an event gallery's event
    titles) are typed before the element's font, too."""
    figure, axis = canvas()
    bind_event_panel(axis, tag="study", event_id=7, start_idx=70, end_idx=71)          # a gallery panel
    identity = axis._nanoscope_event_panel_identity
    panel = panel_identity(axis, 0)
    poor = {"family": POOR_NAME}
    styler = StudyFigureStyle(figure)
    styler.apply(appearance={"font_family": TYPED_IN,
                             "text": {storage_key(panel, "title"): poor, storage_key(panel, "axis.bottom.label"): poor,
                                      storage_key(panel, "axis.left.label"): poor}},
                 panels={"0": {"x_label": "τ_{d} Δ (ms)", "y_label": GREEK}}, event_titles={identity: "Event Δ_{0} τ"})
    shown = words(figure)
    for expected in ("τd Δ (ms)", GREEK_SEEN, "Event Δ0 τ"):
        assert expected in shown, (expected, shown)
    assert "¤" not in "".join(shown), shown
    # restyled with no typed words at all, the labels the figure had come back: nothing is typed over them again
    styler.apply(appearance={"text": {storage_key(panel, "title"): poor, storage_key(panel, "axis.bottom.label"): poor}})
    shown = words(figure)
    assert "Automatic title" in shown and "Time (s)" in shown, shown
    assert not [text for text in shown if "Δ" in text or "τ" in text], shown


def test_the_study_title_typed_in_the_editor_is_spelled_for_the_font_the_element_ends_up_with():
    figure, _axis = canvas()
    poor = {"family": POOR_NAME}
    StudyFigureStyle(figure).apply(appearance={"font_family": TYPED_IN, "title": "Δ I_{0} τ",
                                               "text": {storage_key("figure", "title"): poor}})
    shown = words(figure)
    assert "Δ I0 τ" in shown and "¤" not in "".join(shown), shown


FACE_CHANGES = [[TYPED_IN], POOR, ["DejaVu Sans Mono"], font_family_list(None)]


@pytest.mark.parametrize("typed_words", [typed for typed, _seen in LACKING] + NOWHERE + [GREEK, TYPED, "I_{open-pore}"])
def test_words_typed_again_for_a_new_face_are_what_typing_them_in_that_face_makes(typed_words):
    from nanoscope.plot.typed_text import retype_text, set_typed_text
    for before in FACE_CHANGES:
        for after in FACE_CHANGES:
            figure = Figure(figsize=(4, 3), dpi=100)
            moved = figure.text(.1, .5, "Automatic", family=before)
            direct = figure.text(.1, .3, "Automatic", family=after)
            set_typed_text(moved, typed_words)
            set_typed_text(direct, typed_words)
            moved.set_fontfamily(after)
            retype_text(moved)
            assert (moved.get_text(), moved.get_parse_math()) == (direct.get_text(), direct.get_parse_math()), (before, after)
            assert "¤" not in "".join(words(figure)), (before, after, words(figure))


def test_words_typed_earlier_are_typed_again_for_the_new_face_only_while_the_text_still_shows_them():
    from nanoscope.plot.typed_text import retype_text, set_typed_text
    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    text = figure.text(.1, .5, "Automatic words")
    in_face(figure, [TYPED_IN])
    set_typed_text(text, "Δ I_{0} τ")
    assert "\\mathrm" not in text.get_text()                         # its face draws the Greek itself
    text.set_fontfamily(POOR)                                       # the element's font, applied afterwards
    retype_text(text)
    assert "\\mathrm{Δ}" in text.get_text() and "\\mathrm{τ}" in text.get_text() and text.get_parse_math() is True
    assert text._nanoscope_automatic_text == "Automatic words"        # the words it had are still the automatic ones
    # a Text that was given anything else since (a restyle puts its own words back) is left as it is
    text.set_text("Other words")
    text.set_parse_math(False)
    retype_text(text)
    assert text.get_text() == "Other words" and text.get_parse_math() is False
    retype_text(figure.text(.1, .2, "Never typed"))                 # and one that was never typed over is no business of its own


# ================================================================================================
# a script is set as text (hyphen, minus, plus, comma, colon, comparison signs, apostrophe)
# ================================================================================================
def mathtext_gaps(text) -> list[tuple[str, float]]:
    """``(character, gap)`` of each glyph mathtext lays out for a converted Text: the space it leaves
    between the glyph and the one before it, beyond that one's own advance (kerning is a hair)."""
    glyphs = MathTextParser("path").parse(text.get_text(), 72, text.get_fontproperties())[3]
    out, edge = [], None
    for font, size, code, x, _y in glyphs:
        font.set_size(size, 72)
        out.append((chr(code), 0. if edge is None else x - edge))
        edge = x + font.load_char(code).linearHoriAdvance / 65536
    return out


HYPHENS = [("I_{open-pore}", "Iopen-pore", ("open-pore", "")),
           ("t_{on-off} vs t_{a-b-c}", "ton-off vs ta-b-c", ("on-offa-b-c", "")),
           ("x^{a-b}", "xa-b", ("", "a-b")),
           ("τ_{on-off}", "τon-off", ("on-off", ""))]
MINUS = [("x^{-1}", "x−1", ("", "−1")), ("10^{-3}", "10−3", ("", "−3")), ("x_{i-1}", "xi−1", ("i−1", "")),
         ("x_{1-i}", "x1−i", ("1−i", "")),
         # a minus sign TYPED (or pasted, or filled in from the app's own labels, which carry it) is the same
         # sign as the keyboard's hyphen-minus before a digit: set against its neighbours, not spaced as an operator
         ("x_{n−1}", "xn−1", ("n−1", "")), ("x^{−3}", "x−3", ("", "−3")), ("x_{a−b}", "xa−b", ("a−b", "")),
         ("10^{−3} s^{−1}", "10−3 s−1", ("", "−3−1")), ("x_{i−1,j−1}", "xi−1,j−1", ("i−1,j−1", ""))]
SIGNS = [("Ca^{2+}", "Ca2+", ("", "2+")), ("Fe_{3}^{+}", "Fe3+", ("3", "+")), ("x_{i,j}", "xi,j", ("i,j", "")),
         ("x_{a:b}", "xa:b", ("a:b", "")), ("x_{a<b}", "xa<b", ("a<b", "")), ("x_{a>b}", "xa>b", ("a>b", "")),
         ("x^{a*b}", "xa*b", ("", "a*b")), ("x_{a'}", "xa'", ("a'", "")), ("x_{a,b'}", "xa,b'", ("a,b'", ""))]


@pytest.mark.parametrize("face", sorted(FACES))
@pytest.mark.parametrize("typed_words, seen, expected_scripts", HYPHENS + MINUS + SIGNS)
def test_a_script_is_set_as_text_not_as_mathematics(face, typed_words, seen, expected_scripts):
    figure, axis = canvas()
    in_face(figure, FACES[face])
    type_words(figure, {"title": typed_words})
    assert seen in words(figure), (seen, words(figure))
    assert scripts(line_of(figure, seen)) == expected_scripts
    # no mathematical spacing: every glyph of the line sits against the one before it
    assert max(gap for _char, gap in mathtext_gaps(axis.title)) < .5, mathtext_gaps(axis.title)


def test_a_hyphen_between_two_letters_is_a_hyphen_and_never_a_minus_sign_or_a_prime():
    figure, _axis = canvas()
    type_words(figure, {"title": "I_{open-pore} x_{a'}"})
    shown = "".join(words(figure))
    assert "open-pore" in shown and "a'" in shown, shown
    assert not {"−", "′", "¤"} & set(shown), shown


def test_the_gap_measure_sees_the_spaced_minus_sign_mathtext_makes_of_a_hyphen():
    """The test above would pass vacuously if its measure could not see operator spacing."""
    figure = Figure(figsize=(5, 2), dpi=100)
    FigureCanvasAgg(figure)
    text = figure.text(.05, .5, r"I$_{\mathregular{open-pore}}$", fontsize=12, family=font_family_list(None))
    assert max(gap for _char, gap in mathtext_gaps(text)) > 1.


def test_the_script_markup_is_unchanged_where_there_is_no_script_to_set():
    """A hyphen, plus or comma outside a script is the character typed, drawn by the first face."""
    figure, _axis = canvas()
    type_words(figure, {"title": "open-pore, a+b: <c> d'e I_{0}"})
    assert "open-pore, a+b: <c> d'e I0" in words(figure)
