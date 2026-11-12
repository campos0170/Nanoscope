"""Typed words in a matplotlib figure: the one markup, and how a Text draws it.

Words typed into a figure (a title, an axis label, a legend entry) are LITERAL: ``$5 at 10%``
is drawn as typed. The one markup is ``_{...}`` and ``^{...}``, for sub- and superscripts; every
text editor promises it. The pyqtgraph pages draw it as Qt rich text
(``gui.figure_text.rich_text``, which reads :data:`SUBSUP`); a matplotlib ``Text`` has no rich
text, only mathtext, so :func:`drawn_text` converts the typed words when the Text is DRAWN.
What is stored stays as typed: nothing here is ever written to a project.

Only the scripts become mathtext. Each ``_{...}`` / ``^{...}`` is one ``$...$`` span whose content is
set in ``\\mathregular`` -- the Text's own face, weight and slant, where a bare span would be
mathtext's own italic DejaVu -- and everything else stays outside the dollar signs, laid out in the
figure's font like any other text. Words with no markup are handed over as typed, with
``parse_math`` off, exactly as before: nothing else in them is special to anything. A script
mathtext cannot draw (a ``^``, a quote, a backquote or a tilde in it, a control character, or
whatever this matplotlib's parser refuses) is left as typed, and the other scripts still draw.

A line that holds a script is laid out by mathtext as a whole, and mathtext sets the plain characters
of it, and the content of its scripts, in ONE face: the first the Text's font list finds (Arial). A
character that face lacks (the subscript zero of ΔI/I₀, the superscript minus of s⁻¹, ℃) it does not
take from the next face of the list, as the same words without markup are drawn: in the plain words it
replaces it by the currency sign, and in a script it draws a serif stand-in. So the conversion asks the
face the Text really has, when the words are applied to it (:func:`drawn_text` takes its font
properties): each character that face lacks is spelled in the mathtext font that draws it in the Text's
weight and slant (DejaVu Sans, which matplotlib ships and every figure's font list ends with), and when
mathtext cannot draw one of them either, the Text is drawn as typed, markup visible, as if there were
no markup. The result is checked by parsing it with that face.

The face is the one the Text is DRAWN in, so the words are spelled again whenever it changes: the
Comparison page's Title, X label and Y label boxes, and the study editors' per-panel words, type them
when the figure is built, and the element's own font (family, bold, italic) is applied after;
:func:`set_typed_text` keeps what was typed on the Text and :func:`retype_text` spells it again for the
face the Text now has, as long as the Text still shows what it made of those words.

A script is set as TEXT, not as mathematics. Inside ``$...$`` mathtext reads ``-`` as a minus sign
with space around it, puts space around ``+ = < > :`` and after ``, ; .``, and reads ``'`` as a prime;
the pyqtgraph pages draw a script's characters as typed. So a hyphen between two letters is a hyphen
(``\\text{-}``), any other ``-`` a minus sign, a minus sign that was typed (the app's own labels carry
one, and the inline editor opens filled with them) the same minus sign, and ``+ * , : < >`` and the
apostrophe are spelled so that mathtext sets them with no space and no meaning.

What still differs from the pyqtgraph pages, because mathtext offers no way to set it as text: a
period that is not between two digits, a semicolon and an exclamation mark keep the thin space mathtext
puts after them, and an equals sign the space it puts around it; the hyphen and the apostrophe are
DejaVu Sans' (``\\text`` draws in mathtext's own font) and not the Text's own face. And outside the
scripts: mathtext draws the en and em dashes, the curly quotes and a few other signs of a line that has a
script from DejaVu Sans Display, because it consults that face first for every character.
"""
from __future__ import annotations

import functools
import itertools
import re

__all__ = ["SUBSUP", "drawn_text", "retype_text", "set_typed_text"]

#: ``_{...}`` / ``^{...}``: the innermost braces only, the way the pyqtgraph pages read them.
SUBSUP = re.compile(r"([_^])\{([^{}]*)\}")

#: How a script's character is spelled inside ``\mathregular{...}`` when it is not its own
#: spelling: the characters mathtext reads as syntax, the space (a bare one is dropped) and the
#: backslash (``{...}`` keeps ``\backslash`` from running into the next letter).
_SPELLING = {"#": r"\#", "$": r"\$", "%": r"\%", "_": r"\_", "{": r"\{", "}": r"\}",
             " ": r"\ ", "\\": r"{\backslash}"}

#: Characters mathtext cannot draw as themselves inside a script: a bare one is a syntax error or
#: draws nothing. A script that holds one is left as typed.
_UNDRAWABLE = frozenset('^"`~')

#: The signs mathtext reads as operators or punctuation (space around them, a prime), spelled so that it
#: sets the character itself with nothing around it. A binary operator that opens its group is not
#: spaced, nor is a comma that is alone in its braces; ``\colon``, ``\less`` and ``\greater`` are not in
#: mathtext's lists of relations; ``\text`` draws just what it is given, in mathtext's own font.
_AS_TEXT = {"+": "{+}", "*": "{*}", ",": "{,}", ":": r"{\colon}", "<": r"{\less}", ">": r"{\greater}",
            "'": r"\text{'}"}
_MINUS = "\N{MINUS SIGN}"
_CURRENCY = 0xA4          # what mathtext draws for a character no face of its set has


def _is_control(char: str) -> bool:
    return ord(char) < 32 or ord(char) == 127


class _Face:
    """The first face of a Text: the one font file mathtext draws a plain character, and a script, in.

    ``covers`` says whether it has a glyph for a character; ``fallback`` is the mathtext font command
    that draws a character it lacks in its weight and slant (DejaVu Sans, Bold, Oblique, Bold Oblique).
    """

    def __init__(self, path: str):
        from matplotlib.font_manager import get_font, ttfFontProperty, weight_dict

        self._font = get_font(path)
        bold = italic = False
        try:
            entry = ttfFontProperty(self._font)
            bold = weight_dict.get(entry.weight, entry.weight) >= 600
            italic = entry.style in ("italic", "oblique")
        except (ValueError, TypeError):
            pass
        self.fallback = "\\math" + ("bfit" if bold and italic else "bf" if bold else "it" if italic else "rm")

    def covers(self, char: str) -> bool:
        return self._font.get_char_index(ord(char)) != 0


@functools.lru_cache(maxsize=64)
def _face(path: str) -> _Face:
    return _Face(path)


@functools.lru_cache(maxsize=1)
def _mathtext_parser():
    from matplotlib.mathtext import MathTextParser

    return MathTextParser("path")


@functools.lru_cache(maxsize=1024)
def _codes(text: str, path: str, fontset: str) -> tuple[int, ...] | None:
    """The character codes mathtext lays out for ``text`` with the face at ``path``, as a Text of that
    face is drawn; None when this matplotlib refuses it (an invalid span raises when drawn)."""
    from matplotlib.font_manager import FontProperties

    prop = FontProperties(fname=path, math_fontfamily=fontset)
    try:
        glyphs = _mathtext_parser().parse(text, 72, prop)[3]
    except Exception:  # noqa: BLE001 - a ValueError today; any other refusal is the same "no"
        return None
    return tuple(code for _font, _size, code, _x, _y in glyphs)


def _spell(content: str, face: _Face) -> str | None:
    """A script's characters as the mathtext that sets them as text; None when one cannot be drawn.

    A character the first face lacks is spelled in the mathtext font that has it, neighbours together.
    """
    pieces: list[str] = []
    lacking: list[str] = []

    def flush() -> None:
        if lacking:
            pieces.append(face.fallback + "{" + "".join(lacking) + "}")
            lacking.clear()

    for index, char in enumerate(content):
        if char in _UNDRAWABLE or _is_control(char):
            return None
        if char in ("-", _MINUS):
            # Between two letters a hyphen ("open-pore"); anywhere else a minus sign ("-3", "n-1"). A minus sign that
            # was typed is one wherever it stands: left bare after a letter or a digit, mathtext spaces it as an operator.
            between = (char == "-" and 0 < index < len(content) - 1 and content[index - 1].isalpha()
                       and content[index + 1].isalpha())
            spelled = (r"\text{-}" if between else "{-}" if face.covers(_MINUS)
                       else "{" + face.fallback + "{" + _MINUS + "}}")
        elif char in _AS_TEXT:
            spelled = _AS_TEXT[char]
        elif face.covers(char):
            spelled = _SPELLING.get(char, char)
        else:
            lacking.append(_SPELLING.get(char, char))
            continue
        flush()
        pieces.append(spelled)
    flush()
    return "".join(pieces)


def _script(kind: str, content: str, face: _Face, key: tuple[str, str]) -> str | None:
    """One script as a mathtext span; ``""`` when it draws nothing (``_{}``); None when it
    cannot be drawn and stays as typed."""
    spelled = _spell(content, face)
    if spelled is None:
        return None
    if not content:
        return ""
    span = f"${kind}{{\\mathregular{{{spelled}}}}}$"
    return span if _codes(span, *key) is not None else None


def _runs(line: str, face: _Face, key: tuple[str, str]) -> list[tuple[bool, str]]:
    """One line as ``(is_math, text)`` runs: plain text, and a span for each drawable script."""
    runs: list[tuple[bool, str]] = []
    position = 0

    def plain(text: str) -> None:
        if text and runs and not runs[-1][0]:
            runs[-1] = (False, runs[-1][1] + text)
        elif text:
            runs.append((False, text))

    for found in SUBSUP.finditer(line):
        span = _script(*found.groups(), face, key)
        if span is None:
            continue
        plain(line[position:found.start()])
        position = found.end()
        if span:
            runs.append((True, span))
    plain(line[position:])
    return runs


def _pieces(runs: list[tuple[bool, str]], face: _Face) -> list[tuple[bool, str]] | None:
    """The runs of a line that has a span as ``(is_math, text)`` pieces: the plain words with their
    dollar signs escaped and each stretch of characters the first face lacks spelled in the mathtext font
    that has them, the spans as they are. None when a character cannot be drawn at all."""
    pieces: list[tuple[bool, str]] = []
    for is_math, text in runs:
        if is_math:
            pieces.append((True, text))
            continue
        for covered, group in itertools.groupby(text, face.covers):
            chars = "".join(group)
            if covered:
                pieces.append((False, chars.replace("$", r"\$")))
            elif any(char in _UNDRAWABLE or _is_control(char) for char in chars):
                return None
            else:
                pieces.append((True, "$" + face.fallback + "{" + "".join(_SPELLING.get(c, c) for c in chars) + "}$"))
    return pieces


def _line(pieces: list[tuple[bool, str]]) -> str:
    """The pieces of one line that has a span, as the one mathtext string a Text draws."""
    out = []
    for index, (is_math, text) in enumerate(pieces):
        # A backslash typed just before a span would escape the dollar sign that opens it.
        tail = len(text) - len(text.rstrip("\\"))
        if not is_math and tail and index + 1 < len(pieces):
            out.append(text[:-tail])
            out.append(r"$\mathregular{" + r"{\backslash}" * tail + "}$")
        else:
            out.append(text)
    return "".join(out)


def _draws(line: str, typed: str, key: tuple[str, str]) -> bool:
    """Whether this matplotlib draws ``line`` (mathtext) with the face ``key`` and every character as
    itself: it parses, and no character was replaced by the currency sign (one the user typed is kept)."""
    codes = _codes(line, *key)
    return codes is not None and codes.count(_CURRENCY) <= typed.count(chr(_CURRENCY))


@functools.lru_cache(maxsize=512)
def _drawn(typed: str, path: str, fontset: str) -> tuple[str, bool]:
    from matplotlib import cbook

    key = (path, fontset)
    try:
        face = _face(path)
    except Exception:  # noqa: BLE001 - a face that cannot be read: draw the words as typed, as before the markup
        return typed, False
    originals = typed.split("\n")
    lines = [_runs(line, face, key) for line in originals]
    if not any(is_math for runs in lines for is_math, _ in runs):
        return "\n".join("".join(text for _, text in runs) for runs in lines), False
    drawn = []
    for original, runs in zip(originals, lines):
        if any(is_math for is_math, _ in runs):
            pieces = _pieces(runs, face)
            line = None if pieces is None else _line(pieces)
            if line is None or not cbook.is_math_text(line) or not _draws(line, original, key):
                return typed, False
        else:
            line = "".join(text for _, text in runs).replace("$", r"\$")
            if cbook.is_math_text(line):
                return typed, False
        drawn.append(line)
    return "\n".join(drawn), True


def drawn_text(typed: str, prop=None) -> tuple[str, bool]:
    """``(text, parse_math)`` to hand a matplotlib ``Text`` so that it draws ``typed``.

    ``prop`` is the Text's font properties: which character its first face lacks decides how the
    words are spelled (default: matplotlib's own). No script in the words: ``typed`` itself and False.
    Otherwise the mathtext string and True. A Text reads each line on its own, so a line with no script
    is only given its dollar signs escaped. Should this matplotlib refuse what was built, or a
    character be one that no face it can reach draws, the words are drawn as typed.
    """
    if not SUBSUP.search(typed):
        return typed, False
    from matplotlib.font_manager import FontProperties, findfont

    prop = FontProperties() if prop is None else prop
    return _drawn(typed, findfont(prop), prop.get_math_fontfamily())


def set_typed_text(artist, typed: str) -> None:
    """Give a matplotlib ``Text`` the typed words, to be drawn as :func:`drawn_text` says.

    Call it after the Text has its font, and call :func:`retype_text` after the font changes: the
    conversion asks the face it will be drawn in. What the Text said before is its automatic text, and is
    remembered (if nothing has yet) before the words go over it: the converted string is never that. The
    words, and what was made of them, are kept on the Text for :func:`retype_text`.
    """
    if not hasattr(artist, "_nanoscope_automatic_text"):
        artist._nanoscope_automatic_text = artist.get_text()
    text, parse_math = drawn_text(typed, artist.get_fontproperties())
    artist.set_text(text)
    artist.set_parse_math(parse_math)
    artist._nanoscope_typed = (typed, text, parse_math)


def retype_text(artist) -> None:
    """Spell again, for the face a ``Text`` has NOW, the words :func:`set_typed_text` gave it.

    The spelling depends on which characters the Text's first face lacks, so words typed before its family,
    weight or slant was set are spelled for another face: a character the new one lacks would be drawn as the
    currency sign. A Text that never had words typed over it, or that was given any other text since (a
    restyle that puts the figure's own words back), is left exactly as it is.
    """
    record = getattr(artist, "_nanoscope_typed", None)
    if record is not None and (artist.get_text(), artist.get_parse_math()) == record[1:]:
        set_typed_text(artist, record[0])
