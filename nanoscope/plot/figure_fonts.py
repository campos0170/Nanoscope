"""The figure-font policy: one short list, one default, one fallback rule.

Pure Python (no Qt, no matplotlib) so the live plots, the SVG writer, the matplotlib
figures and the pickers all read the same answer. The list is what journals name
(docs: the 2 November 2026 plotting proposal, section 4): Arial first, then Helvetica,
Times New Roman and Courier New. Greek is typed as Unicode Greek in the chosen family;
no symbol face is offered as a figure font. Any other installed family stays reachable
through the pickers' "Other font…" row, off the main list.

An empty / ``None`` / ``"Default"`` family means "the figure default", which is Arial.
That is resolved HERE, at draw and export time, instead of leaving it to the application
font (which on macOS is the private ``.AppleSystemUIFont`` that no journal has).
"""
from __future__ import annotations

from typing import Iterable

#: The family an untouched figure is drawn and exported in.
DEFAULT_FAMILY = "Arial"

#: The main list, in picker order. Arial first: it is the default.
FIGURE_FAMILIES: tuple[str, ...] = ("Arial", "Helvetica", "Times New Roman", "Courier New")

#: The picker row that opens the full list of installed families.
OTHER_FONT_LABEL = "Other font…"

# family (case-folded) -> (substitutes in preference order, CSS / matplotlib generic).
_STACKS: dict[str, tuple[tuple[str, ...], str]] = {
    "arial": (("Helvetica", "Liberation Sans", "Arimo", "TeX Gyre Heros", "Nimbus Sans"), "sans-serif"),
    "helvetica": (("Arial", "TeX Gyre Heros", "Nimbus Sans", "Liberation Sans", "Arimo"), "sans-serif"),
    "times new roman": (("Times", "Liberation Serif", "Tinos", "TeX Gyre Termes", "Nimbus Roman"), "serif"),
    "courier new": (("Courier", "Liberation Mono", "Cousine", "Nimbus Mono PS"), "monospace"),
}

#: Faces of symbols and dingbats, not of letters. Symbol is never a figure font (Greek is typed as
#: Unicode Greek in the chosen family), so the pickers' Other font… list leaves these out; a family
#: an older project stored still shows, on its own extra row.
SYMBOL_FAMILIES: tuple[str, ...] = ("Symbol", "Webdings", "Wingdings", "Wingdings 2",
                                    "Wingdings 3", "Zapf Dingbats", "Apple Symbols")

_DEFAULT_WORDS = {"", "default"}


def is_default(family: str | None) -> bool:
    """True for an unset family: ``None``, empty, spaces, or the picker's "Default"."""
    return family is None or str(family).strip().lower() in _DEFAULT_WORDS


def effective_family(family: str | None) -> str:
    """The family a figure is drawn in: ``family`` as typed, or Arial when it is unset."""
    return DEFAULT_FAMILY if is_default(family) else str(family).strip()


def is_listed(family: str | None) -> bool:
    """True when ``family`` (after the default rule) is on the main list. Case-insensitive."""
    return effective_family(family).casefold() in {f.casefold() for f in FIGURE_FAMILIES}


def canonical_family(family: str | None) -> str:
    """The main-list spelling of ``family`` when it is listed (``"arial"`` -> ``"Arial"``);
    otherwise the effective family unchanged."""
    eff = effective_family(family)
    for listed in FIGURE_FAMILIES:
        if listed.casefold() == eff.casefold():
            return listed
    return eff


def is_symbol_family(family: str | None) -> bool:
    """True for a symbol or dingbat face (``Wingdings``, ``Zapf Dingbats``...), which no
    figure's text is set in. Case-insensitive."""
    return str(family or "").strip().casefold() in {f.casefold() for f in SYMBOL_FAMILIES}


def generic_family(family: str | None) -> str:
    """``"sans-serif"``, ``"serif"`` or ``"monospace"`` -- the generic a viewer falls back to.
    A family off the main list is assumed sans-serif."""
    entry = _STACKS.get(effective_family(family).casefold())
    return entry[1] if entry else "sans-serif"


def fallback_stack(family: str | None) -> tuple[str, ...]:
    """The named family first, then its metric-compatible substitutes, then the Arial stack
    for a family off the main list, then the generic. No duplicates (case-insensitive).

    >>> fallback_stack("Arial")[:2]
    ('Arial', 'Helvetica')
    """
    eff = canonical_family(family)
    entry = _STACKS.get(eff.casefold())
    names: list[str] = [eff]
    if entry is not None:
        names.extend(entry[0])
        generic = entry[1]
    else:
        # Off the main list: fall back to Arial's stack, so a missing family never leaves a
        # viewer to pick a face of its own.
        names.append(DEFAULT_FAMILY)
        names.extend(_STACKS[DEFAULT_FAMILY.casefold()][0])
        generic = "sans-serif"
    names.append(generic)
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return tuple(out)


def css_font_family(family: str | None, quote: str = "'") -> str:
    """A CSS / SVG ``font-family`` value with the fallback stack: names containing a space are
    quoted with ``quote`` (single quotes unless told otherwise), the generic is bare.

    ``quote='"'`` is for a value that goes inside a single-quoted ``style='...'`` attribute, as
    pyqtgraph's label HTML writes it: a single quote there ends the attribute, and Qt then
    draws the label in the application font without saying so.

    >>> css_font_family("Times New Roman")
    "'Times New Roman', Times, 'Liberation Serif', Tinos, 'TeX Gyre Termes', 'Nimbus Roman', serif"
    """
    generics = {"sans-serif", "serif", "monospace"}
    parts = []
    for name in fallback_stack(family):
        parts.append(name if name in generics or " " not in name else f"{quote}{name}{quote}")
    return ", ".join(parts)


def matplotlib_font_list(family: str | None) -> list[str]:
    """The stack as matplotlib's ``font.sans-serif`` / ``font.serif`` / ``font.monospace`` list
    (the generic itself excluded; matplotlib adds DejaVu as its own last resort)."""
    generics = {"sans-serif", "serif", "monospace"}
    return [n for n in fallback_stack(family) if n not in generics]


def resolve_installed(family: str | None, installed: Iterable[str]) -> str:
    """The first member of the family's stack that is in ``installed`` (case-insensitive,
    returned in the installed spelling). When none is, the effective family itself, so Qt's
    own substitution still has the requested name to work from."""
    have = {str(f).casefold(): str(f) for f in installed}
    for name in fallback_stack(family):
        hit = have.get(name.casefold())
        if hit is not None:
            return hit
    return effective_family(family)
