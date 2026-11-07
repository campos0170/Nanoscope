"""The figure-font policy (nanoscope/plot/figure_fonts.py): list, default, fallback."""
from __future__ import annotations

import pytest

from nanoscope.plot import figure_fonts as ff


def test_the_main_list_is_the_four_journal_families_with_arial_first():
    assert ff.FIGURE_FAMILIES == ("Arial", "Helvetica", "Times New Roman", "Courier New")
    assert ff.DEFAULT_FAMILY == "Arial"


@pytest.mark.parametrize("unset", [None, "", "  ", "Default", "default"])
def test_an_unset_family_resolves_to_arial(unset):
    assert ff.is_default(unset)
    assert ff.effective_family(unset) == "Arial"


def test_a_typed_family_is_kept_and_listing_is_case_insensitive():
    assert ff.effective_family(" Garamond ") == "Garamond"
    assert ff.is_listed("times new roman") and not ff.is_listed("Garamond")
    assert ff.canonical_family("times new roman") == "Times New Roman"
    assert ff.canonical_family("Garamond") == "Garamond"


def test_the_stack_names_the_family_then_substitutes_then_a_generic_without_duplicates():
    st = ff.fallback_stack("Arial")
    assert st[0] == "Arial" and st[1] == "Helvetica" and st[-1] == "sans-serif"
    assert len({s.casefold() for s in st}) == len(st)
    assert ff.fallback_stack("Times New Roman")[-1] == "serif"
    assert ff.fallback_stack("Courier New")[-1] == "monospace"
    # an off-list family falls back to Arial's stack, never to the viewer's own choice
    other = ff.fallback_stack("Garamond")
    assert other[0] == "Garamond" and "Arial" in other and other[-1] == "sans-serif"


def test_css_quotes_names_with_spaces_and_leaves_the_generic_bare():
    css = ff.css_font_family("Times New Roman")
    assert css.startswith("'Times New Roman', Times, ") and css.endswith(", serif")
    assert ff.css_font_family(None).startswith("Arial, Helvetica, ")


def test_resolve_installed_walks_the_stack_in_the_installed_spelling():
    assert ff.resolve_installed("Arial", ["arial", "Helvetica"]) == "arial"
    assert ff.resolve_installed("Arial", ["Liberation Sans", "DejaVu Sans"]) == "Liberation Sans"
    assert ff.resolve_installed("Times New Roman", ["Tinos"]) == "Tinos"
    assert ff.resolve_installed(None, []) == "Arial"          # nothing installed: the name stays
    assert ff.resolve_installed("Garamond", ["Garamond"]) == "Garamond"


def test_matplotlib_list_drops_the_generic():
    lst = ff.matplotlib_font_list("Arial")
    assert lst[0] == "Arial" and "sans-serif" not in lst


@pytest.mark.parametrize("standin, listed, generic", [
    ("Liberation Serif", "Times New Roman", "serif"),
    ("Tinos", "Times New Roman", "serif"),
    ("TeX Gyre Termes", "Times New Roman", "serif"),
    ("Times", "Times New Roman", "serif"),
    ("liberation serif", "Times New Roman", "serif"),
    ("Liberation Mono", "Courier New", "monospace"),
    ("Cousine", "Courier New", "monospace"),
    ("Courier", "Courier New", "monospace"),
    ("Liberation Sans", "Arial", "sans-serif"),       # Arial's and Helvetica's alike: both sans-serif
])
def test_a_stand_in_keeps_the_kind_of_the_family_it_stands_in_for(standin, listed, generic):
    """Qt writes the NAME THE FIGURE WAS DRAWN IN into an SVG: on a computer without Times New
    Roman that is a stand-in, and the list behind it must still end in serif."""
    assert ff.standin_of(standin) == listed
    stack = ff.fallback_stack(standin)
    assert stack[0].casefold() == standin.casefold()      # what was drawn is still named first
    assert stack[1] == listed and stack[-1] == generic    # then the family chosen, then its kind
    assert len({s.casefold() for s in stack}) == len(stack)
    assert ff.generic_family(standin) == generic
    assert ff.css_font_family(standin).endswith(f", {generic}")


def test_a_listed_family_and_a_family_on_no_list_are_no_stand_in():
    for name in (*ff.FIGURE_FAMILIES, "Garamond", "Calibri", "Menlo", "DejaVu Serif", "", None):
        assert ff.standin_of(name) is None, name
    # unchanged: an unknown family is followed by Arial's list and the sans-serif generic
    assert ff.fallback_stack("DejaVu Serif")[1] == "Arial" and ff.generic_family("DejaVu Serif") == "sans-serif"
    # the listed families' own stacks are untouched
    assert ff.fallback_stack("Times New Roman")[:3] == ("Times New Roman", "Times", "Liberation Serif")
    assert ff.fallback_stack("Helvetica")[:2] == ("Helvetica", "Arial")
