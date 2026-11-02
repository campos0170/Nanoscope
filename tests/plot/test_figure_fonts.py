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
