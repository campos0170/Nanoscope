"""Phase 0 final review, FE1: an SVG written where a stand-in was drawn keeps the family's kind.

On a computer without Times New Roman or Courier New the figure is drawn in the first stand-in
it has (``gui.font_picker.figure_family``) and Qt writes THAT name into the SVG. The writer
(``plot.pg_svg``) used to treat the stand-in as a family off the main list and follow it with
Arial's sans-serif list, so a viewer without the one stand-in drew Arial: the serif or the
fixed-width face of the pick was gone. The stand-in is now followed by its family's list.

Drives the real route: a Voltage dependence page in an Analysis Explorer, the font set through
the Appearance widget, the SVG written by the page's own Export ▾ > Export current figure…, with
the installed list of another computer.
"""
from __future__ import annotations

import dataclasses

import pytest

from nanoscope.gui import font_picker
from nanoscope.plot import figure_fonts as ff

from .test_p0_integration import _export_through_the_menu, _svg_families
from .test_p0_reset_on_quantity_change import vdep  # noqa: F401 -- the fixture builds the page

LIBERATION = ["Liberation Sans", "Liberation Serif", "Liberation Mono", "DejaVu Sans"]
CROSCORE = ["Arimo", "Tinos", "Cousine", "DejaVu Sans"]
TEX_LIVE = ["TeX Gyre Heros", "TeX Gyre Termes", "Nimbus Mono PS", "DejaVu Sans"]
WINDOWS = ["Arial", "Times New Roman", "Courier New", "Calibri"]
MAC = ["Arial", "Helvetica", "Times New Roman", "Courier New", "Menlo"]
DEJAVU_ONLY = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]


@pytest.mark.parametrize("installed, family, expected", [
    # the finding: a serif / fixed-width pick where only a stand-in is installed
    (LIBERATION, "Times New Roman",
     "'Liberation Serif', 'Times New Roman', Times, Tinos, 'TeX Gyre Termes', 'Nimbus Roman', serif"),
    (LIBERATION, "Courier New",
     "'Liberation Mono', 'Courier New', Courier, Cousine, 'Nimbus Mono PS', monospace"),
    (CROSCORE, "Times New Roman",
     "Tinos, 'Times New Roman', Times, 'Liberation Serif', 'TeX Gyre Termes', 'Nimbus Roman', serif"),
    (CROSCORE, "Courier New",
     "Cousine, 'Courier New', Courier, 'Liberation Mono', 'Nimbus Mono PS', monospace"),
    (TEX_LIVE, "Times New Roman",
     "'TeX Gyre Termes', 'Times New Roman', Times, 'Liberation Serif', Tinos, 'Nimbus Roman', serif"),
    (TEX_LIVE, "Courier New",
     "'Nimbus Mono PS', 'Courier New', Courier, 'Liberation Mono', Cousine, monospace"),
    # the sans-serif picks were right before and are written as they were
    (LIBERATION, "Arial",
     "'Liberation Sans', Arial, Helvetica, Arimo, 'TeX Gyre Heros', 'Nimbus Sans', sans-serif"),
    (LIBERATION, "Helvetica",
     "'Liberation Sans', Arial, Helvetica, Arimo, 'TeX Gyre Heros', 'Nimbus Sans', sans-serif"),
    # a computer that has the family, or none of its stand-ins, is untouched
    (WINDOWS, "Times New Roman", ff.css_font_family("Times New Roman")),
    (MAC, "Courier New", ff.css_font_family("Courier New")),
    (DEJAVU_ONLY, "Times New Roman", ff.css_font_family("Times New Roman")),
    (DEJAVU_ONLY, "Courier New", ff.css_font_family("Courier New")),
    (WINDOWS, "Helvetica", ff.css_font_family("Arial")),     # Windows draws Helvetica in Arial
])
def test_an_svg_names_the_stand_in_then_its_familys_list(installed, family, expected, vdep,
                                                          monkeypatch, tmp_path):
    page, _explorer = vdep
    monkeypatch.setattr(font_picker, "installed_families", lambda: list(installed))
    widget = page._appearance_widget
    widget.apply_appearance(dataclasses.replace(widget.state(), font_family=family))
    svg = _export_through_the_menu(page, monkeypatch, tmp_path / "x.svg", ".svg")
    families = _svg_families(svg)
    assert families and set(families) == {expected}, set(families)
    assert page._appearance_widget.state().font_family == family, "the figure keeps the name it was given"
