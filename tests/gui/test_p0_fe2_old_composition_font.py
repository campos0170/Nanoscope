"""FE2: a composition an older Mac build saved in the application font -- a private face -- is
revised on a row that says what it is, keeps the family it was saved with, and writes Arial for its
letters in the SVG.

The older composer's default row was "Application default", which stored the application font's own
family (``.AppleSystemUIFont`` on a Mac). Revising such a composition in this build used to show a row
literally named ``.AppleSystemUIFont`` and to name that private face first in the SVG's letters, which
the CHANGELOG says no SVG does. The stored family is unchanged (a composition keeps the family it was
saved with, so its letters keep matching its panels in the PNG and the PDF); only the words of the row
and the SVG's name for it change.

Synthetic data only; offscreen.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore

from nanoscope.core.figure_library import FIGURE_KIND
from nanoscope.core.project import WorkspaceArtifact
from nanoscope.gui import font_picker
from nanoscope.gui.figure_bundle import write_frozen_figure_image
from nanoscope.gui.figure_composer import FigureComposerDialog
from nanoscope.gui.figure_composition_revision import extract_composition
from nanoscope.gui.font_picker import PRIVATE_ROW_LABEL, FontPicker
from nanoscope.gui.plot_export_options import resolve_export_options
from nanoscope.plot import figure_fonts as ff

from nanoscope.gui.figure_composition import compose_figures
from nanoscope.gui.figure_library import FigureLibrary

from .test_figure_composer import entries  # noqa: F401 -- a fixture: three saved figures
from .test_figure_composition_revision import _entry, _snapshot
from .test_p0_figure_font_picker import arial, pdf_font_names, pick  # noqa: F401 -- ``arial`` is a fixture
from .test_p0_g6a_review_fixes import _library_entry, _summaries_page

PRIVATE = ".AppleSystemUIFont"
SVG = "{http://www.w3.org/2000/svg}"
TIP = QtCore.Qt.ItemDataRole.ToolTipRole


def _revised(qtbot, tmp_path, family):
    """The dialog ``FigureLibraryWindow.revise_current`` opens for a composition whose label
    font is ``family`` (what an older Mac build stored when its label font was left at
    "Application default"): two Distribution summaries panels composed, the layout's family
    rewritten, the embedded originals handed back."""
    _page, plot = _summaries_page(qtbot, tmp_path, "")
    entries = [_library_entry("test", "ECDF A", plot), _library_entry("test", "ECDF B", plot)]
    first = FigureComposerDialog(entries, current_id=entries[0].artifact_id)
    qtbot.addWidget(first)
    first.show()
    first.update_preview()
    assert first.snapshot is not None, first._message.text()
    panels, layout = extract_composition(first.snapshot)
    layout["label_style"]["family"] = family
    embedded = []
    for index, panel in enumerate(panels):
        metadata, blobs = panel.snapshot.to_storage()
        embedded.append(WorkspaceArtifact(
            artifact_id=panel.panel_id, kind=FIGURE_KIND, schema_version=1,
            payload={"name": f"Embedded panel {index + 1}", "source_label": "Embedded original",
                     "origin_figure_id": None, "source_recipes": [], "snapshot": metadata},
            blobs=blobs))
    dialog = FigureComposerDialog(embedded, initial_layout=layout, name="revision")
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def _letters(snapshot):
    return [e for e in ET.fromstring(snapshot.svg_bytes).iter(SVG + "text")
            if (e.get("id") or "").endswith("_label")]


def _pdf_fonts(dialog, tmp_path, name):
    out = tmp_path / name
    snapshot = dialog.snapshot
    write_frozen_figure_image(out, snapshot, resolve_export_options(
        (snapshot.width, snapshot.height), extension=".pdf"))
    return pdf_font_names(out)


def test_a_composition_an_older_mac_build_saved_is_revised_on_a_row_that_says_what_it_is(
        qapp, qtbot, tmp_path, arial):
    dialog = _revised(qtbot, tmp_path, PRIVATE)
    picker = dialog._label_font

    # The words: no row names the private face, and the current one says what it is.
    rows = [picker.itemText(i) for i in range(picker.count())]
    assert not [row for row in rows if "AppleSystemUIFont" in row or row.startswith(".")], rows
    assert picker.currentText() == PRIVATE_ROW_LABEL == "Application font (older figure)"
    assert "private system face" in picker.itemData(picker.currentIndex(), TIP)

    # The value: still the family it was saved with, so the letters keep matching its panels.
    assert picker.value() == PRIVATE and dialog.label_family() == PRIVATE
    dialog.update_preview()
    assert dialog.snapshot is not None, dialog._message.text()
    assert dialog.snapshot.config()["label_style"]["family"] == PRIVATE
    if font_picker.is_installed(PRIVATE):            # a Mac: the letters really are drawn in that face
        assert any(name.startswith(".") for name in _pdf_fonts(dialog, tmp_path, "kept.pdf")), (
            "the PDF's letters stay in the private face (the premise of 'keeps its family')")

    # The SVG: the letters are written by the rule of every other SVG (a private face is Arial).
    letters = _letters(dialog.snapshot)
    assert len(letters) == 2
    assert [e.get("font-family") for e in letters] == [ff.css_font_family("Arial")] * 2
    assert "AppleSystemUIFont" not in dialog.snapshot.svg_bytes.decode("utf-8")      # (these panels are Arial)

    # Choosing Arial changes the letters everywhere, and the preview must be made again.
    pick(picker, "Arial (default)")
    assert dialog.snapshot is None and dialog.label_family() == "Arial"
    dialog.update_preview()
    assert dialog.snapshot.config()["label_style"]["family"] == "Arial"
    assert not [name for name in _pdf_fonts(dialog, tmp_path, "arial.pdf") if name.startswith(".")]
    assert [e.get("font-family") for e in _letters(dialog.snapshot)] == [ff.css_font_family("Arial")] * 2


def test_the_library_windows_revise_route_keeps_an_older_mac_composition_on_the_named_row(
        entries, qtbot, monkeypatch):  # noqa: F811
    """The route a person takes: Figure Library, select the composition, Revise composition…. The
    composition below is stored as an older Mac build stored it (its label family is the private
    application font); the revision it saves keeps that family and says Arial in its SVG."""
    old = compose_figures([_snapshot(entry) for entry in entries], ["A", "B", "C"], 2,
                          label_style={"family": PRIVATE, "size_px": 18, "bold": True, "color": "#111111"})
    stored = [_entry(old, entries)]
    library = FigureLibrary(entries=lambda: stored, status=lambda _: ("Saved snapshot", "Frozen"),
                            append=stored.append)
    qtbot.addWidget(library)
    seen = {}

    def revise(dialog):
        picker = dialog._label_font
        seen["rows"] = [picker.itemText(i) for i in range(picker.count())]
        seen["current"], seen["value"] = picker.currentText(), picker.value()
        dialog.update_preview()
        assert dialog.snapshot is not None, dialog._message.text()
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(FigureComposerDialog, "exec", revise)
    result = library.revise_current()
    assert result is stored[-1] and len(stored) == 2

    assert seen["current"] == PRIVATE_ROW_LABEL and seen["value"] == PRIVATE
    assert not [row for row in seen["rows"] if "AppleSystemUIFont" in row or row.startswith(".")], seen["rows"]
    saved = _snapshot(result)
    assert saved.config()["label_style"]["family"] == PRIVATE, "a composition keeps the family it was saved with"
    assert [e.get("font-family") for e in _letters(saved)] == [ff.css_font_family("Arial")] * 3


@pytest.mark.parametrize("installed", [True, False])
def test_the_private_row_stores_the_exact_name_and_reads_the_same_with_or_without_the_face(
        installed, qapp, qtbot, monkeypatch):
    """Not 'not on this computer': a reader of a Windows copy of the project is told what the row
    is, not that a font with a name like a file is missing. The stored name is untouched."""
    monkeypatch.setattr(font_picker, "installed_families", lambda: (
        ([PRIVATE] if installed else []) + ["Arial", "Helvetica", "Times New Roman", "Courier New"]))
    picker = FontPicker()
    qtbot.addWidget(picker)
    picker.set_value(PRIVATE)
    assert picker.currentText() == PRIVATE_ROW_LABEL and picker.has_extra_row()
    assert picker.value() == PRIVATE and picker.currentData() == PRIVATE

    # The row is reused for the next off-list family, with that family's own words and no tip.
    picker.set_value("Calibri")
    assert picker.currentText() == "Calibri — not on this computer"
    assert picker.value() == "Calibri" and not picker.itemData(picker.currentIndex(), TIP)
    picker.set_value(PRIVATE)
    assert picker.currentText() == PRIVATE_ROW_LABEL and picker.itemData(picker.currentIndex(), TIP)
