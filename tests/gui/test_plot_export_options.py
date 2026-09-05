"""Publication sizing remains presentation-only, bounded, and format-explicit."""
import math
from pathlib import Path
import xml.dom.minidom as _xml

import nanoscope  # noqa: F401 -- Qt bootstrap before importing Qt
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.plot_export_options import (
    ExportOptionsDialog,
    choose_export_options,
    resolve_export_options,
)


@pytest.mark.parametrize("extension", ["png", ".pdf", "SVG"])
def test_native_keeps_existing_export_dimensions(extension):
    options = resolve_export_options((800, 600), extension=extension)
    assert options.width == options.height == 0
    assert options.native
    assert options.physical_width_mm is options.physical_height_mm is None
    assert options.extension == "." + extension.lower().lstrip(".")


def test_png_mm_inches_and_pdf_points_are_unambiguous():
    png = resolve_export_options((800, 600), native=False,
                                 width=50.8, height=25.4, dpi=300)
    inches = resolve_export_options((800, 600), native=False,
                                    width=2, height=1, units="in", dpi=300)
    assert png == inches
    assert (png.width, png.height) == (600, 300)
    pdf = resolve_export_options((800, 600), extension="pdf", native=False,
                                 width=2, height=1, units="in", dpi=1200)
    assert (pdf.width, pdf.height) == (144, 72)
    assert pdf.physical_width_mm == 50.8


@pytest.mark.parametrize("dimensions", [(0, 20), (-1, 20), (math.nan, 20),
                                       (20, math.inf)])
def test_rejects_invalid_native_dimensions(dimensions):
    with pytest.raises(ValueError, match="finite, positive"):
        resolve_export_options(dimensions)


@pytest.mark.parametrize("width", [0, -1, math.nan, math.inf, 2001])
def test_rejects_invalid_physical_dimensions(width):
    with pytest.raises(ValueError, match="physical dimension"):
        resolve_export_options((800, 600), native=False, width=width)


@pytest.mark.parametrize("dpi", [0, 2401, 300.5, math.nan, math.inf, "bad"])
def test_rejects_invalid_dpi(dpi):
    with pytest.raises(ValueError, match="DPI"):
        resolve_export_options((800, 600), dpi=dpi)


def test_caps_raster_dimensions_and_total_allocation():
    with pytest.raises(ValueError, match="per side"):
        resolve_export_options((800, 600), native=False,
                               width=1000, height=10, dpi=1200)
    with pytest.raises(ValueError, match="megapixels"):
        resolve_export_options((800, 600), native=False,
                               width=300, height=300, dpi=1200)
    with pytest.raises(ValueError, match="per side"):
        resolve_export_options((100_000, 600))
    with pytest.raises(ValueError, match="megapixels"):
        resolve_export_options((20_000, 20_000))


def test_svg_physical_size_is_metadata_not_ignored_primitive_dimensions():
    options = resolve_export_options((800, 600), extension="svg", native=False,
                                     width=85, height=63.75)
    assert not options.native
    assert options.width == options.height == 0
    assert options.physical_width_mm == 85
    assert options.physical_height_mm == 63.75


def test_dialog_presets_and_aspect_lock(qtbot):
    dialog = ExportOptionsDialog(native_size=QtCore.QSize(800, 400))
    qtbot.addWidget(dialog)
    assert dialog.options().native
    dialog.preset_combo.setCurrentIndex(1)
    assert dialog.width_spin.value() == 85.0
    assert dialog.height_spin.value() == 42.5
    assert dialog.options().physical_width_mm == 85.0
    dialog.width_spin.setValue(120)
    assert dialog.height_spin.value() == 60
    assert dialog.preset_combo.currentIndex() == 0
    dialog.height_spin.setValue(70)
    assert dialog.width_spin.value() == 140
    dialog.aspect_check.setChecked(False)
    dialog.height_spin.setValue(80)
    assert dialog.width_spin.value() == 140
    dialog.aspect_check.setChecked(True)
    assert dialog.height_spin.value() == 70
    dialog.preset_combo.setCurrentIndex(2)
    assert dialog.width_spin.value() == 180
    assert dialog.height_spin.value() == 90


def test_units_are_a_presentation_change_without_rounding_drift(qtbot):
    dialog = ExportOptionsDialog(native_size=(800, 600))
    qtbot.addWidget(dialog)
    dialog.preset_combo.setCurrentIndex(1)
    before = dialog.options()
    for _ in range(10):
        dialog.units_combo.setCurrentIndex(1)
        assert dialog.options() == before
        dialog.units_combo.setCurrentIndex(0)
        assert dialog.options() == before


def test_vector_formats_disable_only_dpi_and_preserve_physical_choice(qtbot):
    dialog = ExportOptionsDialog(native_size=(800, 600))
    qtbot.addWidget(dialog)
    dialog.preset_combo.setCurrentIndex(2)
    before = dialog.options()
    for extension in (".svg", ".pdf"):
        dialog.format_combo.setCurrentIndex(dialog.format_combo.findData(extension))
        assert not dialog.options().native
        assert dialog.options().physical_width_mm == before.physical_width_mm
        assert not dialog.dpi_spin.isEnabled()
        assert dialog.width_spin.isEnabled()
        assert "DPI is not applied" in dialog.summary_label.text()
    dialog.format_combo.setCurrentIndex(dialog.format_combo.findData(".png"))
    assert dialog.options() == before


def test_invalid_size_cannot_be_accepted(qtbot):
    dialog = ExportOptionsDialog(native_size=(800, 600))
    qtbot.addWidget(dialog)
    dialog.preset_combo.setCurrentIndex(2)
    dialog.dpi_spin.setValue(2400)
    dialog.width_spin.setValue(1500)
    assert not dialog.buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()
    assert "Cannot export" in dialog.summary_label.text()
    dialog.accept()
    assert dialog.result() != QtWidgets.QDialog.Accepted


def test_initial_dimensions_follow_existing_px_or_pt_contract(qtbot):
    png = ExportOptionsDialog(native_size=(800, 400), initial_width=1200)
    pdf = ExportOptionsDialog(native_size=(800, 400), initial_format="pdf",
                              initial_height=144)
    for dialog in (png, pdf):
        qtbot.addWidget(dialog)
    assert (png.options().width, png.options().height) == (1200, 600)
    assert (pdf.options().width, pdf.options().height) == (288, 144)
    assert "points" in pdf.summary_label.text()
    assert not pdf.dpi_spin.isEnabled()


def test_cancel_returns_none_without_destination_dialog(qtbot, monkeypatch):
    monkeypatch.setattr(ExportOptionsDialog, "exec", lambda self: QtWidgets.QDialog.Rejected)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: pytest.fail("options must not choose a destination"))
    assert choose_export_options(None, (800, 600)) is None


def test_accept_returns_validated_options_without_writes(qtbot, monkeypatch, tmp_path):
    def accept(dialog):
        dialog.preset_combo.setCurrentIndex(1)
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(ExportOptionsDialog, "exec", accept)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: pytest.fail("options must not choose a destination"))
    before = list(tmp_path.iterdir())
    options = choose_export_options(None, (800, 400), initial_format="pdf")
    assert options.extension == ".pdf"
    assert options.physical_width_mm == 85
    assert list(tmp_path.iterdir()) == before


def test_chooser_confirms_the_normalized_real_target(qtbot, monkeypatch, tmp_path):
    from nanoscope.gui import export_paths, plot_export

    item = QtWidgets.QGraphicsRectItem(0, 0, 800, 400)
    options = resolve_export_options((800, 400), extension="svg", native=False,
                                     width=85, height=42.5)
    monkeypatch.setattr(plot_export, "choose_export_options", lambda *a, **k: options)
    target = tmp_path / "figure.svg"
    target.write_text("earlier export")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "figure.png"), ""))
    confirmed = []
    monkeypatch.setattr(export_paths, "confirm_overwrite",
                        lambda parent, title, paths: confirmed.extend(paths) or True)
    result = plot_export.choose_live_plot_export(None, item, "figure.png")
    assert result == (target, options)
    assert confirmed == [target]
    assert target.read_text() == "earlier export"


def test_cancelled_sizing_does_not_open_destination_dialog(qtbot, monkeypatch):
    from nanoscope.gui import plot_export

    item = QtWidgets.QGraphicsRectItem(0, 0, 800, 400)
    monkeypatch.setattr(plot_export, "choose_export_options", lambda *a, **k: None)
    monkeypatch.setattr(plot_export, "ask_export_path",
                        lambda *a, **k: pytest.fail("cancelled sizing must stop"))
    assert plot_export.choose_live_plot_export(None, item, "figure.png") is None


def test_svg_sizing_is_staged_and_preserves_viewbox_and_geometry(qtbot, monkeypatch, tmp_path):
    from nanoscope.gui import plot_export

    target = tmp_path / "figure.svg"
    target.write_text("earlier export")
    svg = ('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
           'width="800" height="400" viewBox="0 0 800 400">'
           '<path d="M0,0 L100,200 Z" stroke="#333"/></svg>')
    staged_paths = []

    def render(item, staged):
        staged_paths.append(Path(staged))
        assert target.read_text() == "earlier export"
        Path(staged).write_text(svg)

    monkeypatch.setattr(plot_export.pg_svg, "export_svg", render)
    plot_export.write_live_plot_image(target, object(), physical_width_mm=85,
                                      physical_height_mm=42.5)
    assert staged_paths and staged_paths[0] != target
    document = _xml.parse(str(target))
    root = document.documentElement
    assert root.getAttribute("width") == "85mm"
    assert root.getAttribute("height") == "42.5mm"
    assert root.getAttribute("viewBox") == "0 0 800 400"
    assert root.getElementsByTagName("path")[0].getAttribute("d") == "M0,0 L100,200 Z"
    assert root.getElementsByTagName("path")[0].getAttribute("stroke") == "#333"
    document.unlink()


def test_invalid_svg_sizing_preserves_existing_export(qtbot, monkeypatch, tmp_path):
    from nanoscope.gui import plot_export

    target = tmp_path / "figure.svg"
    target.write_text("earlier export")
    monkeypatch.setattr(plot_export.pg_svg, "export_svg",
                        lambda item, staged: Path(staged).write_text("<svg/>"))
    with pytest.raises(ValueError, match="viewBox"):
        plot_export.write_live_plot_image(target, object(), physical_width_mm=85,
                                          physical_height_mm=42.5)
    assert target.read_text() == "earlier export"


def test_actual_vector_and_raster_exports_keep_data_and_zoom(qtbot, tmp_path):
    import numpy as np
    import pyqtgraph as pg
    from PySide6 import QtGui
    from nanoscope.gui.plot_export import write_live_plot_image

    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(800, 400)
    widget.show()
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([3.0, 4.0, 2.0, 1.0])
    curve = widget.plot(x, y)
    widget.setXRange(0.2, 2.8, padding=0)
    widget.setYRange(0.5, 5, padding=0)
    QtWidgets.QApplication.processEvents()
    before = widget.viewRange()
    item = widget.getPlotItem()
    for extension in ("png", "pdf", "svg"):
        options = resolve_export_options((800, 400), extension=extension,
                                         native=False, width=50.8, height=25.4)
        target = tmp_path / f"actual.{extension}"
        write_live_plot_image(target, item, width=options.width, height=options.height,
                              physical_width_mm=options.physical_width_mm,
                              physical_height_mm=options.physical_height_mm)
        assert target.stat().st_size > 0
        assert widget.viewRange() == before
        np.testing.assert_array_equal(curve.getData()[0], x)
        np.testing.assert_array_equal(curve.getData()[1], y)
    raster = QtGui.QImage(str(tmp_path / "actual.png"))
    assert (raster.width(), raster.height()) == (600, 300)
    svg = _xml.parse(str(tmp_path / "actual.svg"))
    assert svg.documentElement.getAttribute("width") == "50.8mm"
    assert svg.documentElement.getAttribute("height") == "25.4mm"
    svg.unlink()


def test_event_panel_only_interactive_exports_use_options(qtbot, tmp_path, monkeypatch):
    from nanoscope.core.session import Session
    from nanoscope.gui import event_plot_panel

    window = event_plot_panel.EventPlotWindow(Session())
    qtbot.addWidget(window)
    options = resolve_export_options((800, 400), extension="pdf", native=False,
                                     width=85, height=42.5)
    target = tmp_path / "event.pdf"
    calls = []
    monkeypatch.setattr(event_plot_panel, "choose_live_plot_export",
                        lambda *a, **k: (target, options))
    monkeypatch.setattr(event_plot_panel, "write_live_plot_image",
                        lambda *a, **k: calls.append((a, k)) or [])
    assert window.save_image() == target
    assert calls[-1][1] == dict(width=options.width, height=options.height,
                                physical_width_mm=85, physical_height_mm=42.5)
    monkeypatch.setattr(event_plot_panel, "choose_live_plot_export",
                        lambda *a, **k: pytest.fail("explicit paths bypass dialog"))
    assert window.save_image(str(target)) == target
    assert calls[-1][1] == {}
