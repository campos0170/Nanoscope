"""Selected physical columns and editable vector fonts use immutable capture state."""
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pandas as pd
import pytest
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from nanoscope.core.figure_style import FigureStyle, style_binding
from nanoscope.gui.figure_bundle import capture_figure_snapshot, write_frozen_figure_image
from nanoscope.gui.figure_export_style import export_style_for, snapshot_export_style
from nanoscope.gui.matplotlib_figure_item import MatplotlibFigureItem
from nanoscope.gui.plot_export_options import ExportOptionsDialog, resolve_export_options


def test_columns_come_from_pinned_style_and_keep_native_custom_choices(qtbot):
    style = FigureStyle('project.custom', 3, 'My columns', columns={'single_mm': 92.5, 'double_mm': 190.25})
    dialog = ExportOptionsDialog(native_size=(800, 400), figure_style=style)
    qtbot.addWidget(dialog)
    assert dialog.size_combo.currentData() is True
    assert dialog.preset_combo.itemData(0) is None
    assert dialog.preset_combo.findData(85.) == -1
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData(92.5))
    options = dialog.options()
    assert (options.physical_width_mm, options.physical_height_mm) == (92.5, 46.25)
    assert 'My columns' in dialog.preset_combo.currentText()
    owner = SimpleNamespace(appearance_state=lambda: {'style_binding': style_binding(style)})
    assert export_style_for(owner) == style


def test_mpl_frozen_svg_keeps_literal_text_and_pdf_embeds_font_without_live_artists(qtbot, tmp_path, monkeypatch):
    figure = Figure(figsize=(4, 3), dpi=100)
    axis = figure.subplots()
    axis.plot([1., 2.], [3., 4.], linewidth=.75)
    axis.set_title('Literal $price$ Ω', family='DejaVu Sans', fontsize=7.25, parse_math=False)
    item = MatplotlibFigureItem(figure)
    view = QtWidgets.QGraphicsView(); qtbot.addWidget(view)
    scene = QtWidgets.QGraphicsScene(view); view.setScene(scene); scene.addItem(item)
    style = FigureStyle('project.capture', 1, 'Capture', columns={'single_mm': 87.5})
    snapshot = capture_figure_snapshot(item, data=pd.DataFrame({'x': [1., 2.]}),
                                      config={'style': {'style_binding': style_binding(style)}})
    assert snapshot_export_style(snapshot) == style
    figure.clear()
    monkeypatch.setattr(figure, 'draw', lambda *a: pytest.fail('export revisited live artists'))
    svg = tmp_path / 'frozen.svg'
    options = resolve_export_options((snapshot.width, snapshot.height), extension='svg',
                                     native=False, width=87.5, height=65.625)
    write_frozen_figure_image(svg, snapshot, options)
    root = ET.fromstring(svg.read_bytes())
    assert root.get('width') == '87.5mm' and root.get('height') == '65.625mm'
    text = [node for node in root.iter() if node.tag.rsplit('}', 1)[-1] in ('text', 'tspan')]
    assert any('Literal $price$ Ω' in ''.join(node.itertext()) for node in text)
    assert any('sans-serif' in node.get('font-family', '') for node in text)
    pdf = tmp_path / 'frozen.pdf'
    options = resolve_export_options((snapshot.width, snapshot.height), extension='pdf',
                                     native=False, width=87.5, height=65.625)
    write_frozen_figure_image(pdf, snapshot, options)
    payload = pdf.read_bytes()
    assert payload.startswith(b'%PDF') and b'/FontFile2' in payload
    assert b'/Subtype /Type3' not in payload


@pytest.mark.parametrize('size', [7., 8.])
def test_native_physical_size_keeps_matplotlib_points_and_shrinking_reports_scaled_points(qtbot, size):
    from nanoscope.gui.plot_print_preflight import inspect_print_metrics
    figure = Figure(figsize=(4, 2), dpi=100)
    figure.text(.2, .5, 'Exact Ω', fontsize=size, family='DejaVu Sans')
    item = MatplotlibFigureItem(figure)
    metrics = inspect_print_metrics(item.native_figure_svg())
    text, _ = metrics.at_size(101.6, 50.8)
    half, _ = metrics.at_size(50.8, 25.4)
    assert text == pytest.approx(size)
    assert half == pytest.approx(size / 2)
    assert item.capture_rendering_metadata()['retained_qt_font_families']


def test_selected_style_width_matches_legacy_builder_optional_sizing():
    from nanoscope.plot.style import figure_size
    style = FigureStyle('project.size', 1, 'Size', columns={'single_mm': 92.5, 'double_mm': 190.25})
    assert figure_size('single', figure_style=style) == pytest.approx((92.5 / 25.4, 92.5 / 25.4 * .72))
    assert figure_size('single')[0] == 3.35  # Existing unbound figures keep their dimensions.


def test_saved_snapshot_replays_recorded_faces_before_pdf_and_refuses_missing_font(qtbot, tmp_path, monkeypatch):
    import json
    from dataclasses import replace
    from nanoscope.gui import matplotlib_figure_item as M
    figure = Figure(figsize=(3, 2), dpi=100)
    figure.text(.2, .5, 'Retained', family='DejaVu Sans', weight='bold', fontsize=8)
    item = MatplotlibFigureItem(figure)
    view = QtWidgets.QGraphicsView(); qtbot.addWidget(view)
    scene = QtWidgets.QGraphicsScene(view); view.setScene(scene); scene.addItem(item)
    snapshot = capture_figure_snapshot(item, data=pd.DataFrame())
    figure.clear()
    observed = []
    restore = M.restore_retained_font_faces
    def recorded(faces):
        observed.extend(faces)
        return restore(faces)
    monkeypatch.setattr(M, 'restore_retained_font_faces', recorded)
    options = resolve_export_options((snapshot.width, snapshot.height), extension='pdf')
    target = tmp_path/'recorded.pdf'
    write_frozen_figure_image(target, snapshot, options)
    assert observed and observed[0]['weight'] == 'bold'
    assert 'DejaVu Sans' in observed[0]['families']
    original = target.read_bytes()
    provenance = snapshot.provenance()
    provenance['nanoscope_capture_rendering']['renderers'][0]['retained_text_faces'][0]['families'] = ['NanoScope missing recorded font 9f0a']
    missing = replace(snapshot, provenance_json_bytes=json.dumps(provenance).encode())
    with pytest.raises(ValueError, match='recorded figure font is unavailable'):
        write_frozen_figure_image(target, missing, options)
    assert target.read_bytes() == original
    svg = tmp_path/'original.svg'
    write_frozen_figure_image(svg, missing, resolve_export_options((snapshot.width, snapshot.height), extension='svg'))
    assert svg.read_bytes() == snapshot.svg_bytes
