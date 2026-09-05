"""Cross-panel appearance persistence and real publication export contracts."""
from __future__ import annotations

import struct

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtGui, QtPdf, QtTest

from nanoscope.core import analysis_workspace as workspace
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.plot_appearance import PlotAppearance
from nanoscope.gui.plot_export import apply_png_resolution
from nanoscope.gui.plot_export_options import resolve_export_options
from .test_shared_plot_appearance import scientific_window, _forbid_analysis
from .test_event_plot import _fed_window as event_window
from .test_psd import _fed_window as psd_window
from .test_population_plot_presentation_updates import window as population_window


def test_style_project_roundtrip_has_no_refit_or_edit_signal(scientific_window, monkeypatch):
    window = scientific_window
    capture, restore = ((workspace.capture_iv, workspace.restore_iv)
                        if isinstance(window, IVWindow) else
                        (workspace.capture_voltage_dependence, workspace.restore_voltage_dependence))
    state = {"title_size": 18, "background": "light", "data_color": "#0072b2", "grid": False}
    window.set_appearance_state(state)
    artifact = capture(window)
    saved_fit = window.fit()
    saved_points = window.points()
    window.set_appearance_state({})
    _forbid_analysis(monkeypatch, window)
    changes = QtTest.QSignalSpy(window.workspaceChanged)
    assert not restore(window, artifact).skipped
    assert window._appearance.to_dict() == state
    assert window.fit() == saved_fit
    if isinstance(window, IVWindow):
        # These synthetic sources are deliberately not in the Session. Restore
        # keeps their values but cannot invent a live document binding.
        fields = ("v_volts", "i_open", "current_units", "voltage_source", "label")
        assert [tuple(getattr(p, key) for key in fields) for p in window.points()] == [
            tuple(getattr(p, key) for key in fields) for p in saved_points]
    else:
        assert window.points() == saved_points
    assert changes.count() == 0
    container = artifact.payload if isinstance(window, IVWindow) else artifact.payload["config"]
    for malformed in (None, "invalid", [1, 2]):
        container["appearance"] = malformed
        assert not restore(window, artifact).skipped
        assert window._appearance == PlotAppearance()
    container.pop("appearance")
    assert not restore(window, artifact).skipped
    assert changes.count() == 0


def test_event_typography_retains_arrays_and_roundtrips(qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import event_shape
    window = event_window(qtbot, tmp_path, n_events=7)
    window._density_main.setRange(xRange=(-2.8, -2.6), yRange=(50, 90), padding=0)
    arrays = (window._density_counts, *window._ensemble)
    copies = [a.copy() for a in arrays]
    ranges = np.asarray(window._density_main.viewRange()).copy()
    def forbidden(*args, **kwargs):
        pytest.fail("Styling/restore invoked analysis")
    monkeypatch.setattr(event_shape, "density_2d", forbidden)
    monkeypatch.setattr(event_shape, "event_ensemble", forbidden)
    window._appearance_widget.title_size.setValue(16)
    window._appearance_widget.line_width.setValue(2)
    qtbot.wait(10)
    for actual, original, expected in zip((window._density_counts, *window._ensemble), arrays, copies):
        assert actual is original
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(window._density_main.viewRange(), ranges, rtol=0, atol=1e-12)
    artifact = workspace.capture_event_plots(window)
    window.set_appearance_state({})
    assert not workspace.restore_event_plots(window, artifact).skipped
    assert window._appearance.title_size == 16
    assert window._appearance.line_width == 2


def _png_chunks(content):
    offset = 8
    chunks = []
    while offset < len(content):
        size = struct.unpack_from(">I", content, offset)[0]
        end = offset + size + 12
        chunks.append((content[offset + 4:offset + 8], content[offset:end]))
        offset = end
    return chunks


def test_png_resolution_changes_no_pixel_chunks(tmp_path):
    target = tmp_path / "resolution.png"
    image = QtGui.QImage(80, 60, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor("#0072b2"))
    assert image.save(str(target))
    original = target.read_bytes()
    apply_png_resolution(target, dpi=600)
    assert [chunk for kind, chunk in _png_chunks(original) if kind != b"pHYs"] == [
        chunk for kind, chunk in _png_chunks(target.read_bytes()) if kind != b"pHYs"]
    reopened = QtGui.QImage(str(target))
    assert reopened == image
    assert reopened.dotsPerMeterX() == round(600 / .0254)
    assert reopened.dotsPerMeterY() == round(600 / .0254)


@pytest.mark.parametrize("extension", ["png", "pdf", "svg"])
def test_population_interactive_physical_export_is_exact(
        population_window, monkeypatch, tmp_path, extension):
    from nanoscope.gui import plots_panel
    window = population_window
    options = resolve_export_options((800, 400), extension=extension, native=False,
                                     width=50.8, height=25.4, dpi=300)
    out = tmp_path / f"population.{extension}"
    monkeypatch.setattr(plots_panel, "choose_live_plot_export", lambda *a, **k: (out, options))
    before = window._plot_data()
    assert window.save_image() == out
    pd.testing.assert_frame_equal(window._plot_data(), before, check_exact=True)
    if extension == "png":
        image = QtGui.QImage(str(out))
        assert (image.width(), image.height()) == (600, 300)
        assert image.dotsPerMeterX() == round(300 / .0254)
    elif extension == "pdf":
        document = QtPdf.QPdfDocument()
        assert document.load(str(out)) == QtPdf.QPdfDocument.Error.None_
        size = document.pagePointSize(0)
        assert size.width() == pytest.approx(144, abs=.1)
        assert size.height() == pytest.approx(72, abs=.1)
    else:
        import xml.etree.ElementTree as ET
        root = ET.parse(out).getroot()
        assert root.attrib["width"] == "50.8mm"
        assert root.attrib["height"] == "25.4mm"


def test_psd_interactive_export_forwards_physical_size(qtbot, tmp_path, monkeypatch):
    from nanoscope.gui import psd_panel
    window, _ = psd_window(qtbot, tmp_path)
    options = resolve_export_options((800, 400), native=False, width=50.8, height=25.4)
    out = tmp_path / "psd.png"
    monkeypatch.setattr(psd_panel, "choose_live_plot_export", lambda *a, **k: (out, options))
    before = window._psd.copy()
    assert window.save_image() == out
    image = QtGui.QImage(str(out))
    assert (image.width(), image.height()) == (600, 300)
    assert image.dotsPerMeterX() == round(300 / .0254)
    np.testing.assert_array_equal(window._psd, before)
