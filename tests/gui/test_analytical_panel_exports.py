"""Focused parity checks for live analytical image and numeric exports."""
from __future__ import annotations

import os
import runpy
from pathlib import Path

import nanoscope  # noqa: F401 -- Qt bootstrap before the panel imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

from nanoscope.analysis.events import EventSet
from nanoscope.analysis.voltage_dep import VDepPoint
from nanoscope.core.session import Session
from nanoscope.gui import event_plot_panel as event_panel
from nanoscope.gui import psd_panel as psd_panel_module
from nanoscope.gui.event_plot_panel import DENSITY, ENSEMBLE, EventPlotWindow
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.psd_panel import (
    VIEW_ALLPOINTS,
    VIEW_FFT,
    VIEW_PSD,
    VIEW_SPECTROGRAM,
    PSDWindow,
)
from nanoscope.gui.voltage_dep_panel import VDepWindow
from nanoscope.gui.plot_export import PlotDataTooLarge


def _test_helper(filename: str, name: str):
    """Reuse the panel fixtures without relying on an environment's ``tests`` package."""
    return runpy.run_path(str(Path(__file__).with_name(filename)))[name]


_event_window = _test_helper("test_event_plot.py", "_fed_window")
_ohmic_session = _test_helper("test_iv.py", "_ohmic_session")
_psd_window = _test_helper("test_psd.py", "_fed_window")
_ladder_session = _test_helper("test_voltage_dep.py", "_ladder_session")
_vdep_doc = _test_helper("test_voltage_dep.py", "_current_doc")
_vdep_df = _test_helper("test_voltage_dep.py", "_constant_df")
_install_vdep = _test_helper("test_voltage_dep.py", "_install_detection")


def _assert_vector_files(window, tmp_path, stem: str) -> None:
    svg = window.save_image(str(tmp_path / f"{stem}.svg"))
    pdf = window.save_image(str(tmp_path / f"{stem}.pdf"))
    assert svg is not None and svg.read_bytes().lstrip().startswith(b"<?xml")
    assert pdf is not None and pdf.read_bytes().startswith(b"%PDF")


def test_event_views_export_vectors_and_reproducible_csv(qtbot, tmp_path):
    window = _event_window(qtbot, tmp_path, n_events=7)
    window.show()
    window.set_view(DENSITY)
    _assert_vector_files(window, tmp_path, "event-density")
    density = window._plot_data()
    bins = density[density["part"] == "density_bin"]
    events = density[density["part"] == "source_event"]
    assert bins["count"].sum() == 7
    assert set(events["event_row_id"]) == set(range(7))
    assert {"x_left", "x_right", "y_left", "y_right", "provenance_json"} <= set(density)
    out = window.export_data(str(tmp_path / "event-density.csv"))
    assert out is not None and len(pd.read_csv(out)) == len(density)

    window.set_view(ENSEMBLE)
    ensemble = window._plot_data()
    assert {"time", "event_current", "mean_current", "median_current"} <= set(ensemble)
    assert ensemble["identity_available"].all()


def test_psd_four_views_export_vectors_and_exact_numeric_tables(qtbot, tmp_path):
    window, _doc_id = _psd_window(qtbot, tmp_path, n=16_384)
    window.show()
    window.set_view(VIEW_PSD)
    _assert_vector_files(window, tmp_path, "psd")
    psd = window._plot_data()
    assert {"curve", "flicker_fit"} <= set(psd["part"])
    assert {"frequency_hz", "display_value", "psd_value", "source_doc_id"} <= set(psd)
    out = window.export_data(str(tmp_path / "psd.csv"))
    assert out is not None and len(pd.read_csv(out)) == len(psd)

    window.set_view(VIEW_FFT)
    fft = window._plot_data()
    assert not fft.empty and {"frequency_hz", "peak_amplitude", "fft_resolution_hz"} <= set(fft)

    window.set_view(VIEW_SPECTROGRAM)
    spectrogram = window._plot_data()
    assert set(spectrogram["part"]) == {"metadata", "spectrogram_cell"}
    assert {"time_index", "frequency_index", "psd_value", "display_log10_psd"} <= set(spectrogram)
    # The colour bar is a sibling item; vector-export the whole visible scene.
    _assert_vector_files(window, tmp_path, "spectrogram")

    window.set_view(VIEW_ALLPOINTS)
    histogram = window._plot_data()
    assert "histogram_bin" in set(histogram["part"])
    assert {"bin_left", "bin_right", "count", "histogram_n_total"} <= set(histogram)


def test_iv_exports_vectors_points_fit_and_source_identity(qtbot, tmp_path):
    window = IVWindow(_ohmic_session(tmp_path, G_nS=80.0, leak_pA=3.0))
    qtbot.addWidget(window)
    window.show()
    _assert_vector_files(window, tmp_path, "iv")
    data = window._plot_data()
    assert {"point", "fit"} <= set(data["part"])
    assert data.loc[data["part"] == "point", "source_doc_id"].notna().all()
    assert {"voltage_V", "current", "fit_slope", "fit_intercept"} <= set(data)
    out = window.export_data(str(tmp_path / "iv.csv"))
    assert out is not None and len(pd.read_csv(out)) == len(data)


def test_voltage_dependence_exports_vectors_points_fit_and_provenance(qtbot, tmp_path):
    window = VDepWindow(_ladder_session(tmp_path))
    qtbot.addWidget(window)
    window.show()
    window.set_fit("exponential")
    _assert_vector_files(window, tmp_path, "voltage-dependence")
    data = window._plot_data()
    assert {"point", "fit"} <= set(data["part"])
    assert {"voltage_V", "metric_value", "baseline_fingerprint",
            "merge_state", "calibration_fingerprint_json"} <= set(data)
    out = window.export_data(str(tmp_path / "voltage-dependence.csv"))
    assert out is not None and len(pd.read_csv(out)) == len(data)


def test_all_four_panels_refuse_empty_csv(qtbot, tmp_path, monkeypatch):
    # A programmatic export follows the same refusal path as the dialog. Suppress the
    # modal warning here; the assertion is that no plausible empty file is left behind.
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning", lambda *args, **kwargs: None)
    windows = [EventPlotWindow(Session()), PSDWindow(Session()),
               IVWindow(Session()), VDepWindow(Session())]
    for index, window in enumerate(windows):
        qtbot.addWidget(window)
        target = tmp_path / f"empty-{index}.csv"
        assert window._plot_data().empty
        assert window.export_data(str(target)) is None
        assert not target.exists()


def test_event_ensemble_budget_refuses_before_long_table_allocation(qtbot, tmp_path,
                                                                    monkeypatch):
    window = _event_window(qtbot, tmp_path, n_events=5)
    window.set_view(ENSEMBLE)
    _time, matrix, _mean, _median = window.ensemble_traces()
    monkeypatch.setattr(event_panel, "MAX_ENSEMBLE_CSV_ROWS", matrix.size - 1)
    with pytest.raises(PlotDataTooLarge, match=r"5 events × .*points") as caught:
        window._plot_data()
    assert "No values were sampled or omitted" in str(caught.value)

    warnings: list[str] = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda _parent, _title, body: warnings.append(body))
    target = tmp_path / "too-large-ensemble.csv"
    assert window.export_data(str(target)) is None
    assert not target.exists()
    assert warnings and "Lower Ensemble" in warnings[-1]


def test_spectrogram_budget_refuses_before_grid_index_allocation(qtbot, tmp_path,
                                                                  monkeypatch):
    window, _doc_id = _psd_window(qtbot, tmp_path, n=16_384)
    window.set_view(VIEW_SPECTROGRAM)
    cells = int(window.spectrogram().psd.size)
    monkeypatch.setattr(psd_panel_module, "MAX_SPECTROGRAM_CSV_ROWS", cells - 1)
    # Both million-scale index arrays are below the budget check; if either is reached,
    # this deliberately fails instead of allocating it.
    monkeypatch.setattr(psd_panel_module.np, "repeat",
                        lambda *args, **kwargs: pytest.fail("allocated grid indices"))
    with pytest.raises(PlotDataTooLarge, match="spectrogram grid") as caught:
        window._plot_data()
    assert f"{cells:,}" in str(caught.value)
    assert "Shorten the analysed Segment" in str(caught.value)


def test_duplicate_event_row_labels_keep_positional_event_id_mapping(qtbot, tmp_path):
    window = _event_window(qtbot, tmp_path, n_events=4)
    frame = window._eventset.df.copy()
    frame.index = [7, 7, 9, 9]
    source = EventSet(frame, dict(window._eventset.provenance))
    window.set_source(window._doc_id, window._channel, source)
    window.set_view(ENSEMBLE)
    data = window._plot_data()
    first_samples = data[data["sample_index"] == 0].sort_values("trace_index")
    assert first_samples["identity_available"].all()
    assert first_samples["event_row_id"].tolist() == [7, 7, 9, 9]
    assert first_samples["event_id"].tolist() == frame["event_id"].tolist()


def test_voltage_dependence_duplicate_labels_export_stable_doc_ids(qtbot, tmp_path):
    session = Session()
    ids = []
    for folder, voltage in (("left", -500.0), ("right", -700.0)):
        doc_id = session.add_document(_vdep_doc(
            tmp_path / folder, nominal_mV=voltage, name="same.dat"))
        _install_vdep(session, doc_id, _vdep_df(
            n=4, dwell_s=1e-3, depth_pA=50.0, frac_depth=0.05), duration_s=1.0)
        ids.append(doc_id)
    window = VDepWindow(session)
    qtbot.addWidget(window)

    assert [point.recording for point in window.points()] == ["same.dat", "same.dat"]
    assert [point.doc_id for point in window.points()] == ids
    points = window._plot_data().query("part == 'point'")
    assert points["source_doc_id"].tolist() == ids
    assert points["source_identity_available"].all()
    # Positional construction used throughout the older corpus remains compatible.
    assert VDepPoint(-0.5, 1.0, "legacy.dat").doc_id == ""
