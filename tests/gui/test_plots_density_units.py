"""Density uses ms and the reciprocal display Jacobian; raw science stays unchanged.

Legacy s/ms/µs recipes are migrated through the real versioned reader. The only
interactive display is milliseconds. Axis ticks show 1000 × stored seconds and
0.001 × stored per-second density, so linear displayed area is one. Log-axis visual
area is not a probability. Raw histogram coordinates/heights, fits and CSV stay in
canonical stored units; typed words remain literal.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import pathlib
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import (PlotsWindow, FrozenMap, plot_config_to_json,
    plot_config_from_json, migrate_plot_quantity)

TIME_UNITS = ("s", "ms", "µs")


def _events(n: int = 982, *, seed: int = 0) -> pd.DataFrame:
    """A full-schema event frame — every histogrammable column, so the unit survey is real.

    Log-normal dwells over roughly the owner's decade, which is what makes the log-x rows
    below a real log axis rather than one bin.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, object] = {}
    for name, (dtype, _unit, _desc) in SCHEMA.items():
        if dtype == "int64":
            cols[name] = np.arange(1, n + 1, dtype=np.int64)
        elif dtype == "bool":
            cols[name] = np.zeros(n, dtype=bool)
        elif dtype == "string":
            cols[name] = pd.Series([""] * n, dtype="string")
        else:
            cols[name] = np.abs(rng.normal(1.0, 0.2, n))
    cols["dwell_s"] = 10.0 ** rng.normal(-2.0, 0.12, n)
    cols["rise_s"] = 10.0 ** rng.normal(-4.0, 0.1, n)
    cols["fall_s"] = 10.0 ** rng.normal(-4.0, 0.1, n)
    cols["t_start_s"] = np.sort(rng.uniform(0.0, 300.0, n))
    cols["baseline_pA"] = np.full(n, 120.0)
    cols["depth_pA"] = np.abs(rng.normal(100.0, 20.0, n))
    cols["frac_depth"] = cols["depth_pA"] / 120.0
    cols["ecd_pA_s"] = cols["depth_pA"] * cols["dwell_s"]
    cols["snr"] = np.abs(rng.normal(9.0, 2.0, n))
    cols["accepted"] = np.ones(n, dtype=bool)
    return pd.DataFrame(cols)


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"apo −700 mV": _events()})
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_bins(30)
    w.set_fit("none")
    w._run_plot()
    return w


def _y_label(window) -> str:
    return window._plot.getAxis("left").labelText.strip()


def _drawn_area(item, xaxis, yaxis) -> float:
    """Area in displayed coordinates; a log-x width is a log width, not time."""
    x, y = item.getData()
    return float((np.asarray(y) * np.diff(np.asarray(x))).sum()) * float(xaxis.scale) * float(yaxis.scale)


def _restore_legacy_unit(window, unit):
    """Real v3 configuration route, before display-basis metadata existed."""
    config = window.config()
    old = replace(config, axes=replace(config.axes, time_unit=unit, quantity_display=FrozenMap({})))
    payload = plot_config_to_json(old)
    payload['version'] = 3
    payload['axes'].pop('quantity_display', None)
    decoded = plot_config_from_json(payload)
    assert decoded.axes.time_unit == unit
    window.apply_config(decoded)
    assert window.config().axes.time_unit == 'ms'
    assert migrate_plot_quantity(window.config()) == window.config()


def _stored_area(item) -> float:
    """∫ y dx over the RAW column. ``xData`` stays in stored units in both x modes."""
    x = np.asarray(item.xData, dtype=np.float64)
    y = np.asarray(item.yData, dtype=np.float64)
    return float((y * np.diff(x)).sum())


# ================================================================================================
# 1. Ticks and labels agree on the displayed ms density; legacy science is retained.
# ================================================================================================

@pytest.mark.parametrize("unit", TIME_UNITS)
@pytest.mark.parametrize("log_x", [False, True])
def test_legacy_density_units_restore_as_ms_ticks_with_inverse_jacobian(window, unit, log_x):
    window.set_normalization("density")
    window.set_log_x(log_x)
    raw = [(item.xData.copy(), item.yData.copy()) for item in window._hist_items]
    _restore_legacy_unit(window, unit)
    assert _y_label(window) == "density (1/ms)"
    assert '(ms' in window._plot.getAxis('bottom').labelText
    assert window._plot.getAxis('left').scale == .001
    xaxis = window._plot.getAxis('bottom')
    assert (xaxis.display_factor if log_x else xaxis.scale) == 1000.
    for item, (x, y) in zip(window._hist_items, raw):
        np.testing.assert_array_equal(item.xData, x)
        np.testing.assert_array_equal(item.yData, y)


def test_the_mode_no_longer_promises_an_area_it_cannot_deliver(window):
    """The name is the persisted value AND the on-screen claim; it stopped making the claim."""
    window.set_normalization("density")
    assert window.normalization() == "density"
    assert "area = 1" not in window.normalization()
    items = [window._norm_combo.itemText(i) for i in range(window._norm_combo.count())]
    assert items == ["count", "density", "probability per bin"]


@pytest.mark.parametrize("col,expected", [
    ("dwell_s", "density (1/ms)"),
    ("t_start_s", "density (1/ms)"),
    ("rise_s", "density (1/ms)"),
    ("depth_pA", "density (1/pA)"),
    ("baseline_pA", "density (1/pA)"),
    # THE ONE COMPOUND UNIT IN THE SCHEMA, and the one branch with no cover from the common
    # columns: ``1/pA·s`` reads as ``(1/pA)·s`` to exactly the reader this label is for, so
    # the reciprocal is parenthesised. A careless simplification here is silent.
    ("ecd_pA_s", "density (1/(pA·s))"),
    # Declared ``sample`` by ``events.SCHEMA`` — honest, and reached through the same one
    # resolver rather than a second table that could drift from it.
    ("start_idx", "density (1/sample)"),
    # Genuinely dimensionless: NO suffix. ``1/`` here would be arithmetic nonsense dressed
    # up as rigour, and these columns never had the defect — they draw at tick scale 1.
    ("frac_depth", "density"),
    ("snr", "density"),
    ("n_levels", "density"),
])
def test_every_column_kind_gets_the_unit_the_schema_declares(window, col, expected):
    window.set_normalization("density")
    window.set_histogram_column(col)
    assert _y_label(window) == expected


def test_log_y_joins_the_units_bracket_instead_of_opening_a_second_one(window):
    """``density (1/ms, log)`` — mirroring the x axis's own ``dwell (ms, log)``.

    The draw site used to concatenate ``" (log)"`` onto the label; with a unit in there
    that produces ``density (1/ms) (log)``, which is why the tag moved inside.
    """
    window.set_normalization("density")
    window.set_log_y(True)
    assert _y_label(window) == "density (1/ms, log)"
    assert ") (" not in _y_label(window)


def test_the_other_two_modes_are_untouched(window):
    """Only a density has a unit to state; ``count`` and ``probability`` are pure numbers."""
    window.set_normalization("count")
    assert _y_label(window) == "count"
    window.set_log_y(True)
    assert _y_label(window) == "count (log)"
    window.set_log_y(False)
    window.set_normalization("probability")
    assert _y_label(window) == "probability"


def test_a_user_label_override_still_replaces_the_text_entirely(window):
    """Deliberate, and matching the x-axis rule: overriding is an act, and silently
    appending a unit to somebody's typed label would be its own small lie. Details still
    states the unit, which is why this is a cost the design accepts rather than a hole."""
    window.set_normalization("density")
    window._y_label_edit.setText("my own label")
    assert _y_label(window) == "my own label"
    assert "a density in 1/s" in window.details_text()


# ================================================================================================
# 2. the exported PICTURE — the half that leaves the app
# ================================================================================================

def test_the_exported_figure_carries_the_unit_on_both_axes(window, tmp_path):
    """``save_image`` renders the live PlotItem, so this file is where a reader of a
    published figure meets the label — and Details, which says the rest, never gets there.

    A regression to a bare ``density`` passes every widget assertion above except this one.
    """
    window.set_normalization("density")
    window.set_log_x(False)
    out = window.save_image(str(tmp_path / "fig.svg"))
    assert out is not None
    text = pathlib.Path(out).read_text(encoding="utf-8", errors="replace")
    assert "dwell (ms)" in text
    assert "density (1/ms)" in text


# ================================================================================================
# 3. Details names the discrepancy — and ONLY where there is one
# ================================================================================================

ALWAYS = "counts / (n × bin width)"
FACTOR = "Integrating the curve"
LOG_NOTE = "AREA UNDER THE DRAWN CURVE is not the integral"


def test_details_names_both_raw_density_and_display_jacobian(window):
    window.set_normalization("density")
    window.set_log_x(False)
    detail = window.details_text()
    assert ALWAYS in detail and "a density in 1/s" in detail
    assert "x ticks read ms" in detail
    assert "inverse Jacobian 1/1000" in detail
    assert "Raw curves and CSV values stay per s" in detail
    assert "linear displayed coordinates is 1" in detail
    assert LOG_NOTE not in detail


def test_linear_displayed_density_is_honest_without_old_factor_warning(window):
    window.set_normalization("density")
    window.set_log_x(False)
    detail = window.details_text()
    assert ALWAYS in detail
    assert "1000, not 1" not in detail
    assert LOG_NOTE not in detail
    assert _drawn_area(window._hist_items[0], window._plot.getAxis('bottom'),
                       window._plot.getAxis('left')) == pytest.approx(1., rel=1e-9)


@pytest.mark.parametrize("unit", TIME_UNITS)
def test_log_axis_details_does_not_claim_probability_after_legacy_restore(window, unit):
    window.set_normalization("density")
    window.set_log_x(True)
    _restore_legacy_unit(window, unit)
    detail = window.details_text()
    assert LOG_NOTE in detail
    assert "log-axis visual area is not probability" in detail
    assert FACTOR not in detail


@pytest.mark.parametrize("unit", TIME_UNITS)
def test_legacy_time_unit_never_scales_a_non_time_column(window, unit):
    window.set_normalization("density")
    window.set_histogram_column("depth_pA")
    window.set_log_x(False)
    _restore_legacy_unit(window, unit)
    assert _y_label(window) == "density (1/pA)"
    assert window._plot.getAxis("bottom").scale == 1.0
    assert window._plot.getAxis("left").scale == 1.0
    assert _drawn_area(window._hist_items[0], window._plot.getAxis("bottom"),
                       window._plot.getAxis("left")) == pytest.approx(1.0, rel=1e-9)
    assert "inverse Jacobian" not in window.details_text()


def test_a_count_histogram_says_none_of_it(window):
    window.set_normalization("count")
    detail = window.details_text()
    assert ALWAYS not in detail and FACTOR not in detail and LOG_NOTE not in detail


def test_a_dimensionless_column_gets_a_dimensionless_sentence(window):
    """``1/`` on a ratio would be nonsense, so neither the label nor Details invents one."""
    window.set_normalization("density")
    window.set_histogram_column("frac_depth")
    detail = window.details_text()
    assert "a dimensionless density" in detail
    assert "1/" not in detail.splitlines()[1]


# ================================================================================================
# 4. AND NOTHING MOVED — the design in one section
# ================================================================================================

@pytest.mark.parametrize("unit", TIME_UNITS)
@pytest.mark.parametrize("log_x", [False, True])
def test_stored_unit_area_remains_one_across_legacy_unit_restore(window, unit, log_x):
    window.set_normalization("density")
    window.set_log_x(log_x)
    _restore_legacy_unit(window, unit)
    for item in window._hist_items:
        assert _stored_area(item) == pytest.approx(1.0, rel=1e-9)


def test_bars_and_raw_csv_are_identical_across_legacy_unit_restore(window, tmp_path):
    window.set_normalization("density")
    window.set_log_x(False)
    item = window._hist_items[0]
    x, y = item.xData.copy(), item.yData.copy()
    table = window._plot_data().copy(deep=True)
    for unit in TIME_UNITS:
        _restore_legacy_unit(window, unit)
        np.testing.assert_array_equal(window._hist_items[0].xData, x)
        np.testing.assert_array_equal(window._hist_items[0].yData, y)
        pd.testing.assert_frame_equal(window._plot_data(), table, check_exact=True)
        path = window.export_data(str(tmp_path / ('density-' + unit + '.csv')))
        assert path is not None
        loaded = pd.read_csv(path)
        for column in ('bin_left', 'bin_right', 'density'):
            np.testing.assert_allclose(loaded[column], table[column], rtol=1e-12, equal_nan=True)


def test_linear_displayed_area_combines_both_axis_scales(window):
    window.set_normalization("density")
    window.set_log_x(False)
    bottom, left = window._plot.getAxis('bottom'), window._plot.getAxis('left')
    assert bottom.scale == 1000. and left.scale == .001
    assert _drawn_area(window._hist_items[0], bottom, left) == pytest.approx(1., rel=1e-9)
    assert _stored_area(window._hist_items[0]) == pytest.approx(1., rel=1e-9)


@pytest.mark.parametrize("unit", TIME_UNITS)
def test_log_visual_area_is_not_probability_after_legacy_units_migrate(window, unit):
    window.set_normalization("density")
    window.set_log_x(True)
    _restore_legacy_unit(window, unit)
    item = window._hist_items[0]
    raw_log_area = float((np.asarray(item.yData) * np.diff(np.log10(item.xData))).sum())
    shown = _drawn_area(item, window._plot.getAxis('bottom'), window._plot.getAxis('left'))
    assert shown == pytest.approx(raw_log_area / 1000., rel=1e-12)
    assert shown != pytest.approx(1., rel=1e-3)
    assert _stored_area(item) == pytest.approx(1., rel=1e-9)


# ================================================================================================
