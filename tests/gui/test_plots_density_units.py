"""S6-03 — the density histogram says what its numbers are IN, and nothing else moved.

THE DEFECT. The *density* mode's heights are ``counts / (n * bin_width)`` with the width
in the column's STORED unit — per second for a dwell, per pA for a depth. ``_axis_label_for``
re-scales the x TICKS for display and deliberately never touches the sample (the exponential
fit takes its detection floor from the detector's ``min_dwell_s``, in seconds, so scaling the
data would bias tau silently). The mode was nevertheless called ``density (area = 1)`` and the
y axis said a bare ``density``, so the area under the curve AS DRAWN was — measured on the
owner's −700 mV apo run, 982 events, ``dwell_s``, 30 bins:

    linear x:  1 (Time unit = s)   1000 (ms, the app's default)   1e6 (µs)
    log x:     23.3011485 — IN ALL THREE UNITS

THE FIX IS NOT A RESCALE, and the assertions here are shaped by that. On a log axis the
drawn abscissa is log10(seconds): the drawn area is a property of the data's RANGE and no
constant factor can make it 1, and the app's own shipped *Dwell-time histogram (log-x,
exponential)* template selects exactly that case. So the FIGURE states its unit and the mode
stops promising an area it cannot keep, while every drawn number stays where it was.

What that makes these tests check, in order:

* the y axis names the stored unit — and does NOT follow the Time-unit combo. That is the
  assertion that distinguishes this design from a rescale, and the one that would fail first
  if somebody "helpfully" made the label track the ticks;
* the unit comes from ``_column_quantity`` — the module's one schema-first resolver — so it is
  right for a depth, a charge and an index as well as a dwell, and absent where the column is
  genuinely dimensionless;
* THE PICTURE, not the widget: ``save_image`` renders the live PlotItem, so the exported SVG
  is where a reader of a published figure meets this label. A regression to a bare ``density``
  would pass every widget-level assertion in this file except that one;
* Details names the ×factor where the drawn area is not 1, says nothing where it IS 1, and on
  a log axis says something different in kind — because there the discrepancy is not a factor;
* and THE VALUES ARE UNCHANGED. If an area assertion in here ever moves, the label design has
  been quietly replaced by a rescale, which is the thing that was measured and rejected.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow

#: The owner's own detected events. Present on his machine, absent on CI — so the numeric
#: pins measured off it live in ONE test that skips, and every structural assertion below
#: runs on the synthetic frame instead. A guard that only ever runs on one laptop is not a
#: guard.
REAL_EVENTS = pathlib.Path(
    "/Users/abrahamcampos/Desktop/nanoscope-example/events/events_nm_0005.csv")

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
    return w


def _y_label(window) -> str:
    return window._plot.getAxis("left").labelText.strip()


def _drawn_area(item, axis) -> float:
    """∫ y dx over the abscissa AS DRAWN — log10(seconds) when the x axis is log."""
    x, y = item.getData()
    return float((np.asarray(y) * np.diff(np.asarray(x))).sum()) * float(axis.scale)


def _stored_area(item) -> float:
    """∫ y dx over the RAW column. ``xData`` stays in stored units in both x modes."""
    x = np.asarray(item.xData, dtype=np.float64)
    y = np.asarray(item.yData, dtype=np.float64)
    return float((y * np.diff(x)).sum())


# ================================================================================================
# 1. the y axis names the unit — and does NOT follow the ticks
# ================================================================================================

@pytest.mark.parametrize("unit", TIME_UNITS)
@pytest.mark.parametrize("log_x", [False, True])
def test_the_density_axis_reads_per_second_whatever_the_ticks_say(window, unit, log_x):
    """THE assertion that separates this design from a rescale.

    The heights are per second in all six of these states — the sample is never re-scaled —
    so the label must be the same in all six. A label that tracked the Time-unit combo would
    read ``1/ms`` over numbers that are per second: a second, subtler lie than the one S6-03
    closed, and one no test would otherwise catch, because the *bars* would look right.
    """
    window.set_normalization("density")
    window.set_time_unit(unit)
    window.set_log_x(log_x)

    assert _y_label(window) == "density (1/s)"
    # ...while the x axis DOES follow it, which is the disagreement the label exists to state.
    assert window._plot.getAxis("bottom").labelText.startswith(f"dwell ({unit}")


def test_the_mode_no_longer_promises_an_area_it_cannot_deliver(window):
    """The name is the persisted value AND the on-screen claim; it stopped making the claim."""
    window.set_normalization("density")
    assert window.normalization() == "density"
    assert "area = 1" not in window.normalization()
    items = [window._norm_combo.itemText(i) for i in range(window._norm_combo.count())]
    assert items == ["count", "density", "probability per bin"]


@pytest.mark.parametrize("col,expected", [
    ("dwell_s", "density (1/s)"),
    ("t_start_s", "density (1/s)"),
    ("rise_s", "density (1/s)"),
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
    """``density (1/s, log)`` — mirroring the x axis's own ``dwell (ms, log)``.

    The draw site used to concatenate ``" (log)"`` onto the label; with a unit in there
    that produces ``density (1/s) (log)``, which is why the tag moved inside.
    """
    window.set_normalization("density")
    window.set_log_y(True)
    assert _y_label(window) == "density (1/s, log)"
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
    window.set_time_unit("ms")
    window.set_log_x(False)
    out = window.save_image(str(tmp_path / "fig.svg"))
    assert out is not None
    text = pathlib.Path(out).read_text(encoding="utf-8", errors="replace")
    assert "dwell (ms)" in text
    assert "density (1/s)" in text


# ================================================================================================
# 3. Details names the discrepancy — and ONLY where there is one
# ================================================================================================

ALWAYS = "counts / (n × bin width)"
FACTOR = "Integrating the curve"
LOG_NOTE = "AREA UNDER THE DRAWN CURVE is not the integral"


def test_details_names_the_factor_on_a_rescaled_linear_axis(window):
    window.set_normalization("density")
    window.set_log_x(False)

    window.set_time_unit("ms")
    detail = window.details_text()
    assert ALWAYS in detail and "a density in 1/s" in detail
    assert "the x ticks read ms" in detail
    assert "1000, not 1" in detail

    window.set_time_unit("µs")
    assert "1e+06, not 1" in window.details_text()


def test_details_says_nothing_when_the_picture_is_already_honest(window):
    """A pane that cries wolf is not read. Time unit = s on a linear axis IS area = 1."""
    window.set_normalization("density")
    window.set_log_x(False)
    window.set_time_unit("s")
    detail = window.details_text()
    assert ALWAYS in detail              # the denominator is always worth stating
    assert FACTOR not in detail          # ...the correction is not
    assert LOG_NOTE not in detail


@pytest.mark.parametrize("unit", TIME_UNITS)
def test_on_a_log_axis_details_says_something_different_in_kind(window, unit):
    """Not a bigger factor — a different statement, and this is the case a rescale design
    could not have covered at all: the drawn log-x area is 23.3011485 in ALL THREE units,
    because the abscissa is a logarithm. Quoting a number here would be true only of this
    dataset's range and would invite exactly the wrong correction."""
    window.set_normalization("density")
    window.set_time_unit(unit)
    window.set_log_x(True)
    detail = window.details_text()
    assert LOG_NOTE in detail
    assert FACTOR not in detail


@pytest.mark.parametrize("unit", TIME_UNITS)
def test_a_non_time_column_is_silent_because_it_never_had_the_defect(window, unit):
    """The Time-unit combo does not touch a pA axis, so a depth histogram DOES integrate
    to 1 as drawn, in every unit — and Details must say nothing about a factor there.

    Worth pinning because it is the shape of the report this fix will generate ("it didn't
    apply on my depth histogram"): the label appears, the correction does not, and both are
    right. It also pins the ``col in _TIME_COLUMNS`` guard, without which the factor
    sentence would fire on every column the moment the user picked ms for a dwell elsewhere.
    """
    window.set_normalization("density")
    window.set_histogram_column("depth_pA")
    window.set_time_unit(unit)
    window.set_log_x(False)

    assert _y_label(window) == "density (1/pA)"
    assert window._plot.getAxis("bottom").scale == 1.0
    assert _drawn_area(window._hist_items[0],
                       window._plot.getAxis("bottom")) == pytest.approx(1.0, rel=1e-9)
    assert FACTOR not in window.details_text()


def test_a_count_histogram_says_none_of_it(window):
    window.set_normalization("count")
    window.set_time_unit("ms")
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
def test_the_stored_unit_area_is_still_exactly_one(window, unit, log_x):
    """If this ever fails, the label design has been replaced by a rescale.

    ∫ y dx = 1 over the RAW column is what the CSV, the fit curve, the axis-range pin and
    seven assertions in ``test_overlay_plots.py`` are all pinned to. It is the invariant the
    y label now merely NAMES.
    """
    window.set_normalization("density")
    window.set_time_unit(unit)
    window.set_log_x(log_x)
    for item in window._hist_items:
        assert _stored_area(item) == pytest.approx(1.0, rel=1e-9)


def test_the_bars_are_identical_across_time_units(window):
    """The Time-unit combo is a DISPLAY control: it may move ticks and labels, never heights.

    Asserted as byte-equality of the height array rather than as a tolerance, because a
    rescale would be a clean factor of 1000 and would sail through any approx().
    """
    window.set_normalization("density")
    window.set_log_x(False)
    heights, peaks = [], []
    for unit in TIME_UNITS:
        window.set_time_unit(unit)
        item = window._hist_items[0]
        heights.append(np.array(item.yData, dtype=np.float64, copy=True))
        peaks.append(float(np.max(item.yData)))
    assert np.array_equal(heights[0], heights[1])
    assert np.array_equal(heights[0], heights[2])
    assert peaks[0] == peaks[1] == peaks[2]


def test_the_drawn_area_is_what_details_says_it_is(window):
    """The honest cost, pinned. The picture does NOT integrate to 1 on the app's default
    ms axis — it integrates to 1000, and Details says 1000. This test exists so that number
    cannot drift away from the sentence without something failing."""
    window.set_normalization("density")
    window.set_log_x(False)
    axis = window._plot.getAxis("bottom")

    window.set_time_unit("s")
    assert _drawn_area(window._hist_items[0], axis) == pytest.approx(1.0, rel=1e-9)

    window.set_time_unit("ms")
    assert _drawn_area(window._hist_items[0], axis) == pytest.approx(1000.0, rel=1e-9)
    assert "1000, not 1" in window.details_text()


@pytest.mark.parametrize("unit", TIME_UNITS)
def test_the_log_x_drawn_area_is_the_same_number_in_every_unit(window, unit):
    """The measurement that decided the design: on a log axis the drawn area does not
    depend on the time unit AT ALL, so no constant factor could ever normalise it. Pinned
    here as an equality between units rather than as a literal, because the value is a
    property of the sample's range."""
    window.set_normalization("density")
    window.set_log_x(True)
    window.set_time_unit("s")
    reference = _drawn_area(window._hist_items[0], window._plot.getAxis("bottom"))
    window.set_time_unit(unit)
    assert _drawn_area(window._hist_items[0],
                       window._plot.getAxis("bottom")) == pytest.approx(reference, rel=1e-12)
    assert reference != pytest.approx(1.0, rel=1e-3), (
        "a log-x histogram whose drawn area IS 1 would make this whole test vacuous")


# ================================================================================================
# 5. the owner's own file — the numbers the report quotes, pinned where they exist
# ================================================================================================

@pytest.mark.skipif(not REAL_EVENTS.exists(),
                    reason="the owner's example events are not on this machine")
def test_the_measured_numbers_on_the_real_982_event_run(qapp, qtbot):
    """Every figure report S6-03 quotes, re-measured through the shipped window.

    Kept apart from the rest of the file, and skipped rather than approximated, because a
    pin on real data is only meaningful against THAT data.
    """
    window = PlotsWindow(Session())
    qtbot.addWidget(window)
    window.set_data({"−700 mV": pd.read_csv(REAL_EVENTS)})
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    window.set_bins(30)
    window.set_fit("none")
    window.set_normalization("density")

    axis = window._plot.getAxis("bottom")
    for unit, expected_drawn in (("s", 1.0), ("ms", 1000.0), ("µs", 1e6)):
        window.set_time_unit(unit)
        window.set_log_x(False)
        item = window._hist_items[0]
        assert float(np.max(item.yData)) == pytest.approx(160.876232, abs=1e-6)
        assert _drawn_area(item, axis) == pytest.approx(expected_drawn, rel=1e-9)
        assert _stored_area(item) == pytest.approx(1.0, rel=1e-12)
        assert _y_label(window) == "density (1/s)"

        window.set_log_x(True)
        item = window._hist_items[0]
        assert float(np.max(item.yData)) == pytest.approx(154.195604, abs=1e-6)
        assert _drawn_area(item, axis) == pytest.approx(23.3011485, rel=1e-8)
        assert _stored_area(item) == pytest.approx(1.0, rel=1e-12)
        assert _y_label(window) == "density (1/s)"
