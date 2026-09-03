"""The comparability rule has THREE halves, and there is no way to ask for two.

REGRESSION FOR apo-run-2026-08-22 ``cross/README.md`` §0 [NEW, Medium] / §7.A, and
for the M12 item beside it.

Measured through the real GUI on the owner's three apoferritin recordings, all
re-detected under ONE common baseline window (2.0 / 0.5 s) -- the state the app's
OWN warning tells the user to reach:

* ``Analysis ▸ Voltage dependence ▸ Median depth`` fitted
  ``V₀ = -0.432 V, R² = 0.9157`` over three UNCALIBRATED pores, on an axis labelled
  ``Median depth`` with no unit, and said **nothing**: ``warning_text() == ""``,
  ``len(warnings()) == 0``, title ``V₀ = -0.432 V · R² = 0.9157 · merged: gap 10 ms
  · baseline sha256:ce8ae31f10bd``;
* ``Analysis ▸ Plots`` on ``depth_pA``, the same three recordings, the same moment,
  printed *"CAUTION — comparing depth_pA across recordings: The current gain is
  UNCALIBRATED …"*.

Two surfaces, one rule, opposite answers -- because ``core.sources`` was made the
rule's single home but ``pool_comparability`` answered only two thirds of it, and
a two-thirds answer is something a caller can mistake for the whole one. Plots
asked it AND ``gain_comparability``; the voltage-dependence window asked it alone.

What is pinned here:

  1. **The half-answer is gone.** :func:`pool_comparability` judges gain as well as
     baseline and merge, from the fingerprint carried on the :class:`PooledItem`.
  2. **The two surfaces agree**, asserted by comparing the sentences they produce
     for the same three recordings -- not by inspecting one of them.
  3. **No second call site.** ``voltage_dep`` and its window never reach for
     ``gain_comparability`` / ``gains_comparable``; there is one call, and it
     returns everything.
  4. **The advice adapts on BOTH surfaces**, not just on Plots: dI/I₀ is offered
     back exactly when the baselines agree.
  5. **The per-recording detection caveats travel with the point** (M12).

Reverted by ``make_revert.py`` in the M3 scratch: every assertion below fails
against that out-of-tree copy of the package with these hunks undone.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

import nanoscope
from nanoscope.analysis import voltage_dep
from nanoscope.core import sources as src_mod
from nanoscope.core.sources import (
    ABSOLUTE_CURRENT_COLUMNS,
    GAIN_SAFE_ALTERNATIVES,
    UNMERGED,
    PlotSource,
    PooledItem,
    pool_comparability,
    source_comparability,
)

# The names this fix ADDS are reached through the module, never imported at the top:
# an out-of-tree revert must fail these tests one by one, on the assertion each is
# about, not vanish into a single collection error that proves nothing about any of
# them.
GAIN_UNSTATED = getattr(src_mod, "GAIN_UNSTATED", "<absent: GAIN_UNSTATED>")

# WHICH TREE THIS RAN AGAINST. Named in every failure below, because a "revert" that
# is really a stub with the wrong signature manufactures failures that look real.
PKG = pathlib.Path(nanoscope.__file__).resolve().parent
WHERE = f"(nanoscope resolved from {PKG})"

# The three real fingerprints from the apoferritin run; FP_600 is the one all three
# recordings carry once they are re-detected under the common 2.0 / 0.5 s window.
FP_500 = "sha256:b64a75ebdf29"
FP_600 = "sha256:ce8ae31f10bd"
FP_700 = "sha256:1a29aabf5fb1"

#: What every recording in this project is in until somebody calibrates it.
UNCAL = (1.0, 0.0)


def test_this_ran_against_a_real_nanoscope_source_tree():
    """The guard on every other assertion in this file."""
    for rel in ("core/sources.py", "analysis/voltage_dep.py", "gui/voltage_dep_panel.py"):
        path = PKG / rel
        assert path.is_file(), f"{rel} missing {WHERE}"
        assert path.stat().st_size > 10_000, f"{rel} is a stub {WHERE}"


# ==================================================================================
# 1. the half-answer is unavailable: ONE call, all three halves
# ==================================================================================

def _items(*fingerprints, baseline=FP_600, merged=UNMERGED):
    return [PooledItem(label=chr(ord("a") + i), baseline_fingerprint=baseline,
                       merge_state=merged, calibration_fingerprint=fp)
            for i, fp in enumerate(fingerprints)]


def test_pool_comparability_judges_the_gain_not_only_the_baseline_and_merge():
    """THE M3 ROOT. One call, and the gain answer is IN it."""
    v = pool_comparability(_items(UNCAL, UNCAL), quantity="median depth",
                           depends_on_baseline=True)
    assert v.gain_warnings, f"the gain half did not fire {WHERE}"
    assert v.gain_warnings[0] in v.warnings, (
        f"the gain warning is not in the list a surface renders {WHERE}")
    assert not v.comparable, WHERE
    assert v.gain_groups == (("uncalibrated", ("a", "b")),), WHERE


def test_a_gain_free_quantity_is_silent_on_the_gain_across_the_same_items():
    """PER QUANTITY, not blanket -- a check that always fires is not a check."""
    v = pool_comparability(_items(UNCAL, UNCAL), quantity="median dI/I0",
                           depends_on_baseline=True, depends_on_gain=False)
    assert v.warnings == (), WHERE
    assert v.gain_warnings == (), WHERE
    # ... and the calibrations are still REPORTED, just not held against it.
    assert v.gain_groups == (("uncalibrated", ("a", "b")),), WHERE


def test_one_shared_real_calibration_pools_cleanly():
    v = pool_comparability(_items((2.5, -3.0), (2.5, -3.0)), quantity="median depth",
                           depends_on_baseline=True)
    assert v.comparable, WHERE
    assert v.gain_groups == (("gain 2.5, offset -3", ("a", "b")),), WHERE


def test_two_different_calibrations_are_not_comparable_either():
    v = pool_comparability(_items((2.5, 0.0), (0.1, 0.0)), quantity="median depth",
                           depends_on_baseline=True)
    assert v.gain_warnings and "do NOT share a current calibration" in v.gain_warnings[0], WHERE
    assert len(v.gain_groups) == 2, WHERE


def test_an_item_that_states_no_calibration_takes_no_part_in_the_gain_question():
    """The manual-measurement table is not a second amplifier -- and says so by
    stating no gain, rather than by a kind-check every future surface must repeat."""
    v = pool_comparability(
        [PooledItem("detection", FP_600, UNMERGED, UNCAL),
         PooledItem("Manual measurements", FP_600, UNMERGED, GAIN_UNSTATED)],
        quantity="depth_pA", depends_on_baseline=True)
    assert v.warnings == (), f"one amplifier cannot disagree with itself {WHERE}"
    assert v.gain_groups == (("uncalibrated", ("detection",)),), WHERE


def test_a_pool_that_states_no_calibration_at_all_is_silent_on_gain():
    """The legacy shape: nothing stated, so there is nothing to compare."""
    v = pool_comparability([PooledItem("a", FP_600), PooledItem("b", FP_600)],
                           quantity="median depth", depends_on_baseline=True)
    assert v.warnings == (), WHERE
    assert v.gain_groups == (), WHERE


def test_the_gain_half_arrives_last_most_fundamental_first():
    """Merge, then baseline, then gain -- two recordings measured under different
    windows are incomparable even when they ARE in the same units."""
    v = pool_comparability(
        [PooledItem("a", FP_500, UNMERGED, UNCAL),
         PooledItem("b", FP_700, "merged: gap 10 ms", UNCAL)],
        quantity="median depth", depends_on_baseline=True)
    assert len(v.warnings) == 3, f"{v.warnings} {WHERE}"
    assert "merge state" in v.warnings[0], WHERE
    assert "estimated baseline" in v.warnings[1], WHERE
    assert v.warnings[2] == v.gain_warnings[0], WHERE


# ==================================================================================
# 2. the two surfaces reach the SAME verdict on the same recordings
# ==================================================================================

def _plot_source(label, fingerprint, gain=UNCAL, kind="detection"):
    return PlotSource(key=label, doc_key=label, label=label, df=None,
                      provenance={"baseline_fingerprint": fingerprint},
                      kind=kind, calibration_fingerprint=gain, display_name=label)


def _why(warning: str) -> str:
    """The gain sentence itself, without each surface's own naming of the quantity."""
    return warning.split(": ", 1)[1]


@pytest.mark.parametrize("baselines,label", [
    ((FP_600, FP_600, FP_600), "one common window"),
    ((FP_500, FP_600, FP_700), "three windows, as probed"),
])
def test_plots_and_voltage_dependence_say_the_same_thing_about_the_same_pores(
        baselines, label):
    """THE INVARIANT. The two pooling surfaces, the owner's three recordings, one
    absolute-current quantity: the same verdict, in the same words."""
    names = ("apo_-500mV.abf", "apo_-600mV.abf", "apo_-700mV.abf")
    plots = source_comparability(
        [_plot_source(n, fp) for n, fp in zip(names, baselines)], ["depth_pA"])
    vdep = voltage_dep.vdep_comparability(
        [voltage_dep.VDepPoint(-0.5 - 0.1 * i, 60.0 + 10.0 * i, n, fp, UNMERGED, UNCAL)
         for i, (n, fp) in enumerate(zip(names, baselines))],
        metric="median_depth")

    plots_gain = [w for w in plots.warnings if w.startswith("CAUTION")]
    vdep_gain = [w for w in vdep.warnings if w.startswith("CAUTION")]
    assert len(plots_gain) == 1, f"Plots said nothing about the gain, {label} {WHERE}"
    assert len(vdep_gain) == 1, (
        f"Voltage dependence said nothing about the gain, {label} -- "
        f"warnings={vdep.warnings} {WHERE}")
    assert _why(vdep_gain[0]) == _why(plots_gain[0]).replace("dwell_s", "median dwell").replace(
        "frac_depth (dI/I0)", "median dI/I0"), (
        f"the two surfaces gave different gain verdicts, {label}:\n"
        f"  Plots: {plots_gain[0]}\n  V-dep: {vdep_gain[0]}\n{WHERE}")
    # ... and they agree about how many things are wrong, not merely that something is.
    assert len(plots.warnings) == len(vdep.warnings), (
        f"{plots.warnings} vs {vdep.warnings}, {label} {WHERE}")


def test_the_gain_advice_adapts_on_the_voltage_window_too():
    """It already did this on Plots. dI/I₀ divides BY the baseline, so it is advice
    only while the baselines agree -- on EITHER surface."""
    agree = voltage_dep.vdep_comparability(
        [voltage_dep.VDepPoint(-0.5, 60.0, "a", FP_600, UNMERGED, UNCAL),
         voltage_dep.VDepPoint(-0.7, 90.0, "b", FP_600, UNMERGED, UNCAL)],
        metric="median_depth").warnings
    differ = voltage_dep.vdep_comparability(
        [voltage_dep.VDepPoint(-0.5, 60.0, "a", FP_500, UNMERGED, UNCAL),
         voltage_dep.VDepPoint(-0.7, 90.0, "b", FP_700, UNMERGED, UNCAL)],
        metric="median_depth").warnings

    caution_agree = [w for w in agree if w.startswith("CAUTION")][0]
    caution_differ = [w for w in differ if w.startswith("CAUTION")][0]
    assert "median dwell and median dI/I0 ARE comparable" in caution_agree, WHERE
    assert "dI/I0" not in caution_differ, (
        f"redirected onto the metric the baseline warning just blocked {WHERE}")
    assert "median dwell IS comparable" in caution_differ, WHERE


def test_every_alternative_this_window_offers_really_is_gain_free():
    """The advice is only worth giving if it is true, per the module's OWN rule."""
    from nanoscope.analysis.voltage_dep import _GAIN_ALTERNATIVES, _MEDIAN_COLUMN, _METRIC_NAME

    assert tuple(_GAIN_ALTERNATIVES) == ("median dwell", "median dI/I0"), (
        f"the window offers {tuple(_GAIN_ALTERNATIVES)} {WHERE}")
    by_name = {name: metric for metric, name in _METRIC_NAME.items()}
    for name in _GAIN_ALTERNATIVES:
        metric = by_name[name]
        assert not voltage_dep.depends_on_gain(metric), f"{name} is an absolute current {WHERE}"
        col = _MEDIAN_COLUMN.get(metric, "")
        assert col not in ABSOLUTE_CURRENT_COLUMNS, WHERE
    # and the flag really is "does it also survive a baseline disagreement?"
    for name, baseline_free in _GAIN_ALTERNATIVES.items():
        assert baseline_free is not voltage_dep.depends_on_baseline(by_name[name]), WHERE


def test_the_plots_alternatives_keep_their_published_names():
    """``GAIN_SAFE_ALTERNATIVES`` is public and the PSD window's sibling call reads
    the same default; folding it into a mapping must not have renamed anything."""
    assert GAIN_SAFE_ALTERNATIVES == ("dwell_s", "frac_depth (dI/I0)"), WHERE
    assert tuple(src_mod.GAIN_ALTERNATIVES) == GAIN_SAFE_ALTERNATIVES, WHERE
    assert src_mod.GAIN_ALTERNATIVES["dwell_s"] is True, WHERE
    assert src_mod.GAIN_ALTERNATIVES["frac_depth (dI/I0)"] is False, WHERE


# ==================================================================================
# 3. no second call site -- structurally
# ==================================================================================

@pytest.mark.parametrize("module_name", [
    "nanoscope.analysis.voltage_dep",
    "nanoscope.gui.voltage_dep_panel",
])
def test_the_voltage_surface_never_reaches_for_a_third_of_the_rule(module_name):
    """The shape of the M3 defect: one caller asking for a SUBSET. There is now one
    call that returns everything, and reaching past it must stay unwritable."""
    import importlib

    mod = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(mod))
    called = {node.func.attr if isinstance(node.func, ast.Attribute)
              else getattr(node.func, "id", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for half in ("gain_comparability", "gains_comparable"):
        assert half not in called, (
            f"{module_name} calls {half} directly; ask pool_comparability, which "
            f"asks it for you with the baseline verdict in hand {WHERE}")


def test_the_column_classification_has_one_home_and_both_surfaces_ask_it():
    """``depth_pA`` may not be an absolute current on one surface and a ratio on
    the other, so neither surface owns the list."""
    assert voltage_dep.column_depends_on_gain is src_mod.column_depends_on_gain, WHERE
    assert src_mod.column_depends_on_gain("depth_pA") is True, WHERE
    assert src_mod.column_depends_on_gain("frac_depth") is False, WHERE
    assert src_mod.column_depends_on_gain("dwell_s") is False, WHERE
    for col in ABSOLUTE_CURRENT_COLUMNS:
        assert src_mod.column_depends_on_gain(col) is True, f"{col} {WHERE}"

    pkg = pathlib.Path(src_mod.__file__).resolve().parent.parent
    hits = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*.py")
                  if "ABSOLUTE_CURRENT_COLUMNS: tuple" in p.read_text())
    assert hits == ["core/sources.py"], f"{hits} {WHERE}"


def test_depends_on_gain_answers_per_metric_and_rejects_an_unknown_one():
    assert voltage_dep.depends_on_gain("median_depth") is True, WHERE
    for metric in ("capture_rate_hz", "median_dwell_s", "median_frac_depth", "n_events"):
        assert voltage_dep.depends_on_gain(metric) is False, f"{metric} {WHERE}"
    with pytest.raises(ValueError, match="unknown metric"):
        voltage_dep.depends_on_gain("median_snr")


# ==================================================================================
# 4. M12 -- a detection's own caveats travel with the point it produced
# ==================================================================================

_M700 = ("the tracked baseline sits 4.20 pA INSIDE the states, measured against the "
         "open pore 5-55 ms either side of each of 982 of them. That is 4.2% of their "
         "own 98.4 pA excursion, and it comes off every depth this baseline measures: "
         "depth_pA and frac_depth read about 4% low, and a voltage series measured "
         "this way puts one recording out of line with the rest.")


def test_a_point_carries_its_detections_own_warnings():
    """M12's root. This caveat reached the Summary panel and the Plots details pane
    and stopped there -- never the window where that recording becomes one term in
    a fitted law, which is the one place it changes a law."""
    series = voltage_dep.vdep_comparability(
        [voltage_dep.VDepPoint(-0.5, 62.8, "apo_-500mV.abf", FP_600, UNMERGED, UNCAL),
         voltage_dep.VDepPoint(-0.7, 99.8, "apo_-700mV.abf", FP_600, UNMERGED, UNCAL,
                               (_M700,))],
        metric="median_depth")
    assert series.detection_warnings == (("apo_-700mV.abf", _M700),), WHERE


def test_a_detection_caveat_does_not_pretend_to_be_a_comparability_verdict():
    """They are different claims: one point is biased, versus these may not be
    pooled. Conflating them would make the ⚠ on the fit mean two things."""
    series = voltage_dep.vdep_comparability(
        [voltage_dep.VDepPoint(-0.5, 62.8, "a", FP_600, UNMERGED, GAIN_UNSTATED),
         voltage_dep.VDepPoint(-0.7, 99.8, "b", FP_600, UNMERGED, GAIN_UNSTATED,
                               (_M700,))],
        metric="median_dwell_s")
    assert series.warnings == (), WHERE
    assert series.comparable is True, WHERE
    assert len(series.detection_warnings) == 1, WHERE


def test_no_caveats_means_no_block():
    series = voltage_dep.vdep_comparability(
        [voltage_dep.VDepPoint(-0.5, 1.0, "a", FP_600, UNMERGED, UNCAL)],
        metric="median_depth")
    assert series.detection_warnings == (), WHERE


# ==================================================================================
# 5. the gain label -- what the table cell and the title both read
# ==================================================================================

def test_the_identity_gain_reads_as_uncalibrated_not_as_a_measurement():
    assert src_mod.gain_label((1.0, 0.0)) == "uncalibrated", WHERE
    assert src_mod.gain_label((0.1, 0.0)) == "gain 0.1, offset 0", WHERE
    assert src_mod.gain_label(None) == "unknown", WHERE
    assert src_mod.gain_label(GAIN_UNSTATED) == "unknown", WHERE
    assert src_mod.gain_label("nonsense") == "unknown", WHERE
    assert src_mod.gain_label((1.0,)) == "unknown", WHERE
