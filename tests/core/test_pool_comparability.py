"""The pooled-provenance comparability rule, in its ONE home — and the proof it is one.

REGRESSION FOR THE apo-run-2026-08-20 SUMMARY §5 ITEM 4.

``Analysis ▸ Voltage dependence`` refuses to pool depths measured under different
baseline windows: it groups the points by ``baseline_fingerprint``, banners the
disagreement and titles the plot with it. ``Analysis ▸ Plots`` — the OTHER surface in
this app that puts two recordings on one axis — did none of it. Measured on the
owner's apoferritin ladder, the three recordings carry three different fingerprints
(``sha256:b64a75ebdf29`` / ``sha256:ce8ae31f10bd`` / ``sha256:1a29aabf5fb1``), because
the probe correctly proposed a different open-pore window for each pore; ticking two
of them and plotting ``frac_depth`` drew one overlay and said nothing.

Worse than nothing. Plots DID have a gain guard, and on ``depth_pA`` it fired and
said *"dwell_s and frac_depth (dI/I0) ARE comparable — plot those instead."*
``frac_depth`` is ``depth_pA / |baseline_pA|``: it divides BY the baseline, so it
moves with the window exactly as the depth does — re-detecting the −700 mV recording
under the −600 mV recording's windows moved its median dI/I0 by 11.3 %. The guard
was pointing the user at the unguarded column.

Two invariants are pinned here:

  1. **ONE rule, ONE home.** The baseline/merge decision lives in
     :mod:`nanoscope.core.sources`, next to the gain rule Plots already consumed,
     and BOTH surfaces call it. ``voltage_dep`` holds no second copy — asserted
     structurally, by identity and by scanning the package for the message text,
     because a second implementation is this project's defining failure mode.
  2. **ONE verdict per metric.** The two guards are reconciled:
     :func:`~nanoscope.core.sources.source_comparability` drops ``frac_depth`` from
     the gain advice exactly when the sources disagree on their baselines, so the
     user is never redirected onto a column that is just as incomparable.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import voltage_dep
from nanoscope.core import sources as src_mod
from nanoscope.core.sources import (
    ABSOLUTE_CURRENT_COLUMNS,
    UNKNOWN_BASELINE,
    UNMERGED,
    PlotSource,
    PooledItem,
    baseline_fingerprint_of,
    column_depends_on_baseline,
    gains_comparable,
    merge_state,
    pool_comparability,
    source_comparability,
)

# The three real fingerprints from the apoferritin run.
FP_500 = "sha256:b64a75ebdf29"
FP_600 = "sha256:ce8ae31f10bd"
FP_700 = "sha256:1a29aabf5fb1"


def _df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "dwell_s": np.linspace(1e-3, 2e-3, n),
        "depth_pA": np.linspace(60.0, 70.0, n),
        "frac_depth": np.linspace(2.0, 2.2, n),
        "baseline_pA": np.full(n, -30.0),
        "accepted": np.ones(n, dtype=bool),
    })


def _plot_source(label: str, fingerprint: str | None, *, merged=None,
                 kind: str = "detection", gain=(1.0, 0.0)) -> PlotSource:
    prov: dict = {"fs": 50_000.0}
    if fingerprint is not None:
        prov["baseline_fingerprint"] = fingerprint
    if merged is not None:
        prov["merged"] = merged
    return PlotSource(key=label, doc_key=label, label=label, df=_df(), provenance=prov,
                      kind=kind, calibration_fingerprint=gain, display_name=label)


# ==================================================================================
# 1. the rule has exactly ONE implementation, and it is in core.sources
# ==================================================================================

def test_voltage_dep_uses_the_core_rule_it_does_not_own_one():
    """The extraction, asserted by IDENTITY: same function object, not same source."""
    assert voltage_dep.merge_state is src_mod.merge_state
    assert voltage_dep.UNMERGED is src_mod.UNMERGED
    assert voltage_dep.UNKNOWN_BASELINE is src_mod.UNKNOWN_BASELINE


@pytest.mark.parametrize("sentinel", [
    "is measured against each ",          # the baseline warning body
    "are not in the same ",               # the merge warning body
    "_BASELINE_FREE_COLUMNS",             # the per-column classification
])
def test_the_rule_is_written_down_in_exactly_one_module(sentinel):
    """A second copy is how the two surfaces drift apart; this fails the moment one appears."""
    pkg = pathlib.Path(src_mod.__file__).resolve().parent.parent
    hits = sorted(p.relative_to(pkg).as_posix()
                  for p in pkg.rglob("*.py") if sentinel in p.read_text())
    assert hits == ["core/sources.py"], f"{sentinel!r} should live in one module, found {hits}"


def test_voltage_dep_verdict_is_the_core_verdict():
    """vdep_comparability is a translation, not a re-decision: same groups, same warnings."""
    pts = [voltage_dep.VDepPoint(-0.5, 1.0, "a", FP_500, UNMERGED),
           voltage_dep.VDepPoint(-0.7, 2.0, "b", FP_700, UNMERGED)]
    series = voltage_dep.vdep_comparability(pts, metric="median_depth")
    verdict = pool_comparability(
        [PooledItem("a", FP_500, UNMERGED), PooledItem("b", FP_700, UNMERGED)],
        quantity="median depth", depends_on_baseline=True)
    assert series.warnings == verdict.warnings
    assert series.baseline_groups == verdict.baseline_groups
    assert series.merge_groups == verdict.merge_groups


# ==================================================================================
# 2. the per-column half — dwell does NOT move with the baseline, amplitudes do
# ==================================================================================

@pytest.mark.parametrize("col", [
    "event_id", "start_idx", "end_idx", "t_start_s", "dwell_s",
    "filter_limited", "accepted",
])
def test_geometry_columns_are_baseline_free(col):
    """Only the detection GEOMETRY survives a change of baseline window.

    Measured, not asserted: the owner's −500 mV detection (1196 events) re-measured
    through ``remeasure_events`` from its own probed 1.0/0.30 s window to the −600's
    2.0/0.50 s moved 0 of 1196 rows on every column here.
    """
    assert not column_depends_on_baseline(col)


@pytest.mark.parametrize("col", [
    "depth_pA", "frac_depth", "snr", "ecd_pA_s", "max_dev_pA", "mean_dev_pA", "baseline_pA",
])
def test_amplitude_columns_depend_on_the_baseline(col):
    """``frac_depth`` is on this list because it DIVIDES BY the baseline. That it was
    not treated as such is the whole defect."""
    assert column_depends_on_baseline(col)


@pytest.mark.parametrize("col", ["rise_s", "fall_s", "n_levels"])
def test_transition_columns_depend_on_the_baseline_too(col):
    """REGRESSION. These three were classified baseline-FREE, so Plots pooled them
    across differently-probed recordings in silence — no Details line, no title, and
    a clean-looking exported PNG.

    They are not free. ``rise_fall_over_set`` takes ``baseline`` as an argument and
    reads its crossings at 0.9 and 0.1 of that event's own ``depth_pA``, and
    ``_count_levels`` segments the deviation FROM the baseline. Measured on the
    owner's −500 mV detection, same 1196 rows, same Schmitt edges, ONLY the baseline
    window moved (1.0/0.30 s → 2.0/0.50 s): rise_s moved on 838 rows, fall_s on 845,
    n_levels on 3, while dwell_s and t_start_s moved on 0. The population medians
    moved 1.49 % (rise_s) and 2.60 % (fall_s) — further than depth_pA (0.33 %) and
    frac_depth (0.09 %), i.e. the guard was silent on columns that move MORE than
    the column it fires on.

    They are first-class here: ``rise_s``/``fall_s`` are in the 20-column schema, in
    the Plots pickers' ``_PREFER_DWELL`` and in ``_PREFER_FEATURES`` (so they seed
    PCA and clustering), and ``n_levels`` is offered by ``_numeric_columns``.
    """
    assert column_depends_on_baseline(col)


def test_the_rule_agrees_with_what_a_re_measure_actually_rewrites():
    """The other module that knows which columns move with the baseline must agree.

    ``analysis.remeasure`` re-measures a curated table against a NEW estimator, and
    it declares both halves of the same fact: ``GEOMETRY_COLUMNS`` is what it
    asserts row-for-row must NOT move, ``REMEASURED_COLUMNS`` is what it rewrites.
    That is the same partition this rule decides, so the two are pinned together —
    they disagreed about ``rise_s``/``fall_s`` for a whole release, and ``remeasure``
    was the one that was right.
    """
    from nanoscope.analysis.remeasure import GEOMETRY_COLUMNS, REMEASURED_COLUMNS

    moved = [c for c in REMEASURED_COLUMNS if not column_depends_on_baseline(c)]
    assert moved == [], f"remeasure rewrites {moved}, but the rule calls them baseline-free"
    fixed = [c for c in GEOMETRY_COLUMNS if column_depends_on_baseline(c)]
    assert fixed == [], f"remeasure pins {fixed} row-for-row, but the rule calls them baseline-bound"


def test_an_unknown_column_fails_closed():
    assert column_depends_on_baseline("some_new_amplitude_column_pA")


def test_frac_depth_is_baseline_bound_but_not_gain_bound():
    """The two guards cover overlapping, non-identical column sets — which is exactly
    why the gain guard's advice cannot be a fixed sentence."""
    assert "frac_depth" not in ABSOLUTE_CURRENT_COLUMNS
    assert column_depends_on_baseline("frac_depth")
    assert "depth_pA" in ABSOLUTE_CURRENT_COLUMNS


# ==================================================================================
# 3. the pooled verdict
# ==================================================================================

def test_two_different_fingerprints_are_not_poolable_for_an_amplitude():
    v = pool_comparability([PooledItem("a", FP_500), PooledItem("b", FP_600)],
                           quantity="depth_pA", depends_on_baseline=True)
    assert not v.comparable
    assert len(v.baseline_groups) == 2
    assert FP_500 in v.warnings[0] and FP_600 in v.warnings[0]
    assert "a" in v.warnings[0] and "b" in v.warnings[0]     # names WHO disagrees


def test_the_same_fingerprint_pools_cleanly():
    v = pool_comparability([PooledItem("a", FP_500), PooledItem("b", FP_500)],
                           quantity="depth_pA", depends_on_baseline=True)
    assert v.comparable and v.baselines_agree


def test_a_missing_fingerprint_is_not_agreement():
    v = pool_comparability([PooledItem("a"), PooledItem("b")],
                           quantity="depth_pA", depends_on_baseline=True)
    assert not v.comparable and not v.baselines_agree
    assert "no baseline_fingerprint" in v.warnings[0]


def test_a_baseline_free_quantity_pools_across_different_windows():
    """The reason the check is PER METRIC: Schmitt edges are set by the noise ladder."""
    v = pool_comparability([PooledItem("a", FP_500), PooledItem("b", FP_700)],
                           quantity="dwell_s", depends_on_baseline=False)
    assert v.comparable
    assert len(v.baseline_groups) == 2      # still reported, just not load-bearing


def test_merge_state_blocks_every_quantity():
    v = pool_comparability(
        [PooledItem("a", FP_500, UNMERGED),
         PooledItem("b", FP_500, merge_state({"merged": {"rule": "gap", "window_s": 0.01}}))],
        quantity="dwell_s", depends_on_baseline=False)
    assert not v.comparable and "merge state" in v.warnings[0]


def test_one_item_warns_about_nothing():
    assert pool_comparability([PooledItem("a")], quantity="depth_pA",
                              depends_on_baseline=True).comparable


def test_nothing_is_dropped_or_refused():
    """The verdict is advisory. An incomparable pool is still returned whole."""
    v = pool_comparability([PooledItem("a", FP_500), PooledItem("b", FP_600)],
                           quantity="depth_pA", depends_on_baseline=True)
    assert [g[0] for g in v.baseline_groups] == [FP_500, FP_600]


def test_merge_state_reads_both_places_a_merge_is_recorded():
    assert merge_state(None) == UNMERGED
    assert merge_state({}) == UNMERGED
    assert merge_state({"merge_gap_s": 0.01}) == "merged: detector gap 10 ms"
    assert merge_state({"merged": {"rule": "gap", "window_s": 0.01}}) == "merged: gap 10 ms"


def test_baseline_fingerprint_of_reads_the_provenance():
    assert baseline_fingerprint_of({"baseline_fingerprint": FP_500}) == FP_500
    assert baseline_fingerprint_of({}) == UNKNOWN_BASELINE
    assert baseline_fingerprint_of(None) == UNKNOWN_BASELINE


# ==================================================================================
# 4. THE MISDIRECTION — the gain advice must not name frac_depth when baselines differ
# ==================================================================================

def test_the_gain_advice_still_names_frac_depth_when_the_baselines_agree():
    """The original advice is CORRECT in the case it was written for, and stays."""
    srcs = [_plot_source("a", FP_500), _plot_source("b", FP_500)]
    v = source_comparability(srcs, ["depth_pA"])
    gain = [w for w in v.warnings if w.startswith("CAUTION")]
    assert len(gain) == 1
    assert "dwell_s and frac_depth (dI/I0) ARE comparable — plot those instead." in gain[0]


def test_the_gain_advice_drops_frac_depth_when_the_baselines_disagree():
    """THE REGRESSION. Two pores probed under their own windows, plotting depth_pA:
    the gain guard fires, and it must not send the user to frac_depth — dI/I0 divides
    by the very baseline they disagree about."""
    srcs = [_plot_source("a", FP_500), _plot_source("b", FP_600)]
    v = source_comparability(srcs, ["depth_pA"])
    gain = [w for w in v.warnings if w.startswith("CAUTION")]
    assert len(gain) == 1
    assert "frac_depth" not in gain[0], "must not redirect onto the unguarded column"
    assert "dwell_s IS comparable — plot that instead." in gain[0]


def test_an_unknown_baseline_also_disqualifies_frac_depth_as_advice():
    srcs = [_plot_source("a", None), _plot_source("b", None)]
    gain = [w for w in source_comparability(srcs, ["depth_pA"]).warnings
            if w.startswith("CAUTION")]
    assert gain and "frac_depth" not in gain[0]


def test_gains_comparable_default_advice_is_unchanged():
    """The default sentence itself is untouched; what changed is who may pass it."""
    ok, why = gains_comparable([(1.0, 0.0), (1.0, 0.0)])
    assert not ok
    assert why.endswith("dwell_s and frac_depth (dI/I0) ARE comparable — plot those instead.")


@pytest.mark.parametrize("fps", [
    [(1.0, 0.0), (1.0, 0.0)],       # nobody calibrated: shared identity gain
    [(2.5, 0.0), (3.1, 0.0)],       # two real, different gains
])
def test_a_caller_with_nothing_to_vouch_for_says_nothing(fps):
    """Empty alternatives ends the caution cleanly — no dangling 'ARE comparable'.

    Both branches of the gain verdict, because the PSD window reaches both.
    """
    ok, why = gains_comparable(fps, alternatives=())
    assert not ok
    assert why.rstrip().endswith(".") and not why.endswith(" ")
    assert "comparable — plot" not in why
    assert "frac_depth" not in why and "dwell_s" not in why


# ==================================================================================
# 5. the Plots adapter: ONE verdict, per metric
# ==================================================================================

def test_frac_depth_across_two_windows_is_no_longer_silent():
    """THE HEADLINE REGRESSION: the column the old guard recommended, unguarded."""
    srcs = [_plot_source("apo -500 mV", FP_500), _plot_source("apo -600 mV", FP_600)]
    v = source_comparability(srcs, ["frac_depth"])
    assert not v.comparable
    assert any("NOT COMPARABLE" in w and "frac_depth" in w for w in v.warnings)
    assert FP_500 in v.warnings[0] and FP_600 in v.warnings[0]
    assert v.headline and "frac_depth" in v.headline


def test_dwell_across_two_windows_stays_quiet():
    """Dwell does not move with the baseline; warning about it would train the user
    to ignore the warning that matters."""
    srcs = [_plot_source("a", FP_500), _plot_source("b", FP_600)]
    v = source_comparability(srcs, ["dwell_s"])
    assert v.comparable and v.headline == ""


def test_a_scatter_of_dwell_against_depth_is_judged_on_the_depth():
    srcs = [_plot_source("a", FP_500), _plot_source("b", FP_600)]
    v = source_comparability(srcs, ["dwell_s", "depth_pA"])
    assert not v.comparable
    assert "depth_pA" in v.warnings[0] and "dwell_s" not in v.warnings[0].split(":")[1]


def test_one_source_is_always_self_comparable():
    v = source_comparability([_plot_source("a", FP_500)], ["depth_pA"])
    assert v.comparable and v.headline == ""


def test_the_manual_table_counts_for_the_baseline_but_not_for_the_gain():
    """A hand-measured depth was measured against whatever the hand picked; it is not
    a second amplifier, so it must not raise the GAIN caution on its own."""
    detection = _plot_source("pore", FP_500)
    manual = PlotSource(key="m", label="Manual measurements", df=_df(), kind="manual")
    v = source_comparability([detection, manual], ["depth_pA"])
    assert any("NOT COMPARABLE" in w for w in v.warnings)
    assert not [w for w in v.warnings if w.startswith("CAUTION")]


def test_the_headline_is_a_one_line_verdict_for_a_plot_title():
    srcs = [_plot_source("a", FP_500), _plot_source("b", FP_600)]
    headline = source_comparability(srcs, ["depth_pA"]).headline
    assert "\n" not in headline and "NOT COMPARABLE" in headline
