"""Analysis ▸ Plots must be as honest as Analysis ▸ Voltage dependence.

REGRESSION FOR THE apo-run-2026-08-20 SUMMARY §5 ITEM 4, on the surface that had it.

The owner's three apoferritin recordings each carry a DIFFERENT ``baseline_fingerprint``
— the Preprocess probe correctly proposes a different open-pore window per pore.
Depths measured under different windows are not poolable, and
``Analysis ▸ Voltage dependence`` enforces that: it banners, it titles the plot with
the verdict, it adds Baseline/Merge columns. ``Analysis ▸ Plots`` did none of it:
``grep -c baseline_fingerprint nanoscope/gui/plots_panel.py`` was **0**. Tick two
recordings, plot ``frac_depth``, and Details listed the sources and their n and said
nothing about the two fingerprints.

And it was worse than silent. On ``depth_pA`` the gain guard fired and said
*"dwell_s and frac_depth (dI/I0) ARE comparable — plot those instead."* — actively
redirecting the user onto the unguarded column. On this data that advice is wrong:
``frac_depth`` is ``depth_pA / |baseline_pA|``, so it moves with the baseline window
too (measured: 11.3 % on the −700 mV recording).

What these pin, in the window:

* the verdict reaches the DETAILS pane, per metric, naming both fingerprints;
* the verdict reaches the plot TITLE — the only thing ``Save image`` keeps — and a
  user-typed title cannot displace it, so an incomparable figure cannot be exported
  looking clean;
* the pool is still DRAWN. Plots is an exploration surface; refusing to overlay two
  distributions would remove the comparison the user came for. Warn and draw;
* ``dwell_s`` stays quiet across the same two sources, because dwell does not move
  with the baseline — a guard that fires on everything trains the user to ignore it;
* the detection's OWN warnings (``AnalysisResult.warnings`` — e.g. "measured at the
  unprobed default baseline") reach the pane too. The pane already rendered a fit's
  ``meta["warnings"]`` and a clustering's ``ClusterResult.warnings``; the detector's
  were the one channel with no outlet here.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import SCHEMA
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import PlotsWindow

FS = 50_000.0

# The real fingerprints the probe stamped on the owner's ladder.
FP_500 = "sha256:b64a75ebdf29"
FP_600 = "sha256:ce8ae31f10bd"


def _frame(n: int = 200, *, seed: int = 0, depth: float = 63.0,
           baseline: float = -27.0) -> pd.DataFrame:
    """A full 20-column event frame, shaped like the apoferritin tables."""
    rng = np.random.default_rng(seed)
    cols = {}
    for name, (dtype, _u, _d) in SCHEMA.items():
        if dtype == "int64":
            cols[name] = np.arange(n, dtype=np.int64)
        elif dtype == "bool":
            cols[name] = np.zeros(n, dtype=bool)
        elif dtype == "string":
            cols[name] = pd.Series([""] * n, dtype="string")
        else:
            cols[name] = np.zeros(n, dtype=np.float64)
    cols["dwell_s"] = 2e-5 + rng.exponential(2e-2, n)
    cols["baseline_pA"] = np.full(n, baseline)
    cols["depth_pA"] = np.abs(rng.normal(depth, depth / 12, n))
    cols["frac_depth"] = cols["depth_pA"] / abs(baseline)
    cols["snr"] = np.abs(rng.normal(9.0, 1.5, n))
    cols["accepted"] = np.ones(n, dtype=bool)
    return pd.DataFrame(cols)


def _source(key: str, fingerprint: str | None, *, seed: int, depth: float,
            baseline: float, warnings: tuple[str, ...] = ()) -> PlotSource:
    prov: dict = {"fs": FS, "min_dwell_s": 2e-5}
    if fingerprint is not None:
        prov["baseline_fingerprint"] = fingerprint
    return PlotSource(
        key=key, doc_key=key, label=key, display_name=key,
        df=_frame(seed=seed, depth=depth, baseline=baseline), provenance=prov,
        doc_id=key, channel=2, module_id="detect.schmitt",
        calibration_fingerprint=(1.0, 0.0), warnings=warnings)


@pytest.fixture
def window(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    return w


def _two_windows(window, fp_a=FP_500, fp_b=FP_600, **kw):
    """The real situation: −500 mV and −600 mV, each probed under its own window."""
    a = _source("apo -500 mV", fp_a, seed=1, depth=63.0, baseline=-27.08, **kw)
    b = _source("apo -600 mV", fp_b, seed=2, depth=89.0, baseline=-34.79)
    window.set_data([a, b])
    window.set_sources([a.key, b.key])
    return a, b


def _title(window) -> str:
    label = window.plot_item().titleLabel
    return "" if label is None else label.text


# ==================================================================================
# 1. the verdict reaches the Details pane — per metric
# ==================================================================================

def test_frac_depth_across_two_baseline_windows_is_no_longer_silent(window):
    """THE DEFECT. Two fingerprints, plot dI/I0, and the pane said nothing at all."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")

    text = window.details_text()
    assert "NOT COMPARABLE" in text
    assert FP_500 in text and FP_600 in text, "the pane must name BOTH fingerprints"
    assert "apo -500 mV" in text and "apo -600 mV" in text
    assert "frac_depth" in text


def test_depth_across_two_baseline_windows_says_so_too(window):
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")

    text = window.details_text()
    assert "NOT COMPARABLE" in text and FP_500 in text and FP_600 in text


def test_the_pane_says_what_to_do_about_it(window):
    """A verdict with no remedy is a scold. V-dep names the remedy; so does this."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    text = window.details_text()
    assert "Re-detect them under ONE baseline window" in text
    assert "Preprocess" in text


def test_one_common_window_goes_quiet(window):
    """Re-detect both under one baseline and the verdict disappears — the remedy works."""
    _two_windows(window, fp_a=FP_500, fp_b=FP_500)
    window.set_plot_type("Histogram")
    for col in ("frac_depth", "depth_pA"):
        window.set_histogram_column(col)
        assert "NOT COMPARABLE" not in window.details_text(), col


def test_dwell_stays_quiet_across_the_same_two_sources(window):
    """Dwell is set by the Schmitt edges, not the baseline level. Per metric, exactly
    as V-dep asks it per metric."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    assert "NOT COMPARABLE" not in window.details_text()
    assert "NOT COMPARABLE" not in _title(window)


@pytest.mark.parametrize("col", ["rise_s", "fall_s", "n_levels"])
def test_the_transition_columns_are_not_silent_either(window, col):
    """REGRESSION, second round. These three were classified baseline-FREE, so the
    same two sources that make ``frac_depth`` shout drew a ``rise_s`` overlay with an
    EMPTY title and no Details line — and ``Save image`` exported it looking clean.

    They are not free. ``rise_fall_over_set`` reads its crossings at 0.9 and 0.1 of
    that event's own ``depth_pA``, off ``pol * (current - baseline)``, and
    ``_count_levels`` segments the deviation FROM the baseline. Re-measured on the
    owner's −500 mV detection with the SAME 1196 rows and the SAME Schmitt edges,
    moving only the window (1.0/0.30 s → 2.0/0.50 s, i.e. onto the −600's), rise_s
    moved on 838 rows and fall_s on 845 while dwell_s moved on 0, and the population
    medians shifted 1.49 % and 2.60 % against depth_pA's 0.33 %.

    ``rise_s``/``fall_s`` are not obscure here: they are in ``_PREFER_DWELL`` and in
    ``_PREFER_FEATURES``, so they seed the pickers, the PCA and the clustering.
    """
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column(col)

    text = window.details_text()
    assert "NOT COMPARABLE" in text, f"{col} moves with the baseline; the pane must say so"
    assert FP_500 in text and FP_600 in text
    assert col in _title(window), "and the title must carry it into the exported PNG"


@pytest.mark.parametrize("col", ["rise_s", "fall_s", "n_levels"])
def test_the_transition_columns_go_quiet_under_one_window(window, col):
    """The other half: the verdict must be about the BASELINE, not about the column."""
    _two_windows(window, fp_a=FP_500, fp_b=FP_500)
    window.set_plot_type("Histogram")
    window.set_histogram_column(col)
    assert "NOT COMPARABLE" not in window.details_text()
    assert "NOT COMPARABLE" not in _title(window)


@pytest.mark.parametrize("col", ["event_id", "start_idx", "end_idx"])
def test_the_index_columns_are_not_accused_of_a_baseline(window, col):
    """The fail-closed default used to fire here and say a sample INDEX 'is measured
    against each recording's own estimated baseline'. It is not: re-measuring the
    owner's −500 mV table under the other window moved 0 of 1196 rows on all three.
    Failing closed is the right default; asserting a mechanism that does not exist is
    how a user learns to stop reading the banner."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column(col)
    assert "NOT COMPARABLE" not in window.details_text()
    assert "NOT COMPARABLE" not in _title(window)


def test_a_single_ticked_source_is_never_warned_about(window):
    _two_windows(window)
    window.set_sources(["apo -500 mV"])
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    assert "NOT COMPARABLE" not in window.details_text()


def test_the_scatter_surface_carries_the_verdict_too(window):
    _two_windows(window)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "frac_depth")
    text = window.details_text()
    assert "NOT COMPARABLE" in text and FP_500 in text and FP_600 in text


def test_a_dwell_vs_dwell_scatter_stays_quiet(window):
    _two_windows(window)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("t_start_s", "dwell_s")
    assert "NOT COMPARABLE" not in window.details_text()


# ==================================================================================
# 2. the misdirection — the gain guard must stop naming frac_depth
# ==================================================================================

def test_the_gain_caution_no_longer_sends_the_user_to_frac_depth(window):
    """It fired on depth_pA and said 'frac_depth ARE comparable — plot those instead'.
    With two baseline windows in play that is false, and it is the sentence that
    walked the user onto the unguarded column."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")

    caution = [ln for ln in window.details_text().splitlines() if ln.startswith("CAUTION")]
    assert len(caution) == 1, "one gain caution, not two banners"
    assert "frac_depth" not in caution[0]
    assert "dwell_s IS comparable" in caution[0]


def test_the_gain_caution_keeps_its_advice_when_the_baselines_agree(window):
    """The original advice is right in the case it was written for, and is preserved."""
    _two_windows(window, fp_a=FP_500, fp_b=FP_500)
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    caution = [ln for ln in window.details_text().splitlines() if ln.startswith("CAUTION")]
    assert len(caution) == 1 and "frac_depth (dI/I0) ARE comparable" in caution[0]


# ==================================================================================
# 3. the plot TITLE — what Save image keeps
# ==================================================================================

def test_the_title_carries_the_verdict(window):
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    assert "NOT COMPARABLE" in _title(window)


def test_a_user_typed_title_cannot_displace_the_verdict(window):
    """The hole this closes: the title is the ONLY thing the PNG keeps, and the Style
    group lets the user overwrite it. If the override won, an incomparable overlay
    could be exported looking clean."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    window.set_title("Apoferritin dI/I0, −500 vs −600 mV")

    title = _title(window)
    assert "NOT COMPARABLE" in title
    assert "Apoferritin dI/I0" in title, "the user's title is kept, not replaced"


def test_the_exported_figure_carries_the_verdict(window, tmp_path):
    """End to end: what is on the PlotItem at export time is what lands in the PNG."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    window.set_title("figure 3b")

    out = window.save_image(str(tmp_path / "fig3b.png"))
    assert out is not None and out.exists() and out.stat().st_size > 0
    assert "NOT COMPARABLE" in _title(window)


def test_a_comparable_overlay_keeps_a_clean_title(window):
    _two_windows(window, fp_a=FP_500, fp_b=FP_500)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    window.set_title("Apoferritin dI/I0")
    assert _title(window) == "Apoferritin dI/I0"


# ==================================================================================
# 4. warn AND DRAW — the judgement call, pinned
# ==================================================================================

def test_an_incomparable_overlay_is_still_drawn(window):
    """Plots is an exploration surface: overlaying two distributions to SEE that they
    differ is exactly what a user ticks two sources for. Nothing is hidden, nothing
    is refused — the pool is drawn, and said out loud."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")

    assert len(window._hist_items) == 2, "both sources still drawn"
    assert sum("apo" in t for t in window.legend_labels()) == 2
    assert "n = " in window.details_text() or "apo -500 mV" in window.details_text()


# ==================================================================================
# 5. the detection's own caveats reach the pane
# ==================================================================================

def test_the_detections_own_warnings_reach_the_details_pane(window):
    """``AnalysisResult.warnings`` — the same tuple StatsPanel.set_warnings renders.
    A table measured at the unprobed default plotted here with no caveat at all,
    while the Summary said so about the same numbers."""
    caveat = ("Detected at the UNPROBED default baseline window "
              "(rough 0.5 s / base 0.1 s): run Preprocess ▸ probe and confirm.")
    a = _source("apo -500 mV", FP_500, seed=1, depth=63.0, baseline=-27.08,
                warnings=(caveat,))
    window.set_data([a])
    window.set_sources([a.key])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")

    assert caveat in window.details_text()


def test_each_sources_caveat_is_attributed_on_an_overlay(window):
    a, _b = _two_windows(window, warnings=("baseline never confirmed",))
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    text = window.details_text()
    assert "apo -500 mV: baseline never confirmed" in text
    assert "apo -600 mV: baseline never confirmed" not in text


def test_the_caveat_survives_on_the_scatter_surface(window):
    a = _source("apo -500 mV", FP_500, seed=1, depth=63.0, baseline=-27.08,
                warnings=("baseline never confirmed",))
    window.set_data([a])
    window.set_sources([a.key])
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "depth_pA")
    assert "baseline never confirmed" in window.details_text()


def test_the_verdict_does_not_outlive_the_plot_that_earned_it(window):
    """pyqtgraph's ``setTitle(None)`` only HIDES the title label — the previous
    draw's text stays on the item. Switch from an incomparable frac_depth overlay to
    dwell_s and the item still read NOT COMPARABLE, so "what does the title say"
    was not a function of the current draw. A warning that fires on the metric it
    does not apply to is the same failure as a warning that never fires."""
    _two_windows(window)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    assert "NOT COMPARABLE" in _title(window)

    window.set_histogram_column("dwell_s")
    assert _title(window) == ""


# ==================================================================================
# 6. the SIGNAL DOMAIN — is the number on this axis picoamps at all?
# ==================================================================================
#
# THE SECOND SILENT OVERLAY, found on the same surface. ``Preprocess ▸ Normalize`` is
# stage 7 of ``preprocess_for_detection``; it rescales the WHOLE trace before either
# detector sees it, and both detectors characterize on the result. Its own docstring
# states the consequence — "NORMALIZE IS LAST, AND IT TAKES THE PICOAMPS WITH IT …
# depth_pA, the baseline, the thresholds and every exported column are still spelled
# 'pA' while carrying sigmas or span-fractions" — and the ONLY disclosure was a live
# banner in the Detection panel, which does not travel with the table.
#
# So: detect one recording with Normalize on, one without, tick both, plot
# ``depth_pA``. Two curves on one axis, one in picoamps and one in noise sigmas, no
# title, no Details line. ``normalize_enabled`` was in provenance the whole time;
# ``pool_comparability`` judged three properties none of which move when the stage is
# toggled, so the rule had nothing to say.

def _domain_source(key: str, prov_extra: dict, *, seed: int = 3,
                   fingerprint: str = FP_500) -> PlotSource:
    """A source identical to ``_source``'s in every way except its provenance extras,
    so a warning that fires here can only have come from the domain half."""
    a = _source(key, fingerprint, seed=seed, depth=63.0, baseline=-27.08)
    return PlotSource(
        key=a.key, doc_key=a.doc_key, label=a.label, display_name=a.display_name,
        df=a.df, provenance={**a.provenance, **prov_extra}, doc_id=a.doc_id,
        channel=a.channel, module_id=a.module_id,
        calibration_fingerprint=a.calibration_fingerprint, warnings=a.warnings)


def _mixed(window, prov_extra: dict, *, other: dict | None = None):
    """One ordinary pA source and one whose provenance says otherwise. SAME baseline
    fingerprint and SAME calibration on both, so the baseline and gain halves cannot
    account for anything these tests see."""
    a = _domain_source("ordinary run", other or {}, seed=1)
    b = _domain_source("odd run", prov_extra, seed=2)
    window.set_data([a, b])
    window.set_sources([a.key, b.key])
    return a, b


NORMALIZED = {"normalize_enabled": True, "normalize_mode": "robust"}


def test_a_normalized_run_overlaid_on_an_ordinary_one_is_no_longer_silent(window):
    """THE DEFECT. One depth column in picoamps, one in noise sigmas, one axis."""
    _mixed(window, NORMALIZED)
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")

    text = window.details_text()
    assert "NOT COMPARABLE" in text
    assert "Normalize" in text
    assert "ordinary run" in text and "odd run" in text
    # The MODE is named: 'robust' divides by the baseline noise sigma and 'minmax' by
    # the trace's full span, which are not the same axis either.
    assert "robust" in text


def test_the_normalized_verdict_reaches_the_title_that_save_image_keeps(window):
    _mixed(window, NORMALIZED)
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    assert "NOT COMPARABLE" in _title(window)


def test_frac_depth_is_guarded_too_which_the_gain_half_would_not_have_done(window):
    """THE REASON THIS IS NOT THE GAIN HALF. ``column_depends_on_gain('frac_depth')``
    is False — a gain is a pure scale and divides out of a ratio — so gating the
    domain check on it would have left dI/I0 silent. Normalize is AFFINE: it subtracts
    a location as well as dividing, and the location is the open-pore level, i.e. the
    dI/I0 DENOMINATOR. Measured through the real detector, frac_depth moves 872x-1048x
    (tests/core/test_signal_domain.py), against depth_pA's mere change of unit."""
    _mixed(window, NORMALIZED)
    window.set_plot_type("Histogram")
    window.set_histogram_column("frac_depth")
    text = window.details_text()
    assert "NOT COMPARABLE" in text and "Normalize" in text
    assert "NOT COMPARABLE" in _title(window)


def test_the_remedy_refuses_to_send_the_user_to_frac_depth(window):
    """The same misdirection the gain guard was fixed for, in a second guard: advice
    that walks the user off one cliff onto another."""
    _mixed(window, NORMALIZED)
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    text = window.details_text()
    assert "frac_depth is NOT a way out" in text
    assert "Re-detect with Normalize OFF" in text


def test_dwell_stays_quiet_across_a_normalized_and_an_ordinary_run(window):
    """Normalize rescales amplitudes. It moves no sample index and no dwell, and a
    guard that fires on everything trains the user to ignore it."""
    _mixed(window, NORMALIZED)
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    assert "NOT COMPARABLE" not in window.details_text()
    assert "NOT COMPARABLE" not in _title(window)


def test_two_identically_normalized_runs_are_not_accused_of_disagreeing(window):
    """Groups DISAGREEING is the test. Two runs on one scale may be pooled; that their
    columns are still mislabelled 'pA' is a per-RUN fact, carried by each detection's
    own AnalysisResult.warnings, not by the pooling verdict."""
    _mixed(window, NORMALIZED, other=NORMALIZED)
    window.set_plot_type("Histogram")
    for col in ("depth_pA", "frac_depth"):
        window.set_histogram_column(col)
        assert "NOT COMPARABLE" not in window.details_text(), col


def test_two_target_units_are_the_same_class_of_bug_and_the_same_rule(window):
    """C3-3. ``ChannelCalibration.fingerprint()`` is ``(gain, offset)`` only,
    deliberately ('target_units is a label'), so a pA recording and an nA one carry
    the SAME fingerprint, agree on every half of the rule, and used to overlay a
    thousand-fold apart in silence. Different remedy, though: nA-to-pA is a pure
    scale, so here frac_depth genuinely IS the way out."""
    _mixed(window, {"units": "nA"}, other={"units": "pA"})
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    text = window.details_text()
    assert "NOT COMPARABLE" in text
    assert "nA" in text and "Calibration" in text
    assert "frac_depth is NOT a way out" not in text


def test_a_recording_that_states_no_units_is_read_as_picoamps(window):
    """"pA" is what the whole app assumes when a channel does not say
    (``ch.units or "pA"``), and the manual-measurement table carries no provenance at
    all. A distinct 'unknown' domain here would fire on every hand-measured overlay."""
    _mixed(window, {"units": "pA"}, other={})
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    assert "NOT COMPARABLE" not in window.details_text()


def test_the_scatter_surface_carries_the_domain_verdict_too(window):
    _mixed(window, NORMALIZED)
    window.set_plot_type("Scatter")
    window.set_scatter_columns("dwell_s", "depth_pA")
    assert "NOT COMPARABLE" in window.details_text()


def test_a_single_normalized_source_still_says_so_through_its_own_warnings(window):
    """The pooling rule is silent on a pool of one — nothing can disagree — so the
    single-source case is closed on the other side: ``modules._build_warnings`` puts
    the sentence on the detection itself, and ``_detection_notes`` prints it. Both
    halves are needed; neither covers the other's case."""
    a = _source("solo", FP_500, seed=1, depth=63.0, baseline=-27.08,
               warnings=("Preprocess > Normalize was ON, so the detector measured a "
                         "DIMENSIONLESS trace.",))
    window.set_data([a])
    window.set_sources([a.key])
    window.set_plot_type("Histogram")
    window.set_histogram_column("depth_pA")
    text = window.details_text()
    assert "NOT COMPARABLE" not in text, "a pool of one cannot disagree with anything"
    assert "DIMENSIONLESS" in text
