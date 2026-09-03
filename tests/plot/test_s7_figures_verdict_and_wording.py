"""Sweep 7 — what the EXPORTED publication figures say (S7-JRN3, MISSED-1, MISSED-2).

Analysis ▸ Export figure… is the artefact that leaves the app, and on the owner's −600 mV
apoferritin recording it went out saying three things the app itself knew were wrong:

* fig_dwell:   "τ = 20411.6 [17554.2, 25035.2] µs" — a 95 % interval for an exponential
               the app's own KS test rejected (D = 0.3966 vs 0.0981 at 1 %, p = 1.6e-39),
               with no verdict anywhere on the page (S7-JRN3).
* fig_depth:   "μ = 2.363 [2.265, 2.455]" — the same, for one Gaussian drawn through a
               visibly bimodal histogram (KS 0.2351 vs 0.0981) (MISSED-2).
* fig_scatter: y axis "blockade depth ΔI/I₀" and "median ΔI/I₀ = 2.524" bare, over a
               cloud in which 236 of 275 events have frac_depth > 1 — enhancements, which
               the depth histogram two functions away already relabels (MISSED-1).

…and none of the four carried the "open pore NOT CERTIFIED" advisory the detection is
stamped with, while the events-CSV sidecar does (S7-JRN3, part 2).

Every test here drives ``nanoscope.plot.figures`` directly, the way the exporter does,
and reads the strings off the axes the way a reader of the PNG would. Each one FAILS
against the pre-fix snapshot and passes on the tree.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from nanoscope.core.document import CalibratedChannel, XAxis
from nanoscope.core.types import ChannelInfo, ChannelKind
from nanoscope.plot import figures

TAU = 200e-6
T_MIN = 20e-6

#: The verdict `analysis.stats._stamp_gof` left on the −600 mV dwell fit, verbatim.
REJECTED_GOF = {"ks": 0.3965529380474438, "ks_critical_001": 0.09814917893233192,
                "describes_data": False, "n": 275, "grid_mass": 1.0,
                "note": "KS D = 0.3966 against a 1% critical value of 0.0981 (n = 275). "
                        "THE FITTED CURVE DOES NOT DESCRIBE THESE DATA — ..."}
CONSISTENT_GOF = {"ks": 0.052, "ks_critical_001": 0.09814917893233192,
                  "describes_data": True, "n": 275, "grid_mass": 1.0,
                  "note": "KS D = 0.0520 against a 1% critical value of 0.0981 (n = 275). "
                          "The fitted curve is consistent with these data."}

NOT_CERTIFIED = ("The open pore here is NOT CERTIFIED, only chosen. This recording has "
                 "TWO conductance levels: the tracked one at -34.6 pA, which holds 90% of "
                 "the recording, and another at -120.7 pA holding 5%. ...")


class _Events:
    """The duck `figures._unpack` eats: ``.df`` + ``.provenance`` (an EventSet's shape)."""

    def __init__(self, df: pd.DataFrame, provenance: dict):
        self.df, self.provenance = df, provenance


def _events(n: int = 600, *, depth_centre: float = 0.3, seed: int = 0,
            provenance: dict | None = None) -> _Events:
    rng = np.random.default_rng(seed)
    dwell = T_MIN + rng.exponential(TAU, n)
    depth = np.abs(rng.normal(depth_centre, 0.05, n))
    df = pd.DataFrame({
        "t_start_s": np.sort(rng.uniform(0.0, 9.0, n)),
        "dwell_s": dwell,
        "frac_depth": depth,
        "filter_limited": np.zeros(n, dtype=bool),
        "accepted": np.ones(n, dtype=bool),
    })
    prov = {"analyzed_duration_s": 10.0, "min_dwell_s": T_MIN}
    prov.update(provenance or {})
    return _Events(df, prov)


class _ExpFit:
    """Stand-in FitResult for a left-truncated exponential, with a chosen verdict."""

    def __init__(self, gof: dict | None, tau: float = TAU, t_min: float = T_MIN):
        self.tau, self.t_min = tau, t_min
        self.params = {"tau_1": tau}
        self.ci = {"tau_1": (tau * 0.95, tau * 1.05)}
        self.n = 600
        self.meta = {"t_min": t_min, "t_min_source": "min_dwell_s"}
        if gof is not None:
            self.meta["gof"] = dict(gof)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= self.t_min, np.exp(-(x - self.t_min) / self.tau) / self.tau, 0.0)


class _GaussFit:
    def __init__(self, gof: dict | None, mu: float = 0.3, sigma: float = 0.05):
        self.mu, self.sigma = mu, sigma
        self.params = {"mu_1": mu, "sigma_1": sigma}
        self.ci = {"mu_1": (mu - 0.01, mu + 0.01), "sigma_1": (sigma * 0.9, sigma * 1.1)}
        self.n = 600
        self.meta = {} if gof is None else {"gof": dict(gof)}

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))


def _blob(ax) -> str:
    """Every string on the axes, with the box's line folds read as spaces."""
    return " ".join(t.get_text().replace("\n", " ") for t in ax.texts)


def _visual_lines(ax) -> list[str]:
    return [line for t in ax.texts for line in t.get_text().split("\n")]


def _axes_width_after_layout(fig, ax) -> float:
    fig.canvas.draw()          # figure.autolayout runs at draw time
    return float(ax.get_position().width)


def _channel(n: int = 200_000, fs: float = 50_000.0) -> tuple[CalibratedChannel, XAxis]:
    rng = np.random.default_rng(1)
    cur = rng.normal(-1000.0, 15.0, n).astype(np.float32)
    info = ChannelInfo(index=0, name="synthetic", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=np.ascontiguousarray(cur), units="pA",
                           sample_rate_hz=fs, t0_s=0.0)
    xax = XAxis(values=np.arange(n, dtype=np.float64) / fs, unit="s", is_time=True)
    return ch, xax


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# ================================================================================================
# S7-JRN3 / MISSED-2 — a rejected fit loses its interval and gains its verdict
# ================================================================================================

def test_a_rejected_dwell_fit_is_drawn_without_its_interval_and_with_the_verdict_on_the_axes():
    events = _events()
    fig, ax = figures.dwell_histogram(events, fit=_ExpFit(REJECTED_GOF))
    text = _blob(ax)

    # The point value stays, labelled; the interval — a claim of precision — does not.
    assert r"$\tau$ = 200.0 $\mu$s (rejected fit)" in text, text
    assert "[" not in text and "]" not in text, text
    # The verdict, in the same currency as the on-screen Plots box.
    assert "KS D = 0.397 > 0.098 (1%): fit REJECTED — parameters are not measurements" in text
    # The fit's own caveats are still there, and n.
    assert "n = 600" in text and r"$t_{min}$ = 20.0 $\mu$s" in text, text

    # NUMBERS DO NOT MOVE: the verdict changes what is WRITTEN, not what is drawn or
    # exported — the source-data CSV behind the figure is byte-for-byte the one an
    # unverdicted copy of the same fit produces.
    fig2, ax2 = figures.dwell_histogram(events, fit=_ExpFit(None))
    a = getattr(fig, "_nanoscope_source_data")
    b = getattr(fig2, "_nanoscope_source_data")
    pd.testing.assert_frame_equal(a, b)

    # ...AND THE PLOT DOES NOT SHRINK. A 75-character verdict written as one line made
    # ``figure.autolayout`` squeeze the histogram into a third of the single-column
    # figure to fit the text on the canvas. Every visual line fits the box's budget,
    # and the axes are the same width with the verdict as without it.
    assert max(len(line) for line in _visual_lines(ax)) <= figures._BOX_LINE_CHARS, \
        _visual_lines(ax)
    assert _axes_width_after_layout(fig, ax) == pytest.approx(
        _axes_width_after_layout(fig2, ax2), rel=0.02)


def test_a_rejected_depth_fit_is_treated_exactly_the_same_way():
    """MISSED-2: the rule is for every fit the exporter draws, not just the dwell one."""
    events = _events()
    fig, ax = figures.depth_histogram(events, fit=_GaussFit(REJECTED_GOF))
    text = _blob(ax)

    assert r"$\mu$ = 0.300 (rejected fit)" in text, text
    assert "[" not in text and "]" not in text, text
    assert "KS D = 0.397 > 0.098 (1%): fit REJECTED — parameters are not measurements" in text
    assert "n = 600" in text, text

    fig2, _ = figures.depth_histogram(events, fit=_GaussFit(None))
    pd.testing.assert_frame_equal(getattr(fig, "_nanoscope_source_data"),
                                  getattr(fig2, "_nanoscope_source_data"))


def test_a_fit_the_test_kept_still_prints_its_interval_and_now_says_so():
    """The interval is the right claim for a fit that survived; the figure says it survived."""
    events = _events()
    _, ax = figures.dwell_histogram(events, fit=_ExpFit(CONSISTENT_GOF))
    text = _blob(ax)
    assert r"$\tau$ = 200.0 [190.0, 210.0] $\mu$s" in text, text
    assert "(rejected fit)" not in text
    assert "KS D = 0.052 ≤ 0.098 (1%): consistent with these data" in text, text

    _, ax = figures.depth_histogram(events, fit=_GaussFit(CONSISTENT_GOF))
    text = _blob(ax)
    assert r"$\mu$ = 0.300 [0.290, 0.310]" in text, text
    assert "consistent with these data" in text, text

    # A stand-in fit that was never stamped (no ``meta['gof']``) has nothing to say
    # about goodness of fit, and keeps its interval — the pre-existing contract.
    _, ax = figures.dwell_histogram(events, fit=_ExpFit(None))
    text = _blob(ax)
    assert r"$\tau$ = 200.0 [190.0, 210.0] $\mu$s" in text, text
    assert "KS" not in text and "rejected" not in text, text


def test_annotate_false_cannot_silence_a_rejection():
    """``annotate=False`` is for a shared caption that restates the NUMBERS. A curve the
    app's own test threw out is not a number the caption can be trusted to restate —
    the on-screen Plots box already refuses to hide it, and the exported figure now
    refuses the same way. A fit that survived writes nothing, as before."""
    events = _events()
    _, ax = figures.dwell_histogram(events, fit=_ExpFit(REJECTED_GOF), annotate=False)
    text = _blob(ax)
    assert "n = " not in text and r"$\tau$" not in text, text          # the numbers obey
    assert "fit REJECTED" in text, text                                  # the verdict does not

    _, ax = figures.depth_histogram(events, fit=_GaussFit(REJECTED_GOF), annotate=False)
    text = _blob(ax)
    assert r"$\mu$" not in text and "fit REJECTED" in text, text

    _, ax = figures.dwell_histogram(events, fit=_ExpFit(CONSISTENT_GOF), annotate=False)
    assert "KS" not in _blob(ax), _blob(ax)


def test_the_real_fitter_s_stamped_verdict_reaches_the_exported_figure():
    """End to end through ``analysis.stats``: the −600 mV dwell sample is a narrow bump
    (quartiles 16.6 / 18.1 / 19.5 ms); an exponential fitted to one is rejected by the
    fitter's own `_stamp_gof`, and THAT verdict — not a recomputation — is what the
    figure prints, with the bootstrap interval the fitter did compute withheld."""
    from nanoscope.analysis.stats import fit_dwell_exponential

    rng = np.random.default_rng(3)
    dwell = np.clip(rng.normal(18e-3, 1.5e-3, 400), 1e-3, None)
    fit = fit_dwell_exponential(dwell, t_min=T_MIN, n_boot=25, seed=0)
    gof = fit.meta.get("gof")
    assert gof is not None and gof["describes_data"] is False, gof   # the precondition
    assert fit.ci, "the fitter computed an interval; the figure must be the one to withhold it"

    events = _events(400)
    events.df["dwell_s"] = dwell
    fig, ax = figures.dwell_histogram(events, fit=fit)
    text = _blob(ax)
    assert "[" not in text, text
    assert "(rejected fit)" in text and "fit REJECTED" in text, text
    assert f"KS D = {gof['ks']:.3f} > {gof['ks_critical_001']:.3f} (1%)" in text, text
    # ...and the drawn curve is the fitter's, unchanged by the verdict.
    src = getattr(fig, "_nanoscope_source_data")
    assert np.isfinite(src["fit_expected_count"]).all()


# ================================================================================================
# MISSED-1 — the scatter's y axis and median line share the depth histogram's guard
# ================================================================================================

def test_the_scatter_and_the_depth_histogram_use_one_wording_for_an_enhancement():
    enhanced = _events(depth_centre=2.5)
    _, ax_s = figures.scatter_dwell_depth(enhanced)
    _, ax_h = figures.depth_histogram(enhanced, fit=False)

    assert "ENHANCEMENT" in ax_s.get_ylabel(), ax_s.get_ylabel()
    assert "blockade depth" not in ax_s.get_ylabel(), ax_s.get_ylabel()
    # One helper, so the two figures cannot drift apart again.
    assert ax_s.get_ylabel() == ax_h.get_xlabel()
    median = float(np.median(enhanced.df["frac_depth"]))
    line = rf"median $\Delta I / I_0$ = {median:.3f} (ENHANCEMENT: > 1, not a blockade)"
    assert line in _blob(ax_s), _blob(ax_s)

    blockade = _events(depth_centre=0.3)
    _, ax_s = figures.scatter_dwell_depth(blockade)
    _, ax_h = figures.depth_histogram(blockade, fit=False)
    assert ax_s.get_ylabel() == ax_h.get_xlabel() == r"blockade depth $\Delta I / I_0$"
    assert "ENHANCEMENT" not in _blob(ax_s), _blob(ax_s)
    median = float(np.median(blockade.df["frac_depth"]))
    assert rf"median $\Delta I / I_0$ = {median:.3f}" in _blob(ax_s)


def test_no_figure_in_the_module_writes_a_bare_blockade_label_outside_the_shared_helper():
    """The wording lives in ``_depth_wording`` and nowhere else, so a third figure cannot
    drift the way the scatter did."""
    import inspect

    src = inspect.getsource(figures)
    helper = inspect.getsource(figures._depth_wording)
    outside = src.replace(helper, "")
    assert outside.count(r"blockade depth $\Delta I / I_0$") == 0, (
        "a figure writes the blockade label without going through _depth_wording")


# ================================================================================================
# S7-JRN3 part 2 — the detection's baseline advisory reaches every exported figure
# ================================================================================================

#: The caveat as the figure now writes it. The pointer used to read "— see the events CSV
#: sidecar"; Analysis > Export figure... writes no events CSV, and the fig_<name>.csv it does
#: write beside the PDF is the figure's own source data (X-SURF-2). It names the command that
#: writes the paragraph instead. Spelled from the module's own constant so the two cannot
#: drift: the point of the test is that the SAME pointer reaches all four figures.
CAVEAT = f"open-pore baseline not certified {figures._ADVISORY_POINTER}"


def _all_four(events):
    ch, xax = _channel()
    _, ax_dwell = figures.dwell_histogram(events, fit=False)
    _, ax_depth = figures.depth_histogram(events, fit=False)
    _, ax_scatter = figures.scatter_dwell_depth(events)
    _, ax_trace = figures.trace_preview(ch, xax, events=events)
    return {"dwell": ax_dwell, "depth": ax_depth, "scatter": ax_scatter, "trace": ax_trace}


def test_every_exported_figure_carries_the_not_certified_advisory_its_detection_carries():
    """The exporter hands each figure the EventSet, whose provenance already holds the
    advisory — the events CSV printed it and the four figures did not."""
    stamped = _events(provenance={"baseline_warnings": [NOT_CERTIFIED]})
    for name, ax in _all_four(stamped).items():
        assert CAVEAT in _blob(ax), (name, _blob(ax))
        assert max(len(line) for line in _visual_lines(ax)) <= figures._BOX_LINE_CHARS, \
            (name, _visual_lines(ax))

    clean = _events()
    for name, ax in _all_four(clean).items():
        assert "certified" not in _blob(ax).lower(), (name, _blob(ax))
        assert "advisor" not in _blob(ax).lower(), (name, _blob(ax))


def test_the_trace_caveat_reaches_the_page_through_the_two_argument_annotate_call_site(monkeypatch):
    """``figures._annotate(ax, lines)`` is the documented single call site for every
    string written onto exported axes, and the sweep-7 repro wraps it with a
    TWO-argument lambda. A first cut of this fix placed the trace's caveat with an
    ``above=`` keyword: inside that wrapper the call raised TypeError, and the
    exporter's best-effort loop dropped the whole trace figure — nine files instead of
    twelve, silently. The placement is a marker on the axes now, and the wrapper sees
    the caveat."""
    captured: list[list[str]] = []
    orig = figures._annotate
    monkeypatch.setattr(figures, "_annotate",
                        lambda ax, lines: (captured.append(list(lines)), orig(ax, lines))[1])

    ch, xax = _channel()
    stamped = _events(provenance={"baseline_warnings": [NOT_CERTIFIED]})
    _, ax = figures.trace_preview(ch, xax, events=stamped)
    assert any(CAVEAT in " ".join(lines).replace("\n", " ") for lines in captured), captured
    # ...and it sits ABOVE the trace axes, not on the trace.
    caveat_texts = [t for t in ax.texts if "certified" in t.get_text()]
    assert caveat_texts and caveat_texts[0].get_position()[1] > 1.0, \
        [(t.get_text(), t.get_position()) for t in ax.texts]


def test_a_baseline_advisory_that_is_not_about_certification_is_counted_not_dropped():
    stamped = _events(provenance={"baseline_warnings": [
        "baseline health: the tracked level drifted by 12% over the recording"]})
    _, ax = figures.dwell_histogram(stamped, fit=False)
    assert f"1 baseline advisory on this detection {figures._ADVISORY_POINTER}" in _blob(ax)

    two = _events(provenance={"baseline_warnings": ["one thing", "another thing"]})
    _, ax = figures.scatter_dwell_depth(two)
    assert f"2 baseline advisories on this detection {figures._ADVISORY_POINTER}" in _blob(ax)


def test_the_advisory_rides_with_the_fit_box_and_survives_annotate_false():
    stamped = _events(provenance={"baseline_warnings": [NOT_CERTIFIED]})
    _, ax = figures.dwell_histogram(stamped, fit=_ExpFit(CONSISTENT_GOF))
    text = _blob(ax)
    assert r"$\tau$ = 200.0" in text and CAVEAT in text, text

    for fn in (lambda: figures.dwell_histogram(stamped, fit=_ExpFit(CONSISTENT_GOF),
                                                annotate=False),
               lambda: figures.depth_histogram(stamped, fit=_GaussFit(CONSISTENT_GOF),
                                                annotate=False),
               lambda: figures.scatter_dwell_depth(stamped, annotate=False)):
        _, ax = fn()
        text = _blob(ax)
        assert CAVEAT in text, text
        assert "n = " not in text, text
