"""Report artwork stays clean; complete scientific caveats travel in details/metadata."""
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



@pytest.mark.parametrize("kind,fit_type,symbol", [
    ("dwell_histogram", _ExpFit, r"$\tau$"),
    ("depth_histogram", _GaussFit, r"$\mu$"),
])
def test_rejected_fit_keeps_parameters_but_moves_verdict_into_details(kind, fit_type, symbol):
    events = _events()
    fit = fit_type(REJECTED_GOF)
    fig, ax = getattr(figures, kind)(events, fit=fit)
    text = _blob(ax)
    assert symbol in text and "n = 600" in text
    assert "[" not in text and "]" not in text
    assert "KS D" not in text and "reject" not in text.lower()
    detail = figures.figure_details(fig)["plots"][0]
    assert detail["fit"]["meta"]["gof"] == REJECTED_GOF
    assert detail["fit"]["ci"] == fit.ci
    assert any("fit REJECTED" in line for line in detail["details"])
    assert REJECTED_GOF["note"] in detail["details"]
    unstamped, _ = getattr(figures, kind)(events, fit=fit_type(None))
    pd.testing.assert_frame_equal(getattr(fig, "_nanoscope_source_data"),
                                  getattr(unstamped, "_nanoscope_source_data"))


@pytest.mark.parametrize("kind,fit_type", [
    ("dwell_histogram", _ExpFit), ("depth_histogram", _GaussFit),
])
def test_consistent_fit_keeps_its_interval_but_not_a_verdict_banner(kind, fit_type):
    fit = fit_type(CONSISTENT_GOF)
    fig, ax = getattr(figures, kind)(_events(), fit=fit)
    assert "[" in _blob(ax)
    assert "consistent with" not in _blob(ax) and "KS D" not in _blob(ax)
    assert figures.figure_details(fig)["plots"][0]["fit"]["meta"]["gof"] == CONSISTENT_GOF


@pytest.mark.parametrize("kind,fit_type", [
    ("dwell_histogram", _ExpFit), ("depth_histogram", _GaussFit),
])
def test_annotate_false_never_restores_warning_artwork(kind, fit_type):
    stamped = _events(provenance={"baseline_warnings": [NOT_CERTIFIED]})
    fig, ax = getattr(figures, kind)(stamped, fit=fit_type(REJECTED_GOF), annotate=False)
    assert not any(word in _blob(ax) for word in ("KS", "REJECTED", "certified", "n ="))
    detail = figures.figure_details(fig)["plots"][0]
    assert NOT_CERTIFIED in detail["details"]
    assert detail["fit"]["meta"]["gof"] == REJECTED_GOF


def _all_four(events):
    ch, xax = _channel()
    return [
        figures.dwell_histogram(events, fit=False),
        figures.depth_histogram(events, fit=False),
        figures.scatter_dwell_depth(events),
        figures.trace_preview(ch, xax, events=events),
    ]


def test_full_baseline_and_bespoke_advisories_survive_in_every_panel_details():
    cautions = [NOT_CERTIFIED, "baseline health: tracked level drifted by 12%"]
    stamped = _events(provenance={"baseline_warnings": cautions, "warnings": ["custom quality caveat"],
                                 "bespoke_flag": {"questionable_reference": True}})
    for fig, ax in _all_four(stamped):
        assert not any(word in _blob(ax).lower() for word in ("certified", "advisory", "caveat", "drifted"))
        detail = figures.figure_details(fig)["plots"][0]
        assert detail["provenance"] == stamped.provenance
        assert all(note in detail["details"] for note in [*cautions, "custom quality caveat"])
        assert detail["provenance"]["bespoke_flag"]["questionable_reference"] is True


def test_fractional_quantity_keeps_honest_axis_without_enhancement_banner():
    enhanced = _events(depth_centre=2.5)
    fig_s, ax_s = figures.scatter_dwell_depth(enhanced)
    fig_h, ax_h = figures.depth_histogram(enhanced, fit=False)
    assert ax_s.get_ylabel() == ax_h.get_xlabel() == r"$\Delta I / I_0$"
    assert "ENHANCEMENT" not in _blob(ax_s)
    assert "blockade depth" not in ax_s.get_ylabel()
    for fig in (fig_s, fig_h):
        assert any("ABOVE 1" in line for line in figures.figure_details(fig)["plots"][0]["details"])
    _, ordinary = figures.depth_histogram(_events(depth_centre=.3), fit=False)
    assert ordinary.get_xlabel() == r"blockade depth $\Delta I / I_0$"


def test_real_fitter_verdict_and_scientific_curve_remain_unchanged():
    from nanoscope.analysis.stats import fit_dwell_exponential
    rng = np.random.default_rng(3)
    dwell = np.clip(rng.normal(18e-3, 1.5e-3, 400), 1e-3, None)
    fit = fit_dwell_exponential(dwell, t_min=T_MIN, n_boot=25, seed=0)
    assert fit.meta["gof"]["describes_data"] is False
    events = _events(400)
    events.df["dwell_s"] = dwell
    original = events.df.copy(deep=True)
    fig, ax = figures.dwell_histogram(events, fit=fit)
    assert "REJECTED" not in _blob(ax) and "[" not in _blob(ax)
    np.testing.assert_equal(figures.figure_details(fig)["plots"][0]["fit"]["meta"], fit.meta)
    assert np.isfinite(getattr(fig, "_nanoscope_source_data")["fit_expected_count"]).all()
    pd.testing.assert_frame_equal(events.df, original)


def test_details_are_detached_from_sources_and_support_multiple_panels():
    events = _events(provenance={"baseline_warnings": [NOT_CERTIFIED]})
    fit = _ExpFit(REJECTED_GOF)
    fig, axes = plt.subplots(1, 2)
    figures.dwell_histogram(events, fit=fit, ax=axes[0])
    figures.depth_histogram(events, fit=False, ax=axes[1])
    events.provenance["baseline_warnings"].clear()
    fit.meta["gof"]["describes_data"] = True
    detail = figures.figure_details(fig)
    assert len(detail["plots"]) == 2
    assert detail["plots"][0]["fit"]["meta"]["gof"]["describes_data"] is False
    assert NOT_CERTIFIED in detail["plots"][1]["details"]
    detail["plots"].clear()
    assert len(figures.figure_details(fig)["plots"]) == 2


def test_saved_pdf_and_png_embed_details_without_changing_three_file_family(tmp_path):
    import json
    from PIL import Image
    from matplotlib.backends.backend_pdf import pdfRepr

    fit = _ExpFit(REJECTED_GOF)
    fit.meta.update(n_excluded_filter_limited=17, floor_biased=True, depth_floor=.02,
                    custom_flag="keep this exact diagnostic", numpy_values=np.array([1., np.nan]))
    events = _events(provenance={"baseline_warnings": [NOT_CERTIFIED]})
    fig, ax = figures.dwell_histogram(events, fit=fit)
    original_data = getattr(fig, "_nanoscope_source_data").copy(deep=True)
    paths = figures.save_figure(fig, tmp_path / "report")
    assert set(paths) == {"pdf", "png", "csv"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.csv", "report.pdf", "report.png"]
    with Image.open(paths["png"]) as png:
        encoded = png.info["Description"]
    metadata = json.loads(encoded)
    detail = metadata["plots"][0]
    assert detail["fit"]["meta"]["gof"] == REJECTED_GOF
    assert detail["fit"]["meta"]["n_excluded_filter_limited"] == 17
    assert detail["fit"]["meta"]["floor_biased"] is True
    assert detail["fit"]["meta"]["custom_flag"] == "keep this exact diagnostic"
    assert detail["fit"]["meta"]["numpy_values"] == [1., None]
    assert NOT_CERTIFIED in detail["details"]
    assert any("floor-biased" in note for note in detail["details"])
    assert b"/Subject " + pdfRepr(encoded) in paths["pdf"].read_bytes()
    assert "NOT CERTIFIED" not in paths["csv"].read_text()
    assert "REJECTED" not in _blob(ax) and "certified" not in _blob(ax)
    pd.testing.assert_frame_equal(getattr(fig, "_nanoscope_source_data"), original_data)
    pd.testing.assert_frame_equal(pd.read_csv(paths["csv"]), original_data)


def test_figure_details_has_safe_empty_default():
    fig, _ = plt.subplots()
    assert figures.figure_details(fig) == {"schema_version": 1, "plots": []}


def test_detection_floor_fallback_warning_is_preserved_outside_artwork():
    events = _events()
    events.provenance.pop("min_dwell_s")
    with pytest.warns(RuntimeWarning, match="min_dwell_s"):
        fig, ax = figures.dwell_histogram(events, fit=False)
    assert "min_dwell_s" not in _blob(ax)
    assert any("min_dwell_s" in note for note in figures.figure_details(fig)["plots"][0]["details"])


def test_sample_axis_shading_refusal_is_preserved_outside_artwork():
    channel, _ = _channel()
    xaxis = XAxis(values=np.arange(channel.data.size, dtype=float), unit="sample", is_time=False)
    with pytest.warns(RuntimeWarning, match="events were not shaded"):
        fig, ax = figures.trace_preview(channel, xaxis, events=_events())
    assert "not shaded" not in _blob(ax)
    assert fig._nanoscope_n_events_shown == 0
    assert any("events were not shaded" in note for note in figures.figure_details(fig)["plots"][0]["details"])
