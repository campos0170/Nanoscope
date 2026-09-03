"""B4DA-08 — what the conditioning smoothers cost the depth has to be SAID somewhere.

``detect_events`` owns the low-pass, so it measures its rise time, marks every affected
row ``filter_limited``, and ``_build_warnings`` prints "N of M event(s) are
filter-limited: depths attenuated, dwell biased long". The SavGol and running-mean
stages in ``preprocess_for_detection`` do the same physical damage one step earlier and
the detector is never told they ran -- so no row could carry the flag, and nothing else
spoke. Measured, 20 square 1.000 ms / 100 pA events at 100 kHz:

    SavGol at the APP DEFAULT window (1 ms, order 3)  depth 76.74 pA (-23.3%)
    SavGol 5.0 ms                                     depth  6.56 pA (-93.4%)
    running mean 3.0 ms                               depth 33.09 pA (-66.9%)

all with 0 rows flagged and not one warning, while the 1 kHz low-pass in the same table
-- with a SMALLER error -- said so.

The disclosure is quantitative and it is measured through the very stages the data went
through (:func:`~nanoscope.analysis.modules._apply_smoothers`, shared with
``preprocess_for_detection``), because a bare "a smoother was enabled" is a caveat the
user cannot act on. It is judged at HALF the median measured dwell: the same stage
inflates the dwell it would otherwise be judged by, which is the trap B4DA-04 fell into.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.modules import (
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW_S,
    DetectSchmitt,
    _build_warnings,
    preprocess_for_detection,
    register_all,
    smoothing_depth_retention,
    smoothing_resolution_dwell_s,
    smoothing_stage_labels,
)

FS = 100_000.0
I0, DEPTH = 1000.0, 100.0
SAVGOL_DEFAULT = {"savgol_enabled": True, "savgol_window_s": DEFAULT_SAVGOL_WINDOW_S,
                  "savgol_polyorder": DEFAULT_SAVGOL_POLYORDER}
RMEAN_3MS = {"rmean_enabled": True, "rmean_window_s": 3e-3}


def _events(dwell_s: float = 1e-3, *, seed: int = 41, sigma: float = 5.0):
    n = int(FS * 4.0)
    pad, step, dw = 5_000, (n - 10_000) // 20, int(round(dwell_s * FS))
    x = np.full(n, I0)
    for i in range(20):
        x[pad + i * step : pad + i * step + dw] -= DEPTH
    return x + np.random.default_rng(seed).normal(0.0, sigma, n)


def _warnings_for(params: dict, dwell_s: float = 1e-3):
    sig = np.asarray(preprocess_for_detection(_events(dwell_s), FS, params),
                     dtype=np.float64)
    df, prov = detect_events(sig, FS, min_dwell_s=1e-4)
    prov = dict(prov)
    prov.update(params)
    return df, _build_warnings(prov, df)


# ---------------------------------------------------------------------------
# the disclosure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("params", [SAVGOL_DEFAULT, RMEAN_3MS,
                                    {"savgol_enabled": True, "savgol_window_s": 5e-3,
                                     "savgol_polyorder": 3}])
def test_a_smoother_that_eats_depth_says_so_on_the_result(params):
    df, warns = _warnings_for(params)
    said = [w for w in warns if "smoothed the trace before detection" in w]
    assert len(said) == 1, (params, warns)
    assert "attenuated" in said[0]
    # It names the stage and its window, so the user knows which control to move.
    for label in smoothing_stage_labels(params):
        assert label in said[0]
    # ...and it does not pretend a row carries it.
    assert int(df["filter_limited"].sum()) == 0
    assert "filter_limited covers only the detector's own low-pass" in said[0]


def test_no_smoother_no_sentence():
    """The chain with nothing enabled must stay silent -- an advisory on a clean run is
    the same defect wearing the other sign."""
    df, warns = _warnings_for({})
    assert not any("smoothed the trace" in w for w in warns), warns
    assert float(df["depth_pA"].median()) == pytest.approx(DEPTH, abs=1.0)


def test_a_harmless_window_stays_silent():
    """A 0.1 ms box on 1 ms events keeps essentially all of the depth. Crying wolf there
    teaches the user to ignore the sentence that matters."""
    df, warns = _warnings_for({"rmean_enabled": True, "rmean_window_s": 1e-4})
    assert float(df["depth_pA"].median()) > 0.97 * DEPTH
    assert not any("smoothed the trace" in w for w in warns), warns


def test_the_sentence_scales_with_the_damage():
    """A wider window must produce a smaller retained fraction in the text, not the
    same words: the numbers are the disclosure."""
    def _pct(params):
        _df, warns = _warnings_for(params)
        said = next(w for w in warns if "smoothed the trace" in w)
        return int(said.split("keeps only ")[1].split("%")[0])

    assert _pct({"rmean_enabled": True, "rmean_window_s": 1e-3}) > _pct(RMEAN_3MS)


# ---------------------------------------------------------------------------
# the measurement behind it
# ---------------------------------------------------------------------------
def test_retention_is_measured_through_the_same_stages_the_data_went_through():
    """The probe agrees with what the events table actually reports.

    A 1 ms square event through the app-default SavGol keeps ~98% of its depth measured
    over its OWN span; the table says 77% because the detected window is wider still --
    so the probe is an upper bound, and the test pins that direction rather than a
    coincidence.
    """
    keep = smoothing_depth_retention(FS, SAVGOL_DEFAULT, 1e-3)
    sig = np.asarray(preprocess_for_detection(_events(1e-3), FS, SAVGOL_DEFAULT),
                     dtype=np.float64)
    df, _prov = detect_events(sig, FS, min_dwell_s=1e-4)
    measured = float(df["depth_pA"].median()) / DEPTH
    assert 0.0 < measured <= keep <= 1.0, (measured, keep)


@pytest.mark.parametrize("window_s", [5e-4, 1e-3, 3e-3])
def test_retention_rises_with_the_event_duration_and_reaches_one(window_s):
    params = {"rmean_enabled": True, "rmean_window_s": window_s}
    curve = [smoothing_depth_retention(FS, params, d)
             for d in (0.1e-3, 0.5e-3, 1e-3, 5e-3, 50e-3)]
    assert curve == sorted(curve)
    assert curve[-1] == pytest.approx(1.0, abs=1e-6)
    assert curve[0] < 1.0


def test_the_resolution_floor_is_a_property_of_the_filter_not_of_the_rows():
    """It is what the ``2 * filter_rise_s`` floor is for the low-pass: a number the
    inflated dwells cannot corrupt."""
    floor = smoothing_resolution_dwell_s(FS, RMEAN_3MS, keep=0.9)
    assert smoothing_depth_retention(FS, RMEAN_3MS, floor) >= 0.9
    assert smoothing_depth_retention(FS, RMEAN_3MS, floor - 2.0 / FS) < 0.9
    # A wider window has a higher floor; no smoother has none at all.
    wider = smoothing_resolution_dwell_s(FS, {"rmean_enabled": True,
                                              "rmean_window_s": 6e-3})
    assert wider > floor
    assert smoothing_resolution_dwell_s(FS, {}) is None
    assert smoothing_depth_retention(FS, {}, 1e-3) is None


def test_stage_labels_name_the_windows_that_ran():
    assert smoothing_stage_labels({}) == ()
    both = smoothing_stage_labels({**SAVGOL_DEFAULT, **RMEAN_3MS})
    assert both == ("running-mean (3 ms window)",
                    "Savitzky-Golay (1 ms window, order 3)")


def test_module_path_carries_the_sentence(context_factory):
    """Through ``DetectSchmitt.run`` with the Preprocess tab's own parameter dict."""
    register_all()
    res = DetectSchmitt().run(context_factory(
        _events().astype(np.float32), fs=FS, params={**RMEAN_3MS, "min_dwell_s": 1e-4}))
    assert res.payload.provenance["rmean_enabled"] is True
    assert any("smoothed the trace before detection" in w for w in res.warnings), res.warnings


def test_preprocess_output_is_unchanged_by_the_refactor():
    """The two smoothing stages moved into a helper the probe shares. Same order, same
    numbers: running-mean FIRST, then SavGol."""
    from nanoscope.analysis.filters import running_mean, savgol
    x = _events()
    params = {**RMEAN_3MS, **SAVGOL_DEFAULT}
    got = np.asarray(preprocess_for_detection(x, FS, params), dtype=np.float64)
    want = savgol(running_mean(np.ascontiguousarray(x, dtype=np.float32), FS,
                               window_s=3e-3),
                  FS, window_s=DEFAULT_SAVGOL_WINDOW_S,
                  polyorder=DEFAULT_SAVGOL_POLYORDER)
    assert np.array_equal(got, np.asarray(want, dtype=np.float64))
