"""E2(a) — what a SECOND fit of ONE event needs, and why it has to be the same fit.

A mask drag retires the edited event's sub-state levels, because they tile a span the event
no longer has (``tests/gui/test_level_fit_reconcile.py``). Putting them back means fitting
that one event again — and a second fit is only useful if it is the SAME fit.

Two numbers decide where a boundary lands: ``sigma`` (the noise scale) and ``tau`` (the
autocorrelation time the z-tests are discounted by). Both were RESOLVED inside
``fit_event_levels`` from a ladder of fallbacks, so a caller that passed neither had no way
to learn what it got. That is fine once and wrong twice, in two separate ways:

* CORRECTNESS. ``tau`` is measured from the open-pore residual AROUND THE EVENTS, so an
  estimate taken after an edit is taken from a different event set. Two neighbouring groups
  in one exported table would answer to two different significance scales, with nothing on
  screen saying so.
* COST. The estimate walks every event's residual in the recording. Measured on a
  60 000-sample synthetic with five two-sub-state events, re-fitting ONE event costs
  **3.4 ms with tau handed in against 26.3 ms without** — the segmentation itself is ~0.3 ms
  and everything else is a whole-recording measurement being paid for again.

So ``fit_event_levels`` grew a ``tau`` parameter and reports what it resolved on
``levels.attrs``; ``modules.fit_levels`` copies both onto ``AnalysisResult.params``, where
they survive a project round-trip. This file pins that contract. The GUI half — the cache,
the guards and the splice — is in ``tests/gui/test_level_fit_reconcile.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.cusum import (
    MIN_SAMPLES_FOR_LEVELS,
    SIGMA_ATTR,
    TAU_ATTR,
    empty_levels,
    fit_event_levels,
)
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.modules import fit_levels

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(5_000, 5_600), (15_000, 15_600), (25_000, 25_600),
          (35_000, 35_600), (45_000, 45_600)]


def _signal(seed: int = 0) -> np.ndarray:
    """Five events, each with two clean sub-states, so there is real structure to resolve."""
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 5.0, N)
    for s, e in _SPANS:
        mid = (s + e) // 2
        x[s:mid] -= 300.0
        x[mid:e] -= 500.0
    return np.ascontiguousarray(x, dtype=np.float64)


@pytest.fixture
def detection():
    sig = _signal()
    df, prov = detect_events(sig, FS, baseline=I0, min_dwell_s=1e-4)
    return sig, df, prov


def _fit(sig, df, prov, **kw):
    return fit_event_levels(sig, FS, df, baseline=float(I0),
                            polarity=int(prov.get("polarity", 1)),
                            provenance=prov, **kw)


# ================================================================================================
# what the fit ran with, reported
# ================================================================================================

def test_the_fit_reports_the_estimators_it_resolved(detection):
    """Without this a caller that passed neither argument cannot reproduce its own fit."""
    sig, df, prov = detection
    levels = _fit(sig, df, prov)
    assert not levels.empty
    for key in (SIGMA_ATTR, TAU_ATTR):
        value = levels.attrs[key]
        assert isinstance(value, float) and np.isfinite(value) and value > 0, (key, value)


def test_an_empty_result_says_tau_is_UNKNOWN_rather_than_zero(detection):
    """``tau`` is measured FROM the rows, so an empty event set resolves none. NaN means
    "not measured" — this app's one sentinel — and a caller reusing the stamp must fall back
    to measuring, not to a number that would silently disable the correction."""
    sig, df, prov = detection
    rejected = df.copy()
    rejected["accepted"] = False
    levels = _fit(sig, rejected, prov, accepted_only=True)
    assert levels.empty
    assert np.isfinite(levels.attrs[SIGMA_ATTR])
    assert np.isnan(levels.attrs[TAU_ATTR])


def test_an_empty_level_table_is_still_the_frozen_schema():
    """The stamp is carried on ``.attrs``, never as a column."""
    empty = empty_levels()
    assert TAU_ATTR not in empty.columns and SIGMA_ATTR not in empty.columns


# ================================================================================================
# reusing them makes the second fit the SAME fit
# ================================================================================================

def test_refitting_one_event_with_the_recorded_pair_reproduces_the_whole_fit(detection):
    """THE PROPERTY THE RE-FIT RESTS ON: one event fitted alone, with the estimators the
    whole-table fit resolved, lands on exactly the boundaries that fit gave it.

    Fitted WITHOUT them the same event is measured against a tau estimated from one event's
    flanks instead of the recording's — a different question, and the reason the numbers are
    handed back in rather than re-derived.
    """
    sig, df, prov = detection
    full = _fit(sig, df, prov)
    ev_id = int(df["event_id"].iloc[0])
    want = full[full["event_id"] == ev_id][["start_idx", "end_idx"]].to_numpy().tolist()

    one = _fit(sig, df.iloc[[0]], prov,
               sigma=full.attrs[SIGMA_ATTR], tau=full.attrs[TAU_ATTR])
    assert one[["start_idx", "end_idx"]].to_numpy().tolist() == want


def test_the_supplied_tau_is_actually_used_and_not_quietly_re_measured(detection):
    """A tau far above what the trace supports discounts every z-test into insignificance,
    so the fit collapses to one level. If the argument were ignored the segmentation would
    not move — which is exactly the failure mode a passed-through-but-unused parameter has."""
    sig, df, prov = detection
    normal = _fit(sig, df.iloc[[0]], prov)
    assert len(normal) >= 2, "the fixture must resolve sub-structure for this to mean anything"

    swamped = _fit(sig, df.iloc[[0]], prov, tau=1e6)
    assert len(swamped) < len(normal)
    assert swamped.attrs[TAU_ATTR] == 1e6, "the stamp must report what RAN, not what was asked"


def test_a_supplied_tau_costs_far_less_than_a_measured_one(detection):
    """Not a benchmark — a guard on the shape of the cost. Re-fitting one event must not pay
    a whole-recording measurement, or the automatic re-fit behind every mask drag becomes a
    stall on the GUI thread. Measured on this fixture: 3.4 ms against 26.3 ms."""
    import time

    sig, df, prov = detection
    full = _fit(sig, df, prov)
    one = df.iloc[[0]]

    t0 = time.perf_counter()
    _fit(sig, one, prov, sigma=full.attrs[SIGMA_ATTR], tau=full.attrs[TAU_ATTR])
    cheap = time.perf_counter() - t0

    t0 = time.perf_counter()
    _fit(sig, one, prov)
    dear = time.perf_counter() - t0

    assert cheap < dear / 2, (
        f"handing tau back saved nothing ({cheap * 1e3:.2f} ms vs {dear * 1e3:.2f} ms) — the "
        "whole-recording autocorrelation is still being paid for per event")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_impossible_tau_is_refused_rather_than_silently_replaced(detection, bad):
    """A tau of 0 divides by zero inside the z-tests and a negative one inverts the discount.
    A caller handing back a number it read from a restored project's params has to be told,
    not given a different segmentation from the groups it is splicing beside."""
    sig, df, prov = detection
    with pytest.raises(ValueError, match="tau"):
        _fit(sig, df.iloc[[0]], prov, tau=bad)


def test_the_default_is_unchanged(detection):
    """``tau=None`` must be exactly the shipped behaviour: nothing about a whole-table fit
    changes because a parameter was added for a second caller."""
    sig, df, prov = detection
    a = _fit(sig, df, prov)
    b = _fit(sig, df, prov, tau=None)
    pd.testing.assert_frame_equal(a, b)


# ================================================================================================
# ...and they reach the result, so they survive a project round-trip
# ================================================================================================

def test_fit_levels_records_the_estimators_on_the_result(detection):
    sig, df, prov = detection
    res = fit_levels(df, sig, FS, doc_id="d", channel=0, baseline=float(I0),
                     polarity=int(prov.get("polarity", 1)), provenance=prov)
    assert res.params["sigma_pA"] == res.payload.attrs[SIGMA_ATTR]
    assert res.params["tau_samples"] == res.payload.attrs[TAU_ATTR]
    # JSON-serialisable, because `params` is what the project writer persists.
    import json
    json.dumps(res.params)


def test_an_unresolved_tau_is_OMITTED_from_params_not_written_as_nan(detection):
    """NaN is not valid JSON, and "the key is absent" is already how this app spells
    "not measured" in provenance. A fit with nothing to fit must not poison the record the
    project writer serialises."""
    import json

    sig, df, prov = detection
    rejected = df.copy()
    rejected["accepted"] = False
    res = fit_levels(rejected, sig, FS, doc_id="d", channel=0, baseline=float(I0),
                     polarity=int(prov.get("polarity", 1)), provenance=prov)
    assert res.payload.empty
    assert "tau_samples" not in res.params
    json.dumps(res.params)


def test_fit_levels_passes_the_pair_through_when_it_is_given_one(detection):
    """A caller holding a previous fit's estimators can reproduce it through the module
    door as well as through the function, or the round-trip only half exists."""
    sig, df, prov = detection
    first = fit_levels(df, sig, FS, doc_id="d", channel=0, baseline=float(I0),
                       polarity=int(prov.get("polarity", 1)), provenance=prov)
    again = fit_levels(df, sig, FS, doc_id="d", channel=0, baseline=float(I0),
                       polarity=int(prov.get("polarity", 1)), provenance=prov,
                       sigma=first.params["sigma_pA"], tau=first.params["tau_samples"])
    pd.testing.assert_frame_equal(first.payload, again.payload)


def test_the_short_event_floor_is_public_and_is_the_one_characterize_uses():
    """``MainWindow._refit_levels_for_event`` has to ask BEFORE calling — the fit does not
    fail on a short span, it returns one "level" spanning the whole event, which would enter
    the export looking like a measurement."""
    from nanoscope.analysis.characterize import _MIN_SAMPLES_FOR_LEVELS as CHAR_FLOOR

    assert MIN_SAMPLES_FOR_LEVELS == CHAR_FLOOR
