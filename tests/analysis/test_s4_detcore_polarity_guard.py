"""B4DA-07 -- polarity is a DIRECTION, and a direction that is not +-1 is refused.

THE DEFECT. Every other knob on `detect_events` validated with a precise message
(``k_out < k_in``, ``k_out > 0``, ``min_dwell_s >= 0``, ``max_dwell_s > min_dwell_s``,
``merge_gap_s >= 0``, ``fs > 0``, finite exclude regions, matching baseline length).
``polarity`` did not. It is used as a plain multiplier against thresholds derived from
the UNSCALED noise, so on 20 blockades of 100 pA against a 1000 pA open pore with 5 pA
noise:

    polarity=-1  ->  20 rows, depth 99.79 pA        (the truth)
    polarity=0   ->   0 rows, polarity_warning=None (a silent, clean, empty result)
    polarity=7   ->  56 rows, median depth 36.02 pA
    polarity=100 -> 403 rows, median depth 375.83 pA

and every one of those values was stamped verbatim into ``provenance['polarity']``,
which is what an export writes and a row rebuild reads back.

THE FIX IS AT THE SEAM. `baseline.require_polarity` is the single gate, and the four
doors that take a polarity from a caller now all pass through it -- `detect_events`,
`estimate_baseline_result`, `characterisation_baseline`, `estimate_baseline` (plus
`baseline_health`). The value reaches all of them from the same places
(``ctx.params`` forwarded by `modules._detect_kwargs`, a restored provenance mapping, a
batch config), and a rule enforced at three doors out of four is not a rule.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis.detect import detect_events

FS = 100_000.0


def _blockades(n: int = 400_000, seed: int = 61) -> np.ndarray:
    pad, step, dw = 5_000, (n - 10_000) // 20, 100
    x = np.full(n, 1000.0)
    for i in range(20):
        x[pad + i * step: pad + i * step + dw] -= 100.0
    return x + np.random.default_rng(seed).normal(0.0, 5.0, n)


@pytest.mark.parametrize("bad", [0, 7, -7, 100, 2, -2, 0.5, -1.5, "up", float("nan")])
def test_detect_events_refuses_a_polarity_that_is_not_a_direction(bad):
    x = _blockades(50_000)
    with pytest.raises(ValueError) as e:
        detect_events(x, FS, polarity=bad, min_dwell_s=1e-4)
    msg = str(e.value)
    assert "polarity must be +1 or -1" in msg
    assert repr(bad) in msg, msg           # the refusal names the value it got
    assert "direction, not a gain" in msg   # ... and the real cause
    assert "Pass None to measure" in msg    # ... and the way out


def test_the_two_directions_and_the_measurement_still_work():
    """The refusal must not cost the legal calls. -1 finds the 20 blockades, +1 finds
    none of them (physics, not an error), and None measures the direction."""
    x = _blockades()
    df_down, prov_down = detect_events(x, FS, polarity=-1, min_dwell_s=1e-4)
    assert len(df_down) == 20 and prov_down["polarity"] == -1
    assert float(df_down["depth_pA"].median()) == pytest.approx(99.79, abs=0.5)
    df_up, prov_up = detect_events(x, FS, polarity=1, min_dwell_s=1e-4)
    assert len(df_up) == 0 and prov_up["polarity"] == 1
    _df, prov_auto = detect_events(x, FS, min_dwell_s=1e-4)
    assert prov_auto["polarity"] == -1 and prov_auto["polarity_source"] == "measured"


def test_numpy_integers_are_still_a_direction():
    """The value arrives from provenance and from params dicts, where it is routinely a
    numpy scalar. Refusing those would break the rebuild path this guard protects."""
    x = _blockades(50_000)
    for pol in (np.int64(-1), np.int32(1), np.float64(-1.0), -1.0):
        df, prov = detect_events(x, FS, polarity=pol, min_dwell_s=1e-4)
        assert prov["polarity"] in (-1, 1)
        assert isinstance(prov["polarity"], int)
        assert len(df) >= 0


def test_no_bad_polarity_can_be_stamped_into_provenance():
    """The reason this is a refusal and not a warning: the number outlived the run --
    an export writes ``provenance['polarity']`` and a row rebuild reads it back."""
    x = _blockades(50_000)
    for bad in (0, 7, 100):
        with pytest.raises(ValueError):
            detect_events(x, FS, polarity=bad, min_dwell_s=1e-4)


@pytest.mark.parametrize("bad", [0, 7, -3, 0.5])
def test_the_estimator_doors_refuse_the_same_value_with_the_same_sentence(bad):
    """The seam, not one call site: the same guard stands at every door that takes a
    polarity from a caller, so a params dict that never reaches `detect_events` cannot
    reach an estimator with a rescaling multiplier either."""
    x = _blockades(50_000)
    for call in (
        lambda: B.characterisation_baseline(x, FS, polarity=bad),
        lambda: B.estimate_baseline_result(x, FS, polarity=bad),
        lambda: B.estimate_baseline(x, FS, polarity=bad),
        lambda: B.baseline_health(x, B.estimate_baseline(x, FS), FS, polarity=bad),
    ):
        with pytest.raises(ValueError) as e:
            call()
        assert "polarity must be +1 or -1" in str(e.value)


def test_require_polarity_returns_the_int_it_accepts():
    assert B.require_polarity(-1) == -1
    assert B.require_polarity(1) == 1
    assert B.require_polarity(np.int64(-1)) == -1
    assert isinstance(B.require_polarity(np.int64(-1)), int)
    with pytest.raises(ValueError):
        B.require_polarity(None)          # "measure it" is each door's own decision
