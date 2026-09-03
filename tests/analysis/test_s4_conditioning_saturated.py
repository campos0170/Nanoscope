"""B4DA-05 — a depth read off a FLAT TOP is a lower bound, and must say so.

The app already makes this distinction for the time axis: ``clipped_by_boundary`` keeps
a row whose event ran past the edge of the analysed span, says its dwell is a lower
bound, and does not accept it. There was no amplitude-domain equivalent, so 20 blockades
of TRUE depth 300 pA clipped against an 800 pA rail came back ``depth_pA`` 200.30 /
``frac_depth`` 0.2002 -- a 33% understatement -- with ``flag=''`` and ``accepted=True``
on every row, no provenance key and no warning. A publication-grade dI/I0 shipped with a
33% error and nothing on the row said the number was a floor.

The evidence was already in the samples ``characterize`` walks: 100 consecutive samples
per event sitting on exactly one value, the extreme of the whole analysed trace. Three
such samples are now enough to flag the row ``saturated``. The row is KEPT and stays
accepted -- the event is real and its dwell is a real measurement -- exactly as
``zero_baseline`` is a caveat rather than a rejection.

The rule must also cost an ordinary recording nothing, which is what the last test here
and the corpus check in ``tests/analysis`` are for: ionic current plus thermal noise does
not repeat a float three times in a row at the extreme of a multi-million-sample array.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis.characterize import (
    FLAG_SATURATED,
    characterize,
    recharacterize_span,
    saturated_warning,
)
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import has_flag
from nanoscope.analysis.modules import DetectSchmitt, _build_warnings, register_all

FS = 100_000.0
I0, TRUE_DEPTH, SIGMA, RAIL = 1000.0, 300.0, 5.0, 800.0


def _railed(*, rail: float = RAIL, n_ev: int = 20, seed: int = 11):
    """``n_ev`` 1 ms / 300 pA blockades on a 1000 pA pore, clipped at ``rail``."""
    n = int(FS * 4.0)
    pad, step, dw = 5_000, (n - 10_000) // n_ev, 100
    x = np.full(n, I0)
    for i in range(n_ev):
        x[pad + i * step : pad + i * step + dw] -= TRUE_DEPTH
    x = x + np.random.default_rng(seed).normal(0.0, SIGMA, n)
    return np.clip(x, rail, 1200.0)


def test_flat_topped_events_are_flagged_and_the_depth_is_named_a_lower_bound():
    df, prov = detect_events(_railed(), FS, min_dwell_s=1e-4)
    assert len(df) == 20
    assert bool(np.asarray(has_flag(df["flag"], FLAG_SATURATED), dtype=bool).all())
    # The number itself cannot be recovered -- the information is gone from the samples.
    # What changed is that the row now says the 200 pA it reports is a floor.
    assert float(df["depth_pA"].median()) == pytest.approx(I0 - RAIL, abs=1.0)
    assert bool(df["accepted"].all()), "a railed event is real; the caveat is not a rejection"


def test_the_warning_counts_the_rows_that_carry_the_token():
    """Counted from the TABLE, never from a parallel computation -- so the sentence
    cannot claim events no row admits to."""
    df, prov = detect_events(_railed(), FS, min_dwell_s=1e-4)
    warns = _build_warnings(dict(prov), df)
    said = [w for w in warns if "flat rail" in w]
    assert len(said) == 1, warns
    assert said[0].startswith("20 of 20 event(s)")
    assert "LOWER BOUNDS" in said[0]
    assert saturated_warning(20, 20) == said[0]


def test_an_unrailed_recording_is_not_flagged_and_says_nothing():
    """The same events with the rail out of reach: no token, no sentence.

    This is the false-positive guard -- the trace is still quantised in the sense that
    every sample is a float32, and its minimum is still attained somewhere.
    """
    df, prov = detect_events(_railed(rail=0.0), FS, min_dwell_s=1e-4)
    assert len(df) == 20
    assert not bool(np.asarray(has_flag(df["flag"], FLAG_SATURATED), dtype=bool).any())
    assert float(df["depth_pA"].median()) == pytest.approx(TRUE_DEPTH, abs=2.0)
    assert not any("flat rail" in w for w in _build_warnings(dict(prov), df))


def test_a_rail_on_the_other_polarity_is_caught_too():
    """An upward (capacitive/positive-going) event on a negative pore rails against the
    trace MAXIMUM. The rule follows the polarity, not a hard-coded direction."""
    n = int(FS * 2.0)
    x = np.full(n, -1500.0)
    for i in range(10):
        s = 5_000 + i * 15_000
        x[s : s + 100] += TRUE_DEPTH
    x = x + np.random.default_rng(2).normal(0.0, SIGMA, n)
    railed = np.clip(x, -2000.0, -1300.0)
    df, _prov = detect_events(railed, FS, min_dwell_s=1e-4)
    assert len(df) == 10
    assert bool(np.asarray(has_flag(df["flag"], FLAG_SATURATED), dtype=bool).all())


def test_only_a_run_at_the_extreme_counts_not_a_single_touching_sample():
    """One sample at the rail is what a deep noise excursion looks like; three in a row
    is what an amplifier sitting on its rail looks like. The rule takes the second."""
    n = 20_000
    cur = np.full(n, I0)
    cur[1_000:1_100] -= TRUE_DEPTH
    cur[5_000:5_100] -= TRUE_DEPTH
    cur = cur + np.random.default_rng(6).normal(0.0, SIGMA, n)
    # Event 0 touches the extreme value once; event 1 holds it for three samples.
    rail = float(cur.min()) - 1.0
    cur[1_050] = rail
    cur[5_050:5_053] = rail
    df = characterize(cur, np.full(n, I0), np.array([1_000, 5_000]),
                      np.array([1_100, 5_100]), FS, -1, noise_rms=SIGMA)
    assert list(np.asarray(has_flag(df["flag"], FLAG_SATURATED), dtype=bool)) == [False, True]


def test_a_rebuild_re_derives_the_token_instead_of_inheriting_a_stale_one():
    """An Event Inspector resize that moves the mask OFF the railed samples must stop
    claiming a flat top -- the token is a statement about the current span."""
    n = 40_000
    cur = np.full(n, I0)
    cur[10_000:10_100] -= TRUE_DEPTH          # this half runs into the rail
    cur[10_200:10_300] -= 100.0               # this one does not
    cur = cur + np.random.default_rng(8).normal(0.0, SIGMA, n)
    cur = np.clip(cur, RAIL, 1200.0)

    prov = {"fs": FS, "polarity": -1, "noise_rms_pA": SIGMA, "span_lo": 0, "span_hi": n}
    railed = recharacterize_span(cur, FS, 10_000, 10_100, provenance=prov)
    clean = recharacterize_span(cur, FS, 10_200, 10_300, provenance=prov,
                                prior=railed.iloc[0])
    assert bool(has_flag(railed["flag"].iloc[0], FLAG_SATURATED))
    assert not bool(has_flag(clean["flag"].iloc[0], FLAG_SATURATED)), (
        "the rebuilt row inherited a flat top it no longer has", clean["flag"].iloc[0])


def test_module_path_puts_the_caveat_on_the_result_a_user_reads(context_factory):
    register_all()
    res = DetectSchmitt().run(context_factory(_railed().astype(np.float32), fs=FS,
                                              params={"min_dwell_s": 1e-4}))
    assert any("flat rail" in w for w in res.warnings), res.warnings
    assert bool(np.asarray(
        has_flag(res.payload.df["flag"], FLAG_SATURATED), dtype=bool).all())
