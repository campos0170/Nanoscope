"""S2 EXPL2-1 / EXPL2-2 — the fitters' truncation/censoring sentences speak the fitted unit.

BL-3 stamped the PARAMETERS of a non-dwell fit with the column's own unit
(``quantity=``/``unit=``), but the sentences the two truncated fitters write about
their own censoring were not routed through the same stamp:

* ``fit_dwell_exponential`` hardcoded ``"right truncation at t_max=300 s ...
  (detector max_dwell_s, or curation)"`` — 300 picoamps labelled seconds, with
  dwell-only provenance ("max_dwell_s") that never existed for a depth column
  (EXPL2-1);
* ``fit_dwell_exp_gamma`` hardcoded ``"DOUBLE truncation, 10 s <= t <= 300 s"`` /
  ``"hard left-truncation at t_min=10 s"`` and a refusal that measured picoamps in
  "min dwell = ... s" (EXPL2-2).

Both notes land in ``meta['warnings']`` and therefore on the Plots Details pane, in
saved projects and through the direct stats API — two lines away from a FIT RANGE
line correctly saying pA. These tests pin the invariant at the source: every
sentence a fit writes about itself is worded in the unit its own ``param_units``
stamp declares, and the dwell default keeps its historical wording bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import stats

RNG = np.random.default_rng(11)
#: An exponential-shaped picoamp sample inside [10, 300] so both truncation
#: corrections engage without tripping the bounds checks.
PA = np.clip(10.0 + RNG.exponential(60.0, 400), None, 300.0 - 1e-6)
#: A dwell control in seconds, for the bit-for-bit historical wording.
DWELL = 2e-4 + RNG.exponential(2e-3, 400)


# ================================================================================================
# EXPL2-1 — fit_dwell_exponential's right-truncation note
# ================================================================================================

def test_exponential_right_truncation_note_speaks_the_fitted_unit():
    fit = stats.fit_dwell_exponential(
        PA, t_min=10.0, t_max=300.0, n_boot=0, quantity="depth_pA", unit="pA")
    note = next(w for w in fit.meta["warnings"] if "right truncation" in w)
    assert "t_max=300 pA is MODELLED" in note, note
    # The dwell-only provenance story must not be told about a picoamp window.
    assert "max_dwell_s" not in note, note
    assert " s " not in note and "300 s" not in note, note
    # ...and the censoring string carries the same unit.
    assert "t_max=300 pA" in fit.meta["censoring"], fit.meta["censoring"]
    assert " s" not in fit.meta["censoring"], fit.meta["censoring"]


def test_exponential_right_truncation_note_converts_through_display():
    """The note goes through the same (factor, name) pair as the degeneracy note,
    so the pane that shows dwells in ms cannot read the bound in raw seconds."""
    t_max = float(np.max(DWELL)) * 1.001
    fit = stats.fit_dwell_exponential(
        DWELL, t_min=2e-4, t_max=t_max, n_boot=0, display=(1e3, "ms"))
    note = next(w for w in fit.meta["warnings"] if "right truncation" in w)
    assert f"t_max={t_max * 1e3:g} ms is MODELLED" in note, note


def test_exponential_dwell_truncation_note_is_bitforbit_historical():
    """CONTROL — the dwell default keeps the exact sentence the censoring tests
    have always pinned, ``detector max_dwell_s`` provenance included."""
    t_max = float(np.max(DWELL)) * 1.001
    fit = stats.fit_dwell_exponential(DWELL, t_min=2e-4, t_max=t_max, n_boot=0)
    note = next(w for w in fit.meta["warnings"] if "right truncation" in w)
    assert note == (
        f"right truncation at t_max={t_max:g} s is MODELLED: events longer than it were "
        f"never reported (detector max_dwell_s, or curation), so the likelihood is "
        f"conditioned on t_min <= t <= t_max. Fitting these events as if only the left "
        f"tail were missing biases tau SHORT (measured -3.7% at t_max = 5 tau, -24% at "
        f"t_max = 2.5 tau)."
    ), note
    assert fit.meta["censoring"] == (
        f"DOUBLE truncation, t_min=0.0002 s <= t <= t_max={t_max:g} s "
        f"(eta == 1 between them); NOT left-truncation only"
    ), fit.meta["censoring"]


def test_exponential_refusals_name_the_quantity_not_dwells():
    """The bounds-mismatch refusals on the same path measured picoamps in
    'max dwell = ... s'; they now name the fitted column and its unit."""
    with pytest.raises(ValueError) as err:
        stats.fit_dwell_exponential(
            PA, t_min=10.0, t_max=50.0, n_boot=0, quantity="depth_pA", unit="pA")
    msg = str(err.value)
    assert "depth_pA value(s) are above the stated upper limit t_max=50 pA" in msg, msg
    assert "dwell" not in msg and " s)" not in msg, msg

    with pytest.raises(ValueError) as err:
        stats.fit_dwell_exponential(
            PA, t_min=50.0, n_boot=0, quantity="depth_pA", unit="pA")
    msg = str(err.value)
    assert "depth_pA value(s) are below the stated lower limit t_min=50 pA" in msg, msg
    assert "dwell" not in msg and "detector is emitting" not in msg, msg


def test_exponential_dwell_refusals_keep_their_historical_wording():
    """CONTROL — the dwell path still says 'longer than the stated upper limit'
    and 'max dwell = ... s', which existing tests match on."""
    with pytest.raises(ValueError, match="longer than the stated upper limit"):
        stats.fit_dwell_exponential(DWELL, t_min=2e-4, t_max=float(DWELL.max()) * 0.5,
                                    n_boot=0)
    with pytest.raises(ValueError, match="shorter than the stated detection limit"):
        stats.fit_dwell_exponential(DWELL, t_min=float(DWELL.max()), n_boot=0)


# ================================================================================================
# EXPL2-2 — fit_dwell_exp_gamma's truncation notes and refusal
# ================================================================================================

def test_exp_gamma_truncation_notes_speak_the_fitted_unit():
    fit = stats.fit_dwell_exp_gamma(
        PA, t_min=10.0, t_max=300.0, quantity="depth_pA", unit="pA")
    note = next(w for w in fit.meta["warnings"] if "DOUBLE truncation" in w)
    assert note == "DOUBLE truncation, 10 pA <= t <= 300 pA", note

    fit = stats.fit_dwell_exp_gamma(PA, t_min=10.0, quantity="depth_pA", unit="pA")
    note = next(w for w in fit.meta["warnings"] if "left-truncation" in w)
    assert note == "hard left-truncation at t_min=10 pA", note


def test_exp_gamma_dwell_truncation_notes_are_bitforbit_historical():
    """CONTROL — the dwell default's notes keep their exact seconds wording."""
    fit = stats.fit_dwell_exp_gamma(DWELL, t_min=2e-4, t_max=float(DWELL.max()) * 1.001)
    hi = float(DWELL.max()) * 1.001
    note = next(w for w in fit.meta["warnings"] if "DOUBLE truncation" in w)
    assert note == f"DOUBLE truncation, 0.0002 s <= t <= {hi:g} s", note


def test_exp_gamma_refusal_names_the_quantity_not_dwells():
    with pytest.raises(ValueError) as err:
        stats.fit_dwell_exp_gamma(PA, t_min=50.0, quantity="depth_pA", unit="pA")
    msg = str(err.value)
    assert "depth_pA value(s) are below the stated lower limit t_min=50 pA" in msg, msg
    assert "dwell" not in msg and "detector is emitting" not in msg, msg

    # The too-few-events door is _clean_1d's, which already names the fitted
    # column; it must keep doing so on this path.
    with pytest.raises(ValueError) as err:
        stats.fit_dwell_exp_gamma(PA[:5], quantity="depth_pA", unit="pA")
    msg = str(err.value)
    assert "depth_pA has only 5 events" in msg, msg
    assert "dwell" not in msg, msg


def test_exp_gamma_dwell_refusal_keeps_its_historical_wording():
    """CONTROL — the dwell refusal still says 'min dwell = ... s'."""
    with pytest.raises(ValueError, match="min dwell ="):
        stats.fit_dwell_exp_gamma(DWELL, t_min=float(DWELL.max()))
