"""S3 (repair) — the RIGHT truncation says who imposed it, or says it does not know.

``fit_dwell_exponential`` prints one sentence about the bound the likelihood was
conditioned on above:

    right truncation at t_max=0.004 s is MODELLED: events longer than it were never
    reported (detector max_dwell_s, or curation), ...

The parenthetical was a hardcoded ASSERTION on the dwell path. EXPL3-04 fixed the
non-dwell columns (``rise_s``, ``depth_pA``, ...) by refusing to call their ceiling a
detector setting at all -- but on ``dwell_s`` itself, the one column that was frozen
"bit-for-bit", the sentence still named ``max_dwell_s`` whatever the caller had passed.
The Plots window intersects the detector's ceiling with a typed Fit range, so on the
owner's runs the pane blamed a ``max_dwell_s`` of 0.05 s for a truncation that happened
at the 0.004 s the analyst typed -- while the real 0.05 s appeared nowhere on the pane.

``t_max`` is a number; WHO imposed it is a second fact, and this module now carries it:
``t_max_source`` + :func:`stats.truncation_source_clause`, one writer for the words, and
``meta['t_max_source']`` so a stored fit can still answer the question. ``None`` is the
honest "not recorded" and keeps both historical hedges bit-for-bit, which is what the
Fits panel (``modules.fit_dwell``, which never had anything but the detector's ceiling
to pass) still gets.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import stats

RNG = np.random.default_rng(7)
DWELL = np.clip(RNG.exponential(1.5e-3, 800) + 2e-4, 2e-4, 5e-3)
T_MIN = 2e-4
T_MAX = 4e-3
DWELL_IN = DWELL[DWELL <= T_MAX]


def _note(fit) -> str:
    return next(w for w in fit.meta["warnings"] if "right truncation" in w)


# ================================================================================================
# the words
# ================================================================================================

def test_the_hedge_is_what_an_unknowing_caller_gets():
    """CONTROL — no ``t_max_source``, and the sentence is the one always printed."""
    fit = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0)
    assert _note(fit) == (
        f"right truncation at t_max={T_MAX:g} s is MODELLED: events longer than it were "
        f"never reported (detector max_dwell_s, or curation), so the likelihood is "
        f"conditioned on t_min <= t <= t_max. Fitting these events as if only the left "
        f"tail were missing biases tau SHORT (measured -3.7% at t_max = 5 tau, -24% at "
        f"t_max = 2.5 tau)."
    ), _note(fit)


def test_a_detector_ceiling_reads_as_the_detector():
    fit = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0,
                                      t_max_source=stats.T_MAX_FROM_DETECTOR)
    line = _note(fit)
    assert "(detector max_dwell_s)" in line, line
    # ...and stops hedging: the detector's ceiling is not "or curation".
    assert "or curation" not in line, line


def test_a_typed_ceiling_does_not_name_the_detector():
    fit = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0,
                                      t_max_source=stats.T_MAX_FROM_FIT_RANGE)
    line = _note(fit)
    assert "max_dwell_s" not in line, line
    assert "NOT a detector limit" in line, line


def test_a_curated_ceiling_does_not_name_the_detector():
    fit = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0,
                                      t_max_source=stats.T_MAX_FROM_CURATION)
    line = _note(fit)
    assert "max_dwell_s" not in line, line
    assert "curation" in line and "NOT a detector limit" in line, line


@pytest.mark.parametrize("source", [None, stats.T_MAX_FROM_FIT_RANGE,
                                    stats.T_MAX_FROM_CURATION])
def test_a_non_dwell_column_never_invents_a_detector_ceiling(source):
    """``depth_pA`` has no ``max_dwell_s`` to be bounded by, whatever the token says."""
    fit = stats.fit_dwell_exponential(DWELL_IN * 1e4, t_min=T_MIN * 1e4, t_max=T_MAX * 1e4,
                                      n_boot=0, quantity="depth_pA", unit="pA",
                                      t_max_source=source)
    line = _note(fit)
    assert "max_dwell_s" not in line, line
    assert "depth_pA values above it" in line, line


def test_the_detector_token_is_refused_over_a_column_it_cannot_describe():
    """The wrong-column fault EXPL3-04 removed must not be reachable through a token.

    ``max_dwell_s`` is a dwell setting. Accepting it for a ``depth_pA`` fit would print
    "(detector max_dwell_s)" under a ceiling of 40 pA -- a mechanism that could not have
    produced that bound. The panel cannot reach this (``_dwell_inputs`` answers None for
    every non-dwell column, so ``_t_max_source`` can only say ``explicit`` there); the
    module refuses it anyway, because the contract is the module's to keep.
    """
    with pytest.raises(ValueError, match="not a dwell column"):
        stats.fit_dwell_exponential(DWELL_IN * 1e4, t_min=T_MIN * 1e4, t_max=T_MAX * 1e4,
                                    n_boot=0, quantity="depth_pA", unit="pA",
                                    t_max_source=stats.T_MAX_FROM_DETECTOR)


# ================================================================================================
# the contract
# ================================================================================================

def test_the_clause_writer_is_shared_and_refuses_a_misspelt_token():
    """A misspelt provenance is a programming error, not a reason to hedge quietly."""
    assert stats.truncation_source_clause(None) == "(detector max_dwell_s, or curation)"
    assert stats.truncation_source_clause(None, quantity="rise_s") == \
        "(a fit range, or curation)"
    with pytest.raises(ValueError, match="unknown t_max_source"):
        stats.truncation_source_clause("max_dwell")


def test_the_fitter_rejects_a_misspelt_token_on_every_call():
    with pytest.raises(ValueError, match="unknown t_max_source"):
        stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0,
                                    t_max_source="detector")


def test_a_provenance_without_a_bound_is_refused_rather_than_ignored():
    """Nothing for the label to describe -- and a silent drop is a silent no-op."""
    with pytest.raises(ValueError, match="no t_max was given"):
        stats.fit_dwell_exponential(DWELL, t_min=T_MIN, n_boot=0,
                                    t_max_source=stats.T_MAX_FROM_FIT_RANGE)


def test_the_source_travels_with_the_fit():
    """Stored beside ``t_max``, as ``t_min_source`` is stored beside ``t_min``."""
    fit = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0,
                                      t_max_source=stats.T_MAX_FROM_FIT_RANGE)
    assert fit.meta["t_max_source"] == "explicit"
    assert fit.meta["t_max"] == pytest.approx(T_MAX)
    plain = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0)
    assert plain.meta["t_max_source"] is None


def test_only_the_words_move():
    """The label is not an input to the likelihood: same params, same BIC, same curve."""
    a = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0)
    b = stats.fit_dwell_exponential(DWELL_IN, t_min=T_MIN, t_max=T_MAX, n_boot=0,
                                    t_max_source=stats.T_MAX_FROM_FIT_RANGE)
    assert a.params == b.params
    assert a.bic == b.bic and a.aic == b.aic
    assert a.meta["censoring"] == b.meta["censoring"]
