"""S2 EXPL2-1..4 — with a typed Fit range, the Details pane tells one truth about it.

Round two of the fit-caption family: BL-3 stamped the PARAMETER units, but the
notes around them still contradicted the fit whenever the user typed a Fit range on
a no-provenance column:

* EXPL2-1: the exponential's right-truncation note called a 300 pA window
  "t_max=300 s ... (detector max_dwell_s, or curation)" two lines above a correct
  "FIT RANGE [10, 300] pA" line;
* EXPL2-2: exp+gamma printed "DOUBLE truncation, 10 s <= t <= 300 s" /
  "hard left-truncation at t_min=10 s" for picoamp bounds;
* EXPL2-3: with a typed Fit range min, the exponential's no-provenance note still
  claimed "the fit falls back to the shortest observed value (...) as t_min" and
  warned tau was biased SHORT — while ``fit.meta['t_min']`` was the typed bound and
  ``t_min_source`` said 'explicit';
* EXPL2-4: gamma / inverse-gaussian / exp+gamma printed "the fit is untruncated"
  in the same pane as "the likelihood is TRUNCATED to this window".

Everything here is read out of ``details_text()`` on a real
:class:`~nanoscope.gui.plots_panel.PlotsWindow` driven through its public setters
and the actual Fit range QLineEdits, in the style of
``tests/gui/test_bl3_dwell_caption_units.py`` (whose contract this file extends).
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow


def _frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """The BL-3 frame: an exponential-shaped picoamp column with NO provenance."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-4 + rng.exponential(2e-3, n),
        "depth_pA": 2.0 + rng.exponential(80.0, n),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


@pytest.fixture
def pane(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"probe": _frame()})
    w.set_plot_type("Histogram")
    w.set_histogram_column("depth_pA")
    return w


def _set_range(pane: PlotsWindow, lo: str = "", hi: str = "") -> None:
    pane._fit_lo.setText(lo)
    pane._fit_hi.setText(hi)
    pane._redraw()


def _drain(pp: PlotsWindow, qtbot, *, timeout_s: float = 240.0) -> None:
    """Wait for the deferred fits/notes to land, loudly if they never do."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qtbot.wait(10)
        if (not pp._pending_notes and not pp._note_timer.isActive()
                and not pp._inflight_notes):
            qtbot.wait(10)
            return
    raise AssertionError(
        f"the panel never settled: pending={sorted(k[0] for k in pp._pending_notes)} "
        f"inflight={sorted(k[0] for k in pp._inflight_notes)}")


# ================================================================================================
# EXPL2-1 — the right-truncation note speaks pA beside the pA FIT RANGE line
# ================================================================================================

def test_exponential_right_truncation_note_speaks_picoamps(pane, qtbot):
    pane.set_fit("exponential")
    _set_range(pane, "10", "300")
    text = pane.details_text()
    assert "FIT RANGE [10, 300] pA" in text, text
    note = next(ln for ln in text.splitlines() if "right truncation" in ln)
    assert "t_max=300 pA is MODELLED" in note, note
    # No seconds label and no dwell-only provenance about a picoamp window.
    assert "300 s" not in note and "max_dwell_s" not in note, note
    # The BL-3 sweep, now with the truncation note on the pane: nothing says " s".
    assert " s)" not in text, f"a seconds label survived on this picoamp pane:\n{text}"


# ================================================================================================
# EXPL2-2 — exp+gamma's truncation notes speak pA
# ================================================================================================

def test_exp_gamma_truncation_notes_speak_picoamps(pane, qtbot):
    pane.set_fit("exponential + gamma (bimodal)")
    _set_range(pane, "10", "300")
    _drain(pane, qtbot)
    text = pane.details_text()
    assert "DOUBLE truncation, 10 pA <= t <= 300 pA" in text, text
    assert "10 s <= t <= 300 s" not in text, text

    _set_range(pane, "10", "")
    _drain(pane, qtbot)
    text = pane.details_text()
    assert "hard left-truncation at t_min=10 pA" in text, text
    assert "t_min=10 s" not in text, text


# ================================================================================================
# EXPL2-3 — a typed floor is reported as the typed floor, not as a data fallback
# ================================================================================================

def test_typed_fit_range_min_is_not_reported_as_a_fallback(pane, qtbot):
    pane.set_fit("exponential")
    _set_range(pane, "10", "")
    fit = pane.fit_result()
    text = pane.details_text()
    # Ground truth on the FitResult: the likelihood used the typed bound.
    assert fit.meta["t_min"] == 10.0
    assert fit.meta.get("t_min_source") == "explicit"
    # The pane says so — and stops claiming the substitution (and its SHORT-bias
    # story) that never ran on this fit.
    assert "falls back to the shortest observed value" not in text, text
    assert "biases tau SHORT" not in text, text
    assert "typed Fit range min (10 pA)" in text, text


def test_without_a_typed_floor_the_fallback_note_still_tells_the_truth(pane, qtbot):
    """CONTROL — no Fit range: the substitution really runs and must still be said
    (this is the sentence tests/gui/test_fit_provenance.py pins)."""
    pane.set_fit("exponential")
    _set_range(pane, "", "")
    text = pane.details_text()
    assert "falls back to the shortest observed value" in text, text
    assert "biases tau SHORT" in text, text


# ================================================================================================
# EXPL2-4 — "untruncated" is never printed beside "TRUNCATED to this window"
# ================================================================================================

@pytest.mark.parametrize("family", [
    "gamma (peaked)", "inverse-gaussian (first passage)",
    "exponential + gamma (bimodal)"])
def test_no_pane_claims_untruncated_beside_a_truncated_fit(pane, qtbot, family):
    pane.set_fit(family)
    _set_range(pane, "10", "300")
    _drain(pane, qtbot)
    fit = pane.fit_result()
    text = pane.details_text()
    # The fit really is truncated to the typed window...
    assert fit.meta["t_min"] == 10.0 and fit.meta["t_max"] == 300.0
    assert "the likelihood is TRUNCATED to this window" in text, text
    # ...so the pane must not also assert the opposite. It still says what is
    # honestly missing: a measured detector limit.
    assert "the fit is untruncated" not in text, text
    assert "no detection limit is known for “depth_pA”" in text, text
    assert "truncated only to the typed fit range" in text, text


@pytest.mark.parametrize("family", [
    "gamma (peaked)", "inverse-gaussian (first passage)"])
def test_without_a_fit_range_the_untruncated_sentence_stands(pane, qtbot, family):
    """CONTROL — with no range typed the fit IS untruncated, and the historical
    sentence keeps saying so."""
    pane.set_fit(family)
    _set_range(pane, "", "")
    _drain(pane, qtbot)
    text = pane.details_text()
    assert "the fit is untruncated" in text, text
    assert "TRUNCATED to this window" not in text, text
