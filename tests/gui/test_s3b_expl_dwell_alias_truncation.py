"""S3b (repair of a repair) — a duration column spelled ``dwell`` is still a dwell.

EXPL3-04 was the pane printing the dwell provenance story over ``rise_s``/``fall_s``,
which are not dwells. The fix asked one question -- "is this a dwell fit?" -- and
answered it with ``quantity is None``, because ``quantity=None`` is
:func:`stats.fit_dwell_exponential`'s historical dwell default.

That answer is wrong for the aliases the Plots window itself declares it supports.
``plots_panel._DWELL_COLUMNS`` is ``("dwell_s", "dwell", "dwell_time_s")`` -- "columns
the detector's own dwell provenance actually describes" -- and for the bare ``dwell``
spelling ``_column_quantity`` answers unit ``""`` (it does not end in ``_s``), so the
panel passes ``quantity="dwell"`` to the fitter. A dwell, under a name
``quantity is None`` cannot see. The consequence on the shipped surface was not a
mis-worded note but a REFUSED FIT: ``truncation_source_clause`` raised
"...which is not a dwell column" for a bound the detector genuinely imposed, and the
Details pane read

    Fit unavailable: t_max_source='max_dwell_s' claims the detector's dwell ceiling
    bounded a fit to 'dwell', which is not a dwell column; max_dwell_s cannot have
    imposed it

for a frame that fits perfectly well the moment its column is renamed ``dwell_time_s``.

The invariant is one shared list of dwell spellings (:data:`stats.DWELL_QUANTITY_NAMES`)
and one predicate over it, so the window and the fitter cannot disagree about which
columns the detector's dwell provenance describes.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import stats
from nanoscope.core.session import Session
from nanoscope.gui.plots_panel import PlotsWindow, _DWELL_COLUMNS

DET_MIN_S = 2.0e-4
DET_MAX_S = 5.0e-2
N = 900


def _frame(duration_col: str) -> pd.DataFrame:
    """The same numbers under a chosen spelling of the duration column."""
    rng = np.random.default_rng(4)
    dwell = np.clip(rng.exponential(1.5e-3, N) + DET_MIN_S, DET_MIN_S, DET_MAX_S)
    return pd.DataFrame({
        duration_col: dwell,
        "depth_pA": np.abs(rng.normal(80.0, 12.0, N)),
        "accepted": np.ones(N, dtype=bool),
        "filter_limited": np.zeros(N, dtype=bool),
    })


def _window(qtbot, duration_col: str) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"run": _frame(duration_col)},
               {"run": {"min_dwell_s": DET_MIN_S, "max_dwell_s": DET_MAX_S,
                        "fs": 50_000.0}})
    w.set_plot_type("Histogram")
    w.set_histogram_column(duration_col)
    w.set_time_unit("s")
    w.set_fit("exponential")
    return w


def _modelled(w: PlotsWindow) -> str:
    line = next((ln for ln in w.details_text().splitlines() if "right truncation at" in ln), "")
    assert line, w.details_text()
    return line


# ================================================================================================
# the shipped surface: the pane a user of a `dwell`-spelled frame actually reads
# ================================================================================================

@pytest.mark.parametrize("duration_col", ["dwell_s", "dwell", "dwell_time_s"])
def test_every_dwell_spelling_the_window_accepts_actually_fits(qapp, qtbot, duration_col):
    """No spelling in the window's own alias list may cost the user their fit."""
    w = _window(qtbot, duration_col)
    text = w.details_text()
    assert "Fit unavailable" not in text, text
    assert "not a dwell column" not in text, text
    fit = w.fit_result()
    assert fit is not None, text
    # Same data, same detector bounds -> the same tau, whatever the column is called.
    assert fit.params["tau_1"] == pytest.approx(1.526e-3, rel=5e-3)


@pytest.mark.parametrize("duration_col", ["dwell_s", "dwell", "dwell_time_s"])
def test_the_detector_ceiling_is_credited_to_the_detector_under_every_spelling(
        qapp, qtbot, duration_col):
    """The bound IS ``max_dwell_s`` here, so the note must say so -- not hedge, not raise."""
    w = _window(qtbot, duration_col)
    line = _modelled(w)
    assert f"t_max={DET_MAX_S:g}" in line, line
    assert "detector max_dwell_s" in line, line
    assert "NOT a detector limit" not in line, line
    assert w.fit_result().meta["t_max_source"] == stats.T_MAX_FROM_DETECTOR


# ================================================================================================
# the module contract underneath it
# ================================================================================================

def test_a_named_dwell_column_gets_the_dwell_hedge_not_the_generic_one():
    """``quantity`` naming a dwell is still a dwell when no token is passed at all."""
    for name in stats.DWELL_QUANTITY_NAMES:
        assert stats.truncation_source_clause(None, quantity=name) == \
            "(detector max_dwell_s, or curation)", name
    # ... and the columns EXPL3-04 was filed about keep the generic hedge.
    for name in ("rise_s", "fall_s", "t_start_s", "depth_pA"):
        assert stats.truncation_source_clause(None, quantity=name) == \
            "(a fit range, or curation)", name


def test_the_detector_token_is_still_refused_over_a_column_it_cannot_describe():
    """The EXPL3-04 guard must survive being widened: only dwells may claim max_dwell_s."""
    for name in stats.DWELL_QUANTITY_NAMES:
        assert stats.truncation_source_clause(
            stats.T_MAX_FROM_DETECTOR, quantity=name) == "(detector max_dwell_s)", name
    for name in ("rise_s", "fall_s", "depth_pA", "Dwell", "dwell_ms"):
        # `Dwell` and `dwell_ms` are deliberately NOT dwells here: the window matches its
        # alias list case-sensitively and by exact name, so a column it would give no dwell
        # provenance to must not be able to claim the detector's ceiling either.
        with pytest.raises(ValueError, match="not a dwell column"):
            stats.truncation_source_clause(stats.T_MAX_FROM_DETECTOR, quantity=name)


def test_the_window_and_the_fitter_share_one_list_of_dwell_spellings():
    """A private copy is how two surfaces end up disagreeing (``_ORIGIN_COLUMNS``' comment).

    ``plots_panel._DWELL_COLUMNS`` should read ``tuple(stats.DWELL_QUANTITY_NAMES)``;
    until it does, this pins them equal so a name added on either side fails loudly here
    instead of silently costing that column its fit.
    """
    assert tuple(_DWELL_COLUMNS) == tuple(stats.DWELL_QUANTITY_NAMES)
