"""S3 EXPL3-02 — an R² with no residual degrees of freedom is arithmetic, not evidence.

A straight line has two free parameters. Through two points it passes exactly and
R² = 1.000 -- for ANY two points, including two points of pure noise. The log-log line
has two parameters as well (coefficient and exponent), and so do both voltage-dependence
trends (slope/intercept, A/V₀).

The app already knew this in one place: ``gui.psd_panel._flicker_lines`` writes "A
log-log line has two free parameters, so R² = … over N points is nearly free and is not
evidence the power law holds". Nowhere else. So the Plots scatter printed

    y = 0.05635·x + 90.73   (x in ms, y in pA)   R² = 1.000   (n = 2)

in its title, its legend and its Details -- in the same window whose histogram refuses
those identical two events outright ("a distribution fit to a handful of events is
decoration, not a measurement") -- and the voltage-dependence panel printed "Goodness of
fit: R² = 1.00000 · n = 2 recording(s)" for a line through two voltages. The sweep-2 fix
on those same lines had already written the full sentence for the OTHER degenerate case
("R² undefined (y does not vary)"); this is its mirror.

ONE writer, in ``analysis.stats``, because both surfaces are fitting surfaces and the
answer must not depend on which window asked. These tests pin the writer and both call
sites.
"""

from __future__ import annotations

import pytest

from nanoscope.analysis import stats


# ================================================================================================
# the writer
# ================================================================================================

def test_the_tag_is_empty_as_soon_as_one_residual_degree_of_freedom_exists():
    """A guard on a degenerate case, not a running commentary on small samples."""
    assert stats.r2_dof_tag(3, 2) == ""
    assert stats.r2_dof_tag(986, 2) == ""
    assert stats.r2_dof_caveat(3, 2) == ""


def test_exactly_as_many_points_as_parameters_is_exact_by_construction():
    assert stats.r2_dof_tag(2, 2) == "exact by construction"
    note = stats.r2_dof_caveat(2, 2)
    assert "EXACT BY CONSTRUCTION" in note
    assert "2 free parameters" in note
    assert "no residual degrees of freedom" in note
    assert "no goodness of fit to report" in note


def test_fewer_points_than_parameters_is_under_determined():
    assert stats.r2_dof_tag(1, 2) == "under-determined"
    note = stats.r2_dof_caveat(1, 2)
    assert "under-determined" in note
    assert "not identified" in note


def test_the_model_and_the_noun_are_the_callers_to_choose():
    """The sentence has to read correctly for recordings as well as for points."""
    note = stats.r2_dof_caveat(2, 2, model="trend", noun="recording")
    assert "a trend has 2 free parameters" in note
    assert "2 recordings were fitted" in note
    assert "point" not in note


def test_the_writer_is_exported():
    assert "r2_dof_tag" in stats.__all__ and "r2_dof_caveat" in stats.__all__


@pytest.mark.parametrize("n", [0, 1, 2, 3, 40])
def test_tag_and_caveat_always_agree_about_whether_there_is_a_problem(n):
    """Two surfaces never disagree about one fact -- including these two strings."""
    assert bool(stats.r2_dof_tag(n, 2)) == bool(stats.r2_dof_caveat(n, 2))
