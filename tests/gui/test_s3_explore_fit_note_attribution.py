"""S3 JAB-03 / EXPL3-05 — one source label per Details line, and only one.

THE DEFECT. With two or more sources ticked, ``PlotsWindow._draw_histogram`` prefixed
every note it got back from ``_overlay_fit`` with ``f"{label}: "`` -- and ``_overlay_fit``
had already prefixed the notes it built itself with a ``who`` that carried NO separator.
Both halves fired, so the pane that carries tau, BIC, the bootstrap interval and the
goodness-of-fit verdict read

    poreA.abf: poreA.abfexponential (left-truncated MLE)
    poreA.abf: poreA.abfBIC = -678.661   AIC = -681.371   components = 1
    poreA.abf: poreA.abf95% CI (percentile bootstrap, ...): tau_1 [15.01, 18.78] ms

with the model name, the KS verdict and the FIT RANGE keyword all corrupted into tokens
like ``abfBIC`` and ``abf95%``. It appeared EXACTLY in the comparison view the overlay
exists for; single-source draws were clean, which is why every shipped assertion (all of
them membership tests -- ``"pore A" in text``) passed over it.

THE RULE THESE PIN. The pane that collects the notes is the one surface that knows how
many series are drawn, and it already attributes every OTHER note it shows
(``_hist_sample``'s exclusions, the per-source detection limits, ``_detection_notes``).
So ``_overlay_fit`` returns UNATTRIBUTED notes and the caller labels each exactly once --
which is also the only way the notes ``_overlay_fit`` merely forwards (``_finite``'s
exclusions, the fitter's own ``meta['warnings']``) get labelled at all.

Written against the pane text a user reads, through the real widgets, with two sources
whose labels are a prefix-free pair so a doubled label cannot hide inside a single one.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import PlotsWindow

LABEL_A = "pore A"
LABEL_B = "pore B"


# ================================================================================================
# builders
# ================================================================================================

def _frame(n: int = 400, *, tau: float = 2e-3, seed: int = 7) -> pd.DataFrame:
    """A frame big enough that every fit family runs (>= 20 events) and returns notes."""
    rng = np.random.default_rng(seed)
    dwell = rng.exponential(tau, n) + 5e-5
    return pd.DataFrame({
        "dwell_s": dwell,
        "depth_pA": np.abs(rng.normal(80.0, 12.0, n)),
        "rise_s": np.abs(rng.normal(6e-5, 1e-5, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _two_source_window(qtbot, *, col: str = "dwell_s", fit: str = "exponential") -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    a = PlotSource(key="a", label=LABEL_A, df=_frame(seed=1),
                   provenance={"min_dwell_s": 5e-5, "fs": 50_000.0})
    b = PlotSource(key="b", label=LABEL_B, df=_frame(seed=2),
                   provenance={"min_dwell_s": 5e-5, "fs": 50_000.0})
    w.set_data([a, b])
    w.set_sources([LABEL_A, LABEL_B])
    w.set_plot_type("Histogram")
    w.set_histogram_column(col)
    w.set_fit(fit)
    return w


def _pane_lines(w: PlotsWindow) -> list[str]:
    return [ln for ln in w.details_text().splitlines() if ln.strip()]


# ================================================================================================
# THE REGRESSION: never twice, and never welded on
# ================================================================================================

@pytest.mark.parametrize("fit", ["exponential", "gamma (peaked)", "gaussian-mixture"])
def test_no_details_line_carries_its_source_label_twice(qapp, qtbot, fit):
    col = "depth_pA" if fit == "gaussian-mixture" else "dwell_s"
    w = _two_source_window(qtbot, col=col, fit=fit)
    doubled = [ln for ln in _pane_lines(w)
               if ln.count(LABEL_A) > 1 or ln.count(LABEL_B) > 1]
    assert not doubled, doubled


@pytest.mark.parametrize("fit", ["exponential", "gamma (peaked)", "gaussian-mixture"])
def test_no_details_line_glues_the_label_onto_the_text(qapp, qtbot, fit):
    """A label must be followed by its separator, never by the note's first character.

    This is the assertion the shipped ``test_the_details_pane_reports_every_source_fit_by_name``
    was missing: it asserted membership, which "pore A: pore Aexponential" satisfies.
    """
    col = "depth_pA" if fit == "gaussian-mixture" else "dwell_s"
    w = _two_source_window(qtbot, col=col, fit=fit)
    glued = [ln for ln in _pane_lines(w)
             if re.search(rf"({re.escape(LABEL_A)}|{re.escape(LABEL_B)})(?=[A-Za-z0-9%])", ln)]
    assert not glued, glued


def test_every_fit_note_is_attributed_exactly_once_and_in_the_house_form(qapp, qtbot):
    """The fit's own lines carry ``"<label>: "`` -- the same form every other note uses."""
    w = _two_source_window(qtbot)
    lines = _pane_lines(w)
    for needle in ("(left-truncated MLE)", "BIC = ", "KS D = "):
        hits = [ln for ln in lines if needle in ln]
        assert hits, f"expected a {needle!r} line in a two-source pane\n" + "\n".join(lines)
        for ln in hits:
            assert ln.startswith((f"{LABEL_A}: ", f"{LABEL_B}: ")), ln
            # ...and the label does not appear again anywhere in the rest of the line.
            assert LABEL_A not in ln[len(LABEL_A) + 2:], ln
            assert LABEL_B not in ln[len(LABEL_B) + 2:], ln


def test_both_sources_are_still_named_in_the_pane(qapp, qtbot):
    """De-duplicating must not de-attribute: each source still labels its own fit."""
    w = _two_source_window(qtbot)
    text = w.details_text()
    assert f"{LABEL_A}: " in text and f"{LABEL_B}: " in text
    assert text.count("BIC = ") >= 2


def test_a_single_source_pane_stays_unprefixed(qapp, qtbot):
    """The one-series case is not a comparison and must not grow a label."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([PlotSource(key="a", label=LABEL_A, df=_frame(),
                           provenance={"min_dwell_s": 5e-5, "fs": 50_000.0})])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    lines = _pane_lines(w)
    assert any(ln.startswith("exponential (") for ln in lines), lines
    assert not any(ln.startswith(f"{LABEL_A}: ") and "BIC = " in ln for ln in lines), lines


def test_the_legend_entry_keeps_its_historical_form(qapp, qtbot):
    """The fix moves the attribution of NOTES; the legend text is unchanged."""
    w = _two_source_window(qtbot)
    entries = [str(t.text) for _s, t in w._plot.legend.items]
    assert any(e.startswith(f"{LABEL_A} fit:") for e in entries), entries
    assert any(e.startswith(f"{LABEL_B} fit:") for e in entries), entries
