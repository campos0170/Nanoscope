"""Fit verdicts remain in Details and metadata, never in plot artwork.

These tests retain the same rejected-fit fixtures and statistical premises while
pinning the requested separation of diagnostics from publication images.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import re
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import PlotsWindow, StyleSettings


def _narrow_peak(n: int = 400, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": np.clip(rng.normal(0.018, 0.0015, n), 0.005, None),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _exponential(n: int = 400, *, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key: str, label: str, df: pd.DataFrame, colour: str) -> PlotSource:
    return PlotSource(key=key, label=label, df=df,
                      provenance={"fs": 250_000.0, "min_dwell_s": 2e-5,
                                  "detector": "schmitt_hysteresis"},
                      colour=colour, doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


def _window(qtbot, *sources: PlotSource) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data(list(sources))
    w.set_sources([s.key for s in sources])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_time_unit("ms")
    return w


def _title(w) -> str:
    return re.sub(r"<[^>]+>", "", str(w._plot.titleLabel.text)).strip()


def _legend(w) -> list[str]:
    legend = getattr(w._plot, "legend", None)
    return [] if legend is None else [label.text for _s, label in legend.items]


@pytest.fixture
def rejected(qapp, qtbot):
    w = _window(qtbot, _source("doc0:0", "apo −600 mV", _narrow_peak(), "#0072B2"))
    w.set_fit("exponential")
    w._redraw()
    gof = w._last_fit.meta.get("gof")
    assert gof and gof.get("describes_data") is False, f"premise: REJECTED -- {gof}"
    return w


# ================================================================================================

@pytest.mark.parametrize("show_gof", [True, False])
@pytest.mark.parametrize("values_box", [False, True])
def test_a_rejected_fit_says_so_in_the_title_and_the_legend_at_every_setting_of_the_two_switches(
        rejected, show_gof, values_box):
    rejected._style = replace(rejected._style, show_gof=show_gof,
                              fit_values_on_plot=values_box)
    rejected._redraw()
    assert "KS REJECTED" not in _title(rejected)
    assert "KS REJECTED" in rejected.details_text()
    fit_rows = [t for t in _legend(rejected) if t.startswith("fit:")]
    assert fit_rows and "KS REJECTED" not in fit_rows[0]
    if values_box:
        assert "REJECTED" not in rejected._fit_values_item.text
        assert ("KS =" in rejected._fit_values_item.text) is show_gof
    else:
        assert rejected._fit_values_item is None


def test_the_shipped_defaults_are_the_case_that_used_to_ship_a_bare_tau(rejected):
    """``StyleSettings()`` is the state the finding measured: verdict switch on, values
    box off. That is the 2x2 cell the figure was silent in."""
    assert StyleSettings().show_gof is True and StyleSettings().fit_values_on_plot is False
    rejected._style = replace(rejected._style, show_gof=True, fit_values_on_plot=False)
    rejected._redraw()
    assert "τ =" in _title(rejected) and "KS REJECTED" not in _title(rejected)
    assert "DOES NOT DESCRIBE THESE DATA" in rejected.details_text()
    assert rejected._fit_values_item is None, "the verdict did not need the box"


def test_the_verdict_survives_save_image(rejected, tmp_path):
    """The title and the legend are the surfaces the export keeps."""
    rejected._style = replace(rejected._style, fit_values_on_plot=False)
    rejected._redraw()
    svg = tmp_path / "rejected.svg"
    rejected.save_image(svg)
    text = svg.read_text(errors="ignore")
    assert "KS REJECTED" not in text
    assert "KS REJECTED" in rejected.details_text()


def test_the_tag_survives_the_legends_middle_elision(rejected):
    """A legend entry over 34 characters is middle-elided to its two ends; the tag is
    sized to be one whole end, so "…S REJECTED" cannot happen."""
    fit_rows = [t for t in _legend(rejected) if t.startswith("fit:")]
    assert "REJECTED" not in fit_rows[0]
    assert "τ" in fit_rows[0]


def test_an_overlay_tags_each_rejected_fit_in_its_own_legend_entry(qapp, qtbot):
    """Two sources: the title stays empty (one summary fits a title, two do not) and the
    legend names every fit, so the verdict has to ride on each entry."""
    w = _window(qtbot,
                _source("doc0:0", "peak", _narrow_peak(), "#0072B2"),
                _source("doc1:0", "expo", _exponential(), "#D55E00"))
    w._style = replace(w._style, fit_values_on_plot=False)
    w.set_fit("exponential")
    w._redraw()
    rows = {t.split(" fit:")[0]: t for t in _legend(w) if " fit:" in t}
    assert set(rows) == {"peak", "expo"}, _legend(w)
    assert "REJECTED" not in rows["peak"]
    assert f"{pp._KS_REJECTED_TAG}: peak" in w.details_text()
    assert "REJECTED" not in rows["expo"], rows["expo"]


def test_a_fit_the_test_accepts_is_not_tagged(qapp, qtbot):
    w = _window(qtbot, _source("doc0:0", "expo", _exponential(), "#0072B2"))
    w._style = replace(w._style, show_gof=True, fit_values_on_plot=False)
    w.set_fit("exponential")
    w._redraw()
    assert w._last_fit.meta["gof"]["describes_data"] is True
    assert "REJECTED" not in _title(w)
    assert not any("REJECTED" in t for t in _legend(w))


def test_the_tooltip_and_the_docstring_now_name_the_surfaces_the_verdict_reaches(rejected):
    tip = rejected._gof_check.toolTip()
    assert "Details" in tip and "provenance" in tip and "Save image" in tip
    assert "not on the plot" in tip
    assert rejected._gof_check.text() == "KS statistic on figure"
