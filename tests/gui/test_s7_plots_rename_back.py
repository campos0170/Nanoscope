"""S7-FIX2 / S7-SK2 — retyping a source's ORIGINAL name into its row removes the override,
and two sources typed to one legend name can no longer ship as two curves under one label.

THE HOLE. ``_on_source_item_changed`` guarded its whole rename branch with
``text_now != src.label``, so typing the source's own name back -- the gesture a person
uses to undo a rename by hand -- returned before the branch that removes the override
could run. Measured with real double-click and per-character key events on the owner's
-700 mV source: rename to "Apo" (row, override and canvas all "Apo"); type "-700 mV"
back -> row "-700 mV", override STILL "Apo", canvas STILL "Apo  (n = 982)"; a re-Plot
left it; the next Refresh put "Apo" back on the row and the typed correction was gone.
The skeptic's escalation: give "Apo" to the -600 mV source next, and the canvas AND the
exported SVG read ['Apo (n = 982)', 'Apo (n = 236)'] -- two voltages, one name, no
warning anywhere, with the Sources list reading ['-700 mV', 'Apo'].

THE FIX, in two halves: the guard is gone (the branch already did the right thing for
exactly this text), and the figure's names go through ONE resolver
(:meth:`PlotsWindow._legend_names`) that tells a collision apart with the source's own
label and says so in Details.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui.plots_panel import PlotsWindow


def _events(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key: str, label: str, *, n: int, seed: int, colour: str) -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(n, seed=seed),
                      provenance={"fs": 250_000.0, "min_dwell_s": 2e-5},
                      colour=colour, doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


def _sources() -> list[PlotSource]:
    return [_source("doc0:0", "−700 mV", n=300, seed=1, colour="#0072B2"),
            _source("doc1:0", "−600 mV", n=200, seed=2, colour="#D55E00")]


@pytest.fixture
def two(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data(_sources())
    w.set_sources(["doc0:0", "doc1:0"])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    return w


def _rows(w) -> list[str]:
    return [w._source_list.item(i).text() for i in range(w._source_list.count())]


def _canvas(w) -> list[str]:
    legend = getattr(w._plot, "legend", None)
    return [] if legend is None else [label.text for _s, label in legend.items]


def _rename(w, row: int, text: str) -> None:
    """The same ``itemChanged`` the item editor's commit emits."""
    w._source_list.item(row).setText(text)


# ================================================================================================
# the guard
# ================================================================================================

def test_retyping_a_sources_original_name_removes_its_override(two):
    key0 = two._source_list.item(0).data(0x0100)      # Qt.UserRole
    real = two._sources[key0].label
    _rename(two, 0, "Apo")
    assert two._style.label_overrides.get(key0) == "Apo"
    assert _canvas(two)[0].startswith("Apo")

    _rename(two, 0, real)
    assert key0 not in two._style.label_overrides, (
        "typing the original name back is the same request as clearing the row")
    assert _rows(two)[0] == real
    assert _canvas(two)[0].startswith(real), "the drawn legend follows the row"


def test_a_refresh_after_typing_the_name_back_keeps_the_correction(two):
    """MainWindow re-pushes ``set_data`` on every detection, and the rows are rebuilt
    from the style; a stranded override would put the old name back on the row."""
    key0 = two._source_list.item(0).data(0x0100)
    real = two._sources[key0].label
    _rename(two, 0, "Apo")
    _rename(two, 0, real)
    two.set_data(_sources())
    assert _rows(two)[0] == real
    assert key0 not in two._style.label_overrides


def test_a_tick_on_an_unrenamed_row_stores_no_override(two):
    """The guard was never load-bearing: a tick reaches the same handler with the row's
    text equal to its label, and that must leave the style exactly as it was."""
    from PySide6.QtCore import Qt
    before = two._style
    two._source_list.item(1).setCheckState(Qt.Unchecked)
    two._source_list.item(1).setCheckState(Qt.Checked)
    assert two._style.label_overrides == before.label_overrides == {}


# ================================================================================================
# two curves under one name
# ================================================================================================

def test_the_stranded_override_route_no_longer_puts_two_curves_under_one_name(two):
    """S7-SK2's exact sequence."""
    key0 = two._source_list.item(0).data(0x0100)
    real0 = two._sources[key0].label
    _rename(two, 0, "Apo")
    _rename(two, 0, real0)
    _rename(two, 1, "Apo")
    names = _canvas(two)
    assert _rows(two)[:2] == [real0, "Apo"]
    assert sum(1 for c in names if c.startswith("Apo")) == 1, names
    assert names[0].startswith(real0), names


def test_two_sources_typed_to_one_name_are_told_apart_on_the_canvas_and_in_details(two, tmp_path):
    """Typed DELIBERATELY to one name: the rows keep what was typed (they are the rename
    affordance), the figure names the two apart with the source's own label, Details
    says so, and the exported SVG carries the two distinct names."""
    _rename(two, 0, "Apo")
    _rename(two, 1, "Apo")
    assert _rows(two)[:2] == ["Apo", "Apo"], "the rows show what the user typed"

    names = _canvas(two)
    apo = [c for c in names if c.startswith("Apo")]
    assert len(apo) == 2 and len(set(apo)) == 2, names
    assert any("−700 mV" in c for c in apo) and any("−600 mV" in c for c in apo), (
        apo, "told apart by the one name guaranteed unique per source")

    note = [ln for ln in two.details_text().splitlines() if ln.startswith("LEGEND:")]
    assert note and "“Apo”" in note[0] and "Retype one" in note[0], note

    svg = tmp_path / "two_apo.svg"
    two.save_image(svg)
    texts = re.findall(r">([^<>]*Apo[^<>]*)<", svg.read_text(errors="ignore"))
    assert len(set(texts)) == 2, texts

    # ...and the legend-entries checklist offers the same two names, not one twice.
    offered = [lbl for eid, lbl in two._legend_offered.items() if eid.startswith("series:")]
    assert len(set(offered)) == 2, offered


def test_unique_names_draw_exactly_as_typed_and_say_nothing(two):
    """The control: no collision, no suffix, no note."""
    _rename(two, 0, "Apo")
    _rename(two, 1, "Holo")
    names = _canvas(two)
    assert names[0].startswith("Apo  (n =") and names[1].startswith("Holo  (n ="), names
    assert not any(ln.startswith("LEGEND:") for ln in two.details_text().splitlines())
