"""Sweep-7 fix-up, Plots window: four holes the skeptics found in the wave-1 fixes.

A1-N1  The KS verdict rides the title's WARN channel (prepended over a typed title),
       not the automatic title text a typed Style ▸ Title displaces. Measured before:
       typed title + "Show legend" off + the values box at its default shipped the
       owner's rejected -600 mV exponential with no verdict anywhere on the saved SVG.
A1-N2  Every figure writer that names a source goes through ``_legend_name`` -- the
       scatter FIT legend entry, the in-figure provenance caption and the export's
       fallback series names still read the raw override, so two rows typed to "Apo"
       were told apart in the series entries and put back under one name beside them.
A1-N3  The density note no longer says the manual-measurement table "does not declare"
       a unit its own ``units`` column declares on every row; it reads that column.
A1-N4  The two import disclosures are one record PER ORIGIN: the second import no
       longer overwrites the first file's account, so an undo back onto file A's own
       state finds A's sentences (and the re-save over A is not silent). The refusal
       sentence still quotes the refused NAME verbatim, not the record that holds it.

Every claim here is read off a driven window: the drawn title item, the drawn legend,
the provenance LabelItem, the saved SVG's text nodes, Details.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import json
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
from nanoscope.gui.plots_panel import PlotsWindow, _ConfigLosses, _NormRefusal, plot_config_to_json


# ================================================================================================
# builders
# ================================================================================================

def _narrow_peak(n: int = 400, *, seed: int = 11) -> pd.DataFrame:
    """A dwell distribution an exponential cannot describe: the KS test REJECTS it."""
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


def _events(n: int = 300, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "rise_s": np.abs(rng.normal(1e-5, 2e-6, n)),
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)),
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


def _svg_texts(path) -> list[str]:
    txt = path.read_text(errors="ignore")
    return [t for t in re.findall(r">([^<>]+)<", txt) if t.strip()]


def _density_note(w) -> str:
    return next(ln.strip() for ln in w.details_text().splitlines() if "bin width" in ln)


# ================================================================================================
# A1-N1 -- the KS verdict is on the title's warn channel
# ================================================================================================

@pytest.fixture
def rejected(qapp, qtbot):
    w = _window(qtbot, _source("doc0:0", "apo −600 mV", _narrow_peak(), "#0072B2"))
    w.set_fit("exponential")
    w._redraw()
    gof = w._last_fit.meta.get("gof")
    assert gof and gof.get("describes_data") is False, f"premise: REJECTED -- {gof}"
    return w


def test_a_typed_title_keeps_the_ks_verdict_ahead_of_the_users_own_text(rejected):
    """The comparability verdict's rule, applied to the fit's: the user may retitle the
    plot; they may not retitle away the fact that the curve does not describe the data."""
    rejected.set_title("Figure 3: apo-ferritin dwell times")
    rejected._redraw()
    title = _title(rejected)
    assert title.startswith("⚠ " + pp._KS_REJECTED_TAG), title
    assert "Figure 3: apo-ferritin dwell times" in title, title


def test_the_verdict_is_said_once_on_the_automatic_title(rejected):
    """Prepended through ``warn`` AND appended to the summary would read 'KS REJECTED ·
    τ = … — KS REJECTED'; the title takes the plain summary."""
    title = _title(rejected)
    assert title.count("KS REJECTED") == 1, title
    assert title.startswith("⚠ " + pp._KS_REJECTED_TAG) and "τ =" in title, title


def test_a_typed_title_with_the_legend_off_still_exports_the_verdict(rejected, tmp_path):
    """The exact figure the skeptic saved: typed title, Show legend off, values box at
    its default (off). The curve is drawn; the SVG must still carry the verdict."""
    rejected._style = replace(rejected._style, fit_values_on_plot=False)
    rejected.set_title("Figure 3: apo-ferritin dwell times, -600 mV")
    rejected._legend_check.setChecked(False)
    rejected._redraw()
    assert _legend(rejected) == [], "premise: the legend is off"
    assert rejected._fit_values_item is None, "premise: the values box is off"
    assert rejected._last_fit is not None, "premise: the curve is still drawn"
    svg = tmp_path / "typed_title_legend_off.svg"
    rejected.save_image(svg)
    texts = _svg_texts(svg)
    assert any("KS REJECTED" in t for t in texts), texts
    assert "DOES NOT DESCRIBE THESE DATA" in rejected.details_text()


def test_the_legend_entry_still_carries_the_tag_while_the_legend_is_shown(rejected):
    """The per-fit tag on the legend entry is kept -- on an overlay it is what says WHICH
    curve was thrown out; only the title's copy moved channels."""
    fit_rows = [t for t in _legend(rejected) if t.startswith("fit:")]
    assert fit_rows and fit_rows[0].endswith(pp._KS_REJECTED_TAG), fit_rows


def test_an_accepted_fit_puts_no_verdict_over_a_typed_title(qapp, qtbot):
    w = _window(qtbot, _source("doc0:0", "expo", _exponential(), "#0072B2"))
    w.set_fit("exponential")
    w.set_title("Figure 4")
    w._redraw()
    assert w._last_fit.meta["gof"]["describes_data"] is True, "premise: a fit that passes"
    assert _title(w) == "Figure 4", _title(w)


def test_an_overlay_names_the_rejected_source_in_the_title_by_the_figures_own_name(qapp, qtbot):
    """Two sources: the automatic title is empty (one summary fits a title, two do not),
    so before this the verdict lived only in the legend entries. The headline names the
    rejected source by the name the LEGEND calls it (the override), not the raw label."""
    w = _window(qtbot,
                _source("doc0:0", "peak", _narrow_peak(), "#0072B2"),
                _source("doc1:0", "expo", _exponential(), "#D55E00"))
    w._style = replace(w._style, fit_values_on_plot=False)
    w._source_list.item(0).setText("Apo")
    w.set_fit("exponential")
    w._redraw()
    assert _title(w) == f"⚠ {pp._KS_REJECTED_TAG}: Apo", _title(w)
    rows = {t.split(" fit:")[0]: t for t in _legend(w) if " fit:" in t}
    assert rows["Apo"].endswith(pp._KS_REJECTED_TAG) and "REJECTED" not in rows["expo"], rows


def test_the_verdict_stacks_behind_the_titles_other_headlines_in_one_channel(rejected):
    """One channel, one joiner: whatever else the title warns about comes first, and the
    KS mark is joined the way ``_warn_headline`` joins everything."""
    assert pp.PlotsWindow._warn_headline("⚠ NOT COMPARABLE", "", pp.PlotsWindow._ks_headline([""])) \
        == f"⚠ NOT COMPARABLE   ·   ⚠ {pp._KS_REJECTED_TAG}"
    assert pp.PlotsWindow._ks_headline([]) == ""
    assert pp.PlotsWindow._ks_headline(["A", "B"]) == f"⚠ {pp._KS_REJECTED_TAG}: A, B"


def test_the_tooltip_says_the_title_carries_it_over_a_typed_title(rejected):
    tip = rejected._gof_check.toolTip()
    assert "AHEAD of any title you type" in tip, tip
    assert "while those are shown" in tip, "the legend entry and the box are conditional"


# ================================================================================================
# A1-N2 -- every figure writer that names a source reads `_legend_name`
# ================================================================================================

@pytest.fixture
def two_apo_scatter(qapp, qtbot):
    """Two sources with SHORT labels (so the legend's middle elision cannot eat the
    distinguishing part), both rows typed to "Apo", scatter with a linear fit and the
    in-figure provenance caption on."""
    w = _window(qtbot,
                _source("doc0:0", "A", _events(seed=1), "#0072B2"),
                _source("doc1:0", "B", _events(seed=2), "#D55E00"))
    w.set_plot_type("Scatter")
    w._scatter_x.setCurrentText("dwell_s")
    w._scatter_y.setCurrentText("depth_pA")
    w._style = replace(w._style, show_provenance=True)
    w.set_scatter_fit("linear")
    w._source_list.item(0).setText("Apo")
    w._source_list.item(1).setText("Apo")
    w._redraw()
    return w


def test_the_scatter_fit_entries_name_the_sources_apart_as_the_series_entries_do(two_apo_scatter):
    names = _legend(two_apo_scatter)
    series = [t for t in names if " fit" not in t]
    fits = [t.split(" fit")[0] for t in names if " fit" in t]
    assert len(fits) == 2 and len(set(fits)) == 2, names
    assert set(fits) == {"Apo — A", "Apo — B"}, names
    assert [s.split("  (")[0] for s in series] == ["Apo — A", "Apo — B"], names


def test_the_provenance_caption_inside_the_figure_names_the_sources_apart(two_apo_scatter):
    prov = two_apo_scatter._prov_item
    assert prov is not None, "premise: the caption is drawn"
    lines = [ln.strip() for ln in re.sub(r"<[^>]+>", "\n", str(prov.text)).splitlines()
             if ln.strip()]
    heads = [ln.split(":")[0] for ln in lines]
    assert heads == ["Apo — A", "Apo — B"], lines


def test_the_saved_image_carries_no_two_sources_under_one_name(two_apo_scatter, tmp_path):
    svg = tmp_path / "two_apo.svg"
    two_apo_scatter.save_image(svg)
    apo = [t for t in _svg_texts(svg) if "Apo" in t]
    assert apo and all("Apo — A" in t or "Apo — B" in t for t in apo), apo


def test_the_exports_fallback_series_names_go_through_the_same_resolver(two_apo_scatter):
    """The fallback runs when a draw offered no legend entries; it must name the drawn
    sources the way the figure does, not by the raw override."""
    two_apo_scatter._legend_offered.clear()
    series, _fits = two_apo_scatter._export_series_labels()
    assert series == ["Apo — A", "Apo — B"], series


def test_no_figure_writer_reads_the_raw_override_any_more():
    """The invariant, pinned at the source: ``_legend_label`` is the ROW's text (the rename
    affordance) and the resolver's input; nothing that writes onto or beside the figure
    calls it directly."""
    src = open(pp.__file__, encoding="utf-8").read()
    body = src[src.index("class PlotsWindow"):]
    callers = [ln.strip() for ln in body.splitlines()
               if "self._legend_label(src)" in ln and "def " not in ln]
    figure_writers = [ln for ln in callers if " fit: " in ln or "lines.append" in ln
                      or "series = " in ln]
    assert figure_writers == [], figure_writers


# ================================================================================================
# A1-N3 / X-SURF-7 -- the density note reads the table's own `units` column
# ================================================================================================

def _manual_table(units, n: int = 300, *, seed: int = 0) -> pd.DataFrame:
    """The manual-measurement table's shape: every MEASUREMENT_SCHEMA column, and the
    ``units`` column set as asked (None = no such column)."""
    from nanoscope.analysis import measure
    rng = np.random.default_rng(seed)
    row = {}
    for c, (d, _u, _desc) in measure.MEASUREMENT_SCHEMA.items():
        if d == "float64":
            row[c] = rng.normal(120.0, 25.0, n)
        elif d == "int64":
            row[c] = np.arange(n)
        else:
            row[c] = pd.Series(["apo -700 mV"] * n, dtype="string")
    row["dwell_s"] = np.abs(rng.normal(0.02, 0.005, n))
    df = pd.DataFrame(row)
    if units is None:
        df = df.drop(columns=["units"])
    else:
        df["units"] = pd.Series(list(units) if not isinstance(units, str) else [units] * n,
                                dtype="string")
    return df


def _density_window(qtbot, sources: dict) -> PlotsWindow:
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data(sources)
    w.set_plot_type("Histogram")
    w.set_normalization("density")
    return w


@pytest.mark.parametrize("col", ["amplitude", "level", "baseline", "area_ecd", "n_samples"])
def test_a_table_whose_units_column_reads_pA_is_not_told_it_declares_nothing(qapp, qtbot, col):
    w = _density_window(qtbot, {"manual measurements": _manual_table("pA")})
    w.set_histogram_column(col)
    w._redraw()
    note = _density_note(w)
    assert "does not declare" not in note, note
    assert "reads “pA” in this table's units column" in note, note
    # A fact about the CHANNEL, never a unit for the column: var is pA², n_samples a
    # count, area_ecd pA·s -- the schema trap the skeptic named.
    assert "1/pA" not in note and "dimensionless" not in note, note


def test_a_table_with_no_units_column_says_so(qapp, qtbot):
    w = _density_window(qtbot, {"manual measurements": _manual_table(None)})
    w.set_histogram_column("amplitude")
    w._redraw()
    note = _density_note(w)
    assert "has no units column and does not declare what that unit is" in note, note


def test_a_blank_units_column_is_called_blank_not_absent(qapp, qtbot):
    w = _density_window(qtbot, {"manual measurements": _manual_table("")})
    w.set_histogram_column("amplitude")
    w._redraw()
    note = _density_note(w)
    assert "units column is blank" in note and "does not declare" in note, note


def test_two_tables_that_disagree_on_the_label_are_told_so(qapp, qtbot):
    w = _density_window(qtbot, {"pA table": _manual_table("pA", seed=1),
                                "nA table": _manual_table("nA", seed=2)})
    w.set_sources(list(w._sources))
    w.set_histogram_column("amplitude")
    w._redraw()
    note = _density_note(w)
    assert "more than one label here (nA, pA)" in note, note


def test_a_units_column_on_only_one_of_two_drawn_tables_is_not_spoken_for_both(qapp, qtbot):
    w = _density_window(qtbot, {"with units": _manual_table("pA", seed=1),
                                "without": _manual_table(None, seed=2)})
    w.set_sources(list(w._sources))
    w.set_histogram_column("amplitude")
    w._redraw()
    note = _density_note(w)
    assert "reads “pA”" not in note and "has no units column" in note, note


def test_declared_units_and_declared_dimensionless_columns_are_untouched(qapp, qtbot):
    df = _manual_table("pA")
    df["depth_pA"] = np.abs(np.random.default_rng(3).normal(40.0, 6.0, len(df)))
    df["frac_depth"] = np.abs(np.random.default_rng(4).normal(0.3, 0.05, len(df)))
    w = _density_window(qtbot, {"manual measurements": df})
    w.set_histogram_column("depth_pA")
    w._redraw()
    assert "a density in 1/pA" in _density_note(w)
    w.set_histogram_column("frac_depth")
    w._redraw()
    assert "a dimensionless density" in _density_note(w)


def test_the_units_reading_is_one_static_rule():
    read = PlotsWindow._units_column_reading
    assert read([]) is None
    assert read([pd.DataFrame({"x": [1.0]})]) is None
    assert read([pd.DataFrame({"units": ["pA", " pA", "pA"]})]) == ("pA",)
    assert read([pd.DataFrame({"units": ["", None, "  "]})]) == ()
    assert read([pd.DataFrame({"units": ["pA"]}), pd.DataFrame({"units": ["nA"]})]) == ("nA", "pA")
    assert read([pd.DataFrame({"units": ["pA"]}), pd.DataFrame({"x": [1.0]})]) is None


# ================================================================================================
# A1-N4 -- one disclosure record per origin
# ================================================================================================

REFUSAL = "is not a y quantity this build offers"
LOSSES = "did not take"


def _both(w) -> tuple[bool, bool]:
    d = w.details_text()
    return REFUSAL in d, LOSSES in d


def _colleague(w, path, norm: str, bins: int):
    payload = plot_config_to_json(w.config())
    payload["histogram"]["normalization"] = norm
    payload["histogram"]["bins"] = bins
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def lineage(qapp, qtbot):
    w = _window(qtbot,
                _source("doc0:0", "−700 mV", _events(seed=1), "#0072B2"),
                _source("doc1:0", "−600 mV", _events(seed=2), "#D55E00"))
    w.set_sources(["doc0:0"])
    return w


def test_an_undo_back_onto_the_first_of_two_imports_finds_that_files_sentences(lineage, tmp_path):
    """Import A, one change, import B, Undo, Undo: the window is on A's own state and
    A's two sentences -- naming A -- are back; the re-save over A is then not silent."""
    w = lineage
    a = _colleague(w, tmp_path / "colleague_A.json", "density (1/s)", 10 ** 9)
    assert w.import_config_json(a) is True
    imported_a = w.config()
    assert _both(w) == (True, True) and "colleague_A.json" in w.details_text()
    w.set_bins(31)
    b = _colleague(w, tmp_path / "colleague_B.json", "density (1/ms)", 10 ** 8)
    assert w.import_config_json(b) is True
    assert _both(w) == (True, True) and "colleague_B.json" in w.details_text()
    assert "colleague_A.json" not in w.details_text(), "B's state is B's account"

    assert w.undo_config() is True                     # back onto the bins change
    assert _both(w) == (True, True) and "colleague_A.json" in w.details_text(), (
        "the bins change was made FROM A's state, so it descends from A's install")
    assert w.undo_config() is True                     # back onto A's own state
    assert w.config() == imported_a, "premise: bit-for-bit on A's imported state"
    assert _both(w) == (True, True), w.details_text()
    assert "colleague_A.json" in w.details_text() and "colleague_B.json" not in w.details_text()
    assert "“density (1/s)” is not a y quantity" in w.details_text(), (
        "the NAME the file asked for, verbatim -- not the record that holds it")

    assert w.redo_config() is True and w.redo_config() is True
    assert _both(w) == (True, True) and "colleague_B.json" in w.details_text(), (
        "forward again onto B's state, and B's account with it")


def test_a_hand_move_of_the_combo_clears_the_on_screen_lineage_only(lineage, tmp_path):
    """The user re-decided the y quantity for the state in front of them -- B's. A's
    refusal, two undos away, is still the true account of A's state.

    A bins change sits BETWEEN the two imports on purpose. Without it B's install is
    bit-identical to A's on-screen state (both names refused, both bin counts clamped
    to the spin box's 500), and an install that moves nothing pushes no undo step --
    measured: depth stays (1, 0) while the origin goes 2 -> 3 -- so the second Undo
    steps past BOTH imports onto the pre-import state, where no sentence should stand.
    That is the history's pre-existing grain (noted in the A1 report), not the records'
    rule, and a test that asserted A's sentences there was asserting a lie.
    """
    w = lineage
    a = _colleague(w, tmp_path / "A.json", "density (1/s)", 10 ** 9)
    assert w.import_config_json(a) is True
    w.set_bins(31)                                   # a step under A's lineage
    b = _colleague(w, tmp_path / "B.json", "density (1/ms)", 10 ** 8)
    assert w.import_config_json(b) is True
    assert w._history.depth()[0] >= 3, "premise: A's import, the bins change and B's import are three steps"
    w._norm_combo.setCurrentText("density")          # not under `_updating`: a hand pick
    assert _both(w) == (False, True)
    assert w.undo_config() is True                   # the hand move, undone: B's own state
    assert _both(w) == (False, True) and "B.json" in w.details_text(), (
        "the hand move re-decided B's y quantity for good; B's losses still stand")
    assert w.undo_config() is True                   # B's import, undone: the bins change
    assert _both(w) == (True, True) and "A.json" in w.details_text()
    assert w.undo_config() is True                   # the bins change, undone: A's own state
    assert _both(w) == (True, True) and "A.json" in w.details_text()
    assert "“density (1/s)” is not a y quantity" in w.details_text()


def test_the_refusal_sentence_quotes_the_refused_name_verbatim_not_the_record(lineage):
    """The record is a dict per origin now; the sentence still quotes the NAME the file
    asked for, because a user searches their own file for it. Measured on the tree
    before this: Details read '“{2: ' density '}” is not a y quantity…' for a config
    whose normalization was ' density ' -- the dict's repr where the spelling belongs --
    and test_plots_undo_templates_json's two-doors test went red on it."""
    w = lineage
    cfg = w.config()
    w.apply_config(replace(cfg, histogram=replace(cfg.histogram, normalization=" density ")))
    line = next(ln.strip() for ln in w.details_text().splitlines() if REFUSAL in ln)
    assert line.startswith("“ density ” is not a y quantity"), line
    assert "{" not in line and "}" not in line, line


def test_reset_state_forgets_every_lineages_record(lineage, tmp_path):
    w = lineage
    assert w.import_config_json(_colleague(w, tmp_path / "A.json", "density (1/s)", 10 ** 9))
    assert w.import_config_json(_colleague(w, tmp_path / "B.json", "density (1/ms)", 10 ** 8))
    w.reset_state()
    assert w._norm_refusal._records == {} and w._config_losses._records == {}


def test_the_records_keep_one_account_per_origin():
    """The rule without a window: what is spoken is the entry for the origin on screen."""
    screen = [1]
    refusal = _NormRefusal(lambda: screen[0])
    refusal.record("density (1/s)", 1)
    screen[0] = 2
    refusal.record("density (1/ms)", 2)
    assert refusal.asked == "density (1/ms)" and refusal.origin == 2
    screen[0] = 1
    assert refusal.asked == "density (1/s)" and refusal.origin == 1
    screen[0] = 3
    assert refusal.asked == "" and refusal.origin == 0
    screen[0] = 2
    refusal.clear()                                   # re-decided for lineage 2 only
    assert refusal.asked == ""
    screen[0] = 1
    assert refusal.asked == "density (1/s)"
    refusal.forget()
    assert refusal.asked == ""

    losses = _ConfigLosses(lambda: screen[0])
    installed = pp.PlotConfig()

    def asked(bins: int) -> pp.PlotConfig:
        """What the file DECODES to -- it has to differ from what was installed, or
        there is no loss to record (a take with nothing lost is an empty account)."""
        return replace(installed, histogram=replace(installed.histogram, bins=bins))

    losses.take(("A.json", {"histogram": {"bins": 10 ** 9}}, asked(10 ** 9)), installed, 1)
    screen[0] = 2
    losses.take(("B.json", {"histogram": {"bins": 10 ** 8}}, asked(10 ** 8)), installed, 2)
    assert losses.source == "B.json" and losses.note().startswith("B.json named")
    assert losses.names == (f"histogram.bins \u201c{10 ** 8}\u201d",), losses.names
    screen[0] = 1
    assert losses.source == "A.json" and losses.note().startswith("A.json named")
    losses.clear()
    assert losses.names == () and losses.note() == ""
    screen[0] = 2
    assert losses.source == "B.json"


def test_a_record_outside_a_window_speaks_its_newest_entry():
    refusal = _NormRefusal()
    refusal.record("x", 1)
    refusal.record("y", 2)
    assert refusal.asked == "y"
    refusal.clear()
    assert refusal.asked == "", "no resolver: clear is clear"


def test_the_record_dict_is_bounded_above_what_the_history_can_reach():
    """Base + 200 past + 200 future + one pending origin is the most the stack can hold,
    so the bound can never drop an entry an undo or a redo could land on."""
    assert pp._ORIGINS_KEPT == 2 * pp._HISTORY_DEPTH + 2
    refusal = _NormRefusal()
    for origin in range(1, pp._ORIGINS_KEPT + 2):
        refusal.record(f"n{origin}", origin)
    assert len(refusal._records) == pp._ORIGINS_KEPT
    assert 1 not in refusal._records and pp._ORIGINS_KEPT + 1 in refusal._records
