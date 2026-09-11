"""Comparison generation is explicit; presentation reuses frozen geometry."""
from copy import deepcopy
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.distribution_summary import summarize_distribution
from nanoscope.plot import comparison_figures as comparison


def source(key="a", n=20, **overrides):
    values = dict(_nanoscope_row=np.arange(n, dtype=np.int64) * 2,
                  event_id=np.arange(n, dtype=np.int64) + 100,
                  dwell_s=np.linspace(.001, .02, n), depth_pA=np.linspace(1., 20., n),
                  ecd_pA_s=np.linspace(.1, 2., n), snr=np.linspace(3., 8., n))
    values.update(overrides)
    return dict(key=key, label="A full recording name " + key * 40, frame=pd.DataFrame(values))


def prepare(kind="facets", **config):
    return comparison.prepare_comparison([source(), source("b")], dict(kind=kind, **config))


def identical_tables(left, right):
    assert left.keys() == right.keys()
    for name in left:
        pd.testing.assert_frame_equal(left[name], right[name], check_exact=True)


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_deterministic_exact_source_values_and_unmodified_inputs(kind):
    sources = [source(), source("b")]
    originals = [s["frame"].copy(deep=True) for s in sources]
    config = dict(kind=kind, labels={"dwell_s": "Dwell (s)", "depth_pA": "Depth (pA)"})
    config_copy = deepcopy(config)
    snapshot, tables = comparison.prepare_comparison(sources, config)
    repeated, repeated_tables = comparison.prepare_comparison(sources, config)
    assert snapshot == repeated
    json.dumps(snapshot, allow_nan=False)
    identical_tables(tables, repeated_tables)
    assert config == config_copy
    assert snapshot["versions"]["seaborn"] != "unavailable"
    assert len(tables["observations"]) == 40
    for i, s in enumerate(sources):
        pd.testing.assert_frame_equal(s["frame"], originals[i], check_exact=True)
        subset = tables["observations"][tables["observations"]["source_key"] == s["key"]]
        np.testing.assert_array_equal(subset["source_row"], originals[i]["_nanoscope_row"])
        np.testing.assert_array_equal(subset["event_id"], originals[i]["event_id"])
        for feature in snapshot["features"]:
            np.testing.assert_array_equal(subset[feature], originals[i][feature])
    # A returned-table edit cannot alter the source event frame.
    tables["observations"].loc[0, snapshot["features"][0]] = 99.
    pd.testing.assert_frame_equal(sources[0]["frame"], originals[0], check_exact=True)


def test_facets_share_both_axes_and_have_bounded_unique_labels():
    snapshot, tables = prepare()
    assert snapshot["panels"][0]["xlim"] == snapshot["panels"][1]["xlim"]
    assert snapshot["panels"][0]["ylim"] == snapshot["panels"][1]["ylim"]
    labels = [s["plot_label"] for s in snapshot["sources"]]
    assert len(set(labels)) == 2 and max(map(len, labels)) <= 32
    assert all(s["label"] in snapshot["details"] for s in snapshot["sources"])
    assert not len(tables["histograms"]) and not len(tables["boxes"])


def test_long_recording_labels_keep_distinguishing_voltage_suffix():
    sources = [source("a"), source("b"), source("c")]
    for s, voltage in zip(sources, ("-500 mV", "-600 mV", "-700 mV")):
        s["label"] = "500 pM Apo ferritin_1 x PBS_22 nm_Process_data_" + voltage
    snapshot, tables = comparison.prepare_comparison(sources, {})
    for src, original in zip(snapshot["sources"], sources):
        assert src["plot_label"].endswith(original["label"][-13:])
        assert original["label"][:14] in src["plot_label"]
        assert len(src["plot_label"]) <= 32
        assert original["label"] in snapshot["details"]


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_render_source_names_and_colors_change_only_presentation(kind):
    snapshot, tables = prepare(kind)
    original = deepcopy(snapshot)
    frozen = {key: table.copy(deep=True) for key, table in tables.items()}
    figure = comparison.render_comparison(snapshot, tables, source_appearance={
        "a": {"label": "Apo -500 mV", "color": "#123456"},
        "b": {"label": "Apo -700 mV", "color": "#654321"}})
    FigureCanvasAgg(figure).draw()
    texts = [text.get_text() for text in figure.findobj(Text)]
    assert "Apo -500 mV" in texts and "Apo -700 mV" in texts
    assert snapshot == original
    identical_tables(tables, frozen)
    colors = [collection.get_facecolors()[0, :3] for axis in figure.axes for collection in axis.collections
              if len(collection.get_facecolors())]
    assert any(np.allclose(color, matplotlib.colors.to_rgb("#123456")) for color in colors)


def test_pairs_use_panel_specific_finite_and_log_masks_not_complete_case():
    s = source(n=5, dwell_s=[1., 2., np.nan, 4., -1.], depth_pA=[1., np.nan, 3., 4., 5.], snr=[np.nan, 2., 3., 4., 5.])
    snapshot, tables = comparison.prepare_comparison([s], dict(kind="pairs", features=["dwell_s", "depth_pA", "snr"], bins=4, log_x=True))
    panels = {p["id"]: p for p in snapshot["panels"]}
    assert panels["pair:0:0"]["counts"][0] == dict(total=5, included=3, excluded_nonfinite=1,
        excluded_log=1, nonfinite_x=1, nonfinite_y=0, source_key="a")
    assert panels["pair:1:0"]["counts"][0]["included"] == 2
    assert panels["pair:1:2"]["counts"][0]["included"] == 3
    assert panels["pair:1:1"]["counts"][0]["included"] == 4
    assert len(tables["observations"]) == 5
    assert "no global complete-case" in snapshot["details"]
    assert "1 non-positive" in snapshot["details"]


def test_common_histogram_bins_are_explicit_counts_with_all_observations():
    snapshot, tables = prepare("pairs", features=["dwell_s", "depth_pA"], bins=7, log_x=True)
    for panel in [p for p in snapshot["panels"] if p["kind"] == "hist"]:
        previous = None
        for key in ("a", "b"):
            bars = tables["histograms"]
            bars = bars[(bars.panel_id == panel["id"]) & (bars.source_key == key)]
            edges = np.r_[bars.left, bars.right.iloc[-1]]
            if previous is not None:
                np.testing.assert_array_equal(edges, previous)
            previous = edges
            values = tables["observations"].loc[tables["observations"].source_key == key, panel["x"]]
            expected, _ = np.histogram(values, bins=edges)
            np.testing.assert_array_equal(bars["count"], expected)
            assert bars["count"].sum() == 20
            assert panel["yscale"] == "linear"
            assert panel["y_label"] == "Event count"


def test_missing_feature_is_disclosed_without_losing_other_panels():
    sources = [source(), source("b")]
    sources[1]["frame"] = sources[1]["frame"].drop(columns="snr")
    snapshot, tables = comparison.prepare_comparison(sources, dict(kind="pairs", features=["depth_pA", "snr"]))
    assert snapshot["sources"][1]["missing_columns"] == ["snr"]
    assert len(tables["observations"]) == 40
    assert snapshot["panels"][0]["counts"][1]["included"] == 20
    assert snapshot["panels"][1]["counts"][1]["excluded_nonfinite"] == 20


def test_all_missing_diagonal_uses_a_readable_zero_count_axis():
    s = source()
    s["frame"] = s["frame"].drop(columns="snr")
    snapshot, tables = comparison.prepare_comparison([s], dict(kind="pairs", features=["depth_pA", "snr"]))
    empty = next(p for p in snapshot["panels"] if p["id"] == "pair:1:1")
    assert empty["ylim"] == [0., 1.]
    assert tables["histograms"].loc[tables["histograms"].panel_id == empty["id"], "count"].sum() == 0


@pytest.mark.parametrize("values", [[0., 1., 2., 40.], [0., 100., 100., 100.], [1.], [3., 3., 3.]])
def test_boxes_use_existing_summary_exactly_and_retain_outliers(values):
    s = source(n=len(values), depth_pA=values)
    snapshot, tables = comparison.prepare_comparison([s], dict(kind="events", show_box=True))
    expected = summarize_distribution(np.array(values), kind="box")
    box = tables["boxes"].iloc[0]
    for name in comparison.BOX_COLUMNS[3:]:
        assert box[name] == expected.stats[name]
    np.testing.assert_array_equal(tables["plotted"].y, values)
    assert len(tables["plotted"]) == len(values)


def test_strip_does_not_touch_global_rng_and_offsets_are_deterministic():
    np.random.seed(57)
    before = np.random.get_state()
    snapshot, tables = prepare("events")
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(tables["plotted"].x, np.r_[rng.uniform(-.2, .2, 20), 1 + rng.uniform(-.2, .2, 20)])


@pytest.mark.parametrize("log_y", [False, True])
def test_swarm_uses_seaborn_once_and_preserves_exact_measured_y(log_y, monkeypatch):
    import seaborn as sns
    calls = []
    original = sns.swarmplot
    def track(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(sns, "swarmplot", track)
    snapshot, tables = prepare("events", points="swarm", log_y=log_y)
    repeated, repeated_tables = prepare("events", points="swarm", log_y=log_y)
    assert calls == [1, 1]
    identical_tables(tables, repeated_tables)
    np.testing.assert_array_equal(tables["plotted"].y, tables["observations"].depth_pA)
    assert "frozen once" in snapshot["details"]


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_render_never_recomputes_and_appearance_never_changes_geometry(kind, monkeypatch):
    snapshot, tables = prepare(kind, show_box=True)
    saved_snapshot, saved_tables = deepcopy(snapshot), {k: v.copy(deep=True) for k, v in tables.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering attempted to recalculate a plot")
    monkeypatch.setattr(comparison, "prepare_comparison", forbidden)
    monkeypatch.setattr(comparison, "summarize_distribution", forbidden)
    monkeypatch.setattr(comparison, "_swarm_positions", forbidden)
    monkeypatch.setattr(np, "histogram", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    import seaborn as sns
    for name in ("swarmplot", "stripplot", "histplot", "kdeplot", "regplot", "lineplot", "boxplot"):
        monkeypatch.setattr(sns, name, forbidden)
    figures = []
    for style in ({}, {"title": "Custom title", "grid": True, "background": "dark", "marker_size": 8., "line_width": 2.,
                       "font_family": "DejaVu Sans", "legend_visible": False}, {"title": "", "grid": False}):
        figure = comparison.render_comparison(snapshot, tables, style)
        FigureCanvasAgg(figure).draw()
        figures.append(figure)
    assert snapshot == saved_snapshot
    identical_tables(tables, saved_tables)
    for first, changed in zip(figures[0].axes, figures[1].axes):
        assert first.get_xlim() == changed.get_xlim()
        assert first.get_ylim() == changed.get_ylim()
        for a, b in zip(first.collections, changed.collections):
            np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
    assert not any(line.get_visible() for ax in figures[2].axes for line in ax.get_xgridlines() + ax.get_ygridlines())
    assert figures[2]._suptitle is None


def test_warning_prose_never_becomes_plot_text_and_pair_units_survive():
    snapshot, tables = prepare("pairs", features=["depth_pA", "snr"], labels={"depth_pA": "Depth (pA)", "snr": "SNR"})
    snapshot["warnings"].append("SENTINEL CALIBRATION WARNING")
    snapshot["details"] += "\nSENTINEL CALIBRATION WARNING"
    figure = comparison.render_comparison(snapshot, tables, {"x_label": "Wrong unit", "y_label": "Wrong unit"})
    FigureCanvasAgg(figure).draw()
    texts = "\n".join(artist.get_text() for artist in figure.findobj(Text))
    assert "SENTINEL" not in texts and "Wrong unit" not in texts
    assert "Depth (pA)" in texts and "SNR" in texts


@pytest.mark.parametrize("mutation", ["infinite", "identity", "x_value", "duplicate", "axis", "source", "counts", "huge", "missing_table"])
def test_restore_rejects_malformed_geometry(mutation):
    snapshot, tables = prepare()
    if mutation == "infinite": tables["plotted"].loc[0, "x"] = np.inf
    elif mutation == "identity": tables["plotted"].loc[0, "observation_index"] = 999
    elif mutation == "x_value": tables["plotted"].loc[0, "x"] += .00001
    elif mutation == "duplicate": tables["plotted"].iloc[1] = tables["plotted"].iloc[0]
    elif mutation == "axis": snapshot["panels"][0]["xlim"] = [2., 1.]
    elif mutation == "source": tables["plotted"].loc[0, "source_key"] = "nonexistent"
    elif mutation == "counts": snapshot["panels"][0]["counts"][0]["included"] -= 1
    elif mutation == "huge": snapshot["layout"]["width"] = 1e9
    elif mutation == "missing_table": tables.pop("observations")
    with pytest.raises(ValueError):
        comparison.render_comparison(snapshot, tables)


@pytest.mark.parametrize("mutation", ["count", "edge", "common_bins", "bin_index", "extra"])
def test_restore_rejects_malformed_histograms(mutation):
    snapshot, tables = prepare("pairs", features=["dwell_s", "depth_pA"], bins=3)
    bars = tables["histograms"]
    if mutation == "count": bars.loc[0, "count"] = -1
    elif mutation == "edge": bars.loc[0, "right"] = bars.loc[0, "left"]
    elif mutation == "common_bins": bars.loc[5, "right"] += .1
    elif mutation == "bin_index": bars.loc[0, "bin_index"] = 2
    elif mutation == "extra": tables["histograms"] = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        comparison.validate_comparison(snapshot, tables)


@pytest.mark.parametrize("config", [dict(kind="bad"), dict(kind="pairs", features=["snr"]),
    dict(kind="pairs", features=["snr", "snr"]), dict(kind="events", log_x=True), dict(bins=0), dict(bins=True), dict(log_y="yes")])
def test_invalid_settings_are_actionable(config):
    with pytest.raises(ValueError):
        comparison.prepare_comparison([source()], config)


def test_explicit_budgets_and_no_silent_sampling():
    with pytest.raises(ValueError, match="one to 8"):
        comparison.prepare_comparison([source(str(i)) for i in range(9)], {})
    with pytest.raises(ValueError, match="no automatic sampling"):
        comparison.prepare_comparison([source(n=comparison.MAX_PAIR_OBSERVATIONS + 1)], dict(kind="pairs"))
    with pytest.raises(ValueError, match="Choose Strip"):
        comparison.prepare_comparison([source(n=comparison.MAX_SWARM_PER_SOURCE + 1)], dict(kind="events", points="swarm"))
    with pytest.raises(ValueError, match="too crowded"):
        comparison.prepare_comparison([source(n=100, depth_pA=np.ones(100))], dict(kind="events", points="swarm"))


def test_unsupported_inputs_rejected_and_nonfinite_empty_is_actionable():
    for bad in (["1", "2"], [True, False], [1 + 2j, 2 + 3j], np.array([2**53 + 1, 2**53 + 2], dtype=np.int64)):
        with pytest.raises(ValueError):
            comparison.prepare_comparison([source(n=2, depth_pA=bad)], dict(kind="events"))
    with pytest.raises(ValueError, match="No observations"):
        comparison.prepare_comparison([source(n=2, depth_pA=[np.nan, np.inf])], dict(kind="events"))
    with pytest.raises(ValueError, match="original event-row"):
        comparison.prepare_comparison([source(n=2, _nanoscope_row=[0, 0])], {})


def test_large_event_ids_are_not_rounded_when_other_source_has_no_ids():
    first, second = source(n=2, event_id=np.array([2**60 + 1, 2**60 + 2], dtype=np.int64)), source("b", n=2)
    second["frame"] = second["frame"].drop(columns="event_id")
    snapshot, tables = comparison.prepare_comparison([first, second], {})
    assert tables["observations"].event_id.iloc[0] == 2**60 + 1
    assert tables["observations"].event_id.iloc[1] == 2**60 + 2


@pytest.mark.parametrize("kind", ["facets", "events"])
@pytest.mark.parametrize("features", [[], ["snr"], ["snr", "snr"], [f"unused{i}" for i in range(7)]])
def test_hidden_incomplete_pair_draft_does_not_block_other_figures(kind, features):
    snapshot, tables = prepare(kind, features=features)
    assert snapshot["config"]["features"] == list(dict.fromkeys(features))
    assert comparison.validate_comparison(snapshot, tables)


def test_empty_pair_panel_does_not_widen_shared_feature_axes():
    s = source(n=4, dwell_s=[.001, .002, np.nan, np.nan], depth_pA=[np.nan, np.nan, 2., 3.])
    snapshot, tables = comparison.prepare_comparison([s], dict(kind="pairs", features=["dwell_s", "depth_pA"]))
    panels = {p["id"]: p for p in snapshot["panels"]}
    assert panels["pair:1:0"]["xlim"] == panels["pair:0:0"]["xlim"]
    assert panels["pair:1:0"]["xlim"][1] < .01
    assert panels["pair:0:1"]["ylim"][1] < .01


@pytest.mark.parametrize("mutation", ["drop_box", "box_axis", "panel_kind", "scale", "layout", "false_exclusion"])
def test_restore_rejects_inconsistent_settings_and_missing_summaries(mutation):
    snapshot, tables = prepare("events", show_box=True)
    if mutation == "drop_box": tables["boxes"] = tables["boxes"].iloc[:0]
    elif mutation == "box_axis": tables["boxes"].loc[0, "upper_whisker"] = 1e5
    elif mutation == "panel_kind": snapshot["panels"][0]["kind"] = "scatter"
    elif mutation == "scale": snapshot["panels"][0]["yscale"] = "log"
    elif mutation == "layout": snapshot["layout"]["columns"] = 2
    elif mutation == "false_exclusion": snapshot["panels"][0]["counts"][0]["nonfinite_y"] = 1
    with pytest.raises(ValueError):
        comparison.validate_comparison(snapshot, tables)


def test_persistent_source_colors_survive_selection_reorder_and_restore():
    from matplotlib.colors import to_rgba
    sources = [source(), source("b")]
    sources[0]["color"], sources[1]["color"] = "#b53697", "#28b681"
    first, first_tables = comparison.prepare_comparison(sources, dict(kind="events"))
    second, second_tables = comparison.prepare_comparison(list(reversed(sources)), dict(kind="events"))
    assert {s["key"]: s["color"] for s in first["sources"]} == {s["key"]: s["color"] for s in second["sources"]}
    restored = json.loads(json.dumps(second))
    figure = comparison.render_comparison(restored, second_tables, {"data_color": "#ffffff"})
    for collection, src in zip(figure.axes[0].collections, restored["sources"]):
        np.testing.assert_allclose(collection.get_facecolors()[0], to_rgba(src["color"], alpha=.75))


def test_bad_source_colors_are_actionable_and_bad_saved_colors_rejected():
    sources = [source()]
    sources[0]["color"] = "not-a-color"
    with pytest.raises(ValueError, match="colors"):
        comparison.prepare_comparison(sources, {})
    snapshot, tables = prepare()
    snapshot["sources"][0]["color"] = "invalid"
    with pytest.raises(ValueError, match="colors"):
        comparison.validate_comparison(snapshot, tables)
