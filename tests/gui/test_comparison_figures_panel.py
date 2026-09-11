"""Comparison UI separates explicit generation from durable presentation edits."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.core.sources import PlotSource
from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.plot_appearance import PlotAppearance


def source(key="first", label="Recording A"):
    frame = pd.DataFrame({"event_id": [5, 5, 9, 11, 14, 17],
        "dwell_s": [.001, .002, .003, .004, .005, .006],
        "depth_pA": [10., 20., 30., 40., 50., 60.],
        "amplitude_pA": [12., 25., 35., 42., 55., 64.],
        "ecd_pA_s": [.01, .04, .09, .16, .25, .36], "snr": [5., 8., 10., 12., 15., 16.],
        "accepted": [True, True, False, True, True, True]}, index=[2, 2, 4, 9, 10, 10])
    return PlotSource(key, label, frame, kind="manual", doc_key=key, current=True)


@pytest.fixture
def page(qtbot):
    page = ComparisonFiguresWindow()
    qtbot.addWidget(page)
    page.resize(1100, 800)
    page.show()
    return page


def assert_tables_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        pd.testing.assert_frame_equal(left[key], right[key], check_exact=True)


def forbid_prepare(monkeypatch):
    from nanoscope.plot import comparison_figures as renderer
    monkeypatch.setattr(renderer, "prepare_comparison", lambda *args, **kwargs: pytest.fail("Unexpected comparison preparation"))


def test_open_source_navigation_and_drafts_never_generate(page, monkeypatch):
    forbid_prepare(monkeypatch)
    page.set_sources([source()])
    page.set_config({"kind": "pairs"})
    for name in ("Analysis", "Appearance", "Export", "Data"):
        page._workflow.set_section(name)
    page.set_appearance_state({"title": "My comparison", "background": "light"})
    assert not page.has_data() and page._plot_data().empty


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_generate_retains_sources_duplicate_ids_and_accepted_positions(page, kind):
    sources = [source(), source("second", "Recording B")]
    originals = [s.df.copy(deep=True) for s in sources]
    page.set_sources(sources, selected_keys=[s.key for s in sources])
    page.set_config({"kind": kind})
    assert page.analyze() and page.has_data() and not page._stale
    for original, offered in zip(originals, sources):
        pd.testing.assert_frame_equal(offered.df, original, check_exact=True)
    observed = page._tables["observations"]
    assert len(observed) == 10
    assert set(observed["source_key"]) == {"first", "second"}
    assert sorted(observed.loc[observed.source_key == "first", "source_row"]) == [0, 1, 3, 4, 5]
    assert len(set(observed.loc[observed.source_key == "first", "observation_index"])) == 5
    assert not page._plot_data().empty
    assert set(page._plot_data()["source_label"]) == {"Recording A", "Recording B"}


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_appearance_and_drafts_preserve_values_geometry_and_zoom(page, monkeypatch, qtbot, kind):
    page.set_sources([source()])
    page.set_config({"kind": kind})
    page.analyze()
    qtbot.wait(10)
    page._view.zoom(1.4)
    scale = page._view.transform().m11()
    before, tables = page.snapshot_state()
    data = page._plot_data()
    forbid_prepare(monkeypatch)
    page._bins.setValue(51)
    assert page._stale
    page.set_appearance_state(PlotAppearance(title="A clear figure", background="light", marker_size=5,
                                            grid=True, font_family="Arial", tick_size=9))
    assert page._view.transform().m11() == scale
    assert page._snapshot == before["snapshot"]
    assert page.figure_applied_config() == before["applied"]
    assert_tables_equal(page._tables, tables)
    pd.testing.assert_frame_equal(page._plot_data(), data, check_exact=True)


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_restore_with_no_sources_never_prepares(page, qtbot, monkeypatch, kind):
    page.set_sources([source()])
    page.set_config({"kind": kind})
    page.analyze()
    page._bins.setValue(51)
    payload, tables = page.snapshot_state()
    other = ComparisonFiguresWindow()
    qtbot.addWidget(other)
    forbid_prepare(monkeypatch)
    other.restore_state(payload, tables)
    assert other.has_data() and other._stale
    assert other.config() == payload["draft"]
    assert other.figure_applied_config() == payload["applied"]
    assert_tables_equal(other._tables, tables)
    pd.testing.assert_frame_equal(other._plot_data(), page._plot_data(), check_exact=True)
    assert other._source_list.item(0).text().startswith("Unavailable source")


@pytest.mark.parametrize("corruption", ["schema", "draft", "stale", "view", "tables", "sources", "warnings", "applied"])
def test_bad_restore_preserves_previous_figure_transactionally(page, corruption):
    page.set_sources([source()])
    page.analyze()
    before, before_tables = page.snapshot_state()
    payload, tables = deepcopy(before), deepcopy(before_tables)
    if corruption == "schema":
        payload["schema_version"] = 2
    elif corruption == "draft":
        payload["draft"]["kind"] = "unknown"
    elif corruption == "stale":
        payload["stale"] = "false"
    elif corruption == "view":
        payload["canvas_view"]["scale"] = float("nan")
    elif corruption == "tables":
        tables["plotted"].loc[0, "x"] = np.inf
    elif corruption == "sources":
        payload["applied"]["source_keys"] = ["wrong"]
    elif corruption == "applied":
        payload["applied"]["kind"] = "events"
    else:
        payload["snapshot"]["warnings"] = [None]
    with pytest.raises((ValueError, TypeError)):
        page.restore_state(payload, tables)
    after, after_tables = page.snapshot_state()
    assert after == before
    assert_tables_equal(after_tables, before_tables)


def test_source_mutation_marks_snapshot_stale_without_changing_it(page, monkeypatch):
    offered = source()
    page.set_sources([offered])
    page.analyze()
    original = page._plot_data()
    forbid_prepare(monkeypatch)
    offered.df.iloc[0, offered.df.columns.get_loc("depth_pA")] = 99.
    page.set_sources([offered])
    assert page._stale
    pd.testing.assert_frame_equal(page._plot_data(), original, check_exact=True)


def test_failed_generate_keeps_prior_picture_and_tables(page):
    page.set_sources([source()])
    page.analyze()
    original = page._plot_data()
    page.set_config({"kind": "pairs", "features": ["dwell_s"]})
    with pytest.raises(ValueError):
        page.analyze()
    pd.testing.assert_frame_equal(page._plot_data(), original, check_exact=True)


@pytest.mark.parametrize("kind", ["facets", "events"])
def test_incomplete_pair_draft_does_not_block_other_kinds(page, kind):
    page.set_sources([source()])
    page.set_config({"kind": "pairs", "features": []})
    page.set_config({"kind": kind})
    assert page.analyze()
    assert page.figure_applied_config()["features"] == []


def test_source_color_change_keeps_frozen_snapshot_and_saved_on_generate(page):
    from dataclasses import replace

    offered = source()
    page.set_sources([offered])
    page.analyze()
    previous = page._snapshot["sources"][0]["color"]
    changed = replace(offered, colour="#ff1234")
    page.set_sources([changed])
    assert not page._stale  # presentation-only defaults do not invalidate measurements
    assert page._snapshot["sources"][0]["color"] == previous
    page.analyze()
    assert not page._stale
    assert page._snapshot["sources"][0]["color"] == "#ff1234"


@pytest.mark.parametrize("kind", ["facets", "pairs", "events"])
def test_source_appearance_replays_frozen_geometry_and_roundtrips(page, qtbot, monkeypatch, kind):
    page.set_sources([source()])
    page.set_config({"kind": kind})
    page.analyze()
    page._view.zoom(1.3)
    before, tables = page.snapshot_state()
    data = page._plot_data()
    scale = page._view.transform().m11()
    forbid_prepare(monkeypatch)
    page.set_source_appearance_state({"first": {"label": "Apo -700 mV", "color": "#123456"}})
    assert page._snapshot == before["snapshot"]
    assert page.figure_applied_config() == before["applied"]
    assert not page._stale
    assert page._view.transform().m11() == scale
    assert_tables_equal(page._tables, tables)
    pd.testing.assert_frame_equal(page._plot_data(), data, check_exact=True)
    payload, saved_tables = page.snapshot_state()
    other = ComparisonFiguresWindow()
    qtbot.addWidget(other)
    other.restore_state(payload, saved_tables)
    assert other.source_appearance_state() == page.source_appearance_state()
    assert other._source_appearance_editor.source.currentData() == "first"
    assert "Apo -700 mV" in other.details_text()
    assert_tables_equal(other._tables, tables)


def test_bad_source_appearance_restore_is_transactional(page):
    page.set_sources([source()])
    page.analyze()
    before, tables = page.snapshot_state()
    payload = deepcopy(before)
    payload["source_appearance"] = {"first": {"color": "broken"}}
    with pytest.raises(ValueError):
        page.restore_state(payload, tables)
    assert page.snapshot_state()[0] == before


def test_csv_export_uses_committed_geometry_not_new_drafts(page, monkeypatch, tmp_path):
    page.set_sources([source()])
    page.set_config({"kind": "events"})
    page.analyze()
    original = page._plot_data()
    page._view_combo.setCurrentIndex(page._view_combo.findData("pairs"))
    forbid_prepare(monkeypatch)
    target = tmp_path / "comparison.csv"
    assert page.save_csv(target) == target
    actual = pd.read_csv(target)
    assert len(actual) == len(original)
    assert set(actual.primitive) == set(original.primitive)
    assert set(actual.source_label) == {"Recording A"}
    np.testing.assert_allclose(actual["x"], original["x"], equal_nan=True)


def test_long_source_label_and_warning_stay_in_details(page, qtbot):
    label = "A long source name with acquisition context " * 20
    page.set_sources([source(label=label)])
    page.analyze()
    qtbot.wait(10)
    assert page.size() == QtCore.QSize(1100, 800)
    assert len(page._status.text()) <= 220
    assert label in page.details_text()
    assert page._source_list.item(0).toolTip() == label
    assert page._view.height() >= 180
