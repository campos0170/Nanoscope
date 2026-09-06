"""Input boundaries and transactional, computation-free summary restoration."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.sources import PlotSource
from nanoscope.gui import distribution_summary_panel as M


@pytest.fixture
def page(qtbot):
    page = M.DistributionSummaryWindow()
    qtbot.addWidget(page)
    page.set_sources([_source([1., 2., 3., 4.])])
    page.analyze()
    return page


def _source(values, label="Measured events"):
    return PlotSource("measurements", label, pd.DataFrame({"dwell_s": values}),
                      kind="manual", doc_key="measurements", current=True)


@pytest.mark.parametrize("values", [
    pd.Series([1, 2, 3], dtype="int64"), pd.Series([1, pd.NA, 3], dtype="Int64"),
    pd.Series([1., "invalid", None, pd.NA, "3"], dtype=object),
])
def test_real_integer_nullable_and_text_observations_are_exact_and_disclosed(page, values):
    source = _source(values)
    original = source.df.copy(deep=True)
    page.set_sources([source])
    page.analyze()
    expected = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = expected[np.isfinite(expected)]
    np.testing.assert_array_equal(page._tables["samples_0"]["value"], finite)
    assert page._snapshot["series"][0]["n_excluded"] == len(values) - len(finite)
    pd.testing.assert_frame_equal(source.df, original, check_exact=True)


@pytest.mark.parametrize("values", [
    pd.Series([2**53+1, pd.NA], dtype="Int64"),
    pd.Series([2**53+1, None], dtype=object),
    ["9007199254740993", None], ["9.007199254740993e15", None],
    [True, False], [1 + 2j, 3 + 0j],
    pd.to_timedelta([1, 2], unit="s"), pd.to_datetime(["2026-01-01", "2026-01-02"]),
])
def test_unsafe_conversion_refused_without_replacing_last_result(page, values):
    before = page._plot_data()
    page.set_sources([_source(values)])
    with pytest.raises(ValueError):
        page.analyze()
    assert page._stale
    pd.testing.assert_frame_equal(page._plot_data(), before, check_exact=True)


@pytest.mark.parametrize("corruption", [
    "draft", "series", "counts", "stats", "primitive", "color", "coordinates",
    "ranges", "recipes", "stale", "source_keys", "description",
])
def test_invalid_restore_is_transactional(page, corruption):
    before, before_tables = page.snapshot_state()
    payload, tables = deepcopy(before), deepcopy(before_tables)
    if corruption == "draft":
        payload["draft"]["kind"] = "unknown"
    elif corruption == "series":
        del payload["snapshot"]["series"][0]["label"]
    elif corruption == "counts":
        payload["snapshot"]["series"][0]["n"] = -1
    elif corruption == "stats":
        payload["snapshot"]["series"][0]["stats"] = {"bad": "not numeric"}
    elif corruption == "primitive":
        del payload["snapshot"]["primitives"][0]["legend"]
    elif corruption == "color":
        payload["snapshot"]["primitives"][0]["color"] = "invalid-color"
    elif corruption == "coordinates":
        tables["draw_0"].loc[0, "x"] = np.inf
    elif corruption == "ranges":
        payload["view_range"] = [[0, "invalid"], [0, 1]]
    elif corruption == "recipes":
        payload["source_recipes"] = "invalid"
    elif corruption == "stale":
        payload["stale"] = "false"
    elif corruption == "source_keys":
        payload["applied"]["source_keys"] = ["different-source"]
    else:
        payload["snapshot"]["warnings"] = [None]
    with pytest.raises(ValueError):
        page.restore_state(payload, tables)
    after, after_tables = page.snapshot_state()
    assert after == before
    for key in before_tables:
        pd.testing.assert_frame_equal(after_tables[key], before_tables[key], check_exact=True)


def test_restore_preserves_stale_draft_and_view_without_statistics(page, qtbot, monkeypatch):
    page._view_combo.setCurrentIndex(page._view_combo.findData("violin"))
    page._plot.setRange(xRange=(1, 2), yRange=(.1, .8), padding=0)
    payload, tables = page.snapshot_state()
    other = M.DistributionSummaryWindow()
    qtbot.addWidget(other)
    monkeypatch.setattr(M, "summarize_distribution", lambda *a, **k: pytest.fail("Restore recomputed"))
    other.restore_state(payload, tables)
    assert other._stale and other.config()["kind"] == "violin"
    assert other.figure_applied_config()["kind"] == "ecdf"
    assert other._plot.viewRange() == page._plot.viewRange()
    pd.testing.assert_frame_equal(other._plot_data(), page._plot_data(), check_exact=True)


def test_csv_chooser_uses_keyword_contract_and_frozen_coordinates(page, tmp_path, monkeypatch):
    target = tmp_path / "summary.csv"
    calls = []
    def choose(parent, *, title, suggestion, filters):
        calls.append((title, suggestion, filters))
        return target
    monkeypatch.setattr(M, "ask_export_path", choose)
    monkeypatch.setattr(M, "summarize_distribution", lambda *a, **k: pytest.fail("Export recomputed"))
    assert page.save_csv() == target and calls
    exported = pd.read_csv(target)
    np.testing.assert_array_equal(exported[["x", "y"]], page._plot_data()[["x", "y"]])


@pytest.mark.parametrize("kind", ["ecdf", "survival", "box", "violin"])
def test_long_source_names_fit_common_screen_with_full_details_available(page, qapp, tmp_path, kind):
    label = "Long recording provenance description · " * 30
    page.set_sources([_source([1., 2., 3., 4.], label)])
    page._view_combo.setCurrentIndex(page._view_combo.findData(kind))
    page.analyze()
    page.resize(1000, 740)
    page.show()
    qapp.processEvents()
    assert page.width() == 1000 and page.height() == 740
    assert len(page._status.text()) <= 220
    assert label in page._status.toolTip()
    assert page._source_list.item(0).toolTip() == label
    assert len(page._snapshot["primitives"][0]["legend"]) < 80
    assert page._snapshot["primitives"][0]["source_label"] == label
    assert page._plot_widget.height() >= 150
    if kind in ("box", "violin"):
        assert not page._plot.getAxis("bottom").autoSIPrefix
        assert page._plot.getAxis("bottom").autoSIPrefixScale == 1
        assert page._plot.viewRange()[0] == [-.6, .6]
    assert page.grab().save(str(tmp_path / "distribution-summary.png"))
