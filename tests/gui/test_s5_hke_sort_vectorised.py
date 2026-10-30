"""HKE-01 — one header click on the 100,000-event table must not freeze the GUI.

The FE-01 fix made header sorts numerically CORRECT, but the proxy doing the ordering
(``QSortFilterProxyModel``) called the source model's Python ``data()`` twice per
comparison — O(n log n) Python round-trips on the GUI thread. Measured through the real
app (MainWindow -> open_paths -> trigger_detection -> sortByColumn, the identical path a
physical header click takes): 48.7 s for the FIRST click at 100k events, 7.8 s for every
re-sort, with no busy indicator anywhere. The replacement (``_VectorSortProxy``) asks the
frame for the whole column once and lets numpy order it — O(n log n) vectorised, ~20-60 ms
at the same scale.

These tests pin BOTH halves: the sort still answers with the numeric truth (and the old
proxy's tie / NA / dynamic-resort semantics), and it answers fast at the 100k scale the
defect was measured at. Timing bounds are deliberately GENEROUS (60-100x the measured
cost, best-of-N) so a loaded CI machine cannot fail them while the O(n log n)-Python
regression — three orders of magnitude slower — cannot pass them.
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
from PySide6.QtCore import Qt

from nanoscope.gui.panels import EventTable

N = 100_000


# ================================================================================================
# builders
# ================================================================================================
def _big_frame(n: int = N, seed: int = 11) -> pd.DataFrame:
    """A 100k-row curatable frame: shuffled numeric columns (so a sort has real work),
    a flag column with a rejecting token here and there, and a few NA depths."""
    rng = np.random.default_rng(seed)
    depth = rng.uniform(5.0, 200.0, n)
    depth[rng.choice(n, size=25, replace=False)] = np.nan
    flag = np.array([""] * n, dtype=object)
    flag[rng.choice(n, size=500, replace=False)] = "clipped_by_boundary"
    return pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "t_start_s": rng.uniform(0.0, 150.0, n),
        "dwell_s": rng.uniform(1e-4, 1e-2, n),
        "depth_pA": depth,
        "accepted": rng.random(n) > 0.3,
        "flag": flag,
    })


def _col(view, name: str) -> int:
    # By the model's column NAME, not the header caption, whose first word stopped being
    # the name when b21e8ae (2026-09-06) captioned depth "median depth (depth_pA)". The
    # proxy never permutes columns, so the source index is the view's index.
    cols = view.model().sourceModel()._columns
    assert name in cols, f"no {name} column"
    return cols.index(name)


def _shown(table: EventTable, colname: str) -> np.ndarray:
    """The column's raw values in view order, via mapToSource (never parsed off the
    display string) — vectorised so reading 100k rows does not itself take minutes."""
    proxy = table._proxy
    order = np.array([proxy.mapToSource(proxy.index(r, 0)).row()
                      for r in range(proxy.rowCount())], dtype=np.intp)
    return table._model.df[colname].to_numpy()[order]


@pytest.fixture
def big_table(qapp, qtbot) -> EventTable:
    t = EventTable()
    qtbot.addWidget(t)
    t.set_events(_big_frame())
    return t


# ================================================================================================
# the stall, measured at the dossier's own scale
# ================================================================================================
def test_first_sort_and_resort_at_100k_stay_interactive(big_table):
    """The defect's two measured numbers, bounded: first click 48.7 s -> must land in
    2 s; re-sort 7.8 s -> best of three must land in 2 s. Both bounds are ~60-100x the
    ~20-60 ms the vectorised sort actually takes, so only the per-comparison-Python
    regression (three orders of magnitude over budget) can fail them."""
    view = big_table.view
    depth_col = _col(view, "depth_pA")
    dwell_col = _col(view, "dwell_s")

    t0 = time.perf_counter()
    view.sortByColumn(depth_col, Qt.DescendingOrder)   # <- the header click
    first = time.perf_counter() - t0
    assert first < 2.0, f"first sort took {first:.2f}s at {N:,} rows (was 48.7 s)"

    resorts = []
    for col, order in ((dwell_col, Qt.AscendingOrder),
                       (depth_col, Qt.AscendingOrder),
                       (dwell_col, Qt.DescendingOrder)):
        t0 = time.perf_counter()
        view.sortByColumn(col, order)
        resorts.append(time.perf_counter() - t0)
    assert min(resorts) < 2.0, f"re-sorts took {resorts} (was 7.8 s each)"


def test_the_100k_sort_is_the_numeric_truth_in_both_directions(big_table):
    """Speed bought with a wrong order would be the FE-01 defect back again. The full
    100k view order must equal numpy's own ordering of the column, NAs excluded and
    grouped at one end (below)."""
    view = big_table.view
    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    shown = _shown(big_table, "depth_pA")
    finite = shown[~np.isnan(shown)]
    truth = np.sort(big_table._model.df["depth_pA"].dropna().to_numpy())[::-1]
    np.testing.assert_array_equal(finite, truth)

    view.sortByColumn(_col(view, "depth_pA"), Qt.AscendingOrder)
    shown = _shown(big_table, "depth_pA")
    finite = shown[~np.isnan(shown)]
    np.testing.assert_array_equal(finite, truth[::-1])


def test_na_rows_sort_together_last_ascending_first_descending(big_table):
    """NA rows travel together at one end: LAST on an ascending click, FIRST on a
    descending one — the ends the pristine ``QSortFilterProxyModel`` MEASURABLY used
    when driven through a shown ``QTableView`` (probed differentially against the
    pre-fix widget, not read off Qt's docs). NaN is this app's 'not measured' marker,
    so ascending leading with the NaN pile would bury every measured row."""
    view = big_table.view
    n_na = int(big_table._model.df["depth_pA"].isna().sum())
    assert n_na > 0, "the fixture must carry NA depths for this test to mean anything"

    view.sortByColumn(_col(view, "depth_pA"), Qt.AscendingOrder)
    assert np.isnan(_shown(big_table, "depth_pA")[-n_na:]).all(), "NA must trail ascending"

    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    assert np.isnan(_shown(big_table, "depth_pA")[:n_na]).all(), "NA must lead descending"


def test_ties_keep_the_previous_sorts_order_like_sequential_header_clicks(qapp, qtbot):
    """The old proxy sorted its EXISTING mapping, so clicking dwell after depth left
    equal dwells in depth order — the spreadsheet multi-level sort users lean on. The
    vectorised proxy must keep that: ties chain through the standing view order."""
    t = EventTable()
    qtbot.addWidget(t)
    t.set_events(pd.DataFrame({
        "event_id": [0, 1, 2, 3],
        "dwell_s": [1e-3, 1e-3, 1e-3, 1e-3],       # all tied
        "depth_pA": [5.0, 20.0, 10.0, 15.0],
        "accepted": [True] * 4,
        "flag": [""] * 4,
    }))
    view = t.view
    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    view.sortByColumn(_col(view, "dwell_s"), Qt.AscendingOrder)
    assert _shown(t, "depth_pA").tolist() == [20.0, 15.0, 10.0, 5.0], (
        "equal dwells must stay in the depth order the previous click established")


def _small_tied_table(qtbot) -> EventTable:
    """Nine rows: depth in three tie groups (2.0 / 1.0 / NaN), ids in source order —
    small enough that every id sequence below can be read against the frame by eye."""
    t = EventTable()
    qtbot.addWidget(t)
    t.set_events(pd.DataFrame({
        "event_id": list(range(9)),
        "t_start_s": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "dwell_s": [1e-4] * 9,
        "depth_pA": [2.0, 1.0, np.nan, 2.0, 1.0, np.nan, 2.0, 1.0, np.nan],
        "accepted": [True] * 9,
        "flag": [""] * 9,
    }))
    return t


def _view_ids(t: EventTable) -> list[int]:
    proxy = t.view.model()
    df = t._model.df
    return [int(df.iat[proxy.mapToSource(proxy.index(r, 0)).row(), 0])
            for r in range(proxy.rowCount())]


def test_descending_ties_keep_the_standing_order_and_a_repeat_click_is_a_no_op(qapp, qtbot):
    """A DESCENDING sort is also a stable sort over the standing view order — ties are
    NOT reversed, and clicking the same header twice shows the same rows twice. Pinned
    because an implementation that reverses a stable ascending order wholesale passes
    every single-click numeric-truth test yet OSCILLATES tie order on repeated clicks
    (measured: the pristine widget, driven through ``sortByColumn`` on a shown view,
    keeps standing order and is idempotent)."""
    t = _small_tied_table(qtbot)
    view = t.view
    # Establish a non-source standing order first: t_start ascending = ids reversed.
    view.sortByColumn(_col(view, "t_start_s"), Qt.AscendingOrder)
    assert _view_ids(t) == [8, 7, 6, 5, 4, 3, 2, 1, 0]

    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    first = _view_ids(t)
    # NaN rows {2,5,8} first, then the 2.0 group {0,3,6}, then 1.0 {1,4,7} — every
    # group in STANDING (reversed-id) order, exactly as the pristine widget showed.
    assert first == [8, 5, 2, 6, 3, 0, 7, 4, 1]

    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    assert _view_ids(t) == first, "a repeated descending click must change nothing"

    view.sortByColumn(_col(view, "depth_pA"), Qt.AscendingOrder)
    assert _view_ids(t) == [7, 4, 1, 6, 3, 0, 8, 5, 2], (
        "ascending chains the same standing ties, NaN rows moving to the end")


def test_a_refeed_under_a_remembered_descending_sort_orders_like_a_fresh_click(qapp, qtbot):
    """ONE ordering rule for both paths: re-feeding the table (detection re-run, bulk
    curation reset) while a descending sort is remembered must show the same order a
    fresh descending click over the new rows would. The pristine proxy's lazy
    ``create_mapping`` path disagreed with its own click path on descending ties; two
    surfaces disagreeing is the shape of defect this tree keeps paying for, so the
    replacement deliberately has no second rule."""
    t = _small_tied_table(qtbot)
    view = t.view
    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    clicked = _view_ids(t)
    assert clicked == [2, 5, 8, 0, 3, 6, 1, 4, 7]   # NaN first, ties in source order

    t.set_events(t._model.df.copy())                # the re-feed, sort remembered
    assert _view_ids(t) == clicked, (
        "a re-feed under the remembered sort must not re-order ties")


def test_selection_follows_its_event_through_a_sort(big_table):
    """A sort is a layout change, not a reset: the selected ROW must still be the same
    EVENT afterwards (persistent indexes are remapped), because the j/k/a/r curation
    keys act on the current row."""
    view = big_table.view
    view.sortByColumn(_col(view, "event_id"), Qt.AscendingOrder)
    view.selectRow(5)
    before = big_table._model._event_id(big_table._current_source_row())
    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    after = big_table._model._event_id(big_table._current_source_row())
    assert after == before, "the selection must follow the event, not squat on row 5"


def test_editing_the_sorted_on_column_resorts_dynamically(qapp, qtbot):
    """The old proxy's dynamicSortFilter re-sorted when the sorted-on column changed —
    that is how a toggled checkbox moves its row while the table is grouped by
    acceptance. Same behaviour, same trigger."""
    t = EventTable()
    qtbot.addWidget(t)
    t.set_events(pd.DataFrame({
        "event_id": [0, 1, 2, 3],
        "depth_pA": [1.0, 2.0, 3.0, 4.0],
        "accepted": [False, False, True, True],
        "flag": [""] * 4,
    }))
    view = t.view
    view.sortByColumn(_col(view, "accepted"), Qt.AscendingOrder)
    assert _shown(t, "accepted").tolist() == [False, False, True, True]
    # Reject one of the accepted rows THROUGH the model (the checkbox door).
    src_row = int(np.nonzero(t._model.df["event_id"].to_numpy() == 2)[0][0])
    t._model.set_accepted(src_row, False)
    assert _shown(t, "accepted").tolist() == [False, False, False, True], (
        "the freshly rejected row must move into the rejected group without a re-click")


def test_string_columns_still_order_as_text(big_table):
    """The flag column orders by its caption (the only ordering a string has here) —
    the rejecting-flag rows group together, not wherever an accidental numeric cast
    would scatter them."""
    view = big_table.view
    view.sortByColumn(_col(view, "flag"), Qt.DescendingOrder)
    shown = _shown(big_table, "flag")
    n_flagged = int((big_table._model.df["flag"] == "clipped_by_boundary").sum())
    assert (shown[:n_flagged] == "clipped_by_boundary").all()
    assert (shown[n_flagged:] == "").all()
