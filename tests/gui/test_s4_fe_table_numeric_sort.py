"""FE-01 — the events table sorts its numeric columns as NUMBERS, not as their captions.

The table advertises itself as "Sortable, curatable table over an EventSet" and its
headers are clickable, but the proxy was given ``Qt.DisplayRole`` as its sort role while
the model serves ``f"{v:.4g}"`` / ``f"{v:.6g}"`` / ``str(int)``. So every "sort" was a
lexicographic sort of the formatted strings:

* clicking ``depth_pA`` for a descending sort put "99.49" above "150.2" — on the owner's
  115-event apo-ferritin detection the top row read 99.49 pA while the deepest event was
  138.855 pA, and NONE of the ten deepest events appeared in the ten rows shown;
* ``.6g`` emits exponents, so a 9.9e-05 s dwell sorted as the LONGEST event;
* ``event_id`` ran 0, 1, 10, 11, 2, 3, 4.

Nothing looked broken, because the visible column still descends. And the j/k/a/r
curation keys walk the view's order, so "reject the worst ten" rejected ten arbitrary
rows.

The sibling table already had the rule and the test (``measure_panel._SORT_ROLE``, and
``test_measure_navigation.py::test_sorting_orders_numerically_not_lexically``); the
events table had zero sort coverage. These tests are that coverage: every case here
would pass under a text sort only by accident, and each one is chosen so a text sort
fails it (9 vs 10 vs 100, exponents, ints past 9).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import EventTable

FS = 50_000.0
N = 60_000
I0 = -1000.0


# ================================================================================================
# builders
# ================================================================================================
def _frame() -> pd.DataFrame:
    """A small frame whose every numeric column orders DIFFERENTLY as text than as numbers.

    ``9 < 10 < 100`` is the canonical trap (as text: "10" < "100" < "9"), the dwells span
    the exponent boundary ``.6g`` formats as ``9.9e-05``, and the ids run past 9.
    """
    return pd.DataFrame({
        "event_id": [0, 1, 2, 3, 10, 11, 100],
        "recording": ["r"] * 7,
        "pore_id": ["p"] * 7,
        "start_idx": [10, 20, 30, 40, 50, 60, 70],
        "end_idx": [15, 25, 35, 45, 55, 65, 75],
        "t_start_s": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "dwell_s": [0.02, 9.9e-05, 0.001, 0.1, 0.00012, 0.009, 0.0004],
        "baseline_pA": [-60.0] * 7,
        "depth_pA": [9.0, 10.0, 100.0, 99.49, 150.2, 8.5, 122.366543],
        "frac_depth": [0.15, 0.17, 1.7, 1.66, 2.5, 0.14, 2.0],
        "ecd_pA_s": [0.18, 0.001, 0.1, 9.9, 0.02, 0.07, 1.99],
        "max_dev_pA": [10.0, 11.0, 110.0, 105.0, 160.0, 9.0, 152.28],
        "mean_dev_pA": [9.0, 10.0, 100.0, 99.0, 150.0, 8.0, 99.5],
        "rise_s": [0.001912] * 7,
        "fall_s": [0.002805] * 7,
        "snr": [3.0, 4.0, 40.0, 30.0, 100.0, 2.0, 21.35],
        "n_levels": [2, 3, 10, 9, 11, 2, 4],
        "filter_limited": [False] * 7,
        "accepted": [True, True, False, True, True, False, True],
        "flag": [""] * 7,
        "added": [False] * 7,
    })


def _signal(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    # Deliberately UNEQUAL depths, so a sorted depth column has something to order.
    for i, (s, e) in enumerate([(10_000, 10_600), (20_000, 20_800),
                                (30_000, 30_500), (40_000, 40_700),
                                (50_000, 50_300)]):
        x[s:e] -= 120.0 + 40.0 * i
    return x.astype(np.float32)


def _doc(tmp_path, signal, name="apo.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _table(qtbot) -> EventTable:
    t = EventTable()
    qtbot.addWidget(t)
    t.set_events(_frame())
    return t


def _col(view, name: str) -> int:
    # By the model's column NAME, not the header caption. The caption is prose: it may
    # carry a badge ("... ⚠"), and since b21e8ae (2026-09-06) the depth header reads
    # "median depth (depth_pA)", so matching its first word found no depth_pA column at
    # all. The proxy never permutes columns, so the source index is the view's index.
    cols = view.model().sourceModel()._columns
    assert name in cols, f"no {name} column"
    return cols.index(name)


def _raw_order(table: EventTable, colname: str) -> list:
    """The column's RAW values in the order the view is showing them.

    Read through ``mapToSource`` out of the DataFrame — never parsed back out of the
    display string, which is rounded to 4 significant figures and cannot distinguish
    122.3665 from 122.4.
    """
    view = table.view
    proxy = view.model()
    df = table._model.df
    out = []
    for r in range(proxy.rowCount()):
        src = proxy.mapToSource(proxy.index(r, 0)).row()
        out.append(df.iloc[src][colname])
    return out


def _click_header(qtbot, view, col: int) -> None:
    """A REAL mouse click on the header section — the gesture the user makes."""
    header = view.horizontalHeader()
    assert header.sectionsClickable(), "the user cannot even click to sort"
    x = header.sectionPosition(col) + header.sectionSize(col) // 2
    y = header.height() // 2
    qtbot.mouseClick(header.viewport(), Qt.LeftButton, pos=QtCore.QPoint(int(x), int(y)))


# ================================================================================================
# the trap, column by column
# ================================================================================================
@pytest.mark.parametrize("colname", [
    "event_id", "start_idx", "end_idx", "t_start_s", "dwell_s", "baseline_pA",
    "depth_pA", "frac_depth", "ecd_pA_s", "max_dev_pA", "mean_dev_pA", "rise_s",
    "fall_s", "snr", "n_levels",
])
def test_every_numeric_column_sorts_numerically_in_both_directions(qtbot, colname):
    """EVERY numeric column, not just the one the finding was reported on.

    Ascending and descending, because a proxy can be handed a correct role and still be
    read in one direction only."""
    table = _table(qtbot)
    view = table.view
    col = _col(view, colname)
    raw = list(table._model.df[colname])

    view.sortByColumn(col, Qt.AscendingOrder)
    assert _raw_order(table, colname) == sorted(raw), (
        f"{colname} ascending is not numeric: {_raw_order(table, colname)}")

    view.sortByColumn(col, Qt.DescendingOrder)
    assert _raw_order(table, colname) == sorted(raw, reverse=True), (
        f"{colname} descending is not numeric: {_raw_order(table, colname)}")


def test_nine_ten_one_hundred_is_the_case_a_text_sort_gets_wrong(qtbot):
    """The one-line statement of the defect: as text the order is 10, 100, 9."""
    table = _table(qtbot)
    view = table.view
    view.sortByColumn(_col(view, "depth_pA"), Qt.AscendingOrder)
    shown = _raw_order(table, "depth_pA")
    trio = [v for v in shown if v in (9.0, 10.0, 100.0)]
    assert trio == [9.0, 10.0, 100.0], f"string sort would give 10, 100, 9 — got {trio}"


def test_the_shortest_dwell_sorts_first_even_though_it_is_written_as_an_exponent(qtbot):
    """``.6g`` writes 9.9e-05 s as "9.9e-05", which text-sorts after "0.02"."""
    table = _table(qtbot)
    view = table.view
    view.sortByColumn(_col(view, "dwell_s"), Qt.AscendingOrder)
    assert _raw_order(table, "dwell_s")[0] == pytest.approx(9.9e-05), (
        "the sub-resolution event must be the SHORTEST, not the longest")


def test_event_id_runs_0_1_2_10_11_100_not_0_1_10_100_11_2(qtbot):
    table = _table(qtbot)
    view = table.view
    view.sortByColumn(_col(view, "event_id"), Qt.AscendingOrder)
    assert _raw_order(table, "event_id") == [0, 1, 2, 3, 10, 11, 100]


def test_the_display_string_is_untouched_only_the_ordering_changed(qtbot):
    """The fix must not change what any cell READS — the numbers keep their formatting;
    it is the comparison that moved off the caption."""
    table = _table(qtbot)
    model = table._model
    col = list(model.df.columns).index("depth_pA")
    idx = model.index(list(model.df["depth_pA"]).index(122.366543), col)
    assert model.data(idx, Qt.DisplayRole) == "122.4"
    assert model.data(idx, Qt.UserRole + 1) == pytest.approx(122.366543)


def test_the_accepted_checkbox_column_groups_rather_than_comparing_nothing(qtbot):
    """The checkbox column served no sort value at all, so clicking it did nothing.
    Its only honest ordering is "the curated-in rows together"."""
    table = _table(qtbot)
    view = table.view
    view.sortByColumn(_col(view, "accepted"), Qt.AscendingOrder)
    order = _raw_order(table, "accepted")
    assert order == sorted(order), order
    assert set(order) == {True, False}


def test_a_real_header_click_sorts_numerically(qtbot):
    """Through the actual gesture: a mouse click on the header section, not sortByColumn."""
    table = _table(qtbot)
    table.show()
    qtbot.waitExposed(table)
    view = table.view
    col = _col(view, "depth_pA")
    _click_header(qtbot, view, col)          # ascending
    _click_header(qtbot, view, col)          # descending
    header = view.horizontalHeader()
    assert header.sortIndicatorSection() == col
    assert header.sortIndicatorOrder() == Qt.DescendingOrder
    shown = _raw_order(table, "depth_pA")
    assert shown == sorted(shown, reverse=True), shown
    assert shown[0] == max(table._model.df["depth_pA"]), (
        "row 1 of a descending depth sort must BE the deepest event")


def test_the_deepest_events_are_the_rows_a_descending_sort_shows(qtbot, win, tmp_path):
    """Wired: a real detection in the real AnalysisPanel's table, the way the owner reads
    "show me the deepest events" off the top of the table."""
    signal = _signal()
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4,
                             recording="apo.dat", pore_id="pore-1")
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    table = win._analysis_panel._table
    view = table.view
    col = _col(view, "depth_pA")
    view.sortByColumn(col, Qt.DescendingOrder)

    shown = _raw_order(table, "depth_pA")
    truth = sorted(table._model.df["depth_pA"], reverse=True)
    assert shown == truth, "the wired panel's table is the one the owner reads"
    top3 = set(np.round(shown[:3], 6))
    assert top3 == set(np.round(truth[:3], 6)), (
        "the three deepest events must be the three rows at the top")


def test_curation_keys_walk_the_numeric_order(qtbot):
    """The j/k/a/r keys walk the VIEW's order, which is why the wrong ordering rejected
    the wrong events. After a descending depth sort, 'r' on the first row must reject
    the deepest event."""
    table = _table(qtbot)
    table.show()
    qtbot.waitExposed(table)
    view = table.view
    view.sortByColumn(_col(view, "depth_pA"), Qt.DescendingOrder)
    view.setCurrentIndex(view.model().index(0, 0))
    view.setFocus()
    qtbot.keyClick(view, Qt.Key_R)
    df = table._model.df
    deepest_id = int(df.loc[df["depth_pA"].idxmax(), "event_id"])
    row = df[df["event_id"] == deepest_id].iloc[0]
    assert not bool(row["accepted"]), (
        "'reject the worst' must reject the deepest event, not an arbitrary one")
