"""Results ▸ Columns…: several columns in one opening, worded as the header is.

Measured before (plotux/columns): the button opened a QMenu of checkable actions and Qt
hides a menu before it fires the entry clicked, so hiding five columns took five openings
and ten clicks. The menu listed schema order and raw names (``depth_pA`` where the header
says ``median depth (depth_pA)``; ``accepted`` 19th of 25 while it is the table's first
column), had no filter or bulk action, let every column be hidden — and saved it — and
its one reset entry, "Restore scientific column order", reset visibility too.

Synthetic detections only. QSettings is the suite's private IniFormat store.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.gui.panels import EventTable


def _frame():
    raw = 100 + np.random.default_rng(9).normal(0, 1, 10000)
    raw[2000:2250] -= 25
    raw[6000:6250] -= 30
    frame, _ = detect_events(raw.astype(np.float32), 50000, min_dwell_s=.0001)
    return frame


@pytest.fixture
def table(qapp, qtbot):
    QtCore.QSettings().remove(EventTable._COLUMN_SETTINGS_KEY)
    widget = EventTable()
    qtbot.addWidget(widget)
    widget.resize(900, 420)
    widget.set_events(_frame())
    widget.show()
    qtbot.waitExposed(widget)
    yield widget
    if widget.columns_popup is not None:
        widget.columns_popup.hide()
    QtCore.QSettings().remove(EventTable._COLUMN_SETTINGS_KEY)


def _open(table, qtbot):
    QTest.mouseClick(table.columns_button, Qt.LeftButton)
    popup = table.columns_popup
    assert popup is not None, "Columns… opened nothing"
    qtbot.waitUntil(popup.isVisible, timeout=2000)
    return popup


def _click_row(checklist, name):
    proxy = checklist.proxy
    for r in range(proxy.rowCount()):
        index = proxy.index(r, 0)
        if index.data(Qt.UserRole) == name:
            checklist.view.scrollTo(index)
            rect = checklist.view.visualRect(index)
            QTest.mouseClick(checklist.view.viewport(), Qt.LeftButton, Qt.NoModifier,
                             QPoint(rect.left() + 90, rect.center().y()))
            return
    raise AssertionError(f"no row {name!r}")


def _hidden(table):
    columns = list(table.dataframe.columns)
    return {str(name) for i, name in enumerate(columns) if table.view.isColumnHidden(i)}


def _visual_order(table):
    columns = list(table.dataframe.columns)
    header = table.view.horizontalHeader()
    return [str(columns[header.logicalIndex(v)]) for v in range(header.count())]


def _saved():
    return json.loads(str(QtCore.QSettings().value(EventTable._COLUMN_SETTINGS_KEY, "{}")))


def test_three_clicks_in_one_opening_hide_three_columns_with_one_save_each(table, qtbot,
                                                                           monkeypatch):
    popup = _open(table, qtbot)
    saves = []
    real_save = table._save_column_preferences
    monkeypatch.setattr(table, "_save_column_preferences",
                        lambda *a: (saves.append(1), real_save(*a)))
    for name in ("snr", "flag", "baseline_pA"):
        _click_row(popup.checklist, name)
        qtbot.wait(20)
        assert popup.isVisible(), f"Columns… closed on the click on {name}"
    assert {"snr", "flag", "baseline_pA"} <= _hidden(table)
    assert len(saves) == 3, "one QSettings write per gesture"
    assert {"snr", "flag", "baseline_pA"} <= set(_saved()["hidden"])


def test_the_rows_are_the_header_in_its_visual_order(table, qtbot):
    table.set_stale_columns(["depth_pA"], "superseded estimator")
    table.set_clock_offset(5.0)
    header = table.view.horizontalHeader()
    header.moveSection(header.visualIndex(list(table.dataframe.columns).index("snr")), 1)
    popup = _open(table, qtbot)
    model = table.view.model()
    expected = [model.headerData(header.logicalIndex(v), Qt.Horizontal, Qt.DisplayRole)
                for v in range(header.count())]
    proxy = popup.checklist.proxy
    rows = [proxy.index(r, 0).data() for r in range(proxy.rowCount())]
    assert rows == expected
    assert rows[:2] == ["accepted", "snr"]
    assert "median depth (depth_pA) ⚠" in rows
    assert "t_start_s (t₀=0)" in rows
    depth = next(proxy.index(r, 0) for r in range(proxy.rowCount())
                 if proxy.index(r, 0).data(Qt.UserRole) == "depth_pA")
    assert "superseded estimator" in depth.data(Qt.ToolTipRole)


def test_filter_dev_then_hide_these_hides_only_those_in_one_save(table, qtbot, monkeypatch):
    popup = _open(table, qtbot)
    checklist = popup.checklist
    assert checklist.filter.hasFocus(), "the keyboard lands in the filter"
    QTest.keyClicks(checklist.filter, "dev")
    assert checklist.shown_keys() == ["max_dev_pA", "mean_dev_pA"]
    assert checklist.none_btn.text() == "Hide these"
    before = _hidden(table)
    saves = []
    real_save = table._save_column_preferences
    monkeypatch.setattr(table, "_save_column_preferences",
                        lambda *a: (saves.append(1), real_save(*a)))
    QTest.mouseClick(checklist.none_btn, Qt.LeftButton)
    assert _hidden(table) == before | {"max_dev_pA", "mean_dev_pA"}
    assert len(saves) == 1, "a bulk gesture is one write, not one per column"
    assert popup.isVisible()


def test_hide_all_keeps_accepted_and_an_all_hidden_save_is_ignored(table, qtbot, qapp):
    header = table.view.horizontalHeader()      # `accepted` kept by name, not by place
    header.moveSection(header.visualIndex(list(table.dataframe.columns).index("snr")), 0)
    popup = _open(table, qtbot)
    QTest.mouseClick(popup.checklist.none_btn, Qt.LeftButton)
    columns = [str(c) for c in table.dataframe.columns]
    assert _hidden(table) == set(columns) - {"accepted"}
    assert "accepted" not in _saved()["hidden"]
    row = next(popup.checklist.model.item(r) for r in range(popup.checklist.model.rowCount())
               if popup.checklist.model.item(r).data(Qt.UserRole) == "accepted")
    assert not (row.flags() & Qt.ItemIsEnabled)
    assert row.toolTip() == "At least one column stays visible"
    table.set_column_visible("accepted", False)
    assert "accepted" not in _hidden(table), "the last visible column cannot be hidden"
    popup.hide()
    QtCore.QSettings().setValue(EventTable._COLUMN_SETTINGS_KEY,
                                json.dumps({"order": columns, "hidden": columns}))
    reopened = EventTable()
    qtbot.addWidget(reopened)
    reopened.set_events(_frame())
    assert "accepted" not in _hidden(reopened)
    assert _hidden(reopened) == {"start_idx", "end_idx"}


def test_default_columns_and_default_order_reset_one_thing_each(table, qtbot):
    header = table.view.horizontalHeader()
    columns = list(table.dataframe.columns)
    header.moveSection(header.visualIndex(columns.index("snr")), 0)
    table.set_column_visible("dwell_s", False)
    table.set_column_visible("start_idx", True)
    custom_order = _visual_order(table)
    popup = _open(table, qtbot)
    QTest.mouseClick(popup.checklist.default_btn, Qt.LeftButton)
    assert popup.checklist.default_btn.text() == "Default columns"
    assert _hidden(table) == set(EventTable._DEFAULT_HIDDEN) & set(map(str, columns))
    assert _visual_order(table) == custom_order, "Default columns moved a column"
    table.set_column_visible("frac_depth", False)
    order_button = next(b for b in popup.findChildren(type(table.columns_button))
                        if b.text() == "Default order")
    QTest.mouseClick(order_button, Qt.LeftButton)
    scientific = [n for n in EventTable._SCIENTIFIC_ORDER if n in columns]
    assert _visual_order(table)[:len(scientific)] == scientific
    assert "frac_depth" in _hidden(table), "Default order changed which columns show"
    assert _saved()["order"][:len(scientific)] == scientific
    assert "frac_depth" in _saved()["hidden"]
    proxy = popup.checklist.proxy
    assert proxy.index(0, 0).data(Qt.UserRole) == "accepted", "the rows follow the new order"


def test_down_space_toggles_the_next_row_esc_closes_and_reopening_matches(table, qtbot):
    popup = _open(table, qtbot)
    checklist = popup.checklist
    QTest.keyClick(checklist.filter, Qt.Key_Return)     # Return in the filter: to the list
    assert checklist.view.hasFocus()
    assert checklist.view.currentIndex().data(Qt.UserRole) == "accepted"
    QTest.keyClick(checklist.view, Qt.Key_Down)
    QTest.keyClick(checklist.view, Qt.Key_Space)
    assert checklist.view.currentIndex().data(Qt.UserRole) == "t_start_s"
    assert "t_start_s" in _hidden(table)
    QTest.keyClick(checklist.view, Qt.Key_Escape)
    qtbot.waitUntil(lambda: not popup.isVisible(), timeout=2000)
    table.set_column_visible("snr", False)          # changed while closed
    popup = _open(table, qtbot)
    shown = {str(c) for c in table.dataframe.columns} - _hidden(table)
    assert set(popup.checklist.checked()) == shown
    assert popup.checklist.count.text() == f"{len(shown)} of {len(table.dataframe.columns)} columns shown"


def test_header_right_click_hides_that_column(table, qtbot, menu_answer):
    header = table.view.horizontalHeader()
    logical = list(table.dataframe.columns).index("snr")
    pos = QPoint(header.sectionViewportPosition(logical) + 4, header.height() // 2)
    menu_answer("Hide “snr”")
    header.customContextMenuRequested.emit(pos)
    assert "snr" in _hidden(table)
    assert "snr" in _saved()["hidden"]
    menu_answer("Columns…")
    header.customContextMenuRequested.emit(pos)
    assert table.columns_popup is not None and table.columns_popup.isVisible()


def test_header_hide_is_greyed_on_the_last_visible_column(table, qtbot, blocked_modals):
    popup = _open(table, qtbot)
    QTest.mouseClick(popup.checklist.none_btn, Qt.LeftButton)
    popup.hide()
    header = table.view.horizontalHeader()
    logical = list(table.dataframe.columns).index("accepted")
    pos = QPoint(header.sectionViewportPosition(logical) + 4, header.height() // 2)
    header.customContextMenuRequested.emit(pos)
    assert blocked_modals and "Hide “accepted” (disabled)" in blocked_modals[-1].text
    assert "accepted" not in _hidden(table)


# -- integration fixes (plotux build 4, 2026-11-01) ------------------------------------------

def test_a_real_right_click_on_a_header_offers_hide(table, qtbot, menu_answer):
    """Through the window, as a user's click arrives: the header must ask for the custom
    menu (its context-menu policy), or the right-click never reaches the handler."""
    header = table.view.horizontalHeader()
    logical = list(table.dataframe.columns).index("snr")
    pos = QPoint(header.sectionViewportPosition(logical) + 4, header.height() // 2)
    menu_answer("Hide “snr”")
    QTest.mouseClick(header.window().windowHandle(), Qt.RightButton, Qt.NoModifier,
                     header.viewport().mapTo(header.window(), pos))
    qtbot.waitUntil(lambda: "snr" in _hidden(table), timeout=2000)


def test_the_greyed_hide_entry_shows_its_reason(table, qtbot, monkeypatch):
    """QMenu shows an action's tooltip only when asked to; the reason must be seen."""
    menus = []
    monkeypatch.setattr(table, "_exec_menu", lambda menu, _pos: (menus.append(menu), None)[1])
    popup = _open(table, qtbot)
    QTest.mouseClick(popup.checklist.none_btn, Qt.LeftButton)
    popup.hide()
    header = table.view.horizontalHeader()
    logical = list(table.dataframe.columns).index("accepted")
    header.customContextMenuRequested.emit(
        QPoint(header.sectionViewportPosition(logical) + 4, header.height() // 2))
    menu = menus[-1]
    hide = next(a for a in menu.actions() if a.text().startswith("Hide"))
    assert not hide.isEnabled() and hide.toolTip() == "At least one column stays visible"
    assert menu.toolTipsVisible()


def test_the_columns_button_still_reads_as_a_list(table):
    assert table.columns_button.text() == "Columns… ▾"
    assert table.columns_button.accessibleName() == "Columns…"


def test_a_refeed_under_the_open_popup_never_hides_a_column_it_did_not_list(table, qtbot):
    """A detection finishing while Columns… is open adds a column; the next row click
    must not hide it (the popup's rows were filled before it existed)."""
    popup = _open(table, qtbot)
    frame = _frame()
    frame["zz_new"] = 1.0
    table.set_events(frame)
    assert "zz_new" in popup.checklist.keys(), "the open popup is refilled"
    _click_row(popup.checklist, "snr")
    assert "snr" in _hidden(table)
    assert "zz_new" not in _hidden(table)
    assert "zz_new" not in _saved()["hidden"]


# -- a column the table on screen lacks (build 5, tables-1) ----------------------------------
#
# One saved view serves every table the Results selector shows. A change made on a table that
# lacks a column (amplitude columns before Curate ▸ Apply, ML or hand-made columns) saved only
# the columns on screen, and the unknown-field rule then took the missing ones for new: back on
# the first table a column the user hid was shown and a default-hidden one they showed was
# hidden. "The view is remembered" has to hold across tables.

def _amplitude_frame():
    """The detection after Curate ▸ Apply: amplitude columns the other tables lack."""
    frame = _frame()
    frame["amplitude_pA"] = frame["depth_pA"]
    frame["amplitude_method"] = "peak"
    frame["amplitude_status"] = "measured"
    frame["amplitude_level_pA"] = 75.0
    return frame


def _toggle(table, qtbot, name):
    """One Columns… gesture: open it, click the row, Esc."""
    popup = _open(table, qtbot)
    _click_row(popup.checklist, name)
    QTest.keyClick(popup.checklist.view, Qt.Key_Escape)
    qtbot.waitUntil(lambda: not popup.isVisible(), timeout=2000)


def _footer_button(popup, text):
    return next(b for b in popup.findChildren(QtWidgets.QPushButton) if b.text() == text)


@pytest.mark.parametrize("gesture", ["Columns… row", "Default order", "header Hide"])
def test_a_choice_survives_a_change_on_a_table_without_that_column(table, qtbot, menu_answer,
                                                                    gesture):
    table.set_events(_amplitude_frame())
    assert "amplitude_level_pA" in _hidden(table) and "amplitude_pA" not in _hidden(table)
    _toggle(table, qtbot, "amplitude_pA")                  # hidden by the user
    _toggle(table, qtbot, "amplitude_level_pA")            # shown by the user (hidden by default)
    assert "amplitude_pA" in _hidden(table) and "amplitude_level_pA" not in _hidden(table)

    table.set_events(_frame())                             # another table: no amplitude columns
    if gesture == "Columns… row":
        _toggle(table, qtbot, "snr")
    elif gesture == "Default order":
        popup = _open(table, qtbot)
        QTest.mouseClick(_footer_button(popup, "Default order"), Qt.LeftButton)
        QTest.keyClick(popup.checklist.view, Qt.Key_Escape)
        qtbot.waitUntil(lambda: not popup.isVisible(), timeout=2000)
    else:
        # The header's own request, as test_header_right_click_hides_that_column sends it (the
        # real right-click is pinned by test_a_real_right_click_on_a_header_offers_hide; a
        # second QWindow right-click in one process never reaches the header offscreen).
        header = table.view.horizontalHeader()
        logical = list(table.dataframe.columns).index("snr")
        menu_answer("Hide “snr”")
        header.customContextMenuRequested.emit(
            QPoint(header.sectionViewportPosition(logical) + 4, header.height() // 2))
        assert "snr" in _hidden(table)
    saved = _saved()
    assert "amplitude_pA" in saved["hidden"] and "amplitude_level_pA" not in saved["hidden"]
    assert {"amplitude_pA", "amplitude_level_pA"} <= set(saved["order"])

    table.set_events(_amplitude_frame())                   # back to the first table
    assert "amplitude_pA" in _hidden(table), "a column the user hid is shown again"
    assert "amplitude_level_pA" not in _hidden(table), "a column the user showed is hidden again"
    if gesture == "Default order":
        columns = [str(c) for c in table.dataframe.columns]
        scientific = [n for n in EventTable._SCIENTIFIC_ORDER if n in columns]
        assert _visual_order(table)[:len(scientific)] == scientific, \
            "Default order puts the amplitude columns in their scientific place too"
    else:
        assert "snr" in _hidden(table)


def test_default_columns_still_forgets_the_choices_for_columns_not_on_screen(table, qtbot):
    table.set_events(_amplitude_frame())
    _toggle(table, qtbot, "amplitude_pA")
    _toggle(table, qtbot, "amplitude_level_pA")
    table.set_events(_frame())
    popup = _open(table, qtbot)
    QTest.mouseClick(popup.checklist.default_btn, Qt.LeftButton)
    QTest.keyClick(popup.checklist.view, Qt.Key_Escape)
    qtbot.waitUntil(lambda: not popup.isVisible(), timeout=2000)
    table.set_events(_amplitude_frame())
    assert "amplitude_pA" not in _hidden(table) and "amplitude_level_pA" in _hidden(table)


def _selector_trace() -> np.ndarray:
    """Forty synthetic blockades, a quarter of them double, on a 1.5 pA open pore."""
    n = 260_000
    rng = np.random.default_rng(3)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    depths = np.linspace(14.0, 46.0, 40)
    rng.shuffle(depths)
    pos = 1_000
    for i, depth in enumerate(depths):
        dwell = 120 + int(rng.integers(0, 140))
        if i % 4 == 0:
            x[pos:pos + 70] -= depth
            x[pos + 90:pos + 160] -= depth
        else:
            x[pos:pos + dwell] -= depth
        pos += 5_000
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot, tmp_path):
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
    from nanoscope.gui.main_window import MainWindow

    QtCore.QSettings().remove(EventTable._COLUMN_SETTINGS_KEY)
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1200, 780)
    win.show()
    QtWidgets.QApplication.processEvents()
    arr = _selector_trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "selector.dat", format_name="synthetic",
                         n_samples=int(arr.size), sample_rate_hz=50_000.0,
                         rate_provenance=Provenance.FILE_META, t0_s=0.0, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(arr,))
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    yield win
    table = win._analysis_panel._table
    if table.columns_popup is not None:
        table.columns_popup.hide()
    win._analysis_controller.abort()
    win._controller.abort()
    win.hide()
    QtCore.QSettings().remove(EventTable._COLUMN_SETTINGS_KEY)


def _detect(win, qtbot):
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._analysis_panel.trigger_detection()
    qtbot.wait(50)


def _pick(win, qtbot, name):
    selector = win._analysis_panel._result_selector
    index = selector.findText(name, Qt.MatchStartsWith)
    assert index >= 0, [selector.itemText(i) for i in range(selector.count())]
    selector.setCurrentIndex(index)
    qtbot.wait(50)


def test_the_table_selector_round_trip_keeps_a_hidden_amplitude_column(window, qtbot):
    """The same in the window: Schmitt and CUSUM on one recording, an amplitude applied on the
    Schmitt table in the Event Inspector, amplitude_pA hidden in Columns…, CUSUM picked in the
    Table selector and snr hidden there, Schmitt picked again."""
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    _detect(window, qtbot)
    panel.detection_panel.set_detector("detect.cusum")
    qtbot.wait(30)
    _detect(window, qtbot)
    _pick(window, qtbot, "Schmitt")
    window._act_event_inspector.trigger()
    inspector = window._event_inspector
    qtbot.wait(50)
    QTest.mouseClick(inspector._amplitude.apply_button, Qt.LeftButton)
    qtbot.wait(50)
    table = panel._table
    assert "amplitude_pA" in table.dataframe.columns and "amplitude_pA" not in _hidden(table)
    inspector.hide()
    window.activateWindow()
    _toggle(table, qtbot, "amplitude_pA")
    assert "amplitude_pA" in _hidden(table)
    _pick(window, qtbot, "CUSUM")
    assert "amplitude_pA" not in table.dataframe.columns
    _toggle(table, qtbot, "snr")
    _pick(window, qtbot, "Schmitt")
    assert "snr" in _hidden(table)
    assert "amplitude_pA" in _hidden(table), "amplitude_pA the user hid is shown again"
