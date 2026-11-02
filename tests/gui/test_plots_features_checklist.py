"""Distributions: the clustering & PCA Features list is the shared ColumnChecklist (G3).

One row click, Space, Select all, Clear or Default is one gesture and one Undo step; the
row identifiers are offered last under "Identifiers" and left out of Default; and the
group title no longer shows a mnemonic in place of "&". Synthetic DataFrames only.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from nanoscope.core.session import Session
from nanoscope.gui.column_checklist import ColumnChecklist
from nanoscope.gui.plots_panel import PlotsWindow


def _window(qtbot) -> PlotsWindow:
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "event_id": np.arange(n), "start_idx": np.arange(n) * 10, "end_idx": np.arange(n) * 10 + 5,
        "dwell_s": rng.exponential(0.002, n), "depth_pA": rng.normal(50, 5, n),
        "snr": rng.normal(8, 1, n), "custom": rng.normal(0, 1, n),
        "accepted": True, "filter_limited": False})
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data({"probe": df})
    return w


def test_the_features_are_a_checklist_with_identifiers_last_and_out_of_default(qapp, qtbot):
    w = _window(qtbot)
    checklist = w._features
    assert isinstance(checklist, ColumnChecklist)
    keys = checklist.keys()
    assert keys[-3:] == ["event_id", "start_idx", "end_idx"], keys
    captions = [checklist.model.item(r).text() for r in range(checklist.model.rowCount())
                if not checklist.model.item(r).isCheckable()]
    assert "Identifiers" in captions
    assert set(w.feature_selection()) == {"dwell_s", "depth_pA", "snr"}
    checklist.set_checked(["custom", "event_id"])
    checklist.default_btn.click()
    assert set(w.feature_selection()) == {"dwell_s", "depth_pA", "snr"}


def test_set_feature_selection_round_trips_and_a_row_click_is_one_undo_step(qapp, qtbot):
    w = _window(qtbot)
    w.set_feature_selection(["dwell_s", "custom"])
    assert w.feature_selection() == ["dwell_s", "custom"]
    w._history_touch()
    back = w._history.depth()[0]
    checklist = w._features
    row = checklist.keys().index("snr")
    index = checklist.proxy.mapFromSource(checklist.model.index(row + 0, 0))
    # Toggle one row the way the delegate does on a click: one gesture.
    item = checklist.model.itemFromIndex(checklist.proxy.mapToSource(index))
    if item.text() != "snr":       # a separator row shifts the index
        item = next(checklist.model.item(r) for r in range(checklist.model.rowCount())
                    if checklist.model.item(r).text() == "snr")
    item.setCheckState(QtCore.Qt.Checked)
    assert w.feature_selection() == ["dwell_s", "snr", "custom"]
    assert w._history.depth()[0] == back + 1
    assert w.undo_config()
    assert w.feature_selection() == ["dwell_s", "custom"]


def test_the_group_title_escapes_the_ampersand(qapp, qtbot):
    w = _window(qtbot)
    boxes = [b for b in w.findChildren(QtWidgets.QGroupBox) if "clustering" in b.title()]
    assert boxes and boxes[0].title() == "Features (clustering && PCA)"
