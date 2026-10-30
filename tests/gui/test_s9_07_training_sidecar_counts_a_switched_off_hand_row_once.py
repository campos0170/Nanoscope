"""S9-07 — With "Include hand-made events" off, the training sidecar counts what it left out once.

MEASURED DEFECT (sweep 9, a synthetic Schmitt table of two detected events and one event added by
hand, nothing rejected). Analysis ▸ Include hand-made events in exports unticked, then Analysis ▸
Export events for training… ▸ Accepted candidates: the CSV correctly held the 2 detected rows, but
``train.meta.json`` said ``n_in_table`` 3, ``n_withheld`` 1, ``n_added_excluded`` 1 and
``withheld_by_status`` {"accepted": 1}, ``withheld_by_source`` {"manual": 1}: the one hand-made
event counted twice, and called withheld under the sidecar's own note that a withheld row is a row
with accepted=False. ``scoring.write_training_export`` handed the WHOLE table to
``curation_stamp`` as ``source_df``. The switch's contract is that ``n_withheld`` keeps meaning the
rows REJECTED and a row the switch kept out is counted in ``n_added_excluded`` only (design 7.2);
the events export keeps it (its own test pins ``n_withheld == 0`` for exactly this case).

THE FIX. With the switch off, the training export's curation block is computed over the rows the
switch KEPT, as the events export's is: ``source_df`` is the table without its hand-made rows, so
``n_in_table``, ``n_withheld`` and ``withheld_by_*`` describe the detected rows and the hand-made
rows the selection would have written are in ``n_added_excluded`` alone. With the switch on the
sidecar is what it was.

These tests drive a real MainWindow: Ctrl+M's door, a rejection through the panel method the table
calls, the include switch on the Analysis menu, and Analysis ▸ Export events for training… with its
own question answered "Accepted candidates" and the save dialog through the export seam; then the
CSV and the sidecar on disk, and the events export's sidecar for the same table beside them.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.gui import main_window as mw

from .test_manual_event_door import (  # noqa: F401  (``win`` is a fixture)
    _MISSED, _open, _table, win)


@pytest.fixture
def rig(win, monkeypatch, tmp_path):
    """The training door's selection box answered "Accepted candidates", the save dialog
    answered with ``rig.dest``, and every other modal recorded."""
    seen = SimpleNamespace(dest=tmp_path / "train.csv", modals=[])

    def exec_(box):
        if "Which events" not in box.text():
            seen.modals.append(box.text())
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", exec_)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "clickedButton",
        lambda box: next((b for b in box.buttons() if b.text() == "Accepted candidates"), None))
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: seen.dest)
    return seen


def _export(win, rig, action: str, name: str) -> dict:
    """Click the Analysis menu item; the curation block of the sidecar it wrote."""
    rig.dest = rig.dest.with_name(name)
    act = getattr(win, action)
    assert act.isEnabled(), f"{act.text()!r} is disabled -- the click would do nothing"
    act.trigger()
    QtWidgets.QApplication.processEvents()
    assert rig.modals == [], rig.modals
    return json.loads(rig.dest.with_suffix(".meta.json").read_text())["curation"]


def _one_hand_made_event(win, tmp_path) -> str:
    doc_id = _open(win, tmp_path)
    assert win._analysis_panel.add_manual_event(doc_id, 0, *_MISSED).ok
    df = _table(win, doc_id, "detect.schmitt")
    assert len(df) == 3 and bool(df["accepted"].all()), "precondition: nothing is rejected"
    return doc_id


def test_a_switched_off_hand_made_event_is_not_also_counted_as_withheld(win, rig, tmp_path):
    """The finding's case: nothing rejected, one hand-made event switched off. Nothing was
    withheld, and the one event is counted once, in n_added_excluded -- as the events export
    of the same table says."""
    _one_hand_made_event(win, tmp_path)
    win._act_include_hand_made.setChecked(False)
    try:
        block = _export(win, rig, "_act_export_training", "train.csv")
        assert len(pd.read_csv(rig.dest)) == 2              # the hand row is left out
        events = _export(win, rig, "_act_export_events", "events.csv")
    finally:
        win._act_include_hand_made.setChecked(True)

    assert block["include_added"] is False and block["n_added_excluded"] == 1
    assert block["n_in_table"] == 2 and block["n_withheld"] == 0, block
    assert not block["withheld_by_status"] and not block["withheld_by_source"], block
    for key in ("include_added", "n_added_excluded", "n_in_table", "n_withheld"):
        assert block[key] == events[key], (key, block[key], events[key])


def test_a_rejected_detected_event_is_still_the_one_withheld(win, rig, tmp_path):
    """With a detected event rejected by hand, n_withheld counts that rejection and nothing
    else: the withheld breakdown names one rejected detector row and no hand-made one."""
    doc_id = _one_hand_made_event(win, tmp_path)
    df = _table(win, doc_id, "detect.schmitt")
    detected = int(df.loc[df["source"] != "manual", "event_id"].iloc[0])
    win._analysis_panel.set_event_accepted(detected, False)
    QtWidgets.QApplication.processEvents()
    win._act_include_hand_made.setChecked(False)
    try:
        block = _export(win, rig, "_act_export_training", "train.csv")
    finally:
        win._act_include_hand_made.setChecked(True)

    assert len(pd.read_csv(rig.dest)) == 1
    assert block["n_in_table"] == 2 and block["n_withheld"] == 1, block
    assert block["n_added_excluded"] == 1
    assert block["withheld_by_status"] == {"rejected": 1}, block
    assert "manual" not in (block["withheld_by_source"] or {}), block


def test_with_the_switch_on_the_sidecar_counts_the_whole_table(win, rig, tmp_path):
    """The switch on (the default): the hand-made event is written and counted, nothing is
    excluded, and nothing is withheld -- the sidecar this export always wrote."""
    _one_hand_made_event(win, tmp_path)
    assert win._act_include_hand_made.isChecked()

    block = _export(win, rig, "_act_export_training", "train.csv")

    assert len(pd.read_csv(rig.dest)) == 3
    assert block["include_added"] is True and block["n_added_excluded"] == 0
    assert block["n_in_table"] == 3 and block["n_withheld"] == 0
    assert block["by_kind"].get("manual") == 1, block
