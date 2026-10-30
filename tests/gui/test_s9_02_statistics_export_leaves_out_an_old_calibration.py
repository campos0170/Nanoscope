"""S9-02 — Export event statistics leaves out a table whose depths are on an old calibration.

MEASURED DEFECT (sweep 9, a synthetic 60,000-sample recording, open pore -1000 pA, a 300 pA event
added by hand). After the gain was corrected to 2 in the Calibration panel the table wore "Depths
stale — the calibration changed", and Analysis ▸ Export event statistics… ("This table") wrote its
row without a word: baseline -1000.6, the ``baseline_pA`` the table stored at the OLD gain, against
a median and level of -1399.4 read off the live, re-gained signal, so amplitude -398.9,
frac_amplitude -0.40 and SNR -48.5, stamped ``cal_gain`` 2.0: neither gain's numbers, every sign
wrong. A plain Schmitt table did the same with no hand-made event in it, every amplitude negative.
A table on another sample rate was already left out ("on another sample rate"); one on another
calibration was not.

THE FIX. ``AnalysisPanel._statistics_staleness`` asks each table the amber line's own two tests,
``stale_rate_reason`` and ``_calibration_moved`` (which the amber line now calls itself, so the two
cannot drift), and the export leaves a stale table out. The table on screen alone is refused
BEFORE the file dialog, way forward first (rebuild the Manual table, re-run detection, or re-run
the source detectors and reconcile again) and then the cause; over several tables the stale ones are
skipped and listed on the status line ("Left out: Schmitt of door.dat (on the old calibration)").
Each table is judged against ITS OWN recording and channel, so "Every Manual table in the project"
leaves out the recording whose gain moved and keeps the rest. Nothing is re-scaled: after the
amber line's Rebuild the same export is written, every number on the new gain.

These tests drive a real MainWindow: Ctrl+M's door, the call the Calibration panel makes
(``Session.set_calibration``), Analysis ▸ Export event statistics… with its scope question and
file dialog answered through their seams, the amber line's own button, and the CSV left on disk.
Every modal is recorded, so a refusal that raised one fails loudly.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.events import MANUAL_MODULE_ID
from nanoscope.core.calibration import ChannelCalibration
import nanoscope.gui.analysis_panel as ap

from .test_manual_event_door import (  # noqa: F401  (``win`` is a fixture)
    DEPTH, I0, _DETECTED, _MISSED, _doc, _open, _signal, _table, win)


@pytest.fixture
def rig(win, monkeypatch, tmp_path):
    """The two questions the export asks, answered, and every modal it might raise, recorded."""
    seen = SimpleNamespace(asked=[], modals=[], scope="shown", dest=tmp_path / "stats.csv",
                           status=[])

    def ask(*args, **kwargs):
        seen.asked.append(kwargs.get("title", ""))
        return seen.dest

    def static_modal(name):
        def raised(*args, **kwargs):
            seen.modals.append(f"{name}: {args[2] if len(args) > 2 else ''}")
            return QtWidgets.QMessageBox.Ok
        return staticmethod(raised)

    monkeypatch.setattr(ap, "ask_export_path", ask)
    monkeypatch.setattr(win, "_choose_statistics_scope", lambda: seen.scope)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: seen.modals.append(self.text()) or 0)
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, static_modal(name))
    win.statusBar().messageChanged.connect(lambda text: text and seen.status.append(text))
    return seen


def _regain(win, doc_id, gain=2.0) -> None:
    """What the Calibration panel does when the gain is corrected."""
    win._session.set_calibration(doc_id, 0, ChannelCalibration(gain=gain, target_units="pA"),
                                 write_sidecar=False)
    QtWidgets.QApplication.processEvents()


def _export(win, rig, scope: str) -> None:
    """Analysis ▸ Export event statistics…, the scope question answered with ``scope``."""
    action = win._act_export_event_statistics
    assert action.isEnabled(), "the menu item would swallow the click"
    rig.scope = scope
    rig.asked.clear()
    rig.status.clear()
    action.trigger()
    QtWidgets.QApplication.processEvents()


def _open_named(win, tmp_path, name: str) -> str:
    """Another recording, selected, as the File Manager selects one."""
    doc_id = win._session.add_document(_doc(tmp_path, _signal(_DETECTED + [_MISSED]), name=name))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    return doc_id


# ================================================================================================
# the table on screen: refused before the file dialog, way forward first
# ================================================================================================
@pytest.mark.parametrize("table, way", [
    (MANUAL_MODULE_ID, "first rebuild the Manual table."),
    ("detect.schmitt", "first re-run detection."),
])
def test_the_table_on_screen_on_an_old_calibration_is_refused_before_the_file_dialog(
        win, rig, tmp_path, table, way):
    """The finding's case, for a Manual table and for a plain Schmitt table: the gain is
    corrected, the amber line says the depths are stale, and the export of that table writes
    nothing and says how to make it writable."""
    doc_id = _open(win, tmp_path, detect=table != MANUAL_MODULE_ID)
    panel = win._analysis_panel
    if table == MANUAL_MODULE_ID:
        assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    assert panel.current_detection().module_id == table
    _regain(win, doc_id)
    assert panel._stale.kind == "calibration" and not panel._stale.isHidden(), (
        "precondition: the app itself flags this table")

    _export(win, rig, "shown")

    said = win.statusBar().currentMessage()
    assert not rig.dest.exists(), "a refused export writes nothing"
    assert rig.asked == [], "no save dialog for a file that cannot be written"
    assert rig.modals == [], f"a modal would hang a scripted export: {rig.modals}"
    assert said.startswith(f"Nothing was exported — {way}"), said
    assert "calibration changed" in said and "old gain" in said, said


# ================================================================================================
# several tables: the stale ones are skipped and named, the rest written on one gain
# ================================================================================================
def test_every_manual_table_leaves_out_the_recording_whose_gain_moved(win, rig, tmp_path):
    """Two recordings, a hand-made event on each, the gain corrected on the first only while
    the second is shown. The one file holds the second recording's event, on its own gain, and
    the status line names the first as left out: each table is judged against its own
    recording's calibration, not against the recording on screen."""
    panel = win._analysis_panel
    first = _open_named(win, tmp_path, "first.dat")
    assert panel.add_manual_event(first, 0, *_MISSED).ok
    second = _open_named(win, tmp_path, "second.dat")
    assert panel.add_manual_event(second, 0, *_MISSED).ok
    _regain(win, first)
    assert panel.current_target() == (second, 0) and panel._stale.isHidden()

    _export(win, rig, "manual")

    assert rig.dest.exists(), rig.status
    frame = pd.read_csv(rig.dest)
    assert list(frame["recording"]) == ["second.dat"]
    row = frame.iloc[0]
    assert row["cal_gain"] == pytest.approx(1.0)
    assert row["baseline"] == pytest.approx(I0, abs=10.0)
    assert row["amplitude"] == pytest.approx(DEPTH, rel=0.1)
    assert any("Left out: Manual of first.dat (on the old calibration)." in m
               for m in rig.status), rig.status


def test_after_the_amber_lines_rebuild_the_rows_are_written_on_the_new_gain(win, rig, qtbot,
                                                                            tmp_path):
    """The way forward, followed. A Schmitt table and a Manual table on one channel, the gain
    corrected, the Manual table rebuilt from its amber line: "Every table on this channel"
    writes the Manual event with every number on the new gain and names Schmitt, still on the
    old one, as left out; "This table" then writes too."""
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    sel = panel._result_selector
    sel.setCurrentIndex(sel.findData(MANUAL_MODULE_ID))
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok
    _regain(win, doc_id)
    assert panel._stale.kind == "calibration" and panel._stale.action_text() == "Rebuild"
    with qtbot.waitSignal(win._session.resultsChanged, timeout=30_000):
        panel._stale._btn.click()
    qtbot.wait(20)
    assert panel._stale.isHidden(), "precondition: the Manual table is current again"

    _export(win, rig, "channel")

    assert rig.dest.exists(), rig.status
    frame = pd.read_csv(rig.dest)
    assert list(frame["table"]) == ["Manual"]
    row = frame.iloc[0]
    stored = _table(win, doc_id, MANUAL_MODULE_ID).iloc[0]
    assert row["cal_gain"] == pytest.approx(2.0)
    assert row["baseline"] == pytest.approx(stored["baseline_pA"]) == pytest.approx(2 * I0,
                                                                                    rel=0.02)
    assert row["amplitude"] == pytest.approx(2 * DEPTH, rel=0.1)
    assert row["snr"] > 0 and 0 < row["frac_amplitude"] < 1
    assert any("Left out: Schmitt of door.dat (on the old calibration)." in m
               for m in rig.status), rig.status

    rig.dest.unlink()
    _export(win, rig, "shown")
    assert rig.dest.exists(), rig.status
