"""Beta regressions: replayed filter inputs and curation navigation remain faithful."""
import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import filters
from nanoscope.gui.panels import DetectionPanel, EventTable


@pytest.mark.parametrize("lines", [
    [60.00025, 120.0],
    [60.0, 25000.000596046462],
    [np.nextafter(25000.0, 0.0), np.nextafter(25000.0, np.inf)],
])
def test_result_reseed_preserves_exact_notch_lines_and_nyquist_decision(qtbot, lines):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_recording(50000.0)
    panel.notch_use_lines.setChecked(True)
    panel.notch_freqs.setText(", ".join(repr(float(f)) for f in lines))
    requested = panel.params()
    assert requested["notch_freqs"] == lines
    resolved = filters.notch_frequencies(50000.0, freqs=lines)
    for _ in range(3):
        panel.set_params(requested)
        replayed = panel.params()["notch_freqs"]
        assert replayed == lines
        assert filters.notch_frequencies(50000.0, freqs=replayed) == resolved
        for f in lines:
            assert repr(float(f)).removesuffix(".0") in panel.notch_lines_note.text()


def test_notch_note_discloses_dropped_lines_and_updates_for_new_recording(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_notch_lines([60, 30000, 50000])
    panel.set_recording(50000.0)
    note = panel.notch_lines_note.text()
    assert "Will notch 1 line(s): 60 Hz" in note
    assert "Skipping 30000, 50000 Hz at or above Nyquist" in note
    assert "Requested: 60, 30000, 50000 Hz" in note
    assert panel.params()["notch_freqs"] == [60, 30000, 50000]

    panel.set_recording(200000.0)
    assert "3 line(s)" in panel.notch_lines_note.text()
    assert "Skipping" not in panel.notch_lines_note.text()
    panel.set_recording(100.0)
    assert "No usable notch lines; detection will be refused" in panel.notch_lines_note.text()
    panel.set_recording(None)
    assert "Skipping" not in panel.notch_lines_note.text()


@pytest.mark.parametrize("schema_change", [False, True])
def test_table_curation_refresh_does_not_echo_restored_current_event(qtbot, schema_change):
    table = EventTable()
    qtbot.addWidget(table)
    frame = pd.DataFrame({"event_id": [10, 20, 30], "accepted": [True] * 3})
    table.set_events(frame)
    table.select_event(20)
    events = []
    table.eventSelected.connect(events.append)
    if schema_change:
        frame["flag"] = ["", "molecule=ferritin", ""]
    else:
        frame.loc[1, "accepted"] = False
    table.refresh_all()
    assert table.selected_event_ids() == [20]
    assert events == []
    table.select_event(30)
    assert events == [30], "real user navigation must still reach the other views"
