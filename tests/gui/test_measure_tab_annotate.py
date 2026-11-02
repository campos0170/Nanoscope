"""The Measure tab's share of manual annotation: the ✎ Annotate button and the key legend at the
top, Clear mask keeping Measure on (as X does), a status line that does not outlive its mask, and
Ctrl+Shift+M on Analysis ▸ Measure event. Plus the guard that the trace modules stay free of
Python event filters (one raised the PySide segfault rate about fourfold on 2026-11-01).
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui

from .test_trace_annotate_mode import (  # noqa: F401  (rig and free_mouse are fixtures)
    EVENT, free_mouse, rig, rows)

_GUI = Path(__file__).resolve().parents[2] / "nanoscope" / "gui"


def test_measure_tab_has_the_annotate_button_and_key_legend(rig):
    win, _ = rig
    panel = win._analysis_panel.measure_panel
    button = panel.annotate_button()
    assert not button.isHidden()
    assert button.defaultAction() is win._act_annotate
    assert button.text() == "✎ Annotate (E)"
    legend = panel._key_legend.text()
    for words in ("drag across an event", "double-click", "Shift at release keeps it",
                  "C catch", "D add", "X discard", "right-click for more"):
        assert words in legend
    # Right under the tab's title, the first control on it, with the legend under it.
    layout = panel.layout()
    assert (layout.indexOf(button), layout.indexOf(panel._key_legend)) == (1, 2)


def test_measure_event_has_ctrl_shift_m(rig):
    win, _ = rig
    assert win._act_measure.shortcut().toString() == "Ctrl+Shift+M"
    assert win._act_add_event.shortcut().toString() == "Ctrl+M"          # the add keeps its key
    taken = [a.text() for a in win.findChildren(QtGui.QAction)
             if a is not win._act_measure and a.shortcut().toString() == "Ctrl+Shift+M"]
    assert taken == []
    win._act_measure.trigger()
    assert win._analysis_panel.measure_panel.is_active()


def test_clear_mask_keeps_measure_and_annotate_on(rig):
    win, doc_id = rig
    panel = win._analysis_panel.measure_panel
    win._trace_keys.set_annotate(True)
    win._trace_view.pointer_sample = lambda: (doc_id, 0, 20_200.0, 10.0)
    win._trace_keys.catch()
    assert win._trace_view.measure_bounds() is not None
    assert panel.clear_mask() is True
    assert win._trace_view.measure_empty()
    assert panel.is_active() and win._trace_keys.annotate() and win._act_annotate.isChecked()
    assert rows(win, doc_id) == 0
    assert panel.clear_mask() is False                                    # nothing left to clear
    assert "no live annotation mask" in panel._status.text()


def test_the_measure_tab_status_line_does_not_outlive_its_mask(rig):
    win, doc_id = rig
    panel = win._analysis_panel.measure_panel
    tv = win._trace_view
    panel._on_trace_measure_placement_requested(tv.x_for_index(doc_id, 0, 30_000))
    assert "Placed a Measure mask at the double-click" in panel._status.text()
    tv.pointer_sample = lambda: (doc_id, 0, 20_200.0, 10.0)
    win._trace_keys.catch()                                               # a mask placed another way
    win._trace_keys.add()
    assert rows(win, doc_id) == 1
    assert panel._status.text() == ""


def test_no_event_filter_in_the_trace_modules():
    for name in ("trace_view.py", "trace_keys.py", "measure_panel.py", "pan_scrollbar.py"):
        text = (_GUI / name).read_text(encoding="utf-8")
        assert "installEventFilter" not in text, name
        assert "def eventFilter" not in text, name


def test_the_placed_sentence_goes_when_the_mask_is_resized(rig):
    """The other half of the stale-status fix: a mask that SETTLES somewhere new (an edge
    tab dragged, a resize) drops the sentence about where it was placed."""
    win, doc_id = rig
    panel = win._analysis_panel.measure_panel
    tv = win._trace_view
    panel._on_trace_measure_placement_requested(tv.x_for_index(doc_id, 0, 30_000))
    assert "Placed a Measure mask at the double-click" in panel._status.text()
    start, stop = tv.measure_bounds()
    tv.set_measure_region(tv.x_for_index(doc_id, 0, start),
                          tv.x_for_index(doc_id, 0, stop + (stop - start) // 2))
    assert tv.measure_bounds()[1] > stop, "premise: the mask was resized"
    assert panel._status.text() == ""


def test_add_with_no_mask_points_to_annotate_mode(rig):
    win, doc_id = rig
    panel = win._analysis_panel.measure_panel
    win._act_measure.setChecked(True)                                     # Measure on, no mask
    if not win._trace_view.measure_empty():
        assert panel.clear_mask() is True
    assert win._trace_view.measure_empty()
    win._trace_keys.add()
    assert "drag in Annotate mode (E)" in win.statusBar().currentMessage()
