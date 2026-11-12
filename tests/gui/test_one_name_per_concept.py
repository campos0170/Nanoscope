"""One name per concept: the strings that still said "Signal Conditioning", "the Plots window" or
"Event plots" now say what the doors say.

The menu item, the Window-menu row and the window title already call the editor window the
**Preprocess workspace**, the analysis page **Distributions** and the waveform page **Event
waveforms** (``analysis_catalog.ANALYSIS_LABELS`` for the last two). Sentences written before those
names settled kept the old ones, so a message could send a user to a "Signal Conditioning" window
that no menu, title or button carries. Each test below reads one such sentence where the user reads
it and checks it against the live label it points at, so the two cannot drift apart again:

* the workspace's window title and its toolbar's name (read aloud by a screen reader);
* the Filter step's tooltip, which says what a click opens -- and a click opens exactly that;
* the status line that names the part of the window that did not follow a recording switch;
* the two refusals under the stale banner's greyed Re-run, which spell the route to the notch list
  with the labels that are on the Preprocess tab, the workspace tab and the stage list;
* the Summary tab's median-depth tooltip.

(``Copy as Python`` and the Fits-tab note are pinned in ``test_signal_conditioning_window.py`` and
``test_s8_rd1_fits_tab_follows_the_shown_table.py``, which already drive those surfaces.)

Stored text is deliberately NOT renamed: "then re-open Plots." (``core.sources``) is carried into saved
Comparison figures, so it stays as it was -- the last test pins that.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.core.analysis_workspace import capture_comparison_figures
from nanoscope.core.document import TraceDocument
from nanoscope.core.sources import PlotSource
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import analysis_panel as AP
from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel, StatsPanel
from nanoscope.gui.signal_conditioning import SignalConditioningWindow
from nanoscope.gui.workflow_status import Stage

FS = 50_000.0
OVER = [60.0 * (i + 1) for i in range(DetectionPanel.NOTCH_MAX_LINES + 1)]


def _doc(tmp_path, name: str, *, rate: float = FS, n: int = 6_000) -> TraceDocument:
    rng = np.random.default_rng(0)
    sig = (np.full(n, -1000.0) + rng.normal(0.0, 8.0, n)).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=int(sig.size),
                         sample_rate_hz=rate, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _load(window, tmp_path, name: str = "a.dat", *, rate: float = FS, select: bool = True) -> str:
    doc = _doc(tmp_path, name, rate=rate)
    did = window._session.add_document(doc)
    window._create_default_layers(window._session.document(did))
    if select:
        window._on_selection(did, 0)
    return did


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 860)
    w.show()
    yield w
    w._analysis_controller.abort()


def _boom(*_args, **_kwargs):
    raise ValueError("injected by the test")


# -- the window and its toolbar ----------------------------------------------------------------

def test_the_bare_workspace_window_is_titled_by_its_one_name(qapp, qtbot):
    ws = SignalConditioningWindow()     # no hosted pages: the case that used to say "Signal Conditioning"
    qtbot.addWidget(ws)
    assert ws.windowTitle() == "Preprocess workspace"


def test_the_workspace_toolbar_carries_the_one_name(qapp, qtbot):
    ws = SignalConditioningWindow()
    qtbot.addWidget(ws)
    assert ws.toolbar.windowTitle() == "Preprocess workspace"   # a screen reader announces this


def test_the_workspace_title_names_the_recording_under_the_one_name(qapp, qtbot):
    ws = SignalConditioningWindow()
    qtbot.addWidget(ws)
    ws.set_source_label("pore.csv · current")
    assert ws.windowTitle() == "Preprocess workspace — pore.csv · current"
    ws.set_source_label("")
    assert ws.windowTitle() == "Preprocess workspace"


# -- the Filter step ---------------------------------------------------------------------------

def test_the_filter_step_says_what_a_click_opens_only_when_a_click_opens_it(window, qtbot, tmp_path):
    strip = window._workflow_strip
    # Nothing open: the step is locked and a click only says why, so the tooltip promises no window.
    assert "Opens the Preprocess workspace" not in strip.tooltip(Stage.FILTER)
    _load(window, tmp_path)
    assert strip.tooltip(Stage.FILTER).endswith(
        "before detection. Opens the Preprocess workspace."), strip.tooltip(Stage.FILTER)


def test_a_click_on_the_filter_step_opens_the_workspace_its_tooltip_names(window, qtbot, tmp_path):
    _load(window, tmp_path)
    assert window._signal_conditioning is None, "built on first open"
    window._workflow_strip.stageActivated.emit(Stage.FILTER.value)
    ws = window._signal_conditioning
    assert ws is not None and ws.isVisible()
    assert ws.windowTitle().startswith("Preprocess workspace")
    # ...and the menu item that opens the same window carries the same name.
    assert window._act_signal_conditioning.text().replace("&", "") == "Preprocess workspace…"


# -- the status line after a recording switch --------------------------------------------------

def test_a_switch_the_workspace_cannot_follow_names_it_by_its_one_name(window, qtbot, tmp_path,
                                                                        monkeypatch, capsys):
    _load(window, tmp_path, "a.dat")
    other = _load(window, tmp_path, "b.dat", select=False)
    window._open_preprocess_workspace()
    assert window._signal_conditioning.isVisible()
    monkeypatch.setattr(window, "_push_conditioning_source", _boom)
    window._on_selection(other, 0)
    msg = window.statusBar().currentMessage()
    assert msg.startswith("Switched recording, but Preprocess workspace could not follow."), msg
    assert "Signal conditioning" not in msg
    assert "ValueError" in capsys.readouterr().err    # the traceback still goes to whoever debugs


def test_a_switch_the_event_waveforms_page_cannot_follow_names_it_by_its_one_name(
        window, qtbot, tmp_path, monkeypatch, capsys):
    """The step only runs while no figure workspace hosts the page and the page is on screen (the
    ``_event_plot_window`` alias), so the page is stood in for by a shown widget."""
    _load(window, tmp_path, "a.dat")
    other = _load(window, tmp_path, "b.dat", select=False)
    assert window._figure_workspace is None
    stand_in = QtWidgets.QWidget()
    qtbot.addWidget(stand_in)
    stand_in.show()
    monkeypatch.setattr(window, "_event_plot_window", stand_in)
    monkeypatch.setattr(window, "_refresh_event_plots", _boom)
    window._on_selection(other, 0)
    msg = window.statusBar().currentMessage()
    assert msg.startswith("Switched recording, but Event waveforms could not follow."), msg
    assert "Event plots" not in msg
    assert "ValueError" in capsys.readouterr().err


# -- the refusals under the stale banner's greyed Re-run ---------------------------------------

def test_the_nyquist_refusal_spells_the_route_with_the_labels_on_the_door(window, qtbot, tmp_path):
    _load(window, tmp_path, rate=100.0)     # Nyquist 50 Hz: no line of a 60/120 Hz list is below it
    panel = window._analysis_panel
    det = panel.detection_panel
    why = panel._cutoff_replay_why({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    assert why is not None and "Nyquist" in why, why
    tab = det.mode_tabs.tabText(det.mode_tabs.indexOf(det._preprocess_page))
    door = det.preprocess_open_button.text()
    assert (tab, door) == ("Preprocess", "Open Preprocess workspace…")
    assert f"fix the list in the Preprocess workspace ({tab} tab ▸ {door}, Notch stage) and press Detect." in why, why
    assert "Signal Conditioning" not in why


def test_the_over_cap_refusal_spells_the_route_with_the_labels_inside_the_workspace(
        window, qtbot, tmp_path):
    _load(window, tmp_path)
    window._open_preprocess_workspace()
    ws = window._signal_conditioning
    # Qt writes a literal "&" in a tab title as "&&"; the user reads "Filters & drift".
    tab = ws.workspace_tabs.tabText(ws.workspace_tabs.indexOf(ws.filters_page)).replace("&&", "&")
    assert tab == "Filters & drift"
    assert any("Notch" in ws.stage_list.item(i).text() for i in range(ws.stage_list.count()))
    why = AP.AnalysisPanel._notch_replay_why({"notch_enabled": True, "notch_freqs": OVER})
    assert why is not None and "at most" in why, why
    assert f"fix the list in the Preprocess workspace ({tab} ▸ Notch stage, " in why, why
    assert "Signal Conditioning" not in why
    assert ws.windowTitle().startswith("Preprocess workspace")      # the window it sends the user to


# -- the Summary tab ---------------------------------------------------------------------------

def test_the_median_depth_tooltip_names_distributions_by_its_one_name(qapp, qtbot):
    panel = StatsPanel()
    qtbot.addWidget(panel)
    row = next(i for i, (_label, key) in enumerate(panel._SUMMARY_ROWS) if key == "median_depth_pA")
    table = panel.summary_table
    for tip in (table.verticalHeaderItem(row).toolTip(), table.item(row, 0).toolTip()):
        assert ("disagreed with Distributions (which filters to accepted events by default) "
                "about the same column of the same detection") in tip, tip
        assert "Plots window" not in tip


# -- stored text keeps its words ---------------------------------------------------------------

def _merged_source(key: str, label: str, provenance: dict, shift: float = 0.0) -> PlotSource:
    frame = pd.DataFrame({"event_id": list(range(1, 25)),
                          "dwell_s": [0.001 * (i + 1) for i in range(24)],
                          "depth_pA": [10.0 + shift + i for i in range(24)],
                          "amplitude_pA": [12.0 + i for i in range(24)],
                          "ecd_pA_s": [0.01 * (i + 1) for i in range(24)],
                          "snr": [5.0 + i for i in range(24)],
                          "accepted": [True] * 24})
    return PlotSource(key, label, frame, kind="detection", doc_key=key, provenance=provenance,
                      module_id="detect.schmitt", calibration_fingerprint=(1.0, 0.0),
                      current=(key == "a"))


def test_the_merge_remedy_a_saved_comparison_figure_carries_keeps_its_wording(qapp, qtbot):
    """The one sentence of the family that is NOT renamed. ``source_comparability`` ends its merge
    warning "... then re-open Plots." and a Comparison figure keeps that warning in its snapshot,
    which the project saves; changing the words would change saved bytes, so it takes an owner
    decision about stored text, not a wording pass."""
    page = ComparisonFiguresWindow()
    qtbot.addWidget(page)
    page.set_sources([_merged_source("a", "Recording A", {"merge_gap_s": 0.001}),   # merged while detecting
                      _merged_source("b", "Recording B", {}, 2.0)],               # not merged
                     selected_keys=["a", "b"])
    page.analyze()
    stored = json.dumps(capture_comparison_figures(page).payload, default=str)
    assert "Merge (or un-merge) them all the same way, then re-open Plots." in stored
