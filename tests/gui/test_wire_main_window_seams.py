"""Three seams that were built, reachable in code, and connected to nothing in the window.

Each test asserts the USER-VISIBLE EFFECT of the connection, never the connection: an inert
seam is proved inert by the thing the user cannot get, not by a missing ``connect`` line.

* **W-EXPORTDOOR** — ``MeasurePanel.export_processed_trace`` wrote a real CSV + sidecar and
  had exactly one door: a button on the Manual-measurement tab, behind the Inspector's tab
  bar. ``Analysis ▸ Export processed trace…`` is the browsable one, gated on the panel's own
  sentence so a greyed item says which of its two conditions is missing.
* **W-EXPORTDISCLOSE** — ``write_events_export`` grew a keyword-only ``n_total`` so the
  sidecar's ``curation.n_withheld`` is the EXACT number of rows the export left behind.
  ``_export_events`` did not pass it, so the sidecar fell back to ``provenance['n_events']``
  — and a provenance a merge or a hand-add has left stale made that block either lie
  (``n_withheld: 0`` over a curated table) or fall silent (``n_withheld: null``).
* **LAY-W1a** — between a 1253 px and a 1336 px window the Inspector opened at the CEILING
  (460-501 px), under the 502 px its own pages need: it took a horizontal scrollbar anyway
  AND spent up to 41 px of the trace on the difference, while the ceiling's docstring
  claimed both things could be had down to 1252. The measured boundary is 1337.

Measured on the owner's display (1512x900) with the suite's redirected QSettings store, and
on synthetic recordings under ``tmp_path`` — nothing here touches a real recording.
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
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.conditioning import conditioned_trace
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui import measure_panel as mp_mod
from nanoscope.gui.main_window import (
    _DOCK_SEPARATOR_WIDTH,
    _FILE_DOCK_DEFAULT_WIDTH,
    _INSPECTOR_CHROME_WIDTH,
    _INSPECTOR_DEFAULT_WIDTH,
    _INSPECTOR_NESTED_CHROME_WIDTH,
    MainWindow,
)

FS = 50_000.0
N = 200_000
I0 = -1000.0
DEPTH = 300.0
SPANS = [(10_000, 10_600), (20_000, 20_800), (30_000, 30_500), (40_000, 40_700),
         (60_000, 60_900), (70_000, 70_400)]
WINDOW_SIZE = (1512, 900)

# ================================================================================================
# builders
# ================================================================================================
def _signal(seed: int = 0) -> np.ndarray:
    """An open pore at -1000 pA with hum and drift, blocked over ``SPANS``."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / FS
    x = (np.full(N, I0) + rng.normal(0.0, 8.0, N)
         + 4.0 * np.sin(2 * np.pi * 60.0 * t) + 6.0 * np.sin(2 * np.pi * 0.4 * t))
    for s, e in SPANS:
        x[s:e] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str = "apo.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    signal = _signal()
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


def _detect(window: MainWindow, doc_id: str, doc: TraceDocument) -> EventSet:
    df, prov = detect_events(np.asarray(doc.channels[0]), FS, baseline=I0,
                             min_dwell_s=1e-4, recording=doc.meta.path.name,
                             pore_id="pore-1")
    events = EventSet(df, prov)
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        payload=events))
    window._sync_export_action()
    return events


def _analysis_menu(window: MainWindow) -> QtWidgets.QMenu:
    menus = [m for m in window.menuBar().findChildren(QtWidgets.QMenu)
             if m.title().replace("&", "") == "Analysis"]
    assert menus, "no Analysis menu"
    return menus[0]


def _processed_action(window: MainWindow):
    """The menu item this round exists to add — found the way a user finds it: by reading
    the Analysis menu, not by knowing an attribute name."""
    hits = [a for a in _analysis_menu(window).actions()
            if "processed trace" in a.text().replace("&", "").lower()]
    assert len(hits) == 1, (
        "Analysis offers no 'Export processed trace…' item, so the conditioned samples "
        "have no browsable door: "
        + ", ".join(a.text() for a in _analysis_menu(window).actions() if a.text()))
    return hits[0]


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(*WINDOW_SIZE)
    w.show()
    QtWidgets.QApplication.processEvents()
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# W-EXPORTDOOR — the processed trace gets a door, and the door says when it is shut
# ================================================================================================
def test_the_processed_trace_export_is_offered_by_the_analysis_menu(window):
    """It exists at all. Before this round the only way to it was a button on a tab."""
    act = _processed_action(window)
    assert "conditioned samples" in act.toolTip()


def test_the_shut_door_names_which_of_its_two_conditions_is_missing(window, tmp_path):
    """A greyed menu item with no reason is the failure this menu bar already fixed once.

    Both gates, in the order a user meets them: nothing open, then a recording open with
    every Preprocess stage off. The sentence is the panel's own, so the menu and the panel
    cannot describe the same refusal differently."""
    act = _processed_action(window)
    _analysis_menu(window).aboutToShow.emit()
    assert not act.isEnabled()
    assert act.statusTip() == "Open a recording and select a channel first."

    doc = _doc(tmp_path)
    _install(window, doc)
    _analysis_menu(window).aboutToShow.emit()
    assert not act.isEnabled(), "offered with no filter stage enabled — it would write raw"
    assert "No filter stage is enabled on the Preprocess tab" in act.statusTip()
    assert window._analysis_panel.measure_panel.processed_export_gate() == act.statusTip()


def _tick_low_pass(window: MainWindow) -> None:
    """Tick *Low-pass ▸ enable* on the Preprocess form, the way a user does."""
    window._analysis_panel._detection.filter_enabled.setChecked(True)
    QtWidgets.QApplication.processEvents()


def test_a_chain_on_the_preprocess_form_opens_the_door(window, tmp_path):
    """The gate the user actually moves: ticking a stage on the PREPROCESS FORM makes the
    item live, without a selection change and without a run. The chain is read back off
    the form when the menu opens, so the item cannot describe a chain that is no longer on."""
    act = _processed_action(window)
    doc = _doc(tmp_path)
    _install(window, doc)
    _tick_low_pass(window)

    _analysis_menu(window).aboutToShow.emit()      # what opening the menu does
    assert act.isEnabled(), (
        "a stage is enabled on the Preprocess form and the menu item is still grey — its "
        "gate never reaches the form, so the door cannot be opened from the app at all")
    assert act.statusTip() == ""                   # the tooltip describes the write again

    window._analysis_panel._detection.filter_enabled.setChecked(False)
    _analysis_menu(window).aboutToShow.emit()
    assert not act.isEnabled(), "the door stayed open after the chain was switched off"


def test_the_menu_item_writes_the_conditioned_samples_and_says_where(
        window, tmp_path, monkeypatch):
    """THE EFFECT. Two files land, they hold the conditioned array rather than the raw one,
    and the window — not only the panel's own status label, which lives behind a tab —
    names both. Nothing in the Inspector moves to tell the user so."""
    doc = _doc(tmp_path)
    _install(window, doc)
    _tick_low_pass(window)
    dest = tmp_path / "processed.csv"
    monkeypatch.setattr(mp_mod, "ask_export_path", lambda *a, **k: dest)

    act = _processed_action(window)
    _analysis_menu(window).aboutToShow.emit()
    tab_before = window._inspector_tabs.currentIndex()
    act.trigger()

    assert dest.exists(), "the menu item is connected to nothing — no file was written"
    frame = pd.read_csv(dest)
    assert len(frame) == N
    chain = window._analysis_panel.measure_panel.conditioning_params()
    assert chain["filter_cutoff_hz"] == 10_000.0, chain
    expected = conditioned_trace(np.asarray(doc.channels[0]), FS, chain)
    current = frame[[c for c in frame.columns if c != "time_s"][0]].to_numpy()
    assert np.allclose(current, expected, atol=1e-3), "the CSV is not the conditioned array"
    assert not np.allclose(current, np.asarray(doc.channels[0]), atol=1e-3), (
        "the CSV is the raw channel — the chain was not applied")

    side = json.loads(dest.with_suffix(".meta.json").read_text())
    assert "bessel" in json.dumps(side), "the sidecar does not carry the chain"

    said = window.statusBar().currentMessage()
    assert dest.name in said and dest.with_suffix(".meta.json").name in said, said
    assert window._inspector_tabs.currentIndex() == tab_before, (
        "the export moved the Inspector under the pointer to report itself")


# ================================================================================================
# W-EXPORTDISCLOSE — the sidecar counts the withheld rows instead of inferring them
# ================================================================================================
def _export_events(window, dest, monkeypatch) -> dict:
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: dest)
    assert window._act_export_events.isEnabled()
    window._act_export_events.trigger()
    return json.loads(dest.with_suffix(".meta.json").read_text())


def test_a_stale_provenance_can_no_longer_make_the_sidecar_claim_nothing_was_withheld(
        window, tmp_path, monkeypatch):
    """The lie. ``n_events`` written before curation equals the ACCEPTED count, so the
    fallback made ``n_withheld: 0`` — an affirmative statement that the curated table is
    the whole table."""
    doc = _doc(tmp_path)
    doc_id, _ = _install(window, doc)
    events = _detect(window, doc_id, doc)
    events.df.loc[events.df.index[:2], "accepted"] = False
    n_total, n_accepted = len(events.df), int(events.df["accepted"].sum())
    assert n_total - n_accepted == 2
    events.provenance["n_events"] = n_accepted          # what a stale reconcile leaves

    meta = _export_events(window, tmp_path / "events.csv", monkeypatch)
    block = meta["curation"]
    assert block["n_withheld"] == 2, (
        "the sidecar took the detection's own stale count over the frame in hand")
    assert block["n_in_table"] == n_total
    assert block["counted_from"] == "caller"
    assert len(pd.read_csv(tmp_path / "events.csv")) == n_accepted


def test_a_provenance_smaller_than_the_export_no_longer_silences_the_disclosure(
        window, tmp_path, monkeypatch):
    """The silence. A provenance count BELOW the exported row count (a hand-add the
    detector never saw) is skipped as stale, and with no other candidate the block read
    ``n_withheld: null`` — "this file does not know" — over a table that does."""
    doc = _doc(tmp_path)
    doc_id, _ = _install(window, doc)
    events = _detect(window, doc_id, doc)
    events.df.loc[events.df.index[:2], "accepted"] = False
    n_total = len(events.df)
    events.provenance["n_events"] = n_total - 5

    block = _export_events(window, tmp_path / "events.csv", monkeypatch)["curation"]
    assert block["n_withheld"] == 2
    assert block["counted_from"] == "caller"
    assert "withheld" in block["note"].lower()


# ================================================================================================
# LAY-W1a — one answer below the boundary, and the boundary is the measured one
# ================================================================================================
def _pages_minimum(window: MainWindow) -> int:
    """The width the Inspector needs to hide nothing — the same arithmetic the window uses."""
    widest = max(area.widget().minimumSizeHint().width()
                 for area in window._inspector_tabs.findChildren(QtWidgets.QScrollArea)
                 if area.widget() is not None)
    return widest + _INSPECTOR_CHROME_WIDTH + _INSPECTOR_NESTED_CHROME_WIDTH


def _measured_boundary(window: MainWindow) -> int:
    """The narrowest window whose ceiling can give the Inspector the width its pages need."""
    wanted = _pages_minimum(window)
    return next(w for w in range(900, 2000)
                if window._inspector_width_ceiling(w) >= wanted)


def _launch_at(window: MainWindow, width: int) -> tuple[int, int]:
    """A first launch at ``width``: (Inspector width, viewer width)."""
    window.resize(width, 900)
    QtWidgets.QApplication.processEvents()
    window._apply_default_dock_widths()
    QtWidgets.QApplication.processEvents()
    return window._inspector.width(), window.centralWidget().width()


def test_the_boundary_is_1337_and_the_docstring_says_so(window):
    """Measured, then compared with the sentence a maintainer reads. The docstring said
    1252 — the width at which the ceiling drops below the FLOOR, which is a different
    number about a different clamp, and 84 px of window width lay between them."""
    boundary = _measured_boundary(window)
    assert boundary == 1337, f"measured boundary moved to {boundary}"
    doc = MainWindow._inspector_width_ceiling.__doc__
    assert f"BELOW {boundary} px" in doc, (
        "the docstring does not state the measured boundary as its claim")
    # The defective CLAIM, as it was written. (The paragraph may still quote the old number
    # while explaining what was wrong with it — that is a correction, not a claim.)
    assert "Below roughly 1252 px" not in doc, (
        "the docstring still names the floor crossover as the boundary of the rule")


def test_below_the_boundary_the_inspector_opens_at_the_floor_not_at_the_ceiling(window):
    """THE BAND. At 1280, 1300 and 1336 the dock opened at 473 / 483 / 501 — under the
    502 its pages need, so the scrollbar came anyway — and the difference came off the
    trace. One answer across the whole band now: the designed floor, with the scrollbar,
    and every unusable pixel handed back to the viewer."""
    wanted = _pages_minimum(window)
    for width in (1253, 1280, 1300, 1336):
        dock, viewer = _launch_at(window, width)
        assert dock == _INSPECTOR_DEFAULT_WIDTH, (
            f"at {width} px the Inspector opened at {dock}: neither the width its pages "
            f"need ({wanted}) nor the floor it falls back to ({_INSPECTOR_DEFAULT_WIDTH})")
        assert viewer > dock, f"at {width} px the viewer ({viewer}) is not the widest column"
        assert viewer == width - 2 * _DOCK_SEPARATOR_WIDTH - _FILE_DOCK_DEFAULT_WIDTH - dock


def test_at_and_above_the_boundary_the_pages_fit_and_the_viewer_still_leads(window):
    """The other side of the same line — unchanged by this fix, asserted so a future
    "simplification" of the clamp cannot quietly take the top half with the bottom."""
    wanted = _pages_minimum(window)
    for width in (_measured_boundary(window), 1440, 1512):
        dock, viewer = _launch_at(window, width)
        assert dock == wanted, f"at {width} px the Inspector opened at {dock}, not {wanted}"
        assert viewer > dock


def test_the_window_never_re_imposes_the_opening_width_on_a_resize(window):
    """Nothing moves under the pointer. The fix above is a change to the width the dock
    OPENS at; a window dragged across the boundary keeps the separator the user last saw."""
    dock, _ = _launch_at(window, 1512)
    for width in (1400, 1336, 1300, 1253):
        window.resize(width, 900)
        QtWidgets.QApplication.processEvents()
        assert window._inspector.width() == dock, (
            f"the Inspector re-flowed itself to {window._inspector.width()} px when the "
            f"window was resized to {width}")
