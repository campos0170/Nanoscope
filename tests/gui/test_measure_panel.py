"""Tests for the manual event-measurement panel (PORT_PLAN Measure, Phase 3, offscreen).

Driven through the real app path — a :class:`MainWindow`, its :class:`AnalysisPanel`,
the owned :class:`MeasurePanel`, and the ONE shared :class:`TraceView` — exactly as a
click would. The panel is the GUI face of the pure ``measure_selection`` core (Phase 1)
on top of the LinearRegionItem cursor tool (Phase 2); these tests assert the wiring:

  - enabling Measure + moving the region fills the LIVE readout, kurtosis included;
  - choosing a different amplitude method recomputes the amplitude live;
  - Add-to-list grows the measurements table (running ids), a row can be removed;
  - Export CSV writes the schema header + the measurement rows;
  - the auto-detected events export writes a CSV (honouring 'accepted');
  - the Analysis ▸ Measure event menu toggle drives the region, and a selection
    change clears it.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

from nanoscope.analysis.conditioning import SIGNAL_BASIS_COLUMN
from nanoscope.analysis.measure import MEASUREMENT_SCHEMA
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
_BLOCKADE = (40_000, 60_000)                 # one clean square: depth 30, dwell 0.2 s
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


# ================================================================================================
# builders
# ================================================================================================

def _blockade_trace(seed: int = 1) -> np.ndarray:
    """~100 pA open pore + 1.5 pA noise with ONE 30 pA-deep square over [40k, 60k)."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    x[_BLOCKADE[0]:_BLOCKADE[1]] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _events_trace(seed: int = 0) -> np.ndarray:
    """The five-event trace (matches the integration fixture) so detection has work."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, arr: np.ndarray, *, name: str, fs: float | None = FS) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name,
        format_name="synthetic",
        n_samples=arr.size,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,),
    )
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Add + layer + select a document's current channel — the real selection path."""
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _enable_measure(window: MainWindow, doc_id: str, i0: int, i1: int):
    """Turn the Measure tool on and drop the cursors on sample span [i0, i1)."""
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[i0]), float(vals[i1]))
    return mp


def _regions(plot: pg.PlotItem) -> list[pg.LinearRegionItem]:
    return [it for it in plot.items if isinstance(it, pg.LinearRegionItem)]


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# live readout + kurtosis
# ================================================================================================

def test_enable_measure_populates_live_readout_including_kurtosis(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)

    stats = mp.current_measurement()
    assert stats is not None, "moving the region must produce a live measurement"

    # KURTOSIS is present, finite, and clearly shown in the readout.
    assert "kurtosis" in stats and np.isfinite(stats["kurtosis"])
    assert mp.readout_value("kurtosis") not in ("", "—")

    # The headline numbers filled and carry the channel units.
    assert mp.readout_value("dwell_s") not in ("", "—")
    assert "pA" in mp.readout_value("amplitude")

    # And they are physically right for this square: 30 pA deep, 100 pA baseline, 0.2 s.
    assert stats["amplitude"] == pytest.approx(30.0, abs=2.0)
    assert stats["baseline"] == pytest.approx(100.0, abs=2.0)
    assert stats["dwell_s"] == pytest.approx(0.2, rel=1e-3)
    assert stats["n_samples"] == _BLOCKADE[1] - _BLOCKADE[0]


def test_readout_updates_live_as_the_region_moves(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    n_first = mp.current_measurement()["n_samples"]

    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[45_000]), float(vals[55_000]))
    assert mp.current_measurement()["n_samples"] == 10_000 != n_first


# ================================================================================================
# amplitude method
# ================================================================================================

def test_amplitude_method_changes_amplitude(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)

    mp.set_method("median")
    amp_median = mp.current_measurement()["amplitude"]
    mp.set_method("min")
    amp_min = mp.current_measurement()["amplitude"]

    assert amp_min != pytest.approx(amp_median)
    # 'min' takes the deepest sample, a strictly larger blockade amplitude on noisy data.
    assert amp_min > amp_median
    assert "pA" in mp.readout_value("amplitude")


# ================================================================================================
# the saved table + CSV
# ================================================================================================

def test_add_to_list_grows_table_with_running_ids(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    assert mp._model.rowCount() == 0

    assert mp.add_current() is True
    assert mp.add_current() is True
    assert mp._model.rowCount() == 2

    df = mp.measurements_df
    assert list(df["measurement_id"]) == [0, 1]          # running, not reused
    assert df["channel"].iloc[0] == "current"
    assert df["recording"].iloc[0] == "b.dat"
    assert np.isfinite(df["kurtosis"].iloc[0])

    # A row can be removed.
    mp._table_view.selectRow(0)
    mp.remove_selected()
    assert mp._model.rowCount() == 1


def test_export_measurements_csv_writes_header_and_row(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    mp.add_current()

    out = mp.export_csv(str(tmp_path / "measurements.csv"))
    assert out is not None and out.exists()

    df = pd.read_csv(out)
    # The full manual schema header — plus the additive `signal_basis` stamp, which the
    # export gained in the 2026-09-02 fix-up (PC-C2: the GUI could choose the basis and
    # the CSV then dropped it). The 32 schema columns are unchanged and still first; see
    # tests/gui/test_pc2_measure_export_contract.py for the additive contract itself.
    assert list(df.columns) == [*MEASUREMENT_SCHEMA, SIGNAL_BASIS_COLUMN]
    assert len(df) == 1
    assert np.isfinite(df["kurtosis"].iloc[0])
    assert df["amplitude"].iloc[0] == pytest.approx(30.0, abs=2.0)
    assert df["amplitude_method"].iloc[0] == "median"


def test_export_measurements_empty_writes_nothing(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = window._analysis_panel.measure_panel
    # Nothing added yet: export is a no-op, no file, no crash.
    assert mp.export_csv(str(tmp_path / "empty.csv")) is None
    assert not (tmp_path / "empty.csv").exists()


# ================================================================================================
# the auto-detected events export
# ================================================================================================

def test_export_events_writes_csv(window, qtbot, tmp_path, monkeypatch):
    doc_id, channel = _install(window, _doc(tmp_path, _events_trace(), name="ev.dat"))

    # Export is disabled until a detection exists.
    assert not window._act_export_events.isEnabled()

    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)
    assert window._act_export_events.isEnabled()

    dest = tmp_path / "events.csv"
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(dest), "")))
    window._export_events()

    assert dest.exists()
    df = pd.read_csv(dest)
    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    n_accepted = int(res.payload.df["accepted"].to_numpy(dtype=bool).sum())
    assert n_accepted >= 1
    assert len(df) == n_accepted                          # honours 'accepted'
    assert "dwell_s" in df.columns and "depth_pA" in df.columns


# ================================================================================================
# menu toggle + selection clearing
# ================================================================================================

def test_measure_menu_action_toggles_region(window, tmp_path):
    doc_id, channel = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = window._analysis_panel.measure_panel
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert window._act_measure.isEnabled()
    assert _regions(pane.plot) == []

    window._act_measure.setChecked(True)                  # menu drives the panel
    assert mp.is_active()
    assert len(_regions(pane.plot)) == 1

    window._act_measure.setChecked(False)
    assert not mp.is_active()
    assert _regions(pane.plot) == []


def test_panel_toggle_reflects_onto_menu_check(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = window._analysis_panel.measure_panel

    mp.set_active(True)                                   # panel drives the menu check
    assert window._act_measure.isChecked()
    mp.set_active(False)
    assert not window._act_measure.isChecked()


def test_selection_change_clears_measure_region(window, tmp_path):
    doc_a = _doc(tmp_path, _blockade_trace(), name="A.dat")
    id_a, _ch_a = _install(window, doc_a)
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    assert mp._trace_view.measure_bounds() is not None

    # Selecting a different document's current channel retargets the panel, which drops
    # the region (it lived on the previous channel's pane) and disarms the menu check.
    id_b, ch_b = _install(window, _doc(tmp_path, _blockade_trace(), name="B.dat"))
    assert not mp.is_active()
    assert mp._trace_view.measure_bounds() is None
    assert not window._act_measure.isChecked()


def test_measure_disabled_without_a_selection(window, tmp_path):
    mp = window._analysis_panel.measure_panel
    assert not mp.can_measure()
    assert not window._act_measure.isEnabled()

    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    assert mp.can_measure()
    assert window._act_measure.isEnabled()

    # Closing the document turns it back off.
    window._session.close_document(doc_id)
    assert not mp.can_measure()
    assert not window._act_measure.isEnabled()


def test_deleting_hand_measurements_asks_first_and_cancel_deletes_nothing(
        window, tmp_path, monkeypatch):
    """MEASURED BEFORE, through a real MainWindow: three hand measurements, ⌘A in the
    table, Remove -> rows 3 -> 0, ``modals shown: []``, and Edit ▸ Undo still reading
    ``'Undo Curate 2 events'`` with ``isEnabled()=True``.

    This list is not an ``EventSet``, so ``core.history`` never sees it: the delete is the
    one destructive click in the application that undo genuinely cannot reach, and reaching
    for Ctrl+Z after it rolls back a DIFFERENT piece of work. Until the history covers the
    measurement frame the guarantee has to be a confirmation that says so.
    """
    doc_id, _ch = _install(window, _doc(tmp_path, _blockade_trace(), name="b.dat"))
    mp = _enable_measure(window, doc_id, *_BLOCKADE)
    for _ in range(3):
        assert mp.add_current() is True
    assert mp._model.rowCount() == 3

    asked: list[int] = []
    monkeypatch.setattr(type(mp), "_confirm_remove",
                        lambda self, n: (asked.append(n), False)[1])
    mp._table_view.selectAll()
    mp.remove_selected()
    assert asked == [3], "the delete must ask, and must name how many rows are at stake"
    assert mp._model.rowCount() == 3, "Cancel deletes nothing"

    monkeypatch.setattr(type(mp), "_confirm_remove", lambda self, n: True)
    mp.remove_selected()
    assert mp._model.rowCount() == 0
