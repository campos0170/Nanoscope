"""Regression tests for MEAS2-02: the manual table's ``t_start_s`` names its clock.

On a recording whose file carries ``t0_s != 0`` (the owner's real ABFs: 10.2526 s,
61.3034 s, 1154.99 s) the trace axis shows the instrument clock while every saved
``t_start_s`` is on the recording clock — a 19-minute gap on the quick ABF with, until
this fix, nothing anywhere in the Manual measurement panel to explain it. The event
table already honours core/clock.py's rule ("every control that shows or takes a time
says which clock it is in"): header badge ``(t₀=0)``, tooltip from ``clock.field_note``
(pinned by tests/gui/test_clock_invariant.py). The manual table now follows the SAME
convention — one clock discipline — plus a per-row ``t0_s`` column, because this table
pools rows from recordings with different t₀ and a header note alone cannot be right
for all of them. ``t_start_s + t0_s`` is the instant the trace axis showed.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
_BLOCKADE = (40_000, 60_000)
T0_APO = 61.3034                     # the owner's real −700 mV ABF t0


def _doc(tmp_path, *, name: str, t0_s: float) -> TraceDocument:
    x = np.full(N, 100.0, dtype=np.float32)
    x[_BLOCKADE[0]:_BLOCKADE[1]] = 70.0
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         t0_s=t0_s, channels=(info,))
    return TraceDocument(meta=meta, channels=(x,))


def _install_and_measure(window: MainWindow, doc: TraceDocument):
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[_BLOCKADE[0]]),
                                          float(vals[_BLOCKADE[1]]))
    return mp, doc_id


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _header(mp, name: str, role):
    model = mp._model
    return model.headerData(model.column_index(name), Qt.Horizontal, role)


# ================================================================================================
# the header names its clock — the event table's convention, on the manual table
# ================================================================================================

def test_t_start_header_is_badged_and_tooltipped_on_a_t0_recording(window, tmp_path):
    mp, _ = _install_and_measure(window, _doc(tmp_path, name="apo.dat", t0_s=T0_APO))

    hdr = str(_header(mp, "t_start_s", Qt.DisplayRole))
    tip = str(_header(mp, "t_start_s", Qt.ToolTipRole))
    assert "t₀=0" in hdr, "the two clocks differ on screen and the header does not say so"
    # The field_note vocabulary the event table uses: names the clock, spells out the
    # conversion with THIS recording's t0.
    assert "recording" in tip.lower() and "clock" in tip.lower()
    assert f"{T0_APO:.4f}" in tip
    # ...and the pooled-table truth: each row's own t0_s column.
    assert "t0_s" in tip


def test_t_start_header_is_plain_when_the_clocks_agree(window, tmp_path):
    """t0 = 0: the badge would be noise (the two clocks read alike), but the tooltip
    still explains the convention — same condition the event table branches on."""
    mp, _ = _install_and_measure(window, _doc(tmp_path, name="zero.dat", t0_s=0.0))

    assert str(_header(mp, "t_start_s", Qt.DisplayRole)) == "t_start_s"
    tip = str(_header(mp, "t_start_s", Qt.ToolTipRole))
    # field_note's agree-case wording ("t₀ is 0, so ... read the same number") plus the
    # pooled-table sentence naming the per-row t0_s column.
    assert "t₀" in tip and "t0_s" in tip


def test_t0_column_header_explains_itself(window, tmp_path):
    mp, _ = _install_and_measure(window, _doc(tmp_path, name="apo.dat", t0_s=T0_APO))
    tip = str(_header(mp, "t0_s", Qt.ToolTipRole))
    assert "t_start_s + t0_s" in tip
    assert "NaN" in tip


# ================================================================================================
# the row stamps its recording's t0, and the two numbers reconcile with the axis
# ================================================================================================

def test_saved_row_carries_t0_and_reconciles_with_the_trace_axis(window, tmp_path):
    mp, doc_id = _install_and_measure(window, _doc(tmp_path, name="apo.dat", t0_s=T0_APO))
    assert mp.add_current() is True

    row = mp.measurements_df.iloc[-1]
    assert float(row["t0_s"]) == pytest.approx(T0_APO)
    # t_start_s + t0_s is the instant the user read off the trace axis: the arithmetic
    # the tooltip promises must actually hold against Session.x_axis.
    axis_at_start = float(window._session.x_axis(doc_id).values[int(row["start_idx"])])
    assert float(row["t_start_s"]) + float(row["t0_s"]) == pytest.approx(axis_at_start)


def test_rows_from_recordings_with_different_t0_each_keep_their_own(window, tmp_path):
    """The pooled-table case the header note alone could not cover: two recordings,
    two t₀s, one table — each row's t0_s is its own recording's."""
    mp, _ = _install_and_measure(window, _doc(tmp_path, name="apo.dat", t0_s=T0_APO))
    mp.add_current()
    mp2, _ = _install_and_measure(window, _doc(tmp_path, name="zero.dat", t0_s=0.0))
    assert mp2 is mp                     # one panel, one pooled table
    mp.add_current()

    t0s = [float(v) for v in mp.measurements_df["t0_s"]]
    assert t0s == pytest.approx([T0_APO, 0.0])


def test_csv_export_carries_the_t0_column(window, tmp_path):
    mp, _ = _install_and_measure(window, _doc(tmp_path, name="apo.dat", t0_s=T0_APO))
    mp.add_current()
    out = mp.export_csv(str(tmp_path / "m.csv"))
    assert out is not None
    df = pd.read_csv(out)
    assert "t0_s" in df.columns
    assert float(df["t0_s"].iloc[0]) == pytest.approx(T0_APO)
