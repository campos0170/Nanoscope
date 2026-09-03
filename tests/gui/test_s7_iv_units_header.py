"""Sweep-7 S7-IV1: the I-V window's Details HEADER names the undeclared-units state.

Measured on the pre-fix tree (docs/sweep7/repros/S7-IV1_units_blank.py): with three
recordings whose current channel declares no units — the state every HDF5/CSV loader
leaves (``raw_units=''``) — ``details_text()`` began "I-V over 3 recording(s), current
in " and stopped, while the plot title, the y-axis and every other line of the same
pane said ``(units unknown)``. One bare ``{units}`` in a file that spells the state
``units or '(units unknown)'`` everywhere else.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.iv_panel import IVWindow


def _unitless_doc(tmp_path: Path, name: str, mV: float, i_open: float,
                  n: int = 4096) -> TraceDocument:
    """A recording as the HDF5/CSV readers leave it: a current channel with NO units
    and a command-voltage channel in mV."""
    cur = np.full(n, i_open, dtype=np.float32)
    volt = np.full(n, mV, dtype=np.float32)
    infos = (ChannelInfo(index=0, name="current", raw_units="", kind=ChannelKind.CURRENT),
             ChannelInfo(index=1, name="command", raw_units="mV", kind=ChannelKind.VOLTAGE))
    meta = RecordingMeta(path=tmp_path / name, format_name="HDF5 flat", n_samples=n,
                         sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META,
                         channels=infos)
    return TraceDocument(meta=meta, channels=(cur, volt))


def test_the_details_header_names_the_undeclared_units_state_like_every_other_line(
        qtbot, tmp_path):
    session = Session()
    for name, mV, i in (("p500.h5", -500.0, -28.0), ("p600.h5", -600.0, -34.8),
                        ("p700.h5", -700.0, -49.1)):
        session.add_document(_unitless_doc(tmp_path, name, mV, i))
    w = IVWindow(session)
    qtbot.addWidget(w)
    w.refresh()

    lines = w.details_text().splitlines()
    assert lines[0] == "I-V over 3 recording(s), current in (units unknown)"
    # No line of the pane may end in a dangling preposition where a unit should be.
    assert not any(line.rstrip().endswith(" in") for line in lines), lines
    # ...and the header agrees with the rest of the pane on the one spelling.
    assert sum("(units unknown)" in line for line in lines) >= 4
