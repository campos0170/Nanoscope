"""CR5-01, at the shipped surface — the next export in a folder with crash debris SAYS so.

The core seam (tests/core/test_s5_persistence_orphan_sweep.py) proves the sweep removes a
dead-pid staging orphan and queues the sentence; this proves the sentence reaches the
USER: Analysis ▸ Export events into that folder ends in the same identity-notes dialog
that already discloses a broken hard link, now carrying the sweep's sentence. One channel
for everything a write wants to say about files (the EXP3-02 seam), so no surface had to
learn a second one.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.atomic_write import take_identity_notes
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0


def _signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600)]:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _dead_pid() -> int:
    p = subprocess.Popen(["/usr/bin/true"])
    p.wait()
    return p.pid


def test_the_next_export_removes_the_orphan_and_shows_the_sentence(qapp, qtbot,
                                                                   tmp_path, monkeypatch):
    take_identity_notes()   # a process-global channel starts each test drained
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic",
                         n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4},
        calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    w._on_selection(doc_id, 0)

    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    orphan = export_dir / f".events.csv.saving-{_dead_pid()}-deadbeef.csv"
    orphan.write_bytes(b"partial bytes from a killed NanoScope" * 100)

    monkeypatch.setattr("nanoscope.gui.main_window.ask_export_path",
                        lambda *a, **k: export_dir / "events.csv")
    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k: dialogs.append((title, text))))

    w._export_events()

    assert (export_dir / "events.csv").exists(), "the export itself must land"
    assert not orphan.exists(), "the crash orphan is gone after the next write there"
    said = " ".join(text for _t, text in dialogs)
    assert "leftover staging file" in said and "removed" in said, (
        f"the removal must reach the user through the export's own dialog: {dialogs!r}")

    try:
        w._analysis_controller.abort()
    except Exception:
        pass
