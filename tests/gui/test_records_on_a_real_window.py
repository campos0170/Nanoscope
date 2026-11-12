"""The two Figure Library defects, on the real main window with a real detection.

* Analysis > Export > Edit standard report figure: Save to Figure Library and Export bundle
  answered "Out of range float values are not JSON compliant" for all four figures. A detection
  on a clean trace records ``polarity_evidence.ratio = n_down / 0 = inf``, so this happens to every
  recording without upward events.
* Analysis Explorer > Event waveforms: Snapshot to Figures, Export bundle and the batch export
  answered "The frozen figure exceeds the 512 MiB storage limit", because the figure's data table
  repeated the detection's 58 KB provenance on every row.

Nothing here is injected: the trace is synthetic, but the detection, its provenance, the editors,
the Explorer and the saves are the application's own.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import math
import zipfile

import numpy as np
import pytest

from nanoscope.core.analysis_workspace import _decode
from nanoscope.core.document import TraceDocument
from nanoscope.core.figure_library import figures as saved_figures
from nanoscope.core.project import load_project
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_plot_panel import ENSEMBLE
from nanoscope.gui.figure_bundle import MAX_SNAPSHOT_BYTES, FigureSnapshot
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N_EVENTS = 60


def _trace(n_events: int, seed: int = 7, dur_s: float = 4.0) -> np.ndarray:
    """A 250 pA open pore with ``n_events`` blockades of two populations: every one lowers the
    current, so the detection's polarity evidence has no upward events."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    x = 250.0 + rng.normal(0.0, 3.0, n)
    slot = n // (n_events + 1)
    for k in range(n_events):
        second = rng.random() < 0.4
        dwell = rng.lognormal(np.log(3e-3 if second else 2.5e-4), 0.7)
        width = int(np.clip(dwell * FS, 4, slot * 0.7))
        start = (k + 1) * slot + int(rng.integers(0, slot // 5))
        x[start:start + width] -= abs(rng.normal(72.0 if second else 38.0, 6.0 if second else 4.0))
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot, tmp_path):
    win = MainWindow(recovery_enabled=False)
    qtbot.addWidget(win)
    win.resize(1400, 900)
    data = _trace(N_EVENTS)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_120mV.csv", format_name="synthetic",
                         n_samples=data.size, sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,), nominal_voltage_mV=120.0,
                         voltage_provenance=Provenance.FILENAME)
    doc = TraceDocument(meta=meta, channels=(data,))
    key = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(key, 0)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=120_000):
        win._analysis_panel.trigger_detection()
    qtbot.wait(100)
    result = win._analysis_panel.current_detection()
    assert result is not None and len(result.payload.df) >= N_EVENTS - 5
    evidence = result.payload.provenance["polarity_evidence"]
    assert evidence["n_up"] == 0 and evidence["ratio"] == math.inf, "the precondition: no upward events"
    yield win
    win._analysis_controller.abort()
    win._dirty = False


def _strict(raw: bytes):
    def refuse(token):
        raise AssertionError(f"not valid JSON: the bare constant {token}")
    return json.loads(raw, parse_constant=refuse)


def test_all_four_report_figures_save_to_the_library_and_bundle(window, tmp_path):
    for kind in ("dwell", "depth", "scatter", "trace"):
        editor = window._edit_report_figure(kind=kind)
        assert editor is not None, f"the {kind} report figure did not open"
        entry = editor.save_to_library(name=f"Report {kind}")
        assert entry is not None, f"{kind}: {editor._status.text()}"
        bundle = editor.export_bundle(path=tmp_path / f"{kind}.zip")
        assert bundle is not None, f"{kind}: {editor._status.text()}"
        with zipfile.ZipFile(bundle) as archive:
            provenance = _strict(archive.read("provenance.json"))
            _strict(archive.read("manifest.json"))
        evidence = provenance["details"]["plots"][0]["provenance"]["polarity_evidence"]
        restored = _decode(evidence, {}, {})
        assert restored["n_up"] == 0 and restored["ratio"] == math.inf
        editor.close()
    assert [e.payload["name"] for e in window._figure_entries()] == [
        f"Report {kind}" for kind in ("dwell", "depth", "scatter", "trace")]
    # Save Project, reopen: every report figure comes back, picture and numbers exactly.
    path = tmp_path / "reports.nanoscope"
    assert window._write_project(str(path))
    reopened = saved_figures(load_project(path).workspace)
    assert [e.payload["name"] for e in reopened] == [e.payload["name"] for e in window._figure_entries()]
    for before, after in zip(window._figure_entries(), reopened):
        assert (FigureSnapshot.from_storage(after.payload["snapshot"], after.blobs)
                == FigureSnapshot.from_storage(before.payload["snapshot"], before.blobs))


def test_event_waveforms_snapshot_bundle_and_batch_export_succeed(window, qtbot, tmp_path):
    window._open_plots()
    qtbot.wait(200)
    owner = window._figure_workspace
    key = owner.create("event_waveforms")
    window._refresh_figure(key)
    qtbot.waitUntil(lambda: owner.get(key).spec.result.status != "empty", timeout=60_000)
    qtbot.wait(300)
    page = owner.page(key)
    page.set_view(ENSEMBLE)
    qtbot.wait(200)
    table = page._plot_data()
    assert len(table) >= 50 * page._npoints.value()               # the ensemble: events x points rows
    assert table["provenance_json"].ne("").sum() == 1             # the provenance, once
    assert len(table.to_csv(index=False).encode("utf-8")) < 2_000 * len(table)

    entry = window._save_analysis_figure(key, name="Event overlay")
    assert entry is not None, "Snapshot to Figures was refused"
    snapshot = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
    assert snapshot.dataframe()["provenance_json"].ne("").sum() == 1
    assert sum(len(blob) for blob in entry.blobs.values()) < MAX_SNAPSHOT_BYTES // 16

    bundle = window._export_analysis_bundle(key, path=str(tmp_path / "events.zip"))
    assert bundle is not None, "Export bundle was refused"
    with zipfile.ZipFile(bundle) as archive:
        assert archive.getinfo("data.csv").file_size < 16 * 1024 * 1024
        assert "Source provenance" in json.loads(archive.read("provenance.json"))["details"]

    folder = tmp_path / "batch"
    folder.mkdir()
    results = window._export_selected_figures(
        directory=folder, formats=("pdf", "png", "zip"), selected_ids=(key,), saved_ids=(),
        interactive=False, overwrite=True)
    assert [r.status for r in results] == ["written"] * 3, [(r.path.name, r.error) for r in results]
