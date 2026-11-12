"""A NaN or an infinity in a figure's provenance must not make the figure unsavable.

MEASURED DEFECT (assessment, run on a real window). Analysis > Export > Edit standard report
figure, any of the four figures, then Save to Figure Library or Export bundle: "Figure was not
added: Out of range float values are not JSON compliant" and "Bundle was not exported: ...".
The value is ``polarity_evidence.ratio`` = n_down / n_up in the detection's provenance, which is
``inf`` for every recording with no upward events (a clean trace: the common case). The frozen
figure writes its provenance with ``json.dumps(..., allow_nan=False)``, which refuses it.

THE FIX writes such a value the way the project's workspace layer already spells it (the tagged
float ``core.analysis_workspace._encode`` writes and ``_decode`` reads), so the provenance stays
strict JSON that any parser reads, one value has one spelling, and the project's own decoder
turns it back into the number. ``allow_nan`` stays off. A NaN in a figure's *configuration* is
still refused: it is a setting that cannot be applied, not a fact about the source.

These tests build the four real report figures with the real builders and open them in the real
editor, through the real ``MainWindow._edit_report_figure``; only the owner (the part of the main
window that holds the project) is a stand-in. Synthetic data only.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import math
import zipfile
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.events import SCHEMA, EventSet
from nanoscope.core import analysis_workspace
from nanoscope.core.analysis_workspace import _decode, _encode, _Resources
from nanoscope.core.document import CalibratedChannel, XAxis
from nanoscope.core.figure_library import figures as saved_figures
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.types import ChannelInfo, ChannelKind
from nanoscope.gui.figure_bundle import FigureSnapshot
from nanoscope.gui.figure_composition import compose_figures
from nanoscope.gui.main_window import MainWindow

KINDS = ("dwell", "depth", "scatter", "trace")
FS = 50_000.0

#: What a detection on a clean trace records: no upward events, so n_down / n_up is infinite.
EVIDENCE = {"n_down": 60, "n_up": 0, "ratio": math.inf, "confidence": math.nan,
            "margin": -math.inf, "verdict": "measured"}


def _spelled(value: float) -> dict:
    """How the project's workspace layer writes this float: the ONE spelling."""
    return _encode(value, _Resources(), "provenance")


def _events(n: int = 60) -> EventSet:
    rng = np.random.default_rng(3)
    dwell = 20e-6 + rng.exponential(200e-6, n)
    depth = np.clip(rng.normal(0.25, 0.05, n), 1e-3, 0.99)
    i0 = -1000.0
    frame = pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "recording": pd.array(["synthetic"] * n, dtype="string"),
        "pore_id": pd.array(["pore0"] * n, dtype="string"),
        "start_idx": (np.arange(n) * 1000).astype(np.int64),
        "end_idx": (np.arange(n) * 1000 + 50).astype(np.int64),
        "t_start_s": np.linspace(0.01, 0.9, n),
        "dwell_s": dwell,
        "baseline_pA": np.full(n, i0),
        "depth_pA": depth * abs(i0),
        "frac_depth": depth,
        "ecd_pA_s": depth * abs(i0) * dwell,
        "max_dev_pA": depth * abs(i0) * 1.1,
        "mean_dev_pA": depth * abs(i0),
        "rise_s": np.full(n, 5e-6),
        "fall_s": np.full(n, 5e-6),
        "snr": depth * abs(i0) / 15.0,
        "n_levels": np.ones(n, dtype=np.int64),
        "filter_limited": np.zeros(n, dtype=bool),
        "accepted": np.ones(n, dtype=bool),
        "flag": pd.array([""] * n, dtype="string"),
    })
    assert list(frame.columns) == list(SCHEMA)
    return EventSet(frame, {"analyzed_duration_s": 1.0, "min_dwell_s": 20e-6,
                            "recording": "synthetic", "polarity_evidence": dict(EVIDENCE)})


def _trace():
    n = 50_000
    data = np.random.default_rng(1).normal(-1000.0, 15.0, n).astype(np.float32)
    info = ChannelInfo(index=0, name="synthetic", raw_units="pA", kind=ChannelKind.CURRENT)
    channel = CalibratedChannel(info=info, data=np.ascontiguousarray(data), units="pA",
                                sample_rate_hz=FS, t0_s=0.0)
    return channel, XAxis(values=np.arange(n, dtype=np.float64) / FS, unit="s", is_time=True)


class _Owner(QtWidgets.QWidget):
    """The slice of the main window the report launcher and the editor use: the active
    detection, the recording's channel and axis, and the project's list of saved figures."""

    def __init__(self, result, channel, axis):
        super().__init__()
        self.entries = []
        self._figure_project_epoch = 1
        self._analysis_panel = SimpleNamespace(current_detection=lambda: result)
        self._session = SimpleNamespace(has_document=lambda _: True,
                                        calibrated=lambda *_: channel, x_axis=lambda _: axis)
        self._figure_library = SimpleNamespace(refresh=lambda _identity: None)

    def _refuse_stale_export(self, _result):
        return False

    def _ensure_figure_workspace(self):
        return SimpleNamespace(style_registry=lambda: None)

    def _stored_fit(self, *_):
        return None, "not computed"

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda *args: None)

    def _append_figure(self, entry):
        self.entries.append(entry)

    def _mark_dirty(self):
        pass

    def _open_figure_library(self):
        pass


@pytest.fixture
def owner(qtbot, monkeypatch):
    result = SimpleNamespace(payload=_events(), doc_id="recording", channel=0)
    channel, axis = _trace()
    widget = _Owner(result, channel, axis)
    qtbot.addWidget(widget)
    source = {"source_key": "durable-source", "channel": 0, "result_fingerprint": "A"}
    monkeypatch.setattr(analysis_workspace, "result_recipe", lambda *_: deepcopy(source))
    return widget


def _edit(owner, kind):
    editor = MainWindow._edit_report_figure(owner, kind=kind)
    assert editor is not None, f"the {kind} report figure did not open"
    return editor


def _evidence(provenance: dict) -> dict:
    """The detection's polarity evidence inside a saved report figure's provenance record."""
    plots = provenance["details"]["plots"]
    assert plots, "the report figure carries its details"
    return plots[0]["provenance"]["polarity_evidence"]


def _strict(raw: bytes):
    """Parse as a standard JSON reader does: NaN, Infinity and -Infinity are refused."""
    def refuse(token):
        raise AssertionError(f"not valid JSON: the bare constant {token}")
    return json.loads(raw, parse_constant=refuse)


@pytest.mark.parametrize("kind", KINDS)
def test_report_figure_saves_to_the_library_with_infinite_and_nan_provenance(owner, kind):
    editor = _edit(owner, kind)
    entry = editor.save_to_library(name=f"Report {kind}")
    assert entry is not None, editor._status.text()
    assert owner.entries == [entry]
    snapshot = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
    evidence = _evidence(snapshot.provenance())
    # One spelling: the one the project's workspace layer uses, in strict JSON.
    assert evidence["ratio"] == _spelled(math.inf)
    assert evidence["margin"] == _spelled(-math.inf)
    assert evidence["confidence"] == _spelled(math.nan)
    assert evidence["n_down"] == 60 and evidence["verdict"] == "measured"
    # ... and the project's own decoder turns it back into the numbers.
    back = _evidence(_decode(snapshot.provenance(), {}, {}))
    assert back["ratio"] == math.inf and back["margin"] == -math.inf
    assert math.isnan(back["confidence"])
    # The library entry's own copy of the provenance uses the same spelling.
    saved = _evidence(entry.payload["analysis"]["payload"]["provenance"])
    assert saved == evidence
    editor.close()


@pytest.mark.parametrize("kind", KINDS)
def test_report_figure_bundle_is_valid_json_and_reads_back(owner, kind, tmp_path):
    editor = _edit(owner, kind)
    path = editor.export_bundle(path=tmp_path / f"{kind}.zip")
    assert path == tmp_path / f"{kind}.zip", editor._status.text()
    with zipfile.ZipFile(path) as archive:
        provenance = _strict(archive.read("provenance.json"))
        manifest = _strict(archive.read("manifest.json"))
        _strict(archive.read("config.json"))
    evidence = _evidence(provenance)
    assert evidence["ratio"] == _spelled(math.inf)
    assert evidence["margin"] == _spelled(-math.inf)
    assert evidence["confidence"] == _spelled(math.nan)
    assert manifest["format"] == "nanoscope-figure-snapshot"
    editor.close()


def test_saved_report_figure_survives_the_project_round_trip(owner, tmp_path):
    editor = _edit(owner, "dwell")
    entry = editor.save_to_library(name="Report dwell")
    assert entry is not None, editor._status.text()
    path = save_project(tmp_path / "report.nanoscope",
                        ProjectState(workspace=AnalyticalWorkspace(artifacts=[entry])))
    loaded = saved_figures(load_project(path).workspace)
    assert [figure.artifact_id for figure in loaded] == [entry.artifact_id]
    again = FigureSnapshot.from_storage(loaded[0].payload["snapshot"], loaded[0].blobs)
    original = FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs)
    assert again == original
    back = _evidence(_decode(again.provenance(), {}, {}))
    assert back["ratio"] == math.inf and math.isnan(back["confidence"])
    # the library entry's copy reads back too, under the same spelling
    assert _evidence(loaded[0].payload["analysis"]["payload"]["provenance"]) == _evidence(again.provenance())
    editor.close()


def test_saved_report_figures_can_be_composed_and_keep_the_provenance(owner):
    entries = []
    for kind in ("dwell", "depth"):
        editor = _edit(owner, kind)
        entries.append(editor.save_to_library(name=f"Report {kind}"))
        assert entries[-1] is not None, editor._status.text()
        editor.close()
    panels = [FigureSnapshot.from_storage(entry.payload["snapshot"], entry.blobs) for entry in entries]
    composed = compose_figures(panels, ["a", "b"], 2)
    stored = composed.provenance()
    assert [_evidence(panel)["ratio"] for panel in stored["panels"]] == [_spelled(math.inf)] * 2
    back = _decode(stored, {}, {})
    assert [_evidence(panel)["ratio"] for panel in back["panels"]] == [math.inf, math.inf]
    _strict(composed.provenance_json_bytes)
    _strict(composed.config_json_bytes)


def test_report_provenance_without_a_nonfinite_number_is_written_exactly_as_before(owner):
    owner._analysis_panel.current_detection().payload.provenance.pop("polarity_evidence")
    editor = _edit(owner, "depth")
    entry = editor.save_to_library(name="Report depth")
    assert entry is not None, editor._status.text()
    raw = entry.blobs["provenance_json"]
    assert b"__nanoscope_workspace_value__" not in raw
    assert raw == json.dumps(json.loads(raw), default=str, allow_nan=False, sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    editor.close()
