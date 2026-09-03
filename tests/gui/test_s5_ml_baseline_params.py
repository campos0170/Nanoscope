"""The measured open-pore baseline reaches ML detections.

THE DEFECT THIS PINS. ``AnalysisPanel.ml_params()`` deliberately carries no
``baseline_*`` key, but :class:`~nanoscope.analysis.ml_modules.MLDetect` reads
exactly those keys for the characterisation baseline every depth is referenced
to. So "Measure this recording" -> "Use these values" -> "Run ML detection"
still measured every depth at the DEFAULT 0.5 s / 0.1 s window with
``baseline_param_source='default'`` — while the result's own advisory told the
user to measure the recording. Found while building the worked example on the
owner's three apo-ferritin recordings: the accepted proposals (1.0/0.3, 2.0/0.5,
1.0/0.3) never reached the exported provenance.

The fix merges ``BaselinePanel.params()`` into the ML run's params at the same
seam the Schmitt run already uses. These tests drive the REAL panel objects and
assert on the params dict the controller is handed — the one place the two
detectors' treatment of the baseline can be compared directly.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------
import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.baseline import BASELINE_PARAM_KEYS


def test_ml_run_carries_the_baseline_panels_params(qapp, qtbot, monkeypatch):
    """Wired: a synthetic doc, real Session/Controller/TraceView/AnalysisPanel;
    accept measured windows on the Preprocess form, press the REAL Run ML path,
    and read the params dict handed to _launch_ml."""
    from pathlib import Path

    from nanoscope.core.session import Session
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import (ChannelInfo, ChannelKind, Provenance,
                                      RecordingMeta)
    from nanoscope.gui.analysis_panel import AnalysisPanel
    from nanoscope.gui.analysis_worker import AnalysisController
    from nanoscope.gui.trace_view import TraceView

    fs = 50_000.0
    rng = np.random.default_rng(7)
    cur = (100.0 + rng.normal(0, 1.5, 200_000)).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA",
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=Path("/tmp/s5mlbp.dat"), format_name="synthetic",
                         n_samples=cur.size, sample_rate_hz=fs,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(cur,))

    session = Session()
    tv = TraceView(session)
    controller = AnalysisController(session)
    panel = AnalysisPanel(session, controller, tv)
    qtbot.addWidget(panel); qtbot.addWidget(tv)
    doc_id = session.add_document(doc)
    session.add_layer(doc_id, 0)
    panel.set_selection(doc_id, 0)

    captured: list[dict] = []
    monkeypatch.setattr(panel, "_launch_ml",
                        lambda full: captured.append(dict(full)) or True)

    # The user's accepted windows on the Preprocess form...
    panel._baseline.rough_s.setValue(2.0)
    panel._baseline.base_s.setValue(0.5)
    # ...and the real ML run entry (model path set the way the suite does).
    panel._detection.set_model_path("/tmp/does-not-matter.keras")
    panel._detection.ml_range.setCurrentIndex(1)   # whole recording
    panel._detection.ml_detect_button.click()

    assert captured, "Run ML never reached _launch_ml"
    got = captured[-1]
    present = [k for k in BASELINE_PARAM_KEYS if k in got]
    assert present, (
        "the ML params carry no baseline_* key at all — depths would be "
        "measured at the default window while the advisory tells the user "
        "to measure this recording")
    assert got.get("baseline_rough_s") == pytest.approx(2.0)
    assert got.get("baseline_base_s") == pytest.approx(0.5)


def test_the_source_merge_exists_in_the_real_assembly():
    """The seam itself: the run assembly must merge self._baseline.params().

    Guards the fix the way test_redo_branch guards _note_step — on the CODE,
    with comments stripped, so a comment quoting the old defect cannot satisfy
    it and a refactor that drops the merge fails by name.
    """
    import inspect
    import re

    from nanoscope.gui.analysis_panel import AnalysisPanel

    src = inspect.getsource(AnalysisPanel)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    # Both detectors' assemblies merge the same object.
    hits = re.findall(r"_baseline\.params\(\)", code)
    assert len(hits) >= 3, (
        f"expected the baseline params merge in the Schmitt run(s) AND the ML "
        f"run assembly; found {len(hits)} merge site(s) — the ML path has "
        f"likely lost its baseline again")
