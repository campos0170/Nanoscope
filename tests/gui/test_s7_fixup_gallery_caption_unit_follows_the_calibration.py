"""Sweep-7 fix-up X-SURF-9 — after a calibration edit the gallery's depth caption and the
Event Inspector's readout print the SAME unit for the same event.

Measured on a copy of the owner's 15 s -1000 mV ABF: set the current-channel gain to 0.001
-> nA in the Calibration panel with a detection on screen and both windows open. Both badged
the depth ⚠, but event 91 read '91 · 139pA ⚠' on the tile and 'depth: 138.85 nA ⚠' in the
inspector -- one stale number under two units on two curation windows at once. The gallery
resolved the channel's unit once, in ``set_events``; a calibration edit reaches it through
``refresh()`` (the same-frame re-feed) and ``set_stale_columns``, and neither re-resolved.
The inspector re-resolves on every re-feed. Now the gallery's re-feed does too: one resolver
(``_resolve_units``), called on both doors.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import re

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
N_EVENTS = 30
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(31)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "units.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == N_EVENTS, len(df)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov),
        baseline_fingerprint=w._analysis_panel.baseline_panel.config().fingerprint(FS)))
    w._act_event_gallery.trigger()
    qtbot.wait(20)
    w._open_event_inspector()
    qtbot.wait(20)
    gal, insp = w._event_gallery, w._event_inspector
    gal.show()
    insp.show()
    gal.set_sort("Depth (deepest first)")
    qtbot.wait(40)
    yield w, doc_id, gal, insp
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _units_of(gal, insp, df, row: int) -> tuple[str, str, str]:
    cap = gal.caption_for(df.iloc[row])
    line = next(l for l in insp._readout.text().splitlines() if l.startswith("median depth"))
    unit_cap = re.search(r"[0-9.]+([a-zA-Z]+)", cap).group(1)
    unit_insp = re.search(r"depth:\s+[-0-9.e+]+\s*([a-zA-Z]+)", line).group(1)
    return cap, unit_cap, unit_insp


def test_the_depth_caption_and_the_inspector_readout_agree_on_the_unit_after_a_gain_edit(
        wired, qtbot):
    w, doc_id, gal, insp = wired
    df = _frame(w, doc_id)
    row = gal._model.df_row(gal._model.index(0, 0))
    eid = int(df.iloc[row]["event_id"])
    w._analysis_panel.select_event(eid)
    qtbot.wait(40)
    assert insp.current_event_id() == eid
    cap, unit_cap, unit_insp = _units_of(gal, insp, df, row)
    assert unit_cap == unit_insp == "pA", (cap, unit_insp)

    # In memory only: nothing is written beside the (synthetic) recording.
    w._session.set_calibration(doc_id, 0, ChannelCalibration(gain=0.001, target_units="nA"),
                               write_sidecar=False)
    qtbot.wait(150)
    assert gal._resolve_units() == "nA"
    cap, unit_cap, unit_insp = _units_of(gal, insp, df, row)
    assert unit_insp == "nA", "the inspector re-resolves on its re-feed"
    assert unit_cap == unit_insp, (
        f"two windows, two units for event {eid}: tile {cap!r}, inspector {unit_insp}")
    assert gal.stale_columns(), "a gain edit badges the depth on both windows (S7-POP1)"
    assert cap.endswith("nA") and "⚠" not in cap, cap
    assert "stale" in gal.status.text(), gal.status.text()

    # ...and back: the caption follows the channel, not the first unit it was fed.
    w._session.set_calibration(doc_id, 0, ChannelCalibration(gain=1.0, target_units="pA"),
                               write_sidecar=False)
    qtbot.wait(150)
    cap, unit_cap, unit_insp = _units_of(gal, insp, df, row)
    assert unit_cap == unit_insp == "pA", (cap, unit_insp)


def test_a_synthetic_recording_gets_no_sidecar_from_this_suite(wired, tmp_path):
    """Data hygiene: the calibration edits above are in-memory; nothing is written."""
    assert not list(tmp_path.glob("*.npcal.json"))
