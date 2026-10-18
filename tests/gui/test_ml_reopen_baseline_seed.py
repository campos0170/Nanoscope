"""An ML detection seeds the open-pore estimator form on reopen, exactly as a Schmitt one does.

``AnalysisPanel._seed_form_from`` restored the baseline form only for the threshold
detectors. An ML run stores the baseline block in its params too (``_launch_ml`` merges
``self._baseline.params()``), but the ML branch seeded only the ML tab, so every reopened
ML project showed the app-default estimator over a table measured with the confirmed
windows: a false "open-pore estimator changed" banner, and a "Re-measure depths" button
that would have applied the defaults.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtWidgets

from nanoscope.analysis.baseline import BASELINE_PARAM_KEYS, DEFAULT_BASELINE
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 400_000
I0 = -100.0
DEPTH = 60.0
ML_MODULE_ID = "detect.keras"


def _signal(seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0) + rng.normal(0.0, 2.0, N)
    t = 500
    while t + 250 < N:
        x[t:t + 250] -= DEPTH
        t += 2000
    return x.astype(np.float32)


def _doc(tmp_path, signal) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "ml.dat", format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _window_with_ml_table(qtbot, tmp_path, *, config, params_override=None):
    """A window holding one ML table measured with ``config``, the panel not yet pointed
    at it: the reopened-project ordering, where results install before the selection."""
    win = MainWindow()
    qtbot.addWidget(win)
    signal = _signal()
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    df, prov = detect_events(signal, FS, baseline_config=config, polarity=-1,
                             min_dwell_s=1e-4, recording="ml", pore_id="pore")
    block = config.to_provenance()
    params = {"min_dwell_s": 1e-4,
              **{k: block[k] for k in BASELINE_PARAM_KEYS if k in block}}
    if params_override is not None:
        params = params_override
    win._session.add_result(AnalysisResult(
        module_id=ML_MODULE_ID, module_version="1.0", doc_id=doc_id, channel=0,
        params=params,
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov),
        baseline_fingerprint=prov.get("baseline_fingerprint")))
    return win, doc_id


def test_reopened_ml_table_seeds_the_baseline_form_and_shows_no_stale_banner(
        qapp, qtbot, tmp_path):
    stored = DEFAULT_BASELINE.replace(rough_s=2.0, base_s=0.5)
    win, doc_id = _window_with_ml_table(qtbot, tmp_path, config=stored)
    panel = win._analysis_panel
    bp = panel._baseline
    assert float(bp.rough_s.value()) == DEFAULT_BASELINE.rough_s   # not seeded yet

    win._on_selection(doc_id, 0)
    QtWidgets.QApplication.processEvents()

    assert float(bp.rough_s.value()) == 2.0
    assert float(bp.base_s.value()) == 0.5
    assert bp.config().fingerprint(FS) == stored.fingerprint(FS)
    assert not panel._stale.isVisibleTo(panel), (
        "a reopened ML table measured with the form's own estimator must not read as stale")


def test_ml_table_without_stored_params_does_not_raise_on_selection(
        qapp, qtbot, tmp_path):
    """A pre-v3 result carries no params at all; seeding must stay silent, not fall back
    to the Schmitt-shaped provenance projection."""
    win, doc_id = _window_with_ml_table(
        qtbot, tmp_path, config=DEFAULT_BASELINE, params_override={})
    panel = win._analysis_panel
    bp = panel._baseline

    win._on_selection(doc_id, 0)
    QtWidgets.QApplication.processEvents()

    assert float(bp.rough_s.value()) == DEFAULT_BASELINE.rough_s
    assert float(bp.base_s.value()) == DEFAULT_BASELINE.base_s
