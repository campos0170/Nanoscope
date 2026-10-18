"""Sweep 7, S7-POP1 (minor), the AnalysisPanel half: ``stalenessChanged`` fires every time
the shown result is re-judged -- when the badge set is written AND when it is cleared.

Measured on the pre-fix tree (docs/sweep7/repros/S7-POP1_stale_depth_unmarked_on_curation_
surfaces.py): one Preprocess knob after a detection badged nine table headers and raised the
banner, while the Event Gallery went on sorting and captioning by ``depth_pA`` and the Event
Inspector printed "depth: 122.37 pA" with no mark. Agent B gave both windows a
``set_stale_columns`` seam; what they could not hear was the moment the badge set moved,
because the Preprocess-knob path (``_on_baseline_params_changed``) and the calibration path
(``_on_calibration_changed``) into ``_check_staleness`` reach no surface refresh. This
signal is that moment. MainWindow wires it to the push (a separate change).

Driven through a real MainWindow with a real detection, so the stamps compared are the ones
the engine wrote.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.modules import register_all
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _signal(spans, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, qtbot, tmp_path):
    """A real Schmitt detection on screen, its stamps written by the engine."""
    did = window._session.add_document(_doc(tmp_path, _signal(_FRAGMENTS), name="a.dat"))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    qtbot.wait(20)
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    assert panel.current_detection() is not None
    assert panel._table.stale_columns == ()
    # isHidden(), not isVisibleTo(panel): the stale line lives in the Results dock now, so the
    # panel is not its ancestor and isVisibleTo would depend on the window being shown.
    assert panel._stale.isHidden()
    return window, did


def test_a_preprocess_knob_after_a_detection_announces_the_badges_and_their_clearing(
        detected, qtbot):
    window, _did = detected
    panel = window._analysis_panel
    bl = panel.baseline_panel
    original = bl.rough_s.value()

    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        bl.rough_s.setValue(original * 4)
    assert panel._table.stale_columns == REMEASURED_COLUMNS
    assert not panel._stale.isHidden()
    assert panel._stale.kind == "baseline"

    # The CLEAR path announces itself too: a surface that only hears "went stale" keeps
    # its marks after the banner is gone.
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        bl.rough_s.setValue(original)
    assert panel._table.stale_columns == ()
    assert panel._stale.isHidden()


def test_a_calibration_edit_after_a_detection_announces_the_badges(detected, qtbot):
    window, did = detected
    panel = window._analysis_panel
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        window._session.set_calibration(did, 0, ChannelCalibration(2.0, 0.0, "pA"),
                                        write_sidecar=False)
    assert panel._table.stale_columns == REMEASURED_COLUMNS
    assert not panel._stale.isHidden()
    assert panel._stale.kind == "calibration"


def test_one_judgement_is_one_announcement(detected, qtbot):
    """The receiver re-reads the panel on every emission, so the count is the cost."""
    window, _did = detected
    panel = window._analysis_panel
    hits: list[int] = []
    panel.stalenessChanged.connect(lambda: hits.append(1))
    panel._check_staleness(panel.current_detection())
    qtbot.wait(10)
    assert len(hits) == 1
