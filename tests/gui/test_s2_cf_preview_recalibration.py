"""S2 regressions — CF2-1 / VIEWER2-002: the live "processed trace" preview must
follow a calibration change on its channel.

``Session.calibrationChanged`` re-set the raw curves and axis label
(``TraceView.refresh_channel``) and re-judged detection staleness
(``AnalysisPanel._on_calibration_changed``) — but nothing recomputed or cleared the
preview curve. Two symptoms of the one root:

  * CF2-1 (same units): Apply gain 2.5 / offset -40 with a savgol preview live →
    the main trace re-renders at the new scale while the orange preview stays at
    the old one — two curves on one pane disagreeing by the whole calibration,
    with USER_GUIDE §5 promising the preview "previews exactly the signal the
    detector sees".
  * VIEWER2-002 (units change): pA -> nA leaves the preview 1000x off; y-autorange
    then spans both curves and squashes the real trace to ~0.08% of pane height —
    the edit looks like it destroyed the recording.

The fix wires ``_on_calibration_changed`` to recompute through the SAME path the
tick uses (``_update_filter_preview``), exactly as the measure panel re-measures on
this signal. Driven offscreen through a real MainWindow (the shape of
tests/gui/test_beta_calib_filters_preview_undo.py).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 50_000            # 1 s


def _trace(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path, name: str = "s2_cf_recal.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(_trace(N),))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _live_preview(window: MainWindow, qtbot) -> None:
    """Tick a savgol preview on the current selection and let it draw."""
    det = window._analysis_panel._detection
    det.savgol_enabled.setChecked(True)
    det.filter_preview.setChecked(True)
    qtbot.wait(20)
    view = window._analysis_panel._trace_view
    assert view.has_filter_preview(), "setup: the preview curve failed to draw"


def _preview_median(window: MainWindow) -> float:
    curve = window._analysis_panel._trace_view._filter_preview
    assert curve is not None
    return float(np.median(np.asarray(curve.yData)))


# ================================================================================================
# CF2-1 — same-unit gain/offset Apply
# ================================================================================================

def test_preview_recomputed_after_same_unit_calibration(window, qtbot, tmp_path):
    """Apply gain 2.5 / offset -40 with the preview live: the orange curve must be
    recomputed from the NEW calibrated data (median ~100*2.5-40 = 210), not keep
    the old-calibration values (~100) beside a main trace at 210."""
    did, ch = _install(window, _synth_doc(tmp_path))
    _live_preview(window, qtbot)
    assert _preview_median(window) == pytest.approx(100.0, abs=1.0)

    window._session.set_calibration(
        did, ch, ChannelCalibration(gain=2.5, offset=-40.0, target_units="pA"),
        write_sidecar=False)
    qtbot.wait(20)

    det = window._analysis_panel._detection
    view = window._analysis_panel._trace_view
    assert view.has_filter_preview(), "the live preview must survive the edit (recomputed)"
    assert det.filter_preview.isChecked()
    assert det.preview_error_label.text() == ""
    assert _preview_median(window) == pytest.approx(210.0, abs=1.0), (
        "the preview kept OLD-calibration data: it no longer shows the signal "
        "the detector would see")


# ================================================================================================
# VIEWER2-002 — units-changing recalibration (pA -> nA)
# ================================================================================================

def test_preview_follows_a_units_change(window, qtbot, tmp_path):
    """pA -> nA (gain 0.001): preview and raw curve must agree (~1x, both ~0.1 nA),
    not disagree 1000x with the real trace squashed to a hairline by y-autorange."""
    did, ch = _install(window, _synth_doc(tmp_path))
    _live_preview(window, qtbot)

    window._session.set_calibration(
        did, ch, ChannelCalibration(gain=0.001, target_units="nA"),
        write_sidecar=False)
    qtbot.wait(20)

    view = window._analysis_panel._trace_view
    assert view.has_filter_preview()
    preview_med = _preview_median(window)
    raw_med = float(np.median(window._session.calibrated(did, ch).data))
    assert preview_med == pytest.approx(raw_med, rel=1e-2), (
        f"preview median {preview_med} vs raw median {raw_med}: the 1000x stale-curve defect")


# ================================================================================================
# guards — the recompute must not create previews nobody asked for
# ================================================================================================

def test_calibration_with_preview_unticked_draws_nothing(window, qtbot, tmp_path):
    """The recompute is gated on the tick: an edit with the box unticked must not
    conjure a preview curve."""
    did, ch = _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.savgol_enabled.setChecked(True)          # a stage on, but preview NOT requested
    qtbot.wait(20)
    view = window._analysis_panel._trace_view
    assert not view.has_filter_preview()

    window._session.set_calibration(
        did, ch, ChannelCalibration(gain=2.5, target_units="pA"), write_sidecar=False)
    qtbot.wait(20)

    assert not view.has_filter_preview(), (
        "a calibration edit must not draw a preview the user never ticked")


def test_calibration_on_another_recording_leaves_the_preview_alone(window, qtbot, tmp_path):
    """An edit on a different (unselected) recording is not about the previewed
    channel: the live preview keeps its values."""
    other_did, _ = _install(window, _synth_doc(tmp_path, name="other.dat"))
    did, ch = _install(window, _synth_doc(tmp_path))          # selection moves here
    _live_preview(window, qtbot)
    before = _preview_median(window)

    window._session.set_calibration(
        other_did, 0, ChannelCalibration(gain=2.5, target_units="pA"), write_sidecar=False)
    qtbot.wait(20)

    assert _preview_median(window) == pytest.approx(before)
