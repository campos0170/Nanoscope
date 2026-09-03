"""S2 regression — CF2-3: a detrend window equal to the trace length degenerated to a
SILENT no-op.

At window == trace the kernel forms exactly one block, warns (its own words) that
"the baseline is effectively constant and cannot track drift", and returns the input
untouched — while the preview drew the unchanged trace with an empty error line,
``res.warnings`` said nothing, and provenance stamped ``detrend_enabled=True`` over a
stage that did nothing. One millisecond more window flipped to the honest
"longer than the recording" refusal: a one-spinbox-step cliff between honest error
and silent no-op.

The fix routes the kernel's own ``ConstantBaselineWarning`` (thread-local sink,
``collect_baseline_warnings``) onto the user surfaces:

  * the preview treats the degenerate window like the neighbouring over-long-window
    refusal — curve cleared, kernel's words on the error line (whose fixed prefix
    says "unavailable", which a cleared curve keeps true);
  * detection carries the warning in ``res.warnings`` (the Stats panel's surface)
    and stamps ``detrend_effective=False`` + ``detrend_warnings`` in provenance.

Driven offscreen through a real MainWindow (the shape of
tests/gui/test_beta_calib_filters_preview_undo.py).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 50_000            # exactly 1 s — the degenerate boundary is window_s == 1.0
_EVENTS = [(10_000, 10_300), (20_000, 20_250), (30_000, 30_400)]


def _drifting_trace(n: int, seed: int = 0) -> np.ndarray:
    """Blockade events on a baseline with 25 pA of linear drift — the thing a working
    detrend removes and a degenerate one leaves untouched."""
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + np.linspace(0.0, 25.0, n) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path, name: str = "s2_detrend.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(_drifting_trace(N),))


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


def _tick_detrend_preview(window: MainWindow, qtbot, window_s: float) -> None:
    det = window._analysis_panel._detection
    det.detrend_enabled.setChecked(True)
    det.detrend_window_s.setValue(window_s)
    det.filter_preview.setChecked(True)
    qtbot.wait(20)


# ================================================================================================
# the preview surface
# ================================================================================================

def test_degenerate_window_puts_the_kernels_warning_on_the_error_line(window, qtbot, tmp_path):
    """window == trace length: the kernel's own 'cannot track drift' sentence must
    reach the preview error line, and the curve is cleared (same treatment as the
    over-long window one spinbox step away — the stage is equally misconfigured,
    it just fails to raise)."""
    _install(window, _synth_doc(tmp_path))
    _tick_detrend_preview(window, qtbot, 1.0)

    det = window._analysis_panel._detection
    text = det.preview_error_label.text()
    assert "cannot track drift" in text, (
        f"the kernel's degeneracy warning never reached the error line: {text!r}")
    assert "1 block" in text                      # the kernel's own words, not a paraphrase
    assert not window._analysis_panel._trace_view.has_filter_preview(), (
        "a curve drawn under an 'unavailable' line would make the two surfaces disagree")


def test_one_step_more_window_is_still_the_honest_refusal(window, qtbot, tmp_path):
    """The other side of the millisecond cliff keeps its first-sweep fix: window >
    trace raises and the error line names it."""
    _install(window, _synth_doc(tmp_path))
    _tick_detrend_preview(window, qtbot, 1.001)

    det = window._analysis_panel._detection
    assert "longer than the recording" in det.preview_error_label.text()
    assert not window._analysis_panel._trace_view.has_filter_preview()


def test_working_window_previews_with_no_message(window, qtbot, tmp_path):
    """A window that forms many blocks detrends for real: curve drawn, line empty —
    the pre-existing healthy path is untouched."""
    _install(window, _synth_doc(tmp_path))
    _tick_detrend_preview(window, qtbot, 0.1)

    det = window._analysis_panel._detection
    assert det.preview_error_label.text() == ""
    view = window._analysis_panel._trace_view
    assert view.has_filter_preview()
    # ...and it actually flattened the drift: ends of the processed trace agree.
    y = np.asarray(view._filter_preview.yData)
    drift = abs(float(np.median(y[-2000:])) - float(np.median(y[:2000])))
    assert drift < 5.0, f"working detrend left {drift:.1f} pA of the 25 pA drift"


def test_recovering_the_window_clears_the_degeneracy_message(window, qtbot, tmp_path):
    """Shrinking the window back to a working size recomputes and clears the line —
    the message must not outlive the misconfiguration it described."""
    _install(window, _synth_doc(tmp_path))
    _tick_detrend_preview(window, qtbot, 1.0)
    assert "cannot track drift" in window._analysis_panel._detection.preview_error_label.text()

    window._analysis_panel._detection.detrend_window_s.setValue(0.1)
    qtbot.wait(20)

    det = window._analysis_panel._detection
    assert det.preview_error_label.text() == ""
    assert window._analysis_panel._trace_view.has_filter_preview()


# ================================================================================================
# the detection surfaces — res.warnings (Stats panel) and provenance
# ================================================================================================

def test_detection_with_degenerate_detrend_warns_and_stamps_honest_provenance(
        window, qtbot, tmp_path):
    _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.detrend_enabled.setChecked(True)
    det.detrend_window_s.setValue(1.0)
    det.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(20)

    res = window._analysis_panel._stored_detection()
    assert res is not None
    assert any("cannot track drift" in w for w in res.warnings), (
        f"the kernel's no-op warning is missing from res.warnings: {res.warnings!r}")
    prov = res.payload.provenance
    assert prov["detrend_enabled"] is True
    assert prov["detrend_effective"] is False, (
        "provenance claims an effective detrend over a stage that did nothing")
    assert any("cannot track drift" in w for w in prov["detrend_warnings"])


def test_detection_with_working_detrend_stays_clean(window, qtbot, tmp_path):
    """A many-block detrend keeps effective=True and adds no drift warning — the
    honest-provenance stamp must not cry wolf on a stage that worked."""
    _install(window, _synth_doc(tmp_path))
    det = window._analysis_panel._detection
    det.detrend_enabled.setChecked(True)
    det.detrend_window_s.setValue(0.1)
    det.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        window._analysis_panel.trigger_detection()
    qtbot.wait(20)

    res = window._analysis_panel._stored_detection()
    assert res is not None
    assert not any("cannot track drift" in w for w in res.warnings)
    prov = res.payload.provenance
    assert prov["detrend_enabled"] is True
    assert prov["detrend_effective"] is True
    assert prov["detrend_warnings"] == ()
