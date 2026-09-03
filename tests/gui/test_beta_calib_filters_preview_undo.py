"""Beta-report regressions CF-2 / CF-3 / CF-4 — the Preprocess tab's two honesty
surfaces (pytest-qt, offscreen, through a real :class:`MainWindow`).

* **CF-2** — the "processed-trace preview unavailable: …" line names one recording's
  numbers ("window_s=60s is longer than the recording (0.5s)"), but it was only ever
  cleared on a SUCCESSFUL preview recompute. Selecting a different recording left it
  standing — a claim about recording A presented as feedback about recording B, where
  it is simply false. ``set_selection`` (and ``_show_empty``) now clear it when the
  target it described goes away, and unticking the preview clears it too (the request
  it described is withdrawn).

* **CF-3** — on a rate-unknown document (the legacy-TDMS shape: ``sample_rate_hz`` is
  honestly ``None``) the preview toggle accepted the tick, drew nothing and said
  nothing, while detection / the rise-time readout / the response plot all refuse the
  same document with a reason. ``_update_filter_preview`` now refuses OUT LOUD through
  the same ``preview_error_label`` as a misconfigured stage.

* **CF-4** — Edit ▸ Undo restores the previous detection through
  ``session.add_result`` → ``resultsChanged`` → ``_on_results_changed``, which swapped
  the shown table WITHOUT re-seeding the form: the tab kept claiming
  "Active: Notch, Savitzky-Golay" over an unfiltered table, and Detect re-ran with
  filters the shown events never had. The module's own rule (``_seed_form_from``:
  "the settings on the tab are the settings that produced the table on screen") is
  now enforced on results-changed-driven swaps exactly as on selection change and
  project reopen.
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
_EVENTS = [(10_000, 10_300), (20_000, 20_250), (30_000, 30_400)]


# ================================================================================================
# builders (the shape of tests/gui/test_analysis_integration.py)
# ================================================================================================

def _event_trace(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        if e <= n:
            x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _synth_doc(tmp_path, *, fs: float | None = FS, n: int = 50_000,
               name: str = "beta_cf.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META if fs else Provenance.UNKNOWN,
        channels=(info,))
    return TraceDocument(meta=meta, channels=(_event_trace(n),))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _run_detection(window: MainWindow, qtbot, *, min_dwell_s: float = 5e-4) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)   # let the queued add_result -> resultsChanged settle


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _trip_preview_error(window: MainWindow, qtbot) -> str:
    """Make the live preview fail honestly on the CURRENT selection: a 60 s detrend
    window on a 0.5 s recording. Returns the error text (asserted non-empty)."""
    det = window._analysis_panel._detection
    det.detrend_enabled.setChecked(True)
    det.detrend_window_s.setValue(60.0)
    det.filter_preview.setChecked(True)
    qtbot.wait(20)
    text = det.preview_error_label.text()
    assert "longer than the recording" in text, f"setup failed to trip the error: {text!r}"
    return text


# ================================================================================================
# CF-2 — a preview error naming recording A must not survive selecting recording B
# ================================================================================================

def test_preview_error_is_cleared_when_a_different_recording_is_selected(window, qtbot, tmp_path):
    _install(window, _synth_doc(tmp_path, n=25_000, name="rec_a_halfsec.dat"))
    _trip_preview_error(window, qtbot)

    # The user clicks the 4 s recording B in the file tree — the real selection path.
    _install(window, _synth_doc(tmp_path, n=200_000, name="rec_b_4sec.dat"))
    qtbot.wait(20)

    det = window._analysis_panel._detection
    assert det.preview_error_label.text() == "", (
        "a failure message naming recording A's length (0.5 s) survived onto "
        "recording B, where it is false")
    assert not det.preview_error_label.isVisibleTo(det)


def test_preview_error_is_cleared_when_the_preview_is_unticked(window, qtbot, tmp_path):
    """Unticking withdraws the request; a standing error would otherwise survive into
    the next tick, where it may no longer be true."""
    _install(window, _synth_doc(tmp_path, n=25_000))
    det = window._analysis_panel._detection
    _trip_preview_error(window, qtbot)

    det.filter_preview.setChecked(False)
    qtbot.wait(20)
    assert det.preview_error_label.text() == ""


def test_preview_error_does_not_survive_closing_the_document(window, qtbot, tmp_path):
    doc_id, _ = _install(window, _synth_doc(tmp_path, n=25_000))
    _trip_preview_error(window, qtbot)

    window._session.close_document(doc_id)
    qtbot.wait(20)
    assert window._analysis_panel._detection.preview_error_label.text() == "", (
        "the error line outlived the recording it was about")


# ================================================================================================
# CF-3 — the preview refuses a rate-unknown recording out loud, like detection does
# ================================================================================================

def test_preview_on_rate_unknown_recording_says_why_instead_of_silently_drawing_nothing(
        window, qtbot, tmp_path):
    _install(window, _synth_doc(tmp_path, fs=None, n=8_000, name="norate.dat"))
    det = window._analysis_panel._detection

    det.notch_enabled.setChecked(True)
    qtbot.wait(10)
    assert det.filter_preview.isEnabled()
    det.filter_preview.setChecked(True)   # the user asks to see the processed trace
    qtbot.wait(20)

    text = det.preview_error_label.text()
    assert "sample rate" in text.lower(), (
        f"rate-unknown preview refused with no reason (or the wrong one): {text!r}")
    assert det.preview_error_label.isVisibleTo(det)
    # No curve was fabricated from a nonexistent time base.
    assert window._trace_view._filter_preview is None


def test_rate_unknown_refusal_lifecycles_like_any_other_preview_error(window, qtbot, tmp_path):
    """The new message obeys CF-2's rule too: it clears when the request is withdrawn
    and when a rate-KNOWN recording is selected (where a preview can actually run)."""
    _install(window, _synth_doc(tmp_path, fs=None, n=8_000, name="norate.dat"))
    det = window._analysis_panel._detection
    det.notch_enabled.setChecked(True)
    det.filter_preview.setChecked(True)
    qtbot.wait(20)
    assert "sample rate" in det.preview_error_label.text().lower()

    # Withdrawing the request clears it...
    det.filter_preview.setChecked(False)
    qtbot.wait(10)
    assert det.preview_error_label.text() == ""

    # ...and so does moving to a recording the message is not about.
    det.filter_preview.setChecked(True)
    qtbot.wait(20)
    assert "sample rate" in det.preview_error_label.text().lower()
    _install(window, _synth_doc(tmp_path, fs=FS, n=25_000, name="rated.dat"))
    qtbot.wait(20)
    assert det.preview_error_label.text() == ""


# ================================================================================================
# CF-4 — undo/redo swap the shown table; the form must follow it
# ================================================================================================

def test_undo_of_a_re_detection_re_seeds_the_form_to_the_restored_run(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path, n=100_000))
    det = window._analysis_panel._detection

    # run 1: unfiltered. run 2: notch + savgol on.
    _run_detection(window, qtbot)
    det.notch_enabled.setChecked(True)
    det.savgol_enabled.setChecked(True)
    qtbot.wait(10)
    _run_detection(window, qtbot)
    p2 = window._session.results.get(doc_id, channel, "detect.schmitt").payload.provenance
    assert p2["notch_enabled"] is True and p2["savgol_enabled"] is True

    # Edit ▸ Undo — the real menu action.
    assert window._act_undo.isEnabled()
    window._act_undo.trigger()
    qtbot.wait(50)

    prov = window._session.results.get(doc_id, channel, "detect.schmitt").payload.provenance
    assert prov["notch_enabled"] is False and prov["savgol_enabled"] is False, (
        "undo did not restore the unfiltered run — the fixture no longer tests the swap")
    assert not det.notch_enabled.isChecked() and not det.savgol_enabled.isChecked(), (
        "the form still claims the NEWER run's filter stages over the restored "
        "unfiltered table — Detect would re-run with filters the shown events never had")
    label = det.active_filters_label.text()
    assert "Notch" not in label and "Savitzky" not in label, label


def test_redo_re_seeds_the_form_forward_again(window, qtbot, tmp_path):
    doc_id, channel = _install(window, _synth_doc(tmp_path, n=100_000))
    det = window._analysis_panel._detection

    _run_detection(window, qtbot)
    det.notch_enabled.setChecked(True)
    qtbot.wait(10)
    _run_detection(window, qtbot)

    window._act_undo.trigger()
    qtbot.wait(50)
    assert not det.notch_enabled.isChecked()

    window._act_redo.trigger()
    qtbot.wait(50)
    prov = window._session.results.get(doc_id, channel, "detect.schmitt").payload.provenance
    assert prov["notch_enabled"] is True
    assert det.notch_enabled.isChecked(), (
        "redo re-installed the filtered run but the form stayed on the unfiltered "
        "settings — the same disagreement in the other direction")


def test_undo_re_seeds_numeric_thresholds_too(window, qtbot, tmp_path):
    """The rule is about the whole form, not the filter checkboxes: a threshold edited
    between runs must revert with the table it produced."""
    doc_id, channel = _install(window, _synth_doc(tmp_path, n=100_000))
    det = window._analysis_panel._detection

    det.k_in.setValue(6.0)
    _run_detection(window, qtbot)
    det.k_in.setValue(4.0)
    _run_detection(window, qtbot)
    assert det.k_in.value() == pytest.approx(4.0)

    window._act_undo.trigger()
    qtbot.wait(50)
    res = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert res.params["k_in"] == pytest.approx(6.0)
    assert det.k_in.value() == pytest.approx(6.0), (
        "the table on screen came from k_in=6.0 but the form still reads 4.0")
