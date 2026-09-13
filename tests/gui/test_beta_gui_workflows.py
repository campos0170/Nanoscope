"""Targeted beta input and first-open workflow regressions."""
import numpy as np
import pytest

from nanoscope.gui.baseline_panel import _OpenPoreReferenceEditor
from nanoscope.gui.first_analysis_guide import FirstAnalysisGuide
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel
from .test_psd import _current_doc, _install


@pytest.mark.parametrize("minimum, maximum", [(0.5, 0.001), (0.001, 0.001)])
@pytest.mark.parametrize("detector", ["detect.schmitt", "detect.cusum"])
def test_invalid_dwell_pair_is_refused_before_any_worker_starts(
        qtbot, tmp_path, monkeypatch, minimum, maximum, detector):
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    params = panel.detection_params() | {
        "min_dwell_s": minimum, "max_dwell_s": maximum, "detector": detector}
    starts, refusals = [], []
    monkeypatch.setattr(panel._controller, "start", lambda *a, **kw: starts.append(kw) or False)
    panel.runRefused.connect(refusals.append)
    assert not panel._run_detection(params)
    assert not starts
    assert "maximum dwell" in refusals[-1] and "minimum dwell" in refusals[-1]
    assert "ValueError" not in refusals[-1]
    # Off and a genuinely wider range retain the ordinary controller route.
    for valid_max in (None, minimum * 2):
        panel._run_detection(params | {"max_dwell_s": valid_max})
    assert len(starts) == 2


def test_baseline_guide_tooltip_explains_baseline_and_preserves_disabled_reason(qtbot):
    available = [True]
    guide = FirstAnalysisGuide(availability_provider=lambda route: (
        available[0], "Optional: pre-filter the current signal" if available[0]
        else "Select a current recording first."))
    qtbot.addWidget(guide)
    guide.steps.setCurrentRow(2)
    button = guide._buttons[0]
    assert button.property("route") == "baseline"
    assert "open-pore reference" in button.toolTip()
    assert "Measure this recording" in button.toolTip()
    assert "pre-filter" not in button.toolTip()
    available[0] = False
    guide.refresh_context()
    assert not button.isEnabled()
    assert button.toolTip() == "Select a current recording first."


def test_numeric_boxes_reject_python_underscores_but_keep_decimal_exponents_and_unicode():
    with pytest.raises(ValueError, match="1_20"):
        DetectionPanel.parse_notch_lines("60, 1_20")
    assert _OpenPoreReferenceEditor._number("1_20") is None
    assert DetectionPanel.parse_notch_lines("1.2e2, ٦٠") == [120, 60]
    assert _OpenPoreReferenceEditor._number("−1.2e2") == -120
    assert _OpenPoreReferenceEditor._number("١٢٠") == 120


@pytest.mark.parametrize("measured_region", [False, True])
def test_first_noise_open_computes_only_the_requested_spectrum(
        qtbot, tmp_path, monkeypatch, measured_region):
    from nanoscope.analysis import psd
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _current_doc(tmp_path)
    doc_id = _install(win, doc)
    win._on_selection(doc_id, 0)
    samples = []
    original = psd.welch_psd
    def measured(data, *args, **kwargs):
        samples.append(len(data))
        return original(data, *args, **kwargs)
    monkeypatch.setattr(psd, "welch_psd", measured)
    if measured_region:
        monkeypatch.setattr(win._trace_view, "measure_target", lambda: (doc_id, 0))
        monkeypatch.setattr(win._trace_view, "measure_bounds", lambda: (25000, 50000))
        win._open_psd_for_selection()
        assert samples == [25000]
    else:
        win._open_psd()
        assert samples == [doc.meta.n_samples]
    assert win._psd_window.has_data()
