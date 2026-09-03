"""User-supplied filter parameters in DetectionPanel (pytest-qt, offscreen).

The panel used to hard-code the two facts that define a low-pass measurement — that it
was 8-pole and Bessel — in the TEXT OF A CHECKBOX LABEL, while the kernel had always
accepted any order and three kinds. Same for the 4-pole Butterworth high-pass, and for
the baseline-drift estimators' shape parameters (a selectable 'percentile' method whose
percentile could not be moved, and an ASLS whose smoothness is measured in block units
and so silently changes meaning with the window).

These tests pin: the controls exist and are built from the kernel's own lists; their
defaults reproduce the previously frozen values; they reach ``params()``; the frequency
spins are clamped to the active recording's Nyquist; the rise-time readout follows the
selected design; and the response preview draws the selected stage (with an honest note
for the two stages that have no transfer function).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import warnings

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import filters
from nanoscope.gui.panels import DetectionPanel

FS = 250_000.0

# Combo index of each stage in the filter selector (== the page-stack index).
LOWPASS, HIGHPASS, NOTCH, MEDIAN, DETREND, RMEAN, SAVGOL = range(7)


@pytest.fixture
def panel(qtbot):
    p = DetectionPanel()
    qtbot.addWidget(p)
    return p


# ================================================================================================
# The controls exist, are built from the kernel's own lists, and default to the frozen values
# ================================================================================================

def test_lowpass_type_combo_offers_exactly_the_kinds_the_kernel_accepts(panel):
    labels = [panel.filter_kind.itemText(i) for i in range(panel.filter_kind.count())]
    assert labels == ["Bessel", "Butterworth", "Gaussian"]
    assert panel.filter_kind.count() == len(filters.KINDS)


def test_highpass_type_combo_offers_exactly_the_kinds_the_kernel_accepts(panel):
    labels = [panel.hp_kind.itemText(i) for i in range(panel.hp_kind.count())]
    assert labels == ["Butterworth", "Bessel"]      # gaussian has no high-pass counterpart
    assert panel.hp_kind.count() == len(filters.HP_KINDS)


def test_defaults_are_exactly_the_previously_hard_coded_values(panel):
    p = panel.params()
    assert p["filter_kind"] == "bessel" and p["filter_order"] == 8
    assert p["hp_kind"] == "butter" and p["hp_order"] == 4
    assert p["detrend_percentile"] == pytest.approx(80.0)
    assert p["detrend_asls_lam"] == pytest.approx(6e4)
    assert p["detrend_asls_p"] == pytest.approx(1e-3)
    assert p["detrend_asls_niter"] == 10
    assert p["detrend_asls_decim"] == pytest.approx(0.1)


def test_the_checkbox_labels_no_longer_hard_code_a_design(panel):
    """The label used to assert '8-pole Bessel' / '4-pole Butterworth' as fact."""
    assert "pole" not in panel.filter_enabled.text()
    assert "Bessel" not in panel.filter_enabled.text()
    assert "pole" not in panel.hp_enabled.text()
    assert "Butterworth" not in panel.hp_enabled.text()


def test_kind_choices_and_setters_round_trip(panel):
    for kind in filters.KINDS:
        panel.set_filter_kind(kind)
        assert panel.filter_kind_choice() == kind
        assert panel.params()["filter_kind"] == kind
    for kind in filters.HP_KINDS:
        panel.set_hp_kind(kind)
        assert panel.hp_kind_choice() == kind
        assert panel.params()["hp_kind"] == kind
    # An unknown string is ignored, leaving the current choice (mirrors set_detrend_method).
    panel.set_filter_kind("chebyshev")
    assert panel.filter_kind_choice() in filters.KINDS


def test_order_setters_reach_params(panel):
    panel.set_filter_order(3)
    panel.set_hp_order(6)
    p = panel.params()
    assert p["filter_order"] == 3
    assert p["hp_order"] == 6


def test_gaussian_disables_the_order_control_because_it_has_none(panel):
    assert panel.filter_order.isEnabled()
    panel.set_filter_kind("gaussian")
    assert not panel.filter_order.isEnabled()
    panel.set_filter_kind("bessel")
    assert panel.filter_order.isEnabled()


# ================================================================================================
# Baseline-drift shape parameters follow the selected method
# ================================================================================================

def test_detrend_shape_parameters_follow_the_selected_method(panel):
    panel.filter_selector.setCurrentIndex(DETREND)     # raise the Baseline drift page
    panel.set_detrend_method("block_median")
    assert panel.detrend_params_stack.currentIndex() == 0
    panel.set_detrend_method("percentile")
    assert panel.detrend_params_stack.currentIndex() == 1
    assert panel.detrend_percentile.isVisibleTo(panel)
    panel.set_detrend_method("asls")
    assert panel.detrend_params_stack.currentIndex() == 2
    for w in (panel.detrend_asls_lam, panel.detrend_asls_p,
              panel.detrend_asls_niter, panel.detrend_asls_decim):
        assert w.isVisibleTo(panel)
    assert not panel.detrend_percentile.isVisibleTo(panel)


def test_detrend_shape_parameters_are_always_in_params_regardless_of_method(panel):
    """params() reads every widget, like every other stage — the module forwards only
    the active method's, so a hidden page's value cannot leak into the estimator."""
    panel.set_detrend_method("block_median")
    panel.detrend_asls_lam.setValue(1e5)
    assert panel.params()["detrend_asls_lam"] == pytest.approx(1e5)


# ================================================================================================
# fs / Nyquist-aware ranges
# ================================================================================================

def test_cutoff_maxima_clamp_to_nyquist_once_the_rate_is_known(panel):
    panel.set_recording(50_000.0)
    nyq = 25_000.0
    for spin in (panel.filter_cutoff_hz, panel.hp_cutoff_hz, panel.notch_freq_hz):
        assert spin.maximum() < nyq
        assert spin.maximum() > nyq * 0.999      # just below, not conservatively low


def test_the_maximum_is_a_value_the_kernel_actually_accepts(panel):
    """The spin's own ceiling must not be a value that raises — that was the old bug."""
    for fs in (10_000.0, 50_000.0, 250_000.0, 1_000_000.0):
        panel.set_recording(fs)
        panel.filter_cutoff_hz.setValue(panel.filter_cutoff_hz.maximum())
        # Round-trips through the kernel's own validation without RAISING. It still
        # warns (any value this close to Nyquist is above fs/2.5) — that advisory is a
        # separate concern, surfaced in the rise-time readout, and is not what this
        # test is about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            filters.lowpass(np.zeros(4096, dtype=np.float32), fs,
                            panel.filter_cutoff_hz.value())


def test_a_cutoff_above_nyquist_cannot_be_dialled_in(panel):
    panel.set_recording(50_000.0)
    panel.filter_cutoff_hz.setValue(5_000_000.0)     # the old static maximum
    assert panel.filter_cutoff_hz.value() < 25_000.0
    panel.hp_cutoff_hz.setValue(5_000_000.0)
    assert panel.hp_cutoff_hz.value() < 25_000.0
    panel.notch_freq_hz.setValue(100_000.0)          # the old static notch maximum
    assert panel.notch_freq_hz.value() < 25_000.0


def test_a_stale_cutoff_is_pulled_down_when_a_slower_recording_is_selected(panel):
    panel.set_recording(250_000.0)
    panel.filter_cutoff_hz.setValue(100_000.0)
    panel.set_recording(50_000.0)
    assert panel.filter_cutoff_hz.value() < 25_000.0


def test_ranges_fall_back_to_the_static_maxima_when_the_rate_is_unknown(panel):
    panel.set_recording(250_000.0)
    panel.set_recording(None)
    assert panel.filter_cutoff_hz.maximum() == pytest.approx(5_000_000.0)
    assert panel.hp_cutoff_hz.maximum() == pytest.approx(5_000_000.0)
    assert panel.notch_freq_hz.maximum() == pytest.approx(100_000.0)


def test_cutoff_accepts_fractional_hz(panel):
    """decimals=0 used to force integer cutoffs AND make every typed digit a new value."""
    assert panel.filter_cutoff_hz.decimals() >= 1
    panel.set_recording(250_000.0)
    panel.filter_cutoff_hz.setValue(1234.5)
    assert panel.filter_cutoff_hz.value() == pytest.approx(1234.5)


# ================================================================================================
# The rise-time readout follows the SELECTED design
# ================================================================================================

def _rise_us(panel) -> float:
    text = panel.rise_time_label.text()
    assert text.startswith("rise time: "), text
    return float(text.split("rise time: ")[1].split(" ")[0])


def test_rise_time_readout_changes_with_the_kind(panel):
    panel.set_recording(FS)
    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(3_000.0)

    panel.set_filter_kind("bessel")
    bessel = _rise_us(panel)
    assert "Bessel" in panel.rise_time_label.text()

    panel.set_filter_kind("butter")
    butter = _rise_us(panel)
    assert "Butterworth" in panel.rise_time_label.text()

    panel.set_filter_kind("gaussian")
    gaussian = _rise_us(panel)
    assert "Gaussian" in panel.rise_time_label.text()
    # Gaussian has no order, so the readout must not claim one.
    assert "pole" not in panel.rise_time_label.text()

    assert len({round(bessel), round(butter), round(gaussian)}) == 3


def test_rise_time_readout_changes_with_the_order(panel):
    panel.set_recording(FS)
    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(3_000.0)
    panel.set_filter_kind("bessel")

    panel.set_filter_order(1)
    first = _rise_us(panel)
    assert "1-pole Bessel" in panel.rise_time_label.text()

    panel.set_filter_order(8)
    eighth = _rise_us(panel)
    assert "8-pole Bessel" in panel.rise_time_label.text()
    assert first != eighth


def test_rise_time_readout_matches_the_kernel_for_the_selected_design(panel):
    """It must be the SAME number detect_events will stamp, not an approximation."""
    panel.set_recording(FS)
    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(5_000.0)
    panel.set_filter_kind("butter")
    panel.set_filter_order(2)
    expected = filters.measure_rise_time(FS, 5_000.0, order=2, kind="butter") * 1e6
    assert _rise_us(panel) == pytest.approx(round(expected), abs=1.0)


def test_rise_time_warns_in_the_gui_when_the_cutoff_is_above_fs_over_2p5(panel):
    """The kernel's RuntimeWarning went to stderr and was invisible; the readout used to
    print a flatteringly small resolution floor for a filter doing almost nothing."""
    panel.set_recording(FS)
    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(FS / 2.2)
    assert "barely attenuates" in panel.rise_time_label.text()

    panel.filter_cutoff_hz.setValue(10_000.0)
    assert "barely attenuates" not in panel.rise_time_label.text()


def test_rise_time_refuses_a_probe_too_large_to_measure_live(panel):
    """~40*fs/cutoff samples: at 1 MHz a 1 Hz cutoff is a 40 M-sample step. The readout
    says so instead of freezing the GUI; detection still measures it off-thread."""
    panel.set_recording(1_000_000.0)
    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(1.0)
    assert "too low to measure live" in panel.rise_time_label.text()


# ================================================================================================
# Frequency-response preview
# ================================================================================================

def test_response_curve_is_drawn_for_every_stage_that_has_one(panel):
    panel.set_recording(FS)
    for idx in (LOWPASS, HIGHPASS, NOTCH, RMEAN, SAVGOL):
        panel.filter_selector.setCurrentIndex(idx)
        x, y = panel.response_curve.getData()
        assert x is not None and len(x) > 100, f"no curve for stage {idx}"
        assert len(x) == len(y)


def test_response_curve_is_the_kernel_helpers_output_not_a_reimplementation(panel):
    panel.set_recording(FS)
    panel.filter_selector.setCurrentIndex(LOWPASS)
    panel.set_filter_kind("butter")
    panel.set_filter_order(2)
    panel.filter_cutoff_hz.setValue(4_000.0)
    x, y = panel.response_curve.getData()
    f, mag = filters.response("lowpass", FS, cutoff_hz=4_000.0, order=2, kind="butter")
    # The plot is in log-x mode, so pyqtgraph hands back log10 of the frequencies.
    assert np.allclose(x, np.log10(f))
    assert np.allclose(y, filters.response_db(mag))


def test_response_updates_live_when_the_design_changes(panel):
    panel.set_recording(FS)
    panel.filter_selector.setCurrentIndex(LOWPASS)
    panel.set_filter_kind("bessel")
    panel.set_filter_order(8)
    _, before = panel.response_curve.getData()
    panel.set_filter_order(2)
    _, after = panel.response_curve.getData()
    assert not np.allclose(before, after)


def test_response_marks_the_nominal_cutoff_and_states_the_6db_reality(panel):
    panel.set_recording(FS)
    panel.filter_selector.setCurrentIndex(LOWPASS)
    panel.filter_cutoff_hz.setValue(10_000.0)
    assert panel.response_cutoff_line.isVisible()
    # Log-x mode: the marker's position is log10 of the frequency.
    assert panel.response_cutoff_line.value() == pytest.approx(np.log10(10_000.0))
    note = panel.response_note.text()
    assert "-6.0 dB" in note and "SQUARED" in note
    assert "true -3 dB corner is 7140 Hz" in note


def test_median_gets_an_explanation_instead_of_a_fabricated_curve(panel):
    """A sliding median is non-linear: any |H(f)| drawn for it would be a lie."""
    panel.set_recording(FS)
    panel.filter_selector.setCurrentIndex(MEDIAN)
    x, _ = panel.response_curve.getData()
    assert x is None or len(x) == 0
    assert not panel.response_cutoff_line.isVisible()
    assert "non-linear" in panel.response_note.text().lower()


def test_baseline_drift_gets_an_explanation_instead_of_a_curve(panel):
    panel.set_recording(FS)
    panel.filter_selector.setCurrentIndex(DETREND)
    x, _ = panel.response_curve.getData()
    assert x is None or len(x) == 0
    assert "estimator" in panel.response_note.text().lower()


def test_response_says_to_select_a_channel_when_the_rate_is_unknown(panel):
    panel.set_recording(None)
    panel.filter_selector.setCurrentIndex(LOWPASS)
    x, _ = panel.response_curve.getData()
    assert x is None or len(x) == 0
    assert "sample rate" in panel.response_note.text()


def test_response_note_flags_a_smoother_that_is_narrower_than_the_lowpass(panel):
    """The Savitzky-Golay default band-limits ~8x harder than the 10 kHz low-pass, and
    only the low-pass has a rise-time readout. The note has to name that."""
    panel.set_recording(FS)
    panel.filter_selector.setCurrentIndex(SAVGOL)
    note = panel.response_note.text()
    assert "Effective -3 dB" in note
    assert "low-pass" in note


def test_response_note_calls_out_a_running_mean_that_is_the_identity(panel):
    """A 10 us window is 1 sample below 150 kHz: enabled, marked active, does nothing."""
    panel.set_recording(50_000.0)
    panel.filter_selector.setCurrentIndex(RMEAN)
    panel.rmean_window_s.setValue(1e-5)
    assert "IDENTITY" in panel.response_note.text()


# ================================================================================================
# The preview-failure readout
# ================================================================================================

def test_preview_error_is_shown_and_cleared(panel):
    assert not panel.preview_error_label.isVisible()
    panel.set_preview_error("window_s=60s is longer than the recording (2s)")
    assert panel.preview_error_label.isVisibleTo(panel)
    assert "longer than the recording" in panel.preview_error_label.text()
    panel.set_preview_error("")
    assert not panel.preview_error_label.isVisibleTo(panel)
