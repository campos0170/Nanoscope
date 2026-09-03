"""Panel-widget tests (PORT_PLAN §6/§8, pytest-qt, offscreen).

Pure-view behaviour of the ported control panels, exercised headless:

* DetectionPanel: ``k_out < k_in`` enforced live after adversarial edits; the
  ``None`` off-sentinels; ``params()`` is a subset of the real ``detect_events``
  kwargs (so the module never gets a key the detector rejects); the rise-time
  readout is monotonic in cutoff and tolerates an unknown sample rate; the
  ``requires_time_base`` gate disables the run buttons with an honest reason.
* StatsPanel: summary / FDR / recovery-gated efficiency / fit request.
* EventTable: curation writes back into the DataFrame in place and row selection
  emits ``eventSelected``.
* SweepDialog: the "use recommended k" button enables only with a plateau.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.detect_cusum import detect_events_cusum
from nanoscope.analysis.events import EventSet
from nanoscope.gui.panels import DetectionPanel, EventTable, StatsPanel, SweepDialog

FS = 50_000.0


def _events() -> EventSet:
    """A tiny real EventSet (three square blockades) via the actual detector."""
    rng = np.random.default_rng(0)
    n = 20_000
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in [(2_000, 2_200), (6_000, 6_300), (12_000, 12_250)]:
        x[s:e] -= 30.0
    df, prov = detect_events(x.astype(np.float32), FS, min_dwell_s=5e-5)
    return EventSet(df, prov)


# ================================================================================================
# DetectionPanel
# ================================================================================================

def test_k_out_clamped_below_k_in_when_lowering_k_in(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.k_in.setValue(5.0)
    panel.k_out.setValue(2.0)
    # Adversarial: drop k_in below the current k_out.
    panel.k_in.setValue(1.5)
    assert panel.k_out.value() < panel.k_in.value()


def test_k_out_cannot_exceed_k_in(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.k_in.setValue(5.0)
    # Adversarial: try to push k_out far above k_in.
    panel.k_out.setValue(40.0)
    assert panel.k_out.value() < panel.k_in.value()


def test_params_off_sentinels(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    p = panel.params()
    # Defaults: filter disabled, merge 0, max_dwell 0 -> all None sentinels.
    assert p["filter_cutoff_hz"] is None
    assert p["merge_gap_s"] is None
    assert p["max_dwell_s"] is None

    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(8_000.0)
    panel.merge_gap_ms.setValue(0.1)  # 0.1 ms == the 1e-4 s this key carries
    panel.max_dwell_s.setValue(0.25)
    p = panel.params()
    assert p["filter_cutoff_hz"] == pytest.approx(8_000.0)
    assert p["merge_gap_s"] == pytest.approx(1e-4)
    assert p["max_dwell_s"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "module_id, detector_fn",
    [("detect.schmitt", detect_events), ("detect.cusum", detect_events_cusum)],
)
def test_params_are_subset_of_detect_events_kwargs(qtbot, module_id, detector_fn):
    """The module forwards these straight into its detector; every key it does NOT
    consume itself must be a real parameter of THAT detector or the call raises a
    TypeError. The ``detrend_*`` keys are consumed by the module (baseline-wander
    removal happens before detection), so they are stripped, not forwarded.

    THE INVARIANT WAS REPHRASED, NOT WEAKENED, WHEN THE SECOND DETECTOR LANDED. It used
    to read "params() is a subset of detect_events' kwargs", which stops being a true
    statement the moment one form carries two detectors' thresholds: ``cusum_h`` is not
    a ``detect_events`` parameter and ``k_in`` is not a ``detect_events_cusum`` one.
    Weakening it to a union would have made it pass while permitting exactly the defect
    it exists to catch -- a Schmitt run receiving ``cusum_h``.

    So it is now checked ONCE PER DETECTOR, against the function that will actually
    receive the dict, through the SAME `_detect_kwargs` the module uses. That is
    strictly stronger than the original: it covers two detectors instead of one, and it
    exercises the foreign-key strip (`_DETECTOR_PRIVATE_KEYS`) rather than assuming it.
    """
    import inspect

    from nanoscope.analysis.modules import _detect_kwargs

    panel = DetectionPanel()
    qtbot.addWidget(panel)
    accepted = set(inspect.signature(detector_fn).parameters)
    forwarded = set(_detect_kwargs(panel.params(), module_id))
    assert forwarded <= accepted, sorted(forwarded - accepted)


def test_neither_detectors_thresholds_reach_the_other(qtbot):
    """The other half of the invariant above, stated positively.

    `params()` carries BOTH forms so switching detectors does not discard the one you
    are not using -- which is only safe because the module strips the foreign pair. If
    that strip ever stopped happening, the assertion above would still pass for the keys
    each function DOES accept while a Schmitt run silently received a CUSUM threshold;
    this pins that neither ever appears in the other's kwargs.
    """
    from nanoscope.analysis.modules import _detect_kwargs

    panel = DetectionPanel()
    qtbot.addWidget(panel)
    params = panel.params()
    assert {"k_in", "k_out", "cusum_k", "cusum_h"} <= set(params)

    schmitt = _detect_kwargs(params, "detect.schmitt")
    cusum = _detect_kwargs(params, "detect.cusum")
    assert {"k_in", "k_out"} <= set(schmitt)
    assert not {"cusum_k", "cusum_h"} & set(schmitt)
    assert {"cusum_k", "cusum_h"} <= set(cusum)
    assert not {"k_in", "k_out"} & set(cusum)
    # And "detector" is not a kwarg of either -- it names the module, it is not a knob.
    assert "detector" not in schmitt and "detector" not in cusum


def test_polarity_choice_mapping(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.polarity_override.setCurrentIndex(0)
    assert panel.polarity_choice() is None
    panel.polarity_override.setCurrentIndex(1)
    assert panel.polarity_choice() == -1
    panel.polarity_override.setCurrentIndex(2)
    assert panel.polarity_choice() == 1


def _rise_us(label_text: str) -> int | None:
    m = re.search(r"rise time:\s*(\d+)", label_text)
    return int(m.group(1)) if m else None


def test_rise_time_readout_monotonic_in_cutoff(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_recording(FS)
    panel.filter_enabled.setChecked(True)

    panel.filter_cutoff_hz.setValue(5_000.0)
    low_cut = _rise_us(panel.rise_time_label.text())
    panel.filter_cutoff_hz.setValue(15_000.0)
    high_cut = _rise_us(panel.rise_time_label.text())

    assert low_cut is not None and high_cut is not None
    # A higher cutoff resolves faster -> a SHORTER rise time.
    assert high_cut < low_cut


def test_set_recording_none_is_tolerated(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_recording(None)   # must NOT raise (legacy-TDMS rate-unknown case)
    panel.filter_enabled.setChecked(True)
    assert "unknown" in panel.rise_time_label.text().lower()


def test_set_runnable_gates_buttons_with_reason(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_runnable(False, "Sample rate unknown — detection disabled.")
    assert not panel.detect_button.isEnabled()
    assert not panel.sweep_button.isEnabled()
    assert not panel._gate_label.isHidden()
    assert "unknown" in panel._gate_label.text().lower()

    panel.set_runnable(True, "")
    assert panel.detect_button.isEnabled()
    assert panel._gate_label.isHidden()


def test_detect_requested_emits_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    got: list[dict] = []
    panel.detectRequested.connect(got.append)
    panel.detect_button.click()
    assert len(got) == 1
    # GREW (never changed) when the hard-coded kernel parameters were exposed: the
    # low-pass/high-pass design (order + kind) and the baseline-drift methods' own shape
    # parameters. Every addition defaults to the value that used to be hard-coded.
    # GREW again with the owner-requested Normalize stage, which defaults off.
    # GREW again for the SECOND DETECTOR and the three estimators that were implemented
    # but unreachable: ``detector`` names which detector runs, ``cusum_k``/``cusum_h``
    # are its thresholds, ``detrend_rough_s``/``base_s``/``mask_sigma``/``dilate_ms``/
    # ``decim`` are the masked and robust medians' shape, ``detrend_sg_*`` are the
    # Savitzky-Golay baseline's, ``notch_freqs`` is the explicit line list and
    # ``normalize_clip_sigma`` is the model-input clip. Every one of them defaults to
    # the value that was previously frozen, so an untouched panel runs what it always
    # ran.
    assert set(got[0]) == {"detector",
                           "polarity", "k_in", "k_out", "cusum_k", "cusum_h",
                           "min_dwell_s",
                           "max_dwell_s", "filter_cutoff_hz", "merge_gap_s",
                           "filter_order", "filter_kind",
                           "detrend_enabled", "detrend_method", "detrend_window_s",
                           "detrend_percentile", "detrend_asls_lam", "detrend_asls_p",
                           "detrend_asls_niter", "detrend_asls_decim",
                           "detrend_rough_s", "detrend_base_s", "detrend_mask_sigma",
                           "detrend_dilate_ms", "detrend_decim",
                           "detrend_sg_poly", "detrend_sg_decim",
                           "notch_enabled", "notch_freq_hz", "notch_harmonics",
                           "notch_q", "notch_freqs",
                           "median_enabled", "median_window_s",
                           "hp_enabled", "hp_cutoff_hz", "hp_order", "hp_kind",
                           "rmean_enabled", "rmean_window_s",
                           "savgol_enabled", "savgol_window_s", "savgol_polyorder",
                           "normalize_enabled", "normalize_mode",
                           "normalize_clip_sigma"}


def test_filter_preview_toggle_signal(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    # Preview is meaningless until the filter is enabled.
    assert not panel.filter_preview.isEnabled()
    panel.filter_enabled.setChecked(True)
    assert panel.filter_preview.isEnabled()

    toggles: list[bool] = []
    panel.previewFilterToggled.connect(toggles.append)
    panel.filter_preview.setChecked(True)
    assert toggles == [True]
    # Disabling the filter clears the preview (emits False).
    panel.filter_enabled.setChecked(False)
    assert toggles[-1] is False
    assert not panel.filter_preview.isEnabled()


# -- Baseline drift (detrend) --------------------------------------------------

def test_detrend_params_exposed_and_default_off(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    p = panel.params()
    assert p["detrend_enabled"] is False
    assert p["detrend_method"] == "block_median"
    assert p["detrend_window_s"] == pytest.approx(0.5)


def test_enabling_detrend_changes_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.detrend_enabled.setChecked(True)
    panel.set_detrend_method("percentile")
    panel.detrend_window_s.setValue(1.25)
    p = panel.params()
    assert p["detrend_enabled"] is True
    assert p["detrend_method"] == "percentile"
    assert p["detrend_window_s"] == pytest.approx(1.25)


def test_detrend_method_choice_mapping(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.detrend_method.setCurrentIndex(0)
    assert panel.detrend_method_choice() == "block_median"
    panel.detrend_method.setCurrentIndex(1)
    assert panel.detrend_method_choice() == "percentile"
    panel.detrend_method.setCurrentIndex(2)
    assert panel.detrend_method_choice() == "asls"
    # set_detrend_method selects by method string; unknown strings are ignored.
    panel.set_detrend_method("block_median")
    assert panel.detrend_method.currentIndex() == 0
    panel.set_detrend_method("nonsense")
    assert panel.detrend_method.currentIndex() == 0


def test_detrend_enables_preview_toggle_without_lowpass(qtbot):
    """Baseline removal alone (low-pass off) still drives the 'show on trace' preview,
    so the user can see the flattened trace."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert not panel.filter_preview.isEnabled()
    panel.detrend_enabled.setChecked(True)
    assert panel.filter_preview.isEnabled()
    # Turning it back off (with the low-pass also off) disables the preview again.
    panel.detrend_enabled.setChecked(False)
    assert not panel.filter_preview.isEnabled()


def test_detrend_preview_stays_enabled_while_either_stage_on(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.filter_enabled.setChecked(True)
    panel.detrend_enabled.setChecked(True)
    assert panel.filter_preview.isEnabled()
    panel.filter_enabled.setChecked(False)   # detrend still on
    assert panel.filter_preview.isEnabled()
    panel.detrend_enabled.setChecked(False)  # now both off
    assert not panel.filter_preview.isEnabled()


def test_detrend_emits_params_changed(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    got: list[dict] = []
    panel.paramsChanged.connect(got.append)
    panel.detrend_enabled.setChecked(True)
    assert got and got[-1]["detrend_enabled"] is True
    panel.detrend_window_s.setValue(0.8)
    assert got[-1]["detrend_window_s"] == pytest.approx(0.8)


# -- High-pass / Notch / Savitzky-Golay ----------------------------------------

def test_new_filter_params_exposed_and_default_off(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    p = panel.params()
    # notch
    assert p["notch_enabled"] is False
    assert p["notch_freq_hz"] == pytest.approx(60.0)
    assert p["notch_harmonics"] == 3
    assert p["notch_q"] == pytest.approx(30.0)
    # high-pass
    assert p["hp_enabled"] is False
    assert p["hp_cutoff_hz"] == pytest.approx(100.0)
    # median
    assert p["median_enabled"] is False
    assert p["median_window_s"] == pytest.approx(1e-4)
    # running mean
    assert p["rmean_enabled"] is False
    assert p["rmean_window_s"] == pytest.approx(5e-4)
    # savgol
    assert p["savgol_enabled"] is False
    assert p["savgol_window_s"] == pytest.approx(1e-3)
    assert p["savgol_polyorder"] == 3


def test_enabling_notch_changes_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_notch_enabled(True)
    panel.set_notch_freq_hz(50.0)
    panel.set_notch_harmonics(5)
    panel.set_notch_q(45.0)
    p = panel.params()
    assert p["notch_enabled"] is True
    assert p["notch_freq_hz"] == pytest.approx(50.0)
    assert p["notch_harmonics"] == 5
    assert p["notch_q"] == pytest.approx(45.0)


def test_enabling_highpass_changes_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_hp_enabled(True)
    panel.set_hp_cutoff_hz(250.0)
    p = panel.params()
    assert p["hp_enabled"] is True
    assert p["hp_cutoff_hz"] == pytest.approx(250.0)


def test_enabling_savgol_changes_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_savgol_enabled(True)
    panel.set_savgol_window_s(2e-3)
    panel.set_savgol_polyorder(4)
    p = panel.params()
    assert p["savgol_enabled"] is True
    assert p["savgol_window_s"] == pytest.approx(2e-3)
    assert p["savgol_polyorder"] == 4


def test_enabling_median_changes_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_median_enabled(True)
    panel.set_median_window_s(3e-4)
    p = panel.params()
    assert p["median_enabled"] is True
    assert p["median_window_s"] == pytest.approx(3e-4)


def test_enabling_rmean_changes_params(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_rmean_enabled(True)
    panel.set_rmean_window_s(1.5e-3)
    p = panel.params()
    assert p["rmean_enabled"] is True
    assert p["rmean_window_s"] == pytest.approx(1.5e-3)


@pytest.mark.parametrize(
    "attr", ["notch_enabled", "median_enabled", "hp_enabled", "rmean_enabled", "savgol_enabled"]
)
def test_each_new_filter_enables_preview_toggle(qtbot, attr):
    """Any pre-detection filter alone (low-pass and detrend off) drives the 'show on
    trace' preview, so the user can see the conditioned trace."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert not panel.filter_preview.isEnabled()
    getattr(panel, attr).setChecked(True)
    assert panel.filter_preview.isEnabled()
    getattr(panel, attr).setChecked(False)
    assert not panel.filter_preview.isEnabled()


def test_preview_stays_enabled_across_mixed_stages(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.notch_enabled.setChecked(True)
    assert panel.filter_preview.isEnabled()
    panel.filter_enabled.setChecked(True)    # low-pass also on
    assert panel.filter_preview.isEnabled()
    panel.notch_enabled.setChecked(False)    # low-pass still on
    assert panel.filter_preview.isEnabled()
    panel.filter_enabled.setChecked(False)   # now every stage off
    assert not panel.filter_preview.isEnabled()


@pytest.mark.parametrize(
    "attr", ["notch_enabled", "median_enabled", "hp_enabled", "rmean_enabled", "savgol_enabled"]
)
def test_new_filter_enable_emits_params_changed(qtbot, attr):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    got: list[dict] = []
    panel.paramsChanged.connect(got.append)
    getattr(panel, attr).setChecked(True)
    assert got and got[-1][attr] is True


def test_highpass_tooltip_warns_about_square_event_distortion(qtbot):
    """The high-pass control must WARN it distorts square events (drift/DC removal only)."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    tip = panel.hp_enabled.toolTip().lower()
    assert "distort" in tip
    assert "square" in tip or "biphasic" in tip


# ================================================================================================
# StatsPanel
# ================================================================================================

def test_stats_summary_and_fdr(qtbot):
    panel = StatsPanel()
    qtbot.addWidget(panel)
    panel.set_summary({
        "n_events": 12, "n_accepted": 10, "rate_hz": 5.5,
        "median_dwell_us": 123.4, "median_frac_depth": 0.21,
        "n_filter_limited": 1, "n_pores": 1,
    })
    assert panel.summary_table.item(0, 0).text() == "12"
    assert panel.summary_table.item(1, 0).text() == "10"

    panel.set_fdr({"estimated_fdr": 0.05, "n_surrogates": 5})
    assert "5.0%" in panel.fdr_label.text()


def test_stats_recovery_enables_efficiency(qtbot):
    panel = StatsPanel()
    qtbot.addWidget(panel)
    assert not panel.use_efficiency.isEnabled()
    panel.set_recovery_available(True)
    assert panel.use_efficiency.isEnabled()
    panel.set_recovery_available(False)
    assert not panel.use_efficiency.isEnabled()


def test_stats_fit_requested(qtbot):
    panel = StatsPanel()
    qtbot.addWidget(panel)
    got: list[dict] = []
    panel.fitRequested.connect(got.append)
    panel.n_components_dwell.setValue(2)
    # The Fit button is GATED on there being a detection to fit (it used to accept the click
    # and report the failure into its own results box afterwards), so open the gate first.
    panel.set_detection_present(True)
    panel.fit_button.click()
    assert len(got) == 1
    assert got[0]["n_components_dwell"] == 2
    assert set(got[0]) == {
        "n_components_dwell", "n_components_depth", "use_efficiency", "use_filter_limited",
    }
    # Filter-limited events are EXCLUDED from the distribution fits unless asked for.
    assert got[0]["use_filter_limited"] is False


# ================================================================================================
# EventTable
# ================================================================================================

def test_event_table_loads_and_curates_in_place(qtbot):
    table = EventTable()
    qtbot.addWidget(table)
    events = _events()
    n = len(events)
    assert n >= 3
    table.set_events(events)
    assert table._proxy.rowCount() == n

    changed: list[tuple[int, bool]] = []
    table.acceptanceChanged.connect(lambda i, a: changed.append((i, a)))

    # Reject row 0 through the model; the write must land in the live DataFrame.
    table._model.set_accepted(0, False)
    assert bool(table.dataframe["accepted"].iloc[0]) is False
    assert changed and changed[-1][1] is False
    ev_id = int(table.dataframe["event_id"].iloc[0])
    assert changed[-1][0] == ev_id


def test_event_table_row_selection_emits_event_id(qtbot):
    table = EventTable()
    qtbot.addWidget(table)
    events = _events()
    table.set_events(events)

    selected: list[int] = []
    table.eventSelected.connect(selected.append)
    table.view.selectRow(0)
    assert selected
    # The emitted id is the event_id of the (source) row 0.
    assert selected[-1] == int(events.df["event_id"].iloc[0])


# ================================================================================================
# SweepDialog
# ================================================================================================

def test_sweep_dialog_enables_use_button_with_plateau(qtbot):
    df = pd.DataFrame({"k": [1.0, 2.0, 3.0, 4.0, 5.0],
                       "n_events": [200.0, 60.0, 48.0, 47.0, 46.0],
                       "n_surrogate": [5.0, 1.0, 0.0, 0.0, 0.0]})
    df.attrs["k_recommended"] = 3.5
    dlg = SweepDialog(df)
    qtbot.addWidget(dlg)
    assert dlg.use_button.isEnabled()

    chosen: list[float] = []
    dlg.kChosen.connect(chosen.append)
    dlg.use_button.click()
    assert chosen == [pytest.approx(3.5)]


def test_sweep_dialog_disables_use_button_without_plateau(qtbot):
    df = pd.DataFrame({"k": [1.0, 2.0, 3.0], "n_events": [200.0, 120.0, 60.0]})
    # No k_recommended in attrs -> no plateau found.
    dlg = SweepDialog(df)
    qtbot.addWidget(dlg)
    assert not dlg.use_button.isEnabled()
