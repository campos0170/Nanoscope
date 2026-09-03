"""DetectionPanel four-tab control (pytest-qt, offscreen).

Preprocessing (the signal-filter chain) is SEPARATE from detection, and each detector gets
its OWN tab. The panel's body is a QTabWidget (``mode_tabs``) with four pages, in order:

* **Preprocess** — the "Signal filters" group (selector + stacked pages + active-filters
  summary), the "show processed trace" preview toggle, and the rise-time readout.
* **Detection** — the Schmitt "Detection" group (k_in/k_out/min/max dwell/merge gap), the
  "Polarity" group, and the Detect + Sweep/Null/Recovery buttons.
* **ML detector** — the ML group (Load model + label + enter/exit + range + Run + gate).
* **Manual measurement** — an EMPTY container here. The measurement tool needs the Session
  and the shared TraceView, and ``panels.py`` imports no sibling gui module, so the owning
  AnalysisPanel installs its widget through ``set_measure_widget``. A bare
  ``DetectionPanel()`` still has four tabs, the last one simply blank. What lands in it,
  and that nothing about the tool broke in the move, is pinned in ``test_measure_tab.py``.

This is a LAYOUT-ONLY reshuffle: the existing widgets are REPARENTED into the tab pages,
so the contract is that every widget stays alive and reachable by attribute; ``params()``
and ``ml_params()`` read every control regardless of the active tab; a control on a hidden
tab stays enabled and its setters still land; and the emitted signals / gating are
untouched. These tests pin that behaviour, re-check a couple of pre-existing param/detect
assertions to prove the contract held, and pin the workflow-rail inner-tab routing wired
through a real MainWindow.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QTabWidget, QWidget

from nanoscope.analysis import ml_detect
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.workflow_status import Stage

FS = 50_000.0
N = 20_000

TAB_TITLES = ["Preprocess", "Detection", "ML detector", "Manual measurement"]

# The three groups of widget attributes, keyed by the tab page they must live on.
PREPROCESS_ATTRS = [
    "filter_selector", "filter_stack", "active_filters_label", "filter_preview",
    "filter_enabled", "filter_cutoff_hz", "rise_time_label",
    "hp_enabled", "hp_cutoff_hz",
    "notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q",
    "median_enabled", "median_window_s",
    "detrend_enabled", "detrend_method", "detrend_window_s",
    "rmean_enabled", "rmean_window_s",
    "savgol_enabled", "savgol_window_s", "savgol_polyorder",
    "normalize_enabled", "normalize_mode",
]
DETECTION_ATTRS = [
    # ``merge_gap_ms``, not ``merge_gap_s``: the widget reads MILLISECONDS now (the
    # params() KEY is still merge_gap_s in seconds — see PARAMS_KEYS below).
    "k_in", "k_out", "min_dwell_s", "max_dwell_s", "merge_gap_ms",
    "polarity_readout", "polarity_override",
    "detect_button", "sweep_button", "null_button", "recovery_button",
]
ML_ATTRS = [
    "ml_load_button", "ml_model_label", "ml_enter", "ml_exit", "ml_range",
    "ml_detect_button", "ml_gate_label",
]

# NOTE: grown (not changed) when the hard-coded kernel parameters were exposed —
# ``filter_order`` / ``filter_kind`` (real detect_events kwargs; the detector owns the
# low-pass), ``hp_order`` / ``hp_kind``, and the baseline-drift methods' own shape
# parameters. Every one defaults to the value that was previously frozen, so the panel's
# behaviour is unchanged; only the key set grew.
PARAMS_KEYS = {
    "polarity", "k_in", "k_out", "min_dwell_s", "max_dwell_s", "filter_cutoff_hz",
    "filter_order", "filter_kind",
    "merge_gap_s", "detrend_enabled", "detrend_method", "detrend_window_s",
    "detrend_percentile", "detrend_asls_lam", "detrend_asls_p", "detrend_asls_niter",
    "detrend_asls_decim",
    "notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q",
    "median_enabled", "median_window_s", "hp_enabled", "hp_cutoff_hz",
    "hp_order", "hp_kind",
    "rmean_enabled", "rmean_window_s", "savgol_enabled", "savgol_window_s",
    "savgol_polyorder",
    # The owner-requested Normalize stage: off by default, and the only stage in the
    # chain that changes the UNIT of the trace the detector measures on.
    "normalize_enabled", "normalize_mode",
    # Round 4. ``detector`` names WHICH detector the Detect button starts
    # ("detect.schmitt" | "detect.cusum"); ``cusum_k``/``cusum_h`` are that detector's two
    # thresholds, in ROBUST SIGMA UNITS, not picoamps. The rest are the shape parameters of
    # the baseline methods the panel can now select (masked/robust median, Savitzky-Golay)
    # plus the explicit notch line list and the model-input normalization clip. Every one
    # defaults to the previously frozen behaviour; only the key set grew.
    "detector", "cusum_k", "cusum_h",
    "detrend_rough_s", "detrend_base_s", "detrend_mask_sigma", "detrend_dilate_ms",
    "detrend_decim", "detrend_sg_poly", "detrend_sg_decim",
    "notch_freqs", "normalize_clip_sigma",
}
ML_PARAMS_KEYS = {
    "model_path", "enter", "exit", "range_source",
    # The ML detector's OWN preprocessing/decode contract (ml_preprocess presets).
    # NOT the general Preprocess-tab keys — an ML run never touches that chain.
    "ml_preset", "ml_preprocess", "ml_decode",
    "ml_preset_customized", "ml_preset_deviations",
}


# ================================================================================================
# helpers
# ================================================================================================

def _page_of(panel: DetectionPanel, attr: str) -> QWidget | None:
    """Walk up from ``panel.<attr>`` to whichever of the four tab pages contains it
    (None if it hangs off the panel root instead of any page)."""
    pages = {panel._preprocess_page, panel._detection_page, panel._ml_page,
             panel._measure_page}
    w = getattr(panel, attr)
    node = w
    while node is not None:
        if node in pages:
            return node
        node = node.parent()
    return None


def _set_mixed_filter_state(panel: DetectionPanel) -> None:
    """A scattered, non-default preprocessing + Schmitt state (via a mix of checkbox and
    setter access) so a stable-params assertion is meaningful."""
    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(8_000.0)
    panel.set_hp_enabled(True)
    panel.set_hp_cutoff_hz(250.0)
    panel.set_notch_enabled(True)
    panel.set_notch_freq_hz(50.0)
    panel.set_notch_harmonics(5)
    panel.set_notch_q(45.0)
    panel.set_median_enabled(True)
    panel.set_median_window_s(3e-4)
    panel.detrend_enabled.setChecked(True)
    panel.set_detrend_method("percentile")
    panel.detrend_window_s.setValue(1.25)
    panel.set_rmean_enabled(True)
    panel.set_rmean_window_s(1.5e-3)
    panel.set_savgol_enabled(True)
    panel.set_savgol_window_s(2e-3)
    panel.set_savgol_polyorder(4)
    panel.k_in.setValue(6.5)
    panel.k_out.setValue(1.5)
    panel.min_dwell_s.setValue(3e-5)
    panel.max_dwell_s.setValue(0.25)
    panel.merge_gap_ms.setValue(0.1)  # ms; params() emits 1e-4 s
    panel.polarity_override.setCurrentIndex(1)


def _current_doc(tmp_path, *, name="rec.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(np.zeros(N, np.float32),))


def _install(win: MainWindow, doc: TraceDocument) -> str:
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    return doc_id


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


# ================================================================================================
# structure: a four-tab QTabWidget, Preprocess / Detection / ML detector / Manual measurement
# ================================================================================================

def test_mode_tabs_is_a_four_tab_qtabwidget_titled_in_order(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert isinstance(panel.mode_tabs, QTabWidget)
    assert panel.mode_tabs.count() == 4
    assert [panel.mode_tabs.tabText(i) for i in range(4)] == TAB_TITLES
    # Preprocess is the default active tab (index 0) — its filter widgets are the ones
    # shown when the panel opens, which a couple of pre-existing tests rely on.
    assert panel.mode_tabs.currentIndex() == 0


def test_the_measure_page_is_an_empty_container_until_an_owner_fills_it(qtbot):
    """The tab exists and is titled whether or not anything is installed in it, so its
    index cannot depend on who constructed the panel; ``set_measure_widget`` is the one
    way in. panels.py never builds the measurement tool itself — it cannot, without
    importing a sibling gui panel."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.mode_tabs.widget(3) is panel._measure_page
    assert panel._measure_page.layout().count() == 0

    installed = QWidget()
    panel.set_measure_widget(installed)
    assert installed.parentWidget() is panel._measure_page
    assert panel.mode_tabs.count() == 4          # installed INTO the page, not as a 5th tab
    assert [panel.mode_tabs.tabText(i) for i in range(4)] == TAB_TITLES


@pytest.mark.parametrize("attr", PREPROCESS_ATTRS)
def test_filter_widgets_live_under_the_preprocess_tab(qtbot, attr):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert _page_of(panel, attr) is panel._preprocess_page


@pytest.mark.parametrize("attr", DETECTION_ATTRS)
def test_schmitt_polarity_and_run_buttons_live_under_the_detection_tab(qtbot, attr):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert _page_of(panel, attr) is panel._detection_page


@pytest.mark.parametrize("attr", ML_ATTRS)
def test_ml_widgets_live_under_the_ml_detector_tab(qtbot, attr):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert _page_of(panel, attr) is panel._ml_page


def test_the_three_groups_do_not_bleed_across_tabs(qtbot):
    """No filter widget on the Detection/ML page, no Schmitt widget on Preprocess/ML, etc.
    — and nothing the detector owns leaks onto the Manual measurement page."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    for attr in PREPROCESS_ATTRS:
        assert _page_of(panel, attr) not in (panel._detection_page, panel._ml_page,
                                             panel._measure_page)
    for attr in DETECTION_ATTRS:
        assert _page_of(panel, attr) not in (panel._preprocess_page, panel._ml_page,
                                             panel._measure_page)
    for attr in ML_ATTRS:
        assert _page_of(panel, attr) not in (panel._preprocess_page, panel._detection_page,
                                             panel._measure_page)


def test_active_tab_shows_only_its_own_widgets_when_panel_shown(qtbot):
    """Real isVisible (panel shown): only the raised tab's widgets are visible — a clean
    separation of preprocessing, detection, ML and manual measurement into their own tabs."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    installed = QWidget()
    panel.set_measure_widget(installed)
    panel.show()
    qtbot.waitExposed(panel)

    panel.show_preprocess_tab()
    assert panel.filter_cutoff_hz.isVisible()
    assert not panel.detect_button.isVisible()
    assert not panel.ml_detect_button.isVisible()
    assert not installed.isVisible()

    panel.show_detection_tab()
    assert panel.detect_button.isVisible()
    assert not panel.filter_cutoff_hz.isVisible()
    assert not panel.ml_detect_button.isVisible()
    assert not installed.isVisible()

    panel.show_ml_tab()
    assert panel.ml_detect_button.isVisible()
    assert not panel.filter_cutoff_hz.isVisible()
    assert not panel.detect_button.isVisible()
    assert not installed.isVisible()

    panel.show_measure_tab()
    assert installed.isVisible()
    assert not panel.filter_cutoff_hz.isVisible()
    assert not panel.detect_button.isVisible()
    assert not panel.ml_detect_button.isVisible()


# ================================================================================================
# show_*_tab() raises the matching page
# ================================================================================================

def test_show_tab_helpers_raise_the_right_index(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.show_detection_tab()
    assert panel.mode_tabs.currentIndex() == 1
    assert panel.mode_tabs.currentWidget() is panel._detection_page
    panel.show_ml_tab()
    assert panel.mode_tabs.currentIndex() == 2
    assert panel.mode_tabs.currentWidget() is panel._ml_page
    panel.show_measure_tab()
    assert panel.mode_tabs.currentIndex() == 3
    assert panel.mode_tabs.currentWidget() is panel._measure_page
    panel.show_preprocess_tab()
    assert panel.mode_tabs.currentIndex() == 0
    assert panel.mode_tabs.currentWidget() is panel._preprocess_page


# ================================================================================================
# params() / ml_params() are identical no matter which tab is active
# ================================================================================================

def test_params_identical_no_matter_which_tab_is_active(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    _set_mixed_filter_state(panel)

    panel.show_preprocess_tab()
    reference = dict(panel.params())
    assert set(reference) == PARAMS_KEYS
    # Switch through every tab — including the Manual measurement one, which shares the
    # strip but contributes nothing to the detector's params. params() must not budge.
    for show in (panel.show_detection_tab, panel.show_ml_tab, panel.show_measure_tab,
                 panel.show_preprocess_tab):
        show()
        assert panel.params() == reference


def test_ml_params_identical_no_matter_which_tab_is_active(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_model_path("/models/detector.keras")
    panel.ml_enter.setValue(0.8)
    panel.ml_exit.setValue(0.4)
    panel.ml_range.setCurrentIndex(1)   # Whole recording

    panel.show_ml_tab()
    reference = dict(panel.ml_params())
    assert set(reference) == ML_PARAMS_KEYS
    assert {k: reference[k] for k in ("model_path", "enter", "exit", "range_source")} == {
        "model_path": "/models/detector.keras", "enter": 0.8,
        "exit": 0.4, "range_source": "whole"}
    # The model pipeline rides along, and its thresholds mirror the enter/exit spins.
    assert reference["ml_decode"]["enter_threshold"] == pytest.approx(0.8)
    assert reference["ml_decode"]["exit_threshold"] == pytest.approx(0.4)
    for show in (panel.show_preprocess_tab, panel.show_detection_tab,
                 panel.show_measure_tab, panel.show_ml_tab):
        show()
        assert panel.ml_params() == reference


# ================================================================================================
# setters land in params()/ml_params() even when a DIFFERENT tab is active
# ================================================================================================

def test_filter_setters_land_in_params_with_a_non_preprocess_tab_active(qtbot):
    """A filter control on the (hidden) Preprocess page stays live: its setter lands in
    params() while the ML tab is the one raised."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.show_ml_tab()   # Preprocess page hidden

    panel.set_notch_enabled(True)
    panel.set_notch_harmonics(7)
    panel.set_hp_enabled(True)
    panel.set_savgol_enabled(True)
    panel.set_savgol_polyorder(5)
    panel.detrend_enabled.setChecked(True)

    p = panel.params()
    assert p["notch_enabled"] is True and p["notch_harmonics"] == 7
    assert p["hp_enabled"] is True
    assert p["savgol_enabled"] is True and p["savgol_polyorder"] == 5
    assert p["detrend_enabled"] is True
    # The active tab never moved.
    assert panel.mode_tabs.currentIndex() == 2


def test_ml_setters_land_in_ml_params_with_a_non_ml_tab_active(qtbot, monkeypatch):
    """The ML controls on the (hidden) ML page stay live: their setters land in
    ml_params() while the Detection tab is raised."""
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.show_detection_tab()   # ML page hidden

    panel.set_model_path("/x/y/model.keras")
    panel.ml_enter.setValue(0.9)
    panel.ml_exit.setValue(0.6)
    panel.ml_range.setCurrentIndex(0)   # Visible view

    mp = panel.ml_params()
    assert {k: mp[k] for k in ("model_path", "enter", "exit", "range_source")} == {
        "model_path": "/x/y/model.keras", "enter": 0.9,
        "exit": 0.6, "range_source": "visible"}
    assert set(mp) == ML_PARAMS_KEYS
    assert panel.mode_tabs.currentIndex() == 1


# ================================================================================================
# the run buttons fire their signals from their own tabs
# ================================================================================================

def test_detect_button_fires_detect_requested_from_the_detection_tab(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.show_detection_tab()

    got: list[dict] = []
    panel.detectRequested.connect(got.append)
    panel.detect_button.click()
    assert len(got) == 1
    # Full frozen key set — the button still emits the same params() contract.
    assert set(got[0]) == PARAMS_KEYS


def test_ml_detect_button_fires_ml_detect_requested_from_the_ml_tab(qtbot, monkeypatch):
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    # Open the gate: a model is loaded AND the channel is runnable, so the Run button
    # is actually enabled and a click reaches the signal.
    panel.set_model_path("/models/detector.keras")
    panel.set_ml_runnable(True, "")
    panel.show_ml_tab()
    assert panel.ml_detect_button.isEnabled()

    got: list[dict] = []
    panel.mlDetectRequested.connect(got.append)
    panel.ml_detect_button.click()
    assert len(got) == 1
    assert set(got[0]) == ML_PARAMS_KEYS
    assert got[0]["model_path"] == "/models/detector.keras"


# ================================================================================================
# Pre-existing contract re-checks (prove the tab reshuffle kept detection behaviour)
# ================================================================================================

def test_contract_detect_requested_key_set_unchanged(qtbot):
    """Identical to the pre-existing DetectionPanel test — the detect button still emits
    exactly the frozen params() key set."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    got: list[dict] = []
    panel.detectRequested.connect(got.append)
    panel.detect_button.click()
    assert len(got) == 1
    assert set(got[0]) == PARAMS_KEYS


def test_contract_off_sentinels_and_gate(qtbot):
    """The None off-sentinels and the set_runnable gate are unchanged by the reshuffle."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    p = panel.params()
    assert p["filter_cutoff_hz"] is None
    assert p["merge_gap_s"] is None
    assert p["max_dwell_s"] is None

    panel.set_runnable(False, "Sample rate unknown — detection disabled.")
    assert not panel.detect_button.isEnabled()
    assert not panel.sweep_button.isEnabled()
    assert not panel._gate_label.isHidden()
    panel.set_runnable(True, "")
    assert panel.detect_button.isEnabled()
    assert panel._gate_label.isHidden()


# ================================================================================================
# workflow-strip routing to the inner tabs (wired through a real MainWindow)
# ================================================================================================

def test_strip_filter_reveals_analysis_tab_and_raises_the_preprocess_inner_tab(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    # Park on a different inspector tab AND a different inner tab, to prove both move.
    win._inspector_tabs.setCurrentWidget(win._cal_scroll)
    win._analysis_panel.detection_panel.show_ml_tab()

    win._workflow_strip.stageActivated.emit(Stage.FILTER.value)

    assert win._inspector_tabs.currentWidget() is win._analysis_panel
    assert win._analysis_panel.detection_panel.mode_tabs.currentIndex() == 0


def test_strip_detect_reveals_analysis_tab_and_raises_the_detection_inner_tab(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    win._inspector_tabs.setCurrentWidget(win._cal_scroll)
    win._analysis_panel.detection_panel.show_preprocess_tab()

    win._workflow_strip.stageActivated.emit(Stage.DETECT.value)

    assert win._inspector_tabs.currentWidget() is win._analysis_panel
    assert win._analysis_panel.detection_panel.mode_tabs.currentIndex() == 1
