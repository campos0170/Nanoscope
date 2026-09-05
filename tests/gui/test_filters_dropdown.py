"""DetectionPanel filter dropdown + stacked-page layout (pytest-qt, offscreen).

The pre-detection filter stages (Low-pass, High-pass, Notch, Median, Baseline drift,
Running mean, Savitzky-Golay, Normalize) used to stack as one QGroupBox each. They now
live in ONE "Signal filters" group: a combo selects which stage's controls are shown, a
QStackedWidget holds one page per stage, and an active-filters indicator (a ``● `` combo
marker plus an execution-order summary label) keeps the whole chain visible.

This is a LAYOUT-ONLY consolidation, so the contract is: every filter widget stays alive
and reachable by attribute; ``params()`` reads every stage regardless of which page is
shown; a filter stays enabled while its page is hidden; and the setters / emitted signals
are untouched. These tests pin that behaviour, and a couple of pre-existing param/detect
assertions are re-checked here to prove the contract held.

The stage list GREW (it never changed) when the owner-requested Normalize stage landed
at index 7: a whole-trace rescale, off by default, and the one stage in the chain that
changes the UNIT of the signal the detector measures on.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from nanoscope.analysis.conditioning import chain_description
from nanoscope.gui.panels import DetectionPanel

# (combo index, label, widget attrs that belong on this page, enable-checkbox attr)
FILTERS: list[tuple[int, str, list[str], str]] = [
    (0, "Low-pass", ["filter_enabled", "filter_cutoff_hz", "rise_time_label"], "filter_enabled"),
    (1, "High-pass", ["hp_enabled", "hp_cutoff_hz"], "hp_enabled"),
    (2, "Notch", ["notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q"], "notch_enabled"),
    (3, "Median", ["median_enabled", "median_window_s"], "median_enabled"),
    (4, "Baseline drift", ["detrend_enabled", "detrend_method", "detrend_window_s"], "detrend_enabled"),
    (5, "Running mean", ["rmean_enabled", "rmean_window_s"], "rmean_enabled"),
    (6, "Savitzky-Golay", ["savgol_enabled", "savgol_window_s", "savgol_polyorder"], "savgol_enabled"),
    (7, "Normalize", ["normalize_enabled", "normalize_mode"], "normalize_enabled"),
]

# The stages that expose a set_*_enabled helper (Low-pass and Baseline drift are
# toggled via their checkbox, not a dedicated enable setter): (setter, params key, index, label).
ENABLE_SETTERS: list[tuple[str, str, int, str]] = [
    ("set_hp_enabled", "hp_enabled", 1, "High-pass"),
    ("set_notch_enabled", "notch_enabled", 2, "Notch"),
    ("set_median_enabled", "median_enabled", 3, "Median"),
    ("set_rmean_enabled", "rmean_enabled", 5, "Running mean"),
    ("set_savgol_enabled", "savgol_enabled", 6, "Savitzky-Golay"),
    ("set_normalize_enabled", "normalize_enabled", 7, "Normalize"),
]

EXPECTED_LABELS = ["Low-pass", "High-pass", "Notch", "Median",
                   "Baseline drift", "Running mean", "Savitzky-Golay", "Normalize"]


def _base_label(text: str) -> str:
    """Strip the active-filter ``● `` marker so we can compare against the bare name."""
    return text[2:] if text.startswith("● ") else text


def _execution_summary(panel: DetectionPanel) -> str:
    params = panel.params()
    description = chain_description(params)
    if panel.hp_enabled.isChecked():
        return (
            "Candidate-localization order: "
            + (description or "raw calibrated signal")
            + ". Quantitative measurement: same order with high-pass omitted."
        )
    return "Active execution order: " + (
        description or "raw calibrated signal (no stages enabled)")


# ================================================================================================
# Combo lists every filter, default = Low-pass
# ================================================================================================

def test_combo_lists_all_eight_filters_in_order_default_lowpass(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.filter_selector.count() == len(EXPECTED_LABELS)
    labels = [_base_label(panel.filter_selector.itemText(i))
              for i in range(len(EXPECTED_LABELS))]
    assert labels == EXPECTED_LABELS
    # Default selection is Low-pass, and the stack shows its page.
    assert panel.filter_selector.currentIndex() == 0
    assert panel.filter_stack.currentIndex() == 0
    # Nothing enabled by default -> no markers.
    assert panel.filter_selector.itemText(0) == "Low-pass"


def test_stack_has_one_page_per_filter(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.filter_stack.count() == len(EXPECTED_LABELS)


# ================================================================================================
# Selecting a filter raises its page and shows the right widgets
# ================================================================================================

@pytest.mark.parametrize("idx,label,visible_attrs,_enable", FILTERS)
def test_selecting_filter_raises_page_and_shows_its_widgets(qtbot, idx, label, visible_attrs, _enable):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.filter_selector.setCurrentIndex(idx)
    assert panel.filter_stack.currentIndex() == idx
    # Every widget on this page is visible within the panel...
    for attr in visible_attrs:
        assert getattr(panel, attr).isVisibleTo(panel), f"{attr} should be visible on the {label} page"
    # ...and a widget that belongs to a DIFFERENT page is not.
    other = "notch_freq_hz" if label != "Notch" else "filter_cutoff_hz"
    assert not getattr(panel, other).isVisibleTo(panel)


def test_selected_page_widgets_are_really_visible_when_shown(qtbot):
    """Same as above but with the panel actually shown (real isVisible, not isVisibleTo)."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)
    panel.filter_selector.setCurrentIndex(0)
    assert panel.filter_cutoff_hz.isVisible()
    assert panel.rise_time_label.isVisible()
    assert not panel.notch_freq_hz.isVisible()
    panel.filter_selector.setCurrentIndex(2)
    assert panel.notch_freq_hz.isVisible()
    assert not panel.filter_cutoff_hz.isVisible()


def test_preview_toggle_is_group_level_visible_on_every_page(qtbot):
    """The 'show processed trace' preview reflects the whole chain, so it stays visible
    no matter which filter page is raised (it is not tucked onto one page)."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    for idx in range(len(EXPECTED_LABELS)):
        panel.filter_selector.setCurrentIndex(idx)
        assert panel.filter_preview.isVisibleTo(panel)


def test_highpass_labels_the_orange_preview_as_candidate_localization(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.filter_preview.text() == "show processed trace on the plot"
    panel.set_hp_enabled(True)
    assert panel.filter_preview.text() == \
        "show candidate-localization trace on the plot"
    tip = panel.filter_preview.toolTip()
    assert "orange trace" in tip
    assert "without high-pass" in tip
    panel.set_hp_enabled(False)
    assert panel.filter_preview.text() == "show processed trace on the plot"


# ================================================================================================
# params() is identical regardless of which page is shown
# ================================================================================================

def _set_mixed_state(panel: DetectionPanel) -> None:
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


def test_params_identical_no_matter_which_page_is_shown(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    _set_mixed_state(panel)

    panel.filter_selector.setCurrentIndex(0)
    reference = dict(panel.params())
    # Walk through every page; params() must not budge as the shown page changes.
    for idx in range(len(EXPECTED_LABELS)):
        panel.filter_selector.setCurrentIndex(idx)
        assert panel.filter_stack.currentIndex() == idx
        assert panel.params() == reference


def test_switching_pages_emits_no_params_changed(qtbot):
    """Raising a different filter page is layout-only: it must NOT re-emit paramsChanged
    (nothing about the detection parameters changed)."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    _set_mixed_state(panel)
    got: list[dict] = []
    panel.paramsChanged.connect(got.append)
    for idx in (6, 3, 1, 4, 0, 5, 2):
        panel.filter_selector.setCurrentIndex(idx)
    assert got == []


# ================================================================================================
# A hidden filter stays enabled and shows up in params() + the indicator
# ================================================================================================

@pytest.mark.parametrize("setter,key,idx,label", ENABLE_SETTERS)
def test_toggling_hidden_filter_via_setter_shows_in_params(qtbot, setter, key, idx, label):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    # Show the Low-pass page so the target filter's page is HIDDEN. For these five
    # stages the enable-checkbox attribute name is identical to the params key.
    panel.filter_selector.setCurrentIndex(0)
    assert not getattr(panel, key).isVisibleTo(panel)

    getattr(panel, setter)(True)   # enable it while its page is hidden

    # It is enabled in params()...
    assert panel.params()[key] is True
    # ...its page is still hidden (we never switched to it)...
    assert panel.filter_stack.currentIndex() == 0
    assert not getattr(panel, key).isVisibleTo(panel)
    # ...and the active indicator reflects it live (combo marker + summary label).
    assert panel.filter_selector.itemText(idx) == f"● {label}"
    assert panel.active_filters_label.text() == _execution_summary(panel)


def test_hidden_filter_param_value_setter_also_lands(qtbot):
    """A *value* setter (not just the enable) applied to a hidden page still lands in
    params() — proves the widget is alive, not a dead reparented stub."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.filter_selector.setCurrentIndex(0)   # notch page hidden
    panel.set_notch_enabled(True)
    panel.set_notch_harmonics(7)
    panel.set_notch_q(42.0)
    p = panel.params()
    assert p["notch_enabled"] is True
    assert p["notch_harmonics"] == 7
    assert p["notch_q"] == pytest.approx(42.0)


# ================================================================================================
# The active-filters indicator reflects the enabled set live
# ================================================================================================

def test_active_indicator_starts_at_none(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.active_filters_label.text() == _execution_summary(panel)
    # No item carries a marker while everything is off.
    for i in range(len(EXPECTED_LABELS)):
        assert not panel.filter_selector.itemText(i).startswith("● ")


def test_active_indicator_uses_the_kernels_execution_order(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    # Enable a scattered subset via a mix of checkbox + setter access.
    panel.set_notch_enabled(True)          # index 2
    panel.filter_enabled.setChecked(True)  # index 0
    panel.set_savgol_enabled(True)         # index 6
    # The summary is the actual recipe, not the editor combo's order: low-pass executes
    # last even though its page is first.
    summary = panel.active_filters_label.text()
    assert summary == _execution_summary(panel)
    assert (
        summary.index("notch")
        < summary.index("Savitzky-Golay")
        < summary.index("low-pass")
    )
    # And exactly those combo items carry the ● marker.
    marked = [_base_label(panel.filter_selector.itemText(i))
              for i in range(len(EXPECTED_LABELS))
              if panel.filter_selector.itemText(i).startswith("● ")]
    assert marked == ["Low-pass", "Notch", "Savitzky-Golay"]


def test_active_indicator_updates_when_a_filter_is_disabled(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_hp_enabled(True)
    panel.set_median_enabled(True)
    assert panel.active_filters_label.text() == _execution_summary(panel)
    panel.set_hp_enabled(False)
    assert panel.active_filters_label.text() == _execution_summary(panel)
    assert panel.filter_selector.itemText(1) == "High-pass"      # marker cleared
    assert panel.filter_selector.itemText(3) == "● Median"
    panel.set_median_enabled(False)
    assert panel.active_filters_label.text() == _execution_summary(panel)


def test_active_indicator_refreshes_after_an_enabled_stages_parameters_change(qtbot):
    """The summary is a recipe, so it must follow the method it actually runs.

    Regression for the fresh apo-ferritin detrending run: enabling the default
    ``block_median`` stage refreshed the label, but changing the combo to
    ``masked_median`` only changed the stored/run parameters.  The label then kept
    saying ``block_median`` over a detection whose provenance said
    ``masked_median``.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.detrend_enabled.setChecked(True)
    assert "block_median" in panel.active_filters_label.text()

    assert panel.set_detrend_method("masked_median") is None
    assert panel.params()["detrend_method"] == "masked_median"
    assert panel.active_filters_label.text() == _execution_summary(panel)
    assert "masked_median" in panel.active_filters_label.text()

    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(7_500.0)
    assert panel.active_filters_label.text() == _execution_summary(panel)
    assert "7500 Hz" in panel.active_filters_label.text()


def test_highpass_active_summary_shows_both_execution_paths(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_notch_enabled(True)
    panel.set_hp_enabled(True)
    panel.filter_enabled.setChecked(True)

    summary = panel.active_filters_label.text()
    assert summary.startswith("Candidate-localization order:")
    assert "high-pass" in summary
    assert summary.endswith(
        "Quantitative measurement: same order with high-pass omitted.")
    assert "notch" in summary and "low-pass" in summary


def test_highpass_preview_names_detector_rows_and_manual_processed_signal(qtbot):
    """The orange branch is not universally the quantitative branch: classical
    detector rows omit high-pass, while manual Processed-signal measurement does not."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    tip = panel.filter_preview.toolTip()
    assert "classical detector event-table amplitudes" in tip
    assert "Manual measurement" in tip and "including high-pass" in tip


def test_whole_chain_preview_sits_immediately_under_execution_summary(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    layout = panel.active_filters_label.parentWidget().layout()
    assert (
        layout.indexOf(panel.filter_preview)
        == layout.indexOf(panel.active_filters_label) + 1)
    assert (
        layout.indexOf(panel.preview_error_label)
        == layout.indexOf(panel.filter_preview) + 1)


def test_active_indicator_does_not_change_shown_page(qtbot):
    """Rewriting combo item text for the marker must not switch the shown page or the
    current selection (it rewrites text, not the current index)."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.filter_selector.setCurrentIndex(4)   # Baseline drift page shown
    panel.set_notch_enabled(True)              # toggling a different, hidden filter
    assert panel.filter_selector.currentIndex() == 4
    assert panel.filter_stack.currentIndex() == 4


# ================================================================================================
# Pre-existing contract re-checks (prove the layout change kept detection behaviour)
# ================================================================================================

def test_contract_params_key_set_is_unchanged(qtbot):
    """The full params() key set the detector/preprocess consume.

    GREW (never changed) when the hard-coded kernel parameters were exposed: the low-pass
    and high-pass design (order/kind) and the baseline-drift methods' shape parameters are
    now the user's, and again when the Normalize stage landed. Every addition defaults to
    the value that used to be frozen, and Normalize defaults OFF."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    got: list[dict] = []
    panel.detectRequested.connect(got.append)
    panel.detect_button.click()
    assert len(got) == 1
    assert set(got[0]) == {"polarity", "k_in", "k_out", "min_dwell_s",
                           "max_dwell_s", "filter_cutoff_hz", "merge_gap_s",
                           "filter_order", "filter_kind",
                           "detrend_enabled", "detrend_method", "detrend_window_s",
                           "detrend_percentile", "detrend_asls_lam", "detrend_asls_p",
                           "detrend_asls_niter", "detrend_asls_decim",
                           "notch_enabled", "notch_freq_hz", "notch_harmonics", "notch_q",
                           "median_enabled", "median_window_s",
                           "hp_enabled", "hp_cutoff_hz", "hp_order", "hp_kind",
                           "rmean_enabled", "rmean_window_s",
                           "savgol_enabled", "savgol_window_s", "savgol_polyorder",
                           "normalize_enabled", "normalize_mode",
                           # Round 4: the detector choice and the CUSUM detector's two
                           # thresholds (robust SIGMA units, not pA), the shape parameters
                           # of the newly selectable baseline methods, the explicit notch
                           # line list, and the model-input normalization clip.
                           "detector", "cusum_k", "cusum_h",
                           "detrend_rough_s", "detrend_base_s", "detrend_mask_sigma",
                           "detrend_dilate_ms", "detrend_decim", "detrend_sg_poly",
                           "detrend_sg_decim",
                           "notch_freqs", "normalize_clip_sigma"}


def test_contract_off_sentinels_and_defaults(qtbot):
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    p = panel.params()
    # Off sentinels (filter disabled, merge/max_dwell zero -> None).
    assert p["filter_cutoff_hz"] is None
    assert p["merge_gap_s"] is None
    assert p["max_dwell_s"] is None
    # Every pre-detection stage defaults off with its documented default value.
    assert p["detrend_enabled"] is False and p["detrend_method"] == "block_median"
    assert p["notch_enabled"] is False and p["notch_harmonics"] == 3
    assert p["hp_enabled"] is False and p["hp_cutoff_hz"] == pytest.approx(100.0)
    assert p["median_enabled"] is False and p["savgol_polyorder"] == 3


def test_contract_preview_enable_tracks_any_stage(qtbot):
    """The group-level preview toggle still enables whenever ANY stage is on and clears
    when every stage is off — unchanged from before the consolidation."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert not panel.filter_preview.isEnabled()
    panel.set_notch_enabled(True)          # a hidden pre-detection stage
    assert panel.filter_preview.isEnabled()
    panel.set_notch_enabled(False)
    assert not panel.filter_preview.isEnabled()


def test_contract_rise_time_readout_still_on_lowpass_page(qtbot):
    """The rise-time readout still lives on the Low-pass page and updates with cutoff."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_recording(50_000.0)
    panel.filter_enabled.setChecked(True)
    panel.filter_cutoff_hz.setValue(5_000.0)
    assert "rise time" in panel.rise_time_label.text()
    # It is a child of the Low-pass page (index 0), not the group root.
    assert panel.rise_time_label.isVisibleTo(panel)          # low-pass shown by default
    panel.filter_selector.setCurrentIndex(2)                 # hide the low-pass page
    assert not panel.rise_time_label.isVisibleTo(panel)

# ================================================================================================
# owner request: named preprocessing protocols — save the whole chain, recall it by name
# ================================================================================================

def test_protocol_save_recall_and_manual_drop(qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL

    # Dial a distinctive chain: notch at 50 Hz / 5 harmonics + median on.
    panel.notch_enabled.setChecked(True)
    panel.notch_freq_hz.setValue(50.0)
    panel.notch_harmonics.setValue(5)
    panel.median_enabled.setChecked(True)

    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("eu-mains", True)))
    panel._save_protocol()
    assert panel.protocol_combo.currentText() == "eu-mains"

    # Change the chain by hand → the name comes off (a protocol name over a
    # hand-edited chain would be a lie)...
    panel.notch_freq_hz.setValue(60.0)
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL

    # ...and recalling the protocol restores the saved chain exactly.
    panel.protocol_combo.setCurrentText("eu-mains")
    assert panel.notch_freq_hz.value() == pytest.approx(50.0)
    assert panel.notch_harmonics.value() == 5
    assert panel.notch_enabled.isChecked() and panel.median_enabled.isChecked()

    # A protocol captures PREPROCESSING only — detection thresholds are untouched.
    k_in_before = panel.k_in.value()
    panel.protocol_combo.setCurrentText(panel._PROTO_MANUAL)
    panel.protocol_combo.setCurrentText("eu-mains")
    assert panel.k_in.value() == k_in_before

    # Second panel, same settings store: the protocol survives.
    panel2 = DetectionPanel()
    qtbot.addWidget(panel2)
    names = [panel2.protocol_combo.itemText(i) for i in range(panel2.protocol_combo.count())]
    assert "eu-mains" in names
    panel2.protocol_combo.setCurrentText("eu-mains")
    assert panel2.notch_harmonics.value() == 5

    # Delete forgets it everywhere.
    panel._delete_protocol() if panel.protocol_combo.currentText() == "eu-mains" else None
    panel.protocol_combo.setCurrentText("eu-mains") if any(
        panel.protocol_combo.itemText(i) == "eu-mains"
        for i in range(panel.protocol_combo.count())) else None
    # (state may already be manual after the delete; the store is what matters)
    assert "eu-mains" not in DetectionPanel._load_protocols()
