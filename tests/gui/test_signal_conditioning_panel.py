"""DetectionPanel's Signal Conditioning seam and the EventTable notice chip, standalone.

SP-04 (a protocol recall sets EVERY stage), the hosted/standalone split, the stage API the
window drives, the fixed-height card and the once-built chip.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.conditioning import chain_description
from nanoscope.analysis.conditioning_recipes import BUILTIN_RECIPES, CHAIN_KEYS, complete_recipe
from nanoscope.gui.inspector_widgets import PinnedActions
from nanoscope.gui.panels import DetectionPanel, EventTable


def _typed(monkeypatch, *answers):
    seq = list(answers)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: (seq.pop(0), True)))


@pytest.fixture(autouse=True)
def _empty_protocol_store():
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)


@pytest.fixture
def panel(qtbot):
    p = DetectionPanel()
    qtbot.addWidget(p)
    return p


def _all_stages_on(p: DetectionPanel) -> None:
    p.filter_enabled.setChecked(True)
    for key in ("highpass", "notch", "median", "detrend", "rmean", "savgol", "normalize"):
        p.set_stage_enabled(key, True)


# -- SP-04 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(BUILTIN_RECIPES))
def test_builtin_protocol_recall_sets_every_stage(panel, name):
    _all_stages_on(panel)
    panel.protocol_combo.setCurrentText(name)
    got = panel.params()
    assert chain_description(got) == chain_description(BUILTIN_RECIPES[name])
    assert {k: got[k] for k in CHAIN_KEYS} == BUILTIN_RECIPES[name]
    assert got["filter_cutoff_hz"] is None, "the low-pass stayed on under a protocol's name"
    assert panel.protocol_combo.currentText() == name


def test_a_user_protocol_saved_before_a_key_existed_is_recalled_complete(panel, monkeypatch):
    # A store written by an older build: partial, without normalize_clip_sigma.
    DetectionPanel._store_protocols({"old": {"notch_enabled": True, "notch_q": 20.0}})
    panel._load_protocols_into_combo()
    _all_stages_on(panel)
    panel.normalize_clip_sigma.setValue(5.0)
    panel.protocol_combo.setCurrentText("old")
    got = panel.params()
    assert {k: got[k] for k in CHAIN_KEYS} == complete_recipe({"notch_enabled": True,
                                                             "notch_q": 20.0})
    assert got["normalize_clip_sigma"] == 30.0 and not got["median_enabled"]


def test_the_combo_names_a_protocol_only_while_the_form_equals_it(panel):
    name = "Cu2-insulin (SavGol baseline)"
    panel.protocol_combo.setCurrentText(name)
    # An unrelated non-chain edit keeps the name...
    panel.k_in.setValue(panel.k_in.value() + 1.0)
    assert panel.protocol_combo.currentText() == name
    # ...a set_params that lands on the exact recipe keeps it too...
    panel.set_params(BUILTIN_RECIPES[name])
    assert panel.protocol_combo.currentText() == name
    # ...and any chain edit takes it off, a disabled stage's knob included.
    panel.hp_cutoff_hz.setValue(50.0)
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL
    assert panel.current_protocol() == (None, None)
    panel.protocol_combo.setCurrentText(name)
    assert panel.current_protocol() == (name, BUILTIN_RECIPES[name])
    panel.set_params({"median_enabled": True})
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL


# -- standalone vs hosted --------------------------------------------------------------------
def test_a_standalone_panel_keeps_its_in_page_editor(panel):
    assert not panel.is_signal_chain_hosted()
    assert type(panel.filter_stack) is QtWidgets.QStackedWidget
    assert panel._signal_filters_box.isVisibleTo(panel._signal_chain_page)
    assert not panel.signal_chain_summary.isVisibleTo(panel._signal_chain_page)
    assert not panel.open_conditioning_button.isVisibleTo(panel)
    assert panel._signal_filters_box.isAncestorOf(panel.protocol_combo)
    assert panel._signal_filters_box.isAncestorOf(panel.preview_error_label)


def test_hosting_hides_the_editor_and_shows_the_open_button(panel):
    panel.show_signal_chain()
    before = panel.params()
    panel.set_signal_chain_hosted(True)
    page = panel._signal_chain_page
    assert not panel._signal_filters_box.isVisibleTo(page)
    assert panel.signal_chain_summary.isVisibleTo(page)
    assert panel.open_conditioning_button.objectName() == "openSignalConditioningButton"
    pinned = panel.page_actions[panel._preprocess_page]
    assert pinned.isAncestorOf(panel.open_conditioning_button)
    assert pinned.isAncestorOf(panel.filter_preview)
    assert panel.open_conditioning_button.isVisibleTo(pinned)
    assert panel.signal_chain_summary.isAncestorOf(panel.preview_error_label)
    assert panel.params() == before
    with_signal = []
    panel.openSignalConditioningRequested.connect(with_signal.append)
    panel.select_stage("notch")
    panel.open_conditioning_button.click()
    assert with_signal == ["notch"]


def test_editor_parts_are_built_once_and_keep_the_stack_contract(panel):
    panel.set_signal_chain_hosted(True)
    panel.select_stage("median")
    pages = [panel.filter_stack.widget(i) for i in range(panel.filter_stack.count())]
    parts = panel.signal_chain_editor_parts()
    assert panel.signal_chain_editor_parts() is parts
    stack = panel.filter_stack
    assert parts.stage_stack is stack and stack.count() == 8
    assert [stack.widget(i) for i in range(8)] == pages
    assert stack.indexOf(pages[3]) == 3 and stack.indexOf(QtWidgets.QWidget()) == -1
    assert panel.current_stage_key() == "median" and stack.currentIndex() == 3
    assert parts.protocol_bar.isAncestorOf(panel.protocol_combo)
    assert parts.protocol_bar.isAncestorOf(panel.protocol_rename_btn)
    assert parts.notices.isAncestorOf(panel.normalize_units_label)
    assert parts.response_view.isAncestorOf(panel.response_plot)
    assert parts.chain_response_toggle is panel.response_chain_check
    panel.select_stage("notch")
    assert stack.currentIndex() == 2


def test_select_stage_and_set_stage_enabled_write_the_real_widgets(panel):
    shown = []
    panel.stageShown.connect(shown.append)
    for key in DetectionPanel.stage_keys():
        panel.select_stage(key)
        assert panel.current_stage_key() == key
    assert shown[-1] == "lowpass" and set(shown) == set(DetectionPanel.stage_keys())
    panel.select_stage("not-a-stage")
    assert panel.current_stage_key() == "lowpass"
    panel.set_stage_enabled("lowpass", True)
    assert panel.filter_enabled.isChecked() and panel.params()["filter_cutoff_hz"] is not None
    panel.set_stage_enabled("savgol", True)
    assert panel.savgol_enabled.isChecked() and panel.params()["savgol_enabled"]
    panel.set_stage_enabled("savgol", False)
    assert not panel.params()["savgol_enabled"]
    panel.show_baseline_drift()
    assert panel.current_stage_key() == "detrend"


def test_show_baseline_drift_asks_for_the_window_only_while_hosted(panel):
    asked = []
    panel.openSignalConditioningRequested.connect(asked.append)
    panel.show_open_pore_baseline()
    panel.show_baseline_drift()
    assert asked == [] and panel.preprocess_section_selector.currentIndex() == 0
    panel.set_signal_chain_hosted(True)
    panel.show_open_pore_baseline()
    panel.show_baseline_drift()
    assert asked == ["detrend"] and panel.current_stage_key() == "detrend"
    # Review D-A7: the editor is in the window, so the dock stays on the baseline workspace.
    assert panel.preprocess_section_selector.currentIndex() == 1


def test_essential_level_does_not_hide_the_hosted_response(panel):
    panel.set_control_level("essential")
    assert panel.response_plot.isHidden()
    panel.set_signal_chain_hosted(True)
    assert not panel.response_plot.isHidden()
    assert not panel.response_chain_check.isHidden()
    panel.set_control_level("advanced")
    assert not panel.response_plot.isHidden()


def test_the_card_rows_have_fixed_heights_whatever_they_say(panel, qtbot):
    panel.set_signal_chain_hosted(True)
    panel.resize(420, 900)
    panel.show()
    panel.show_signal_chain()
    qtbot.wait(20)
    notes = (panel.chain_status_note, panel.chain_warning_note, panel.chain_protocol_note,
             panel.chain_scope_note, panel.chain_order_note)
    heights = [n.height() for n in notes]
    ys = [n.y() for n in notes]
    panel.set_chain_status("⚠ The form differs from the chain the Schmitt detection on "
                           "screen ran with: low-pass 10000 → 5000 Hz; notch Q 30 → 20; "
                           "median off → on (0.1 ms) (+3 more). Revert, or run detection.",
                           True)
    _all_stages_on(panel)
    panel.protocol_combo.setCurrentText("Zn2-insulin (zn_30h)")
    qtbot.wait(20)
    assert [n.height() for n in notes] == heights
    assert [n.y() for n in notes] == ys
    text = panel.chain_card_text()
    assert text["protocol"] == "Protocol: Zn2-insulin (zn_30h)"
    assert "NORMALIZE ON" in text["warning"]
    assert text["order"] == panel.active_filters_label.text()
    assert "not by the ML detector" in text["scope"]


def test_the_card_mirrors_a_detrend_refusal_set_and_cleared_alone(panel):
    """Review DA-R1: while hosted the card is the only place in the dock that shows the
    refusal, and a refusal can come and go with no widget value (so no paramsChanged)."""
    panel.set_signal_chain_hosted(True)
    panel.set_params({"detrend_enabled": True, "detrend_method": "block_median"})
    assert "not applied" not in panel.chain_card_text()["warning"]
    form = panel.params()
    form["detrend_method"] = "bogus_method"         # refused: the combo does not move
    panel.set_params(form)
    assert panel.detrend_refusal_label.text()
    assert "Baseline-drift method not applied" in panel.chain_card_text()["warning"]
    form["detrend_method"] = "block_median"         # the combo's own value: clears it
    panel.set_params(form)
    assert not panel.detrend_refusal_label.text()
    assert "not applied" not in panel.chain_card_text()["warning"]


def test_a_recall_the_form_clamps_drops_to_manual_and_says_what_changed(panel):
    """Review D2: on a 50 kHz recording a 40 kHz low-pass is clamped to Nyquist, so the
    form is not the protocol and the combo must not keep its name."""
    panel.set_recording(50_000.0)
    DetectionPanel._store_protocols({"imp": complete_recipe({"filter_cutoff_hz": 40_000.0})})
    panel._load_protocols_into_combo()
    said = []
    panel.protocolNotice.connect(said.append)
    panel.protocol_combo.setCurrentText("imp")
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL
    assert panel.params()["filter_cutoff_hz"] < 25_000.0
    assert said and "“imp”" in said[-1] and "low-pass 40000 →" in said[-1]
    assert panel.chain_card_text()["protocol"] == f"Protocol: {panel._PROTO_MANUAL}"
    # A disabled stage's clamped knob is named by key.
    DetectionPanel._store_protocols({"imp": {"hp_cutoff_hz": 0.05}})
    panel._load_protocols_into_combo()
    panel.protocol_combo.setCurrentText("imp")
    assert panel.protocol_combo.currentText() == panel._PROTO_MANUAL
    assert "hp_cutoff_hz 0.05 → 0.1" in said[-1]
    # A recall the form holds exactly keeps the name and says nothing.
    n = len(said)
    panel.protocol_combo.setCurrentText("Zn2-insulin (zn_30h)")
    assert panel.protocol_combo.currentText() == "Zn2-insulin (zn_30h)" and len(said) == n


def test_import_ranges_are_the_boxes_static_ranges():
    """validate_recipe (Qt-free) restates the frequency boxes' static ranges; keep them equal
    so an imported value is never one the form would clamp."""
    from nanoscope.analysis import conditioning_recipes as cr

    assert cr._FLOAT_RANGES["filter_cutoff_hz"] == DetectionPanel._STATIC_LP_RANGE
    assert cr._FLOAT_RANGES["hp_cutoff_hz"] == DetectionPanel._STATIC_HP_RANGE
    assert cr._FLOAT_RANGES["notch_freq_hz"] == DetectionPanel._STATIC_NOTCH_RANGE


def test_a_corrupt_store_entry_is_skipped_never_raised(panel):
    """Review D3: one bad entry must not block exporting (or importing) every good one."""
    good = complete_recipe({"median_enabled": True})
    DetectionPanel._store_protocols({"ok": good, "bad": {"notch_freqs": 5},
                                     "worse": "not a dict"})
    panel._load_protocols_into_combo()
    library, unreadable = panel.protocol_library_report()
    assert library == {"ok": good} and unreadable == ["bad", "worse"]
    assert panel.protocol_library() == {"ok": good}
    said = []
    panel.protocolNotice.connect(said.append)
    before = panel.params()
    panel.protocol_combo.setCurrentText("bad")
    assert panel.params() == before and panel.protocol_combo.currentText() == panel._PROTO_MANUAL
    assert "“bad” could not be read" in said[-1]
    # An import onto an unreadable name renames instead of replacing it.
    report = panel.merge_protocols({"bad": complete_recipe({"savgol_enabled": True})})
    assert report.renamed == [("bad", "bad (imported)")]
    assert DetectionPanel._load_protocols()["bad"] == {"notch_freqs": 5}


def test_merge_protocols_never_overwrites_and_keeps_the_selection(panel, monkeypatch):
    panel.set_notch_enabled(True)
    _typed(monkeypatch, "mine")
    panel._save_protocol()
    assert panel.protocol_combo.currentText() == "mine"
    report = panel.merge_protocols({"mine": complete_recipe({"savgol_enabled": True}),
                                    "theirs": complete_recipe({"median_enabled": True})})
    assert report.added == ["theirs"] and report.renamed == [("mine", "mine (imported)")]
    assert panel.protocol_combo.currentText() == "mine"
    lib = panel.protocol_library()
    assert set(lib) == {"mine", "mine (imported)", "theirs"}
    assert lib["mine"]["notch_enabled"] and not lib["mine"]["savgol_enabled"]
    names = [panel.protocol_combo.itemText(i) for i in range(panel.protocol_combo.count())]
    assert "theirs" in names and "mine (imported)" in names


# -- the event table's notice chip ---------------------------------------------------------
def test_event_table_notice_chip_floors_nothing(qtbot):
    table = EventTable()
    qtbot.addWidget(table)
    table.column_toolbar.insertWidget(0, QtWidgets.QLabel("Table"))
    selector = QtWidgets.QComboBox()
    selector.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    table.column_toolbar.insertWidget(1, selector, 1)
    table.resize(420, 300)
    table.show()
    qtbot.wait(20)
    floor = table.minimumSizeHint().width()
    columns = table.columns_button.geometry()
    selector_geo = selector.geometry()
    view_y = table.view.y()
    table.set_notice("⚠ Chain changed", "the long sentence")
    qtbot.wait(20)
    assert table.notice_chip.isVisible()
    assert table.notice_chip.minimumSizeHint().width() <= 32
    assert table.minimumSizeHint().width() <= floor + 32
    # Review D-A2: the chip's slot is reserved, so NOTHING in the row moves or resizes.
    assert table.columns_button.geometry() == columns
    assert selector.geometry() == selector_geo
    assert table.view.y() == view_y
    assert table.notice_text() == "⚠ Chain changed"
    chip_geo = table.notice_chip.geometry()
    table.set_notice("⚠ A much longer notice than the slot was sized for", "tip")
    qtbot.wait(20)
    assert table.notice_chip.geometry() == chip_geo and selector.geometry() == selector_geo
    table.set_notice("")
    qtbot.wait(20)
    assert not table.notice_chip.isVisible() and table.notice_text() == ""
    assert selector.geometry() == selector_geo and table.columns_button.geometry() == columns
    # The spare width goes to the selector, not to an empty gap.
    assert selector.width() > table.notice_chip.sizeHint().width()


def test_a_narrow_row_takes_width_from_the_empty_slot_too(qtbot):
    """Review DA-2: the hidden chip's reserved slot kept its full width at every Inspector
    width, so at the narrowest the table selector painted one letter."""
    table = EventTable()
    qtbot.addWidget(table)
    table.column_toolbar.insertWidget(0, QtWidgets.QLabel("Table"))
    selector = QtWidgets.QComboBox()
    selector.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    table.column_toolbar.insertWidget(1, selector, 1)
    chip = table.notice_chip
    table.resize(640, 300)
    table.show()
    qtbot.wait(20)
    assert chip.width() == chip.sizeHint().width(), "wide: the full slot"
    wide_selector = selector.width()
    table.resize(300, 300)
    qtbot.wait(20)
    assert chip.width() < chip.sizeHint().width()
    assert selector.width() >= chip.width() - 1, (selector.width(), chip.width())
    assert wide_selector > selector.width()
    # ...and appearing still moves nothing at that width.
    geos = [w.geometry() for w in (selector, chip, table.columns_button)]
    table.set_notice("⚠ Chain changed", "tip")
    qtbot.wait(20)
    assert [w.geometry() for w in (selector, chip, table.columns_button)] == geos
    assert chip.minimumSizeHint().width() <= 32


def test_the_notice_actions_are_built_once(qtbot):
    table = EventTable()
    qtbot.addWidget(table)
    actions = list(table.notice_menu.actions())
    assert [a.data() for a in actions] == ["revert", "run", "open"]
    for i in range(50):
        table.set_notice("⚠ Chain changed" if i % 2 else "", f"tip {i}")
    assert list(table.notice_menu.actions()) == actions
    assert table.notice_menu.findChildren(QtWidgets.QMenu) == []


def test_the_chip_menu_goes_through_the_exec_seam(qtbot, menu_answer):
    table = EventTable()
    qtbot.addWidget(table)
    table.set_notice("⚠ Chain changed", "tip")
    got = []
    table.noticeActionRequested.connect(got.append)
    menu_answer("Revert form to this detection's chain")
    table.notice_chip.click()
    assert got == ["revert"]
    assert isinstance(table.notice_chip, QtWidgets.QPushButton)
    assert not isinstance(table.notice_chip, QtWidgets.QToolButton)


def test_the_pinned_area_still_floors_nothing_once_hosted(panel):
    floor = panel.minimumSizeHint().width()
    panel.set_signal_chain_hosted(True)
    for actions in panel.findChildren(PinnedActions):
        actions._sync_height()
    assert panel.minimumSizeHint().width() <= floor


# -- D-beta: the spectrum's Send to notch, and the Response tab's stage split -------------------
def test_add_notch_lines_spells_out_a_live_series_and_turns_the_notch_on(panel):
    panel.set_notch_enabled(True)                    # the 60 Hz ×3 series is live
    said = panel.add_notch_lines([4570.83, 120.2])
    assert said == ("Notch: added 4570.8 Hz — the explicit line list is now "
                    "60, 120, 180, 4570.8 Hz.")
    params = panel.params()
    assert params["notch_enabled"] and params["notch_freqs"] == [60.0, 120.0, 180.0, 4570.8]
    assert panel.notch_use_lines.isChecked()
    assert panel.add_notch_lines([4570.8]).endswith("is already removed — nothing was added.")


def test_add_notch_lines_on_an_off_notch_starts_a_list_of_its_own(panel):
    assert not panel.notch_enabled.isChecked()
    panel.add_notch_lines([300.0])
    assert panel.params()["notch_freqs"] == [300.0], "the off series is not smuggled in"


def test_add_notch_lines_keeps_an_off_notchs_list_and_leaves_it_off(panel):
    """An off notch's kept list is neither switched back on with the new line nor lost:
    the line joins it and the notch stays off, and the sentence leads with that."""
    panel.set_notch_lines([60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0, 660.0])
    panel.set_notch_enabled(False)
    said = panel.add_notch_lines([4570.8])
    params = panel.params()
    assert not params["notch_enabled"]
    assert params["notch_freqs"] == [60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 540.0,
                                     660.0, 4570.8]
    assert said == ("Notch still OFF: added 4570.8 Hz to the list it keeps, now 10 lines: "
                    "60, 120, 180 … 4570.8 Hz — tick Notch to apply them.")
    again = panel.add_notch_lines([4570.8])
    assert again.startswith("Notch still OFF: 4570.8 Hz is already in the list it keeps")
    assert panel.params()["notch_freqs"][-1] == 4570.8 and not panel.params()["notch_enabled"]


def test_add_notch_lines_refuses_past_the_cap_and_touches_nothing(panel):
    many = [100.0 * (k + 1) for k in range(DetectionPanel.NOTCH_MAX_LINES)]
    panel.set_notch_lines(many)
    # On: those 32 lines are in force, so one more is past the cap. (Off, they would not
    # be, and the new line would replace them.)
    panel.set_notch_enabled(True)
    before = (panel.params(), panel.notch_freqs.text())
    said = panel.add_notch_lines([5_123.0])
    assert said.startswith("Refused:") and f"at most {DetectionPanel.NOTCH_MAX_LINES}" in said
    assert (panel.params(), panel.notch_freqs.text()) == before


def _random_chains(n: int, seed: int = 7):
    import random

    rng = random.Random(seed)
    for _ in range(n):
        yield {
            "notch_enabled": rng.random() < 0.5,
            "notch_freq_hz": rng.choice([50.0, 60.0]), "notch_harmonics": rng.randint(1, 5),
            "notch_q": rng.choice([10.0, 30.0, 50.0]),
            "notch_freqs": rng.choice([None, [60.0, 180.0, 300.0]]),
            "median_enabled": rng.random() < 0.5,
            "hp_enabled": rng.random() < 0.5, "hp_cutoff_hz": rng.choice([1.0, 20.0, 100.0]),
            "hp_order": rng.randint(1, 6), "hp_kind": rng.choice(["butter", "bessel"]),
            "detrend_enabled": rng.random() < 0.5,
            "rmean_enabled": rng.random() < 0.5, "rmean_window_s": rng.choice([2e-4, 5e-4]),
            "savgol_enabled": rng.random() < 0.5, "savgol_window_s": rng.choice([5e-4, 1e-3]),
            "savgol_polyorder": rng.randint(2, 4),
            "normalize_enabled": rng.random() < 0.5,
            "filter_cutoff_hz": rng.choice([None, 5_000.0, 10_000.0]),
            "filter_order": rng.choice([2, 4, 8]),
            "filter_kind": rng.choice(["bessel", "butter", "gaussian"]),
        }


def test_linear_stage_specs_equal_the_panels_active_stages(panel):
    """The Response tab's Phase / Group delay / Step derive their stages from params; the
    Magnitude view (the panel) from its widgets. They must be the same stages."""
    from nanoscope.analysis.filter_views import linear_stage_specs

    for chain in _random_chains(24):
        panel.set_params(complete_recipe(chain))
        assert linear_stage_specs(panel.params()) == panel._active_linear_response_stages(), chain
