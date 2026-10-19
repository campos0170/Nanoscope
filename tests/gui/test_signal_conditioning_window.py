"""The Signal Conditioning window, driven through a real MainWindow.

Built once, opened from four doors, hosting the DetectionPanel's own editor widgets, an
execution-order stage list that reflects and writes the real checkboxes, the
chain-versus-detection judge (coalesced, exception-safe) with its once-built chip,
protocol JSON exchange, Copy as Python, placement and settings.
"""

from __future__ import annotations

import ast
import importlib
import json
import os

import nanoscope  # noqa: F401  -> Qt plugin bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from nanoscope.analysis import conditioning_recipes as cr
from nanoscope.analysis.modules import register_all
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import analysis_panel as AP
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel
from nanoscope.gui.workflow_status import Stage

FS = 50_000.0


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)
    yield
    QtCore.QSettings().remove("signalConditioning")
    QtCore.QSettings().remove(DetectionPanel._PROTO_SETTINGS_KEY)


def _signal(n=60_000, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, -1000.0) + rng.normal(0.0, 8.0, n)
    for s, e in ((10_000, 10_300), (30_000, 30_400), (45_000, 45_250)):
        x[s:e] += 300.0
    return x.astype(np.float32)


def _doc(tmp_path, name="a.dat") -> TraceDocument:
    sig = _signal()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1512, 875)
    w.show()
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def loaded(window, qtbot, tmp_path):
    did = window._session.add_document(_doc(tmp_path))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    qtbot.wait(20)
    return window, did


@pytest.fixture
def detected(loaded, qtbot):
    window, did = loaded
    panel = window._analysis_panel
    panel.detection_panel.set_params({"filter_cutoff_hz": 10_000.0})
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    assert panel.current_detection() is not None
    return window, did


def _ws(window):
    return window._signal_conditioning


# -- one window, four doors ----------------------------------------------------------------
def test_menu_action_opens_one_window_built_once(window, qtbot):
    act = window._act_signal_conditioning
    assert act.shortcut() == QtGui.QKeySequence("Ctrl+Shift+F") and act.isEnabled()
    assert window._signal_conditioning is None, "built lazily"
    act.trigger()
    qtbot.wait(10)
    ws = _ws(window)
    assert ws is not None and ws.isVisible()
    assert ws.objectName() == "SignalConditioningWindow"
    assert ws.parent() is window and ws.isWindow()
    assert ws.minimumWidth() >= 760 and ws.minimumHeight() >= 560
    assert not ws.testAttribute(QtCore.Qt.WA_DeleteOnClose)
    act.trigger()
    assert _ws(window) is ws
    # No ⌘W in this window: that key is File ▸ Close document.
    close = QtGui.QKeySequence(QtGui.QKeySequence.Close)
    assert not [s for s in ws.findChildren(QtGui.QShortcut) if s.key() == close]
    assert not [a for a in ws.findChildren(QtGui.QAction) if a.shortcut() == close]


def test_card_button_filter_chip_and_baseline_advanced_all_open_it(window, qtbot):
    det = window._analysis_panel.detection_panel
    assert det.is_signal_chain_hosted()
    det.select_stage("savgol")
    det.open_conditioning_button.click()
    ws = _ws(window)
    assert ws.isVisible() and ws.stage_list.currentRow() == cr.STAGE_KEYS.index("savgol")
    ws.close()
    assert not ws.isVisible() and _ws(window) is ws
    window._reveal_stage(Stage.FILTER)
    assert ws.isVisible()
    assert window._inspector_tabs.currentWidget() is window._analysis_panel
    ws.close()
    det.show_open_pore_baseline()
    window._analysis_panel.baseline_workspace.advancedRequested.emit()
    assert ws.isVisible() and det.current_stage_key() == "detrend"
    assert ws.stage_list.currentRow() == cr.STAGE_KEYS.index("detrend")
    # Review D-A7: the dock stays in the baseline workspace the user clicked from.
    assert det.preprocess_section_selector.currentIndex() == 1
    ws.close()
    assert det.preprocess_section_selector.currentIndex() == 1


def test_the_editor_widgets_live_in_the_workspace(window, qtbot):
    det = window._analysis_panel.detection_panel
    before = det.params()
    window._open_signal_conditioning()
    ws = _ws(window)
    for widget in (det.filter_stack, det.protocol_combo, det.protocol_save_btn,
                   det.response_plot, det.response_chain_check, det.normalize_units_label,
                   det.hp_enabled, det.notch_freqs):
        assert widget.window() is ws, widget
    assert det.params() == before
    # The Preprocess card stays in the dock, with the preview toggle where it was.
    assert det.signal_chain_summary.window() is window
    assert det.filter_preview.window() is window
    assert ws.preview_tabs.objectName() == "signalConditioningPreview"
    # D-beta added the Time and Spectrum previews ahead of the relocated Response view.
    assert [ws.preview_tabs.tabText(i) for i in range(ws.preview_tabs.count())] == [
        "Time", "Spectrum", "Response"]


# -- the stage list ------------------------------------------------------------------------
def test_stage_list_is_execution_order_and_reflects_and_writes_enable_state(loaded, qtbot):
    window, _ = loaded
    det = window._analysis_panel.detection_panel
    window._open_signal_conditioning()
    ws = _ws(window)
    rows = ws.stage_rows()
    assert tuple(r.key for r in rows) == cr.STAGE_KEYS
    texts = [ws.stage_list.item(i).text() for i in range(8)]
    assert texts[0].startswith("1  Notch") and texts[-1].startswith("8  Low-pass")
    # UPDATED (review DA-5): the setting precedes the role, so right-elision keeps the cutoff.
    assert texts[-1].startswith("8  Low-pass — ") and texts[-1].endswith(
        "(applied inside detection)")
    assert ws.stage_list.verticalScrollBar().maximum() == 0
    # The form reflects into the list...
    det.set_stage_enabled("median", True)
    ws.flush()
    assert ws.stage_list.item(1).checkState() == QtCore.Qt.Checked
    # ...and a tick in the list writes the real checkbox.
    ws.stage_list.item(4).setCheckState(QtCore.Qt.Checked)
    assert det.rmean_enabled.isChecked()
    ws.stage_list.item(7).setCheckState(QtCore.Qt.Checked)
    assert det.params()["filter_cutoff_hz"] is not None
    ws.flush()
    # UPDATED (review D-A3): the true corner is in the row's tooltip, not its label.
    assert "(−3 dB" in ws.stage_list.item(7).toolTip(), "the rate is known: the true corner"
    assert "−3 dB" not in ws.stage_list.item(7).text()
    # A protocol recall re-renders every row without echoing enable requests back.
    asked = []
    ws.stageEnableRequested.connect(lambda k, on: asked.append((k, on)))
    det.protocol_combo.setCurrentText("Zn2-insulin (zn_30h)")
    ws.flush()
    assert asked == []
    assert [r.enabled for r in ws.stage_rows()] == [True, False, False, True, False, False,
                                                    True, False]


def test_stage_list_selection_raises_the_form_and_follows_the_panel(window, qtbot):
    det = window._analysis_panel.detection_panel
    window._open_signal_conditioning()
    ws = _ws(window)
    for row, key in enumerate(cr.STAGE_KEYS):
        ws.stage_list.setCurrentRow(row)
        assert det.current_stage_key() == key
        assert det.filter_stack.currentWidget().isVisibleTo(ws)
    det.show_baseline_drift()
    assert ws.stage_list.currentRow() == cr.STAGE_KEYS.index("detrend")
    det.select_stage("notch")
    assert ws.stage_list.currentRow() == 0


@pytest.mark.parametrize("recipe", ["", *cr.BUILTIN_RECIPES])
def test_stage_settings_survive_the_default_column_width(window, qtbot, recipe):
    """UPDATED (review DA-5, replacing D-A3's role-first test): at the default column the
    label's TAIL elides, so each row's setting comes before its role note (which the row's
    tooltip repeats) and is painted in full: the low-pass cutoff, the windows.

    The one exception is the event-masked median, whose method and two windows are wider
    than the column on their own; its method still fits and the tooltip has the windows."""
    det = window._analysis_panel.detection_panel
    if recipe:
        det.protocol_combo.setCurrentText(recipe)
    else:
        det.set_params({"filter_cutoff_hz": 10_000.0, "hp_enabled": True})
    window._open_signal_conditioning()
    ws = _ws(window)
    ws.resize(ws.minimumWidth(), ws.minimumHeight())
    qtbot.wait(20)
    fm = ws.stage_list.fontMetrics()
    room = ws.stage_list.viewport().width() - 2 * fm.height()   # minus the check box
    for row in ws.stage_rows():
        item = ws.stage_list.item(row.position - 1)
        head = f"{row.position}  {row.title} — {row.summary}"
        assert item.text().startswith(head)
        if row.role:
            assert item.text().endswith(f"({row.role})") and row.role in item.toolTip()
        if row.key == "detrend" and row.summary.startswith("event-masked"):
            head = head[:head.index(" · rough")]
        assert fm.horizontalAdvance(head) < room, (head, room)


def test_the_rise_readout_and_the_magnitude_title_are_never_clipped(qapp, qtbot, tmp_path):
    """Review DB-R2-09: the form layout gave the four-line rise readout its size-hint
    height (60 of 80 px), cutting its last line; the hosted magnitude title was clipped
    mid-word in the ~380 px preview pane. It shows with the app's own font set before the
    window is built (as the app does), so this test builds its own window under it."""
    previous = QtWidgets.QApplication.font()
    QtWidgets.QApplication.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        window = MainWindow()
        qtbot.addWidget(window)
        window.resize(1512, 875)
        window.show()
        did = window._session.add_document(_doc(tmp_path))
        window._create_default_layers(window._session.document(did))
        window._on_selection(did, 0)
        qtbot.wait(20)
        det = window._analysis_panel.detection_panel
        window._open_signal_conditioning()
        ws = _ws(window)
        qtbot.wait(50)
        det.select_stage("lowpass")
        qtbot.wait(20)
        det.set_params({"filter_cutoff_hz": 10_000.0})       # 2 lines -> 4 at this width
        qtbot.wait(50)
        label = det.rise_time_label
        assert label.heightForWidth(label.width()) > 2 * label.fontMetrics().lineSpacing()
        assert label.height() >= label.heightForWidth(label.width())
        ws.splitter.setSizes([ws.LEFT_MIN_WIDTH, ws.width() - ws.LEFT_MIN_WIDTH])
        qtbot.wait(30)
        assert label.height() >= label.heightForWidth(label.width())
        ws.set_preview_tab("response")
        qtbot.wait(30)
        title = det.response_plot.getPlotItem().titleLabel
        assert title.text == "Low-pass response — ON"
        assert title.boundingRect().width() <= det.response_plot.width()
        window._analysis_controller.abort()
    finally:
        QtWidgets.QApplication.setFont(previous)


# -- focus and keyboard (review D-A1 / D-A5) ------------------------------------------------
def test_every_open_lands_on_the_stage_list_and_space_toggles_the_stage(loaded, qtbot,
                                                                       monkeypatch):
    window, _ = loaded
    ap = window._analysis_panel
    det = ap.detection_panel
    runs = []
    monkeypatch.setattr(ap, "trigger_detection_from_menu", lambda *a, **k: runs.append(1))
    window.activateWindow()
    ap._table.view.setFocus()
    qtbot.wait(10)
    QtTest.QTest.keyClick(ap._table.view, QtCore.Qt.Key_F,
                          QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)
    qtbot.wait(20)
    ws = _ws(window)
    assert ws is not None and ws.isVisible()
    assert QtWidgets.QApplication.focusWidget() is ws.stage_list
    assert ws.stage_list.currentRow() == cr.STAGE_KEYS.index(det.current_stage_key())
    det.select_stage("median")
    was = det.median_enabled.isChecked()
    QtTest.QTest.keyClick(ws.stage_list, QtCore.Qt.Key_Space)
    qtbot.wait(10)
    assert det.median_enabled.isChecked() is not was, "Space toggles the current stage"
    assert runs == []
    QtTest.QTest.keyClick(ws.stage_list, QtCore.Qt.Key_Down)
    assert det.current_stage_key() == "highpass", "Down moves the stage selection"
    # The status buttons never take focus, not even by a click.
    for button in (ws.revert_button, ws.run_button):
        assert button.focusPolicy() == QtCore.Qt.TabFocus
    # A re-open from the card (window hidden) lands on the list again.
    ws.close()
    det.open_conditioning_button.setFocus()
    det.open_conditioning_button.click()
    qtbot.wait(10)
    assert QtWidgets.QApplication.focusWidget() is ws.stage_list


def test_opening_from_the_chip_after_a_change_cannot_revert_by_space(detected, qtbot,
                                                                     menu_answer):
    window, _ = detected
    ap = window._analysis_panel
    det = ap.detection_panel
    det.set_stage_enabled("detrend", True)
    qtbot.waitUntil(lambda: ap._table.notice_chip.isVisible(), timeout=2000)
    menu_answer("Open Signal Conditioning…")
    ap._table.notice_chip.click()
    qtbot.wait(20)
    ws = _ws(window)
    assert ws.revert_button.isEnabled()
    assert QtWidgets.QApplication.focusWidget() is ws.stage_list
    det.select_stage("detrend")
    QtTest.QTest.keyClick(ws.stage_list, QtCore.Qt.Key_Space)
    qtbot.wait(10)
    # Space toggled the selected (detrend) row: off again — through the stage, not Revert.
    assert not det.detrend_enabled.isChecked()
    assert ap.chain_status().kind in ("match", "differs")


@pytest.mark.parametrize("key", cr.STAGE_KEYS)
def test_tab_from_the_stage_list_enters_the_shown_stage_form(window, qtbot, key):
    det = window._analysis_panel.detection_panel
    window._open_signal_conditioning(key)
    ws = _ws(window)
    qtbot.wait(10)
    assert ws.form_area.focusPolicy() == QtCore.Qt.NoFocus
    ws.stage_list.setFocus()
    QtTest.QTest.keyClick(ws.stage_list, QtCore.Qt.Key_Tab)
    qtbot.wait(5)
    focused = QtWidgets.QApplication.focusWidget()
    page = det.filter_stack.currentWidget()
    assert focused is not None and page.isAncestorOf(focused), (key, focused)
    # ...and Tab keeps cycling inside this window, back to the list.
    for _ in range(80):
        QtTest.QTest.keyClick(QtWidgets.QApplication.focusWidget(), QtCore.Qt.Key_Tab)
        if QtWidgets.QApplication.focusWidget() is ws.stage_list:
            break
        assert QtWidgets.QApplication.focusWidget().window() is ws
    assert QtWidgets.QApplication.focusWidget() is ws.stage_list


def test_the_form_width_never_depends_on_how_tall_the_form_is(window, qtbot):
    """Review D-A6: an as-needed scrollbar narrowed every control when a notice or a
    protocol recall made the form overflow."""
    det = window._analysis_panel.detection_panel
    window._open_signal_conditioning("normalize")
    ws = _ws(window)
    ws.resize(ws.minimumWidth(), ws.minimumHeight())
    qtbot.wait(20)
    assert ws.form_area.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOn
    width = ws.form_area.viewport().width()
    combo_w = det.normalize_mode.width()
    det.set_stage_enabled("normalize", True)
    qtbot.wait(20)
    assert ws.form_area.viewport().width() == width and det.normalize_mode.width() == combo_w
    det.select_stage("notch")
    det.protocol_combo.setCurrentText("Zn2-insulin (zn_30h)")
    qtbot.wait(20)
    assert ws.form_area.viewport().width() == width


def test_follow_main_trace_is_on_screen_at_the_minimum_width(window, qtbot):
    """Review D5: the one stateful toggle must never be hidden in the » overflow."""
    window._open_signal_conditioning()
    ws = _ws(window)
    ws.resize(ws.minimumWidth(), ws.minimumHeight())
    qtbot.wait(20)
    actions = [a for a in ws.toolbar.actions() if not a.isSeparator()]
    assert actions[0] is ws.act_follow
    for act in actions:
        if act.isCheckable():
            button = ws.toolbar.widgetForAction(act)
            assert button is not None and button.isVisible(), act.text()


def test_a_stage_change_does_not_scroll_the_form_area(window, qtbot):
    """The hosted twin of test_fin_stage_switch_does_not_scroll: with focus on a field of
    the page being hidden, the switch parks focus on the stage list (the partner), so Qt
    hands nothing to the incoming page and the form area does not jump."""
    det = window._analysis_panel.detection_panel
    window._open_signal_conditioning("detrend")
    ws = _ws(window)
    ws.resize(ws.minimumWidth(), ws.minimumHeight())
    qtbot.wait(20)
    bar = ws.form_area.verticalScrollBar()
    for i, key in enumerate(cr.STAGE_KEYS):
        det.select_stage(key)
        qtbot.wait(5)
        page = det.filter_stack.currentWidget()
        kinds = (QtWidgets.QAbstractSpinBox, QtWidgets.QCheckBox, QtWidgets.QComboBox,
                 QtWidgets.QLineEdit)
        fields = [w for w in page.findChildren(QtWidgets.QWidget)
                  if isinstance(w, kinds) and w.isVisibleTo(ws) and w.isEnabled()
                  and not isinstance(w.parentWidget(), QtWidgets.QAbstractSpinBox)]
        field = fields[-1]                      # the lowest field: the scroll-prone case
        field.setFocus()
        ws.form_area.ensureWidgetVisible(field)
        qtbot.wait(5)
        assert ws.focusWidget() is field
        before = bar.value()
        det.select_stage(cr.STAGE_KEYS[(i + 1) % len(cr.STAGE_KEYS)])
        qtbot.wait(5)
        assert bar.value() <= before, f"{key}: the form scrolled {bar.value() - before} px"
        assert ws.focusWidget() is ws.stage_list


def test_notices_are_below_the_stage_form(window, qtbot):
    det = window._analysis_panel.detection_panel
    window._open_signal_conditioning("highpass")
    ws = _ws(window)
    qtbot.wait(20)
    y_before = det.hp_enabled.mapTo(ws, QtCore.QPoint(0, 0)).y()
    det.hp_enabled.setChecked(True)
    qtbot.wait(20)
    assert det.highpass_measurement_label.isVisibleTo(ws)
    assert det.hp_enabled.mapTo(ws, QtCore.QPoint(0, 0)).y() == y_before
    stack_bottom = det.filter_stack.mapTo(ws, QtCore.QPoint(0, det.filter_stack.height())).y()
    assert det.highpass_measurement_label.mapTo(ws, QtCore.QPoint(0, 0)).y() >= stack_bottom


def _combo_y(combo, ws):
    return combo.mapTo(ws.form_area.viewport(), QtCore.QPoint(0, 0)).y()


def _overflow_by(ws, bar, px, qtbot):
    """Make the window just tall enough that the form overflows by about ``px``."""
    ws.resize(ws.width(), ws.minimumHeight())
    qtbot.wait(30)
    ws.resize(ws.width(), ws.height() + bar.maximum() - px)
    qtbot.wait(30)
    assert 0 < bar.maximum() <= px + 20, ("precondition: the form overflows a little",
                                          bar.maximum())


def _pick(combo, word, avoid=None):
    return next(i for i in range(combo.count()) if word in combo.itemText(i).lower()
                and not (avoid and avoid in combo.itemText(i).lower()))


def test_a_shorter_form_never_slides_the_edited_control(window, qtbot):
    """Review DA-1: a choice that shortened the shown stage (fewer detrend knobs, a smaller
    notice) clamped the scrolled form and moved the combo just used ~30 px down — onto the
    stage's own enable box, for a second click to untick."""
    det = window._analysis_panel.detection_panel
    window._open_signal_conditioning("detrend")
    ws = _ws(window)
    ws.resize(ws.minimumWidth(), ws.minimumHeight())
    det.protocol_combo.setCurrentText("Zn2-insulin (zn_30h)")     # masked median, long
    det.set_stage_enabled("highpass", True)
    det.select_stage("detrend")
    qtbot.wait(30)
    bar = ws.form_area.verticalScrollBar()
    method = det.detrend_method
    _overflow_by(ws, bar, 40, qtbot)        # as at 1512x900: a little, so a shrink clamps
    bar.setValue(min(bar.maximum(), max(1, _combo_y(method, ws) + bar.value() - 4)))
    qtbot.wait(10)
    scrolled, y = bar.value(), _combo_y(method, ws)
    assert scrolled > 0 and 0 <= y < ws.form_area.viewport().height()
    method.setCurrentIndex(_pick(method, "block"))
    qtbot.wait(30)
    assert bar.value() == scrolled and _combo_y(method, ws) == y
    # A stage switch starts again from the content's own height.
    det.select_stage("notch")
    qtbot.wait(10)
    det.select_stage("detrend")
    qtbot.wait(30)
    lay = ws._form_host.layout()
    natural = max(lay.totalHeightForWidth(ws._form_host.width()),
                  lay.totalMinimumSize().height(), ws.form_area.viewport().height())
    assert ws._form_host.height() <= natural
    # The normalize mode's notice shrinking at the bottom of the form moves nothing either.
    det.protocol_combo.setCurrentText("Zn2-insulin (zn_30h)")
    det.set_stage_enabled("highpass", True)
    det.select_stage("normalize")
    qtbot.wait(30)
    mode = det.normalize_mode
    _overflow_by(ws, bar, 30, qtbot)
    bar.setValue(bar.maximum())
    qtbot.wait(10)
    scrolled, y = bar.value(), _combo_y(mode, ws)
    assert 0 <= y < ws.form_area.viewport().height(), "the combo is on screen"
    mode.setCurrentIndex(_pick(mode, "z", avoid="model"))
    qtbot.wait(30)
    assert bar.value() == scrolled and _combo_y(mode, ws) == y


def test_the_status_line_leads_with_what_differs(detected, qtbot):
    """Review DA-3: at the 760 px default the fixed prefix of the sentence filled the line,
    so the window never showed WHAT differs."""
    window, _ = detected
    ap = window._analysis_panel
    det = ap.detection_panel
    window._open_signal_conditioning()
    ws = _ws(window)
    ws.resize(ws.DEFAULT_WIDTH, ws.minimumHeight())
    det.set_stage_enabled("median", True)
    det.filter_cutoff_hz.setValue(5_000.0)
    qtbot.waitUntil(lambda: ap.chain_status().kind == "differs", timeout=2000)
    qtbot.wait(20)
    status = ap.chain_status()
    line = ws.status_line
    assert line.text().startswith(f"⚠ Differs from the {status.detector_label} run: ")
    assert line.toolTip() == status.sentence
    first = status.diffs[0].text
    lead = line.text()[:line.text().index(first) + len(first)]
    assert line.fontMetrics().horizontalAdvance(lead) < line.width(), "the first diff shows"


def test_run_detection_follows_the_panels_own_gate(loaded, qtbot):
    """Review DA-4: the button copied the menu action, which only re-synced on a retarget
    or a menu open, so an open-pore reference edit left it live-but-refused or dead."""
    window, _ = loaded
    ap = window._analysis_panel
    window._open_signal_conditioning()
    ws = _ws(window)
    editor = ap._baseline.reference_editor

    def states():
        return [ap.can_run(), window._act_run_detection.isEnabled(), ws.run_button.isEnabled()]

    assert states() == [True, True, True]
    editor.mode.setCurrentIndex(1)          # a known open-pore current, box still empty
    qtbot.wait(20)
    assert ap._baseline.reference_error()
    assert states() == [False, False, False]
    analysis = next(m for m in window.menuBar().findChildren(QtWidgets.QMenu)
                    if m.menuAction().text().replace("&", "") == "Analysis")
    analysis.aboutToShow.emit()
    editor.mode.setCurrentIndex(0)          # back to automatic
    qtbot.wait(20)
    assert states() == [True, True, True]


# -- the judge -----------------------------------------------------------------------------
def test_chain_change_after_a_schmitt_run_flags_table_card_and_window_and_revert_restores(
        detected, qtbot):
    window, _ = detected
    ap = window._analysis_panel
    det = ap.detection_panel
    table = ap._table
    qtbot.waitUntil(lambda: ap.chain_status().kind == "match", timeout=2000)
    assert not table.notice_chip.isVisible()
    window._open_signal_conditioning()
    ws = _ws(window)
    det.filter_cutoff_hz.setValue(5_000.0)
    qtbot.waitUntil(lambda: ap.chain_status().kind == "differs", timeout=2000)
    assert table.notice_text() == "⚠ Chain changed"
    assert "low-pass 10000 → 5000 Hz" in table.notice_chip.toolTip()
    assert "low-pass 10000 → 5000 Hz" in det.chain_card_text()["status"]
    assert "low-pass" in ws.status_line.text() and ws.revert_button.isEnabled()
    ws.revert_button.click()
    qtbot.waitUntil(lambda: ap.chain_status().kind == "match", timeout=2000)
    assert det.params()["filter_cutoff_hz"] == 10_000.0
    assert not table.notice_chip.isVisible() and not ws.revert_button.isEnabled()


def test_the_chip_menu_reverts_and_opens_through_the_seam(detected, qtbot, menu_answer):
    window, _ = detected
    ap = window._analysis_panel
    det = ap.detection_panel
    det.set_stage_enabled("median", True)
    qtbot.waitUntil(lambda: ap.chain_status().kind == "differs", timeout=2000)
    menu_answer("Revert form to this detection's chain")
    ap._table.notice_chip.click()
    assert not det.median_enabled.isChecked()
    det.set_stage_enabled("median", True)
    qtbot.waitUntil(lambda: ap._table.notice_chip.isVisible(), timeout=2000)
    menu_answer("Open Signal Conditioning…")
    ap._table.notice_chip.click()
    assert _ws(window) is not None and _ws(window).isVisible()


def test_the_judge_is_coalesced(detected, qtbot, monkeypatch):
    window, _ = detected
    ap = window._analysis_panel
    calls = []
    real = AP.judge_chain
    monkeypatch.setattr(AP, "judge_chain", lambda *a, **k: calls.append(1) or real(*a, **k))
    qtbot.wait(20)
    calls.clear()
    ap.detection_panel.protocol_combo.setCurrentText("Cu2-insulin (SavGol baseline)")
    qtbot.waitUntil(lambda: ap.chain_status().kind == "differs", timeout=2000)
    qtbot.wait(20)
    assert len(calls) == 1, f"{len(calls)} judgements for one protocol recall"


def test_a_judge_that_raises_never_stops_the_staleness_or_the_preview(
        detected, qtbot, monkeypatch):
    window, did = detected
    ap = window._analysis_panel
    det = ap.detection_panel

    def boom(*_a, **_k):
        raise TypeError("legacy provenance will not project")

    monkeypatch.setattr(AP, "judge_chain", boom)
    det.filter_preview.setChecked(True)
    with qtbot.waitSignal(ap.stalenessChanged, timeout=2000):
        ap._check_staleness(ap.current_detection())
    det.filter_cutoff_hz.setValue(4_000.0)
    qtbot.waitUntil(lambda: ap.chain_status().kind == "unknown", timeout=2000)
    assert window._trace_view.has_filter_preview() if hasattr(
        window._trace_view, "has_filter_preview") else True
    assert not ap._table.notice_chip.isVisible()


def test_an_ml_table_is_labelled_not_produced_by_this_chain(detected, qtbot):
    window, did = detected
    ap = window._analysis_panel
    schmitt = ap.current_detection()
    ml = AnalysisResult(module_id="detect.keras", module_version="test", doc_id=did,
                        channel=schmitt.channel, params={}, payload=schmitt.payload,
                        calibration_fingerprint=schmitt.calibration_fingerprint)
    window._session.add_result(ml)
    qtbot.wait(20)
    idx = ap._result_selector.findData("detect.keras")
    assert idx >= 0
    ap._result_selector.setCurrentIndex(idx)
    qtbot.waitUntil(lambda: ap.chain_status().kind == "not_classical", timeout=2000)
    assert "ML detector" in ap.chain_status().sentence
    assert "ML detector" in ap.detection_panel.chain_card_text()["status"]
    assert not ap._table.notice_chip.isVisible()


# -- protocols and Copy as Python ----------------------------------------------------------
def test_export_and_import_protocols_round_trip(window, qtbot, tmp_path, monkeypatch):
    ap = window._analysis_panel
    det = ap.detection_panel
    window._open_signal_conditioning()
    ws = _ws(window)
    DetectionPanel._store_protocols({"mine": cr.complete_recipe({"median_enabled": True})})
    det._load_protocols_into_combo()
    target = tmp_path / "out.json"
    monkeypatch.setattr(AP, "ask_export_path", lambda *a, **k: target)
    said = []
    ap.statusMessage.connect(said.append)
    ws.act_export.trigger()
    assert target.exists() and "Exported 1 protocol" in said[-1]
    doc = json.loads(target.read_text())
    assert doc["format"] == cr.PROTOCOL_FILE_FORMAT and list(doc["protocols"]) == ["mine"]
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    ws.act_import.trigger()
    assert said[-1] == "Imported 0 protocols (1 already present)."
    doc["protocols"]["theirs"] = cr.complete_recipe({"savgol_enabled": True})
    target.write_text(json.dumps(doc))
    ws.act_import.trigger()
    assert said[-1] == "Imported 1 protocol (1 already present)."
    names = [det.protocol_combo.itemText(i) for i in range(det.protocol_combo.count())]
    assert "theirs" in names


def test_export_without_suffix_asks_about_the_json_it_writes(window, qtbot, tmp_path,
                                                             monkeypatch):
    """Review D1: the overwrite prompt must judge the file really written (name + .json)."""
    ap = window._analysis_panel
    DetectionPanel._store_protocols({"mine": cr.complete_recipe({"median_enabled": True})})
    keep = tmp_path / "keep.json"
    keep.write_text("precious")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "keep"), "")))
    from nanoscope.gui import export_paths
    asked = []
    monkeypatch.setattr(export_paths, "confirm_overwrite",
                        lambda parent, title, existing: asked.append(list(existing)) or False)
    assert not ap.export_protocols()
    assert asked == [[keep]] and keep.read_text() == "precious"


def test_export_skips_and_names_an_unreadable_stored_protocol(window, qtbot, tmp_path,
                                                             monkeypatch):
    """Review D3: a corrupt store entry blocked every export with a TypeError."""
    ap = window._analysis_panel
    DetectionPanel._store_protocols({"ok": cr.complete_recipe({"median_enabled": True}),
                                     "bad": {"notch_freqs": 5}})
    target = tmp_path / "out.json"
    monkeypatch.setattr(AP, "ask_export_path", lambda *a, **k: target)
    said = []
    ap.statusMessage.connect(said.append)
    assert ap.export_protocols()
    assert list(json.loads(target.read_text())["protocols"]) == ["ok"]
    assert "Exported 1 protocol" in said[-1] and "“bad”" in said[-1]


def test_import_refuses_a_foreign_file_by_name(window, qtbot, tmp_path, monkeypatch,
                                               blocked_modals):
    ap = window._analysis_panel
    bad = tmp_path / "foreign.json"
    bad.write_text(json.dumps({"format": "x", "protocols": {}}))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(bad), "")))
    said = []
    ap.statusMessage.connect(said.append)
    assert not ap.import_protocols()
    assert "foreign.json is not a NanoScope conditioning-protocol file" in said[-1]
    assert blocked_modals and "foreign.json" in blocked_modals[-1].text


def test_copy_as_python_puts_a_parseable_script_on_the_clipboard(loaded, qtbot):
    window, did = loaded
    ap = window._analysis_panel
    ap.detection_panel.set_params({"filter_cutoff_hz": 8_000.0, "median_enabled": True})
    window._open_signal_conditioning()
    ws = _ws(window)
    said = []
    ap.statusMessage.connect(said.append)
    ws.act_copy_python.trigger()
    script = QtGui.QGuiApplication.clipboard().text()
    ast.parse(script)
    doc = window._session.document(did)
    assert f"PATH = {str(doc.meta.path)!r}" in script
    assert "conditioned_trace(raw, fs, PARAMS)" in script
    assert "'median_enabled': True" in script and "'filter_cutoff_hz': 8000.0" in script
    assert said[-1].startswith("Copied a ") and "Python script" in said[-1]


# -- placement, settings, lifetime ---------------------------------------------------------
@pytest.mark.parametrize("screen", [(1512, 875), (1280, 775), (1920, 1055)])
def test_default_geometry_is_right_anchored_and_leaves_the_trace_visible(
        window, qtbot, monkeypatch, screen):
    w, h = screen
    avail = QtCore.QRect(0, 25, w, h)
    monkeypatch.setattr(window, "_available_screen_geometry", lambda: avail)
    window.setGeometry(0, 25, w, h)
    qtbot.wait(20)
    window._open_signal_conditioning()
    ws = _ws(window)
    geo = ws.geometry()
    assert geo.right() == min(window.frameGeometry().right(), avail.right())
    assert geo.width() == max(ws.minimumWidth(), ws.DEFAULT_WIDTH)
    assert avail.contains(geo)
    # The stated budget (default_window_rect): ~416 px of the trace stay visible at 1512,
    # ~824 at 1920 and only ~184 at 1280 (the window cannot be narrower than 760 px).
    trace = window._trace_view
    left = trace.mapToGlobal(QtCore.QPoint(0, 0)).x()
    assert geo.left() - left >= 150, "the main trace is hidden behind the window"
    assert geo.left() >= window._file_dock.mapToGlobal(
        QtCore.QPoint(window._file_dock.width(), 0)).x()


def test_geometry_persists_and_bad_values_fall_back(qapp, qtbot, monkeypatch):
    s = QtCore.QSettings()
    s.setValue("signalConditioning/geometry", "not bytes")
    s.setValue("signalConditioning/splitter", "nor this")
    s.setValue("signalConditioning/followMain", "maybe")
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    monkeypatch.setattr(w, "_available_screen_geometry", lambda: QtCore.QRect(0, 0, 1512, 875))
    w._open_signal_conditioning()
    ws = w._signal_conditioning
    assert ws.isVisible() and ws.width() >= ws.minimumWidth()
    assert "Ignored saved Signal Conditioning" in w.statusBar().currentMessage()
    assert ws.follows_main()
    ws.setGeometry(10, 20, 780, 580)   # inside the offscreen screen, so Qt keeps it
    ws.set_follow_main(False)
    w._save_settings()
    assert isinstance(s.value("signalConditioning/geometry"), QtCore.QByteArray)
    w2 = MainWindow()
    qtbot.addWidget(w2)
    w2.show()
    monkeypatch.setattr(w2, "_available_screen_geometry",
                        lambda: QtCore.QRect(0, 0, 1512, 875))
    w2._open_signal_conditioning()
    ws2 = w2._signal_conditioning
    assert (ws2.width(), ws2.height()) == (780, 580)
    assert not ws2.follows_main()


def test_close_hides_and_keeps_the_object(window, qtbot):
    window._open_signal_conditioning()
    ws = _ws(window)
    ws.close()
    qtbot.wait(10)
    assert not ws.isVisible()
    window._open_signal_conditioning()
    assert _ws(window) is ws and ws.isVisible()


def test_show_main_trace_and_ml_chain_lower_the_window(window, qtbot, monkeypatch):
    window._open_signal_conditioning()
    ws = _ws(window)
    lowered = []
    monkeypatch.setattr(ws, "lower", lambda: lowered.append(1))
    ws.act_show_main.trigger()
    assert lowered == [1]
    ws.act_ml_chain.trigger()
    assert lowered == [1, 1]
    det = window._analysis_panel.detection_panel
    assert det.mode_tabs.currentWidget() is det._ml_page


def test_going_to_an_event_lowers_the_window_too(detected, qtbot, monkeypatch):
    window, _ = detected
    window._open_signal_conditioning()
    ws = _ws(window)
    lowered = []
    monkeypatch.setattr(ws, "lower", lambda: lowered.append(1))
    window._goto_event_in_trace(0)
    assert lowered == [1]


def test_an_unfollowed_window_is_dropped_on_a_recording_switch(loaded, qtbot, tmp_path):
    """Review D4: a range kept for recording A must not be sliced out of recording B."""
    window, did = loaded
    window._open_signal_conditioning()
    ws = _ws(window)
    ws.act_follow.setChecked(False)
    ws._window_range = (40_000, 50_001)
    other = _doc(tmp_path, "b.dat")
    other_id = window._session.add_document(other)
    window._create_default_layers(window._session.document(other_id))
    window._on_selection(other_id, 0)
    qtbot.wait(30)
    assert "b.dat" in ws.windowTitle()
    assert ws.current_window() is None
    script = window._analysis_panel.copy_chain_as_python(ws.current_window())
    assert "START, STOP = 0, raw.size" in script
    # Same source again (window re-pushed without a switch): a kept range survives.
    ws._window_range = (100, 200)
    window._push_conditioning_source()
    assert ws.current_window() == (100, 200)


def test_follow_main_tracks_the_visible_range(loaded, qtbot):
    window, did = loaded
    window._open_signal_conditioning()
    ws = _ws(window)
    window._trace_view.zoom_x_to(0.2, 0.4) if hasattr(window._trace_view, "zoom_x_to") else None
    qtbot.wait(30)
    rng = window._analysis_panel.visible_sample_range()
    assert rng is not None and ws.current_window() == rng
    ws.act_follow.setChecked(False)
    frozen = ws.current_window()
    window._trace_view.zoom_x_to(0.5, 0.9)
    qtbot.wait(30)
    assert ws.current_window() == frozen


def test_the_run_button_mirrors_the_run_action(loaded, qtbot):
    window, _ = loaded
    window._open_signal_conditioning()
    ws = _ws(window)
    assert ws.run_button.isEnabled() == window._act_run_detection.isEnabled()
    assert ws.run_button.isEnabled()


def test_the_module_imports_no_sibling_gui_module():
    """MainWindow is the only wirer: the window reaches no panel in gui/ (every spelling of
    a sibling import — the canonicaliser of test_measure_navigation, inlined)."""
    import pathlib

    mod = importlib.import_module("nanoscope.gui.signal_conditioning")
    reached = set()
    for node in ast.walk(ast.parse(pathlib.Path(mod.__file__).read_text())):
        if isinstance(node, ast.Import):
            reached |= {a.name for a in node.names if a.name.startswith("nanoscope.gui.")}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 1 or (node.level == 2 and (node.module or "").startswith("gui")) \
                    or (node.level == 0 and (node.module or "").startswith("nanoscope.gui")):
                reached.add(f"{node.level}:{node.module}")
    # D-beta: the previews need the mandated worker-thread seam, the readable axes and the
    # trace view's curve factory — all on test_measure_navigation's non-panel allowlist.
    assert reached == {"1:worker_thread", "1:axes", "1:trace_view"}


def test_hosting_does_not_raise_the_inspector_floor(window, qtbot):
    det = window._analysis_panel.detection_panel
    hosted = window._inspector.minimumSizeHint().width()
    det.set_signal_chain_hosted(False)
    qtbot.wait(10)
    standalone = window._inspector.minimumSizeHint().width()
    det.set_signal_chain_hosted(True)
    assert hosted <= standalone


def test_protocol_prompts_open_over_the_workspace(window, qtbot, monkeypatch):
    window._open_signal_conditioning()
    ws = _ws(window)
    parents = []

    def fake_get_text(parent, *_a, **_k):
        parents.append(parent)
        return "", False

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", staticmethod(fake_get_text))
    window._analysis_panel.detection_panel.protocol_save_btn.click()
    assert parents == [ws]


def test_closing_the_recording_refreshes_an_open_window(loaded, qtbot):
    window, did = loaded
    window._open_signal_conditioning()
    ws = _ws(window)
    assert "current" in ws.windowTitle()
    window._session.close_document(did)
    qtbot.wait(20)
    assert ws.windowTitle() == "Signal Conditioning"
    assert ws.chain_status().kind == "none"
