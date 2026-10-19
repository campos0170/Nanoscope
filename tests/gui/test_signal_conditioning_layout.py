"""Signal Conditioning layout: it fits the owner's screens, its rows never reflow, and the
dock's compact card keeps its Open button pinned and moves nothing.

Re-aims the D-relevant assertions of three files that already fail at HEAD on their
fixture (test_lay_detection_panel_reachable, test_lay2_detection_actions_above_the_fold,
test_wire_fold2r_stage_selector): the protocol verbs, the response plot and the stage
selector now live in the Signal Conditioning window.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin bootstrap
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.signal_conditioning import SignalConditioningWindow, default_window_rect

SCREENS = {"1512x900": (1512, 875), "1280x800": (1280, 775)}


@pytest.fixture(autouse=True)
def _fresh_settings():
    QtCore.QSettings().remove("signalConditioning")
    yield
    QtCore.QSettings().remove("signalConditioning")


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    w._analysis_controller.abort()


def _settle(qtbot, w):
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
        if w.layout() is not None:
            w.layout().activate()
    qtbot.wait(10)


@pytest.mark.parametrize("screen", list(SCREENS))
def test_the_workspace_fits_the_owners_screens(window, qtbot, monkeypatch, screen):
    aw, ah = SCREENS[screen]
    avail = QtCore.QRect(0, 25, aw, ah)
    monkeypatch.setattr(window, "_available_screen_geometry", lambda: avail)
    window.setGeometry(avail)
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    det = window._analysis_panel.detection_panel
    # The offscreen screen is 800x800, so lay the window out at the computed size by hand.
    rect = default_window_rect(window.frameGeometry(), window.geometry(), avail,
                               width=ws.DEFAULT_WIDTH)
    ws.resize(rect.size())
    _settle(qtbot, ws)
    hint = ws.minimumSizeHint()
    assert hint.width() <= ws.MIN_WIDTH and hint.height() <= ws.MIN_HEIGHT
    assert ws.stage_list.height() >= 8 * ws.stage_list.sizeHintForRow(0)
    assert ws.stage_list.verticalScrollBar().maximum() == 0
    for i in range(det.filter_stack.count()):
        det.filter_stack.setCurrentIndex(i)
        _settle(qtbot, ws)
        host = ws.form_area.widget()
        assert host.minimumSizeHint().width() <= ws.form_area.viewport().width(), i
    bar = ws.toolbar
    for action in bar.actions():
        if action.isSeparator():
            continue
        shown = bar.widgetForAction(action)
        ext = bar.findChild(QtWidgets.QToolButton, "qt_toolbar_ext_button")
        assert (shown is not None and shown.isVisible()) or (ext is not None and ext.isVisible())


def test_protocol_verbs_are_reachable_and_left_aligned_in_the_workspace(window, qtbot):
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    ws.resize(ws.MIN_WIDTH, ws.MIN_HEIGHT)
    _settle(qtbot, ws)
    det = window._analysis_panel.detection_panel
    combo_rect = det.protocol_combo.geometry()
    combo_bottom = det.protocol_combo.mapTo(ws, QtCore.QPoint(0, combo_rect.height())).y()
    xs = []
    for button in (det.protocol_save_btn, det.protocol_rename_btn, det.protocol_delete_btn):
        assert button.isVisibleTo(ws)
        top_left = button.mapTo(ws, QtCore.QPoint(0, 0))
        assert top_left.y() >= combo_bottom, "the verbs sit below the combo"
        assert ws.rect().contains(QtCore.QRect(top_left, button.size()))
        xs.append(top_left.x())
    left = det.protocol_combo.parentWidget().mapTo(ws, QtCore.QPoint(0, 0)).x()
    assert xs == sorted(xs) and xs[0] - left <= 12, "left-aligned under the combo"


def test_the_response_plot_gets_the_workspace_height(window, qtbot):
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    # D-beta: Time is now the first tab; the |H(f)| plot is the Response tab's Magnitude view.
    ws.set_preview_tab("response")
    ws.set_response_view("magnitude")
    ws.resize(ws.MIN_WIDTH, ws.MIN_HEIGHT)
    _settle(qtbot, ws)
    plot = window._analysis_panel.detection_panel.response_plot
    assert plot.isVisibleTo(ws)
    assert plot.height() > 190, "the dock-era 190 px cap was lifted"


def test_the_status_row_never_reflows(window, qtbot):
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    ws.resize(900, 640)
    _settle(qtbot, ws)
    from nanoscope.analysis.conditioning_recipes import ChainStatus

    def geometry():
        return [w.geometry() for w in (ws.status_line, ws.revert_button, ws.run_button)]

    before = geometry()
    long = ChainStatus("differs", "⚠ " + "a very long sentence " * 30, (), {"x": 1}, "Schmitt")
    for status in (long, ChainStatus.none(), ChainStatus("match", "short", detector_label="S")):
        ws.set_chain_status(status)
        ws.set_run_enabled(status.kind == "match")
        _settle(qtbot, ws)
        assert geometry() == before
        assert ws.revert_button.isVisible() and ws.run_button.isVisible()


@pytest.mark.parametrize("screen", list(SCREENS))
def test_the_open_button_is_pinned_above_the_fold(window, qtbot, screen):
    w, h = SCREENS[screen]
    window.resize(w, h)
    det = window._analysis_panel.detection_panel
    window._reveal_detection_panel()
    det.show_signal_chain()
    _settle(qtbot, window)
    for stale in (False, True):
        if stale:
            window._analysis_panel._stale.show_message(
                "Calibration changed since this detection ran — " + "long reason " * 40,
                kind="calibration", action_label="Re-run")
            _settle(qtbot, window)
        for widget in (det.open_conditioning_button, det.filter_preview):
            assert widget.isVisible(), widget
            assert not widget.visibleRegion().isEmpty(), (screen, stale, widget)


def test_the_card_moves_nothing_across_states(window, qtbot):
    window.resize(1512, 875)
    det = window._analysis_panel.detection_panel
    window._reveal_detection_panel()
    det.show_signal_chain()
    _settle(qtbot, window)
    rows = (det.chain_status_note, det.chain_warning_note, det.chain_protocol_note,
            det.chain_scope_note, det.chain_order_note)

    def geometry():
        return [r.geometry() for r in rows] + [det.open_conditioning_button.geometry(),
                                               det.filter_preview.geometry()]

    before = geometry()
    states = (
        lambda: det.set_hp_enabled(True),
        lambda: det.set_normalize_enabled(True),
        lambda: det.protocol_combo.setCurrentText("Zn2-insulin (zn_30h)"),
        lambda: det.set_chain_status("⚠ The form differs from the chain the Schmitt "
                                     "detection on screen ran with: " + "x → y; " * 20, True),
        lambda: det.set_preview_error("a detrend window longer than the trace"),
        lambda: det.set_chain_status("", False),
    )
    for state in states:
        state()
        _settle(qtbot, window)
        assert geometry() == before
    assert det.preview_error_label.isVisible()
    assert det.preview_error_label.y() >= det.chain_order_note.geometry().bottom()


def test_the_window_class_is_built_from_its_own_defaults():
    assert SignalConditioningWindow.MIN_WIDTH == 760 and SignalConditioningWindow.MIN_HEIGHT == 560
    assert SignalConditioningWindow.DEFAULT_WIDTH >= SignalConditioningWindow.MIN_WIDTH


def test_every_preview_tab_fits_the_minimum_window(window, qtbot):
    """D-beta: the Time, Spectrum and Response controls never floor the window, and the
    stage editor keeps its column."""
    window._open_signal_conditioning()
    ws = window._signal_conditioning
    for tab in ("spectrum", "response", "time"):
        ws.set_preview_tab(tab)
        if tab == "response":
            for view in ("phase", "group_delay", "step", "magnitude"):
                ws.set_response_view(view)
    ws.resize(ws.MIN_WIDTH, ws.MIN_HEIGHT)
    _settle(qtbot, ws)
    hint = ws.minimumSizeHint()
    assert hint.width() <= ws.MIN_WIDTH and hint.height() <= ws.MIN_HEIGHT
    assert ws.splitter.sizes()[0] >= ws.LEFT_MIN_WIDTH
    pane = ws.preview_tabs
    for widget in (*ws.time_mode_buttons.values(), ws.exact_button):
        top_left = widget.mapTo(pane, QtCore.QPoint(0, 0))
        assert pane.rect().contains(QtCore.QRect(top_left, widget.size())), widget.text()
