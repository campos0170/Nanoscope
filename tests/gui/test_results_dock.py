"""The Results dock (change A): the event table and its tabs under the trace, not in the Inspector.

Before it, the Inspector's Analysis tab split ONE column between the parameter forms and the
results: measured at 1512x900 with the 13 pt system font, the forms got a 410x256 viewport
(34 px with the stale banner up — the banner wrapped to 254 px) and the event table 2 rows
(0 with the banner). Now the forms keep the Inspector's full height and the results surface —
owned and wired by AnalysisPanel exactly as before — is the ``ResultsDock`` under the trace,
with two fixed-height header rows: the toolbar (Table selector, Columns…, Merge…, the table's
constant columns, advisories) and the status line (the event count, or the one-line amber
stale notice in the same slot).

Every test here that touches the dock's place, height or visibility runs in a PRIVATE
QSettings namespace: MainWindow saves its layout on close, and a closed or floating Results
dock leaking into the suite's shared namespace would follow every later window.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pathlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from nanoscope.analysis.modules import register_all  # noqa: E402
from nanoscope.core.calibration import ChannelCalibration  # noqa: E402
from nanoscope.core.document import TraceDocument  # noqa: E402
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta  # noqa: E402
from nanoscope.gui import analysis_panel as AP  # noqa: E402
from nanoscope.gui import main_window as MW  # noqa: E402
from nanoscope.gui.analysis_panel import AnalysisPanel  # noqa: E402
from nanoscope.gui.main_window import MainWindow  # noqa: E402

FS = 50_000.0
I0 = -1000.0
SIZES = ((1512, 900), (1280, 800), (1920, 1080))

#: The house number for "this widget does not pin a dock" (test_window_layout.py's).
_SMALL_ENOUGH_TO_SHRINK = 220


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


@pytest.fixture
def isolated_settings(qapp):
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("ResultsDock")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


@pytest.fixture
def real_font(qapp):
    """The owner's real look: the 13 pt system font (the suite runs at 9 pt). Widgets read
    the font when they are built, so windows must be created inside this fixture."""
    old = qapp.font()
    qapp.setFont(QtGui.QFont(".AppleSystemUIFont", 13))
    try:
        yield
    finally:
        qapp.setFont(old)


def _settle(qapp, ms: int = 30) -> None:
    QTest.qWait(ms)
    qapp.processEvents()


def _open(qapp, qtbot, size=(1512, 900)) -> MainWindow:
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(*size)
    win.show()
    _settle(qapp)
    win._inspector_tabs.setCurrentWidget(win._analysis_panel)
    win.resize(*size)
    _settle(qapp)
    assert (win.width(), win.height()) == tuple(size)
    return win


@pytest.fixture
def window(qapp, isolated_settings, qtbot):
    win = _open(qapp, qtbot)
    yield win
    win._analysis_controller.abort()
    win.close()


def _signal(n: int = 60_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in ((10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300),
                 (45_000, 45_400), (52_000, 52_250)):
        x[s:e] -= 300.0
    return x.astype(np.float32)


def _doc(path: Path) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    sig = _signal()
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=int(sig.size),
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _detect(win: MainWindow, qtbot, tmp_path, name: str = "a.dat") -> str:
    """A real Schmitt detection on screen, its stamps written by the engine (so the table
    starts CLEAN — neither stamp stale)."""
    did = win._session.add_document(_doc(tmp_path / name))
    win._create_default_layers(win._session.document(did))
    win._on_selection(did, 0)
    qtbot.wait(20)
    panel = win._analysis_panel
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    assert panel.current_detection() is not None
    assert panel._stale.isHidden()
    return did


def _rect(widget: QtWidgets.QWidget, host: QtWidgets.QWidget) -> QtCore.QRect:
    return QtCore.QRect(widget.mapTo(host, QtCore.QPoint(0, 0)), widget.size())


def _panels_menu(win: MainWindow) -> QtWidgets.QMenu:
    # findChildren + menuAction, NEVER QAction.menu(): that hands the live menu to Python,
    # which deletes it when the wrapper dies.
    for menu in win.menuBar().findChildren(QtWidgets.QMenu):
        if menu.menuAction().text().replace("&", "") == "Panels":
            return menu
    raise AssertionError("no View ▸ Panels menu")


def _stale_by_gain(win: MainWindow, did: str, gain: float = 2.0) -> None:
    win._session.set_calibration(did, 0, ChannelCalibration(gain, 0.0, "pA"),
                                 write_sidecar=False)


def _as_voltage(gain: float) -> ChannelCalibration:
    return ChannelCalibration(gain, 0.0, "pA", Provenance.MEASURED,
                              "really the command channel", kind=ChannelKind.VOLTAGE)


# ================================================================================================
# 1. the dock exists, is a real dock, and the panel still owns what is in it
# ================================================================================================

def test_the_results_dock_is_a_real_dock_with_a_stable_object_name(window):
    dock = window._results_dock
    assert dock.objectName() == "ResultsDock"      # what saveState/restoreState key on
    assert dock.windowTitle() == "Results"
    assert window.dockWidgetArea(dock) == QtCore.Qt.BottomDockWidgetArea
    feats = dock.features()
    for flag in (QtWidgets.QDockWidget.DockWidgetClosable,
                 QtWidgets.QDockWidget.DockWidgetMovable,
                 QtWidgets.QDockWidget.DockWidgetFloatable,
                 QtWidgets.QDockWidget.DockWidgetVerticalTitleBar):
        assert feats & flag, flag
    assert dock.isVisible()


def test_view_panels_lists_the_docks_and_results_has_its_key(window, qapp):
    actions = [a for a in _panels_menu(window).actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == ["File Manager", "Inspector", "Results"]
    toggle = actions[2]
    assert toggle is window._act_toggle_results
    assert toggle.shortcut() == QtGui.QKeySequence("Ctrl+Shift+R")
    assert toggle.statusTip()
    toggle.trigger()
    _settle(qapp)
    assert window._results_dock.isHidden()
    toggle.trigger()
    _settle(qapp)
    assert window._results_dock.isVisible()


def test_no_other_action_claims_the_results_shortcut(window):
    key = QtGui.QKeySequence("Ctrl+Shift+R")
    owners = [a for a in window.findChildren(QtGui.QAction) if key in a.shortcuts()]
    assert owners == [window._act_toggle_results]


def test_the_results_surface_lives_in_the_dock_and_the_panel_still_owns_it(window):
    panel = window._analysis_panel
    dock, inspector = window._results_dock, window._inspector
    for name in ("_table", "results_tabs", "_stale", "merge_controls", "_result_selector",
                 "_result_notice", "_stats", "_score", "merge_toggle", "_results_summary"):
        widget = getattr(panel, name)
        assert dock.isAncestorOf(widget), name
        assert not inspector.isAncestorOf(widget), name
    assert dock.widget() is panel._results_host
    assert panel.findChild(QtWidgets.QSplitter) is None
    root = panel.layout()
    assert root.indexOf(panel._results_host) == -1
    assert root.stretch(root.indexOf(panel.detection_panel)) == 1
    assert [panel.results_tabs.tabText(i) for i in range(4)] == [
        "Events", "Summary", "Fits", "Quality"]
    assert all(panel.results_tabs.tabToolTip(i) for i in range(4))


def test_a_standalone_panel_keeps_its_results_inline(qapp, qtbot):
    from nanoscope.core.session import Session
    from nanoscope.gui.analysis_worker import AnalysisController
    from nanoscope.gui.trace_view import TraceView

    session = Session()
    view = TraceView(session)
    qtbot.addWidget(view)
    controller = AnalysisController(session)
    panel = AnalysisPanel(session, controller, view)
    qtbot.addWidget(panel)
    assert panel.isAncestorOf(panel._table)
    assert panel.layout().indexOf(panel._results_host) >= 0
    host = panel.take_results_widget()
    assert host is panel._results_host
    assert panel.layout().indexOf(host) == -1
    controller.abort()


def test_no_new_sibling_import():
    """The move is a reparent done by the owner: analysis_panel reaches no new gui module."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    try:
        from test_measure_navigation import _gui_modules_reached
    finally:
        sys.path.pop(0)
    assert _gui_modules_reached(AP) == {
        "analysis_worker", "baseline_panel", "baseline_workspace", "categorical",
        "export_paths", "hand_row_preview", "inspector_widgets", "measure_panel",
        "merge_panel", "open_pore_preview", "overlays", "panels", "trace_view"}


# ================================================================================================
# 2. where it sits and how tall it opens
# ================================================================================================

@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_bottom_dock_lives_between_the_side_columns(qapp, isolated_settings, qtbot, size):
    win = _open(qapp, qtbot, size)
    try:
        assert win.corner(QtCore.Qt.BottomLeftCorner) == QtCore.Qt.LeftDockWidgetArea
        assert win.corner(QtCore.Qt.BottomRightCorner) == QtCore.Qt.RightDockWidgetArea
        dock, central = win._results_dock, win.centralWidget()
        assert abs(dock.x() - central.x()) <= 2
        assert abs(dock.width() - central.width()) <= 2
        column = central.height() + dock.height() + MW._DOCK_SEPARATOR_WIDTH
        for side in (win._file_dock, win._inspector):
            assert abs(side.height() - column) <= 2, (side.objectName(), side.height(), column)
    finally:
        win.close()


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_first_launch_height_is_derived_and_capped(qapp, isolated_settings, qtbot, size):
    win = _open(qapp, qtbot, size)
    try:
        height = win._results_dock.height()
        assert abs(height - win._results_default_height()) <= 6, height
        assert height <= win._dock_area_height() * MW._RESULTS_MAX_HEIGHT_FRACTION + 2
        assert height >= MW._RESULTS_MIN_DEFAULT_HEIGHT - 2
        assert not win._results_layout_pending
    finally:
        win.close()


def test_the_rows_the_height_promises_are_visible(qapp, isolated_settings, qtbot, tmp_path):
    """``results_height_for_rows`` is layout arithmetic; the painted table must agree."""
    win = _open(qapp, qtbot, (1920, 1080))
    try:
        _detect(win, qtbot, tmp_path)
        panel = win._analysis_panel
        view = panel._table.view
        row = view.verticalHeader().defaultSectionSize()
        dock_h = win._results_dock.height()
        promised = max(r for r in range(0, 40) if panel.results_height_for_rows(r) <= dock_h)
        assert view.viewport().height() >= promised * row, (view.viewport().height(), promised)
        assert promised >= 5
    finally:
        win._analysis_controller.abort()
        win.close()


def test_compact_rows(window):
    view = window._analysis_panel._table.view
    assert view.verticalHeader().defaultSectionSize() == max(20, view.fontMetrics().height() + 8)


def test_rows_are_24_px_at_the_macs_system_font(qapp, isolated_settings, qtbot):
    """The ~24 px target is the Mac's: 13 pt is 13 px at its 72 dpi (line height 16). The
    offscreen 96 dpi reference inflates the same font to 17 px, and rows to 28."""
    old = qapp.font()
    font = QtGui.QFont(".AppleSystemUIFont")
    font.setPixelSize(13)
    qapp.setFont(font)
    try:
        win = MainWindow()
        qtbot.addWidget(win)
        row = win._analysis_panel._table.view.verticalHeader().defaultSectionSize()
        assert abs(row - 24) <= 1, row
        win.close()
    finally:
        qapp.setFont(old)


def test_four_full_rows_open_at_1512x900_at_the_real_look(real_font, qapp, isolated_settings,
                                                          qtbot, tmp_path):
    """The owner's screen, first launch, a table whose constant columns are captioned: the
    dock opens at a third of the dock area and that third holds four whole rows."""
    win = _open(qapp, qtbot)
    try:
        _detect(win, qtbot, tmp_path)
        _settle(qapp)
        table = win._analysis_panel._table
        view = table.view
        assert not table.constants_label.isHidden()
        assert win._analysis_panel._results_toolbar.isAncestorOf(table.constants_label), (
            "the constants caption costs the dock a row of its own again")
        row = view.verticalHeader().defaultSectionSize()
        assert view.viewport().height() >= 4 * row, (view.viewport().height(), row)
    finally:
        win._analysis_controller.abort()
        win.close()


def test_the_inspector_parameters_get_their_height_back(real_font, qapp, isolated_settings,
                                                        qtbot, tmp_path):
    """HEAD: the Detection page's viewport was 256 px at 1512x900, 34 with the banner up."""
    win = _open(qapp, qtbot)
    try:
        did = _detect(win, qtbot, tmp_path)
        dp = win._analysis_panel.detection_panel
        dp.show_detection_tab()
        _settle(qapp)
        scroll = dp.page_scrolls[dp.mode_tabs.currentWidget()]
        clean = scroll.viewport().height()
        assert clean >= 500, clean
        _stale_by_gain(win, did)
        _settle(qapp)
        assert not win._analysis_panel._stale.isHidden()
        assert scroll.viewport().height() == clean
    finally:
        win._analysis_controller.abort()
        win.close()


def test_closing_the_results_dock_returns_the_trace_height(window, qapp, qtbot, tmp_path):
    _detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    dock = window._results_dock
    panel._table.view.selectRow(1)
    _settle(qapp)
    frame_before = panel._table.dataframe.copy()
    selected = panel._table.view.selectionModel().selectedRows()
    open_trace, open_dock = window._trace_view.height(), dock.height()
    dock.close()
    _settle(qapp)
    assert window._trace_view.height() >= open_trace + open_dock - 10
    dock.show()
    _settle(qapp)
    assert abs(dock.height() - open_dock) <= 6
    assert panel._table.dataframe.equals(frame_before)
    assert panel._table.view.selectionModel().selectedRows() == selected


# ================================================================================================
# 3. persistence: versioned, migrated, validated
# ================================================================================================

def test_a_legacy_window_state_keeps_the_users_widths_and_gains_the_dock(
        qapp, isolated_settings, qtbot):
    """A layout saved before the Results dock (saveState version 0) carries Qt's default
    corners and no ResultsDock. Restored, the user's widths survive and the dock arrives
    under the trace at its opening height; nothing goes on the info bar."""
    old = _open(qapp, qtbot)
    old.removeDockWidget(old._results_dock)
    for corner, area in ((QtCore.Qt.TopLeftCorner, QtCore.Qt.TopDockWidgetArea),
                         (QtCore.Qt.BottomLeftCorner, QtCore.Qt.BottomDockWidgetArea),
                         (QtCore.Qt.TopRightCorner, QtCore.Qt.TopDockWidgetArea),
                         (QtCore.Qt.BottomRightCorner, QtCore.Qt.BottomDockWidgetArea)):
        old.setCorner(corner, area)
    old.resizeDocks([old._inspector], [400], QtCore.Qt.Horizontal)
    _settle(qapp)
    legacy = old.saveState(0)
    old.close()
    QtCore.QSettings().setValue("windowState", legacy)

    win = _open(qapp, qtbot)
    try:
        dock = win._results_dock
        assert abs(win._inspector.width() - 400) <= 6, win._inspector.width()
        assert dock.isVisible() and not dock.isFloating()
        assert win.dockWidgetArea(dock) == QtCore.Qt.BottomDockWidgetArea
        assert win.corner(QtCore.Qt.BottomLeftCorner) == QtCore.Qt.LeftDockWidgetArea
        assert win.corner(QtCore.Qt.BottomRightCorner) == QtCore.Qt.RightDockWidgetArea
        assert abs(dock.height() - win._results_default_height()) <= 6, dock.height()
        assert win._info_bar.isHidden(), win._info_bar._label.text()
        assert "Results panel" in win.statusBar().currentMessage()
    finally:
        win.close()


def test_a_current_window_state_round_trips(qapp, isolated_settings, qtbot):
    first = _open(qapp, qtbot)
    first.resizeDocks([first._results_dock], [350], QtCore.Qt.Vertical)
    _settle(qapp)
    dragged = first._results_dock.height()
    first.close()
    win = _open(qapp, qtbot)
    assert not win._results_layout_pending, "a v1 restore must keep the user's height"
    assert abs(win._results_dock.height() - dragged) <= 6
    win._results_dock.close()
    _settle(qapp)
    win.close()
    closed = _open(qapp, qtbot)
    assert closed._results_dock.isHidden(), "closed stays closed"
    closed._results_dock.show()
    closed._results_dock.setFloating(True)
    _settle(qapp)
    closed.close()
    floating = _open(qapp, qtbot)
    try:
        assert floating._results_dock.isFloating(), "floating stays floating"
    finally:
        floating.close()


def test_the_corners_are_reapplied_after_any_restore(qapp, isolated_settings, qtbot):
    first = _open(qapp, qtbot)
    first.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.BottomDockWidgetArea)
    first.close()
    win = _open(qapp, qtbot)
    try:
        assert win.corner(QtCore.Qt.BottomLeftCorner) == QtCore.Qt.LeftDockWidgetArea
    finally:
        win.close()


def test_a_garbage_window_state_is_still_refused_and_named(qapp, isolated_settings, qtbot):
    QtCore.QSettings().setValue("windowState", QtCore.QByteArray(b"junk"))
    win = _open(qapp, qtbot)
    try:
        assert "windowState" in win._info_bar._label.text()
        assert abs(win._results_dock.height() - win._results_default_height()) <= 6
    finally:
        win.close()


def test_the_window_state_is_written_at_the_current_version(qapp, isolated_settings, qtbot):
    win = _open(qapp, qtbot)
    win.close()
    state = QtCore.QSettings().value("windowState")
    probe = MainWindow()
    qtbot.addWidget(probe)
    assert not probe.restoreState(state, 0), "saved at version 0"
    assert probe.restoreState(state, MW._WINDOW_STATE_VERSION)


# ================================================================================================
# 4. the two header rows: fixed, and the stale line moves nothing
# ================================================================================================

def _snapshot(win: MainWindow) -> dict[str, QtCore.QRect]:
    panel = win._analysis_panel
    dp = panel.detection_panel
    return {name: _rect(w, win) for name, w in (
        ("tabbar", panel.results_tabs.tabBar()),
        ("table_viewport", panel._table.view.viewport()),
        ("toolbar", panel._results_toolbar),
        ("status_line", panel._results_status),
        ("selector", panel._result_selector),
        ("columns", panel._table.columns_button),
        ("merge", panel.merge_toggle),
        ("mode_tabs", dp.mode_tabs),
        ("detect", dp.detect_button),
        ("dock", win._results_dock),
        ("trace", win._trace_view))}


@pytest.mark.parametrize("cause", ["calibration", "baseline knob"])
def test_the_stale_line_moves_nothing(window, qapp, qtbot, tmp_path, cause):
    did = _detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _settle(qapp)
    before = _snapshot(window)
    bl = panel.baseline_panel
    rough = bl.rough_s.value()
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        if cause == "calibration":
            _stale_by_gain(window, did)
        else:
            bl.rough_s.setValue(rough * 4)
    _settle(qapp)
    assert not panel._stale.isHidden() and panel._results_summary.isHidden()
    assert _snapshot(window) == before, "the stale line moved something"
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        if cause == "calibration":
            _stale_by_gain(window, did, 1.0)
        else:
            bl.rough_s.setValue(rough)
    _settle(qapp)
    assert panel._stale.isHidden() and not panel._results_summary.isHidden()
    assert _snapshot(window) == before


def test_a_gate_change_on_the_stale_line_moves_nothing(window, qapp, qtbot, tmp_path):
    did = _detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _stale_by_gain(window, did, 2.5)
    _settle(qapp)
    banner = panel._stale
    assert banner._btn.isEnabled()
    before, btn_before = _snapshot(window), _rect(banner._btn, window)
    window._session.set_calibration(did, 0, _as_voltage(2.5), write_sidecar=False)
    _settle(qapp)
    assert not banner._btn.isEnabled()
    after = _snapshot(window)
    # The Detect page's own gate note is the DetectionPanel's business, not this line's.
    for name in ("mode_tabs", "detect"):
        before.pop(name), after.pop(name)
    assert after == before
    assert _rect(banner._btn, window) == btn_before, "the gated reason pushed the button"


def test_the_stale_line_is_one_line_and_details_holds_the_whole_sentence(
        window, qapp, qtbot, tmp_path):
    did = _detect(window, qtbot, tmp_path)
    _stale_by_gain(window, did)
    _settle(qapp)
    panel = window._analysis_panel
    banner = panel._stale
    assert banner.height() <= panel._results_status.height()
    assert banner._summary.text() == "⚠ " + AP._STALE_SUMMARIES["calibration"]
    assert banner._summary.toolTip() == banner.message()
    assert banner._label.text() == banner.message()
    assert banner.message().startswith("Calibration changed since this detection ran")
    actions = banner._details_menu.actions()
    assert len(actions) == 1 and isinstance(actions[0], QtWidgets.QWidgetAction)
    pane = actions[0].defaultWidget()
    assert pane.isAncestorOf(banner._label) and pane.isAncestorOf(banner._why)
    assert banner._label.wordWrap() and banner._label.width() <= 480
    assert banner._label.textInteractionFlags() & QtCore.Qt.TextSelectableByMouse
    assert banner._details.focusPolicy() == QtCore.Qt.StrongFocus
    assert banner.minimumSizeHint().width() == 0
    # The popup is a plain popup(), never a QToolButton menu (whose C++ exec would block).
    assert banner._details.popupMode() != QtWidgets.QToolButton.InstantPopup
    banner._details.click()          # the suite's popup net closes it again
    _settle(qapp)
    assert banner._details_menu.actions() == actions, "the popup is built once, not per click"


@pytest.mark.parametrize("size", [(1512, 900), (1280, 800)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("cause", ["calibration", "baseline knob"])
def test_a_gated_action_says_why_on_the_visible_line(real_font, qapp, isolated_settings,
                                                     qtbot, tmp_path, size, cause):
    """The relaxed D2/CW-1 invariant: the refusal STARTS on screen before any click — in the
    painted text of the line, not only in a tooltip — and the full sentence is in Details.
    On a narrow dock the painted copy gives up the verdict, never the refusal's opening."""
    win = _open(qapp, qtbot, size)
    try:
        did = _detect(win, qtbot, tmp_path)
        panel = win._analysis_panel
        if cause == "calibration":
            win._session.set_calibration(did, 0, _as_voltage(2.5), write_sidecar=False)
            verdict = "⚠ " + AP._STALE_SUMMARIES["calibration"]
        else:
            win._session.set_calibration(did, 0, _as_voltage(1.0), write_sidecar=False)
            _settle(qapp)
            assert panel._stale.isHidden()
            panel.baseline_panel.rough_s.setValue(panel.baseline_panel.rough_s.value() * 4)
            verdict = "⚠ " + AP._STALE_SUMMARIES["baseline"]
        _settle(qapp)
        banner = panel._stale
        assert not banner.isHidden() and not banner._btn.isEnabled()
        painted = QtWidgets.QLabel.text(banner._summary)
        reason = banner.action_refusal().split(" is unavailable: ", 1)[1]
        assert "unavailable: " + AP._opening_words(reason) in painted, painted
        if size == (1512, 900):
            assert painted.startswith("⚠ Depths stale"), painted
        assert banner._summary.text().startswith(verdict)
        assert banner._why.text() == banner.action_refusal()
        assert banner._why.text().startswith(banner._btn.text() + " is unavailable:")
        assert not banner._why.isHidden()
        assert banner.action_refusal() in banner._summary.toolTip()
        assert banner._btn.width() >= banner._btn.sizeHint().width()
    finally:
        win._analysis_controller.abort()
        win.close()


#: Every (kind, short verdict) `_judge_staleness` can put on the line, and every action label.
_VERDICTS = (("calibration", None), ("baseline", None), ("reconciliation_source", None),
             ("calibration", "Calibration and open-pore reference changed"),
             ("baseline", "Open-pore reference changed"),
             ("baseline", "Estimator changed — refresh the sources"),
             ("baseline", "Depths stale — predates the current estimator"))
_ACTIONS = ("Re-run", "Run detection", "Re-measure depths", "Refresh sources first",
            "Review sources first")


@pytest.mark.parametrize("size", [(1512, 900), (1280, 800)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_every_gated_line_starts_its_reason_at_the_real_look(real_font, qapp, isolated_settings,
                                                             qtbot, tmp_path, size):
    """Every verdict x action x refusal, including the reconciled tables' (always gated) and
    the reference-changed ones: "unavailable: " and the reason's first word are PAINTED."""
    win = _open(qapp, qtbot, size)
    try:
        did = win._session.add_document(_doc(tmp_path / "v.dat"))
        win._session.set_calibration(did, 0, _as_voltage(1.0), write_sidecar=False)
        gate_reason = win._analysis_panel._gate_for(did, 0)[1]
        assert gate_reason
        reasons = (gate_reason, AP._RECONCILED_STALE_REFUSAL,
                   AP._RECONCILED_SOURCE_CHANGE_REFUSAL, AP._RECONCILED_BASELINE_REFUSAL)
        banner = win._analysis_panel._stale
        missed = []
        for kind, verdict in _VERDICTS:
            for action in _ACTIONS:
                for reason in reasons:
                    banner.show_message("The full sentence.", kind=kind, action_label=action,
                                        summary=verdict)
                    banner.set_action_gate(False, reason)
                    qapp.processEvents()
                    painted = QtWidgets.QLabel.text(banner._summary)
                    if "unavailable: " + AP._opening_words(reason) not in painted:
                        missed.append((verdict or kind, action, painted))
        assert not missed, missed
    finally:
        win.close()


@pytest.mark.parametrize("size", [(1512, 900), (1280, 800)], ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("cause", ["calibration", "baseline knob"])
def test_the_stale_line_fits_its_budget_at_the_real_look(real_font, qapp, isolated_settings,
                                                         qtbot, tmp_path, size, cause):
    """At 1512x900 the whole short verdict reads; at 1280x800 at least "⚠ Depths stale",
    and the action button is never squeezed below its text."""
    win = _open(qapp, qtbot, size)
    try:
        did = _detect(win, qtbot, tmp_path)
        panel = win._analysis_panel
        if cause == "calibration":
            _stale_by_gain(win, did)
        else:
            panel.baseline_panel.rough_s.setValue(panel.baseline_panel.rough_s.value() * 4)
        _settle(qapp)
        banner = panel._stale
        assert not banner.isHidden()
        painted = QtWidgets.QLabel.text(banner._summary)
        if size == (1512, 900):
            assert painted == banner._summary.text(), painted
        else:
            assert painted.startswith("⚠ Depths stale"), painted
        for button in (banner._btn, banner._details):
            assert button.width() >= button.sizeHint().width(), button.text()
            assert win.rect().contains(_rect(button, win)), button.text()
    finally:
        win._analysis_controller.abort()
        win.close()


def test_the_header_rows_are_fixed_height(window):
    panel = window._analysis_panel
    for row in (panel._results_toolbar, panel._results_status):
        assert row.minimumHeight() == row.maximumHeight() == row.height()
        assert row.minimumSizeHint().width() == 0, row.objectName()
    assert panel._results_toolbar.objectName() == "resultsToolbar"
    assert panel._results_status.objectName() == "resultsStatusLine"
    assert panel._result_selector.objectName() == "eventResultSelector"


def test_the_result_selector_gives_way_first_on_a_narrow_dock(real_font, qapp,
                                                              isolated_settings, qtbot):
    window = _open(qapp, qtbot)
    panel = window._analysis_panel
    toolbar = panel._results_toolbar.layout()
    assert toolbar.indexOf(panel._table.columns_button) >= 0, "Columns… is in the toolbar"
    assert toolbar.indexOf(panel.merge_toggle) >= 0
    assert panel.results_tabs.cornerWidget(QtCore.Qt.TopRightCorner) is None
    selector = panel._result_selector
    assert selector.minimumWidth() <= 100
    room = window.width() - 2 * 6 - window._file_dock.width()
    window.resizeDocks([window._inspector], [room - 400], QtCore.Qt.Horizontal)
    _settle(qapp)
    assert selector.width() < selector.sizeHint().width()
    for button in (panel._table.columns_button, panel.merge_toggle):
        assert button.width() >= button.sizeHint().width()


def test_the_toolbar_fits_the_1280_dock_at_the_real_look(real_font, qapp, isolated_settings,
                                                         qtbot, tmp_path):
    """With a result shown (so the advisories link is up) nothing on the row is squeezed
    or clipped; the selector has given way instead."""
    win = _open(qapp, qtbot, (1280, 800))
    try:
        _detect(win, qtbot, tmp_path)
        _settle(qapp)
        panel = win._analysis_panel
        bar = panel._results_toolbar
        assert not panel._result_notice.isHidden()
        for widget in (panel._table.columns_button, panel.merge_toggle, panel._result_notice):
            assert widget.width() >= widget.sizeHint().width(), widget
            assert bar.rect().contains(_rect(widget, bar)), widget
    finally:
        win._analysis_controller.abort()
        win.close()


@pytest.mark.parametrize("size", [(1512, 900), (1280, 800)], ids=["1512x900", "1280x800"])
def test_the_chain_chip_lives_in_the_results_toolbar_and_moves_no_button(
        size, real_font, qapp, isolated_settings, qtbot, tmp_path):
    """The Signal Conditioning chip used to sit in the table's own toolbar row, which this dock
    empties; here it shares the Results toolbar. Its appearance (after a chain edit, made in the
    Signal Conditioning window) never squeezes a button at 1512x900 and never pushes one out
    of the row at 1280x800."""
    win = _open(qapp, qtbot, size)
    try:
        _detect(win, qtbot, tmp_path)
        _settle(qapp)
        panel = win._analysis_panel
        chip = panel._table.notice_chip
        assert panel._results_toolbar.isAncestorOf(chip)
        assert panel._table.column_toolbar.indexOf(chip) == -1
        buttons = (panel._table.columns_button, panel.merge_toggle, panel._result_notice)
        panel._table.set_notice("⚠ Chain changed", "the form's chain differs")
        _settle(qapp)
        assert chip.isVisible()
        bar = panel._results_toolbar
        if size == (1512, 900):
            # At 1280x800 the row has no spare width, so a shown chip (only while the form's
            # chain differs from the detection) costs Columns… a few px; nothing leaves the row.
            for widget in buttons:
                assert widget.width() >= widget.sizeHint().width(), widget
        for widget in (*buttons, chip):
            assert bar.rect().contains(_rect(widget, bar)), widget
    finally:
        win._analysis_controller.abort()
        win.close()


# ================================================================================================
# 5. floors
# ================================================================================================

def test_the_results_dock_imposes_no_width_floor(window, qapp):
    dock = window._results_dock
    assert dock.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK
    room = window.width() - 2 * 6 - window._file_dock.width()
    window.resizeDocks([window._inspector], [room - 300], QtCore.Qt.Horizontal)
    _settle(qapp)
    assert abs(dock.width() - window.centralWidget().width()) <= 2


def test_the_results_dock_adds_only_its_header_to_the_minimum_height(window, qapp, qtbot):
    assert window._results_dock.minimumSizeHint().height() <= 80
    window._results_dock.close()
    qtbot.wait(30)
    window.resize(775, 450)
    _settle(qapp)
    assert window.height() <= 460


# ================================================================================================
# 6. what lives in it still works
# ================================================================================================

def test_merge_toggle_shows_the_strip_above_the_table_and_switches_to_events(window, qapp):
    panel = window._analysis_panel
    toggle = panel.merge_toggle
    assert toggle.isCheckable() and not toggle.isChecked()
    assert panel._merge_scroll.isHidden()
    assert toggle.focusPolicy() == QtCore.Qt.StrongFocus
    panel.results_tabs.setCurrentIndex(1)
    toggle_at = _rect(toggle, window)
    toggle.setChecked(True)
    _settle(qapp)
    assert panel.results_tabs.currentIndex() == 0, "Merge… on another tab showed nothing"
    assert panel._merge_scroll.isVisible()
    viewport = panel._merge_scroll.viewport()
    box = QtCore.QRect(QtCore.QPoint(0, 0), viewport.size())
    mp = panel.merge_controls
    for widget in (mp.window_ms, mp.merge_all_btn, mp.merge_region_btn):
        assert box.contains(_rect(widget, viewport))
    assert _rect(toggle, window) == toggle_at
    assert _rect(panel._merge_scroll, window).bottom() < _rect(panel._table, window).top()


def test_merge_toggle_reads_on_from_any_tab(window, qapp):
    """Fusion's own checked shade differs by 7/255: once the user is on Summary nothing said
    Merge… was on. The checked style is unmistakable and moves nothing."""
    panel = window._analysis_panel
    toggle = panel.merge_toggle
    rect = _rect(toggle, window)

    def mean_rgb() -> tuple[float, float, float]:
        image = toggle.grab().toImage()
        pixels = [QtGui.QColor(image.pixel(x, y)) for y in range(image.height())
                  for x in range(image.width())]
        return tuple(sum(getattr(p, c)() for p in pixels) / len(pixels)
                     for c in ("red", "green", "blue"))

    off = mean_rgb()
    toggle.setChecked(True)
    panel.results_tabs.setCurrentIndex(1)
    _settle(qapp)
    on = mean_rgb()
    assert max(abs(a - b) for a, b in zip(on, off)) >= 20, (off, on)
    assert _rect(toggle, window) == rect


@pytest.mark.parametrize("how", ["merge open", "dock dragged short"])
def test_a_short_events_page_scrolls_instead_of_squeezing_the_table(
        real_font, qapp, isolated_settings, qtbot, tmp_path, how):
    """Squeezed below its header + a row + its scrollbar the table painted its header over
    the scrollbar and the caption into the header; now the Events page scrolls instead,
    and the dock still has no height floor beyond its header rows."""
    win = _open(qapp, qtbot)
    try:
        _detect(win, qtbot, tmp_path)
        panel = win._analysis_panel
        if how == "merge open":
            panel.merge_toggle.setChecked(True)
        else:
            win.resizeDocks([win._results_dock], [150], QtCore.Qt.Vertical)
        _settle(qapp)
        view = panel._table.view
        assert view.height() >= view.minimumHeight() > view.horizontalHeader().height()
        assert view.viewport().height() >= view.verticalHeader().defaultSectionSize()
        page = panel.results_tabs.widget(0)
        assert isinstance(page, QtWidgets.QScrollArea) and page.widget() is panel._events_page
        assert page.verticalScrollBar().maximum() > 0, "the page should scroll, not squeeze"
        if how == "merge open":
            assert panel._merge_scroll.height() == panel._merge_scroll.maximumHeight()
        assert win._results_dock.minimumSizeHint().height() <= 80
    finally:
        win._analysis_controller.abort()
        win.close()


def test_the_advisories_link_opens_summary_and_reveals_the_dock(window, qapp, qtbot, tmp_path):
    _detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    notice = panel._result_notice
    assert not notice.isHidden(), "every result carries its recorded limitations"
    n = len(panel._stats._warnings) if hasattr(panel._stats, "_warnings") else None
    assert "⚠" in notice.text() and "advisories" in notice.text()
    assert notice.accessibleName().startswith("Analysis advisories (")
    assert notice.toolTip().startswith("Analysis advisories (")
    if n is not None:
        assert f"({n})" in notice.accessibleName()
    window._results_dock.close()
    _settle(qapp)
    with qtbot.waitSignal(panel.resultsRevealRequested, timeout=1_000):
        notice.linkActivated.emit("summary")
    _settle(qapp)
    assert window._results_dock.isVisible()
    assert panel.results_tabs.currentIndex() == 1


def test_the_status_line_counts_events(window, qapp, qtbot, tmp_path):
    _detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    df = panel._table.dataframe
    n, acc = len(df), int(df["accepted"].sum())
    assert n > 0
    assert panel._results_summary.text() == \
        f"{n:,} event{'s' if n != 1 else ''} · {acc:,} accepted"
    view = panel._table.view
    view.setFocus()
    view.selectRow(0)
    _settle(qapp)
    QTest.keyClick(view, QtCore.Qt.Key_R)
    _settle(qapp)
    assert panel._results_summary.text().endswith(f"{acc - 1:,} accepted")
    window._session.close_document(window._session.documents()[0].doc_id)
    _settle(qapp)
    assert panel._results_summary.text() == AP._RESULTS_EMPTY_TEXT


def test_table_keys_and_undo_work_when_the_dock_floats(window, qapp, qtbot, tmp_path):
    _detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    dock = window._results_dock
    dock.setFloating(True)
    _settle(qapp)
    view = panel._table.view
    col = list(panel._table.dataframe.columns).index("accepted")
    dock.activateWindow()
    view.setFocus()
    view.selectRow(0)
    _settle(qapp)
    assert bool(panel._table.dataframe.iat[0, col]) is True
    QTest.keyClick(view, QtCore.Qt.Key_R)
    _settle(qapp)
    assert bool(panel._table.dataframe.iat[0, col]) is False
    assert window._act_undo.isEnabled()
    QTest.keyClick(view, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
    qtbot.waitUntil(lambda: bool(panel._table.dataframe.iat[0, col]) is True, timeout=2_000)


def test_the_title_bar_is_vertical_only_in_top_and_bottom_areas(window, qapp):
    dock = window._results_dock
    flag = QtWidgets.QDockWidget.DockWidgetVerticalTitleBar
    window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    _settle(qapp)
    assert not dock.features() & flag
    dock.setFloating(True)
    _settle(qapp)
    assert not dock.features() & flag
    dock.setFloating(False)
    window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
    _settle(qapp)
    assert dock.features() & flag


# ================================================================================================
# 7. a closed dock is announced, never re-opened
# ================================================================================================

def test_a_hidden_results_dock_is_announced_not_reopened(window, qapp, qtbot, tmp_path):
    window._results_dock.close()
    _settle(qapp)
    _detect(window, qtbot, tmp_path)
    assert window._results_dock.isHidden(), "a detection re-opened the dock under the pointer"
    assert MW._RESULTS_PANEL_PATH in window.statusBar().currentMessage()


def test_a_table_turning_stale_behind_a_closed_dock_is_said_once(window, qapp, qtbot, tmp_path):
    did = _detect(window, qtbot, tmp_path)
    window._results_dock.close()
    _settle(qapp)
    window.statusBar().clearMessage()
    _stale_by_gain(window, did)
    _settle(qapp)
    message = window.statusBar().currentMessage()
    assert "stale" in message and MW._RESULTS_PANEL_PATH in message
    window.statusBar().clearMessage()
    _stale_by_gain(window, did, 3.0)      # still stale: a re-judgement, not a transition
    _settle(qapp)
    assert window.statusBar().currentMessage() == ""
    assert window._results_dock.isHidden()


def test_a_stale_table_shown_after_an_empty_recording_is_announced(window, qapp, qtbot,
                                                                   tmp_path):
    """The hint follows the line's REAL visibility: a repoint to a recording with nothing to
    show hides the line without re-judging anything, and the next stale table shown behind
    the closed dock must still be said."""
    a = _detect(window, qtbot, tmp_path, "a.dat")
    c = _detect(window, qtbot, tmp_path, "c.dat")
    b = window._session.add_document(_doc(tmp_path / "b.dat"))      # never detected
    window._create_default_layers(window._session.document(b))
    window._results_dock.close()
    _settle(qapp)
    panel = window._analysis_panel
    window._on_selection(a, 0)
    _settle(qapp)
    window.statusBar().clearMessage()
    _stale_by_gain(window, a)
    _settle(qapp)
    assert "stale" in window.statusBar().currentMessage()
    window._on_selection(b, 0)
    _settle(qapp)
    assert panel._stale.isHidden()
    _stale_by_gain(window, c, 3.0)
    _settle(qapp)
    window.statusBar().clearMessage()
    window._on_selection(c, 0)
    _settle(qapp)
    assert not panel._stale.isHidden()
    assert MW._RESULTS_PANEL_PATH in window.statusBar().currentMessage()
    assert window._results_dock.isHidden()


# ================================================================================================
# 8. ⇧⌘R is a keyboard round trip
# ================================================================================================

def _focus_table(window, qapp, qtbot) -> QtWidgets.QTableView:
    window.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is window, timeout=2_000)
    view = window._analysis_panel._table.view
    view.setFocus()
    view.selectRow(0)
    _settle(qapp)
    assert QtWidgets.QApplication.focusWidget() is view
    return view


def test_the_results_toggle_hands_the_table_its_focus_back(window, qapp, qtbot, tmp_path):
    """Hidden with the focus in the table, Qt handed it to the next widget in the chain —
    the workflow strip's first chip, where Space opens a file — and showing the dock left
    it there, so 'r' silently did nothing. The focus waits on the trace and comes back."""
    _detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    view = _focus_table(window, qapp, qtbot)
    toggle = window._act_toggle_results
    toggle.trigger()
    _settle(qapp)
    assert window._results_dock.isHidden()
    assert QtWidgets.QApplication.focusWidget() is window._trace_view
    toggle.trigger()
    _settle(qapp)
    assert window._results_dock.isVisible()
    assert QtWidgets.QApplication.focusWidget() is view
    col = list(panel._table.dataframe.columns).index("accepted")
    assert bool(panel._table.dataframe.iat[0, col]) is True
    QTest.keyClick(view, QtCore.Qt.Key_R)
    _settle(qapp)
    assert bool(panel._table.dataframe.iat[0, col]) is False, "'r' no longer curates"


def test_showing_the_dock_another_way_never_takes_the_focus(window, qapp, qtbot, tmp_path):
    _detect(window, qtbot, tmp_path)
    _focus_table(window, qapp, qtbot)
    window._results_dock.close()               # its ✕: the focus is parked on the trace
    _settle(qapp)
    assert QtWidgets.QApplication.focusWidget() is window._trace_view
    window._reveal_results_panel()
    _settle(qapp)
    assert window._results_dock.isVisible()
    assert QtWidgets.QApplication.focusWidget() is window._trace_view
