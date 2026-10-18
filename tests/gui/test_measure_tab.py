"""The manual measurement tool as the FOURTH Analysis mode tab (pytest-qt, offscreen).

The owner's complaint, verbatim: *"looking at the manual detector, I dont like its location.
It should have its own tab in the Analysis section."* It used to sit at the bottom of the
Analysis tab's vertical splitter — below a long control scroll AND below the event table — so
the one tool you reach for by hand was the one you had to scroll to find.

It is now the fourth page of ``DetectionPanel.mode_tabs``:

    Preprocess | Detection | ML detector | **Manual measurement**

which puts all four ways of producing (or measuring) an event side by side as peers. The
EventTable deliberately stayed where it was, directly below the tab strip: it is the shared
result surface every one of the four feeds, and moving it was not asked for.

What these tests exist to stop coming back — each is a bug that was fixed once already:

* **the dock floor.** The tool's three-button row has a hard ~438 px layout minimum; a page's
  minimum width propagates through the tab widget into ``QDockWidget::minimumSize()``, which
  is the exact bound ``QDockAreaLayout`` will not drag a separator past. Unwrapped it pinned
  the Inspector at 458 px. It is installed SCROLL-WRAPPED, and the dock still shrinks to its
  ~137 px floor.
* **the wiring.** Every signal (``measureActiveChanged`` / ``markMissedRequested`` /
  ``measurementsChanged`` / ``navigateRequested``), every pushed setter and the
  ``AnalysisPanel.measure_panel`` property survive the move, exercised end to end rather
  than merely asserted to exist.
* **cursors while the tab is hidden.** DELIBERATE DECISION, pinned here: leaving the tab does
  NOT deactivate the tool. The region and the saved-measurement markers live on the shared
  TraceView, and the whole point of measuring is to watch the trace — an app that tore the
  cursors down because you clicked "Detection" would throw away the selection you were
  working on. Hiding a tab is layout, not state.
* **the menu action reveals what it turns on.** ``Analysis ▸ Measure event`` now raises the
  Inspector's Analysis tab AND this inner tab, on the way IN only.
* **"Mark selection as missed event" spans two tabs.** The detection is produced on the
  Detection / ML tab and the missed event is bracketed on this one; the gate and the append
  still work across that seam.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.scoring import ADDED_COLUMN
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.measure_panel import MeasurePanel
from nanoscope.gui.panels import EventTable
from nanoscope.gui.workflow_status import Stage

FS = 100_000.0
N = 100_000

TAB_TITLES = ["Preprocess", "Detection", "ML", "Measure"]
MEASURE_TAB_INDEX = 3

# Five clean 30 pA-deep squares the detector finds at a 1 ms dwell floor...
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]
# ...plus ONE short 0.3 ms blockade it MISSES, for the hand-marked path.
_MISSED = (60_000, 60_030)
_MIN_DWELL_S = 1e-3
# A clean wide span for plain measurement work.
_SPAN = (40_000, 44_000)

# The Inspector floored at 458 px before the scroll-wrapping; the measured value after it is
# ~137 (the tab bar's own minimum). Same threshold test_window_layout.py pins.
_SMALL_ENOUGH_TO_SHRINK = 220
_NARROW = 180
_WIDE = 560


# ================================================================================================
# builders
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    x[_MISSED[0]:_MISSED[1]] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, *, name: str = "pore.dat", arr: np.ndarray | None = None) -> TraceDocument:
    arr = _trace() if arr is None else arr
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Add + layer + select a document's current channel — the real selection path."""
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _run_detection(window: MainWindow, qtbot) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(_MIN_DWELL_S)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _bracket(window: MainWindow, doc_id: str, span=_SPAN) -> MeasurePanel:
    """Turn the cursors on and drop them on ``span``. Returns the panel."""
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[span[0]]), float(vals[span[1]]))
    return mp


def _mode_tabs(window: MainWindow):
    return window._analysis_panel.detection_panel.mode_tabs


def _measure_page(window: MainWindow):
    return window._analysis_panel.detection_panel._measure_page


def _regions(plot: pg.PlotItem) -> list[pg.LinearRegionItem]:
    """The MEASURE CURSORS on this pane — the one MOVABLE ``LinearRegionItem``.

    ``movable`` is load-bearing, not tidiness: ``overlays.render_marker`` draws the
    "selected saved measurement" marker as a NON-movable ``LinearRegionItem`` on the same
    pane. Counting both would let "the cursors survived the tab switch" pass on a run where
    the cursors were torn down and only a marker was left, and would break the
    ``== []`` assertions for a reason that has nothing to do with the tool being off."""
    return [it for it in plot.items
            if isinstance(it, pg.LinearRegionItem) and it.movable]


def _ancestors(widget, stop) -> list[QtWidgets.QWidget]:
    """The widget's parent chain up to (excluding) ``stop``."""
    out, node = [], widget.parentWidget()
    while node is not None and node is not stop:
        out.append(node)
        node = node.parentWidget()
    return out


def _set_dock_width(win: MainWindow, dock, width: int) -> int:
    win.resizeDocks([dock], [width], QtCore.Qt.Horizontal)
    QtWidgets.QApplication.processEvents()
    return dock.width()


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def shown_window(window):
    """A window big enough that the dock layout has real space to distribute.

    Also the ONLY way the hidden-tab tests mean anything: a widget that was never shown is
    never hidden either, so on an unshown window ``show_detection_tab()`` fires no
    ``hideEvent`` and a panel that tore its cursors down on hide would sail straight past
    the test written to catch it."""
    window.resize(1600, 900)
    window.show()
    QtWidgets.QApplication.processEvents()
    return window


def _raise_measure_tab(window: MainWindow) -> MeasurePanel:
    """Bring the measure tab genuinely to the front and assert it really is visible."""
    window._inspector.show()
    window._inspector_tabs.setCurrentWidget(window._analysis_panel)
    window._analysis_panel.detection_panel.show_measure_tab()
    QtWidgets.QApplication.processEvents()
    mp = window._analysis_panel.measure_panel
    assert mp.isVisible(), "the fixture failed to actually show the measure tab"
    return mp


def _raise_other_tab(window: MainWindow, which: str = "detection") -> None:
    """Switch away, and assert the measure panel is now REALLY hidden (a real hideEvent)."""
    getattr(window._analysis_panel.detection_panel, f"show_{which}_tab")()
    QtWidgets.QApplication.processEvents()
    assert not window._analysis_panel.measure_panel.isVisible(), (
        "the tab switch did not actually hide the panel; the test proves nothing")


# ================================================================================================
# 1. placement: the fourth tab, and the table left where it was
# ================================================================================================

def test_manual_measurement_is_the_fourth_mode_tab(window):
    """The request, literally: its own tab in the Analysis section, beside the other three."""
    tabs = _mode_tabs(window)
    assert tabs.count() == 4
    assert [tabs.tabText(i) for i in range(4)] == TAB_TITLES
    assert tabs.tabText(MEASURE_TAB_INDEX) == "Measure"
    assert "Manual measurement" in tabs.tabToolTip(MEASURE_TAB_INDEX)
    assert tabs.widget(MEASURE_TAB_INDEX) is _measure_page(window)


def test_the_measure_tab_holds_the_real_measure_widgets(window, tmp_path):
    """Not a placeholder: the live toggle, the readout, the mark-missed button and the
    saved-measurements table are all on that page."""
    _install(window, _doc(tmp_path))
    mp = window._analysis_panel.measure_panel
    page = _measure_page(window)

    assert page in _ancestors(mp, window)
    for widget in (mp._toggle, mp._snap_view_btn, mp._mark_missed_btn,
                   mp._table_view, mp._goto_button):
        assert page in _ancestors(widget, window), f"{widget} is not on the measure page"


def test_the_measure_panel_is_no_longer_a_splitter_child(window):
    """It is reached only through the mode tabs — never as a pane beside the controls.

    Its old home was pane 2 — under a long control scroll AND under the table, which is the
    "I don't like its location" the owner reported. (Updated for change A: the Analysis page
    has no splitter at all any more; the results half moved to the Results dock.)"""
    panel = window._analysis_panel
    mp = panel.measure_panel
    assert panel.findChild(QtWidgets.QSplitter) is None
    wrapper = next(w for w in _ancestors(mp, panel)
                   if isinstance(w, QtWidgets.QScrollArea) and w.widget() is mp)
    assert wrapper is not None
    assert _mode_tabs(window) in _ancestors(mp, panel)
    assert _measure_page(window) in _ancestors(mp, panel)


def test_the_event_table_stayed_below_the_tabs(window):
    """The shared result surface every detector feeds stays OUTSIDE the mode tabs — in the
    Results dock under the trace since change A, not in the Inspector."""
    panel = window._analysis_panel
    table = panel._table
    assert isinstance(table, EventTable)
    assert window._results_dock.isAncestorOf(table)
    assert not window._inspector.isAncestorOf(table)
    assert _measure_page(window) not in _ancestors(table, window)
    assert _mode_tabs(window) not in _ancestors(table, window)


def test_the_other_three_tabs_are_unaffected(window):
    """The measure page is a sibling, not a rearrangement: the detector's own controls are
    exactly where they were, and Preprocess is still the tab that opens."""
    dp = window._analysis_panel.detection_panel
    tabs = _mode_tabs(window)
    assert tabs.currentIndex() == 0
    assert tabs.widget(0) is dp._preprocess_page
    assert tabs.widget(1) is dp._detection_page
    assert tabs.widget(2) is dp._ml_page
    page = _measure_page(window)
    for widget in (dp.filter_cutoff_hz, dp.detect_button, dp.ml_detect_button,
                   dp.k_in, dp.polarity_override, dp.ml_range):
        assert page not in _ancestors(widget, window)


# ================================================================================================
# 2. the scroll wrapping, and the dock floor it exists to protect
# ================================================================================================

def test_the_measure_panel_is_installed_inside_a_widget_resizable_scroll_area(window):
    """The load-bearing detail. Its immediate parent chain up to the tab page must pass
    through a ``widgetResizable`` QScrollArea whose child IS the panel."""
    mp = window._analysis_panel.measure_panel
    chain = _ancestors(mp, _measure_page(window))
    areas = [w for w in chain if isinstance(w, QtWidgets.QScrollArea)]
    assert areas, "the measure panel must be scroll-wrapped inside its tab page"
    area = areas[0]
    assert area.widget() is mp
    assert area.widgetResizable()
    assert area.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
    assert area.minimumWidth() == 0


def test_the_measure_page_imposes_no_width_floor(window):
    """The panel still WANTS ~438 px (its button row cannot wrap) — the page must not
    inherit that as a minimum, or the dock is pinned again.

    The tab widget itself keeps a large minimum (the ML page is the widest, ~570) and that
    is fine: the ``controls_scroll`` above it is what clamps the whole stack. What matters
    is that the NEW page contributes nothing to that number."""
    mp = window._analysis_panel.measure_panel
    assert mp.minimumSizeHint().width() > _SMALL_ENOUGH_TO_SHRINK, (
        "if the panel itself got narrow this test stopped proving anything")
    assert _measure_page(window).minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK
    assert (_measure_page(window).minimumSizeHint().width()
            < _mode_tabs(window).minimumSizeHint().width()), (
        "the measure page must not be what sets the tab widget's floor")
    # Navigation stays outside the scroller; the per-page viewport now clamps
    # content width without making the mode selector disappear while scrolling.
    clamp = window._analysis_panel.detection_panel.page_scrolls[_measure_page(window)]
    assert clamp.widgetResizable()
    assert clamp.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK


def test_the_inspector_dock_still_shrinks_after_the_move(window):
    """The regression the whole wrapping exists for: the effective floor QDockAreaLayout
    will not let the separator cross. It was 458; it is the tab bar's own ~137."""
    assert window._inspector.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK
    assert window._inspector.minimumWidth() <= _SMALL_ENOUGH_TO_SHRINK
    assert window._analysis_panel.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK


def test_dragging_the_inspector_narrow_still_sticks_with_the_measure_tab_raised(shown_window):
    """And it holds with the widest page actually shown — the real user gesture."""
    shown_window._inspector_tabs.setCurrentWidget(shown_window._analysis_panel)
    shown_window._analysis_panel.detection_panel.show_measure_tab()
    QtWidgets.QApplication.processEvents()

    insp = shown_window._inspector
    _set_dock_width(shown_window, insp, _WIDE)
    viewer_before = shown_window.centralWidget().width()

    assert _set_dock_width(shown_window, insp, _NARROW) == pytest.approx(_NARROW, abs=12)
    assert shown_window.centralWidget().width() > viewer_before
    # It scrolls rather than resisting.
    assert _set_dock_width(shown_window, insp, _WIDE) == pytest.approx(_WIDE, abs=12)


# ================================================================================================
# 3. every signal, setter and accessor survived the move
# ================================================================================================

def test_the_measure_panel_property_still_returns_the_owned_widget(window):
    mp = window._analysis_panel.measure_panel
    assert isinstance(mp, MeasurePanel)
    assert mp is window._analysis_panel._measure


@pytest.mark.parametrize("name", ["measureActiveChanged", "markMissedRequested",
                                  "measurementsChanged", "navigateRequested"])
def test_the_four_signals_still_exist_on_the_moved_panel(window, name):
    assert hasattr(window._analysis_panel.measure_panel, name)


@pytest.mark.parametrize("name", ["set_selection", "set_selected_event",
                                  "set_detection_present", "set_measurements_df",
                                  "measurements_df", "clear", "can_measure",
                                  "is_active", "set_active"])
def test_the_pushed_api_still_exists_on_the_moved_panel(window, name):
    assert hasattr(window._analysis_panel.measure_panel, name)


def test_measure_active_changed_still_reaches_the_menu_check(window, tmp_path):
    _install(window, _doc(tmp_path))
    mp = window._analysis_panel.measure_panel

    mp.set_active(True)                       # panel drives the menu
    assert window._act_measure.isChecked()
    mp.set_active(False)
    assert not window._act_measure.isChecked()


def test_measurements_changed_still_marks_the_session_dirty(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path))
    window._dirty = False                      # opening a recording is not unsaved work

    mp = _bracket(window, doc_id)
    assert mp.add_current() is True
    assert window._dirty, "measurementsChanged must still reach the unsaved-work tracker"


def test_set_selection_still_repoints_the_moved_panel(window, tmp_path):
    """The owner pushes the resolved CURRENT channel in; a target change drops the region
    that belonged to the previous channel's pane."""
    id_a, _ch = _install(window, _doc(tmp_path, name="A.dat"))
    mp = _bracket(window, id_a)
    assert window._trace_view.measure_bounds() is not None

    _install(window, _doc(tmp_path, name="B.dat"))
    assert not mp.is_active()
    assert window._trace_view.measure_bounds() is None
    assert not window._act_measure.isChecked()


def test_set_measurements_df_and_measurements_df_still_round_trip(window, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path))
    mp = _bracket(window, doc_id)
    assert mp.add_current() is True
    saved = mp.measurements_df.copy()
    assert len(saved) == 1

    mp.set_measurements_df(None)
    assert len(mp.measurements_df) == 0
    mp.set_measurements_df(saved)
    assert len(mp.measurements_df) == 1
    assert isinstance(mp.measurements_df, pd.DataFrame)


def test_set_selected_event_still_lands_from_a_table_row(window, qtbot, tmp_path):
    """Clicking an event row hands the span to the measure tool ("Snap to selected event"),
    across the tab seam."""
    doc_id, channel = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel
    mp = panel.measure_panel
    assert mp._selected_event is None

    first_id = int(panel.current_events_df()["event_id"].iloc[0])
    panel._on_event_selected(first_id)
    assert mp._selected_event is not None
    assert mp._selected_event[0] == doc_id and mp._selected_event[1] == channel


def test_set_detection_present_still_gates_the_mark_missed_button(window, qtbot, tmp_path):
    doc_id, _ch = _install(window, _doc(tmp_path))
    mp = _bracket(window, doc_id, _MISSED)
    assert not mp._mark_missed_btn.isEnabled(), "no detection yet -> nothing to add to"

    _run_detection(window, qtbot)
    _bracket(window, doc_id, _MISSED)
    assert mp._mark_missed_btn.isEnabled()


def test_navigate_requested_still_routes_through_main_window(window, tmp_path):
    """A saved row belonging to ANOTHER open recording announces its target; MainWindow —
    still the sole wirer — switches the app selection and the jump finishes."""
    id_a, ch_a = _install(window, _doc(tmp_path, name="A.dat"))
    mp = _bracket(window, id_a)
    assert mp.add_current() is True

    id_b, _ch_b = _install(window, _doc(tmp_path, name="B.dat"))
    assert mp._target[0] == id_b, "the panel followed the new selection"

    seen: list[tuple[str, int]] = []
    mp.navigateRequested.connect(lambda d, c: seen.append((d, c)))
    mp.set_measurements_df(mp.measurements_df)     # rows are A's; the view shows them
    mp._table_view.selectRow(0)
    assert mp.go_to_selected() is True

    assert seen == [(id_a, ch_a)]
    assert mp._target == (id_a, ch_a), "MainWindow routed the selection back to A"


# ================================================================================================
# 4. cursors while the tab is HIDDEN — the deliberate decision
#
# These all run on a SHOWN window on purpose. A widget that was never shown is never hidden
# either, so on an unshown window a tab switch fires no hideEvent — and a panel that
# deactivated itself on hide would pass every one of these tests without the shown fixture.
# ================================================================================================

def test_leaving_the_measure_tab_does_not_deactivate_the_cursors(shown_window, tmp_path):
    """DELIBERATE: hiding the tab is layout, not state.

    The region lives on the shared TraceView and the user is measuring precisely so she can
    look at the trace — tearing the cursors down because she clicked "Detection" would
    silently discard the selection she was working on."""
    doc_id, channel = _install(shown_window, _doc(tmp_path))
    mp = _raise_measure_tab(shown_window)
    _bracket(shown_window, doc_id)
    pane = shown_window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert len(_regions(pane.plot)) == 1
    bounds = shown_window._trace_view.measure_bounds()
    assert bounds == _SPAN

    _raise_other_tab(shown_window, "detection")

    assert mp.is_active(), "leaving the tab must not disarm the tool"
    assert shown_window._act_measure.isChecked(), "...nor un-check the menu"
    assert len(_regions(pane.plot)) == 1, "the on-trace region must survive the tab switch"
    assert shown_window._trace_view.measure_bounds() == bounds
    assert mp.current_measurement() is not None


def test_the_readout_keeps_updating_while_the_tab_is_hidden(shown_window, tmp_path):
    """Not merely alive — still LIVE. Dragging the region with the ML tab raised recomputes
    the measurement, so coming back to the tab shows the truth, not a stale number."""
    doc_id, _ch = _install(shown_window, _doc(tmp_path))
    mp = _raise_measure_tab(shown_window)
    _bracket(shown_window, doc_id)
    n_first = mp.current_measurement()["n_samples"]

    _raise_other_tab(shown_window, "ml")
    vals = shown_window._session.x_axis(doc_id).values
    shown_window._trace_view.set_measure_region(
        float(vals[_SPAN[0]]), float(vals[_SPAN[0] + 1_000]))

    stats = mp.current_measurement()
    assert stats["n_samples"] == 1_000
    assert stats["n_samples"] != n_first

    _raise_measure_tab(shown_window)
    assert mp.current_measurement()["n_samples"] == 1_000


def test_saved_measurement_markers_survive_a_tab_switch(shown_window, tmp_path):
    """The markers are TraceView overlays; a hidden panel must not tear them down."""
    doc_id, _ch = _install(shown_window, _doc(tmp_path))
    mp = _raise_measure_tab(shown_window)
    _bracket(shown_window, doc_id)
    assert mp.add_current() is True
    span_id = mp._span_overlay_id
    assert span_id is not None
    assert shown_window._trace_view._overlays[span_id][0].n_spans == 1

    _raise_other_tab(shown_window, "preprocess")
    assert mp._span_overlay_id == span_id
    assert shown_window._trace_view._overlays[span_id][0].n_spans == 1

    # ...and a row saved from the OTHER tab still lands on the trace.
    vals = shown_window._session.x_axis(doc_id).values
    shown_window._trace_view.set_measure_region(float(vals[10_000]), float(vals[12_000]))
    assert mp.add_current() is True
    assert shown_window._trace_view._overlays[mp._span_overlay_id][0].n_spans == 2


def test_row_to_trace_navigation_works_from_another_tab(shown_window, tmp_path):
    """Double-clicking a saved row restores its exact span even with the tab hidden — the
    zoom + cursors are TraceView work, not tab work."""
    doc_id, _ch = _install(shown_window, _doc(tmp_path))
    mp = _raise_measure_tab(shown_window)
    _bracket(shown_window, doc_id)
    assert mp.add_current() is True

    vals = shown_window._session.x_axis(doc_id).values
    shown_window._trace_view.set_measure_region(float(vals[5_000]), float(vals[7_000]))
    shown_window._trace_view.reset_zoom()
    assert shown_window._trace_view.measure_bounds() == (5_000, 7_000)

    _raise_other_tab(shown_window, "detection")
    mp._table_view.doubleClicked.emit(mp._proxy.index(0, 0))

    assert shown_window._trace_view.measure_bounds() == _SPAN
    assert mp._marker_overlay_id is not None


def test_turning_the_tool_off_still_clears_the_region_from_any_tab(shown_window, tmp_path):
    """The other half of the decision: the tab does not own the cursors, but the TOGGLE
    still does — switching tabs is not a way to leak a live region."""
    doc_id, channel = _install(shown_window, _doc(tmp_path))
    mp = _raise_measure_tab(shown_window)
    _bracket(shown_window, doc_id)
    pane = shown_window._trace_view.pane_for_layer(f"{doc_id}#{channel}")

    _raise_other_tab(shown_window, "ml")
    mp.set_active(False)
    assert _regions(pane.plot) == []
    assert not shown_window._act_measure.isChecked()


# ================================================================================================
# 5. the menu action reveals the tab it turns on
# ================================================================================================

def test_measure_menu_action_raises_the_manual_measurement_tab(window, tmp_path):
    """``Analysis ▸ Measure event`` arms a tool whose readout, Add button and table are on
    this tab; it must bring the user there instead of arming something invisible."""
    doc_id, channel = _install(window, _doc(tmp_path))
    # Park somewhere else entirely: a different Inspector tab AND a different inner tab.
    window._inspector_tabs.setCurrentWidget(window._cal_scroll)
    window._analysis_panel.detection_panel.show_preprocess_tab()

    assert window._act_measure.isEnabled()
    window._act_measure.setChecked(True)

    assert window._inspector_tabs.currentWidget() is window._analysis_panel
    assert _mode_tabs(window).currentIndex() == MEASURE_TAB_INDEX
    assert _mode_tabs(window).currentWidget() is _measure_page(window)
    # ...and it really did turn the tool on, not just move furniture.
    assert window._analysis_panel.measure_panel.is_active()
    pane = window._trace_view.pane_for_layer(f"{doc_id}#{channel}")
    assert len(_regions(pane.plot)) == 1


def test_unchecking_the_menu_action_does_not_move_the_tabs(window, tmp_path):
    """Asymmetric on purpose: revealing on the way IN is help, yanking the Inspector
    somewhere on the way OUT is the app moving furniture for no reason."""
    _install(window, _doc(tmp_path))
    window._act_measure.setChecked(True)
    assert _mode_tabs(window).currentIndex() == MEASURE_TAB_INDEX

    window._analysis_panel.detection_panel.show_ml_tab()
    window._inspector_tabs.setCurrentWidget(window._cal_scroll)
    window._act_measure.setChecked(False)

    assert not window._analysis_panel.measure_panel.is_active()
    assert _mode_tabs(window).currentIndex() == 2
    assert window._inspector_tabs.currentWidget() is window._cal_scroll


def test_the_panels_own_toggle_does_not_yank_the_tab(window, tmp_path):
    """The panel's own button is already ON this tab; mirroring it back into a reveal would
    fight whichever tab the user just chose."""
    _install(window, _doc(tmp_path))
    window._analysis_panel.detection_panel.show_detection_tab()
    window._inspector_tabs.setCurrentWidget(window._cal_scroll)

    window._analysis_panel.measure_panel.set_active(True)

    assert window._act_measure.isChecked()
    assert _mode_tabs(window).currentIndex() == 1
    assert window._inspector_tabs.currentWidget() is window._cal_scroll


# ================================================================================================
# 6. "Mark selection as missed event" across the two tabs
# ================================================================================================

def test_mark_missed_reaches_the_detection_produced_on_another_tab(window, qtbot, tmp_path):
    """The full seam: detect on the Detection tab, bracket on the Manual measurement tab,
    and the hand-marked span is appended to that detection as accepted + added."""
    doc_id, channel = _install(window, _doc(tmp_path))
    dp = window._analysis_panel.detection_panel

    dp.show_detection_tab()
    _run_detection(window, qtbot)
    panel = window._analysis_panel
    before = panel.current_events_df()
    assert len(before) == len(_EVENTS), "the short blockade must be MISSED at this dwell floor"

    dp.show_measure_tab()
    mp = _bracket(window, doc_id, _MISSED)
    assert mp._mark_missed_btn.isEnabled()
    with qtbot.waitSignal(window._session.resultsChanged, timeout=5_000):
        mp._mark_missed_btn.click()
    qtbot.wait(20)

    after = panel.current_events_df()
    assert len(after) == len(before) + 1
    added = after[after[ADDED_COLUMN].to_numpy(dtype=bool)]
    assert len(added) == 1
    row = added.iloc[0]
    assert bool(row["accepted"]) is True
    assert int(row["start_idx"]) == _MISSED[0] and int(row["end_idx"]) == _MISSED[1]
    # It reached the shared event table below the tabs, which never moved.
    assert len(panel._table.dataframe) == len(after)


def test_mark_missed_still_works_when_the_measure_tab_is_hidden(window, qtbot, tmp_path):
    """The gate is "a detection + live cursors", not "the tab is showing" — the button is
    reachable from the Event Inspector's routed path too."""
    doc_id, _ch = _install(window, _doc(tmp_path))
    _run_detection(window, qtbot)
    panel = window._analysis_panel
    n_before = len(panel.current_events_df())

    _bracket(window, doc_id, _MISSED)
    panel.detection_panel.show_ml_tab()          # tab hidden, tool still armed
    with qtbot.waitSignal(window._session.resultsChanged, timeout=5_000):
        assert panel.add_current_selection_as_event() is not None
    qtbot.wait(20)

    assert len(panel.current_events_df()) == n_before + 1


# ================================================================================================
# 7. the other tabs' contract is untouched, and the strip still routes
# ================================================================================================

def test_detection_still_runs_after_visiting_the_measure_tab(window, qtbot, tmp_path):
    """The detector's params are read off live widgets, not off the raised tab: a detour
    through Manual measurement must not change what Detect does."""
    doc_id, channel = _install(window, _doc(tmp_path))
    dp = window._analysis_panel.detection_panel
    dp.min_dwell_s.setValue(_MIN_DWELL_S)
    reference = dict(dp.params())

    dp.show_measure_tab()
    _bracket(window, doc_id)
    assert dp.params() == reference, "the measure tab contributes nothing to params()"

    _run_detection(window, qtbot)
    assert len(window._analysis_panel.current_events_df()) == len(_EVENTS)


def test_strip_filter_and_detect_still_route_from_the_measure_tab(window, tmp_path):
    """The workflow strip's two inner-tab reveals still land on index 0 / 1 — adding a
    fourth page did not shift them, and being parked on it is no obstacle."""
    doc_id, _ch = _install(window, _doc(tmp_path))
    dp = window._analysis_panel.detection_panel

    dp.show_measure_tab()
    window._inspector_tabs.setCurrentWidget(window._cal_scroll)
    window._workflow_strip.stageActivated.emit(Stage.FILTER.value)
    assert window._inspector_tabs.currentWidget() is window._analysis_panel
    assert _mode_tabs(window).currentIndex() == 0

    dp.show_measure_tab()
    window._inspector_tabs.setCurrentWidget(window._cal_scroll)
    window._workflow_strip.stageActivated.emit(Stage.DETECT.value)
    assert window._inspector_tabs.currentWidget() is window._analysis_panel
    assert _mode_tabs(window).currentIndex() == 1


def test_panels_module_still_imports_no_sibling_gui_panel():
    """The tab page is an EMPTY container on purpose. If panels.py ever built the
    measurement tool itself it would have to import ``measure_panel`` — a panel-to-panel
    import, and the end of "MainWindow is the sole wirer".

    Checked by parsing the imports rather than grepping the text, so a docstring that
    merely NAMES the tool (this module's does) is not mistaken for reaching it, and every
    spelling of the import — including ``from . import measure_panel`` — is caught."""
    import ast
    import pathlib

    import nanoscope.gui.panels as panels_mod

    src = pathlib.Path(panels_mod.__file__).read_text()
    reached: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nanoscope.gui."):
                    reached.add(alias.name.split(".")[2])
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level == 1 and name:                       # from .x import y
                reached.add(name.split(".")[0])
            elif node.level == 1:                              # from . import x
                reached |= {a.name for a in node.names}
            elif node.level == 2 and name.startswith("gui."):  # from ..gui.x import y
                reached.add(name.split(".")[1])
            elif node.level == 0 and name.startswith("nanoscope.gui."):
                reached.add(name.split(".")[2])

    # Pure layout helpers are shared; numerical/interactive sibling panels still
    # belong to their existing owners and must not be constructed from panels.py.
    # numeric_text (2026-09-13) is the same kind of shared, panel-free leaf as
    # inspector_widgets: pure ASCII number-parsing, no widget, no gui/ import of its
    # own.
    assert reached <= {'inspector_widgets', 'numeric_text'}, (
        f"panels.py imported a sibling panel: {reached}")
    # ...and the checker above really would catch the import it is standing guard over.
    guarded = ast.parse("from .measure_panel import MeasurePanel")
    assert any(isinstance(n, ast.ImportFrom) and n.level == 1
               and (n.module or "").split(".")[0] == "measure_panel"
               for n in ast.walk(guarded))
