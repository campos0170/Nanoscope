"""How the main window spends its horizontal space — the three-column contract.

The window is [File Manager dock] · [signal viewer] · [Inspector dock], with a thin workflow
strip along the top of the middle column. Two things had gone wrong and are pinned here so
they cannot come back:

* **The Inspector could only grow.** Nothing set an explicit width; the floor was an implicit
  ``minimumSizeHint`` chain — an un-shrinkable row of push buttons deep inside the Analysis
  tab (``MeasurePanel``) propagated ~438 px up through the vertical splitter and the tab
  widget into ``QDockWidget::minimumSize()``, which is the exact bound ``QDockAreaLayout``
  refuses to drag a separator past. Dragging right did nothing and the signal viewer could
  never be widened. Every Inspector page is now a ``widgetResizable`` ``QScrollArea``, so
  content scrolls horizontally instead of pushing back.
* **A fixed 172 px left column** held a six-row vertical workflow rail. It is a horizontal
  strip above the viewer now, and the left column belongs to the file manager.

These tests assert the resulting BEHAVIOUR — what the user can drag, what stays wired — plus
the one structural invariant (no Inspector page may impose a width floor) that would let the
regression back in. Nothing here touches the frozen 20-column EventSet schema.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.layer import ViewMode
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import (
    _FILE_DOCK_DEFAULT_WIDTH,
    _INSPECTOR_DEFAULT_WIDTH,
    _INSPECTOR_MAX_DEFAULT_WIDTH,
    MainWindow,
)

FS = 50_000.0
N = 20_000

# The Inspector used to floor at 458 px. Anything at or under this leaves the user real room
# to widen the viewer; the measured value after the fix is ~137 (the tab bar's own minimum).
_SMALL_ENOUGH_TO_SHRINK = 220
# A width the user might drag to when she wants the viewer as wide as possible.
_NARROW = 180
_WIDE = 560


def _doc(tmp_path, *, name="rec.dat", with_voltage=False) -> TraceDocument:
    chans = [ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)]
    if with_voltage:
        chans.append(ChannelInfo(index=1, name="voltage", raw_units="mV",
                                 kind=ChannelKind.VOLTAGE))
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=tuple(chans))
    return TraceDocument(meta=meta, channels=tuple(np.zeros(N, np.float32) for _ in chans))


def _install(win: MainWindow, doc: TraceDocument) -> str:
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    return doc_id


@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace for the whole module.

    Every test here is about geometry, and ``MainWindow._restore_settings`` reads the REAL
    user's saved ``windowState`` — a developer who happens to have closed or floated a dock in
    the actual app would otherwise see these tests measure that layout instead of the one
    under test. Here "first launch" really means first launch."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("WindowLayoutDefaults")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


@pytest.fixture
def win(isolated_settings, qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def shown_win(win):
    """A window big enough that the dock layout has real space to distribute."""
    win.resize(1600, 900)
    win.show()
    QtWidgets.QApplication.processEvents()
    return win


def _set_dock_width(win: MainWindow, dock, width: int) -> int:
    """Drag ``dock``'s separator to ``width`` the way QMainWindow does, and report the result."""
    win.resizeDocks([dock], [width], QtCore.Qt.Horizontal)
    QtWidgets.QApplication.processEvents()
    return dock.width()


# ================================================================================================
# the Inspector shrinks as well as grows
# ================================================================================================

def test_inspector_dock_has_no_meaningful_width_floor(win):
    """The EFFECTIVE minimum — the number QDockAreaLayout will not let the separator cross.
    It was 458; the tab bar's own ~133 is all that is left of it."""
    insp = win._inspector
    assert insp.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK
    assert insp.minimumWidth() <= _SMALL_ENOUGH_TO_SHRINK


def test_no_inspector_page_imposes_a_width_floor(win):
    """The structural invariant, and the one that catches a regression before a user does: a
    page added to the Inspector unwrapped would floor the whole dock at its content width."""
    tabs = win._inspector_tabs
    assert tabs.count() == 3
    for i in range(tabs.count()):
        page = tabs.widget(i)
        assert page.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK, (
            f"Inspector tab {tabs.tabText(i)!r} ({type(page).__name__}) imposes a width floor; "
            f"wrap it in a widgetResizable QScrollArea")


def test_dragging_the_inspector_narrow_sticks_and_widens_the_viewer(shown_win):
    """The actual complaint: the separator would not move right."""
    insp = shown_win._inspector
    _set_dock_width(shown_win, insp, _WIDE)
    viewer_before = shown_win.centralWidget().width()

    assert _set_dock_width(shown_win, insp, _NARROW) == pytest.approx(_NARROW, abs=12)
    assert shown_win.centralWidget().width() > viewer_before


def test_the_inspector_can_be_dragged_back_out_again(shown_win):
    insp = shown_win._inspector
    _set_dock_width(shown_win, insp, _NARROW)
    assert _set_dock_width(shown_win, insp, _WIDE) == pytest.approx(_WIDE, abs=12)


def test_a_narrow_inspector_scrolls_its_content_instead_of_resisting(shown_win):
    """Where the width went: the panels keep their natural size and the scroll areas hand the
    user a horizontal scrollbar, rather than the panels squeezing the dock back open."""
    shown_win._inspector_tabs.setCurrentWidget(shown_win._analysis_panel)
    _set_dock_width(shown_win, shown_win._inspector, _NARROW)

    areas = shown_win._analysis_panel.findChildren(QtWidgets.QScrollArea)
    assert areas, "the Analysis tab no longer wraps its sub-panels in scroll areas"
    for area in areas:
        assert area.widgetResizable()
        assert area.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded
    # …and at this width at least one of them really is scrolling.
    assert any(a.horizontalScrollBar().maximum() > 0 for a in areas)


def test_the_analysis_title_elides_rather_than_widening_the_dock(win, tmp_path):
    """A long "recording · channel · kind · units" title used to be a floor of its own."""
    doc_id = _install(win, _doc(tmp_path, name="a_very_long_recording_name_indeed.abf"))
    win._on_selection(doc_id, 0)
    title = win._analysis_panel._title

    assert "a_very_long_recording_name_indeed.abf" in title.text()   # full string preserved
    assert title.minimumSizeHint().width() == 0                      # but it demands nothing


# ================================================================================================
# sensible defaults on a first launch
# ================================================================================================

def test_first_launch_opens_both_docks_at_their_designed_widths(isolated_settings, qapp, qtbot):
    """Shrinkable must not mean "opens collapsed": with no saved state the window states its
    own layout."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1600, 900)
    w.show()
    QtWidgets.QApplication.processEvents()

    assert w._file_dock.width() == pytest.approx(_FILE_DOCK_DEFAULT_WIDTH, abs=12)
    # The Inspector's designed width is DERIVED (main_window._inspector_default_width): wide
    # enough to show its widest page, clamped between the 460 floor and the 620 cap. It was a
    # flat 460, which opened the Analysis controls in a 420 px viewport against an 836 px
    # control column — everything past x=420 (protocol Save as…/Rename/Delete, the ML preset
    # buttons, Null control…, Recovery map…) was off the right edge of a 1512 px screen with
    # no menu item behind it. Assert the derivation, not the old constant.
    assert _INSPECTOR_DEFAULT_WIDTH <= w._inspector_default_width() <= _INSPECTOR_MAX_DEFAULT_WIDTH
    assert w._inspector.width() == pytest.approx(w._inspector_default_width(), abs=12)
    # ...and the viewer keeps the width the cap exists to defend. This replaces "the viewer
    # gets more than both docks combined", which cannot survive an Inspector wide enough to
    # show its own controls (320 + 620 = 940 against a 1512 px screen) and was only ever a
    # proxy for this: the signal viewer is never squeezed to a strip.
    assert w.centralWidget().width() >= 560
    assert w.centralWidget().width() > w._file_dock.width()


def test_a_saved_window_state_beats_the_defaults(isolated_settings, qapp, qtbot):
    """Once the user has dragged the separators, her widths win on every later launch."""
    first = MainWindow()
    qtbot.addWidget(first)
    first.resize(1600, 900)
    first.show()
    QtWidgets.QApplication.processEvents()
    _set_dock_width(first, first._inspector, _NARROW)
    saved = first._inspector.width()
    first._save_settings()

    second = MainWindow()
    qtbot.addWidget(second)
    second.resize(1600, 900)
    second.show()
    QtWidgets.QApplication.processEvents()
    assert second._inspector.width() == pytest.approx(saved, abs=12)


# ================================================================================================
# the left column is the File Manager
# ================================================================================================

def test_the_file_manager_owns_the_left_column(win):
    dock = win._file_dock
    assert win.dockWidgetArea(dock) == QtCore.Qt.LeftDockWidgetArea
    assert dock.windowTitle() == "File Manager"
    # objectName is what saveState/restoreState key on — it must survive the rename
    assert dock.objectName() == "FileDock"
    # nothing else competes for the left column any more
    assert win.centralWidget().findChildren(type(win._workflow_strip))


def test_the_file_manager_has_no_meaningful_width_floor(win):
    """The left column must shrink for the same reason the Inspector must.

    Regression: the dock's toolbar is a row of push buttons, which never wraps — its ~293 px
    width propagated into ``QDockWidget::minimumSize()`` and floored the LEFT column at 293,
    so the width the rework handed the file manager could not be handed back to the viewer.
    Same remedy as the Inspector: the row lives in a widgetResizable QScrollArea."""
    dock = win._file_dock
    assert dock.minimumSizeHint().width() <= _SMALL_ENOUGH_TO_SHRINK
    assert dock.minimumWidth() <= _SMALL_ENOUGH_TO_SHRINK


def test_the_file_manager_toolbar_scrolls_instead_of_flooring_the_dock(shown_win):
    """Where that width went — and the row stays one row tall while there is room for it."""
    dock = shown_win._file_dock
    area = dock._toolbar_scroll
    assert area.widgetResizable()
    assert area.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarAsNeeded

    _set_dock_width(shown_win, dock, 420)
    roomy_row, roomy_tree = area.height(), dock._tree.height()
    assert area.horizontalScrollBar().maximum() == 0        # nothing to scroll when roomy

    assert _set_dock_width(shown_win, dock, 120) == pytest.approx(120, abs=12)
    assert area.horizontalScrollBar().maximum() > 0         # squeezed: it scrolls…
    # …at the cost of the scrollbar's own height and nothing more
    lost = roomy_tree - dock._tree.height()
    assert 0 <= lost <= area.horizontalScrollBar().sizeHint().height() + 2
    # every control is still there to be scrolled to
    for btn in (dock._btn_overlay, dock._btn_stacked, dock._btn_segment, dock._btn_close):
        assert btn.isVisible()

    # and the affordance costs nothing once the dock is roomy again
    _set_dock_width(shown_win, dock, 420)
    assert area.height() == roomy_row


def test_both_docks_at_their_floors_leave_the_window_narrow(shown_win):
    """The two floors together are what bounds how narrow the whole window may get; neither
    dock may quietly re-impose a wide one."""
    _set_dock_width(shown_win, shown_win._file_dock, 40)
    _set_dock_width(shown_win, shown_win._inspector, 40)
    assert shown_win._file_dock.width() + shown_win._inspector.width() < 300
    assert shown_win.minimumWidth() <= 640


def test_the_file_manager_is_resizable_and_collapsible(shown_win):
    dock = shown_win._file_dock
    features = dock.features()
    assert features & QtWidgets.QDockWidget.DockWidgetClosable
    assert features & QtWidgets.QDockWidget.DockWidgetMovable

    wide = _set_dock_width(shown_win, dock, 420)
    assert wide == pytest.approx(420, abs=12)
    # narrower than the 293 px the un-wrapped button row used to floor it at, and back again
    assert _set_dock_width(shown_win, dock, _NARROW) == pytest.approx(_NARROW, abs=12)
    assert _set_dock_width(shown_win, dock, 420) == pytest.approx(420, abs=12)

    dock.close()                              # collapse it entirely…
    assert dock.isHidden()
    dock.toggleViewAction().trigger()         # …and get it back from View ▸ Panels
    assert not dock.isHidden()


def test_the_file_manager_keeps_its_selection_wiring(win, tmp_path):
    doc_id = _install(win, _doc(tmp_path, with_voltage=True))
    win._file_dock.selectionChanged.emit(doc_id, 1)
    assert win._current_selection == (doc_id, 1)


def test_the_file_manager_keeps_its_layer_and_close_controls(win, tmp_path):
    dock = win._file_dock
    doc_id = _install(win, _doc(tmp_path))

    dock._btn_stacked.click()
    assert win._session.view_mode is ViewMode.STACKED
    dock._btn_overlay.click()
    assert win._session.view_mode is ViewMode.OVERLAY

    # Segment stays gated by MainWindow (no Measure region drawn here)…
    assert not dock._btn_segment.isEnabled()
    dock.set_segment_enabled(True)
    assert dock._btn_segment.isEnabled()

    # …and Close still closes the selected recording.
    dock.focus_document(doc_id)
    dock._btn_close.click()
    assert doc_id not in win._session.documents()


# ================================================================================================
# the workflow strip sits above the viewer, not beside it
# ================================================================================================

def test_the_strip_is_the_top_row_of_the_centre_column(win):
    layout = win.centralWidget().layout()
    assert isinstance(layout, QtWidgets.QVBoxLayout)      # a column of rows, not side-by-side
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert order[0] is win._workflow_strip
    assert order.index(win._workflow_strip) < order.index(win._trace_view)


def test_nothing_in_the_centre_column_is_a_fixed_width_column(win):
    """The rail was ``setFixedWidth(172)``. Every centre-column widget must now be free to use
    whatever width the window has."""
    unconstrained = QtWidgets.QWidget().maximumWidth()      # QWIDGETSIZE_MAX
    layout = win.centralWidget().layout()
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        if widget is None:
            continue
        assert widget.maximumWidth() == unconstrained, (
            f"{type(widget).__name__} caps the centre column's width")


@pytest.mark.parametrize("window_width", [900, 1400])
def test_the_strip_fits_the_centre_column_at_any_window_width(shown_win, window_width):
    """In the REAL window (not a detached strip): the centre column is whatever is left after
    both docks, and the strip has to live in it — never clipped below its own minimum, and
    never wide enough to need a scroll area of its own."""
    shown_win.resize(window_width, 800)
    QtWidgets.QApplication.processEvents()
    strip = shown_win._workflow_strip

    assert shown_win.width() == window_width          # the strip did not veto the width
    assert strip.width() >= strip.minimumSizeHint().width()
    assert strip.width() <= shown_win.centralWidget().width()
    # it is chrome: one short row, and it never grows a scroll area
    assert 24 <= strip.height() <= 32
    assert not strip.findChildren(QtWidgets.QAbstractScrollArea)
