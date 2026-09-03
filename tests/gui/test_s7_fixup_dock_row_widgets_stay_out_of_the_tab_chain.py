"""X9 — Tab out of a text field may not walk into the File Manager and repoint the app.

MEASURED DEFECT (sweep-7 skeptic, owner's −1000 mV recording, IN15 selected). Typing
``60, 120`` into the Preprocess ▸ notch "lines" box and pressing Tab — the natural way to
commit a text field — landed keyboard focus in a ``QDoubleSpinBox`` that the File Manager
installs on a channel row with ``setItemWidget`` (the per-layer "Shift this layer along the
time axis" control). ``QAbstractItemView::eventFilter`` turns a *focus-in* on a persistent
index widget into ``setCurrentIndex(...)``: it cannot tell a click from a Tab. So the tree
selected that row, the dock emitted ``selectionChanged``, and the app selection moved
IN15 → IN14 — the VOLTAGE channel — taking ``MainWindow._current_selection`` and the
Calibration panel with it. The Calibration panel's Apply is what writes a ``.npcal.json``
sidecar beside the user's recording, so an unrequested keystroke pointed the sidecar writer
at the wrong channel; the Analysis panel meanwhile stayed on IN15 (it resolves through
``Session.resolve_current_channel``), leaving the window split across two channels. Return
never did this — it stays in the box — so only the traversal key was affected.

THE FIX is at the one place the dock hands a widget to ``setItemWidget``
(``FileDock._install_row_widget``): an embedded row widget takes CLICK focus only, so Tab
skips it. Clicking one still focuses and edits it, and a click on a row's own control
legitimately selects that row.

These tests drive the real MainWindow, the real dock and real key events; the invariant test
reads the focus policy off the widgets actually installed in the tree. Both fail against the
pre-sweep7 snapshot (the defect is older than the fix batch). Numbers are untouched: nothing
here reaches detection, export or figure data.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 20_000


# ================================================================================================
# fixtures / builders
# ================================================================================================

@pytest.fixture
def isolated_settings(qapp):
    """A private QSettings namespace, so a real saved ``windowState`` cannot hand these
    focus-sensitive tests a collapsed File Manager."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("S7FixupDockTabChain")
    QtCore.QSettings().clear()
    try:
        yield
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(app_name)


@pytest.fixture
def win(isolated_settings, qapp, qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()          # focus traversal only walks widgets that are visible to the top level
    QTest.qWait(10)
    return w


def _two_channel_doc(tmp_path) -> TraceDocument:
    """current + voltage, the shape the defect was measured on (IN15 current, IN14 voltage)."""
    chans = (
        ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="voltage", raw_units="mV", kind=ChannelKind.VOLTAGE),
    )
    arrays = (
        np.ascontiguousarray(np.full(N, -50.0, dtype=np.float32)),
        np.ascontiguousarray(np.full(N, -500.0, dtype=np.float32)),
    )
    meta = RecordingMeta(
        path=tmp_path / "rec.dat", format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=chans, nominal_voltage_mV=-500.0)
    return TraceDocument(meta=meta, channels=arrays)


def _row_widgets(win: MainWindow) -> list[QtWidgets.QWidget]:
    """Every widget the dock actually installed on a tree row, read off the tree itself
    (not off the dock's bookkeeping list, so this cannot pass by agreeing with the fix)."""
    tree = win._file_dock._tree
    found: list[QtWidgets.QWidget] = []
    it = QtWidgets.QTreeWidgetItemIterator(tree)
    while it.value():
        item = it.value()
        for col in range(tree.columnCount()):
            w = tree.itemWidget(item, col)
            if w is not None:
                found.append(w)
        it += 1
    return found


def _takes_tab_focus(widget: QtWidgets.QWidget) -> list[QtWidgets.QWidget]:
    out = []
    for w in [widget, *widget.findChildren(QtWidgets.QWidget)]:
        if int(w.focusPolicy().value) & int(QtCore.Qt.TabFocus.value):
            out.append(w)
    return out


def _row_widget_holding(win: MainWindow, focused) -> QtWidgets.QWidget | None:
    """The installed row widget ``focused`` belongs to, if any.

    The tree ITSELF is allowed in the Tab chain — focusing a view changes no selection. What
    must never happen is focus landing on a persistent *index* widget, because that is what
    ``QAbstractItemView::eventFilter`` converts into ``setCurrentIndex``."""
    if focused is None:
        return None
    for w in _row_widgets(win):
        if w is focused or w.isAncestorOf(focused):
            return w
    return None


# ================================================================================================
# 1. the keystroke
# ================================================================================================

def test_tab_out_of_the_notch_lines_box_leaves_the_selected_channel_alone(win, tmp_path):
    """THE DEFECT, as the user meets it: type the notch lines, press Tab to commit."""
    doc = _two_channel_doc(tmp_path)
    win._on_loaded(doc)
    QTest.qWait(10)
    win._file_dock.focus_channel(doc.doc_id, 0)
    QTest.qWait(10)

    before = win._current_selection
    assert before == (doc.doc_id, 0), "setup: the current channel should be selected"
    cal_before = win._calibration_panel._sel

    box = win._analysis_panel._detection.notch_freqs
    win._analysis_panel._detection.notch_use_lines.setChecked(True)
    QTest.qWait(10)
    box.setFocus()
    QTest.keyClicks(box, "60, 120")
    QTest.keyClick(box, QtCore.Qt.Key_Tab)
    QTest.qWait(50)

    landed = QtWidgets.QApplication.focusWidget()
    tree = win._file_dock._tree
    assert _row_widget_holding(win, landed) is None, (
        f"Tab walked into a File Manager row widget ({type(landed).__name__})")
    assert win._current_selection == before, (
        f"Tab moved the app selection {before} -> {win._current_selection}")
    assert win._calibration_panel._sel == cal_before, (
        "Tab repointed the Calibration panel — the panel whose Apply writes a sidecar")
    assert [i.text(0) for i in tree.selectedItems()] == ["current"], (
        "the dock's own selected row moved")


def test_repeated_tabs_never_reach_a_file_manager_row_widget(win, tmp_path):
    """Each further Tab used to step to the next row's spin (IN14, then IN15 again)."""
    doc = _two_channel_doc(tmp_path)
    win._on_loaded(doc)
    QTest.qWait(10)
    win._file_dock.focus_channel(doc.doc_id, 0)
    QTest.qWait(10)
    before = win._current_selection

    win._analysis_panel._detection.notch_use_lines.setChecked(True)   # the box is enabled by it
    QTest.qWait(10)
    box = win._analysis_panel._detection.notch_freqs
    box.setFocus()
    assert QtWidgets.QApplication.focusWidget() is box, "setup: focus should start in the box"
    visited = []
    for _ in range(12):
        w = QtWidgets.QApplication.focusWidget()
        if w is None:
            break
        QTest.keyClick(w, QtCore.Qt.Key_Tab)
        QTest.qWait(5)
        landed = QtWidgets.QApplication.focusWidget()
        visited.append(type(landed).__name__ if landed is not None else "None")
        assert _row_widget_holding(win, landed) is None, (
            f"Tab #{len(visited)} landed on a File Manager row widget "
            f"({type(landed).__name__}); chain so far: {visited}")
    assert win._current_selection == before, (
        f"tabbing around the window moved the app selection {before} -> "
        f"{win._current_selection} (chain: {visited})")


# ================================================================================================
# 2. the invariant, at the install site
# ================================================================================================

def test_no_widget_installed_on_a_file_manager_row_takes_tab_focus(win, tmp_path):
    """The rule ``_install_row_widget`` exists to enforce, checked on the live tree.

    Stated on the widgets themselves rather than on one keystroke, because Qt's focus-in →
    ``setCurrentIndex`` is a property of EVERY persistent index widget: any future row
    control inherits the defect the moment it accepts Tab."""
    win._on_loaded(_two_channel_doc(tmp_path))
    QTest.qWait(10)

    installed = _row_widgets(win)
    assert installed, "setup: the dock should have installed swatch + Δt widgets on the rows"
    offenders = [(type(w).__name__, [type(c).__name__ for c in _takes_tab_focus(w)])
                 for w in installed if _takes_tab_focus(w)]
    assert not offenders, (
        f"row widgets still in the Tab chain: {offenders} — a focus-in on any of them makes "
        f"the tree select that row")


def test_the_delta_t_spin_is_still_clickable_and_still_shifts_its_layer(win, tmp_path):
    """The fix removes Tab, not the control: the spin still takes a click and still writes.

    Guards the obvious over-correction (NoFocus / setEnabled(False) / dropped signal), which
    would silently cost the user the only way to align two recordings in time."""
    doc = _two_channel_doc(tmp_path)
    win._on_loaded(doc)
    QTest.qWait(10)

    layer_id = f"{doc.doc_id}#0"
    spin = win._file_dock._spins[layer_id]
    assert spin.isEnabled()
    assert spin.focusPolicy() == QtCore.Qt.ClickFocus, (
        f"the Δt spin should take click focus, got {spin.focusPolicy()}")

    QTest.mouseClick(spin, QtCore.Qt.LeftButton)
    QTest.qWait(10)
    focused = QtWidgets.QApplication.focusWidget()
    assert focused is spin or (focused is not None and spin.isAncestorOf(focused)), (
        f"clicking the Δt spin no longer focuses it (focus on {type(focused).__name__})")

    spin.setValue(0.25)
    QTest.qWait(10)
    assert win._session.layer(layer_id).time_offset_s == pytest.approx(0.25), (
        "editing the Δt spin no longer shifts the layer")
