"""Change C -- the Event inspector fits a docked column, and nothing in it moves by itself.

Measured before: a 1105x807 minimum (two rows of 1089 and 775 px, an 841 px amplitude
row, a 480x300 plot floor), and pyqtgraph's title widened the PlotItem past its widget,
cutting off 13 px of the waveform in a 486 px column (267 px with an amplitude clause).
Now: five short rows when narrow (two when wide), a 180 px plot floor, a readout that
scrolls and collapses (Details), and a title that never widens the plot.
"""

from __future__ import annotations

from ._inspector_hosting import real_font  # noqa: F401  (fixture)


import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_inspector import EventInspectorWindow

FS = 50_000.0
DOCK_WIDTHS = (460, 502, 560)
STALE = ("The open-pore estimator changed since this detection ran: these events were "
         "measured with one estimator and another is configured now. Depth, frac_depth, "
         "ECD, mean/max deviation and SNR are stale.")


def _settle(ms: int = 30) -> None:
    QTest.qWait(ms)
    QtWidgets.QApplication.processEvents()


@pytest.fixture
def inspector(qapp, qtbot, real_font, tmp_path):
    data = np.full(40_000, 100.0) + np.random.default_rng(3).normal(0, .5, 40_000)
    for start in range(2_000, 38_000, 4_000):
        data[start:start + 300] -= 60.0
    data = data.astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "layout.dat", format_name="synthetic",
                         n_samples=len(data), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(data,)))
    frame, prov = detect_events(data, FS, baseline=100.0, polarity=-1, min_dwell_s=2e-5)
    insp = EventInspectorWindow(session)
    qtbot.addWidget(insp)
    insp.set_events(doc_id, 0, frame, provenance=prov)
    assert insp._amplitude.apply_current(method="mean")   # the title's amplitude clause
    return insp


@pytest.fixture
def column(qapp, qtbot, inspector):
    """The inspector docked in a plain main window's right column, as MainWindow docks it."""
    main = QtWidgets.QMainWindow()
    qtbot.addWidget(main)
    main.setCentralWidget(QtWidgets.QWidget())
    dock = QtWidgets.QDockWidget("Event inspector", main)
    main.addDockWidget(Qt.RightDockWidgetArea, dock)
    dock.setWidget(inspector)
    inspector.setWindowFlags(Qt.Widget)
    inspector.set_host("dock")
    inspector.show()
    main.resize(1512, 847)
    main.show()
    _settle()
    return main, dock, inspector


def _with_counter_for(insp, n: int) -> None:
    """Size the Event box and the "of N" count for ``n`` events (what a 10 000-event
    detection does), without building one."""
    real = insp._df
    insp._df = pd.DataFrame(index=range(n))
    try:
        insp._fix_counter_widths()
    finally:
        insp._df = real


def test_every_narrow_row_fits_a_docked_column_with_ten_thousand_events(inspector):
    _with_counter_for(inspector, 10_000)
    inspector._arrange_toolbar(False)
    for i in range(inspector._toolbar_rows.count()):
        width = inspector._toolbar_rows.itemAt(i).layout().minimumSize().width()
        assert width <= 436, (i, width)
    for row in inspector._amplitude.findChildren(QtWidgets.QHBoxLayout):
        assert row.minimumSize().width() <= 436, row.minimumSize().width()
    assert inspector._summary_bar.minimumSizeHint().width() <= 448


def test_the_minimum_never_floors_a_docked_column(inspector):
    inspector.set_stale_columns(REMEASURED_COLUMNS, STALE)
    inspector._amplitude.method.setCurrentIndex(
        inspector._amplitude.method.findData("selected_median"))
    _settle()
    for host in ("window", "dock"):
        inspector.set_host(host)
        for shown in (True, False):
            inspector.set_details_shown(shown)
            hint = inspector.minimumSizeHint()
            assert hint.width() <= 460 and hint.height() <= 640, (host, shown, hint)


def _inside(widget, insp) -> bool:
    top_left = widget.mapTo(insp, QtCore.QPoint(0, 0))
    return insp.rect().contains(QtCore.QRect(top_left, widget.size()))


@pytest.mark.parametrize("width", DOCK_WIDTHS)
def test_at_docked_widths_nothing_is_clipped_and_the_waveform_fits(column, width):
    main, dock, insp = column
    main.resizeDocks([dock], [width], Qt.Horizontal)
    _settle()
    assert not insp._toolbar_is_wide
    assert insp._toolbar.horizontalScrollBar().maximum() == 0
    controls = [w for w in insp.findChildren(QtWidgets.QAbstractButton)
                + insp.findChildren(QtWidgets.QComboBox)
                + insp.findChildren(QtWidgets.QAbstractSpinBox)
                if w.isVisible() and w.window() is main]
    assert len(controls) > 15
    for widget in controls:
        assert _inside(widget, insp), widget.accessibleName() or widget.objectName() or widget
    view = insp._plot_widget.viewport().width()
    assert insp._plot.geometry().width() <= view + 1
    title = insp._plot.titleLabel
    assert "amplitude" in title.text                          # the whole title is kept
    painted = title.item.toPlainText()
    assert _drawn_whole(title) is False and title.toolTip() == title.text
    # ...and painted in a shorter form that keeps depth and dI/I0 whole.
    clauses = dict(c.strip().split(" ", 1) for c in painted.split("·") if " " in c.strip())
    full = dict(c.split(" ", 1) for c in title.text.split("   ·   "))
    assert clauses["depth"] == full["median"].removeprefix("depth "), (painted, title.text)
    assert clauses["dI/I0"] == full["dI/I0"], (painted, title.text)


def _drawn_whole(title) -> bool:
    # The label is HTML: it draws runs of spaces as one.
    return " ".join(title.item.toPlainText().split()) == " ".join(title.text.split())


def _clauses(text: str) -> dict[str, str]:
    """``"depth 59.93 pA · dI/I0 0.599"`` -> {"depth": "59.93 pA", "dI/I0": "0.599"}."""
    out = {}
    for clause in text.replace("…", "").split("·"):
        name, _, value = clause.strip().partition(" ")
        out[{"median": "depth"}.get(name, name)] = value.removeprefix("depth ")
    return out


def test_a_docked_title_always_shows_depth_and_di_i0_whole(column):
    """Review C-R2: elided at the right in a 460 px column, the title showed 0 of 60
    complete depths and no dI/I0 at all ("median depth 1…")."""
    main, dock, insp = column
    main.resizeDocks([dock], [460], Qt.Horizontal)
    _settle()
    title = insp._plot.titleLabel
    for event_id in insp._df["event_id"]:
        insp.show_event(int(event_id))
        _settle(5)
        full, painted = _clauses(title.text), _clauses(title.item.toPlainText())
        assert painted["depth"] == full["depth"] and painted["dI/I0"] == full["dI/I0"], (
            title.item.toPlainText(), title.text)
        assert title.toolTip() == ("" if _drawn_whole(title) else title.text)


def test_the_docked_readout_leads_with_depth_and_di_i0(column):
    """Review C-R2: Details shown in a short docked column, the pane had room for the box
    title and a blank reserved line only."""
    main, dock, insp = column
    insp.set_details_shown(True)
    _settle()
    lines = insp._readout.text().splitlines()
    assert lines[0].startswith("Event ") and lines[1].startswith("median depth:")
    assert lines[2].startswith("dI/I0:")
    assert insp._stale_label.isHidden(), "no blank reserved line while nothing is stale"
    insp.set_stale_columns(REMEASURED_COLUMNS, STALE)
    assert not insp._stale_label.isHidden()
    insp.set_stale_columns((), "")
    # At the pane's smallest, its first three lines are still in view.
    splitter = insp._detail_splitter
    splitter.setSizes([sum(splitter.sizes()), 0])
    _settle()
    area = insp._readout_area
    assert area.height() == area.minimumHeight()
    third = insp._readout.mapTo(area.viewport(), QtCore.QPoint(0, 0)).y() + 3 * \
        insp._readout.fontMetrics().lineSpacing()
    assert third <= area.viewport().height(), (third, area.viewport().height())
    # In its own window the order and the reserved line are as before.
    insp.setParent(main, Qt.Window)
    dock.hide()
    insp.set_host("window")
    lines = insp._readout.text().splitlines()
    assert lines[1].startswith("accepted:") and not insp._stale_label.isHidden()


def test_in_a_short_docked_column_the_body_scrolls_and_nothing_overlaps(column):
    """Review C-R3/C-R4: docked, the inspector no longer floors the column's height; below
    its content's own minimum the body scrolls, and every row keeps its height."""
    main, dock, insp = column
    main.resize(1512, 560)
    insp.set_details_shown(True)
    amp = insp._amplitude
    amp.method.setCurrentIndex(amp.method.findData("selected_median"))
    insp._level_note.setText("No sub-state levels for this event: the fit could not run on "
                             "this mask (412 samples). " * 2)
    insp._level_note.setVisible(True)
    _settle(60)
    assert main.minimumSizeHint().height() <= 560
    assert insp._body.verticalScrollBar().maximum() > 0, "the body scrolls"
    plot, readout = insp._plot_widget, insp._readout_area
    assert plot.height() >= plot.minimumHeight()
    assert readout.height() >= readout.minimumHeight()
    plot_bottom = plot.mapTo(insp._body.widget(), QtCore.QPoint(0, plot.height())).y()
    assert plot_bottom <= readout.mapTo(insp._body.widget(), QtCore.QPoint(0, 0)).y()
    amp_bottom = amp.mapTo(insp._body.widget(), QtCore.QPoint(0, amp.height())).y()
    assert amp_bottom <= plot.mapTo(insp._body.widget(), QtCore.QPoint(0, 0)).y()


def test_a_short_axis_abbreviates_its_label(column):
    """Review C-R6: the 206 px "Measurement current (pA)" was cut at both ends on a
    128-187 px axis."""
    main, dock, insp = column
    axis = insp._plot.getAxis("left")
    insp.set_details_shown(False)
    _settle()
    assert axis.height() > axis.label.boundingRect().width() + 4
    assert axis.labelText == "Measurement current"
    main.resize(1512, 520)
    insp.set_details_shown(True)
    _settle(60)
    assert axis.labelText == "I" and axis.labelUnits == insp._units, axis.labelText
    assert axis.label.boundingRect().width() <= axis.height()
    main.resize(1512, 847)
    insp.set_details_shown(False)
    _settle(60)
    assert axis.labelText == "Measurement current"


def test_the_toolbar_is_two_rows_when_wide_and_five_when_narrow(qtbot, inspector):
    inspector.resize(1100, 860)
    inspector.show()
    _settle()
    rows = [inspector._toolbar_rows.itemAt(i).layout() for i in range(5)]
    assert inspector._toolbar_is_wide and [r.count() > 0 for r in rows] == [True] * 2 + [False] * 3
    assert rows[0].indexOf(inspector._accept_btn) >= 0 <= rows[0].indexOf(inspector._prev_btn)
    inspector.resize(600, 860)
    _settle()
    assert not inspector._toolbar_is_wide and all(r.count() > 0 for r in rows)
    assert rows[0].indexOf(inspector._accept_btn) >= 0 <= rows[0].indexOf(inspector._prev_btn)
    assert inspector._toolbar.horizontalScrollBar().maximum() == 0


def test_out_of_the_dock_the_two_row_toolbar_lies_inside_its_strip(column):
    """Measured: re-arranged while the window was being reparented out of the dock, the
    rows kept the five-row strip's height, and the second row fell out of sight."""
    main, dock, insp = column
    assert not insp._toolbar_is_wide
    insp.setParent(main, Qt.Window)
    dock.hide()
    insp.set_host("window")
    insp.resize(1100, 860)
    insp.show()
    _settle(60)
    assert insp._toolbar_is_wide
    strip = insp._toolbar.viewport().rect()
    for widget in (insp._prev_btn, insp._accept_btn, insp._filter_combo, insp._merge_next_btn):
        top_left = widget.mapTo(insp._toolbar.viewport(), QtCore.QPoint(0, 0))
        assert strip.contains(QtCore.QRect(top_left, widget.size())), widget.accessibleName()
    # With the room, the whole title is drawn and needs no tooltip.
    assert _drawn_whole(insp._plot.titleLabel) and insp._plot.titleLabel.toolTip() == ""


def test_the_details_toggle_moves_neither_itself_nor_the_waveforms_origin(column):
    _main, _dock, insp = column
    assert not insp.details_shown()                   # docked default: the waveform's room
    spot = insp._details_btn.mapToGlobal(QtCore.QPoint(0, 0))
    origin = insp._plot_widget.mapTo(insp, QtCore.QPoint(0, 0))
    height = insp._plot_widget.height()
    insp._details_btn.click()
    _settle()
    assert insp.details_shown() and not insp._readout_area.isHidden()
    assert insp._details_btn.mapToGlobal(QtCore.QPoint(0, 0)) == spot
    assert insp._plot_widget.mapTo(insp, QtCore.QPoint(0, 0)) == origin
    assert insp._plot_widget.height() < height
    assert insp._readout_area.height() <= 0.4 * (height + 8) + 2
    insp._details_btn.click()
    _settle()
    assert insp._plot_widget.height() == height
    assert insp._details_btn.mapToGlobal(QtCore.QPoint(0, 0)) == spot


def test_details_are_remembered_per_host(inspector):
    assert inspector.details_state() == {"window": True, "dock": False}
    inspector.set_host("dock")
    inspector._details_btn.click()
    assert inspector.details_state() == {"window": True, "dock": True}
    inspector.set_host("window")
    assert inspector.details_shown() and inspector._details_btn.isChecked()
    inspector._details_btn.click()
    assert inspector.details_state() == {"window": False, "dock": True}
    inspector.restore_details_state({"window": True, "dock": "yes"})   # a bad value is ignored
    assert inspector.details_state() == {"window": True, "dock": True}
    inspector.reset_details_state()
    assert inspector.details_state() == {"window": True, "dock": False}


def _geometry(insp) -> tuple:
    def rect(w):
        return (w.mapTo(insp, QtCore.QPoint(0, 0)).toTuple(), w.size().toTuple())
    vb = insp._plot.getViewBox().geometry()
    return (rect(insp._plot_widget), rect(insp._summary_bar), rect(insp._toolbar),
            rect(insp._details_btn), (round(vb.width()), round(vb.height())))


@pytest.mark.parametrize("shown", [False, True])
def test_the_staleness_flip_moves_nothing_in_the_docked_column(column, shown):
    main, dock, insp = column
    main.resizeDocks([dock], [502], Qt.Horizontal)
    insp.set_details_shown(shown)
    _settle(60)
    clean = _geometry(insp)
    insp.set_stale_columns(REMEASURED_COLUMNS, STALE)
    _settle(60)
    assert _geometry(insp) == clean
    assert insp._details_btn.text().startswith("⚠ ")
    assert STALE in insp._details_btn.toolTip() and insp.stale_note() == STALE
    insp.set_stale_columns((), "")
    _settle(60)
    assert _geometry(insp) == clean and not insp._details_btn.text().startswith("⚠")


def test_the_host_button_appears_only_once_hosted(inspector):
    assert inspector._host_btn.isHidden(), "a standalone window has nobody to move it"
    emitted: list[int] = []
    inspector.hostToggleRequested.connect(lambda: emitted.append(1))
    inspector.set_host("window")
    assert not inspector._host_btn.isHidden()
    assert inspector._host_btn.text() == "Dock beside trace"
    inspector._host_btn.click()
    assert emitted == [1]
    inspector.set_host("dock")
    assert inspector._host_btn.text() == "Own window"
