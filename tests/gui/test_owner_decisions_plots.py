"""The owner's decisions on four plot-window questions (after sweep 10; design 13.4 and 13.6).

* **A Plots fit over a typed Fit range is not filed** (S10-LIN3 lead). Closing the window
  filed every dwell or depth fit under ``fit.dwell`` / ``fit.depth``, one made over a typed Fit
  range included: it records neither the range nor its count, so Generate Report figures drew
  it over the whole column as "your saved fit". It is now not filed, the report fits the
  column afresh, and the FIT RANGE line says so. A fit with no range is filed as before.
* **An emptied Title, X or Y label box is the automatic label in every window** (13.4 Q1).
  Comparison and Study figures drew no label for an emptied box while the other windows drew
  their own, with both boxes showing the same "Automatic" placeholder. A box of spaces now
  hides the label, everywhere.
* **The Plots window's STALE mark follows a Baseline-form edit** (13.4 Q4), once the knob
  stops: ``stalenessChanged`` reached no Plots redraw, so the mark waited for the next draw.
* **A plain text field follows the ⌘Z rule** (13.6). ``_UndoKeyGuard`` let ⌘Z through a spin
  box or an editable combo with nothing typed in it; a focused plain QLineEdit (a plot's Title
  box) kept the key even then. Keys are sent with ``QTest.keyClick`` to the focused field.

Synthetic recordings only; every ``meta.path`` is under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.gui import main_window as mw
from nanoscope.gui.key_sequences import redo_sequences, undo_sequences
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui import plot_appearance
from nanoscope.gui.plot_appearance import PlotAppearance

#: What a box of spaces stores (``plot_appearance.HIDDEN_LABEL``); spelled out so each test
#: here fails on its own behaviour on a tree without the decision, not on the import.
HIDDEN_LABEL = " "

from .test_editing_dependencies import _headline, _open, _plots
from .test_editing_undo_reach import _accepted, _frame, _reject, make_window  # noqa: F401
from .test_plot_appearance_history import PAUSE_MS, _build, _study_window
from .test_sweep10_keys import _focus, _key, _plot_window

REFIT = "depth: no saved fit — the figure was refit fresh at k=1"


@pytest.fixture
def window(qapp, qtbot, monkeypatch, tmp_path):
    """Every export dialog is answered with ``tmp_path / 'figs.pdf'``."""
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: tmp_path / "figs.pdf")
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._controller.abort()


def _export(w: MainWindow) -> str:
    """Analysis ▸ Generate Report figures…, and the status line it leaves."""
    w._export_figures()
    QtWidgets.QApplication.processEvents()
    return w.statusBar().currentMessage()


# ================================================================================================
# 1. a Plots fit over a typed Fit range is not the table's fit
# ================================================================================================

def _depth_fit(w: MainWindow, fit_min: str = "") -> str:
    """Analysis ▸ Plots, a Gaussian mixture on frac_depth (over ``fit_min`` when given); the
    Details text, read before the window closes."""
    plots = _plots(w)
    plots.set_histogram_column("frac_depth")
    plots.set_fit("gaussian-mixture")
    if fit_min:
        plots._fit_lo.setText(fit_min)
        plots._changed()
    QtWidgets.QApplication.processEvents()
    details = plots.details_text()
    plots.close()
    QtWidgets.QApplication.processEvents()
    return details


def test_a_fit_over_a_typed_fit_range_is_not_filed_and_the_report_fits_afresh(window, tmp_path):
    doc = _open(window, tmp_path)
    details = _depth_fit(window, fit_min="0.26")
    line = next(ln for ln in details.splitlines() if "FIT RANGE" in ln)
    assert "This fit is not kept for Generate Report figures" in line, line

    assert window._session.results.get(doc, 0, "fit.depth") is None, \
        "a fit of a narrowed sample was filed as the table's depth fit"
    said = _export(window)
    assert REFIT in said, said


def test_a_fit_without_a_range_is_filed_as_before(window, tmp_path):
    """CONTROL: the owner kept this half. No FIT RANGE line, no word about the report."""
    doc = _open(window, tmp_path)
    details = _depth_fit(window)
    assert "FIT RANGE" not in details and "Generate Report" not in details

    stored = window._session.results.get(doc, 0, "fit.depth")
    assert stored is not None and stored.params["column"] == "frac_depth"
    assert "depth: used your saved fit (k=1)" in _export(window)


def test_a_range_typed_after_the_draw_still_keeps_the_fit_off_the_report(window, tmp_path):
    """The boxes are asked as the window closes too: a fit restored with a project's range,
    or a range typed and not yet applied, is not filed as a whole-column fit."""
    doc = _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("frac_depth")
    plots.set_fit("gaussian-mixture")
    QtWidgets.QApplication.processEvents()
    plots._fit_lo.setText("0.26")          # no editingFinished: the fit on screen is unranged
    plots.close()
    QtWidgets.QApplication.processEvents()
    assert window._session.results.get(doc, 0, "fit.depth") is None


# ================================================================================================
# 2. an emptied label box is the automatic label; a box of spaces hides it
# ================================================================================================

def test_an_emptied_box_is_stored_as_automatic_and_spaces_as_hidden():
    assert getattr(plot_appearance, "HIDDEN_LABEL", None) == HIDDEN_LABEL
    assert PlotAppearance.from_mapping({"title": "", "x_label": "", "y_label": ""}) \
        == PlotAppearance()
    assert PlotAppearance.from_mapping({"title": "   ", "x_label": "\t"}) \
        == PlotAppearance(title=HIDDEN_LABEL, x_label=HIDDEN_LABEL)
    assert PlotAppearance.from_mapping({"y_label": "  Rate (Hz) "}).y_label == "Rate (Hz)"


def _empty(box) -> None:
    """Type a character into the box and take it out again, as the user empties it."""
    QTest.keyClicks(box, "x")
    QTest.keyClick(box, Qt.Key_Backspace)


def test_an_emptied_box_draws_the_automatic_label_on_a_comparison_figure(qtbot):
    from nanoscope.plot.comparison_figures import render_comparison

    comparison = _build("comparison")
    qtbot.addWidget(comparison)
    original = render_comparison(comparison._snapshot, comparison._tables, comparison._appearance)
    auto_title = original._suptitle.get_text()
    auto_x = {a.get_xlabel() for a in original.axes if a.get_visible()}
    assert auto_title and auto_x != {""}, "precondition: the figure has labels of its own"

    widget = comparison._appearance_widget
    for box in (widget.title, widget.x_label, widget.y_label):
        _empty(box)
    assert comparison.appearance_state() == {}
    figure = render_comparison(comparison._snapshot, comparison._tables, comparison._appearance)
    assert figure._suptitle is not None and figure._suptitle.get_text() == auto_title
    assert {a.get_xlabel() for a in figure.axes if a.get_visible()} == auto_x


def test_an_emptied_box_draws_the_original_labels_on_a_study_figure(qtbot):
    study = _study_window(suptitle="Original suptitle")
    qtbot.addWidget(study)
    _empty(study._appearance_widget.title)
    for name in ("title", "x_label", "y_label"):          # the per-panel boxes
        study._panel_edits[name].setText("x")
        study._panel_edits[name].textEdited.emit("x")
        study._panel_edits[name].setText("")
        study._panel_edits[name].textEdited.emit("")
    assert study.apply_appearance()
    axis = study._figure.axes[0]
    assert study._figure._suptitle.get_text() == "Original suptitle"
    assert (axis.get_title(), axis.get_xlabel(), axis.get_ylabel()) \
        == ("Accepted events", "Dwell (ms)", "Current (pA)")


def test_a_box_of_spaces_hides_the_label_in_every_kind_of_window(qtbot):
    from nanoscope.plot.comparison_figures import render_comparison

    # Comparison figures
    comparison = _build("comparison")
    qtbot.addWidget(comparison)
    for box in (comparison._appearance_widget.title, comparison._appearance_widget.x_label):
        QTest.keyClicks(box, " ")
    figure = render_comparison(comparison._snapshot, comparison._tables, comparison._appearance)
    assert figure._suptitle is None
    assert {a.get_xlabel() for a in figure.axes if a.get_visible()} == {""}

    # Study figures: the figure's title and a panel's own boxes
    study = _study_window(suptitle="Original suptitle")
    qtbot.addWidget(study)
    QTest.keyClicks(study._appearance_widget.title, " ")
    study._panel_edits["x_label"].setText(" ")
    study._panel_edits["x_label"].textEdited.emit(" ")
    assert study.apply_appearance()
    assert study._figure._suptitle.get_text().strip() == ""
    assert study._figure.axes[0].get_xlabel() == ""

    # a pyqtgraph window (I-V): the title row is hidden, not a blank band
    iv = _build("iv")
    qtbot.addWidget(iv)
    QTest.keyClicks(iv._appearance_widget.title, "Custom")
    assert iv._plot.titleLabel.isVisible()
    iv._appearance_widget.title.selectAll()
    QTest.keyClicks(iv._appearance_widget.title, " ")
    assert iv.appearance_state().get("title") == HIDDEN_LABEL
    assert not iv._plot.titleLabel.isVisible()
    QTest.keyClicks(iv._appearance_widget.x_label, " ")
    assert iv._plot.getAxis("bottom").labelText.strip() == ""


def test_a_box_of_spaces_hides_the_plots_windows_title_and_axis_label(window, tmp_path):
    _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("frac_depth")
    plots.set_fit("gaussian-mixture")           # plotux G3: a fit's numbers are in the fit label, never the title
    QtWidgets.QApplication.processEvents()
    assert plots._fit_label_item.plain_text() and plots._plot.getAxis("bottom").labelText
    widget = plots._appearance_widget
    QTest.keyClicks(widget.title, " ")
    QTest.keyClicks(widget.x_label, " ")
    assert not plots._plot.titleLabel.isVisible() or not plots._plot.titleLabel.text.strip()
    assert plots._plot.getAxis("bottom").labelText.strip() == ""
    widget.title.clear()
    widget.title.textEdited.emit("")
    widget.x_label.clear()
    widget.x_label.textEdited.emit("")
    # An emptied box is the automatic title -- none for a histogram, now that the fit
    # summary is not the title -- and the fit's numbers stayed on the figure throughout.
    assert not plots._plot.titleLabel.isVisible() or not plots._plot.titleLabel.text.strip()
    assert plots._fit_label_item.plain_text()
    assert plots._plot.getAxis("bottom").labelText.strip()


# ================================================================================================
# 3. the STALE mark follows a Baseline-form edit, once the knob stops
# ================================================================================================

def _count_redraws(monkeypatch, plots) -> list[int]:
    drawn: list[int] = []
    real = plots._redraw

    def counted():
        drawn.append(1)
        real()
    monkeypatch.setattr(plots, "_redraw", counted)
    return drawn


def test_a_baseline_edit_marks_the_open_plots_window_once_the_knob_stops(window, tmp_path, qtbot,
                                                                         monkeypatch):
    _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("depth_pA")
    QtWidgets.QApplication.processEvents()
    assert plots.isVisible() and "STALE" not in _headline(plots)
    drawn = _count_redraws(monkeypatch, plots)

    rough = window._analysis_panel.baseline_panel.rough_s
    for value in (1.5, 2.0, 2.5):                       # a knob held down: three steps
        rough.setValue(value)
        QtWidgets.QApplication.processEvents()
    assert window._analysis_panel._stale.kind == "baseline", "precondition: the line moved"
    assert not drawn, "the window redrew on every knob step"

    qtbot.waitUntil(lambda: "⚠ STALE: depth_pA" in _headline(plots), timeout=3000)
    assert "STALE: depth_pA on this plot" in plots.details_text()
    qtbot.wait(400)
    assert len(drawn) == 1, f"{len(drawn)} redraws for one run of knob steps"

    window._analysis_panel.stalenessChanged.emit()      # re-judged, and nothing moved
    qtbot.wait(450)
    assert len(drawn) == 1, "a verdict that moved no mark redrew the window"

    rough.setValue(3.0)                 # the sentence names the window configured now
    qtbot.wait(450)
    assert len(drawn) == 2


def test_a_baseline_edit_reverted_takes_the_mark_down(window, tmp_path, qtbot):
    _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("depth_pA")
    rough = window._analysis_panel.baseline_panel.rough_s
    before = rough.value()
    rough.setValue(2.0)
    qtbot.waitUntil(lambda: "⚠ STALE: depth_pA" in _headline(plots), timeout=3000)
    rough.setValue(before)
    qtbot.waitUntil(lambda: "STALE" not in _headline(plots), timeout=3000)
    assert "STALE" not in plots.details_text()


def test_a_closed_plots_window_is_left_alone(window, tmp_path, qtbot, monkeypatch):
    """CONTROL: a closed window is not redrawn, and nothing is left running for it."""
    _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("depth_pA")
    QtWidgets.QApplication.processEvents()
    plots.close()
    QtWidgets.QApplication.processEvents()
    drawn = _count_redraws(monkeypatch, plots)
    window._analysis_panel.baseline_panel.rough_s.setValue(2.0)
    qtbot.wait(450)
    timer = getattr(plots, "_stale_redraw_timer", None)
    assert not drawn and (timer is None or not timer.isActive())


# ================================================================================================
# 4. a plain text field lets ⌘Z through unless it has typing of its own
# ================================================================================================

def _iv_with_one_step(qtbot):
    window, w = _plot_window(qtbot, "iv")
    w.title_size.setValue(14)
    qtbot.wait(PAUSE_MS)
    assert window.appearance_state() == {"title_size": 14.0}
    return window, w


def test_cmd_z_in_an_untouched_title_box_undoes_the_plots_appearance(qtbot):
    window, w = _iv_with_one_step(qtbot)
    _focus(qtbot, w.title)
    _key(undo_sequences()[0])
    assert window.appearance_state() == {}, "⌘Z in an untouched Title box was kept by the box"
    for key in redo_sequences():
        assert QtWidgets.QApplication.focusWidget() is w.title
        _key(key)
        assert window.appearance_state() == {"title_size": 14.0}, key.toString()
        _key(undo_sequences()[0])
        assert window.appearance_state() == {}


def test_a_title_box_the_user_typed_in_keeps_cmd_z_for_its_typing(qtbot):
    """CONTROL: the documented exception holds for a plain field too."""
    window, w = _iv_with_one_step(qtbot)
    _focus(qtbot, w.title)
    QTest.keyClicks(w.title, "abc")
    qtbot.wait(PAUSE_MS)
    depth = w._history.depth()
    assert w.title.isUndoAvailable()
    _key(undo_sequences()[0])
    assert w.title.text() == "", "⌘Z did not take back what was typed"
    assert w._history.depth() == depth, "⌘Z typed in the Title box moved the appearance history"
    assert window.appearance_state().get("title_size") == 14.0


def _plain_field(w: MainWindow) -> QtWidgets.QLineEdit:
    """The Calibration tab's Target units box, its tab brought forward as a click does. A
    plain field: it applies on Return, so typing in it moves nothing else."""
    from nanoscope.gui.calibration_panel import CalibrationPanel

    panel = w.findChild(CalibrationPanel)
    page = panel
    while not isinstance(page.parentWidget().parentWidget(), QtWidgets.QTabWidget):
        page = page.parentWidget()
    page.parentWidget().parentWidget().setCurrentWidget(page)
    QTest.qWait(10)
    field = panel._units
    assert field.isVisible() and field.isEnabled() and not field.text(), "precondition"
    return field


@pytest.mark.parametrize("read_only", [False, True], ids=["editable", "read-only"])
def test_cmd_z_in_an_untouched_main_window_field_reaches_edit_undo(make_window, read_only):
    w, _doc_id = make_window()
    w.activateWindow()
    QTest.qWait(20)
    fired: list[int] = []
    w._act_undo.triggered.connect(lambda *_: fired.append(1))
    event_id = int(_frame(w)["event_id"].iloc[1])
    _reject(w, event_id)
    field = _plain_field(w)
    field.setReadOnly(read_only)
    field.setFocus(Qt.OtherFocusReason)
    QTest.qWait(10)
    assert QtWidgets.QApplication.focusWidget() is field
    _key(undo_sequences()[0])
    assert fired, "⌘Z in an untouched text field never reached Edit ▸ Undo"
    assert _accepted(w, event_id)


def test_a_main_window_field_the_user_typed_in_keeps_cmd_z(make_window):
    """CONTROL in the main window: typing in the field keeps ⌘Z for the typing."""
    w, _doc_id = make_window()
    w.activateWindow()
    QTest.qWait(20)
    fired: list[int] = []
    w._act_undo.triggered.connect(lambda *_: fired.append(1))
    event_id = int(_frame(w)["event_id"].iloc[1])
    _reject(w, event_id)
    field = _plain_field(w)
    field.setFocus(Qt.OtherFocusReason)
    QTest.qWait(10)
    before = field.text()
    QTest.keyClicks(field, "7")
    assert field.text() != before
    _key(undo_sequences()[0])
    assert field.text() == before, "⌘Z did not take back what was typed"
    assert not fired and not _accepted(w, event_id), "⌘Z typed in a field undid the table"
