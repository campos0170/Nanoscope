"""The Plots window's RUN GATE, its title hygiene, and the room its controls take.

Three round-4 complaints about the same window, and none of them is about a number:

* **Q1 "the data should not be plotted by default — the user needs to make the
  selection of the data and type of plot, and then run it."** Every tick of a source
  and every walk down the plot-type combo used to re-histogram, re-fit and re-run the
  advisory batch behind it. The plot now waits: a change of DATA SELECTION, PLOT TYPE
  or AXIS COLUMN marks the picture stale and says so, and ``Plot`` is what redraws.
  Cosmetics — fonts, colours, grid, legend — stay LIVE, because making someone press a
  button to see a colour is the same complaint pointing the other way.
* **Q6 "the title from one plot should not transfer automatically to another … the
  title for a dwell time histogram is not the same as the title for a frac_io vs. time
  KDE plot."** Asked whether a DATA change should clear it too, the owner said no. So
  the rule is exactly: plot type or axis column clears it, sources do not, and there is
  no dialog.
* **layout "the Population Plots panel needs more room to expand."** Two measured bugs:
  the controls column was capped 24 px NARROWER than its own contents (so a horizontal
  scrollbar was pinned to the panel permanently and the splitter could not be dragged),
  and the splitter allotted it 446 px while the widget painted 420 (a 26 px dead
  gutter). The numbers in this file are the before/after of those measurements.

WHAT THE GATE DELIBERATELY DOES NOT GATE: the programmatic surface. ``set_data``,
``set_plot_type``, ``set_fit`` and the rest draw immediately, exactly as they always
have — naming a plot in code IS the run gesture, MainWindow's Refresh path depends on
it, and so does the ~96-site test corpus around this window. The gate is about the
mouse, which is where the cost the owner is complaining about actually is. The
``shown`` fixture is therefore load-bearing here for the same reason it is for the
Apply-fit gate: a person's gesture is one with the popup open or the widget holding
focus, and a never-shown widget can have neither.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import PlotsWindow


# ================================================================================================
# builders
# ================================================================================================

def _events(n: int = 200, *, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(3e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "frac_depth": np.abs(rng.normal(0.3, 0.05, n)),
        "accepted": np.ones(n, dtype=bool),
    })


def _source(key: str, df: pd.DataFrame, *, label: str | None = None) -> PlotSource:
    return PlotSource(key=key, label=label or key, df=df, provenance={},
                      colour="#0072B2", doc_id=key, channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source("a", _events(seed=1), label="apo_-500 mV.csv"),
                _source("b", _events(seed=2), label="apo_-700 mV.csv")])
    w.set_sources(["a"])
    return w


@pytest.fixture
def shown(win, qapp):
    """On screen, so a real gesture can be told from code's — see the module docstring."""
    win.show()
    qapp.processEvents()
    return win


def _user_picks(qapp, combo, text: str) -> None:
    """Choose ``text`` the way a PERSON does: with the dropdown open."""
    combo.showPopup()
    qapp.processEvents()
    combo.setCurrentText(text)
    qapp.processEvents()
    combo.hidePopup()
    qapp.processEvents()


def _tick(w, row: int, on: bool) -> None:
    """Tick a source row the way a click does — through the item, not through
    ``set_sources``, because ``set_sources`` is the programmatic contract that must
    keep drawing."""
    w._source_list.item(row).setCheckState(Qt.Checked if on else Qt.Unchecked)


# ================================================================================================
# Q1 — the run gate
# ================================================================================================

def test_the_programmatic_surface_still_draws_immediately(win):
    """The contract ~96 test sites and MainWindow's Refresh path stand on."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert w._hist_item is not None
    assert w._plot_armed and not w._stale, "code naming a plot IS the run gesture"


def test_a_window_told_not_to_autoplot_opens_on_the_prompt(qapp, qtbot):
    """The "opens showing an empty plot" half of Q1. Off by default so that pushing
    data stays a draw for every existing caller; one flag turns it on."""
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.auto_plot = False
    w.set_data([_source("a", _events())])

    assert not w._plot_armed
    assert w._hist_item is None, "nothing was plotted"
    assert pp._PLOT_PROMPT in w.details_text()
    assert w._message_item is not None and pp._PLOT_PROMPT in w._message_item.toPlainText()
    # ...and the sources really did arrive; only the picture is waiting.
    assert w.source_labels() == ["a"]
    assert "dwell_s" in [w._hist_col.itemText(i) for i in range(w._hist_col.count())]

    w._plot_btn.click()
    assert w._plot_armed and w._hist_item is not None


def test_ticking_a_source_marks_the_plot_stale_instead_of_redrawing(shown):
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    drawn = list(w.legend_labels()), w.details_text()
    assert not w._stale

    _tick(w, 1, True)                       # a second pore, by hand
    assert w._stale, "a data-selection change must not redraw on its own"
    assert (list(w.legend_labels()), w.details_text()) == drawn, "the picture moved"
    # ...but the column pickers DID re-sync, or the user could not choose what to plot.
    assert w.selected_source_keys() == ["a", "b"]

    w._plot_btn.click()
    assert not w._stale
    assert w.details_text() != drawn[1], "Plot is what redraws"


def test_the_stale_state_is_visible_not_merely_recorded(shown):
    """A gate the user cannot see is a control that stopped working — the same rule
    that put the ● on Apply fit."""
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert w._plot_btn.text() == "Plot"
    assert not w._plot_state.isVisible()

    _tick(w, 1, True)
    assert w._plot_btn.text().endswith("●")
    assert w._plot_state.isVisible()
    banner = w._plot_state.text()
    assert "data selection" in banner and "Plot" in banner
    assert pp._WARN_HEX in w._plot_state.styleSheet()

    w._plot_btn.click()
    assert w._plot_btn.text() == "Plot"
    assert not w._plot_state.isVisible()


def test_a_hand_picked_plot_type_waits_but_a_coded_one_does_not(shown, qapp):
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert w._hist_item is not None

    _user_picks(qapp, w._plot_type_combo, "Scatter")
    assert w._stale and "plot type" in w._plot_state.text()
    assert w._hist_item is not None, "the previous picture is still the one on screen"

    w._plot_btn.click()
    assert not w._stale
    assert w._hist_item is None and w._scatter_item is not None


def test_a_hand_picked_axis_column_waits(shown, qapp):
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    before = w.details_text()

    _user_picks(qapp, w._hist_col, "depth_pA")
    assert w._stale and "axis column" in w._plot_state.text()
    assert w.details_text() == before

    w._plot_btn.click()
    assert "depth_pA" in w.details_text()


@pytest.mark.parametrize("move, what", [
    (lambda w: w._font_size.setValue(15), "body font size"),
    (lambda w: w._grid_check.setChecked(False), "gridlines"),
    (lambda w: w._legend_check.setChecked(False), "the legend"),
    (lambda w: w._bg_combo.setCurrentText("Light"), "the background"),
    (lambda w: w._title_edit.setText("Figure 3b"), "the title"),
    (lambda w: w._point_size.setValue(11), "the marker size"),
])
def test_cosmetic_controls_stay_live(shown, move, what):
    """The other half of the gate. None of these re-reads a row, so none of them is
    allowed to make the user press Plot again."""
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    move(w)
    assert not w._stale, f"{what} is cosmetic and must not gate the plot"
    assert not w._plot_state.isVisible()


def test_the_fit_apply_buttons_are_the_same_gate_as_plot(shown, qapp):
    """One gate, three doors. Two buttons that each commit a DIFFERENT half of the
    picture is how a user reads a gamma's k off a plot still drawing an exponential."""
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")

    _user_picks(qapp, w._fit_combo, "exponential")   # a proposal, not a fit
    _tick(w, 1, True)                                # ...and a stale data selection
    assert w._stale and w._fit_apply_btn.isEnabled()
    assert w._plot_btn.text().endswith("●"), "the Plot button badges the fit too"

    w._fit_apply_btn.click()                         # the OTHER door
    assert not w._stale, "Apply fit commits the whole selection, not half of it"
    assert not w._fit_apply_btn.isEnabled()
    assert w._plot_btn.text() == "Plot"
    assert w._fit_item is not None
    assert len(w.selected_source_keys()) == 2


def test_a_background_note_never_walks_through_the_gate(shown, qapp):
    """The advisory worker redraws when its batch lands. That redraw is about the
    picture that was DRAWN, so it must not quietly render the selection the user has
    changed but not yet run."""
    w = shown
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_fit("exponential")
    before = w.details_text()

    _tick(w, 1, True)
    assert w._stale
    w._on_notes_ready([])          # exactly what the worker's `done` signal delivers
    assert w.details_text() == before, "a note redrew the plot behind the run gate"
    assert w._stale, "...and it silently cleared the gate as well"


# ================================================================================================
# Q6 — the title clears when the ANALYSIS changes, and only then
# ================================================================================================

def test_the_title_and_labels_go_blank_when_the_plot_type_changes(win):
    """"The title for a dwell time histogram is not the same as the title for a
    frac_io vs. time KDE plot." No dialog: the owner resolved that himself."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_title("Apo dwell distribution, −500 mV")
    w._x_label_edit.setText("Dwell time (ms)")
    w._y_label_edit.setText("Events per bin")

    w.set_plot_type("KDE (density)")
    assert w.style().title == ""
    assert w.style().x_label == "" and w.style().y_label == ""
    # the WIDGETS too, or the box and the plot disagree about what the title is
    assert w._title_edit.text() == ""
    assert w._x_label_edit.text() == "" and w._y_label_edit.text() == ""


def test_the_title_goes_blank_when_an_axis_column_changes(win):
    """An axis label is the stronger case: "Dwell time (ms)" left standing over a
    frac_depth axis is not a stale caption, it is a figure stating the wrong quantity —
    in the one place ``Save image`` exports."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_title("Apo dwell distribution")
    w._x_label_edit.setText("Dwell time (ms)")

    w.set_histogram_column("depth_pA")
    assert w.style().title == "" and w.style().x_label == ""


def test_the_title_survives_a_change_of_DATA(win):
    """Asked directly, the owner said no: "no need to clear the data. the user selects
    the data from the file manager as is done now." A caption naming the molecule and
    the voltage survives a third pore being ticked; it does not survive becoming a KDE.
    """
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_title("Apoferritin, −500 vs −700 mV")
    w._x_label_edit.setText("Dwell time (ms)")

    w.set_sources(["a", "b"])
    assert w.style().title == "Apoferritin, −500 vs −700 mV"
    assert w.style().x_label == "Dwell time (ms)"

    w.set_sources(["a"])
    assert w.style().title == "Apoferritin, −500 vs −700 mV"

    w._accepted_check.setChecked(False)
    assert w.style().title == "Apoferritin, −500 vs −700 mV"


def test_the_repopulation_that_follows_a_refresh_does_not_clear_the_title(win):
    """``set_data`` refills every column combo, which fires the same signal a user's
    pick does. If that counted as an axis-column change, a Refresh would silently blank
    the caption of a figure the user was in the middle of composing."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_title("figure 3b")

    w.set_data([_source("a", _events(seed=4), label="apo_-500 mV.csv")])
    assert w.style().title == "figure 3b"


def test_clearing_the_title_asks_nothing(win, monkeypatch):
    """No prompt, in his words. A modal here would also be unanswerable from the
    programmatic path every other window drives this one through."""
    def _boom(*_a, **_k):       # pragma: no cover - the point is that it never runs
        raise AssertionError("a dialog was raised on a plot-type change")

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, staticmethod(_boom))
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_title("figure 3b")
    w.set_plot_type("Scatter")
    assert w.style().title == ""


def test_run_pca_switches_the_plot_type_without_clearing_the_gate_or_the_title(win):
    """``run_pca`` moves the plot-type combo itself, under ``_updating``. That is a bulk
    widget update, not a user changing analysis: it must not draw twice, and it must not
    blank a caption the user typed for the plot it is about to show."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_feature_selection(["dwell_s", "depth_pA"])
    w.set_title("figure 3b")

    assert w.run_pca() is not None
    assert w._plot_type_combo.currentText() == "PCA"
    assert w.style().title == "figure 3b", "a Run button blanked the user's title"
    assert not w._stale, "an explicit Run leaves nothing pending"


def test_run_clustering_is_a_run_gesture_and_clears_the_gate(shown):
    w = shown
    w.set_plot_type("Scatter")
    w.set_scatter_columns("dwell_s", "depth_pA")
    w.set_feature_selection(["dwell_s", "depth_pA"])
    _tick(w, 1, True)
    assert w._stale

    assert w.run_clustering() is not None
    assert not w._stale, "Run clustering draws, so it owns the gate it just satisfied"


def test_a_reopened_window_goes_back_behind_the_gate(win):
    """``reset_state`` is "everything needs to be blank", and the gate is part of that:
    a window that still counted as run would restyle live over a picture the user has
    not asked for yet."""
    w = win
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    assert w._plot_armed

    w.close()          # closeEvent files the results and resets
    assert not w._plot_armed and not w._stale
    assert w._legend_entries.count() == 0

    w.auto_plot = False
    w.set_data([_source("a", _events())])
    assert pp._PLOT_PROMPT in w.details_text()


def test_the_long_fit_combo_still_holds_its_full_text(win):
    """The combos were capped in WIDTH, not in content — a cap that truncated the
    selection would be a far worse bug than the scrollbar it removed."""
    w = win
    w.set_fit("exponential + gamma (bimodal)")
    assert w._fit_combo.currentText() == "exponential + gamma (bimodal)"
    assert w._fit_combo.sizeAdjustPolicy() == QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon


# ================================================================================================
# LAYOUT — the two measured bugs
# ================================================================================================

def _parts(w):
    splitter = w.centralWidget()
    scroll = splitter.widget(0)
    return splitter, scroll, scroll.widget()


def test_the_controls_column_is_never_capped_below_its_own_contents(win, qapp):
    """MEASURED BEFORE: contents minimumSizeHint 444 px against a maximumWidth of 420 —
    24 px narrower than the thing it contained — so the scroll area's viewport was
    400 px and a horizontal scrollbar of range 0–44 was on screen at rest, always."""
    w = win
    w.show()
    qapp.processEvents()
    splitter, scroll, controls = _parts(w)

    assert scroll.maximumWidth() >= 16777215, "the cap is what pinned the scrollbar"
    need = controls.minimumSizeHint().width()
    assert scroll.minimumWidth() >= need, (
        f"the column may not be narrower than its contents ({scroll.minimumWidth()} < {need})")
    assert scroll.viewport().width() >= need, "the viewport still cannot show a whole row"
    assert not scroll.horizontalScrollBar().isVisible()
    # ...and the content minimum itself came down, which is what made all of that fit:
    # the histogram page's Fit row was unstacked and the long combos capped.
    assert need <= 360, f"the controls got wide again ({need} px)"


def test_no_dead_gutter_between_the_controls_and_the_plot(win, qapp):
    """MEASURED BEFORE: the splitter allotted the column 446 px and the widget painted
    420 of them, leaving 26 px of window doing nothing at rest."""
    w = win
    w.show()
    qapp.processEvents()
    splitter, scroll, controls = _parts(w)
    allotted = splitter.sizes()[0]
    assert scroll.width() == allotted, (
        f"{allotted - scroll.width()} px of the splitter's allotment is unpainted")
    assert controls.width() == scroll.viewport().width()


def test_the_splitter_can_actually_be_dragged(win, qapp):
    """MEASURED BEFORE: every request, for any width, came back as 420."""
    w = win
    w.show()
    qapp.processEvents()
    splitter, scroll, _controls = _parts(w)
    total = sum(splitter.sizes())

    splitter.setSizes([700, total - 700])
    qapp.processEvents()
    wide = splitter.sizes()[0]
    assert wide > 600, f"the column still refuses to widen ({wide} px)"

    splitter.setSizes([scroll.minimumWidth(), total - scroll.minimumWidth()])
    qapp.processEvents()
    assert splitter.sizes()[0] == scroll.minimumWidth()
    assert splitter.sizes()[0] < wide, "...and it can be given back to the plot"
    assert not scroll.horizontalScrollBar().isVisible()


def test_the_plot_gets_the_room_the_window_gains(win, qapp):
    """The whole point of "needs more room to expand": widening the window must widen
    the PLOT, not split the gain with a controls column that has nothing to do with it.
    """
    w = win
    w.resize(1120, 720)
    w.show()
    qapp.processEvents()
    splitter, scroll, _c = _parts(w)
    before = splitter.sizes()

    w.resize(1520, 720)
    qapp.processEvents()
    after = splitter.sizes()
    assert after[1] - before[1] >= 380, f"the plot gained only {after[1] - before[1]} px"
    assert after[0] == before[0], "the controls column ate part of the new width"
