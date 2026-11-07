"""The plot decisions after the integrated review (fixes3, group plots).

* **The STALE look keeps the user's view, and looks only at the drawn columns.** The debounced
  look after a Baseline-form edit was a full draw, which auto-fits: a mouse zoom was lost on
  every knob burst, and a ``dwell_s`` plot (no mark to move) was redrawn on every burst, because
  the look keyed on the whole table's amber sentence.
* **A Plots fit over a typed Fit range carries its range** (``modules.FIT_RANGE_KEY``). A fit
  restored from a project was known ranged only by the boxes: emptied and closed without Return,
  the narrowed fit was filed as the column's. The report refuses a stored ranged fit too.
* **The FIT RANGE line says what the report will draw**: the saved whole-column fit when the
  report would take it, otherwise a fresh fit.
* **A composed figure keeps a blank title hidden** (" " hidden, "" none), as the window does,
  and a titled panel still lines up with an untitled one on shared axes.
* **A table's open cell editor keeps ⌘Z** (as on 3938d32): an Undo that resets the table took
  the editor away and left the keyboard focus nowhere. Esc, then ⌘Z, reaches Edit ▸ Undo.
* **fixes4: a cluster split's amplitude note is a mark** (its STALE line followed only the next
  draw once the look keyed on the drawn columns), and **a hidden title takes no room on screen**
  in any window that styles its plots with ``apply_plot_appearance`` (a 23 px band stayed).

Synthetic recordings only; every ``meta.path`` is under ``tmp_path``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis import modules
from nanoscope.core.analysis_workspace import PLOTS_ARTIFACT_ID
from nanoscope.gui import figure_composition as C
from nanoscope.gui import main_window as mw
from nanoscope.gui.figure_bundle import capture_figure_snapshot
from nanoscope.gui.key_sequences import undo_sequences
from nanoscope.gui.main_window import MainWindow

from .test_editing_dependencies import _headline, _open, _plots
from .test_manual_measurement_edits import _FIRST, _install, _place
from .test_owner_decisions_plots import REFIT, _count_redraws, _depth_fit, _export
from .test_plot_appearance_history import PAUSE_MS, _build, _iv_window
from .test_sweep10_keys import _key

AFRESH = "This fit is not kept for Generate Report figures, which fit the whole column afresh."
SAVED = ("This fit is not kept for Generate Report figures, which draw the whole-column fit "
         "already saved for this table.")


@pytest.fixture
def window(qapp, qtbot, monkeypatch, tmp_path):
    """Every export dialog is answered with ``tmp_path / 'figs.pdf'``."""
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: tmp_path / "figs.pdf")
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._controller.abort()


# ================================================================================================
# 1. the STALE look keeps the view, and moves only for a drawn column
# ================================================================================================

def _zoom(plots):
    """What a wheel zoom does to the ViewBox: a narrower x range, auto-range off."""
    vb = plots._plot.getViewBox()
    (lo, hi), _y = vb.viewRange()
    vb.setRange(xRange=(lo + 0.3 * (hi - lo), lo + 0.5 * (hi - lo)), padding=0)
    QtWidgets.QApplication.processEvents()
    return vb, np.asarray(vb.viewRange())


def test_the_stale_mark_keeps_the_users_zoom(window, tmp_path, qtbot):
    _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("depth_pA")
    QtWidgets.QApplication.processEvents()
    vb, zoomed = _zoom(plots)

    rough = window._analysis_panel.baseline_panel.rough_s
    rough.setValue(2.0)
    qtbot.waitUntil(lambda: "⚠ STALE: depth_pA" in _headline(plots), timeout=3000)
    assert np.allclose(vb.viewRange(), zoomed), "the STALE look threw the zoom away"
    rough.setValue(2.5)                 # a further burst: the sentence moves, the view does not
    qtbot.wait(450)
    assert np.allclose(vb.viewRange(), zoomed)


def test_a_baseline_edit_under_a_dwell_plot_redraws_nothing(window, tmp_path, qtbot,
                                                             monkeypatch):
    _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("dwell_s")
    QtWidgets.QApplication.processEvents()
    details = plots.details_text()
    vb, zoomed = _zoom(plots)
    drawn = _count_redraws(monkeypatch, plots)
    judged: list[int] = []
    window._analysis_panel.stalenessChanged.connect(lambda *_: judged.append(1))

    rough = window._analysis_panel.baseline_panel.rough_s
    for value in (1.5, 2.0, 2.5):                       # three bursts, a look after each
        rough.setValue(value)
        qtbot.wait(450)
    assert judged and window._analysis_panel._stale.kind == "baseline", "precondition"
    assert not drawn, f"{len(drawn)} redraws, and no mark of a dwell_s plot moved"
    assert "STALE" not in _headline(plots) and plots.details_text() == details
    assert np.allclose(vb.viewRange(), zoomed)


# ================================================================================================
# 2. a ranged fit carries its range: restored, it is still not filed; stored, it is refused
# ================================================================================================

def _ranged_depth_fit(w: MainWindow, fit_min: str = "0.26"):
    plots = _plots(w)
    plots.set_histogram_column("frac_depth")
    plots.set_fit("gaussian-mixture")
    plots._fit_lo.setText(fit_min)
    plots._changed()
    QtWidgets.QApplication.processEvents()
    return plots


def test_a_restored_ranged_fit_is_not_filed_when_its_box_is_emptied(window, tmp_path):
    doc = _open(window, tmp_path)
    plots = _ranged_depth_fit(window)
    n_rows = len(window._analysis_panel.current_detection().payload.df)

    owner = window._figure_workspace
    key = owner.id_for_page(plots)
    workspace = window._capture_analytical_workspace()
    owner.install_workspace(workspace)
    assert not owner.resident_ids()
    assert owner.activate(key)
    plots = owner.page(key)
    assert plots._fits and "restored snapshot; not recomputed" in plots.details_text()
    restored = next(iter(plots._fits.values()))
    assert int(restored.n) < n_rows

    plots._fit_lo.setText("")       # emptied, and the window closed before Return or Tab
    plots.close()
    QtWidgets.QApplication.processEvents()
    assert window._session.results.get(doc, 0, "fit.depth") is None, \
        "a restored fit of a narrowed sample was filed as the table's depth fit"
    # what told it: the range the fit carried through the project
    assert restored.meta.get(getattr(modules, "FIT_RANGE_KEY", "")) == [0.26, None]


def test_the_report_refuses_a_stored_fit_made_over_a_fit_range(window, tmp_path):
    """Whatever path stored it: here, put in the store as the window would have filed it."""
    doc = _open(window, tmp_path)
    plots = _ranged_depth_fit(window)
    src = plots.primary_source()
    fit = plots._fits[src.key]
    window._session.results.put(plots._as_result(
        "fit.depth", src, payload=fit, params={"column": "frac_depth"}, scalars={},
        warnings=()))
    plots.close()
    QtWidgets.QApplication.processEvents()
    assert window._session.results.get(doc, 0, "fit.depth").payload is fit
    said = _export(window)
    assert ("depth: saved fit was made over a typed Fit range, not the whole column — the "
            "figure was refit fresh at k=1") in said, said


# ================================================================================================
# 3. the FIT RANGE line says what the report will draw
# ================================================================================================

def _fit_range_line(details: str) -> str:
    return next(ln for ln in details.splitlines() if "FIT RANGE" in ln)


def test_the_fit_range_line_names_the_saved_fit_the_report_draws(window, tmp_path):
    doc = _open(window, tmp_path)
    _depth_fit(window)                                  # a whole-column fit, filed on close
    saved = window._session.results.get(doc, 0, "fit.depth")
    assert saved is not None, "precondition"

    line = _fit_range_line(_depth_fit(window, fit_min="0.26"))
    assert line.endswith(SAVED), line
    assert window._session.results.get(doc, 0, "fit.depth") is saved
    assert "depth: used your saved fit (k=1)" in _export(window)


def test_the_fit_range_line_says_afresh_when_the_report_would_refuse_the_saved_fit(window,
                                                                                  tmp_path):
    """CONTROL: a saved fit the report refuses (one of depth_pA, under fit.depth) is not
    drawn, so the report fits afresh, and the line says so."""
    doc = _open(window, tmp_path)
    plots = _plots(window)
    plots.set_histogram_column("depth_pA")
    plots.set_fit("gaussian-mixture")
    QtWidgets.QApplication.processEvents()
    plots._run_plot()
    plots.close()
    QtWidgets.QApplication.processEvents()
    assert window._session.results.get(doc, 0, "fit.depth").params["column"] == "depth_pA"

    line = _fit_range_line(_depth_fit(window, fit_min="0.26"))
    assert line.endswith(AFRESH), line
    said = _export(window)
    assert "depth: saved fit is of depth_pA" in said and "refit fresh at k=1" in said, said


# ================================================================================================
# 4. a composed figure keeps a blank title hidden, and still aligns
# ================================================================================================

def _figure(qtbot, title: str | None):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(400, 300)
    widget.show()
    widget.plot([1, 2, 3], [4, 5, 6])
    widget.setLabel("bottom", "Time", units="s")
    widget.setLabel("left", "Current", units="pA")
    if title is not None:
        widget.setTitle(title)
    qtbot.wait(10)
    return widget


def _model(plot) -> dict:
    payload = C.capture_cartesian_model(plot)
    assert payload is not None
    return json.loads(payload)


def _clone(qtbot, model):
    """The detached widget compose draws a shared-axes panel in (kept by the caller)."""
    clone = C._clone_model(model, model["canvas"], model["ranges"])
    qtbot.addWidget(clone)
    return clone


def _no_title_row(widget) -> bool:
    plot = widget.getPlotItem()
    return not plot.titleLabel.isVisible() and plot.layout.rowMaximumHeight(0) == 0


def test_a_hidden_title_has_no_row_in_the_composed_panel(qtbot):
    """The I-V window, its Title box a space: no row on screen, none in the clone."""
    w = _iv_window()
    qtbot.addWidget(w)
    w.show()
    qtbot.wait(20)
    QTest.keyClicks(w._appearance_widget.title, " ")
    qtbot.wait(PAUSE_MS)
    assert w.appearance_state().get("title") == " " and _no_title_row(w._plot_widget), \
        "precondition"
    # The I-V fit is in a fit label since 2026-11-01 (plotux G4), and a panel carrying one is
    # composed as a frozen picture; hidden, the panel is redrawn from its model as before.
    w._appearance_widget.fit_label_show.setChecked(False)
    qtbot.wait(PAUSE_MS)
    clone = _clone(qtbot, _model(w._plot_widget.getPlotItem()))
    assert _no_title_row(clone)


def test_an_untitled_figure_has_no_title_row_in_the_composed_panel(qtbot):
    widget = _figure(qtbot, None)
    clone = _clone(qtbot, _model(widget.getPlotItem()))
    assert _no_title_row(clone)


def test_a_title_hidden_after_it_was_shown_is_not_replayed(qtbot):
    """pyqtgraph's setTitle(None) hides the row and keeps the old text."""
    widget = _figure(qtbot, "An old fit title")
    widget.setTitle(None)
    model = _model(widget.getPlotItem())
    assert model["title"]["text"] == ""
    clone = _clone(qtbot, model)
    assert _no_title_row(clone)


def test_a_titled_and_an_untitled_panel_still_align_on_shared_axes(qtbot, monkeypatch):
    widgets, snapshots = [], []
    for title in ("A title", None):
        widgets.append(_figure(qtbot, title))
        plot = widgets[-1].getPlotItem()
        snapshots.append(capture_figure_snapshot(
            plot, data=pd.DataFrame(),
            additional_resources={C.CARTESIAN_RESOURCE: C.capture_cartesian_model(plot)}))
    seen = []
    original = C.capture_figure_snapshot

    def spy(plot, **kwargs):
        seen.append((plot.titleLabel.isVisible(), plot.vb.sceneBoundingRect()))
        return original(plot, **kwargs)
    monkeypatch.setattr(C, "capture_figure_snapshot", spy)
    C.compose_figures(snapshots, ["A", "B"], 2, share_x=True, share_y=True)
    (titled, a), (untitled, b) = seen
    assert titled and not untitled
    assert a == b, f"plot rectangles differ: {a} vs {b}"


# ================================================================================================
# 5. a table's open cell editor keeps ⌘Z
# ================================================================================================

def test_an_open_cell_editor_keeps_cmd_z_and_esc_then_cmd_z_undoes(window, qtbot, tmp_path):
    doc_id, _channel = _install(window, tmp_path)
    window.show()
    window.raise_()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=3000)
    panel = window._analysis_panel.measure_panel
    panel.set_active(True)
    _place(window, doc_id, *_FIRST)
    assert panel.add_current() is True
    qtbot.wait(900)
    col = panel._model.column_index("label")
    panel._model.setData(panel._model.index(0, col), "A")      # an "Edit measurement" step
    qtbot.wait(900)

    window._reveal_measure_panel()
    QTest.qWait(100)
    view = panel._table_view
    index = panel._proxy.mapFromSource(panel._model.index(0, col))
    view.setFocus(Qt.MouseFocusReason)
    view.setCurrentIndex(index)
    view.edit(index)
    QTest.qWait(50)
    editor = QtWidgets.QApplication.focusWidget()
    assert isinstance(editor, QtWidgets.QLineEdit) and editor.text() == "A", "precondition"
    assert not editor.isUndoAvailable()

    _key(undo_sequences()[0])
    assert panel.measurements_df["label"].tolist() == ["A"], "⌘Z in the open editor undid the table"
    assert QtWidgets.QApplication.focusWidget() is editor, "the editor lost the keyboard"

    QTest.keyClick(editor, Qt.Key_Escape)
    QTest.qWait(20)
    assert view.hasFocus()
    _key(undo_sequences()[0])
    assert panel.measurements_df["label"].tolist() == [""], "Esc, then ⌘Z, did not reach Undo"


# ================================================================================================
# 6. a cluster split's amplitude note is a mark of its draw (fixes4)
# ================================================================================================

def _clustered_on_amplitude(w: MainWindow, tmp_path, kind: str, colour: str = "Cluster"):
    """dwell_s on the axes (PCA: of dwell_s and t_start_s), amplitude_pA among the
    clustering features."""
    _open(w, tmp_path)
    table = w._analysis_panel.current_detection().payload.df
    table["amplitude_pA"] = table["depth_pA"].astype(float)     # a synthetic amplitude column
    plots = _plots(w)
    plots.set_feature_selection(["dwell_s", "amplitude_pA"])
    assert plots.run_clustering() is not None
    if kind == "PCA":
        plots.set_feature_selection(["dwell_s", "t_start_s"])
        assert plots.run_pca() is not None
    else:
        plots.set_plot_type(kind)
    if kind == "Histogram":
        plots.set_histogram_column("dwell_s")
        plots.set_histogram_colour_by(colour)
    elif kind == "Scatter":
        plots.set_scatter_columns("dwell_s", "t_start_s")
        plots.set_colour_by(colour)
    QtWidgets.QApplication.processEvents()
    assert plots._plot_armed and "STALE" not in plots.details_text(), "precondition"
    return plots


@pytest.mark.parametrize("kind", ["Histogram", "Scatter", "PCA"])
def test_a_baseline_edit_marks_a_cluster_split_on_amplitude_live(window, tmp_path, qtbot,
                                                                 monkeypatch, kind):
    """The draw writes the clustering features' amplitude note, amplitude_pA's STALE line
    with it, so the look after a Baseline edit writes it too."""
    plots = _clustered_on_amplitude(window, tmp_path, kind)
    drawn = _count_redraws(monkeypatch, plots)
    window._analysis_panel.baseline_panel.rough_s.setValue(1.5)
    qtbot.wait(450)
    assert drawn == [1], f"{len(drawn)} redraws: the amplitude note waited for the next draw"
    assert "STALE: amplitude_pA" in plots.details_text()


def test_the_amplitude_note_is_no_mark_when_the_split_is_off(window, tmp_path, qtbot,
                                                             monkeypatch):
    """CONTROL: the same clustering, the histogram not split, so no note is drawn."""
    plots = _clustered_on_amplitude(window, tmp_path, "Histogram", colour="None")
    drawn = _count_redraws(monkeypatch, plots)
    window._analysis_panel.baseline_panel.rough_s.setValue(1.5)
    qtbot.wait(450)
    assert not drawn and "STALE" not in plots.details_text()


# ================================================================================================
# 7. a hidden title takes no room on screen either (fixes4)
# ================================================================================================

def _titled_plots(kind: str, w) -> list:
    """The plots each pyqtgraph window styles with ``apply_plot_appearance``."""
    return {"event": lambda: [w._density_main, w._ensemble_plot],
            "psd": lambda: [w._plot, w._fft_plot, w._spec_plot, w._hist_plot],
            "iv": lambda: [w._plot], "vdep": lambda: [w._plot], "summary": lambda: [w._plot],
            "subpop": lambda: [w._embed_view._plot.getPlotItem(),
                               w._wave_view._plot.getPlotItem()]}[kind]()


def _title_room(plot) -> tuple[float, float]:
    """The title label's height and how far below the item's top the ViewBox starts."""
    return (plot.titleLabel.geometry().height(),
            plot.vb.sceneBoundingRect().top() - plot.sceneBoundingRect().top())


@pytest.mark.parametrize("kind", ["event", "psd", "iv", "vdep", "summary", "subpop"])
def test_a_hidden_title_takes_no_room_on_screen_and_comes_back(qtbot, kind):
    """A title shown, then hidden with a space in the Title box: no band (the label's own
    minimum height used to win over the row's 0: 23 px); then shown again: its row back."""
    window = _build(kind)
    qtbot.addWidget(window)
    window.resize(900, 650)
    window.show()
    box = window._appearance_widget.title
    for text in ("Shown", " ", "Again"):
        box.selectAll()
        QTest.keyClicks(box, text)
        qtbot.wait(20)
        for plot in _titled_plots(kind, window):
            height, top = _title_room(plot)
            margin = plot.layout.getContentsMargins()[1]
            if text.strip():
                assert plot.titleLabel.isVisible() and height >= 30 and top >= margin + 30, \
                    (text, height, top)
            else:
                assert not plot.titleLabel.isVisible()
                assert (height, top) == (0, margin), f"a {height:g} px band over the plot"
