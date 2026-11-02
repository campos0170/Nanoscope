"""Distributions after a project is reopened: the figure that was saved (plotux, round 5).

Opening a project draws a histogram fit from the SAVED ``FitResult``
(``PlotsWindow._overlay_workspace_fit``), never by refitting. That path built the fit label
from the one-line summary, so a reopened figure lost every parameter but the headline, BIC
and KS; a typed template's ``{n}`` read "—"; and the legend's fit row got its numbers back
("fit: τ = 2.304 ms (n = 120)" where the saved figure read "fit (exponential)"). It now
uses the live draw's helpers, and the saved KS numbers (``meta['gof']``) are read, not
recomputed.

Real routes throughout: a MainWindow, synthetic CSV recordings with synthetic event tables,
the real ``_write_project``, a FRESH MainWindow and ``_open_project`` until
``projectRestored``. Offscreen.
"""
from __future__ import annotations

import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from nanoscope.analysis.events import SCHEMA, EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.project import DocumentRef, ProjectState
from nanoscope.gui import figure_text
from nanoscope.gui.figure_text import TextStyle, figure_text_for
from nanoscope.gui.main_window import MainWindow

LEFT = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    figure_text.GATE.reset()
    yield
    figure_text.GATE.reset()


def _csv(path, n: int = 20000) -> None:
    pd.DataFrame({"time": np.arange(n) / 10000.0,
                  "current": np.full(n, 100.0, dtype=np.float32)}).to_csv(path, index=False)


def _install_events(window: MainWindow, doc_id: str, seed: int = 0, n: int = 120) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        start = 100 + i * 150
        dwell = float(rng.exponential(0.002) + 1e-4)
        depth = float(rng.normal(40, 5))
        rows.append({"event_id": i, "recording": "r.csv", "pore_id": "P",
                     "start_idx": start, "end_idx": start + max(2, int(dwell * 10000)),
                     "t_start_s": start / 10000.0, "dwell_s": dwell, "baseline_pA": 100.0,
                     "depth_pA": depth, "frac_depth": depth / 100.0, "ecd_pA_s": depth * dwell,
                     "max_dev_pA": depth, "mean_dev_pA": depth, "rise_s": 1e-4, "fall_s": 1e-4,
                     "snr": 10.0, "n_levels": 1, "filter_limited": False, "accepted": True,
                     "flag": ""})
    frame = pd.concat([empty_df(), pd.DataFrame(rows)], ignore_index=True)
    frame = frame.astype({name: spec[0] for name, spec in SCHEMA.items()})
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=tuple(window._session.calibration(doc_id, 0).fingerprint()),
        payload=EventSet(frame, {})))
    window._on_selection(doc_id, 0)


def _main(qtbot) -> MainWindow:
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    window.resize(1200, 900)
    window.show()
    return window


def _recordings(qtbot, tmp_path, names=("rec",)):
    """A MainWindow holding one synthetic recording per name, each with its own events,
    and its Distributions window open on them."""
    paths = []
    for name in names:
        path = tmp_path / f"{name}.csv"
        _csv(path)
        paths.append(path)
    window = _main(qtbot)
    with qtbot.waitSignal(window.projectRestored, timeout=20000):
        window._begin_restore(ProjectState(documents=[DocumentRef(str(p)) for p in paths]))
    for seed, document in enumerate(window._session.documents()):
        _install_events(window, document.doc_id, seed=seed)
    window._open_plots()
    plots = window._plots_window
    if len(names) > 1:
        plots.set_sources(list(plots._sources))
    return window, plots


def _settle() -> None:
    for _ in range(5):
        QtWidgets.QApplication.processEvents()


def _reopen(qtbot, window: MainWindow, tmp_path):
    project = tmp_path / "figure.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False
    again = _main(qtbot)
    with qtbot.waitSignal(again.projectRestored, timeout=20000):
        again._open_project(str(project))
    again._open_plots()
    _settle()
    return again, again._plots_window


def _figure(plots) -> dict:
    legend = plots._plot.legend
    return {"label": plots._fit_label_item.plain_text() if plots._fit_label_item.isVisible()
            else "(hidden)",
            "legend": [label.text for _sample, label in legend.items] if legend else []}


@pytest.mark.parametrize("kind,col,shown", [
    ("exponential", "dwell_s", ("τ = ", " ms")),
    ("inverse-gaussian", "depth_pA", ("μ = ", "λ = ")),
])
def test_a_reopened_fit_keeps_every_number_and_a_number_free_legend_row(
        qapp, qtbot, tmp_path, kind, col, shown):
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column(col)
    plots.set_fit(kind)
    for name in ("bic", "ks"):               # the fit label's Numbers row: BIC and KS
        plots._appearance_widget.fit_label_fields[name].click()
    _settle()
    before = _figure(plots)
    assert all(part in before["label"] for part in (*shown, "n = 120", "BIC = ", "KS = ")), before
    assert any(row.startswith("fit (") for row in before["legend"]), before

    again, reopened = _reopen(qtbot, window, tmp_path)
    assert _figure(reopened) == before
    # The saved figure, not a re-derived one: nothing was refitted to draw it.
    assert reopened.fit_results() and "restored snapshot; not recomputed" in reopened.details_text()
    again._dirty = False


@pytest.mark.parametrize("names,kind,col,number", [
    (("rec",), "inverse-gaussian", "depth_pA", "λ = "),
    (("apo", "holo"), "exponential", "dwell_s", "τ = "),
])
def test_in_legend_instead_keeps_the_numbers_in_the_fit_row_through_a_redraw_and_a_reopen(
        qapp, qtbot, tmp_path, names, kind, col, number):
    """A fit's legend row was added before its readout was recorded, so with In legend
    instead every draw but a restyle built the row with no numbers, and the fit label was
    hidden because the row "carries" them: a Bins change, Plot, or a reopened project
    showed the fit with no number anywhere on the figure."""
    window, plots = _recordings(qtbot, tmp_path, names=names)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column(col)
    plots.set_fit(kind)
    _settle()
    plots._appearance_widget.fit_label_in_legend.click()
    _settle()

    def fit_rows(figure):
        rows = [row for row in figure["legend"] if "fit (" in row]
        assert len(rows) == len(names), figure
        return rows

    before = _figure(plots)
    assert before["label"] == "(hidden)"
    assert all("<br>" in row and number in row and "n = 120" in row
               for row in fit_rows(before)), before
    plots.set_bins(17)                      # a live redraw
    _settle()
    assert _figure(plots) == before
    plots._plot_btn.click()
    _settle()
    assert _figure(plots) == before

    again, reopened = _reopen(qtbot, window, tmp_path)
    assert _figure(reopened) == before
    again._dirty = False


def test_a_typed_fit_label_template_keeps_its_numbers_through_a_reopen(qapp, qtbot, tmp_path):
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    _settle()
    controller = figure_text_for(plots._plot)
    label = plots._fit_label_item
    view = plots._plot_widget
    at = view.viewport().mapTo(view.window(), view.mapFromScene(label.sceneBoundingRect().center()))
    QTest.mouseDClick(view.window().windowHandle(), LEFT, NOMOD, at)
    _settle()
    assert controller.editor is not None, "a double-click on the fit label opened nothing"
    controller.editor.selectAll()
    QTest.keyClicks(controller.editor, "Fit: {equation} ({n})")
    QTest.keyClick(controller.editor, QtCore.Qt.Key.Key_Return)
    _settle()
    before = _figure(plots)
    assert before["label"].startswith("Fit: τ = ") and before["label"].endswith("(n = 120)"), before

    again, reopened = _reopen(qtbot, window, tmp_path)
    after = _figure(reopened)
    assert after == before
    assert "—" not in after["label"], after
    again._dirty = False


def test_an_overlay_reopens_with_named_rows_and_one_label_line_per_source(qapp, qtbot, tmp_path):
    window, plots = _recordings(qtbot, tmp_path, names=("apo", "holo"))
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    _settle()
    before = _figure(plots)
    fit_rows = [row for row in before["legend"] if " fit (" in row]
    assert len(fit_rows) == 2 and not any("=" in row for row in fit_rows), before
    assert len(plots._fit_label_item.lines()) == 2, before

    again, reopened = _reopen(qtbot, window, tmp_path)
    assert _figure(reopened) == before
    assert len(reopened._fit_label_item.lines()) == 2
    again._dirty = False


def test_a_reopened_scatter_fit_is_pending_and_the_plot_button_says_so(qapp, qtbot, tmp_path):
    """Opening a project never runs a regression, so the saved scatter's line, its fit
    label and its legend rows are not on the reopened picture. The KDE overlay's rule:
    the gate's ● and its banner say the controls ask for more than is drawn."""
    window, plots = _recordings(qtbot, tmp_path, names=("apo", "holo"))
    plots.set_plot_type("Scatter")
    plots.set_scatter_columns("dwell_s", "depth_pA")
    plots.set_scatter_fit("linear")
    _settle()
    before = _figure(plots)
    assert plots._plot_btn.text() == "Plot" and "y = " in before["label"], before

    again, reopened = _reopen(qtbot, window, tmp_path)
    assert _figure(reopened)["label"] == "(hidden)"
    assert reopened._plot_btn.text() == "Plot ●"
    assert "scatter fit" in reopened._plot_btn.toolTip()
    reopened._plot_btn.click()
    _settle()
    assert _figure(reopened) == before
    assert reopened._plot_btn.text() == "Plot"
    again._dirty = False


def test_a_reopened_fit_curve_takes_a_new_fit_colour(qapp, qtbot, tmp_path):
    """The restored curve had no presentation role, so the restyle path skipped it: a
    Fit colour picked after a reopen recoloured the label but not the curve."""
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    _settle()
    again, reopened = _reopen(qtbot, window, tmp_path)
    widget = reopened._appearance_widget
    widget.apply_appearance(replace(widget.state(), fit_color="#009e73", line_width=3.0))
    _settle()
    pen = reopened._fit_item.opts["pen"]
    assert pen.color().name() == "#009e73"
    assert pen.widthF() == pytest.approx(4.0)        # the fit draws one px over the line width
    # ...and the one-source fit row is still unnamed after the restyle rebuilt the legend.
    assert "fit (exponential)" in _figure(reopened)["legend"]
    again._dirty = False


def test_a_reopened_fit_the_ks_test_rejected_still_says_so(qapp, qtbot, tmp_path):
    """The verdict is the one saved with the fit (``meta['gof']``), not recomputed: Details
    carries the rejection and the KS sentence after a reopen as it did before."""
    window, plots = _recordings(qtbot, tmp_path)
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("depth_pA")      # a normal sample: an exponential misfits it
    plots.set_fit("exponential")
    _settle()
    details = plots.details_text()
    assert "KS REJECTED" in details, details
    verdict = next(line for line in details.splitlines() if line.startswith("KS D = "))

    again, reopened = _reopen(qtbot, window, tmp_path)
    after = reopened.details_text()
    assert "KS REJECTED" in after and verdict in after, after
    assert _figure(reopened)["label"] == _figure(plots)["label"]
    again._dirty = False


def test_legend_row_and_detection_limit_styles_survive_a_reopen(qapp, qtbot, tmp_path):
    window, plots = _recordings(qtbot, tmp_path, names=("apo", "holo"))
    plots.set_plot_type("Histogram")
    plots.set_histogram_column("dwell_s")
    plots.set_fit("exponential")
    _settle()
    host = figure_text_for(plots._plot).host
    targets = host.text_targets()
    row = next(key for key in targets if key.startswith("legend.entry.fit:"))
    host.apply_text(row, replace(targets[row].style, bold=True, color="#d55e00"))
    host.apply_text("annotation.limit", TextStyle(text="detector floor", italic=True))
    _settle()
    styled = [label.text for _s, label in plots._plot.legend.items if "font-weight: bold" in label.text]
    assert len(styled) == 1 and "#d55e00" in styled[0], styled
    assert plots._limit_line.label.toPlainText() == "detector floor"

    again, reopened = _reopen(qtbot, window, tmp_path)
    rows = [label.text for _s, label in reopened._plot.legend.items]
    assert rows == [label.text for _s, label in plots._plot.legend.items], rows
    text = reopened._limit_line.label
    assert text.toPlainText() == "detector floor" and "italic" in text.toHtml()
    again._dirty = False
