"""Templates for the windows other than Distributions, and the project's copy of every plot
preset and template (Plot Studio step 6). Synthetic data only.

F2: each of the five windows saves its analysis settings and appearance under a name -- never
its sources, span or results -- and recalls them as the user's own gesture: a draft where the
window has Generate or Apply fit, one redraw where it has none, and one appearance Undo step.
Delete asks first; Rename moves what was saved.

F1: Save Project writes a copy of this computer's presets and templates into the project;
opening it adds what this computer lacks, never over its own, and says so in the banner.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import voltage_dep
from nanoscope.core.figure_quantity_basis import read_quantity_bases
from nanoscope.core.plot_preset_state import (
    APPEARANCE_PRESETS, DISTRIBUTIONS_TEMPLATES, WINDOW_TEMPLATES, WORKSPACE_KEY,
)
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plot_preset_sync
from nanoscope.gui import plot_templates as pt
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plot_appearance import PlotAppearance, PlotAppearanceWidget


def _clear_stores() -> None:
    settings = QtCore.QSettings()
    for key in (PlotAppearanceWidget._PRESETS_KEY, pp.PlotsWindow._TPL_SETTINGS_KEY,
                *(pt._settings_key(kind) for kind in pt.WINDOW_KINDS)):
        settings.remove(key)
    settings.sync()


@pytest.fixture(autouse=True)
def _stores():
    """Every test starts and ends with no presets or templates: a leftover one would ride
    into the next test's saved project."""
    _clear_stores()
    yield
    _clear_stores()


def _events_source(key="measurements", label="Measured events") -> PlotSource:
    frame = pd.DataFrame({"event_id": [5, 6, 9, 11, 14, 17],
                          "dwell_s": [.001, .002, .003, .004, .005, .006],
                          "depth_pA": [10., 20., 30., 40., 50., 60.],
                          "amplitude_pA": [12., 25., 35., 42., 55., 64.],
                          "frac_depth": [.1, .2, .3, .4, .5, .6],
                          "ecd_pA_s": [.01, .04, .09, .16, .25, .36],
                          "snr": [5., 8., 10., 12., 15., 16.],
                          "accepted": [True, True, False, True, True, True]})
    return PlotSource(key, label, frame, kind="manual", doc_key=key, current=True)


def _summary(qtbot):
    from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
    window = DistributionSummaryWindow()
    qtbot.addWidget(window)
    window.set_sources([_events_source(), _events_source("other", "Other events")],
                       selected_keys=["measurements"])
    window.analyze()
    return window


def _comparison(qtbot):
    from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
    window = ComparisonFiguresWindow()
    qtbot.addWidget(window)
    window.set_sources([_events_source(), _events_source("other", "Other events")],
                       selected_keys=["measurements"])
    assert window.analyze()
    return window


def _vdep(qtbot):
    from nanoscope.gui.voltage_dep_panel import VDepWindow
    window = VDepWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(window)
    window._points = [voltage_dep.VDepPoint(
        v, -100 * v + 2, f"run {i}", baseline_fingerprint="baseline-a",
        analyzed_duration_s=10, voltage_source="measured", doc_id=str(i))
        for i, v in enumerate((-.2, -.5, -.8, -1.0))]
    window._series = voltage_dep.vdep_comparability(window._points, metric="capture_rate_hz")
    window._applied_fit = "linear"
    with QtCore.QSignalBlocker(window._fit_combo):
        window._fit_combo.setCurrentText("linear")
    window._redraw()
    return window


def _psd(qtbot):
    from nanoscope.gui.psd_panel import PSDWindow
    window = PSDWindow(Session(), defer_initial_refresh=True)
    qtbot.addWidget(window)
    return window


def _events(qtbot):
    from nanoscope.gui.event_plot_panel import EventPlotWindow
    window = EventPlotWindow(Session())
    qtbot.addWidget(window)
    return window


BUILDERS = {"distribution-summaries": _summary, "comparison-figures": _comparison,
            "voltage-dependence": _vdep, "noise-spectrum": _psd, "event-waveforms": _events}


def _appearance_history(window) -> tuple[int, int]:
    return window._appearance_widget._history.depth()


#: 2026-11-11: a page that presents units (Voltage dependence, Distribution summaries, Noise
#: spectrum) always carries the witness of the quantity each axis shows (8661d35, quantity
#: display). It is not look: a template keeps it, because the axis pins it keeps are numbers in
#: the display unit, and Reset figure keeps it. A preset takes it away.
WITNESS = "quantity_display"


def _look(appearance: dict) -> dict:
    """An appearance mapping without the witness of the quantity each axis shows."""
    return {key: value for key, value in appearance.items() if key != WITNESS}


def _bases(appearance: dict) -> dict[str, tuple[str, str]]:
    """The quantity each axis shows according to the witness, as ``{axis: (quantity key,
    display unit)}``; empty for a page that records none (Comparison figures, Event waveforms)."""
    return {axis: (basis.quantity.key, basis.quantity.display_unit)
            for axis, basis in read_quantity_bases(appearance.get(WITNESS)).items()}


# ================================================================================================
# F2: one shared control, in each window's Export section
# ================================================================================================

@pytest.mark.parametrize("kind", list(BUILDERS))
def test_each_window_offers_templates_in_its_export_section(qtbot, kind):
    window = BUILDERS[kind](qtbot)
    controls = window._templates
    assert isinstance(controls, pt.WindowTemplateControls)
    window._workflow.set_section("Export")
    window.show()
    assert controls.isVisibleTo(window), "the Export section shows the templates"
    window._workflow.set_section("Appearance")
    assert not controls.isVisibleTo(window), "and another section hides them"
    assert [b.text() for b in (controls.save_button, controls.rename_button,
                                controls.delete_button)] == [
        "Save as template…", "Rename", "Delete"]
    assert not controls.rename_button.isEnabled() and not controls.delete_button.isEnabled()


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_a_template_keeps_settings_and_appearance_never_sources(qtbot, kind):
    window = BUILDERS[kind](qtbot)
    window._appearance_widget.apply_appearance(PlotAppearance(title_size=17, grid=True))
    assert window._templates.save_template("Mine")
    (payload,) = pt.load_window_templates(kind).values()
    assert payload["format"] == pt.TEMPLATE_FORMAT and payload["window"] == kind
    saved = payload["appearance"]
    assert _look(saved) == {"title_size": 17.0, "grid": True}, "the look, and no source's look"
    witness = window.appearance_state().get(WITNESS)
    assert saved.get(WITNESS) == witness, "the quantity its pins are numbers of, as drawn"
    if witness is not None:
        assert set(witness) == {"version", "axes"} and witness["axes"]
        assert all(set(axis) == {"coordinate_frame", "extra", "quantity"}
                   for axis in witness["axes"].values()), "a quantity per axis, never a source"
    text = json.dumps({**payload, "appearance": _look(saved)})      # (the witness is keyed by view)
    if witness is not None:      # its values are read too, without the view-keyed axis names
        text += json.dumps(list(saved[WITNESS]["axes"].values()))
    for word in ("source_keys", "measurements", "start_s", "end_s", "target", "compare",
                 "snapshot", "series", "curves", "points\": [", "density_counts"):
        assert word not in text, f"a template carries {word!r}"
    assert window._templates.selected() == "Mine"
    assert window._templates.rename_button.isEnabled()


def test_summaries_recall_is_a_draft_that_waits_for_generate(qtbot, monkeypatch):
    window = _summary(qtbot)
    window._view_combo.setCurrentIndex(window._view_combo.findData("box"))
    window._metric_combo.setCurrentIndex(window._metric_combo.findData("depth_pA"))
    window._violin_points.setValue(512)
    window._accepted_only.setChecked(False)
    window._appearance_widget.apply_appearance(PlotAppearance(axis_size=15))
    assert window._templates.save_template("Box depth")
    window.analyze()                            # the committed figure is now the box one
    committed = json.dumps(window._snapshot, default=str)
    window._view_combo.setCurrentIndex(window._view_combo.findData("ecdf"))
    window._metric_combo.setCurrentIndex(window._metric_combo.findData("dwell_s"))
    window._violin_points.setValue(256)
    window._accepted_only.setChecked(True)
    window._appearance_widget.apply_appearance(PlotAppearance())
    window._source_list.item(1).setCheckState(QtCore.Qt.Checked)   # tick a second source
    sources = window.config()["source_keys"]
    window.analyze()
    committed = json.dumps(window._snapshot, default=str)
    steps = _appearance_history(window)[0]

    monkeypatch.setattr(window, "analyze", lambda: pytest.fail("a recall ran Generate"))
    changed: list = []
    window.workspaceChanged.connect(lambda: changed.append(True))
    assert window._templates.recall_template("Box depth")
    config = window.config()
    assert (config["kind"], config["metric"], config["violin_points"], config["accepted_only"]) == (
        "box", "depth_pA", 512, False)
    assert config["source_keys"] == sources == ["measurements", "other"], "sources stay"
    assert json.dumps(window._snapshot, default=str) == committed, "no number changed"
    assert window._stale and "Unapplied" in window._status.text()
    state = window.appearance_state()
    assert _look({key: value for key, value in state.items() if key != "source_appearance"}) == {
        "axis_size": 15.0}
    assert not state.get("source_appearance"), "a template never carries a source's look"
    assert _bases(state)["bottom"] == ("dwell_s", "ms"), "the figure drawn is still the dwell ECDF"
    assert _appearance_history(window)[0] == steps + 1, "the look is one Undo step"
    assert changed, "the project is told the draft changed"
    assert window._templates.selected() == "Box depth"


def test_comparison_recall_is_a_draft_and_a_damaged_template_moves_nothing(
        qtbot, monkeypatch):
    window = _comparison(qtbot)
    window._view_combo.setCurrentIndex(window._view_combo.findData("pairs"))
    window._bins.setValue(33)
    assert window._templates.save_template("Pairs")
    window._view_combo.setCurrentIndex(window._view_combo.findData("facets"))
    window._bins.setValue(20)
    window.analyze()
    committed = json.dumps(window._snapshot, default=str)
    monkeypatch.setattr(window, "analyze", lambda: pytest.fail("a recall ran Generate"))
    assert window._templates.recall_template("Pairs")
    assert window.config()["kind"] == "pairs" and window.config()["bins"] == 33
    assert window.config()["source_keys"] == ["measurements"]
    assert json.dumps(window._snapshot, default=str) == committed
    assert window._stale

    stored = pt.load_window_templates("comparison-figures")
    stored["Broken"] = {**stored["Pairs"], "settings": {**stored["Pairs"]["settings"],
                                                         "bins": 9999}}
    pt.store_window_templates("comparison-figures", stored)
    window._templates.refresh()
    before = window.config()
    reports: list[str] = []
    monkeypatch.setattr(window._templates, "_report", reports.append)
    assert not window._templates.recall_template("Broken")
    assert window.config() == before, "refused whole, not half-applied"
    assert reports and "“Broken” could not be recalled" in reports[0]


def test_voltage_dependence_recall_proposes_the_fit_and_waits_for_apply(qtbot, monkeypatch):
    window = _vdep(qtbot)
    with QtCore.QSignalBlocker(window._fit_combo):
        window._fit_combo.setCurrentText("exponential")
    window._log_y.setChecked(True)
    window._abs_v.setChecked(True)
    assert window._templates.save_template("Exp |V|")
    stored = pt.load_window_templates("voltage-dependence")["Exp |V|"]["settings"]
    assert stored["selected_fit"] == "exponential" and "applied_fit" not in stored
    with QtCore.QSignalBlocker(window._fit_combo):
        window._fit_combo.setCurrentText("linear")
    window._sync_fit_apply_btn()
    window._log_y.setChecked(False)
    window._abs_v.setChecked(False)
    monkeypatch.setattr(window, "refresh", lambda: pytest.fail("the metric did not change"))
    assert window._templates.recall_template("Exp |V|")
    assert window._fit_combo.currentText() == "exponential"
    assert window._applied_fit == "linear", "the drawn trend is still the applied one"
    assert window._fit_apply_btn.isEnabled() and window._fit_apply_btn.text() == "Apply fit ●"
    assert window._log_y.isChecked() and window._abs_v.isChecked()


def test_noise_spectrum_recall_recomputes_once_and_keeps_the_recording_and_span(
        qtbot, monkeypatch):
    window = _psd(qtbot)
    window._nperseg_combo.setCurrentText("8192")
    window._window_combo.setCurrentText("blackman")
    window._overlap_spin.setValue(75)
    window._asd_check.setChecked(True)
    window._spec_cmap_combo.setCurrentText("magma")
    window._bg_combo.setCurrentText("Light")
    assert window._templates.save_template("Welch 8k")
    settings = pt.load_window_templates("noise-spectrum")["Welch 8k"]["settings"]
    assert settings["nperseg"] == "8192" and settings["style"]["background"] == "light"
    window._nperseg_combo.setCurrentText("Auto")
    window._window_combo.setCurrentText("hann")
    window._overlap_spin.setValue(50)
    window._asd_check.setChecked(False)
    window._spec_cmap_combo.setCurrentText("viridis")
    window._bg_combo.setCurrentText("Dark")
    span = (window._start_spin.value(), window._end_spin.value())
    calls: list = []
    monkeypatch.setattr(window, "_recompute", lambda: calls.append(True))
    assert window._templates.recall_template("Welch 8k")
    assert calls == [True], "one recompute for the whole recall"
    assert (window._nperseg_combo.currentText(), window._window_combo.currentText(),
            window._overlap_spin.value(), window._asd_check.isChecked()) == (
        "8192", "blackman", 75, True)
    assert window._spec_cmap_combo.currentText() == "magma"
    assert window.style().background == "light" and window._bg_combo.currentText() == "Light"
    assert (window._start_spin.value(), window._end_spin.value()) == span


def test_event_waveforms_recall_redraws_each_changed_view_once(qtbot, monkeypatch):
    window = _events(qtbot)
    window._bins.setValue(77)
    window._logx.setChecked(False)
    window._cmap_combo.setCurrentText("magma")
    window._normalize.setChecked(False)
    window._align_combo.setCurrentText("center")
    window.set_view("Ensemble")
    assert window._templates.save_template("Fine bins")
    window._bins.setValue(40)
    window._logx.setChecked(True)
    window._cmap_combo.setCurrentText("viridis")
    window._normalize.setChecked(True)
    window.set_view("Density")
    calls: list = []
    monkeypatch.setattr(window, "_redraw_density", lambda: calls.append("density"))
    monkeypatch.setattr(window, "_redraw_ensemble", lambda: calls.append("ensemble"))
    assert window._templates.recall_template("Fine bins")
    assert calls == ["density", "ensemble"], "both views' settings moved: one redraw each"
    assert window._bins.value() == 77 and not window._logx.isChecked()
    assert window._cmap_combo.currentText() == "magma"
    assert not window._normalize.isChecked() and window._align_combo.isEnabled()
    assert window._align_combo.currentText() == "center"
    assert window.current_view() == "Ensemble"


def test_voltage_dependence_recall_of_another_metric_regathers_once(qtbot, monkeypatch):
    window = _vdep(qtbot)
    window.set_metric("median_depth")
    with QtCore.QSignalBlocker(window._fit_combo):
        window._fit_combo.setCurrentText("exponential")
    assert window._templates.save_template("Depth exp")
    window.set_metric("capture_rate_hz")
    with QtCore.QSignalBlocker(window._fit_combo):
        window._fit_combo.setCurrentText("linear")
    window._applied_fit = "linear"
    window._sync_fit_apply_btn()
    calls: list = []
    regather = window.refresh
    monkeypatch.setattr(window, "refresh", lambda: (calls.append(True), regather()))
    assert window._templates.recall_template("Depth exp")
    assert calls == [True], "a changed metric re-gathers once, as picking it does"
    assert window._metric() == "median_depth"
    assert window._fit_combo.currentText() == "exponential"
    assert window._applied_fit == "linear", "the recalled fit is a proposal"
    assert window._fit_apply_btn.isEnabled() and window._fit_apply_btn.text() == "Apply fit ●"


#: One value per window that the window's own check refuses (a template edited by hand, or
#: written by a build with other choices).
DAMAGE = {"noise-spectrum": ("window", "no such taper"),
          "event-waveforms": ("bins", 0),
          "voltage-dependence": ("metric", "no such metric"),
          "distribution-summaries": ("violin_points", 9999)}


@pytest.mark.parametrize("kind", list(DAMAGE))
def test_a_damaged_template_is_refused_whole_and_moves_nothing(qtbot, monkeypatch, kind):
    window = BUILDERS[kind](qtbot)
    window._appearance_widget.apply_appearance(PlotAppearance(title_size=17))
    assert window._templates.save_template("Good")
    good = pt.load_window_templates(kind)["Good"]
    key, bad = DAMAGE[kind]
    assert key in good["settings"]
    damaged = {**good, "settings": {**good["settings"], key: bad},
               "appearance": {"title_size": 9.0, "grid": True}}
    pt.store_window_templates(kind, {"Good": good, "Damaged": damaged})
    window._templates.refresh()
    window._appearance_widget.apply_appearance(PlotAppearance(title_size=15))
    before = window.template_settings()
    reports: list[str] = []
    monkeypatch.setattr(window._templates, "_report", reports.append)
    assert not window._templates.recall_template("Damaged")
    assert window.template_settings() == before, "neither a control nor the look moved"
    assert reports and "“Damaged” could not be recalled" in reports[0]


def _zoom(viewbox) -> list:
    return [[round(v, 6) for v in pair] for pair in viewbox.viewRange()]


def test_an_appearance_only_recall_keeps_the_zoom_and_computes_nothing(
        qtbot, tmp_path, monkeypatch):
    """The rule every appearance edit follows: no recompute, no re-gather, no zoom reset."""
    from .test_event_plot import _fed_window as fed_events
    from .test_psd import _fed_window as fed_psd

    psd, _doc_id = fed_psd(qtbot, tmp_path)
    events = fed_events(qtbot, tmp_path)
    vdep = _vdep(qtbot)
    targets = ((psd, lambda: psd.active_plot_item().getViewBox(), "_recompute"),
               (events, lambda: events._density_main.getViewBox(), "_redraw_density"),
               (vdep, lambda: vdep._plot.getViewBox(), "refresh"))
    for window, viewbox, _compute in targets:
        window.show()
        assert window._templates.save_template("Same settings")
        window._appearance_widget.apply_appearance(PlotAppearance(title_size=18))
        viewbox().setRange(xRange=(1, 2), yRange=(1, 2), padding=0)
    qtbot.wait(10)
    zooms = [_zoom(viewbox()) for _window, viewbox, _compute in targets]
    counts, spectrum = events._density_counts, psd._psd
    for window, _viewbox, compute in targets:
        monkeypatch.setattr(window, compute, lambda: pytest.fail("an appearance-only recall "
                                                                 "computed"))
    monkeypatch.setattr(events, "_redraw_ensemble", lambda: pytest.fail("ensemble redrawn"))
    monkeypatch.setattr(vdep, "_redraw", lambda: pytest.fail("points redrawn"))
    for window, _viewbox, _compute in targets:
        steps = _appearance_history(window)[0]
        drawn = window.appearance_state().get(WITNESS)
        assert window._templates.recall_template("Same settings")
        assert _look(window.appearance_state()) == {}, "the template's look arrived"
        assert window.appearance_state().get(WITNESS) == drawn, "and what is drawn is still what it was"
        assert _appearance_history(window)[0] == steps + 1
    qtbot.wait(10)
    assert [_zoom(viewbox()) for _window, viewbox, _compute in targets] == zooms
    assert events._density_counts is counts and psd._psd is spectrum, "no result replaced"


def test_an_identical_recall_keeps_a_restored_snapshot_whose_recording_is_gone(
        qtbot, tmp_path):
    """A project reopened without its recording shows the SAVED spectrum and count grid;
    only a deliberate edit may replace them. Recalling a template that matches the window
    is not one, so the next save still carries them."""
    from nanoscope.core.analysis_workspace import (
        capture_event_plots, capture_psd, restore_event_plots, restore_psd,
    )
    from nanoscope.gui.event_plot_panel import EventPlotWindow
    from nanoscope.gui.psd_panel import PSDWindow
    from .test_event_plot import _fed_window as fed_events
    from .test_psd import _fed_window as fed_psd

    live_psd, _doc_id = fed_psd(qtbot, tmp_path)
    psd = PSDWindow(Session(), defer_initial_refresh=True)     # no recording in this session
    qtbot.addWidget(psd)
    restore_psd(psd, capture_psd(live_psd))
    assert psd.has_data()
    assert psd._templates.save_template("Same")
    title = psd.style().title
    psd._title_edit.setText("Saved figure")      # a style edit the template takes back
    assert psd._templates.recall_template("Same")
    assert psd.has_data() and capture_psd(psd).payload.get("curves")
    assert psd.style().title == title

    live_events = fed_events(qtbot, tmp_path)
    events = EventPlotWindow(Session())
    qtbot.addWidget(events)
    restore_event_plots(events, capture_event_plots(live_events))
    saved = events.density_counts()
    assert saved is not None and saved.sum() > 0
    assert events._templates.save_template("Same")
    kept = events._cmap_combo.currentText()
    events._cmap_combo.setCurrentText("cividis" if kept != "cividis" else "magma")
    assert events._templates.recall_template("Same")   # takes the colormap back
    assert events._cmap_combo.currentText() == kept
    assert events.density_counts() is not None
    assert (events.density_counts() == saved).all()


def test_a_draft_recall_that_changes_nothing_does_not_mark_the_figure_unapplied(qtbot):
    for window in (_summary(qtbot), _comparison(qtbot)):
        assert window._templates.save_template("As generated")
        assert not window._stale
        assert window._templates.recall_template("As generated")
        assert not window._stale, "a hand edit marks it only when a control moves"
        assert "Unapplied" not in window._status.text()


def test_a_noise_spectrum_recall_without_a_recording_shows_the_view_it_names(qtbot):
    from nanoscope.gui.psd_panel import VIEWS
    window = _psd(qtbot)
    settings, appearance = window.template_settings()
    window.apply_template_settings(dict(settings, view="spectrogram"), appearance)
    index = [key for key, _label in VIEWS].index("spectrogram")
    assert window.current_view() == "spectrogram"
    assert window._view_stack.currentIndex() == index
    assert window._view_options.currentIndex() == index


def test_rename_moves_what_was_saved_and_delete_asks_first(qtbot, monkeypatch, blocked_modals):
    window = _summary(qtbot)
    window._view_combo.setCurrentIndex(window._view_combo.findData("violin"))
    controls = window._templates
    assert controls.save_template("Violin <b>x</b>")
    saved = pt.load_window_templates("distribution-summaries")["Violin <b>x</b>"]
    window._view_combo.setCurrentIndex(window._view_combo.findData("ecdf"))
    assert controls.rename_template("Violin <b>x</b>", "Violin")
    assert pt.load_window_templates("distribution-summaries") == {"Violin": saved}, (
        "the stored settings moved; what is on screen was not re-saved")
    assert not controls.save_template("Violin"), "an existing name is never silently replaced"

    asked: list = []
    guard = QtWidgets.QDialog.exec

    def cancel(box, *a, **k):
        asked.append(box)
        return guard(box, *a, **k)

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", cancel)
    controls._ask_delete()
    assert "Violin" in pt.load_window_templates("distribution-summaries"), "Cancel keeps it"
    (box,) = asked
    assert box.textFormat() == QtCore.Qt.TextFormat.PlainText and "“Violin”" in box.text()
    assert box.defaultButton().text() == "Cancel" and box.escapeButton().text() == "Cancel"

    def delete(box, *a, **k):
        next(b for b in box.buttons() if b.text() == "Delete template").click()
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", delete)
    controls._ask_delete()
    assert pt.load_window_templates("distribution-summaries") == {}
    assert controls.selected() is None


def test_a_template_from_a_newer_build_is_refused_by_number(qtbot, monkeypatch):
    window = _summary(qtbot)
    pt.store_window_templates("distribution-summaries", {
        "Future": {"format": pt.TEMPLATE_FORMAT, "version": 7,
                   "window": "distribution-summaries", "settings": {}, "appearance": {}},
        "Wrong window": pt.encode_window_template("noise-spectrum", {}, {})})
    window._templates.refresh()
    reports: list[str] = []
    monkeypatch.setattr(window._templates, "_report", reports.append)
    before = window.config()
    assert not window._templates.recall_template("Future")
    assert not window._templates.recall_template("Wrong window")
    assert "template version 7" in reports[0]
    assert "a template for Noise spectrum" in reports[1]
    assert window.config() == before


# ================================================================================================
# F1: the project carries a copy
# ================================================================================================

def _main(qtbot):
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow(recovery_enabled=False)
    qtbot.addWidget(window)
    return window


def _seed_local() -> dict:
    """One of each: an appearance preset, a Distributions template, a window template."""
    look = PlotAppearance(background="light", title_size=13).to_dict()
    settings = QtCore.QSettings()
    settings.setValue(PlotAppearanceWidget._PRESETS_KEY, json.dumps({"Paper": look}))
    settings.sync()
    dwell = pp.plot_config_to_json(pp.PlotsWindow._BUILTIN_TEMPLATES[
        "Dwell-time histogram (log-x, exponential)"])
    pp.PlotsWindow._store_templates({"Dwell": dwell})
    welch = pt.encode_window_template("noise-spectrum", {"nperseg": "8192"}, {"grid": True})
    pt.store_window_templates("noise-spectrum", {"Welch 8k": welch})
    return {"look": look, "dwell": dwell, "welch": welch}


def _open(qtbot, path):
    window = _main(qtbot)
    with qtbot.waitSignal(window.projectRestored, timeout=10_000):
        window._open_project(str(path))
    return window


def test_a_saved_project_carries_presets_and_another_computer_gets_them(qtbot, tmp_path):
    seeded = _seed_local()
    source = _main(qtbot)
    project = tmp_path / "carried.nanoscope"
    assert source._write_project(str(project))
    from nanoscope.core.project import load_project
    section = load_project(project).workspace.extra[WORKSPACE_KEY]
    assert section[APPEARANCE_PRESETS] == {"Paper": seeded["look"]}
    assert section[DISTRIBUTIONS_TEMPLATES] == {"Dwell": seeded["dwell"]}
    assert section[WINDOW_TEMPLATES] == {"noise-spectrum": {"Welch 8k": seeded["welch"]}}

    _clear_stores()                       # "another computer": nothing saved locally
    restored = _open(qtbot, project)
    assert PlotAppearanceWidget.user_preset_states() == {"Paper": seeded["look"]}
    assert pp.PlotsWindow._load_templates() == {"Dwell": seeded["dwell"]}
    assert pt.load_window_templates("noise-spectrum") == {"Welch 8k": seeded["welch"]}
    banner = restored._info_bar._label.text()
    assert restored._info_bar.isVisibleTo(restored)
    assert ("3 plot preset(s) or template(s) this computer did not have" in banner
            and "“Paper”" in banner and "“Dwell”" in banner and "“Welch 8k”" in banner)
    assert not restored._dirty, "adding the project's copies is not a project edit"


def test_opening_never_replaces_a_local_entry(qtbot, tmp_path):
    seeded = _seed_local()
    project = tmp_path / "theirs.nanoscope"
    assert _main(qtbot)._write_project(str(project))
    mine = PlotAppearance(background="dark", grid=True).to_dict()
    settings = QtCore.QSettings()
    settings.setValue(PlotAppearanceWidget._PRESETS_KEY, json.dumps({"Paper": mine}))
    settings.sync()

    restored = _open(qtbot, project)
    assert PlotAppearanceWidget.user_preset_states() == {
        "Paper": mine, "Paper (from project)": seeded["look"]}
    banner = restored._info_bar._label.text()
    assert "“Paper” → “Paper (from project)”" in banner and "yours is unchanged" in banner
    assert "“Dwell”" not in banner, "an identical template is not announced as added"

    _open(qtbot, project)                 # opening again adds nothing more
    assert set(PlotAppearanceWidget.user_preset_states()) == {"Paper", "Paper (from project)"}


def test_without_presets_a_project_gets_no_section(qtbot, tmp_path):
    project = tmp_path / "plain.nanoscope"
    assert _main(qtbot)._write_project(str(project))
    from nanoscope.core.project import load_project
    assert WORKSPACE_KEY not in load_project(project).workspace.extra


def test_what_this_build_cannot_read_stays_in_the_project(qtbot, tmp_path):
    from nanoscope.core.project import load_project, save_project
    _seed_local()
    project = tmp_path / "newer.nanoscope"
    assert _main(qtbot)._write_project(str(project))
    state = load_project(project)
    section = state.workspace.extra[WORKSPACE_KEY]
    future = {"format": pt.TEMPLATE_FORMAT, "version": 9, "window": "noise-spectrum",
              "settings": {}, "appearance": {}}
    section[WINDOW_TEMPLATES]["noise-spectrum"]["From a newer build"] = future
    section[WINDOW_TEMPLATES]["a-future-window"] = {"Theirs": {"anything": 1}}
    section["figure_templates"] = {"A newer family": {}}
    save_project(project, state)
    _clear_stores()

    restored = _open(qtbot, project)
    assert "From a newer build" not in pt.load_window_templates("noise-spectrum")
    assert "could not be read by this version" in restored._info_bar._label.text()
    resaved = tmp_path / "resaved.nanoscope"
    assert restored._write_project(str(resaved))
    carried = load_project(resaved).workspace.extra[WORKSPACE_KEY]
    assert carried[WINDOW_TEMPLATES]["noise-spectrum"]["From a newer build"] == future
    assert carried[WINDOW_TEMPLATES]["a-future-window"] == {"Theirs": {"anything": 1}}
    assert carried["figure_templates"] == {"A newer family": {}}
    assert "Welch 8k" in carried[WINDOW_TEMPLATES]["noise-spectrum"]


def test_a_damaged_carried_entry_is_named_and_the_project_still_opens(qtbot, tmp_path):
    """A manifest is read with plain json.loads, so NaN and 1e400 arrive as floats. They
    must make an entry unreadable, never abort the restore half-way."""
    from nanoscope.core.project import load_project, save_project
    _seed_local()
    project = tmp_path / "damaged.nanoscope"
    assert _main(qtbot)._write_project(str(project))
    state = load_project(project)
    section = state.workspace.extra[WORKSPACE_KEY]
    section[WINDOW_TEMPLATES]["noise-spectrum"]["Welch 8k"]["settings"]["rms_lo_hz"] = (
        float("nan"))
    section[DISTRIBUTIONS_TEMPLATES]["Dwell"]["version"] = json.loads("1e400")
    save_project(project, state)
    _clear_stores()

    restored = _open(qtbot, project)
    assert not restored._restoring, "the restore finished"
    assert pt.load_window_templates("noise-spectrum") == {}
    assert pp.PlotsWindow._load_templates() == {}
    assert "Paper" in PlotAppearanceWidget.user_preset_states(), "the readable one came in"
    banner = restored._info_bar._label.text()
    assert "2 of the project's plot presets or templates could not be read" in banner
    assert "“Welch 8k”" in banner and "“Dwell”" in banner


def test_a_carried_template_under_a_shipped_name_comes_in_pickable(qtbot):
    shipped = "Publication figure (light)"
    other = pp.plot_config_to_json(pp.PlotsWindow._BUILTIN_TEMPLATES[
        "Dwell-time histogram (log-x, exponential)"])
    same = pp.plot_config_to_json(pp.PlotsWindow._BUILTIN_TEMPLATES[shipped])
    notes = plot_preset_sync.install_project_presets({WORKSPACE_KEY: {
        DISTRIBUTIONS_TEMPLATES: {shipped: other}}})
    assert pp.PlotsWindow._load_templates() == {f"{shipped} (from project)": other}
    assert f"“{shipped}” → “{shipped} (from project)”" in " ".join(notes)
    plots = pp.PlotsWindow(Session())
    qtbot.addWidget(plots)
    assert plots._template_combo.findText(f"{shipped} (from project)") > 0
    _clear_stores()
    assert plot_preset_sync.install_project_presets({WORKSPACE_KEY: {
        DISTRIBUTIONS_TEMPLATES: {shipped: same}}}) == ()
    assert pp.PlotsWindow._load_templates() == {}, "a copy of a shipped one is not added"


def test_an_open_window_lists_what_a_project_added(qtbot):
    window = _summary(qtbot)
    look = PlotAppearanceWidget()
    qtbot.addWidget(look)                        # a second open page's appearance controls
    plots = pp.PlotsWindow(Session())
    qtbot.addWidget(plots)
    plots._select_template("Publication figure (light)")
    gone = PlotAppearanceWidget()                # a lister destroyed before the project opens
    gone.deleteLater()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    scatter = pp.plot_config_to_json(pp.PlotsWindow._BUILTIN_TEMPLATES[
        "Dwell vs depth scatter (log-x)"])
    extra = {WORKSPACE_KEY: {
        APPEARANCE_PRESETS: {"Theirs": {"grid": True}},
        DISTRIBUTIONS_TEMPLATES: {"Their scatter": scatter},
        WINDOW_TEMPLATES: {"distribution-summaries": {
            "Box": pt.encode_window_template("distribution-summaries", {"kind": "box"}, {})}}}}
    notes = plot_preset_sync.install_project_presets(extra)
    assert notes and "3 plot preset(s) or template(s)" in notes[0]
    assert window._templates.combo.findData("Box") > 0
    assert look.user_presets.findData("Theirs") > 0
    assert plots._template_combo.findText("Their scatter") > 0
    assert plots._template_combo.currentText() == "Publication figure (light)", (
        "adding other templates keeps the named selection")


# ================================================================================================
# the same recalls over live synthetic data, nothing replaced by a stand-in
# ================================================================================================

def test_a_live_noise_spectrum_recomputes_with_the_recalled_welch_settings(qtbot, tmp_path):
    from .test_psd import _fed_window
    window, _doc_id = _fed_window(qtbot, tmp_path)
    window._nperseg_combo.setCurrentText("8192")
    window._window_combo.setCurrentText("blackman")
    assert window._templates.save_template("Welch 8k")
    window._nperseg_combo.setCurrentText("1024")
    assert window._nperseg_used == 1024
    assert window._templates.recall_template("Welch 8k")
    assert window._nperseg_used == 8192 and window.has_data()
    assert window._curves[0].nperseg == 8192


def test_live_event_waveforms_rebin_the_bound_events(qtbot, tmp_path):
    from .test_event_plot import _fed_window
    window = _fed_window(qtbot, tmp_path)
    window._bins.setValue(12)
    assert window._templates.save_template("Coarse")
    window._bins.setValue(40)
    assert window.density_counts().shape == (40, 40)
    assert window._templates.recall_template("Coarse")
    assert window.density_counts().shape == (12, 12)
    assert window.has_data()


# ================================================================================================
# the Event waveforms and Noise spectrum settings of Plot Studio step 5 (track E) in a template
# ================================================================================================

def test_an_event_waveforms_template_keeps_extremum_alignment_band_and_subset(
        qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import event_shape
    from nanoscope.gui import plot_interaction as pi
    from nanoscope.gui.event_plot_panel import ENSEMBLE
    from .test_event_plot import _fed_window
    window = _fed_window(qtbot, tmp_path)
    window._eventset.provenance["polarity"] = -1
    window.set_view(ENSEMBLE)
    window.set_normalize_time(False)
    window.set_align("extremum")
    window.set_band("iqr")
    settings, _appearance = window.template_settings()
    assert (settings["align"], settings["ensemble_band"],
            settings["ensemble_subset"]) == ("extremum", "iqr", "Accepted only")
    assert window._templates.save_template("Deepest")
    window.set_align("start")
    window.set_band("")
    aligns = []
    real = event_shape.event_ensemble
    monkeypatch.setattr(event_shape, "event_ensemble",
                        lambda *a, **k: aligns.append(k.get("align")) or real(*a, **k))
    assert window._templates.recall_template("Deepest")
    assert aligns == ["extremum"], "one ensemble run, on the recalled alignment"
    assert window._band_kind() == "iqr" and window.ensemble_band() is not None
    # A band-only difference re-shades the traces on screen: no kernel, zoom and history kept.
    view = window._ensemble_plot.getViewBox()
    nav = pi.navigation_for(window._ensemble_plot)
    (x0, x1), _y = view.viewRange()
    view.setRange(xRange=(x0, 0.5 * (x0 + x1)), padding=0)
    view.sigRangeChangedManually.emit([True, True])
    qtbot.wait(pi.SETTLE_MS + 100)
    window.set_band("")
    zoom, kept = _zoom(view), nav.history()
    monkeypatch.setattr(event_shape, "event_ensemble",
                        lambda *a, **k: pytest.fail("a band-only recall ran the ensemble"))
    assert window._templates.recall_template("Deepest")
    qtbot.wait(pi.SETTLE_MS + 100)
    assert window._band_kind() == "iqr" and len(window._band_items) == 3
    assert _zoom(view) == zoom and nav.history() == kept


def test_a_noise_spectrum_template_keeps_the_spectrogram_levels_and_frequency_axis(
        qtbot, tmp_path, monkeypatch):
    from nanoscope.analysis import psd as psd_mod
    from nanoscope.gui.psd_panel import VIEW_SPECTROGRAM
    from .test_psd import _fed_window
    window, _doc_id = _fed_window(qtbot, tmp_path, n=65_536)
    window.set_view(VIEW_SPECTROGRAM)
    window._spec_levels_combo.setCurrentIndex(window._spec_levels_combo.findData("manual"))
    window._spec_level_lo.setValue(-4.5)
    window._spec_level_hi.setValue(-2.0)
    window._spec_logf_check.setChecked(True)
    assert window._templates.save_template("Log spec")
    window._spec_logf_check.setChecked(False)
    window._spec_levels_combo.setCurrentIndex(window._spec_levels_combo.findData("auto"))
    monkeypatch.setattr(psd_mod, "spectrogram_psd",
                        lambda *a, **k: pytest.fail("presentation recomputed the spectrogram"))
    assert window._templates.recall_template("Log spec")
    assert window._spec_levels_mode() == "manual"
    assert tuple(window._spec_image.getLevels()) == (-4.5, -2.0)
    assert window._spec_log_edges is not None
    # A damaged pair is refused whole, before any control moves.
    payload = pt.load_window_templates("noise-spectrum")["Log spec"]
    payload["settings"]["spectrogram_levels_mode"] = "sometimes"
    pt.store_window_templates("noise-spectrum", {"Bad": payload})
    window._templates.refresh()
    monkeypatch.setattr(window._templates, "_report", lambda _text: None)
    assert not window._templates.recall_template("Bad")
    assert window._spec_levels_mode() == "manual"
