"""Each plot window's own Undo and Redo of its appearance (design §4, decision 9).

The widget alone first -- what one step is, the 700 ms pause, ``set_state``, the 200-step
ceiling, the buttons and the keys -- then the eight windows that host it, each driven
through its own widget, then the Plots window, which must keep exactly ONE stack, and its
template Delete, which now asks first (§3.8). Synthetic data only.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from nanoscope.analysis import iv, voltage_dep
from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.iv_panel import IVWindow
from nanoscope.gui.key_sequences import redo_sequences, undo_sequences
from nanoscope.gui.plot_appearance import (
    APPEARANCE_COALESCE_MS,
    APPEARANCE_HISTORY_DEPTH,
    PlotAppearance,
    PlotAppearanceWidget,
)
from nanoscope.gui.voltage_dep_panel import VDepWindow

#: The two shipped presets, by their place in the preset combo.
PUBLICATION = PlotAppearance(
    font_family="Arial", title_size=14, axis_size=12, tick_size=10, grid=False,
    background="light", data_color="#0072b2", fit_color="#d55e00", line_width=2,
    marker_size=8)
PRESENTATION = replace(PUBLICATION, grid=True, background="dark", marker_size=11)

#: Long enough for the coalescing timer to have fired.
PAUSE_MS = APPEARANCE_COALESCE_MS + 150


def _widget(qtbot, **kwargs) -> PlotAppearanceWidget:
    widget = PlotAppearanceWidget(**kwargs)
    qtbot.addWidget(widget)
    return widget


def _depth(widget) -> tuple[int, int]:
    return widget._history.depth()


def _press(target, sequence) -> None:
    """One press of ``sequence`` as the window receives it, so a QShortcut answers."""
    combo = QtGui.QKeySequence(sequence)[0]
    QtWidgets.QApplication.sendEvent(
        target, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, combo.key(), combo.keyboardModifiers()))


def _activate(qtbot, window) -> None:
    window.show()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=3000)


# ================================================================================================
# the widget alone
# ================================================================================================

def test_the_depth_and_the_pause_are_the_plots_windows():
    """Two windows of one app must not disagree about how far back Undo reaches or how
    long a pause ends a run. Restated in plot_appearance (plots_panel imports it), so held
    equal here."""
    assert APPEARANCE_HISTORY_DEPTH == pp._HISTORY_DEPTH == 200
    assert APPEARANCE_COALESCE_MS == pp._HISTORY_COALESCE_MS == 700


def test_an_edit_a_preset_a_saved_look_and_reset_are_one_step_each(qtbot):
    w = _widget(qtbot)
    w.title_size.setValue(1)
    w.title_size.setValue(14)            # "14" typed: two values, one edit
    qtbot.waitUntil(lambda: _depth(w) == (1, 0))
    assert w.save_user_preset("History test look", overwrite=True)
    try:
        assert _depth(w) == (1, 0), "saving a look changes no appearance, so it is no step"
        w.preset.activated.emit(1)
        assert _depth(w) == (2, 0) and w.state() == PUBLICATION
        assert w.apply_user_preset("History test look")
        assert _depth(w) == (3, 0) and w.state() == PlotAppearance(title_size=14)
    finally:
        w.delete_user_preset("History test look")
    w.reset_button.click()
    assert _depth(w) == (4, 0) and w.state() == PlotAppearance()


def test_undo_and_redo_emit_the_exact_states(qtbot):
    """An undo is not a restore: it emits ``changed`` with the state it lands on, exactly
    as the user's own edit would, so the host re-applies it the same way."""
    w = _widget(qtbot)
    w.title_size.setValue(14)
    w.preset.activated.emit(2)           # closes the edit's run as its own step
    w.reset_button.click()
    assert _depth(w) == (3, 0)
    seen: list = []
    w.changed.connect(seen.append)
    while w.undo():
        pass
    while w.redo():
        pass
    assert seen == [PRESENTATION, PlotAppearance(title_size=14), PlotAppearance(),
                    PlotAppearance(title_size=14), PRESENTATION, PlotAppearance()]
    assert not w.redo() and seen[-1] == PlotAppearance(), "the top of the stack emits nothing"


def test_a_run_of_edits_inside_the_pause_is_one_step(qtbot):
    w = _widget(qtbot)
    assert w._burst.isSingleShot() and w._burst.interval() == APPEARANCE_COALESCE_MS
    w.title_size.setValue(12)
    qtbot.keyClicks(w.title, "Fig 3")    # five keystrokes, five ``changed``
    w.grid.setCurrentIndex(w.grid.findData(True))
    assert _depth(w) == (0, 0), "nothing is committed while the run is open"
    assert w.undo_button.isEnabled(), "...but the open run is already something to undo"
    qtbot.waitUntil(lambda: _depth(w) == (1, 0))
    assert w.undo_button.toolTip() == "Undo: 3 appearance changes"
    seen: list = []
    w.changed.connect(seen.append)
    assert w.undo()
    assert seen == [PlotAppearance()], "one press takes the whole run back"

    # A PAUSE ends a run: two edits further apart than the pause are two steps.
    w.redo()
    w.tick_size.setValue(9)
    qtbot.wait(PAUSE_MS)
    w.tick_size.setValue(11)
    qtbot.wait(PAUSE_MS)
    assert _depth(w) == (3, 0)


def test_an_emptied_label_box_is_a_blank_label_and_one_step(qtbot):
    """An emptied box holds "", not the None it started from: Comparison and Study figures
    draw "" as NO title or axis label, the GUI's one way to an untitled figure (the host
    test below). So typing a character and deleting it is a real step, called "blank"."""
    w = _widget(qtbot)
    qtbot.keyClicks(w.title, "x")
    qtbot.keyClick(w.title, QtCore.Qt.Key_Backspace)
    qtbot.wait(PAUSE_MS)
    assert w.state() == PlotAppearance(title="") and _depth(w) == (1, 0)
    assert w.undo_button.toolTip() == "Undo: title automatic → blank"
    assert w.undo() and w.state() == PlotAppearance()


def test_set_state_adds_no_step_and_closes_a_run(qtbot):
    w = _widget(qtbot)
    seen: list = []
    w.changed.connect(seen.append)
    restored = PlotAppearance(title_size=20)
    w.set_state(restored)                 # a restore: silent, and no step
    assert _depth(w) == (0, 0) and not w.undo_button.isEnabled() and seen == []

    w.background.setCurrentIndex(w.background.findData("light"))       # a run opens...
    loaded = PlotAppearance(title_size=20, background="light", grid=True)
    w.set_state(loaded)                   # ...and an install from outside closes it
    assert _depth(w) == (1, 0), "the run is the user's step; set_state recorded none"
    assert len(seen) == 1, "set_state never emits"
    w.tick_size.setValue(9)               # inside the pause, and still a NEW step
    assert w.undo() and w.state() == loaded
    assert w.undo() and w.state() == restored
    assert not w.undo()


def test_what_the_host_leaves_on_screen_is_what_is_recorded(qtbot):
    """A host may answer ``changed`` with ``set_state``: Comparison figures puts the old
    value back when a render fails, Subpopulations strips the colours. That answer is the
    step -- a refused edit records nothing, a normalised one records the normal form."""
    w = _widget(qtbot)

    def refuse(_state):
        w.set_state(PlotAppearance())

    w.changed.connect(refuse)
    w.preset.activated.emit(1)
    assert w.state() == PlotAppearance() and _depth(w) == (0, 0)
    assert not w.undo_button.isEnabled()
    w.changed.disconnect(refuse)

    def strip(state):
        w.set_state(replace(state, data_color=None, fit_color=None))

    w.changed.connect(strip)
    w.preset.activated.emit(1)
    plain = replace(PUBLICATION, data_color=None, fit_color=None)
    assert w.state() == plain and _depth(w) == (1, 0)
    assert w.undo() and w.state() == PlotAppearance()
    assert w.redo() and w.state() == plain
    assert _depth(w) == (1, 0), "an undo the host answered pushed no phantom step"


def test_an_undo_or_redo_the_host_refuses_moves_nothing(qtbot):
    """A host that answers with what it showed before has refused the step, and then the
    stack does not move either: the step stays where it was, offered again, neither lost
    nor replaced by a Redo that replays a step to where the plot already is."""
    w = _widget(qtbot)
    w.preset.activated.emit(1)
    w.preset.activated.emit(2)
    assert w.undo() and w.state() == PUBLICATION and _depth(w) == (1, 1)
    tips = (w.undo_button.toolTip(), w.redo_button.toolTip())

    def refuse(_state):               # Comparison figures, when the figure will not render
        w.set_state(PUBLICATION)

    w.changed.connect(refuse)
    assert not w.undo() and not w.redo()
    assert w.state() == PUBLICATION and _depth(w) == (1, 1)
    assert (w.undo_button.toolTip(), w.redo_button.toolTip()) == tips
    assert w.undo_button.isEnabled() and w.redo_button.isEnabled()
    w.changed.disconnect(refuse)
    assert w.undo() and w.state() == PlotAppearance(), "the refused step is still there"
    assert w.redo() and w.redo() and w.state() == PRESENTATION and _depth(w) == (2, 0)


def test_the_ceiling_drops_the_oldest_steps_and_counts_them(qtbot):
    w = _widget(qtbot)
    over = 5
    for i in range(APPEARANCE_HISTORY_DEPTH + over):
        w.preset.activated.emit(1 + i % 2)
    assert _depth(w) == (APPEARANCE_HISTORY_DEPTH, 0)
    assert w._history.dropped == over
    assert w.undo_button.toolTip().endswith(
        "Its oldest history was dropped at the 200-step ceiling (5 steps lost).")
    while w.undo():
        pass
    # Before the drops, the floor was the empty start; the five oldest steps went.
    assert w.state() == PUBLICATION
    assert not w.undo_button.isEnabled()
    assert w.undo_button.toolTip() == (
        "Nothing left to undo — this plot's oldest appearance history was dropped at the "
        "200-step ceiling (5 steps lost).")
    w.clear_history()
    assert w._history.dropped == 0 and _depth(w) == (0, 0)
    assert w.undo_button.toolTip() == "Nothing to undo yet."
    assert not w.redo_button.isEnabled()


def test_the_buttons_follow_the_stack_and_name_the_step(qtbot, monkeypatch):
    w = _widget(qtbot)
    assert not w.undo_button.isEnabled() and not w.redo_button.isEnabled()
    assert w.undo_button.toolTip() == "Nothing to undo yet."
    assert w.redo_button.toolTip() == "Nothing to redo."

    w.title_size.setValue(12)
    assert w.undo_button.isEnabled(), "an open run is offered to Undo"
    assert w.undo_button.toolTip() == "Undo: title size default → 12 pt"
    qtbot.wait(PAUSE_MS)
    w.title_size.setValue(14)
    qtbot.wait(PAUSE_MS)
    assert w.undo_button.toolTip() == "Undo: title size 12 → 14 pt"

    w.undo_button.click()
    assert w.redo_button.isEnabled()
    assert w.redo_button.toolTip() == "Redo: title size 12 → 14 pt"
    assert w.undo_button.toolTip() == "Undo: title size default → 12 pt"

    w.background.setCurrentIndex(w.background.findData("dark"))
    assert not w.redo_button.isEnabled(), "committing this run will clear Redo"
    assert w.redo_button.toolTip() == "Redo is cleared by a new change."
    assert w.undo_button.toolTip() == "Undo: background original → dark"
    qtbot.wait(PAUSE_MS)
    assert w.redo_button.toolTip() == "Nothing to redo."

    qtbot.keyClicks(w.title, "Figure 3")
    assert w.undo_button.toolTip() == "Undo: title automatic → ‘Figure 3’"
    monkeypatch.setattr(QtWidgets.QColorDialog, "getColor",
                        staticmethod(lambda *a, **k: QtGui.QColor("#7f00ff")))
    w.data_color.click()                  # closes the typing's run, then one step
    assert w.undo_button.toolTip() == "Undo: data color default → #7f00ff"
    w.preset.activated.emit(1)
    assert w.undo_button.toolTip() == "Undo: Publication · light preset"
    w.reset_button.click()
    assert w.undo_button.toolTip() == "Undo: reset appearance"
    w.undo_button.click()
    assert w.redo_button.toolTip() == "Redo: reset appearance"

    # One field is named, whatever moved it: a size typed back to Default lands on no
    # override at all, and is still that size's step, not a reset nobody pressed.
    one = _widget(qtbot)
    one.title_size.setValue(12)
    qtbot.wait(PAUSE_MS)
    one.title_size.setValue(0)
    assert one.undo_button.toolTip() == "Undo: title size 12 pt → default"


def test_each_key_is_registered_once_and_none_without_shortcuts(qtbot):
    w = _widget(qtbot)
    shortcuts = w.findChildren(QtGui.QShortcut)
    keys = sorted(s.key().toString() for s in shortcuts)
    assert keys == sorted(k.toString() for k in undo_sequences() + redo_sequences())
    assert len(set(keys)) == len(keys), "a key bound twice in one window fires nothing"
    assert all(s.parent() is w for s in shortcuts)
    assert all(s.context() == QtCore.Qt.ShortcutContext.WindowShortcut for s in shortcuts)

    keyless = _widget(qtbot, shortcuts=False)
    assert keyless.findChildren(QtGui.QShortcut) == []
    assert keyless.undo_button is not None, "the buttons stay; only the keys are the window's"

    plain = _widget(qtbot, history=False)
    assert plain.findChildren(QtGui.QShortcut) == []
    assert plain.undo_button is None and plain.redo_button is None
    plain.preset.activated.emit(1)
    assert not plain.undo() and plain.state() == PUBLICATION


def test_the_keys_undo_and_redo_in_the_widgets_window(qtbot):
    w = _widget(qtbot)
    _activate(qtbot, w)
    w.title_size.setValue(14)
    _press(w, undo_sequences()[0])
    assert w.state() == PlotAppearance()
    for key in redo_sequences():         # every Redo key this platform resolves
        _press(w, key)
        assert w.state() == PlotAppearance(title_size=14), key.toString()
        _press(w, undo_sequences()[0])


# ================================================================================================
# the eight windows that host it
# ================================================================================================

HOSTS = ("event", "psd", "iv", "vdep", "summary", "comparison", "subpop", "study")


def _iv_window():
    window = IVWindow(Session(), defer_initial_refresh=True)
    window._points = [iv.IVPoint(v, 100 * v + 2, "pA", "measured", str(i), f"run {i}")
                      for i, v in enumerate((-.2, -.5, -.8, -1.0))]
    window._redraw()
    return window


def _vdep_window():
    window = VDepWindow(Session(), defer_initial_refresh=True)
    window._points = [voltage_dep.VDepPoint(
        v, -100 * v + 2, f"run {i}", baseline_fingerprint="baseline-a",
        analyzed_duration_s=10, voltage_source="measured", doc_id=str(i))
        for i, v in enumerate((-.2, -.5, -.8, -1.0))]
    window._series = voltage_dep.vdep_comparability(window._points, metric="capture_rate_hz")
    window._applied_fit = "linear"
    window._redraw()
    return window


def _events_source(key="measurements", label="Measured events") -> PlotSource:
    frame = pd.DataFrame({"event_id": [5, 6, 9, 11, 14, 17],
                          "dwell_s": [.001, .002, .003, .004, .005, .006],
                          "depth_pA": [10., 20., 30., 40., 50., 60.],
                          "amplitude_pA": [12., 25., 35., 42., 55., 64.],
                          "ecd_pA_s": [.01, .04, .09, .16, .25, .36],
                          "snr": [5., 8., 10., 12., 15., 16.],
                          "accepted": [True, True, False, True, True, True]})
    return PlotSource(key, label, frame, kind="manual", doc_key=key, current=True)


def _study_window(suptitle=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from nanoscope.gui.study_figure_window import StudyFigureWindow

    figure = Figure(figsize=(4, 3), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot([1., 2., 3.], [.2, .5, .3], "o-", color="#147d92", label="2 hours")
    axis.set(xlabel="Dwell (ms)", ylabel="Current (pA)", title="Accepted events")
    axis.legend()
    if suptitle is not None:
        figure.suptitle(suptitle)
    frame = pd.DataFrame({"dwell_ms": [1., 2., 3.], "current_pA": [.2, .5, .3]})
    return StudyFigureWindow(figure, frame, {}, name="history_fixture",
                             provenance={"study": "Synthetic fixture"})


def _build(kind):
    if kind == "event":
        from nanoscope.gui.event_plot_panel import EventPlotWindow
        return EventPlotWindow(Session())
    if kind == "psd":
        from nanoscope.gui.psd_panel import PSDWindow
        return PSDWindow(Session(), defer_initial_refresh=True)
    if kind == "iv":
        return _iv_window()
    if kind == "vdep":
        return _vdep_window()
    if kind == "summary":
        from nanoscope.gui.distribution_summary_panel import DistributionSummaryWindow
        window = DistributionSummaryWindow()
        window.set_sources([_events_source()])
        window.analyze()
        return window
    if kind == "comparison":
        from nanoscope.gui.comparison_figures_panel import ComparisonFiguresWindow
        window = ComparisonFiguresWindow()
        window.set_sources([_events_source()])
        assert window.analyze()
        return window
    if kind == "subpop":
        from nanoscope.gui.subpop_window import SubpopReportWindow
        return SubpopReportWindow()
    return _study_window()


@pytest.fixture(params=HOSTS)
def host(request, qtbot):
    window = _build(request.param)
    qtbot.addWidget(window)
    return request.param, window


def _applied(kind, window) -> PlotAppearance:
    """What the HOST holds -- what it drew with and what it saves -- not the widget's."""
    if kind == "study":
        return PlotAppearance.from_mapping(window.style()["appearance"])
    return PlotAppearance.from_mapping(window.appearance_state())


def _canvas_ink(kind, window) -> str | None:
    """The background the host actually painted, where it paints one with pyqtgraph."""
    canvas = {"event": lambda: window._density_glw,
              "psd": lambda: window._plot_widget,
              "iv": lambda: window._plot_widget,
              "vdep": lambda: window._plot_widget,
              "summary": lambda: window._plot_widget,
              "subpop": lambda: window._embed_view._plot}.get(kind)
    return None if canvas is None else canvas().backgroundBrush().color().name()


#: The canvas each override paints (``apply_plot_appearance``). With no override the host's
#: own canvas shows, which is neither -- and not comparable with the canvas before the
#: first edit: the Noise window paints its own dark canvas only on its first restyle.
_PAINT = {"light": "#ffffff", "dark": "#151a21"}


def _assert_painted(kind, window, state) -> None:
    ink = _canvas_ink(kind, window)
    if ink is None:
        return
    if state.background in _PAINT:
        assert ink == _PAINT[state.background], (kind, state.background)
    else:
        assert ink not in _PAINT.values(), (kind, ink)


def test_each_window_undoes_and_redoes_an_edit_a_preset_and_a_reset(host):
    kind, window = host
    widget = window._appearance_widget
    assert widget.undo_button is not None and widget.findChildren(QtGui.QShortcut), (
        "no window an eight-host widget lives in binds Undo or Redo itself")
    start = _applied(kind, window)
    figure = getattr(window, "_figure_item", None)

    widget.background.setCurrentIndex(widget.background.findData("light"))     # an edit
    edited = _applied(kind, window)
    assert edited == replace(start, background="light")
    widget.preset.activated.emit(2)                                              # a preset
    preset = _applied(kind, window)
    assert preset.background == "dark" and preset.title_size == 14
    widget.reset_button.click()                                                  # a reset
    assert _applied(kind, window) == PlotAppearance()

    # Back through all three, then forward again. At every step the HOST holds -- and,
    # where it paints a pyqtgraph canvas, shows -- the state the widget stepped to:
    # an undo reaches the host through the same ``changed`` a user edit does.
    for expected in (preset, edited, start):
        widget.undo_button.click()
        assert widget.state() == expected and _applied(kind, window) == expected
        _assert_painted(kind, window, expected)
    assert not widget.undo_button.isEnabled()
    for expected in (edited, preset, PlotAppearance()):
        widget.redo_button.click()
        assert widget.state() == expected and _applied(kind, window) == expected
        _assert_painted(kind, window, expected)
    assert not widget.redo_button.isEnabled()
    if kind == "comparison":
        assert window._figure_item is not figure, "the figure was re-rendered, not just noted"


def _restore_doors(kind, window):
    """Every way the host installs a figure from outside: each must end the history."""
    if kind == "study":
        return [window.reset_style]
    doors = [lambda: window.set_appearance_state({"title_size": 16})]
    if kind in ("summary", "comparison"):
        def restore():
            payload, tables = window.snapshot_state()
            window.restore_state(payload, tables)
        doors.append(restore)
    if kind == "subpop":
        doors.append(lambda: window.set_report(None))
    return doors


def test_a_restored_or_blanked_figure_starts_its_history_over(host):
    kind, window = host
    widget = window._appearance_widget
    for door in _restore_doors(kind, window):
        widget.background.setCurrentIndex(widget.background.findData("dark"))
        widget.preset.activated.emit(1)
        assert widget.undo_button.isEnabled()
        door()
        assert _depth(widget) == (0, 0)
        assert not widget.undo_button.isEnabled() and not widget.redo_button.isEnabled()


def test_comparison_figures_refusing_an_undo_keeps_the_step(qtbot, monkeypatch,
                                                             blocked_modals):
    """THE HOST THAT REFUSES, for real: when the figure will not render, Comparison
    figures warns and puts its previous appearance back. That undo moved nothing, so the
    stack must not have moved either."""
    window = _build("comparison")
    qtbot.addWidget(window)
    widget = window._appearance_widget
    widget.preset.activated.emit(1)
    render = window._make_figure_item

    def fails_unstyled(snapshot, tables, appearance, source_appearance=None):
        if appearance == PlotAppearance():
            raise ValueError("synthetic render failure")
        return render(snapshot, tables, appearance, source_appearance)

    monkeypatch.setattr(window, "_make_figure_item", fails_unstyled)
    assert not widget.undo()
    assert [m.kind for m in blocked_modals] == ["QMessageBox.warning"]
    assert widget.state() == _applied("comparison", window) == PUBLICATION
    assert _depth(widget) == (1, 0) and not widget.redo_button.isEnabled()
    assert widget.undo_button.toolTip() == "Undo: Publication · light preset"
    monkeypatch.setattr(window, "_make_figure_item", render)
    assert widget.undo() and _applied("comparison", window) == PlotAppearance()


def test_emptying_a_label_box_still_takes_the_label_off_the_figure(qtbot):
    """Comparison and Study figures draw an EMPTY title as none and an empty axis label as
    blank, and their own only when there is no override at all (``render_comparison``,
    ``StudyFigureStyle.apply``). Emptying the box is the GUI's one way to the first, as it
    was at 27ec0eb, and an undo gives the figure its own label back."""
    from nanoscope.plot.comparison_figures import render_comparison

    comparison = _build("comparison")
    qtbot.addWidget(comparison)
    widget = comparison._appearance_widget
    for box in (widget.title, widget.x_label):
        qtbot.keyClicks(box, "x")
        qtbot.keyClick(box, QtCore.Qt.Key_Backspace)
    assert comparison.appearance_state() == {"title": "", "x_label": ""}
    figure = render_comparison(comparison._snapshot, comparison._tables, comparison._appearance)
    assert figure._suptitle is None
    assert {axis.get_xlabel() for axis in figure.axes if axis.get_visible()} == {""}
    assert widget.undo() and comparison.appearance_state() == {}
    figure = render_comparison(comparison._snapshot, comparison._tables, comparison._appearance)
    assert figure._suptitle is not None and figure._suptitle.get_text()

    study = _study_window(suptitle="Original suptitle")
    qtbot.addWidget(study)
    qtbot.keyClicks(study._appearance_widget.title, "x")
    qtbot.keyClick(study._appearance_widget.title, QtCore.Qt.Key_Backspace)
    assert study.style()["appearance"] == {"title": ""}
    assert study.apply_appearance()
    assert study._figure._suptitle.get_text() == ""


def test_in_the_explorer_a_key_reaches_only_the_page_on_screen(qtbot):
    """THE CONFIGURATION THE APP RUNS. Every one of these windows is a page of the one
    Analysis Explorer, so several appearance widgets bind the same keys in ONE window.
    That is unambiguous only because a QShortcut answers while its widget is on screen,
    and only the current page is: the key reaches the page in front, never the other."""
    from nanoscope.gui.analysis_explorer import PAGE_IV, PAGE_VOLTAGE_SERIES, AnalysisExplorer

    explorer = AnalysisExplorer()
    qtbot.addWidget(explorer)
    pages = {PAGE_IV: _iv_window(), PAGE_VOLTAGE_SERIES: _vdep_window()}
    for page_id, page in pages.items():
        explorer.show_page(page_id, page)
        page._workflow.set_section("Appearance")
        widget = page._appearance_widget
        widget.background.setCurrentIndex(widget.background.findData("light"))
    qtbot.waitUntil(explorer.isActiveWindow, timeout=3000)

    _press(explorer, undo_sequences()[0])     # the voltage series page is in front
    assert pages[PAGE_VOLTAGE_SERIES].appearance_state() == {}
    assert pages[PAGE_IV].appearance_state() == {"background": "light"}

    explorer.raise_page(PAGE_IV)
    _press(explorer, undo_sequences()[0])
    assert pages[PAGE_IV].appearance_state() == {}
    assert pages[PAGE_VOLTAGE_SERIES].appearance_state() == {}


# ================================================================================================
# the Plots window: one stack, and a Delete that asks
# ================================================================================================

@pytest.fixture
def plots(qtbot):
    def _clear():
        settings = QtCore.QSettings()
        settings.remove(pp.PlotsWindow._TPL_SETTINGS_KEY)
        settings.sync()

    _clear()
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"dwell_s": 2e-5 + rng.exponential(2.94e-4, 300),
                          "depth_pA": np.abs(rng.normal(40.0, 6.0, 300)),
                          "frac_depth": np.abs(rng.normal(0.30, 0.05, 300)),
                          "snr": np.abs(rng.normal(9.0, 2.0, 300)),
                          "accepted": np.ones(300, dtype=bool),
                          "filter_limited": np.zeros(300, dtype=bool)})
    window = pp.PlotsWindow(Session())
    qtbot.addWidget(window)
    window.set_data([PlotSource(key="doc0:0", label="synthetic −500 mV", df=frame,
                                provenance={"fs": 250_000.0, "min_dwell_s": 2e-5},
                                colour="#0072B2", doc_id="doc0", channel=0,
                                module_id="detect.threshold",
                                calibration_fingerprint=(1.0, 0.0))])
    window.set_plot_type("Histogram")
    window.set_histogram_column("dwell_s")
    yield window
    _clear()


def test_the_plots_window_keeps_exactly_one_undo_stack(plots, qtbot):
    widget = plots._appearance_widget
    assert widget._history is None, "its PlotConfig stack already holds appearance"
    assert widget.undo_button is None and widget.redo_button is None
    assert widget.findChildren(QtGui.QShortcut) == []
    keys = [s.key().toString() for s in plots.findChildren(QtGui.QShortcut)]
    assert len(keys) == len(set(keys)), f"a key was bound twice: {keys}"

    before = plots.config()
    back = plots._history.depth()[0]
    widget.background.setCurrentIndex(widget.background.findData("light"))
    assert plots._style.background == "light"
    assert plots._history.depth()[0] == back + 1, "the appearance edit is the window's step"
    _activate(qtbot, plots)
    _press(plots, undo_sequences()[0])         # one stack, so the window's own key answers
    assert plots.config() == before
    assert widget.state() == PlotAppearance.from_mapping(plots.appearance_state()), (
        "and the widget shows what the window went back to")
    assert widget.state().background == before.style.background


def test_deleting_a_template_asks_and_cancel_keeps_it(plots, monkeypatch, blocked_modals):
    name = "kept <b>figure</b>"
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: (name, True)))
    plots._save_template()
    assert plots._template_combo.currentText() == name

    asked: list = []
    guard = QtWidgets.QDialog.exec           # the suite's modal guard: Rejected, no click

    def cancel(box, *a, **k):
        asked.append(box)
        return guard(box, *a, **k)

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", cancel)
    plots._delete_template()
    assert name in pp.PlotsWindow._load_templates(), "Cancel keeps the template"
    assert plots._template_combo.currentText() == name
    (box,) = asked
    assert box.textFormat() == QtCore.Qt.TextFormat.PlainText
    assert name in box.text(), "the question names the template, verbatim"
    roles = {button.text(): box.buttonRole(button) for button in box.buttons()}
    assert roles == {"Delete template": QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
                     "Cancel": QtWidgets.QMessageBox.ButtonRole.RejectRole}
    assert box.defaultButton().text() == "Cancel", "Return keeps it"
    assert box.escapeButton().text() == "Cancel", "and so does Esc"
    assert [m.kind for m in blocked_modals] == ["QMessageBox"]

    def delete(box, *a, **k):
        next(b for b in box.buttons() if b.text() == "Delete template").click()
        return 0

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", delete)
    plots._delete_template()
    assert name not in pp.PlotsWindow._load_templates()
    assert plots._template_combo.currentText() == pp.PlotsWindow._TPL_MANUAL
