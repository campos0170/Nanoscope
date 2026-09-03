"""Undo/redo, saved templates and JSON sidecars for the Plots window — owner A2 / A3 / B3.

All three of these fall out of one thing: ``PlotsWindow.config()`` returns a frozen VALUE.
An undo stack is a list of them, a template is one of them under a name, an exported
sidecar is one of them in a file. So the three features share their whole surface —
:meth:`PlotsWindow._set_config` writes, :meth:`PlotsWindow.config` reads — and they share
this file, because the interesting failures are at the seams BETWEEN them:

* an undo that restores a config but leaves the widgets showing the old one, or that
  cascades change handlers and pushes entries of its own;
* a template that quietly moves the user's ticked sources, or that comes back claiming a
  fit has been computed;
* a sidecar that round-trips the fields somebody remembered and silently drops the rest.

The last one is pinned by MECHANISM rather than by a list: :func:`_busy_config` must give
every field of every section a non-default value, and
``test_the_fixture_actually_exercises_every_field`` fails if it does not. Add a field to
``StyleSettings`` and that test goes red until the round-trip covers it — which is the
same trick ``test_no_derived_state_ever_enters_the_config`` plays one file over.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import json
import os
from dataclasses import fields, replace
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import (
    AxesConfig,
    ClusterConfig,
    FitConfig,
    FrozenMap,
    HistogramConfig,
    KdeConfig,
    PLOT_CONFIG_FORMAT,
    PLOT_CONFIG_MIN_READ_VERSION,
    PLOT_CONFIG_VERSION,
    PlotConfig,
    PlotConfigError,
    PlotConfigVersionError,
    PlotsWindow,
    ScatterConfig,
    SelectionConfig,
    StyleSettings,
    _ConfigHistory,
    plot_config_from_json,
    plot_config_to_json,
    read_plot_config,
    write_plot_config,
)


# ================================================================================================
# builders and fixtures
# ================================================================================================

def _events(n: int = 300, *, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)),
        "snr": np.abs(rng.normal(9.0, 2.0, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key: str = "doc0:0", label: str = "apo −500 mV") -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(),
                      provenance={"fs": 250_000.0, "min_dwell_s": 2e-5},
                      colour="#0072B2", doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


@pytest.fixture(autouse=True)
def _clean_template_store():
    """This module's templates start and end absent.

    The suite redirects QSettings to a temp .ini (tests/conftest.py), but the store is
    process-wide, so a template left behind by one test is in the next one's combo."""
    def _clear():
        s = QtCore.QSettings()
        s.remove(PlotsWindow._TPL_SETTINGS_KEY)
        s.sync()
    _clear()
    yield
    _clear()


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data([_source()])
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_time_unit("ms")
    return w


def _busy_config() -> PlotConfig:
    """A config with EVERY field of every section moved off its default.

    The point of the exercise is that a codec which drops a field cannot pass the
    round-trip test, and the point of ``test_the_fixture_actually_exercises_every_field``
    is that this builder cannot quietly stop covering one.
    """
    return PlotConfig(
        plot_type=pp.KDE,
        norm_user_set=True,
        selection=SelectionConfig(source_keys=("doc0:0", "doc1:1"), accepted_only=False),
        histogram=HistogramConfig(column="frac_depth", bins=64,
                                  normalization=pp._NORM_PROB,
                                  colour_by=pp._COLOUR_CLUSTER, log_x=True, log_y=True,
                                  include_filter_limited=True),
        scatter=ScatterConfig(x="dwell_s", y="snr", log_x=True, log_y=True,
                              colour_by=pp._COLOUR_CLUSTER),
        kde=KdeConfig(x="depth_pA", y="snr", log_x=False, log_y=True, levels=13),
        cluster=ClusterConfig(features=("dwell_s", "snr"), method=pp._METHOD_KMEANS,
                              k_mode=pp._K_FIXED, k=5),
        fit=FitConfig(kind=pp._FIT_GAMMA, k=3, k_auto=True,
                      scatter_kind=pp._SFIT_POWER, applied=pp._FIT_EXP, applied_k=2,
                      applied_k_auto=True, applied_scatter=pp._SFIT_LINEAR,
                      # VERBATIM box text, deliberately unparseable — see `_decoded`.
                      range_lo="nan", range_hi="abc", range_unit="µs"),
        axes=AxesConfig(time_unit="s", x_min="1.5", x_max="2.5", y_min="3.5",
                        y_max="4.5", manual_x=(1e-4, 2e-3), manual_y=(10.0, 80.0)),
        style=StyleSettings(
            title="figure 3b", x_label="dwell (ms)", y_label="events",
            show_legend=False, font_family="Helvetica", font_size_pt=13,
            data_color="#ff0000", fit_color="#00ff00", point_size=9, line_width=3,
            background="light", export_w=1600, export_h=1200,
            fit_values_on_plot=True,
            label_overrides=FrozenMap({"doc0:0": "pore A"}),
            legend_pos=(0.31, 0.42), legend_font_size_pt=8, legend_counts=False,
            legend_include=FrozenMap({"series:doc0:0": False, "cluster:1": False}),
            show_grid=False, title_font_size_pt=16, axis_label_font_size_pt=14,
            tick_font_size_pt=9, title_bold=True, title_italic=True,
            show_r2=False, show_gof=False, show_ci_band=True, show_provenance=True,
            show_residuals=True),
    )


# ================================================================================================
# A2 — THE STACK, AS A VALUE (no window, no widgets)
# ================================================================================================

def test_the_stack_records_only_real_movement():
    """The DIFF is the mechanism (see :meth:`_ConfigHistory.note`): a caller that says
    "something may have moved" too often costs a comparison and records nothing, which is
    what lets three unrelated choke points and a generic signal walk all call the hook."""
    a = PlotConfig()
    h = _ConfigHistory(a)
    assert h.note(a) is False and h.depth() == (0, 0), "no movement is no step"

    b = replace(a, style=replace(a.style, title="one"))
    assert h.note(b) is True and h.depth() == (1, 0)
    assert h.note(b) is False, "the same config twice is still one step"
    assert h.base == b


def test_undo_and_redo_walk_the_same_line():
    a = PlotConfig()
    b = replace(a, style=replace(a.style, title="one"))
    c = replace(b, style=replace(b.style, title="two"))
    h = _ConfigHistory(a)
    h.note(b)
    h.note(c)

    assert h.undo() == b and h.depth() == (1, 1)
    assert h.undo() == a and h.depth() == (0, 2)
    assert h.undo() is None, "nothing behind the first state"
    assert h.redo() == b and h.redo() == c
    assert h.redo() is None


def test_a_new_change_destroys_the_redo_branch():
    """Keeping it would offer a "forward" into a configuration no longer reachable from
    where the user is."""
    a = PlotConfig()
    b = replace(a, style=replace(a.style, title="one"))
    c = replace(a, style=replace(a.style, title="two"))
    h = _ConfigHistory(a)
    h.note(b)
    assert h.undo() == a and h.can_redo
    h.note(c)
    assert not h.can_redo and h.depth() == (1, 0)


def test_the_stack_is_bounded_and_says_what_it_dropped():
    """A silent bound is indistinguishable from a broken Undo button, so the count is
    kept. The shipped depth is :data:`pp._HISTORY_DEPTH`; this drives a small one so the
    eviction is reached without 200 configs."""
    h = _ConfigHistory(PlotConfig(), depth=3)
    for i in range(6):
        h.note(replace(PlotConfig(), style=StyleSettings(title=str(i))))
    back, _forward = h.depth()
    assert back == 3, "the bound holds"
    assert h.dropped == 3, "and it says how many steps it cost"
    assert pp._HISTORY_DEPTH == 200, (
        "the shipped depth is a measured number — see the comment on _HISTORY_DEPTH")


def test_reset_forgets_both_directions():
    h = _ConfigHistory(PlotConfig())
    h.note(replace(PlotConfig(), style=StyleSettings(title="x")))
    h.undo()
    h.reset(PlotConfig())
    assert h.depth() == (0, 0) and not h.can_undo and not h.can_redo


# ================================================================================================
# A2 — THE STACK, WIRED
# ================================================================================================

def test_undo_restores_the_config_and_the_widgets(win):
    """Both halves, because restoring the VALUE and leaving the controls showing the old
    one is the failure this feature would have: the next change would be read off widgets
    that disagree with the config the window thinks it has."""
    before = win.config()
    win.set_bins(77)
    win.set_normalization("density")
    win.set_log_x(True)

    assert win.undo_config() and win.undo_config() and win.undo_config()

    assert win.config() == before, "the value came back"
    assert win._bins.value() == before.histogram.bins, "and so did the widget"
    assert win._norm_combo.currentText() == before.histogram.normalization
    assert win._hist_logx.isChecked() is before.histogram.log_x


def test_undo_does_not_cascade_into_new_entries(win):
    """The single ``_set_config`` writer is what makes this tractable: it writes ~40
    controls under ``_updating`` so no handler runs, and ``_history_suspended`` covers the
    draw at the end, which runs with ``_updating`` already restored."""
    win.set_bins(77)
    back, forward = win._history.depth()

    win.undo_config()

    assert win._history.depth() == (back - 1, forward + 1), (
        "an undo must move ONE step, not push entries of its own")
    assert win._history.base == win.config(), (
        "and the stack's idea of the window must be the window")


def test_redo_puts_it_back(win):
    win.set_title("figure 3b")
    win._title_edit.editingFinished.emit()
    assert win.undo_config()
    assert win._style.title == "" and win._title_edit.text() == ""
    assert win.redo_config()
    assert win._style.title == "figure 3b" and win._title_edit.text() == "figure 3b"


def test_a_run_of_typing_is_ONE_undo_entry(win):
    """THE GRAIN RULE. Every keystroke in the Title box emits ``textChanged``, so a
    per-signal stack would need eleven Ctrl+Z to walk back out of "figure 3b" — and the
    twelfth would undo the colour change before it, which is the change the user was
    actually trying to reach."""
    win.set_bins(41)                     # a committed step to walk back to
    before = win.config()
    back, _ = win._history.depth()

    for text in ("f", "fi", "fig", "fig ", "fig 3", "fig 3b"):
        win._title_edit.setText(text)
    assert win._history.depth()[0] == back, "a keystroke is not a step"
    assert win._history.base != win.config(), "...but the step IS open"

    win._title_edit.editingFinished.emit()   # what focus-out and Return both do
    assert win._history.depth()[0] == back + 1, "one entry for the whole run"

    assert win.undo_config()
    assert win.config() == before, "and one Ctrl+Z walks back out of all six keystrokes"


def test_a_pause_also_closes_the_text_run(win, qtbot):
    """The other half of "until focus-out or a pause": a user who types a title, stops to
    look at the plot and types more would otherwise get the whole title as one entry
    however long they paused."""
    back, _ = win._history.depth()
    win._title_edit.setText("figure")
    assert win._history.depth()[0] == back
    qtbot.wait(pp._HISTORY_COALESCE_MS + 250)
    assert win._history.depth()[0] == back + 1, "the pause closed it"


def test_a_change_that_is_text_PLUS_something_else_is_not_held_open(win):
    """The deferral is keyed on the SHAPE OF THE CHANGE, not on which signal delivered it
    (see :meth:`PlotsWindow._typed_text_only`), so a gesture that moves a real control is
    never left dangling by a title that happens to be half-typed."""
    win._title_edit.setText("half typed")
    back, _ = win._history.depth()
    win.set_bins(55)
    assert win._history.depth()[0] == back + 1
    assert win._history.base == win.config(), "committed, not left open"


def test_typed_text_only_is_about_the_fields_not_the_signal():
    """Pure, so the rule can be read without a window. A combo's string is a PICK — one
    gesture, one step — and must not be mistaken for typing."""
    base = PlotConfig()
    typed = replace(base, style=replace(base.style, title="x"),
                    axes=replace(base.axes, x_min="0.1"))
    picked = replace(base, histogram=replace(base.histogram, normalization=pp._NORM_PROB))

    assert PlotsWindow._typed_text_only(base, typed)
    assert not PlotsWindow._typed_text_only(base, picked), (
        "`normalization` is a string but it is chosen from a combo")
    assert not PlotsWindow._typed_text_only(base, base), "no change is not a text change"
    assert not PlotsWindow._typed_text_only(
        base, replace(typed, histogram=replace(base.histogram, bins=99))), "mixed commits"


def test_arriving_data_is_not_an_undo_step(win):
    """``EventHistory.set_suspended``'s rule one object down: which columns the pickers
    land on and which rows are ticked are a CONSEQUENCE of a push, not something the user
    did — and MainWindow re-pushes on every detection."""
    win.set_bins(63)
    depth = win._history.depth()

    win.set_data([_source(), _source("doc1:0", "apo −600 mV")])

    assert win._history.depth() == depth, "a data push records nothing"
    assert win._history.base == win.config(), "but the base tracks the new window"
    # ...and the styling step underneath is still there to walk back to.
    assert win.undo_config() and win._bins.value() == PlotConfig().histogram.bins


def test_reset_state_drops_the_stack_with_the_data_it_described(win):
    win.set_bins(77)
    win.set_title("figure 3b")
    assert win._history.depth()[0] > 0

    win.reset_state()

    assert win._history.depth() == (0, 0), (
        "every entry named sources this window no longer has")
    assert win._history.base == win.config()
    assert not win.undo_config(), "Undo must not name a step it cannot take"


def test_undo_respects_the_run_gate(win, qtbot):
    """On an armed window undo DRAWS — it has to show you what you got back. On a window
    that has never been run it must not: drawing a picture the user never asked for is the
    complaint the gate exists to answer."""
    fresh = PlotsWindow(Session())
    qtbot.addWidget(fresh)
    fresh.auto_plot = False
    fresh.set_data([_source()])
    assert not fresh._plot_armed
    fresh.set_bins(44)
    assert not fresh._plot_armed, "changing a control does not run the plot"

    assert fresh.undo_config()
    assert not fresh._plot_armed, "and neither does undoing it"

    win._redraw()
    assert win._plot_armed
    win.set_bins(44)
    win.undo_config()
    assert win._plot_armed, "an armed window redraws what the undo restored"


def test_the_buttons_offer_an_open_text_run(win):
    """Undo is available from the first keystroke — committing is the first thing
    :meth:`PlotsWindow.undo_config` does — and Redo is withdrawn during one, because the
    commit will destroy the branch it points at."""
    win.set_bins(51)
    win.undo_config()
    assert win._redo_btn.isEnabled()

    win._title_edit.setText("t")
    assert win._undo_btn.isEnabled()
    assert not win._redo_btn.isEnabled(), (
        "a button that does something other than what it says is worse than a dim one")


# ================================================================================================
# A3 — TEMPLATES: what the split is, and that it is real
# ================================================================================================

def test_a_template_drops_exactly_the_three_unportable_groups():
    cfg = _busy_config()
    tpl = cfg.as_template()

    assert tpl.selection.source_keys == (), "source keys name THIS session's documents"
    assert (tpl.fit.applied, tpl.fit.applied_k, tpl.fit.applied_k_auto,
            tpl.fit.applied_scatter) == (pp._FIT_NONE, 1, False, pp._SFIT_NONE)
    assert tpl.style.label_overrides == {} and tpl.style.legend_include == {}
    # ...and NOTHING else moved.
    assert tpl == replace(
        cfg,
        selection=replace(cfg.selection, source_keys=()),
        fit=replace(cfg.fit, applied=pp._FIT_NONE, applied_k=1, applied_k_auto=False,
                    applied_scatter=pp._SFIT_NONE),
        style=replace(cfg.style, label_overrides=FrozenMap(), legend_include=FrozenMap()))
    assert tpl.as_template() == tpl, "idempotent, so a hand-edited file is sanitised too"


def test_a_template_keeps_the_row_POLICIES_that_name_no_row():
    """``accepted_only`` and ``include_filter_limited`` look like data selection and are
    not: they mean the same thing on any recording. Dropping ``accepted_only`` would be a
    recalled template that silently re-admits events the user rejected by hand."""
    tpl = _busy_config().as_template()
    assert tpl.selection.accepted_only is False
    assert tpl.histogram.include_filter_limited is True


def test_recalling_a_template_never_moves_the_ticked_sources(win):
    win.set_data([_source(), _source("doc1:0", "apo −600 mV")])
    win.set_sources(["apo −500 mV", "apo −600 mV"])
    ticked = win.selected_source_keys()
    assert len(ticked) == 2

    win.apply_template(PlotConfig(style=StyleSettings(background="light")))

    assert win.selected_source_keys() == ticked
    assert win._style.background == "light"


def test_recalling_a_fit_template_arms_apply_rather_than_claiming_a_fit(win):
    """Owner note #12's failure, re-created by the restore path: a window badged as
    already showing a gamma of rows nothing has fitted."""
    win.apply_template(
        win._BUILTIN_TEMPLATES["Dwell-time histogram (log-x, exponential)"])

    assert win._fit_combo.currentText() == pp._FIT_EXP, "the model is selected"
    assert win._applied_fit == pp._FIT_NONE, "and it is NOT claimed to be drawn"
    assert win._fit_apply_btn.isEnabled() and "●" in win._fit_apply_btn.text()


def test_a_blank_column_in_a_template_keeps_the_windows_own(win):
    """What makes a purely presentational template possible — and what the shipped
    "Publication figure (light)" is."""
    win.set_histogram_column("frac_depth")
    win.apply_template(win._BUILTIN_TEMPLATES["Publication figure (light)"])

    assert win._hist_col.currentText() == "frac_depth"
    assert win._style.background == "light" and win._style.show_grid is False
    assert win._identity.x_column == "frac_depth", "and the anchor still says so"


def test_a_template_naming_a_column_this_recording_lacks_leaves_the_anchor_honest(win):
    """THE STALE-ANCHOR HAZARD ``_installable`` closes. ``_set_config`` declines to write
    a column the picker does not carry — but then took ``_identity`` from the CONFIG, so
    the anchor named a column that was not on screen and the next rebase cleared the pins,
    the fit window and the title of a plot nothing had changed about."""
    win.apply_template(PlotConfig(histogram=HistogramConfig(column="no_such_column")))

    assert win._hist_col.currentText() == "dwell_s"
    assert win._identity.x_column == "dwell_s", "the anchor names what is on screen"

    # Now type against THAT plot. A rebase measuring from the phantom column would read
    # "the analysis moved" and throw both of these away on the next unrelated change.
    win.set_title("figure 3b")
    win.set_axis_range(xmin=0.1, xmax=2.0)
    win.set_bins(33)     # changes no quantity, so it must clear nothing
    assert win._style.title == "figure 3b"
    assert win._manual_x is not None, "the pin survived a change that changed no quantity"


def test_save_recall_rename_and_delete_a_template(win, monkeypatch):
    """The DetectionPanel named-protocol pattern: QSettings JSON, save-as / recall /
    rename / delete, and a rename that MOVES the stored settings rather than re-capturing
    what is on screen."""
    names = iter(["figure 3 panels", "figure 3"])
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: (next(names), True)))
    win.set_bins(64)
    win.set_title("figure 3b")
    win._title_edit.editingFinished.emit()

    win._save_template()
    assert win._template_combo.currentText() == "figure 3 panels"
    assert "figure 3 panels" in PlotsWindow._load_templates()

    # Recall on a window that has been changed since.
    win.set_bins(12)
    win.set_title("something else")
    win._on_template_selected("figure 3 panels")
    assert win._bins.value() == 64 and win._style.title == "figure 3b"
    assert win._template_combo.currentText() == "figure 3 panels"

    # RENAME MOVES THE STORED SETTINGS. It captures nothing from the screen, so an edit
    # made after the save cannot sneak into the renamed template. `_select_template` is
    # the door that puts a name back on the combo WITHOUT re-applying it, which is the
    # only way to hold "the name is selected" and "the screen has been edited" at once.
    win.set_bins(99)
    assert win._template_combo.currentText() == PlotsWindow._TPL_MANUAL, (
        "an edit takes the template's name off the settings")
    win._select_template("figure 3 panels")
    win._rename_template()
    stored = PlotsWindow._load_templates()
    assert set(stored) == {"figure 3"}
    assert plot_config_from_json(stored["figure 3"]).histogram.bins == 64, (
        "a rename is not a re-save of what is on screen")

    win._select_template("figure 3")
    win._delete_template()
    assert PlotsWindow._load_templates() == {}
    assert win._template_combo.currentText() == PlotsWindow._TPL_MANUAL
    assert win._bins.value() == 99, "deleting a NAME is not a request to change the plot"


def test_a_builtin_name_cannot_be_taken(win, monkeypatch):
    """The shipped set stays reachable; two different configurations behind one name in
    one combo is what the refusal prevents."""
    shown: list = []
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Publication figure (light)", True)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a)))
    win._save_template()
    assert PlotsWindow._load_templates() == {}, "the built-in was not replaced"
    assert shown, "and the refusal was reported, not silent"

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: (PlotsWindow._TPL_MANUAL, True)))
    win._save_template()
    assert PlotsWindow._load_templates() == {}, "the sentinel is reserved too"


def test_builtin_templates_are_not_editable(win):
    win._select_template("Publication figure (light)")
    assert not win._template_rename_btn.isEnabled()
    assert not win._template_delete_btn.isEnabled()
    win._select_template(PlotsWindow._TPL_MANUAL)


def test_the_name_drops_to_manual_only_when_it_stops_being_true(win):
    """A COMPARISON, not a blanket drop. ``DetectionPanel`` drops on any change at all;
    this one asks whether the change made a difference to WHAT THE TEMPLATE DESCRIBES, so
    the two things a template deliberately does not carry cannot take its name off it."""
    name = "Dwell-time histogram (log-x, exponential)"
    win._on_template_selected(name)
    assert win._template_combo.currentText() == name

    # A retyped legend entry: `style.label_overrides`, which no template carries.
    win._source_list.item(0).setText("pore A")
    assert win._style.label_overrides.get("doc0:0") == "pore A"
    assert win._template_combo.currentText() == name, (
        "a legend rename is not part of any template")

    # Pressing Apply: `fit.applied*` moves and the PROPOSAL does not, and applied is the
    # other thing a template deliberately never carries.
    assert win._fit_combo.currentText() == pp._FIT_EXP and win._applied_fit == pp._FIT_NONE
    win._apply_fit()
    assert win._applied_fit == pp._FIT_EXP
    assert win._template_combo.currentText() == name, (
        "a fit APPLIED is not part of any template — only the model proposed is")

    # ...and something the template DOES carry takes its name off at once.
    win.set_line_width(7)
    assert win._template_combo.currentText() == PlotsWindow._TPL_MANUAL


def test_a_corrupt_template_store_is_no_templates_not_a_crash():
    QtCore.QSettings().setValue(PlotsWindow._TPL_SETTINGS_KEY, "{not json")
    assert PlotsWindow._load_templates() == {}


# ================================================================================================
# B3 — THE JSON SIDECAR
# ================================================================================================

def test_the_fixture_actually_exercises_every_field():
    """THE MECHANISM behind the round-trip test. A codec that silently drops a field can
    only be caught by a fixture that sets it, and a hand-written fixture goes stale the
    first time a field is added — so this fails instead, naming the field, until
    :func:`_busy_config` covers it."""
    busy, blank = _busy_config(), PlotConfig()
    assert busy.plot_type != blank.plot_type
    assert busy.norm_user_set != blank.norm_user_set
    for section in pp._CONFIG_SECTIONS:
        got, default = getattr(busy, section), getattr(blank, section)
        for f in fields(default):
            assert getattr(got, f.name) != getattr(default, f.name), (
                f"_busy_config leaves {section}.{f.name} at its default, so the "
                f"round-trip test proves nothing about it")


def test_every_field_survives_the_round_trip():
    cfg = _busy_config()
    assert plot_config_from_json(plot_config_to_json(cfg)) == cfg


def test_the_typed_boxes_come_back_BYTE_FOR_BYTE():
    """``fit.range_lo``/``range_hi`` and the four axis boxes are the user's TEXT, not
    numbers: "nan", "inf" and "abc" are legal values there, ``_fit_range`` declines them,
    and ``_fit_range_input_notes`` quotes them back. A codec that parsed them would
    destroy the very text the message is about."""
    cfg = replace(PlotConfig(),
                  fit=FitConfig(range_lo="nan", range_hi="inf", range_unit="µs"),
                  axes=AxesConfig(x_min="abc", x_max="0.", y_min="1e-4", y_max=""))
    got = plot_config_from_json(plot_config_to_json(cfg))
    assert (got.fit.range_lo, got.fit.range_hi) == ("nan", "inf")
    assert got.fit.range_unit == "µs", "the unit is what makes the bounds interpretable"
    assert (got.axes.x_min, got.axes.x_max, got.axes.y_min) == ("abc", "0.", "1e-4")


def test_the_two_representations_of_an_axis_limit_are_both_kept():
    """The typed text is in the DISPLAY unit and the pin is in DATA units; a number can
    sit in a box unapplied, so they are not redundant."""
    cfg = replace(PlotConfig(), axes=AxesConfig(time_unit="ms", x_min="0.1", x_max="2.0",
                                               manual_x=(1e-4, 2e-3)))
    got = plot_config_from_json(plot_config_to_json(cfg))
    assert (got.axes.x_min, got.axes.x_max) == ("0.1", "2.0")
    assert got.axes.manual_x == (1e-4, 2e-3) and got.axes.manual_y is None


def test_the_frozen_collections_come_back_frozen():
    """``dict`` on the wire, :class:`FrozenMap` in the value — a decoded config that held
    a plain dict would be an undo entry every other entry could write into."""
    got = plot_config_from_json(plot_config_to_json(_busy_config()))
    assert isinstance(got.style.label_overrides, FrozenMap)
    assert isinstance(got.style.legend_include, FrozenMap)
    assert isinstance(got.selection.source_keys, tuple)
    assert isinstance(got.cluster.features, tuple)


def test_the_file_declares_what_it_is(tmp_path):
    out = write_plot_config(tmp_path / "cfg.json", _busy_config())
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == PLOT_CONFIG_FORMAT
    assert payload["version"] == PLOT_CONFIG_VERSION
    assert set(pp._CONFIG_SECTIONS) <= set(payload)
    assert read_plot_config(out) == _busy_config()


def test_a_version_this_build_cannot_read_is_REFUSED_BY_NAME(tmp_path):
    """Not a traceback, and not "the file is bad": the version the file declares and the
    versions this build reads are what the user can act on."""
    p = tmp_path / "cfg.json"
    payload = plot_config_to_json(PlotConfig())
    payload["version"] = PLOT_CONFIG_VERSION + 7
    p.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PlotConfigVersionError) as exc:
        read_plot_config(p)
    assert str(PLOT_CONFIG_VERSION + 7) in str(exc.value)
    assert str(PLOT_CONFIG_VERSION) in str(exc.value)
    assert p.name in str(exc.value)


def test_an_older_missing_key_reads_as_that_fields_default():
    """MIN_READ_VERSION is only honest if the reader is additive: it walks the DATACLASS's
    fields, not the payload's keys."""
    payload = plot_config_to_json(_busy_config())
    del payload["style"]["title_italic"]
    del payload["fit"]["range_unit"]
    payload["style"]["a_key_from_the_future"] = 12

    got = plot_config_from_json(payload)
    assert got.style.title_italic is StyleSettings().title_italic
    assert got.fit.range_unit == FitConfig().range_unit
    assert got.style.title == "figure 3b", "everything else still read"


@pytest.mark.parametrize("payload, kind", [
    ([1, 2, 3], "not an object"),
    ({"format": "something-else", "version": 1}, "wrong format"),
    ({"format": PLOT_CONFIG_FORMAT}, "no version"),
    ({"format": PLOT_CONFIG_FORMAT, "version": "one"}, "unreadable version"),
    ({"format": PLOT_CONFIG_FORMAT, "version": PLOT_CONFIG_MIN_READ_VERSION - 1},
     "too old"),
    ({"format": PLOT_CONFIG_FORMAT, "version": 1, "axes": ["not", "a", "section"]},
     "malformed section"),
])
def test_every_malformation_arrives_as_one_exception_type(payload, kind):
    """One class, so the GUI catches once and shows one sentence — ``project``'s promise,
    and the reason its ``_manifest_section`` guard exists."""
    with pytest.raises(PlotConfigError):
        plot_config_from_json(payload)


def test_export_and_import_through_the_window(win, tmp_path):
    win.set_bins(64)
    win.set_title("figure 3b")
    win._title_edit.editingFinished.emit()
    win.set_axis_range(xmin=0.1, xmax=2.0)
    saved = win.config()

    out = win.export_config_json(tmp_path / "figure3b.json")
    assert out is not None and out.exists()

    win.reset_state()
    win.set_data([_source()])
    assert win.config().style.title == ""

    assert win.import_config_json(out)
    assert win.config() == saved


def test_import_refuses_a_bad_file_with_a_message_not_a_traceback(win, tmp_path,
                                                                 monkeypatch):
    said: list = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda _p, _t, text, *a, **k: said.append(text)))
    p = tmp_path / "cfg.json"
    payload = plot_config_to_json(PlotConfig())
    payload["version"] = PLOT_CONFIG_VERSION + 1
    p.write_text(json.dumps(payload), encoding="utf-8")
    before = win.config()

    assert win.import_config_json(p) is False
    assert win.config() == before, "a refused import changes nothing"
    assert said and str(PLOT_CONFIG_VERSION + 1) in said[0]

    p.write_text("{not json", encoding="utf-8")
    assert win.import_config_json(p) is False
    assert len(said) == 2 and p.name in said[1]


def test_an_imported_config_naming_absent_sources_restores_what_it_can(win, tmp_path):
    """``_apply_checked_keys`` is deliberately silent about keys this session has not
    loaded: restoring what CAN be restored beats refusing the whole file."""
    cfg = replace(win.config(),
                  selection=SelectionConfig(source_keys=("doc0:0", "gone:9")),
                  histogram=replace(win.config().histogram, bins=71))
    p = write_plot_config(tmp_path / "cfg.json", cfg)

    assert win.import_config_json(p)
    assert win.selected_source_keys() == ["doc0:0"]
    assert win._bins.value() == 71


def test_a_template_and_an_export_differ_in_exactly_one_thing(win, tmp_path):
    """The distinction worth stating: an EXPORT is exact (source keys included), a
    TEMPLATE is portable. Both go through the same codec, which is why the template store
    gets the version refusal for free."""
    win.set_sources(["apo −500 mV"])
    exported = plot_config_to_json(win.config())
    templated = plot_config_to_json(win.config().as_template())

    assert exported["selection"]["source_keys"] == ["doc0:0"]
    assert templated["selection"]["source_keys"] == []
    assert exported["style"] == templated["style"] or (
        exported["style"]["label_overrides"] != templated["style"]["label_overrides"])


def test_undo_and_redo_are_on_the_keyboard(win):
    """A configuration undo the user has to hunt for in a control column is not the undo
    anybody's hands expect. Redo is registered twice ON PURPOSE — see
    :meth:`PlotsWindow._build_config_group` for which platform gives which — and the
    registration is deduped, so this also pins that nothing was bound twice (Qt answers
    an ambiguous overload by doing nothing at all)."""
    from PySide6 import QtGui

    keys = [sc.key().toString() for sc in win.findChildren(QtGui.QShortcut)]
    assert len(keys) == len(set(keys)), f"a sequence was registered twice: {keys}"
    assert QtGui.QKeySequence(QtGui.QKeySequence.Undo).toString() in keys
    assert "Ctrl+Shift+Z" in keys, "the sequence every other app spells Redo"


def test_every_shipped_template_survives_the_codec(win):
    """The built-ins are :class:`PlotConfig` values rather than stored JSON, so nothing
    forces them through the reader in normal use — which is exactly how one comes to name
    a field that no longer exists. This forces them."""
    for name, tpl in PlotsWindow._BUILTIN_TEMPLATES.items():
        assert plot_config_from_json(plot_config_to_json(tpl)) == tpl, name
        assert tpl.as_template() == tpl, f"{name} carries something a template may not"
        win.apply_template(tpl)          # and it installs without complaint
        assert win._identity == win.config().identity, (
            f"{name} left the identity anchor pointing at something else")


def test_recalling_a_template_is_one_undo_step(win):
    """The three features meet here: a template is installed through ``apply_config``,
    whose contract is that naming a whole configuration IS the run gesture — so it lands
    on the stack like any other change and one Ctrl+Z takes the figure back."""
    before = win.config()
    back, _ = win._history.depth()

    win._on_template_selected("Publication figure (light)")
    assert win._style.background == "light"
    assert win._history.depth()[0] == back + 1, "one step, not one per control written"

    assert win.undo_config()
    assert win.config() == before, "and one step back out of the whole template"


def test_importing_a_config_is_one_undo_step(win, tmp_path):
    """Same argument as the template, and the same seam — which is the point of both
    going through ``apply_config``."""
    cfg = replace(win.config(), histogram=replace(win.config().histogram, bins=88))
    p = write_plot_config(tmp_path / "cfg.json", cfg)
    before = win.config()
    back, _ = win._history.depth()

    assert win.import_config_json(p)
    assert win._bins.value() == 88
    assert win._history.depth()[0] == back + 1

    assert win.undo_config() and win.config() == before


# ================================================================================================
# S6-03 — THE RENAME'S MIGRATION: "density (area = 1)" is a value users have on disk
# ================================================================================================
#
# The Plots window's y-quantity is a COMBO ITEM'S LABEL, and that label is what
# ``plot_config_to_json`` writes into every sidecar and every stored template. So renaming
# the item — which S6-03 did, because "density (area = 1)" was false on the drawn picture in
# every unit but seconds and unfixable on a log axis — renamed a persisted value.
#
# The failure that makes these tests mandatory was DRIVEN, not imagined. ``_set_config``
# installs the normalization with ``QComboBox.setCurrentText``, which on a non-editable combo
# is a silent no-op for text the combo does not carry. Handed a file naming a mode this build
# does not offer, the window reported ``import_config_json() -> True``, left the combo on
# ``count``, made ``config().histogram.normalization`` read ``count``, said nothing in
# Details, and dropped ``_template_combo`` to "— manual —" because
# ``_sync_template_selection`` compares configs and the two spellings never match. A user
# would watch their density figure become a count figure, and their own template's name
# vanish as they picked it, with no message anywhere.
#
# It was first closed at the READ boundary alone, and that was one door short. The migration
# now lives IN THE VALUE (``HistogramConfig.__post_init__``), because a config does not have
# to come from a file: ``apply_config`` and ``apply_template`` were both measured taking a
# ``PlotConfig`` carrying the old spelling and silently drawing something else —
# ``apply_config`` because ``setCurrentText`` is a no-op for an item the combo has not got,
# and ``apply_template`` because ``_installable`` cannot tell yesterday's word for the mode
# you are in from a mode this build has never heard of, and resolves both away to whatever
# is on screen. Canonicalising in the constructor puts ONE point on every door: the file
# reader builds the dataclass, the template store goes through that reader, an undo entry
# comes from ``config()``, and a script's ``dataclasses.replace`` calls it too.
#
# The rename also BUMPED ``PLOT_CONFIG_VERSION`` to 2. That is not about this build reading
# old files (it still does, ``MIN_READ`` is 1); it is about an older build reading a file
# this one writes, which no table here can fix — it would accept the file, fail to recognise
# "density", and draw counts. The bump turns that into a refusal naming the version.

def _v1_sidecar_payload(cfg: PlotConfig) -> dict:
    """``cfg`` as a PRE-RENAME build wrote it: version 1, ``density (area = 1)``.

    Reconstructed rather than checked in, because it must stay a faithful v1 file as the
    rest of the format evolves — a checked-in file cannot grow the key a section gains
    tomorrow. The checked-in REAL one
    (``fixtures/plot_config_v1_pre_rename.json``, exported by the frozen pre-rename build
    and read by ``test_a_real_v1_sidecar_from_the_pre_rename_build_still_opens``) is the
    other half of the same guarantee: this payload keeps up with the format, that file
    proves the format's claim about history is true of an actual artifact.

    ``version`` is written back down to 1 explicitly. ``plot_config_to_json`` stamps
    :data:`PLOT_CONFIG_VERSION`, which is 2 since the rename, and a "v1 sidecar" that
    declares 2 would be quietly testing nothing.
    """
    payload = plot_config_to_json(replace(
        cfg, histogram=replace(cfg.histogram, normalization="density")))
    payload["histogram"]["normalization"] = "density (area = 1)"
    payload["version"] = 1
    return payload


def test_a_v1_sidecar_naming_the_old_density_still_selects_density(win, tmp_path):
    """The whole point: the file must not turn a density figure into a count figure.

    ``normalization()`` AND ``config()`` are both asserted because they are different
    surfaces — the first is what is drawn, the second is what the template comparison, the
    undo diff and the next export all read.
    """
    p = tmp_path / "old.json"
    p.write_text(json.dumps(_v1_sidecar_payload(win.config())), encoding="utf-8")

    assert win.import_config_json(p)
    assert win.normalization() == "density"
    assert win.config().histogram.normalization == "density"


def test_a_v1_payload_still_reads_after_the_bump(tmp_path):
    """``MIN_READ_VERSION`` is the promise that yesterday's file still opens, and the
    rename did not touch it: only ``PLOT_CONFIG_VERSION`` moved."""
    payload = _v1_sidecar_payload(PlotConfig())
    assert payload["version"] == 1
    assert PLOT_CONFIG_MIN_READ_VERSION == 1
    assert plot_config_from_json(payload).histogram.normalization == "density"


def test_a_real_v1_sidecar_from_the_pre_rename_build_still_opens(win):
    """THE PROMISE, KEPT AGAINST AN ACTUAL ARTIFACT rather than a payload this suite built.

    ``fixtures/plot_config_v1_pre_rename.json`` was exported by the frozen pre-rename
    ``plots_panel.py`` (sha256 337d1080…) driving a real window over the owner's
    982-event ``events_nm_0005.csv``: version 1, ``"normalization": "density (area = 1)"``,
    41 bins, log x, an exponential fit and the title "figure 3b". Opened on that build it
    drew a density whose bar heights hash to ``82809ccdfb559400`` (y_sum 2316.5122829391244);
    opened here, with the same events, it draws the same array to the bit.

    A hand-built dict cannot make this claim, because the thing under test is whether the
    file a user already has on disk — every key of it, spelled as that build spelled it —
    survives both the rename and the version bump.
    """
    p = Path(__file__).with_name("fixtures") / "plot_config_v1_pre_rename.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["version"] == 1, "premise: the fixture is genuinely a v1 file"
    assert payload["histogram"]["normalization"] == "density (area = 1)"

    assert win.import_config_json(p)
    assert win.normalization() == "density"
    assert win.config().histogram.normalization == "density"
    assert win._bins.value() == 41 and win._hist_logx.isChecked()
    assert win.config().style.title == "figure 3b"
    assert win.config().fit.applied == "exponential"


def test_the_rename_bumped_the_version_so_an_older_build_refuses_by_name(tmp_path,
                                                                        monkeypatch):
    """THE OTHER DIRECTION, which ``_LEGACY_NORMS`` cannot reach.

    A build that predates the rename accepts a v1 file, does not recognise ``density``,
    installs it as nothing, and draws counts — measured on the frozen pre-rename tree with
    a real sidecar this build wrote: ``import_config_json`` returned True, the y axis said
    "count", and it summed 982 events as integers where the author had saved the density
    summing to 1628.66. Only the version field can stop that, so the rename bumped it.

    The older reader is simulated by putting its own ``PLOT_CONFIG_VERSION`` back — the
    reader looks the constant up when it runs — so the sentence asserted here is character
    for character the one that build produces.
    """
    p = write_plot_config(tmp_path / "today.json", replace(
        PlotConfig(), histogram=HistogramConfig(column="dwell_s", normalization="density")))
    assert json.loads(p.read_text())["version"] == 2 == PLOT_CONFIG_VERSION, (
        "every sidecar this build writes declares the vocabulary it was written in")

    monkeypatch.setattr(pp, "PLOT_CONFIG_VERSION", 1)
    with pytest.raises(PlotConfigVersionError) as exc:
        read_plot_config(p)
    assert "version 2" in str(exc.value) and "versions 1 to 1" in str(exc.value)
    assert "Update NanoScope" in str(exc.value)


def test_a_stored_template_written_before_the_rename_recalls_under_its_own_name(win):
    """THE HALF A NORMALIZATION-ONLY ASSERTION WOULD MISS, and the half the user sees.

    Translating at the reader is what makes the recalled config compare EQUAL to the window
    it just configured. Fix it at the widget instead and the mode would be right while the
    combo dropped to "— manual —" the instant the user picked their own template.
    """
    old = _v1_sidecar_payload(replace(
        win.config(), histogram=replace(win.config().histogram, bins=41)))
    PlotsWindow._store_templates({"my dwell figure": old})
    win._load_templates_into_combo()

    win._on_template_selected("my dwell figure")

    assert win.normalization() == "density"
    assert win._bins.value() == 41
    assert win._template_combo.currentText() == "my dwell figure", (
        "the template that just configured this window must still compare equal to it")


def test_the_setter_translates_the_old_name_and_only_the_old_name(win):
    """``density (area = 1)`` names a mode that still exists, and it is what every script,
    notebook and pre-S6-03 test says — so it is translated, not refused. Anything else
    unmatched still raises WITH THE OFFERED ITEMS NAMED, which is the invariant
    ``nanoscope.gui.combos`` exists to hold and the reason this setter has never used
    ``setCurrentText``."""
    win.set_normalization("density (area = 1)")
    assert win.normalization() == "density"

    with pytest.raises(ValueError) as excinfo:
        win.set_normalization("density (1/s)")
    assert "density (1/s)" in str(excinfo.value)
    assert "'count', 'density', 'probability per bin'" in str(excinfo.value)
    assert win.normalization() == "density", "a refused setter changes nothing"


def test_a_normalization_this_build_cannot_offer_is_refused_out_loud(win):
    """A name the combo has not got installs as NOTHING — ``setCurrentText`` is a silent
    no-op for text a non-editable combo does not carry — so the window has to say so.

    The half that was always right: the rest of the template lands and ``config()`` goes
    on reporting the mode that is DRAWN, because it reads the widget. The half that was
    missing: the request was swallowed. It is now recorded by the one writer
    (``_set_config`` -> ``_NormRefusal``) and re-stated in Details on every draw, naming
    both the refused quantity and the ones on offer — the same courtesy
    ``set_normalization`` does when it raises, which is the difference between a refusal
    and a shrug."""
    win.set_normalization("count")
    alien = replace(win.config(),
                    histogram=replace(win.config().histogram,
                                      normalization="quantile mass", bins=57))

    win.apply_template(alien)

    assert win._bins.value() == 57, "the rest of the template still installs"
    assert win.normalization() == "count"
    assert win.config().histogram.normalization == "count", (
        "config() must report the mode that is drawn, not the one that was asked for")
    said = win.details_text()
    assert "quantile mass" in said, "the refused quantity is not named anywhere"
    assert "count, density, probability per bin" in said, "the offered ones are not named"
    assert win._norm_refusal.asked == "quantile mass"


def test_the_public_in_process_door_is_refused_out_loud_too(win):
    """``apply_config`` does NOT pass ``_installable`` — a script or a notebook holding a
    ``PlotConfig`` arrives straight at the writer. That is exactly why the record is kept
    at the writer and not at ``_installable``: a fix bolted onto the latter closes the
    template door and leaves the public one open, which is the mistake S6-03 made once
    already (see ``test_apply_template_with_the_old_spelling_still_draws_a_density``)."""
    win.set_normalization("count")
    cfg = win.config()

    win.apply_config(replace(cfg, histogram=replace(cfg.histogram, bins=61,
                                                    normalization="density (1/s)")))

    assert win._bins.value() == 61
    assert win.normalization() == "count"
    assert "density (1/s)" in win.details_text()


def test_the_template_name_does_not_stand_over_a_y_quantity_that_did_not_land(win):
    """THE LOST TELL. The combo means "these settings ARE that template", and a recall
    that could not install the y quantity has to knock it back to "— manual —".

    ``_sync_template_selection`` decides that by comparing ``_template_would_give`` —
    which runs ``_installable`` — against ``config()``. While ``_installable`` resolved an
    unofferable normalization back to the one on screen, those two AGREED, so the
    comparison that exists to catch "this template no longer describes what you see" was
    answered with the substitution instead of the request, and the name stood over a
    window drawing something else. Measured on the owner's 982-event file: the bins came
    across (47), the y axis read ``count``, and the combo went on reading the template's
    own name — where the build before that clause dropped it to "— manual —".

    The sync is called directly because IT is the comparison under test: nudging a real
    control to provoke the change hook would break the match by itself and prove nothing
    about the normalization.
    """
    stored = plot_config_to_json(replace(
        win.config(), histogram=replace(win.config().histogram, bins=47)))
    stored["histogram"]["normalization"] = "density (1/s)"
    PlotsWindow._store_templates({"colleague's template": stored})
    win._load_templates_into_combo()

    win._on_template_selected("colleague's template")
    win._sync_template_selection()

    assert win._bins.value() == 47, "premise: the recall really did half-land"
    assert win.normalization() == "count"
    assert win._template_combo.currentText() == PlotsWindow._TPL_MANUAL, (
        "the window is asserting that these settings ARE that template")
    assert "density (1/s)" in win.details_text()


def test_a_template_that_did_land_keeps_its_name(win):
    """The other side of the same comparison, and the reason the fix is not "drop the
    name whenever anything is resolved": a template whose every setting installed is
    still describing the window, and a combo that dropped to "— manual —" the instant the
    user picked their own template would be the pre-fix bug wearing the fix's name."""
    stored = plot_config_to_json(replace(
        win.config(), histogram=replace(win.config().histogram, bins=44,
                                        normalization="density")))
    PlotsWindow._store_templates({"my dwell figure": stored})
    win._load_templates_into_combo()

    win._on_template_selected("my dwell figure")
    win._sync_template_selection()

    assert win._bins.value() == 44 and win.normalization() == "density"
    assert win._template_combo.currentText() == "my dwell figure"
    assert win._norm_refusal.asked == ""
    assert "not a y quantity" not in win.details_text(), (
        "a pane that cries wolf is not read")


def test_the_refusal_stands_until_the_y_quantity_is_re_decided(win):
    """``_NormRefusal``'s whole rule, driven. Details is rewritten IN FULL by every draw,
    so a sentence printed once at the door would be gone with the next picture — the
    record is what makes the disclosure a statement about the window rather than an
    artefact of one gesture, exactly as ``_RefusalLedger`` does for refused events.

    An ordinary edit does not clear it (nobody re-decided the y quantity); a hand pick of
    the combo does; and an install that CAN honour the name clears it, because the one
    writer re-decides the record every time it runs."""
    cfg = win.config()
    win.apply_config(replace(cfg, histogram=replace(cfg.histogram,
                                                    normalization="quantile mass")))
    assert "quantile mass" in win.details_text()

    win.set_bins(52)                       # an ordinary edit, and a fresh draw
    assert win._bins.value() == 52
    assert "quantile mass" in win.details_text(), "the record did not survive the redraw"

    win._norm_combo.setCurrentIndex(1)     # a hand pick: not under `_updating`
    assert win.normalization() == "density"
    assert win._norm_refusal.asked == ""
    assert "quantile mass" not in win.details_text()

    # ...and an install that CAN be honoured clears it as surely as a hand pick.
    cfg = win.config()
    win.apply_config(replace(cfg, histogram=replace(cfg.histogram,
                                                    normalization="quantile mass")))
    assert win._norm_refusal.asked == "quantile mass"
    win.apply_config(replace(cfg, histogram=replace(cfg.histogram,
                                                    normalization="count")))
    assert win._norm_refusal.asked == ""
    assert "quantile mass" not in win.details_text()


def test_the_refusal_is_not_a_modal_and_does_not_move_a_single_bar(win, monkeypatch):
    """A name in somebody's file is not worth stopping the app for, and the invariant that
    outranks the whole round is that nothing DRAWN moves: the refused install leaves the
    same bars the window was already showing, to the bit."""
    said: list[str] = []
    for name in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            staticmethod(lambda *a, _n=name, **k: said.append(_n)))

    win.set_normalization("count")
    win._plot_btn.click()
    before = _drawn_heights(win)

    cfg = win.config()
    win.apply_config(replace(cfg, histogram=replace(cfg.histogram,
                                                    normalization="quantile mass")))

    assert said == [], "a foreign name in a file must not raise a dialog"
    assert np.array_equal(_drawn_heights(win), before)
    assert win.plot_item().getAxis("left").labelText == "count"


def _every_page(win) -> list[str]:
    """Every plot type this build offers, read off the combo rather than listed.

    A fifth page added to that combo has to speak the refusal too, and the way this file
    makes that unforgettable is to enumerate the control instead of a literal — the same
    trick ``test_the_migration_table_is_read_in_exactly_one_place`` plays on the table."""
    win.set_scatter_columns("dwell_s", "depth_pA")
    win.set_kde_columns("dwell_s", "depth_pA")
    win.set_feature_selection(["dwell_s", "depth_pA", "snr"])
    return [win._plot_type_combo.itemText(i)
            for i in range(win._plot_type_combo.count())]


def _draw_page(win, kind: str, normalization: str) -> str:
    """Install a config for page ``kind`` naming ``normalization``; return Details."""
    cfg = win.config()
    win.apply_config(replace(cfg, plot_type=kind,
                             histogram=replace(cfg.histogram,
                                               normalization=normalization)))
    if win._pca_result is None and kind == pp.PCA:
        win.run_pca()          # PCA draws nothing until it has been run once
    return win.details_text()


def test_a_refused_y_quantity_is_spoken_on_every_page_not_just_the_histogram(win):
    """THE CONFIGURATION NAMES A Y QUANTITY WHATEVER ITS PLOT TYPE, so every page has to
    say when one did not land.

    The sentence was rendered into the HISTOGRAM's notes list only. Driven end to end on
    a v2 sidecar whose ``plot_type`` is "Scatter" and whose normalization is
    ``density (1/s)``: ``import_config_json`` returned True, the window landed on
    Scatter, the record was set, Details said NOTHING and no dialog appeared — the one
    tell left was the template combo dropping to "— manual —" with no reason attached to
    it. A colleague's scatter config imported, edited and saved back therefore dropped a
    field of the sender's file on a path where nothing had ever said it was refused.

    Switching to Histogram did surface it, but nothing told the user to, and the page
    they are on when they re-save is the page their colleague's file chose."""
    for kind in _every_page(win):
        said = _draw_page(win, kind, "quantile mass")
        assert win._norm_refusal.asked == "quantile mass", kind
        line = [ln for ln in said.splitlines() if "not a y quantity" in ln]
        assert line, f"{kind} refused the y quantity in silence"
        assert "quantile mass" in line[0], kind
        assert "count, density, probability per bin" in line[0], (
            f"{kind} does not name what IS on offer, which is what makes it a refusal "
            f"rather than a shrug")


def test_no_page_claims_a_y_axis_it_is_not_drawing(win):
    """The same record, two sentences, because one of them would be FALSE on three of the
    four pages: a scatter's y is a column, so "y is drawn as count" there would trade one
    silence for one wrong statement. The histogram keeps the direct wording; the others
    name the histogram's setting and say this page is not using it."""
    for kind in _every_page(win):
        line = [ln for ln in _draw_page(win, kind, "quantile mass").splitlines()
                if "not a y quantity" in ln][0]
        if kind == pp.HISTOGRAM:
            assert "y is drawn as “count”" in line
        else:
            assert "y is drawn as" not in line, (
                f"{kind} is claiming an axis it does not draw")
            assert "the histogram's y quantity stays “count”" in line, kind


def test_no_page_cries_wolf_when_the_y_quantity_did_land(win):
    """The other side of it, on all four pages: a name this build CAN offer is installed
    and nothing is said anywhere. A pane that carries a refusal for a request that landed
    is not read the next time it carries a real one."""
    for kind in _every_page(win):
        said = _draw_page(win, kind, "density")
        assert win._norm_refusal.asked == "", kind
        assert "not a y quantity" not in said, kind


def test_the_window_is_saying_it_at_the_moment_the_sidecar_is_re_saved(win, tmp_path):
    """THE HARM, DRIVEN WHOLE: import a colleague's scatter config carrying a y quantity
    this build has not got, change one thing, save it back.

    The sidecar is NOT rewritten to carry the refused name forward. ``export_config_json``
    writes :meth:`PlotsWindow.config` — the settings actually on screen — and it is the
    same gesture ``save_image`` is, on the other half of one figure; a sidecar naming a y
    quantity the figure is not drawn in would be a second lie in place of the first. What
    the round owes the user is that the drop is DISCLOSED before they make it, which is
    what this pins: the refusal is on screen in the same pane the save reports into."""
    sender = plot_config_to_json(replace(
        win.config(), plot_type=pp.SCATTER,
        scatter=replace(win.config().scatter, x="dwell_s", y="depth_pA")))
    sender["histogram"]["normalization"] = "density (1/s)"
    src = tmp_path / "colleague.json"
    src.write_text(json.dumps(sender), encoding="utf-8")

    assert win.import_config_json(src) is True
    assert win._plot_type_combo.currentText() == pp.SCATTER, "premise: not a histogram"
    assert "density (1/s)" in win.details_text(), (
        "the sender's y quantity was dropped without a word on the page they chose")

    win.set_bins(52)                                   # the one thing the user changed
    out = tmp_path / "resaved.json"
    assert win.export_config_json(out) == out
    back = json.loads(out.read_text(encoding="utf-8"))

    assert back["histogram"]["bins"] == 52
    assert back["histogram"]["normalization"] == "count", (
        "the sidecar must describe the figure it was saved beside")
    assert "density (1/s)" in win.details_text(), (
        "…and the pane the save reports into must still be saying what was dropped")


def test_the_value_and_the_setter_read_the_migration_table_the_same_way(win):
    """ONE TABLE, ONE READING. The two readers had drifted over whitespace:
    ``set_normalization`` looked the name up stripped and ``HistogramConfig`` did not, so
    ``" density (area = 1) "`` — reachable from a hand-edited sidecar or QSettings entry,
    the app never writes padding itself — migrated through the setter and came back
    UNCHANGED from the value, where it then met a combo that could not match it either.

    Both now ask ``_canonical_norm``, and it strips for the LOOKUP only: an unmatched name
    is returned exactly as given, so a refusal can still quote what the file said."""
    for spelling in ("density (area = 1)", " density (area = 1) ",
                     "density (area = 1) ", "\tdensity (area = 1)\n"):
        assert HistogramConfig(normalization=spelling).normalization == "density", spelling
        win.set_normalization(spelling)
        assert win.normalization() == "density", spelling

    assert HistogramConfig(normalization="  quantile mass  ").normalization == \
        "  quantile mass  ", "a foreign name is refused, not tidied"


def test_the_two_doors_match_a_current_spelling_by_different_rules_on_purpose(win):
    """WHAT ONE TABLE, ONE READING DOES *NOT* MAKE AGREE — pinned so the docstring above
    ``_canonical_norm`` cannot drift back into claiming it does.

    A LEGACY spelling now migrates through both doors however it is padded (the half the
    one reading closed). A CURRENT spelling that needs tidying does not, and that is a
    different mechanism: ``set_normalization`` resolves through ``combos.choose_combo_item``
    — strip, case-fold, unique prefix — because a bare string is a caller TYPING, while
    ``_set_config`` matches with ``QComboBox.findText``, which is exact, because a whole
    ``PlotConfig`` out of a file is not typing. Two shapes of caller, two answers, and
    neither of them is silence: the exact door REFUSES OUT LOUD, naming what it was
    handed.

    If someone gives the install door the lenient rule, this test goes red — and that is
    the moment ``_canonical_norm``'s docstring has to be rewritten with it."""
    for spelling in (" density ", "DENSITY", "Density", "dens"):
        win.set_normalization("count")
        win.set_normalization(spelling)
        assert win.normalization() == "density", f"the setter is lenient: {spelling!r}"

        win.set_normalization("count")
        cfg = win.config()
        win.apply_config(replace(cfg, histogram=replace(cfg.histogram,
                                                        normalization=spelling)))
        assert win.normalization() == "count", f"the install door is exact: {spelling!r}"
        assert f"“{spelling}”" in win.details_text(), (
            f"{spelling!r} was refused in silence, which is the one answer neither door "
            f"is allowed to give")

    # ...and the half that IS closed: padding no longer splits the legacy spelling.
    win.set_normalization("count")
    cfg = win.config()
    win.apply_config(replace(cfg, histogram=replace(
        cfg.histogram, normalization=" density (area = 1) ")))
    assert win.normalization() == "density"
    assert win._norm_refusal.asked == ""


def test_the_migration_table_is_read_in_exactly_one_place():
    """PINNED BY MECHANISM, because "two readers of one table" is the defect itself and a
    third would be added the same way the second was. ``_LEGACY_NORMS`` may be NAMED only
    where it is defined and inside ``_canonical_norm``; every other door asks the
    function."""
    import ast

    tree = ast.parse(Path(pp.__file__).read_text(encoding="utf-8"))
    readers = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Name) and inner.id == "_LEGACY_NORMS":
                    readers.add(node.name)
    assert readers == {"_canonical_norm"}, (
        f"_LEGACY_NORMS is read in {sorted(readers)}; it is read in one place")


def _drawn_heights(w) -> np.ndarray:
    """The bar heights actually on the plot item — the only assertion that cannot be
    satisfied by a control that merely reads right."""
    for item in w.plot_item().listDataItems():
        y = item.getData()[1]
        if y is not None and len(y):
            return np.asarray(y, float)
    raise AssertionError("nothing is drawn")


def test_apply_config_with_the_old_spelling_still_draws_a_density(win):
    """THE IN-PROCESS DOOR, and the one no GUI gesture reaches — which is why it went
    unnoticed: Import JSON, templates and undo all pass through the file reader, so only a
    script or a notebook holding a ``PlotConfig`` can arrive here.

    Measured before the migration moved into the value: this call left the combo on
    ``count``, made ``config()`` report ``count``, labelled the y axis ``count`` and drew
    raw integers (1, 4, 3, 17, 18 against the density's 1.66, 6.63, 4.98, 28.19, 29.85 on
    the owner's file) — no exception, nothing in Details, nothing in the banner.
    ``_set_config`` writes the mode with ``setCurrentText``, a silent no-op for an item the
    combo has not got.

    The heights are asserted, not just the labels: a fix that moved the combo without
    moving the numbers would be the same bug wearing the right hat.
    """
    win.set_normalization("count")
    win._plot_btn.click()
    counts = _drawn_heights(win)

    cfg = win.config()
    win.apply_config(replace(cfg, histogram=replace(cfg.histogram,
                                                    normalization="density (area = 1)")))

    assert win.normalization() == "density"
    assert win.config().histogram.normalization == "density"
    assert win.plot_item().getAxis("left").labelText.startswith("density")
    assert "y = density" in win.details_text()
    heights = _drawn_heights(win)
    assert not np.array_equal(heights, counts), "the bars are a density, not the counts"
    assert np.allclose(counts, np.round(counts)), "premise: the count mode drew integers"
    assert not np.allclose(heights, np.round(heights)), "the density mode did not"


def test_apply_template_with_the_old_spelling_still_draws_a_density(win):
    """THE SECOND IN-PROCESS DOOR, which fails for the OPPOSITE reason and is the reason
    the fix is not a clause in ``_installable``.

    ``apply_template`` does go through ``_installable``, and that method's whole job is to
    replace a name this window cannot select with the one on screen. It cannot tell
    "yesterday's word for density" from "a mode from a build that does not exist yet", so
    before the value canonicalised itself it resolved the old spelling away too — measured:
    combo left on ``count``. Closing only this door would have left ``apply_config``, the
    public one, exactly as broken.
    """
    win.set_normalization("count")
    template = replace(win.config(),
                       histogram=replace(win.config().histogram, bins=37,
                                         normalization="density (area = 1)"))

    win.apply_template(template)

    assert win._bins.value() == 37, "premise: the template really was applied"
    assert win.normalization() == "density"
    assert win.config().histogram.normalization == "density"


def test_the_value_itself_cannot_carry_the_old_spelling():
    """WHERE THE MIGRATION LIVES, stated at the value rather than at any door.

    Every door in this module ends up constructing one of these, so pinning the
    constructor pins all of them at once — including doors nobody has written yet. The
    equality assertion is the half the template comparison and the undo diff rest on: two
    configs that name the same mode in two spellings have to BE the same config.
    """
    assert HistogramConfig(normalization="density (area = 1)").normalization == "density"
    assert replace(HistogramConfig(), normalization="density (area = 1)") == \
        HistogramConfig(normalization="density")
    assert PlotConfig(histogram=HistogramConfig(normalization="density (area = 1)")) == \
        PlotConfig(histogram=HistogramConfig(normalization="density"))

    # A NAME THAT IS GENUINELY FOREIGN IS LEFT EXACTLY AS GIVEN. A value object is not the
    # place to judge what a widget can offer: a newer build's mode name has to survive the
    # dataclass so `_installable` can resolve it against the live combo and
    # `set_normalization` can refuse it with the offered items named.
    assert HistogramConfig(normalization="quantile mass").normalization == "quantile mass"


def test_the_translation_table_is_the_only_thing_that_knows_the_old_name(win, tmp_path,
                                                                        monkeypatch):
    """PINNED BY MECHANISM: empty ``_LEGACY_NORMS`` and EVERY door must forget together.

    Three places that must agree is how this defect was born — the reader knew the old
    spelling, the setter knew it, and the two install doors did not. If a second hard-coded
    copy of "density (area = 1)" ever reappears at one of these doors, that door goes on
    working with the table emptied and this test goes red.
    """
    monkeypatch.setattr(pp, "_LEGACY_NORMS", {})

    assert HistogramConfig(normalization="density (area = 1)").normalization == \
        "density (area = 1)", "the value's translation reads the table, it does not inline it"

    p = tmp_path / "old.json"
    p.write_text(json.dumps(_v1_sidecar_payload(win.config())), encoding="utf-8")
    assert plot_config_from_json(json.loads(p.read_text())).histogram.normalization == \
        "density (area = 1)", "the reader has no private copy of the table"

    with pytest.raises(ValueError):
        win.set_normalization("density (area = 1)")


def test_a_blank_font_family_round_trips_through_the_json_sidecar(win, tmp_path):
    """S6-01's other half: "" is a VALUE, and the codec has to keep treating it as one.

    The fix gave the blank a row of its own in the Style panel's Font combo, which is a
    widget change — but the reason it could stay a widget change is that ``""`` already
    survives the sidecar as itself. Pinned here so a future "helpfully" written codec
    (``font_family or DEFAULT``, or omitting empty strings to keep the file tidy) cannot
    turn "the application's own font" into a concrete family nobody chose.
    """
    w = win
    assert w._style.font_family == "", "premise: no family has been chosen"

    p = write_plot_config(tmp_path / "blank-font.json", w.config())
    payload = json.loads(p.read_text())
    assert payload["style"]["font_family"] == "", (
        "the blank is written out as itself, not dropped and not filled in")
    assert read_plot_config(p).style == w.config().style

    assert w.import_config_json(p)
    assert w._style.font_family == ""
    assert w._font_combo.currentIndex() == 0
    assert w._font_combo.currentData() == ""
    assert w._font_combo.currentText() == pp._FONT_DEFAULT_LABEL

    # A SIDECAR WRITTEN BEFORE THIS FIELD EXISTED — the key simply absent. The codec is
    # field-driven off the dataclass defaults, so it lands back on "" and the file still
    # loads; a pre-fix config must not become unreadable, nor arrive naming a font.
    del payload["style"]["font_family"]
    old = tmp_path / "no-font-key.json"
    old.write_text(json.dumps(payload))
    assert w.import_config_json(old)
    assert w._style.font_family == ""
    assert w._font_combo.currentIndex() == 0


# ================================================================================================
# THE STATES A DRAW NEVER REACHES, and the six losses beside the y quantity
# ================================================================================================

def _colleague_payload(win, kind: str) -> dict:
    """A sidecar from somebody else's build: a foreign y quantity plus six other things.

    Six MORE than the y quantity, one of each kind the round trip can lose — a fit this
    build cannot fit, a column these events have not got, a colour channel that is not on
    offer, a source this session never loaded, a style key the format has no field for,
    and a whole top-level block it has never heard of."""
    payload = plot_config_to_json(replace(
        win.config(), plot_type=kind,
        scatter=replace(win.config().scatter, x="dwell_s", y="depth_pA"),
        kde=replace(win.config().kde, x="dwell_s", y="depth_pA")))
    payload["histogram"]["normalization"] = "density (1/s)"
    payload["histogram"]["column"] = "conductance_nS"
    payload["scatter"]["colour_by"] = "pore_id"
    payload["fit"]["kind"] = "stretched exponential"
    payload["selection"]["source_keys"] = ["doc0:0", "a run this session never loaded"]
    payload["style"]["halo_width_px"] = 3
    payload["their_own_block"] = {"lab": "somewhere else"}
    return payload


def test_a_page_that_cannot_draw_still_says_what_the_configuration_asked_for(win):
    """AN EARLY RETURN THAT SKIPS THE DISCLOSURE IS THE BUG, NOT THE PAGE.

    Every empty state in this window goes through ``_message``, which replaced Details
    wholesale — so a window that could not draw said why it could not draw and NOTHING
    about a configuration it had just been handed and could not honour. Driven here by
    unticking every source, which puts all four pages on the one shared early return:
    measured on a tree with only the ``_message`` line reverted, the sentence is absent on
    4 of 4 pages and present on 4 of 4 with it.

    Enumerated off the plot-type combo, like its sibling above, so a fifth page cannot be
    forgotten — and this is the half that WAS forgotten, because the shipped page walk
    presses Run PCA on the user's behalf and so never sees an empty one."""
    pages = _every_page(win)
    for kind in pages:
        _draw_page(win, kind, "quantile mass")
        win._updating = True
        for i in range(win._source_list.count()):
            win._source_list.item(i).setCheckState(QtCore.Qt.Unchecked)
        win._updating = False
        win._on_sources_changed()
        said = win.details_text()
        assert "rows to plot" in said or "Tick a source" in said, (
            f"premise: {kind} is not in an empty state, so this proves nothing")
        assert "quantile mass" in said, (
            f"{kind} drawing nothing said nothing about the y quantity it was refused")
        # the instruction is still there — the disclosure is added, not substituted
        assert said.splitlines()[0].strip(), kind
        win._updating = True
        for i in range(win._source_list.count()):
            win._source_list.item(i).setCheckState(QtCore.Qt.Checked)
        win._updating = False
        win._on_sources_changed()


def test_the_pca_page_speaks_in_the_state_every_import_actually_lands_in(win, tmp_path):
    """A CONFIGURATION CANNOT CARRY A PCA RESULT, so this state is not a corner: it is
    where every import onto the PCA page lands, and it is where the user is standing when
    they decide whether to save the file back.

    ``_draw_pca`` returns at its first line while ``_pca_result`` is None and the refusal
    was injected 22 lines later, so the page that round 4 scored as speaking was speaking
    only after Run PCA — a gesture the user has not made and the shipped page walk makes
    for them. Measured before the fix on this same file: import returned True, the record
    read ``density (1/s)``, Details held the eleven-word instruction and nothing else, and
    the re-save wrote ``count`` over the sender's file with 0 dialogs.

    NO ``run_pca()`` HERE, deliberately. That call is what hid this."""
    payload = plot_config_to_json(replace(win.config(), plot_type=pp.PCA))
    payload["histogram"]["normalization"] = "density (1/s)"
    src = tmp_path / "colleague_pca.json"
    src.write_text(json.dumps(payload), encoding="utf-8")

    assert win.import_config_json(src) is True
    assert win._plot_type_combo.currentText() == pp.PCA
    assert win._pca_result is None, (
        "premise: an imported config leaves the PCA page in its EMPTY state")
    assert "Run PCA" in win.details_text(), "premise: this is the empty state"
    assert "density (1/s)" in win.details_text(), (
        "the page an imported PCA config lands on said nothing about what it dropped")

    win.set_title("re-saved by the recipient")
    out = tmp_path / "resaved.json"
    assert win.export_config_json(out) == out
    assert json.loads(out.read_text())["histogram"]["normalization"] == "count"
    assert "density (1/s)" in win.details_text(), (
        "…and it must still be saying it in the pane the save reports into")


def test_an_import_says_how_many_settings_it_could_not_take_and_names_them(win, tmp_path):
    """THE SIX LOSSES BESIDE THE Y QUANTITY, WHICH WERE ALL SILENT.

    Driven before the fix on a sidecar carrying seven droppable things: the re-save
    rewrote ``fit.kind`` to ``none``, ``histogram.column`` to the one on screen,
    ``scatter.colour_by`` to ``None`` and ``selection.source_keys`` to the one loaded run,
    and dropped two keys the format has no field for — seven losses, of which exactly ONE
    (the y quantity) was disclosed anywhere, and 0 dialogs appeared.

    The export contract is not what changed: ``export_config_json`` writes the settings on
    screen, deliberately. What changed is that the user is told, on whatever page their
    colleague's file chose, before they touch Save."""
    for kind in _every_page(win):
        src = tmp_path / "colleague.json"
        src.write_text(json.dumps(_colleague_payload(win, kind)), encoding="utf-8")
        assert win.import_config_json(src) is True

        names = win._config_losses.names
        assert len(names) == 7, (kind, names)
        assert {n.split(" ")[0] for n in names} == {
            "fit.kind", "histogram.column", "histogram.normalization",
            "scatter.colour_by", "selection.source_keys", "style.halo_width_px",
            "their_own_block.lab"}, (kind, names)

        line = [ln for ln in win.details_text().splitlines() if "did not take" in ln]
        assert line, f"{kind} lost seven settings and said nothing"
        assert "colleague.json" in line[0], "the user has to know WHICH file"
        assert "7 settings" in line[0], (
            "the COUNT is the number a user acts on, and it is stated whole")
        for name in names:
            assert name in line[0], (
                f"{kind}: seven losses fit under the cap, so all seven must be NAMED — "
                f"a value the user cannot find in their own file is half a disclosure")
        assert "stretched exponential" in line[0] and "conductance_nS" in line[0], (
            "a lost value has to be quoted, or the user cannot find it in their file")
        assert "a run this session never loaded" in line[0], (
            "a dropped list entry names the entry, not the whole list")
        assert "doc0:0" not in line[0], (
            "…and NOT the entry that survived: a user must not have to diff two lists")

        win.set_bins(51)                                # the one thing the user changed
        out = tmp_path / "resaved.json"
        assert win.export_config_json(out) == out
        back = json.loads(out.read_text(encoding="utf-8"))
        assert back["histogram"]["bins"] == 51
        assert "halo_width_px" not in back["style"] and "their_own_block" not in back
        assert "did not take" in win.details_text(), (
            f"{kind}: the pane the save reports into stopped saying it")


def test_a_faithful_round_trip_does_not_cry_wolf(win, tmp_path):
    """NOTHING LOST, NOTHING SAID — including across the one MIGRATION this format has.

    The count is measured against the payload as this build DECODES it, so a v1 file's
    ``density (area = 1)`` is already ``density`` on both sides of the comparison and the
    rename is not reported as a loss; the envelope keys are identical there for the same
    reason, so a v1 file reports nothing even though its ``version`` is 1 and the re-save
    writes 2. A disclosure that fires on a file this build understood completely would be
    worth less than no disclosure at all."""
    for kind in _every_page(win):
        win.set_plot_type(kind)
        src = tmp_path / "mine.json"
        assert win.export_config_json(src) == src

        assert win.import_config_json(src) is True
        assert win._config_losses.names == (), (kind, win._config_losses.names)
        assert "did not take" not in win.details_text(), kind

        legacy = json.loads(src.read_text(encoding="utf-8"))
        legacy["version"] = PLOT_CONFIG_MIN_READ_VERSION
        legacy["histogram"]["normalization"] = "density (area = 1)"
        old = tmp_path / "v1.json"
        old.write_text(json.dumps(legacy), encoding="utf-8")

        assert win.import_config_json(old) is True
        assert win._norm_combo.currentText() == "density", (
            "premise: the migration ran, so there is something that COULD be miscounted")
        assert win._config_losses.names == (), (
            f"{kind}: the v1 rename was counted as a loss")
        assert "did not take" not in win.details_text(), kind


def test_the_losses_sentence_stands_exactly_while_the_files_state_is_on_screen(win, tmp_path):
    """SCOPED TO THE IMPORT'S LINEAGE, so it outlives neither the file nor its own truth.

    This test used to pin "cleared by the one writer on every other install", including
    the redo that puts the imported state BACK — and that rule was S7-FIX1: an undo that
    landed on the imported state (bit-for-bit, ``config()`` equal) erased both
    disclosures, and a re-save then wrote over the colleague's file with 0 dialogs. The
    rule now is the one :class:`_ConfigLosses` states: the sentence is spoken while the
    configuration on screen DESCENDS FROM the import, whichever writer put it there.

    So: an undo that steps back PAST the import hides it (the pre-import state is not
    one the file produced); a redo that steps forward onto the file's state shows it
    again; a template recall opens a new origin and hides it; undoing that recall shows
    it again; another import re-takes it."""
    src = tmp_path / "colleague.json"
    src.write_text(json.dumps(_colleague_payload(win, pp.HISTOGRAM)), encoding="utf-8")
    assert win.import_config_json(src) is True
    assert len(win._config_losses.names) == 7
    assert "did not take" in win.details_text()

    win.undo_config()
    assert win._config_losses.names == (), (
        "the pre-import state is not one the file produced")
    assert "did not take" not in win.details_text()

    win.redo_config()
    assert len(win._config_losses.names) == 7, (
        "the redo put the file's own state back on screen, so the sentence about that "
        "state is exactly true again -- whichever writer re-installed it")
    assert "did not take" in win.details_text()

    imported = win.config()
    win.apply_template(replace(imported, histogram=replace(imported.histogram, bins=41)))
    assert win._bins.value() == 41, "premise: the recall moved something, so it is a step"
    assert win._config_losses.names == (), "a template recall left it standing"
    assert "did not take" not in win.details_text()

    win.undo_config()
    assert win.config() == imported, "premise: back on the file's own state"
    assert len(win._config_losses.names) == 7, (
        "undoing the recall is back on the file's state; the sentence returns with it")

    # A recall of a template EQUAL to the state on screen is still an eraser (it opens a
    # new origin) but pushes no step, so the next Undo steps past the import instead.
    win.apply_template(win.config())
    assert win._config_losses.names == (), "an identical template recall left it standing"

    assert win.import_config_json(src) is True
    assert len(win._config_losses.names) == 7, "another import re-takes the record"


def test_no_draw_path_can_return_early_without_going_through_the_one_funnel():
    """THE MECHANISM THE PAGE-BY-PAGE FIX DID NOT HAVE.

    Speaking the refusal on four pages is a list; every early return going through
    ``_message`` is a rule, and only the rule survives a fifth page or a twelfth empty
    state. This walks the AST of the five methods that can end a draw and asserts that
    every ``return`` in them is immediately preceded by a ``self._message(...)`` call in
    the same block — which is what makes "``_message`` carries the disclosure" equal to
    "every state carries the disclosure".

    Eleven returns across the five methods at the time of writing; the exact count is not
    asserted, because the rule is what matters and a twelfth is welcome as long as it says
    something first."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(pp))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "PlotsWindow")
    wanted = {"_redraw_current", "_draw_histogram", "_draw_scatter", "_draw_kde",
              "_draw_pca"}
    seen, silent, total = set(), [], 0

    def walk(fn_name, body):
        nonlocal total
        for i, st in enumerate(body):
            for sub in ("body", "orelse", "finalbody"):
                if hasattr(st, sub):
                    walk(fn_name, getattr(st, sub))
            if isinstance(st, ast.Return):
                total += 1
                prev = body[i - 1] if i else None
                if prev is None or "self._message(" not in ast.unparse(prev):
                    silent.append((fn_name, st.lineno))

    for fn in cls.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in wanted:
            seen.add(fn.name)
            walk(fn.name, fn.body)

    assert seen == wanted, f"a draw path was renamed away from this guard: {wanted - seen}"
    assert total >= 11, "premise: the draw paths still have early returns to check"
    assert not silent, (
        f"these returns end a draw without going through _message, so whatever the "
        f"window was asked for and could not honour is not said in that state: {silent}")

    # …and the funnel really does ask for the notes, rather than this being a guard on a
    # method that has stopped carrying them.
    body = inspect.getsource(pp.PlotsWindow._message).split('"""')[-1]
    assert "_incoming_notes()" in body
