"""S7-FIX1 / S7-SK1 — an import's two disclosures stand for as long as the configuration
on screen DESCENDS FROM that import, whichever writer put it there.

THE HOLE. ``_NormRefusal`` and ``_ConfigLosses`` exist so a user who opens a colleague's
sidecar, changes one thing and saves it back is told first that parts of that file did
not land. Both were "re-decided by the one writer on every install", and two writers
re-install the window's OWN state: ``undo_config -> _install_history -> _set_config``
and ``_on_time_unit_changed -> _rebase -> _set_config``. Measured on the owner's
982-event -700 mV table with a v2 sidecar naming ``density (1/s)`` and 10**9 bins: after
the import both sentences were in Details; one ``set_bins(31)`` and one Undo later the
window was bit-for-bit back in the imported state (``config()`` equal to the post-import
value) and both sentences were gone; Export JSON… over the colleague's file then wrote
``count`` and 500 with 0 dialogs. The Time-unit combo -- which the app's own S6-03
density note tells the user to move -- reached the same clearing without touching Undo.

THE RULE, pinned here gesture by gesture: the records are tagged with the ORIGIN the
install opened on the history (:meth:`_ConfigHistory.open_origin`) and spoken while the
state on screen carries that origin. An undo, a redo or a rebase opens none; an import,
a template recall or a script's ``apply_config`` opens one. The seventeen-gesture
survey from the skeptic's ``s5_erasers.py`` is the last test.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import json
import os
from dataclasses import replace

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.session import Session
from nanoscope.core.sources import PlotSource
from nanoscope.gui import plots_panel as pp
from nanoscope.gui.plots_panel import PlotConfig, PlotsWindow, _ConfigHistory, plot_config_to_json

REFUSAL = "is not a y quantity this build offers"
LOSSES = "did not take"


def _events(n: int = 300, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "dwell_s": 2e-5 + rng.exponential(2.94e-4, n),
        "depth_pA": np.abs(rng.normal(40.0, 6.0, n)),
        "rise_s": np.abs(rng.normal(1e-5, 2e-6, n)),
        "frac_depth": np.abs(rng.normal(0.30, 0.05, n)),
        "accepted": np.ones(n, dtype=bool),
        "filter_limited": np.zeros(n, dtype=bool),
    })


def _source(key: str, label: str, *, seed: int, colour: str) -> PlotSource:
    return PlotSource(key=key, label=label, df=_events(seed=seed),
                      provenance={"fs": 250_000.0, "min_dwell_s": 2e-5,
                                  "detector": "schmitt_hysteresis"},
                      colour=colour, doc_id=key.split(":")[0], channel=0,
                      module_id="detect.threshold", calibration_fingerprint=(1.0, 0.0))


def _sources() -> list[PlotSource]:
    return [_source("doc0:0", "−700 mV", seed=1, colour="#0072B2"),
            _source("doc1:0", "−600 mV", seed=2, colour="#D55E00")]


@pytest.fixture
def win(qapp, qtbot):
    w = PlotsWindow(Session())
    qtbot.addWidget(w)
    w.set_data(_sources())
    w.set_plot_type("Histogram")
    w.set_histogram_column("dwell_s")
    w.set_time_unit("ms")
    return w


def _colleague_file(win, tmp_path):
    """A sidecar from somebody else's build: a y quantity this build has not got (the
    refusal) plus a bin count the spin cannot take (a loss)."""
    payload = plot_config_to_json(win.config())
    payload["histogram"]["normalization"] = "density (1/s)"
    payload["histogram"]["bins"] = 10 ** 9
    f = tmp_path / "colleague.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    return f


def _both(win) -> tuple[bool, bool]:
    d = win.details_text()
    return REFUSAL in d, LOSSES in d


def _imported(win, tmp_path):
    f = _colleague_file(win, tmp_path)
    assert win.import_config_json(f) is True
    assert _both(win) == (True, True), "premise: the import discloses both"
    return f


# ================================================================================================
# the two writers that re-install the window's OWN state
# ================================================================================================

def test_an_undo_that_lands_back_on_the_imported_state_keeps_both_disclosures(win, tmp_path):
    """S7-FIX1 verbatim: import, one change, Undo. The window is in the state the file
    produced; the two sentences that describe that state are still on the page."""
    _imported(win, tmp_path)
    after_import = win.config()
    win.set_bins(31)
    assert _both(win) == (True, True)

    assert win.undo_config() is True
    assert win.config() == after_import, "premise: bit-for-bit back on the imported state"
    assert _both(win) == (True, True), (
        "the undo re-installed the file's own state and erased the sentences about it")
    assert win._norm_refusal.asked == "density (1/s)"
    assert win._config_losses.names and "histogram.bins" in win._config_losses.names[0]


@pytest.mark.parametrize("unit", ["s", "µs"])
def test_a_time_unit_change_keeps_both_disclosures(win, tmp_path, unit):
    """S7-SK1: the Time-unit combo moves the analysis identity, so its handler re-installs
    through ``_rebase -> _set_config`` -- the window's own state, corrected, and not a
    re-decision of anything the file asked for. The app's own density note is what
    tells the user to make this exact move."""
    _imported(win, tmp_path)
    win._time_unit.setCurrentText(unit)
    assert win._time_unit.currentText() == unit
    assert _both(win) == (True, True), f"Time unit = {unit} erased the disclosure"


def test_a_redo_that_reinstalls_the_imported_state_shows_them_again(win, tmp_path):
    """Undo PAST the import lands on a state the file did not produce, so the sentences
    go; Redo lands back on the file's state, so they come back -- whichever writer."""
    _imported(win, tmp_path)
    assert win.undo_config() is True          # steps back past the import itself
    assert _both(win) == (False, False), (
        "the pre-import configuration is not one the file produced")
    assert win._norm_refusal.asked == "" and win._config_losses.names == ()

    assert win.redo_config() is True
    assert _both(win) == (True, True), "the file's state is back on screen; so is its account"


def test_a_template_recall_hides_them_and_undoing_the_recall_shows_them_again(win, tmp_path):
    """The documented eraser, and its undo. A recall installs a configuration from
    another origin, so the sentences about the file are not spoken over it; stepping
    back off the recall puts the file's state back and the sentences with it.

    The template MOVES something (bins), so the recall is a history step of its own to
    undo. A recall of a template equal to the state on screen is still an eraser (the
    survey below pins that) but pushes no step, so Undo after it steps past the import."""
    _imported(win, tmp_path)
    after_import = win.config()
    win.apply_template(replace(after_import,
                               histogram=replace(after_import.histogram, bins=41)))
    assert win._bins.value() == 41, "premise: the recall moved something"
    assert _both(win) == (False, False), "a template recall left the sentences standing"

    assert win.undo_config() is True
    assert win.config() == after_import, "premise: back on the file's own state"
    assert _both(win) == (True, True)


def test_another_import_re_decides_both_records(win, tmp_path):
    """A second file replaces the first file's account with its own -- here, a faithful
    file, so nothing is left to say."""
    _imported(win, tmp_path)
    mine = tmp_path / "mine.json"
    assert win.export_config_json(mine) == mine
    assert win.import_config_json(mine) is True
    assert _both(win) == (False, False)
    assert win._config_losses.names == () and win._norm_refusal.asked == ""


def test_a_hand_move_of_the_normalise_combo_clears_the_refusal_only(win, tmp_path):
    """The user re-decided the y quantity; nothing about the other losses changed."""
    _imported(win, tmp_path)
    win._norm_combo.setCurrentText("density")      # not under `_updating`: a hand pick
    assert _both(win) == (False, True)


def test_a_scripts_apply_config_over_the_import_hides_them_like_a_template(win, tmp_path):
    """``apply_config`` is the public door with no file behind it -- another origin."""
    _imported(win, tmp_path)
    cfg = win.config()
    win.apply_config(replace(cfg, histogram=replace(cfg.histogram, bins=41)))
    assert win._bins.value() == 41
    assert _both(win) == (False, False)


# ================================================================================================
# the stack itself: origins ride with the entries
# ================================================================================================

def test_the_history_carries_an_origin_with_every_entry():
    a, b, c = PlotConfig(), PlotConfig(plot_type=pp.SCATTER), PlotConfig(plot_type=pp.KDE)
    h = _ConfigHistory(a)
    o0 = h.origin
    assert h.note(b) is True and h.origin == o0, "an ordinary edit inherits its origin"

    o1 = h.open_origin()
    assert o1 != o0 and h.origin == o1, "open: the state on screen is ALREADY the new origin"
    assert h.note(c) is True and h.origin == o1

    assert h.undo() == b and h.origin == o0
    assert h.redo() == c and h.origin == o1
    assert h.undo() == b and h.undo() == a and h.origin == o0

    o2 = h.open_origin()
    assert h.note(a) is False, "an install that moves nothing is still an install..."
    assert h.origin == o2, "...and it changes what the state on screen descends from"


# ================================================================================================
# the seventeen-gesture survey (docs/sweep7/scratch/SKEPTIC/s5_erasers.py), pinned
# ================================================================================================

def _plot_button(win):
    from PySide6 import QtWidgets
    return next(b for b in win.findChildren(QtWidgets.QPushButton)
                if b.text().strip().lower().startswith("plot"))


def _tick_second_source(win):
    from PySide6.QtCore import Qt
    win._source_list.item(1).setCheckState(Qt.Checked)


_GESTURES = {
    "nothing (control)":            (lambda w: None,                                   (True, True)),
    "bins 500 -> 31":               (lambda w: w.set_bins(31),                         (True, True)),
    "histogram column -> depth_pA": (lambda w: w.set_histogram_column("depth_pA"),     (True, True)),
    "histogram column -> rise_s":   (lambda w: w.set_histogram_column("rise_s"),       (True, True)),
    "Time unit ms -> s":            (lambda w: w._time_unit.setCurrentText("s"),       (True, True)),
    "Time unit ms -> us":           (lambda w: w._time_unit.setCurrentText("µs"),      (True, True)),
    "log x on":                     (lambda w: w._hist_logx.setChecked(True),          (True, True)),
    "log y on":                     (lambda w: w._hist_logy.setChecked(True),          (True, True)),
    "plot type -> Scatter":         (lambda w: w.set_plot_type("Scatter"),             (True, True)),
    "Normalise combo -> density":   (lambda w: w.set_normalization("density"),         (False, True)),
    "tick 2nd source":              (_tick_second_source,                              (True, True)),
    "accepted-only on":             (lambda w: w._accepted_check.setChecked(True),     (True, True)),
    "press Plot":                   (lambda w: _plot_button(w).click(),                (True, True)),
    "Refresh (set_data)":           (lambda w: w.set_data(_sources()),                 (True, True)),
    "undo":                         (lambda w: (w.set_bins(31), w.undo_config()),      (True, True)),
    "redo":                         (lambda w: (w.set_bins(31), w.undo_config(),
                                                w.redo_config()),                      (True, True)),
    "rename a source row":          (lambda w: w._source_list.item(0).setText("Apo"),  (True, True)),
    "title text":                   (lambda w: w.set_title("figure 3"),                (True, True)),
    "template recall":              (lambda w: w.apply_template(w.config()),           (False, False)),
}


@pytest.mark.parametrize("gesture", list(_GESTURES))
def test_which_gestures_erase_an_imports_disclosures(win, tmp_path, gesture):
    """Each gesture from a FRESH import. Everything that merely edits or re-installs the
    window's own state preserves both sentences; a hand move of the Normalise combo
    clears the refusal only; a template recall hides both."""
    _imported(win, tmp_path)
    act, expected = _GESTURES[gesture]
    act(win)
    assert _both(win) == expected, gesture
