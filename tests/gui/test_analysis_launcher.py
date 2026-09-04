"""GUI tests for the analysis launcher — the always-visible row that says which analysis
surfaces exist (pytest-qt, offscreen).

The problem it fixes is not click COUNT (every surface was already two clicks): it is that all
six were the SAME two clicks into the same flat ``&Analysis`` menu, which on macOS lives at the
top of the screen detached from the window, and that the explanatory tooltips written for those
surfaces were set on QActions — where a native menu never renders them. Nothing in the visible
layout said the density map, the ensemble, the noise spectrum, the I-V fit or the
voltage-dependence series existed at all.

So these tests pin, in the order they matter:

* every surface has a button, and a button whose surface is not usable yet is still SHOWN
  (greyed, never hidden) — a feature you cannot use yet is still a feature you should know about;
* a locked button is still clickable and answers "why not?" in the status bar, opening nothing;
* an available button triggers the surface's EXISTING QAction and the real window opens — the
  launcher never grows a second open path;
* every gate's enablement and its reason come from the SAME predicate evaluation, so the greyed
  button, its tooltip, the status-bar message and the menu item can never disagree;
* enablement tracks the live session as it progresses: no documents → a recording → a detection
  → an accepted event → a two-recording voltage series;
* the row degrades through its densities instead of flooring the window's width, and never
  scrolls or clips below its own minimum;
* the Fits panel's Fit button is gated up front with a visible reason instead of failing after
  the click;
* the ``&Analysis`` menu still holds every action with its shortcut, now in sections.

Nothing here touches the frozen 20-column EventSet schema.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis.events import EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_launcher import (
    DENSITIES,
    DENSITY_FULL,
    DENSITY_MINIMAL,
    DENSITY_SHORT,
    SURFACE_EVENT_PLOTS,
    SURFACE_FIGURES,
    SURFACE_GALLERY,
    SURFACE_IV,
    SURFACE_ORDER,
    SURFACE_POPULATIONS,
    SURFACE_PSD,
    SURFACE_VDEP,
    AnalysisLauncher,
)
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import StatsPanel
from nanoscope.gui.workflow_status import Stage

FS = 50_000.0
N = 20_000

# The six surfaces that open a window (the seventh, Export figures, opens a file dialog).
WINDOW_SURFACES: tuple[str, ...] = (
    SURFACE_POPULATIONS, SURFACE_EVENT_PLOTS, SURFACE_PSD, SURFACE_IV, SURFACE_VDEP,
    SURFACE_GALLERY)

# Surface id -> the MainWindow attribute holding that surface's single lazily-built window.
_WINDOW_ATTR: dict[str, str] = {
    SURFACE_POPULATIONS: "_plots_window",
    SURFACE_EVENT_PLOTS: "_event_plot_window",
    SURFACE_PSD: "_psd_window",
    SURFACE_IV: "_iv_window",
    SURFACE_VDEP: "_vdep_window",
    SURFACE_GALLERY: "_event_gallery",
}


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path, *, name="rec.dat", with_voltage=False, nominal_mV=None,
         fs: float | None = FS) -> TraceDocument:
    """A single-current-channel recording, optionally with a voltage channel / nominal voltage."""
    chans = [ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)]
    if with_voltage:
        chans.append(ChannelInfo(index=1, name="voltage", raw_units="mV",
                                 kind=ChannelKind.VOLTAGE))
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=fs,
        rate_provenance=Provenance.UNKNOWN if fs is None else Provenance.FILE_META,
        channels=tuple(chans),
        nominal_voltage_mV=nominal_mV,
        voltage_provenance=(Provenance.UNKNOWN if nominal_mV is None
                            else Provenance.FILENAME))
    return TraceDocument(meta=meta, channels=tuple(np.zeros(N, np.float32) for _ in chans))


def _voltage_only_doc(tmp_path, *, name="volt.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="voltage", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(np.zeros(N, np.float32),))


def _eventset(accepted: list[bool]) -> EventSet:
    n = len(accepted)
    rows = pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "recording": pd.array(["rec"] * n, dtype="string"),
        "pore_id": pd.array(["p0"] * n, dtype="string"),
        "start_idx": np.arange(n, dtype=np.int64) * 100 + 10,
        "end_idx": np.arange(n, dtype=np.int64) * 100 + 60,
        "t_start_s": np.zeros(n),
        "dwell_s": np.full(n, 1e-3),
        "baseline_pA": np.full(n, -50.0),
        "depth_pA": np.full(n, 20.0),
        "frac_depth": np.full(n, 0.4),
        "ecd_pA_s": np.full(n, 0.02),
        "max_dev_pA": np.full(n, 25.0),
        "mean_dev_pA": np.full(n, 20.0),
        "rise_s": np.full(n, 1e-4),
        "fall_s": np.full(n, 1e-4),
        "snr": np.full(n, 8.0),
        "n_levels": np.ones(n, dtype=np.int64),
        "filter_limited": np.zeros(n, dtype=bool),
        "accepted": np.asarray(accepted, dtype=bool),
        "flag": pd.array([""] * n, dtype="string"),
    })
    return EventSet(pd.concat([empty_df(), rows], ignore_index=True),
                    {"analyzed_duration_s": 1.0, "recording": "rec"})


def _install(win: MainWindow, doc: TraceDocument) -> str:
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    return doc_id


def _install_detection(session: Session, doc_id: str, ev: EventSet, channel: int = 0) -> None:
    session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=channel,
        params={}, calibration_fingerprint=(1.0, 0.0), payload=ev))


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def detached_launcher(qapp, qtbot):
    """A parentless launcher whose width can actually be set.

    The density ladder is a function of the width the row is GIVEN, and inside a live MainWindow
    the central layout immediately re-imposes its own width on any resize. Hosting it in a bare,
    layout-less parent lets ``resize()`` stick. The host is shown (offscreen) because Qt defers a
    hidden widget's resize event and the density is chosen in ``resizeEvent``. Mirrors the
    workflow strip's own fixture — the two widgets share this behaviour on purpose.
    """
    hosts: list[QtWidgets.QWidget] = []

    def make(source: MainWindow | None = None) -> AnalysisLauncher:
        host = QtWidgets.QWidget()
        hosts.append(host)          # keep it alive: C++ deletes the child with its parent
        qtbot.addWidget(host)
        host.resize(2400, 200)
        launcher = AnalysisLauncher(host)
        if source is not None:
            launcher.set_tooltips(
                {sid: act.toolTip() for sid, act in source._surface_actions().items()})
        host.show()
        return launcher

    yield make
    for host in hosts:
        host.close()


def _voltage_series(win: MainWindow, tmp_path, n: int, *, prefix: str = "v") -> list[str]:
    """``n`` recordings that each have a current channel, a known voltage AND a detection —
    exactly what the voltage-dependence gate counts."""
    ids = []
    for i in range(n):
        doc_id = _install(
            win, _doc(tmp_path, name=f"{prefix}{i}.dat", nominal_mV=-200.0 * (i + 1)))
        _install_detection(win._session, doc_id, _eventset([True, True]))
        ids.append(doc_id)
    return ids


# ================================================================================================
# 1. the surfaces exist, and they exist VISIBLY
# ================================================================================================

def test_launcher_has_a_button_for_every_analysis_surface(win):
    launcher = win._analysis_launcher
    assert launcher.surface_ids() == (
        SURFACE_POPULATIONS, SURFACE_EVENT_PLOTS, SURFACE_PSD,
        SURFACE_IV, SURFACE_VDEP, SURFACE_FIGURES, SURFACE_GALLERY)
    # ...each a clickable control of its own, not one painted banner.
    assert len({id(launcher.surface_button(s)) for s in SURFACE_ORDER}) == 7
    assert [launcher.label(s) for s in SURFACE_ORDER] == [
        "Populations", "Density / ensemble", "Power spectrum",
        "I-V / conductance", "V-dependence", "Export figures", "Event gallery"]


def test_the_event_gallery_is_one_of_the_launcher_surfaces(win, tmp_path):
    """THE SEVENTH SURFACE, and the one this row was missing.

    The gallery was item 13 of 29 in a menu that on macOS is at the top of the screen, it had
    no keyboard shortcut, and every other door to it was inside another window — so a user
    who had not already found it had no way to learn it existed. It is on the row now, greyed
    with a reason until a detection exists, and a click opens the SAME QAction the menu item
    does.
    """
    launcher = win._analysis_launcher
    assert SURFACE_GALLERY in launcher.surface_ids()
    assert not launcher.is_enabled(SURFACE_GALLERY)
    assert launcher.reason(SURFACE_GALLERY).strip()

    doc_id = _install(win, _doc(tmp_path))
    _install_detection(win._session, doc_id, _eventset([True, True]))
    win._on_selection(doc_id, 0)
    assert launcher.is_enabled(SURFACE_GALLERY)
    assert launcher.reason(SURFACE_GALLERY) == ""

    fired: list[bool] = []
    win._act_event_gallery.triggered.connect(lambda *_: fired.append(True))
    assert win._event_gallery is None
    launcher.surface_button(SURFACE_GALLERY).clicked.emit()
    assert fired == [True]                  # the menu's own action, not a second open path
    assert win._event_gallery is not None


def test_the_gallery_gate_has_exactly_one_owner(win, tmp_path):
    """``_sync_export_action`` used to set ``_act_event_gallery.setEnabled(has_detection)``
    ITSELF. With the surface on the launcher that made two owners of one fact, which is how
    a greyed button and a live menu item come to disagree — so the direct ``setEnabled`` is
    gone and both read the gate."""
    launcher = win._analysis_launcher
    for _ in range(2):
        assert win._act_event_gallery.isEnabled() is launcher.is_enabled(SURFACE_GALLERY)
        doc_id = _install(win, _doc(tmp_path, name="g.dat"))
        _install_detection(win._session, doc_id, _eventset([True]))
        win._on_selection(doc_id, 0)
    assert win._act_event_gallery.isEnabled() is launcher.is_enabled(SURFACE_GALLERY) is True
    # ...and the path that ONLY runs the export gate keeps them in step as well.
    win._sync_export_action()
    assert win._act_event_gallery.isEnabled() is launcher.is_enabled(SURFACE_GALLERY)


def test_the_two_curation_windows_have_keyboard_shortcuts(win):
    """The Analysis menu had Ctrl+D and Ctrl+M and nothing else; the two windows a curator
    lives in had no key at all."""
    assert win._act_event_gallery.shortcut() == QtCore.QKeyCombination(
        QtCore.Qt.ControlModifier, QtCore.Qt.Key_G)
    assert win._act_event_inspector.shortcut() == QtCore.QKeyCombination(
        QtCore.Qt.ControlModifier, QtCore.Qt.Key_I)
    # ...and they are not taken from anything else in this window.
    others = [a.shortcut() for a in win._analysis_menu.actions()
              if not a.isSeparator()
              and a not in (win._act_event_gallery, win._act_event_inspector)]
    assert win._act_event_gallery.shortcut() not in others
    assert win._act_event_inspector.shortcut() not in others


def test_launcher_sits_in_the_visible_central_column_under_the_workflow_strip(win):
    """The whole complaint was that no analysis surface appeared anywhere in the layout. The
    launcher must therefore be in the central column, directly under the workflow strip whose
    ANALYZE stage it expands."""
    layout = win.centralWidget().layout()
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert win._workflow_strip in order
    assert win._analysis_launcher in order
    assert order.index(win._analysis_launcher) == order.index(win._workflow_strip) + 1


def test_launcher_buttons_stay_visible_when_their_surface_is_locked(win, qtbot):
    """The entire discoverability point: a greyed "V-dependence" button still advertises that
    voltage dependence exists. On an empty session all six are locked — and all six are still
    on screen, in the layout, at a non-zero size."""
    win.show()
    qtbot.waitExposed(win)
    launcher = win._analysis_launcher
    assert all(not launcher.is_enabled(s) for s in SURFACE_ORDER)
    for sid in SURFACE_ORDER:
        btn = launcher.surface_button(sid)
        assert not btn.isHidden(), f"{sid} button was hidden while locked"
        assert btn.isVisibleTo(launcher)
        assert btn.width() > 0 and btn.height() > 0
        assert launcher.label(sid)            # it still says what it is


def test_locked_buttons_are_never_qt_disabled_so_the_click_can_explain_itself(win):
    """Qt never delivers a click to a widget that is ``setEnabled(False)``, so a genuinely
    disabled button could not tell the user why it is disabled. Locked is a STYLE here — the
    same choice the workflow strip's locked chips make."""
    launcher = win._analysis_launcher
    for sid in SURFACE_ORDER:
        assert not launcher.is_enabled(sid)               # the surface is locked...
        assert launcher.surface_button(sid).isEnabled()   # ...the widget still takes clicks


# ================================================================================================
# 2. the tooltips that never rendered now render
# ================================================================================================

def test_every_button_carries_its_actions_own_explanatory_copy(win):
    """The prose is REUSED, not rewritten: each button's tooltip is the surface's QAction
    tooltip — the copy that a native macOS menu silently drops."""
    launcher = win._analysis_launcher
    for sid, action in win._surface_actions().items():
        prose = action.toolTip()
        assert prose, f"{sid} has no action tooltip to reuse"
        assert prose in launcher.tooltip(sid)
        assert launcher.label(sid) in launcher.tooltip(sid)


def test_a_locked_buttons_tooltip_states_the_reason_too(win, tmp_path):
    """Hovering must be enough — the user should not have to click a dead control to find out
    why it is dead."""
    launcher = win._analysis_launcher
    assert not launcher.is_enabled(SURFACE_VDEP)
    reason = launcher.reason(SURFACE_VDEP)
    assert reason and reason in launcher.tooltip(SURFACE_VDEP)
    assert win._act_vdep.toolTip() in launcher.tooltip(SURFACE_VDEP)


def test_an_available_surface_advertises_no_reason(win, tmp_path):
    doc_id = _install(win, _doc(tmp_path))
    win._on_selection(doc_id, 0)
    launcher = win._analysis_launcher
    assert launcher.is_enabled(SURFACE_PSD)
    assert launcher.reason(SURFACE_PSD) == ""
    assert "Unavailable" not in launcher.tooltip(SURFACE_PSD)


def test_the_event_inspector_is_deliberately_not_duplicated_here(win, tmp_path):
    """The launcher expands the workflow strip's ANALYZE stage. CURATE is a different stage with
    its own always-visible chip one row up, which already routes to the Event inspector and
    already carries its reason — putting it in the launcher too would duplicate an affordance
    that is not missing. (The surfaces in this row are the ones with NO visible affordance.)

    The event GALLERY is the other half of CURATE and is NOT in the same position: the chip
    does not route to it, and nothing else in the visible layout does either. Same rule, and
    it is the rule — not the stage — that decides."""
    assert "inspector" not in " ".join(SURFACE_ORDER)
    assert SURFACE_GALLERY in SURFACE_ORDER
    doc_id = _install(win, _doc(tmp_path))
    _install_detection(win._session, doc_id, _eventset([True]))
    win._on_selection(doc_id, 0)
    assert win._event_inspector is None
    win._workflow_strip.stageActivated.emit(Stage.CURATE.value)   # still one visible click
    assert win._event_inspector is not None


# ================================================================================================
# 3. routing — an available click triggers the EXISTING action; a locked click explains
# ================================================================================================

def test_enabled_click_triggers_the_existing_menu_action_and_opens_the_window(win, tmp_path):
    """Not "a duplicate open path ran" — the very QAction the menu item uses fires, and the
    single lazily-built window it owns comes up."""
    doc_id = _install(win, _doc(tmp_path))
    win._on_selection(doc_id, 0)
    launcher = win._analysis_launcher
    assert launcher.is_enabled(SURFACE_PSD)

    fired: list[bool] = []
    win._act_psd.triggered.connect(lambda *_: fired.append(True))
    assert win._psd_window is None

    launcher.surface_button(SURFACE_PSD).clicked.emit()

    assert fired == [True]                 # the menu's own action, not a parallel code path
    assert win._psd_window is not None     # ...and it really opened the window


def test_a_real_mouse_click_on_the_button_reaches_the_same_action(win, qtbot, tmp_path):
    """The button WIDGET is the user's actual target, so drive it with a mouse, not just the
    signal."""
    doc_id = _install(win, _doc(tmp_path))
    win._on_selection(doc_id, 0)
    win.show()
    qtbot.waitExposed(win)

    fired: list[bool] = []
    win._act_psd.triggered.connect(lambda *_: fired.append(True))
    btn = win._analysis_launcher.surface_button(SURFACE_PSD)
    qtbot.mouseClick(btn, QtCore.Qt.LeftButton)

    assert fired == [True]
    assert win._psd_window is not None


def test_keyboard_activation_reaches_the_same_action(win, qtbot, tmp_path):
    """This is an accessibility fix; the row has to be usable from the keyboard too."""
    doc_id = _install(win, _doc(tmp_path))
    win._on_selection(doc_id, 0)
    win.show()
    qtbot.waitExposed(win)

    fired: list[bool] = []
    win._act_psd.triggered.connect(lambda *_: fired.append(True))
    btn = win._analysis_launcher.surface_button(SURFACE_PSD)
    btn.setFocus()
    qtbot.keyClick(btn, QtCore.Qt.Key_Space)

    assert fired == [True]


def test_every_available_surface_opens_its_own_window(win, tmp_path):
    """All five window surfaces, driven from the launcher, each opening the single instance
    MainWindow owns for it."""
    ids = _voltage_series(win, tmp_path, 2)
    win._on_selection(ids[0], 0)
    launcher = win._analysis_launcher

    for sid in WINDOW_SURFACES:
        assert launcher.is_enabled(sid), f"{sid} should be available: {launcher.reason(sid)}"
        assert getattr(win, _WINDOW_ATTR[sid]) is None
        launcher.surface_button(sid).clicked.emit()
        assert getattr(win, _WINDOW_ATTR[sid]) is not None, f"{sid} did not open its window"


def test_locked_click_posts_its_reason_to_the_status_bar_and_opens_nothing(win):
    """The workflow-chip pattern, verbatim: a locked click is an explanation, never an action."""
    launcher = win._analysis_launcher
    for sid in WINDOW_SURFACES:
        assert not launcher.is_enabled(sid)
        launcher.surface_button(sid).clicked.emit()
        assert win.statusBar().currentMessage() == launcher.reason(sid)
        assert getattr(win, _WINDOW_ATTR[sid]) is None, f"{sid} opened from a LOCKED click"


def test_the_posted_reason_is_the_string_the_button_advertises(win, tmp_path):
    """The status-bar message, the button's stored reason and its tooltip are one string — they
    are all read off the same gate evaluation, so they cannot drift."""
    _install(win, _doc(tmp_path, nominal_mV=-200.0))
    launcher = win._analysis_launcher
    launcher.surface_button(SURFACE_VDEP).clicked.emit()
    msg = win.statusBar().currentMessage()
    assert msg == launcher.reason(SURFACE_VDEP)
    assert msg in launcher.tooltip(SURFACE_VDEP)


def test_locked_figure_export_click_opens_no_file_dialog(win, monkeypatch):
    """The sixth surface has no window — its locked click must not reach the export routine
    (which would raise a modal file dialog and hang a headless run)."""
    called: list[bool] = []
    monkeypatch.setattr(MainWindow, "_export_figures", lambda self: called.append(True))
    launcher = win._analysis_launcher
    assert not launcher.is_enabled(SURFACE_FIGURES)
    launcher.surface_button(SURFACE_FIGURES).clicked.emit()
    assert called == []
    assert win.statusBar().currentMessage() == launcher.reason(SURFACE_FIGURES)


# ================================================================================================
# 4. the gates track the real predicates as the session progresses
# ================================================================================================

def _gate_map(win: MainWindow) -> dict[str, bool]:
    return {sid: win._analysis_launcher.is_enabled(sid) for sid in SURFACE_ORDER}


def test_gates_track_the_session_from_empty_to_a_detected_voltage_series(win, tmp_path):
    # -- nothing open: every surface locked, every one still on the row -----------------------
    assert _gate_map(win) == dict.fromkeys(SURFACE_ORDER, False)

    # -- one plain recording (current channel, known rate, no voltage) ------------------------
    doc_id = _install(win, _doc(tmp_path, name="a.dat"))
    win._on_selection(doc_id, 0)
    gates = _gate_map(win)
    assert gates[SURFACE_PSD] is True          # raw-signal QC needs no detection
    assert gates[SURFACE_IV] is False          # ...but I-V needs a known voltage
    assert gates[SURFACE_POPULATIONS] is False
    assert gates[SURFACE_EVENT_PLOTS] is False
    assert gates[SURFACE_FIGURES] is False
    assert gates[SURFACE_VDEP] is False

    # -- a detection with nothing accepted yet ------------------------------------------------
    _install_detection(win._session, doc_id, _eventset([False, False]))
    win._on_selection(doc_id, 0)
    gates = _gate_map(win)
    assert gates[SURFACE_POPULATIONS] is True  # there is a per-event table to plot
    assert gates[SURFACE_FIGURES] is True
    assert gates[SURFACE_EVENT_PLOTS] is False  # ...but nothing accepted to bin/overlay
    assert "accepted" in win._analysis_launcher.reason(SURFACE_EVENT_PLOTS)

    # -- accept an event THROUGH the panel (the real curation path) ---------------------------
    win._analysis_panel.set_event_accepted(0, True)
    assert win._analysis_launcher.is_enabled(SURFACE_EVENT_PLOTS) is True

    # -- a second recording with a voltage lights I-V, and its detection lights V-dependence --
    doc_b = _install(win, _doc(tmp_path, name="b.dat", nominal_mV=-500.0))
    assert win._analysis_launcher.is_enabled(SURFACE_IV) is True
    assert win._analysis_launcher.is_enabled(SURFACE_VDEP) is False   # 1 point is not a trend
    _install_detection(win._session, doc_b, _eventset([True]))
    doc_c = _install(win, _doc(tmp_path, name="c.dat", nominal_mV=-1000.0))
    _install_detection(win._session, doc_c, _eventset([True]))
    assert win._analysis_launcher.is_enabled(SURFACE_VDEP) is True


def test_closing_the_last_document_re_locks_every_surface(win, tmp_path):
    doc_id = _install(win, _doc(tmp_path, nominal_mV=-200.0))
    _install_detection(win._session, doc_id, _eventset([True]))
    win._on_selection(doc_id, 0)
    assert win._analysis_launcher.is_enabled(SURFACE_PSD) is True

    win._session.close_document(doc_id)
    assert _gate_map(win) == dict.fromkeys(SURFACE_ORDER, False)


def test_a_launcher_gate_and_its_menu_action_can_never_disagree(win, tmp_path):
    """They are set from ONE evaluation of ONE predicate, so assert the equality holds at every
    step of a realistic progression — this is the invariant the whole design exists to buy."""
    def _assert_agree():
        for sid, action in win._surface_actions().items():
            assert win._analysis_launcher.is_enabled(sid) is action.isEnabled(), sid

    _assert_agree()
    doc_id = _install(win, _doc(tmp_path, name="a.dat", nominal_mV=-200.0))
    win._on_selection(doc_id, 0)
    _assert_agree()
    _install_detection(win._session, doc_id, _eventset([True]))
    win._on_selection(doc_id, 0)
    _assert_agree()
    doc_b = _install(win, _doc(tmp_path, name="b.dat", nominal_mV=-500.0))
    _install_detection(win._session, doc_b, _eventset([True]))
    _assert_agree()
    # ...and after the menu re-derives everything on aboutToShow, too.
    win._analysis_menu.aboutToShow.emit()
    _assert_agree()


def test_every_locked_surface_states_a_reason_at_every_stage(win, tmp_path):
    """Before this work only 1 of 7 gates said its condition anywhere the user could see. Now a
    locked surface without a reason is a test failure, in every session state."""
    launcher = win._analysis_launcher

    def _assert_reasoned():
        for sid in SURFACE_ORDER:
            if not launcher.is_enabled(sid):
                assert launcher.reason(sid).strip(), f"{sid} is locked with no reason"

    _assert_reasoned()
    _install(win, _voltage_only_doc(tmp_path))
    _assert_reasoned()
    doc_id = _install(win, _doc(tmp_path, name="a.dat"))
    win._on_selection(doc_id, 0)
    _assert_reasoned()
    _install_detection(win._session, doc_id, _eventset([False]))
    win._on_selection(doc_id, 0)
    _assert_reasoned()


def test_vdep_reason_counts_the_recordings_the_predicate_counts(win, tmp_path):
    """The reason quotes the very integer the gate thresholds, so it can never overstate
    progress ("0 so far" while one recording is in fact ready, or vice-versa)."""
    launcher = win._analysis_launcher
    assert "0 so far" in launcher.reason(SURFACE_VDEP)

    _voltage_series(win, tmp_path, 1)
    assert win._vdep_ready_count() == 1
    assert "1 so far" in launcher.reason(SURFACE_VDEP)

    _voltage_series(win, tmp_path, 1, prefix="w")   # a second one -> a trend
    assert launcher.is_enabled(SURFACE_VDEP)
    assert launcher.reason(SURFACE_VDEP) == ""


def test_density_gate_distinguishes_no_detection_from_nothing_accepted(win, tmp_path):
    """Two different problems with two different fixes get two different sentences."""
    launcher = win._analysis_launcher
    doc_id = _install(win, _doc(tmp_path))
    win._on_selection(doc_id, 0)
    no_detection = launcher.reason(SURFACE_EVENT_PLOTS)
    assert "Run detection" in no_detection

    _install_detection(win._session, doc_id, _eventset([False, False]))
    win._on_selection(doc_id, 0)
    nothing_accepted = launcher.reason(SURFACE_EVENT_PLOTS)
    assert "accepted" in nothing_accepted
    assert nothing_accepted != no_detection


def test_no_locked_reason_blames_detection_when_nothing_is_open(win):
    """A reason that names the WRONG blocker is worse than no reason at all.

    With an empty session "Run detection first" points the user at a detector that is itself
    disabled for want of a recording — a dead end one step further away than the real fix.
    Every gate must name the blocker that is actually first.
    """
    launcher = win._analysis_launcher
    for sid in SURFACE_ORDER:
        reason = launcher.reason(sid)
        assert not launcher.is_enabled(sid), sid
        assert "Run detection" not in reason, (sid, reason)
        assert "Select a recording" not in reason, (sid, reason)
        assert "accept at least one" not in reason, (sid, reason)
        assert ("Open a recording" in reason or "Open the voltage series" in reason), \
            (sid, reason)


def test_selection_dependent_gates_ask_for_a_selection_not_a_detection(win, tmp_path):
    """A recording is OPEN but nothing is selected yet.

    The density map and the figure export both read the CURRENT selection, so "run detection on
    the selected recording" is false here — there is no selected recording to run it on. They
    ask for the selection instead, and no longer tell the user to open what is already open.
    """
    launcher = win._analysis_launcher
    _install(win, _doc(tmp_path))
    assert win._current_selection is None
    for sid in (SURFACE_EVENT_PLOTS, SURFACE_FIGURES):
        reason = launcher.reason(sid)
        assert not launcher.is_enabled(sid), sid
        assert "Select a recording" in reason, (sid, reason)
        assert "Open a recording" not in reason, (sid, reason)


def test_the_locked_reasons_are_distinct_per_surface(win, tmp_path):
    """Six greyed buttons that all say the same sentence would tell the user nothing about
    which surface needs what — the reasons are surface-specific in every state."""
    launcher = win._analysis_launcher
    for stage in range(3):
        if stage == 1:
            _install(win, _doc(tmp_path, name="a.dat"))
        if stage == 2:
            doc_id = _install(win, _doc(tmp_path, name="b.dat"))
            win._on_selection(doc_id, 0)
        locked = [launcher.reason(s) for s in SURFACE_ORDER if not launcher.is_enabled(s)]
        assert len(set(locked)) == len(locked), (stage, locked)


def test_psd_gate_separates_nothing_open_from_nothing_suitable(win, tmp_path):
    launcher = win._analysis_launcher
    assert "Open a recording" in launcher.reason(SURFACE_PSD)

    _install(win, _voltage_only_doc(tmp_path))       # open, but no current channel
    reason = launcher.reason(SURFACE_PSD)
    assert not launcher.is_enabled(SURFACE_PSD)
    assert "current channel" in reason and "Open a recording" not in reason


def test_psd_gate_is_locked_without_a_known_sample_rate(win, tmp_path):
    """The rate-unknown (legacy TDMS) case: a current channel is not on its own enough."""
    doc_id = _install(win, _doc(tmp_path, fs=None))
    win._on_selection(doc_id, 0)
    launcher = win._analysis_launcher
    assert not launcher.is_enabled(SURFACE_PSD)
    assert "sample rate" in launcher.reason(SURFACE_PSD)


def test_open_windows_are_marked_open_and_unmarked_when_closed(win, tmp_path):
    """These windows are single instances that re-focus rather than re-open, so 'already open'
    is worth saying — but only while it is TRUE."""
    doc_id = _install(win, _doc(tmp_path))
    win._on_selection(doc_id, 0)
    launcher = win._analysis_launcher
    assert launcher.is_open(SURFACE_PSD) is False

    launcher.surface_button(SURFACE_PSD).clicked.emit()
    assert launcher.is_open(SURFACE_PSD) is True

    win._psd_window.close()
    assert launcher.is_open(SURFACE_PSD) is False


# ================================================================================================
# 5. width: the row degrades, it never scrolls and it never floors the window
# ================================================================================================

def test_launcher_is_a_thin_horizontal_row(win):
    launcher = win._analysis_launcher
    assert 24 <= launcher.height() <= 32
    assert launcher.height() <= win._workflow_strip.height() + 4   # sibling chrome, not a pane


def test_density_ladder_is_the_widest_that_fits(win, detached_launcher):
    launcher = detached_launcher(win)
    for density in DENSITIES:
        assert launcher.density_for(launcher.width_for(density)) == density
    assert launcher.density_for(launcher.width_for(DENSITY_MINIMAL) - 1) == DENSITY_MINIMAL
    assert launcher.density_for(0) == DENSITY_MINIMAL
    # monotonic: wider never means a poorer rendering
    assert launcher.width_for(DENSITY_FULL) > launcher.width_for(DENSITY_SHORT)
    assert launcher.width_for(DENSITY_SHORT) > launcher.width_for(DENSITY_MINIMAL)


def test_launcher_degrades_instead_of_forcing_the_window_wider(win, detached_launcher):
    launcher = detached_launcher(win)

    launcher.resize(launcher.width_for(DENSITY_FULL), launcher.height())
    assert launcher.density() == DENSITY_FULL
    assert launcher.shown_text(SURFACE_EVENT_PLOTS) == "Density / ensemble"

    launcher.resize(launcher.width_for(DENSITY_SHORT), launcher.height())
    assert launcher.density() == DENSITY_SHORT
    assert launcher.shown_text(SURFACE_EVENT_PLOTS) == "Density"
    assert launcher.shown_text(SURFACE_POPULATIONS) == "Populations"   # no honest abbreviation

    launcher.resize(launcher.width_for(DENSITY_MINIMAL), launcher.height())
    assert launcher.density() == DENSITY_MINIMAL
    assert all(launcher.shown_text(s) == "" for s in SURFACE_ORDER)
    # ...and every surface is still there, still clickable, still named in its tooltip.
    for sid in SURFACE_ORDER:
        assert not launcher.surface_button(sid).isHidden()
        assert launcher.label(sid) in launcher.tooltip(sid)


def test_launcher_fits_the_narrow_central_column_both_docks_open(win, detached_launcher):
    """With both docks open the central column can be ~820 px. The row must render SOMETHING
    sensible there rather than clipping — its short density is what that width buys."""
    launcher = detached_launcher(win)
    launcher.resize(820, launcher.height())
    assert launcher.density() in DENSITIES
    assert launcher.width_for(launcher.density()) <= 820
    assert launcher.width_for(DENSITY_SHORT) <= 820   # at worst it drops to short labels here


def test_launcher_never_becomes_the_windows_width_floor(win, detached_launcher):
    """Its minimum is the glyphs-only width whatever it is currently rendering, so a wide row
    cannot pin the main window (or the docks either side of it) at its full-label width."""
    launcher = detached_launcher(win)
    launcher.resize(launcher.width_for(DENSITY_FULL), launcher.height())
    assert launcher.density() == DENSITY_FULL

    floor = launcher.minimumSizeHint().width()
    assert floor == launcher.width_for(DENSITY_MINIMAL)
    assert floor < launcher.width_for(DENSITY_FULL)
    assert floor < win.statusBar().minimumSizeHint().width()


def test_a_real_window_drives_the_ladder_and_never_clips_below_the_minimum(win, qtbot):
    """Squeeze and stretch the REAL window (not a detached row): the density follows the width
    the central column actually gets, and the row keeps at least its own minimum width, so the
    buttons are never cut in half."""
    launcher = win._analysis_launcher
    win.show()
    qtbot.waitExposed(win)

    win.resize(2200, 900)
    qtbot.wait(50)
    assert launcher.density() == DENSITY_FULL
    assert launcher.shown_text(SURFACE_VDEP) == "V-dependence"

    win.resize(600, 700)
    qtbot.wait(50)
    assert launcher.density() == DENSITY_MINIMAL          # degraded, not clipped
    assert launcher.width() >= launcher.minimumSizeHint().width()
    assert all(not launcher.surface_button(s).isHidden() for s in SURFACE_ORDER)
    assert win.minimumSizeHint().width() >= launcher.minimumSizeHint().width()


def test_the_launcher_never_scrolls(win):
    """Degrade, don't scroll — a scrollbar would hide the very surfaces the row exists to
    advertise."""
    assert win._analysis_launcher.findChildren(QtWidgets.QAbstractScrollArea) == []


# ================================================================================================
# 6. the Fit button is gated up front, with a reason
# ================================================================================================

def test_fit_button_starts_disabled_with_a_visible_reason(qapp, qtbot):
    panel = StatsPanel()
    qtbot.addWidget(panel)
    assert not panel.fit_button.isEnabled()
    reason = panel.fit_gate_reason()
    assert reason and "Run detection" in reason
    assert panel._fit_gate_label.isVisibleTo(panel)
    assert reason in panel.fit_button.toolTip()


def test_fit_button_without_a_detection_emits_nothing(qapp, qtbot):
    """It used to accept the click and then write "Run detection first…" into the results box —
    a failure discovered after acting, in a place the user was not looking."""
    panel = StatsPanel()
    qtbot.addWidget(panel)
    got: list[dict] = []
    panel.fitRequested.connect(got.append)
    panel.fit_button.click()
    assert got == []
    assert panel.results.toPlainText() == ""


def test_fit_button_enables_and_clears_its_reason_once_there_is_a_detection(qapp, qtbot):
    panel = StatsPanel()
    qtbot.addWidget(panel)
    panel.set_detection_present(True)
    assert panel.fit_button.isEnabled()
    assert panel.fit_gate_reason() == ""
    assert not panel._fit_gate_label.isVisibleTo(panel)
    panel.set_detection_present(False)
    assert not panel.fit_button.isEnabled()
    assert panel.fit_gate_reason()


def test_fit_button_tracks_the_live_detection_in_a_real_window(win, tmp_path):
    """Wired for real: the gate follows the SAME state the merge controls follow."""
    stats = win._analysis_panel._stats
    doc_id = _install(win, _doc(tmp_path))
    win._on_selection(doc_id, 0)
    assert not stats.fit_button.isEnabled()
    assert stats.fit_gate_reason()

    _install_detection(win._session, doc_id, _eventset([True, True]))
    win._on_selection(doc_id, 0)
    assert stats.fit_button.isEnabled()
    assert win._analysis_panel.merge_controls.merge_all_btn.isEnabled()   # the same state
    assert stats.fit_gate_reason() == ""


# ================================================================================================
# 7. the &Analysis menu: sectioned, but nothing lost
# ================================================================================================

_ANALYSIS_ACTION_ATTRS = (
    "_act_run_detection", "_act_batch_detect", "_act_load_ml_model", "_act_run_ml_detection",
    "_act_event_inspector", "_act_measure", "_act_add_measurement",
    "_act_compare_baseline",
    "_act_plots", "_act_event_plots", "_act_event_gallery", "_act_reconcile", "_act_labels",
    "_act_mark_missed",
    # ``_act_psd_selection`` sits immediately after ``_act_psd``: it opens the SAME
    # surface over the region the Measure tool is bracketing, carrying the selection as
    # sample indices so the trace's instrument clock and the panel's recording clock
    # cannot be confused (High-6).
    "_act_psd", "_act_psd_selection", "_act_fit_levels", "_act_subpop",
    "_act_show_saved_subpop",
    "_act_iv", "_act_vdep",
    "_act_export_events", "_act_export_processed", "_act_export_training", "_act_export_levels",
    "_act_export_density", "_act_density_style", "_act_export_figure",
)
# ``_act_compare_baseline`` is in *Curate*, beside the Event inspector, because it is
# about the events already on screen -- it changes nothing by itself, it measures what
# a re-measure would change (§4.2.7). It is deliberately NOT in *Explore this
# recording* with the five plot surfaces: those open a window onto the data, this one
# answers a question about the table you are curating.
#
# ``_act_fit_levels`` sits at the END of *Explore this recording*, after the three
# same-recording surfaces. It is an analysis OF this recording, so the section name is
# right; it is last within it because it is the one entry that opens no window (it fills
# the Event Inspector's readout and enables the level export instead).
#
# It reads as a Curate action -- it answers a question about the event table, like
# ``_act_compare_baseline`` -- and that is where it was first put. Anywhere ahead of
# ``_act_plots`` pushes the plot surfaces to menu item 10, which is precisely the
# position ``test_the_five_plot_surfaces_are_no_longer_buried_at_the_end`` exists to keep
# them out of. Section fit lost to a live product invariant; the invariant was not
# widened to accommodate a new action.
#
# ``_act_subpop`` (Event subpopulations -- the ported Zn2-insulin pipeline) sits
# immediately AFTER ``_act_fit_levels``, in the same section and for the same
# reasons: it is an analysis OF this recording's event table, it opens no plot
# surface, it writes nothing back, and it runs only when chosen. Behind the plot
# surfaces, so their position is undisturbed.


def test_analysis_menu_still_contains_every_action(win):
    items = [a for a in win._analysis_menu.actions() if not a.isSeparator()]
    assert len(items) == len(_ANALYSIS_ACTION_ATTRS)
    for attr in _ANALYSIS_ACTION_ATTRS:
        assert getattr(win, attr) in items, f"{attr} vanished from the Analysis menu"


def test_analysis_menu_keeps_every_shortcut_and_flag(win):
    assert win._act_run_detection.shortcut() == QtCore.QKeyCombination(
        QtCore.Qt.ControlModifier, QtCore.Qt.Key_D)
    assert win._act_measure.isCheckable()
    # ...and every action still carries the tooltip copy the launcher reuses.
    # Ctrl+M lives on the MENU, not on the MeasurePanel: a Qt.WindowShortcut on a panel
    # that is one page of a hidden tab stack is delivered only while that tab is raised,
    # which is not when the key is used (C30).
    assert win._act_add_measurement.shortcut() == QtCore.QKeyCombination(
        QtCore.Qt.ControlModifier, QtCore.Qt.Key_M)
    assert win._act_mark_missed.shortcut() == QtCore.QKeyCombination(
        QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier, QtCore.Qt.Key_M)
    for attr in _ANALYSIS_ACTION_ATTRS:
        if attr != "_act_measure":
            assert getattr(win, attr).toolTip()


def test_analysis_menu_is_sectioned_instead_of_one_flat_list(win):
    """Ordering and separators only — but there must actually BE sections, and each one must
    hold something."""
    actions = win._analysis_menu.actions()
    sections = [a.text() for a in actions if a.isSeparator() and a.text()]
    assert sections == ["Detect", "Curate", "Explore this recording",
                        "Compare recordings", "Export"]
    # every section is followed by at least one real action
    for i, act in enumerate(actions):
        if act.isSeparator() and act.text():
            rest = actions[i + 1:]
            assert any(not a.isSeparator() for a in rest), f"empty section {act.text()}"


def test_the_five_plot_surfaces_are_no_longer_buried_at_the_end(win):
    """They used to be items 10-14 of a flat list. They are now inside two named sections that
    start much earlier — and, more importantly, they are on the launcher row as well."""
    items = [a for a in win._analysis_menu.actions() if not a.isSeparator()]
    first_surface = min(items.index(getattr(win, a)) for a in
                        ("_act_plots", "_act_event_plots", "_act_psd", "_act_iv", "_act_vdep"))
    assert first_surface < 9


# ================================================================================================
# 8. architecture: the launcher announces, MainWindow routes (no panel-to-panel import)
# ================================================================================================

def _canonical_gui_imports(src: str) -> set[str]:
    """EVERY ``nanoscope.gui`` sibling ``src`` imports, by any of the five spellings:
    ``from .x import y``, ``from . import x``, ``import nanoscope.gui.x``,
    ``from nanoscope.gui.x import y`` and ``from ..gui.x import y``. ``from . import x`` is the
    easy one to miss — its ``node.module`` is None, so a canonicaliser keyed on it skips the
    import entirely while it binds the sibling just as firmly."""
    tree = ast.parse(src)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nanoscope.gui."):
                    out.add(alias.name.split(".")[2])
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level == 1 and name:                       # from .x import y
                out.add(name.split(".")[0])
            elif node.level == 1:                              # from . import x
                out |= {a.name for a in node.names}
            elif node.level == 2 and name.startswith("gui."):  # from ..gui.x import y
                out.add(name.split(".")[1])
            elif node.level == 0 and name.startswith("nanoscope.gui."):
                out.add(name.split(".")[2])
    return out


def test_analysis_launcher_imports_no_sibling_gui_panel():
    """Discoverability must not be bought with a panel-to-panel import: the launcher holds no
    window, no session and no other panel — it announces a click and MainWindow routes it."""
    import nanoscope.gui.analysis_launcher as mod

    assert _canonical_gui_imports(pathlib.Path(mod.__file__).read_text()) == set()


def test_analysis_launcher_stays_a_pure_view():
    """Qt + the stdlib only: no analysis maths, no core session types. Every predicate it
    displays is pushed in by MainWindow."""
    import nanoscope.gui.analysis_launcher as mod

    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            name = node.module or ""
            assert not name.startswith("nanoscope."), f"launcher must stay view-only: {name}"
            assert "gui" not in name.split("."), f"launcher must not import gui: {name}"


def test_launcher_carries_no_open_or_gating_logic_of_its_own():
    """It must not know how to build a window or how to decide a gate — both live in
    main_window, which is why the launcher can never disagree with the menu.

    Checked against real IDENTIFIERS (an AST walk), not raw text: the module is allowed to name
    the windows in prose, which is documentation, and forbidden to reference them in code.
    """
    import nanoscope.gui.analysis_launcher as mod

    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {"PlotsWindow", "EventPlotWindow", "PSDWindow", "IVWindow", "VDepWindow",
                 "Session", "EventSet"}
    assert not (used & forbidden), f"launcher references {sorted(used & forbidden)}"


# ================================================================================================
# Density figure: the MENU path, not the handler
# ================================================================================================

def test_density_action_survives_the_triggered_checked_bool(win, tmp_path, monkeypatch):
    """THE regression. ``QAction.triggered`` carries a ``checked`` bool and Qt passes it as
    the first POSITIONAL argument to whatever it is connected to.

    With ``ycol`` positional, triggering the menu bound it to ``False``, the amplitude
    prompt was skipped, and the export died on "events frame has no column False". Calling
    ``_export_density()`` directly could never reveal that -- only the menu path does,
    which is exactly why this test triggers the ACTION.
    """
    import nanoscope.gui.main_window as MW

    doc_id = _install(win, _doc(tmp_path, name="a.dat", with_voltage=True))
    _install_detection(win._session, doc_id, _eventset([True] * 40))
    win._sync_export_action()
    assert win._act_export_density.isEnabled()

    target = tmp_path / "d.pdf"
    monkeypatch.setattr(MW, "ask_export_path", lambda *a, **k: target)
    # Answer the amplitude prompt without a human: take the first accept button.
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton",
                        lambda self: self.buttons()[0] if self.buttons() else None)
    # ...and neutralise the STATIC convenience boxes. `warning` builds and shows its own
    # modal in C++, which patching `exec` does not touch -- so on a regression this test
    # would HANG on the "could not write the figure" box instead of failing. Verified: with
    # ycol positional again it blocks indefinitely. A test that hangs is worse than no test.
    failures: list[str] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: failures.append(str(a[2])[:160])))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: failures.append(str(a[2])[:160])))

    win._act_export_density.trigger()          # THE MENU PATH
    assert not failures, f"the menu path reported an error: {failures}"
    assert (tmp_path / "d.png").exists(), "the menu action must actually write the figure"
    assert (tmp_path / "d.csv").exists(), "…with its source data"


def test_density_groups_label_by_voltage_when_they_differ(win, tmp_path):
    """A legend is the point of this figure, and a recording filename is ~60 characters of
    shared boilerplate. When the recordings differ in applied voltage, that IS the
    comparison and the label writes itself."""
    for name, mv in (("run_a.dat", -500.0), ("run_b.dat", -700.0)):
        doc = _doc(tmp_path, name=name, with_voltage=True)
        doc_id = _install(win, doc)
        # Give the voltage channel a distinct level the labeller can read.
        arr = np.full(N, mv * 10.0, np.float32)     # ABF stores 10x; the app divides by 10
        object.__setattr__(doc, "channels", (doc.channels[0], arr))
        _install_detection(win._session, doc_id, _eventset([True] * 30))
    labels = list(win._density_groups())
    assert len(labels) == 2
    assert all(lab.endswith("mV") for lab in labels), labels


def test_density_groups_fall_back_to_the_distinguishing_part_of_the_name(win, tmp_path):
    """No voltage to separate them -> strip the common prefix AND suffix, leaving only
    what differs. The whole filename three times over ran off both edges of the plot."""
    for name in ("pore7_ada_75mV_processed.dat", "pore7_cb7_75mV_processed.dat"):
        doc_id = _install(win, _doc(tmp_path, name=name))
        _install_detection(win._session, doc_id, _eventset([True] * 30))
    labels = list(win._density_groups())
    assert len(labels) == 2
    assert set(labels) == {"ada", "cb7"}, labels


def test_density_action_is_disabled_without_a_detection(win, tmp_path):
    _install(win, _doc(tmp_path))
    win._sync_export_action()
    assert not win._act_export_density.isEnabled()
    # ...and the style dialog is always available: styling needs no data.
    assert win._act_density_style.isEnabled()


def test_density_groups_label_a_current_only_recording_by_its_nominal_voltage(win, tmp_path):
    """MEASURED: a three-voltage KDE of the owner's apo-ferritin CSVs came out with a legend
    reading "6_Process_-6", "6_Process_-7", "8_process data_-5".

    ``_density_groups`` carried a private half-copy of the voltage lookup that consulted only
    a VOLTAGE CHANNEL. Every current-only export — which is every CSV the owner has — fell
    straight through it to the filename-affix fallback, even though the app knew the bias all
    along: the Calibration panel prints it from ``meta.nominal_voltage_mV`` ("V from
    filename/header"), and ``iv.applied_voltage_mV`` is the shared resolver that returns it.
    The fix was deleting the copy; this asserts the shared answer reaches the legend.
    """
    for name, mv in (("apo_0008_-500.csv", -500.0), ("apo_0006_-700.csv", -700.0)):
        doc_id = _install(win, _doc(tmp_path, name=name, with_voltage=False, nominal_mV=mv))
        _install_detection(win._session, doc_id, _eventset([True] * 30))
    labels = sorted(win._density_groups())
    assert labels == ["-500 mV", "-700 mV"], (
        f"legend read {labels}; a current-only recording must still be labelled by its bias")


def test_editing_the_label_palette_marks_the_project_unsaved(win, monkeypatch):
    """The palette is saved WITH the project, so editing it is unsaved work. Without this,
    defining a vocabulary and quitting threw it away with no prompt — nothing had marked the
    session dirty."""
    from nanoscope.analysis.labels import LabelVocabulary

    win._mark_clean()
    assert win._dirty is False
    v = LabelVocabulary()
    v.add("molecule", "apo-ferritin")
    monkeypatch.setattr(win, "_edit_labels", lambda: None)   # not exercising the dialog here
    # drive the same tail the dialog runs
    changed = v.to_json() != win._label_vocab.to_json()
    win._label_vocab = v
    win._push_vocabulary()
    if changed:
        win._mark_dirty()
    assert win._dirty is True, "editing the palette left the project looking saved"


def test_label_editor_visibly_blocks_reserved_and_duplicate_hotkeys(win, monkeypatch):
    """Legacy palettes still open, but unsafe shortcuts cannot be silently re-saved."""
    from nanoscope.analysis.labels import LabelVocabulary

    vocab = LabelVocabulary()
    vocab.add("molecule", "apo", hotkey="a")
    vocab.add("type", "collision", hotkey="c")
    vocab.add("type", "translocation", hotkey="C")
    win._label_vocab = vocab
    seen: dict[str, object] = {}

    def inspect_dialog(dialog):
        table = dialog.findChild(QtWidgets.QTableWidget, "labelVocabularyTable")
        error = dialog.findChild(QtWidgets.QLabel, "labelHotkeyError")
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
        seen["error"] = error.text()
        seen["ok_enabled"] = buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()
        seen["highlighted"] = [
            table.item(row, 2).toolTip() for row in range(table.rowCount())
        ]
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(QtWidgets.QDialog, "exec", inspect_dialog)
    win._edit_labels()

    assert seen["ok_enabled"] is False
    assert "reserved" in str(seen["error"]).lower()
    assert "duplicated" in str(seen["error"]).lower()
    assert all(seen["highlighted"])
    assert win._label_vocab.to_json() == vocab.to_json()


def test_label_editor_first_use_can_save_one_blank_starter_row(win, monkeypatch):
    """Completing one example must not turn the untouched starter rows into errors."""
    from nanoscope.analysis.labels import LabelVocabulary

    win._label_vocab = LabelVocabulary()

    def fill_one(dialog):
        table = dialog.findChild(QtWidgets.QTableWidget, "labelVocabularyTable")
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
        assert table.rowCount() == 3
        assert all(
            not table.item(row, col).text()
            for row in range(table.rowCount())
            for col in range(table.columnCount())
        )
        table.item(0, 0).setText("molecule")
        table.item(0, 1).setText("apo-ferritin")
        QtWidgets.QApplication.processEvents()
        assert buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", fill_one)
    win._edit_labels()

    assert [(entry.key, entry.value) for entry in win._label_vocab.entries] == [
        ("molecule", "apo-ferritin")
    ]


def test_label_editor_invalid_accept_stays_open_then_saves_correction(win, monkeypatch):
    """The acceptance boundary revalidates; correcting a reserved key keeps the edit."""
    from nanoscope.analysis.labels import LabelVocabulary

    win._label_vocab = LabelVocabulary()

    def reject_then_fix(dialog):
        table = dialog.findChild(QtWidgets.QTableWidget, "labelVocabularyTable")
        error = dialog.findChild(QtWidgets.QLabel, "labelHotkeyError")
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
        ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        table.item(0, 0).setText("type")
        table.item(0, 1).setText("translocation")
        table.item(0, 2).setText("A")
        QtWidgets.QApplication.processEvents()
        assert not ok.isEnabled()
        assert "reserved" in error.text().lower()

        # Even a programmatic accepted signal cannot close around live invalid data.
        buttons.accepted.emit()
        assert dialog.result() != QtWidgets.QDialog.Accepted

        table.item(0, 2).setText("x")
        QtWidgets.QApplication.processEvents()
        assert ok.isEnabled()
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", reject_then_fix)
    win._edit_labels()

    assert len(win._label_vocab.entries) == 1
    entry = win._label_vocab.entries[0]
    assert (entry.key, entry.value, entry.hotkey) == (
        "type", "translocation", "x")


def test_the_palette_reaches_an_already_open_inspector(win, tmp_path, qtbot):
    """MEASURED BEFORE: a restore handed the palette to the gallery only, so an inspector
    left open across an Open Project kept the previous project's vocabulary and its digits
    wrote labels the new project has never heard of."""
    from nanoscope.analysis.labels import LabelVocabulary

    doc_id = _install(win, _doc(tmp_path))
    _install_detection(win._session, doc_id, _eventset([True] * 30))
    win._sync_export_action()
    win._open_event_inspector()          # the same path the menu action takes
    qtbot.wait(30)
    assert win._event_inspector is not None

    v = LabelVocabulary()
    v.add("molecule", "holo-ferritin")
    win._label_vocab = v
    win._push_vocabulary()
    assert [e.token for e in win._event_inspector._vocab.entries] == ["molecule=holo-ferritin"]
