"""GUI tests for the two ease-of-use affordances wired through a real MainWindow (pytest-qt).

Exercised the way the user progresses through the app:

* the calibration chip's colour + text track NO_CHANNEL → UNCALIBRATED(amber) →
  CALIBRATED(green) as a current recording is opened and a gain is dialled in;
* the workflow strip's per-stage glyphs track OPEN → CALIBRATE → DETECT → CURATE → ANALYZE
  as a synth recording is opened, the gain set, a detection stored and an event accepted;
* clicking the chip reveals the Calibration panel; clicking a strip stage routes to that
  stage's existing reveal action; clicking a locked stage posts its reason to the status bar;
* the strip degrades through its densities as the window narrows, so it can never be the
  thing that stops the window (or the panes either side of it) getting smaller.

The strip is the former left-hand *rail* laid out horizontally above the signal viewer. These
tests assert its stage STATES, tooltips and routing — never a fixed column geometry.

Both features read a pure PipelineStatus (unit-tested in test_workflow_status.py); this file
pins the WIRING. Nothing here touches the frozen 20-column EventSet schema.
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
from PySide6 import QtWidgets

from nanoscope.analysis.events import EventSet, empty_df
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.status_chip import AMBER, GREEN, NEUTRAL
from nanoscope.gui.workflow_rail import (
    DENSITY_CURRENT,
    DENSITY_FULL,
    DENSITY_MINIMAL,
    GLYPH_CURRENT,
    GLYPH_DONE,
    GLYPH_LOCKED,
    GLYPH_TODO,
    WorkflowStrip,
)
from nanoscope.gui.workflow_status import PIPELINE_ORDER, CalibrationState, Stage

FS = 50_000.0
N = 20_000


# ================================================================================================
# builders
# ================================================================================================

def _current_doc(tmp_path, *, name="rec.dat", with_voltage=False) -> TraceDocument:
    chans = [ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)]
    if with_voltage:
        chans.append(ChannelInfo(index=1, name="voltage", raw_units="mV",
                                 kind=ChannelKind.VOLTAGE))
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=tuple(chans))
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
def detached_strip(qapp, qtbot):
    """A parentless strip fed the SAME PipelineStatus the wired one gets.

    The density ladder is a function of the width the strip is GIVEN, and inside a live
    MainWindow the central layout immediately re-imposes its own width on any resize. Hosting
    it in a bare, layout-less parent lets ``resize()`` stick, so the ladder can be exercised at
    the exact widths that matter instead of at whatever the window happens to be. The host is
    shown (offscreen) because Qt defers a hidden widget's resize event and the density is
    chosen in ``resizeEvent``."""
    hosts: list[QtWidgets.QWidget] = []

    def make(status) -> WorkflowStrip:
        host = QtWidgets.QWidget()
        hosts.append(host)          # keep it alive: C++ deletes the strip with its parent
        qtbot.addWidget(host)
        host.resize(2400, 200)
        strip = WorkflowStrip(host)
        strip.set_status(status)
        host.show()
        return strip

    yield make
    for host in hosts:
        host.close()


# ================================================================================================
# the chip: NO_CHANNEL -> UNCALIBRATED(amber) -> CALIBRATED(green)
# ================================================================================================

def test_chip_starts_no_channel_on_empty_session(win):
    assert win._calibration_chip.status().state is CalibrationState.NO_CHANNEL
    assert win._calibration_chip.accent() == NEUTRAL
    assert win._calibration_chip.text() == "No current channel"


def test_chip_goes_amber_uncalibrated_then_green_calibrated(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    # a current channel at its default gain -> uncalibrated, amber, the loud warning
    assert win._calibration_chip.status().state is CalibrationState.UNCALIBRATED
    assert win._calibration_chip.accent() == AMBER
    assert "not physical" in win._calibration_chip.text()

    # dial in a real gain -> calibrated, green, shows the gain/units detail
    win._session.set_calibration(doc_id, 0, ChannelCalibration(gain=2500.0, target_units="nS"))
    assert win._calibration_chip.status().state is CalibrationState.CALIBRATED
    assert win._calibration_chip.accent() == GREEN
    assert "2500" in win._calibration_chip.text() and "nS" in win._calibration_chip.text()


def test_chip_no_channel_on_voltage_only_recording(win, tmp_path):
    doc_id = _install(win, _voltage_only_doc(tmp_path))
    win._on_selection(doc_id, 0)
    assert win._calibration_chip.status().state is CalibrationState.NO_CHANNEL
    assert win._calibration_chip.accent() == NEUTRAL


def test_chip_follows_current_channel_when_voltage_is_selected(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path, with_voltage=True))
    win._on_selection(doc_id, 1)                 # user clicks the voltage channel
    assert win._calibration_chip.status().state is CalibrationState.UNCALIBRATED
    assert win._calibration_chip.status().channel == 0    # resolved to the current channel


# ================================================================================================
# the strip: glyphs track the pipeline
# ================================================================================================

def test_strip_glyphs_track_open_calibrate_detect_curate_analyze(win, tmp_path):
    strip = win._workflow_strip

    # empty session: OPEN is current, everything else locked
    assert strip.glyph(Stage.OPEN) == GLYPH_CURRENT
    assert strip.glyph(Stage.CALIBRATE) == GLYPH_LOCKED
    assert win._workflow_status.current_stage is Stage.OPEN

    # open + select a current recording: OPEN done, CALIBRATE current, DETECT a to-do
    doc_id = _install(win, _current_doc(tmp_path))
    ev = _eventset([False, False])
    win._on_selection(doc_id, 0)
    assert strip.glyph(Stage.OPEN) == GLYPH_DONE
    assert strip.glyph(Stage.CALIBRATE) == GLYPH_CURRENT
    assert strip.glyph(Stage.DETECT) == GLYPH_TODO
    assert strip.glyph(Stage.CURATE) == GLYPH_LOCKED

    # set the gain: CALIBRATE done, DETECT current
    win._session.set_calibration(doc_id, 0, ChannelCalibration(gain=2500.0, target_units="nS"))
    assert strip.glyph(Stage.CALIBRATE) == GLYPH_DONE
    assert strip.glyph(Stage.DETECT) == GLYPH_CURRENT

    # store a detection (nothing accepted yet): DETECT done, CURATE current, ANALYZE a to-do
    _install_detection(win._session, doc_id, ev)
    assert strip.glyph(Stage.DETECT) == GLYPH_DONE
    assert strip.glyph(Stage.CURATE) == GLYPH_CURRENT
    assert strip.glyph(Stage.ANALYZE) == GLYPH_TODO

    # accept an event THROUGH the panel (the real curation path): CURATE done, ANALYZE current
    win._analysis_panel.set_event_accepted(0, True)
    assert strip.glyph(Stage.CURATE) == GLYPH_DONE
    assert strip.glyph(Stage.ANALYZE) == GLYPH_CURRENT


def test_strip_renders_all_six_stages_in_pipeline_order(win, tmp_path):
    """Every stage has a chip carrying its name (regression: the label was styled but never
    set, so the stepper rendered as a bare glyph column)."""
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    strip = win._workflow_strip

    assert [strip.label(s) for s in PIPELINE_ORDER] == [
        "Open", "Calibrate", "Filter", "Detect", "Curate", "Analyze"]
    # …and each one is clickable chrome of its own, not a single painted banner.
    assert len({id(strip.stage_chip(s)) for s in PIPELINE_ORDER}) == 6


def test_strip_marks_filter_optional_when_it_has_room_to_say_so(win, tmp_path, detached_strip):
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    strip = detached_strip(win._workflow_status)
    strip.resize(strip.width_for(DENSITY_FULL), strip.height())

    assert strip.density() == DENSITY_FULL
    assert strip.shown_text(Stage.OPEN) == "Open"
    assert "optional" in strip.shown_text(Stage.FILTER).lower()


def test_strip_tooltips_carry_each_stage_reason(win, tmp_path):
    """The reason is what the tooltip is FOR — it is the same string a locked click posts to
    the status bar, so the user can read why a stage is unavailable without clicking it."""
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)                     # a current doc, no detection yet
    strip = win._workflow_strip

    for stage in PIPELINE_ORDER:
        reason = win._workflow_status.stage(stage).reason
        assert reason and reason in strip.tooltip(stage)
        assert strip.label(stage) in strip.tooltip(stage)
    assert "Detect events first" in strip.tooltip(Stage.ANALYZE)     # the locked one


def test_strip_filter_is_optional_todo_never_current(win, tmp_path):
    strip = win._workflow_strip
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    # FILTER is available (a current channel exists) but shown as an optional to-do, and it is
    # never the current step (current is CALIBRATE here).
    assert strip.glyph(Stage.FILTER) == GLYPH_TODO
    assert win._workflow_status.stage(Stage.FILTER).optional is True
    assert win._workflow_status.current_stage is not Stage.FILTER


def test_strip_locks_calibrate_and_detect_without_a_current_channel(win, tmp_path):
    strip = win._workflow_strip
    doc_id = _install(win, _voltage_only_doc(tmp_path))
    win._on_selection(doc_id, 0)
    assert strip.glyph(Stage.OPEN) == GLYPH_DONE
    assert strip.glyph(Stage.CALIBRATE) == GLYPH_LOCKED
    assert strip.glyph(Stage.DETECT) == GLYPH_LOCKED
    assert win._workflow_status.current_stage is None


# ================================================================================================
# routing: chip click, strip stage click, locked-stage reason
# ================================================================================================

def test_chip_click_reveals_the_calibration_panel(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    win._inspector_tabs.setCurrentWidget(win._analysis_panel)   # park on a different tab

    win._calibration_chip.clicked.emit()

    assert win._inspector_tabs.currentWidget() is win._cal_scroll
    assert not win._inspector.isHidden()


def test_strip_stage_click_routes_to_the_matching_reveal(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)

    # CALIBRATE -> the Calibration tab
    win._inspector_tabs.setCurrentWidget(win._analysis_panel)
    win._workflow_strip.stageActivated.emit(Stage.CALIBRATE.value)
    assert win._inspector_tabs.currentWidget() is win._cal_scroll

    # DETECT -> the Analysis tab (DetectionPanel lives there)
    win._inspector_tabs.setCurrentWidget(win._cal_scroll)
    win._workflow_strip.stageActivated.emit(Stage.DETECT.value)
    assert win._inspector_tabs.currentWidget() is win._analysis_panel


def test_clicking_a_strip_chip_routes_the_same_as_the_signal(win, tmp_path):
    """The chip WIDGET is the user's actual target, so drive it, not just stageActivated:
    a real click must reach the same reveal action."""
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    win._inspector_tabs.setCurrentWidget(win._analysis_panel)

    win._workflow_strip.stage_chip(Stage.CALIBRATE).clicked.emit()
    assert win._inspector_tabs.currentWidget() is win._cal_scroll

    win._workflow_strip.stage_chip(Stage.DETECT).clicked.emit()
    assert win._inspector_tabs.currentWidget() is win._analysis_panel


def test_strip_curate_click_opens_the_event_inspector(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path))
    _install_detection(win._session, doc_id, _eventset([True, False]))
    win._on_selection(doc_id, 0)
    assert win._event_inspector is None
    win._workflow_strip.stageActivated.emit(Stage.CURATE.value)     # CURATE is available now
    assert win._event_inspector is not None


def test_locked_stage_click_posts_its_reason_to_the_status_bar(win, tmp_path):
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)                 # a current doc, but no detection yet
    # ANALYZE is locked (no detection) -> clicking it must explain why, not act.
    win._workflow_strip.stageActivated.emit(Stage.ANALYZE.value)
    msg = win.statusBar().currentMessage()
    assert "Detect events first" in msg
    # the Event Inspector was NOT opened by the locked click
    assert win._event_inspector is None


def test_open_stage_click_on_empty_session_does_not_crash(win):
    # OPEN is available on an empty session; routing it must not raise (it opens the file dialog,
    # which we cannot drive headlessly — so just assert the path is reachable via the handler).
    assert win._workflow_status.current_stage is Stage.OPEN
    # locked stages still emit; OPEN is available -> handler will call _open_dialog. Guard the
    # dialog so the test stays headless.
    win._open_dialog = lambda: None  # type: ignore[method-assign]
    win._workflow_strip.stageActivated.emit(Stage.OPEN.value)


# ================================================================================================
# the strip is chrome: one short row that yields instead of pushing the window wider
# ================================================================================================

def test_strip_is_a_thin_horizontal_row(win):
    """It replaced a 172 px column; it must cost one toolbar-height row, not a second one."""
    strip = win._workflow_strip
    assert 24 <= strip.height() <= 32
    # wide enough to be a breadcrumb, not a column: the six stages read left-to-right.
    assert strip.width_for(DENSITY_FULL) > 4 * strip.height()


def test_strip_density_ladder_is_widest_that_fits(win, tmp_path, detached_strip):
    """The decision itself, with no geometry involved: each density is chosen exactly at the
    width it needs, and anything narrower than glyphs-only still resolves to glyphs-only."""
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)
    strip = detached_strip(win._workflow_status)

    for density in (DENSITY_FULL, DENSITY_CURRENT, DENSITY_MINIMAL):
        need = strip.width_for(density)
        assert strip.density_for(need) == density
    assert strip.density_for(strip.width_for(DENSITY_MINIMAL) - 1) == DENSITY_MINIMAL
    assert strip.density_for(0) == DENSITY_MINIMAL
    # the ladder is monotonic: wider never means a poorer rendering
    assert strip.width_for(DENSITY_FULL) > strip.width_for(DENSITY_CURRENT)
    assert strip.width_for(DENSITY_CURRENT) > strip.width_for(DENSITY_MINIMAL)


def test_strip_degrades_instead_of_forcing_the_window_wider(win, tmp_path, detached_strip):
    """As it gets squeezed the strip drops the "(optional)" note, then every label except the
    one that says where you are, then the labels entirely — the glyphs and the click targets
    survive all the way down."""
    doc_id = _install(win, _current_doc(tmp_path))
    win._on_selection(doc_id, 0)                     # CALIBRATE is the current stage here
    strip = detached_strip(win._workflow_status)

    strip.resize(strip.width_for(DENSITY_FULL), strip.height())
    assert strip.density() == DENSITY_FULL

    strip.resize(strip.width_for(DENSITY_CURRENT), strip.height())
    assert strip.density() == DENSITY_CURRENT
    assert strip.shown_text(Stage.CALIBRATE) == "Calibrate"      # "you are here" survives
    assert strip.shown_text(Stage.ANALYZE) == ""

    strip.resize(strip.width_for(DENSITY_MINIMAL), strip.height())
    assert strip.density() == DENSITY_MINIMAL
    assert all(strip.shown_text(s) == "" for s in PIPELINE_ORDER)
    # …and every stage still shows its glyph and can still be clicked.
    assert strip.glyph(Stage.CALIBRATE) == GLYPH_CURRENT
    assert "Calibrate" in strip.tooltip(Stage.CALIBRATE)


def test_strip_never_becomes_the_windows_width_floor(win, detached_strip):
    """Its minimum is the glyphs-only width whatever it happens to be rendering, so a wide
    strip cannot pin the main window (or the docks either side) at its full-label width."""
    strip = detached_strip(win._workflow_status)
    strip.resize(strip.width_for(DENSITY_FULL), strip.height())
    assert strip.density() == DENSITY_FULL

    floor = strip.minimumSizeHint().width()
    assert floor == strip.width_for(DENSITY_MINIMAL)
    assert floor < strip.width_for(DENSITY_FULL)
    # and it is not what stops the window shrinking — the status bar already asks for more.
    assert floor < win.statusBar().minimumSizeHint().width()


# ================================================================================================
# the no-cross-panel-import invariant, for the NEW code (main_window is the only wirer)
# ================================================================================================

def _relative_sibling_imports(module) -> list[str]:
    """Names imported via ``from .X import ...`` (level-1 relative) in a gui module — i.e. its
    sibling-gui dependencies."""
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    return [n.module for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module]


def test_new_widgets_import_only_the_pure_status_module_no_sibling_panel():
    """The chip and the strip may import Qt + the pure ``workflow_status`` only — never another
    gui panel. main_window stays the sole importer that wires panels together."""
    import nanoscope.gui.status_chip as chip_mod
    import nanoscope.gui.workflow_rail as strip_mod

    for mod in (chip_mod, strip_mod):
        siblings = _relative_sibling_imports(mod)
        assert siblings == ["workflow_status"] or siblings == [], (
            f"{mod.__name__} imports sibling gui module(s) {siblings}; only 'workflow_status' "
            f"is allowed")


def test_pure_status_module_imports_no_gui_and_no_qt():
    """``workflow_status`` derives everything from core/analysis value types; it must not import
    any gui module and must stay Qt-free at runtime (Session is typing-only)."""
    import nanoscope.gui.workflow_status as ws

    tree = ast.parse(pathlib.Path(ws.__file__).read_text())
    runtime = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in runtime:
        mod = getattr(node, "module", None) or ""
        names = [a.name for a in node.names]
        assert "gui" not in mod.split("."), f"workflow_status must not import gui: {mod}"
        assert not mod.startswith("PySide6"), "workflow_status must stay Qt-free"
        assert not any(n.startswith("PySide6") for n in names), "workflow_status must stay Qt-free"
