"""Sweep-3 regressions JM-01 / JM-04 — and the RULE both of them broke.

THE RULE
========
Every readout on the Analysis panel describes ONE ``(doc_id, channel)``. Pointing the
panel at another recording is not "leave the screen as it was": it is the moment the
previous recording's answers become claims about this one, under this one's title.
``AnalysisPanel._clear_results`` is the negative twin of ``_show_result`` and the two
halves must agree —

    whatever ``_show_result`` writes from a result, ``_clear_results`` writes from its
    absence.

A readout with only a POSITIVE writer states the last recording that had one. Two did:

* **JM-01** — the Fits box. ``_refresh_fit_text`` wrote only when the new target HAD a
  stored fit (``if parts:``, no ``else``), and the only writer of the empty string was
  reached only when the new target had no DETECTION at all. So with two detected
  recordings, fitting A and then selecting B left A's dwell/depth summary verbatim
  beside B's own event table and Summary. ``FitResult.summary()`` carries no doc_id,
  path or channel, so nothing on screen said whose numbers those were.

* **JM-04** — the Detection tab's "measured" polarity line. Pushed only when a
  detection's provenance carried a dict; nothing reset it. One recording's
  "blockade (down): 18 down / 0 up (confidence 1.00, measured)" therefore stayed on
  screen for a recording whose own events go the other way — and after the recording
  was CLOSED, under a title reading "No channel selected".

The last test in this file is the rule itself rather than either instance: it repoints
at an undetected recording and asserts that NO readout the result path writes is still
carrying the previous recording's answer — the Fits box, the polarity line, the Summary
block, the warnings block, the baseline health block, the detector score, and the clock
the event table's ``t_*`` headers are labelled on.

pytest-qt, offscreen, through a real :class:`MainWindow` — every gesture is one a user
makes (the selection path the file tree uses, the Fits box's own Fit button, closing a
document).
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 200_000


# ================================================================================================
# builders (the shape of tests/gui/test_project_io.py + test_beta_calib_filters_preview_undo.py)
# ================================================================================================

def _fittable_trace(seed: int = 0, *, up: bool = False) -> np.ndarray:
    """40 events of varied dwell — enough for fit_dwell / fit_depth to stand behind a
    number. ``up`` flips the excursions, which is what makes the measured polarity of
    the two recordings in these tests genuinely opposite."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    sign = 1.0 if up else -1.0
    for i in range(40):
        s = 1_000 + i * 4_800
        e = s + 120 + int(rng.integers(0, 140))
        x[s:e] += sign * 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str, arr: np.ndarray, *, t0_s: float = 0.0) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(arr.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, t0_s=float(t0_s),
        channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    """Add + layer + select, the way MainWindow's own loader does."""
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _detect(window: MainWindow, qtbot) -> None:
    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(30)   # let the queued add_result -> resultsChanged settle


def _fit(window: MainWindow, qtbot) -> None:
    """Press the Fits box's own Fit button, the way the user does."""
    stats = window._analysis_panel._stats
    stats.n_components_dwell.setValue(1)
    stats.n_components_depth.setValue(1)
    stats.fit_button.click()
    qtbot.wait(30)


def _fits_text(window: MainWindow) -> str:
    return window._analysis_panel._stats.results.toPlainText()


def _polarity(window: MainWindow) -> str:
    return window._analysis_panel._detection.polarity_readout.text()


def _t_start_header(window: MainWindow) -> str:
    """The event table's ``t_start_s`` header AS DISPLAYED — where the clock note lives."""
    model = window._analysis_panel._table._model
    for section in range(model.columnCount()):
        text = str(model.headerData(section, Qt.Horizontal, Qt.DisplayRole) or "")
        if text.startswith("t_start_s"):
            return text
    raise AssertionError("the event table has no t_start_s column")


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def two_detected(window, qtbot, tmp_path):
    """Recording A (blockades, t₀ = 61.3034 s) and recording B (enhancements, t₀ = 0),
    both detected, selection left on B. The owner's own files differ in t₀ exactly like
    this (10.2526 / 61.3034 / 0 s)."""
    doc_a, ch_a = _install(window, _doc(tmp_path, "pore_a_down.dat",
                                       _fittable_trace(0), t0_s=61.3034))
    _detect(window, qtbot)
    doc_b, ch_b = _install(window, _doc(tmp_path, "pore_b_up.dat",
                                       _fittable_trace(1, up=True), t0_s=0.0))
    _detect(window, qtbot)
    assert window._session.results.get(doc_a, ch_a, "detect.schmitt") is not None
    assert window._session.results.get(doc_b, ch_b, "detect.schmitt") is not None
    return (doc_a, ch_a), (doc_b, ch_b)


# ================================================================================================
# JM-01 — the Fits box
# ================================================================================================

def test_the_fits_box_does_not_survive_onto_a_recording_with_no_fit(
        window, qtbot, two_detected):
    """Fit A, then select B (detected, never fitted): the box must not read A's fit."""
    (doc_a, ch_a), (doc_b, ch_b) = two_detected

    window._on_selection(doc_a, ch_a)
    qtbot.wait(20)
    _fit(window, qtbot)
    a_text = _fits_text(window)
    assert "exponential" in a_text, f"setup: recording A was never fitted ({a_text!r})"

    window._on_selection(doc_b, ch_b)
    qtbot.wait(20)

    assert window._session.results.get(doc_b, ch_b, "fit.dwell") is None, (
        "setup: recording B must have no stored fit of its own")
    assert window._analysis_panel._sel == (doc_b, ch_b)
    assert window._analysis_panel._table._model.rowCount() > 0, (
        "setup: B's own event table should be on screen beside the Fits box")
    assert _fits_text(window) == "", (
        "the Fits box still shows recording A's dwell/depth fit while the panel is "
        f"pointed at recording B: {_fits_text(window)!r}")


def test_a_targets_own_fit_still_comes_back_on_the_round_trip(window, qtbot, two_detected):
    """The negative half must not cost the positive one: A's fit reappears on return,
    and it is A's — clearing B's empty box did not delete anything."""
    (doc_a, ch_a), (doc_b, ch_b) = two_detected

    window._on_selection(doc_a, ch_a)
    qtbot.wait(20)
    _fit(window, qtbot)
    a_text = _fits_text(window)

    window._on_selection(doc_b, ch_b)
    qtbot.wait(20)
    assert _fits_text(window) == ""

    window._on_selection(doc_a, ch_a)
    qtbot.wait(20)
    assert _fits_text(window) == a_text, "recording A's own fit did not come back"
    assert window._session.results.get(doc_a, ch_a, "fit.dwell") is not None


def test_the_fits_box_is_empty_once_the_fitted_recording_is_closed(
        window, qtbot, two_detected):
    """Closing the recording a fit belongs to leaves nothing that fit could be about."""
    (doc_a, ch_a), _b = two_detected

    window._on_selection(doc_a, ch_a)
    qtbot.wait(20)
    _fit(window, qtbot)
    assert _fits_text(window) != ""

    window._session.close_document(doc_a)
    qtbot.wait(30)

    assert _fits_text(window) == "", (
        f"the Fits box outlived the recording it describes: {_fits_text(window)!r}")


# ================================================================================================
# JM-04 — the measured-polarity line
# ================================================================================================

def test_the_polarity_line_does_not_survive_onto_an_undetected_recording(
        window, qtbot, tmp_path):
    doc_a, ch_a = _install(window, _doc(tmp_path, "pore_down.dat", _fittable_trace(0)))
    _detect(window, qtbot)
    measured_a = _polarity(window)
    assert "measured" in measured_a, f"setup: A's polarity was never measured ({measured_a!r})"

    doc_b, ch_b = _install(window, _doc(tmp_path, "pore_up.dat",
                                        _fittable_trace(1, up=True)))
    qtbot.wait(20)

    assert window._session.results.get(doc_b, ch_b, "detect.schmitt") is None
    assert window._analysis_panel._table._model.rowCount() == 0, (
        "setup: B has no detection, so its event table is empty")
    assert _polarity(window) == "not yet measured", (
        "the Detection tab states another recording's measured polarity for a "
        f"recording that was never detected: {_polarity(window)!r} (A's line was "
        f"{measured_a!r})")


def test_the_polarity_line_states_each_recordings_own_physics(window, qtbot, two_detected):
    """Two recordings whose events go opposite ways: each selection reads its own."""
    (doc_a, ch_a), (doc_b, ch_b) = two_detected

    window._on_selection(doc_a, ch_a)
    qtbot.wait(20)
    a_line = _polarity(window)
    window._on_selection(doc_b, ch_b)
    qtbot.wait(20)
    b_line = _polarity(window)

    assert "down" in a_line and "measured" in a_line, a_line
    assert "up" in b_line and "measured" in b_line, b_line
    assert a_line != b_line, (
        "both recordings show the same measured polarity although their events go "
        f"opposite ways: {a_line!r}")
    # ...and back, so it is a repoint and not a one-way wipe.
    window._on_selection(doc_a, ch_a)
    qtbot.wait(20)
    assert _polarity(window) == a_line


def test_the_polarity_line_does_not_outlive_the_recording_it_measured(
        window, qtbot, tmp_path):
    """With nothing open the panel reads "No channel selected"; the polarity line must
    not go on asserting a closed file's physics with confidence 1.00."""
    doc_a, _ch = _install(window, _doc(tmp_path, "pore_down.dat", _fittable_trace(0)))
    _detect(window, qtbot)
    assert "measured" in _polarity(window)

    window._session.close_document(doc_a)
    qtbot.wait(30)

    assert window._analysis_panel._sel is None
    assert _polarity(window) == "not yet measured", (
        f"the closed recording's polarity is still on screen: {_polarity(window)!r}")


# ================================================================================================
# THE RULE — no readout the result path writes may outlive the result
# ================================================================================================

def test_no_readout_survives_a_repoint_at_a_recording_with_nothing_to_show(
        window, qtbot, tmp_path):
    """The generalisation both defects are instances of, asserted readout by readout.

    Recording A is detected, fitted, and carries a clock offset (t₀ = 61.3034 s) and a
    baseline health block. Recording B is opened and never detected. Every surface the
    result path writes must be showing B's answer (there is none) rather than A's.
    """
    panel = window._analysis_panel
    doc_a, ch_a = _install(window, _doc(tmp_path, "pore_a.dat",
                                        _fittable_trace(0), t0_s=61.3034))
    _detect(window, qtbot)
    _fit(window, qtbot)

    # Everything is up on A.
    assert _fits_text(window) != ""
    assert "measured" in _polarity(window)
    assert panel._stats.summary_text("n_events") not in ("", "—")
    assert "(t₀=0)" in _t_start_header(window), (
        "setup: A's t₀ differs from 0, so its t_start_s header must carry the clock note")
    assert panel._baseline.health_text() != "", (
        "setup: a detection's baseline health block should be on screen")

    _install(window, _doc(tmp_path, "pore_b.dat", _fittable_trace(2), t0_s=0.0))
    qtbot.wait(30)

    left_standing = []
    if _fits_text(window) != "":
        left_standing.append(f"Fits box: {_fits_text(window)!r}")
    if _polarity(window) != "not yet measured":
        left_standing.append(f"polarity: {_polarity(window)!r}")
    if panel._stats.summary_text("n_events") != "—":
        left_standing.append(f"summary n_events: {panel._stats.summary_text('n_events')!r}")
    if panel._stats.warnings_text() != "":
        left_standing.append(f"warnings: {panel._stats.warnings_text()!r}")
    if panel._baseline.health_text() != "":
        left_standing.append(f"baseline health: {panel._baseline.health_text()!r}")
    if "(t₀=0)" in _t_start_header(window):
        left_standing.append(f"table clock header: {_t_start_header(window)!r}")
    if panel._score.value_text("n_detected") != "—":
        left_standing.append(
            f"detector score: {panel._score.value_text('n_detected')!r}")
    if panel._table._model.rowCount() != 0:
        left_standing.append("the event table still has rows")

    assert not left_standing, (
        "recording A's answers are still on screen with the panel pointed at "
        "recording B: " + " | ".join(left_standing))
