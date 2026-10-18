"""Sweep 7, S7-NOTCH1 (major), the launcher half: a STORED or REPLAYED Schmitt param dict
whose explicit notch list the Preprocess tab's box would refuse is refused by the launcher
too -- by name, through the same status line -- and one notch-line cap exists.

Agent F gave the DetectionPanel's "lines" box the cap (``NOTCH_MAX_LINES``) and the
refuse-by-name policy, so ``params()`` can never carry a flooded or illegal list. Two
doors go around the box: the stale banner's Re-run replays the result's STORED dict
(``schmitt_params_from_result``), and a project written before the cap can hold the
10,923-line paste that started all this (42 s per detection on the owner's -600 mV file;
a truncated '54.' notch stamped into provenance). The ML launcher (``_launch_ml``) already
refused such a dict; ``_run_detection`` now does the same, and ``_ML_NOTCH_MAX_LINES`` is
an alias of the DetectionPanel's constant rather than a second 32.

Also here: the box's own refusal (the orange note under it) now reaches the status bar
through ``runRefused``, the road the ML box's refusal already took -- once per refused
EDIT, never on a Tab out of the rolled-back box.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.modules import register_all
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import analysis_panel as AP
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import DetectionPanel

FS = 50_000.0
I0 = -1000.0
_FRAGMENTS = [(10_000, 10_200), (10_220, 10_500), (10_520, 11_000), (30_000, 30_300)]
# ``getattr`` so the file COLLECTS on the pre-fix snapshot (which has no panel cap) and
# each test fails there on its own claim.
CAP = getattr(DetectionPanel, "NOTCH_MAX_LINES", AP._ML_NOTCH_MAX_LINES)
OVER = [60.0 * (i + 1) for i in range(CAP + 1)]


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _signal(spans, *, n=60_000, depth=300.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] -= depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def selected(window, qtbot, tmp_path):
    did = window._session.add_document(_doc(tmp_path, _signal(_FRAGMENTS), name="a.dat"))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    qtbot.wait(20)
    assert window._analysis_panel.can_run()
    return window, did


# ================================================================================================
# one cap
# ================================================================================================

def test_one_notch_line_cap_exists_and_the_ml_name_is_an_alias_of_it():
    assert AP._ML_NOTCH_MAX_LINES == DetectionPanel.NOTCH_MAX_LINES
    src = open(AP.__file__, encoding="utf-8").read()
    assert re.search(r"^_ML_NOTCH_MAX_LINES\s*=\s*DetectionPanel\.NOTCH_MAX_LINES\s*$",
                     src, re.M), "two 32s can drift apart; the ML name must alias the panel's"


# ================================================================================================
# the launcher refuses a replayed dict the box would refuse
# ================================================================================================

def test_a_replayed_dict_over_the_cap_is_refused_by_name_and_nothing_starts(selected, qtbot):
    window, did = selected
    panel = window._analysis_panel
    params = {**panel.detection_params(), "notch_enabled": True, "notch_freqs": OVER}
    with qtbot.waitSignal(panel.runRefused, timeout=1_000) as refused:
        assert panel._run_detection(params) is False
    msg = refused.args[0]
    assert f"{CAP + 1} lines" in msg and f"at most {CAP}" in msg, msg
    assert "Preprocess tab" in msg
    assert panel._pending is None
    assert not list(window._session.results.for_doc(did))
    qtbot.wait(20)
    assert window.statusBar().currentMessage() == msg, "the refusal must reach the status bar"


def test_a_replayed_dict_with_a_non_positive_or_non_finite_line_is_refused_by_name(
        selected, qtbot):
    window, did = selected
    panel = window._analysis_panel
    params = {**panel.detection_params(), "notch_enabled": True,
              "notch_freqs": [60.0, -120.0, float("inf"), 180.0]}
    with qtbot.waitSignal(panel.runRefused, timeout=1_000) as refused:
        assert panel._run_detection(params) is False
    msg = refused.args[0]
    assert "‘-120.0’" in msg and "‘inf’" in msg, msg
    assert "Nothing was dropped and nothing ran" in msg
    assert not list(window._session.results.for_doc(did))


def test_only_an_enabled_explicit_list_is_judged_and_a_legal_one_passes():
    judge = AP.AnalysisPanel._notch_replay_refusal
    assert judge({"notch_enabled": False, "notch_freqs": OVER}) is None
    assert judge({"notch_enabled": True, "notch_freqs": None}) is None
    assert judge({"notch_enabled": True}) is None
    assert judge({"notch_enabled": True, "notch_freqs": [60.0, 120.0, 180.0]}) is None
    assert judge({"notch_enabled": True, "notch_freqs": OVER[:CAP]}) is None
    assert judge({"notch_enabled": True, "notch_freqs": OVER}) is not None


def test_the_stale_banners_rerun_of_a_flooded_stored_run_is_refused_and_the_banner_stays(
        selected, qtbot):
    """The door the finding named: Re-run replays the STORED dict, never the box.

    Sweep-7 fix-up CW-1 moved the refusal BEFORE the click: the banner's button is
    withheld with the same sentence printed under the message (a live Re-run refused
    on every press with advice the button cannot take -- "fix the list on the Preprocess
    tab" -- looped the user on one sentence). The replay-time refusal stays for the
    programmatic door (``_on_rerun``), and both name the door that runs the box."""
    window, did = selected
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    res = panel.current_detection()
    assert res is not None
    flooded = replace(res, params={**dict(res.params), "notch_enabled": True,
                                   "notch_freqs": OVER})
    window._session.add_result(flooded)
    qtbot.wait(20)
    assert panel.current_detection() is flooded
    # Make it stale the calibration way, which offers Re-run.
    window._session.set_calibration(did, 0, ChannelCalibration(2.0, 0.0, "pA"),
                                    write_sidecar=False)
    qtbot.wait(20)
    # isHidden(), not isVisibleTo(panel): the stale line lives in the Results dock now, so the
    # panel is not its ancestor and isVisibleTo would depend on the window being shown.
    assert not panel._stale.isHidden() and panel._stale.kind == "calibration"

    # Withheld before any click, by name, with the door named.
    assert not panel._stale._btn.isEnabled()
    why = panel._stale.action_refusal()
    assert f"{CAP + 1} lines" in why and "press Detect" in why, why
    panel._stale._btn.click()
    qtbot.wait(20)
    assert panel._pending is None
    # The programmatic replay is refused with the same sentence, and nothing moves.
    with qtbot.waitSignal(panel.runRefused, timeout=1_000) as refused:
        panel._on_rerun()
    assert f"{CAP + 1} lines" in refused.args[0]
    assert "press Detect" in refused.args[0]
    assert panel._pending is None, "a refused replay must not be recorded as a pending job"
    assert not panel._stale.isHidden(), "the banner is dismissed by a RUN, not a click"
    assert panel.current_detection() is flooded, "the stored table was not replaced"


# ================================================================================================
# the box's own refusal reaches the status bar
# ================================================================================================

def _paste(box: QtWidgets.QLineEdit, qapp, text: str) -> None:
    QtWidgets.QApplication.clipboard().setText(text)
    box.setFocus()
    box.selectAll()
    box.paste()
    qapp.processEvents()


def test_the_lines_box_refusal_reaches_the_status_bar_once_per_refused_edit(
        selected, qtbot, qapp):
    window, _did = selected
    panel = window._analysis_panel
    det = panel._detection
    det.notch_use_lines.setChecked(True)
    box = det.notch_freqs
    # A legal commit says nothing on the status line.
    with qtbot.assertNotEmitted(panel.runRefused, wait=100):
        _paste(box, qapp, "60, 120")
        box.editingFinished.emit()
    assert det.params()["notch_freqs"] == [60.0, 120.0]

    # The refused paste: the same sentence on the note under the box and on the status bar.
    _paste(box, qapp, ", ".join(f"{f:g}" for f in OVER))
    with qtbot.waitSignal(panel.runRefused, timeout=1_000) as refused:
        box.editingFinished.emit()
    msg = refused.args[0]
    note = det.notch_lines_note.text()
    assert "\n⚠ " in note
    assert msg.endswith(note.split("\n⚠ ", 1)[1].strip()), (msg, note)
    assert f"{CAP + 1} lines" in msg and f"at most {CAP}" in msg
    assert det.params()["notch_freqs"] == [60.0, 120.0], "the box was rolled back"
    qtbot.wait(20)
    assert window.statusBar().currentMessage() == msg

    # A Tab out of the untouched, rolled-back box keeps the note (the panel's design) but
    # does not re-post it.
    with qtbot.assertNotEmitted(panel.runRefused, wait=100):
        box.editingFinished.emit()
    assert "\n⚠ " in det.notch_lines_note.text()
