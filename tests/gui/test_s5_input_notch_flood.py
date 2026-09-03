"""Sweep-5 IF-02 regression: the "notch lines" box can no longer silently commit
thousands of filtfilt passes.

The defect, measured on the pristine code: pasting a spreadsheet column into the ML
tab's freeform "notch lines" QLineEdit (Qt silently truncates the paste at 32,767
chars) put 10,923 duplicate '60' entries into ``MLPreprocessConfig.notch_freqs`` with
no cap and no disclosure. ``notch_sequential`` runs one zero-phase filtfilt over the
whole analysed range PER LINE by contract (~0.8 ms/line on 60k samples, ~200 ms/line
at the 15 M-sample ceiling), so one accidental paste committed every subsequent ML
preprocess run to ~8 s / ~36 min of silent filtering; the only on-screen trace was a
preset-status line naming the field, never the count.

The fix (gui/analysis_panel.py — the single launcher of analysis jobs): a cap,
``_ML_NOTCH_MAX_LINES`` (32), enforced twice and out loud both times:

* commit-time — an over-cap editingFinished is refused via ``runRefused`` (the status
  bar), NAMING the cap and the count, and the box is rolled back to its last under-cap
  contents (so the box never displays a config the refusal just said may not run);
* launch-time — ``_launch_ml`` refuses an over-cap ENABLED notch list however the
  params arrived (a stored run replayed by the stale banner, a programmatic caller),
  by the same name, before it costs anything.

Wired through a real MainWindow (status bar included), offscreen.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import _ML_NOTCH_MAX_LINES
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000


def _synth_doc(tmp_path: Path, *, name: str = "notch.dat") -> TraceDocument:
    rng = np.random.default_rng(11)
    arr = np.ascontiguousarray(rng.normal(-50.0, 2.0, N), dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id, 0


@pytest.fixture
def window(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    _install(w, _synth_doc(tmp_path))
    qtbot.wait(30)
    yield w
    w._analysis_controller.abort()


def _paste_and_commit(window, qapp, text: str) -> None:
    """The shipped gesture: a clipboard paste into the notch box, then the commit that
    Tab / Return / clicking Run performs (editingFinished)."""
    nf = window._analysis_panel._detection.ml_notch_freqs
    QtWidgets.QApplication.clipboard().setText(text)
    nf.setFocus()
    nf.selectAll()
    nf.paste()
    qapp.processEvents()
    nf.editingFinished.emit()
    qapp.processEvents()


def test_flood_paste_is_refused_named_and_rolled_back(window, qapp):
    """The dossier's exact gesture: a 1M-entry column paste. The commit is refused on
    the status bar BY THE CAP'S NAME with the count, and the box + config are rolled
    back to what they held before — never a silent multi-thousand-line config."""
    det = window._analysis_panel._detection
    before = det.ml_notch_freqs.text()

    _paste_and_commit(window, qapp, ",".join(["60"] * 1_000_000))

    # The box and the config agree, and both are the pre-paste contents.
    assert det.ml_notch_freqs.text() == before
    cfg = det.ml_preprocess_config()
    assert len(cfg.notch_freqs) <= _ML_NOTCH_MAX_LINES
    # The refusal is on the status bar and names the limit and the magnitude.
    msg = window.statusBar().currentMessage()
    assert f"at most {_ML_NOTCH_MAX_LINES} lines" in msg
    assert "10,923" in msg          # what one max-length paste yields — the user sees the count
    assert "filtfilt" in msg        # ...and WHY the limit exists
    # The preset-status surface was re-derived from the ROLLED-BACK text: it must not
    # keep describing the refused paste as the panel's configuration.
    assert "notch_freqs" not in window._analysis_panel._detection.ml_params()[
        "ml_preset_deviations"]


def test_legitimate_edits_pass_untouched(window, qapp):
    """The cap refuses floods by name — it must not tax real configuration: a hand edit,
    and a list at exactly the cap, both commit exactly as typed with no refusal."""
    det = window._analysis_panel._detection
    window.statusBar().clearMessage()

    _paste_and_commit(window, qapp, "50, 100, 150")
    assert det.ml_preprocess_config().notch_freqs == (50.0, 100.0, 150.0)
    assert window.statusBar().currentMessage() == ""

    at_cap = ", ".join(str(60 * (i + 1)) for i in range(_ML_NOTCH_MAX_LINES))
    _paste_and_commit(window, qapp, at_cap)
    assert len(det.ml_preprocess_config().notch_freqs) == _ML_NOTCH_MAX_LINES
    assert window.statusBar().currentMessage() == ""

    # One past the cap is the boundary the refusal names.
    over = ", ".join(["60"] * (_ML_NOTCH_MAX_LINES + 1))
    _paste_and_commit(window, qapp, over)
    assert len(det.ml_preprocess_config().notch_freqs) == _ML_NOTCH_MAX_LINES
    assert f"at most {_ML_NOTCH_MAX_LINES} lines" in window.statusBar().currentMessage()


def test_rollback_target_tracks_stored_run_replays(window, qapp):
    """set_ml_params (a stored run replayed onto the panel) writes the box without a
    commit; a later flood must roll back to THAT value, not to a config from before the
    replay — the rollback target is tracked on textChanged, not editingFinished."""
    det = window._analysis_panel._detection
    params = det.ml_params()
    params["ml_preprocess"] = dict(params["ml_preprocess"], notch_freqs=[50.0, 100.0])
    det.set_ml_params(params)
    assert det.ml_preprocess_config().notch_freqs == (50.0, 100.0)

    _paste_and_commit(window, qapp, ",".join(["60"] * 5_000))
    assert det.ml_preprocess_config().notch_freqs == (50.0, 100.0)


def test_launch_gate_refuses_flooded_stored_config_by_name(window, qapp):
    """The other half: a flooded config that never went through the box (an old stored
    run replayed by the stale banner's Re-run) is refused at the launch seam, before it
    costs anything — and the refusal reaches the status bar with the count."""
    panel = window._analysis_panel
    started = []
    panel._controller.start = lambda *a, **k: (started.append(a), False)[1]  # record + decline

    flooded = {
        "model_path": "/nonexistent/model.keras", "enter": 0.5, "exit": 0.4,
        "range_source": "whole", "range_start": 0, "range_stop": N,
        "ml_preprocess": {"notch_enabled": True, "notch_freqs": [60.0] * 10_923},
    }
    assert panel._launch_ml(flooded) is False
    assert started == []                      # refused BEFORE the controller — zero cost
    msg = window.statusBar().currentMessage()
    assert "10,923" in msg and f"capped at {_ML_NOTCH_MAX_LINES}" in msg

    # A DISABLED flooded notch costs nothing, so the cap does not refuse it (a false
    # refusal would name a cost the run does not pay) — it proceeds to the controller.
    window.statusBar().clearMessage()
    disabled = dict(flooded,
                    ml_preprocess={"notch_enabled": False, "notch_freqs": [60.0] * 10_923})
    assert panel._launch_ml(disabled) is False    # the recording controller stub declined
    assert started, "the gate must not block a config whose notch stage is off"
    assert window.statusBar().currentMessage() == ""
