"""C12 — committing a calibration must never delete the user's ``.npcal.json`` silently.

``<datafile>.npcal.json`` is a USER-OWNED file sitting next to their raw data. It carries a
``note`` field the code's own comment says a project file cannot round-trip. ``save_sidecar``
deletes it whenever every channel ends up at its default — and *Reset to default* does that by
definition, with no confirmation, no status line and no undo.

The one-call-site trap here is that ``_reset`` is not the only door: *Apply* deletes the same
file the moment the typed values happen to equal the defaults. So the guarantee lives on a
single ``_commit`` path (AST-guarded at the bottom of this file), fed by a ``sidecar_effect``
predicate that shares its determination with ``save_sidecar`` itself.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import json
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.calibration import (
    SIDECAR_NONE,
    SIDECAR_REMOVED,
    SIDECAR_WRITTEN,
    CalibrationSet,
    ChannelCalibration,
    save_sidecar,
    sidecar_effect,
    sidecar_path,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel

FS = 50_000.0
N = 4_000
NOTE = "measured against the 1 MOhm resistor on 2026-02-14 — do not lose this"


# ================================================================================================
# builders
# ================================================================================================
def _doc(tmp_path, *, name="rec.abf") -> TraceDocument:
    """A recording with a CURRENT and a VOLTAGE channel (the voltage one has a non-identity
    default, which is what makes the removal branch reachable at all)."""
    cur = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    volt = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    rng = np.random.default_rng(0)
    arrs = (rng.normal(-1000.0, 8.0, N).astype(np.float32),
            np.full(N, -6000.0, dtype=np.float32))
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(cur, volt))
    return TraceDocument(meta=meta, channels=arrs)


def _session(tmp_path):
    doc = _doc(tmp_path)
    (tmp_path / "rec.abf").write_bytes(b"ABF ")
    s = Session()
    return s, s.add_document(doc), doc


def _write_sidecar_with_note(tmp_path, gain=0.25):
    p = sidecar_path(tmp_path / "rec.abf")
    p.write_text(json.dumps({"version": 1, "channels": {
        "0": {"gain": gain, "offset": 0.0, "target_units": "pA", "note": NOTE}}}, indent=2))
    return p


def _note_in(path, channel="0") -> str:
    """The stored note, read as JSON — json.dumps escapes the em dash, so a substring test
    against the raw text would fail for the wrong reason."""
    return json.loads(path.read_text())["channels"][channel]["note"]


def _panel(qtbot, session, doc_id, channel=0, *, confirm=True):
    p = CalibrationPanel(session)
    qtbot.addWidget(p)
    # `**kw` because the seam now also carries the ACTION that raised it (Apply vs
    # Reset), which the real dialog puts in its title; a test that pinned the old
    # arity would have hidden that the two buttons share one prompt.
    p._confirm_sidecar_removal = lambda path, **kw: confirm    # the injectable seam
    p.set_selection(doc_id, channel)
    return p


# ================================================================================================
# the predicate and the writer agree — that is what makes "ask first" possible
# ================================================================================================
def test_sidecar_effect_predicts_exactly_what_save_sidecar_does(tmp_path):
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    doc = _doc(tmp_path)
    calib = CalibrationSet()
    for info in doc.meta.channels:
        calib.set("d", info.index, CalibrationSet.default_for(info))

    # all default, nothing on disk
    assert sidecar_effect(data, "d", calib, doc.meta.channels) == SIDECAR_NONE
    assert save_sidecar(data, "d", calib, doc.meta.channels) == SIDECAR_NONE

    # a real edit
    calib.set("d", 0, ChannelCalibration(gain=0.25, offset=0.0, target_units="pA", note=NOTE))
    assert sidecar_effect(data, "d", calib, doc.meta.channels) == SIDECAR_WRITTEN
    assert save_sidecar(data, "d", calib, doc.meta.channels) == SIDECAR_WRITTEN
    assert sidecar_path(data).exists()

    # back to default, with the file now present
    calib.set("d", 0, CalibrationSet.default_for(doc.meta.channels[0]))
    assert sidecar_effect(data, "d", calib, doc.meta.channels) == SIDECAR_REMOVED
    assert save_sidecar(data, "d", calib, doc.meta.channels) == SIDECAR_REMOVED
    assert not sidecar_path(data).exists()


def test_session_can_be_asked_before_it_commits(tmp_path):
    """The question must be answerable WITHOUT performing the deletion, and must not
    disturb the live calibration set."""
    s, doc_id, doc = _session(tmp_path)
    s.set_calibration(doc_id, 0, ChannelCalibration(gain=0.25, offset=0.0, note=NOTE))
    assert sidecar_path(tmp_path / "rec.abf").exists()

    default = CalibrationSet.default_for(doc.meta.channels[0])
    assert s.sidecar_effect_of(doc_id, 0, default) == SIDECAR_REMOVED
    # asking changed nothing
    assert sidecar_path(tmp_path / "rec.abf").exists()
    assert s.calibration(doc_id, 0).gain == 0.25


def test_set_calibration_reports_what_it_did_to_the_file(tmp_path):
    """A caller that cannot see the deletion cannot tell the user about it."""
    s, doc_id, doc = _session(tmp_path)
    assert s.set_calibration(doc_id, 0,
                             ChannelCalibration(gain=0.25, offset=0.0)) == SIDECAR_WRITTEN
    assert s.set_calibration(
        doc_id, 0, CalibrationSet.default_for(doc.meta.channels[0])) == SIDECAR_REMOVED
    assert s.set_calibration(doc_id, 0,
                             ChannelCalibration(gain=0.3), write_sidecar=False) == SIDECAR_NONE


# ================================================================================================
# THE regression: the button
# ================================================================================================
def test_c12_reset_to_default_does_not_delete_the_sidecar_without_asking(tmp_path, qtbot):
    """Pre-fix this unlinked the file — note and all — with no prompt and no status line."""
    s, doc_id, doc = _session(tmp_path)
    p = _write_sidecar_with_note(tmp_path)
    panel = _panel(qtbot, s, doc_id, 0, confirm=False)      # user says "Cancel"

    panel._reset()

    assert p.exists(), "Reset deleted the user's calibration sidecar without consent"
    assert _note_in(p) == NOTE, "the note the app cannot round-trip was destroyed"
    assert "Nothing changed" in panel.status_text()


def test_reset_deletes_the_sidecar_when_the_user_agrees_and_says_so(tmp_path, qtbot):
    """Consent given, deletion happens — and is REPORTED. Silent success is still silent."""
    s, doc_id, doc = _session(tmp_path)
    p = _write_sidecar_with_note(tmp_path)
    panel = _panel(qtbot, s, doc_id, 0, confirm=True)

    panel._reset()

    assert not p.exists()
    assert p.name in panel.status_text()
    assert "deleted" in panel.status_text()


def test_c12_the_same_deletion_through_APPLY_is_equally_guarded(tmp_path, qtbot):
    """The other door.

    *Apply* with the default values typed in reaches the identical ``unlink``. Guarding only
    the Reset button would have left this one keystroke away — the exact shape of the
    "fixed at one call site" mistake.
    """
    s, doc_id, doc = _session(tmp_path)
    p = _write_sidecar_with_note(tmp_path)
    panel = _panel(qtbot, s, doc_id, 0, confirm=False)

    default = CalibrationSet.default_for(doc.meta.channels[0])
    panel._gain.setText(f"{default.gain:g}")
    panel._offset.setText(f"{default.offset:g}")
    panel._units.setText(default.target_units)
    panel._apply()

    assert p.exists(), "Apply-at-defaults deleted the sidecar without consent"
    assert _note_in(p) == NOTE


def test_a_normal_edit_still_writes_the_sidecar_without_any_prompt(tmp_path, qtbot):
    """The guard must not turn every calibration edit into a dialog."""
    s, doc_id, _ = _session(tmp_path)
    asked = {"n": 0}
    panel = CalibrationPanel(s)
    qtbot.addWidget(panel)

    def counting(path, **kw):
        asked["n"] += 1
        return True

    panel._confirm_sidecar_removal = counting
    panel.set_selection(doc_id, 0)
    panel._gain.setText("0.42")
    panel._offset.setText("0")
    panel._apply()

    assert asked["n"] == 0
    p = sidecar_path(tmp_path / "rec.abf")
    assert p.exists() and json.loads(p.read_text())["channels"]["0"]["gain"] == 0.42
    assert p.name in panel.status_text()


def test_resetting_when_there_is_no_sidecar_asks_nothing(tmp_path, qtbot):
    """Nothing to destroy, nothing to confirm."""
    s, doc_id, _ = _session(tmp_path)
    asked = {"n": 0}
    panel = CalibrationPanel(s)
    qtbot.addWidget(panel)
    panel._confirm_sidecar_removal = lambda path, **kw: (
        asked.__setitem__("n", asked["n"] + 1), True)[1]
    panel.set_selection(doc_id, 0)

    panel._reset()

    assert asked["n"] == 0
    assert not sidecar_path(tmp_path / "rec.abf").exists()


# ================================================================================================
# THE GUARD — one commit path, so a third door cannot be added quietly
# ================================================================================================
def _set_calibration_callers(src: str) -> list[str]:
    """Names of the functions that call ``session.set_calibration`` in ``src``."""
    tree = ast.parse(src)
    out: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_calibration"):
                out.append(fn.name)
    return out


def test_the_calibration_panel_commits_through_exactly_one_function():
    """Every path that can delete the sidecar must pass the confirmation.

    ``_apply`` and ``_reset`` both reach the same ``unlink``; a future third button would
    too. Pinning the commit to one function is what makes the guard checkable at all.
    """
    src = (pathlib.Path(nanoscope.__file__).parent / "gui" / "calibration_panel.py").read_text()
    assert _set_calibration_callers(src) == ["_commit"]


def test_the_commit_guard_would_actually_catch_a_violation():
    src = (
        "class P:\n"
        "    def _commit(self, cal):\n"
        "        self._session.set_calibration(1, 2, cal)\n"
        "    def _reset(self):\n"
        "        self._session.set_calibration(1, 2, DEFAULT)\n"
    )
    assert sorted(_set_calibration_callers(src)) == ["_commit", "_reset"]
