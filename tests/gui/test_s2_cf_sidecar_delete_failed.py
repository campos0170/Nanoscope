"""S2 regression CF2-2 — a Reset that cannot DELETE the old sidecar must not claim the
calibration "will be gone when you reopen".

One fixed SIDECAR_FAILED string used to serve BOTH failure branches of ``save_sidecar``:

* the failed WRITE — nothing was saved, so "the calibration is NOT stored next to your
  data and will be gone when you reopen" is the honest truth;
* the failed DELETE — the OLD sidecar survived the refused unlink, so the same sentence
  is exactly backwards: the reset is what is session-only, and the surviving file
  silently re-applies its gain (and note) on next open.

A user who confirms "Reset and delete the file" on a read-only folder (network share,
archived drive) was told their recording would reopen at defaults; it reopened with the
gain they had just tried to remove. The invariant under test: the two failures are two
outcomes (``SIDECAR_FAILED`` vs ``SIDECAR_DELETE_FAILED``) all the way from
``save_sidecar``'s return to the panel's status line, and each status sentence states
the consequence that is true for ITS branch — with the OS's own reason in both.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import stat

import numpy as np
import pytest

from nanoscope.core.calibration import (
    SIDECAR_DELETE_FAILED,
    SIDECAR_FAILED,
    SIDECAR_REMOVED,
    SIDECAR_WRITTEN,
    CalibrationSet,
    ChannelCalibration,
    save_sidecar,
    sidecar_path,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.calibration_panel import CalibrationPanel


# ================================================================================================
# builders
# ================================================================================================

def _current_doc(tmp_path, name: str = "rec.dat") -> TraceDocument:
    """A single current channel shaped like the real recordings — enough for a sidecar."""
    n = 2048
    cur = np.ascontiguousarray(np.full(n, -2000.0, dtype=np.float32))
    c = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(c,))
    return TraceDocument(meta=meta, channels=(cur,))


def _info() -> ChannelInfo:
    return ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)


@pytest.fixture
def read_only_dir(tmp_path):
    """Yield tmp_path, flipped read-only by the test itself; always restored for cleanup."""
    try:
        yield tmp_path
    finally:
        os.chmod(tmp_path, stat.S_IRWXU)


# ================================================================================================
# core — save_sidecar's return contract distinguishes the two failures
# ================================================================================================

def test_refused_unlink_returns_delete_failed_with_the_os_reason(read_only_dir):
    """All-default payload + undeletable existing sidecar -> SIDECAR_DELETE_FAILED, never
    the write-failure code, and ``reason`` carries the OS's own words."""
    tmp_path = read_only_dir
    data = tmp_path / "rec.dat"
    data.write_bytes(b"DATA")
    calib = CalibrationSet()
    calib.set("d", 0, ChannelCalibration(gain=2.0, offset=0.0, target_units="pA"))
    assert save_sidecar(data, "d", calib, (_info(),)) == SIDECAR_WRITTEN
    old_blob = sidecar_path(data).read_text()

    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)         # unlink is now refused
    reason: list[str] = []
    outcome = save_sidecar(data, "d", CalibrationSet(), (_info(),), reason)

    assert outcome == SIDECAR_DELETE_FAILED
    assert outcome != SIDECAR_FAILED                        # the codes must never merge again
    assert reason and ("denied" in reason[0].lower() or "read-only" in reason[0].lower()), \
        reason
    os.chmod(tmp_path, stat.S_IRWXU)
    # The whole point: the old file survived, byte for byte, and will re-apply on load.
    assert sidecar_path(data).read_text() == old_blob


def test_refused_write_still_returns_the_write_failure_code(read_only_dir):
    """The split must not disturb the write branch: a refused WRITE stays SIDECAR_FAILED
    (pinned wording in the panel and tests/core/test_persist_invariant.py depend on it)."""
    tmp_path = read_only_dir
    data = tmp_path / "rec.dat"
    data.write_bytes(b"DATA")
    calib = CalibrationSet()
    calib.set("d", 0, ChannelCalibration(gain=2.0, offset=0.0, target_units="pA"))

    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)         # no create possible
    reason: list[str] = []
    outcome = save_sidecar(data, "d", calib, (_info(),), reason)

    assert outcome == SIDECAR_FAILED
    assert reason, "the write failure must still carry the OS reason"
    os.chmod(tmp_path, stat.S_IRWXU)
    assert not sidecar_path(data).exists()                  # nothing was saved — 'gone' is true


def test_absent_sidecar_and_default_payload_is_still_none_when_dir_is_read_only(read_only_dir):
    """No sidecar on disk, all-default payload, read-only dir: nothing to delete, so this is
    SIDECAR_NONE — not a failure of any kind (missing_ok covers it without touching the dir)."""
    tmp_path = read_only_dir
    data = tmp_path / "rec.dat"
    data.write_bytes(b"DATA")
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    outcome = save_sidecar(data, "d", CalibrationSet(), (_info(),))
    assert outcome not in (SIDECAR_FAILED, SIDECAR_DELETE_FAILED, SIDECAR_REMOVED)


# ================================================================================================
# gui — the status line states the consequence that is true for its branch
# ================================================================================================

@pytest.fixture
def cal_panel(qapp, qtbot):
    session = Session()
    panel = CalibrationPanel(session)
    qtbot.addWidget(panel)
    return panel, session


def _calibrated_doc_with_sidecar(panel, session, tmp_path):
    """A doc whose gain=2 sidecar is already on disk, selected in the panel."""
    doc = _current_doc(tmp_path)
    doc_id = session.add_document(doc)
    session.set_calibration(doc_id, 0, ChannelCalibration(
        gain=2.0, offset=0.0, target_units="pA", provenance=Provenance.UNKNOWN))
    sc = sidecar_path(doc.meta.path)
    assert sc.exists(), "setup: the gain=2 sidecar must be on disk before the reset"
    panel.set_selection(doc_id, 0)
    return doc_id, sc


def test_reset_with_undeletable_sidecar_says_the_old_file_survives(cal_panel, read_only_dir):
    """Reset, confirmation accepted, unlink refused: the status must say the OLD file could
    not be deleted and WILL re-apply — and must NOT claim the calibration will be gone."""
    panel, session = cal_panel
    tmp_path = read_only_dir
    doc_id, sc = _calibrated_doc_with_sidecar(panel, session, tmp_path)
    old_blob = sc.read_text()

    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    panel._confirm_sidecar_removal = lambda path, **kw: True    # the injectable seam
    panel._reset()

    status = panel.status_text()
    # The reset itself did land in-session; the FILE is what failed.
    assert session.calibration(doc_id, 0).gain == 1.0
    # The consequence sentence of the WRITE branch is forbidden here — it is the inversion
    # this test exists to pin: the calibration is NOT gone on reopen, it comes back.
    assert "will be gone when you reopen" not in status, status
    # The delete branch's own truths: which act failed, the OS's reason, and the fact that
    # the surviving file re-applies on reopen.
    assert "could NOT delete" in status, status
    assert "denied" in status.lower() or "errno" in status.lower(), status
    assert "re-apply" in status, status
    os.chmod(tmp_path, stat.S_IRWXU)
    assert sc.read_text() == old_blob                           # the file it talks about


def test_apply_with_default_values_hits_the_same_delete_branch_honestly(cal_panel,
                                                                        read_only_dir):
    """Apply with gain=1/offset=0 raises the same deletion (and the same refused unlink);
    the honest delete-failure sentence must cover this path too, not only the Reset button."""
    panel, session = cal_panel
    tmp_path = read_only_dir
    # target_units "" here: Apply reads the units field back, and only values that ALL
    # equal the channel's default (gain 1, offset 0, units "") make Apply a deletion.
    doc = _current_doc(tmp_path)
    doc_id = session.add_document(doc)
    session.set_calibration(doc_id, 0, ChannelCalibration(
        gain=2.0, offset=0.0, target_units="", provenance=Provenance.UNKNOWN))
    sc = sidecar_path(doc.meta.path)
    assert sc.exists(), "setup: the gain=2 sidecar must be on disk before the Apply"
    panel.set_selection(doc_id, 0)

    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    panel._confirm_sidecar_removal = lambda path, **kw: True
    panel._gain.setText("1")
    panel._offset.setText("0")
    panel._units.setText("")
    panel._apply()

    status = panel.status_text()
    assert "will be gone when you reopen" not in status, status
    assert "could NOT delete" in status and "re-apply" in status, status
    os.chmod(tmp_path, stat.S_IRWXU)
    assert sc.exists()


def test_refused_write_status_keeps_its_true_gone_on_reopen_sentence(cal_panel,
                                                                     read_only_dir):
    """The split must not weaken the write branch's message: Apply of a NON-default gain
    into a read-only dir still says the calibration is not stored and will be gone —
    which is true there, because nothing was written."""
    panel, session = cal_panel
    tmp_path = read_only_dir
    doc = _current_doc(tmp_path)
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)

    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)
    panel._gain.setText("2")
    panel._apply()

    status = panel.status_text()
    assert "could NOT update" in status, status
    assert "will be gone when you reopen" in status, status
    assert session.calibration(doc_id, 0).gain == 2.0           # applied in-session
    os.chmod(tmp_path, stat.S_IRWXU)
    assert not sidecar_path(doc.meta.path).exists()
