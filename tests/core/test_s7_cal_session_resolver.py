"""Sweep-7 calibration fixes at the core: ONE resolver for what a channel is, exact raw
extremes on the session, and no sidecar written beside a recording that has moved.

These pin :class:`~nanoscope.core.session.Session` driven exactly as the app drives it —
open a synthetic recording, set a calibration, ask — with no QApplication (a Session is a
plain QObject that constructs and emits without an event loop).

Findings covered (docs/sweep7/FINDINGS.md): S7-CAL1 + the skeptics' MISSED rail/chip and
Calibration-panel items (root cause: three surfaces read the header while two read the
override), S7-CAL3 (strided extremes printed as exact) and S7-CAL5 (a "saved" that landed
beside nothing).
"""

from __future__ import annotations

import nanoscope  # noqa: F401  -> runs bootstrap.py (import-order discipline)

import json
import shutil

import numpy as np
import pytest

from nanoscope.core import calibration as calmod
from nanoscope.core.calibration import SIDECAR_WRITTEN, ChannelCalibration, sidecar_path
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import trace_view

FS = 50_000.0


def _doc(path, channels: tuple[ChannelInfo, ...], arrays=None) -> TraceDocument:
    n = 4_000 if arrays is None else int(arrays[0].size)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=channels)
    if arrays is None:
        arrays = tuple(np.zeros(n, dtype=np.float32) for _ in channels)
    return TraceDocument(meta=meta, channels=tuple(arrays))


def _undeclared(tmp_path) -> TraceDocument:
    """The shape of the owner's nisin TDMS / an unlabelled CSV: nothing declared."""
    return _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)))


def _declared(tmp_path) -> TraceDocument:
    """A normally declared ABF: IN15 current on channel 1, the command voltage on 0."""
    return _doc(tmp_path / "declared.abf", (
        ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
        ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)))


# ================================================================================================
# effective_kind: the override wins, the header stands otherwise
# ================================================================================================

def test_effective_kind_returns_the_override_when_one_is_set_and_the_header_otherwise(tmp_path):
    s = Session()
    did = s.add_document(_undeclared(tmp_path))
    assert s.effective_kind(did, 0) is ChannelKind.UNKNOWN
    s.set_calibration(did, 0, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    assert s.effective_kind(did, 0) is ChannelKind.CURRENT
    assert s.effective_kind(did, 1) is ChannelKind.UNKNOWN, "the sibling is untouched"


def test_the_trace_view_resolver_is_a_delegate_to_the_session_and_not_a_second_opinion(
        tmp_path, monkeypatch):
    """Two implementations of one question is how the app came to say "No current channel"
    about a channel it was drawing with a pA axis. The trace view must ANSWER FROM the
    session, so a change there reaches the panes without a second edit."""
    s = Session()
    did = s.add_document(_undeclared(tmp_path))
    seen: list[tuple[str, int]] = []

    def sentinel(doc_id, ch):
        seen.append((doc_id, ch))
        return ChannelKind.CONDUCTANCE

    monkeypatch.setattr(s, "effective_kind", sentinel)
    assert trace_view.effective_kind(s, did, 1) is ChannelKind.CONDUCTANCE
    assert seen == [(did, 1)]


# ================================================================================================
# resolve_current_channel: the pipeline's channel, by effective kind
# ================================================================================================

def test_a_channel_corrected_to_current_resolves_as_the_pipeline_channel(tmp_path):
    s = Session()
    did = s.add_document(_undeclared(tmp_path))
    assert s.resolve_current_channel(did, 0) is None, "nothing declared, nothing overridden"
    s.set_calibration(did, 0, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    assert s.resolve_current_channel(did, 0) == 0
    assert s.resolve_current_channel(did, 1) == 0, "selecting the sibling falls back to it"
    assert s.resolve_current_channel(did, None) == 0
    assert s.is_live_current(did, 0) and not s.is_live_current(did, 1)


def test_a_header_current_channel_re_declared_as_voltage_stops_resolving(tmp_path):
    """The reverse override: the user says IN15 is really the command channel. Every
    surface that asks here now agrees there is no current channel — before, Batch detect
    and the PSD greyed out while Run detection stayed enabled on it."""
    s = Session()
    did = s.add_document(_declared(tmp_path))
    assert s.resolve_current_channel(did, 0) == 1, "voltage selected -> the analyte channel"
    s.set_calibration(did, 1, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.VOLTAGE),
                      write_sidecar=False)
    assert s.effective_kind(did, 1) is ChannelKind.VOLTAGE
    assert s.resolve_current_channel(did, 1) is None
    assert s.resolve_current_channel(did, None) is None


def test_a_dead_channel_does_not_resolve_even_when_corrected_to_current(tmp_path):
    s = Session()
    did = s.add_document(_doc(tmp_path / "dead.csv", (
        ChannelInfo(index=0, name="flat", raw_units="", kind=ChannelKind.UNKNOWN,
                    is_dead=True, dead_reason="no signal"),)))
    s.set_calibration(did, 0, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    assert s.effective_kind(did, 0) is ChannelKind.CURRENT
    assert s.resolve_current_channel(did, 0) is None


# ================================================================================================
# raw_extremes: exact, not strided
# ================================================================================================

def test_raw_extremes_are_exact_over_every_sample_not_a_stride(tmp_path):
    """A 200,000-sample channel whose only positive sample sits at an index no 20,000-sample
    stride visits. The strided maximum is -50 (the current "never goes positive"); the
    exact maximum is +5.49 — the sign flip measured on the owner's -500 mV recording."""
    n = 200_000
    arr = np.full(n, -50.0, dtype=np.float32)
    arr[7] = 5.49          # 7 is not a multiple of the stride (10) the panel would use
    arr[13] = -281.677
    s = Session()
    did = s.add_document(_doc(tmp_path / "spiky.csv", (
        ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),), (arr,)))
    lo, hi = s.raw_extremes(did, 0)
    assert hi == pytest.approx(5.49, rel=1e-6)
    assert lo == pytest.approx(-281.677, rel=1e-6)
    strided = arr[::max(1, n // 20_000)]
    assert float(strided.max()) == -50.0, "the stride misses the spike; the session must not"


def test_raw_extremes_ignore_nan_and_are_nan_for_a_channel_with_no_finite_sample(tmp_path):
    s = Session()
    a = np.array([np.nan, -3.0, 2.0, np.nan], dtype=np.float32)
    b = np.full(4, np.nan, dtype=np.float32)
    did = s.add_document(_doc(tmp_path / "nan.csv", (
        ChannelInfo(index=0, name="a", raw_units="pA", kind=ChannelKind.CURRENT),
        ChannelInfo(index=1, name="b", raw_units="pA", kind=ChannelKind.CURRENT)), (a, b)))
    assert s.raw_extremes(did, 0) == (-3.0, 2.0)
    lo, hi = s.raw_extremes(did, 1)
    assert np.isnan(lo) and np.isnan(hi)


# ================================================================================================
# a moved recording: nothing is written beside nothing, and the outcome says so
# ================================================================================================

def _real_file_doc(path) -> TraceDocument:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real recording; the samples are in memory")
    return _doc(path, (
        ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),))


def test_set_calibration_on_a_moved_recording_writes_nothing_beside_the_old_path_and_says_so(
        tmp_path):
    old, new = tmp_path / "cal5_old", tmp_path / "cal5_new"
    data = old / "pore.abf"
    s = Session()
    did = s.add_document(_real_file_doc(data))
    assert s.recording_moved(did) is False

    assert s.set_calibration(did, 0, ChannelCalibration(0.75, 0.0, "pA")) == SIDECAR_WRITTEN
    side = sidecar_path(data)
    assert json.loads(side.read_text())["channels"]["0"]["gain"] == 0.75

    # The user tidies the recording into another folder in Finder; the app keeps working.
    new.mkdir()
    shutil.move(str(data), str(new / "pore.abf"))
    assert s.recording_moved(did) is True

    cal = ChannelCalibration(0.9, 0.0, "pA")
    moved = calmod.SIDECAR_RECORDING_MOVED      # attribute access: absent on the old tree
    assert s.sidecar_effect_of(did, 0, cal) == moved
    assert s.sidecar_notes_lost_by(did, 0, cal) == ()
    assert s.set_calibration(did, 0, cal) == moved
    assert s.calibration(did, 0).gain == 0.9, "the calibration is live in the session"
    assert json.loads(side.read_text())["channels"]["0"]["gain"] == 0.75, (
        "the orphan in the folder the recording left is NOT rewritten")
    assert not sidecar_path(new / "pore.abf").exists(), "and nothing is invented beside the new path"

    # A document that never had a file is not "moved": its sidecar path is still the
    # place to write, exactly as before (every synthetic recording in this suite).
    did2 = s.add_document(_doc(tmp_path / "never_on_disk.csv", (
        ChannelInfo(index=0, name="c", raw_units="pA", kind=ChannelKind.CURRENT),)))
    assert s.recording_moved(did2) is False
    assert s.set_calibration(did2, 0, ChannelCalibration(2.0, 0.0, "pA")) == SIDECAR_WRITTEN
    assert sidecar_path(tmp_path / "never_on_disk.csv").exists()


def test_the_moved_flag_is_per_document_and_cleared_on_close(tmp_path):
    s = Session()
    data = tmp_path / "a" / "pore.abf"
    did = s.add_document(_real_file_doc(data))
    data.unlink()
    assert s.recording_moved(did) is True
    s.close_document(did)
    assert s.recording_moved(did) is False, "an unknown document has not moved"
