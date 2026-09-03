"""NE-02 — the float64->float32 downcast may no longer destroy a recording in silence.

A valid HDF5 whose float64 samples all lie below float32's smallest subnormal (~1.4e-45)
used to load as identically zero with ``meta.issues == ()``, and the only diagnostic shown
— dead_reason "constant 0.0 (sampled N)" — asserted a property no sample in the file has.
The overflow direction of the very same cast was loud twice over (numpy's RuntimeWarning,
core.document's every-sample-non-finite issue) and integer rounding past 2^24 had its own
advisory, so underflow was the one representability failure of this cast that said nothing.

The invariant restored, at the seam: ``io.base.DowncastAudit`` is one rule for the three
readers whose sources can hold what float32 cannot (HDF5 and TDMS store float64 natively;
a text file can spell any decimal). Each reader passes every source block and its float32
destination through the shared audit; destruction is reported through one sentence, and a
channel whose flat line is the CAST's doing gets a dead hint that says so instead of
blaming the file. The ABF reader is deliberately not wired: its disk dtypes are int16 and
float32, so no value narrower than float32 can exist in the file (see abf.load's comment).
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest
from nptdms import ChannelObject, TdmsWriter

from nanoscope.core.types import SampleRange
from nanoscope.io.base import open_document

FS = 50_000.0
TINY = 1e-310            # nonzero in float64, exactly 0.0 in float32
SURVIVOR = 1e-40         # below float32 *normal* range but kept as a subnormal


def _write_h5(path: Path, current: np.ndarray) -> Path:
    with h5py.File(path, "w") as h:
        h["current"] = current
        h["time"] = np.arange(len(current), dtype=np.float64) / FS
    return path


def _audit_issues(doc) -> list[str]:
    return [s for s in doc.meta.issues if "float32" in s and "0.0" in s and "below" in s]


# ================================================================================================
# 1. HDF5: total destruction is reported, and the dead hint stops blaming the file
# ================================================================================================

def test_h5_total_underflow_gets_an_issue_and_an_honest_dead_hint(tmp_path):
    rng = np.random.default_rng(7)
    cur = rng.normal(0.0, 1.0, 20_000) * TINY
    cur[cur == 0.0] = TINY                       # every sample nonzero in the file
    p = _write_h5(tmp_path / "under.h5", cur)

    doc = open_document(p)

    # The samples really were destroyed (that is the float32 contract, unchanged)...
    assert doc.channels[0].dtype == np.float32
    assert int(np.count_nonzero(doc.channels[0])) == 0

    # ...but no longer in silence: one issue names the destruction, its scale, and the cast.
    issues = _audit_issues(doc)
    assert len(issues) == 1, f"expected one downcast issue, got {doc.meta.issues!r}"
    assert "/current" in issues[0]
    assert "float64" in issues[0]
    assert "every one of its 20,000 nonzero samples" in issues[0]
    assert "identically zero" in issues[0]

    # And the dead hint tells the same story instead of asserting the FILE is constant 0.0
    # — a property no sample in the file has.
    info = doc.meta.channels[0]
    assert info.is_dead
    assert not info.dead_reason.startswith("constant 0.0 (sampled")
    assert "float32" in info.dead_reason
    assert "see Issues" in info.dead_reason


def test_h5_partial_underflow_counts_what_was_destroyed(tmp_path):
    cur = np.full(10_000, 100.0)
    cur[:2_500] = TINY
    p = _write_h5(tmp_path / "partial.h5", cur)

    doc = open_document(p)
    issues = _audit_issues(doc)
    assert len(issues) == 1
    assert "2,500 of its 10,000 nonzero samples" in issues[0]
    # The surviving samples are intact.
    assert int(np.count_nonzero(doc.channels[0])) == 7_500


def test_h5_windowed_load_of_a_column_shaped_channel_is_audited_too(tmp_path):
    # The dossier's verifier variant: shape (N, 1), windowed load. The audit must see the
    # window it read, not depend on a whole-file pass.
    cur = np.full((60_000, 1), 4e-46)            # below half the smallest subnormal
    p = tmp_path / "col.h5"
    with h5py.File(p, "w") as h:
        h["current"] = cur
        h["time"] = np.arange(60_000, dtype=np.float64) / FS
    doc = open_document(p, SampleRange(10_000, 50_000))
    issues = _audit_issues(doc)
    assert len(issues) == 1
    assert "40,000 nonzero samples" in issues[0]


# ================================================================================================
# 2. the audit stays silent when the cast is faithful — no new noise on healthy files
# ================================================================================================

def test_h5_float32_subnormals_survive_and_say_nothing(tmp_path):
    # 1e-40 is below float32's NORMAL range but representable as a subnormal: the cast keeps
    # it, so there is nothing to report. This bounds the destruction band the issue covers.
    p = _write_h5(tmp_path / "sub.h5", np.full(10_000, SURVIVOR))
    doc = open_document(p)
    assert int(np.count_nonzero(doc.channels[0])) == 10_000
    assert _audit_issues(doc) == []


def test_h5_ordinary_recording_gains_no_issue(tmp_path):
    rng = np.random.default_rng(3)
    p = _write_h5(tmp_path / "ok.h5", rng.normal(100.0, 1.5, 20_000))
    doc = open_document(p)
    assert _audit_issues(doc) == []
    assert not doc.meta.channels[0].is_dead


def test_h5_float32_source_needs_no_audit(tmp_path):
    # A float32 dataset casts to itself; nothing can be flushed, so nothing is said —
    # downcast_needs_audit gates the per-chunk pass off entirely.
    p = tmp_path / "f32.h5"
    with h5py.File(p, "w") as h:
        h["current"] = np.full(5_000, 1.0, dtype=np.float32)
        h["time"] = np.arange(5_000, dtype=np.float64) / FS
    doc = open_document(p)
    assert _audit_issues(doc) == []


# ================================================================================================
# 3. the same rule holds for the other wide-source readers: TDMS and text
# ================================================================================================

def test_tdms_float64_underflow_gets_the_same_sentence(tmp_path):
    p = tmp_path / "under.tdms"
    props = {"wf_increment": 1.0 / FS, "wf_start_offset": 0.0, "unit_string": "pA"}
    with TdmsWriter(str(p)) as w:
        w.write_segment([ChannelObject("g", "current",
                                       np.full(5_000, TINY), properties=props)])
    doc = open_document(p)
    issues = _audit_issues(doc)
    assert len(issues) == 1
    assert "float64" in issues[0]
    assert "every one of its 5,000 nonzero samples" in issues[0]
    info = doc.meta.channels[0]
    assert info.is_dead
    assert not info.dead_reason.startswith("constant 0.0 (sampled")
    assert "see Issues" in info.dead_reason


def test_csv_sub_float32_decimals_get_the_same_sentence(tmp_path):
    p = tmp_path / "under.csv"
    with p.open("w") as f:
        f.write("time,current\n")
        for i in range(3_000):
            f.write(f"{i / FS:.6e},1e-310\n")
    doc = open_document(p)
    issues = _audit_issues(doc)
    assert len(issues) == 1
    assert "'current'" in issues[0]
    assert "decimal text" in issues[0]
    info = doc.meta.channels[0]
    assert info.is_dead
    assert not info.dead_reason.startswith("constant 0.0 (sampled")


def test_csv_ordinary_recording_gains_no_issue_and_still_loads_float32(tmp_path):
    p = tmp_path / "ok.csv"
    with p.open("w") as f:
        f.write("time,current\n")
        for i in range(3_000):
            f.write(f"{i / FS:.6e},{100.0 + (i % 7) * 0.5}\n")
    doc = open_document(p)
    assert _audit_issues(doc) == []
    assert doc.channels[0].dtype == np.float32
    assert doc.channels[0][0] == np.float32(100.0)


# ================================================================================================
# 4. a genuinely all-zero file keeps its honest "constant 0.0" — the substitution is only
#    for flat lines the cast produced
# ================================================================================================

def test_h5_true_zero_file_keeps_the_constant_zero_hint(tmp_path):
    p = _write_h5(tmp_path / "zero.h5", np.zeros(10_000))
    doc = open_document(p)
    assert _audit_issues(doc) == []
    info = doc.meta.channels[0]
    assert info.is_dead
    assert info.dead_reason.startswith("constant 0.0 (sampled")
