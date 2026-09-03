"""Round-trip of a project containing a SEGMENT (derived recording) — ``core/project.py``.

No Qt: this exercises the pure (de)serializer plus the pure re-derivation
(:func:`build_segment_document`) that the app performs on load. It asserts that

  * a project with a parent + a segment, each carrying its OWN detection result, round-trips:
    the segment spec comes back with the right uid / window / name; the two results come
    back keyed by PERSISTENT KEY (the parent's path, the segment's uid) and are DISTINCT
    (different event tables), never confused for one another;
  * re-deriving the segment from the loaded spec reproduces exactly the parent's crop —
    so the spec alone (uid + window + parent key) is enough to reconstruct the recording;
  * an OLD project (version 1, no ``segments`` key) still loads unchanged.

The detection tables come from the REAL Schmitt detector so the two results genuinely
differ (the segment is a sub-window with a subset of the parent's events).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectSchmitt
from nanoscope.core.document import CalibratedChannel, TraceDocument
from nanoscope.core.project import (
    CalibrationRec,
    DocumentRef,
    ProjectState,
    ResultRec,
    SegmentRec,
    load_project,
    save_project,
)
from nanoscope.core.segment import build_segment_document, persistent_key
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta

FS = 100_000.0
N = 100_000
PARENT_PATH = "/data/pore_a.abf"
SEG_UID = "seg-abc12345-20000-60000"
SEG_I0, SEG_I1 = 20_000, 60_000
# Five events; the segment window [20k, 60k) brackets exactly two of them.
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _parent_doc() -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=Path(PARENT_PATH), format_name="Axon ABF", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(_event_trace(),))


def _detect(arr: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Run the REAL Schmitt detector on a raw current array; return (df, provenance)."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=np.ascontiguousarray(arr, dtype=np.float32),
                           units="pA", sample_rate_hz=FS, t0_s=0.0)
    meta = RecordingMeta(path=Path(PARENT_PATH), format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    res = DetectSchmitt().run(SimpleAnalysisContext(channel=ch, meta=meta,
                                                    params={"min_dwell_s": 5e-4}))
    return res.payload.df.copy(), dict(res.payload.provenance)


# ================================================================================================
# the round-trip
# ================================================================================================

def test_project_with_parent_and_segment_round_trips_with_distinct_results(tmp_path):
    parent = _parent_doc()
    seg = build_segment_document(parent, SEG_I0, SEG_I1, name="pore_a [0.2–0.6 s]",
                                 segment_uid=SEG_UID)

    df_parent, prov_parent = _detect(parent.channels[0])
    df_segment, prov_segment = _detect(seg.channels[0])
    # Sanity: the two detections are genuinely different (the segment is a sub-window).
    assert len(df_parent) == len(_EVENTS)
    assert len(df_segment) == 2
    assert len(df_segment) != len(df_parent)

    state = ProjectState(
        documents=[DocumentRef(path=PARENT_PATH, loaded_range=None)],
        segments=[SegmentRec(segment_uid=SEG_UID, parent_key=PARENT_PATH,
                             i0=SEG_I0, i1=SEG_I1, name="pore_a [0.2–0.6 s]",
                             t0_abs_s=SEG_I0 / FS)],
        calibrations=[
            CalibrationRec(PARENT_PATH, 0, 0.1, 0.0, "pA", "file_meta"),   # keyed by path
            CalibrationRec(SEG_UID, 0, 0.1, 0.0, "pA", "file_meta"),       # keyed by uid
        ],
        # Results keyed by PERSISTENT KEY: the parent by its path, the segment by its uid.
        results=[
            ResultRec(PARENT_PATH, 0, "detect.schmitt", df_parent, prov_parent),
            ResultRec(SEG_UID, 0, "detect.schmitt", df_segment, prov_segment),
        ],
    )

    out = save_project(tmp_path / "seg.nanoscope", state)
    back = load_project(out)

    # -- the segment spec survives (uid / window / name) --------------------------------------
    assert len(back.segments) == 1
    s = back.segments[0]
    assert (s.segment_uid, s.parent_key, s.i0, s.i1, s.name) == (
        SEG_UID, PARENT_PATH, SEG_I0, SEG_I1, "pore_a [0.2–0.6 s]")
    assert s.t0_abs_s == pytest.approx(SEG_I0 / FS)

    # -- both results survive, keyed distinctly, tables NOT confused ---------------------------
    by_key = {r.doc_id: r for r in back.results}
    assert set(by_key) == {PARENT_PATH, SEG_UID}
    assert_frame_equal(by_key[PARENT_PATH].events, df_parent)
    assert_frame_equal(by_key[SEG_UID].events, df_segment)
    assert len(by_key[SEG_UID].events) != len(by_key[PARENT_PATH].events)   # distinct

    # -- the spec alone re-derives the SAME crop (uid + window is enough) ----------------------
    rederived = build_segment_document(parent, s.i0, s.i1, name=s.name,
                                       segment_uid=s.segment_uid)
    # The identity that had to survive is the PERSISTENT one — that is what the stored
    # calibrations and results are keyed by. ``doc_id`` deliberately does NOT survive: it
    # carries a per-load token, and a re-derivation is a new load (see core/identity.py).
    assert persistent_key(rederived) == persistent_key(seg) == SEG_UID
    assert rederived.source_key == seg.source_key
    assert rederived.doc_id != seg.doc_id
    assert rederived.meta.n_samples == SEG_I1 - SEG_I0
    assert np.array_equal(rederived.channels[0], parent.channels[0][SEG_I0:SEG_I1])


def test_segment_calibration_keyed_by_uid_round_trips(tmp_path):
    """A segment's own calibration is stored under its uid (not the parent's path), so it
    re-attaches to the re-derived segment and never bleeds into the parent."""
    state = ProjectState(
        documents=[DocumentRef(path=PARENT_PATH)],
        segments=[SegmentRec(SEG_UID, PARENT_PATH, SEG_I0, SEG_I1, "seg")],
        calibrations=[
            CalibrationRec(PARENT_PATH, 0, 1.0, 0.0, "", "unknown"),
            CalibrationRec(SEG_UID, 0, 0.037, -1.0, "pA", "unknown"),   # an INDEPENDENT edit
        ],
    )
    back = load_project(save_project(tmp_path / "c.nanoscope", state))
    cals = {(c.doc_id): c for c in back.calibrations}
    assert (cals[SEG_UID].gain, cals[SEG_UID].offset) == (0.037, -1.0)
    assert (cals[PARENT_PATH].gain, cals[PARENT_PATH].offset) == (1.0, 0.0)


# ================================================================================================
# back-compat: an old (version 1, no `segments`) project still loads
# ================================================================================================

def test_old_project_without_segments_still_loads(tmp_path):
    # Save a normal project, then rewrite its manifest to version 1 with the `segments` key
    # DROPPED — exactly what a project written before segmentation looks like on disk.
    df, prov = _detect(_event_trace())
    state = ProjectState(
        documents=[DocumentRef(path=PARENT_PATH, loaded_range=(0, 50_000))],
        results=[ResultRec(PARENT_PATH, 0, "detect.schmitt", df, prov)],
    )
    out = save_project(tmp_path / "v2.nanoscope", state)

    members = {n: zipfile.ZipFile(out).read(n) for n in zipfile.ZipFile(out).namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["version"] = 1
    manifest.pop("segments", None)                     # a real v1 project has no such key
    members["manifest.json"] = json.dumps(manifest).encode()
    old = tmp_path / "v1_no_segments.nanoscope"
    with zipfile.ZipFile(old, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    back = load_project(old)                            # must NOT raise a version error
    assert back.segments == []                          # no segments -> empty, unchanged behaviour
    assert [(d.path, d.loaded_range) for d in back.documents] == [(PARENT_PATH, (0, 50_000))]
    assert len(back.results) == 1
    assert_frame_equal(back.results[0].events, df)      # the parent's result is intact
