"""S07 — a document with no physical reference must refuse frac_depth from ANY detector.

Before this fix ``fractional_blockage_reference_unavailable`` was known only inside
``study_event_bridge.prepare_review_export``, which stamped it onto its own synthetic
"detect.study" result. That document (a baseline-subtracted, events-down HDF5 export) can
also be reopened and run through the ORDINARY Schmitt/CUSUM/Keras detectors — see
docs/USER_GUIDE.md — and none of those ever set the flag, so a near-zero baseline on an
already-detrended trace produced a fabricated, physically meaningless ``frac_depth``.

The fix makes "no physical reference" a property of the DOCUMENT
(``RecordingMeta.physical_reference_unavailable``), read once by ``io/hdf5.py`` from a
marker ``study_event_bridge.py`` already writes into the file
(``core.types.STUDY_BRIDGE_ATTR_KEY``), and consumed generically by every detector via
``core.acquisition.stamp_reference_availability`` at the same moment each already stamps
``stamp_acquisition_bandwidth``.

This file checks each hop: the reader sets the field only for a bridge-marked file
(``test_hdf5_reader_*``), the shared stamp only ever ADDS the key, never removes it
(``test_stamp_reference_availability_*``), the two threshold detectors refuse frac_depth
end to end (``test_schmitt_*`` / ``test_cusum_*``), the ML detector's source is guarded the
same way the codebase already guards its other Keras-path stamps -- structurally, not by
running a model (``test_ml_modules_*``) -- and, closing the loop, that reopening an actual
bridge-shaped file and running Schmitt on it directly refuses frac_depth with no synthetic
"detect.study" result involved at all (``test_ordinary_detection_on_a_reopened_bridge_export``).
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from nanoscope.analysis import ml_modules
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectCusum, DetectSchmitt
from nanoscope.core.acquisition import stamp_reference_availability
from nanoscope.core.document import CalibratedChannel
from nanoscope.core.types import (
    STUDY_BRIDGE_ATTR_KEY,
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.io.hdf5 import Hdf5Reader

FS = 50_000.0

#: Five square blockades on a flat, noisy open pore -- same shape used by
#: tests/analysis/test_src_detectors_stamp.py for the same two detectors.
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]


def _trace(n: int = 100_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, 100.0) + rng.normal(0.0, 1.5, n)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return x.astype(np.float32)


def _context(data: np.ndarray, *, physical_reference_unavailable: bool) -> SimpleAnalysisContext:
    """A minimal `SimpleAnalysisContext`, built directly rather than through the
    function-scoped `context_factory` fixture so the new RecordingMeta field can be set."""
    info = ChannelInfo(index=0, name="synthetic", raw_units="pA", kind=ChannelKind.CURRENT)
    ch = CalibratedChannel(info=info, data=np.ascontiguousarray(data, dtype=np.float32),
                           units="pA", sample_rate_hz=FS, t0_s=0.0)
    meta = RecordingMeta(path=Path("synthetic.dat"), format_name="synthetic",
                         n_samples=int(ch.data.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,), t0_s=0.0,
                         physical_reference_unavailable=physical_reference_unavailable)
    return SimpleAnalysisContext(channel=ch, meta=meta, params={})


def _write_bridge_style_h5(path: Path, *, stamped: bool) -> Path:
    """A file shaped exactly like study_event_bridge.prepare_review_export's own output:
    /current + /time datasets and, when ``stamped``, the same top-level attribute key the
    bridge writes (``STUDY_BRIDGE_ATTR_KEY`` / the bridge's own ``BRIDGE_KEY``)."""
    signal = _trace()
    t = np.arange(signal.size, dtype=np.float64) / FS
    with h5py.File(path, "w") as h:
        h["current"] = signal
        h["time"] = t
        if stamped:
            h.attrs[STUDY_BRIDGE_ATTR_KEY] = json.dumps(
                dict(version=1, recording_id="r1", units="pA", orientation="events_down"))
    return path


# ===========================================================================
# RecordingMeta — the new field itself
# ===========================================================================
def test_recording_meta_defaults_to_having_a_physical_reference():
    meta = RecordingMeta(path=Path("x.abf"), format_name="synthetic", n_samples=10,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=())
    assert meta.physical_reference_unavailable is False


# ===========================================================================
# io/hdf5.py — the reader is the one place the fact is DISCOVERED
# ===========================================================================
def test_hdf5_reader_load_sets_the_flag_from_the_bridge_marker(tmp_path):
    path = _write_bridge_style_h5(tmp_path / "prepared-review.h5", stamped=True)
    document = Hdf5Reader(path).load()
    assert document.meta.physical_reference_unavailable is True


def test_hdf5_reader_probe_also_sets_the_flag(tmp_path):
    """Both call sites funnel through _build_meta; probe() must not skip it."""
    path = _write_bridge_style_h5(tmp_path / "prepared-review.h5", stamped=True)
    probe = Hdf5Reader(path).probe()
    assert probe.meta.physical_reference_unavailable is True


def test_hdf5_reader_leaves_the_flag_false_for_an_ordinary_recording(tmp_path):
    path = _write_bridge_style_h5(tmp_path / "ordinary.h5", stamped=False)
    document = Hdf5Reader(path).load()
    assert document.meta.physical_reference_unavailable is False


# ===========================================================================
# core/acquisition.py — the shared stamp every detector calls
# ===========================================================================
def test_stamp_reference_availability_adds_the_key_only_when_the_document_has_no_reference():
    ctx = _context(_trace(), physical_reference_unavailable=True)
    prov: dict = {}
    stamp_reference_availability(prov, ctx)
    assert prov == {"fractional_blockage_reference_unavailable": True}


def test_stamp_reference_availability_is_a_no_op_for_an_ordinary_document():
    ctx = _context(_trace(), physical_reference_unavailable=False)
    prov: dict = {}
    stamp_reference_availability(prov, ctx)
    assert prov == {}


# ===========================================================================
# analysis/modules.py — the two threshold detectors, end to end
# ===========================================================================
def test_schmitt_on_a_document_with_no_physical_reference_refuses_frac_depth():
    ctx = _context(_trace(), physical_reference_unavailable=True)
    res = DetectSchmitt().run(ctx)
    df = res.payload.df
    assert len(df) == len(_EVENTS), "fixture produced no events; the rest proves nothing"
    assert res.payload.provenance["fractional_blockage_reference_unavailable"] is True
    assert df["frac_depth"].isna().all()


def test_schmitt_on_an_ordinary_document_still_reports_real_frac_depth():
    """Control: the guard must not blanket-NaN every detection, only the ones on a
    document that says it has no physical reference."""
    ctx = _context(_trace(), physical_reference_unavailable=False)
    res = DetectSchmitt().run(ctx)
    df = res.payload.df
    assert len(df) == len(_EVENTS)
    assert "fractional_blockage_reference_unavailable" not in res.payload.provenance
    assert df["frac_depth"].notna().all()


def test_cusum_on_a_document_with_no_physical_reference_refuses_frac_depth():
    ctx = _context(_trace(), physical_reference_unavailable=True)
    res = DetectCusum().run(ctx)
    df = res.payload.df
    assert len(df) > 0, "fixture produced no events; the rest proves nothing"
    assert df["frac_depth"].isna().all()


# ===========================================================================
# analysis/ml_modules.py — asserted on the SOURCE, as
# test_src_detectors_stamp.test_the_ml_detector_stamps_the_same_pair_at_the_same_moment
# already does for stamp_source_status: a real MLDetect.run needs a .keras file and a
# TensorFlow load, and the property under test is structural.
# ===========================================================================
def test_ml_modules_stamps_reference_availability_right_after_acquisition_bandwidth():
    src = Path(ml_modules.__file__).read_text()
    tree = ast.parse(src)
    calls = [node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("stamp_acquisition_bandwidth", "stamp_reference_availability")]
    assert "stamp_reference_availability" in calls, (
        "ml_modules no longer stamps reference availability, so a Keras detection on a "
        "bridge-exported document would fabricate frac_depth")
    i = calls.index("stamp_acquisition_bandwidth")
    assert calls[i + 1] == "stamp_reference_availability", (
        "stamp_reference_availability must be called immediately after "
        "stamp_acquisition_bandwidth, the same moment every other detector calls it")


# ===========================================================================
# closing the loop: an ACTUAL reopened bridge export, run through Schmitt directly --
# the regression itself, with no synthetic detect.study result anywhere in the path.
# ===========================================================================
def test_ordinary_detection_on_a_reopened_bridge_export_refuses_frac_depth(tmp_path):
    path = _write_bridge_style_h5(tmp_path / "prepared-review.h5", stamped=True)
    document = Hdf5Reader(path).load()
    assert document.meta.physical_reference_unavailable is True

    info = document.meta.channels[0]
    channel = CalibratedChannel(info=info, data=document.raw(0), units="pA",
                                sample_rate_hz=document.meta.sample_rate_hz, t0_s=0.0)
    ctx = SimpleAnalysisContext(channel=channel, meta=document.meta, params={})

    res = DetectSchmitt().run(ctx)
    df = res.payload.df
    assert len(df) == len(_EVENTS), "fixture produced no events; the rest proves nothing"
    assert df["frac_depth"].isna().all()
