"""End-to-end integration test over the REAL corpus, through the REAL app path.

For every file this test:
  1. probes it (io.base.probe_path) — the cheap metadata read the GUI does first,
  2. loads it (io.base.open_document) — the eager read the LoadWorker runs off-thread,
  3. installs it in a Session and builds the real TraceView (the viewer widget),
  4. asserts the curves the widget would paint are non-empty and finite,
  5. asserts sample-rate / N / duration are physically sane.

It also asserts the two negative contracts:
  * the CPYA-variant ABF is CLAIMED (Claim.STRONG) but REFUSED with an honest message — no crash,
  * Keras/model/split .h5 files are NOT misidentified as recordings — refused, no crash.

Corpus lives in /Users/abrahamcampos/ (outside the repo). A missing file skips its case.

Run:  QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python3 -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import (contract §2). -------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from nanoscope.core.session import Session
from nanoscope.core.types import SampleRange
from nanoscope.gui.trace_view import TraceView
from nanoscope.io.abf import AbfReader
from nanoscope.io.base import Claim, open_document, probe_path, select_reader
from nanoscope.io.errors import ReaderError

CORPUS = Path("/Users/abrahamcampos")
LOAD_CAP = 2_000_000   # cap the loaded window so slow formats stay fast; the FULL facts come from probe()
FS = 50_000.0          # nominal acquisition rate for this lab's rig


# ================================================================================================
# Expectations. N/fs for the deterministic formats are the VERIFIED full-file facts from probing.
# ================================================================================================

class Rec:
    def __init__(self, fname, fmt, *, fs=FS, n=None, min_ch=1, exceeds=False, t0=None):
        self.fname = fname
        self.fmt = fmt
        self.fs = fs               # None => rate genuinely unknown (legacy)
        self.n = n                 # exact full N when deterministic; None => tolerant (text estimate)
        self.min_ch = min_ch
        self.exceeds = exceeds
        self.t0 = t0               # optional exact t0 sanity


RECORDINGS = [
    Rec("500 pM Apo ferritin_1 x PBS_22 nm_0000_Process_-1000 mV.abf",
        "Axon ABF", n=1_003_927, min_ch=3),
    Rec("500 pM Apo ferritin_1 x PBS_22 nm_0004_process _ -1000 mV_ 385-400 s time trace.abf",
        "Axon ABF", n=749_651, min_ch=3),
    Rec("500 pM Apo ferritin_1 x PBS_22 nm_0005_Process_-700 mV.abf",
        "Axon ABF", n=9_074_042, min_ch=3),
    Rec("500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf",
        "Axon ABF", n=2_993_798, min_ch=3),
    Rec("500 pM Apo ferritin_1 x PBS_22 nm_0008_process data_-500 mV.abf",
        "Axon ABF", n=12_502_631, min_ch=3),
    Rec("March_25_ATP 10 nm_12 nm_3_ 1 hydrogel_0005_Process.abf",
        "Axon ABF", n=50_495_999, min_ch=3, exceeds=True),
    Rec("2_3 electroporation-005_Cut_1.h5",
        "HDF5 flat", n=2_000_000, min_ch=2, t0=60.00002),
    Rec("4_detection 100 pM nisin with fibrin -014_Cut_1-Copy1.h5",
        "HDF5 flat", n=9_800_724, min_ch=3),
    Rec("4_detection 100 pM nisin with fibrin -008.tdms",
        "TDMS (modern)", n=13_428_995, min_ch=2),
    Rec("example.tdms",
        "TDMS (legacy)", fs=None, n=20_200, min_ch=2),
    Rec("500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.csv",
        "Text trace", n=None, min_ch=1),          # text N is a probe estimate -> tolerant
    Rec("nisin_fibrin_trace.csv",
        "Text trace", n=None, min_ch=1),
]

# .h5 files that are NOT recordings (Keras models / ML train-val splits). Must be refused, not loaded.
NON_RECORDING_H5 = [
    "complete_spike_detector_model.h5",
    "my_gem_model.h5",
    "training_split.h5",
    "validation_split.h5",
    "mearec_test_10s.h5",
    "mearec_resampled_50k.h5",
]

CPYA_FILE = "500 pM Apo ferritin_1 x PBS_22 nm_0011_Process_-200 mV_30-35.abf"


# ================================================================================================
# Fixtures / helpers
# ================================================================================================

# NO LOCAL ``qapp`` FIXTURE HERE. There was one — ``return create_app([])`` — and because
# a fixture defined in a test module shadows conftest's, every test in this file ran with
# the SHIPPED application name, which is the name the real preferences store is keyed on.
# Measured: a suite run rewrote ``~/Library/Preferences/com.nanoscope-local.NanoScope.plist``
# and left eleven pytest temporary-directory paths in the owner's Recent Projects.
# ``tests/conftest.py`` now redirects the STORE (not the name), so this file is no longer
# the hole it was; the fixture is gone anyway, because the shared ``qapp`` is what every
# other GUI test uses and one QApplication per session is what Qt allows.
# The now-unused ``create_app`` import went with it; conftest's ``qapp`` is the one place
# the QApplication is built.


def _require(fname: str) -> Path:
    p = CORPUS / fname
    if not p.exists():
        pytest.skip(f"corpus file absent: {fname}")
    return p


def _render_through_app(qapp, doc) -> dict:
    """Drive the real viewer: Session -> default layers -> TraceView. Return render facts.

    Mirrors main_window's default-layer policy for the first document: one layer per non-dead
    channel, in channel order. Then inspects the curves the widget would actually paint.
    """
    session = Session()
    session.add_document(doc)

    live = [c.index for c in doc.meta.channels if not c.is_dead]
    assert live, "every channel flagged dead -> nothing to view"
    for ch in live:
        session.add_layer(doc.doc_id, ch)

    view = TraceView(session)
    view.resize(1200, 600)   # give the ViewBox a real extent so downsampling computes

    n_curves = 0
    for pane in view._panes:
        for curve in pane._curves.values():
            x, y = curve.getData()   # the DOWNSAMPLED data pyqtgraph paints (peak-decimated)
            assert x is not None and y is not None, "curve produced no display data"
            assert len(x) > 0 and len(y) > 0, "curve display data is empty"
            assert len(x) == len(y)
            assert np.isfinite(y).all(), "curve carries non-finite y (NaN/inf) into the paint path"
            n_curves += 1

    assert view._panes, "viewer built zero panes for a loaded recording"
    assert n_curves == len(live), "a live channel did not become a painted curve"

    xaxis = session.x_axis(session.layers()[0].doc_id)
    return {"panes": len(view._panes), "curves": n_curves, "x_unit": xaxis.unit,
            "live_channels": len(live)}


# ================================================================================================
# POSITIVE: every real recording opens, renders, and reports sane rate/N/duration.
# ================================================================================================

@pytest.mark.parametrize("rec", RECORDINGS, ids=lambda r: r.fname[:48])
def test_recording_opens_renders_and_is_sane(qapp, rec):
    path = _require(rec.fname)

    # -- probe: the cheap full-file metadata read --------------------------------------------
    probe = probe_path(path)
    meta = probe.meta

    assert meta.format_name == rec.fmt
    assert len(meta.channels) >= rec.min_ch
    assert meta.n_samples > 0
    assert probe.exceeds_ceiling is rec.exceeds

    # -- rate / duration sanity --------------------------------------------------------------
    if rec.fs is None:
        assert meta.sample_rate_hz is None, "fabricated a rate for a rate-unknown file"
        assert meta.duration_s is None
    else:
        assert meta.sample_rate_hz == pytest.approx(rec.fs, rel=0.01)
        assert meta.duration_s == pytest.approx(meta.n_samples / meta.sample_rate_hz)
        assert 0.1 < meta.duration_s < 5000.0, "duration outside any plausible recording length"

    # -- N sanity ----------------------------------------------------------------------------
    if rec.n is not None:
        assert meta.n_samples == rec.n
    else:
        assert meta.n_samples > 1000   # text N is an estimate; just require it isn't degenerate

    if rec.t0 is not None:
        assert meta.t0_s == pytest.approx(rec.t0, abs=1e-3)

    # -- load through the real reader path (bounded window for the huge ones) ----------------
    rng = None if meta.n_samples <= LOAD_CAP else SampleRange(0, LOAD_CAP)
    doc = open_document(path, rng)

    expected_n = meta.n_samples if rng is None else len(rng)
    assert doc.meta.n_samples == expected_n
    for i, arr in enumerate(doc.channels):
        assert arr.dtype == np.float32, f"ch{i} not float32"
        assert arr.ndim == 1 and arr.flags["C_CONTIGUOUS"]
        assert len(arr) == doc.meta.n_samples

    # -- render through the real viewer widget -----------------------------------------------
    facts = _render_through_app(qapp, doc)
    assert facts["x_unit"] == ("sample" if rec.fs is None else "s")


# ================================================================================================
# NEGATIVE: the CPYA-variant ABF is claimed but honestly refused (no crash).
# ================================================================================================

def test_cpya_abf_claimed_strong_but_refused_honestly():
    path = _require(CPYA_FILE)

    # We CLAIM it (so we can tell the truth about it) — Claim.STRONG, and the ABF reader is selected.
    assert AbfReader.sniff(path) is Claim.STRONG
    assert select_reader(path) is AbfReader

    with pytest.raises(ReaderError) as exc:
        probe_path(path)
    assert "CPYA" in str(exc.value), "CPYA refusal must name the variant, not leak a neo traceback"

    # And the generic open path refuses the same way — never a raw TypeError from neo.
    with pytest.raises(ReaderError):
        open_document(path)


# ================================================================================================
# NEGATIVE: ML / model / split .h5 files are NOT misidentified as recordings.
# ================================================================================================

@pytest.mark.parametrize("fname", NON_RECORDING_H5)
def test_ml_h5_not_misidentified_as_recording(fname):
    path = _require(fname)

    # HDF5 magic is real, so the HDF5 reader legitimately CLAIMS the file...
    reader = select_reader(path)
    assert reader.format_name == "HDF5 flat"

    # ...but probing must REFUSE it as a recording with a typed, user-facing error — never crash,
    # never return a bogus RecordingProbe.
    with pytest.raises(ReaderError):
        probe_path(path)
    with pytest.raises(ReaderError):
        open_document(path)
