"""JAB-02 / JHT-01 — the export call sites hand the DOCUMENT to the clock seam.

THE SEAM WAS ALREADY THERE AND NOBODY CALLED IT. ``analysis.scoring`` grew
:func:`~nanoscope.analysis.scoring.clock_stamp` and a ``meta=``/``t0_s=`` parameter on
all three writers, and ``tests/analysis/test_s3_clock_export_sidecar.py`` pins that seam
in isolation. The defect survived anyway, because ``main_window`` called
``write_events_export(...)`` with neither: ``clock_stamp`` fell through to
``provenance["t0_s"]``, found nothing, and wrote ``"t0_s": null`` — a sidecar that says
"the surface that wrote this file did not know the origin" about a window that was
displaying ``t₀ = 160.0000 s`` in the event table's own tooltip at that very moment.

So this file tests THE CALL SITE, not the writer. Every assertion below fires the real
``Analysis ▸ …`` QAction on a wired :class:`MainWindow` and reads the VALUE out of the
file on disk. Presence of the key is not the property under test — the key was present
and null while the app knew the number, which is exactly the failure this pins:

* whole-file load  -> the sidecar's ``t0_s`` IS ``meta.t0_s``;
* windowed load    -> it is the WINDOW's own origin, so two windows of one recording
  no longer produce two key-identical sidecars dating the same physical event
  6.00000 s apart (the JAB-02 independent reproduction);
* segment (rebased) -> the PARENT's instrument clock, not the rebased 0.0 — the window
  hands over the meta and never picks between ``clock.instrument_origin_s`` and
  ``clock.clock_offset_s``, which differ by the whole cut offset;
* recording closed -> an explicit null and ``t0_s_known: False``, never a fabricated 0;
* and all three scoring exports — events, training, sub-state levels — are wired, not
  just the one the report named.

The note in the file is compared against the on-screen event-table header tooltip
verbatim: one clock, one sentence, whether it is read on the screen or in the folder a
co-author receives.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import scoring
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.segment import build_segment_document
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
SPANS = [(10_000, 10_600), (20_000, 20_800), (30_000, 30_500), (40_000, 40_700)]

#: The owner's quick apoferritin ABF's real origin, measured (not quoted) in the report:
#: an event the CSV dates at 0.2524 s happened at 1155.2433 s on the instrument clock.
T0_APO = 1154.99091246287


# ================================================================================================
# builders — a document whose t0_s is a REAL recording's, so a dropped origin is visible
# ================================================================================================
def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in SPANS:
        x[s:e] -= DEPTH
    return x.astype(np.float32)


def _doc(tmp_path, *, t0_s: float, name: str = "apo.dat", seed: int = 0) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    signal = _signal(seed)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,),
        t0_s=float(t0_s))
    return TraceDocument(meta=meta, channels=(signal,))


def _adopt(win, doc) -> str:
    """Add ``doc`` to the window's session, select it, and give it a real detection."""
    doc_id = win._session.add_document(doc)
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(np.asarray(doc.channels[0]), FS, baseline=I0,
                             min_dwell_s=1e-4, recording=doc.meta.path.name,
                             pore_id="pore-1")
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    # The gate that owns the Analysis menu's enabled state. Asked for explicitly because
    # a QAction that is disabled swallows trigger() silently, which would make every
    # assertion below vacuous.
    win._sync_export_action()
    return doc_id


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _fire(win, action, dest, monkeypatch):
    """Trigger a real export QAction, answering only the file dialog it would raise."""
    monkeypatch.setattr(mw, "ask_export_path", lambda *a, **k: dest)
    assert action.isEnabled(), f"{action.text()!r} is disabled — nothing was exported"
    action.trigger()
    return json.loads(dest.with_suffix(".meta.json").read_text())


def _table_tooltip(win) -> str:
    """What the on-screen event table says about the ``t_start_s`` column's clock."""
    model = win._analysis_panel._table._model
    col = list(model.df.columns).index("t_start_s")
    return model.headerData(col, QtCore.Qt.Horizontal, QtCore.Qt.ToolTipRole)


# ================================================================================================
# Analysis ▸ Export events… — the surface the report named
# ================================================================================================
def test_the_events_export_stamps_the_documents_own_origin(win, tmp_path, monkeypatch):
    """THE VALUE, not the key. ``t0_s`` was already in the file and it was ``null``."""
    _adopt(win, _doc(tmp_path, t0_s=T0_APO))
    dest = tmp_path / "events.csv"
    meta = _fire(win, win._act_export_events, dest, monkeypatch)

    assert meta[scoring.CLOCK_KEY] == pytest.approx(T0_APO, abs=1e-9)
    assert meta[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is True
    # ...and the frozen 20-column schema is still frozen: the origin rode the sidecar.
    assert "t0_s" not in pd.read_csv(dest).columns


def test_the_exported_time_plus_the_stamp_is_the_instant_the_trace_showed(win, tmp_path,
                                                                          monkeypatch):
    """THE WHOLE POINT: the export is now reversible onto the instrument clock.

    ``t_start_s + t0_s`` is where the user saw the event, which is the definition
    ``analysis.measure.MEASUREMENT_SCHEMA`` gives the identically-spelled column of the
    hand-measurement CSV written into the same folder.
    """
    doc_id = _adopt(win, _doc(tmp_path, t0_s=T0_APO))
    dest = tmp_path / "events.csv"
    meta = _fire(win, win._act_export_events, dest, monkeypatch)

    first = float(pd.read_csv(dest)["t_start_s"].iloc[0])
    on_the_trace = first + meta[scoring.CLOCK_KEY]
    x = win._session.x_axis(doc_id).values
    i0 = int(win._analysis_panel.current_detection().payload.accepted.df["start_idx"].iloc[0])
    assert on_the_trace == pytest.approx(float(x[i0]), abs=1e-6)


def test_the_file_and_the_screen_say_the_same_sentence(win, tmp_path, monkeypatch):
    """ONE CLOCK, ONE WORDING. The app annotated the column on screen and stripped the
    annotation on the way out; the sentence in the file is now the table header's own."""
    _adopt(win, _doc(tmp_path, t0_s=T0_APO))
    meta = _fire(win, win._act_export_events, tmp_path / "events.csv", monkeypatch)
    assert meta[scoring.CLOCK_BLOCK_KEY]["note"] == _table_tooltip(win)


# ================================================================================================
# the windowed load — where the missing origin was UNRECOVERABLE
# ================================================================================================
def test_two_windows_of_one_recording_no_longer_write_indistinguishable_sidecars(
        win, qapp, qtbot, tmp_path, monkeypatch):
    """JAB-02's independent reproduction, inverted.

    Two partial loads of one file produced sidecars that were key-identical
    (``analysed_start_idx: 0`` in both, no window origin anywhere) while dating the same
    physical event 6.00000 s apart. A partial load's ``meta.t0_s`` IS the window's
    origin, so stamping it is what tells the two exports apart — and what puts either of
    them back on the instrument clock.
    """
    a_t0, b_t0 = T0_APO, T0_APO + 6.0
    metas = []
    for i, t0 in enumerate((a_t0, b_t0)):
        w = MainWindow()
        qtbot.addWidget(w)
        _adopt(w, _doc(tmp_path, t0_s=t0, name=f"win{i}.dat"))
        metas.append(_fire(w, w._act_export_events, tmp_path / f"w{i}.csv", monkeypatch))

    assert metas[0][scoring.CLOCK_KEY] == pytest.approx(a_t0, abs=1e-9)
    assert metas[1][scoring.CLOCK_KEY] == pytest.approx(b_t0, abs=1e-9)
    assert metas[1][scoring.CLOCK_KEY] - metas[0][scoring.CLOCK_KEY] == pytest.approx(6.0)


def test_a_segment_export_carries_its_parents_clock_not_its_rebased_zero(
        win, tmp_path, monkeypatch):
    """The call site hands over the META and never picks a clock function.

    A cut's own ``meta.t0_s`` is 0 by construction (``core.segment`` rebases it) while
    ``clock.instrument_origin_s`` — MEASUREMENT_SCHEMA's definition of this key — is the
    instrument instant of the cut's first sample. Reading ``meta.t0_s`` at the call site
    would have written 0.0 here: a measurements.csv and an events.meta.json in ONE folder
    disagreeing by the whole cut offset, under one key name.
    """
    parent = _doc(tmp_path, t0_s=T0_APO, name="parent.dat")
    i0 = 5_000
    seg = build_segment_document(parent, i0, N, name="cut", segment_uid="seg-1",
                                 parent_key=str(parent.meta.path))
    assert seg.meta.t0_s == 0.0                      # the rebase this test exists about
    _adopt(win, seg)
    meta = _fire(win, win._act_export_events, tmp_path / "seg.csv", monkeypatch)

    expected = T0_APO + i0 / FS
    assert meta[scoring.CLOCK_KEY] == pytest.approx(expected, abs=1e-9)
    assert meta[scoring.CLOCK_BLOCK_KEY]["rebased_from_s"] == pytest.approx(expected, abs=1e-9)


# ================================================================================================
# UNKNOWN IS SAID, NEVER GUESSED
# ================================================================================================
def test_a_recording_with_no_time_base_is_not_given_a_fabricated_zero(win, tmp_path,
                                                                      monkeypatch):
    """A document with no sample rate HAS no clock, and the export says so rather than
    stamping 0.0 — which would be indistinguishable from a recording that really began
    at the start of the experiment."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    sig = _signal()
    doc = TraceDocument(
        meta=RecordingMeta(path=tmp_path / "norate.dat", format_name="synthetic",
                           n_samples=int(sig.size), sample_rate_hz=None,
                           rate_provenance=Provenance.UNKNOWN, channels=(info,)),
        channels=(sig,))
    _adopt(win, doc)
    meta = _fire(win, win._act_export_events, tmp_path / "norate.csv", monkeypatch)

    assert meta[scoring.CLOCK_KEY] is None
    assert meta[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is False


# ================================================================================================
# the OTHER two scoring exports — the report named one surface; three share the writer
# ================================================================================================
def test_the_training_export_is_wired_to_the_same_seam(win, tmp_path, monkeypatch):
    """``Analysis ▸ Export events for training…`` writes the table a StudentModel
    notebook trains on. Its ``TRAINING_COLUMNS`` are frozen too, and its clock was
    dropped the same way."""
    from nanoscope.analysis.labels import LabelVocabulary

    _adopt(win, _doc(tmp_path, t0_s=T0_APO))
    vocabulary = LabelVocabulary()
    vocabulary.add("molecule", "apo-ferritin", hotkey="c", note="500 pM standard")
    win._label_vocab = vocabulary
    dest = tmp_path / "training.csv"
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "clickedButton",
        lambda self: next(b for b in self.buttons() if "Accepted candidates" in b.text()))
    meta = _fire(win, win._act_export_training, dest, monkeypatch)

    assert meta[scoring.CLOCK_KEY] == pytest.approx(T0_APO, abs=1e-9)
    assert meta[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is True
    assert meta["label_vocabulary"] == vocabulary.to_json()


def test_the_levels_export_is_wired_to_the_same_seam(win, tmp_path, monkeypatch):
    """A level's ``t_start_s`` is on the event table's clock by construction, so a level
    table joined back to an events CSV inherits whatever the events CSV got wrong."""
    _adopt(win, _doc(tmp_path, t0_s=T0_APO))
    win._fit_levels()
    if getattr(win, "_level_fit", None) is None or win._level_fit.payload is None:
        pytest.skip("CUSUM found no sub-states in this synthetic trace")
    win._sync_export_action()
    meta = _fire(win, win._act_export_levels, tmp_path / "levels.csv", monkeypatch)

    assert meta[scoring.CLOCK_KEY] == pytest.approx(T0_APO, abs=1e-9)
    assert meta[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is True


# ================================================================================================
# the helper itself — one document answers both questions
# ================================================================================================
def test_the_source_file_and_the_clock_come_from_one_document(win, tmp_path):
    """``_export_source`` exists so a sidecar cannot name one recording and stamp
    another's origin: both halves are read from the same live document, or neither is."""
    doc_id = _adopt(win, _doc(tmp_path, t0_s=T0_APO))
    source_file, meta = win._export_source(doc_id)
    assert source_file == str(tmp_path / "apo.dat")
    assert meta is win._session.document(doc_id).meta

    # ...and once the recording is closed there is no meta to read, which the sidecar
    # records as an honest unknown rather than a fabricated zero.
    win._session.close_document(doc_id)
    closed_source, closed_meta = win._export_source(doc_id)
    assert closed_meta is None
    assert closed_source  # source_of() still names the file the rows came from
    assert scoring.clock_stamp(meta=closed_meta)[scoring.CLOCK_KEY] is None
