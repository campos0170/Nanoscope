"""S3 / JAB-02 + JHT-01: every table this app exports names the clock its seconds are on.

THE DEFECT
----------
``t_start_s`` is on the RECORDING clock — 0 at the first LOADED sample
(:mod:`nanoscope.core.clock`, ``analysis/events.py`` SCHEMA). The trace the user read
those events off is on the INSTRUMENT clock, and so is the ``figs_trace.csv`` the figure
export drops in the same folder. On screen the app is explicit about this: the event
table's header renders ``t_start_s (t₀=0)`` with a tooltip spelling out "add t₀ =
1154.9909 s", and ``measure.MEASUREMENT_SCHEMA`` gives the hand-measurement export a
``t0_s`` column for exactly this reason.

The files ``analysis.scoring`` writes said nothing at all: one session on the owner's
quick apoferritin ABF produced measurements.csv carrying ``t0_s = 1154.99091``,
events.csv carrying ``t_start_s = 0.2523`` with no origin anywhere in it or in its
``.meta.json``, and figs_trace.csv putting those same samples at 1154.997 s. On a
windowed load it is unrecoverable: two windows of one recording date the same physical
event 6.000 s apart and their sidecars are key-identical.

THE RULE, and where it now lives: one seam — :func:`scoring.clock_stamp` — used by
EVERY sidecar this module writes (events, levels, training). The frozen 20-column schema
is untouched (``tests/analysis/test_baseline_seam.py`` pins that the writer adds no
column of its own); the clock rides in the sidecar, at the top level, in
MEASUREMENT_SCHEMA's own spelling. An origin the caller did not supply is written as an
explicit null with a note saying so — never as 0, which would be indistinguishable from
a recording that really starts at 0.

...AND IN THAT SCHEMA'S MEANING, not merely its spelling (sweep-3 repair, §4 below).
``MEASUREMENT_SCHEMA``'s ``t0_s`` is ``clock.instrument_origin_s`` — a segment carries
its PARENT recording's clock, and a layer alignment nudge is never included — which is
1158.9909 s where ``clock.clock_offset_s`` is 0.0 on the owner's quick apo cut. So the
writers take the document's ``meta`` and pick the quantity themselves; the caller never
picks between two functions that put two files in one folder 19 minutes apart under one
key name. A bare number and a ``provenance["t0_s"]`` still work, and a
``ClockOffset`` handed in keeps the rebase origin it carries.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the Anaconda Qt plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import scoring
from nanoscope.analysis.events import SCHEMA
from nanoscope.analysis.measure import MEASUREMENT_SCHEMA
from nanoscope.core import clock
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 40_000
T0_APO = 1154.99091246287            # the owner's quick apoferritin ABF
PROV = {"fs": FS, "polarity": -1, "units": "pA"}


def _events(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id": np.arange(n),
        "start_idx": np.arange(n) * 1000,
        "end_idx": np.arange(n) * 1000 + 100,
        "t_start_s": np.arange(n) / 10.0,
        "dwell_s": np.full(n, 1e-3),
        "accepted": np.ones(n, dtype=bool),
    })


def _sidecar(out: dict) -> dict:
    return json.loads(Path(out["meta"]).read_text())


# ================================================================================================
# 1. the seam itself
# ================================================================================================

def test_the_block_speaks_measure_pys_language():
    """One voice: the key is spelled as MEASUREMENT_SCHEMA's column, and the sentence is
    ``core.clock``'s own — the same one the event table's header tooltip shows."""
    assert scoring.CLOCK_KEY in MEASUREMENT_SCHEMA
    stamp = scoring.clock_stamp(_events(), t0_s=T0_APO)
    assert stamp["t0_s"] == pytest.approx(T0_APO)
    block = stamp[scoring.CLOCK_BLOCK_KEY]
    assert block["columns_clock"] == clock.RECORDING
    assert block["columns"] == ["t_start_s"]
    assert block["t0_s_known"] is True
    assert block["note"] == clock.field_note(clock.RECORDING, T0_APO)
    assert "1154.9909" in block["note"]


def test_an_unsupplied_origin_is_said_and_never_guessed():
    stamp = scoring.clock_stamp(_events())
    assert stamp["t0_s"] is None
    block = stamp[scoring.CLOCK_BLOCK_KEY]
    assert block["t0_s_known"] is False
    assert block["columns_clock"] == clock.RECORDING     # the clock is known even so
    assert "NOT supplied" in block["note"]
    # A zero here would be indistinguishable from a recording that starts at zero.
    assert stamp["t0_s"] != 0.0


def test_a_nonfinite_origin_reads_as_unknown():
    """MEASUREMENT_SCHEMA's convention: NaN is "unknown", never "matches"."""
    for bad in (float("nan"), float("inf"), None, "", "not a number"):
        stamp = scoring.clock_stamp(_events(), t0_s=bad)
        assert stamp["t0_s"] is None, bad
        assert stamp[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is False, bad


def test_the_origin_may_arrive_through_the_provenance_dict():
    """The channel a caller that hands over only a provenance dict can use."""
    stamp = scoring.clock_stamp(_events(), provenance={**PROV, "t0_s": T0_APO})
    assert stamp["t0_s"] == pytest.approx(T0_APO)
    # An explicit argument wins over the dict.
    stamp2 = scoring.clock_stamp(_events(), t0_s=1.5, provenance={"t0_s": T0_APO})
    assert stamp2["t0_s"] == pytest.approx(1.5)


def test_dwell_s_is_not_annotated_because_a_duration_is_on_no_clock():
    assert "dwell_s" not in scoring.RECORDING_CLOCK_COLUMNS
    block = scoring.clock_stamp(_events())[scoring.CLOCK_BLOCK_KEY]
    assert "dwell_s" not in block["columns"]


# ================================================================================================
# 2. EVERY writer in the module carries it — the audit, as a test
# ================================================================================================

def test_events_export_sidecar_carries_the_clock(tmp_path):
    out = scoring.write_events_export(_events(), tmp_path / "ev.csv", provenance=dict(PROV),
                                      source_file="pore.abf", detector="detect.schmitt",
                                      t0_s=T0_APO)
    meta = _sidecar(out)
    assert meta["t0_s"] == pytest.approx(T0_APO)
    assert meta["clock"]["columns"] == ["t_start_s"]
    assert meta["clock"]["columns_clock"] == clock.RECORDING


def test_events_export_still_adds_no_column_of_its_own(tmp_path):
    """The clock rides in the sidecar precisely because the 20 columns are frozen."""
    df = _events()
    out = scoring.write_events_export(df, tmp_path / "ev.csv", provenance=dict(PROV),
                                      t0_s=T0_APO)
    written = pd.read_csv(out["data"])
    assert list(written.columns) == list(df.columns)
    assert "t0_s" not in written.columns
    assert set(written.columns) <= set(SCHEMA) | {"accepted"}


def test_levels_export_sidecar_carries_the_clock(tmp_path):
    levels = pd.DataFrame({"event_id": [0, 0], "level_idx": [0, 1], "n_levels": [2, 2],
                           "depth_pA": [10.0, 20.0], "dwell_s": [1e-3, 2e-3],
                           "t_start_s": [0.1, 0.2]})
    out = scoring.write_levels_export(levels, tmp_path / "lv.csv", provenance=dict(PROV),
                                      t0_s=T0_APO)
    meta = _sidecar(out)
    assert meta["t0_s"] == pytest.approx(T0_APO)
    assert meta["clock"]["columns"] == ["t_start_s"]

    # The WIDE layout has no time column; the block says so and still states the origin,
    # which is what lets a wide table be joined back to the events CSV.
    wide = scoring.write_levels_export(levels, tmp_path / "wide.csv", provenance=dict(PROV),
                                       t0_s=T0_APO, wide=True)
    wmeta = _sidecar(wide)
    assert wmeta["clock"]["columns"] == []
    assert wmeta["t0_s"] == pytest.approx(T0_APO)


def test_training_export_sidecar_carries_the_clock(tmp_path):
    df = _events()
    df["added"] = False
    out = scoring.write_training_export(df, tmp_path / "train.csv", provenance=dict(PROV),
                                        t0_s=T0_APO)
    meta = _sidecar(out)
    assert meta["t0_s"] == pytest.approx(T0_APO)
    assert meta["clock"]["columns"] == ["t_start_s"]
    # ...and the training table's own column contract is untouched.
    assert list(pd.read_csv(out["data"]).columns) == scoring.TRAINING_COLUMNS


def test_every_sidecar_writer_stamps_the_clock(tmp_path):
    """The audit itself: no writer in this module may emit a sidecar without one.

    A new export added next month is meant to fail this the day it forgets.
    """
    levels = pd.DataFrame({"event_id": [0], "level_idx": [0], "n_levels": [1],
                           "depth_pA": [10.0], "dwell_s": [1e-3], "t_start_s": [0.1]})
    df = _events()
    df["added"] = False
    written = [
        scoring.write_events_export(df, tmp_path / "a.csv", provenance=dict(PROV)),
        scoring.write_levels_export(levels, tmp_path / "b.csv", provenance=dict(PROV)),
        scoring.write_training_export(df, tmp_path / "c.csv", provenance=dict(PROV)),
    ]
    for out in written:
        meta = _sidecar(out)
        assert scoring.CLOCK_KEY in meta, out["meta"].name
        assert scoring.CLOCK_BLOCK_KEY in meta, out["meta"].name
        assert meta["clock"]["columns_clock"] == clock.RECORDING, out["meta"].name


def test_the_sidecar_is_valid_json_with_an_unknown_origin(tmp_path):
    """``null``, not ``NaN``: a NaN would make the file unreadable to a strict parser."""
    out = scoring.write_events_export(_events(), tmp_path / "ev.csv", provenance=dict(PROV))
    text = Path(out["meta"]).read_text()
    assert "NaN" not in text
    assert json.loads(text)["t0_s"] is None


# ================================================================================================
# 3. THE GUI SURFACE — Analysis ▸ Export events… on a real window, real detection
# ================================================================================================

def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def detected(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, t0_s=T0_APO,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(arr,))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    w._analysis_panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=60_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert w._analysis_panel.current_detection() is not None
    yield w, doc_id
    w._analysis_controller.abort()


def test_gui_events_export_names_the_clock_and_never_lies_about_the_origin(
        detected, qtbot, tmp_path, monkeypatch):
    """The shipped events.meta.json states the clock, and whatever it says about the
    origin is TRUE of the document the rows came from.

    Deliberately not pinned to "unknown": the moment ``MainWindow._export_events`` hands
    ``t0_s`` to the writer this test gets stronger rather than failing.
    """
    w, doc_id = detected
    target = tmp_path / "events.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "CSV (*.csv)")))

    assert w._act_export_events.isEnabled()
    w._act_export_events.trigger()
    qtbot.wait(50)

    assert target.exists()
    meta = json.loads((tmp_path / "events.meta.json").read_text())

    # The clock is named, in one place a reader already looks (beside `fs` and `units`).
    assert scoring.CLOCK_KEY in meta and scoring.CLOCK_BLOCK_KEY in meta
    block = meta["clock"]
    assert block["columns_clock"] == clock.RECORDING
    assert "t_start_s" in block["columns"]
    assert block["note"]

    # The two halves of the file never disagree...
    assert block["t0_s_known"] is (meta["t0_s"] is not None)
    # ...and a stated origin is the document's real one, never a fabricated 0.
    doc_meta = w._session.document(doc_id).meta
    # `clock.instrument_origin_s`, NOT `clock_offset_s`: MEASUREMENT_SCHEMA defines this
    # key as the row's instrument-clock origin, and gui/measure_panel stamps exactly that
    # into measurements.csv. The two functions differ by 1158.99 s on a segment, and a
    # folder holding both files must not hold two meanings of one key name.
    doc_t0 = clock.instrument_origin_s(doc_meta)
    if meta["t0_s"] is not None:
        assert meta["t0_s"] == pytest.approx(doc_t0)
        assert block["note"] == clock.field_note(clock.RECORDING,
                                                 clock.clock_offset_s(doc_meta))
    else:
        assert "NOT supplied" in block["note"]

    # The CSV itself is untouched: the frozen schema gained no column.
    written = pd.read_csv(target)
    assert "t0_s" not in written.columns
    assert list(written.columns) == meta["columns"]


def test_gui_export_and_the_on_screen_header_describe_the_clock_alike(detected, qtbot,
                                                                     tmp_path, monkeypatch):
    """The annotation the screen shows must not be stripped on the way out."""
    w, _doc_id = detected
    target = tmp_path / "ev2.csv"
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "CSV (*.csv)")))
    w._act_export_events.trigger()
    qtbot.wait(50)

    meta = json.loads((tmp_path / "ev2.meta.json").read_text())
    # Whatever the file says, it is one of core.clock's own sentences about the
    # RECORDING clock — the same module the event table's tooltip is built from.
    note = meta["clock"]["note"]
    assert note.startswith("Time seconds from the start of this recording")


# ================================================================================================
# 4. ONE VOICE — ``t0_s`` means what MEASUREMENT_SCHEMA says it means (sweep-3 repair)
# ================================================================================================
#
# Spelling the key like ``measure.MEASUREMENT_SCHEMA``'s column is not yet speaking with
# one voice. That column is DEFINED as ``clock.instrument_origin_s`` — "for a segment,
# its parent recording's clock, not the segment's own rebased axis; a layer alignment
# nudge is a display shift and is never included" — and ``gui/measure_panel`` stamps
# exactly that, with a comment naming ``clock.clock_offset_s`` as the WRONG function for
# this key. The two differ on the two document classes the user really makes: 1158.9909 s
# vs 0.0 on a segment of the owner's quick apo ABF, and by the nudge under a layer
# alignment. A measurements.csv and an events.meta.json in ONE folder disagreeing by
# 1158.99 s under one key name is the defect this whole module exists to close.
#
# So the origin is taken from the DOCUMENT, here, and the caller never picks a function.

REBASED_FROM = 1158.99091246287


def _meta(t0_s: float = T0_APO, n: int = N, fs: float | None = FS) -> RecordingMeta:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    return RecordingMeta(path=Path("/tmp/pore.dat"), format_name="synthetic", n_samples=n,
                         sample_rate_hz=fs,
                         rate_provenance=(Provenance.FILE_META if fs else Provenance.UNKNOWN),
                         t0_s=float(t0_s), channels=(info,))


def _cut_meta() -> RecordingMeta:
    """A REBASED document, exactly as ``core.segment`` builds one."""
    from nanoscope.core import segment as segment_mod
    return segment_mod._rebased_meta(_meta(0.0), REBASED_FROM)


def test_the_meta_route_stamps_the_quantity_measurement_schema_defines():
    meta = _meta()
    stamp = scoring.clock_stamp(_events(), meta=meta)
    assert stamp["t0_s"] == pytest.approx(clock.instrument_origin_s(meta))
    assert stamp[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is True


def test_a_cuts_origin_is_its_parents_clock_and_never_its_manufactured_zero():
    """The document class where the two candidate functions differ by 19 minutes."""
    cut = _cut_meta()
    stamp = scoring.clock_stamp(_events(), meta=cut)

    assert stamp["t0_s"] == pytest.approx(REBASED_FROM)
    assert stamp["t0_s"] == pytest.approx(clock.instrument_origin_s(cut))
    # ...and NOT the on-screen distance, which is 0 here and would have read as "this
    # recording started at the beginning of the experiment".
    assert float(clock.clock_offset_s(cut)) == 0.0
    assert stamp["t0_s"] != 0.0

    block = stamp[scoring.CLOCK_BLOCK_KEY]
    assert block["rebased_from_s"] == pytest.approx(REBASED_FROM)
    # The sentence is the cut's own — the one the event table's header shows for this
    # same document, not the "t₀ is 0, both clocks read alike" one it used to get.
    assert block["note"] == clock.field_note(clock.RECORDING, clock.clock_offset_s(cut))
    assert "CUT of another" in block["note"]
    assert "1158.9909" in block["note"]
    # ...and specifically NOT the sentence core/clock.py's docstring names as the lie it
    # exists to stop, which is what a bare float origin used to produce here.
    assert "This recording's t\u2080 is 0" not in block["note"]


def test_a_layer_alignment_nudge_is_never_stamped_as_an_origin():
    """A nudge is a display shift the user can undo; MEASUREMENT_SCHEMA excludes it and
    so does this. Handing the meta over is what makes that unmissable."""
    meta = _meta()
    stamp = scoring.clock_stamp(_events(), meta=meta)
    assert stamp["t0_s"] == pytest.approx(meta.t0_s)
    # The number a caller would have produced with the other function under a 2.5 s nudge.
    assert stamp["t0_s"] != pytest.approx(float(clock.clock_offset_s(meta, 2.5)))


def test_a_clock_offset_handed_in_directly_keeps_its_rebase():
    """``float()`` used to strip :attr:`ClockOffset.rebased_from_s`, so a caller that
    passed ``clock_offset_s`` of a cut got 0.0 and the wrong sentence."""
    cut = _cut_meta()
    stamp = scoring.clock_stamp(_events(), t0_s=clock.clock_offset_s(cut))
    assert stamp["t0_s"] == pytest.approx(REBASED_FROM)
    block = stamp[scoring.CLOCK_BLOCK_KEY]
    assert block["rebased_from_s"] == pytest.approx(REBASED_FROM)
    assert block["note"] == clock.field_note(clock.RECORDING, clock.clock_offset_s(cut))


def test_the_document_beats_a_bare_number_and_a_provenance_dict():
    meta = _meta()
    stamp = scoring.clock_stamp(_events(), meta=meta, t0_s=1.5,
                                provenance={"t0_s": 99.0})
    assert stamp["t0_s"] == pytest.approx(T0_APO)


def test_a_document_with_no_clock_says_that_instead_of_saying_nothing():
    """Refusals name their real cause: no sample rate is not the same unknown as an
    origin the surface forgot to pass."""
    stamp = scoring.clock_stamp(_events(), meta=_meta(0.0, fs=None))
    assert stamp["t0_s"] is None
    note = stamp[scoring.CLOCK_BLOCK_KEY]["note"]
    assert note == clock.field_note(clock.RECORDING, None)
    assert "no sample rate" in note
    assert "NOT supplied" not in note


def test_every_writer_takes_the_document_and_stamps_the_parents_clock(tmp_path):
    """The audit, again, through the new seam: all three writers accept ``meta=``."""
    cut = _cut_meta()
    levels = pd.DataFrame({"event_id": [0], "level_idx": [0], "n_levels": [1],
                           "depth_pA": [10.0], "dwell_s": [1e-3], "t_start_s": [0.1]})
    df = _events()
    df["added"] = False
    written = [
        scoring.write_events_export(df, tmp_path / "a.csv", provenance=dict(PROV), meta=cut),
        scoring.write_levels_export(levels, tmp_path / "b.csv", provenance=dict(PROV), meta=cut),
        scoring.write_training_export(df, tmp_path / "c.csv", provenance=dict(PROV), meta=cut),
    ]
    for out in written:
        meta = _sidecar(out)
        assert meta["t0_s"] == pytest.approx(REBASED_FROM), out["meta"].name
        assert meta["clock"]["rebased_from_s"] == pytest.approx(REBASED_FROM), out["meta"].name
        assert "CUT of another" in meta["clock"]["note"], out["meta"].name


def test_one_folder_one_meaning_of_t0_s(qapp, qtbot, tmp_path, monkeypatch):
    """THE point of the repair, driven through real widgets on a real SEGMENT.

    A hand measurement taken on a cut writes ``t0_s`` into measurements.csv through
    ``gui/measure_panel``; an events export of the same cut writes ``t0_s`` into its
    sidecar through this module. Both files land in one folder. They must be the same
    number, because they carry the same name.
    """
    w = MainWindow()
    qtbot.addWidget(w)
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, t0_s=T0_APO,
                         channels=(info,))
    doc_id = w._session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    w._create_default_layers(w._session.document(doc_id))
    w._on_selection(doc_id, 0)
    qtbot.wait(20)

    # File ▸ Create segment from selection… — the real action; its name prompt is modal.
    monkeypatch.setattr(MainWindow, "_ask_segment_name", lambda self, default: default)
    vals = w._session.x_axis(doc_id).values
    w._analysis_panel.measure_panel.set_active(True)
    qtbot.wait(20)
    w._trace_view.set_measure_region(float(vals[4_000]), float(vals[24_000]))
    qtbot.wait(20)
    w._sync_create_segment_action()
    assert w._act_create_segment.isEnabled()
    w._act_create_segment.trigger()
    qtbot.wait(50)

    seg_ids = [d.doc_id for d in w._session.documents() if d.doc_id != doc_id]
    assert seg_ids
    seg = w._session.document(seg_ids[-1])
    assert getattr(seg.meta, clock.REBASE_ATTR, None) is not None
    w._on_selection(seg.doc_id, 0)
    qtbot.wait(50)

    # A hand measurement ON THE CUT, through the panel's own button path.
    panel = w._analysis_panel.measure_panel
    panel.set_active(True)
    qtbot.wait(20)
    seg_vals = w._session.x_axis(seg.doc_id).values
    w._trace_view.set_measure_region(float(seg_vals[100]), float(seg_vals[5_000]))
    qtbot.wait(50)
    assert panel.add_current(), panel.refusal_note()
    row_t0 = float(panel.measurements_df["t0_s"].iloc[0])

    # ...and an events export of the same cut, through this module's writer.
    out = scoring.write_events_export(_events(), tmp_path / "ev.csv",
                                      provenance=dict(PROV), meta=seg.meta)
    sidecar_t0 = _sidecar(out)["t0_s"]

    assert sidecar_t0 == pytest.approx(row_t0), (
        f"measurements.csv says t0_s={row_t0}, events.meta.json says {sidecar_t0}")
    assert row_t0 == pytest.approx(getattr(seg.meta, clock.REBASE_ATTR))
