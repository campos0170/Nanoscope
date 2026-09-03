"""Regression tests for JAB-04 and MEAS3-02: every clock surface in the window agrees
about a SEGMENT, and it agrees with the number the same window already prints for it.

THE DEFECT. Cut a segment out of the owner's quick apoferritin ABF (``meta.t0_s =
1154.9909 s``) at sample 200,000 and the window named it ``… [1159–1165 s]``, its File
Info notes said ``t₀=1158.99s`` and the Files dock tooltip said the same — while the
trace x-axis label said *"instrument clock; t₀ = 0, so both clocks read alike here"*
over an axis running 0–6 s, the event table's ``t_start_s`` header lost its badge and
its tooltip asserted *"This recording's t₀ is 0"*, and a hand measurement stamped
``t0_s = 0.0`` into the column added (MEAS2-02) precisely so a pooled table could be
read back onto one clock. Four surfaces, one number, and the three that were wrong were
the three built to prevent a clock error.

WHAT IS PINNED. Not each sentence — the RULE, at the seam: ``clock.clock_offset_s``
(which every one of those surfaces reads, through ``TraceView.clock_offset_for``) hands
on the origin ``core.segment`` rebased away, so no clock surface can answer without it.
These tests therefore check the surfaces TOGETHER: what matters is that no two of them
can disagree, not that any one of them contains a particular phrase.

Driven through a real MainWindow, its real TraceView, its real event table and its real
MeasurePanel, and through ``Session.create_segment`` — the door File ▸ Create segment
from selection goes through.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt

from nanoscope.core import clock
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 400_000
#: The owner's quick −1000 mV ABF t₀, verbatim.
T0_QUICK = 1154.9909123675027
CUT = (200_000, 300_000)                   # 4.0 s – 6.0 s into the parent
CUT_ORIGIN = T0_QUICK + CUT[0] / FS        # 1158.9909…
_BLOCKADE = (40_000, 60_000)               # inside the cut, in the SEGMENT's indices


def _doc(tmp_path, *, name: str = "apo_quick.dat", t0_s: float = T0_QUICK) -> TraceDocument:
    x = np.full(N, 100.0, dtype=np.float32)
    # one clean square inside the window the segment will cover
    x[CUT[0] + _BLOCKADE[0]:CUT[0] + _BLOCKADE[1]] = 70.0
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         t0_s=t0_s, channels=(info,))
    return TraceDocument(meta=meta, channels=(x,))


def _install(window: MainWindow, doc: TraceDocument) -> str:
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id


def _segment(window: MainWindow, parent_id: str, name: str = "clean stretch") -> str:
    seg_id = window._session.create_segment(parent_id, CUT[0], CUT[1], name=name)
    window._create_default_layers(window._session.document(seg_id))
    window._on_selection(seg_id, 0)
    return seg_id


def _measure(window: MainWindow, doc_id: str, i0: int, i1: int):
    mp = window._analysis_panel.measure_panel
    mp.set_active(True)
    vals = window._session.x_axis(doc_id).values
    window._trace_view.set_measure_region(float(vals[i0]), float(vals[i1]))
    return mp


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# ================================================================================================
# the four surfaces, together
# ================================================================================================

def test_no_clock_surface_on_a_segment_claims_its_t0_is_zero(window, tmp_path):
    """The headline. Whatever each surface says, none of them may say the reassuring
    thing — that there is nothing to convert — about data recorded 19 minutes in."""
    parent_id = _install(window, _doc(tmp_path))
    seg_id = _segment(window, parent_id)
    seg = window._session.document(seg_id)

    offset = window._trace_view.clock_offset_for(seg_id, 0)
    axis_label = window._trace_view.x_axis_label()
    note = clock.field_note(clock.RECORDING, offset)

    assert float(offset) == 0.0, "the segment's own axis really does start at 0"
    assert clock.clocks_agree(offset) is False, \
        "'no distinction is needed' is exactly what is false here"
    for text in (axis_label, note):
        assert "both clocks read alike here" not in text
        assert "This recording's t₀ is 0" not in text
    # ...and the number every OTHER surface of this window already prints is in them.
    assert "1158.9909" in axis_label and "1158.9909" in note
    assert "1158.99" in seg.meta.notes[0]
    assert seg.origin.t0_abs_s == pytest.approx(CUT_ORIGIN)


def test_the_segment_axis_label_and_its_name_tell_one_story(window, tmp_path):
    """The window NAMES the segment off the parent's instrument axis (``[1159–1165 s]``).
    Two surfaces, one fact: if the name says 1159 the axis label may not say t₀ = 0."""
    parent_id = _install(window, _doc(tmp_path))
    seg_id = _segment(window, parent_id)

    name = window._session.document(seg_id).display_name
    label = window._trace_view.x_axis_label()

    named_late = "1159" in name or "1158" in name
    label_says_zero = "t₀ = 0, so both clocks read alike" in label
    assert not (named_late and label_says_zero), (
        f"the name says {name!r} and the axis label says {label!r}")
    assert "1158.9909" in label


def test_the_event_table_header_on_a_segment_names_its_clock(window, tmp_path):
    """The badge and the tooltip are pushed from the same offset the axis is labelled
    with (analysis_panel -> _EventTableModel.set_clock_offset), so the two tables and
    the axis cannot drift apart."""
    from nanoscope.gui import panels

    parent_id = _install(window, _doc(tmp_path))
    seg_id = _segment(window, parent_id)

    model = panels._EventTableModel()
    model.set_clock_offset(window._trace_view.clock_offset_for(seg_id, 0))
    col = [i for i in range(model.columnCount())
           if str(model.headerData(i, Qt.Horizontal, Qt.DisplayRole)).startswith("t_start_s")]
    assert col, "no t_start_s column"
    header = str(model.headerData(col[0], Qt.Horizontal, Qt.DisplayRole))
    tip = str(model.headerData(col[0], Qt.Horizontal, Qt.ToolTipRole))

    assert "t₀" in header, f"the header says nothing about a clock: {header!r}"
    assert "This recording's t₀ is 0" not in tip
    assert "1158.9909" in tip


def test_switching_from_a_genuine_t0_zero_file_to_a_segment_repaints_the_header(
        window, tmp_path):
    """Both offsets are 0.0. A repaint guard comparing them as plain floats left the
    previous recording's clock note standing over the new one — the stale-label failure
    the clock module exists to close, one level up."""
    from nanoscope.gui import panels

    zero_id = _install(window, _doc(tmp_path, name="zero.dat", t0_s=0.0))
    parent_id = _install(window, _doc(tmp_path))
    seg_id = _segment(window, parent_id)

    model = panels._EventTableModel()
    model.set_clock_offset(window._trace_view.clock_offset_for(zero_id, 0))
    col = [i for i in range(model.columnCount())
           if str(model.headerData(i, Qt.Horizontal, Qt.DisplayRole)).startswith("t_start_s")][0]
    before = str(model.headerData(col, Qt.Horizontal, Qt.ToolTipRole))

    model.set_clock_offset(window._trace_view.clock_offset_for(seg_id, 0))
    after = str(model.headerData(col, Qt.Horizontal, Qt.ToolTipRole))

    assert before != after, "the header kept the previous recording's clock note"
    assert "1158.9909" in after


# ================================================================================================
# MEAS3-02: the pooled manual table
# ================================================================================================

def test_a_parent_row_and_a_segment_row_for_the_same_samples_land_on_one_clock(
        window, tmp_path):
    """THE pooled-table case, measured twice through two doors. The same physical samples
    addressed through the parent and through a segment of it used to land 1,158.99 s
    (19.3 min) apart under a column that calls itself the row's instrument-clock origin."""
    parent_id = _install(window, _doc(tmp_path))
    mp = _measure(window, parent_id, CUT[0] + _BLOCKADE[0], CUT[0] + _BLOCKADE[1])
    assert mp.add_current() is True

    seg_id = _segment(window, parent_id)
    mp = _measure(window, seg_id, _BLOCKADE[0], _BLOCKADE[1])
    assert mp.add_current() is True

    df = mp.measurements_df
    assert len(df) == 2
    instants = [float(r["t_start_s"]) + float(r["t0_s"]) for _, r in df.iterrows()]
    true_instant = T0_QUICK + (CUT[0] + _BLOCKADE[0]) / FS
    assert instants[0] == pytest.approx(true_instant, abs=1e-6)
    assert instants[1] == pytest.approx(true_instant, abs=1e-6), \
        "the segment row is on a different clock from the parent row it duplicates"


def test_the_segment_row_stamps_the_origin_the_files_dock_prints(window, tmp_path):
    """The column's tooltip promises the row's instrument-clock origin. The app already
    publishes that number for a segment in three places; the stamp is now the fourth,
    not a fabricated 0."""
    parent_id = _install(window, _doc(tmp_path))
    seg_id = _segment(window, parent_id)
    mp = _measure(window, seg_id, _BLOCKADE[0], _BLOCKADE[1])
    assert mp.add_current() is True

    row = mp.measurements_df.iloc[0]
    seg = window._session.document(seg_id)
    assert float(row["t0_s"]) == pytest.approx(seg.origin.t0_abs_s)
    assert float(row["t0_s"]) == pytest.approx(CUT_ORIGIN)


def test_the_stamp_rides_the_csv_export(window, tmp_path):
    """t0_s is the extraction surface: a pooled CSV that silently mixed a real origin and
    a fabricated 0 under one column is the version of this defect that leaves the app."""
    import pandas as pd

    parent_id = _install(window, _doc(tmp_path))
    mp = _measure(window, parent_id, CUT[0] + _BLOCKADE[0], CUT[0] + _BLOCKADE[1])
    assert mp.add_current() is True
    seg_id = _segment(window, parent_id)
    mp = _measure(window, seg_id, _BLOCKADE[0], _BLOCKADE[1])
    assert mp.add_current() is True

    out = tmp_path / "measurements.csv"
    mp.export_csv(out)
    df = pd.read_csv(out)
    assert sorted(round(float(v), 4) for v in df["t0_s"]) == [
        round(T0_QUICK, 4), round(CUT_ORIGIN, 4)]


def test_a_segment_of_a_segment_is_stamped_at_its_own_depth(window, tmp_path):
    """The next boundary the same rule reaches: a nested cut used to be stamped off its
    parent's manufactured 0, so a cut of a cut of the 1154.99 s recording claimed its
    data was recorded seconds after the epoch."""
    parent_id = _install(window, _doc(tmp_path))
    seg_id = _segment(window, parent_id)
    inner_id = window._session.create_segment(seg_id, 20_000, 80_000, name="deeper")
    window._create_default_layers(window._session.document(inner_id))
    window._on_selection(inner_id, 0)

    expected = CUT_ORIGIN + 20_000 / FS
    inner = window._session.document(inner_id)
    assert inner.origin.t0_abs_s == pytest.approx(expected)
    assert clock.instrument_origin_s(inner.meta) == pytest.approx(expected)
    assert f"{expected:.4f}"[:9] in window._trace_view.x_axis_label()

    mp = _measure(window, inner_id, 10_000, 30_000)
    assert mp.add_current() is True
    assert float(mp.measurements_df.iloc[0]["t0_s"]) == pytest.approx(expected)
