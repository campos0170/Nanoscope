"""GUI tests for signal segmentation, wired through a real MainWindow (pytest-qt, offscreen).

A segment is a DERIVED RECORDING: cropping the on-trace Measure selection produces a new
document in the Files list, analysed with the exact same per-recording machinery as any
file — its own results, independent of the parent.

These tests assert, on a live MainWindow + a synthetic recording:

* selecting a region and invoking *Create segment* adds a new document to the Files list,
  with the right crop, and selects it (panels switch to it);
* the Create-segment affordance (menu action + Files-dock button, gated together) is OFF
  with no active recording and with no on-trace region, and ON once a region is drawn;
* creating a segment marks the session dirty (for free, via documentOpened);
* running real detection on the segment stores its result under the SEGMENT, distinct from
  and independent of the parent's;
* a project with a parent + a segment (each detected) round-trips through the app: the
  segment is re-derived and its result re-attaches, still distinct from the parent's;
* the Files dock imports no sibling gui panel — MainWindow stays the sole wirer.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import ast
import os
import pathlib

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.identity import source_of
from nanoscope.core.segment import SegmentDocument, persistent_key
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
# Five events; the segment window [20k, 60k) brackets exactly two of them.
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]
SEG_I0, SEG_I1 = 20_000, 60_000


# ================================================================================================
# builders
# ================================================================================================

def _event_trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path, arr: np.ndarray) -> None:
    """A time,current CSV the TextTraceReader reopens as one CURRENT channel at FS."""
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(path, index=False)


def _doc(path, arr: np.ndarray) -> TraceDocument:
    """An in-memory doc whose meta.path IS the tmp CSV, so its doc_id == the reopened one."""
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def win(qapp, qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    # Never face a live name-prompt modal: default to a fixed name (tests override per-case).
    w._ask_segment_name = lambda default: "SEG-A"
    return w


def _open_parent(win: MainWindow, tmp_path, *, seed: int = 0, name: str = "pore.csv"):
    """Open a real-file-backed parent recording, layered + selected (so a restore can reopen it)."""
    csv = tmp_path / name
    arr = _event_trace(seed=seed)
    _write_csv(csv, arr)
    doc = _doc(csv, arr)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    return doc_id, arr


def _select_region(win: MainWindow, doc_id: str, i0: int, i1: int) -> None:
    """Turn on the Measure tool and drag its region to [i0, i1) — the SAME LinearRegionItem the
    Create-segment affordance reads."""
    win._analysis_panel.measure_panel.set_active(True)
    vals = win._session.x_axis(doc_id).values
    win._trace_view.set_measure_region(float(vals[i0]), float(vals[i1]))


def _detect_active(win: MainWindow, qtbot, min_dwell_s: float = 5e-4) -> None:
    """Run real Schmitt detection on the panel's current selection (through the worker)."""
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(min_dwell_s)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)


def _the_segment(win: MainWindow) -> SegmentDocument:
    segs = win._session.segment_documents()
    assert len(segs) == 1
    return segs[0]


def _find_doc_item(dock, doc_id: str):
    """The FileDock tree item for a document row (payload ("doc", doc_id)), or None."""
    from PySide6 import QtCore, QtWidgets

    it = QtWidgets.QTreeWidgetItemIterator(dock._tree)
    while it.value():
        item = it.value()
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload and payload[0] == "doc" and payload[1] == doc_id:
            return item
        it += 1
    return None


# ================================================================================================
# create adds a document + selects it
# ================================================================================================

def test_create_segment_adds_document_to_files_list_and_selects_it(win, tmp_path):
    parent_id, _ = _open_parent(win, tmp_path)
    _select_region(win, parent_id, SEG_I0, SEG_I1)
    assert win._can_create_segment()

    n_before = len(win._session.documents())
    win._ask_segment_name = lambda default: "my segment"
    win._create_segment_from_selection()

    assert len(win._session.documents()) == n_before + 1
    seg = _the_segment(win)
    assert (seg.origin.i0, seg.origin.i1) == (SEG_I0, SEG_I1)
    assert seg.meta.n_samples == SEG_I1 - SEG_I0
    assert seg.display_name == "my segment"

    # It appears in the Files dock as a selectable "doc" row…
    dock = win._file_dock
    it = _find_doc_item(dock, seg.doc_id)
    assert it is not None
    assert "✂" in it.text(0)                       # …distinguished with a segment marker
    assert "segment of" in it.toolTip(0)           # …and a tooltip tracing it to the parent

    # …and it is now the active selection (panels switched to it).
    assert win._current_selection[0] == seg.doc_id
    assert win._analysis_panel._sel[0] == seg.doc_id


def test_prefilled_name_encodes_parent_and_range(win, tmp_path):
    parent_id, _ = _open_parent(win, tmp_path)
    _select_region(win, parent_id, SEG_I0, SEG_I1)
    captured: list[str] = []
    win._ask_segment_name = lambda default: captured.append(default) or default
    win._create_segment_from_selection()
    assert captured, "the name prompt was never shown"
    default = captured[0]
    assert "pore.csv" in default
    assert "s]" in default                          # a seconds window, e.g. "pore.csv [0.2–0.6 s]"


def test_cancelling_the_name_prompt_creates_nothing(win, tmp_path):
    parent_id, _ = _open_parent(win, tmp_path)
    _select_region(win, parent_id, SEG_I0, SEG_I1)
    n_before = len(win._session.documents())
    win._ask_segment_name = lambda default: None   # user hit Cancel
    win._create_segment_from_selection()
    assert len(win._session.documents()) == n_before
    assert win._session.segment_documents() == ()


# ================================================================================================
# gating
# ================================================================================================

def test_gated_off_with_no_active_recording(win):
    # Empty session: nothing selected, nothing to segment.
    assert not win._can_create_segment()
    assert not win._act_create_segment.isEnabled()
    assert not win._file_dock._btn_segment.isEnabled()


def test_gated_off_with_a_recording_but_no_region(win, tmp_path):
    _open_parent(win, tmp_path)                     # selected, but Measure is off -> no region
    assert win._trace_view.measure_bounds() is None
    assert not win._can_create_segment()
    assert not win._act_create_segment.isEnabled()
    assert not win._file_dock._btn_segment.isEnabled()


def test_gate_turns_on_once_a_region_is_drawn_then_off_when_measure_stops(win, tmp_path):
    parent_id, _ = _open_parent(win, tmp_path)
    _select_region(win, parent_id, SEG_I0, SEG_I1)
    assert win._can_create_segment()
    assert win._act_create_segment.isEnabled()
    assert win._file_dock._btn_segment.isEnabled()

    win._analysis_panel.measure_panel.set_active(False)   # Measure off -> selection gone
    assert win._trace_view.measure_bounds() is None
    assert not win._can_create_segment()
    assert not win._act_create_segment.isEnabled()


def test_handler_is_a_noop_when_ungated(win, tmp_path):
    _open_parent(win, tmp_path)                     # no region drawn
    win._create_segment_from_selection()            # must not raise, must create nothing
    assert win._session.segment_documents() == ()


def test_files_dock_button_emits_and_creates_via_mainwindow(win, tmp_path):
    """The Files-dock 'Segment' button routes through MainWindow (the sole wirer)."""
    parent_id, _ = _open_parent(win, tmp_path)
    _select_region(win, parent_id, SEG_I0, SEG_I1)
    win._ask_segment_name = lambda default: "via button"
    win._file_dock._btn_segment.click()
    seg = _the_segment(win)
    assert seg.display_name == "via button"


# ================================================================================================
# dirty tracking
# ================================================================================================

def test_creating_a_segment_marks_the_session_dirty(win, tmp_path):
    """A hand-cut segment is work: nothing on disk holds it, and only a project can.

    It is marked dirty EXPLICITLY, not via ``documentOpened`` — opening a recording is not
    unsaved work (B3) and no longer sets the flag, so riding on it would have quietly
    disarmed the quit guard for every segment the user cuts."""
    parent_id, _ = _open_parent(win, tmp_path)
    _select_region(win, parent_id, SEG_I0, SEG_I1)
    win._mark_clean()
    assert win._dirty is False
    win._create_segment_from_selection()
    assert win._dirty is True


# ================================================================================================
# independent analysis: detection under the segment, not the parent
# ================================================================================================

def test_detection_on_segment_is_independent_of_parent(win, tmp_path, qtbot):
    parent_id, _ = _open_parent(win, tmp_path)

    # Detect on the PARENT (five events).
    win._on_selection(parent_id, 0)
    _detect_active(win, qtbot)
    parent_res = win._session.results.get(parent_id, 0, "detect.schmitt")
    assert parent_res is not None
    n_parent = len(parent_res.payload.df)
    assert n_parent == len(_EVENTS)

    # Cut a segment bracketing two events and detect on IT.
    _select_region(win, parent_id, SEG_I0, SEG_I1)
    win._ask_segment_name = lambda default: "seg"
    win._create_segment_from_selection()
    seg = _the_segment(win)
    assert win._analysis_panel._sel == (seg.doc_id, 0)
    _detect_active(win, qtbot)

    seg_res = win._session.results.get(seg.doc_id, 0, "detect.schmitt")
    assert seg_res is not None
    assert seg_res.doc_id == seg.doc_id            # keyed under the SEGMENT
    assert len(seg_res.payload.df) == 2            # only the two bracketed events

    # The two results are independent objects with distinct tables; the parent's is untouched.
    assert seg_res is not parent_res
    assert len(seg_res.payload.df) != n_parent
    assert win._session.results.get(parent_id, 0, "detect.schmitt") is parent_res
    assert len(parent_res.payload.df) == n_parent


# ================================================================================================
# full-app project round-trip: re-derive the segment + re-attach its result
# ================================================================================================

def test_project_roundtrip_rederives_segment_and_reattaches_distinct_result(win, tmp_path, qtbot):
    parent_id, _ = _open_parent(win, tmp_path)
    win._on_selection(parent_id, 0)
    _detect_active(win, qtbot)
    df_parent = win._session.results.get(parent_id, 0, "detect.schmitt").payload.df.copy()

    _select_region(win, parent_id, SEG_I0, SEG_I1)
    win._ask_segment_name = lambda default: "pore_seg"
    win._create_segment_from_selection()
    seg = _the_segment(win)
    seg_uid = seg.origin.segment_uid
    _detect_active(win, qtbot)
    df_segment = win._session.results.get(seg.doc_id, 0, "detect.schmitt").payload.df.copy()
    assert len(df_segment) != len(df_parent)

    proj = tmp_path / "seg_session.nanoscope"
    win._write_project(str(proj))
    win._analysis_controller.abort()

    # Reopen in a FRESH window.
    w2 = MainWindow()
    qtbot.addWidget(w2)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    sess2 = w2._session

    # -- the parent reopened, its result intact ----------------------------------------------
    # By SOURCE, not by the source window's doc_id: reopening a file is a new LOAD and gets a
    # new id (core/identity.py). What has to survive is the persistent key, which for a file
    # is its path — and that is what the project stored.
    parent2 = sess2.doc_id_for_source(source_of(parent_id))
    assert parent2 is not None and parent2 != parent_id
    rp = sess2.results.get(parent2, 0, "detect.schmitt")
    assert rp is not None and len(rp.payload.df) == len(df_parent)

    # -- the segment was RE-DERIVED (same uid/window/name) ------------------------------------
    segs2 = sess2.segment_documents()
    assert len(segs2) == 1
    seg2 = segs2[0]
    assert seg2.origin.segment_uid == seg_uid
    assert (seg2.origin.i0, seg2.origin.i1) == (SEG_I0, SEG_I1)
    assert seg2.display_name == "pore_seg"
    assert seg2.meta.n_samples == SEG_I1 - SEG_I0
    # Stable PERSISTENT identity across the round-trip — the uid, which is what the results
    # were keyed under. The doc_id is deliberately fresh: a re-derivation is a new load.
    assert persistent_key(seg2) == persistent_key(seg) == seg_uid
    assert seg2.source_key == seg.source_key
    assert seg2.doc_id != seg.doc_id

    # -- the segment's result re-attached under the segment, distinct from the parent's -------
    rs = sess2.results.get(seg2.doc_id, 0, "detect.schmitt")
    assert rs is not None
    assert len(rs.payload.df) == len(df_segment)
    assert len(rs.payload.df) != len(rp.payload.df)
    pd.testing.assert_frame_equal(rs.payload.df, df_segment)

    w2._analysis_controller.abort()
    w2._controller.abort()


# ================================================================================================
# the no-cross-panel-import invariant (MainWindow is the sole wirer)
# ================================================================================================

def _relative_sibling_imports(module) -> list[str]:
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    return [n.module for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module]


def test_file_dock_imports_no_sibling_gui_panel():
    """FileDock gained a segment button + signal but stays a pure dock: it imports only core
    (level-2 relative) — never another gui panel. MainWindow wires createSegmentRequested to
    Session.create_segment."""
    import nanoscope.gui.file_dock as fd

    assert _relative_sibling_imports(fd) == []
