"""HKE-02 — batch curation and the undo/redo table re-feed must not freeze the GUI at 100k.

Measured through the real app at 100,000 events: "Reject all" 2.5 s, "Accept all" 4.0 s,
Edit->Undo 7.8 s and Edit->Redo 7.9 s of synchronous GUI-thread stall per gesture (a 50 ms
repaint timer starved throughout), and undoing even a SINGLE one-row gesture cost the same
seconds — the cost was never the gesture, it was per-row bookkeeping:

* ``EventGalleryWindow._set_rows_accepted`` decided and wrote row by row — five pandas
  scalar (``iat``) round-trips per row, ~2-4 s per bulk gesture at 100k. It now decides
  the same ``derived AND requested`` per row but VECTORISED, with ONE frame write for the
  whole gesture.
* every undo/redo re-feeds the table, and ``EventTable`` re-measured its column widths
  with ``QTableView.resizeColumnsToContents`` — a delegate size-hint per cell, ~3.6 s per
  call, called TWICE per undo (the re-feed, then an unchanged stale-badge pass). It now
  measures a bounded row sample (``_autosize_columns``), and the stale-badge pass
  re-measures only when the badge set actually changed.

These tests pin the timings at the dossier's own 100k scale with GENEROUS bounds
(~10-40x the measured post-fix cost, best-of-N where repeatable) and — first — that the
batched decision is bit-identical to the per-row rule it replaced.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_gallery import EventGalleryWindow, _derived_accepted
from nanoscope.gui.panels import EventTable

FS = 100_000.0
N = 100_000


# ================================================================================================
# builders
# ================================================================================================
def _session_with_doc(tmp_path) -> tuple[Session, str, int]:
    arr = np.full(200_000, 100.0, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "hke02.dat", format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    return session, doc_id, 0


def _big_frame(n: int = N, seed: int = 5) -> pd.DataFrame:
    """100k events; a sprinkling of rejecting flags so 'Accept all' has refusals to
    disclose, plus a curator token and a caveat token that must NOT block accepts."""
    rng = np.random.default_rng(seed)
    flag = np.array([""] * n, dtype=object)
    flag[rng.choice(n, size=700, replace=False)] = "clipped_by_boundary"
    flag[rng.choice(n, size=300, replace=False)] = "flagged"        # curator's own: no block
    starts = np.arange(n, dtype=np.int64) % 150_000
    return pd.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "start_idx": starts,
        "end_idx": starts + 40,
        "t_start_s": starts / FS,
        "dwell_s": rng.uniform(1e-4, 1e-2, n),
        "depth_pA": rng.uniform(5.0, 200.0, n),
        "accepted": np.ones(n, dtype=bool),
        "flag": flag,
    })


@pytest.fixture
def gallery(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    df = _big_frame()
    # The frame's own truth: rejecting flags start rejected (the detector's half).
    df["accepted"] = np.array([_derived_accepted(f) for f in df["flag"]], dtype=bool)
    gal = EventGalleryWindow(session)
    qtbot.addWidget(gal)
    gal.set_events(doc_id, ch, df)
    return gal, df


# ================================================================================================
# the decision is unchanged — only the fan-out was batched
# ================================================================================================
def test_batched_accept_all_lands_exactly_what_the_per_row_rule_lands(gallery):
    """``derived AND requested``, row by row, compared against the rule's own function
    over every one of the 100k rows: rejecting flags stay rejected, the curator token
    and the empty flag accept. Speed bought with a different decision is a defect."""
    gal, df = gallery
    gal.reject_all()
    assert not df["accepted"].to_numpy().any(), "reject_all must land on every row"

    gal.accept_all()
    expected = np.array([_derived_accepted(f) for f in df["flag"]], dtype=bool)
    np.testing.assert_array_equal(df["accepted"].to_numpy().astype(bool), expected)
    assert not expected.all(), "the fixture must carry rows the accept is refused on"


def test_accept_all_still_discloses_its_refusals_on_the_note(gallery):
    """The refusal ledger is fed from the vectorised buckets now; the note must still
    count the rows 'Accept all' kept rejected — the M6d disclosure, at 100k."""
    gal, df = gallery
    gal.reject_all()
    gal.accept_all()
    n_refused = int((~np.array([_derived_accepted(f) for f in df["flag"]])).sum())
    note = gal.refusal_note()
    assert str(n_refused) in note, (
        f"the note must count the {n_refused} kept-rejected rows; note was {note!r}")


def test_a_single_tile_gesture_still_announces_the_written_value(gallery):
    """One changed row still emits per-event ``acceptanceChanged`` (the owner mirrors it
    onto the table checkbox); the batch signal stays for real batches."""
    gal, df = gallery
    row = int(np.nonzero((df["flag"] == "").to_numpy())[0][0])
    assert bool(df["accepted"].iloc[row])
    sm = gal.view.selectionModel()
    idx = gal._model.index_for_df_row(row)
    assert idx.isValid(), "the fixture row must sit on the first page"
    sm.select(QItemSelection(idx, idx), QItemSelectionModel.ClearAndSelect)
    heard: list[tuple[int, bool]] = []
    gal.acceptanceChanged.connect(lambda ev, acc: heard.append((ev, acc)))
    gal.set_accepted_selected(False)
    assert heard == [(int(df["event_id"].iloc[row]), False)]
    assert not bool(df["accepted"].iloc[row])


def test_a_bulk_gesture_is_one_batch_signal_not_a_storm(gallery):
    """One gesture, one ``curationBatchChanged`` — the per-row signal storm is the
    freeze this defect measured downstream of."""
    gal, df = gallery
    batch: list[int] = []
    per_row: list[int] = []
    gal.curationBatchChanged.connect(lambda: batch.append(1))
    gal.acceptanceChanged.connect(lambda *a: per_row.append(1))
    gal.reject_all()
    assert len(batch) == 1 and not per_row
    gal.accept_all()
    assert len(batch) == 2 and not per_row


# ================================================================================================
# the timings, at the dossier's own scale
# ================================================================================================
def test_bulk_curation_at_100k_stays_under_the_stall_bar(gallery):
    """Reject-all / accept-all measured 2.5-4.0 s of stall at 100k; post-fix ~0.2 s.
    Bound: every gesture < 2.0 s AND the best < 1.0 s — generous on a loaded machine,
    unreachable for the per-row-iat regression."""
    gal, _df = gallery
    timings = []
    for gesture in (gal.reject_all, gal.accept_all, gal.reject_all, gal.accept_all):
        t0 = time.perf_counter()
        gesture()
        timings.append(time.perf_counter() - t0)
    assert max(timings) < 2.0, f"bulk gestures took {timings} (were 2.5-4.0 s each)"
    assert min(timings) < 1.0, f"even the best gesture is slow: {timings}"


def test_the_table_refeed_at_100k_stays_under_the_stall_bar(qapp, qtbot):
    """``EventTable.set_events`` is the dominant cost inside every undo/redo re-install
    (measured 7.2 of 7.8 s: two full-table width passes). Post-fix it is a bounded
    sample. Bound: best of three < 2.0 s (post-fix measures ~0.2 s)."""
    table = EventTable()
    qtbot.addWidget(table)
    df = _big_frame()
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        table.set_events(df)
        timings.append(time.perf_counter() - t0)
    assert min(timings) < 2.0, f"set_events at {N:,} rows took {timings} (was ~3.6 s)"

    # ...and the stale-badge pass that runs after EVERY re-feed must be ~free when the
    # badge set did not change (it was the second 3.6 s width pass inside every undo).
    t0 = time.perf_counter()
    for _ in range(5):
        table.set_stale_columns((), "")
    dt = time.perf_counter() - t0
    assert dt < 0.5, f"5 unchanged stale-badge passes took {dt:.2f}s — the guard is gone"


def test_autosized_columns_still_fit_their_contents(qapp, qtbot):
    """The bounded sample must still MEASURE: a column whose widest value sits at the
    end of a 100k-row frame (past the 1000 rows Qt's own pass read) gets a width that
    fits it, and the badge pass widens the header it changed."""
    from PySide6.QtGui import QFontMetrics

    table = EventTable()
    qtbot.addWidget(table)
    df = _big_frame(n=2_000)
    df.loc[len(df) - 1, "flag"] = "clipped_by_boundary;dwell>max_dwell_s"   # the tail row
    table.set_events(df)
    fm = QFontMetrics(table.view.font())
    col = df.columns.get_loc("flag")
    want = fm.horizontalAdvance("clipped_by_boundary;dwell>max_dwell_s")
    assert table.view.columnWidth(col) >= want, (
        "the widest value lives in the tail and the sample must have measured it")

    before = table.view.columnWidth(df.columns.get_loc("depth_pA"))
    table.set_stale_columns(("depth_pA",), "superseded estimator")
    after = table.view.columnWidth(df.columns.get_loc("depth_pA"))
    assert after >= before, "the ⚠ badge widened the header; the width must follow"
