"""S7-POP5 / POP6 / POP7 / POP1 (gallery half) — what the gallery's captions and status line
say, driven through the real window.

POP5  the depth caption carries the channel's unit, like the dwell caption beside it.
POP6  the status line names only the digits the palette ARMS, and an unarmed digit says so.
POP7  an empty subset reads "0 of 0", never "0–-1 of 0".
POP1  ``set_stale_columns`` — the seam MainWindow pushes the events table's own badge set
      through — marks the sorted caption and the status line, and clears with ``()``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.analysis.labels import LabelVocabulary, parse_labels
from nanoscope.analysis.remeasure import REMEASURED_COLUMNS
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
N_EVENTS = 30
_SPANS = [(4_000 + i * 5_000, 4_000 + i * 5_000 + 300 + (i * 137) % 900)
          for i in range(N_EVENTS)]
N = _SPANS[-1][1] + 4_000


def _signal() -> np.ndarray:
    rng = np.random.default_rng(5)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def wired(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pop5.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    doc = TraceDocument(meta=meta, channels=(_signal(),))
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) == N_EVENTS, len(df)
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov),
        # Stamped with the live estimator, so the table -- and, now that MainWindow pushes
        # the table's badge set onto this window on every feed (S7-POP1 wiring), the
        # gallery -- start CLEAN. An unstamped result predates the estimator and is
        # judged stale on open, which is the owner's-project case, not this fixture's.
        baseline_fingerprint=w._analysis_panel.baseline_panel.config().fingerprint(FS)))
    w._act_event_gallery.trigger()
    qtbot.wait(20)
    gal = w._event_gallery
    gal.page_size_box.setCurrentText("200")
    gal.set_filter("All")
    gal.show()
    gal.activateWindow()
    qtbot.wait(40)
    yield w, doc_id, gal
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _click_tile(gal, df_row: int) -> None:
    idx = gal._model.index_for_df_row(df_row)
    assert idx.isValid(), f"row {df_row} is not on this page"
    QTest.mouseClick(gal.view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     gal.view.visualRect(idx).center())


def _captions(gal, df, n: int = 4) -> list[str]:
    return [gal.caption_for(df.iloc[r]) for r in gal._model.page_df_rows()[:n]]


# ================================================================================================
# POP5
# ================================================================================================
def test_the_depth_caption_carries_the_channels_unit_like_the_dwell_caption_does(wired, qtbot):
    """'86 · 139' beside '24 · 33.88ms' was the one caption of three without its unit, on a
    window that had already resolved it."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    assert gal._resolve_units() == "pA"
    gal.set_sort("Dwell (longest first)")
    qtbot.wait(10)
    assert all(c.endswith(("ms", "µs")) for c in _captions(gal, df)), _captions(gal, df)
    gal.set_sort("Depth (deepest first)")
    qtbot.wait(10)
    caps = _captions(gal, df)
    assert all(c.endswith("pA") for c in caps), caps
    depth = float(df.iloc[gal._model.page_df_rows()[0]]["depth_pA"])
    assert caps[0] == f"{int(df.iloc[gal._model.page_df_rows()[0]]['event_id'])} · {depth:.3g}pA"


# ================================================================================================
# POP7
# ================================================================================================
def test_an_empty_subset_reads_0_of_0_and_never_a_negative_range(wired, qtbot):
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    df.iloc[:, df.columns.get_loc("accepted")] = False
    gal.refresh()
    gal.set_filter("Accepted only")
    qtbot.wait(20)
    assert gal.event_count() == 0
    first = gal.status.text().split("   ·   ")[0]
    assert first == "0 of 0 (accepted only)", first
    assert "–-1" not in gal.status.text()


# ================================================================================================
# POP6
# ================================================================================================
def test_the_status_line_names_only_the_digits_the_palette_arms(wired, qtbot):
    """Empty palette: no digit is advertised and the line says where to define labels.
    Two entries: '1–2 to label'. Nine: '1–9 to label'."""
    w, doc_id, gal = wired
    assert gal.vocabulary().entries == []
    status = gal.status.text()
    assert "1–9 to label" not in status, status
    assert "click to select, R to reject" in status, status
    assert "Analysis ▸ Event labels…" in status, status
    assert gal.armed_digits() == ()

    vocab = LabelVocabulary()
    vocab.add("molecule", "apo")
    vocab.add("molecule", "holo")
    gal.set_vocabulary(vocab)
    qtbot.wait(10)
    assert gal.armed_digits() == ("1", "2")
    status = gal.status.text()
    assert "click to select, R to reject, 1–2 to label" in status, status
    assert "1–9" not in status, status

    for i in range(3, 10):
        vocab.add("molecule", f"kind{i}")
    gal.set_vocabulary(vocab)
    qtbot.wait(10)
    assert "click to select, R to reject, 1–9 to label" in gal.status.text(), gal.status.text()


def test_an_unarmed_digit_says_so_on_the_status_line_instead_of_doing_nothing(wired, qtbot):
    """A real key press on the view. With no palette, 1 writes nothing — and says so; with a
    two-entry palette, 5 says the palette arms 1–2 only; 1 with nothing selected says so;
    1 over a selected tile writes the label and takes the note away."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    flags_before = df["flag"].astype(str).tolist()
    _click_tile(gal, 0)
    qtbot.wait(10)
    QTest.keyClick(gal.view, Qt.Key_1)
    qtbot.wait(20)
    assert df["flag"].astype(str).tolist() == flags_before
    status = gal.status.text()
    assert "1 is not armed — no labels defined" in status, status
    assert "Analysis ▸ Event labels…" in status, status

    vocab = LabelVocabulary()
    vocab.add("molecule", "apo")
    vocab.add("molecule", "holo")
    gal.set_vocabulary(vocab)
    qtbot.wait(10)
    assert "is not armed" not in gal.status.text(), "a new palette retires the old note"
    QTest.keyClick(gal.view, Qt.Key_5)
    qtbot.wait(20)
    assert df["flag"].astype(str).tolist() == flags_before
    assert "5 is not armed — the palette arms 1–2 only" in gal.status.text(), gal.status.text()

    gal.view.clearSelection()
    QTest.keyClick(gal.view, Qt.Key_1)
    qtbot.wait(20)
    assert df["flag"].astype(str).tolist() == flags_before
    assert "1 labels the SELECTED tiles and none is selected" in gal.status.text(), (
        gal.status.text())

    _click_tile(gal, 0)
    qtbot.wait(10)
    QTest.keyClick(gal.view, Qt.Key_1)
    qtbot.wait(20)
    assert parse_labels(df["flag"].iloc[0]).get("molecule") == "apo"
    assert "is not armed" not in gal.status.text()
    assert "none is selected" not in gal.status.text()


# ================================================================================================
# POP1 — the gallery half of the seam
# ================================================================================================
_NOTE = "Depth, frac_depth, ECD, mean/max deviation and SNR are stale."


def test_set_stale_columns_marks_the_sorted_caption_and_the_status_line_and_clears(wired, qtbot):
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    gal.set_sort("Depth (deepest first)")
    qtbot.wait(10)
    assert not any(c.endswith("⚠") for c in _captions(gal, df))
    assert "⚠" not in gal.status.text()

    gal.set_stale_columns(REMEASURED_COLUMNS, _NOTE)
    qtbot.wait(10)
    assert gal.stale_columns() == tuple(REMEASURED_COLUMNS)
    caps = _captions(gal, df)
    assert all(c.endswith("pA ⚠") for c in caps), caps
    assert f"⚠ sorted and captioned by stale depth_pA: {_NOTE}" in gal.status.text(), (
        gal.status.text())

    # dwell_s is not in the tuple: the dwell caption stays clean, the sentence stays.
    gal.set_sort("Dwell (longest first)")
    qtbot.wait(10)
    assert not any(c.endswith("⚠") for c in _captions(gal, df)), _captions(gal, df)
    assert f"⚠ {_NOTE}" in gal.status.text(), gal.status.text()
    assert "sorted and captioned" not in gal.status.text()

    gal.set_sort("dI/I0 (deepest first)")
    qtbot.wait(10)
    assert all(c.endswith(" ⚠") for c in _captions(gal, df)), _captions(gal, df)

    gal.set_stale_columns(())
    qtbot.wait(10)
    assert not any(c.endswith("⚠") for c in _captions(gal, df))
    assert "⚠" not in gal.status.text(), gal.status.text()


def test_the_tuple_the_events_table_badges_is_the_tuple_the_seam_takes(wired, qtbot):
    """The finding's gesture: one knob in the Preprocess tab. The events table derives its
    badge set (``REMEASURED_COLUMNS``) and its banner; pushing exactly those through the
    seam is what the MainWindow wiring does, and the gallery then says the banner's
    sentence over a badged caption."""
    w, doc_id, gal = wired
    df = _frame(w, doc_id)
    ap = w._analysis_panel
    res = ap.current_detection()
    assert res is not None
    before = float(ap._baseline.rough_s.value())
    ap._baseline.rough_s.setValue(before * 4.0)
    qtbot.wait(30)
    ap._check_staleness(res)
    qtbot.wait(20)
    cols = tuple(ap._table.stale_columns)
    assert cols == tuple(REMEASURED_COLUMNS), cols
    note = ap._stale.message()
    assert "stale" in note or "re-measured" in note, note

    gal.set_sort("Depth (deepest first)")
    gal.set_stale_columns(cols, note)
    qtbot.wait(10)
    assert all(c.endswith("⚠") for c in _captions(gal, df))
    assert note in gal.status.text(), gal.status.text()
