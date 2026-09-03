"""EXPL3-06, on the shipped surface: detect → curate → ``Analysis ▸ Event
subpopulations`` → the exported file.

The analysis-layer invariant is pinned in
``tests/analysis/test_s3b_expl3_subpop_curation_naming.py``. What is at stake HERE is that
the fact reaches the user through the real path, with nothing hand-built in the middle:

  * the detection is run by the real detector through ``AnalysisPanel.trigger_detection``,
    so the provenance carrying ``n_events`` is the DETECTOR'S own stamp and not a dict a
    test wrote;
  * the curation is done by writing the ``accepted`` checkbox in the real events table
    model — the same ``setData`` a click lands on;
  * the run is started by triggering the real ``QAction``;
  * the assertions are made on the app's own status line, on the report window's header,
    and on the bytes ``write_report`` puts on disk through the real export path.

The defect was that these four surfaces disagreed: the status bar said "120 accepted
event(s)", the file said "detected regions : 120" over a detection of 300, and a reader of
the file alone could not tell that 60 % of the events had been removed before the pipeline
started.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re

import numpy as np
import pytest
from PySide6.QtCore import Qt

from nanoscope.analysis import subpop_report as SR
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N_SAMPLES = 300_000
N_EVENTS = 150
N_CURATED_OUT = 55


class _Cheap(SR.SubpopConfig):
    """The same pipeline over a small grid — a GUI test cannot afford the 5-fold, K<=8,
    200-resample default, and the full grid's numbers are pinned in the analysis tests."""

    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        kw.setdefault("with_tsne", False)
        kw.setdefault("with_umap", False)
        super().__init__(**kw)


def _trace(seed: int = 3):
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 2.0, N_SAMPLES)
    pos = 500
    for i in range(N_EVENTS):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(30, 45))
        if pos + dwell + 400 >= N_SAMPLES:
            break
        x[pos:pos + dwell] -= 60.0 if deep else 30.0
        pos += int(rng.integers(500, 1100))
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "curated.dat", format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, qtbot, tmp_path, monkeypatch):
    """A real detection over a real (synthetic) recording, through the app."""
    monkeypatch.setattr(SR, "SubpopConfig", _Cheap)
    doc = _doc(tmp_path)
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(4e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(30)
    res = window._session.results.get(doc_id, 0, "detect.schmitt")
    assert res is not None and len(res.payload.df) > 60, "the detector found too little"
    return window, doc_id, res.payload


def _curate_out(window, qtbot, n: int) -> int:
    """Uncheck ``accepted`` on ``n`` rows through the real table model — the same write a
    checkbox click performs. Returns how many rows are still accepted."""
    # Through the VIEW's model — the sort proxy the click really goes through, not the
    # source frame underneath it.
    model = window._analysis_panel._table.view.model()
    header = [model.headerData(c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
              for c in range(model.columnCount())]
    col = header.index("accepted")
    done = 0
    row = 0
    while done < n and row < model.rowCount():
        idx = model.index(row, col)
        if model.setData(idx, Qt.CheckState.Unchecked.value, Qt.ItemDataRole.CheckStateRole):
            done += 1
        row += 1
    qtbot.wait(10)
    assert done == n, f"only {done} of {n} rows could be curated out"
    df = window._analysis_panel.current_detection().payload.df
    return int(df["accepted"].to_numpy(dtype=bool).sum())


def _run(window, qtbot, timeout_ms: int = 180_000):
    """Trigger the real menu action and collect every status-bar line the run wrote."""
    said: list[str] = []
    window.statusBar().messageChanged.connect(said.append)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=timeout_ms):
        window._act_subpop.trigger()
    qtbot.wait(30)
    said.append(window.statusBar().currentMessage())
    return window._subpop_window, [s for s in said if s]


@pytest.fixture
def curated_run(detected, qtbot):
    window, _doc_id, events = detected
    n_detected = len(events.df)
    n_accepted = _curate_out(window, qtbot, N_CURATED_OUT)
    assert n_accepted == n_detected - N_CURATED_OUT
    win, said = _run(window, qtbot)
    assert win is not None and win.report() is not None
    return window, win, n_detected, n_accepted, said


# ================================================================================================
# the four surfaces that used to disagree
# ================================================================================================

def test_the_status_line_and_the_report_are_about_the_same_named_set(curated_run):
    _window, win, n_detected, n_accepted, said = curated_run
    # What the app says while the run starts...
    assert any(f"Grouping {n_accepted} accepted event(s)" in s for s in said), said
    # ...and what it says when it finishes (the report's own headline, verbatim).
    assert any(f"{n_accepted} accepted region(s) passed the gate" in s for s in said), said
    # ...and the file no longer calls that same number the detected count.
    text = win.report().text()
    assert f"accepted regions        : {n_accepted}" in text
    assert "detected regions" not in text


def test_the_exported_file_names_the_curation_with_the_detectors_own_count(
        curated_run, tmp_path, monkeypatch):
    """The bytes on disk — ``write_report`` produces a document read WITHOUT the app."""
    _window, win, n_detected, n_accepted, _said = curated_run
    target = tmp_path / "subpops"

    def fake_ask(parent, *, title, suggestion, filters, outputs=None):
        return target

    monkeypatch.setattr("nanoscope.gui.main_window.ask_export_path", fake_ask)
    win.exportRequested.emit()

    text = target.with_suffix(".txt").read_text()
    assert f"curated out (not below) : {n_detected - n_accepted}" in text
    assert f"of {n_detected} detected region(s)" in text
    assert re.search(r"CURATION: .* curated OUT before this run started", text)
    # The count the detector really produced is recoverable from the file alone.
    assert str(n_detected) in text


def test_the_report_window_header_repeats_the_corrected_claim(curated_run):
    _window, win, _n_detected, n_accepted, _said = curated_run
    head = win.header()
    rep = win.report()
    assert f"{rep.n_kept} of {n_accepted} accepted region(s) passed the gate" in head
    assert "detected region(s) passed the gate" not in head


def test_the_detectors_own_count_is_the_one_in_the_ledger(curated_run):
    """Not reconstructed from event ids, not passed in by the GUI: read off the provenance
    the DETECTOR stamped, which ``EventSet.accepted`` carries forward."""
    _window, win, n_detected, _n_accepted, _said = curated_run
    assert win.report().meta["n_detected"] == n_detected
    assert win.report().meta["rows_filtered"] == "accepted"


# ================================================================================================
# a run with nothing curated out discloses nothing — but is still named honestly
# ================================================================================================

def test_a_run_with_nothing_removed_is_named_accepted_and_discloses_nothing(detected, qtbot):
    """The app ALWAYS hands over ``events.accepted``, so an app report is always about the
    curated-in set and says so — that is the true statement, and it costs nothing when the
    set happens to be all of them. What must not appear is a removal caveat or a
    "curated out" line for a run where nothing was removed: an accurate report does not
    grow a warning. (The "detected regions" wording is still what a non-app caller handing
    over an unfiltered table gets — pinned in the analysis test file.)"""
    window, _doc_id, events = detected
    win, _said = _run(window, qtbot)
    text = win.report().text()
    assert f"accepted regions        : {len(events.df)}" in text
    assert "curated out (not below)" not in text
    assert not [w for w in win.report().warnings if w.startswith("CURATION:")]
