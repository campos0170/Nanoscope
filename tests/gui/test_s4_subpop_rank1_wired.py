"""CS4-03 on the shipped surface: ``Analysis ▸ Event subpopulations`` over a recording
whose pooled PCA keeps ONE component.

The analysis-layer invariant is pinned in
``tests/analysis/test_s4_subpop_embedding_refusal.py``. What is at stake HERE is what the
researcher actually gets: the real QAction, the app's own detector, the real report
window. Before the fix this run ended in a modal titled "Analysis failed" reading

    ValueError: n_components=2 must be between 1 and min(n_samples, n_features)=1
    with svd_solver='randomized'

with no report, no subpopulation assignment adopted, and the whole feature + gate +
clustering stage discarded — for a t-SNE picture the module itself exports and never
prints.

The recording is the one from the defect report: 120 events of randomised width and
depth, seed 120. One leptokurtic event is enough to put PC1 past the 95 % variance target
(measured here: n_kept = 1, PC1 = 0.985, loading 0.99 on ``kurtosis``), which is the
condition — not a knife edge, and not a small-n effect.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import subpop_report as SR
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N_EVENTS = 120
SEED = 120


class _CheapWithTsne(SR.SubpopConfig):
    """The product's t-SNE default (``with_tsne=True``) over a grid a GUI test can afford.

    ``with_tsne`` is the point of this file: the analysis tests' ``_cheap`` turns it off,
    which is exactly why no existing test ever drove the embedding stage on a degenerate
    scope.
    """

    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        kw.setdefault("with_tsne", True)
        kw.setdefault("with_umap", False)
        super().__init__(**kw)


def _varied_trace(seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 60_000 + N_EVENTS * 4_000
    x = np.full(n, 100.0) + rng.normal(0.0, 1.2, n)
    pos = 20_000
    for _i in range(N_EVENTS):
        width = int(rng.integers(80, 700))
        depth = float(rng.uniform(15.0, 70.0))
        seg = np.full(width, depth) + rng.normal(0.0, 2.0, width)
        x[pos:pos + width] -= seg
        pos += 4_000
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def rank_one_run(window, qtbot, tmp_path, monkeypatch):
    """Detect through the app, then trigger the real menu action. Returns what was shown."""
    monkeypatch.setattr(SR, "SubpopConfig", _CheapWithTsne)

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question",
        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical",
        staticmethod(lambda parent, title, text, *a, **k: shown.append((title, text))))
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        staticmethod(lambda parent, title, text, *a, **k: shown.append((title, text))))
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k: shown.append((title, text))))

    arr = _varied_trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "varied.dat", format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(arr,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    panel = window._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    # 5 ms: the widest events (up to 700 samples = 7 ms) fall outside the dwell
    # window and are curated out by the detector itself, which is what leaves the
    # 75-event accepted set whose PCA keeps a single component.
    panel._detection.max_dwell_s.setValue(5e-3)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=120_000):
        panel.trigger_detection()
    qtbot.wait(30)
    assert len(panel.current_detection().payload.df) > 100

    failed: list[str] = []
    window._analysis_controller.failed.connect(failed.append)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=300_000):
        window._act_subpop.trigger()
    qtbot.wait(50)
    return window, failed, shown


def test_the_run_finishes_instead_of_dying_on_a_solver_message(rank_one_run):
    window, failed, shown = rank_one_run
    assert failed == [], failed
    assert not [t for t in shown if t[0] == "Analysis failed"], shown
    assert not any("svd_solver" in text for _title, text in shown), shown
    # The report window is open with a report in it, and the assignment was adopted.
    assert window._subpop_window is not None
    assert window._subpop_window.report() is not None
    assert window._subpop_assignment


def test_the_scope_really_is_the_degenerate_one(rank_one_run):
    """Guard on the FIXTURE: if PCA ever keeps 2 components here, this file stops
    testing the condition it was written for and must be told so."""
    window, _failed, _shown = rank_one_run
    rep = window._subpop_window.report()
    assert rep.clustering.scopes["ALL"].pca.n_kept == 1
    assert rep.embeddings is None


def test_the_caveat_strip_tells_the_reader_the_picture_is_missing_and_why(rank_one_run):
    """The reader looking at the (empty) figures pane reads it verbatim, numbered."""
    window, _failed, _shown = rank_one_run
    win = window._subpop_window
    strip = win._caveats.toPlainText()
    assert "NO T-SNE:" in strip
    assert "at least 2 components wide" in strip
    assert "PC1 alone carries" in strip and "kurtosis" in strip
    assert "svd_solver" not in strip
    # ...and the caveat count on the box title and the header agree with the report.
    assert str(len(win.report().warnings)) in win._caveat_box.title()


def test_the_note_also_rides_in_the_report_text_and_on_the_main_window(rank_one_run):
    window, _failed, _shown = rank_one_run
    assert "NO T-SNE:" in window._subpop_window.report().text()
    assert any(w.startswith("NO T-SNE:") for w in window._subpop_caveats)
