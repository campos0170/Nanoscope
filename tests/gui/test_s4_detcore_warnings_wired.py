"""B4DA-06 / B4DA-02 through the widget the user actually reads.

B4DA-06. `BaselineResult.warnings` prepended ``self.fallback`` and then extended with
``self.health.warnings`` -- and the health block already carries the identical sentence,
written onto a ``tracking_window_fits`` check by `_with_fallback`. Every short or
hand-cut segment (a legal document: ``core.segment.MIN_SEGMENT_SAMPLES`` is 2) therefore
handed the Stats panel the same ~300-character paragraph twice, and
:meth:`StatsPanel.set_warnings` renders what it is given WHOLE and unfiltered -- by
design, since it must not know which warning it is showing. Measured through the live
widget before the fix: 2 bullets under "About these numbers", 1 distinct.

B4DA-02. The other half of the same surface: the ``edge_tracking`` row's sentence has to
REACH that block, or a baseline that never tracked the head of the recording is still
green everywhere the user looks.

Both are asserted on the rendered text of a real `StatsPanel`, and the second one
through a real `MainWindow` detection -- the gesture is Analysis > Detect, and what is
quoted is what the panel then reads.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.modules import _build_warnings
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import StatsPanel

FS = 100_000.0


def _bullets(panel: StatsPanel) -> list[str]:
    """The advisory block as the user reads it: one string per bullet."""
    text = panel.warnings_text()
    if not text:
        return []
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return [b.strip() for b in body.split("\n•") if b.strip().strip("•")]


@pytest.fixture
def stats(qapp, qtbot):
    p = StatsPanel()
    qtbot.addWidget(p)
    return p


def test_a_hand_cut_segment_gets_the_fallback_advisory_once(stats, qtbot):
    """50 ms of signal: the tracking window is clamped, the fallback is real, and it is
    ONE fact -- so it gets one sentence."""
    rng = np.random.default_rng(3)
    sig = 1000.0 + rng.normal(0.0, 5.0, 5_000)
    sig[2_500:2_600] -= 100.0
    df, prov = detect_events(sig, FS)
    warns = _build_warnings(prov, df)

    stats.set_warnings(warns)
    bullets = _bullets(stats)
    assert bullets, stats.warnings_text()
    assert len(bullets) == len(set(bullets)), bullets
    clamped = [b for b in bullets if "shorter than the 0.1 s tracking window" in b]
    assert len(clamped) == 1, clamped
    # The sentence itself is unchanged -- this is a de-duplication, not a re-wording.
    assert "Depths are NOT comparable" in clamped[0]


@pytest.mark.parametrize("n", [2, 500, 5_000])
def test_no_length_of_segment_prints_a_sentence_twice(stats, n):
    """The three lengths the dossier measured (2 samples / 5 ms / 50 ms), each of which
    rendered its fallback twice."""
    rng = np.random.default_rng(3)
    sig = 1000.0 + rng.normal(0.0, 5.0, n)
    sig[n // 2: n // 2 + max(1, n // 20)] -= 100.0
    df, prov = detect_events(sig, FS)
    warns = _build_warnings(prov, df)
    assert len(warns) == len(set(warns)), warns
    stats.set_warnings(warns)
    bullets = _bullets(stats)
    assert len(bullets) == len(set(bullets)), bullets


def _wander_doc(tmp_path) -> TraceDocument:
    """4 s of open-pore wander and ZERO events -- B4DA-02's fixture, as a document."""
    n = int(FS * 4.0)
    t = np.arange(n) / FS
    rng = np.random.default_rng(2)
    x = (1000.0 - 100.0 * t + 250.0 * np.sin(2 * np.pi * 0.5 * t)
         + rng.normal(0.0, 5.0, n))
    info = ChannelInfo(index=0, name="current", raw_units="pA",
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "b4da02_wander.dat", format_name="synthetic", n_samples=n,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta,
                         channels=(np.ascontiguousarray(x, dtype=np.float32),))


def test_the_untracked_edge_reaches_the_stats_panel(qapp, qtbot, tmp_path):
    """The shipped surface for B4DA-02: run the detection the user runs, then read the
    "About these numbers" block. Before the fix it said nothing about the head at all
    while 17 rows of pure artefact sat in the table."""
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        doc = _wander_doc(tmp_path)
        doc_id = w._session.add_document(doc)
        w._create_default_layers(doc)
        w._on_selection(doc_id, 0)
        qtbot.wait(20)
        with qtbot.waitSignal(w._analysis_controller.finished, timeout=120_000):
            w._analysis_panel.trigger_detection()
        qtbot.wait(50)

        res = w._session.results.get(doc_id, 0, "detect.schmitt")
        assert res is not None
        said = w._analysis_panel._stats.warnings_text()
        assert "never TRACKED the trace at its own edge" in said, said
        assert "DO NOT read a depth" in said
        bullets = _bullets(w._analysis_panel._stats)
        assert len(bullets) == len(set(bullets)), bullets
    finally:
        w._analysis_controller.abort()
