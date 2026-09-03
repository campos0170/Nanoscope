"""GA5-01, through the chrome the user watches -- the status-bar progress bar.

The dossier's surface: 'Load all anyway' + Analysis > Detect on the 606 MB ABF ran
~30 s with the shared QProgressBar determinate at 0% throughout, because
`DetectSchmitt.run` reported only 0.0 and 1.0. The 606 MB measurement itself lives in
the fix's verification run (a test must not open a 606 MB recording); what is pinned
here is the same wiring at test scale:

* a real MainWindow detection's ``progressed`` signal now carries fractions strictly
  between 0 and 1, in monotone order -- the signal `_on_progress` feeds the bar from;
* `_on_progress` renders such a fraction as an intermediate bar value with the
  stage-naming format, so any mid fraction that arrives IS visible chrome.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0


def _doc(tmp_path) -> TraceDocument:
    """4 s with real blockades -- big enough that every baseline stage runs."""
    n = int(FS * 4.0)
    rng = np.random.default_rng(3)
    x = 1000.0 + rng.normal(0.0, 5.0, n)
    for s in range(20_000, n - 2_000, 40_000):
        x[s:s + 300] -= 80.0
    info = ChannelInfo(index=0, name="current", raw_units="pA",
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "ga501_progress.dat", format_name="synthetic", n_samples=n,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta,
                         channels=(np.ascontiguousarray(x, dtype=np.float32),))


def test_a_real_detection_progresses_the_bar_signal(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        doc = _doc(tmp_path)
        doc_id = w._session.add_document(doc)
        w._create_default_layers(doc)
        w._on_selection(doc_id, 0)
        qtbot.wait(20)

        fracs: list[float] = []
        w._analysis_controller.progressed.connect(
            lambda f, m: fracs.append(float(f)))
        with qtbot.waitSignal(w._analysis_controller.finished, timeout=120_000):
            w._analysis_panel.trigger_detection()
        qtbot.wait(50)

        assert w._session.results.get(doc_id, 0, "detect.schmitt") is not None
        mid = [f for f in fracs if 0.0 < f < 1.0]
        # The whole defect: this list was empty on every detect, regardless of size.
        assert len(mid) >= 4, fracs
        assert all(a <= b for a, b in zip(fracs, fracs[1:])), fracs
    finally:
        w._analysis_controller.abort()


def test_on_progress_renders_a_mid_fraction_as_visible_chrome(qapp, qtbot):
    """The half of the surface the signal test cannot see: a mid fraction reaching
    `_on_progress` becomes an intermediate bar value with the stage in its format --
    so the fractions asserted above are, by this wiring, pixels."""
    w = MainWindow()
    qtbot.addWidget(w)
    try:
        w._on_progress(0.37, "detecting events (baseline health (QC block))")
        assert w._progress.isVisible() or w._progress.isVisibleTo(w)
        assert w._progress.value() == 37
        assert "baseline health" in w._progress.format()
    finally:
        w._analysis_controller.abort()
