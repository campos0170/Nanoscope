"""CF2-4, wired through the real widgets — enabling Savitzky-Golay at the
Preprocess tab's DEFAULTS must not flip the polarity and vanish the events.

The user-visible failure: a clean 5-blockade recording detects 5 events with
savgol off; ticking the savgol stage (1 ms window, order 3 — the panel's own
defaults) flipped the measured polarity to +1 on the filter's edge lobes,
detection searched the wrong way and reported a clean-looking ZERO, and neither
``res.warnings`` (the Stats panel's honesty channel) nor ``polarity_warning``
said a word — the readout claimed 'measured' at confidence 0.33.

Pinned here through a real :class:`MainWindow`: the savgol run finds the same
5 events, the polarity readout says blockade/down, and the result's warning
surface carries the arbitration sentence naming the filter as the likely cause.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. -----------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
_BLOCKADES = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
              (70_000, 70_200), (85_000, 85_350)]


def _five_blockade_doc(tmp_path) -> TraceDocument:
    rng = np.random.default_rng(0)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _BLOCKADES:
        x[s:e] -= 30.0
    x = np.ascontiguousarray(x, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA",
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "cf2_savgol.dat", format_name="synthetic", n_samples=N,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
        channels=(info,))
    return TraceDocument(meta=meta, channels=(x,))


def _install(window: MainWindow, doc: TraceDocument) -> tuple[str, int]:
    session = window._session
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    channel = next(c.index for c in doc.meta.channels
                   if c.kind is ChannelKind.CURRENT and not c.is_dead)
    window._on_selection(doc_id, channel)
    return doc_id, channel


def _run_detection(window: MainWindow, qtbot) -> None:
    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)   # let the queued add_result -> resultsChanged settle


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def test_savgol_at_panel_defaults_keeps_the_events_and_warns_honestly(
    window, qtbot, tmp_path
):
    doc_id, channel = _install(window, _five_blockade_doc(tmp_path))
    qtbot.wait(20)
    det = window._analysis_panel._detection

    # Ground truth first: the unfiltered run finds the 5 blockades, decisively.
    _run_detection(window, qtbot)
    r0 = window._session.results.get(doc_id, channel, "detect.schmitt")
    assert len(r0.payload.df) == 5, "unfiltered fixture drifted"
    assert r0.payload.provenance["polarity"] == -1
    ev0 = r0.payload.provenance["polarity_evidence"]
    assert ev0["confidence"] == 1.0 and "decided_by" not in ev0

    # The regression: tick the savgol stage and re-detect at its DEFAULTS.
    det.savgol_enabled.setChecked(True)
    qtbot.wait(20)
    _run_detection(window, qtbot)
    r = window._session.results.get(doc_id, channel, "detect.schmitt")
    prov = r.payload.provenance

    # The filter's edge lobes must not flip the direction or eat the events.
    assert prov["polarity"] == -1
    assert len(r.payload.df) == 5, (
        f"savgol at defaults reported {len(r.payload.df)} of 5 real blockades"
    )

    # The readout and the table tell ONE story: blockade/down, not enhancement.
    assert "blockade" in det.polarity_readout.text()

    # And the honesty channel the Stats panel prints carries the arbitration
    # sentence, naming the filter as the likely cause with a way out.
    assert prov["polarity_warning"] is not None
    polar_warns = [w for w in r.warnings if "polarity" in w.lower()]
    assert polar_warns, f"no polarity sentence in res.warnings: {r.warnings!r}"
    assert any("Savitzky-Golay" in w for w in polar_warns)
    assert any("polarity override" in w for w in polar_warns)
