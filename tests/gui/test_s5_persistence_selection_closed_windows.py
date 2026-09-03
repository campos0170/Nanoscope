"""MAR-01 — ``_on_selection`` does not feed windows the user has CLOSED.

The END-001 class, one seam over. ``_refresh_surface_windows`` already states the design
rule ("A window the user has CLOSED is not re-fed... once a window existed it was
refreshed forever, with nothing on screen to show for it") and guards its feeds with
``_on_screen``; ``_on_selection`` stepped the same four refreshers UNCONDITIONALLY, so
every recording switch after the PSD or Event-plots window had EVER been opened paid its
full synchronous recompute — a Welch PSD, the whole calibrated array — against an
invisible window. Measured on the real -600 mV / 385-400 s ABFs: 12/12 hidden feeds per
run, +86–304 ms per switch, forever.

Nothing is lost by the skip, and that is a property of the openers: each one refreshes
immediately before ``show()`` — the reopen control below pins it.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600)]


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= 300.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, name: str) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(_signal(),))


@pytest.fixture
def two(qapp, qtbot, tmp_path):
    """Two detected recordings and the MainWindow SHOWN — visibility is the whole point."""
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    ids = []
    for name in ("pore_A.dat", "pore_B.dat"):
        doc = _doc(tmp_path, name)
        doc_id = w._session.add_document(doc)
        w._create_default_layers(doc)
        ids.append(doc_id)
    for doc_id in ids:
        w._on_selection(doc_id, 0)
        df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
        w._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={"min_dwell_s": 1e-4},
            calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
            payload=EventSet(df, prov)))
    yield w, ids[0], ids[1]
    w._analysis_controller.abort()


def _spies(w, monkeypatch) -> dict[str, list[bool]]:
    """Record every feed each surface window receives, tagged with its visibility at the
    moment of the feed — the isVisible()==False entries are the defect."""
    fed: dict[str, list[bool]] = {"psd": [], "event_plots": [], "inspector": [],
                                  "gallery": []}
    psd, ep = w._psd_window, w._event_plot_window
    insp, gal = w._event_inspector, w._event_gallery

    def spy(name, obj, meth):
        real = getattr(obj, meth)
        def wrapped(*a, **k):
            fed[name].append(obj.isVisible())
            return real(*a, **k)
        monkeypatch.setattr(obj, meth, wrapped)

    spy("psd", psd, "set_target")
    spy("event_plots", ep, "set_source")
    spy("inspector", insp, "set_events")
    spy("gallery", gal, "set_events")
    monkeypatch.setattr(insp, "refresh", lambda *a, **k: fed["inspector"].append(
        insp.isVisible()), raising=False)
    return fed


def _open_all(w):
    w._open_psd()
    w._open_event_plots()
    w._open_event_inspector()
    w._open_event_gallery()


def _close_all(w):
    for win in (w._psd_window, w._event_plot_window, w._event_inspector,
                w._event_gallery):
        win.close()
        assert not win.isVisible()


def test_switching_recordings_feeds_no_closed_window(two, monkeypatch):
    w, a, b = two
    _open_all(w)
    _close_all(w)
    fed = _spies(w, monkeypatch)

    for target in (b, a, b, a):
        w._on_selection(target, 0)

    hidden = {name: feeds for name, feeds in fed.items() if feeds}
    assert hidden == {}, (
        f"closed windows were fed on selection switches: {hidden} "
        f"(each False is a synchronous recompute nothing on screen shows)")


def test_open_windows_are_still_fed_on_every_switch(two, monkeypatch):
    """CONTROL: the guard skips CLOSED windows only — an open one must follow the
    selection, or the fix trades a latency defect for a staleness defect."""
    w, a, b = two
    _open_all(w)
    fed = _spies(w, monkeypatch)

    w._on_selection(b, 0)
    w._on_selection(a, 0)

    assert fed["psd"], "the open PSD window must follow the selection"
    assert all(fed["psd"]), "and every feed must land on a VISIBLE window"
    assert fed["event_plots"] and all(fed["event_plots"])


def test_a_reopened_window_is_fed_on_its_way_back_up(two, monkeypatch):
    """The property the skip leans on, pinned: every opener refreshes immediately before
    show(), so no switch that happened while the window was closed is ever missed."""
    w, a, b = two
    _open_all(w)
    _close_all(w)
    w._on_selection(b, 0)          # switches while everything is closed
    w._on_selection(a, 0)

    fed = _spies(w, monkeypatch)
    _open_all(w)                   # reopen through the real openers

    assert fed["psd"], "the opener must feed the window it is about to show"
    assert fed["event_plots"]
    assert fed["inspector"]
    assert fed["gallery"]
    # ...and the windows are on screen afterwards, showing the CURRENT selection's data.
    assert w._psd_window.isVisible() and w._event_plot_window.isVisible()


def test_subpop_views_stay_unconditional(two, monkeypatch):
    """The one non-window entry keeps its unconditional step: it owns app-wide state
    (report colouring) that outlives any window — the `_refresh_surface_windows`
    docstring's own carve-out."""
    w, a, b = two
    calls = []
    monkeypatch.setattr(w, "_refresh_subpop_views", lambda: calls.append(True))
    w._on_selection(b, 0)
    assert calls, "_refresh_subpop_views must run on every selection, windows or none"
