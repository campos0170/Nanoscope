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


def _stage_spies(w, monkeypatch) -> dict[str, list[str]]:
    """Record the source every Explorer figure is STAGED with (the only thing a selection
    change may do to a Noise or Event waveforms figure), as the recording's file name."""
    staged: dict[str, list[str]] = {"psd": [], "event_plots": []}

    def spy(name, page):
        real = page.stage_source

        def wrapped(recipe, *a, **k):
            staged[name].append(str((recipe or {}).get("source_key", "")).rsplit("/", 1)[-1])
            return real(recipe, *a, **k)
        monkeypatch.setattr(page, "stage_source", wrapped)

    spy("psd", w._psd_window)
    spy("event_plots", w._event_plot_window)
    return staged


def _press_update(w, qtbot):
    """The user's Update plot: the Explorer toolbar's own button."""
    w._analysis_explorer.toolbar().update_action.trigger()
    qtbot.wait(10)


def _close_all(w):
    """The user closes every analysis window. The Noise and Event waveforms figures are pages
    of the Analysis Explorer (2026-11-11, 8661d35): they are closed by closing the Explorer,
    which hides them and resets nothing; the curation windows close themselves."""
    w._analysis_explorer.close()
    for win in (w._psd_window, w._event_plot_window):
        assert not win.isVisible()
    for win in (w._event_inspector, w._event_gallery):
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


def test_a_selection_switch_stages_followers_and_never_recomputes_an_open_page(
        two, monkeypatch, qtbot):
    """Only the selected tab is on screen, and neither tab is recomputed by a switch.

    2026-11-11: this test used to pin that the visible Event plots page was FED on every
    switch and the hidden PSD tab refreshed when its tab was activated. Since 8661d35
    (plotting redesign) Noise and Event waveforms are managed figures: a selection change
    reaches only the figures that opted in with Follow selection, STAGES the new source on
    them (a pending change, shown as "Update plot (n)") and leaves the displayed result alone
    until Update plot; showing a tab computes nothing."""
    w, a, b = two
    w._on_selection(a, 0)          # the fixture leaves B selected
    _open_all(w)
    owner = w._figure_workspace
    kpsd = owner.id_for_page(w._psd_window)
    kep = owner.id_for_page(w._event_plot_window)
    for key in (kpsd, kep):        # each figure has its result for recording A
        assert owner.activate(key)
        _press_update(w, qtbot)
        assert owner.get(key).spec.result.status == "current"
    fed = _spies(w, monkeypatch)
    staged = _stage_spies(w, monkeypatch)
    drawn = w._event_plot_window.density_counts().copy()

    # 1. The default figures do not follow the selection: nothing fed, nothing staged.
    w._on_selection(b, 0)
    w._on_selection(a, 0)
    assert fed["event_plots"] == [] and fed["psd"] == []
    assert staged == {"psd": [], "event_plots": []}

    # 2. A follower is told, in order, and stages; its drawn result stays what it was.
    owner.set_follow_selection(kpsd, True)
    owner.set_follow_selection(kep, True)
    for record in (*fed.values(), *staged.values()):
        record.clear()
    w._on_selection(b, 0)
    assert staged == {"psd": ["pore_B.dat"], "event_plots": ["pore_B.dat"]}
    for page in (w._psd_window, w._event_plot_window):      # the visible tab AND the hidden one
        assert page.inspector_adapter().pending().count >= 1
    w._on_selection(a, 0)
    assert staged == {"psd": ["pore_B.dat", "pore_A.dat"],
                      "event_plots": ["pore_B.dat", "pore_A.dat"]}
    assert fed["event_plots"] == [] and fed["psd"] == [], (
        "a selection change recomputed a figure instead of staging its source")
    np.testing.assert_array_equal(w._event_plot_window.density_counts(), drawn)

    # 3. Showing the hidden tab is not a refresh either.
    w._on_selection(b, 0)
    assert owner.activate(kpsd)
    assert w._psd_window.isVisible() and not w._event_plot_window.isVisible()
    assert fed["psd"] == [], "activating the PSD tab recomputed it"

    # 4. Update plot is the one route that moves the figure to the staged recording.
    _press_update(w, qtbot)
    assert fed["psd"] and all(fed["psd"]), "Update plot must feed the page while it is visible"
    assert w._psd_window.curves()[0].doc_id == b
    assert owner.get(kpsd).spec.result.status == "current"


def test_a_reopened_window_is_fed_on_its_way_back_up(two, monkeypatch, qtbot):
    """The property the skip leans on, pinned: every opener refreshes immediately before
    show(), so no switch that happened while the window was closed is ever missed.

    2026-11-11: that still holds for the two curation windows (Event inspector, Event
    gallery), which are fed on the way back up. The Noise and Event waveforms figures are
    Explorer pages now (8661d35): reopening raises the SAME saved figure and computes nothing,
    and "no switch is missed" means the figure that follows the selection kept staging while
    the Explorer was closed, so its binding names the recording selected last."""
    w, a, b = two
    _open_all(w)
    owner = w._figure_workspace
    kpsd = owner.id_for_page(w._psd_window)
    kep = owner.id_for_page(w._event_plot_window)
    owner.set_follow_selection(kpsd, True)
    owner.set_follow_selection(kep, True)
    psd, events = w._psd_window, w._event_plot_window
    _close_all(w)
    w._on_selection(b, 0)          # switches while everything is closed
    w._on_selection(a, 0)
    assert owner.binding(kpsd)["source"]["source_key"].endswith("pore_A.dat")

    fed = _spies(w, monkeypatch)
    _open_all(w)                   # reopen through the real openers

    assert fed["inspector"], "the opener must feed the window it is about to show"
    assert fed["gallery"]
    # ...the two figures come back as themselves, computing nothing ...
    assert w._psd_window is psd and w._event_plot_window is events
    assert len(owner.records()) == 2
    assert fed["psd"] == [] and fed["event_plots"] == [], (
        "reopening the Explorer recomputed a saved figure")
    # ...and no switch was missed while they were closed.
    assert owner.binding(kpsd)["source"]["source_key"].endswith("pore_A.dat")
    assert owner.binding(kep)["source"]["source_key"].endswith("pore_A.dat")
    # Only the selected tab is on screen; the hidden one is still installed.
    assert w._event_plot_window.isVisible() and not w._psd_window.isVisible()
    assert owner.activate(kpsd)
    assert w._psd_window.isVisible() and not w._event_plot_window.isVisible()
    assert fed["psd"] == [], "showing a tab is not a refresh"


def test_subpop_views_stay_unconditional(two, monkeypatch):
    """The one non-window entry keeps its unconditional step: it owns app-wide state
    (report colouring) that outlives any window — the `_refresh_surface_windows`
    docstring's own carve-out."""
    w, a, b = two
    calls = []
    monkeypatch.setattr(w, "_refresh_subpop_views", lambda: calls.append(True))
    w._on_selection(b, 0)
    assert calls, "_refresh_subpop_views must run on every selection, windows or none"
