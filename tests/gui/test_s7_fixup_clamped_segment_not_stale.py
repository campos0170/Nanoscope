"""Sweep-7 fix-up, X4 (major): a run whose open-pore windows were CLAMPED to a short
recording is not "stale" against a Preprocess tab that would clamp to the same windows.

Measured on a 0.08 s cut of the owner's -600 mV recording, detected at the defaults
(0.5 s / 0.1 s): the estimator clamped to 80 / 16 ms and stamped that config's
fingerprint (sha256:31991df77507); ``_baseline_staleness`` compared it with the FORM's
500 / 100 ms fingerprint and raised "The open-pore estimator changed since this detection
ran ... rough 500 ms / base 100 ms is configured now" over DET3's CLAMPED line saying the
opposite, badged nine columns on the table, the gallery and the inspector, and offered a
"Re-measure depths" that re-measured to the identical fingerprint and left the banner up
for good. Nothing had changed: every fresh run on that recording clamps to the same
windows.

"Stale" now means "differs from what a fresh run WOULD use on THIS recording": the live
config is resolved through ``baseline.adapt_to_length`` -- the same clamp
``characterisation_baseline`` applies -- before it is fingerprinted, so the judge and the
estimator read one config. A synthetic 4 000-sample recording at 50 kHz stands in for the
owner's segment; the clamp (80 / 16 ms at the 0.5 / 0.1 s default) is the same.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.baseline import FALLBACK_CLAMPED
from nanoscope.analysis.modules import register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
I0 = -1000.0
N = 4_000                       # 0.08 s: shorter than the default 0.1 s base window


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


def _short_signal(*, n=N, depth=300.0, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    x[1_000:1_200] -= depth
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, name) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def clamped(window, qtbot, tmp_path):
    """A real Schmitt detection at the app defaults on a recording the windows must
    clamp to; the stamps are the engine's."""
    did = window._session.add_document(_doc(tmp_path, _short_signal(), name="seg.dat"))
    window._create_default_layers(window._session.document(did))
    window._on_selection(did, 0)
    qtbot.wait(20)
    panel = window._analysis_panel
    bl = panel.baseline_panel
    assert (bl.rough_s.value(), bl.base_s.value()) == (0.5, 0.1), "the app defaults"
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        assert panel._run_detection(panel.detection_params())
    qtbot.wait(50)
    res = panel.current_detection()
    assert res is not None
    prov = res.payload.provenance
    assert prov.get("baseline_fallback") == FALLBACK_CLAMPED
    assert abs(float(prov["baseline_rough_s"]) - N / FS) < 1e-6
    return window, did


def test_a_clamped_run_is_not_stale_against_the_form_it_was_run_from(clamped, qtbot):
    window, _did = clamped
    panel = window._analysis_panel
    res = panel.current_detection()
    assert not panel._stale.isVisibleTo(panel), (
        "a false 'estimator changed' banner over a run nothing has changed: "
        + panel._stale.message())
    assert panel._table.stale_columns == ()
    # ...while the CLAMPED line under the form still says what actually ran.
    bl = panel.baseline_panel
    assert bl.run_windows_label.isVisibleTo(bl)
    assert "CLAMPED" in bl.run_windows_label.text()
    # The judge and the estimator read one config: the resolver's fingerprint IS the stamp.
    cfg, token, live, fs, n = panel._live_baseline_for_run()
    assert token == FALLBACK_CLAMPED and n == N and fs == FS
    assert cfg.fingerprint(fs) == res.baseline_fingerprint
    assert live.fingerprint(fs) != res.baseline_fingerprint, (
        "the control: the FORM's own fingerprint never matches a clamped stamp")


def test_a_form_change_that_clamps_to_the_same_windows_is_not_stale_either(clamped, qtbot):
    """1.0 / 0.2 s keeps the 5:1 ratio, so a fresh run here would clamp to the same
    80 / 16 ms: nothing a re-measure could change."""
    window, _did = clamped
    panel = window._analysis_panel
    bl = panel.baseline_panel
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        bl.rough_s.setValue(1.0)
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        bl.base_s.setValue(0.2)
    assert (bl.rough_s.value(), bl.base_s.value()) == (1.0, 0.2)
    assert not panel._stale.isVisibleTo(panel), panel._stale.message()
    assert panel._table.stale_columns == ()


def test_a_form_change_that_clamps_differently_is_stale_says_so_and_a_re_measure_clears_it(
        clamped, qtbot):
    """base 0.2 s makes the ratio 2.5:1 -- a fresh run would clamp to 80 / 32 ms, which
    is a different estimator: the banner is right to rise, its sentence names the
    clamp, and the offered Re-measure lands on the resolver's fingerprint and clears it."""
    window, _did = clamped
    panel = window._analysis_panel
    bl = panel.baseline_panel
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        bl.base_s.setValue(0.2)
    assert bl.base_s.value() == 0.2
    banner = panel._stale
    assert banner.isVisibleTo(panel) and banner.kind == "baseline"
    msg = banner.message()
    assert "changed since this detection ran" in msg
    assert "CLAMPED to this recording's 0.08 s length" in msg, msg
    assert "rough 80 ms / base 32 ms" in msg, msg
    assert banner._btn.isEnabled() and banner.action_text() == "Re-measure depths"

    cfg, token, _live, fs, _n = panel._live_baseline_for_run()
    assert token == FALLBACK_CLAMPED
    want = cfg.fingerprint(fs)
    assert panel.remeasure_active_detection() == "started"
    qtbot.waitUntil(lambda: panel._pending is None, timeout=30_000)
    qtbot.wait(50)
    assert panel.current_detection().baseline_fingerprint == want
    assert not banner.isVisibleTo(panel)
    assert panel._table.stale_columns == ()


def test_a_form_change_the_ladder_does_not_clamp_is_judged_against_the_form_itself(
        clamped, qtbot):
    """The ladder clamps on the TRACKING window (the base pass for a masked median), so
    base 0.05 s -- 2 500 samples, inside the 4 000 -- takes rung 1 and a fresh run here
    measures at the form's own 0.5 / 0.05 s. The judge must say exactly that: the form's
    windows named, no CLAMPED clause (there is no clamp to name), and the Re-measure
    lands on the form's own fingerprint. This is the control that the resolver mirrors
    the ladder rather than assuming a short recording always clamps."""
    window, _did = clamped
    panel = window._analysis_panel
    bl = panel.baseline_panel
    with qtbot.waitSignal(panel.stalenessChanged, timeout=1_000):
        bl.base_s.setValue(0.05)
    assert bl.base_s.value() == 0.05
    cfg, token, live, fs, _n = panel._live_baseline_for_run()
    assert token is None and cfg == live
    banner = panel._stale
    assert banner.isVisibleTo(panel) and banner.kind == "baseline"
    msg = banner.message()
    assert "rough 500 ms / base 50 ms" in msg and "is configured now" in msg, msg
    assert "CLAMPED" not in msg, msg

    want = live.fingerprint(fs)
    assert panel.remeasure_active_detection() == "started"
    qtbot.waitUntil(lambda: panel._pending is None, timeout=30_000)
    qtbot.wait(50)
    assert panel.current_detection().baseline_fingerprint == want
    assert not banner.isVisibleTo(panel)
    assert panel._table.stale_columns == ()
