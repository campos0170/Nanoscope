"""CLOCK — the report's halves are the ANALYSED span's, read off a WIRED MainWindow.

THE DEFECT. ``subpop_report.feature_table`` split ``time_half`` at the midpoint of
``len(detrended)/fs`` — the whole raw trace — while the events under it came from
whatever span the detector was pointed at. The ML detector's range combo DEFAULTS
to "Visible view", so on the app's own default path the two clocks routinely
differ, and on a recording whose events are spread perfectly evenly the app
printed:

    THE RECORDING IS NOT STATIONARY: 100 % of all gated events fall in the second
    half of it. ...

which is false of the events it measured, and — because the pooled fraction
collapses to 0.00 — it also suppressed the per-class check underneath it.

WHERE THE OLD TESTS LOOK, AND THE STEP PAST IT. Every wired subpopulation test in
this suite drives the SCHMITT detector, which always analyses ``(0, n)``, so the
two clocks coincide and the bug is invisible. The step past that is the
**range-limited detection** — the non-default code path — driven here through the
real ML module with the suite's fake Keras model, so the whole app path (panel ->
AnalysisController -> MLDetect -> ResultStore -> Analysis > Event
subpopulations…) runs for real.

The same recording is run BOTH ways in the same window — whole recording and
visible sub-range — so the assertion is not "the number is nice" but "the two
reports each split their OWN span", which is the invariant and which no single
run can state.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. -----------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import functools
import pathlib

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import event_features as EF
from nanoscope.analysis import ml_detect
from nanoscope.analysis import subpop_gmm as SG
from nanoscope.analysis import subpop_report as SR
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N_SAMPLES = 1_100_000                 # 22.000 s
N_EVENTS = 700
VISIBLE_FROM_S = 13.2                 # the framed sub-range: the last 8.8 s
FAKE_W = 3072                         # the real model's window size


# ===========================================================================
# 0 — which tree is this?
# ===========================================================================


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """Passes in BOTH trees, so nothing below can be an import that never happened."""
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    for mod in (SR, SG, EF):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents, f"{mod.__name__} at {p} is not under {root}"
        assert p.stat().st_size > 20_000, f"{p} is {p.stat().st_size} B — a stub?"
    print(f"\n[CLOCK] nanoscope resolved from {root}")


# ===========================================================================
# the recording: 700 events, PERFECTLY EVENLY spread over the whole clock
# ===========================================================================


def _starts() -> np.ndarray:
    """Event onsets, evenly spaced across the whole recording. No jitter, no drift."""
    return np.linspace(6_000, N_SAMPLES - 12_000, N_EVENTS).astype(np.int64)


def _trace(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 200.0, dtype=np.float64) + rng.normal(0.0, 1.5, N_SAMPLES)
    for i, s in enumerate(_starts()):
        deep = bool(i % 2)
        depth = (58.0 if deep else 26.0) + rng.normal(0.0, 2.0)
        dwell = int(rng.integers(150, 220) if deep else rng.integers(60, 110))
        x[s:s + dwell] -= depth
    return np.ascontiguousarray(x, dtype=np.float32)


def _document(tmp_dir: pathlib.Path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_dir / "synthetic_even.csv", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


# ===========================================================================
# the fake Keras model — the suite's own, so no TensorFlow is imported
# ===========================================================================


class _FakeEventModel:
    def __init__(self, window: int) -> None:
        self.window = window

    def predict(self, arr, verbose=0):
        x = np.asarray(arr, dtype=np.float32)
        return (np.abs(x) > 0.5).astype(np.float16)


def _use_fake_model(monkeypatch) -> None:
    wrapper = ml_detect.KerasEventModel(
        model=_FakeEventModel(FAKE_W), window=FAKE_W, path="<fake>", mtime=0.0,
        input_shape=(None, FAKE_W, 1), output_shape=(None, FAKE_W, 1))
    monkeypatch.setattr(ml_detect, "TENSORFLOW_AVAILABLE", True)
    monkeypatch.setattr(ml_detect, "load_keras_model", lambda _path: wrapper)


# ===========================================================================
# the wired run
# ===========================================================================


def _pump(qapp, ms: int = 60) -> None:
    import time
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)
    qapp.processEvents(QtCore.QEventLoop.AllEvents, 10)


def _until(qapp, pred, timeout: float, label: str) -> None:
    import time
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        qapp.processEvents(QtCore.QEventLoop.AllEvents, 20)
        if pred():
            return
        time.sleep(0.002)
    raise AssertionError(f"timed out after {timeout}s waiting for {label}")


class _Run:
    """One wired subpopulation report, plus the span the detector actually used."""

    def __init__(self, win, report, det_prov):
        self.report = report
        self.prov = dict(det_prov)
        sw = win._subpop_window
        self.caveats = sw.caveats()
        self.text = sw.text()

    @property
    def q(self):
        return self.report.quantification

    @property
    def span_s(self) -> tuple[float, float]:
        from nanoscope.analysis.events import analysed_span
        i0, i1 = analysed_span(self.prov, N_SAMPLES)
        return i0 / FS, i1 / FS

    def caveat(self, prefix: str) -> str:
        for w in self.report.warnings:
            if w.startswith(prefix):
                assert w in self.caveats, "a caveat the report raised is not on screen"
                return w
        return ""

    def pooled(self) -> float:
        ef = self.q.table["early_frac"].to_numpy(dtype=float)
        n = self.q.table["n"].to_numpy(dtype=float)
        return float((ef * n).sum() / n.sum())


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """A PRIVATE QSettings namespace — the suite must never touch the real store."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("RPStationarityClockPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


@pytest.fixture(scope="module")
def wired(qapp, tmp_path_factory, monkeypatch_module):
    """ONE window, ONE recording, TWO detections: whole recording, then the last 8.8 s."""
    mp = monkeypatch_module
    tmp = tmp_path_factory.mktemp("rp_clock")
    mp.setattr(
        SR, "SubpopConfig",
        functools.partial(SR.SubpopConfig, k_grid=(1, 2, 3), n_splits=2,
                          n_boot=0, with_tsne=False, with_umap=False),
    )
    mp.setattr(QtWidgets.QMessageBox, "question",
               staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    boxed: list[str] = []

    def _dialog_exec(self, *a, **k):
        try:
            boxed.append(f"{self.windowTitle()!r}: {self.text()!r}")
        except Exception:                       # noqa: BLE001 — not a QMessageBox
            boxed.append(repr(self))
        return QtWidgets.QDialog.Rejected

    for _cls in (QtWidgets.QDialog, QtWidgets.QMessageBox):
        mp.setattr(_cls, "exec", _dialog_exec, raising=False)
        mp.setattr(_cls, "exec_", _dialog_exec, raising=False)
    for _nm in ("information", "warning", "critical", "about"):
        mp.setattr(QtWidgets.QMessageBox, _nm,
                   staticmethod(lambda *a, _n=_nm, **k: (
                       boxed.append(f"{_n}: {a[2] if len(a) > 2 else a}"),
                       QtWidgets.QMessageBox.StandardButton.Cancel)[1]))
    _use_fake_model(mp)

    win = MainWindow()
    win.resize(1500, 900)
    win.show()
    _pump(qapp, 200)
    doc = _document(tmp)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    _pump(qapp, 120)

    ap = win._analysis_panel
    dp = ap.detection_panel
    dp.set_model_path("/fake/model.keras")
    _pump(qapp, 60)

    runs: dict[str, _Run] = {}

    def _one(kind: str) -> _Run:
        if kind == "visible":
            x0 = win._trace_view.x_for_index(doc_id, 0, int(VISIBLE_FROM_S * FS))
            x1 = win._trace_view.x_for_index(doc_id, 0, N_SAMPLES - 1)
            win._trace_view.zoom_x_to(x0, x1)
            _pump(qapp, 80)
        dp.ml_range.setCurrentIndex(0 if kind == "visible" else 1)
        _pump(qapp, 40)
        done = []
        conn = win._analysis_controller.finished.connect(lambda *_: done.append(1))
        win._run_ml_detection()
        _until(qapp, lambda: bool(done), 300.0, f"the {kind} ML detection")
        _pump(qapp, 200)
        win._analysis_controller.finished.disconnect(conn)
        res = ap.current_detection()
        assert res is not None, f"no {kind} detection landed"
        prov = dict(res.payload.provenance or {})
        n_acc = int(res.payload.accepted.df.shape[0])
        assert n_acc > 120, f"fixture guard ({kind}): only {n_acc} accepted events"
        win._subpop_window = None
        assert win._act_subpop.isEnabled()
        win._act_subpop.trigger()
        _pump(qapp, 200)
        assert win._subpop_pending, "the wired action did not start a run"
        _until(qapp,
               lambda: win._subpop_window is not None
               and win._subpop_window.report() is not None,
               900.0, f"the {kind} subpopulation report (modals: {boxed})")
        _pump(qapp, 200)
        assert not boxed, f"the wired {kind} run raised a modal: {boxed}"
        run = _Run(win, win._subpop_window.report(), prov)
        print(f"\n[CLOCK/{kind}] span={run.span_s} accepted={n_acc} "
              f"pooled={run.pooled():.4f}")
        print(f"[CLOCK/{kind}] caveats:\n{run.caveats}")
        return run

    runs["whole"] = _one("whole")
    runs["visible"] = _one("visible")
    win._analysis_controller.abort()
    win._controller.abort()
    _pump(qapp, 100)
    yield runs
    win.close()
    _pump(qapp, 50)


# ===========================================================================
# the defect
# ===========================================================================


def test_the_sub_range_run_does_not_print_a_false_non_stationarity_claim(wired):
    """THE REPRODUCTION. Events perfectly evenly spread; the sub-range report claimed
    100 % of them fall in one half."""
    run = wired["visible"]
    t0, t1 = run.span_s
    assert (t1 - t0) < 0.6 * (N_SAMPLES / FS), "fixture guard: the span is not a sub-range"
    note = run.caveat("THE RECORDING IS NOT STATIONARY")
    assert not note, f"the app printed a false claim about evenly-spread events: {note}"


def test_pooled_early_frac_is_a_measurement_on_the_analysed_span(wired):
    """A perfectly even supply, split on its OWN span, is ~0.50 — measured, not decreed."""
    for kind, lo, hi in (("whole", 0.40, 0.60), ("visible", 0.40, 0.60)):
        p = wired[kind].pooled()
        assert lo <= p <= hi, f"[{kind}] pooled early_frac {p:.4f} outside {lo}-{hi}"


def test_every_gated_event_is_inside_the_span_the_halves_were_cut_from(wired):
    """The invariant itself: the clock the report split is the span it describes."""
    for kind in ("whole", "visible"):
        run = wired[kind]
        t0, t1 = run.span_s
        feats = run.report.features
        onset = feats["onset_s"].to_numpy(dtype=float)
        early = feats["time_half"].to_numpy() == "early"
        mid = 0.5 * (t0 + t1)
        assert onset.min() >= t0 - 1e-9 and onset.max() <= t1 + 1e-9, (
            f"[{kind}] onsets {onset.min():.3f}-{onset.max():.3f} outside span {t0}-{t1}")
        assert np.array_equal(early, onset <= mid), (
            f"[{kind}] time_half was not cut at the analysed span's midpoint {mid:.3f}")


def test_the_two_runs_have_different_clocks_and_each_uses_its_own(wired):
    """The step past a single run: the same recording, two spans, two midpoints."""
    w, v = wired["whole"], wired["visible"]
    assert w.span_s != v.span_s, "fixture guard: both runs analysed the same span"
    mids = [0.5 * (r.span_s[0] + r.span_s[1]) for r in (w, v)]
    for run, mid in zip((w, v), mids):
        onset = run.report.features["onset_s"].to_numpy(dtype=float)
        early = run.report.features["time_half"].to_numpy() == "early"
        assert np.array_equal(early, onset <= mid)
    assert abs(mids[0] - mids[1]) > 1.0, "fixture guard: the two midpoints coincide"
