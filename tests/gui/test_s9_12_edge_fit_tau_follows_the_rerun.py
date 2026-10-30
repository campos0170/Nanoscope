"""S9-12 — The edge fit uses the open-pore tau of the table now shown, not of the first run it saw.

MEASURED DEFECT (sweep 9, a synthetic 8 s / 50 kHz trace, nine 300 pA events the detector sees and
twenty-seven short 3.75-sigma events a person adds by hand). ``AnalysisPanel._open_pore_tau``
cached a detector table's tau under ``(doc_id, channel, module_id, fs, baseline_fingerprint)``. The
baseline fingerprint names the open-pore ESTIMATOR, which a filter change does not touch (the
10 kHz run and the unfiltered one carried the same ``sha256:8e331d27a59a``), and only closing the
recording cleared the cache. Detect with a 10 kHz low-pass, press S once, re-run with the low-pass
off: C, S, F and the draw fit kept dividing their z by the 10 kHz table's tau, 3.598, where the
unfiltered table's own is 1.000, and S / F then left 29 of 54 edges unfitted against 27 in a window
that only ever saw the unfiltered table ("the step is not significant against the noise"). After a
re-run at 2 kHz the fit, and the ``fit_json`` stored with the row, used 3.598 under a table whose
low-pass floor is 16.81, though the design says "the fit floors its tau at the detector's low-pass
autocorrelation time when the table has one" (CURSOR_ERGONOMICS_DESIGN section 14).

THE FIX. The cache is keyed on the table's own RUN: beside the estimator, the digest of the signal
chain the tau is measured on (``modules.detection_signal_key``: the low-pass cutoff, kind and order
and the rest of the chain) and the run stamp ``detected_at``, which every run writes anew and every
add or merge keeps. A re-run with another low-pass, or none, measures its own tau with its own
floor; an add, which installs a new table object of the same run, still finds the tau it had.

These tests drive real MainWindows: the Detection tab's low-pass box and cutoff, Analysis ▸ Run
detection on the worker, and the trace keys' own slots (S, F, D) with the pointer stood in for by
``TraceView.pointer_sample``, as in ``test_trace_keys``. Synthetic only.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.cusum import filter_autocorr_time
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
import nanoscope.gui.analysis_panel as ap
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 400_000
#: What the detector sees: nine unmistakable blockades.
BIG = [(10_000 + 40_000 * k, 10_300 + 40_000 * k) for k in range(9)]
#: What a person adds by hand: short 3.75-sigma events between them.
WEAK = [(25_000 + 40_000 * k + 997 * j, 25_000 + 40_000 * k + 997 * j + length)
        for k in range(9) for j, length in enumerate((12, 16, 20))]
#: A stretch of open pore, far from both kinds of event: somewhere an add cannot overlap a row.
QUIET = (31_000, 31_100)


def _signal() -> np.ndarray:
    rng = np.random.default_rng(5)
    x = -1000.0 + rng.normal(0.0, 8.0, N)
    for s, e in BIG:
        x[s:e] += 300.0
    for s, e in WEAK:
        x[s:e] += 30.0
    return x.astype(np.float32)


def _window(qtbot, tmp_path):
    win = MainWindow()
    qtbot.addWidget(win)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "weak.dat", format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(_signal(),)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    return win, doc_id


def _run(win, qtbot, cutoff: float | None) -> dict:
    """Analysis ▸ Run detection with the Detection tab's low-pass ticked at ``cutoff`` Hz, or
    unticked; the provenance of the table it installed."""
    form = win._analysis_panel._detection
    form.min_dwell_s.setValue(1e-4)
    form.filter_enabled.setChecked(cutoff is not None)
    if cutoff is not None:
        form.filter_cutoff_hz.setValue(float(cutoff))
    assert win._act_run_detection.isEnabled(), "the menu item would swallow the click"
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._act_run_detection.trigger()
    qtbot.wait(30)
    res = win._analysis_panel.current_detection()
    assert res is not None and res.module_id == "detect.schmitt"
    return dict(res.payload.provenance)


def _point(win, doc_id, sample, px=1.0) -> None:
    win._trace_view.pointer_sample = lambda: (doc_id, 0, float(sample), float(px))


def _s_then_f(win, doc_id):
    """S on each weak event's start, F on its end, as a person does: each edge's source and the
    tau the start's fit recorded."""
    keys = win._trace_keys
    out = []
    for s, e in WEAK:
        keys.discard()
        keys._reset()
        _point(win, doc_id, s)
        keys.place_edge("start", fit=True)
        start_source, start_json = keys._start.source, keys._fit_json
        _point(win, doc_id, e)
        keys.place_edge("end", fit=True)
        out.append((start_source, keys._end.source, json.loads(start_json)["tau"]))
    keys.discard()
    keys._reset()
    return out


def _tau(win, doc_id) -> float:
    return win._analysis_panel.fit_context(doc_id, 0).tau


# ================================================================================================
# the measured cases
# ================================================================================================
def test_a_rerun_without_the_low_pass_fits_like_a_window_that_only_saw_that_run(
        qapp, qtbot, tmp_path):
    fresh, fresh_doc = _window(qtbot, tmp_path)
    _run(fresh, qtbot, None)
    expected = _s_then_f(fresh, fresh_doc)
    fresh_tau = _tau(fresh, fresh_doc)

    win, doc_id = _window(qtbot, tmp_path)
    first = _run(win, qtbot, 10_000.0)
    _point(win, doc_id, WEAK[0][0])
    win._trace_keys.place_edge("start", fit=True)          # S once, on the 10 kHz table
    tau_10k = _tau(win, doc_id)
    win._trace_keys.discard()
    win._trace_keys._reset()
    second = _run(win, qtbot, None)
    assert second.get("filter_cutoff_hz") is None
    assert second.get("baseline_fingerprint") == first.get("baseline_fingerprint"), (
        "precondition: the re-run keeps the estimator, which is what the old key stopped at")

    got = _s_then_f(win, doc_id)

    assert tau_10k > fresh_tau + 1.0, "precondition: the two runs' taus differ"
    assert _tau(win, doc_id) == pytest.approx(fresh_tau)
    assert [row[2] for row in got] == pytest.approx([row[2] for row in expected])
    assert [row[:2] for row in got] == [row[:2] for row in expected]


def test_a_rerun_at_a_slower_low_pass_floors_the_tau_at_that_filter(qapp, qtbot, tmp_path):
    win, doc_id = _window(qtbot, tmp_path)
    _run(win, qtbot, 10_000.0)
    _point(win, doc_id, WEAK[0][0])
    win._trace_keys.place_edge("start", fit=True)
    win._trace_keys.discard()
    win._trace_keys._reset()
    prov = _run(win, qtbot, 2_000.0)
    floor = filter_autocorr_time(FS, float(prov["filter_cutoff_hz"]),
                                 order=int(prov.get("filter_order") or 8),
                                 kind=str(prov.get("filter_kind") or "bessel"))

    _point(win, doc_id, WEAK[1][0])
    win._trace_keys.place_edge("start", fit=True)

    stored = json.loads(win._trace_keys._fit_json)["tau"]
    assert stored >= floor - 1e-6, (
        f"fit_json tau {stored} under a 2 kHz table whose low-pass floor is {floor:.2f}")
    assert _tau(win, doc_id) == pytest.approx(stored, abs=1e-3)      # fit_json rounds it


# ================================================================================================
# what the key must keep: one measurement per run, not one per edit
# ================================================================================================
def test_the_tau_is_measured_once_per_run_and_an_add_does_not_measure_it_again(
        qapp, qtbot, tmp_path, monkeypatch):
    """S measures the run's tau. Shift+S, Shift+F and D then put a hand-made event in the table,
    which installs a new table object of the SAME run, and the next S reuses the tau. A re-run,
    even with the same settings, is another run and measures its own."""
    calls = []
    real = ap.estimate_open_pore_tau
    monkeypatch.setattr(ap, "estimate_open_pore_tau",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    win, doc_id = _window(qtbot, tmp_path)
    keys = win._trace_keys
    _run(win, qtbot, 10_000.0)
    _point(win, doc_id, WEAK[0][0])
    keys.place_edge("start", fit=True)                      # S
    assert len(calls) == 1
    keys.discard()
    keys._reset()
    _point(win, doc_id, QUIET[0])
    keys.place_edge("start", fit=False)                     # Shift+S
    _point(win, doc_id, QUIET[1])
    keys.place_edge("end", fit=False)                       # Shift+F
    keys.add()                                              # D
    QtWidgets.QApplication.processEvents()
    assert win._analysis_panel.last_add_outcome.ok, win.statusBar().currentMessage()
    _point(win, doc_id, WEAK[5][0])
    keys.place_edge("start", fit=True)                      # S
    assert len(calls) == 1, "the add's new table object is the same run: same tau"

    keys.discard()
    keys._reset()
    _run(win, qtbot, 10_000.0)
    _point(win, doc_id, WEAK[6][0])
    keys.place_edge("start", fit=True)
    assert len(calls) == 2, "a re-run measures its own tau"
