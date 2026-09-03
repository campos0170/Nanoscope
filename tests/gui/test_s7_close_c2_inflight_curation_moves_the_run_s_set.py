"""Sweep 7 CLOSE — C2's standing line must also catch the curation made WHILE the
mixture is in flight.

C2's first repair made the report window's standing line compare the accepted SET with
the run's instead of comparing counts, and that closed the swap-after-the-run case.
The re-verifier's own probes then measured the hole it did not close
(``scratchpad/reverify/C2/probe_R1_inflight_curation.py`` and ``probe_R3_…``, on the
owner's ABF 0004): the two halves of the claim were snapshotted at DIFFERENT INSTANTS.

* ``report.n_fired`` counts the rows the job was HANDED — ``events.accepted``, a
  ``.copy()`` taken in :meth:`MainWindow._run_subpopulations` at LAUNCH.
* ``_subpop_accepted_ids`` was rebuilt from the LIVE frame at ADOPTION, a minute or many
  later.

A mixture takes that minute, the status line invites the user to keep working and nothing
gates the File dock while it runs — so a curation made in flight lands INSIDE the
adoption-time snapshot, and the line then compared the current set with itself. Measured
on 0004 before this change: reject one event while the run flies and the report lands with
``n_fired 110``, 109 accepted, and the standing line ``''`` — the count predicate the
first repair replaced would have caught exactly that one ("accepted now: 109 of 110").

Both halves are now taken at the same instant: the SET is snapshotted at launch off the
very ``EventSet`` the job is handed, and is only installed if its size is the count the
report printed — so a set that is not the run's can never be the thing the line compares
against, and a snapshot that cannot be squared with ``n_fired`` degrades to the count
rather than to silence.

Driven through the real doors: Analysis ▸ Event subpopulations (the menu action), a REAL
off-thread run of the real pipeline (only the CV grid is the cheap one, and the worker is
held at the gate so "in flight" is a fact and not a race), and the events table's own
checkbox through its model.

MEASURED BEFORE (this tree with a pre-change copy of ``main_window.py``, run from a
neutral cwd on PYTHONPATH so nothing was edited back and forth): the two in-flight tests
fail with the standing line ``''``; the control passes there and here.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.modules import register_all
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

SCHMITT = "detect.schmitt"

FS = 100_000.0
N_SAMPLES = 420_000
N_EVENTS = 100
_SPACING = 4_000


@pytest.fixture(autouse=True)
def _registered():
    register_all()
    yield


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    """This module's own QSettings namespace — nothing here reads or leaves app state."""
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("S7CloseC2Inflight")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


# ================================================================================================
# builders — the subpopulation suite's synthetic recording, verbatim in shape: events of
# different dwells and depths so the mixture stage has something to fit
# ================================================================================================

def _trace(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N_SAMPLES, 100.0) + rng.normal(0.0, 1.5, N_SAMPLES)
    for i in range(N_EVENTS):
        s = 2_000 + i * _SPACING
        length = int(rng.integers(60, 600))
        x[s:s + length] -= float(rng.uniform(15.0, 45.0))
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=path, format_name="synthetic", n_samples=arr.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    d = tmp_path_factory.mktemp("s7_close_c2")
    p = d / "a.csv"
    return p, _trace(3)


class _Cheap(SR.SubpopConfig):
    """The same pipeline over a small grid — see ``test_subpop_gui._Cheap``."""

    def __init__(self, **kw):
        kw.setdefault("k_grid", (1, 2, 3))
        kw.setdefault("n_splits", 2)
        kw.setdefault("n_boot", 0)
        super().__init__(**kw)


@pytest.fixture
def win(qapp, qtbot, recording, monkeypatch):
    """A window with a REAL Schmitt detection on screen, and the mixture made cheap.

    ``_run_subpopulations`` imports both names from :mod:`nanoscope.analysis.subpop_report`
    at call time, so patching the module is what the launch actually sees.
    """
    monkeypatch.setattr(SR, "SubpopConfig", _Cheap)
    path, arr = recording
    w = MainWindow()
    qtbot.addWidget(w)
    doc_id = w._session.add_document(_doc(path, arr))
    w._create_default_layers(w._session.document(doc_id))
    w._on_selection(doc_id, 0)
    w._analysis_panel._detection.set_detector(SCHMITT)
    w._analysis_panel._detection.min_dwell_s.setValue(3e-4)
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=60_000):
        w._analysis_panel.trigger_detection()
    qtbot.wait(20)
    assert w._session.results.get(doc_id, 0, SCHMITT) is not None
    yield w, doc_id
    w._analysis_controller.abort()
    w._controller.abort()


# ================================================================================================
# doors
# ================================================================================================

def _frame(w, doc_id) -> pd.DataFrame:
    return w._session.results.get(doc_id, 0, SCHMITT).payload.df


def _event_id_at(w, doc_id, row: int) -> int:
    return int(_frame(w, doc_id)["event_id"].to_numpy()[row])


def _checkbox(w, doc_id, event_id: int, accepted: bool) -> None:
    """The events table's own checkbox, driven through the model the view edits."""
    df = _frame(w, doc_id)
    model = w._analysis_panel._table._model
    row = int(np.flatnonzero(df["event_id"].to_numpy() == int(event_id))[0])
    col = int(df.columns.get_loc("accepted"))
    state = (QtCore.Qt.CheckState.Checked if accepted
             else QtCore.Qt.CheckState.Unchecked)
    assert model.setData(model.index(row, col), state.value,
                         QtCore.Qt.ItemDataRole.CheckStateRole), (
        "the events table refused the curation this test is about")
    assert bool(_frame(w, doc_id)["accepted"].to_numpy()[row]) is bool(accepted), (
        "the checkbox did not move the accepted flag")


def _accepted_ids(w, doc_id) -> set[int]:
    df = _frame(w, doc_id)
    return {int(e) for e in df.loc[df["accepted"].to_numpy().astype(bool), "event_id"]}


def _launch_held(w, qtbot, monkeypatch) -> threading.Event:
    """Analysis ▸ Event subpopulations, with the worker held at the door.

    The run has to be genuinely IN FLIGHT while the curation below happens — the whole
    point of the defect — and a sleep would be a race. The real pipeline still runs; it
    only starts when this test says so.
    """
    gate = threading.Event()
    real = SR.run_subpopulations

    def held(*a, **kw):
        assert gate.wait(timeout=120), "the test never released the worker"
        return real(*a, **kw)

    monkeypatch.setattr(SR, "run_subpopulations", held)
    w._act_subpop.trigger()
    qtbot.wait(30)
    assert w._subpop_pending, "Analysis ▸ Event subpopulations did not start a run"
    return gate


def _land(w, qtbot, gate: threading.Event) -> None:
    # The worker is released INSIDE the wait, never before it: released first, a fast run
    # could finish before anything was listening and this would hang out its whole timeout.
    with qtbot.waitSignal(w._analysis_controller.finished, timeout=180_000):
        gate.set()
    qtbot.waitUntil(lambda: not w._subpop_pending, timeout=10_000)
    qtbot.wait(40)
    assert w._subpop_window is not None, "the report never landed"
    assert w._subpop_assignment, "the run produced no assignment to stand on"


# ================================================================================================
# the residue: the accepted set moved while the report was being computed
# ================================================================================================

def test_a_rejection_made_while_the_run_was_in_flight_is_disclosed_when_it_lands(
        win, qtbot, monkeypatch):
    w, doc_id = win
    n_rows = len(_frame(w, doc_id))
    at_launch = _accepted_ids(w, doc_id)

    gate = _launch_held(w, qtbot, monkeypatch)
    victim = _event_id_at(w, doc_id, 3)
    _checkbox(w, doc_id, victim, False)
    assert w._subpop_pending, "the curation was not made while the run was in flight"
    _land(w, qtbot, gate)

    report = w._subpop_window.report()
    assert int(report.n_fired) == len(at_launch), (
        "the fixture did not reproduce the defect's shape: the report must describe the "
        "set as it was at LAUNCH")
    assert _accepted_ids(w, doc_id) != at_launch, "the in-flight curation was undone"

    note = w._subpop_curation_note()
    assert note, (
        f"a rejection made while the run flew was not disclosed at all: the report "
        f"describes {report.n_fired} accepted event(s) and "
        f"{len(_accepted_ids(w, doc_id))} are accepted now")
    assert note == (
        f"accepted now: {len(at_launch) - 1} of {n_rows}; this run's fractions describe "
        f"the {len(at_launch)} accepted at run time — re-run to update"), note
    assert w._subpop_window.live_note().endswith(note), (
        "the report window is not showing the line")


def test_an_in_flight_swap_that_keeps_the_accepted_count_is_disclosed_too(
        win, qtbot, monkeypatch):
    """The case no count can catch, on the in-flight door — one out, one back in."""
    w, doc_id = win
    n_rows = len(_frame(w, doc_id))
    spare = _event_id_at(w, doc_id, 11)
    _checkbox(w, doc_id, spare, False)          # rejected BEFORE the run, so not in it
    at_launch = _accepted_ids(w, doc_id)

    gate = _launch_held(w, qtbot, monkeypatch)
    victim = _event_id_at(w, doc_id, 3)
    _checkbox(w, doc_id, victim, False)         # ...and the swap, both halves in flight
    _checkbox(w, doc_id, spare, True)
    assert w._subpop_pending, "the swap was not made while the run was in flight"
    _land(w, qtbot, gate)

    now = _accepted_ids(w, doc_id)
    assert len(now) == len(at_launch) == int(w._subpop_window.report().n_fired), (
        "the fixture did not keep the count equal, which is what this test is for")
    assert now != at_launch, "the swap did not move the set"

    note = w._subpop_curation_note()
    assert note, (
        "a swap made while the run flew left the count equal and the line silent — the "
        "run's fractions describe a set that is no longer the accepted one")
    assert note == (
        f"accepted now: {len(now)} of {n_rows}; this run's fractions describe "
        f"the {len(at_launch)} accepted at run time — re-run to update"), note


def test_a_run_nobody_touched_while_it_flew_lands_with_nothing_to_disclose(
        win, qtbot, monkeypatch):
    """The control: the line exists to fire on a moved set, not on the passage of time."""
    w, doc_id = win
    at_launch = _accepted_ids(w, doc_id)
    gate = _launch_held(w, qtbot, monkeypatch)
    _land(w, qtbot, gate)

    assert _accepted_ids(w, doc_id) == at_launch
    assert w._subpop_accepted_ids == frozenset(at_launch), (
        "the run's own set was not the one snapshotted")
    assert w._subpop_curation_note() == "", (
        "an untouched run was called stale the moment it landed")
