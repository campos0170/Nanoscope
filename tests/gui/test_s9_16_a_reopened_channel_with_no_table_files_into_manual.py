"""S9-16 — After File ▸ Open Project, a channel with no table still files a hand-made event into
the Manual table, which the add creates.

MEASURED DEFECT (found while integrating the sweep-9 fixes; the S9-01 fix caused it). S9-01 made
``AnalysisPanel._set_active_detection`` write the selected channel's remembered table, so that
the remembered table follows the table on screen. The project restore
(``analysis_workspace.restore_analysis_panel``) also calls it, with the panel-wide
``active_detection`` the project saved (``detect.schmitt`` by default, or the last table shown on
another recording), without putting any table on screen. A channel with no table therefore
reopened remembering Schmitt: Ctrl+M said "Nothing was added: Schmitt has no stored table on this
channel. Show a table, or the Manual table.", C, S and F refused the same way, and the next save
wrote the phantom choice into ``event_table_choices_v1``. On e2c92cb the reopened channel files
"Added event 0 to Manual." Design 4.2: a channel with no table "adds to the Manual table, creating
it", and where Ctrl+M files must not depend on a side effect such as a reopen.

THE FIX. ``_set_active_detection`` records the choice only when the channel shown holds that
module's table, so the remembered table is always one that is (or was) on screen.

These tests drive the real save (``MainWindow._write_project``) and the real reopen
(``MainWindow._open_project`` until ``projectRestored``), then what a person does next: Ctrl+M,
the C and D keys, and another save. Synthetic recordings only (a CSV written to pytest's
tmp_path).
"""


from __future__ import annotations

import inspect
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import MANUAL_MODULE_ID, EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import load_project
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.analysis_panel import AnalysisPanel
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
_DETECTED = [(10_000, 10_300), (30_000, 30_300)]
_MISSED = (20_000, 20_300)
SCHMITT = "detect.schmitt"
NO_TABLE = "has no stored table on this channel"
CHOICES = "event_table_choices_v1"


def _signal(spans, *, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in spans:
        x[s:e] += DEPTH
    return x.astype(np.float32)


def _settle() -> None:
    QtWidgets.QApplication.processEvents()


def _recording(win: MainWindow, tmp_path: Path, name: str) -> tuple[str, Path]:
    """A CSV on disk (so the project can reopen it), opened as a document and selected."""
    arr = _signal(_DETECTED + [_MISSED])
    csv = tmp_path / name
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": arr.astype(np.float64)}).to_csv(csv, index=False)
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=csv, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, t0_s=0.0, channels=(info,))
    doc_id = win._session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    _settle()
    return doc_id, csv


def _schmitt_lands(win: MainWindow, doc_id: str) -> None:
    """A Schmitt run's table installed on the channel, the way a finished run installs it."""
    df, prov = detect_events(_signal(_DETECTED, seed=1), FS, baseline=I0, min_dwell_s=1e-4)
    win._session.add_result(AnalysisResult(
        module_id=SCHMITT, module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    _settle()


@pytest.fixture
def spy(monkeypatch):
    """Every ``_set_active_detection`` call: the module, whether the selected channel holds that
    table, and the caller (file:line), so a failure names who recorded the phantom choice."""
    calls: list[tuple[str, bool, str]] = []
    original = AnalysisPanel._set_active_detection

    def recording(self, module_id):
        sel = self._sel
        has = bool(sel is not None and self._session.has_document(sel[0])
                   and self._session.results.get(sel[0], sel[1], module_id) is not None)
        frame = inspect.stack()[1]
        calls.append((module_id, has, f"{Path(frame.filename).name}:{frame.lineno}"))
        return original(self, module_id)

    monkeypatch.setattr(AnalysisPanel, "_set_active_detection", recording)
    return calls


def _reopen(qtbot, proj: Path) -> MainWindow:
    """File > Open Project after its dialog: the real restore, until it says it has settled."""
    w = MainWindow()
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(50)
    return w


def _live(w: MainWindow, csv: Path) -> str:
    live = w._session.doc_id_for_source(str(csv.resolve()))
    assert w._analysis_panel.current_target() == (live, 0), w._analysis_panel.current_target()
    return live


def _state(w: MainWindow, doc_id: str) -> dict:
    p = w._analysis_panel
    return {"target": p.target_module_id(doc_id, 0),
            "trace label": p.target_table_name(doc_id, 0),
            "remembered": p.chosen_tables().get((doc_id, 0)),
            "selector": p._result_selector.currentText()}


def _manual_rows(w: MainWindow, doc_id: str) -> int:
    res = w._session.results.get(doc_id, 0, MANUAL_MODULE_ID)
    return 0 if res is None else len(res.payload.df)


def _ctrl_m(w: MainWindow, span) -> str:
    """Measure on, a region over ``span``, Analysis > Add event from cursors (Ctrl+M)."""
    w._analysis_panel.measure_panel.set_active(True)
    w._trace_view.set_measure_region(span[0] / FS, span[1] / FS)
    _settle()
    w._act_add_event.trigger()
    _settle()
    return w.statusBar().currentMessage()


def test_a_recording_with_no_table_reopens_still_filing_ctrl_m_into_manual(
        qapp, qtbot, tmp_path, spy):
    """Open a recording, detect nothing, save; reopen. The channel, the Table selector and the
    trace label are as they were, and Ctrl+M starts the Manual table as it did before the save."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc_id, csv = _recording(win, tmp_path, "rec_a.csv")
    before = _state(win, doc_id)
    assert before["target"] == MANUAL_MODULE_ID and before["remembered"] is None, before
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))

    spy.clear()
    w2 = _reopen(qtbot, proj)
    live = _live(w2, csv)
    after = _state(w2, live)
    unbacked = [c for c in spy if not c[1]]
    assert after == before, (
        f"before the save {before}; after the reopen {after}; _set_active_detection calls "
        f"naming a table the selected channel does not have: {unbacked}")

    said = _ctrl_m(w2, _MISSED)
    assert _manual_rows(w2, live) == 1 and "to Manual" in said, said


def test_c_then_d_after_the_reopen_catch_the_event_and_file_it_into_manual(qapp, qtbot, tmp_path):
    """The trace keys on the reopened channel: C over the missed event catches it, D files it."""
    win = MainWindow()
    qtbot.addWidget(win)
    _doc_id, csv = _recording(win, tmp_path, "rec_a.csv")
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    live = _live(w2, csv)
    w2._trace_view.pointer_sample = lambda: (live, 0, float(_MISSED[0] + 150), 10.0)
    w2._trace_keys.catch()
    caught = w2.statusBar().currentMessage()
    assert NO_TABLE not in caught and w2._trace_view.measure_bounds() is not None, caught

    w2._trace_keys.add()
    if _manual_rows(w2, live) == 0 and "Press again" in w2.statusBar().currentMessage():
        w2._trace_keys.add()          # an unfitted edge asks for a second press
    said = w2.statusBar().currentMessage()
    assert _manual_rows(w2, live) == 1 and "to Manual" in said, (caught, said)


def test_the_next_save_records_no_table_the_channel_lacks_and_a_second_reopen_files_manual(
        qapp, qtbot, tmp_path):
    """Reopen and save again at once: the project must not now remember Schmitt for a channel
    with no Schmitt table, and the reopen after that still files into Manual."""
    win = MainWindow()
    qtbot.addWidget(win)
    _doc_id, csv = _recording(win, tmp_path, "rec_a.csv")
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))
    first = load_project(proj).workspace.extra.get(CHOICES)

    w2 = _reopen(qtbot, proj)
    _live(w2, csv)
    proj2 = tmp_path / "p2.nanoscope"
    assert w2._write_project(str(proj2))
    second = load_project(proj2).workspace.extra.get(CHOICES)
    phantom = [r for r in (second or []) if isinstance(r, dict) and r.get("module_id") == SCHMITT]
    assert not phantom, f"first save {first!r}; the save after the reopen {second!r}"

    w3 = _reopen(qtbot, proj2)
    live3 = _live(w3, csv)
    assert _state(w3, live3)["target"] == MANUAL_MODULE_ID, _state(w3, live3)


def test_a_recording_with_no_table_saved_on_screen_beside_one_with_schmitt(qapp, qtbot, tmp_path):
    """Recording A has a Schmitt table; the person moves to B, which has none, and saves. After
    the reopen B (on screen again) files Ctrl+M into Manual, and A still files into Schmitt."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc_a, csv_a = _recording(win, tmp_path, "rec_a.csv")
    _schmitt_lands(win, doc_a)
    assert win._analysis_panel.current_detection().module_id == SCHMITT
    doc_b, csv_b = _recording(win, tmp_path, "rec_b.csv")
    before_b = _state(win, doc_b)
    assert before_b["target"] == MANUAL_MODULE_ID, before_b
    proj = tmp_path / "p.nanoscope"
    assert win._write_project(str(proj))

    w2 = _reopen(qtbot, proj)
    live_b = _live(w2, csv_b)
    after_b = _state(w2, live_b)
    said = _ctrl_m(w2, _MISSED)
    assert _manual_rows(w2, live_b) == 1 and "to Manual" in said, (before_b, after_b, said)

    live_a = w2._session.doc_id_for_source(str(csv_a.resolve()))
    w2._on_selection(live_a, 0)
    _settle()
    assert w2._analysis_panel.target_module_id(live_a, 0) == SCHMITT
