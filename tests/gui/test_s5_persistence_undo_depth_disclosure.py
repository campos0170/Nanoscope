"""IF-03 — the DEPTH ceiling's losses are disclosed like the byte ceiling's.

``core/history.py`` bounds one table's stack two ways, and until this fix only one of them
could speak: the MAX_BYTES eviction counted itself (``_evicted``) and announced itself
(``stepsEvicted``), while the MAX_DEPTH branch of ``_trim`` silently discarded the oldest
steps. Measured through a real MainWindow: 50 curation gestures, Ctrl+Z walks back exactly
30, 20 events stay rejected — and the Edit menu read *"Undo — nothing to undo on this
event table"*, the wording ``evicted()``'s own contract reserves for a table whose history
the app never touched ("the two states look identical to the user and mean opposite
things"). The MAX_DEPTH docstring calls the identical prior outcome "a data-loss bug, not
a policy".

Now the depth branch counts (``depth_dropped``), announces (``stepsDepthDropped``), and
the menu names the ceiling that actually bound — each limit refuses by its own name, so
the byte wording ("memory ceiling", pinned by test_s3_dl_history_scope) is untouched.
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
from nanoscope.core.history import MAX_DEPTH
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
_SPANS = [(1_000 + i * 800, 1_400 + i * 800) for i in range(40)]


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
def one(qapp, qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    doc = _doc(tmp_path, "pore_A.dat")
    doc_id = w._session.add_document(doc)
    w._create_default_layers(doc)
    w._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(), FS, baseline=I0, min_dwell_s=1e-4)
    assert len(df) > MAX_DEPTH + 5, "the fixture needs more events than the depth cap"
    w._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"min_dwell_s": 1e-4},
        calibration_fingerprint=w._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    yield w, doc_id
    w._analysis_controller.abort()


def _frame(w, doc_id):
    return w._session.results.get(doc_id, 0, "detect.schmitt").payload.df


def _reject_row(w, doc_id, row: int) -> None:
    df = _frame(w, doc_id)
    w._analysis_panel._table.set_accepted_by_id(int(df["event_id"].iloc[row]), False)
    w._history.checkpoint_all()


def _key(doc_id):
    return (doc_id, 0, "detect.schmitt")


def _sprint(w, doc_id, n: int) -> None:
    for row in range(n):
        _reject_row(w, doc_id, row)


def test_the_menu_at_the_undo_floor_names_the_depth_ceiling(one):
    w, a = one
    n = MAX_DEPTH + 8
    _sprint(w, a, n)

    undone = 0
    while w._history.can_undo(_key(a)):
        w._undo_events()
        undone += 1
    assert undone == MAX_DEPTH, "the stack walks back exactly the cap"

    still_rejected = int((~_frame(w, a)["accepted"].to_numpy(dtype=bool)).sum())
    assert still_rejected == n - MAX_DEPTH, "the overflow steps are genuinely lost"

    w._sync_undo_actions()
    text = w._act_undo.text()
    assert "nothing to undo on this event table" not in text, (
        f"the never-edited wording must not cover a loss: {text}")
    assert f"{MAX_DEPTH}-step ceiling" in text, text
    assert f"{n - MAX_DEPTH} steps lost" in text, text


def test_the_counter_and_signal_track_every_depth_drop(one):
    w, a = one
    said: list[tuple[str, int]] = []
    w._history.stepsDepthDropped.connect(lambda d, n: said.append((d, n)))
    n = MAX_DEPTH + 5
    _sprint(w, a, n)
    assert w._history.depth_dropped(_key(a)) == n - MAX_DEPTH
    assert said, "each drop announces itself"
    assert said[-1] == (a, n - MAX_DEPTH)
    assert w._history.evicted(_key(a)) == 0, (
        "depth losses must not masquerade as byte evictions — the wordings differ")


def test_the_status_bar_says_it_while_it_happens(one, qtbot):
    w, a = one
    messages: list[str] = []
    w.statusBar().messageChanged.connect(lambda m: messages.append(m) if m else None)
    _sprint(w, a, MAX_DEPTH + 2)
    qtbot.wait(5)
    assert any(f"{MAX_DEPTH}-step ceiling" in m for m in messages), messages[-4:]

    # BOTH NUMBERS OF THE CLAUSE, spelled. The sentence used to say "N of its oldest
    # step(s) have been dropped", which is ungrammatical on the first drop — the one this
    # toast is most likely to be read on, since it fires the instant the ceiling starts
    # biting — while `_sync_undo_actions` 60 lines away had always pluralised properly.
    # The sprint above crosses the cap twice, so both sentences really are produced here,
    # and the Plots window's standing line borrows them verbatim
    # (test_s6_plots_undo_depth_disclosure::test_the_two_windows_speak_one_dialect).
    ceiling = [m for m in messages if f"{MAX_DEPTH}-step ceiling" in m]
    assert any("its oldest step has been dropped and cannot be undone." in m
               for m in ceiling), ceiling
    assert any("its 2 oldest steps have been dropped and cannot be undone." in m
               for m in ceiling), ceiling
    assert not any("step(s)" in m for m in messages), messages


def test_under_the_cap_the_plain_wording_survives(one):
    """CONTROL: the disclosure must not appear for a table that lost nothing."""
    w, a = one
    _sprint(w, a, 3)
    for _ in range(3):
        w._undo_events()
    w._sync_undo_actions()
    text = w._act_undo.text()
    assert "ceiling" not in text, text
    assert w._history.depth_dropped(_key(a)) == 0


def test_closing_the_recording_takes_its_depth_record_with_it(one):
    w, a = one
    _sprint(w, a, MAX_DEPTH + 3)
    assert w._history.depth_dropped(_key(a)) > 0
    w._history.drop_doc(a)
    assert w._history.depth_dropped(_key(a)) == 0

    _clear_check = w._history
    _clear_check.clear()
    assert _clear_check.depth_dropped(_key(a)) == 0
