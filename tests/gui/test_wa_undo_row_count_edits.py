"""WA-UNDO-1 (wired) — CTRL+Z DID NOT UNDO AN ADD OR A MERGE, AND THE MENU SAID IT DID.

Measured on the -1000 mV apo-ferritin recording through a real ``MainWindow`` at 1512x900:
"Mark selection as missed event" took the table 115 → 116 rows, Edit read **"Undo Add
event", enabled** — and triggering it left 116 rows with the added row still there, after
which Redo read "nothing to redo". "Merge all" (116 → 88) behaved the same way. A plain
reject, in the same session, undid perfectly.

The cause was not the stack. A hand-added row has no detector bounds, so
``orig_start_idx``/``orig_end_idx`` are ``pd.NA`` in a masked ``Int64`` column, and
``core.history._column_equal`` sent masked dtypes down its integer fast path, where
``to_numpy()`` had already turned the null into ``NaN`` and ``NaN == NaN`` is False. Every
later checkpoint therefore saw a mutation that never happened and pushed a phantom step
whose "pre-edit" snapshot already held the added row. ``EventHistory._move`` checkpoints
before it moves, so each press minted a fresh phantom and undid that instead.
``tests/core/test_wa_history_masked_na.py`` holds the comparison itself; this file holds
what the user actually does.

These tests drive the SHIPPED doors: the Manual measurement tab's "Mark selection as missed
event" button and the panel's merge engine, then MainWindow's own Edit ▸ Undo / Redo
actions — because the defect was invisible to anything that called ``EventHistory``
directly with a frame it had built itself.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet, stamp_original_bounds
from nanoscope.analysis.scoring import ADDED_COLUMN
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
#: Five well-separated events plus a close PAIR the merge engine can collapse — and one
#: stretch (58 000-58 600) left undetected on purpose, for the hand-add to bracket.
_SPANS = [(5_000, 5_400), (12_000, 12_500), (20_000, 20_600),
          (30_000, 30_500), (40_000, 40_400), (40_450, 40_900)]
_MISSED = (58_000, 58_600)


def _signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for s, e in _SPANS:
        x[s:e] -= DEPTH
    # The one the detector will not be asked to find: a shallow dip the hand-add brackets.
    x[_MISSED[0]:_MISSED[1]] -= DEPTH
    return np.ascontiguousarray(x, dtype=np.float32)


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


@pytest.fixture
def detected(window, tmp_path):
    """One recording, one stored detection that deliberately STOPS before ``_MISSED``."""
    sig = _signal()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "wa_rows.dat", format_name="synthetic",
                         n_samples=int(sig.size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    doc = TraceDocument(meta=meta, channels=(sig,))
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    # Detect over the first 55 000 samples only, so `_MISSED` is a genuine detector miss.
    df, prov = detect_events(sig[:55_000], FS, baseline=I0, min_dwell_s=1e-4)
    prov = dict(prov)
    prov.update(span_start=0, span_stop=55_000)
    window._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={"detector": "detect.schmitt", "min_dwell_s": 1e-4},
        calibration_fingerprint=window._session.calibration(doc_id, 0).fingerprint(),
        # `stamp_original_bounds` exactly as `analysis/modules.py` does when the real
        # detector installs a run: the two Int64 columns it writes are what a hand-added
        # row leaves as `pd.NA`, and therefore what the defect lived in.
        payload=EventSet(stamp_original_bounds(df), prov)))
    return window, doc_id, sig


def _rows(window) -> int:
    return len(window._analysis_panel.current_detection().payload.df)


def _n_added(window) -> int:
    df = window._analysis_panel.current_detection().payload.df
    return int(df[ADDED_COLUMN].to_numpy(dtype=bool).sum()) if ADDED_COLUMN in df else 0


def _hand_add(window, qtbot) -> int | None:
    """Bracket ``_MISSED`` with the Measure cursors and press the panel's own add.

    Through :meth:`AnalysisPanel.add_current_selection_as_event` — the method the "Mark
    selection as missed event" button is wired to — with the cursors set on the trace view
    exactly as the button requires (it refuses a selection that is not on this recording).
    """
    window._act_measure.trigger()
    qtbot.wait(20)
    view = window._trace_view
    layer = window._session.layer(view._measure_layer_id)
    xs = window._session.x_axis(layer.doc_id, layer.time_offset_s).values
    view.set_measure_region(float(xs[_MISSED[0] - 40]), float(xs[_MISSED[1] + 40]))
    qtbot.wait(20)
    return window._analysis_panel.add_current_selection_as_event()


def _sync_menu(window):
    """What Edit ▸ Undo / Redo would read if the user opened the menu right now."""
    window._checkpoint_everything()
    window._sync_undo_actions()
    return window._act_undo.text(), window._act_undo.isEnabled(), \
        window._act_redo.text(), window._act_redo.isEnabled()


# ================================================================================================
# the add
# ================================================================================================

def test_undo_of_a_hand_added_event_removes_the_row_it_named(detected, qtbot):
    """116 rows in, 115 rows out. The measured headline, in the wired app."""
    window, _doc_id, _sig = detected
    before = _rows(window)
    assert _hand_add(window, qtbot) is not None, "the hand-add itself did not happen"
    qtbot.wait(20)
    assert _rows(window) == before + 1
    assert _n_added(window) == 1

    undo_text, undo_on, _redo_text, _redo_on = _sync_menu(window)
    assert undo_on and undo_text == "Undo Add event", undo_text

    window._undo_events()
    qtbot.wait(20)
    assert _rows(window) == before, "Edit ▸ Undo left the hand-added row in the table"
    assert _n_added(window) == 0


def test_redo_puts_the_hand_added_event_back(detected, qtbot):
    """Redo used to read "nothing to redo" because a phantom consumed the press."""
    window, _doc_id, _sig = detected
    before = _rows(window)
    _hand_add(window, qtbot)
    qtbot.wait(20)
    window._undo_events()
    qtbot.wait(20)

    _u, _uo, redo_text, redo_on = _sync_menu(window)
    assert redo_on and redo_text == "Redo Add event", redo_text
    window._redo_events()
    qtbot.wait(20)
    assert _rows(window) == before + 1
    assert _n_added(window) == 1


def test_opening_the_edit_menu_after_an_add_invents_no_step(detected, qtbot):
    """``checkpoint_all`` runs on every Edit-menu open, a save and a quit.

    Each one used to push a phantom over a table nobody had touched, which is both how the
    undo was consumed and how a just-saved project came back dirty.
    """
    window, _doc_id, _sig = detected
    _hand_add(window, qtbot)
    qtbot.wait(20)
    depth = window._history.depth()
    for _ in range(3):
        window._checkpoint_everything()
    assert window._history.depth() == depth, (
        "opening the Edit menu pushed steps for edits that never happened")


def test_a_reject_after_an_add_undoes_the_reject_and_then_the_add(detected, qtbot):
    """Two real edits, two presses. The control experiment and the defect, in one session."""
    window, _doc_id, _sig = detected
    before = _rows(window)
    _hand_add(window, qtbot)
    qtbot.wait(20)
    panel = window._analysis_panel
    df = panel.current_detection().payload.df
    panel._table.set_accepted_by_id(int(df["event_id"].iloc[0]), False)
    qtbot.wait(20)
    window._checkpoint_current()

    window._undo_events()
    qtbot.wait(20)
    df = panel.current_detection().payload.df
    assert len(df) == before + 1 and bool(df["accepted"].to_numpy(bool).all()), (
        "the first press did not undo the rejection")

    window._undo_events()
    qtbot.wait(20)
    assert _rows(window) == before, "the second press never reached the add"


# ================================================================================================
# the merge
# ================================================================================================

def test_undo_of_a_merge_restores_every_row_it_collapsed(detected, qtbot):
    """The other row-count-changing edit, through the panel's own merge engine.

    Driven after a hand-add on purpose: the ``pd.NA`` the add leaves behind is what
    poisoned the comparison, so a merge in a table that holds one is the case that failed.
    """
    window, _doc_id, _sig = detected
    _hand_add(window, qtbot)
    qtbot.wait(20)
    panel = window._analysis_panel
    before = _rows(window)

    panel.merge_all_subevents(0.005)
    qtbot.waitUntil(lambda: _rows(window) != before, timeout=60_000)
    qtbot.wait(50)
    merged = _rows(window)
    assert merged < before

    undo_text, undo_on, _rt, _ro = _sync_menu(window)
    assert undo_on and undo_text.startswith("Undo Merge events"), undo_text
    window._undo_events()
    qtbot.wait(50)
    assert _rows(window) == before, "Edit ▸ Undo left the merged table on screen"

    window._redo_events()
    qtbot.wait(50)
    assert _rows(window) == merged
