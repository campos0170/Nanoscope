"""BL-1 — THE EVENTS TABLE'S PER-ROW ACCEPT IS THE SAME AND AS EVERY OTHER SURFACE.

The gallery and the inspector both derive ``accepted`` (``derived AND requested``, from
:data:`~nanoscope.analysis.events.AUTOMATIC_REJECTION_FLAGS`), refuse an accept of a row
the detector could not measure, and SAY SO. The events table's own per-row path —
``_EventTableModel.setData`` (the checkbox) and ``set_accepted`` (the a/r keys and the
mirror door) — wrote ``requested`` straight over the cell: one tick put ``accepted=True``
beside ``clipped_by_boundary``, past the very rule "Accept all" had just enforced over
the whole frame, with the gallery tile beside it still painted, and captioned, rejected.
That hole was reported (not patched) by tests/test_rk_refusal_counter_wired.py, whose
module docstring names the exact methods; this file pins its closure.

THE FIX THESE TESTS PIN
-----------------------
1. the model's one write (``_curate``) applies the same AND, read from the same named
   set — byte-identical machinery with the two windows (the import fence between gui
   modules forbids sharing it by import; sharing the RULE means naming
   ``AUTOMATIC_REJECTION_FLAGS`` and never restating its tokens);
2. a refused accept writes nothing, announces nothing on ``acceptanceChanged`` (a signal
   with no change behind it reaches the dirty flag and the undo recorder), and is NOT
   silent: it is recorded on the refusal ledger and stated on the table's own
   ``refusal_label`` — the same sentence, from the same ``_refusal_note``, as the
   gallery's status clause and the inspector's batch note;
3. in the wired app the ledger is the ONE SHARED per-lineage record MainWindow keeps
   (see tests/gui/test_bl_refusal_note_lineage.py), so a refusal made at the table is
   disclosed by the gallery and the inspector too — one voice;
4. rejection is never blocked, a hand-rejected ordinary row is re-acceptable, the repair
   verb is "corrected back to rejected", and a frame with no ``flag`` column fails
   CLOSED — all exactly as at the other two surfaces.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from PySide6.QtCore import Qt

import nanoscope.gui.event_gallery as gal_mod
import nanoscope.gui.event_inspector as insp_mod
import nanoscope.gui.main_window as mw_mod
import nanoscope.gui.panels as panels_mod
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
#: Laid out so the three unmeasurable events carry the LOWEST ids (clipped opening, two
#: over the dwell limit), with five ordinary events after them.
_LONG = [(10_000, 20_000), (30_000, 40_000)]
_CLEAN = [(50_000, 50_300), (60_000, 60_250), (70_000, 70_300),
          (90_000, 90_250), (110_000, 110_300)]
_MAX_DWELL_S = 5e-3


# ================================================================================================
# builders — a real recording, a real MainWindow, the real detector
# ================================================================================================
def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    x[0:300] -= 30.0                        # the recording OPENS mid-event -> clipped row 0
    for s, e in _LONG:
        x[s:e] -= 30.0                      # two events over the dwell limit
    for s, e in _CLEAN:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_bl1.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def settings_ns(qapp):
    """A PRIVATE QSettings namespace: this file opens standalone windows and must neither
    read nor leave behind the developer's own saved layout."""
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("NanoScopeBL1TableGuard")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app)


def _drain(qtbot, ms: int = 120) -> None:
    qtbot.wait(ms)
    QtCore.QCoreApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


@pytest.fixture
def wired(qapp, qtbot, tmp_path, settings_ns):
    """A real MainWindow with a real detection — the only configuration in which the
    table's refusal ledger is the one the gallery and inspector share."""
    win = MainWindow()
    qtbot.addWidget(win)
    doc = _doc(tmp_path)
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    panel = win._analysis_panel
    panel._detection.min_dwell_s.setValue(5e-4)
    panel._detection.max_dwell_s.setValue(_MAX_DWELL_S)
    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    _drain(qtbot)
    df = panel.current_detection().payload.df
    unmeasurable = [i for i in range(len(df))
                    if not panels_mod._derived_accepted(df["flag"].iloc[i])]
    assert len(unmeasurable) == 3, (
        "the fixture needs three of the detector's own rejections; got "
        f"{df['flag'].tolist()}")
    assert len(df) - len(unmeasurable) >= 3, "and at least three ordinary rows"
    try:
        yield win, df, unmeasurable
    finally:
        win._analysis_controller.abort()


def _table(win):
    return win._analysis_panel._table


def _checkbox_click(table, df_row: int, checked: bool) -> bool:
    """Tick the REAL checkbox: what the view's delegate does on a click is a
    ``setData(…, CheckStateRole)`` through the sort proxy. Returns setData's verdict."""
    col = list(table.dataframe.columns).index("accepted")
    src = table._model.index(df_row, col)
    pidx = table._proxy.mapFromSource(src)
    assert pidx.isValid()
    return bool(table._proxy.setData(
        pidx, Qt.Checked if checked else Qt.Unchecked, Qt.CheckStateRole))


def _gallery(win, qtbot):
    win._act_event_gallery.trigger()
    _drain(qtbot)
    return win._event_gallery


def _inspector(win, qtbot):
    win._act_event_inspector.trigger()
    _drain(qtbot)
    return win._event_inspector


# ================================================================================================
# 0. provenance — which tree, and the machinery is the windows' machinery, verbatim
# ================================================================================================
def test_the_package_under_test_is_the_one_beside_this_file():
    here = Path(__file__).resolve().parents[2]
    for mod in (nanoscope, panels_mod, gal_mod, insp_mod, mw_mod):
        assert Path(mod.__file__).resolve().is_relative_to(here), \
            f"{mod.__name__} resolved from {mod.__file__}, not from {here}"
    assert callable(panels_mod._EventTableModel.__dict__.get("_curate")), \
        "the model has no single guarded write"
    for name in ("refusal_note", "set_refusals_source"):
        assert callable(getattr(panels_mod.EventTable, name, None)), \
            f"EventTable has no {name}()"


def test_the_tables_machinery_is_byte_identical_with_the_windows():
    """The import fence (MainWindow is the only gui module that may import other panels)
    forbids sharing by import, so the table carries the same three pieces the two windows
    already duplicate — and they must not drift, exactly as the windows' copies must not
    (tests/test_rp_refusal_clear_wired.py pins those two against each other)."""
    for name in ("_derived_accepted", "_refusal_note", "_RefusalLedger"):
        assert inspect.getsource(getattr(panels_mod, name)) == \
            inspect.getsource(getattr(gal_mod, name)), \
            f"panels.{name} drifted from the windows' copy"


# ================================================================================================
# 1. THE HEADLINE — the checkbox cannot accept a row the detector could not measure
# ================================================================================================
def test_the_checkbox_cannot_accept_an_unmeasurable_row_and_says_so(wired, qtbot):
    """One tick on the table used to put ``accepted=True`` beside the rejecting flag —
    the forbidden state, silently. Now: nothing lands, nothing is announced, and the
    refusal is stated on the table's own label in the windows' own sentence."""
    win, df, unmeasurable = wired
    table = _table(win)
    row = unmeasurable[0]
    assert bool(df["accepted"].iloc[row]) is False

    seen: list[tuple[int, bool]] = []
    table.acceptanceChanged.connect(lambda i, a: seen.append((int(i), bool(a))))
    landed = _checkbox_click(table, row, True)
    _drain(qtbot)

    assert landed is False, "setData claimed success for a write it refused"
    assert bool(df["accepted"].iloc[row]) is False, \
        "one tick put a row the detector could not measure into the accepted set"
    assert seen == [], (
        "nothing was written, so nothing may be announced: a signal with no change "
        f"behind it reaches the dirty flag and the undo recorder. {seen}")
    assert "1 event kept rejected: flagged by the detector as not measurable" \
        in table.refusal_label.text(), table.refusal_label.text()
    # ``isHidden``, not ``isVisible``: the fixture's MainWindow is never shown, and an
    # unshown ancestor makes every child report not-visible. Hidden-ness is the label's
    # own state, which is what the fix controls.
    assert not table.refusal_label.isHidden(), "the note is set but not shown"


def test_a_table_refusal_is_one_voice_across_all_three_surfaces(wired, qtbot):
    """BL-2's shared state, seen from the table: the refusal it records is THE record,
    so the gallery's status line and the inspector's batch note say the same thing —
    without waiting for a curation signal, because a refusal emits none."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)
    table = _table(win)
    assert "kept rejected" not in gal.status.text()

    _checkbox_click(table, unmeasurable[1], True)
    _drain(qtbot)

    assert "1 tile kept rejected: flagged by the detector as not measurable" \
        in gal.status.text(), gal.status.text()
    assert "1 event kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    assert table.refusal_note() == insp.refusal_note(), (
        "two surfaces, two sentences about one record:\n"
        f"  table    : {table.refusal_note()!r}\n"
        f"  inspector: {insp.refusal_note()!r}")


def test_the_a_key_is_the_same_write_and_the_same_guard(wired, qtbot):
    """The keyboard reaches the same choke point — a guard installed in one gesture is a
    guard missing from the others."""
    win, df, unmeasurable = wired
    table = _table(win)
    row = unmeasurable[2]
    src = table._model.index(row, 0)
    table.view.setCurrentIndex(table._proxy.mapFromSource(src))
    qtbot.keyClick(table.view, Qt.Key_A)
    _drain(qtbot)
    assert bool(df["accepted"].iloc[row]) is False
    assert "1 event kept rejected" in table.refusal_label.text(), \
        table.refusal_label.text()


# ================================================================================================
# 2. what must NOT move — rejection, re-acceptance, and the mirror
# ================================================================================================
def test_rejection_is_still_never_blocked_and_says_nothing(wired, qtbot):
    win, df, unmeasurable = wired
    table = _table(win)
    ordinary = next(r for r in range(len(df)) if r not in unmeasurable)
    assert bool(df["accepted"].iloc[ordinary]) is True

    assert _checkbox_click(table, ordinary, False) is True
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is False
    assert table.refusal_note() == "", table.refusal_note()


def test_a_hand_rejected_ordinary_row_is_still_re_acceptable(wired, qtbot):
    """``flagged`` / labels / ``zero_baseline`` reject nothing, and a curator's own
    earlier rejection is the curator's to undo — the AND must not become a lock."""
    win, df, unmeasurable = wired
    table = _table(win)
    ordinary = next(r for r in range(len(df)) if r not in unmeasurable)
    _checkbox_click(table, ordinary, False)
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is False

    assert _checkbox_click(table, ordinary, True) is True
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is True, \
        "a hand-rejected row with no rejecting flag became un-acceptable"
    assert table.refusal_note() == "", table.refusal_note()


def test_the_windows_written_values_still_land_through_the_mirror(wired, qtbot):
    """The mirror door (``set_accepted_by_id`` -> ``set_accepted``) now runs under the
    guard, and must cost the mirror nothing: the windows announce what they WROTE, and a
    written True re-derives to True."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)
    ordinary = next(r for r in range(len(df)) if r not in unmeasurable)
    eid = int(df["event_id"].iloc[ordinary])
    insp.show_event(eid)
    _drain(qtbot)

    insp.set_accepted_current(False)
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is False
    insp.set_accepted_current(True)
    _drain(qtbot)
    assert bool(df["accepted"].iloc[ordinary]) is True, "the mirror stopped landing"


# ================================================================================================
# 3. the repair verb, and the fail-closed frame — same words, same rules as the windows
# ================================================================================================
def test_a_row_repaired_out_of_the_forbidden_state_is_announced_as_what_was_written(
        wired, qtbot):
    """``accepted=True`` beside a rejecting flag is reachable from any table written
    before the guard. Accepting there TAKES AN ACCEPTANCE AWAY: the cell goes back to
    False, that write IS announced (the mirror repaints from it), and the verb is
    "corrected back to rejected" — "kept" would be a lie about a destructive edit."""
    win, df, unmeasurable = wired
    table = _table(win)
    row = unmeasurable[0]
    df.iat[row, df.columns.get_loc("accepted")] = True     # out-of-band (pre-guard table)
    win._analysis_panel.resync_curation()
    _drain(qtbot)

    seen: list[tuple[int, bool]] = []
    table.acceptanceChanged.connect(lambda i, a: seen.append((int(i), bool(a))))
    landed = _checkbox_click(table, row, True)
    _drain(qtbot)

    assert bool(df["accepted"].iloc[row]) is False, "the repair did not happen"
    assert landed is True, "a write that moved the cell reported failure"
    assert seen == [(int(df["event_id"].iloc[row]), False)], seen
    assert "1 event corrected back to rejected" in table.refusal_label.text(), \
        table.refusal_label.text()
    assert "kept rejected" not in table.refusal_label.text()


def test_a_frame_with_no_flag_column_fails_closed_on_the_standalone_table(qapp, qtbot):
    """Same rule as both windows: nothing vouches for the row, so the accept is refused,
    the cell is left alone, and the label says why. Rejection still lands. A STANDALONE
    EventTable (no MainWindow, no shared ledger) — the guard lives in the model and the
    fallback ledger, not in the wiring."""
    import pandas as pd

    table = panels_mod.EventTable()
    qtbot.addWidget(table)
    df = pd.DataFrame({"event_id": np.arange(3, dtype=np.int64),
                       "accepted": np.array([False, True, True]),
                       "dwell_s": np.full(3, 1e-3)})
    table.set_events(df)

    assert _checkbox_click(table, 0, True) is False
    assert bool(df["accepted"].iloc[0]) is False, \
        "the guard failed OPEN: it wrote a column it cannot vouch for"
    text = table.refusal_label.text()
    assert "NOT accepted" in text and "'flag' column" in text, text

    assert _checkbox_click(table, 1, False) is True
    assert bool(df["accepted"].iloc[1]) is False, "rejection was blocked"
