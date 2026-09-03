"""M6d — THE REFUSAL NOTE IS ABOUT THE FRAME, NOT ABOUT THE LAST GESTURE.

MEASURED BY ME, in a real ``MainWindow``, on the owner's read-only
``500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf`` (copied first; ch 2
``IN15``, 50 kHz, 317 events, ``max_dwell_s`` 26 ms, so the detector refuses three rows —
0 ``clipped_by_boundary``, 4 and 260 ``dwell>max_dwell_s``), event loop drained after
every gesture. Left column is an out-of-tree revert of this change, right column is this
tree::

                                            REVERTED              THIS TREE
  gallery Accept all                        3 tiles kept rejected 3 tiles kept rejected
  …then R on ONE ORDINARY tile              (no clause at all)    3 tiles kept rejected
  rows genuinely still refused              [0, 4, 260]           [0, 4, 260]
  …then Accept selected on a MEASURABLE row (no clause at all)    3 tiles kept rejected
  gallery Accept all, then the INSPECTOR's
    Reject all (the identical whole-frame
    reject, made elsewhere)                 3 tiles kept rejected 3 tiles kept rejected
  gallery Accept all, then the GALLERY's OWN
    Reject all (the same edit, same frame)  (no clause at all)    3 tiles kept rejected
  inspector Accept all, then R on an
    ordinary event                          (no clause at all)    3 events kept rejected

Rows two, four and seven are the chartered defect, verbatim: **four rows unaccepted,
three of them because the detector could not measure them, and nothing said so.** Rows
five and six are the same defect seen from the side — one identical edit to one identical
frame, described two different ways depending on which window the click landed in, because
the mirrored path (``refresh()``) never cleared the record and the window's own write path
always did.

THE INVARIANT
-------------
The note is a **count of currently-refused rows**: not a latch (M6c fixed that), and not a
per-gesture artefact (this). It is now recomputed from a ``_RefusalLedger``:

* an **accept** re-decides the events IN ITS OWN SCOPE and no others — it is the only
  gesture that can create or retire a refusal, because a refusal is what an accept turned
  into;
* a **reject** records nothing and forgets nothing — rejection is never refused, so it has
  no refusal of its own, and the events an earlier accept was refused over are still
  refused. This is what ``reflect_acceptance`` (the same decision arriving from the events
  table) has always done, which is why the two now agree;
* a **new frame** clears it.

WHAT THE 17 EXISTING WIRED TESTS ON THIS SURFACE DID NOT DO, AND THIS FILE DOES
------------------------------------------------------------------------------
None of them takes a SECOND gesture after the bulk accept. Every test below does: R on an
ordinary tile, R on a refused tile, a table tick and then R, an accept of an unrelated
measurable row, reject-all then accept-all, and a page turn plus a filter change wedged in
between — plus the cross-window comparison that no test made at all.

TWO EXISTING ASSERTIONS PINNED THE DEFECT AS CORRECT and are reported as fence
violations rather than edited here (``tests/`` outside this file is not this change's to
touch):

* ``tests/test_fup_refusal_note_wired.py``
  ``test_the_gallery_note_is_cleared_by_the_next_gesture_and_by_a_new_detection`` —
  asserts the clause is gone after ``reject_all``;
* ``tests/test_fix_accept_all_derives_the_column.py``
  ``test_gallery_says_on_the_status_line_how_many_tiles_it_kept_rejected`` line 486 —
  asserts the clause is gone after an unrelated allowed acceptance.

THE HUNKS THIS FILE PINS
------------------------
1. ``_RefusalLedger`` — one class, byte-identical in both windows, holding the rule that
   was previously five copies of ``self._clear_refusals()`` at the top of five write paths;
2. gallery ``_set_rows_accepted`` — an accept ``forget``s only its own scope; a reject
   touches nothing;
3. inspector ``set_accepted_current`` / ``_set_all_accepted`` — the same, at single-event
   and whole-frame size;
4. the ledger remembers ``event_id`` and not row position, so a merge cannot make the
   count describe somebody else's events;
5. the recount is VECTORISED and counted per distinct flag string, because it runs on
   every page turn, filter change, curation click and resize.

NOT FIXED HERE, AND SINCE FIXED ELSEWHERE: the events table's own per-row checkbox
(``nanoscope/gui/panels.py``) was still not an AND when this file was written
(``panels.py`` was outside that change's fence). BL-1 closed it — the table's per-row
write now applies the same derived-AND and discloses on the ledger every surface shares;
see tests/gui/test_bl_table_accept_guard.py. The "tick one row" tests below therefore
install the refuted state OUT-OF-BAND (see ``_table_tick``), which is how it remains
reachable in the field: a table written before the guard.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import ast
import inspect
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore
from PySide6.QtCore import QItemSelection, QItemSelectionModel

import nanoscope.gui.event_gallery as gal_mod
import nanoscope.gui.event_inspector as insp_mod
import nanoscope.gui.main_window as mw_mod
from nanoscope.analysis.events import AUTOMATIC_REJECTION_FLAGS
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
_N = 220_000
_CLEAN = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400), (70_000, 70_200),
          (110_000, 110_260), (130_000, 130_310)]
_LONG = [(90_000, 100_000), (150_000, 160_000)]
_MAX_DWELL_S = 5e-3


# ================================================================================================
# builders — a real recording, a real MainWindow, the real detector
# ================================================================================================
def _trace() -> np.ndarray:
    """Six ordinary events, one clipped at the open (row 0) and two over the dwell limit:
    THREE of the detector's own rejections, and enough ordinary rows to press R on one
    without touching any of them."""
    rng = np.random.default_rng(0)
    x = np.full(_N, 100.0) + rng.normal(0.0, 1.5, _N)
    x[0:300] -= 30.0                        # the recording OPENS mid-event -> clipped row 0
    for s, e in _CLEAN:
        x[s:e] -= 30.0
    for s, e in _LONG:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path) -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore_m6d.dat", format_name="synthetic",
                         n_samples=arr.size, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


@pytest.fixture
def settings_ns(qapp):
    """A PRIVATE QSettings namespace: this file opens standalone windows and must neither
    read nor leave behind the developer's own saved layout."""
    org, app = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("NanoScopeM6dRefusalClear")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app)


def _drain(qtbot, ms: int = 120) -> None:
    """Let every queued connection run. MainWindow's surface refresher — the thing that
    re-feeds both windows on the very signal the gesture emits — is queued work."""
    qtbot.wait(ms)
    QtCore.QCoreApplication.instance().processEvents(QtCore.QEventLoop.AllEvents, 50)
    qtbot.wait(ms)


@pytest.fixture
def wired(qapp, qtbot, tmp_path, settings_ns):
    """A real MainWindow with a real detection — the only configuration in which the
    surface refresher, the events table and both windows are connected to each other."""
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
                    if not gal_mod._derived_accepted(df["flag"].iloc[i])]
    assert len(unmeasurable) == 3, (
        "this file needs THREE of the detector's own rejections — with one, a record that "
        "is wiped by the next gesture is indistinguishable from one that is not; got "
        f"{df['flag'].tolist()}")
    assert len(df) - len(unmeasurable) >= 3, (
        "and at least three ORDINARY rows, so R can be pressed on one that is not counted")
    try:
        yield win, df, unmeasurable
    finally:
        win._analysis_controller.abort()


def _gallery(win, qtbot):
    win._act_event_gallery.trigger()
    _drain(qtbot)
    return win._event_gallery


def _inspector(win, qtbot):
    win._act_event_inspector.trigger()
    _drain(qtbot)
    return win._event_inspector


def _really_refused(df) -> list[int]:
    """The rows that are, RIGHT NOW, rejected and unmeasurable — computed from the FRAME
    and not from either window, so a test cannot pass by agreeing with the thing it tests."""
    return [i for i in range(len(df))
            if not bool(df["accepted"].iloc[i])
            and not gal_mod._derived_accepted(df["flag"].iloc[i])]


def _ordinary(df) -> int:
    """A row the detector measured fine — the tile the user presses R on."""
    return next(i for i in range(len(df))
                if gal_mod._derived_accepted(df["flag"].iloc[i]))


def _select_rows(gal, rows: list[int]) -> None:
    sm = gal.view.selectionModel()
    sm.clearSelection()
    wanted = set(rows)
    for r in range(gal._model.rowCount()):
        idx = gal._model.index(r, 0)
        if gal._model.df_row(idx) in wanted:
            sm.select(QItemSelection(idx, idx), QItemSelectionModel.Select)


def _table_tick(win, df, row: int, accepted: bool, qtbot) -> None:
    """Put ONE row's ``accepted`` into the given state OUT-OF-BAND, and announce it the
    way any externally edited frame is announced (the owner's resync + refreshers).

    [EDITED with BL-1] This helper used to drive ``set_event_accepted`` — the events
    table's own per-row write, then deliberately unguarded. BL-1 closed that door (the
    table now applies the same derived-AND as every other surface; see
    tests/gui/test_bl_table_accept_guard.py), so the state these tests need — an
    unmeasurable row inside the accepted set — is installed the one way it remains
    reachable in the field: a table written before the guard (an old project file), i.e.
    a direct write on the shared frame. Nothing here records on the refusal ledger: an
    out-of-band write is not an accept gesture, so it can refute a counted row but never
    create a refusal — which is exactly what the stays-silent test below still pins."""
    df.iat[row, df.columns.get_loc("accepted")] = bool(accepted)
    win._analysis_panel.resync_curation()
    win._refresh_surface_windows()
    _drain(qtbot)


def _note_of(w, surface: str) -> str:
    return w.status.text() if surface == "gallery" else w._batch_note.text()


def _expected(df, noun: str) -> str:
    n = len(_really_refused(df))
    return "" if n == 0 else f"{n} {noun}" + ("" if n == 1 else "s") + " kept rejected"


def _synthetic_frame(n: int, unmeasurable: int, seed: int = 0) -> pd.DataFrame:
    """A frame off the real schema's two columns this machinery reads, plus ``event_id``."""
    rng = np.random.default_rng(seed)
    flag = np.array([""] * n, dtype=object)
    bad = rng.choice(n, size=unmeasurable, replace=False)
    flag[bad] = "clipped_by_boundary"
    return pd.DataFrame({"event_id": np.arange(n, dtype=np.int64),
                         "accepted": np.zeros(n, dtype=bool),
                         "flag": flag,
                         "dwell_s": rng.random(n)})


# ================================================================================================
# 0. provenance — which tree, and are these the real methods
# ================================================================================================
def test_the_package_under_test_is_the_one_beside_this_file():
    """A revert run from inside the repo silently tests the LIVE tree: cwd beats PYTHONPATH
    when cwd holds the package. Every module asserted about below must resolve from the tree
    this file lives in — and the methods must be the real ones with the real signatures, a
    revert stub with a wrong signature having manufactured fake failures here before."""
    here = Path(__file__).resolve().parents[1]
    for mod in (nanoscope, gal_mod, insp_mod, mw_mod):
        assert Path(mod.__file__).resolve().is_relative_to(here), \
            f"{mod.__name__} resolved from {mod.__file__}, not from {here}"

    assert list(inspect.signature(gal_mod.EventGalleryWindow._set_rows_accepted)
                .parameters) == ["self", "rows", "accepted"]
    assert list(inspect.signature(insp_mod.EventInspectorWindow._set_all_accepted)
                .parameters) == ["self", "accepted"]
    assert list(inspect.signature(insp_mod.EventInspectorWindow.set_accepted_current)
                .parameters) == ["self", "accepted"]
    for cls in (gal_mod.EventGalleryWindow, insp_mod.EventInspectorWindow):
        assert list(inspect.signature(cls.refusal_note).parameters) == ["self"]
        assert isinstance(cls.__dict__.get("_refused"), property)
    assert "curationBatchChanged" in mw_mod.SURFACE_MUTATION_SIGNALS["event_gallery"]
    assert "curationBatchChanged" in mw_mod.SURFACE_MUTATION_SIGNALS["event_inspector"]


# ================================================================================================
# 1. HUNK 1 — one ledger, not two conditionals
# ================================================================================================
def test_the_machinery_is_one_class_and_the_two_copies_cannot_drift():
    """It was duplicated byte-for-byte across both windows as five ``_clear_refusals()``
    calls at the top of five write paths, which is how the last fix reached one window's
    inbound path and not either window's own. It is now ONE class; this is the same guard
    ``_derived_accepted`` and ``_refusal_note`` already carry, applied to the rule that
    kept being half-applied."""
    for mod in (gal_mod, insp_mod):
        assert inspect.isclass(getattr(mod, "_RefusalLedger", None)), \
            f"{mod.__name__} has no refusal ledger"
    assert inspect.getsource(gal_mod._RefusalLedger) == \
        inspect.getsource(insp_mod._RefusalLedger), \
        "the two windows' refusal ledgers drifted apart"
    # ...and the rule is IN the class, not beside it: every window method that reads or
    # writes the record must go through the ledger instance.
    for cls in (gal_mod.EventGalleryWindow, insp_mod.EventInspectorWindow):
        for name in ("_clear_refusals", "_refused", "_refused_kept", "_refused_repaired"):
            member = cls.__dict__[name]
            src = inspect.getsource(member.fget if isinstance(member, property) else member)
            assert "_refusals" in src, f"{cls.__name__}.{name} keeps its own copy of the state"


def test_no_curation_write_path_clears_the_record_any_more():
    """HUNK 2 + 3, structurally. ``_clear_refusals`` may be reached from the two
    frame-installing doors and from NOWHERE else: it is the "a different detection arrived"
    door, and it was being used as the "a gesture happened" door — which is the defect."""
    allowed = {"set_events", "clear"}
    for mod, cls in ((gal_mod, gal_mod.EventGalleryWindow),
                     (insp_mod, insp_mod.EventInspectorWindow)):
        tree = ast.parse(Path(mod.__file__).read_text())
        callers: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "_clear_refusals"):
                    callers.add(node.name)
        assert callers == allowed, (
            f"{cls.__name__}: _clear_refusals is called from {sorted(callers)}; only "
            f"{sorted(allowed)} install a new frame")


def test_the_recount_is_vectorised_and_counted_per_distinct_flag():
    """HUNK 5, structurally. ``_refused`` is read by ``_sync_controls``, i.e. on every page
    turn, filter change, curation click and resize; a Python loop calling
    ``_derived_accepted`` once per row is 833 ms at 100 000 events with 90 % unmeasurable.
    The behaviour is measured further down; this is the method it must be measured on."""
    src = textwrap.dedent(inspect.getsource(gal_mod._RefusalLedger.refused_in))
    names = {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    assert "value_counts" in names, "the per-distinct-flag trick is gone"
    assert "isin" in names, "the membership test went back to a per-row loop"
    # ...and the RULE is still asked, not re-implemented beside it.
    ids = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    assert "_derived_accepted" in ids
    literals = {n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not (literals & set(AUTOMATIC_REJECTION_FLAGS)), \
        f"the ledger enumerates flag tokens instead of naming the set: {literals}"


# ================================================================================================
# 2. THE HEADLINE — the second gesture, on both surfaces
# ================================================================================================
def test_wired_gallery_keeps_the_disclosure_when_r_is_pressed_on_an_ordinary_tile(
        wired, qtbot):
    """THE measured defect, verbatim. Accept all, then press R on ONE ORDINARY tile: the
    clause vanished while all three unmeasurable rows were still refused."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)

    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()

    row = _ordinary(df)
    assert row not in unmeasurable and bool(df["accepted"].iloc[row])
    _select_rows(gal, [row])
    gal.reject_selected_btn.click()          # the R gesture's one write path
    _drain(qtbot)

    assert bool(df["accepted"].iloc[row]) is False, "the reject itself stopped working"
    assert _really_refused(df) == unmeasurable, "the frame moved; this is no longer the case"
    assert "3 tiles kept rejected: flagged by the detector as not measurable" \
        in gal.status.text(), (
            "R on one ordinary tile took down a disclosure about three OTHER rows that are "
            "still refused:\n" + gal.status.text())


def test_wired_inspector_keeps_the_disclosure_when_r_is_pressed_on_an_ordinary_event(
        wired, qtbot):
    """The same defect on the other surface, at single-event size, on its own widget."""
    win, df, unmeasurable = wired
    insp = _inspector(win, qtbot)

    insp._accept_all_btn.click()
    _drain(qtbot)
    assert "3 events kept rejected" in insp._batch_note.text(), insp._batch_note.text()

    row = _ordinary(df)
    insp.show_event(int(df["event_id"].iloc[row]))
    _drain(qtbot)
    insp.set_accepted_current(False)          # the A/R key's one write path
    _drain(qtbot)

    assert bool(df["accepted"].iloc[row]) is False
    assert _really_refused(df) == unmeasurable
    assert "3 events kept rejected" in insp._batch_note.text(), insp._batch_note.text()
    assert insp._batch_note.isVisible(), "the note is set but not shown"


@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_r_on_an_ALREADY_REFUSED_tile_leaves_the_count_where_it_was(wired, qtbot, surface):
    """ONE STEP PAST the obvious case. Rejecting a row that is already refused moves
    nothing, so the count must not move either — a record that dropped the row it was just
    told to reject would read "2" with three rows still refused, which is the M6c latch
    wearing this fix's name."""
    win, df, unmeasurable = wired
    w = _gallery(win, qtbot) if surface == "gallery" else _inspector(win, qtbot)
    (w.accept_all_btn if surface == "gallery" else w._accept_all_btn).click()
    _drain(qtbot)
    noun = "tile" if surface == "gallery" else "event"
    assert f"3 {noun}s kept rejected" in _note_of(w, surface)

    victim = unmeasurable[1]
    if surface == "gallery":
        _select_rows(w, [victim])
        w.reject_selected_btn.click()
    else:
        w.show_event(int(df["event_id"].iloc[victim]))
        _drain(qtbot)
        w.set_accepted_current(False)
    _drain(qtbot)

    assert _really_refused(df) == unmeasurable
    assert f"3 {noun}s kept rejected" in _note_of(w, surface), _note_of(w, surface)


@pytest.mark.parametrize("surface", ["gallery", "inspector"])
def test_bulk_accept_then_a_table_tick_then_R_recounts_at_every_step(wired, qtbot, surface):
    """THREE gestures across TWO surfaces, which is where the last two fixes each broke.
    The table's tick is unguarded, so it genuinely retires a refusal; R puts it back. The
    number on screen is compared against a count taken from the FRAME at every step."""
    win, df, unmeasurable = wired
    w = _gallery(win, qtbot) if surface == "gallery" else _inspector(win, qtbot)
    (w.accept_all_btn if surface == "gallery" else w._accept_all_btn).click()
    _drain(qtbot)
    noun = "tile" if surface == "gallery" else "event"
    assert _expected(df, noun) in _note_of(w, surface)

    _table_tick(win, df, unmeasurable[0], True, qtbot)          # from ELSEWHERE
    assert len(_really_refused(df)) == 2
    assert _expected(df, noun) in _note_of(w, surface), _note_of(w, surface)

    # ...and now the window's OWN reject on an ordinary row, which used to erase all of it.
    row = _ordinary(df)
    if surface == "gallery":
        _select_rows(w, [row])
        w.reject_selected_btn.click()
    else:
        w.show_event(int(df["event_id"].iloc[row]))
        _drain(qtbot)
        w.set_accepted_current(False)
    _drain(qtbot)
    assert len(_really_refused(df)) == 2
    assert f"2 {noun}s kept rejected" in _note_of(w, surface), _note_of(w, surface)

    # ...and R on the row the TABLE accepted puts it back UP, which a latch cannot do.
    _table_tick(win, df, unmeasurable[0], False, qtbot)
    assert len(_really_refused(df)) == 3
    assert f"3 {noun}s kept rejected" in _note_of(w, surface), _note_of(w, surface)


def test_an_accept_of_an_unrelated_MEASURABLE_row_does_not_take_the_note_down(wired, qtbot):
    """The other second gesture: an ACCEPT that refuses nothing. It re-decides its own
    scope — one measurable row — and has no standing over three rows it never touched.
    ``tests/test_fix_accept_all_derives_the_column.py:486`` asserts the opposite; that
    assertion is reported as a fence violation."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text()

    row = _ordinary(df)
    _select_rows(gal, [row])
    gal.reject_selected_btn.click()          # take it out of the accepted set first
    _drain(qtbot)
    _select_rows(gal, [row])
    gal.accept_selected_btn.click()          # ...and put it back: an allowed acceptance
    _drain(qtbot)

    assert bool(df["accepted"].iloc[row]) is True, "the allowed acceptance did not land"
    assert _really_refused(df) == unmeasurable
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()


def test_reject_all_then_accept_all_says_the_same_thing_at_both_ends(wired, qtbot):
    """Reject-all FIRST — so there is nothing to disclose — then accept-all, which refuses
    three. A record that a reject can wipe passes the first half and a record that a reject
    can CREATE fails it, so both directions are pinned here."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)

    gal.reject_all_btn.click()
    _drain(qtbot)
    assert gal.refusal_note() == "", (
        "a reject invented a refusal it never made:\n" + gal.refusal_note())
    assert len(_really_refused(df)) == 3, "the rows ARE refused; nothing has asked to accept"

    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()


def test_a_page_turn_and_a_filter_change_between_the_two_gestures_change_nothing(
        wired, qtbot):
    """A rebuild of the line with no curation anywhere near it, wedged between the accept
    and the R — the ordering trick that was the batch-before-last's fix does not survive
    this, and neither does a note held as text written over the line's output."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text()

    gal._model.set_page_size(2)
    gal._model.set_page(1)                    # a page turn
    gal._sync_controls()
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()
    gal.filter_box.setCurrentText("Accepted only")   # a filter change
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()
    gal.refresh()                             # the refresher's own door
    assert gal.status.text().count("kept rejected") == 1, (
        "the note doubled on rebuild:\n" + gal.status.text())

    gal.filter_box.setCurrentText("All")
    _drain(qtbot)
    row = _ordinary(df)
    _select_rows(gal, [row])
    gal.reject_selected_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()


# ================================================================================================
# 3. THE ASYMMETRY — the same edit, made in two places, described two ways
# ================================================================================================
def test_the_gallerys_own_reject_all_and_the_inspectors_leave_the_same_sentence(
        wired, qtbot):
    """THE defect stated as an equality, and the one no test made. Two runs, one frame,
    one identical resulting state (0 accepted, three rows refused): the whole-frame reject
    made in the INSPECTOR reached the gallery through ``refresh()``, which never cleared,
    while the gallery's own went through its write path, which always did. The window
    disagreed with itself about which hand held the mouse."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)

    gal.accept_all_btn.click()
    _drain(qtbot)
    insp.reject_all()                         # the SAME edit, made elsewhere
    _drain(qtbot)
    from_elsewhere = gal.refusal_note()
    state_elsewhere = (int(df["accepted"].to_numpy(dtype=bool).sum()), _really_refused(df))

    gal.accept_all_btn.click()
    _drain(qtbot)
    gal.reject_all_btn.click()                # ...and made here
    _drain(qtbot)
    from_here = gal.refusal_note()
    state_here = (int(df["accepted"].to_numpy(dtype=bool).sum()), _really_refused(df))

    assert state_here == state_elsewhere, "the two paths did not leave the same frame"
    assert from_here == from_elsewhere, (
        "one frame, two sentences, decided by which window the click landed in:\n"
        f"  from the inspector: {from_elsewhere!r}\n"
        f"  from the gallery  : {from_here!r}")
    assert "3 tiles kept rejected" in from_here, from_here


# ================================================================================================
# 4. the false-positive direction — a window that refused nothing says nothing
# ================================================================================================
def test_a_window_that_has_refused_nothing_stays_silent_through_every_gesture(
        wired, qtbot):
    """The mirror-image lie. A record that only a new frame clears must still never CLAIM a
    refusal: nothing but an accept may put an event in it."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    insp = _inspector(win, qtbot)

    for gesture in (gal.reject_all_btn.click,
                    lambda: gal._model.set_page(0),
                    gal.refresh,
                    insp.reject_all,
                    lambda: _table_tick(win, df, unmeasurable[0], True, qtbot),
                    lambda: _table_tick(win, df, unmeasurable[0], False, qtbot)):
        gesture()
        _drain(qtbot)
        assert gal.refusal_note() == "", (gesture, gal.refusal_note())
        assert insp.refusal_note() == "", (gesture, insp.refusal_note())
        assert "kept rejected" not in gal.status.text(), gal.status.text()

    # ...and an accept that refuses nothing still says nothing.
    ok = [r for r in range(len(df)) if r not in unmeasurable]
    _select_rows(gal, ok)
    gal.accept_selected_btn.click()
    _drain(qtbot)
    assert gal.refusal_note() == "", gal.refusal_note()


def test_a_new_detection_still_takes_the_note_and_the_counted_events_with_it(wired, qtbot):
    """The one thing that DOES clear it. The remembered ids name events of one detection;
    a different frame makes them somebody else's."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "kept rejected" in gal.status.text()

    with qtbot.waitSignal(win._analysis_controller.finished, timeout=60_000):
        win._analysis_panel.trigger_detection()
    _drain(qtbot)
    assert gal._refused_kept == [] and gal._refused_repaired == []
    assert "kept rejected" not in gal.status.text(), gal.status.text()


# ================================================================================================
# 5. the repair verb, and the fail-closed frame — neither may move
# ================================================================================================
@pytest.mark.parametrize("mod_name", ["event_gallery", "event_inspector"])
def test_a_row_repaired_out_of_the_forbidden_state_is_still_not_called_kept(qapp, mod_name):
    """"kept rejected" is a LIE about a destructive edit: the row was sitting
    ``accepted=True`` beside a rejecting flag (reachable from any table written before M6)
    and the accept put it back, so the user's accepted count went DOWN on a click that said
    "accept". Three outcomes, three verbs — this is the one that must not collapse into the
    first, and the ledger keeps the two buckets apart."""
    mod = gal_mod if mod_name == "event_gallery" else insp_mod
    noun = "tile" if mod_name == "event_gallery" else "event"
    ledger = mod._RefusalLedger()
    df = _synthetic_frame(6, 2, seed=3)
    bad = [i for i in range(6) if df["flag"].iloc[i]]
    df.iat[bad[0], df.columns.get_loc("accepted")] = True     # the forbidden state
    eid = df["event_id"].to_numpy()

    ledger.record(kept=eid[[bad[1]]], repaired=eid[[bad[0]]])
    df.iat[bad[0], df.columns.get_loc("accepted")] = False    # ...the accept repairs it
    note = mod._refusal_note(*ledger.counts(df), noun=noun)
    assert f"1 {noun} corrected back to rejected" in note, note
    assert f"1 {noun} kept rejected" in note, note
    assert note.endswith(": flagged by the detector as not measurable"), note


@pytest.mark.parametrize("mod_name", ["event_gallery", "event_inspector"])
def test_a_frame_with_no_flag_column_still_fails_closed_and_still_says_so(qapp, mod_name):
    """The derived half cannot be read, so nothing can be vouched for and NOTHING is
    accepted. The disclosure now outlives the gesture like every other one — the table
    still has no ``flag`` column after a reject, so the refusal still stands."""
    mod = gal_mod if mod_name == "event_gallery" else insp_mod
    noun = "tile" if mod_name == "event_gallery" else "event"
    ledger = mod._RefusalLedger()
    flagless = _synthetic_frame(5, 0).drop(columns=["flag"])

    ledger.record_blocked(len(flagless))
    note = mod._refusal_note(*ledger.counts(flagless), noun=noun)
    assert note == (f"5 {noun}s NOT accepted: this table has no 'flag' column, so the "
                    "detector's own rejections cannot be read — refusing rather than "
                    "guessing"), note
    # a reject afterwards records nothing and forgets nothing: the frame has not changed.
    assert mod._refusal_note(*ledger.counts(flagless), noun=noun) == note
    # ...and the NEXT accept re-decides it, scope and all.
    ledger.forget([0, 1])
    ledger.record_blocked(2)
    assert f"2 {noun}s NOT accepted" in mod._refusal_note(*ledger.counts(flagless), noun=noun)
    ledger.clear()
    assert mod._refusal_note(*ledger.counts(flagless), noun=noun) == ""


def test_rejection_is_still_never_blocked_on_a_frame_with_no_flag_column(wired, qtbot):
    """MUST NOT MOVE. A person may always reject an event the app was willing to keep, and
    that is true of a frame the guard cannot read at all."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    flagless = df.drop(columns=["flag"]).copy()
    flagless["accepted"] = True
    gal.set_events(gal._doc_id, gal._channel, flagless)
    _drain(qtbot)

    gal.accept_all_btn.click()
    _drain(qtbot)
    assert bool(flagless["accepted"].all()), "the accept wrote to a frame it refused"
    assert "NOT accepted" in gal.status.text(), gal.status.text()

    gal.reject_all_btn.click()
    _drain(qtbot)
    assert not bool(flagless["accepted"].any()), "rejection was blocked"


def test_the_rejecting_flags_are_still_the_three_the_schema_names():
    """MUST NOT MOVE, and named rather than listed inside the window."""
    assert AUTOMATIC_REJECTION_FLAGS == {
        "clipped_by_boundary", "dwell>max_dwell_s", "non_finite"}
    for mod in (gal_mod, insp_mod):
        assert mod.AUTOMATIC_REJECTION_FLAGS is AUTOMATIC_REJECTION_FLAGS


def test_a_hand_rejected_unflagged_row_is_still_re_acceptable(wired, qtbot):
    """MUST NOT MOVE. The guard is about the DETECTOR's rejections; a person's own reject
    carries no rejecting token and must come back on the next accept."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    row = _ordinary(df)
    _select_rows(gal, [row])
    gal.reject_selected_btn.click()
    _drain(qtbot)
    assert bool(df["accepted"].iloc[row]) is False

    gal.accept_all_btn.click()
    _drain(qtbot)
    assert bool(df["accepted"].iloc[row]) is True, "a hand rejection became permanent"
    assert _really_refused(df) == unmeasurable


# ================================================================================================
# 6. HUNK 4 — the record names events, not positions
# ================================================================================================
@pytest.mark.parametrize("mod_name", ["event_gallery", "event_inspector"])
def test_the_record_survives_the_rows_moving_under_it(qapp, mod_name):
    """A row index survives an append and NOT a merge, and this record now outlives many
    more gestures than the per-gesture counter did. Dropping a row ahead of a refused one
    shifts every position after it: a ledger of positions then counts somebody else's
    event, silently and with a plausible number."""
    mod = gal_mod if mod_name == "event_gallery" else insp_mod
    df = _synthetic_frame(10, 3, seed=11)
    bad = [i for i in range(10) if df["flag"].iloc[i]]
    good = [i for i in range(10) if not df["flag"].iloc[i]]
    eid = df["event_id"].to_numpy()

    ledger = mod._RefusalLedger()
    ledger.record(kept=eid[bad])
    assert ledger.counts(df) == (3, 0, 0)

    # a merge drops two MEASURABLE rows from in front of the refused ones...
    merged = df.drop(index=[good[0], good[1]]).reset_index(drop=True)
    assert ledger.counts(merged) == (3, 0, 0), (
        "the record moved onto other events when the rows shifted")
    # ...and an event that leaves the frame altogether simply stops being counted.
    shorter = merged[merged["event_id"] != eid[bad[0]]].reset_index(drop=True)
    assert ledger.counts(shorter) == (2, 0, 0)


# ================================================================================================
# 7. HUNK 5 — the interactive cost of an honest disclosure
# ================================================================================================
@pytest.mark.parametrize("n,bad", [(20_000, 18_000), (100_000, 90_000)])
def test_the_recount_is_not_a_python_loop_over_the_refused_rows(qapp, n, bad):
    """``_refused`` is read by ``_sync_controls``, which runs on every page turn, filter
    change, curation click and resize. At the owner's real sizes (317 events, 3 refused)
    the per-row loop was ~30 µs and invisible; at 100 000 events with 90 % unmeasurable —
    reachable through a mis-set ``max_dwell_s`` — it was 844 ms measured, i.e. an honest
    disclosure would have been by far the slowest thing in the window. The ceiling is
    deliberately loose: what is pinned is the ORDER, not a machine's clock.

    DRIVEN THROUGH ``_refused`` ITSELF, over a stub holding nothing but the frame and the
    record, so that it measures WHATEVER IMPLEMENTATION THIS TREE HAS rather than the one
    this change introduced — on an out-of-tree revert this fails with a time, not with an
    AttributeError. The wired counterpart below measures the same thing where the user
    meets it.
    """
    df = _synthetic_frame(n, bad, seed=5)
    rows = np.nonzero(df["flag"].to_numpy() != "")[0]
    stub = types.SimpleNamespace(_df=df)
    if hasattr(gal_mod, "_RefusalLedger"):                 # this tree
        ledger = gal_mod._RefusalLedger()
        ledger.record(kept=df["event_id"].to_numpy()[rows])
        stub._refusals = ledger
    else:                                                  # a revert: the same record, its shape
        stub._refused_kept = rows.tolist()
        stub._refused_repaired = []
        stub._refused_blocked = 0
        stub._still_refused = lambda r: gal_mod.EventGalleryWindow._still_refused(stub, r)
    recount = gal_mod.EventGalleryWindow.__dict__["_refused"].fget
    assert recount(stub) == (bad, 0, 0), "the stub does not reproduce the real record"

    best = min(_time_recount(recount, stub) for _ in range(5))
    assert best < 0.060, f"{n} events / {bad} refused: recount took {best * 1e3:.1f} ms"


def _time_recount(recount, stub) -> float:
    t0 = time.perf_counter()
    recount(stub)
    return time.perf_counter() - t0


def test_the_wired_status_line_is_not_slower_for_being_honest(wired, qtbot):
    """The same cost where the user meets it: a bare rebuild of the line, on the real
    surface, after the accept that fills the record."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    gal.accept_all_btn.click()
    _drain(qtbot)
    assert "3 tiles kept rejected" in gal.status.text()

    best = min(_time_sync(gal) for _ in range(20))
    assert best < 0.050, f"_sync_controls took {best * 1e3:.1f} ms"


def _time_sync(gal) -> float:
    t0 = time.perf_counter()
    gal._sync_controls()
    return time.perf_counter() - t0


# ================================================================================================
# 8. undo — the frame the record describes must come back bit for bit
# ================================================================================================
def test_undo_after_the_bulk_accept_and_the_R_restores_the_frame_exactly(wired, qtbot):
    """MUST NOT MOVE: ``atol=0``, NaN-equal, every column. The note is about the frame, so
    a fix that touched the frame to keep the note true would show up here."""
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    snapshot = df.copy(deep=True)

    gal.accept_all_btn.click()
    _drain(qtbot)
    _select_rows(gal, [_ordinary(df)])
    gal.reject_selected_btn.click()
    _drain(qtbot)
    assert not df["accepted"].equals(snapshot["accepted"]), "nothing to undo"

    win._act_undo.trigger()
    _drain(qtbot)
    win._act_undo.trigger()
    _drain(qtbot)
    restored = win._analysis_panel.current_detection().payload.df

    assert list(restored.columns) == list(snapshot.columns)
    for col in snapshot.columns:
        a, b = snapshot[col], restored[col]
        if pd.api.types.is_float_dtype(a):
            assert np.allclose(a.to_numpy(), b.to_numpy(), atol=0, rtol=0, equal_nan=True), col
        else:
            assert [x if pd.notna(x) else "<NA>" for x in a.tolist()] == \
                [x if pd.notna(x) else "<NA>" for x in b.tolist()], col


# ================================================================================================
# 9. what I went looking for INSIDE the fix
# ================================================================================================
@pytest.mark.parametrize("mod_name", ["event_gallery", "event_inspector"])
def test_the_recount_does_not_raise_on_a_frame_with_no_event_id_column(qapp, mod_name):
    """FOUND IN MY OWN FIX. The ledger names events by ``event_id``, so the recount reads a
    column the implementation it replaced never touched — and it is read by
    ``_sync_controls``, i.e. on every resize. Neither window can DRAW such a frame at all
    (its tile captions and its row lookups have always needed the column), so the guard is
    not a claim that the window works without it; what it buys is that the disclosure is
    not the new thing that raises, and that the answer is the silent one rather than a
    number invented out of positions."""
    mod = gal_mod if mod_name == "event_gallery" else insp_mod
    noun = "tile" if mod_name == "event_gallery" else "event"
    df = _synthetic_frame(6, 2, seed=9)
    ledger = mod._RefusalLedger()
    ledger.record(kept=df["event_id"].to_numpy()[df["flag"].to_numpy() != ""])
    assert ledger.counts(df) == (2, 0, 0)

    headless = df.drop(columns=["event_id"])
    assert ledger.counts(headless) == (0, 0, 0)
    assert mod._refusal_note(*ledger.counts(headless), noun=noun) == ""
    # ...and the frame-level refusal, which needs no identity at all, still answers.
    ledger.record_blocked(6)
    assert mod._refusal_note(*ledger.counts(headless.drop(columns=["flag"])), noun=noun)


def test_the_bucket_is_re_decided_by_the_next_accept_over_the_same_row(qapp):
    """A DELIBERATE CONSEQUENCE, pinned so it is a decision and not an accident. "corrected
    back to rejected" is a statement about where the row STARTED when the accept reached it.
    A second accept over the same row finds it already rejected, so the honest verb for THAT
    gesture is "kept rejected" — the destructive edit is not repeated, and claiming it was
    would be the mirror-image lie of the one this whole file is about."""
    ledger = gal_mod._RefusalLedger()
    df = _synthetic_frame(4, 1, seed=7)
    bad = next(i for i in range(4) if df["flag"].iloc[i])
    eid = df["event_id"].to_numpy()

    df.iat[bad, df.columns.get_loc("accepted")] = True         # the forbidden state
    ledger.record(repaired=eid[[bad]])
    df.iat[bad, df.columns.get_loc("accepted")] = False        # ...repaired by that accept
    assert ledger.counts(df) == (0, 1, 0)

    ledger.forget(eid[[bad]])                                  # a SECOND accept, same row
    ledger.record(kept=eid[[bad]])
    assert ledger.counts(df) == (1, 0, 0)


def test_a_hand_installed_frame_is_replaced_by_the_owners_own_the_moment_anything_emits(
        wired, qtbot):
    """FOUND WHILE HUNTING INSIDE MY OWN FIX, and the reason the frame-level refusal's
    survival is pinned at the ledger and not out here.

    On a frame MainWindow was never told about, the first gesture that EMITS sends the
    owner's surface refresher straight back with the real detection — synchronously, on the
    same call stack — and the record goes with the frame, which is correct. So a wired test
    of "the blocked note survives a reject" would have been measuring the re-feed rather
    than the rule, and would have passed for the wrong reason on a tree where the rule was
    broken. Pinned here so the next reader does not write that test.
    """
    win, df, unmeasurable = wired
    gal = _gallery(win, qtbot)
    flagless = df.drop(columns=["flag"]).copy()
    flagless["accepted"] = True
    gal.set_events(gal._doc_id, gal._channel, flagless)
    _drain(qtbot)

    gal.accept_all_btn.click()              # fails closed, writes nothing, EMITS NOTHING
    _drain(qtbot)
    assert gal._df is flagless, "a refused accept re-fed the window"
    assert "NOT accepted: this table has no 'flag' column" in gal.refusal_note()

    gal.reject_all_btn.click()              # rejection is never blocked, and it emits
    assert not bool(flagless["accepted"].any()), "rejection was blocked"
    assert gal._df is not flagless, (
        "the owner's refresher did not re-feed the window; the assertion below is then "
        "about the rule and belongs at the ledger")
    assert gal.refusal_note() == "", (
        "the record outlived the frame it named:\n" + gal.refusal_note())
