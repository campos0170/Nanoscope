"""M6 — "Accept all" must not accept a row the DETECTOR rejected.

``accepted`` is not a slot a button may write; it is derived, and the derivation is an
AND (``analysis/characterize.py``)::

    accepted = ~(too_long | clipped | non_finite) & inherit_accepted

The left half is a statement about the MEASUREMENT, the right half is the curator's own
decision. Both bulk-curation surfaces used to write the curator's half over BOTH halves:

* ``EventInspectorWindow._set_all_accepted``  ->  ``self._df["accepted"] = bool(accepted)``
* ``EventGalleryWindow._set_rows_accepted``   ->  ``self._df.iat[r, col] = bool(accepted)``

Measured on the owner's -600 mV recording through the real windows (276 events, one of
them ``flag='clipped_by_boundary', accepted=False`` at ``start_idx=0``): one click on
"Accept all" took the accepted count from 275 to **276** and left the clipped row
``accepted=True`` with its flag intact -- on the same window whose readout printed
``accepted: no`` a moment earlier. Four gestures reached it: the inspector's "Accept all",
the gallery's "Accept all N", select-then-A, and Space (invert).

WHAT THESE TESTS PIN
--------------------
1. the derivation is read from ONE named set
   (:data:`~nanoscope.analysis.events.AUTOMATIC_REJECTION_FLAGS`) in both windows, not
   from a second list of tokens that would rot when a fourth rejecting rule is added;
2. every gesture on both surfaces writes ``derived AND requested``;
3. what the gallery ANNOUNCES is what it wrote, not what was asked for -- the owner
   mirrors that signal straight onto the table's checkbox;
4. the two things the fix must not break: a hand REJECT of an accepted event, and a
   re-accept of a hand-rejected (unflagged) event. Only flag-derived rejection is
   inviolable, and rejection itself is never blocked;
5. the windows' own controls agree with the frame afterwards.

Every module here is asserted to resolve from the tree this file lives in
(:func:`test_the_package_under_test_is_the_one_beside_this_file`) -- ``nanoscope`` is
importable from an editable install on this machine, so a run against a reverted copy of
the tree could otherwise import the FIXED package and report a false pass.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionModel

import nanoscope.analysis.events as events_mod
import nanoscope.gui.event_gallery as gal_mod
import nanoscope.gui.event_inspector as insp_mod
from nanoscope.analysis.characterize import characterize
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    AUTOMATIC_REJECTION_FLAGS,
    FLAG_CLIPPED,
    FLAG_CURATED,
    FLAG_MAX_DWELL,
    FLAG_NON_FINITE,
    FLAG_ZERO_BASELINE,
    combine_flags,
    has_flag,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.event_gallery import EventGalleryWindow
from nanoscope.gui.event_inspector import EventInspectorWindow

FS = 100_000.0

# The trace: it STARTS inside an event (so the detector clips row 0 against the analysed
# span, exactly as it does on the owner's -600 mV recording), carries four clean events,
# and ends with an 80 ms one that is over the declared 5 ms limit.
_CLEAN = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400), (70_000, 70_200)]
_LONG = (90_000, 98_000)
_MAX_DWELL_S = 5e-3

ROW_CLIPPED = 0          # flag=clipped_by_boundary, accepted=False   (detector's own)
ROW_NON_FINITE = 1       # stamped with what characterize writes for a poisoned window
ROW_CAVEAT = 2           # zero_baseline: DERIVED, but it rejects nothing
ROW_LABELLED = 3         # a curator's label + 'flagged': never a rejection reason
ROW_HAND = 4             # a bare hand reject: no reason token at all
ROW_TOO_LONG = 5         # flag=dwell>max_dwell_s, accepted=False     (detector's own)


# ================================================================================================
# builders
# ================================================================================================

def _trace() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = np.full(200_000, 100.0) + rng.normal(0.0, 1.5, 200_000)
    x[0:300] -= 30.0                       # the recording opens mid-event
    for s, e in _CLEAN:
        x[s:e] -= 30.0
    x[_LONG[0]:_LONG[1]] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _session_with_doc(tmp_path) -> tuple[Session, str, int]:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / "pore_m6.dat",
        format_name="synthetic",
        n_samples=arr.size,
        sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META,
        channels=(info,),
    )
    session = Session()
    doc_id = session.add_document(TraceDocument(meta=meta, channels=(arr,)))
    return session, doc_id, 0


def _detection(session: Session, doc_id: str, channel: int):
    """The REAL detector's frame, then the curation states the GUI has to respect.

    Rows 0 and 5 are the detector's own rejections. The other three states are written
    the way ``characterize`` writes them (token + ``accepted`` together) -- see
    :func:`test_characterize_writes_the_non_finite_pairing_this_file_stamps`, which
    proves the stamp is faithful rather than assumed.
    """
    ch = session.calibrated(doc_id, channel)
    df, prov = detect_events(np.asarray(ch.data), float(ch.sample_rate_hz),
                             min_dwell_s=5e-4, max_dwell_s=_MAX_DWELL_S)
    assert len(df) == 6, "the fixture trace must produce exactly six events"
    assert str(df["flag"].iloc[ROW_CLIPPED]) == FLAG_CLIPPED
    assert str(df["flag"].iloc[ROW_TOO_LONG]) == FLAG_MAX_DWELL
    assert not bool(df["accepted"].iloc[ROW_CLIPPED])
    assert not bool(df["accepted"].iloc[ROW_TOO_LONG])

    fcol = df.columns.get_loc("flag")
    acol = df.columns.get_loc("accepted")
    df.iat[ROW_NON_FINITE, fcol] = FLAG_NON_FINITE
    df.iat[ROW_NON_FINITE, acol] = False
    df.iat[ROW_CAVEAT, fcol] = FLAG_ZERO_BASELINE          # a caveat, not a rejection
    df.iat[ROW_LABELLED, fcol] = combine_flags("molecule=apo-ferritin", FLAG_CURATED)
    df.iat[ROW_HAND, acol] = False                          # a bare reject == by hand
    return df, prov


def _inspector(qtbot, session, df, doc_id, channel, prov=None) -> EventInspectorWindow:
    insp = EventInspectorWindow(session)
    qtbot.addWidget(insp)
    insp.set_events(doc_id, channel, df, provenance=dict(prov or {}))
    return insp


def _gallery(qtbot, session, df, doc_id, channel) -> EventGalleryWindow:
    gal = EventGalleryWindow(session)
    qtbot.addWidget(gal)
    gal.set_events(doc_id, channel, df)
    return gal


def _select_df_rows(gal: EventGalleryWindow, rows: list[int]) -> None:
    """Select exactly the tiles standing for these DF rows, through the real selection model."""
    sm = gal.view.selectionModel()
    sm.clearSelection()
    for r in range(gal._model.rowCount()):
        idx = gal._model.index(r, 0)
        if gal._model.df_row(idx) in rows:
            sm.select(QItemSelection(idx, idx), QItemSelectionModel.Select)


def _accepted(df) -> list[bool]:
    return [bool(v) for v in df["accepted"].to_numpy(dtype=bool)]


#: What every gesture that says "accept" must land on this fixture: the two rows the
#: detector rejected stay rejected, and every other row -- caveat, label, hand reject --
#: is accepted.
_EXPECTED_AFTER_ACCEPT = [False, False, True, True, True, False]


# ================================================================================================
# 0. provenance — which package these tests are actually exercising
# ================================================================================================

def test_the_package_under_test_is_the_one_beside_this_file():
    """``nanoscope`` is installed editable on this machine, so a run against a REVERTED
    copy of the tree can silently import the fixed package from site-packages and report a
    false pass. Every module this file asserts about must live in the same tree as this
    file, and the two windows must be the real classes with the real signatures -- a stub
    with the wrong signature has manufactured fake failures in this project before."""
    here = Path(__file__).resolve().parents[1]
    for mod in (nanoscope, events_mod, insp_mod, gal_mod):
        assert Path(mod.__file__).resolve().parents[1] == here or \
            Path(mod.__file__).resolve().is_relative_to(here), \
            f"{mod.__name__} resolved from {mod.__file__}, not from {here}"

    assert list(inspect.signature(insp_mod.EventInspectorWindow._set_all_accepted)
                .parameters) == ["self", "accepted"]
    assert list(inspect.signature(gal_mod.EventGalleryWindow._set_rows_accepted)
                .parameters) == ["self", "rows", "accepted"]
    assert list(inspect.signature(insp_mod.EventInspectorWindow.accept_all)
                .parameters) == ["self"]
    assert list(inspect.signature(gal_mod.EventGalleryWindow.accept_all)
                .parameters) == ["self"]


def test_characterize_writes_the_non_finite_pairing_this_file_stamps():
    """The fixture stamps ``flag=non_finite, accepted=False`` on one row because the
    detector refuses a non-finite input outright. This is the rule-owner writing that same
    pairing, so the stamp is a copy of a real state and not an invented one."""
    x = np.full(3000, 100.0)
    x[1000:1300] -= 30.0
    x[1100] = np.inf
    row = characterize(x, np.full(3000, 100.0), np.array([1000]), np.array([1300]),
                       FS, -1, noise_rms=1.5).iloc[0]
    assert str(row["flag"]) == FLAG_NON_FINITE
    assert bool(row["accepted"]) is False


# ================================================================================================
# 1. the derivation is read from ONE named set, in both windows
# ================================================================================================

def test_both_windows_derive_accepted_from_the_one_named_set():
    """HUNK 1. Both windows must ask :data:`AUTOMATIC_REJECTION_FLAGS` -- the same object,
    not a copy and not a second list of tokens. A fix that spelled the three rejecting
    tokens out at a second site would go stale the moment a fourth rule is added, which is
    the failure mode this project has already paid for twice (C1, C21)."""
    for mod in (insp_mod, gal_mod):
        fn = getattr(mod, "_derived_accepted", None)
        assert fn is not None, f"{mod.__name__} has no derivation helper"
        assert list(inspect.signature(fn).parameters) == ["flag"]
        assert mod.AUTOMATIC_REJECTION_FLAGS is events_mod.AUTOMATIC_REJECTION_FLAGS

        body = ast.parse(inspect.getsource(fn).strip()).body[0]
        names = {n.id for n in ast.walk(body) if isinstance(n, ast.Name)}
        assert "AUTOMATIC_REJECTION_FLAGS" in names
        literals = {n.value for n in ast.walk(body)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        literals.discard(ast.get_docstring(body) or "")
        assert not (literals & set(events_mod.DERIVED_FLAGS)), \
            f"{mod.__name__} enumerates flag tokens instead of naming the set: {literals}"

    assert inspect.getsource(insp_mod._derived_accepted) == \
        inspect.getsource(gal_mod._derived_accepted), \
        "the two windows' derivation drifted apart"


@pytest.mark.parametrize("mod_name", ["event_inspector", "event_gallery"])
def test_the_derivation_answers_every_token_the_way_characterize_does(mod_name):
    mod = insp_mod if mod_name == "event_inspector" else gal_mod
    d = mod._derived_accepted
    # rejects
    assert d(FLAG_CLIPPED) is False
    assert d(FLAG_MAX_DWELL) is False
    assert d(FLAG_NON_FINITE) is False
    assert d(combine_flags(FLAG_ZERO_BASELINE, FLAG_CLIPPED)) is False
    assert d(combine_flags(FLAG_CLIPPED, FLAG_CURATED)) is False   # a mark cannot excuse it
    # does not reject
    assert d("") is True
    assert d(None) is True
    assert d(pd.NA) is True                       # the schema's string dtype missing value
    assert d(float("nan")) is True
    assert d(FLAG_ZERO_BASELINE) is True          # DERIVED, but it rejected nothing
    assert d(FLAG_CURATED) is True
    assert d("molecule=apo-ferritin") is True


@pytest.mark.parametrize("mod_name", ["event_inspector", "event_gallery"])
def test_a_fourth_rejecting_rule_reaches_the_gui_with_no_second_edit(mod_name, monkeypatch):
    """The point of naming the set rather than the tokens: add a rule to the set and both
    windows honour it, with nothing edited here."""
    mod = insp_mod if mod_name == "event_inspector" else gal_mod
    assert mod._derived_accepted("clogged") is True
    monkeypatch.setattr(mod, "AUTOMATIC_REJECTION_FLAGS",
                        frozenset(AUTOMATIC_REJECTION_FLAGS | {"clogged"}))
    assert mod._derived_accepted("clogged") is False


# ================================================================================================
# 2. the Event Inspector's bulk curation
# ================================================================================================

def test_inspector_accept_all_leaves_the_detectors_rejections_rejected(qapp, qtbot, tmp_path):
    """HUNK 2, and the defect itself. Reproduced on the owner's -600 mV recording: 275
    accepted -> 276 after one click, the clipped row accepted with its flag intact."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    insp = _inspector(qtbot, session, df, doc_id, ch, prov)

    batched: list[int] = []
    insp.curationBatchChanged.connect(lambda: batched.append(1))
    insp._accept_all_btn.click()                      # the real button

    assert _accepted(df) == _EXPECTED_AFTER_ACCEPT
    # the reasons are still on the rows -- a rejection with no reason is C21
    assert has_flag(df["flag"].iloc[ROW_CLIPPED], FLAG_CLIPPED)
    assert has_flag(df["flag"].iloc[ROW_TOO_LONG], FLAG_MAX_DWELL)
    assert batched == [1]                             # still exactly one announcement


def test_inspector_accept_all_still_re_accepts_a_hand_rejected_event(qapp, qtbot, tmp_path):
    """The other half of the guarantee: ONLY flag-derived rejection is inviolable. A bare
    hand reject carries no reason token, so "Accept all" is entitled to reverse it."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    insp = _inspector(qtbot, session, df, doc_id, ch, prov)

    eid = int(df["event_id"].iloc[ROW_CAVEAT])
    insp.show_event(eid)
    insp.set_accepted_current(False)                  # hand-reject an accepted event
    assert bool(df["accepted"].iloc[ROW_CAVEAT]) is False

    insp._accept_all_btn.click()
    assert bool(df["accepted"].iloc[ROW_CAVEAT]) is True
    assert bool(df["accepted"].iloc[ROW_HAND]) is True
    assert bool(df["accepted"].iloc[ROW_LABELLED]) is True


def test_inspector_reject_all_is_never_blocked(qapp, qtbot, tmp_path):
    """Rejection is a person's to make about any row; the guard is on the accepting
    direction only."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    insp = _inspector(qtbot, session, df, doc_id, ch, prov)
    insp._reject_all_btn.click()
    assert _accepted(df) == [False] * 6


def test_inspector_accept_all_repairs_a_row_that_is_already_in_the_forbidden_state(
        qapp, qtbot, tmp_path):
    """The state the defect left behind -- ``accepted=True`` beside ``clipped_by_boundary``
    -- is one the schema forbids, and the derivation writes it back out rather than
    preserving it. This is what makes the fix a fix at the invariant and not a veto."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    df.iat[ROW_CLIPPED, df.columns.get_loc("accepted")] = True   # the leak, pre-installed
    insp = _inspector(qtbot, session, df, doc_id, ch, prov)

    insp._accept_all_btn.click()
    assert bool(df["accepted"].iloc[ROW_CLIPPED]) is False


def test_inspector_controls_agree_with_the_frame_after_accept_all(qapp, qtbot, tmp_path):
    """The defect was a window claiming something its own frame denied. Syncing the
    checkbox from the REQUESTED value rather than from the row would move that defect one
    layer up instead of fixing it, so the displayed clipped event must still read
    ``accepted: no`` with the box unticked."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    insp = _inspector(qtbot, session, df, doc_id, ch, prov)
    insp.show_event(int(df["event_id"].iloc[ROW_CLIPPED]))

    insp._accept_all_btn.click()
    qtbot.wait(10)
    assert insp._accept_check.isChecked() is False
    assert "accepted:        no" in insp._readout.text()

    insp.show_event(int(df["event_id"].iloc[ROW_HAND]))
    assert insp._accept_check.isChecked() is True


def test_inspector_curated_column_is_the_and_not_a_broadcast(qapp, qtbot, tmp_path):
    """HUNK 2, at the seam. ``_curated_column`` is the whole column as the schema derives
    it under one bulk decision; rejecting needs no flags read at all."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    insp = _inspector(qtbot, session, df, doc_id, ch, prov)

    col = insp._curated_column(True)
    assert col.dtype == bool and list(col) == _EXPECTED_AFTER_ACCEPT
    assert list(insp._curated_column(False)) == [False] * 6

    # The column is still a bool column afterwards, not an object one: every reader of it
    # (the table model, the span overlays, the summary, Export events) calls
    # ``to_numpy(dtype=bool)`` on it, and the frozen schema declares it bool.
    insp._accept_all_btn.click()
    assert df["accepted"].dtype == np.dtype(bool)
    insp._reject_all_btn.click()
    assert df["accepted"].dtype == np.dtype(bool)

    src = inspect.getsource(insp_mod.EventInspectorWindow._set_all_accepted)
    assert 'self._df["accepted"] = bool(accepted)' not in src, "the broadcast is back"
    assert "_curated_column" in src


# ================================================================================================
# 3. the Event Gallery — every gesture goes through the same choke point
# ================================================================================================

def test_gallery_accept_all_shown_leaves_the_detectors_rejections_rejected(
        qapp, qtbot, tmp_path):
    """HUNK 3, through the real button. M6c: the label promises what the click will
    actually do — three of the six tiles shown are the detector's own rejections, so it
    reads "Accept 3 of 6"."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    gal = _gallery(qtbot, session, df, doc_id, ch)

    assert gal.accept_all_btn.text() == "Accept 3 of 6"
    gal.accept_all_btn.click()
    assert _accepted(df) == _EXPECTED_AFTER_ACCEPT


def test_gallery_accept_selected_leaves_the_detectors_rejections_rejected(
        qapp, qtbot, tmp_path):
    """The A key / "Accept selected" button reaches the same writer, so it is guarded by
    the same rule -- a guard installed in one gesture is a guard missing from the others."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    gal = _gallery(qtbot, session, df, doc_id, ch)

    _select_df_rows(gal, [ROW_CLIPPED, ROW_TOO_LONG, ROW_HAND])
    gal.accept_selected_btn.click()
    assert _accepted(df) == [False, False, True, True, True, False]


def test_gallery_space_inverts_without_accepting_a_flagged_tile(qapp, qtbot, tmp_path):
    """Space on an all-rejected selection means "accept these" -- and must not mean it for
    the row whose flag says the measurement is not one."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    gal = _gallery(qtbot, session, df, doc_id, ch)

    _select_df_rows(gal, [ROW_CLIPPED, ROW_HAND])
    gal._toggle_selected()                    # nothing selected is accepted -> accept them
    assert bool(df["accepted"].iloc[ROW_CLIPPED]) is False
    assert bool(df["accepted"].iloc[ROW_HAND]) is True


def test_gallery_reject_all_shown_is_never_blocked(qapp, qtbot, tmp_path):
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    gal = _gallery(qtbot, session, df, doc_id, ch)
    gal.reject_all_btn.click()
    assert _accepted(df) == [False] * 6


def test_gallery_announces_what_it_wrote_and_not_what_was_asked_for(qapp, qtbot, tmp_path):
    """HUNK 4. The owner mirrors ``acceptanceChanged`` straight onto the EventTable's
    checkbox, so emitting the REQUEST would re-open the leak one signal later and one panel
    over: the frame says rejected, the table's box ticks anyway.

    The case that produces a single-row change here is the forbidden state being repaired
    (a clipped row sitting ``accepted=True``), which is exactly the case that must not
    announce ``True``."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    df.iat[ROW_CLIPPED, df.columns.get_loc("accepted")] = True
    gal = _gallery(qtbot, session, df, doc_id, ch)

    seen: list[tuple] = []
    gal.acceptanceChanged.connect(lambda eid, ok: seen.append((int(eid), bool(ok))))
    _select_df_rows(gal, [ROW_CLIPPED])
    gal.accept_selected_btn.click()

    assert bool(df["accepted"].iloc[ROW_CLIPPED]) is False
    assert seen == [(int(df["event_id"].iloc[ROW_CLIPPED]), False)]


def test_gallery_says_on_the_status_line_how_many_tiles_it_kept_rejected(
        qapp, qtbot, tmp_path):
    """A refused gesture that moves nothing and says nothing is a silent failure. The note
    is NOT transient (M6d — it counts the rows currently refused, not the last gesture) and
    must not double when the gesture is repeated."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    gal = _gallery(qtbot, session, df, doc_id, ch)

    _select_df_rows(gal, [ROW_CLIPPED])
    gal.accept_selected_btn.click()           # already rejected: nothing changes at all
    first = gal.status.text()
    assert "1 tile kept rejected" in first
    gal.accept_selected_btn.click()
    assert gal.status.text().count("kept rejected") == 1

    gal.accept_all_btn.click()                # three refusals in one gesture
    assert "3 tiles kept rejected" in gal.status.text()

    _select_df_rows(gal, [ROW_HAND])
    gal.accept_selected_btn.click()           # an allowed acceptance refuses nothing...
    # ...and retires nothing either. An accept re-decides ITS OWN SCOPE; the three rows the
    # detector could not measure are untouched by it and are still refused (M6d). The
    # assertion here used to be `"kept rejected" not in ...`, i.e. the app going silent
    # about three still-refused rows the moment an unrelated row was accepted.
    assert "3 tiles kept rejected" in gal.status.text(), gal.status.text()


def test_gallery_reflect_acceptance_stays_an_inbound_mirror(qapp, qtbot, tmp_path):
    """NOT guarded, deliberately. ``reflect_acceptance`` is told what another panel has
    already written to the shared frame; a guard here would make this window disagree with
    the frame it is mirroring, which is a worse failure than the one being fixed. The
    curation ACTIONS are the surface that owes the invariant."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df, prov = _detection(session, doc_id, ch)
    gal = _gallery(qtbot, session, df, doc_id, ch)

    fired: list = []
    gal.acceptanceChanged.connect(lambda *a: fired.append(a))
    eid = int(df["event_id"].iloc[ROW_CLIPPED])
    df.iat[ROW_CLIPPED, df.columns.get_loc("accepted")] = True   # "the table wrote this"
    gal.reflect_acceptance(eid, True)
    assert bool(df["accepted"].iloc[ROW_CLIPPED]) is True
    assert fired == []


# ================================================================================================
# 4. the two surfaces agree with each other
# ================================================================================================

def test_both_windows_land_on_the_same_column_from_the_same_start(qapp, qtbot, tmp_path):
    """The inspector's "Accept all" (whole frame) and the gallery's "Accept all shown"
    under the All filter are the same question asked twice; they must not answer it
    differently."""
    session, doc_id, ch = _session_with_doc(tmp_path)
    df_a, prov = _detection(session, doc_id, ch)
    insp = _inspector(qtbot, session, df_a, doc_id, ch, prov)
    insp._accept_all_btn.click()

    session_b, doc_b, ch_b = _session_with_doc(tmp_path)
    df_b, _ = _detection(session_b, doc_b, ch_b)
    gal = _gallery(qtbot, session_b, df_b, doc_b, ch_b)
    gal.accept_all_btn.click()

    assert _accepted(df_a) == _accepted(df_b) == _EXPECTED_AFTER_ACCEPT
