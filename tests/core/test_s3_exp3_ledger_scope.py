"""EXP3-02, second pass — the identity DISCLOSURE CHANNEL itself has to be honest.

The first pass made ``identity_notes`` unconditional at every seam, so a writer inherits the
hard-link / followed-symlink disclosure by passing through the door instead of by remembering
to ask (``test_s3_exp3_write_seam.py``). Verifying that fix measured three ways the channel
could still lie to the surface that eventually drains it, all of them at the seam and none of
them visible from any single call site:

1. **A note for a write that never happened.** ``StagedSet.stage`` recorded the sentences as
   soon as a member was reserved. A figure export whose every figure fails to render abandons
   the set — nothing is placed, the user is told so — and the sentences stayed queued, ready
   to appear beside the next export as "those other names still hold the PREVIOUS contents"
   for a file this app never wrote. ``staged_write``/``staged_path`` had the same shape: they
   recorded before the ``os.replace`` that may raise.
2. **A note beside the wrong file.** The ledger held bare sentences with no idea which write
   produced them, so a drain after an events export would hand out a sentence about a
   calibration sidecar written ten minutes earlier for another document.
3. **A note discarded in silence.** Past ``LEDGER_MAX`` the oldest entries were deleted with
   no trace — measured at 84 hard-linked writes: 64 kept, 20 gone unread. That is exactly the
   silent-failure shape the channel exists to close, one level up, and it is unrecoverable:
   an atomic replace leaves nothing behind that could reconstruct the link it broke.

Every test here fails on the pristine module.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanoscope.core import atomic_write
from nanoscope.core.atomic_write import (StagedSet, peek_identity_notes, staged_path,
                                         staged_write, staged_write_set,
                                         take_identity_notes)


@pytest.fixture(autouse=True)
def _drained():
    """The ledger is process-wide: start and end empty so a note asserted here is this
    test's."""
    take_identity_notes()
    yield
    take_identity_notes()


def _hard_linked(dirpath: Path, name: str, body: bytes = b"old") -> Path:
    """A file with a second name on the same disk — the situation §11 promises to disclose."""
    target = dirpath / name
    target.write_bytes(body)
    os.link(target, dirpath / f"{target.stem}_alias{target.suffix}")
    return target


# ------------------------------------------------------------------------------------------------
# 1. a note describes a write that actually happened
# ------------------------------------------------------------------------------------------------

def test_an_abandoned_set_leaves_no_note_behind(tmp_path: Path):
    """The all-figures-failed ending: nothing was placed, so there is nothing to disclose."""
    target = _hard_linked(tmp_path, "figures_dwell.pdf")

    group = StagedSet()
    Path(group.stage(target)).write_bytes(b"new")
    assert peek_identity_notes() == (), "staging is not writing; nothing to say yet"
    group.abandon()

    assert target.read_bytes() == b"old", "the abandon must leave the file alone"
    assert peek_identity_notes() == ()


def test_a_set_that_raises_on_its_way_leaves_no_note_behind(tmp_path: Path):
    """``staged_write_set`` abandons on any exception — the same ending, through the door
    the exports actually use."""
    target = _hard_linked(tmp_path, "events.csv")

    with pytest.raises(RuntimeError):
        with staged_write_set() as group:
            Path(group.stage(target)).write_bytes(b"new")
            raise RuntimeError("the renderer blew up")

    assert peek_identity_notes() == ()


def test_a_withdrawn_member_takes_its_note_with_it(tmp_path: Path):
    """``unstage`` exists so one failing family does not sink the healthy ones. The family
    that withdrew was not written, so its sentences leave with it — and the survivors' do
    not."""
    withdrawn = _hard_linked(tmp_path, "figures_dwell.pdf")
    kept = _hard_linked(tmp_path, "figures_iv.pdf")

    group = StagedSet()
    Path(group.stage(withdrawn)).write_bytes(b"new")
    Path(group.stage(kept)).write_bytes(b"new")
    group.unstage([withdrawn])
    group.commit()

    notes = take_identity_notes()
    assert any("figures_iv.pdf" in n for n in notes), notes
    assert not any("figures_dwell.pdf" in n for n in notes), notes


def test_a_failed_single_write_leaves_no_note_behind(tmp_path: Path):
    """``staged_write`` reads ``st_nlink`` before the replace, but must not SAY it before
    the replace: the block raised, the file is untouched, its other names are not stale."""
    target = _hard_linked(tmp_path, "study.nanoscope")

    with pytest.raises(RuntimeError):
        with staged_write(target) as fh:
            fh.write(b"half")
            raise RuntimeError("interrupted")

    assert target.read_bytes() == b"old"
    assert peek_identity_notes() == ()


def test_a_staged_path_that_is_never_written_leaves_no_note_behind(tmp_path: Path):
    """The ``savefig`` shape: the caller was handed a path and wrote nothing to it."""
    target = _hard_linked(tmp_path, "figures_dwell.csv")

    with pytest.raises(FileNotFoundError):
        with staged_path(target):
            pass

    assert peek_identity_notes() == ()


def test_a_write_that_lands_still_says_so(tmp_path: Path):
    """The control for all of the above: the disclosure is deferred, not lost."""
    target = _hard_linked(tmp_path, "events.csv")
    with staged_write(target) as fh:
        fh.write(b"new")
    assert any("hard link" in n for n in take_identity_notes())


# ------------------------------------------------------------------------------------------------
# 2. a note knows which file it is about
# ------------------------------------------------------------------------------------------------

def test_a_drain_scoped_to_one_export_does_not_take_another_files_note(tmp_path: Path):
    """Two documents, two writes, one process. The events export must not print the
    sidecar's sentence, and the sidecar's sentence must not be lost by being printed."""
    sidecar = _hard_linked(tmp_path, "rec.csv.npcal.json")
    events = _hard_linked(tmp_path, "events.csv")
    for path in (sidecar, events):
        with staged_write(path) as fh:
            fh.write(b"new")

    mine = take_identity_notes(for_paths=[events])
    assert mine and all("events.csv" in n for n in mine), mine

    still_pending = peek_identity_notes()
    assert still_pending and all("npcal" in n for n in still_pending), still_pending


def test_a_scoped_drain_finds_the_note_under_the_name_the_caller_used(tmp_path: Path):
    """A write through a symlink is filed under the DESTINATION (that is the file that
    changed). A surface that only knows the path it handed the seam must still find it."""
    real_dir = tmp_path / "dated"
    real_dir.mkdir()
    destination = _hard_linked(real_dir, "study.nanoscope")
    link = tmp_path / "current.nanoscope"
    link.symlink_to(destination)

    with staged_write(link) as fh:
        fh.write(b"new")

    notes = take_identity_notes(for_paths=[link])
    assert any("current.nanoscope" in n and str(destination) in n for n in notes), notes
    assert any("hard link" in n for n in notes), notes
    assert peek_identity_notes() == ()


def test_one_path_may_be_passed_bare(tmp_path: Path):
    """A surface that wrote one file will spell it that way. Iterating a bare ``str`` would
    scope the drain to its characters — silence indistinguishable from "nothing to say"."""
    target = _hard_linked(tmp_path, "events.csv")
    with staged_write(target) as fh:
        fh.write(b"new")
    assert take_identity_notes(for_paths=str(target))

    # The first replace broke the link (that is the trade being disclosed), so make a new
    # one before asking the same question with a Path.
    os.link(target, tmp_path / "events_alias2.csv")
    with staged_write(target) as fh:
        fh.write(b"newer")
    assert take_identity_notes(for_paths=target)


def test_an_unscoped_drain_still_takes_everything(tmp_path: Path):
    """A surface that speaks for the whole session (a quit-time summary) is unchanged."""
    for name in ("a.csv", "b.csv"):
        with staged_write(_hard_linked(tmp_path, name)) as fh:
            fh.write(b"new")
    assert len(take_identity_notes()) == 2
    assert peek_identity_notes() == ()


def test_a_scoped_peek_matches_what_a_scoped_take_would_return(tmp_path: Path):
    with staged_write(_hard_linked(tmp_path, "a.csv")) as fh:
        fh.write(b"new")
    with staged_write(_hard_linked(tmp_path, "b.csv")) as fh:
        fh.write(b"new")

    scope = [tmp_path / "b.csv"]
    peeked = peek_identity_notes(for_paths=scope)
    assert peeked == tuple(take_identity_notes(for_paths=scope))


def test_a_set_files_each_members_note_under_that_member(tmp_path: Path):
    """The figure export writes twelve files in one set; a per-file drain must still work."""
    dwell = _hard_linked(tmp_path, "figures_dwell.pdf")
    iv = _hard_linked(tmp_path, "figures_iv.pdf")
    with staged_write_set() as group:
        for member in (dwell, iv):
            Path(group.stage(member)).write_bytes(b"new")

    only_iv = take_identity_notes(for_paths=[iv])
    assert only_iv and all("figures_iv.pdf" in n for n in only_iv), only_iv
    assert all("figures_dwell.pdf" in n for n in peek_identity_notes())


# ------------------------------------------------------------------------------------------------
# 3. nothing is dropped in silence
# ------------------------------------------------------------------------------------------------

def test_the_cap_says_how_many_disclosures_it_threw_away(tmp_path: Path):
    """84 hard-linked writes, a 64-entry cap: the 20 that fell off are named as a count, not
    dropped. Silence here would be the very failure this channel exists to end."""
    overflow = 20
    for i in range(atomic_write.LEDGER_MAX + overflow):
        with staged_write(_hard_linked(tmp_path, f"f{i}.bin")) as fh:
            fh.write(b"new")

    notes = take_identity_notes()
    assert str(overflow) in notes[0], notes[0]
    assert "discarded" in notes[0]
    assert "cannot be recovered" in notes[0]
    assert len([n for n in notes if "discarded" not in n]) == atomic_write.LEDGER_MAX


def test_the_discard_disclosure_is_delivered_once(tmp_path: Path):
    """It is a disclosure, so it drains like one: shown once, never twice."""
    for i in range(atomic_write.LEDGER_MAX + 3):
        with staged_write(_hard_linked(tmp_path, f"f{i}.bin")) as fh:
            fh.write(b"new")

    first = take_identity_notes()
    assert any("discarded" in n for n in first)
    assert take_identity_notes() == []


def test_a_scoped_drain_also_carries_the_discard_disclosure(tmp_path: Path):
    """A surface draining its own file is still the next surface to speak; withholding the
    count until some unscoped drain that may never come is the silence again."""
    for i in range(atomic_write.LEDGER_MAX + 5):
        with staged_write(_hard_linked(tmp_path, f"f{i}.bin")) as fh:
            fh.write(b"new")
    mine = _hard_linked(tmp_path, "events.csv")
    with staged_write(mine) as fh:
        fh.write(b"new")

    notes = take_identity_notes(for_paths=[mine])
    assert any("discarded" in n for n in notes), notes
    assert any("events.csv" in n for n in notes), notes


def test_an_ordinary_session_never_mentions_a_discard(tmp_path: Path):
    """The count counts what it says: no drop, no sentence."""
    with staged_write(_hard_linked(tmp_path, "events.csv")) as fh:
        fh.write(b"new")
    assert not any("discarded" in n for n in take_identity_notes())


# ------------------------------------------------------------------------------------------------
# ...through the writers the app actually calls, not only the primitive
# ------------------------------------------------------------------------------------------------

def _events_frame(n: int = 3):
    import numpy as np
    import pandas as pd
    return pd.DataFrame({
        "event_id": np.arange(n),
        "start_s": np.linspace(0.0, 1.0, n),
        "dwell_s": np.full(n, 1e-3),
        "depth_pA": np.full(n, 20.0),
        "accepted": np.ones(n, dtype=bool),
    })


def test_an_in_app_export_files_its_note_under_its_own_path(tmp_path: Path):
    """``Analysis ▸ Export events…``: the surface that will print the sentence knows the
    path it asked for, and draining by that path is what it will do."""
    from nanoscope.analysis.scoring import write_events_export

    df = _events_frame()
    out = write_events_export(df, tmp_path / "events.csv")
    take_identity_notes()
    os.link(out["data"], tmp_path / "table_S1_events.csv")

    write_events_export(df, tmp_path / "events.csv")
    notes = take_identity_notes(for_paths=[tmp_path / "events.csv"])
    assert any("hard link" in n and "events.csv" in n for n in notes), notes
    assert peek_identity_notes() == ()


def test_an_in_app_export_that_refuses_leaves_no_note_behind(tmp_path: Path):
    """The export was refused (its folder is not writable), so nothing about the target's
    identity changed — and the next export must not inherit a sentence saying it did."""
    from nanoscope.analysis.scoring import write_events_export

    share = tmp_path / "archive_share"
    share.mkdir()
    victim = _hard_linked(share, "events.csv")
    share.chmod(0o555)
    try:
        with pytest.raises(OSError) as caught:
            write_events_export(_events_frame(), victim)
        assert "events.csv" in str(caught.value)          # EXP3-05: the chosen file
        assert ".saving-" not in str(caught.value)        # ...never the staging sibling
        assert peek_identity_notes() == ()
    finally:
        share.chmod(0o755)
