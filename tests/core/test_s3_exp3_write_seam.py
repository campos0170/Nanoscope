"""The write seam's two promises, pinned for EVERY writer instead of for one of them.

Two defects, one shape: ``core/atomic_write`` states a guarantee for "every file the app
writes over user data" and implemented it at a door only one caller walked through
knowingly.

EXP3-02 — **the identity disclosure was an opt-in.** ``identity_notes`` was computed only
when a caller passed a ``notes`` list, and ``core.project.save_project`` was the only caller
in the app that did. ``docs/USER_GUIDE.md`` §11 makes the promise for "the project, the
calibration sidecar, exported events, exported figures"; the module docstring says the
hard-link trade "is deliberate — and instead of being silent it is REPORTED, through
``notes``". A researcher who hard-links an exported ``events.csv`` into a manuscript folder
got a manuscript copy that silently stopped tracking at the first re-export, while the same
situation on a ``.nanoscope`` produced a dialog in the same session. These tests pin the
disclosure at the SEAM, for every entry point and for the in-app writers that never asked.

EXP3-05 — **a failure named the hidden staging file.** A save or export into a folder the
process cannot write fails at the temporary, so the raw ``OSError`` the GUI prints verbatim
read ``Permission denied: '/…/.events.csv.saving-83391-d33ad5de.csv'`` — a path the user
never chose and that does not exist. ``temp_sibling``'s own docstring names that as the harm
the design must not inflict. These tests pin the restatement: the file the user named, the
reason, and an action — and never the temp.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.core import atomic_write
from nanoscope.core.atomic_write import (StagedSet, peek_identity_notes, staged_path,
                                         staged_write, staged_write_set,
                                         take_identity_notes)

#: The shape of a staging sibling, exactly as the reporter's repro matched it.
TEMP_RE = re.compile(r"/\.[^/']*\.saving-\d+-[0-9a-f]{8}")


@pytest.fixture(autouse=True)
def _drained():
    """The ledger is process-wide. Start every test with it empty so a note asserted here is
    a note THIS test's write produced, and leave it empty for the next one."""
    take_identity_notes()
    yield
    take_identity_notes()


@pytest.fixture
def readonly_dir(tmp_path: Path) -> Path:
    """A folder the process cannot write into — a mounted share, a read-only volume, a
    folder owned by someone else. Restored so pytest can clean up."""
    d = tmp_path / "archive_share"
    d.mkdir()
    d.chmod(0o555)
    yield d
    d.chmod(0o755)


# ================================================================================================
# EXP3-02 — the identity disclosure is produced by every seam, asked for or not
# ================================================================================================

def test_staged_write_reports_a_hard_link_without_being_asked(tmp_path: Path):
    """The events-export shape: a file with a second name, replaced. No ``notes=`` anywhere."""
    target = tmp_path / "events.csv"
    target.write_bytes(b"first export\n")
    alias = tmp_path / "table_S1_events.csv"
    os.link(target, alias)

    with staged_write(target) as fh:
        fh.write(b"second export\n")

    # The trade itself is unchanged and correct: the other name kept the previous export.
    assert alias.read_bytes() == b"first export\n"
    assert target.stat().st_ino != alias.stat().st_ino
    # ...and it is no longer silent.
    notes = take_identity_notes()
    assert any("hard link" in n for n in notes), notes
    assert any("events.csv" in n for n in notes), notes


def test_staged_path_reports_a_hard_link_without_being_asked(tmp_path: Path):
    """The figure/measurement shape — ``to_csv``/``savefig`` opening the staged path."""
    target = tmp_path / "figures_dwell.csv"
    target.write_text("a,b\n1,2\n")
    alias = tmp_path / "fig1_source_data.csv"
    os.link(target, alias)

    with staged_path(target) as staged:
        Path(staged).write_text("a,b\n3,4\n")

    assert alias.read_text() == "a,b\n1,2\n"
    assert any("hard link" in n for n in take_identity_notes())


def test_a_staged_set_reports_a_hard_link_without_being_asked(tmp_path: Path):
    """The Export-figures shape: a bare ``StagedSet()``, which is what MainWindow builds."""
    target = tmp_path / "figures_dwell.pdf"
    target.write_bytes(b"old figure")
    os.link(target, tmp_path / "fig1.pdf")

    with staged_write_set() as group:
        Path(group.stage(target)).write_bytes(b"new figure")

    assert (tmp_path / "fig1.pdf").read_bytes() == b"old figure"
    assert any("hard link" in n for n in take_identity_notes())


def test_a_followed_symlink_is_reported_by_every_seam(tmp_path: Path):
    """"The save says so the first time — the file that changed is not the one you clicked."
    Three seams, three writes, three notes."""
    real = tmp_path / "real"
    real.mkdir()
    for i, writer in enumerate(("w", "p", "s")):
        destination = real / f"target{i}.csv"
        destination.write_text("old\n")
        link = tmp_path / f"link{i}.csv"
        link.symlink_to(destination)

        if writer == "w":
            with staged_write(link) as fh:
                fh.write(b"new\n")
        elif writer == "p":
            with staged_path(link) as staged:
                Path(staged).write_text("new\n")
        else:
            with staged_write_set() as group:
                Path(group.stage(link)).write_text("new\n")

        assert link.is_symlink(), "the link itself must survive the write"
        assert destination.read_text() == "new\n"
        notes = take_identity_notes()
        assert any(f"link{i}.csv" in n and str(destination) in n for n in notes), notes


def test_an_ordinary_write_says_nothing(tmp_path: Path):
    """Silence means "nothing unusual", never "not checked" — so an ordinary export must not
    fill the ledger with noise the next disclosure would have to be sifted out of."""
    target = tmp_path / "plain.csv"
    with staged_write(target) as fh:
        fh.write(b"x\n")
    with staged_path(tmp_path / "plain2.csv") as staged:
        Path(staged).write_text("x\n")
    assert peek_identity_notes() == ()


def test_a_caller_that_takes_delivery_is_not_double_reported(tmp_path: Path):
    """The project save's channel is unchanged, and its notes do NOT also queue in the ledger.

    A caller that passes ``notes`` has somewhere to put them and is going to — leaving a copy
    behind would let the next surface to drain the ledger repeat a disclosure the user has
    already read, attached to a different file. The ledger holds what nobody has taken."""
    target = tmp_path / "study.nanoscope"
    target.write_bytes(b"old")
    os.link(target, tmp_path / "study_backup.nanoscope")

    notes: list[str] = []
    with staged_write(target, notes=notes) as fh:
        fh.write(b"new")

    assert any("hard link" in n for n in notes)
    assert peek_identity_notes() == ()


def test_the_project_save_still_reports_through_its_own_channel(tmp_path: Path):
    """``core.project.save_project`` is the one writer that already had a place to show the
    sentences (MainWindow puts them in a dialog). Nothing about that changed."""
    from nanoscope.core.project import ProjectState, save_project

    target = tmp_path / "study.nanoscope"
    save_project(target, ProjectState())
    os.link(target, tmp_path / "study_backup.nanoscope")

    notes: list[str] = []
    save_project(target, ProjectState(), notes)
    assert any("hard link" in n and "study.nanoscope" in n for n in notes), notes
    assert peek_identity_notes() == ()


def test_taking_the_notes_drains_them(tmp_path: Path):
    """A note that has been shown must not be shown again beside the NEXT export."""
    target = tmp_path / "e.csv"
    target.write_bytes(b"1")
    os.link(target, tmp_path / "e_alias.csv")
    with staged_write(target) as fh:
        fh.write(b"2")

    assert peek_identity_notes()          # peek does not drain
    first = take_identity_notes()
    assert first
    assert take_identity_notes() == []
    assert peek_identity_notes() == ()


def test_the_ledger_cannot_grow_without_bound(tmp_path: Path):
    """It is a disclosure channel, not a log. A long unattended session must not accumulate.

    The cap still holds; what changed is that reaching it is no longer silent — see
    ``test_s3_exp3_ledger_scope.py`` for the sentence itself."""
    for i in range(atomic_write.LEDGER_MAX + 20):
        target = tmp_path / f"f{i}.bin"
        target.write_bytes(b"0")
        os.link(target, tmp_path / f"f{i}_alias.bin")
        with staged_write(target) as fh:
            fh.write(b"1")
    pending = [n for n in peek_identity_notes() if "discarded" not in n]
    assert len(pending) == atomic_write.LEDGER_MAX


# ------------------------------------------------------------------------------------------------
# ...and the writers that never asked. These are the surfaces USER_GUIDE §11 names.
# ------------------------------------------------------------------------------------------------

def test_the_calibration_sidecar_now_discloses(tmp_path: Path):
    """``calibration.save_sidecar`` has no ``notes`` parameter AT ALL, so before the seam
    change the disclosure was not merely unused there — it was unreachable."""
    from nanoscope.core.calibration import (CalibrationSet, ChannelCalibration,
                                            save_sidecar, sidecar_path)
    from nanoscope.core.types import ChannelInfo, ChannelKind

    data = tmp_path / "rec.csv"
    data.write_text("time,current\n0,1\n")
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    calib = CalibrationSet()
    # A real, non-default gain — otherwise the sidecar is DELETED rather than written.
    calib.set("doc-1", 0, ChannelCalibration(gain=2500.0, target_units="pA"))

    save_sidecar(data, "doc-1", calib, (info,))
    first = sidecar_path(data)
    assert first.is_file()
    take_identity_notes()                     # the first write has nothing to disclose
    alias = tmp_path / "rec_calibration_backup.json"
    os.link(first, alias)

    save_sidecar(data, "doc-1", calib, (info,))
    assert any("hard link" in n for n in take_identity_notes())


def test_the_measurement_csv_export_now_discloses(tmp_path: Path):
    """Manual measurement ▸ Export CSV… — ``analysis.measure`` stages through ``staged_path``
    and passes no notes."""
    from nanoscope.analysis.measure import write_measurements_csv

    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = Path(write_measurements_csv(df, tmp_path / "measurements.csv"))
    take_identity_notes()
    os.link(out, tmp_path / "measurements_for_paper.csv")

    write_measurements_csv(df, out)
    assert any("hard link" in n for n in take_identity_notes())


def test_the_events_export_now_discloses(tmp_path: Path):
    """Analysis ▸ Export events… — ``scoring.write_events_export`` builds a bare
    ``staged_write_set()``. This is the exact file the reporter hard-linked."""
    from nanoscope.analysis.scoring import write_events_export

    df = pd.DataFrame({
        "event_id": np.arange(3),
        "start_s": np.linspace(0.0, 1.0, 3),
        "dwell_s": np.full(3, 1e-3),
        "depth_pA": np.full(3, 20.0),
        "accepted": np.ones(3, dtype=bool),
    })
    out = write_events_export(df, tmp_path / "events.csv")
    take_identity_notes()
    os.link(out["data"], tmp_path / "table_S1_events.csv")

    write_events_export(df, tmp_path / "events.csv")
    notes = take_identity_notes()
    assert any("hard link" in n and "events.csv" in n for n in notes), notes


# ================================================================================================
# EXP3-05 — a failure names the file the USER chose
# ================================================================================================

def _assert_addressed_to(exc: BaseException, target: Path) -> None:
    text = str(exc)
    assert not TEMP_RE.search(text), f"the message names a hidden staging file: {text}"
    assert str(target) in text, text
    assert target.name in text, text


def test_staged_write_into_an_unwritable_folder_names_the_chosen_file(readonly_dir: Path):
    target = readonly_dir / "study.nanoscope"
    with pytest.raises(PermissionError) as excinfo:
        with staged_write(target) as fh:
            fh.write(b"x")
    _assert_addressed_to(excinfo.value, target)
    # An action, the way refuse_if_write_protected has one.
    assert "write protection" in str(excinfo.value)
    assert not list(readonly_dir.iterdir()), "nothing may be left behind"


def test_staged_path_into_an_unwritable_folder_names_the_chosen_file(readonly_dir: Path):
    """The caller opens the staged path itself, so the OSError is raised inside the block."""
    target = readonly_dir / "measurements.csv"
    with pytest.raises(PermissionError) as excinfo:
        with staged_path(target) as staged:
            Path(staged).write_text("a,b\n")
    _assert_addressed_to(excinfo.value, target)


def test_staging_a_set_member_refuses_before_handing_out_a_path(readonly_dir: Path):
    """``StagedSet.stage`` returns a path the CALLER writes, outside anything this class can
    wrap — so an unwritable folder used to surface as ``savefig``'s own OSError naming the
    staging file. It is now refused by ``stage`` itself, before a member is written."""
    target = readonly_dir / "figures_dwell.pdf"
    group = StagedSet()
    with pytest.raises(PermissionError) as excinfo:
        group.stage(target)
    _assert_addressed_to(excinfo.value, target)
    assert group.targets == ()
    assert not list(readonly_dir.iterdir()), "the probe must not survive"


def test_the_restatement_keeps_the_original_as_its_cause(readonly_dir: Path):
    """The errno and the staging path are still there for whoever is debugging the app
    rather than using it — they are just not what the user is shown."""
    target = readonly_dir / "events.csv"
    with pytest.raises(PermissionError) as excinfo:
        with staged_write(target) as fh:
            fh.write(b"x")
    cause = excinfo.value.__cause__
    assert isinstance(cause, OSError) and cause.errno is not None
    assert TEMP_RE.search(str(cause)), "the cause is the original, temp path and all"


def test_a_read_only_target_still_gets_its_own_sentence(tmp_path: Path):
    """The neighbouring refusal is untouched: it already named the file and an action, and
    it must not be restated into the folder-level wording."""
    target = tmp_path / "events.csv"
    target.write_bytes(b"x")
    target.chmod(stat.S_IRUSR)
    try:
        with pytest.raises(PermissionError) as excinfo:
            with staged_write(target) as fh:
                fh.write(b"y")
    finally:
        target.chmod(0o644)
    assert "marked read-only and was not changed" in str(excinfo.value)
    assert not TEMP_RE.search(str(excinfo.value))


def test_a_failure_that_never_touched_the_staging_file_is_passed_through(tmp_path: Path):
    """Only a message that pointed at the hidden file is re-addressed. A caller's own
    ValueError must arrive exactly as it was raised — restating it would be the app
    inventing a filesystem story for something that is not one."""
    boom = ValueError("the frame has no dwell column")
    with pytest.raises(ValueError) as excinfo:
        with staged_path(tmp_path / "x.csv"):
            raise boom
    assert excinfo.value is boom
