"""§2.5 / §2.7 / §2.10 — nothing the app writes may destroy what it did not author.

One guarantee, three failures, and they are all the same shape at different scales:

* **§2.7** a write over ONE file silently changed WHICH file it wrote — a symlink became a
  regular file (the canonical project keeping the previous save), extended attributes were
  destroyed on every save, a 239-character name could not be staged at all, and a
  ``chmod 400`` project was overwritten without a word;
* **§2.10** a write over a SET of files landed part way — three new figures beside nine
  survivors of a previous export, under one ``figures_*`` stem, with no marker;
* **§2.5** a write of a VALUE landed part way — a project restore built a calibration with
  no ``note`` because the record had no such field, and the next ordinary *Apply* wrote that
  emptiness into the user's ``.npcal.json``.

PORT_PLAN rule 21 already stated the first of these for "every file the app writes over user
data", and was true of exactly one of them. The guarantee now lives in
:mod:`nanoscope.core.atomic_write`, and the AST guard at the bottom of this file is what
keeps the plan's sentence honest.

Every test here is written against the FAILURE, never the happy path.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import nanoscope  # noqa: F401
from nanoscope.core import atomic_write
from nanoscope.core.atomic_write import (
    StagedSet,
    resolve_write_target,
    staged_path,
    staged_write,
    staged_write_set,
    temp_sibling,
)
from nanoscope.core.calibration import (
    SIDECAR_WRITTEN,
    CalibrationSet,
    ChannelCalibration,
    notes_lost_by,
    save_sidecar,
    sidecar_path,
)
from nanoscope.core.project import (
    VERSION,
    CalibrationRec,
    DocumentRef,
    ProjectState,
    load_project,
    save_project,
)
from nanoscope.core.types import Provenance

PKG = pathlib.Path(nanoscope.__file__).resolve().parent


def _state(tag: str = "a") -> ProjectState:
    return ProjectState(documents=[DocumentRef(path=f"/data/{tag}.abf", loaded_range=None)])


def _xattrs(p) -> list[str]:
    """Extended-attribute names, read by the OS's own tool rather than by the code
    under test — a helper sharing the module's ctypes binding could hide a broken one."""
    if sys.platform != "darwin":
        return sorted(os.listxattr(str(p))) if hasattr(os, "listxattr") else []
    return sorted(subprocess.run(["xattr", str(p)], capture_output=True,
                                 text=True).stdout.split())


def _set_xattr(p, name: str, value: str) -> bool:
    if sys.platform == "darwin":
        return subprocess.run(["xattr", "-w", name, value, str(p)]).returncode == 0
    if not hasattr(os, "setxattr"):
        return False
    try:
        os.setxattr(str(p), name, value.encode())
        return True
    except OSError:
        return False


# ================================================================================================
# §2.7 — a write must not change WHICH file it wrote
# ================================================================================================
def test_saving_through_a_symlink_writes_the_file_the_link_points_at(tmp_path):
    """The measured failure: ``link.is_symlink()`` False, the real file's inode unchanged,
    and the canonical project still holding the OLD content — so the user's next open read
    the save they thought they had just made, and nothing said so."""
    real = tmp_path / "real.nanoscope"
    save_project(real, _state("first"))
    link = tmp_path / "link.nanoscope"
    link.symlink_to(real)
    inode_before = real.stat().st_ino

    save_project(link, _state("second"))

    assert link.is_symlink(), "the link was replaced by a regular file"
    assert real.stat().st_ino != inode_before, "the real file was never touched"
    assert load_project(real).documents[0].path == "/data/second.abf"
    assert load_project(link).documents[0].path == "/data/second.abf"


def test_a_symlinked_save_says_which_file_it_actually_wrote(tmp_path):
    """Resolving silently would be its own kind of surprise: the user named one path and a
    different one changed. The note is the only place that is stated."""
    real = tmp_path / "real.nanoscope"
    save_project(real, _state())
    link = tmp_path / "link.nanoscope"
    link.symlink_to(real)

    notes: list[str] = []
    save_project(link, _state("b"), notes)
    assert any("link.nanoscope" in n and str(real) in n for n in notes), notes


def test_an_ordinary_save_reports_nothing_at_all(tmp_path):
    """Silence has to MEAN something. A note on every save trains the user past all of them."""
    notes: list[str] = []
    save_project(tmp_path / "p.nanoscope", _state(), notes)
    save_project(tmp_path / "p.nanoscope", _state("b"), notes)
    assert notes == []


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"),
                    reason="extended attributes are a POSIX feature")
def test_extended_attributes_survive_a_save(tmp_path):
    """Measured before: ``['com.apple.provenance', 'user.note']`` → ``['com.apple.provenance']``
    on EVERY save. Finder tags and hand-set metadata on a project the user has kept for
    months, destroyed by opening and saving it."""
    p = tmp_path / "p.nanoscope"
    save_project(p, _state())
    if not _set_xattr(p, "user.note", "my tag"):
        pytest.skip("this filesystem does not support extended attributes")
    before = _xattrs(p)
    assert "user.note" in before

    save_project(p, _state("b"))

    assert _xattrs(p) == before
    assert load_project(p).documents[0].path == "/data/b.abf"


def test_a_name_at_the_filesystem_limit_can_still_be_saved(tmp_path):
    """``OSError[63] File name too long`` for a 239-character project — and the message named
    a hidden ``.tmp`` path the user has never seen, for a save they had every reason to
    expect to work."""
    long = tmp_path / ("L" * 239 + ".nanoscope")
    save_project(long, _state())
    assert load_project(long).documents[0].path == "/data/a.abf"
    save_project(long, _state("b"))
    assert load_project(long).documents[0].path == "/data/b.abf"


def test_the_temp_name_fits_the_filesystems_own_limit(tmp_path):
    name_max = os.pathconf(str(tmp_path), "PC_NAME_MAX")
    for length in (1, 60, 200, name_max - 11, name_max - 1):
        target = tmp_path / ("x" * length + ".nanoscope")
        tmp = temp_sibling(target)
        assert len(tmp.name.encode()) <= name_max, (length, tmp.name)
        assert tmp.parent == tmp_path


def test_a_read_only_project_is_not_silently_overwritten(tmp_path):
    """A ``chmod 400`` project was overwritten without a word: ``os.replace`` needs write
    permission on the DIRECTORY and pays no attention to the file's own mode, so the
    read-only bit was decorative. The refusal must name the file, not a temp."""
    p = tmp_path / "p.nanoscope"
    save_project(p, _state())
    os.chmod(p, 0o400)
    try:
        with pytest.raises(PermissionError) as exc:
            save_project(p, _state("b"))
        assert "p.nanoscope" in str(exc.value) and ".tmp" not in str(exc.value)
        assert load_project(p).documents[0].path == "/data/a.abf"
    finally:
        os.chmod(p, 0o600)


def test_a_hard_linked_project_is_saved_and_the_other_names_are_named(tmp_path):
    """An atomic replace is a NEW INODE by definition, so a second name for the same file
    necessarily goes stale. Writing in place is the only alternative and it is the
    destructive path the staging exists to forbid — so the trade stands, and is reported."""
    a = tmp_path / "a.nanoscope"
    save_project(a, _state())
    b = tmp_path / "b.nanoscope"
    os.link(a, b)

    notes: list[str] = []
    save_project(a, _state("b"), notes)

    assert load_project(a).documents[0].path == "/data/b.abf"
    assert any("hard link" in n for n in notes), notes


def test_the_saved_project_still_keeps_the_permissions_it_had(tmp_path):
    p = tmp_path / "p.nanoscope"
    save_project(p, _state())
    os.chmod(p, 0o640)
    save_project(p, _state("b"))
    assert (p.stat().st_mode & 0o777) == 0o640


def test_resolve_write_target_does_not_call_a_symlinked_DIRECTORY_a_followed_link(tmp_path):
    """``/tmp`` is a symlink to ``/private/tmp`` on every Mac. Reporting that would put a
    dialog in front of the user on ordinary saves, which is how a real warning gets ignored."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    resolved, followed = resolve_write_target(link_dir / "p.nanoscope")
    assert followed is None
    assert resolved == (real_dir / "p.nanoscope").resolve()


# ================================================================================================
# §2.7 — a write must still never half-land (the C13 rule, now in the shared door)
# ================================================================================================
def test_a_staged_write_that_raises_leaves_the_target_untouched(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"ORIGINAL")
    with pytest.raises(RuntimeError):
        with staged_write(p) as fh:
            fh.write(b"HALF")
            raise RuntimeError("disk died")
    assert p.read_bytes() == b"ORIGINAL"
    assert list(tmp_path.glob(".*tmp")) == []


def test_a_staged_path_that_was_never_written_is_an_error_not_an_empty_file(tmp_path):
    """A writer that silently produced nothing used to leave the caller believing it had
    written. An empty target is worse than a raised error, because it opens."""
    p = tmp_path / "f.csv"
    p.write_text("ORIGINAL")
    with pytest.raises(FileNotFoundError):
        with staged_path(p):
            pass
    assert p.read_text() == "ORIGINAL"


def test_the_staged_path_keeps_the_targets_suffix(tmp_path):
    """matplotlib picks its backend from the extension; a ``.tmp`` handed to ``savefig``
    raises instead of writing a PDF."""
    with staged_path(tmp_path / "figure.pdf") as staged:
        assert staged.suffix == ".pdf"
        staged.write_bytes(b"%PDF-1.4")
    assert (tmp_path / "figure.pdf").read_bytes() == b"%PDF-1.4"


# ================================================================================================
# §2.10 — a SET of files lands together or not at all
# ================================================================================================
def _seed(dirpath, names, text=b"OLD"):
    for n in names:
        (dirpath / n).write_bytes(text)


def test_a_set_whose_last_member_fails_replaces_nothing(tmp_path):
    """The measured failure, scaled down: ``figures_depth.pdf`` new while
    ``figures_depth.png`` and ``.csv`` are old, under one stem, with no marker."""
    names = ["f_depth.pdf", "f_depth.png", "f_depth.csv"]
    _seed(tmp_path, names)
    with pytest.raises(OSError):
        with staged_write_set() as group:
            group.stage(tmp_path / names[0]).write_bytes(b"NEW")
            group.stage(tmp_path / names[1]).write_bytes(b"NEW")
            raise OSError(28, "No space left on device")
    assert [(tmp_path / n).read_bytes() for n in names] == [b"OLD"] * 3
    assert list(tmp_path.glob(".*tmp")) == []


def test_a_set_that_completes_replaces_every_member(tmp_path):
    names = ["f_depth.pdf", "f_depth.png", "f_depth.csv"]
    _seed(tmp_path, names)
    with staged_write_set() as group:
        for n in names:
            group.stage(tmp_path / n).write_bytes(b"NEW")
    assert [(tmp_path / n).read_bytes() for n in names] == [b"NEW"] * 3


def test_a_member_that_was_staged_but_never_written_sinks_the_whole_set(tmp_path):
    names = ["a.txt", "b.txt"]
    _seed(tmp_path, names)
    group = StagedSet()
    group.stage(tmp_path / names[0]).write_bytes(b"NEW")
    group.stage(tmp_path / names[1])            # reserved, never written
    with pytest.raises(FileNotFoundError):
        group.commit()
    assert [(tmp_path / n).read_bytes() for n in names] == [b"OLD"] * 2


def test_a_commit_that_fails_PART_WAY_restores_what_it_had_already_replaced(tmp_path):
    """The narrow window the staging alone does not close: staging succeeded for every
    member and the replace itself failed on a later one. Without a rollback the folder
    splits exactly the way it split before, only more rarely."""
    names = ["a.txt", "b.txt", "c.txt"]
    _seed(tmp_path, names)
    group = StagedSet()
    for n in names:
        group.stage(tmp_path / n).write_bytes(b"NEW")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst, *a, **kw):
        if str(dst).endswith("c.txt"):
            raise OSError(28, "No space left on device")
        calls["n"] += 1
        return real_replace(src, dst, *a, **kw)

    atomic_write.os.replace = flaky
    try:
        with pytest.raises(OSError):
            group.commit()
    finally:
        atomic_write.os.replace = real_replace
    assert calls["n"] >= 1, "the test never reached the partial state it exists to cover"
    assert [(tmp_path / n).read_bytes() for n in names] == [b"OLD"] * 3
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)


def test_a_set_creates_files_that_did_not_exist_and_removes_them_all_on_a_rollback(tmp_path):
    names = ["a.txt", "b.txt"]
    group = StagedSet()
    for n in names:
        group.stage(tmp_path / n).write_bytes(b"NEW")

    real_replace = os.replace

    def flaky(src, dst, *a, **kw):
        if str(dst).endswith("b.txt"):
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *a, **kw)

    atomic_write.os.replace = flaky
    try:
        with pytest.raises(OSError):
            group.commit()
    finally:
        atomic_write.os.replace = real_replace
    assert list(tmp_path.iterdir()) == [], "a half-created set was left behind"


def test_a_failing_figure_family_leaves_no_mixed_vintage(tmp_path):
    """Through the real ``save_figure``: ENOSPC on the PNG used to leave a NEW pdf beside an
    OLD png and an OLD csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nanoscope.plot import figures

    names = ["fig_depth.pdf", "fig_depth.png", "fig_depth.csv"]
    _seed(tmp_path, names)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    real_savefig = type(fig).savefig
    calls = {"n": 0}

    def flaky(self, f, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        return real_savefig(self, f, *a, **kw)

    type(fig).savefig = flaky
    try:
        with pytest.raises(OSError):
            figures.save_figure(fig, tmp_path / "fig_depth.pdf",
                                data=pd.DataFrame({"a": [1.0, 2.0]}))
    finally:
        type(fig).savefig = real_savefig
        plt.close(fig)

    assert [(tmp_path / n).read_bytes() for n in names] == [b"OLD"] * 3
    assert list(tmp_path.glob(".*tmp")) == []


def test_the_events_export_and_its_provenance_sidecar_land_together(tmp_path):
    """The sidecar does not accompany the CSV, it DESCRIBES it. A table from today beside a
    sidecar from last month is not a partial export, it is one with a confidently wrong
    provenance record."""
    from nanoscope.analysis import scoring

    data = tmp_path / "events.csv"
    meta = tmp_path / "events.meta.json"
    data.write_text("OLD")
    meta.write_text('{"old": true}')

    real_to_csv = pd.DataFrame.to_csv

    def flaky(self, path_or_buf=None, *a, **kw):
        out = real_to_csv(self, path_or_buf, *a, **kw)
        raise OSError(28, "No space left on device")

    pd.DataFrame.to_csv = flaky
    try:
        with pytest.raises(OSError):
            scoring.write_events_export(pd.DataFrame({"start_idx": [1, 2]}), data)
    finally:
        pd.DataFrame.to_csv = real_to_csv

    assert data.read_text() == "OLD"
    assert meta.read_text() == '{"old": true}'


# ================================================================================================
# §2.5 — the project carries every field the value object has
# ================================================================================================
def test_every_ChannelCalibration_field_has_a_home_in_CalibrationRec():
    """THE invariant, and the reason `note` was lost.

    A field that exists on the value object and not in the record is not merely "not
    persisted": the restore builds the object with that field's DEFAULT, the panel's Apply
    faithfully carries the default forward, and it is written over the user's own file. The
    check is on the FIELD LISTS so the next field cannot be forgotten the same way.
    """
    import dataclasses

    value_fields = {f.name for f in dataclasses.fields(ChannelCalibration)}
    record_fields = {f.name for f in dataclasses.fields(CalibrationRec)}
    assert value_fields <= record_fields, (
        f"{sorted(value_fields - record_fields)} would be silently blanked by a restore")


def test_a_calibration_note_survives_a_project_round_trip(tmp_path):
    cal = ChannelCalibration(gain=0.25, offset=1.5, target_units="pA",
                             provenance=Provenance.MEASURED, note="1 MOhm standard, 2026-07-01")
    state = ProjectState(documents=[DocumentRef(path="/data/rec.abf")],
                         calibrations=[CalibrationRec.from_calibration("/data/rec.abf", 0, cal)])
    p = save_project(tmp_path / "p.nanoscope", state)
    back = load_project(p).calibrations[0].to_calibration()
    assert back == cal


def test_a_pre_v5_project_keeps_the_note_the_sidecar_already_installed(tmp_path):
    """THE measured failure. A v4 project has nothing to say about the note, and reading its
    silence as "there is none" is what blanked ``.npcal.json`` on the next ordinary Apply."""
    rec = CalibrationRec(doc_id="/data/rec.abf", channel=0, gain=0.75, offset=0.0,
                         target_units="pA", provenance="unknown")       # note is None
    assert rec.note is None
    existing = ChannelCalibration(gain=0.5, note="electrode replaced 2026-07-01")
    assert rec.to_calibration(existing).note == "electrode replaced 2026-07-01"
    assert rec.to_calibration(existing).gain == 0.75


def test_a_v5_project_that_records_an_EMPTY_note_really_means_there_is_none(tmp_path):
    """The other half of the three-state rule: "" is an answer, absent is not."""
    rec = CalibrationRec(doc_id="/d", channel=0, gain=0.75, offset=0.0,
                         target_units="", provenance="unknown", note="")
    assert rec.to_calibration(ChannelCalibration(note="stale")).note == ""


def test_re_saving_a_pre_v5_project_does_not_invent_an_empty_note(tmp_path):
    """Load a v4 project and save it: writing ``"note": ""`` for a record that knows nothing
    would turn "I have no opinion" into a confident "there is none", and the defect would
    come back one save later."""
    state = ProjectState(
        documents=[DocumentRef(path="/data/rec.abf")],
        calibrations=[CalibrationRec(doc_id="/data/rec.abf", channel=0, gain=0.75, offset=0.0,
                                     target_units="", provenance="unknown")])
    p = save_project(tmp_path / "p.nanoscope", state)
    import zipfile
    with zipfile.ZipFile(p) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert "note" not in manifest["calibrations"][0], manifest["calibrations"][0]
    assert load_project(p).calibrations[0].note is None


def test_the_project_version_moved_and_old_projects_still_load(tmp_path):
    """A pin, so the format version only ever moves ON PURPOSE.

    Both halves matter and they pull against each other: VERSION going up is what stops an
    OLDER build from opening a file it would silently strip on its next save, and
    MIN_READ_VERSION staying at 1 is what stops the bump from orphaning any project the
    owner already has.

    Moved to 6 for the `fits` list (the derived fit.dwell / fit.depth / fit.levels results,
    previously dropped on save without a word). `fits` is additive on READ, which is why it
    was first written without a bump -- wrongly: a build predating the key accepts a
    version-5 file, ignores the key and the `fits/<i>.parquet` members, and drops them when
    it next saves. Additive-ness governs whether THIS build can read OLDER files; it says
    nothing about whether older builds can safely read this one.

    Moved to 7 for `channel_name` / `channel_kind` on all four per-channel records. Same
    argument in both directions: additive on read (a v1-v6 record carries None and resolves
    by index exactly as before), and an older build would accept a v7 file, ignore the
    identity and strip it on its next save -- putting back the defect where a re-export that
    reorders the signal columns lands a pA calibration on the voltage channel.

    Moved to 8 for `CalibrationRec.kind`, the user's "treat this channel as" override. Note
    it is a DIFFERENT field from v7's `channel_kind`: that one records what the file header
    declared, this one the correction the user made when the header was wrong or silent.
    Same argument in both directions once more -- additive on read (a v1-v7 record carries
    None, meaning "no override, the header stands", and `to_calibration` then leaves the
    sidecar's own override alone), and an older build would accept a v8 file, ignore the
    key and strip the override on its next save, putting back the pane-ordering complaint
    it exists to fix.

    Moved to 9 for `colours`, the project's identity-keyed colour map
    (`core.layer.ColourRegistry`): which slot each recording+channel holds in the
    categorical cycle, and the ink the user assigned it by hand. Before it, a recording's
    colour was persisted for exactly ONE of the four surfaces that draw it (`layers[].color`
    carried the trace view's ink) while the Plots panel recoloured from POSITION on every
    refresh -- so "the Zn run is the green one" was true in one window and false in the
    next, and a hand-picked colour did not survive a reopen at all. Keyed by the PERSISTENT
    key like `calibrations` / `layers` / `results`, so it re-attaches to a re-derived
    segment. The same argument in both directions a fourth time: additive on read (a v1-v8
    project has no key and loads with an empty registry, which is visually what it last
    showed, since a project without one was drawing positional colours anyway), and an older
    build accepts a v9 file, ignores the key and strips every hand-picked colour on its next
    save.

    Moved to 11 for the analytical ``workspace``: ordered, stable-id plot/PCA/GMM/plugin
    artifacts, whose bulk DataFrame tables and opaque blobs live in archive members rather
    than manifest JSON. A v1-v10 project has no workspace and still opens as an empty one;
    a v10 reader must refuse v11 because accepting it would strip the entire workspace on
    the next save.

    Moved to 12 for new fields INSIDE three artifacts a v11 build already owns: the
    spectrogram colour limits and log-frequency axis (builtin.psd), the ensemble's subset,
    band and frozen alignment (builtin.event-plots), and the PlotConfig v3 KDE and heatmap
    sections (builtin.plots). A v11 build restores its own artifacts from the keys it knows
    and re-captures them on save, so it would show a heatmap as a histogram and strip the
    rest; v11's opaque carry-through covers only kinds and fields it does not own.

    Moved to 13 for named live FigureSpec records, their frozen resources and applied/draft
    recipes. A v12 build cannot restore or safely edit those records, so it must refuse
    the project rather than silently dropping live analytical views on its next save.
    """
    assert VERSION == 13
    from nanoscope.core.project import MIN_READ_VERSION
    assert MIN_READ_VERSION == 1


# ================================================================================================
# §2.5 — and the sidecar write is staged like everything else
# ================================================================================================
def test_the_sidecar_write_is_staged_not_a_truncate_in_place(tmp_path):
    """It was a ``path.write_text`` — on the exact file the removal confirmation exists to
    protect. A failure part way through left a user-owned calibration file that no longer
    parsed, and ``load_sidecar`` then discards the whole thing."""
    from nanoscope.core.types import ChannelInfo, ChannelKind

    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    sc = sidecar_path(data)
    sc.write_text(json.dumps({"version": 1, "channels": {
        "0": {"gain": 0.5, "offset": 0.0, "target_units": "pA", "note": "keep me"}}}))
    info = ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    calib = CalibrationSet()
    calib.set("d", 0, ChannelCalibration(gain=0.25, offset=0.0, target_units="pA"))

    real_replace = os.replace
    atomic_write.os.replace = lambda *a, **kw: (_ for _ in ()).throw(
        OSError(28, "No space left on device"))
    try:
        reason: list[str] = []
        outcome = save_sidecar(data, "d", calib, (info,), reason)
    finally:
        atomic_write.os.replace = real_replace

    assert outcome != SIDECAR_WRITTEN
    assert json.loads(sc.read_text())["channels"]["0"]["note"] == "keep me"
    assert reason and "space" in reason[0].lower(), reason


def _two_channels():
    from nanoscope.core.types import ChannelInfo, ChannelKind
    return (ChannelInfo(index=0, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT),
            ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE))


def _noted_sidecar(tmp_path):
    """A file whose ch0 carries a note and whose ch1 is calibrated too — so a reset of ch0
    leaves the FILE in place. That is the case the file-level question cannot see."""
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    ch0, ch1 = _two_channels()
    calib = CalibrationSet()
    calib.set("d", 0, ChannelCalibration(gain=0.5, offset=0.0, target_units="pA",
                                         note="1 MOhm standard, 2026-07-01"))
    calib.set("d", 1, ChannelCalibration(gain=0.2, offset=0.0, target_units="mV"))
    assert save_sidecar(data, "d", calib, (ch0, ch1)) == SIDECAR_WRITTEN
    return data, calib, (ch0, ch1)


def test_a_write_that_drops_a_notes_channel_reports_the_note_it_would_erase(tmp_path):
    """The question is "what does this write DESTROY", not "does the file survive".

    ``sidecar_effect`` answers WRITTEN here — correctly, the file is rewritten and stays —
    and the confirmation built on it therefore never fires, while the one field a project
    cannot round-trip goes out of the user's file under a status line that reads like an
    ordinary save.
    """
    from nanoscope.core.calibration import SIDECAR_WRITTEN as _W, sidecar_effect

    data, calib, channels = _noted_sidecar(tmp_path)
    calib.set("d", 0, CalibrationSet.default_for(channels[0]))     # Reset to default, ch0

    assert sidecar_effect(data, "d", calib, channels) == _W       # the file is NOT at stake
    assert notes_lost_by(data, "d", calib, channels) == ((0, "1 MOhm standard, 2026-07-01"),)


def test_an_ordinary_gain_change_loses_no_note_and_reports_none(tmp_path):
    """A prompt that fires on the ordinary edit is worse than none: it is the one the user
    learns to click through, and the same click then answers the prompt that matters."""
    data, calib, channels = _noted_sidecar(tmp_path)
    calib.set("d", 0, ChannelCalibration(gain=0.75, offset=0.0, target_units="pA",
                                         note="1 MOhm standard, 2026-07-01"))
    assert notes_lost_by(data, "d", calib, channels) == ()

    # replacing a note with a DIFFERENT note is an edit, not a loss
    calib.set("d", 0, ChannelCalibration(gain=0.75, offset=0.0, target_units="pA",
                                         note="re-measured 2026-08-02"))
    assert notes_lost_by(data, "d", calib, channels) == ()


def test_a_channel_with_no_note_and_a_file_with_no_sidecar_report_nothing(tmp_path):
    """Silence must mean "nothing to lose", so it may not be produced by an empty note or by
    a missing file being read as an erasure."""
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    ch0, ch1 = _two_channels()
    calib = CalibrationSet()
    calib.set("d", 0, ChannelCalibration(gain=0.5, offset=0.0, target_units="pA"))
    calib.set("d", 1, ChannelCalibration(gain=0.2, offset=0.0, target_units="mV"))
    assert notes_lost_by(data, "d", calib, (ch0, ch1)) == ()       # no sidecar at all
    save_sidecar(data, "d", calib, (ch0, ch1))
    calib.set("d", 0, CalibrationSet.default_for(ch0))
    assert notes_lost_by(data, "d", calib, (ch0, ch1)) == ()       # entry goes, no note in it


def test_deleting_the_whole_file_reports_every_note_in_it(tmp_path):
    """The removal case still answers, so a caller cannot be told "nothing is at stake" for
    the most destructive write there is."""
    data, calib, channels = _noted_sidecar(tmp_path)
    for info in channels:
        calib.set("d", info.index, CalibrationSet.default_for(info))
    assert notes_lost_by(data, "d", calib, channels) == ((0, "1 MOhm standard, 2026-07-01"),)


# ================================================================================================
# THE GUARD — every write over user data goes through the door
# ================================================================================================
_WRITE_CALLS = {"write_text", "write_bytes", "to_csv", "to_parquet", "savefig"}
_NUMPY_WRITERS = {"save", "savez", "savez_compressed", "savetxt"}
_STAGERS = {"staged_path", "staged_write", "stage"}
# An in-memory buffer is not a file on the user's disk. `project._df_to_parquet_bytes`
# serialises into one and the ZIP member is written through the staged handle, which is the
# whole point — the guard must not push it into inventing a temp file.
_BUFFERS = {"BytesIO", "StringIO"}


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)


def unstaged_writes(src: str, filename: str = "<src>") -> list[str]:
    """Conservative lexical check of file serializers and their staged destinations.

    Track a staged name only after it is bound and only inside its context. A fresh
    TemporaryDirectory and streams derived from it are also safe: they cannot truncate
    an existing user file. Nested functions are inspected once, in their own scope.
    This remains a syntactic build guard, not a proof about arbitrary Python programs.
    """
    out: list[str] = []

    class Guard(ast.NodeVisitor):
        def __init__(self):
            self.safe: set[str] = set()
            self.hashes: set[str] = set()
            self.memory_sinks: set[str] = set()
            self.function = "<module>"

        def staged(self, node):
            if isinstance(node, ast.Name):
                return node.id in self.safe
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                return self.staged(node.left)
            if not isinstance(node, ast.Call):
                return False
            name = _call_name(node)
            if name in _STAGERS | _BUFFERS | self.memory_sinks | {"TemporaryDirectory"}:
                return True
            if name in {"Path", "TextIOWrapper"} and node.args:
                return self.staged(node.args[0])
            if name == "fdopen" and node.args:
                return self.staged(node.args[0])
            if name == "open" and isinstance(node.func, ast.Attribute):
                return self.staged(node.func.value)
            if name == "GzipFile":
                return any(kw.arg == "fileobj" and self.staged(kw.value) for kw in node.keywords)
            return False

        def visit_FunctionDef(self, node):
            before, function = self.safe, self.function
            hashes, sinks = self.hashes, self.memory_sinks
            self.safe, self.function = set(before), node.name
            self.hashes, self.memory_sinks = set(hashes), set(sinks)
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                self.safe.discard(argument.arg)
                self.hashes.discard(argument.arg)
                self.memory_sinks.discard(argument.arg)
            for statement in node.body:
                self.visit(statement)
            self.safe, self.function = before, function
            self.hashes, self.memory_sinks = hashes, sinks

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Assign(self, node):
            self.visit(node.value)
            staged = self.staged(node.value)
            for target in node.targets:
                if isinstance(target, ast.Tuple) and _call_name(node.value) == "mkstemp":
                    # Both the descriptor and path came from a fresh exclusive file.
                    self.safe.update(item.id for item in target.elts if isinstance(item, ast.Name))
                if isinstance(target, ast.Name):
                    if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                            and isinstance(node.value.func.value, ast.Name)
                            and node.value.func.value.id == "hashlib"):
                        self.hashes.add(target.id)
                    else:
                        self.hashes.discard(target.id)
                    if staged:
                        self.safe.add(target.id)
                    else:
                        self.safe.discard(target.id)

        def visit_ClassDef(self, node):
            # Recognize the small streaming hash sink used to compute an NPY digest.
            # It must only feed bytes into a known hashlib object and return len(data).
            # An arbitrary class named Sink is not sufficient evidence.
            methods = {method.name: method for method in node.body if isinstance(method, ast.FunctionDef)}
            write = methods.get("write")
            if write and len(write.args.args) == 2 and len(write.body) == 2:
                data = write.args.args[1].arg
                first, last = write.body
                update = first.value if isinstance(first, ast.Expr) else None
                returned = last.value if isinstance(last, ast.Return) else None
                if (isinstance(update, ast.Call) and isinstance(update.func, ast.Attribute)
                        and isinstance(update.func.value, ast.Name) and update.func.value.id in self.hashes
                        and update.func.attr == "update" and len(update.args) == 1
                        and isinstance(update.args[0], ast.Name) and update.args[0].id == data
                        and isinstance(returned, ast.Call) and isinstance(returned.func, ast.Name)
                        and returned.func.id == "len" and len(returned.args) == 1
                        and isinstance(returned.args[0], ast.Name) and returned.args[0].id == data
                        and set(methods) <= {"write", "flush"}
                        and ("flush" not in methods or all(isinstance(x, ast.Pass) for x in methods["flush"].body))):
                    self.memory_sinks.add(node.name)
            self.generic_visit(node)

        def visit_With(self, node):
            before = set(self.safe)
            for item in node.items:
                self.visit(item.context_expr)
                if isinstance(item.optional_vars, ast.Name):
                    if self.staged(item.context_expr):
                        self.safe.add(item.optional_vars.id)
                    else:
                        self.safe.discard(item.optional_vars.id)
            for statement in node.body:
                self.visit(statement)
            self.safe = before

        visit_AsyncWith = visit_With

        def visit_If(self, node):
            self.visit(node.test)
            before = set(self.safe)
            for statement in node.body:
                self.visit(statement)
            left = self.safe
            self.safe = set(before)
            for statement in node.orelse:
                self.visit(statement)
            self.safe &= left

        def visit_Call(self, node):
            self.generic_visit(node)
            if not isinstance(node.func, ast.Attribute):
                return
            name = node.func.attr
            numpy_write = (name in _NUMPY_WRITERS and isinstance(node.func.value, ast.Name)
                           and node.func.value.id in {"np", "numpy"})
            if name not in _WRITE_CALLS and not numpy_write:
                return
            # FigureFormatPane.write_text edits an ElementRef. Its required 'run'
            # keyword is not accepted by pathlib.Path.write_text; no file API is exempt.
            if name == "write_text" and any(kw.arg == "run" for kw in node.keywords):
                return
            if name in {"write_text", "write_bytes"}:
                target = node.func.value
            else:
                keyword = {"to_csv": "path_or_buf", "to_parquet": "path", "savefig": "fname",
                           "save": "file", "savez": "file", "savez_compressed": "file", "savetxt": "fname"}[name]
                target = node.args[0] if node.args else next(
                    (kw.value for kw in node.keywords if kw.arg == keyword), None)
            if target is None or (isinstance(target, ast.Constant) and target.value is None):
                return  # pandas returns an in-memory string/bytes when no path is supplied
            if not self.staged(target):
                out.append(f"{self.function}:{node.lineno}: {ast.unparse(node)[:90]}")

    Guard().visit(ast.parse(src, filename))
    return out


def test_no_module_in_the_app_writes_a_path_it_did_not_stage():
    """PORT_PLAN rule 21's sentence, made checkable.

    It claimed every write over user data was staged, and four writers were not. This is the
    guard that keeps the claim true, and the reason the rule lives in one module rather than
    in whichever writer someone remembered.
    """
    offenders: dict[str, list[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        if path.name == "atomic_write.py":
            continue        # the door itself
        found = unstaged_writes(path.read_text(), str(path))
        if found:
            offenders[str(path.relative_to(PKG))] = found
    assert offenders == {}, offenders


def test_the_guard_would_actually_catch_a_violation():
    """Probed, not assumed. A guard nobody has broken on purpose pins nothing."""
    bad = (
        "def export(df, path):\n"
        "    path.write_text('x')\n"
        "    df.to_csv(path, index=False)\n"
    )
    assert len(unstaged_writes(bad)) == 2

    good = (
        "def export(df, path):\n"
        "    with staged_path(path) as staged:\n"
        "        staged.write_text('x')\n"
        "    with staged_write_set() as group:\n"
        "        df.to_csv(group.stage(path), index=False)\n"
        "        out = group.stage(path)\n"
        "        out.write_bytes(b'y')\n"
    )
    assert unstaged_writes(good) == []


def test_the_guard_is_not_satisfied_by_staging_a_DIFFERENT_path():
    """The plausible near-miss: stage one file, write another."""
    sneaky = (
        "def export(df, path, other):\n"
        "    with staged_path(path) as staged:\n"
        "        staged.write_text('x')\n"
        "        other.write_text('and this one too')\n"
    )
    assert len(unstaged_writes(sneaky)) == 1


def test_guard_tracks_staging_order_scope_and_reassignment():
    source = '''
def export(path, other):
    staged.write_text('before staging')
    with staged_path(path) as staged:
        staged.write_text('safe')
        staged = other
        staged.write_text('reassigned')
    staged.write_text('after context')
'''
    assert len(unstaged_writes(source)) == 3


def test_guard_counts_nested_writers_once_and_checks_keyword_destinations():
    source = '''
def outer(frame, path):
    def inner():
        frame.to_csv(path_or_buf=path)
        frame.to_parquet(path=path)
        fig.savefig(fname=path)
'''
    found = unstaged_writes(source)
    assert len(found) == 3
    assert all(item.startswith('inner:') for item in found)


def test_guard_proves_compressed_stream_comes_from_private_directory():
    source = '''
def export(frame, folder):
    with tempfile.TemporaryDirectory(dir=folder) as stage:
        stage = Path(stage)
        staged_trace = stage / 'trace.csv.gz'
        with staged_trace.open('wb') as binary:
            with gzip.GzipFile(fileobj=binary, mode='wb') as zipped:
                with io.TextIOWrapper(zipped) as text:
                    frame.to_csv(text)
'''
    assert unstaged_writes(source) == []
    assert len(unstaged_writes(source.replace('staged_trace.open', 'destination.open'))) == 1


def test_guard_distinguishes_figure_text_edits_from_path_writes():
    assert unstaged_writes("pane.write_text(ref, 'text', value, run=True)") == []
    assert len(unstaged_writes("path.write_text(value, encoding='utf-8')")) == 1


def test_guard_checks_numpy_paths_and_their_keyword_spellings():
    assert len(unstaged_writes('''
np.save(path, values)
np.savez_compressed(file=path, array=values)
numpy.savetxt(fname=path, X=values)
''')) == 3
    assert unstaged_writes('''
with staged_path(path) as temporary:
    np.savez_compressed(temporary, array=values)
''') == []


@pytest.mark.parametrize("suffix,serializer", [(".npy", "save"), (".npz", "savez_compressed")])
def test_numpy_partial_archive_failure_preserves_existing_file(tmp_path, monkeypatch, suffix, serializer):
    from nanoscope.core.atomic_write import write_staged

    path = tmp_path / ("arrays" + suffix)
    original = np.arange(8, dtype=np.float64)
    getattr(np, serializer)(path, original)
    before = path.read_bytes()

    def partial_write(stream, *args, **kwargs):
        stream.write(b"incomplete NPY payload")
        raise OSError(28, "injected disk full after writing archive data")

    with monkeypatch.context() as patch:
        patch.setattr(np.lib.format, "write_array", partial_write)
        with pytest.raises(OSError):
            write_staged(path, getattr(np, serializer), np.arange(100))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    write_staged(path, getattr(np, serializer), original + 3)
    if suffix == ".npz":
        with np.load(path) as loaded:
            np.testing.assert_array_equal(loaded["arr_0"], original + 3)
    else:
        np.testing.assert_array_equal(np.load(path), original + 3)
    assert list(tmp_path.iterdir()) == [path], "NumPy must not append a second suffix to staging"


def test_path_serializer_failure_preserves_destination_and_cleans_staging(tmp_path):
    from nanoscope.core.atomic_write import write_staged

    path = tmp_path / "table.csv"
    path.write_bytes(b"previous complete export")

    def failing_writer(staged, *, marker):
        assert staged != path and staged.suffix == ".csv"
        staged.write_text(marker)
        raise OSError(28, "injected disk full")

    with pytest.raises(OSError):
        write_staged(path, failing_writer, marker="partial")
    assert path.read_bytes() == b"previous complete export"
    assert list(tmp_path.iterdir()) == [path]


def test_path_serializer_preserves_return_value_and_real_suffix(tmp_path):
    from pathlib import Path
    from nanoscope.core.atomic_write import write_staged

    path = tmp_path / "figure.svg"
    assert write_staged(path, Path.write_text, "<svg/>", encoding="utf-8") == 6
    assert path.read_text() == "<svg/>"
    assert list(tmp_path.iterdir()) == [path]


# ================================================================================================
# §2.6 — the "treat this channel as" override survives a project round trip (v8)
# ================================================================================================
def test_the_channel_kind_override_round_trips_through_a_project(tmp_path):
    """`ChannelCalibration.kind` is what makes a channel the user relabelled to pA sort
    ABOVE the voltage pane, because `trace_view._kind` feeds both the pane order and the
    overlay grouping key. A record that could not carry it would restore None, and the
    panel's Apply would then write that default over the user's own sidecar -- the exact
    failure `note` was added in v5 to end.
    """
    from nanoscope.core.calibration import ChannelCalibration
    from nanoscope.core.types import ChannelKind

    cal = ChannelCalibration(gain=0.5, offset=0.0, target_units="pA",
                             note="relabelled by hand", kind=ChannelKind.CURRENT)
    rec = CalibrationRec.from_calibration("d", 0, cal)
    assert rec.kind == ChannelKind.CURRENT.value

    state = ProjectState(calibrations=[rec])
    back = load_project(save_project(tmp_path / "p.nanoscope", state)).calibrations[0]
    assert back.kind == ChannelKind.CURRENT.value
    assert back.to_calibration().kind is ChannelKind.CURRENT


def test_a_pre_v8_project_does_not_blank_the_sidecars_kind_override(tmp_path):
    """Three-state, like `note`: an ABSENT key is a project with no opinion, and it must
    leave the override the sidecar installed moments ago in `Session.add_document` alone
    rather than assert "no override" over it."""
    from nanoscope.core.calibration import ChannelCalibration
    from nanoscope.core.types import ChannelKind

    state = ProjectState(calibrations=[CalibrationRec(
        doc_id="d", channel=0, gain=1.0, offset=0.0,
        target_units="pA", provenance="unknown")])
    p = save_project(tmp_path / "p.nanoscope", state)
    import zipfile
    with zipfile.ZipFile(p) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert "kind" not in manifest["calibrations"][0], manifest["calibrations"][0]

    rec = load_project(p).calibrations[0]
    assert rec.kind is None
    existing = ChannelCalibration(gain=1.0, offset=0.0, target_units="pA",
                                  kind=ChannelKind.CURRENT)
    assert rec.to_calibration(existing).kind is ChannelKind.CURRENT   # not blanked
    assert rec.to_calibration().kind is None                          # nothing to keep
